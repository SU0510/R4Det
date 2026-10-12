"""C1 minimal arm: observation-validity-conditioned update (candidate C, arm 1).

Pre-registration: ``docs/rssm_c_validity_update_prereg_20261012.md`` (equations
and gates are FROZEN there; this file implements exactly that, nothing more).

Single research factor vs the standard ``no2d_igdr`` mainline
(``MotionAlignedRSSMFusion``): the observation-driven correction is re-scaled
per BEV cell by a gain that is a function of the recurrent state and of an
input-derived radar-return indicator ``valid_t`` (built from the *point cloud
the network sees* -- no GT, no labels, no boxes, no diagnostic mask):

    g_t = 1 + tanh(W_v * [h_t ; valid_t] + b_v)        (B, latent_dim, H, W)
    z_t = mu_q + (g_t - 1) * (mu_q - mu_p)             (== mu_p + g_t * (mu_q - mu_p))

Everything else is inherited unchanged: encoder, ConvGRU transition, prior,
posterior mu/logstd, KL (still computed on the un-gated mu_q/logstd_q vs the
prior), decoder/reconstruction, output_proj + residual feat, deformable state
alignment, schedule, losses, detection head.

Identity-at-init contract
-------------------------
``W_v`` and ``b_v`` are zero-initialised, so ``g_t == 1`` exactly and
``z_t == mu_q + 0 * (mu_q - mu_p) == mu_q`` bitwise. With the standard
mainline weights loaded the arm is therefore bitwise identical to the baseline
until the gate moves. The gate is NOT an idle parameter: its gradient is

    dL/dW_v = (dL/dz_t) * (mu_q - mu_p) * valid-part * ...

which is non-zero as soon as the upstream gradient is non-zero, so DDP sees
the module as used from step 0.

The deterministic control arm (``ValidityGatedDeterministicRSSMFusion``)
applies the SAME gain module (same shape, same init) to the project's
canonical deterministic latent (``DeterministicMotionAlignedLatentFusion``,
the class already used by the ``..._deterministic_latent_...`` mainline
control): there the latent has no prior, so the gain multiplies the
observation-driven latent directly, ``z = z_mu + (g - 1) * z_mu``. It answers
the pre-registered question "with the same update strategy, does the
stochastic state beat a matched deterministic state", it does NOT enter the
first arm's decision.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.models.builder import FUSION_LAYERS
from mmdet3d.models.fusion_layers.rssm_fusion import (
    DeterministicMotionAlignedLatentFusion, MotionAlignedRSSMFusion)


def build_radar_validity_mask(points, num_samples, height, width,
                              point_cloud_range, layout='yx', device=None,
                              dtype=torch.float32):
    """Per-BEV-cell radar-return indicator built from the input point cloud.

    ``valid[i, 0, r, c] = 1`` iff at least one loaded point of sample ``i``
    falls into that BEV cell. This is the *only* coverage signal the model is
    allowed to consume at inference (prereg SS4): it is a pure function of the
    point tensor the radar branch already sees, so it cannot leak GT boxes,
    labels or the diagnostic masks used by the SS79 experiment.

    Args:
        points (list[Tensor]): per-sample point tensors (N_i, C>=2); columns
            0/1 are x/y in the ego frame of that frame.
        num_samples (int): batch size B.
        height (int): number of rows of the output grid.
        width (int): number of columns of the output grid.
        point_cloud_range (list[float]): [x_min, y_min, z_min, x_max, y_max, z_max].
        layout (str): 'yx' -> rows are the y cells, columns are the x cells
            (the pillar-scatter layout used by ``PointPillarsScatter``);
            'xy' -> rows are the x cells, columns are the y cells (the layout
            of the fused BEV at the RSSM call site). The caller must pass the
            layout of the tensor the mask will be concatenated with; the
            detector asserts the final shape, so a wrong layout fails loudly
            instead of silently mis-aligning the mask.
        device: optional device override.
        dtype: output dtype (float, so it can be concatenated with features).

    Returns:
        Tensor (B, 1, height, width) with values in {0, 1}.
    """
    if layout not in ('yx', 'xy'):
        raise ValueError(f"layout must be 'yx' or 'xy', got {layout!r}")
    x_min, y_min = float(point_cloud_range[0]), float(point_cloud_range[1])
    x_max, y_max = float(point_cloud_range[3]), float(point_cloud_range[4])
    if x_max <= x_min or y_max <= y_min:
        raise ValueError('point_cloud_range must have x_max > x_min, y_max > y_min')

    if device is None:
        for pts in points:
            if torch.is_tensor(pts) and pts.numel() > 0:
                device = pts.device
                break
    if device is None:
        device = torch.device('cpu')

    out = torch.zeros((num_samples, 1, height, width), device=device, dtype=dtype)
    n_x, n_y = (height, width) if layout == 'xy' else (width, height)
    for i in range(num_samples):
        pts = points[i] if i < len(points) else None
        if pts is None or not torch.is_tensor(pts) or pts.numel() == 0:
            continue
        if pts.dim() != 2 or pts.shape[1] < 2:
            raise ValueError(f'points[{i}] must be (N, C>=2), got {tuple(pts.shape)}')
        pts = pts.to(device)
        x, y = pts[:, 0], pts[:, 1]
        ix = torch.floor((x - x_min) / (x_max - x_min) * n_x).long()
        iy = torch.floor((y - y_min) / (y_max - y_min) * n_y).long()
        keep = (ix >= 0) & (ix < n_x) & (iy >= 0) & (iy < n_y)
        if not bool(keep.any()):
            continue
        ix, iy = ix[keep], iy[keep]
        flat = torch.zeros(n_y * n_x, device=device, dtype=dtype)
        flat[iy * n_x + ix] = 1.0
        grid = flat.view(n_y, n_x)  # rows = y cells, cols = x cells
        out[i, 0] = grid if layout == 'yx' else grid.t()
    return out


class _ValidityGainMixin(object):
    """Shared gain module for both arms of the C1 pair (frozen formula)."""

    accepts_obs_valid = True

    def _build_validity_gain(self, state_channels, latent_dim):
        self.use_validity_gain = True
        self.validity_gain = nn.Conv2d(
            state_channels + 1, latent_dim, kernel_size=1)
        # zero-init => g == 1 exactly (identity-at-init contract)
        nn.init.zeros_(self.validity_gain.weight)
        nn.init.zeros_(self.validity_gain.bias)

    def _validity_gate(self, h_t, obs_valid, latent_dim):
        if (not self.use_validity_gain) or obs_valid is None:
            return None
        if obs_valid.dim() == 3:
            obs_valid = obs_valid.unsqueeze(1)
        if obs_valid.shape[-2:] != h_t.shape[-2:]:
            obs_valid = F.interpolate(
                obs_valid, size=h_t.shape[-2:], mode='nearest')
        obs_valid = obs_valid.to(dtype=h_t.dtype, device=h_t.device)
        cat = torch.cat([h_t, obs_valid], dim=1)
        return 1.0 + torch.tanh(self.validity_gain(cat))

    @staticmethod
    def _gate_stats(gate, obs_valid):
        """Cheap detached diagnostics (channel-mean then spatial means)."""
        if gate is None:
            return {}
        with torch.no_grad():
            g = gate.mean(dim=1)  # (B, H, W)
            # 0-dim tensors: the runner's log buffer rejects python floats.
            stats = {'v_gate_mean': g.mean()}
            if obs_valid is not None:
                v = obs_valid
                if v.dim() == 3:
                    v = v.unsqueeze(1)
                if v.shape[-2:] != g.shape[-2:]:
                    v = F.interpolate(v, size=g.shape[-2:], mode='nearest')
                v = (v[:, 0] > 0.5).to(g.dtype)  # (B,H,W): no channel broadcast
                n_v = v.sum()
                if float(n_v.item()) > 0:
                    stats['v_gate_valid_mean'] = (g * v).sum() / n_v
                n_i = v.numel() - n_v
                if float(n_i.item()) > 0:
                    stats['v_gate_invalid_mean'] = (g * (1.0 - v)).sum() / n_i
            return stats


@FUSION_LAYERS.register_module()
class ValidityGatedRSSMFusion(_ValidityGainMixin, MotionAlignedRSSMFusion):
    """Arm C1: standard MotionAlignedRSSMFusion + one validity-conditioned gain.

    Single variable vs the mainline: the posterior correction is re-scaled by
    ``g_t`` (see module docstring). All other arguments are inherited from
    ``MotionAlignedRSSMFusion`` and must stay identical to the mainline config.
    """

    def __init__(self,
                 point_cloud_range=(0, -39.68, -4, 69.12, 39.68, 2),
                 valid_layout='xy',
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.point_cloud_range = list(point_cloud_range)
        self.valid_layout = valid_layout
        self._build_validity_gain(self.channels, self.latent_dim)

    def _obs_valid_from_points(self, obs_points, feat):
        """Input points -> (B, 1, H, W) mask matching ``feat``'s BEV layout."""
        if obs_points is None:
            return None
        pts = obs_points if isinstance(obs_points, (list, tuple)) else [obs_points]
        B, _, H, W = feat.shape
        return build_radar_validity_mask(
            pts, B, H, W, self.point_cloud_range, layout=self.valid_layout,
            device=feat.device, dtype=feat.dtype)

    def forward(self, feat, velocity=None, use_posterior=True,
                deterministic=True, detach_state=True, obs_valid=None,
                obs_points=None):
        B, C, H, W = feat.shape
        if obs_valid is None:
            obs_valid = self._obs_valid_from_points(obs_points, feat)

        if self.use_action:
            if velocity is None:
                velocity = torch.zeros(B, self.action_dim, device=feat.device)
            velocity_map = velocity[:, :, None, None].expand(
                B, self.action_dim, H, W)
        else:
            velocity_map = None

        if self.h_state is None or self.h_state.shape[0] != B:
            self.h_state = torch.zeros(
                B, self.channels, H, W, device=feat.device, dtype=feat.dtype)
            self.z_state = torch.zeros(
                B, self.latent_dim, H, W, device=feat.device, dtype=feat.dtype)

        h_aligned = self._deform_align(
            self.h_state, feat,
            self.align_h_offset_mask, self.align_h_deform_conv)
        if self.align_z_state:
            z_aligned = self._deform_align(
                self.z_state, feat,
                self.align_z_offset_mask, self.align_z_deform_conv)
        else:
            z_aligned = self.z_state

        if self.use_action:
            x = torch.cat([z_aligned, velocity_map], dim=1)
        else:
            x = z_aligned
        h_t = self.transition(x, h_aligned)

        mu_p = self.prior_mu(h_t)
        logstd_p = self.prior_logstd(h_t)

        e_t = self.encoder(feat)
        mu_q = self.posterior_mu(torch.cat([h_t, e_t], dim=1))
        logstd_q = self.posterior_logstd(torch.cat([h_t, e_t], dim=1))

        gate = self._validity_gate(h_t, obs_valid, self.latent_dim)
        if use_posterior:
            # g == 1 at init: mu_q + 0 * (mu_q - mu_p) == mu_q bitwise.
            mu_sel = mu_q if gate is None else mu_q + (gate - 1.0) * (mu_q - mu_p)
            z_t = mu_sel if deterministic else self.sample(mu_sel, logstd_q)
        else:
            z_t = mu_p if deterministic else self.sample(mu_p, logstd_p)

        kl, stats = self.kl_loss(mu_q, logstd_q, mu_p, logstd_p)
        stats.update(self._gate_stats(gate, obs_valid))

        reconstruction = self.decoder(torch.cat([h_t, z_t], dim=1))
        output = self.output_proj(z_t)
        output = output + feat

        if detach_state:
            self.h_state = h_t.detach()
            self.z_state = z_t.detach()
        else:
            self.h_state = h_t
            self.z_state = z_t

        return output, reconstruction, kl * self.kl_scale, h_t, z_t, stats


@FUSION_LAYERS.register_module()
class ValidityGatedDeterministicRSSMFusion(
        _ValidityGainMixin, DeterministicMotionAlignedLatentFusion):
    """Arm C1-D: the pre-registered matched deterministic control.

    Same gain module (identical shape/init) applied to the deterministic
    latent: ``z = z_mu + (g - 1) * z_mu``. No prior/logstd/KL and no sampling,
    exactly like the mainline ``DeterministicMotionAlignedLatentFusion``.
    """

    def __init__(self,
                 point_cloud_range=(0, -39.68, -4, 69.12, 39.68, 2),
                 valid_layout='xy',
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.point_cloud_range = list(point_cloud_range)
        self.valid_layout = valid_layout
        self._build_validity_gain(self.channels, self.latent_dim)

    def _obs_valid_from_points(self, obs_points, feat):
        if obs_points is None:
            return None
        pts = obs_points if isinstance(obs_points, (list, tuple)) else [obs_points]
        B, _, H, W = feat.shape
        return build_radar_validity_mask(
            pts, B, H, W, self.point_cloud_range, layout=self.valid_layout,
            device=feat.device, dtype=feat.dtype)

    def forward(self, feat, velocity=None, use_posterior=True,
                deterministic=True, detach_state=True, obs_valid=None,
                obs_points=None):
        B, C, H, W = feat.shape
        if obs_valid is None:
            obs_valid = self._obs_valid_from_points(obs_points, feat)

        if self.use_action:
            if velocity is None:
                velocity = torch.zeros(B, self.action_dim, device=feat.device)
            velocity_map = velocity[:, :, None, None].expand(
                B, self.action_dim, H, W)
        else:
            velocity_map = None

        if self.h_state is None or self.h_state.shape[0] != B:
            self.h_state = torch.zeros(
                B, self.channels, H, W, device=feat.device, dtype=feat.dtype)
            self.z_state = torch.zeros(
                B, self.latent_dim, H, W, device=feat.device, dtype=feat.dtype)

        h_aligned = self._deform_align(
            self.h_state, feat,
            self.align_h_offset_mask, self.align_h_deform_conv)
        if self.align_z_state:
            z_aligned = self._deform_align(
                self.z_state, feat,
                self.align_z_offset_mask, self.align_z_deform_conv)
        else:
            z_aligned = self.z_state

        if self.use_action:
            x = torch.cat([z_aligned, velocity_map], dim=1)
        else:
            x = z_aligned
        h_t = self.transition(x, h_aligned)

        e_t = self.encoder(feat)
        z_mu = self.posterior_mu(torch.cat([h_t, e_t], dim=1))

        gate = self._validity_gate(h_t, obs_valid, self.latent_dim)
        z_t = z_mu if gate is None else z_mu + (gate - 1.0) * z_mu
        if not deterministic and self.posterior_noise_std > 0:
            z_t = z_t + self.posterior_noise_std * torch.randn_like(z_t)

        stats = {}
        stats.update(self._gate_stats(gate, obs_valid))

        reconstruction = self.decoder(torch.cat([h_t, z_t], dim=1))
        output = self.output_proj(z_t)
        output = output + feat

        if detach_state:
            self.h_state = h_t.detach()
            self.z_state = z_t.detach()
        else:
            self.h_state = h_t
            self.z_state = z_t

        return output, reconstruction, None, h_t, z_t, stats
