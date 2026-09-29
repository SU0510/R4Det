import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from mmcv.ops import ModulatedDeformConv2d
from mmcv.runner import BaseModule, auto_fp16
from mmcv.cnn import ConvModule, xavier_init

from mmdet3d.models.builder import FUSION_LAYERS


class ConvGRUCell(nn.Module):
    """Convolutional GRU cell for 2D feature maps.

    Standard GRU equations adapted for spatial (B, C, H, W) tensors.
    Input x = [z_{t-1}, action_t], hidden state h = h_{t-1}.

        r_t = σ(conv_r([x, h]))          # reset gate
        u_t = σ(conv_u([x, h]))          # update gate
        h̃_t = tanh(conv_h([x, r_t*h]))  # candidate
        h_t = (1 - u_t)*h + u_t*h̃_t     # output
    """

    def __init__(self, input_dim, hidden_dim, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2

        self.reset_conv = nn.Conv2d(
            input_dim + hidden_dim, hidden_dim,
            kernel_size, padding=padding
        )

        self.update_conv = nn.Conv2d(
            input_dim + hidden_dim, hidden_dim,
            kernel_size, padding=padding
        )

        self.candidate_conv = nn.Conv2d(
            input_dim + hidden_dim, hidden_dim,
            kernel_size, padding=padding
        )

    def forward(self, x, h):
        """
        Args:
            x: Input tensor (B, input_dim, H, W) — concat of [z_{t-1}, action].
            h: Hidden state (B, hidden_dim, H, W) — h_{t-1}.
        Returns:
            h_new: (B, hidden_dim, H, W)
        """
        combined = torch.cat([x, h], dim=1)

        r = torch.sigmoid(self.reset_conv(combined))
        u = torch.sigmoid(self.update_conv(combined))

        combined_c = torch.cat([x, r * h], dim=1)
        h_candidate = torch.tanh(self.candidate_conv(combined_c))

        h_new = (1.0 - u) * h + u * h_candidate
        return h_new


@FUSION_LAYERS.register_module()
class BEVRSSMTemporalFusion(BaseModule):
    """RSSM for BEV temporal fusion — per-frame recurrent processing (filtering mode).

    Each forward call processes ONE frame. The model maintains internal
    deterministic state h and stochastic state z across calls:

        h_t = ConvGRU(h_{t-1}, z_{t-1}, action)    ← recurrent transition
        prior:     p(z_t | h_t)                     ← KL regularizer only
        posterior: q(z_t | h_t, encoder(feat))      ← used for both train & inference
        KL = KL(q || p)
        output = output_proj(z_t) + feat

    h_0 is initialized to zeros (NOT from any observation).

    By default, both training and validation use the posterior mean mu_q
    deterministically (use_posterior=True, deterministic=True). The prior
    only serves as a KL regularizer — it is never used as the detection
    feature source at inference time.

    Args:
        in_channels (int): Input BEV feature channels.
        out_channels (int): Output BEV feature channels.
        kernel_size (int): Conv kernel size for GRU cell.
        latent_dim (int | None): Stochastic state dim (defaults to in_channels).
        hidden_dim (int): Encoder hidden dimension.
        action_dim (int): Action (velocity) dimension.
        kl_scale (float): KL loss weight.
        free_nats (float): Free-bits threshold for KL (0 = disabled).
        min_std (float): Minimum std for logstd constraint.
        init_std (float): Initial std for prior/posterior logstd layers.
        norm_cfg (dict): Normalization config.
        act_cfg (dict): Activation config.
        init_cfg (dict): Init config.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        latent_dim=None,
        hidden_dim=64,
        action_dim=2,
        kl_scale=1.0,
        free_nats=0.0,
        min_std=0.1,
        init_std=0.2,
        norm_cfg=dict(type='BN', requires_grad=True),
        act_cfg=dict(type='ReLU', inplace=True),
        init_cfg=None
    ):
        super().__init__(init_cfg)

        if in_channels != out_channels:
            out_channels = in_channels

        if latent_dim is None:
            latent_dim = in_channels

        self.channels = out_channels
        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.kernel_size = kernel_size
        self.kl_scale = kl_scale
        self.free_nats = free_nats
        self.min_std = min_std
        self.min_logstd = math.log(min_std)
        self.init_std = init_std

        ################################################
        # Encoder: observation → latent_dim
        ################################################
        self.encoder = nn.Sequential(
            ConvModule(
                in_channels,
                hidden_dim,
                3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            ),
            ConvModule(
                hidden_dim,
                latent_dim,
                3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=None
            )
        )

        # Build-time switch: action_dim <= 0 disables the velocity branch
        # entirely (no extra channels, no zero-input in the graph). Set
        # action_dim>0 AND pass velocity in forward to re-enable. The action
        # code paths below are preserved for that future use.
        self.use_action = action_dim > 0

        ################################################
        # ConvGRU: h_{t-1} + z_{t-1} [+ action] → h_t
        ################################################
        gru_input_dim = latent_dim + (action_dim if self.use_action else 0)
        self.transition = ConvGRUCell(
            input_dim=gru_input_dim,
            hidden_dim=out_channels,
            kernel_size=kernel_size
        )

        ################################################
        # Prior: p(z | h)
        ################################################
        self.prior_mu = nn.Conv2d(
            out_channels,
            latent_dim,
            3,
            padding=1
        )
        self.prior_logstd = nn.Conv2d(
            out_channels,
            latent_dim,
            3,
            padding=1
        )

        ################################################
        # Posterior: q(z | h, e)
        ################################################
        self.posterior_mu = nn.Conv2d(
            out_channels + latent_dim,
            latent_dim,
            3,
            padding=1
        )
        self.posterior_logstd = nn.Conv2d(
            out_channels + latent_dim,
            latent_dim,
            3,
            padding=1
        )

        ################################################
        # Decoder: h + z → in_channels (reconstruct feat)
        ################################################
        self.decoder = nn.Sequential(
            ConvModule(
                out_channels + latent_dim,
                hidden_dim,
                3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg
            ),
            ConvModule(
                hidden_dim,
                in_channels,
                3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=None
            )
        )

        ################################################
        # Output: z_t → out_channels
        # z_t already fuses temporal context (h_t) + current observation (e_t)
        # via the posterior q(z | h_t, e_t), so we project z_t directly.
        # Xavier-init (see init_weights); residual `+ feat` in forward keeps
        # output ≈ feat at the start while letting z_t contribute from step 1.
        ################################################
        self.output_proj = nn.Conv2d(
            latent_dim,
            out_channels,
            3,
            padding=1
        )

        # Internal recurrent state (per-batch, maintained across forward calls)
        self.h_state = None
        self.z_state = None

        self.init_weights()

    def _logstd_bias_value(self):
        """Compute bias for logstd layers so initial std ≈ init_std.

        Under the softplus constraint:
            logstd = min_logstd + softplus(raw_logstd - min_logstd)

        At init, raw_logstd = bias (weights are zero). Solving:
            bias = min_logstd + log(init_std / min_std - 1)

        For init_std=0.2, min_std=0.1: bias = log(0.1) ≈ -2.303, giving std ≈ 0.2.
        Requires init_std > min_std.
        """
        assert self.init_std > self.min_std, \
            f"init_std ({self.init_std}) must be > min_std ({self.min_std})"
        return self.min_logstd + math.log(self.init_std / self.min_std - 1.0)

    def init_weights(self):
        """Initialize weights with Xavier, then override logstd bias.

        output_proj is Xavier-initialized (Group 1 below) — NOT zero-init —
        so z_t contributes to the output from step 1. The residual `+ feat`
        in forward keeps output ≈ feat when z_t is small, but the projection
        is no longer frozen at zero, so the RSSM branch starts learning
        immediately instead of being gated off at init.
        """
        super().init_weights()

        # Group 1: Xavier init for mu/logstd/output layers
        for m in [self.prior_mu, self.prior_logstd,
                  self.posterior_mu, self.posterior_logstd,
                  self.output_proj]:
            if hasattr(m, 'weight'):
                xavier_init(m, distribution='uniform')

        # Group 2: Xavier init for ConvGRU layers
        for m in [self.transition.reset_conv,
                  self.transition.update_conv,
                  self.transition.candidate_conv]:
            if hasattr(m, 'weight'):
                xavier_init(m, distribution='uniform')
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.uniform_(m.bias, -0.1, 0.1)

        # Override: init prior/posterior logstd bias for controlled initial std
        bias_val = self._logstd_bias_value()
        nn.init.constant_(self.prior_logstd.bias, bias_val)
        nn.init.constant_(self.posterior_logstd.bias, bias_val)

    def reset_state(self):
        """Reset internal recurrent state (call at sequence boundaries)."""
        self.h_state = None
        self.z_state = None

    def reset_for_samples(self, mask):
        """Zero out internal state for specific samples (e.g. invalid prev frames).

        Args:
            mask: (B,) bool tensor, True = samples to reset.
        """
        if self.h_state is not None and mask.any():
            keep = ~mask
            self.h_state = torch.where(
                keep[:, None, None, None], self.h_state,
                torch.zeros_like(self.h_state))
            self.z_state = torch.where(
                keep[:, None, None, None], self.z_state,
                torch.zeros_like(self.z_state))

    def _constrained_logstd(self, logstd):
        """Constrain logstd to [min_logstd, 0] using a soft lower bound."""
        logstd = self.min_logstd + F.softplus(logstd - self.min_logstd)
        # Upper bound: std ≤ 1.0 to prevent variance explosion
        return torch.clamp(logstd, max=0.0)

    def sample(self, mu, logstd):
        """Sample from distribution with smooth minimum std constraint."""
        logstd = self._constrained_logstd(logstd)
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def kl_loss(self, mu_q, logstd_q, mu_p, logstd_p):
        """KL divergence with minimum variance, free-bits, and debug statistics.

        Returns:
            kl_mean: Averaged KL loss (after free-bits clamp).
            stats: dict of detached scalar tensors for logging.
        """
        # Smooth lower bound on logstd
        logstd_q = self.min_logstd + F.softplus(logstd_q - self.min_logstd)
        logstd_p = self.min_logstd + F.softplus(logstd_p - self.min_logstd)
        # Upper bound: std ≤ 1.0 to prevent variance explosion
        logstd_q = torch.clamp(logstd_q, max=0.0)
        logstd_p = torch.clamp(logstd_p, max=0.0)

        var_q = torch.exp(2.0 * logstd_q)
        var_p = torch.exp(2.0 * logstd_p)
        std_q = torch.exp(logstd_q)
        std_p = torch.exp(logstd_p)

        # Per-element KL (before free-bits clamp)
        kl_raw = (
            logstd_p - logstd_q
            + (var_q + (mu_q - mu_p) ** 2) / (2.0 * var_p)
            - 0.5
        )

        # Free-bits: clamp per-element KL to at least `free_nats`.
        if self.free_nats > 0:
            kl_clamped = torch.clamp(kl_raw, min=self.free_nats)
        else:
            kl_clamped = kl_raw

        # Debug statistics (all detached — no graph retained)
        stats = dict(
            stat_kl_raw_mean=kl_raw.mean().detach(),
            stat_kl_effective_mean=kl_clamped.mean().detach(),
            stat_clamped_ratio=(kl_raw < self.free_nats).float().mean().detach() if self.free_nats > 0 else
                torch.zeros((), device=kl_raw.device),
            stat_mu_diff_sq=(mu_q - mu_p).square().mean().detach(),
            stat_posterior_std=std_q.mean().detach(),
            stat_prior_std=std_p.mean().detach(),
        )

        return kl_clamped.mean(), stats

    @auto_fp16(apply_to=['feat', 'velocity'])
    def forward(self, feat, velocity=None, use_posterior=True,
                deterministic=True, detach_state=True):
        """Process ONE frame through the RSSM.

        Uses internal h_{t-1}, z_{t-1} from the previous call.
        Stores h_t, z_t for the next call.

        Args:
            feat: Current BEV feature (B, C, H, W).
            velocity: Ego-motion (B, action_dim). Defaults to zeros.
            use_posterior (bool): If True, use q(z|h,encoder(feat)).
                If False, use p(z|h). Default: True (filtering mode).
            deterministic (bool): If True, use the distribution mean (mu).
                If False, sample with noise. Default: True.
            detach_state (bool): If True, detach h/z before storing as the
                internal state for the next call. Set False only inside a
                code span that will fold the sequence, then reset the state
                or detach it at the fold boundary.

        Returns:
            output: Fused BEV feature (B, out_channels, H, W).
            reconstruction: Reconstructed feat (B, in_channels, H, W).
            kl: KL divergence loss (scalar).
            h_t: Deterministic state (B, out_channels, H, W).
            z_t: Stochastic state (B, latent_dim, H, W).
            stats: dict of debug statistics (detached scalars).
        """
        B, C, H, W = feat.shape

        ################################################
        # Velocity / action (skipped when use_action is False — keeps the
        # branch out of the graph without deleting the code).
        ################################################
        if self.use_action:
            if velocity is None:
                velocity = torch.zeros(
                    B, self.action_dim, device=feat.device
                )
            velocity_map = velocity[:, :, None, None].expand(
                B, self.action_dim, H, W
            )
        else:
            velocity_map = None

        ################################################
        # Initialize state if first call
        # h_0 = zeros — NOT derived from any observation!
        ################################################
        if self.h_state is None or self.h_state.shape[0] != B:
            self.h_state = torch.zeros(
                B, self.channels, H, W,
                device=feat.device, dtype=feat.dtype
            )
            self.z_state = torch.zeros(
                B, self.latent_dim, H, W,
                device=feat.device, dtype=feat.dtype
            )

        ################################################
        # 1. Deterministic transition: h_t = ConvGRU(h_{t-1}, z_{t-1} [, action])
        ################################################
        if self.use_action:
            x = torch.cat([self.z_state, velocity_map], dim=1)
        else:
            x = self.z_state
        h_t = self.transition(x, self.h_state)

        ################################################
        # 2. Prior: p(z_t | h_t)
        ################################################
        mu_p = self.prior_mu(h_t)
        logstd_p = self.prior_logstd(h_t)

        ################################################
        # 3. Encode observation: e_t = encoder(feat)
        ################################################
        e_t = self.encoder(feat)

        ################################################
        # 4. Posterior: q(z_t | h_t, e_t)
        ################################################
        mu_q = self.posterior_mu(torch.cat([h_t, e_t], dim=1))
        logstd_q = self.posterior_logstd(torch.cat([h_t, e_t], dim=1))

        ################################################
        # 5. Select z_t based on use_posterior / deterministic
        ################################################
        if use_posterior:
            if deterministic:
                z_t = mu_q
            else:
                z_t = self.sample(mu_q, logstd_q)
        else:
            if deterministic:
                z_t = mu_p
            else:
                z_t = self.sample(mu_p, logstd_p)

        ################################################
        # 6. KL divergence (with debug statistics)
        ################################################
        kl, stats = self.kl_loss(mu_q, logstd_q, mu_p, logstd_p)

        ################################################
        # 7. Reconstruction: decoder(h_t, z_t) → feat
        ################################################
        reconstruction = self.decoder(
            torch.cat([h_t, z_t], dim=1)
        )

        ################################################
        # 8. Output: z_t → out_channels → residual to feat
        ################################################
        output = self.output_proj(z_t)
        output = output + feat  # residual: preserves original BEV features

        ################################################
        # 9. Store state for next frame. By default detach, otherwise the
        # state is kept in the graph for a bounded BPTT window.
        ################################################
        if detach_state:
            self.h_state = h_t.detach()
            self.z_state = z_t.detach()
        else:
            self.h_state = h_t
            self.z_state = z_t

        return output, reconstruction, kl * self.kl_scale, h_t, z_t, stats

@FUSION_LAYERS.register_module()
class MotionAlignedRSSMFusion(BEVRSSMTemporalFusion):
    """RSSM with motion-aware deformable alignment of historical states.

    Before the ConvGRU transition, h_{t-1} and z_{t-1} are warped via
    ModulatedDeformConv2d using offsets/masks predicted from
    concat(feat_cur, h_{t-1}) and concat(feat_cur, z_{t-1}) respectively.

    This compensates for object motion between frames — without alignment,
    the GRU consumes spatially misaligned state at each pixel, injecting
    noise into the reset/update gates.

    Inherits all RSSM components (encoder, ConvGRU, prior, posterior,
    decoder, output_proj, KL loss, state management) from
    BEVRSSMTemporalFusion. Only the transition step is modified.

    Args:
        align_kernel_size (int): Kernel size for deformable alignment conv.
        align_deform_groups (int): Deformable groups.
        align_z_state (bool): Whether to also align z_state (default True).
        (All other args inherited from BEVRSSMTemporalFusion.)
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        latent_dim=None,
        hidden_dim=64,
        action_dim=2,
        kl_scale=1.0,
        free_nats=0.0,
        min_std=0.1,
        init_std=0.2,
        norm_cfg=dict(type='BN', requires_grad=True),
        act_cfg=dict(type='ReLU', inplace=True),
        align_kernel_size=3,
        align_deform_groups=1,
        align_z_state=True,
        init_cfg=None
    ):
        # Set alignment attrs BEFORE super().__init__() because it calls
        # init_weights() → _init_alignment_weights() which reads these.
        self.align_kernel_size = align_kernel_size
        self.align_deform_groups = align_deform_groups
        self.align_z_state = align_z_state

        # Parent sets up: encoder, transition, prior, posterior, decoder,
        # output_proj, state attributes, and calls init_weights().
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            kl_scale=kl_scale,
            free_nats=free_nats,
            min_std=min_std,
            init_std=init_std,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=init_cfg
        )

        k2 = align_kernel_size * align_kernel_size
        align_offset_channels = 3 * align_deform_groups * k2

        # Alignment for h_state: feat + h_state → offsets + mask → warp h_state
        self.align_h_offset_mask = nn.Conv2d(
            in_channels + out_channels,
            align_offset_channels,
            kernel_size=align_kernel_size,
            padding=align_kernel_size // 2,
        )
        self.align_h_deform_conv = ModulatedDeformConv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=align_kernel_size,
            padding=align_kernel_size // 2,
            deform_groups=align_deform_groups,
            bias=False,
        )

        # Alignment for z_state
        if align_z_state:
            self.align_z_offset_mask = nn.Conv2d(
                in_channels + latent_dim,
                align_offset_channels,
                kernel_size=align_kernel_size,
                padding=align_kernel_size // 2,
            )
            self.align_z_deform_conv = ModulatedDeformConv2d(
                in_channels=latent_dim,
                out_channels=latent_dim,
                kernel_size=align_kernel_size,
                padding=align_kernel_size // 2,
                deform_groups=align_deform_groups,
                bias=False,
            )

        # Re-init: parent init_weights() already ran, overlay alignment zero-init
        self._init_alignment_weights()

    def _init_alignment_weights(self):
        """Zero-init offset/mask generators so alignment starts as identity."""
        for prefix in ['align_h_', 'align_z_']:
            if not self.align_z_state and prefix == 'align_z_':
                continue
            offset_mask = getattr(self, f'{prefix}offset_mask', None)
            deform_conv = getattr(self, f'{prefix}deform_conv', None)
            if offset_mask is not None:
                nn.init.constant_(offset_mask.weight, 0)
                nn.init.constant_(offset_mask.bias, 0)
            if deform_conv is not None:
                xavier_init(deform_conv, distribution='uniform')

    def init_weights(self):
        """Override: parent init + zero-init for alignment layers."""
        super().init_weights()
        self._init_alignment_weights()

    def _deform_align(self, state, feat, offset_mask_conv, deform_conv):
        """Align a state tensor to the current frame via deformable convolution.

        Args:
            state: (B, C, H, W) historical state to warp.
            feat: (B, in_channels, H, W) current BEV feature (reference).
            offset_mask_conv: Conv2d to predict offsets and mask.
            deform_conv: ModulatedDeformConv2d to apply the warp.

        Returns:
            aligned: (B, C, H, W) warped state.
        """
        concat = torch.cat([feat, state], dim=1)
        offset_and_mask = offset_mask_conv(concat)
        k2 = self.align_kernel_size * self.align_kernel_size
        o1 = 2 * self.align_deform_groups * k2
        offset = offset_and_mask[:, :o1, :, :]
        mask = offset_and_mask[:, o1:, :, :].sigmoid()
        return deform_conv(state, offset, mask)

    @auto_fp16(apply_to=['feat', 'velocity'])
    def forward(self, feat, velocity=None, use_posterior=True,
                deterministic=True, detach_state=True):
        """Process ONE frame through the motion-aligned RSSM.

        Same interface as BEVRSSMTemporalFusion.forward().
        The only difference: h_{t-1} and z_{t-1} are deformably aligned
        to the current feat before the ConvGRU transition.

        Returns:
            output, reconstruction, kl, h_t, z_t, stats
        """
        B, C, H, W = feat.shape

        # ---- velocity (skipped when use_action is False) ---------------
        if self.use_action:
            if velocity is None:
                velocity = torch.zeros(B, self.action_dim, device=feat.device)
            velocity_map = velocity[:, :, None, None].expand(B, self.action_dim, H, W)
        else:
            velocity_map = None

        # ---- state init -------------------------------------------------
        if self.h_state is None or self.h_state.shape[0] != B:
            self.h_state = torch.zeros(
                B, self.channels, H, W, device=feat.device, dtype=feat.dtype)
            self.z_state = torch.zeros(
                B, self.latent_dim, H, W, device=feat.device, dtype=feat.dtype)

        # ---- motion-aware alignment of historical state ----------------
        h_aligned = self._deform_align(
            self.h_state, feat,
            self.align_h_offset_mask, self.align_h_deform_conv)
        if self.align_z_state:
            z_aligned = self._deform_align(
                self.z_state, feat,
                self.align_z_offset_mask, self.align_z_deform_conv)
        else:
            z_aligned = self.z_state

        # ---- 1. Deterministic transition (uses aligned states) ---------
        if self.use_action:
            x = torch.cat([z_aligned, velocity_map], dim=1)
        else:
            x = z_aligned
        h_t = self.transition(x, h_aligned)

        # ---- 2. Prior ---------------------------------------------------
        mu_p = self.prior_mu(h_t)
        logstd_p = self.prior_logstd(h_t)

        # ---- 3. Encode observation --------------------------------------
        e_t = self.encoder(feat)

        # ---- 4. Posterior -----------------------------------------------
        mu_q = self.posterior_mu(torch.cat([h_t, e_t], dim=1))
        logstd_q = self.posterior_logstd(torch.cat([h_t, e_t], dim=1))

        # ---- 5. Select z_t ----------------------------------------------
        if use_posterior:
            z_t = mu_q if deterministic else self.sample(mu_q, logstd_q)
        else:
            z_t = mu_p if deterministic else self.sample(mu_p, logstd_p)

        # ---- 6. KL divergence -------------------------------------------
        kl, stats = self.kl_loss(mu_q, logstd_q, mu_p, logstd_p)

        # ---- 7. Reconstruction ------------------------------------------
        reconstruction = self.decoder(torch.cat([h_t, z_t], dim=1))

        # ---- 8. Output --------------------------------------------------
        output = self.output_proj(z_t)
        output = output + feat

        # ---- 9. Store state for next frame -------------------------------
        if detach_state:
            self.h_state = h_t.detach()
            self.z_state = z_t.detach()
        else:
            self.h_state = h_t
            self.z_state = z_t

        return output, reconstruction, kl * self.kl_scale, h_t, z_t, stats


@FUSION_LAYERS.register_module()
class DeterministicMotionAlignedLatentFusion(MotionAlignedRSSMFusion):
    """Motion-aligned recurrent fusion with a deterministic latent state.

    Keeps the same recurrent backbone as MotionAlignedRSSMFusion (h/z state,
    deformable h/z alignment, encoder, ConvGRU transition, posterior_mu,
    decoder/reconstruction branch, and output_proj + residual feat).
    Removes the stochastic components: prior_mu, prior_logstd,
    posterior_logstd, sampling, and the KL loss. Training and inference always
    use z_t = posterior_mu([h_t, e_t]).
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        latent_dim=None,
        hidden_dim=128,
        action_dim=0,
        posterior_noise_std=0.0,
        norm_cfg=dict(type='BN', requires_grad=True),
        act_cfg=dict(type='ReLU', inplace=True),
        align_kernel_size=3,
        align_deform_groups=1,
        align_z_state=True,
        init_cfg=None
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            kl_scale=1.0,
            free_nats=0.0,
            min_std=0.1,
            init_std=0.2,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            align_kernel_size=align_kernel_size,
            align_deform_groups=align_deform_groups,
            align_z_state=align_z_state,
            init_cfg=init_cfg
        )
        self.posterior_noise_std = posterior_noise_std

        # Unregister the stochastic branches. The parent constructor created
        # them in the standard RSSM layout; this class never touches them.
        self.prior_mu = None
        self.prior_logstd = None
        self.posterior_logstd = None

    def init_weights(self):
        """Init shared recurrent modules once, then only alignment on rebuilds.

        The parent constructor calls this method before the stochastic
        branches are unregistered. `build_model` calls it again after the
        removal, so the second pass must not touch the now-None prior layers.
        """
        if self.prior_mu is not None:
            super().init_weights()
        else:
            self._init_alignment_weights()

    @auto_fp16(apply_to=['feat', 'velocity'])
    def forward(self, feat, velocity=None, use_posterior=True,
                deterministic=True, detach_state=True):
        B, C, H, W = feat.shape

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
        if not deterministic and self.posterior_noise_std > 0:
            z_t = z_mu + self.posterior_noise_std * torch.randn_like(z_mu)
        else:
            z_t = z_mu

        reconstruction = self.decoder(torch.cat([h_t, z_t], dim=1))
        output = self.output_proj(z_t)
        output = output + feat

        if detach_state:
            self.h_state = h_t.detach()
            self.z_state = z_t.detach()
        else:
            self.h_state = h_t
            self.z_state = z_t

        return output, reconstruction, None, h_t, z_t, None


@FUSION_LAYERS.register_module()
class FixedNoisePosteriorLatentFusion(DeterministicMotionAlignedLatentFusion):
    """Posterior fusion with fixed Gaussian noise during training.

    Keeps the recurrent backbone (h/z state, deformable alignment, encoder,
    ConvGRU transition, posterior_mu, decoder/reconstruction, and output_proj +
    residual feat) and removes prior/prior_logstd/posterior_logstd/sampling/KL.
    Training draws z_t = posterior_mu + posterior_noise_std * N(0, I);
    inference uses z_t = posterior_mu.
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('posterior_noise_std', 0.1)
        super().__init__(*args, **kwargs)


@FUSION_LAYERS.register_module()
class PosteriorOnlyLearnableStdLatentFusion(MotionAlignedRSSMFusion):
    """Posterior-only latent fusion with a learnable conditional std.

    Keeps the same recurrent backbone as Deterministic/FixedNoise (h/z state,
    deformable h/z alignment, encoder, ConvGRU transition, posterior_mu,
    decoder/reconstruction, and output_proj + residual feat) while removing
    prior_mu, prior_logstd, and the KL loss. Training samples
    z_t = posterior_mu + eps * posterior_std; inference uses z_t = posterior_mu.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        latent_dim=None,
        hidden_dim=128,
        action_dim=0,
        min_std=0.1,
        init_std=0.2,
        norm_cfg=dict(type='BN', requires_grad=True),
        act_cfg=dict(type='ReLU', inplace=True),
        align_kernel_size=3,
        align_deform_groups=1,
        align_z_state=True,
        init_cfg=None
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            kl_scale=1.0,
            free_nats=0.0,
            min_std=min_std,
            init_std=init_std,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            align_kernel_size=align_kernel_size,
            align_deform_groups=align_deform_groups,
            align_z_state=align_z_state,
            init_cfg=init_cfg
        )

        # Unregister the prior-only branches. posterior_logstd is kept and
        # remains initialized by the same parent path as Deterministic/FixedNoise.
        self.prior_mu = None
        self.prior_logstd = None

    def init_weights(self):
        """First pass initializes the full parent; later rebuilds only alignment."""
        if self.prior_mu is not None:
            super().init_weights()
        else:
            self._init_alignment_weights()

    def _posterior_std_stats(self, logstd, max_samples=65536):
        """Estimate posterior std statistics from a deterministic subsample."""
        with torch.no_grad():
            flat = logstd.detach().flatten()

            if flat.numel() > max_samples:
                step = (flat.numel() + max_samples - 1) // max_samples
                flat = flat[::step][:max_samples]

            std = torch.exp(self._constrained_logstd(flat.float()))
            quantiles = torch.quantile(
                std,
                std.new_tensor([0.1, 0.5, 0.9]),
            )

            std_mean = std.mean()
            return dict(
                stat_posterior_std=std_mean,
                stat_posterior_std_mean=std_mean,
                stat_posterior_std_std=std.std(),
                stat_posterior_std_p10=quantiles[0],
                stat_posterior_std_p50=quantiles[1],
                stat_posterior_std_p90=quantiles[2],
            )

    @auto_fp16(apply_to=['feat', 'velocity'])
    def forward(self, feat, velocity=None, use_posterior=True,
                deterministic=True, detach_state=True):
        B, C, H, W = feat.shape

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
        mu_q = self.posterior_mu(torch.cat([h_t, e_t], dim=1))
        logstd_q = self.posterior_logstd(torch.cat([h_t, e_t], dim=1))
        z_t = mu_q if deterministic else self.sample(mu_q, logstd_q)
        stats = self._posterior_std_stats(logstd_q)

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


@FUSION_LAYERS.register_module()
class LowDimFutureConsistentLatentFusion(BaseModule):
    """Low-dimensional latent fusion with an exclusive next-frame task.

    The previous full-resolution RSSM used a pixel-wise 256-channel posterior
    that could be reconstructed from ``feat``/``h_t``.  Counterfactual
    diagnostics showed that replacing ``z_t`` with another sample's posterior
    mean changed the 3D-head output by only a few percent.

    This variant removes that redundancy:

    1. z is pooled to a small spatial grid (or a global vector);
    2. z modulates the recurrent state with residual FiLM plus a gate;
    3. during training z must predict the raw next-frame BEV feature.

    Two elements of the standard RSSM layout are required for z to stay
    informative and are kept here:

    * ``p(z_t | h_t)`` and ``q(z_t | h_t, e_t)``: the prior is a prediction
      from the deterministic state, the posterior corrects it with the current
      observation. Conditioning on ``z_prev`` instead would make the
      recurrence a pure z -> z chain and take ``h_t`` off the latent path.
    * an observation likelihood: ``decoder(h_t, z_t)`` reconstructs a pooled,
      normalized view of the current BEV feature. This is the only term that
      anchors ``z_t`` to the current observation; without it, free-bits can
      clamp the whole KL term and leave z with no gradient that requires it to
      carry observation information.

    The reconstruction is folded into the primary objective at index 2 with
    ``recon_loss_weight`` (index 1 stays ``None`` so the detector's legacy
    reconstruction path is untouched); ``stat_recon_mse`` exposes it for
    logging.

    The detector-facing six-tuple and state-reset interface are unchanged.
    """

    def __init__(
        self,
        in_channels,
        out_channels=None,
        latent_dim=32,
        hidden_dim=128,
        action_dim=0,
        kl_scale=1.0,
        free_nats=0.0,
        min_std=0.1,
        init_std=0.2,
        latent_pool='adaptive',
        latent_size=(16, 16),
        predict_future_channels=None,
        future_loss_weight=0.1,
        recon_loss_weight=0.1,
        discr_loss_weight=0.0,
        discr_margin=0.2,
        recon_target_mode='current',
        recon_input_mode='state_latent',
        direct_correction_readout=False,
        readout_mode='film',
        posterior_obs_mode='l2',
        posterior_obs_scale=0.1,
        z_proj_init='small',
        posterior_struct='standard',
        obs_pred_loss_weight=0.05,
        z_prev_normalize=True,
        latent_object_loss_weight=0.0,
        num_object_classes=4,
        object_head_hidden_dim=64,
        object_min_radius=1,
        modulation_scale=0.1,
        gate_init_bias=-1.0,
        norm_cfg=dict(type='BN', requires_grad=True),
        act_cfg=dict(type='ReLU', inplace=True),
        init_cfg=None,
    ):
        super().__init__(init_cfg)

        if out_channels is not None and out_channels != in_channels:
            raise ValueError(
                'LowDimFutureConsistentLatentFusion keeps the BEV channel '
                'count unchanged; use output_proj modulation instead of '
                'changing out_channels.')
        if latent_pool not in ('adaptive', 'global'):
            raise ValueError("latent_pool must be 'adaptive' or 'global'")

        self.channels = in_channels
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.use_action = action_dim > 0
        self.kl_scale = kl_scale
        self.free_nats = free_nats
        self.min_std = min_std
        self.min_logstd = math.log(min_std)
        self.init_std = init_std
        self.latent_pool = latent_pool
        self.latent_size = tuple(latent_size)
        self.predict_future_channels = (
            in_channels if predict_future_channels is None
            else predict_future_channels
        )
        self.future_loss_weight = future_loss_weight
        self.recon_loss_weight = recon_loss_weight
        if discr_loss_weight < 0:
            raise ValueError('discr_loss_weight must be >= 0')
        if discr_margin < 0:
            raise ValueError('discr_margin must be >= 0')
        self.discr_loss_weight = discr_loss_weight
        self.discr_margin = discr_margin
        if recon_target_mode not in ('current', 'temporal_delta'):
            raise ValueError(
                "recon_target_mode must be 'current' or 'temporal_delta'")
        if recon_input_mode not in ('state_latent', 'posterior_correction'):
            raise ValueError(
                "recon_input_mode must be 'state_latent' or "
                "'posterior_correction'")
        self.recon_target_mode = recon_target_mode
        self.recon_input_mode = recon_input_mode
        self.direct_correction_readout = direct_correction_readout
        if readout_mode not in ('film', 'spatial'):
            raise ValueError("readout_mode must be 'film' or 'spatial'")
        if readout_mode == 'spatial' and latent_pool == 'global':
            # The spatial readout exists to stop the correction being crushed
            # into a global vector; a global pool defeats that.
            raise ValueError(
                "readout_mode='spatial' requires latent_pool='adaptive'")
        self.readout_mode = readout_mode
        if posterior_obs_mode not in ('l2', 'scaled_raw'):
            raise ValueError(
                "posterior_obs_mode must be 'l2' or 'scaled_raw'")
        if posterior_obs_scale <= 0:
            raise ValueError('posterior_obs_scale must be > 0')
        self.posterior_obs_mode = posterior_obs_mode
        self.posterior_obs_scale = posterior_obs_scale
        if z_proj_init not in ('small', 'xavier'):
            raise ValueError("z_proj_init must be 'small' or 'xavier'")
        if z_proj_init == 'xavier' and readout_mode != 'spatial':
            raise ValueError(
                "z_proj_init='xavier' requires readout_mode='spatial'")
        self.z_proj_init = z_proj_init
        if posterior_struct not in ('standard', 'innovation'):
            raise ValueError(
                "posterior_struct must be 'standard' or 'innovation'")
        if obs_pred_loss_weight < 0:
            raise ValueError('obs_pred_loss_weight must be >= 0')
        self.posterior_struct = posterior_struct
        self.obs_pred_loss_weight = obs_pred_loss_weight
        if not isinstance(z_prev_normalize, bool):
            raise ValueError('z_prev_normalize must be a bool')
        # Route-B ep6 ablation switch. `normalize` reproduces the route-B and
        # every earlier lowdim run bit-for-bit; `raw` feeds the pooled previous
        # latent to the GRU unnormalized so its magnitude survives the
        # recurrence. Only the scale is at stake: both branches keep the
        # channel-wise direction.
        self.z_prev_normalize = z_prev_normalize
        if latent_object_loss_weight < 0:
            raise ValueError('latent_object_loss_weight must be >= 0')
        if num_object_classes < 1:
            raise ValueError('num_object_classes must be >= 1')
        if object_head_hidden_dim < 1:
            raise ValueError('object_head_hidden_dim must be >= 1')
        if object_min_radius < 0:
            raise ValueError('object_min_radius must be >= 0')
        # Route-C object-centric supervision. The auxiliary head consumes
        # only `posterior_correction`, so it cannot borrow the recurrent state
        # or the pooled observation and call that object evidence. It is
        # built unconditionally so a config can toggle the weight without
        # changing the state-dict layout; inference drops the branch entirely.
        self.latent_object_loss_weight = latent_object_loss_weight
        self.num_object_classes = num_object_classes
        self.object_head_hidden_dim = object_head_hidden_dim
        self.object_min_radius = object_min_radius
        if self.recon_loss_weight > 0 and (
                self.latent_pool == 'global' or min(self.latent_size) < 2):
            # The recon target is standardized across each sample, so a 1x1
            # grid collapses to a constant and would be trivially predictable.
            raise ValueError(
                'reconstruction needs a spatial latent target; use '
                "latent_pool='adaptive' with latent_size >= 2 in both dims")
        self.modulation_scale = modulation_scale
        self.gate_init_bias = gate_init_bias
        self.use_future_consistency = (
            self.predict_future_channels > 0 and self.future_loss_weight > 0)
        # Tells the detector that index 2 of the forward tuple is the complete
        # per-frame objective (KL + future consistency) rather than a bare KL
        # term that still needs the reconstruction loss paired with it.
        self.latent_loss_is_primary = True

        self.encoder = nn.Sequential(
            ConvModule(
                in_channels,
                hidden_dim,
                3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            ),
            ConvModule(
                hidden_dim,
                latent_dim,
                3,
                padding=1,
                norm_cfg=None,
                act_cfg=None,
            ),
        )
        self.latent_gru = ConvGRUCell(
            input_dim=latent_dim + (action_dim if self.use_action else 0),
            hidden_dim=latent_dim,
            kernel_size=3,
        )

        self.prior_mu = nn.Conv2d(latent_dim, latent_dim, 3, padding=1)
        self.prior_logstd = nn.Conv2d(latent_dim, latent_dim, 3, padding=1)
        self.posterior_mu = nn.Conv2d(
            2 * latent_dim, latent_dim, 3, padding=1)
        self.posterior_logstd = nn.Conv2d(
            2 * latent_dim, latent_dim, 3, padding=1)
        if self.posterior_struct == 'innovation':
            # Kalman-style predict/correct split: `obs_prior` predicts the
            # observation encoding from the deterministic state alone, and the
            # posterior head is re-interpreted as a *delta* on top of mu_p
            # (see forward). Only trained by the observation-prediction loss.
            self.obs_prior = nn.Conv2d(
                latent_dim, latent_dim, 3, padding=1)
        else:
            self.obs_prior = None

        decoder_in_channels = (
            2 * latent_dim
            if self.recon_input_mode == 'state_latent'
            else latent_dim
        )
        self.decoder = nn.Sequential(
            ConvModule(
                decoder_in_channels,
                hidden_dim,
                3,
                padding=1,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
            ),
            nn.Conv2d(hidden_dim, in_channels, 3, padding=1),
        )

        if self.readout_mode == 'spatial':
            # Both streams are projected at the latent grid and only then
            # upsampled, so the correction keeps its 16x16 spatial structure
            # all the way to the detector instead of being average-pooled into
            # a global vector and re-broadcast through a scalar gate.
            self.h_proj = nn.Conv2d(latent_dim, in_channels, 1)
            self.z_proj = nn.Conv2d(latent_dim, in_channels, 1)
            self.film_proj = None
            self.gate_proj = None
            self.output_proj = None
            self.correction_output_proj = None
        else:
            self.h_proj = None
            self.z_proj = None
            self.film_proj = nn.Linear(latent_dim, 2 * latent_dim)
            self.gate_proj = nn.Linear(2 * latent_dim, latent_dim)
            self.output_proj = nn.Conv2d(latent_dim, in_channels, 1)
            self.correction_output_proj = (
                nn.Conv2d(latent_dim, in_channels, 1)
                if self.direct_correction_readout else None
            )

        if self.use_future_consistency:
            self.prior_future = nn.Conv2d(
                latent_dim, 1, 1)
            self.posterior_future = nn.Conv2d(
                latent_dim, 1, 1)
        else:
            self.prior_future = None
            self.posterior_future = None

        # Route-C auxiliary branch: correction -> class center heatmap. Kept
        # off the detection path (the forward output never reads it) so
        # removing it at inference is free.
        self.object_head = nn.Sequential(
            nn.Conv2d(
                latent_dim, object_head_hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(object_head_hidden_dim, num_object_classes, 1),
        )

        if self.latent_pool == 'adaptive':
            self.latent_pool_layer = nn.AdaptiveAvgPool2d(self.latent_size)
        else:
            self.latent_pool_layer = nn.AdaptiveAvgPool2d(1)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        if self.use_future_consistency:
            self.future_target_pool = nn.AdaptiveAvgPool2d(self.latent_size)

        self.h_state = None
        self.z_state = None
        # Prediction made at frame t, consumed at frame t+1 against that
        # frame's BEV. Holding it across the two calls is what makes the
        # exclusive task forward-looking instead of a copy of the input.
        self.pending_future_pred = None
        self.pending_future_valid = None
        self.prev_recon_target = None
        self.prev_recon_valid = None
        # Set by the detector for the current frame only; history frames must
        # not contribute object supervision (their GT is not the prediction
        # target of the current correction).
        self.current_object_target = None
        self.init_weights()

    def init_weights(self):
        super().init_weights()
        xavier_modules = [
            self.prior_mu,
            self.prior_logstd,
            self.posterior_mu,
            self.posterior_logstd,
            self.decoder[0].conv,
            self.decoder[1],
        ]
        if self.readout_mode == 'spatial':
            xavier_modules += [self.h_proj, self.z_proj]
        else:
            xavier_modules += [
                self.film_proj, self.gate_proj, self.output_proj]
        if self.obs_prior is not None:
            xavier_modules.append(self.obs_prior)
        for module in xavier_modules:
            xavier_init(module, distribution='uniform')
        if self.correction_output_proj is not None:
            xavier_init(
                self.correction_output_proj, distribution='uniform')
        for module in (
            self.latent_gru.reset_conv,
            self.latent_gru.update_conv,
            self.latent_gru.candidate_conv,
        ):
            xavier_init(module, distribution='uniform')
            if module.bias is not None:
                nn.init.uniform_(module.bias, -0.1, 0.1)
        bias_value = self.min_logstd + math.log(
            self.init_std / self.min_std - 1.0)
        nn.init.constant_(self.prior_logstd.bias, bias_value)
        nn.init.constant_(self.posterior_logstd.bias, bias_value)
        if self.readout_mode == 'spatial':
            # h_proj stays small so the detection residual starts near
            # identity. z_proj follows `z_proj_init`: 'small' keeps the
            # historical std=0.01, 'xavier' restores the same scale the other
            # 1x1/3x3 projections in this class receive (for a 32->256 1x1
            # this is ~8-10x larger than std=0.01).
            nn.init.normal_(self.h_proj.weight, std=0.01)
            nn.init.zeros_(self.h_proj.bias)
            if self.z_proj_init == 'xavier':
                xavier_init(self.z_proj, distribution='uniform')
                nn.init.zeros_(self.z_proj.bias)
            else:
                nn.init.normal_(self.z_proj.weight, std=0.01)
                nn.init.zeros_(self.z_proj.bias)
        else:
            nn.init.zeros_(self.film_proj.bias)
            nn.init.zeros_(self.gate_proj.bias)
            self.gate_proj.bias.data.fill_(self.gate_init_bias)
            # Keep the detection residual near identity at init, while
            # retaining a non-zero gradient path from the low-dimensional
            # latent.
            nn.init.normal_(self.output_proj.weight, std=0.01)
            nn.init.zeros_(self.output_proj.bias)
            if self.correction_output_proj is not None:
                nn.init.normal_(self.correction_output_proj.weight, std=0.01)
                nn.init.zeros_(self.correction_output_proj.bias)
        if self.use_future_consistency:
            xavier_init(self.prior_future, distribution='uniform')
            xavier_init(self.posterior_future, distribution='uniform')
        # Initialised last so adding the route-C branch does not shift the
        # RNG consumption (and therefore the weights) of any pre-existing
        # module, keeping old checkpoints and ablations bit-comparable.
        xavier_init(self.object_head[0], distribution='uniform')
        xavier_init(self.object_head[2], distribution='uniform')
        # CenterNet init: start at low objectness so the focal term does not
        # open with a huge negative-gradient spike on background pixels.
        nn.init.constant_(self.object_head[2].bias, -2.19)

    def reset_state(self):
        self.h_state = None
        self.z_state = None
        self.pending_future_pred = None
        self.pending_future_valid = None
        self.prev_recon_target = None
        self.prev_recon_valid = None
        self.current_object_target = None

    def set_current_object_target(self, target):
        """Attach the current-frame objectness heatmap for route-C training.

        ``target`` is ``(B, num_object_classes, H, W)`` on any device; it is
        moved to the fusion device at loss time. The detector must call this
        immediately before the current-frame forward and clear it afterwards
        (``reset_state`` also clears it) so history frames never train the
        auxiliary head against the wrong frame's GT.
        """
        if target is None:
            self.current_object_target = None
            return
        if target.dim() != 4:
            raise ValueError(
                'object target must be (B, C, H, W), got shape '
                f'{tuple(target.shape)}')
        if target.shape[1] != self.num_object_classes:
            raise ValueError(
                'object target channel mismatch: expected '
                f'{self.num_object_classes}, got {target.shape[1]}')
        self.current_object_target = target

    def reset_for_samples(self, mask):
        if mask.any() and self.h_state is not None:
            keep = ~mask
            self.h_state = torch.where(
                keep[:, None, None, None], self.h_state,
                torch.zeros_like(self.h_state))
            self.z_state = torch.where(
                keep[:, None, None, None], self.z_state,
                torch.zeros_like(self.z_state))
            if self.pending_future_valid is not None:
                # Invalid samples must be dropped from the next frame's
                # future-consistency mean instead of being scored against a
                # stale target.
                self.pending_future_valid = self.pending_future_valid & keep
            if self.prev_recon_valid is not None:
                self.prev_recon_valid = self.prev_recon_valid & keep

    def _pool_latent(self, value):
        return self.latent_pool_layer(value)

    def _normalized_recon_target(self, feat):
        target = self._pool_latent(feat).detach()
        target_mean = target.mean(dim=(1, 2, 3), keepdim=True)
        target_std = target.std(
            dim=(1, 2, 3), keepdim=True).clamp_min(1e-4)
        return (target - target_mean) / target_std

    def _posterior_observation(self, e_pooled_raw):
        """Prepare the observation encoding fed to the posterior.

        ``l2`` is the historical behaviour: per-sample channel-wise L2
        normalization. Its denominator is dominated by the batch-common
        component (measured: 1.90 RMS common vs 0.53 sample-specific), so it
        divides away most of the sample identity along with the scale.

        ``scaled_raw`` applies one global scalar instead. A single shared
        factor rescales the magnitude without touching the relative
        differences between samples, which is what the probes showed is being
        lost.
        """
        if self.posterior_obs_mode == 'scaled_raw':
            return e_pooled_raw * self.posterior_obs_scale
        return F.normalize(e_pooled_raw, dim=1)

    def _constrained_logstd(self, logstd):
        logstd = self.min_logstd + F.softplus(logstd - self.min_logstd)
        return torch.clamp(logstd, max=0.0)

    def _object_heatmap_loss(self, posterior_correction):
        """CenterNet focal loss on the correction-derived class heatmap.

        The head is fed ``posterior_correction`` only: allowing it to see
        ``h_t`` would let the recurrent state solve the target and remove the
        pressure on the correction. Returns ``(loss, recall, num_pos)``; the
        loss is a zero scalar when disabled or when no current-frame target
        was attached so the objective stays exactly unchanged in that case.
        Returns ``(loss, hit1, num_pos, thr50)``, where ``hit1`` is the
        per-occupied-channel argmax hit rate and ``thr50`` is the fraction of
        GT centres whose probability exceeds 0.5.
        """
        zero = torch.zeros(
            (), device=posterior_correction.device,
            dtype=posterior_correction.dtype)
        if (not self.training or self.latent_object_loss_weight <= 0
                or self.current_object_target is None):
            return zero, zero, zero, zero
        target = self.current_object_target.to(
            device=posterior_correction.device, dtype=torch.float32)
        if target.shape[0] != posterior_correction.shape[0]:
            raise ValueError(
                'object target batch mismatch: expected '
                f'{posterior_correction.shape[0]}, got {target.shape[0]}')
        if target.shape[-2:] != posterior_correction.shape[-2:]:
            raise ValueError(
                'object target spatial mismatch: expected '
                f'{tuple(posterior_correction.shape[-2:])}, got '
                f'{tuple(target.shape[-2:])}')
        # The head emits logits and the loss is evaluated in logit space.
        # `logsigmoid` is the numerically stable log(sigmoid(.)), so a
        # saturated logit keeps a finite gradient instead of the flat plateau
        # that probability-space clamping (clamp(p, 1e-4, 1-1e-4)) creates.
        logits = self.object_head(
            posterior_correction.to(torch.float32)).float()
        prob = logits.sigmoid()
        pos = target.eq(1.0).float()
        neg = 1.0 - target
        neg_weights = neg.pow(4)
        pos_loss = -F.logsigmoid(logits) * (1.0 - prob).pow(2) * pos
        neg_loss = (
            -F.logsigmoid(-logits) * prob.pow(2) * neg_weights * neg)
        num_pos = pos.sum()
        # Match the reference CenterNet behaviour: an image with no object
        # must contribute no gradient instead of being normalised by a
        # clamped 1.0 denominator, which would blow the term up ~200x.
        if num_pos.item() == 0:
            return (zero, zero, num_pos.detach(), zero)
        loss = (pos_loss.sum() + neg_loss.sum()) / num_pos
        # A CenterNet head is trained with a Gaussian soft target and is not
        # driven to exceed 0.5 at every centre early in training, so a fixed
        # 0.5 threshold reads 0 for a long time without saying whether the
        # map carries object evidence. Hit@1 is the meaningful diagnostic:
        # in each *occupied* class channel, the argmax cell counts as a hit
        # when it is one of that class's GT centres. Background-only classes
        # are excluded, otherwise their arbitrary uniform argmax would be
        # scored against every centre. The 0.5 threshold is reported too.
        flat = prob.flatten(2)
        peak_idx = flat.argmax(dim=2)
        peak = torch.zeros_like(flat).scatter_(
            2, peak_idx[:, :, None], 1.0).view_as(prob)
        hits = (peak * pos).sum()
        occupied = (pos.sum(dim=(2, 3)) > 0).sum().clamp_min(1.0)
        recall = hits / occupied
        thr_hits = ((prob > 0.5) & pos.bool()).sum().float() / num_pos
        return (
            loss.to(posterior_correction.dtype),
            recall.detach().to(posterior_correction.dtype),
            num_pos.detach().to(posterior_correction.dtype),
            thr_hits.detach().to(posterior_correction.dtype),
        )

    def sample(self, mu, logstd):
        logstd = self._constrained_logstd(logstd)
        std = torch.exp(logstd)
        return mu + std * torch.randn_like(std)

    def kl_loss(self, mu_q, logstd_q, mu_p, logstd_p):
        logstd_q = self._constrained_logstd(logstd_q)
        logstd_p = self._constrained_logstd(logstd_p)
        var_q = torch.exp(2.0 * logstd_q)
        var_p = torch.exp(2.0 * logstd_p)
        kl_raw = (
            logstd_p - logstd_q
            + (var_q + (mu_q - mu_p).square()) / (2.0 * var_p)
            - 0.5
        )
        if self.free_nats > 0:
            kl_effective = torch.clamp(kl_raw, min=self.free_nats)
        else:
            kl_effective = kl_raw
        stats = dict(
            stat_kl_raw_mean=kl_raw.mean().detach(),
            stat_kl_effective_mean=kl_effective.mean().detach(),
            stat_clamped_ratio=(
                (kl_raw < self.free_nats).float().mean().detach()
                if self.free_nats > 0 else torch.zeros((), device=kl_raw.device)
            ),
            stat_mu_diff_sq=(mu_q - mu_p).square().mean().detach(),
            stat_posterior_std=torch.exp(logstd_q).mean().detach(),
            stat_prior_std=torch.exp(logstd_p).mean().detach(),
        )
        # kl_effective is already a per-element mean over
        # (batch, latent_dim, latent_h, latent_w); do NOT divide by the
        # element count again.
        return kl_effective.mean(), stats

    @auto_fp16(apply_to=['feat', 'velocity'])
    def forward(self, feat, velocity=None, use_posterior=True,
                deterministic=True, detach_state=True):
        B, C, H, W = feat.shape
        if self.use_action:
            if velocity is None:
                velocity = torch.zeros(
                    B, self.action_dim, device=feat.device)
            velocity_map = velocity[:, :, None, None].expand(
                B, self.action_dim, H, W)
        else:
            velocity_map = None

        if self.h_state is None or self.h_state.shape[0] != B:
            self.h_state = torch.zeros(
                B, self.latent_dim, *self.latent_size,
                device=feat.device, dtype=feat.dtype)
            self.z_state = torch.zeros(
                B, self.latent_dim, *self.latent_size,
                device=feat.device, dtype=feat.dtype)

        pooled_z = self._pool_latent(self.z_state)
        if self.z_prev_normalize:
            z_prev = F.normalize(pooled_z, dim=1)
        else:
            z_prev = pooled_z
        if self.use_action:
            x = torch.cat([z_prev, velocity_map], dim=1)
        else:
            x = z_prev
        h_t = self._pool_latent(self.h_state)
        h_t = self.latent_gru(x, h_t)

        e_t = self.encoder(feat)
        e_pooled = self._posterior_observation(self._pool_latent(e_t))
        # Standard RSSM: the prior predicts from the deterministic state and
        # the posterior corrects it with the observation encoding.
        mu_p = self.prior_mu(h_t)
        logstd_p = self.prior_logstd(h_t)
        delta_mu = None
        obs_pred = torch.zeros((), device=feat.device, dtype=feat.dtype)
        innovation_sq = torch.zeros((), device=feat.device, dtype=feat.dtype)
        if self.posterior_struct == 'innovation':
            # Kalman-style predict/correct. `obs_prior` predicts the current
            # observation encoding from h_t alone and is trained *only* by the
            # observation-prediction term below. stopgrad on e_hat_t keeps it
            # out of every other gradient path, so the posterior sees purely
            # the part of the observation that h_t could not predict.
            e_hat_t = self.obs_prior(h_t)
            obs_pred = F.smooth_l1_loss(e_hat_t, e_pooled.detach())
            innovation = e_pooled - e_hat_t.detach()
            innovation_sq = innovation.square().mean().detach()
            posterior_input = torch.cat([h_t, innovation], dim=1)
            # The posterior head is re-interpreted as a delta on top of the
            # prior mean, so the correction is the model's own prediction
            # rather than the difference of two batch-common vectors.
            delta_mu = self.posterior_mu(posterior_input)
            mu_q = mu_p + delta_mu
        else:
            posterior_input = torch.cat([h_t, e_pooled], dim=1)
            mu_q = self.posterior_mu(posterior_input)
        logstd_q = self.posterior_logstd(posterior_input)

        if use_posterior:
            z_t = mu_q if deterministic else self.sample(mu_q, logstd_q)
        else:
            z_t = mu_p if deterministic else self.sample(mu_p, logstd_p)
        if use_posterior:
            posterior_correction = (
                delta_mu if delta_mu is not None
                else mu_q - mu_p.detach())
        else:
            posterior_correction = torch.zeros_like(mu_p)

        kl, stats = self.kl_loss(mu_q, logstd_q, mu_p, logstd_p)

        target = self._normalized_recon_target(feat)
        decoder_input = (
            torch.cat([h_t, z_t], dim=1)
            if self.recon_input_mode == 'state_latent'
            else posterior_correction
        )
        reconstruction = self.decoder(decoder_input)
        recon_loss = torch.zeros(
            (), device=feat.device, dtype=feat.dtype)
        has_recon_target = False
        recon_valid_frac = torch.zeros(
            (), device=feat.device, dtype=feat.dtype)
        delta_rms = torch.zeros(
            (), device=feat.device, dtype=feat.dtype)
        discr_loss = torch.zeros(
            (), device=feat.device, dtype=feat.dtype)
        discr_positive = torch.zeros(
            (), device=feat.device, dtype=feat.dtype)
        discr_negative = torch.zeros(
            (), device=feat.device, dtype=feat.dtype)
        if self.recon_target_mode == 'current':
            recon_loss = F.mse_loss(reconstruction, target)
            has_recon_target = True
            recon_valid_frac = torch.ones_like(recon_valid_frac)
        elif self.prev_recon_target is not None:
            valid = self.prev_recon_valid
            if valid is None:
                valid = torch.ones(
                    B, dtype=torch.bool, device=feat.device)
            if valid.any():
                delta_target = target - self.prev_recon_target
                delta_rms = delta_target.square().mean().sqrt().detach()
                delta_mean = delta_target.mean(
                    dim=(1, 2, 3), keepdim=True)
                delta_std = delta_target.std(
                    dim=(1, 2, 3), keepdim=True).clamp_min(1e-4)
                delta_target = (delta_target - delta_mean) / delta_std
                valid_map = valid[:, None, None, None].to(
                    target.dtype).expand_as(target)
                denom = valid_map.sum().clamp_min(1.0)
                recon_loss = (
                    (reconstruction - delta_target).square() *
                    valid_map
                ).sum() / denom
                has_recon_target = True
                recon_valid_frac = valid.float().mean().detach()

                # Discrimination: this target must be reconstructed better by
                # its own correction than by another sample's. Without it, a
                # near-common correction can satisfy the reconstruction term
                # (measured: swapping samples moved recon by -0.09%..+9.07%),
                # so the objective never asks z_t to identify its own sample.
                if (self.discr_loss_weight > 0 and B >= 2
                        and valid.sum() >= 2):
                    rolled_valid = torch.roll(valid, 1, dims=0)
                    pair_valid = valid & rolled_valid
                    if pair_valid.any():
                        correction_roll = torch.roll(
                            posterior_correction, 1, dims=0)
                        recon_roll = self.decoder(correction_roll)
                        pair_map = pair_valid[:, None, None, None].to(
                            target.dtype).expand_as(target)
                        pair_denom = pair_map.sum().clamp_min(1.0)
                        discr_positive = (
                            (reconstruction - delta_target).square()
                            * pair_map
                        ).sum() / pair_denom
                        discr_negative = (
                            (recon_roll - delta_target).square()
                            * pair_map
                        ).sum() / pair_denom
                        discr_loss = F.relu(
                            self.discr_margin
                            + discr_positive
                            - discr_negative).mean()
        if self.recon_target_mode == 'temporal_delta':
            self.prev_recon_target = target
            self.prev_recon_valid = torch.ones(
                B, dtype=torch.bool, device=feat.device)
        stats['stat_recon_mse'] = recon_loss.detach()
        stats['stat_recon_valid_frac'] = recon_valid_frac
        stats['stat_recon_delta_rms'] = delta_rms
        stats['stat_obs_pred'] = obs_pred.detach()
        stats['stat_innovation_sq'] = innovation_sq
        stats['stat_correction_sq'] = posterior_correction.square().mean(
        ).detach()
        stats['stat_discr_positive'] = discr_positive.detach()
        stats['stat_discr_negative'] = discr_negative.detach()
        stats['stat_discr_gap'] = (
            discr_negative - discr_positive).detach()

        future_loss = torch.zeros((), device=feat.device, dtype=feat.dtype)
        has_future_target = False
        if self.training and self.use_future_consistency:
            # Predict the NEXT frame's BEV from the latent of the CURRENT
            # frame. Consuming the stored prediction one frame later is what
            # keeps the task forward-looking; scoring z_t against the frame it
            # was just encoded from would be trivially solvable from e_t.
            z_t_future_input = F.normalize(z_t, dim=1)
            mu_p_future_input = F.normalize(mu_p, dim=1)
            posterior_future_pred = self.posterior_future(z_t_future_input)
            prior_future_pred = self.prior_future(mu_p_future_input)

            if self.pending_future_pred is not None:
                valid = self.pending_future_valid
                if valid is None:
                    valid = torch.ones(
                        feat.shape[0], dtype=torch.bool, device=feat.device)
                if valid.any():
                    target = feat.detach()
                    target = target.square().mean(dim=1, keepdim=True).sqrt()
                    target = self.future_target_pool(target)
                    target_mean = target.mean(dim=(2, 3), keepdim=True)
                    target_std = target.std(
                        dim=(2, 3), keepdim=True).clamp_min(1e-4)
                    target = (target - target_mean) / target_std
                    pending_post, pending_prior = self.pending_future_pred
                    # Broadcast to the target's full shape before summing:
                    # `valid[:, None, None, None]` is (B, 1, 1, 1), so summing
                    # it directly would count samples instead of
                    # (sample, spatial) elements and inflate the mean by the
                    # number of latent cells.
                    valid_map = valid[:, None, None, None].to(
                        target.dtype).expand_as(target)
                    denom = valid_map.sum().clamp_min(1.0)
                    future_loss = (
                        ((pending_post - target).square() * valid_map).sum()
                        + ((pending_prior - target).square() * valid_map).sum()
                    ) / (2.0 * denom)
                    has_future_target = True

            self.pending_future_pred = (
                posterior_future_pred, prior_future_pred)
            self.pending_future_valid = torch.ones(
                feat.shape[0], dtype=torch.bool, device=feat.device)
        else:
            self.pending_future_pred = None
            self.pending_future_valid = None

        if self.readout_mode == 'spatial':
            # Both streams stay on the latent grid, so a sample-specific
            # correction reaches the detector as a spatial residual instead of
            # being pooled into a global vector and squashed through a gate.
            latent_residual = (
                self.h_proj(h_t) + self.z_proj(posterior_correction))
        else:
            z_global = self.global_pool(z_t).flatten(1)
            h_global = self.global_pool(h_t).flatten(1)
            film = self.film_proj(z_global)
            gamma = 1.0 + self.modulation_scale * torch.tanh(
                film[:, :self.latent_dim])
            beta = film[:, self.latent_dim:]
            gate = torch.sigmoid(self.gate_proj(
                torch.cat([z_global, h_global], dim=1)))
            h_modulated = (
                gamma[:, :, None, None] * h_t + beta[:, :, None, None])
            h_gated = (1.0 - gate[:, :, None, None]) * h_t + \
                gate[:, :, None, None] * h_modulated
            latent_residual = self.output_proj(h_gated)
            if self.correction_output_proj is not None:
                latent_residual = \
                    latent_residual + self.correction_output_proj(
                        posterior_correction)
        if latent_residual.shape[-2:] != feat.shape[-2:]:
            latent_residual = F.interpolate(
                latent_residual,
                size=feat.shape[-2:],
                mode='bilinear',
                align_corners=False)
        output = feat + latent_residual

        if detach_state:
            self.h_state = h_t.detach()
            self.z_state = z_t.detach()
        else:
            self.h_state = h_t
            self.z_state = z_t

        latent_loss = kl * self.kl_scale
        if self.obs_pred_loss_weight > 0 and self.posterior_struct == 'innovation':
            latent_loss = latent_loss + self.obs_pred_loss_weight * obs_pred
        (object_hm_loss, object_hm_recall, object_hm_num_pos,
         object_hm_thr50) = (
            self._object_heatmap_loss(posterior_correction))
        if self.latent_object_loss_weight > 0:
            latent_loss = (
                latent_loss
                + self.latent_object_loss_weight * object_hm_loss)
        if self.recon_loss_weight > 0 and has_recon_target:
            latent_loss = latent_loss + self.recon_loss_weight * recon_loss
        if (self.discr_loss_weight > 0 and has_recon_target
                and self.recon_target_mode == 'temporal_delta'):
            latent_loss = latent_loss + self.discr_loss_weight * discr_loss
        if has_future_target:
            latent_loss = latent_loss + self.future_loss_weight * future_loss
        # NOTE: the key must not contain the substring 'loss'. mmdet's
        # _parse_losses() sums every log_var whose name contains 'loss' into
        # the reported total, so a 'stat_*loss*' name would silently inflate
        # (or, for negative values, deflate) the training loss. Keep this as a
        # pure diagnostic.
        stats['stat_future_mse'] = future_loss.detach()
        # Name deliberately avoids the substring 'loss': mmdet's
        # _parse_losses() would otherwise fold this diagnostic into the
        # reported total.
        stats['stat_object_hm'] = object_hm_loss.detach()
        stats['stat_object_hm_recall'] = object_hm_recall
        stats['stat_object_hm_pos'] = object_hm_num_pos
        stats['stat_object_hm_thr50'] = object_hm_thr50
        # Index 1 stays None: the reconstruction here is pooled/normalized and
        # its MSE is already folded into index 2 with recon_loss_weight, so
        # the detector's legacy full-resolution recon path must not double-count
        # it (nor compare a 16x16 map against the full BEV grid).
        return output, None, latent_loss, h_t, z_t, stats


@FUSION_LAYERS.register_module()
class MotionAlignedInnovationRSSMFusion(MotionAlignedRSSMFusion):
    """Full-resolution motion-aligned RSSM with an innovation-conditioned posterior.

    This is the main-line successor to route B. Route B's innovation posterior
    was validated on the *low-dim* (16x16x32) family, where it fixed the
    predict/correct semantics but could not recover AP: route C ended at
    ep8-12 35.0289 versus the no2d_igdr baseline 38.8906 (-3.8617). The
    low-dim replacement had also removed three main-line capabilities at once
    (deformable motion alignment, native 216x248 resolution, and the
    full-resolution readout), so that gap cannot be attributed to the
    posterior alone.

    This class keeps the entire MotionAlignedRSSMFusion backbone intact --
    encoder, ConvGRU `transition`, deformable h/z alignment, prior, logstd
    heads, decoder, output_proj residual readout, state management, and the
    legacy KL + reconstruction losses -- and changes only how the posterior
    mean is produced:

        standard:   mu_q = posterior_mu([h_t, e_t])
        innovation: e_hat = obs_prior(h_t)
                    innovation = e_t - stopgrad(e_hat)
                    delta = posterior_mu([h_t, innovation])
                    mu_q = mu_p + delta

    `obs_prior` is trained only by the observation-prediction term
    (smooth L1 against a stop-gradient of the encoder output), so the
    posterior conditions on the part of the current observation that the
    recurrent state could not already predict. `delta` is the model's own
    correction rather than the difference of two batch-common vectors.

    Every added parameter is created after `super().__init__()`, so all
    pre-existing modules keep their initialization RNG order and existing
    checkpoints remain bitwise loadable for the shared keys.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        latent_dim=None,
        hidden_dim=64,
        action_dim=2,
        kl_scale=1.0,
        free_nats=0.0,
        min_std=0.1,
        init_std=0.2,
        obs_pred_loss_weight=0.05,
        norm_cfg=dict(type='BN', requires_grad=True),
        act_cfg=dict(type='ReLU', inplace=True),
        align_kernel_size=3,
        align_deform_groups=1,
        align_z_state=True,
        init_cfg=None
    ):
        if obs_pred_loss_weight < 0:
            raise ValueError('obs_pred_loss_weight must be >= 0')
        self.obs_pred_loss_weight = obs_pred_loss_weight
        self.posterior_struct = 'innovation'

        # Parent builds encoder / transition / prior / posterior / decoder /
        # output_proj / alignment and runs init_weights().
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            kl_scale=kl_scale,
            free_nats=free_nats,
            min_std=min_std,
            init_std=init_std,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            align_kernel_size=align_kernel_size,
            align_deform_groups=align_deform_groups,
            align_z_state=align_z_state,
            init_cfg=init_cfg,
        )

        # Added AFTER the parent's init_weights(): obs_prior must not perturb
        # the RNG stream consumed by the inherited modules.
        self.obs_prior = nn.Conv2d(
            out_channels, self.latent_dim, 3, padding=1)
        xavier_init(self.obs_prior, distribution='uniform')

    def forward(self, feat, velocity=None, use_posterior=True,
                deterministic=True, detach_state=True):
        """Same interface as MotionAlignedRSSMFusion.forward().

        Returns:
            output, reconstruction, latent_loss, h_t, z_t, stats
        """
        B, C, H, W = feat.shape

        if self.use_action:
            if velocity is None:
                velocity = torch.zeros(
                    B, self.action_dim, device=feat.device)
            velocity_map = velocity[:, :, None, None].expand(
                B, self.action_dim, H, W)
        else:
            velocity_map = None

        if self.h_state is None or self.h_state.shape[0] != B:
            self.h_state = torch.zeros(
                B, self.channels, H, W, device=feat.device, dtype=feat.dtype)
            self.z_state = torch.zeros(
                B, self.latent_dim, H, W, device=feat.device, dtype=feat.dtype)

        # ---- motion-aware alignment of historical state ----------------
        h_aligned = self._deform_align(
            self.h_state, feat,
            self.align_h_offset_mask, self.align_h_deform_conv)
        if self.align_z_state:
            z_aligned = self._deform_align(
                self.z_state, feat,
                self.align_z_offset_mask, self.align_z_deform_conv)
        else:
            z_aligned = self.z_state

        # ---- deterministic transition -----------------------------------
        if self.use_action:
            x = torch.cat([z_aligned, velocity_map], dim=1)
        else:
            x = z_aligned
        h_t = self.transition(x, h_aligned)

        # ---- prior ------------------------------------------------------
        mu_p = self.prior_mu(h_t)
        logstd_p = self.prior_logstd(h_t)

        # ---- observation encoding (native resolution) -------------------
        e_t = self.encoder(feat)

        # ---- innovation-conditioned posterior ---------------------------
        e_hat_t = self.obs_prior(h_t)
        obs_pred = F.smooth_l1_loss(e_hat_t, e_t.detach())
        innovation = e_t - e_hat_t.detach()
        posterior_input = torch.cat([h_t, innovation], dim=1)
        delta_mu = self.posterior_mu(posterior_input)
        mu_q = mu_p + delta_mu
        logstd_q = self.posterior_logstd(posterior_input)

        # ---- select z_t -------------------------------------------------
        if use_posterior:
            z_t = mu_q if deterministic else self.sample(mu_q, logstd_q)
        else:
            z_t = mu_p if deterministic else self.sample(mu_p, logstd_p)

        # ---- KL ---------------------------------------------------------
        kl, stats = self.kl_loss(mu_q, logstd_q, mu_p, logstd_p)

        # ---- reconstruction ---------------------------------------------
        reconstruction = self.decoder(torch.cat([h_t, z_t], dim=1))

        # ---- output -----------------------------------------------------
        output = self.output_proj(z_t)
        output = output + feat

        # ---- state ------------------------------------------------------
        if detach_state:
            self.h_state = h_t.detach()
            self.z_state = z_t.detach()
        else:
            self.h_state = h_t
            self.z_state = z_t

        latent_loss = kl * self.kl_scale
        if self.obs_pred_loss_weight > 0:
            latent_loss = latent_loss + self.obs_pred_loss_weight * obs_pred

        stats['stat_obs_pred'] = obs_pred.detach()
        stats['stat_innovation_sq'] = innovation.square().mean().detach()

        # The detector consumes index 1 as a full-resolution reconstruction
        # for its own MSE term; keep it so the legacy recon path keeps working
        # exactly as for MotionAlignedRSSMFusion.
        return output, reconstruction, latent_loss, h_t, z_t, stats
