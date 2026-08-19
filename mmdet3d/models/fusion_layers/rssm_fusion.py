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
            self.h_state[mask] = 0.0
            self.z_state[mask] = 0.0

    def sample(self, mu, logstd):
        """Sample from distribution with smooth minimum std constraint."""
        # Smooth lower bound on logstd via softplus: guarantees std >= min_std
        logstd = self.min_logstd + F.softplus(logstd - self.min_logstd)
        # Upper bound: std ≤ 1.0 to prevent variance explosion
        logstd = torch.clamp(logstd, max=0.0)
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
