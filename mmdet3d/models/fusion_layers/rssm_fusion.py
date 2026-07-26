import torch
import torch.nn as nn
import torch.nn.functional as F

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
    """True RSSM for BEV temporal fusion — per-frame recurrent processing.

    Each forward call processes ONE frame. The model maintains internal
    deterministic state h and stochastic state z across calls:

        h_t = ConvGRU(h_{t-1}, z_{t-1}, action)    ← recurrent transition
        prior:     p(z_t | h_t)
        posterior: q(z_t | h_t, encoder(feat))
        z_t ~ q (train) / ~ p (inference)
        KL = KL(q || p)

    h_0 is initialized to zeros (NOT from any observation).

    Args:
        in_channels (int): Input BEV feature channels.
        out_channels (int): Output BEV feature channels.
        kernel_size (int): Conv kernel size for GRU cell.
        latent_dim (int | None): Stochastic state dim (defaults to in_channels).
        hidden_dim (int): Encoder hidden dimension.
        action_dim (int): Action (velocity) dimension.
        kl_scale (float): KL loss weight.
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

        ################################################
        # ConvGRU: h_{t-1} + z_{t-1} + action → h_t
        ################################################
        gru_input_dim = latent_dim + action_dim  # z_{t-1} + action
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
        # Output: h + z → out_channels
        ################################################
        self.output_layer = ConvModule(
            out_channels + latent_dim,
            out_channels,
            3,
            padding=1,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )

        # Internal recurrent state (per-batch, maintained across forward calls)
        self.h_state = None
        self.z_state = None

        self.init_weights()

    def init_weights(self):
        super().init_weights()

        for m in [self.prior_mu, self.prior_logstd,
                  self.posterior_mu, self.posterior_logstd]:
            if hasattr(m, 'weight'):
                xavier_init(m, distribution='uniform')

        for m in [self.transition.reset_conv,
                  self.transition.update_conv,
                  self.transition.candidate_conv]:
            if hasattr(m, 'weight'):
                xavier_init(m, distribution='uniform')
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.uniform_(m.bias, -0.1, 0.1)

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

    @staticmethod
    def sample(mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    @staticmethod
    def kl_loss(mu_q, logstd_q, mu_p, logstd_p):
        var_q = torch.exp(2.0 * logstd_q)
        var_p = torch.exp(2.0 * logstd_p)

        kl = (
            logstd_p - logstd_q
            + (var_q + (mu_q - mu_p) ** 2) / (2.0 * var_p)
            - 0.5
        )
        return kl.mean()

    @auto_fp16(apply_to=['feat', 'velocity'])
    def forward(self, feat, velocity=None):
        """Process ONE frame through the RSSM.

        Uses internal h_{t-1}, z_{t-1} from the previous call.
        Stores h_t, z_t for the next call.

        Args:
            feat: Current BEV feature (B, C, H, W).
            velocity: Ego-motion (B, action_dim). Defaults to zeros.

        Returns:
            output: Fused BEV feature (B, out_channels, H, W).
            reconstruction: Reconstructed feat (B, in_channels, H, W).
            kl: KL divergence loss (scalar).
            h_t: Deterministic state (B, out_channels, H, W).
            z_t: Stochastic state (B, latent_dim, H, W).
        """
        B, C, H, W = feat.shape

        ################################################
        # Velocity / action
        ################################################
        if velocity is None:
            velocity = torch.zeros(
                B, self.action_dim, device=feat.device
            )

        velocity_map = velocity[:, :, None, None].expand(
            B, self.action_dim, H, W
        )

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
        # 1. Deterministic transition: h_t = ConvGRU(h_{t-1}, z_{t-1}, action)
        ################################################
        x = torch.cat([self.z_state, velocity_map], dim=1)
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
        # 5. Sample z_t (posterior during training, prior during inference)
        ################################################
        if self.training:
            z_t = self.sample(mu_q, logstd_q)
        else:
            z_t = self.sample(mu_p, logstd_p)

        ################################################
        # 6. KL divergence
        ################################################
        kl = self.kl_loss(mu_q, logstd_q, mu_p, logstd_p)

        ################################################
        # 7. Reconstruction: decoder(h_t, z_t) → feat
        ################################################
        reconstruction = self.decoder(
            torch.cat([h_t, z_t], dim=1)
        )

        ################################################
        # 8. Output: output_layer(h_t, z_t) → fused feature
        ################################################
        output = self.output_layer(
            torch.cat([h_t, z_t], dim=1)
        )

        ################################################
        # 9. Store state for next frame (detach — no grad across timesteps)
        ################################################
        self.h_state = h_t.detach()
        self.z_state = z_t.detach()

        return output, reconstruction, kl * self.kl_scale, h_t, z_t