"""Reliability-gated camera/radar BEV fusion (single-variable arm vs ConcatConv).

Pre-registered in docs/rssm_reliability_fusion_prereg_20261012.md.

Motivation (evidence-backed, see the prereg): the historical Cross_Modal_Fusion
swap (ep12-16 window, seed0) lost Overall -1.30, Car -2.08 and Ped -5.24 vs the
plain concat baseline, while Truck +1.76 / Cyc +0.35 -- i.e. replacing concat
with an unguarded attention-style module is not free, and that module carried
neither a reliability estimate nor any missing-modality handling.

This arm changes exactly ONE thing relative to ``ConcatConvFusion``: the radar
half of the concatenation is scaled by a per-cell reliability gate ``alpha``
that is a function of input-visible evidence only (radar point occupancy from
the input point cloud + the radar feature magnitude statistics). The fusion
convolution, its shape and its weights are untouched::

    occ   = occupancy(points)                       # (B, 1, H, W), GT-free
    alpha = 1 + tanh(gate([occ, |radar|_mean_c, |radar|_max_c]))   # (B, 1, H, W)
    out   = fusion_conv(cat([img_bev, alpha * radar_bev], dim=1))

Properties frozen by the prereg:

* ``alpha`` is zero-initialised (last conv layer) so ``alpha == 1`` exactly at
  step 0 -> the arm is **bitwise identical** to the baseline before training.
* ``alpha in (0, 2)``: the gate may attenuate the radar branch when the return
  evidence is weak, and may amplify it up to 2x when it is strong.
* The camera tensor enters the concatenation unchanged -- the camera path is
  never attenuated by this module.
* No new loss, no KL change, no semantic head, no Doppler gate.
* ``obs_points=None`` reproduces the baseline exactly, so any call path that
  does not opt in keeps the old behaviour bit for bit.
"""
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet3d.models.builder import FUSION_LAYERS
from mmdet3d.models.fusion_layers.concat_conv_fusion import ConcatConvFusion

from .validity_gain_rssm import build_radar_validity_mask

DEFAULT_PC_RANGE = [0, -39.68, -4, 69.12, 39.68, 2]


@FUSION_LAYERS.register_module()
class ReliabilityGatedConcatFusion(ConcatConvFusion):
    """ConcatConv fusion with an input-evidence reliability gate on radar.

    Args:
        img_channels (int): Image BEV channels (default 256).
        rad_channels (int): Radar BEV channels (default 384).
        out_channels (int): Fused channels (default 256).
        kernel_size (int): Fusion conv kernel (default 3) -- unchanged vs
            baseline so the parameter budget stays matched.
        point_cloud_range (list[float]): BEV extent used to build the occupancy
            evidence from the input points. Must match the detector's grid.
        gate_hidden (int): Hidden width of the gate MLP (default 16).
        gate_kernel (int): Kernel of the gate MLP (default 3).
        valid_layout (str): 'yx' for the pillar-scatter/radar-feature layout
            (default; the mask multiplies the radar tensor), 'xy' for the
            fused/BEV layout used by the RSSM call site.
    """

    accepts_obs_valid = True

    def __init__(self,
                 img_channels=256,
                 rad_channels=384,
                 out_channels=256,
                 kernel_size=3,
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                 act_cfg=dict(type='ReLU', inplace=True),
                 point_cloud_range=None,
                 gate_hidden=16,
                 gate_kernel=3,
                 valid_layout='yx'):
        super(ReliabilityGatedConcatFusion, self).__init__(
            img_channels=img_channels,
            rad_channels=rad_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        if valid_layout not in ('yx', 'xy'):
            raise ValueError("valid_layout must be 'yx' or 'xy', got %r"
                             % (valid_layout,))
        self.point_cloud_range = list(
            DEFAULT_PC_RANGE if point_cloud_range is None else point_cloud_range)
        self.valid_layout = valid_layout
        pad = gate_kernel // 2
        # Evidence channels: occupancy + radar |.| channel-mean and channel-max.
        self.gate = nn.Sequential(
            nn.Conv2d(3, gate_hidden, gate_kernel, padding=pad),
            nn.ReLU(inplace=True),
            nn.Conv2d(gate_hidden, 1, gate_kernel, padding=pad),
        )
        # Zero-init the last layer: alpha = 1 + tanh(0) = 1 -> baseline exactly.
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    # ------------------------------------------------------------------ mask
    def build_occupancy(self, obs_points, radar_bev):
        """(B, 1, H, W) radar-return occupancy at the radar feature grid."""
        pts = obs_points if isinstance(obs_points, (list, tuple)) else [obs_points]
        bsz, _, height, width = radar_bev.shape
        return build_radar_validity_mask(
            pts, bsz, height, width, self.point_cloud_range,
            layout=self.valid_layout, device=radar_bev.device,
            dtype=radar_bev.dtype)

    def evidence(self, radar_bev, obs_points):
        occ = self.build_occupancy(obs_points, radar_bev)
        mag = radar_bev.abs()
        return torch.cat([occ, mag.mean(dim=1, keepdim=True),
                          mag.amax(dim=1, keepdim=True)], dim=1)

    def reliability_gate(self, radar_bev, obs_points):
        """alpha in (0, 2); identically 1.0 with the frozen zero-init."""
        return 1.0 + torch.tanh(self.gate(self.evidence(radar_bev, obs_points)))

    # --------------------------------------------------------------- forward
    def forward(self, img_bev, radar_bev, obs_points=None):
        """Fuse (img_bev, radar_bev); ``obs_points=None`` == baseline exactly."""
        if obs_points is None:
            alpha = torch.ones_like(radar_bev[:, :1])
        else:
            alpha = self.reliability_gate(radar_bev, obs_points)
        return self.fusion_conv(torch.cat([img_bev, alpha * radar_bev], dim=1))
