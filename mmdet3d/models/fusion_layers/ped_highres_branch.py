import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmdet3d.models.builder import FUSION_LAYERS


@FUSION_LAYERS.register_module()
class PedHighresBranch(nn.Module):
    """Ped-only high-resolution radar branch.

    The radar scatter is already on the doubled grid (0.16 m), while the
    frozen fused BEV is on the original grid (0.32 m). This branch does not
    alter either contract; it only produces features for the Ped head.
    """

    def __init__(self,
                 radar_channels=64,
                 fused_channels=256,
                 out_channels=256,
                 channels=64,
                 num_convs=2,
                 kernel_size=3,
                 norm_cfg=dict(type='BN2d'),
                 act_cfg=dict(type='ReLU', inplace=True)):
        super().__init__()
        self.radar_conv = ConvModule(
            radar_channels,
            channels,
            kernel_size,
            padding=kernel_size // 2,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)
        convs = []
        for _ in range(num_convs - 1):
            convs.append(
                ConvModule(
                    channels,
                    channels,
                    kernel_size,
                    padding=kernel_size // 2,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
        self.convs = nn.Sequential(*convs)
        self.fusion_conv = nn.Conv2d(
            channels + fused_channels,
            out_channels,
            kernel_size,
            padding=kernel_size // 2,
            bias=True)
        nn.init.zeros_(self.fusion_conv.weight)
        nn.init.zeros_(self.fusion_conv.bias)

    def forward(self, highres_radar, lowres_fused):
        highres = self.convs(self.radar_conv(highres_radar))
        high_h, high_w = highres.shape[-2:]
        base = F.interpolate(
            lowres_fused,
            size=(high_h, high_w),
            mode='nearest')
        residual = self.fusion_conv(torch.cat((highres, base), dim=1))
        return base + residual
