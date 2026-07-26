import torch
import torch.nn as nn
from mmdet3d.models.builder import FUSION_LAYERS
from mmcv.cnn import ConvModule


@FUSION_LAYERS.register_module()
class ConcatConvFusion(nn.Module):
    """Simple concatenation + convolution fusion for camera and radar BEV features.

    Args:
        img_channels (int): Channels of image BEV features.
        rad_channels (int): Channels of radar BEV features.
        out_channels (int): Output channels after fusion.
        kernel_size (int): Convolution kernel size. Default: 3.
        norm_cfg (dict): Normalization config. Default: BN.
        act_cfg (dict): Activation config. Default: ReLU.
    """

    def __init__(self,
                 img_channels=256,
                 rad_channels=384,
                 out_channels=256,
                 kernel_size=3,
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                 act_cfg=dict(type='ReLU', inplace=True)):
        super(ConcatConvFusion, self).__init__()
        padding = kernel_size // 2
        self.fusion_conv = ConvModule(
            img_channels + rad_channels,
            out_channels,
            kernel_size,
            padding=padding,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            inplace=False)

    def forward(self, img_bev, radar_bev):
        """Forward pass.

        Args:
            img_bev (torch.Tensor): Image BEV features. (B, img_channels, H, W)
            radar_bev (torch.Tensor): Radar BEV features. (B, rad_channels, H, W)

        Returns:
            torch.Tensor: Fused BEV features. (B, out_channels, H, W)
        """
        return self.fusion_conv(torch.cat([img_bev, radar_bev], dim=1))