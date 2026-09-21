import torch
import torch.nn as nn
from mmcv.ops import ModulatedDeformConv2d
from mmdet3d.models.builder import FUSION_LAYERS
from mmcv.runner import BaseModule, auto_fp16
from mmcv.cnn import ConvModule, build_norm_layer, build_activation_layer, bias_init_with_prob, xavier_init
import warnings

@FUSION_LAYERS.register_module()
class TemporalDeformableFusion(BaseModule):

    @auto_fp16()
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 deform_groups: int = 1,
                 gate_kernel_size: int = 1,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True),
                 init_cfg=None):

        super(TemporalDeformableFusion, self).__init__(init_cfg)

        if in_channels != out_channels:
            out_channels = in_channels

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = (kernel_size - 1) // 2
        self.deform_groups = deform_groups
        self.gate_kernel_size = gate_kernel_size
        self.gate_padding = gate_kernel_size // 2
        self.offset_mask_generator = nn.Conv2d(
            in_channels=2 * in_channels,
            out_channels=3 * deform_groups * kernel_size * kernel_size,
            kernel_size=kernel_size,
            padding=self.padding,
            stride=1
        )

        self.deform_conv = ModulatedDeformConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=self.padding,
            stride=1,
            deform_groups=deform_groups,
            bias=False
        )

        conv_in_channels = in_channels * 2
        self.conv_z = ConvModule(
            conv_in_channels,
            out_channels,
            kernel_size=gate_kernel_size,
            padding=self.gate_padding,
            act_cfg=None,
            norm_cfg=norm_cfg)

        self.conv_r = ConvModule(
            conv_in_channels,
            out_channels,
            kernel_size=gate_kernel_size,
            padding=self.gate_padding,
            act_cfg=None,
            norm_cfg=norm_cfg)

        self.conv_h = ConvModule(
            conv_in_channels,
            out_channels,
            kernel_size=gate_kernel_size,
            padding=self.gate_padding,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg)

        self.output_layer = ConvModule(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg)

        self.init_weights()

    def init_weights(self):
        super(TemporalDeformableFusion, self).init_weights()

        if hasattr(self.offset_mask_generator, 'weight'):
            nn.init.constant_(self.offset_mask_generator.weight, 0)
        if hasattr(self.offset_mask_generator, 'bias') and self.offset_mask_generator.bias is not None:
            nn.init.constant_(self.offset_mask_generator.bias, 0)

        if hasattr(self.deform_conv, 'weight'):
            xavier_init(self.deform_conv, distribution='uniform')

        for m in [self.conv_z, self.conv_r, self.conv_h, self.output_layer]:
            if hasattr(m, 'conv'):
                xavier_init(m.conv, distribution='uniform')

    @auto_fp16(apply_to=('feat_curr', 'feat_prev'))
    def forward(self, feat_curr: torch.Tensor, feat_prev: torch.Tensor) -> torch.Tensor:

        concat_feat_for_offset = torch.cat([feat_curr, feat_prev], dim=1)

        offset_and_mask = self.offset_mask_generator(concat_feat_for_offset)
        k2 = self.kernel_size * self.kernel_size
        offset = offset_and_mask[:, :2 * self.deform_groups * k2, :, :]
        mask = offset_and_mask[:, 2 * self.deform_groups * k2:, :, :].sigmoid()
        h_t_minus_1_aligned = self.deform_conv(feat_prev, offset, mask)
        concat_feat_for_gate = torch.cat([feat_curr, h_t_minus_1_aligned], dim=1)
        z_t = torch.sigmoid(self.conv_z(concat_feat_for_gate))
        r_t = torch.sigmoid(self.conv_r(concat_feat_for_gate))
        h_tilde_t = self.conv_h(torch.cat([feat_curr, r_t * h_t_minus_1_aligned], dim=1))
        h_t = (1 - z_t) * feat_curr + z_t * h_tilde_t
        output = self.output_layer(h_t)
        return output


@FUSION_LAYERS.register_module()
class TemporalDeformableFusionBaseline(TemporalDeformableFusion):
    """RSSM-interface adapter for the original TemporalDeformableFusion.

    The current R4Det temporal loop expects a six-item return:

        output, reconstruction, kl, h_t, z_t, stats

    The original one-step GRU baseline only returns ``output``. This subclass
    keeps that implementation unchanged while exposing the same interface.
    It stores the previous *raw* BEV feature, exactly the input the original
    baseline consumed, and applies the deformable GRU only on the current
    frame. No reconstruction or KL terms are produced. Subclassing (instead
    of wrapping) keeps the pretrained checkpoint keys under
    ``temporal_fusion.*`` compatible with the original GRU baseline weights.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 deform_groups: int = 1,
                 gate_kernel_size: int = 1,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='ReLU', inplace=True),
                 init_cfg=None):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            deform_groups=deform_groups,
            gate_kernel_size=gate_kernel_size,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            init_cfg=init_cfg)
        self.prev_feat = None

    def reset_state(self):
        self.prev_feat = None

    def reset_for_samples(self, mask):
        if self.prev_feat is not None and mask.any():
            keep = ~mask
            self.prev_feat = torch.where(
                keep[:, None, None, None], self.prev_feat,
                torch.zeros_like(self.prev_feat))

    @auto_fp16(apply_to=('feat', 'velocity'))
    def forward(self, feat, velocity=None, use_posterior=True,
                deterministic=True, detach_state=True):
        if self.prev_feat is None or self.prev_feat.shape != feat.shape:
            self.prev_feat = feat.detach() if detach_state else feat
            return feat, None, None, None, None, None

        output = super().forward(feat, self.prev_feat)
        self.prev_feat = feat.detach() if detach_state else feat
        return output, None, None, None, None, None

