"""ConcatConvFusion subclass that stashes its inputs on ModalityContext.

The fused-BEV computation is *identical* to the baseline
``ConcatConvFusion`` (super().forward, same parameters, same pretrained
keys ``cross_attention.fusion_conv.*``). The only addition is that the two
un-fused modality BEVs are published on the ``ModalityContext`` bus so the
Motion-Evidence RSSM can build modality-specific observations and
confidence maps downstream.

Stashing (not returning) keeps the ``R4Det.extract_feat`` call site
untouched:

    bev_feats = self.cross_attention(img_bev_feats, pts_bev_feats)   # unchanged
    ...
    self.temporal_fusion(bev_feats, ...)                             # unchanged
"""

from mmdet3d.models.builder import FUSION_LAYERS
from mmdet3d.models.fusion_layers.concat_conv_fusion import ConcatConvFusion

from .modality_bus import ModalityContext

__all__ = ['ConcatConvFusionStash']


@FUSION_LAYERS.register_module()
class ConcatConvFusionStash(ConcatConvFusion):
    """Baseline concat+conv fusion + modality publication on the bus."""

    def forward(self, img_bev, radar_bev):
        ModalityContext.set('camera_bev', img_bev)
        ModalityContext.set('radar_bev', radar_bev)
        return super().forward(img_bev, radar_bev)
