"""ME-RSSM: Motion-Evidence RSSM for Camera + 4D Radar BEV perception.

Importing this package registers all new modules with the mmdet3d
registries so they can be selected purely through config files:

    model = dict(
        pts_voxel_encoder  = dict(type='RadarPillarFeatureNetMotion', ...),
        pts_middle_encoder = dict(type='PointPillarsScatterMotion', ...),
        RCFusion           = dict(type='ConcatConvFusionStash', ...),
        temporal_fusion    = dict(type='MotionEvidenceRSSMFusion', ...))

Nothing in the existing mmdet3d tree is modified; the unmodified
``R4Det`` detector consumes these modules through their unchanged
interfaces. See me_rssm/docs/ for the full research documentation.
"""

from .modality_bus import ModalityContext
from .radar_motion_chain import (MOTION_STAT_CHANNELS,
                                 PointPillarsScatterMotion,
                                 RadarPillarFeatureNetMotion)
from .stash_fusion import ConcatConvFusionStash
from .motion_evidence_rssm import MotionEvidenceRSSMFusion

__all__ = [
    'ModalityContext',
    'MOTION_STAT_CHANNELS',
    'RadarPillarFeatureNetMotion',
    'PointPillarsScatterMotion',
    'ConcatConvFusionStash',
    'MotionEvidenceRSSMFusion',
]
