# ---------------------------------------------------------------------------
# ME-RSSM mainline config: Motion-Evidence RSSM for Camera + 4D Radar BEV.
#
# Inherits the clean mainline (N=4, 24e, pretrained, head-v2) and swaps in
# four new modules, all defined in the me_rssm package (imported via
# custom_imports). The original baseline config/classes remain untouched;
# set the type names back to the baseline values to reproduce it exactly.
#
# Differences vs. the inherited mainline:
#   pts_voxel_encoder : RadarPillarFeatureNet -> RadarPillarFeatureNetMotion
#                       (same parameters, adds raw Doppler stat outputs;
#                       pretrained keys load unchanged)
#   pts_middle_encoder: PointPillarsScatter -> PointPillarsScatterMotion
#                       (same scatter + publishes the velocity-evidence
#                       canvas on the ModalityContext bus)
#   RCFusion          : ConcatConvFusion -> ConcatConvFusionStash
#                       (identical fusion, stashes cam/radar BEVs; the
#                       pretrained cross_attention.* keys still load)
#   temporal_fusion   : MotionAlignedRSSMFusion -> MotionEvidenceRSSMFusion
#                       (Doppler-conditioned alignment offsets, dynamic-
#                       gated GRU update, modality observations, Kalman-
#                       like reliability gain; reduces exactly to the
#                       baseline when the bus is empty)
#   pts_bbox_head     : shared_stem=False  -- clean-baseline parity (the
#                       inherited config still carries the un-reverted
#                       shared_stem=True from the closed section-41
#                       experiment; audit U7).
# ---------------------------------------------------------------------------

import os
import sys

# Bootstrap: make the repo root importable so `me_rssm` resolves no matter
# which working directory mmcv's temp-dir config import uses.
try:
    _cfg_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:  # pragma: no cover - very old mmcv without __file__
    _cfg_dir = os.path.join(os.getcwd(), 'me_rssm', 'configs')
_repo_root = os.path.abspath(os.path.join(_cfg_dir, '..', '..'))
if not os.path.isdir(os.path.join(_repo_root, 'me_rssm')):
    _p = os.getcwd()
    for _ in range(5):
        if os.path.isdir(os.path.join(_p, 'me_rssm')):
            _repo_root = _p
            break
        _p = os.path.dirname(_p)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# Import at config-exec time, not only through custom_imports:
# tools/test_vod.py (the evaluation entry) does not process
# custom_imports, so registration must happen while the config loads.
# For train_vod.py this is simply an idempotent re-import.
import me_rssm  # noqa: E402,F401  (registers all me_rssm modules)

_base_ = [
    '../../configs/r4det/'
    'TJ4D-R4Det_motion_align_rssm_det3d_N4_2x4_24e_pretrained_v2_head.py'
]

custom_imports = dict(imports=[
    'mmdet3d',
    'mmdet3d.core.hook.kl_scale_scheduler',
    'me_rssm',
])

model = dict(
    pts_voxel_encoder=dict(
        type='RadarPillarFeatureNetMotion',
    ),
    pts_middle_encoder=dict(
        type='PointPillarsScatterMotion',
    ),
    RCFusion=dict(
        type='ConcatConvFusionStash',
    ),
    temporal_fusion=dict(
        type='MotionEvidenceRSSMFusion',
        # --- new-architecture switches (identity-start by design) -------
        motion_channels=7,
        camera_bev_channels=256,
        motion_hidden=32,
        motion_latent=32,
        camera_hidden=128,
        camera_latent=64,
        use_motion_offsets=True,
        use_reliability_gain=True,
        use_dynamic_gate=True,
        use_modality_obs=True,
        motion_grad=False,
        modality_dropout=0.0,
        gain_init_bias=4.0,
        conf_init_bias=2.0,
        dyn_init_bias=-2.0,
    ),
    pts_bbox_head=dict(
        shared_stem=False,
    ),
)
