# Geometry-paired multi-scale fusion (GP-MSF), front-end single-variable arm.
# Pre-registration (FROZEN before the first run):
#   docs/gpmsf_prereg_20261013.md
#
# Single variable vs ..._no2d_igdr (the standard mainline): ONLY the RCFusion
# type changes, from ``ConcatConvFusion`` to ``GeometryPairedMultiScaleFusion``.
# The fusion keeps the same 640->256 k=3 conv on cat([camera, radar]) and adds a
# zero-initialised, geometry-grounded evidence path on top of it:
#   C' = C + valid_R * W_RtoC(M_R); R' = R + valid_I * W_ItoR(M_I)
#   out = conv([C', R']); validity gates learned biases as well as weights.
# where M_R / M_I come from bounded metric addresses that read the *native*
# radar levels (0.32/0.64/1.28 m) and P2/P3/P4 through three train-only height
# references.  W_RtoC / W_ItoR are zero at step 0, so the arm is bitwise
# identical to the baseline before the first update; the 450-iter smoke writes
# evidence that they leave zero (the geometry path really ran).
#
# Everything else (backbone, head, temporal_fusion RSSM, KL schedule, data
# pipeline, schedule) is inherited unchanged.  Frozen geometry below is copied
# from the resolved mainline config (grid_config / pts_backbone / img_neck).
import os
import sys

try:
    _cfg_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:  # pragma: no cover
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

import me_rssm  # noqa: E402,F401

_base_ = (
    '../../configs/r4det/'
    'TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr.py'
)

custom_imports = dict(imports=[
    'mmdet3d',
    'mmdet3d.core.hook.kl_scale_scheduler',
    'me_rssm',
])

model = dict(
    RCFusion=dict(
        # --- the resolved mainline args, unchanged -------------------------
        type='GeometryPairedMultiScaleFusion',
        img_channels=256,
        rad_channels=384,
        out_channels=256,
        kernel_size=3,
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        act_cfg=dict(type='ReLU', inplace=True),
        # --- arm-frozen BEV canvas (resolved grid_config) ------------------
        # fusion input layout is (B, C, 248, 216) = rows y / cols x
        bev_shape=(248, 216),
        bev_origin=(0.0, -39.68),      # xbound[0], ybound[0]
        bev_cell=0.32,
        # --- arm-frozen native radar geometry ------------------------------
        # PointPillarsScatter canvas: 496 rows (y) x 432 cols (x) at 0.16 m;
        # pixel-0 *centre* is (0 + 0.16/2, -39.68 + 0.16/2).
        radar_origin=(0.08, -39.60),
        radar_base_cell=0.16,
        radar_base_shape=(496, 432),
        radar_level_channels=(64, 128, 256),   # SECOND out_channels
        radar_level_steps=(0.32, 0.64, 1.28),  # k3/p1/s2 twice
        # --- arm-frozen image geometry -------------------------------------
        image_channels=256,                    # FPN out_channels
        image_size=(480, 640),                 # ida_aug_conf['final_dim']
        image_level_strides=(4, 8, 16),        # P2/P3/P4 of num_outs=5 FPN
        depth_downsample=8,                    # GeometryDepth_Net downsample
        # --- arm-frozen fusion capacity ------------------------------------
        embed_dim=64,
        num_heads=4,
        num_offsets=4,
        height_refs=(-1.1230436265, -0.3003779650, 0.6601446867),  # train-only
        gamma_init=1.0,
        null_score_init=0.0,
        query_chunk=4096,
        grad_checkpoint=True,
        new_param_seed=20261013,
        # Train/eval must fail loudly if the detector ever stops passing `geo`;
        # a silent unwired arm would just be the concat baseline.
        require_geo=True,
    ),
)
