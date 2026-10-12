# Reliability-gated camera/radar BEV fusion (front-end single-variable arm).
# Pre-registration (FROZEN before the first run):
#   docs/rssm_reliability_fusion_prereg_20261012.md
#
# Single variable vs ..._no2d_igdr (the standard mainline): ONLY the RCFusion
# type changes, from ``ConcatConvFusion`` to ``ReliabilityGatedConcatFusion``:
#   occ   = radar-point occupancy of the *input* point cloud (GT-free)
#   alpha = 1 + tanh(gate([occ, |radar|_mean_c, |radar|_max_c]))   # in (0,2)
#   out   = fusion_conv(cat([img_bev, alpha * radar_bev]))         # same conv
# The fusion convolution (640->256, k=3) is byte-identical in shape to the
# baseline's; the gate adds 593 parameters (+0.04%) and is zero-initialised, so
# at step 0 alpha == 1 and the arm is bitwise identical to the baseline.
#
# Everything else (backbone, head, temporal_fusion RSSM, KL schedule, data
# pipeline, schedule) is inherited unchanged.
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
        type='ReliabilityGatedConcatFusion',
        img_channels=256,
        rad_channels=384,
        out_channels=256,
        kernel_size=3,
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        act_cfg=dict(type='ReLU', inplace=True),
        # --- arm-frozen inputs (prereg §2) ---------------------------------
        point_cloud_range=[0, -39.68, -4, 69.12, 39.68, 2],
        # PointPillarsScatter lays the canvas out as rows=y, cols=x
        # (coors are b,z,y,x -> canvas.view(ny, nx)), so the occupancy mask must
        # use the 'yx' convention to align with the radar tensor it scales.
        valid_layout='yx',
        gate_hidden=16,
        gate_kernel=3,
    ),
)
