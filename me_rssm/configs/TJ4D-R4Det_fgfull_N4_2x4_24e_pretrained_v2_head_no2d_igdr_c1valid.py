# Candidate C, minimal arm 1 (C1): observation-validity-conditioned update.
# Pre-registration (FROZEN before the first run):
#   docs/rssm_c_validity_update_prereg_20261012.md
#
# Single variable vs ..._no2d_igdr (the standard full-resolution
# MotionAlignedRSSMFusion mainline): the fusion type adds ONE gain
#   g_t = 1 + tanh(W_v [h_t ; valid_t] + b_v),  z_t = mu_q + (g_t-1)(mu_q-mu_p)
# with valid_t built ONLY from the input point cloud (GT-free). All inherited
# args below are byte-for-byte the resolved mainline args; W_v/b_v are
# zero-init, so at init the arm is bitwise identical to the mainline.
#
# Launch (identical protocol to the mainline seed0):
#   CUDA_VISIBLE_DEVICES=5,6,7 bash tools/dist_train.sh <this config> 3 \
#     --seed 0 --deterministic \
#     --work-dir /data/lurui/work_dirs/fgfull_N4_no2d_igdr_2x4_24e_c1valid_seed0
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
    temporal_fusion=dict(
        type='ValidityGatedRSSMFusion',
        # --- the resolved mainline args, unchanged -------------------------
        in_channels=256,
        out_channels=256,
        kernel_size=3,
        latent_dim=256,
        hidden_dim=128,
        action_dim=0,
        kl_scale=1.0,
        free_nats=1.0,
        min_std=0.1,
        init_std=0.2,
        align_kernel_size=3,
        align_deform_groups=1,
        align_z_state=True,
        norm_cfg=dict(type='BN', requires_grad=True),
        act_cfg=dict(type='ReLU', inplace=True),
        # --- C1 frozen inputs (prereg §2) ---------------------------------
        point_cloud_range=[0, -39.68, -4, 69.12, 39.68, 2],
        valid_layout='xy',
    ),
)
