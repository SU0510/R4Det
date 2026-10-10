# Candidate A: z-semantic supervision + detection cls-logit residual readout,
# on the standard no2d_igdr main line. Prepared 2026-10-10.
# Pre-registration: docs/rssm_candidate_A_prereg_20261010.md
#   (numeric gates in its §5 are PROPOSED and need user approval; the 450-iter
#    smoke needs none and makes no performance claim).
#
# Why this config exists
# ----------------------
# The stage-0 target audit (§70) killed the "reconstruction target is 95%
# shared template" story that was going to justify a target reform: the real
# target is 56.5% sample-specific (§70.6), and a frame-difference target sits
# exactly on the (S-1)/S pure-noise floor. What the audit did establish is a
# supervision/readout break, measured on the mainline:
#   * recon = 0.18% of the total loss from ep4 on (§70.2) — no pressure on z;
#   * KL fully absorbed by free-bits, mu_q -> mu_p (mu_diff^2 -99.5%) (§70.2);
#   * zero_z moves the head 38.02% but shuffle_z only 4.67% (§49) — z is a
#     sample-common bias; and shuffle_z is ALREADY foreground-concentrated
#     (fg/bg_hi 4.3x, §69), so the readout's spatial attention is not missing
#     either. What is missing is *class-discriminative content* in z and a path
#     that turns it into a class decision.
#
# What changes relative to ..._no2d_igdr (the direct base) — exactly two, both
# on the supervision/readout axis:
#   1. sem_head: z-exclusive centre-heatmap head, CenterNet focal loss against
#      the CURRENT frame's GT centres -> loss_rssm_sem. h_t is not an input, so
#      the only way to reduce it is to put class evidence into z (unit T3).
#   2. z_cls_logits: zero-init 1x1 conv on z_t -> 4 class logits, broadcast over
#      the anchor slots of each class and added to conv_cls inside Anchor3DHead.
#      Geometry (conv_reg) and conv_dir_cls never see it.
# The reconstruction target, posterior, prior, KL/free-nats, output_proj,
# backbone, detector head geometry, optimiser, schedule, data pipeline and seed
# protocol are byte-identical to the base — this is NOT a DC-family run and it
# carries no evidence field, no L1 ego-warp and no difference target.
#
# Zero-init contract (prereg §2, unit tests T1/T2): with W_cls == 0 the residual
# is exactly 0, so a run hot-started from checkpoints/pretrained_tj4d.pth — or a
# probe loading the baseline epoch_14.pth — starts bit-identical to the
# baseline detector. Ablations are single flags:
#   sem_supervision_weight=0  -> readout only   (C2)
#   cls_residual=False        -> supervision only (C1)
#   sem_head_input='h'|'feat' -> control arms    (C3 / M2)
#
# Launch (same protocol as the baseline):
#   CUDA_VISIBLE_DEVICES=5,6,7 bash tools/dist_train.sh <this config> 3 \
#     --seed 0 --deterministic \
#     --work-dir /data/lurui/work_dirs/fgfull_N4_no2d_igdr_2x4_24e_semA_seed0
# Smoke first (no val, no AP claim):
#   CUDA_VISIBLE_DEVICES=5,6,7 bash tools/dist_train.sh \
#     me_rssm/sanity/semreadout_smoke_450iter.py 3 --seed 0 --deterministic \
#     --work-dir /data/lurui/work_dirs/semA_smoke450_seed0

import os
import sys

# Bootstrap: make the repo root importable so `me_rssm` resolves no matter
# which working directory mmcv's temp-dir config import uses.
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

# Register me_rssm modules at config-exec time: tools/test_vod.py (the eval
# entry) does not process custom_imports.
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
        type='SemanticReadoutRSSMFusion',
        # --- baseline RSSM args, identical to ..._no2d_igdr ---------------
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
        # --- candidate A --------------------------------------------------
        sem_num_classes=4,        # Pedestrian, Cyclist, Car, Truck
        sem_head_input='z',       # the arm; 'h' / 'feat' are the controls
        sem_head_hidden=64,
        sem_supervision_weight=1.0,
        # 0.32 m/BEV cell: 2 cells ~ 0.64 m, i.e. no object is thinner than the
        # target's core. Radius otherwise grows with the box extent.
        object_min_radius=2,
        cls_residual=True,        # z -> conv_cls additive class evidence
        cls_residual_gate=1.0,    # probes set 0.0 for the paired toggle
        log_sem_maps=False,
    ),
)
