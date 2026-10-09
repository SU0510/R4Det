# DC-RSSM-2: + L1 analytic ego-warp (docs/13 §2 L1, step 2, the flagship
# layer) on top of DC-RSSM-1's physical evidence field. Prepared 2026-10-08.
# NOT yet trained — every launch is a user-approved operation.
#
# Why this config exists
# ----------------------
# DC-RSSM-1 verdict (training_runs_full.md §68.9h): AP gate FAIL, mechanism
# 3/4. The L3 evidence field fixed MCI-v2's Ped collapse (23.10 -> 27.07,
# +3.97) and produced the family's first fully monotone curve, but the
# ALIGNMENT root cause (docs/08: alignment is purely appearance-driven;
# static content presents as -v_ego in Doppler) is untouched by L3 — nothing
# in the chain compensates ego motion, so the deformable alignment keeps
# paying attention to a state that has moved. L1 is the flagship layer for
# exactly that root cause (docs/13 §0/§2): solve v_ego per frame in closed
# form from the static Doppler set, warp the carried state by v_ego * dt
# BEFORE the deformable alignment so the deformable only sees the residual.
# Pose-free: TJ4D has no ego pose and no timestamps, so this is detection-
# time self-calibration (novelty vs CRT-Fusion, which consumes nuScenes
# poses; here it is a necessity, not a choice — docs/09).
#
# Per the DC-RSSM-1 verdict, L3 is KEPT (Ped-repair asset) and L1 rides on
# top: step 2 is the pre-registered "+L1 ego-warp" increment.
#
# Robust solve (pre-run mechanism correction, 2026-10-09, BEFORE any 24e
# data; evidence in the smoke work_dir's probes):
#   The pre-registered static rule (v_std <= q90 AND |v_r| <= q90, weighted
#   L2 closed form) was measured to be inflated by +24%..+96% on val frames
#   (idx 3: 4.20-4.34 vs LMedS truth 2.96-3.03; idx 703: 3.19-3.62 vs
#   1.84-1.86; idx 1403: 1.73-1.87 vs 1.37-1.42). Cause: a handful of
#   strong-return moving cells carry count*SNR weights up to ~100x a static
#   cell's, so the weighted fit follows the mover tail. The robust
#   replacement is the same pre-registered intent realised properly:
#   quantile pre-filter -> LMedS seed (median absolute residual, coarse 1-D
#   over vx then a local 2-D refine) -> MAD residual trim
#   clamp(3*1.4826*MAD, 0.2, 1.0) m/s -> the same weighted closed form on
#   the trimmed set. Landed within 0.2% of the LMedS truth on all 12
#   probed frames (unit test W11 reproduces the mover-bias/recovery pair).
#   Geometry was NOT the problem: 8 canvas<->world mapping hypotheses all
#   fit within 10% residual RMS and the assumed one is best; the rigid
#   field explains 58-80% of measured cells with a 0.013 m/s median
#   residual. F1's 7-13 m/s is a p98-max tail statistic, not a mean.
#
# Scalar learning rate (pre-run design decision, same session, before any
# 24e data): the smoke's warp_scale trajectory (1.4e-3 at iter 50 ->
# 4.54e-3 at iter 450, dt 0.1013 -> 0.1082) extrapolates to warp_scale ~0.3
# by ep16 under AdamW at the base lr (effective per-step update ~7% of lr),
# i.e. ~30% compensation — the layer would be only half exercised and the
# ep16 toggle test correspondingly weak. lr_mult=5 on the two new scalars
# ONLY (dt, warp_scale) accelerates them to ~1.0 (full compensation scale)
# within the run without touching the mechanism's semantics: zero-init
# identity start, same math, same gradient path; the loss still decides the
# value. This is an optimisation-rate choice, not a mechanism variable.
#
# Changes relative to ..._no2d_igdr_dcrssm1 (the direct base):
#   temporal_fusion : type -> EgoWarpEvidenceFieldRSSMFusion, + use_ego_warp,
#                     + pc_range (pinned below), + static_q / min_static_cells
#                     / dt_init / det_eps / warp_z_state. Everything else
#                     byte-identical (L3 evidence field stays ON).
#   optimizer       : paramwise_cfg lr_mult=5 for dt / warp_scale only.
#
# New parameters: dt (1 scalar, init 0.1 s = nominal 10 Hz TJ4D frame
# interval; the dataset has no timestamps — documented assumption) and
# warp_scale (1 scalar, ZERO init = bitwise identity start). The closed-form
# v_ego is detached (a measurement), never a parameter.
#
# Temporal geometry pinning: R4Det.py:872-873 asserts the temporal fusion
# feature grid is (216, 248); with voxel_size 0.32 that is x in [0, 69.12]
# (216 cells, dim2) and y in [-39.68, 39.68] (248 cells, dim3). The module
# derives cell sizes from this range and the runtime feature shape, so a
# wrong pc_range would produce wrong displacements — caught by the smoke's
# v_ego magnitude gate and the ep16 probe's sign test.
#
# Pre-registered gates (docs/13 §4 step 2, written before any data):
#   mechanism (450-iter smoke): stat_ego_warp_scale leaves exactly 0 and the
#     dt gradient arrives (5th instance of the family deadlock rule;
#     unit-tested in test_ego_warp_rssm.py W6); stat_ego_solved > 0.5 on
#     measured frames; stat_ego_speed inside [0.5, 20] m/s (TJ4D ego flow is
#     7-13 m/s on ego-moving frames, docs/08 F1; the train-set mean over all
#     frames sits well below that because many frames are near-stationary --
#     this is a geometry-wiring check, not a calibration); static set
#     non-vanishing; no NaN; and warp NOT inert (E8 below).
#     E5 CORRECTION (found in the first smoke, before any 24e data):
#     stat_ego_static_frac is normalized over the WHOLE canvas, and TJ4D
#     radar measures only ~0.3-1% of it (L3 calibration: 99.68% empty), so
#     the originally drafted 0.05 line was mis-scaled by ~100x. The
#     pre-registered intent ("the static set does not vanish") is kept at
#     5e-4 in canvas units; the probe reports |static|/|measured| directly.
#   mechanism (ep16 probe, PAIRED TOGGLE + INERTNESS -- pre-registered
#     formulation, three-way classification so a null result cannot be
#     misread):
#     (a) INERTNESS: mean |warp_scale| * dt * |v_ego| must exceed 1e-3 m at
#         ep16. At iter 450 of the smoke it is 1.3e-3 m (0.0046 * 0.108 *
#         2.6), i.e. the layer is alive but its applied displacement is
#         millimetres against a needed ~0.26 m per frame; if it is still
#         inertial at ep16 the warp was never exercised and the AP verdict
#         is uninformative about the mechanism -> record as "unexercised",
#         NOT as falsified.
#     (b) If non-inert: with the SAME ep16 weights and SAME val clips, the
#         static-region cross-frame alignment residual (mean
#         |h_warped - h_curr| over static cells, per clip, normalized) must
#         be LOWER with the warp active than with warp_scale forced to 0, in
#         >= 8 of 10 clips (sign test). Sign test rather than a magnitude
#         threshold so the gate stays falsifiable without a post-hoc
#         threshold sweep. Also: |v_ego| in [3, 20] m/s on ego-moving val
#         frames (geometry teeth; an independent occupancy cross-correlation
#         shift is measured by me_rssm/sanity/probe_ego_warp_sanity.py and
#         must agree with the Doppler solve in magnitude and sign).
#   AP (24e): ep12-16 window mean >= 39.2809 (no2d_igdr seed0 baseline) AND
#     >= 38.8786 (MCI-v2 39.3786 - 0.5, the chain non-inferiority line);
#     Ped paired delta >= -1.5 (keep the L3 Ped repair); Car/Truck no
#     regression (docs/13 step-2 addendum).
#   Failure of either gate = stop-and-record (no hyperparameter sweeps on an
#   unverified mechanism). Per the DC-RSSM-1 verdict, L1 is the DC-RSSM
#   direction's LAST attempt: if it fails the AP gate the direction stops
#   and the paper pivots to the mechanism-analysis positioning
#   (docs/rssm_paper_directions_20261008.md).

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
    '../../me_rssm/configs/'
    'TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_dcrssm1.py'
)

custom_imports = dict(imports=[
    'mmdet3d',
    'mmdet3d.core.hook.kl_scale_scheduler',
    'me_rssm',
])

optimizer = dict(
    # Only the two new scalars get the accelerated rate (see header);
    # custom_keys match the module path in named_parameters.
    paramwise_cfg=dict(
        custom_keys={
            'temporal_fusion.dt': dict(lr_mult=5.0),
            'temporal_fusion.warp_scale': dict(lr_mult=5.0),
        }),
)

model = dict(
    temporal_fusion=dict(
        type='EgoWarpEvidenceFieldRSSMFusion',
        # --- L1 analytic ego-warp (docs/13 step 2, flagship) -------------
        use_ego_warp=True,
        # Pinned by R4Det.py:872-873 (216 x = 69.12/0.32, 248 y = 79.36/0.32).
        pc_range=[0.0, -39.68, -4.0, 69.12, 39.68, 2.0],
        # Static set: quantile gates over measured cells, no hyperparameter
        # sweep (docs/13 L1: tau from quantiles by construction).
        static_q=0.9,
        min_static_cells=32,
        # Nominal 10 Hz TJ4D frame interval (no timestamps in the dataset);
        # learnable from there.
        dt_init=0.1,
        det_eps=1e-6,
        # Align h and z across the previous ego frame (align_z_state chain:
        # z is carried in the previous ego frame just like h).
        warp_z_state=True,
    ),
)
