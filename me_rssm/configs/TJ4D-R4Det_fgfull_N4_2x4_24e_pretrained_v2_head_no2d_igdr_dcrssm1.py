# DC-RSSM-1: physical evidence field (docs/13 L3, step 1) on the MCI main
# line. Prepared 2026-10-08. Design: me_rssm/docs/13_dc_rssm.md. NOT yet
# trained — every launch is a user-approved operation.
#
# Why this config exists
# ----------------------
# MCI-v2 ep16 verdict (training_runs_full.md §68.9e): AP gate FAIL
# (39.3786 < 39.6061), mechanism gate 3/4 — the innovation channel is
# alive (deadlock fix verified end-to-end) but its magnitude gate input
# carries only 5.4% sample-specific energy (94.6% batch-common), and the
# learned confidences c_cam/c_mot collapsed to spatial constants
# (§68.9b). Detection gradients do not produce spatial structure; the
# structure must come from the measurement, by construction.
#
# DC-RSSM-1 replaces the learned gate inputs c_mot / dyn with a
# PARAMETER-FREE physical evidence field computed from the raw Doppler
# canvas ('velocity_bev'): c_mot0 = sigmoid(alpha1 * count * SNR -
# beta1), dyn0 = sigmoid(alpha2 * v_std * count - beta2), cov = count /
# log(2) (count is the canvas's already-normalized occupancy). The only
# learned part is a zero-init (small-normal) 1x1 refinement conv —
# learning corrects the physical layer, never replaces it. Coverage
# semantics (cov ~ 0 = "no measurement", NOT "static") is the
# pre-registered Ped dead-zone answer.
#
# Changes relative to ..._no2d_igdr_mci (the direct base):
#   temporal_fusion   : -> EvidenceFieldInnovationRSSMFusion (+ L3 field,
#                        use_physical_evidence=True). Everything else is
#                        byte-identical MCI mainline: same producer chain,
#                        same switches, same MAI-inherited args.
#
# Pre-registered gates (docs/13 §4 step 1, written before any data):
#   mechanism (450 iter / probe): c_mot/dyn map spatial std/mean >= 0.1
#     (learned version ~0.001); class separation at GT centers appears.
#   AP (18e): ep12-16 >= 39.2809 (no2d_igdr baseline) AND >= v2 - 0.5
#     (39.379 - 0.5 = 38.879); Ped paired delta >= -1.5 (the cov+refine
#     direct test).
#   Failure of either gate = stop-and-record (no hyperparameter sweeps on
#   an unverified mechanism).

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
    'TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_mci.py'
)

custom_imports = dict(imports=[
    'mmdet3d',
    'mmdet3d.core.hook.kl_scale_scheduler',
    'me_rssm',
])

model = dict(
    temporal_fusion=dict(
        type='EvidenceFieldInnovationRSSMFusion',
        # --- L3 physical evidence field (docs/13 step 1) -----------------
        use_physical_evidence=True,
        evidence_refine=True,
        # Fixed calibration constants (sigmoid argument scale/shift).
        # Defaults are the identity calibration (alpha=1, beta=0): the
        # quantile-based one-shot calibration tool can overwrite them
        # before launch; the gate compares whatever is written here.
        c_mot_alpha1=1.0,
        c_mot_beta1=0.0,
        dyn_alpha2=1.0,
        dyn_beta2=0.0,
        cov_floor=0.0,
    ),
)
