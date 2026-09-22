# ---------------------------------------------------------------------------
# CLEAN MAINLINE SNAPSHOT (reproducibility guard, research-package output).
#
# PROVENANCE: identical to
#   configs/r4det/TJ4D-R4Det_motion_align_rssm_det3d_N4_2x4_24e_pretrained_v2_head.py
# (commit b293d51, 2026-09-19) with ONE change:
#   pts_bbox_head.shared_stem: True -> False
#
# WHY: the committed mainline config still carries shared_stem=True from the
# closed section-41 experiment (added in b4fd7be, never reverted; audit U7
# of docs/R4Det_RSSM_full_report.md 3.3). The Run 10 multiseed mainline
# (40.42 +- 0.51, the paper main table) trained with shared_stem=False.
# Re-running the committed config as-is would NOT reproduce the mainline.
# This snapshot freezes the clean setting without touching any original
# file. The ME-RSSM mainline config already overrides shared_stem=False,
# so all queued ME runs are unaffected.
#
# Historical hygiene issue, since fixed: BEVRSSMTemporalFusion.forward lost
# its final `return` in commit 99c74ff (report 3.3 item 2), which made a
# `baseline_rssm` rerun crash on the None tuple. The line was restored in the
# rssm_fusion module; MotionAlignedRSSMFusion (the mainline) was never
# affected because it overrides forward entirely. This note is kept for
# provenance, and no config values are changed by that fix.
# ---------------------------------------------------------------------------

import os

try:
    _cfg_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:  # pragma: no cover - mmcv exec's configs without __file__
    _cfg_dir = os.path.join(os.getcwd(), 'me_rssm', 'configs')
_repo_root = os.path.abspath(os.path.join(_cfg_dir, '..', '..'))
if not os.path.isdir(os.path.join(_repo_root, 'configs')):
    _p = os.getcwd()
    for _ in range(5):
        if os.path.isdir(os.path.join(_p, 'configs', 'r4det')):
            _repo_root = _p
            break
        _p = os.path.dirname(_p)

_base = os.path.join(
    _repo_root, 'configs/r4det',
    'TJ4D-R4Det_motion_align_rssm_det3d_N4_2x4_24e_pretrained_v2_head.py')

_base_ = [_base]

model = dict(
    pts_bbox_head=dict(
        shared_stem=False,
    ),
)
