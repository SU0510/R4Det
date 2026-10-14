# PROBE-ONLY config: frozen GP-MSF arm + active_cells_only=True + rssm_bptt_steps=0.
#
# Purpose (training record section 104.10, user directive 2026-10-14): isolate
#   (a) the cost of the *history-frame* BPTT edge (t2 in a 4-frame N4 window),
# by pairing against the same arm with the default rssm_bptt_steps=1.
#
# Semantics: N=4 window, default rssm_bptt_steps=1 keeps t0,t1 as no-grad burn-in
# and t2,t3 gradient-enabled.  rssm_bptt_steps=0 makes t0,t1,t2 all no-grad
# burn-in; only the current frame t3 is gradient-enabled.  No frame is dropped,
# 4 extract_feat / 4 GP-MSF calls remain; only the t2 backward + checkpoint
# recompute is removed.
#
# This is *not* a mechanism change to GP-MSF: the geometry path is unchanged and
# still trains from the current-frame detection loss.  It is an RSSM temporal
# variable (BPTT1 vs BPTT0) and must be ablated, not silently baked in.
#
# This file must NOT be used for a training run.
_base_ = './TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_gpmsf_activecells_probe.py'

model = dict(
    rssm_bptt_steps=0,
)
