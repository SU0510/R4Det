# PROBE-ONLY config: standard mainline B0 + rssm_bptt_steps=0.
#
# Purpose (training record section 104.10): the fair baseline for the GP-MSF
# BPTT0 pair.  A resource comparison between arm(BPTT0) and B0(BPTT1) would be
# confounded, so B0 must carry the same temporal variable.  See
# ..._gpmsf_activecells_bptt0_probe.py for the arm side.
#
# Semantics: default rssm_bptt_steps=1 keeps t0,t1 burn-in and t2,t3
# gradient-enabled in the N=4 window; =0 makes t0,t1,t2 burn-in and keeps only
# the current frame t3.  4 extract_feat calls remain.
#
# This file must NOT be used for a training run.
_base_ = '../../configs/r4det/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr.py'

model = dict(
    rssm_bptt_steps=0,
)
