# PROBE-ONLY config: the frozen GP-MSF arm plus ``active_cells_only=True``.
#
# Purpose: measure the *real whole-step* cost of the active-cell compaction
# (training record section 104.9) on the same N4/batch/DDP3 recipe, without
# editing the pre-registered arm config.  The switch is an exactly-equivalent
# reformulation of the dense candidate loop -- a cell whose every candidate is
# invalid (no image validity AND no radar echo support) provably returns
# M_R = M_I = 0 with valid_r = valid_i = False, so it can be skipped.  It is
# *not* a new mechanism: no candidate is dropped while it is valid, all three
# image levels, all three native radar levels, three height refs and the
# camera-only fallback stay in force.
#
# Equivalence evidence: me_rssm/sanity/gpmsf_active_cells_checks.py (A1-A10,
# bit-exact at equal chunking) and the arm-config 13/13 + golden compare.
#
# This file must NOT be used for a training run.  If the whole-step probe meets
# the resource gate, the flag is added to the arm config itself.
_base_ = './TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_gpmsf.py'

model = dict(
    RCFusion=dict(
        active_cells_only=True,
    ),
)
