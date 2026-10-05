# MCI + modality dropout 0.1: Ped-loose red-flag mitigation (68.8e/g).
#
# Single variable vs ..._mci: modality_dropout 0.0 -> 0.1. At training time
# each sample's camera/motion confidence maps are zeroed with p=0.1 per
# modality, forcing the filter to practise the prior-fallback path
# (me_rssm/docs/06 P-item: "modality dropout, dual identity: robustness +
# fallback training pressure") instead of only ever seeing confident
# observations. Targets the diagnosed Ped dead-zone: Ped is the only class
# that is dynamic while nearly invisible to Doppler (dynamic score 0.145,
# docs/08), so both the observation and the fallback prediction are weak
# there; dropout trains the prior/transition to carry more of the load and
# regularizes the gate toward less over-confident shrinkage.
#
# Pre-registered gate (written before any val exists, 2026-10-05):
#   - Overall ep12-16 window >= no2d_igdr seed0 39.2809 (baseline kept)
#     AND >= MCI mainline seed0 40.1061 - 0.5 (no loss vs mainline)
#   - Ped ep12-16 paired delta vs baseline >= -1.5 (recovered from -4.07)
#   - Mechanism: three-window shuffle_z >= 0.2%, h/e < 2 (unchanged)
# Success -> multi-seed queue. Everything else inherited unchanged.

_base_ = [
    './TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_mci.py'
]

model = dict(
    temporal_fusion=dict(
        modality_dropout=0.1,
    ),
)
