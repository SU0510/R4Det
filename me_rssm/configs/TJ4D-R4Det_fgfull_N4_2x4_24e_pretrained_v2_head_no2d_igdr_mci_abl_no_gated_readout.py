# MCI ablation R-off: remove the reliability-gated readout
# (output = feat + res + tanh(.)*res). Single variable vs ..._mci:
# use_gated_readout=False. The correct-step gate stays on. This is THE
# ablation for the P1 fix (docs/rssm_review_20261002.md): if the full model
# beats MAI but this ablation does not, the gain comes from the readout.
_base_ = ['./TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_mci.py']

model = dict(
    temporal_fusion=dict(
        use_gated_readout=False,
    ),
)
