# MCI ablation P2-off: gates see only the modality confidences, not the
# innovation magnitude (06_second_layer P2 was the pre-registration FOR
# adding it; this ablation measures its contribution). Single variable vs
# ..._mci: innov_in_gain=False.
_base_ = ['./TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_mci.py']

model = dict(
    temporal_fusion=dict(
        innov_in_gain=False,
    ),
)
