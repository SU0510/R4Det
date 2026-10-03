# MCI ablation A-off: remove the Doppler-conditioned alignment offsets
# (ME-RSSM C1). Single variable vs ..._mci: use_motion_offsets=False.
# Isolates the value of measurement-driven predict-side alignment; the
# deformable alignment itself (appearance-driven, baseline) stays on.
_base_ = ['./TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_mci.py']

model = dict(
    temporal_fusion=dict(
        use_motion_offsets=False,
    ),
)
