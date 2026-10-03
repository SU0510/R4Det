# MCI ablation C-off: remove the reliability-gated correct step
# (z_t = z_sel + tanh(.)*(z_sel - mu_p)). Single variable vs ..._mci:
# use_reliability_gain=False. The gated READOUT stays on, so this isolates
# the correct-step gate alone.
_base_ = ['./TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_mci.py']

model = dict(
    temporal_fusion=dict(
        use_reliability_gain=False,
    ),
)
