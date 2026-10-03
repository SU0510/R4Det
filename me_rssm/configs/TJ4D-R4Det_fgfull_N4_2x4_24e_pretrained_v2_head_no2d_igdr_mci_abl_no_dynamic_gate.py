# MCI ablation D-off: remove the dynamic-evidence-modulated GRU update
# gate (ME-RSSM C3). Single variable vs ..._mci: use_dynamic_gate=False.
_base_ = ['./TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_mci.py']

model = dict(
    temporal_fusion=dict(
        use_dynamic_gate=False,
    ),
)
