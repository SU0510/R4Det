# Ablation D-off: disable the dynamic-evidence modulation of the GRU update
# gate (dyn_beta fixed at 0 semantics). Static background and dynamic
# objects again share one update rate.
_base_ = ['./TJ4D-R4Det_me_rssm_N4_24e_pretrained_v2_head.py']

model = dict(
    temporal_fusion=dict(
        use_dynamic_gate=False,
    ),
)
