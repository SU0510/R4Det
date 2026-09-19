# Ablation B-off: the temporal module consumes no side-channel context at
# all (equivalent to running on the plain fused BEV). The velocity-evidence
# chain still runs upstream but nothing reads it. This is the runtime
# equivalent of the empty-bus baseline and isolates the value of the
# modality-specific observation model as a whole.
_base_ = ['./TJ4D-R4Det_me_rssm_N4_24e_pretrained_v2_head.py']

model = dict(
    temporal_fusion=dict(
        use_modality_obs=False,
    ),
)
