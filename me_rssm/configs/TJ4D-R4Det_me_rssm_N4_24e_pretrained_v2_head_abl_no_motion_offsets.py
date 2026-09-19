# Ablation A-off: remove the Doppler-conditioned term from the deformable
# state-alignment offsets. The alignment falls back to the learned
# appearance-driven offsets of the baseline. Everything else (modality
# observations, reliability gain, dynamic gate) stays on.
_base_ = ['./TJ4D-R4Det_me_rssm_N4_24e_pretrained_v2_head.py']

model = dict(
    temporal_fusion=dict(
        use_motion_offsets=False,
    ),
)
