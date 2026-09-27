# Posterior-innovation repair with an unpooled spatial detection readout.
#
# Base: lowdim_z_delta. Both 450-iter gates before this one failed for the same
# reason: the reconstruction loss fell only ~11%, the batch-shuffle ranking gap
# reached just 0.0111 against a 0.1 target, and swapping corrections changed
# reconstruction by ~1%. The bottleneck is therefore the readout, not the
# supervision: the correction was average-pooled to 32 numbers and squashed
# through FiLM/gating, so the detector never saw where the innovation lived.
#
# This variant deletes the global pool, FiLM and gate, and feeds
# `feat + h_proj(h_t) + z_proj(correction_t)` with 1x1 convolutions on the
# 16x16 latent grid that are upsampled to the BEV resolution. Differential
# reconstruction and every KL hyperparameter are unchanged; the discriminating
# ranking term is deliberately off so the readout is the only variable.

_base_ = (
    './TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_'
    'no2d_igdr_lowdim_z_delta.py'
)

model = dict(
    temporal_fusion=dict(
        readout_mode='spatial',
    ),
)
