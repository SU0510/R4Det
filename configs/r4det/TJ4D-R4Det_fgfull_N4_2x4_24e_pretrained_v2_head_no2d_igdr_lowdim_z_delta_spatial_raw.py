# Route A cell 2/4: information-preserving posterior input, small readout.
#
# Base: ..._lowdim_z_delta_spatial (cell 1/4 = l2 + std=0.01).
# Only change: the posterior sees `pool(e_t) * 0.1` instead of
# `normalize(pool(e_t))`. The measured L2 denominator is dominated by the
# batch-common component (1.90 RMS common vs 0.53 sample-specific), so it
# divides away most of the sample identity. A single shared scalar keeps the
# relative amplitudes. 0.1 maps the raw 1.97 RMS to ~0.197, close to h_t.
# readout_mode / z_proj_init / reconstruction / KL are unchanged.

_base_ = (
    './TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_'
    'no2d_igdr_lowdim_z_delta_spatial.py'
)

model = dict(
    temporal_fusion=dict(
        posterior_obs_mode='scaled_raw',
        posterior_obs_scale=0.1,
    ),
)
