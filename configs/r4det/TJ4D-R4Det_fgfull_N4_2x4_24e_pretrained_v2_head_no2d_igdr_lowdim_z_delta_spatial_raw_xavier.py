# Route A cell 4/4: information-preserving posterior input + Xavier readout.
#
# Base: ..._lowdim_z_delta_spatial (cell 1/4 = l2 + std=0.01).
# Applies both route-A changes at once. The 2x2 matrix separates an upstream
# information bottleneck (posterior input normalization) from a downstream
# gain bottleneck (z_proj initialization); this cell is the combined repair
# and is the candidate for the formal ep6/ep12 gate.

_base_ = (
    './TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_'
    'no2d_igdr_lowdim_z_delta_spatial.py'
)

model = dict(
    temporal_fusion=dict(
        posterior_obs_mode='scaled_raw',
        posterior_obs_scale=0.1,
        z_proj_init='xavier',
    ),
)
