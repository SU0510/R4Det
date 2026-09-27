# Route A cell 3/4: L2 posterior input, Xavier readout init.
#
# Base: ..._lowdim_z_delta_spatial (cell 1/4 = l2 + std=0.01).
# Only change: z_proj uses the same xavier_init the other projections in the
# class receive, instead of a hard-coded std=0.01. For a 32->256 1x1 this is
# ~8-10x larger, matching the scale probe where a x10 gain lifted the
# shuffle_z head ratio from 2.9e-5 to 0.0126. The posterior input is left as
# the historical L2 normalization so this cell isolates the downstream side.

_base_ = (
    './TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_'
    'no2d_igdr_lowdim_z_delta_spatial.py'
)

model = dict(
    temporal_fusion=dict(
        z_proj_init='xavier',
    ),
)
