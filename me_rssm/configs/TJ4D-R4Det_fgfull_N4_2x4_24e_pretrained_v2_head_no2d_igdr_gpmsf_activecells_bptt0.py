# Formal GP-MSF BPTT0 arm.
#
# Relative to the versioned BPTT0 B0 reference, the only model variable is
# RCFusion.  active_cells_only skips only BEV cells whose complete candidate
# set is provably zero; all valid image/radar levels, height references, and
# camera-only fallback remain in the computation.

_base_ = (
    '../../configs/r4det/'
    'TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_bptt0.py'
)

custom_imports = dict(imports=[
    'mmdet3d',
    'mmdet3d.core.hook.kl_scale_scheduler',
    'me_rssm',
])

model = dict(
    RCFusion=dict(
        type='GeometryPairedMultiScaleFusion',
        img_channels=256,
        rad_channels=384,
        out_channels=256,
        kernel_size=3,
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        act_cfg=dict(type='ReLU', inplace=True),
        bev_shape=(248, 216),
        bev_origin=(0.0, -39.68),
        bev_cell=0.32,
        radar_origin=(0.08, -39.60),
        radar_base_cell=0.16,
        radar_base_shape=(496, 432),
        radar_level_channels=(64, 128, 256),
        radar_level_steps=(0.32, 0.64, 1.28),
        image_channels=256,
        image_size=(480, 640),
        image_level_strides=(4, 8, 16),
        depth_downsample=8,
        embed_dim=64,
        num_heads=4,
        num_offsets=4,
        height_refs=(-1.1230436265, -0.3003779650, 0.6601446867),
        gamma_init=1.0,
        null_score_init=0.0,
        query_chunk=4096,
        grad_checkpoint=True,
        new_param_seed=20261013,
        require_geo=True,
        active_cells_only=True,
    ),
)
