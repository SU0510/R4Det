# 450-iter DDP smoke for route-A cell 3/4 (L2 posterior input, Xavier z_proj
# init). Not a formal run.
#
# Gate: e_pooled specific >= 5%, correction specific >= 3%, correction^2
# >= 0.005 and plateaued, shuffle_z head >= 1%, recon drop >= 15%, raw KL
# bounded.
_base_ = [
    '../../configs/r4det/'
    'TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_'
    'no2d_igdr_lowdim_z_delta_spatial_xavier.py'
]

runner = dict(_delete_=True, type='IterBasedRunner', max_iters=450)
lr_config = dict(
    _delete_=True,
    policy='CosineAnnealing',
    warmup=None,
    warmup_iters=0,
    warmup_ratio=0.1,
    min_lr_ratio=1e-5,
    by_epoch=False)
evaluation = dict(interval=450)
checkpoint_config = dict(interval=450)
