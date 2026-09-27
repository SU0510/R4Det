# 450-iter DDP smoke for route-A cell 4/4 (scaled_raw posterior input +
# Xavier z_proj init) -- the combined repair. Not a formal run.
#
# Gate: e_pooled specific >= 5%, correction specific >= 3%, correction^2
# >= 0.005 and plateaued, shuffle_z head >= 1%, recon drop >= 15%, raw KL
# bounded. This cell is the ep6/ep12 candidate if it passes.
_base_ = [
    '../../configs/r4det/'
    'TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_'
    'no2d_igdr_lowdim_z_delta_spatial_raw_xavier.py'
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
# No val during the mechanism smoke: the route-A gate is
# training-log + checkpoint-probe only, so evaluating here would add ~7.5 min
# per cell of KITTI AP/IoU that cannot pass or fail the gate.
evaluation = dict(interval=4500)
checkpoint_config = dict(interval=450)
