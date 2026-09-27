# 450-iter DDP smoke for the unpooled spatial readout. Not a formal run.
#
# Gate before any formal training: correction^2 must stay >= 0.005 without a
# continued decline over the last 150 iters, recon must fall >= 15%, and the
# free-bits clamped ratio must stay below 0.95.
_base_ = [
    '../../configs/r4det/'
    'TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_'
    'no2d_igdr_lowdim_z_delta_spatial.py'
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
