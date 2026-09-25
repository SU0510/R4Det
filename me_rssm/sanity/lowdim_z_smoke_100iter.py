# 100-iter DDP smoke for the low-dimensional z variant. Not a formal run.
_base_ = ['../../configs/r4det/'
          'TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_lowdim_z.py']

runner = dict(_delete_=True, type='IterBasedRunner', max_iters=100)
lr_config = dict(
    _delete_=True,
    policy='CosineAnnealing',
    warmup=None,
    warmup_iters=0,
    warmup_ratio=0.1,
    min_lr_ratio=1e-5,
    by_epoch=False)
evaluation = dict(interval=100)
checkpoint_config = dict(interval=100)
