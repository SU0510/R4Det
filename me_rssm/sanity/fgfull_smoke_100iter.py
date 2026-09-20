# Temporary 100-iteration DDP smoke config. Do not use for the formal run.
_base_ = ['../../configs/r4det/'
          'TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head.py']

# Replace, not merge, so the inherited max_epochs key is removed.
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
