# 450-iter DDP smoke for the route-B `z_prev` normalize ablation.
#
# Not a formal run; no val is configured because the gate is mechanism-only.
# Compare against the route-B 450-iter smoke on the same iter budget:
#   innovation specific > e_pooled specific
#   delta_mu specific >= 5%
#   correction^2 forms a plateau instead of monotone collapse
#   shuffle/zero head ratio improves over the route-B 0.064-0.092 band

_base_ = [
    '../../configs/r4det/'
    'TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_'
    'no2d_igdr_lowdim_z_innovation_rawnorm.py'
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
evaluation = dict(interval=4500)
checkpoint_config = dict(interval=450)
