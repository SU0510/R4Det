# 450-iter DDP smoke for route B (Innovation-Conditioned RSSM). Not a formal
# run; no val is configured because the route-B gate is mechanism-only.
#
# Gate: innovation specific > e_pooled specific, delta_mu specific >= 5%,
# prior_only head clearly below posterior, shuffle_z head >= 2%.
_base_ = [
    '../../configs/r4det/'
    'TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_'
    'no2d_igdr_lowdim_z_innovation.py'
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
