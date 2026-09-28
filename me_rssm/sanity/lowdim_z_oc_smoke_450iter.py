# 450-iter DDP smoke for route C (object-centric latent supervision).
#
# Mechanism-only gate, no val configured. Compare against the rawnorm ep6
# control, which reached:
#   shuffle/zero head ratio 0.077-0.125, shuffle_z head 0.84-1.38%
#   innovation specific 46.7% > e_pooled 16.3%
#   correction^2 0.2218 -> 0.0685 (still monotone collapse)
#
# Route-C pass conditions at 450 iter:
#   clear downward trend in stat_object_hm on the current frame
#   stat_object_hm_recall rising from its ~0.1 logit-bias init
#   stat_correction_sq plateau (no monotone decay to near zero)
#   shuffle/zero and shuffle_z head not worse than the rawnorm control

_base_ = [
    '../../configs/r4det/'
    'TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_'
    'no2d_igdr_lowdim_z_innovation_rawnorm_oc.py'
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
