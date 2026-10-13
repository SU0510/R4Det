# 450-iter DDP smoke for the GP-MSF arm (real-data S0; NO val, no AP claim).
# Pre-registration: docs/gpmsf_prereg_20261013.md §5.
#
# S0 gates read from the work dir log.json / iter_450.pth / gpmsf_wiring_stats:
#   S0-1 loss finite, no NaN/OOM, reaches iter 450 and writes iter_450.pth.
#   S0-2 single variable vs the mainline, residual projections zero-initialised,
#        fusion_conv shape/params identical to ConcatConvFusion.
#   S0-3 the geometry path really ran: proj_r2c / proj_i2r left zero (they only
#        receive gradient through the native multi-scale evidence path).
#   S0-4 capacity frozen: new params 177,726 / shared fusion_conv 1,475,072.
# Memory and step time are reported against the existing baseline-front-end
# smokes; they flag a review, they are not an automatic stop (see prereg §5).
_base_ = [
    '../configs/'
    'TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_gpmsf.py'
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

custom_imports = dict(imports=[
    'mmdet3d',
    'mmdet3d.core.hook.kl_scale_scheduler',
    'me_rssm',
    'me_rssm.gpmsf_stats_hook',
])
# the mainline KL schedule, plus the GP-MSF wiring probe
custom_hooks = [
    dict(
        type='KLScaleSchedulerHook',
        start_epoch=0,
        end_epoch=10,
        start_value=0.0,
        end_value=1.0,
    ),
    dict(type='GPMSFWiringStatsHook', interval=50),
]

# collect_stats costs a few python-scalar syncs per forward; it is only on for
# the smoke (real-data wiring evidence) and off in the 24e arm.
model = dict(RCFusion=dict(collect_stats=True))
