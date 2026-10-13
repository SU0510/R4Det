# 450-iter DDP smoke for the GP-MSF arm (real-data S0; NO val, no AP claim).
# Pre-registration: docs/gpmsf_prereg_20261013.md §5.
#
# Required S0: fresh successful exit, all logs/checkpoint finite, real-batch
# train/eval initial equivalence and two-step gradients, complete rank-zero
# diagnostics and 3-rank parameter sync, frozen single-variable recipe/capacity.
# Matched B0 resource smoke is gpmsf_baseline_smoke_450iter.py. Time AND memory
# must be <=1.20x; missing evidence or over-budget means no automatic 24e run.
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

# The hook collects every 50th iteration only, with gradient checkpointing
# still enabled. Formal 24e keeps statistics off.
model = dict(RCFusion=dict(collect_stats=True))
