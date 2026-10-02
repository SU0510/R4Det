# MotionAlignedRSSMFusion baseline, shortened to a complete 18-epoch run.
#
# Base: ..._no2d_igdr.py (the 24-epoch, 3-GPU mainline baseline:
# ep12-16 = 40.4400 +- 1.3475 over three seeds, all-seed peak
# 41.8296 +- 1.3843; this config intentionally changes ONLY the schedule
# horizon, not the architecture or the data pipeline).
#
# Why a separate config instead of `max_epochs=18` on the CLI:
# mmcv's CosineAnnealingLrUpdaterHook reads runner.max_epochs, so changing
# only the runner would leave the LR schedule assuming 24 epochs while the
# run stops at 18 (LR at ep18 would still be ~2.9e-5, not the intended
# minimum).  Keeping a standalone config makes the complete 18-epoch
# horizon explicit and auditable.
#
# Schedule changes relative to the 24e baseline:
#   runner.max_epochs      24 -> 18
#   lr_config              24e cosine -> 18e cosine (same peak lr)
#   KLScaleSchedulerHook   ep0-10 linear -> ep0-8 linear
#
# The KL warmup is shortened proportionally (10/24 -> 8/18) so that the
# posterior is still unconstrained while the detector stabilises, then KL
# reaches its full weight for the final ten epochs.  Peak LR stays 1.5e-4;
# only the decay horizon changes.

_base_ = './TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr.py'

runner = dict(type='EpochBasedRunner', max_epochs=18)

lr_config = dict(
    policy='CosineAnnealing',
    warmup=None,
    min_lr_ratio=1e-5,
)

custom_hooks = [
    dict(
        type='KLScaleSchedulerHook',
        start_epoch=0,
        end_epoch=8,
        start_value=0.0,
        end_value=1.0,
    ),
]
