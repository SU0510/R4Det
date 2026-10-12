# 450-iter DDP smoke for the reliability-gated fusion arm (wiring smoke only;
# NO val, no AP claim). Pre-registration:
#   docs/rssm_reliability_fusion_prereg_20261012.md §5 (S0)
#
# S0 gates (read from the work_dir log.json / iter_450.pth / the unit tests):
#   S0-1 reaches iter 450, writes iter_450.pth, no NaN in any logged loss.
#   S0-2 zero-init identity contract: the gate's last layer starts at zero, so
#        alpha == 1 and the arm is bitwise identical to ConcatConvFusion.
#   S0-3 the gate is live: its weights leave zero and ``alpha`` deviates from 1.
#   S0-4 layout: occupancy uses the pillar-scatter convention (rows=y, cols=x),
#        verified against PointPillarsScatter's source and the corner unit test.
_base_ = [
    '../configs/'
    'TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_relfusion.py'
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
