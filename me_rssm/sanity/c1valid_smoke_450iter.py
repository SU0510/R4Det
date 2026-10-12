# 450-iter DDP smoke for C1 (wiring/mechanism smoke only; NO val, no AP claim).
# Pre-registration: docs/rssm_c_validity_update_prereg_20261012.md §5 S0.
#
# S0 gates read from the work_dir log.json / the wiring probe:
#   S0-1 loss finite, no NaN/OOM, reaches iter 450 and writes iter_450.pth.
#   S0-2 v_gate_mean == 1.0 at step 0 (zero-init identity contract) and the
#        gate weights receive non-zero grad once training starts.
#   S0-3 valid_t is non-empty on every frame and its invalid fraction matches
#        the frozen coverage statistics (§76: 19.8% of val objects are zero
#        returns; cell-level invalid share is reported, not gated).
#   S0-4 baseline arms are untouched: fusions without accepts_obs_valid keep
#        the exact old call signature (regression test), and this run uses the
#        same batch/seed/schedule protocol as every other 450-iter smoke.
_base_ = [
    '../configs/'
    'TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_c1valid.py'
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
