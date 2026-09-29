# 450-iter DDP smoke for the Motion-Aligned Innovation RSSM (MAI).
#
# Mechanism-only gate, no val configured. The point of this smoke is to prove
# the innovation posterior trains on the full-resolution main line before
# spending an ep6 run on it:
#   - stat_obs_pred falls (obs_prior actually learns to predict the encoding)
#   - stat_innovation_sq stays bounded and does not collapse to zero
#   - stat_mu_diff_sq does not decay monotonically toward zero
#   - reconstruction / KL stay comparable to the no2d_igdr main line
#   - no NaN/Inf, memory roughly at the main-line level
#
# Strict same-epoch mechanism baseline for the paired comparison is the
# standard RSSM (no2d_igdr seed1), whose three-window probes give:
#   ep6  shuffle_z head 49.73% / 3.30% / 3.36%, zero 36.97% / 40.87% / 44.29%
#   ep12 shuffle_z head  2.71% / 4.71% / 4.14%, zero 26.39% / 47.66% / 54.34%

_base_ = [
    '../../configs/r4det/'
    'TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_mai.py'
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
