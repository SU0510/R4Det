# 450-iter DDP smoke for MCI-RSSM (merged model).
#
# Mechanism-only gate, no val configured. Purpose: prove the merged model
# trains on the full-resolution main line BEFORE spending an ep6 run:
#   - stat_obs_pred falls (obs_prior learns to predict the observation)
#   - stat_innovation_sq stays bounded, does not collapse to zero
#   - stat_gain_* shows a non-degenerate dynamic range
#     (std > 0 across iters; per-iter spatial std > 0; mean not pinned at 1)
#   - stat_readout_gain_* same non-degeneracy
#   - reconstruction / KL comparable to the no2d_igdr/MAI main line
#   - no NaN/Inf, memory roughly at main-line level (+ bus consumers)
#
# Gate numbers are read from the work_dir log.json afterwards with:
#   python3 tools/verdict_window.py --stats <work_dir>
#   python3 me_rssm/sanity/mci_gate_report.py <work_dir>
#
# The AP+mechanism gate at ep6 (shuffle_z >= 2% three-window probe,
# paired ep6 AP vs no2d_igdr seed0) is a separate, user-approved run —
# see me_rssm/sanity/watch_mci_ep6.sh.

_base_ = [
    '../configs/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_mci.py'
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
