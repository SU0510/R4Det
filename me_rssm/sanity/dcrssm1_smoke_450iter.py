# 450-iter DDP smoke for DC-RSSM-1 (physical evidence field, docs/13 L3).
#
# Mechanism-only gate, no val configured. Purpose: prove the evidence
# field trains on the full-resolution main line BEFORE spending an 18e
# run. The DC-RSSM-1 specific gates (in addition to the MCI smoke gates):
#   - stat_ev_c_mot_std / stat_ev_dyn_std: the physical field's spatial
#     std is non-trivial BY CONSTRUCTION (the property the learned gates
#     lacked; docs/13 step-1 mechanism gate: std/mean >= 0.1)
#   - stat_ev_cov_mean: coverage coverage statistics present
#   - evidence_refine_conv receives gradient (verified in unit tests;
#     here we check the weight moves off its small-normal init)
#   - stat_gain_* / readout non-degenerate, identity start intact at
#     iter 0 (stat_gain_mean == 1.0 exactly at the first iterations)
#
# Gate numbers are read from the work_dir log.json afterwards with:
#   python3 me_rssm/sanity/mci_gate_report.py <work_dir>
#
# The AP+mechanism gate at ep16 (c_mot/dyn spatial std/mean >= 0.1,
# ep12-16 >= 39.2809 AND >= v2 - 0.5, Ped paired >= -1.5) is a separate,
# user-approved run.

_base_ = [
    '../configs/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_dcrssm1.py'
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
