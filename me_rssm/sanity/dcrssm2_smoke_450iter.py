# 450-iter DDP smoke for DC-RSSM-2 (= DC-RSSM-1 L3 evidence field + L1
# analytic ego-warp, docs/13 step 2).
#
# Mechanism-only gate, no val configured. Purpose: prove the L1 ego-warp
# trains on the full-resolution main line BEFORE spending a 24e run.
#
# DC-RSSM-2 specific gates (on top of the DC-RSSM-1 smoke gates):
#   - stat_ego_warp_scale: exactly 0.0 at the first iterations (bitwise
#     identity start) and OFF ZERO by the end of the smoke (weight
#     liveness; unit-tested to receive gradient from step 1 because the
#     closed-form v_ego is a measurement, not a zero-init learnable).
#   - stat_ego_dt / dt gradient: dt starts at 0.1 and its gradient is
#     chained through warp_scale (arrives from step 2; unit test W6).
#   - stat_ego_solved > 0.5 on measured frames; stat_ego_static_frac
#     >= 0.05.
#   - stat_ego_speed inside [0.5, 20] m/s (TJ4D ego flow 7-13 m/s per
#     docs/08 F1; the band is a geometry-wiring sanity check, deliberately
#     wider than the true flow).
#   - no NaN in any loss or stat.
#   - L3 gates unchanged (c_mot/dyn structure; see
#     me_rssm/sanity/mci_gate_report.py for the log.json readout).
#
# Gate numbers are read from the work_dir log.json afterwards with:
#   python3 me_rssm/sanity/mci_gate_report.py <work_dir>
# plus the ego readout:
#   python3 me_rssm/sanity/dcrssm2_gate_report.py <work_dir>
#
# The ep16 probe (paired-toggle static alignment residual sign test) and
# the AP gate (ep12-16 >= 39.2809 AND >= 38.8786, Ped paired >= -1.5,
# Car/Truck no regression) are a separate, user-approved run.

_base_ = [
    '../configs/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_dcrssm2.py'
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
