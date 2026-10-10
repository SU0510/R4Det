# 450-iter DDP smoke for candidate A (z-semantic supervision + cls residual).
# Mechanism/wiring gate only — NO val, so no interval mean / peak / best saved,
# and no AP claim whatsoever (prereg §5 S0).
#
# Gates (read from the work_dir log.json + the wiring probe):
#   S0-c  loss_rssm_sem finite and DOWN over the run; no NaN; no OOM; the
#         run reaches iter 450 and writes iter_450.pth.
#   S0-d  pressure: ||d loss_rssm_sem / d z_t|| / ||d total / d z_t|| >= 5%
#         (me_rssm/sanity/probe_semreadout_wiring.py; this is the direct
#         lesson of §70.2 — a supervision term with no gradient pressure
#         cannot move anything, and 0.18%-of-total recon is the failure we are
#         NOT allowed to repeat).
#   S0-e  stat_cls_res_rms leaves 0 at step 0 (the zero-init identity
#         contract) and becomes NON-zero once W_cls moves — i.e. the readout
#         is alive and the residual reaches the class logits, not just the
#         log dict. ||d loss_cls / d W_cls|| > 0 is the direct reading and
#         is checked by probe_semreadout_wiring.py.
#   baseline path untouched: loss_rssm_recon / loss_rssm_kl behave as in the
#         baseline smoke ranges, and the first iterations show the class
#         logits identical to the baseline because W_cls starts at 0.
_base_ = [
    '../configs/'
    'TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_semreadout.py'
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
