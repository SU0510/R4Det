# Restore the seed_1 clean full-RSSM BEST (epoch 15, Overall 40.51), which the
# original 24e run never checkpointed because its checkpoint_interval was 2.
# Resume from seed_1 epoch_14 with checkpoint_interval 1 and the unchanged 24e
# cosine schedule so epoch 15 reproduces exactly. Stop the run as soon as
# epoch_15.pth is written; this is a checkpoint-recovery run, not new training.
_base_ = './TJ4D-R4Det_motion_align_rssm_det3d_N4_2x4_24e_pretrained_v2_head.py'

load_from = None
resume_from = '/data/lurui/work_dirs/run10_headv2_multiseed/seed_1/epoch_14.pth'
checkpoint_config = dict(interval=1)
work_dir = '/data/lurui/work_dirs/seed1_ep15_restore'
