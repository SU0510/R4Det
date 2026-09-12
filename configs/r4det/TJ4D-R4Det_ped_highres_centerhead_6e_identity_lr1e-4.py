_base_ = './TJ4D-R4Det_ped_highres_centerhead_3x2x2_3e_raw.py'

seed = 0
lr = 1e-4
max_epochs = 6
optimizer = dict(type='AdamW', lr=lr, betas=(0.95, 0.99), weight_decay=0.01)
runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)

work_dir = '/data/lurui/work_dirs/ped_highres_centerhead_identity_3x2x2_6e_lr1e-4_seed0'
load_from = '/data/lurui/work_dirs/ped_centerhead_stage1_3x2x2_12e_seed0/epoch_7.pth'
resume_from = None
