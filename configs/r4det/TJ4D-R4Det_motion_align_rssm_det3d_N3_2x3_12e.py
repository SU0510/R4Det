# initialization
custom_imports = dict(imports=[
    'mmdet3d',
    'mmdet3d.core.hook.kl_scale_scheduler',
])
find_unused_parameters = False
# dataset settings
dataset_type = 'TJ4DDataset'
data_root = 'data/TJ4D/'
class_names = ['Pedestrian', 'Cyclist', 'Car', 'Truck']
input_modality = dict(use_lidar=True, use_camera=True)
file_client_args = dict(backend='disk')

# dataset BEV grid and pc range configs
point_cloud_range = [0, -39.68, -4, 69.12, 39.68, 2]
post_center_range = [x + y for x, y in zip(point_cloud_range, [-10, -10, -5, 10, 10, 5])]
voxel_size = [0.32, 0.32, 6.00]
grid_config = {
    'xbound': [point_cloud_range[0], point_cloud_range[3], voxel_size[0]],
    'ybound': [point_cloud_range[1], point_cloud_range[4], voxel_size[1]],
    'zbound': [point_cloud_range[2], point_cloud_range[5], voxel_size[2]],
    'dbound': [1.0, 73, 1.0]}
code_weights = [2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
code_size = len(code_weights)

# ablation settings (disabled for baseline)
use_backward_projection = False              # PDF disabled
use_sa_radarnet = False                      # SRP disabled
use_radar_depth = False                      # GDC disabled
backward_ablation = dict(pv_logits=True, depth_prob=True, cross=3, self=2)
focusradardepth_ablation = dict(foreground_loss_alpha=5.0)
painting_ablation = dict(pv_logits=False, depth_prob=True)

# supervision settings — only 3D detection kept
use_box3d_supervision = True                 # 3D object detection task
use_depth_supervision = False                # depth supervision disabled
use_props_supervision = False                # BEV proposal supervision disabled
use_msk2d_supervision = False                # 2D supervision disabled for baseline
use_inst_supervision = False                 # instance supervision disabled

# default settings
freeze_images = False                        # train image backbone
freeze_depths = False
freeze_radars = False
use_grid_mask = False
camera_stream = 'LSS'
loss_depth_prob = 0.0
loss_bev_seg = 0.0
loss_range_seg = 0.0

# image augmentation
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675],
    std=[1.0, 1.0, 1.0], to_rgb=False,
)
ida_aug_conf = {
    'resize_lim': (0.40, 0.50),
    'final_dim': (480, 640),
    'final_dim_test': (480, 640),
    'bot_pct_lim': (0.0, 0.0),
    'top_pct_lim': (0.0, 0.1),
    'rot_lim': (-2.7, 2.7),
    'rand_flip': True,
}
# BEVDataAugmentation
bda_aug_conf = dict(
    rot_range=(-0.3925, 0.3925),
    scale_ratio_range=(0.95, 1.05),
    translation_std=(1.0, 0.0, 0.0),
    flip_dx_ratio=0.0,
    flip_dy_ratio=0.5,
)
inst_feat_dim = 256
# model parameter
bev_h_ = int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])
bev_w_ = int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1])
img_channels = 256
rad_channels = 384
downsample = 8
num_in_height = 8
_num_layers_cross_ = backward_ablation['cross']
_num_points_pillr_ = 8
_num_points_cross_ = 8
_num_levels_ = 1
_num_cams_ = 1
_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
num_3d_proposals = 300

_num_layers_self_ = backward_ablation['self']
_num_points_self_ = 8

bev_h_ = int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])
bev_w_ = int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1])
bev_grid_size = [bev_h_, bev_w_, 1]

# model settings
model = dict(
    type='R4Det',
    mask_save_dir='./work_dirs/temp_mask_outputs/',
    bev_h_=bev_h_,
    bev_w_=bev_w_,
    img_channels=img_channels,
    rad_channels=rad_channels,
    num_classes=len(class_names),
    num_in_height=num_in_height,
    use_depth_supervision=use_depth_supervision,
    use_props_supervision=use_props_supervision,
    use_box3d_supervision=use_box3d_supervision,
    use_backward_projection=use_backward_projection,
    use_sa_radarnet=use_sa_radarnet,
    use_grid_mask=use_grid_mask,
    freeze_images=freeze_images,
    freeze_depths=freeze_depths,
    freeze_radars=freeze_radars,
    camera_stream=camera_stream,
    point_cloud_range=point_cloud_range,
    grid_config=grid_config,
    img_norm_cfg=img_norm_cfg,
    backward_ablation=backward_ablation,
    focusradardepth_ablation=focusradardepth_ablation,
    painting_ablation=painting_ablation,

    # NOTE: No img_rpn_head, no img_roi_head — 2D detection heads disabled
    # NOTE: No instance_feature_fusion_layer — IGDR disabled

    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe'),
    img_neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=img_channels,
        norm_cfg=dict(type='BN', requires_grad=False),
        num_outs=5),
    img_view_transformer=dict(
        type='ViewTransformerLSS',
        downsample=downsample,
        grid_config=grid_config,
        data_config=ida_aug_conf),
    depth_net=dict(
        type='GeometryDepth_Net',
        use_radar_depth=use_radar_depth,
        use_extra_depth=False,
        data_config=ida_aug_conf,
        downsample=downsample,
        input_channels=img_channels * 5,
        numC_input=_dim_,
        numC_Trans=_dim_,
        cam_channels=33,
        grid_config=grid_config,
        loss_depth_type='kld',
        loss_prob_weight=loss_depth_prob,
        loss_abs_weight=0.01,
        loss_sam2_weight=0.02,
        foreground_loss_alpha=focusradardepth_ablation['foreground_loss_alpha'],
        relative_loss_weight=0.04),
    rangeview_foreground=None,                  # disabled for baseline

    pts_voxel_layer=dict(
        max_num_points=10,
        point_cloud_range=point_cloud_range,
        voxel_size=[voxel_size[0]/2, voxel_size[1]/2, voxel_size[2]],
        max_voxels=(16000, 40000)),
    pts_voxel_encoder=dict(
        type='RadarPillarFeatureNet',
        in_channels=5,
        feat_channels=[64],
        with_distance=False,
        voxel_size=[voxel_size[0]/2, voxel_size[1]/2, voxel_size[2]],
        point_cloud_range=point_cloud_range,
        legacy=False,
        with_velocity_snr_center=True),
    pts_middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=[bev_w_*2, bev_h_*2]),
    pts_backbone=dict(
        type='SECOND',
        in_channels=64,
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[64, 128, 256]),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128]),

    # Simple concat+conv fusion instead of Cross_Modal_Fusion
    RCFusion=dict(
        type='ConcatConvFusion',
        img_channels=_dim_,
        rad_channels=rad_channels,
        out_channels=_dim_,
        kernel_size=3),
    proposal_layer=None,                         # disabled for baseline

    # temporal fusion with Motion-Aligned RSSM (deformable state alignment)
    temporal_fusion=dict(
        type='MotionAlignedRSSMFusion',
        in_channels=_dim_,
        out_channels=_dim_,
        kernel_size=3,
        latent_dim=_dim_,
        hidden_dim=64,
        action_dim=0,  # velocity branch disabled (TJ4D has no ego pose); set >0 + pass velocity to re-enable
        kl_scale=1.0,
        free_nats=1.0,
        min_std=0.1,
        init_std=0.2,
        align_kernel_size=3,
        align_deform_groups=1,
        align_z_state=True,
        norm_cfg=dict(
            type='BN',
            requires_grad=True
        ),
        act_cfg=dict(
            type='ReLU',
            inplace=True
        )
    ),
    # NOTE: backward_projection (PDF) removed — disabled
    seq_len=3,  # temporal sequence length (current + 2 history frames) for RSSM
    pts_bbox_head=dict(
        type='Anchor3DHead',
        num_classes=len(class_names),
        in_channels=_dim_,
        feat_channels=_dim_,
        use_direction_classifier=True,
        dir_offset=0.0,
        dir_limit_offset=1.0,
        anchor_generator=dict(
            type='Anchor3DRangeGenerator',
            ranges=[
                [0, -40.0, -1.163, 70.4, 40.0, -1.163],
                [0, -40.0, -1.353, 70.4, 40.0, -1.353],
                [0, -40.0, -1.363, 70.4, 40.0, -1.363],
                [0, -40.0, -1.403, 70.4, 40.0, -1.403],
            ],
            sizes=[[0.6, 0.8, 1.69], [0.78, 1.77, 1.60], [1.84, 4.56, 1.70],[2.66, 10.76, 3.47]],
            rotations=[0, 1.57],
            reshape_out=False),
        assigner_per_size=True,
        diff_rad_by_sin=True,
        assign_per_class=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
        loss_dir=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.2)),

    # model training and testing settings — removed all 2D training/test configs
    train_cfg=dict(
        pts=dict(
            assigner=[
                dict(  # for Pedestrian
                    type='MaxIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.35,
                    neg_iou_thr=0.2,
                    min_pos_iou=0.2,
                    ignore_iof_thr=-1),
                dict(  # for Cyclist
                    type='MaxIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.35,
                    neg_iou_thr=0.2,
                    min_pos_iou=0.2,
                    ignore_iof_thr=-1),
                dict(  # for Car
                    type='MaxIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.35,
                    min_pos_iou=0.35,
                    ignore_iof_thr=-1),
                dict(  # for Truck
                    type='MaxIoUAssigner',
                    iou_calculator=dict(type='BboxOverlapsNearest3D'),
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.35,
                    min_pos_iou=0.35,
                    ignore_iof_thr=-1),
            ],
            allowed_border=0,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        pts=dict(
            use_rotate_nms=True,
            nms_across_levels=False,
            nms_pre=1000,
            nms_post=num_3d_proposals,
            max_num=num_3d_proposals,
            score_thr=0.0,
            nms_thr=0.5,
            min_bbox_size=0
        )
    )
)

# pipeline settings — simplified: removed 2D-related transforms
train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=8, use_dim=[0,1,2,3,5]),
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ImageAug3D2', data_aug_conf=ida_aug_conf, is_train=True),
    dict(type='GlobalRotScaleTransFlipAll', bda_aug_conf=bda_aug_conf, is_train=True),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CustomCollect3D2', keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d'],
         meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                    'depth2img', 'cam2img', 'pad_shape',
                    'scale_factor', 'flip', 'pcd_horizontal_flip',
                    'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                    'img_norm_cfg', 'pcd_trans', 'sample_idx',
                    'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                    'transformation_3d_flow', 'img_aug_matrix', 'lidar_aug_matrix',
                    'lidar2cam', 'gt_depths', 'cam_aware', 'bda_rot')),
]
test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=8, use_dim=[0,1,2,3,5]),
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ImageAug3D2', data_aug_conf=ida_aug_conf, is_train=False),
    dict(type='GlobalRotScaleTransFlipAll', bda_aug_conf=bda_aug_conf, is_train=False),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(type='CustomCollect3D2', keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d'],
         meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                    'depth2img', 'cam2img', 'pad_shape',
                    'scale_factor', 'flip', 'pcd_horizontal_flip',
                    'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                    'img_norm_cfg', 'pcd_trans', 'sample_idx',
                    'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                    'transformation_3d_flow', 'img_aug_matrix', 'lidar_aug_matrix',
                    'lidar2cam', 'gt_depths', 'cam_aware', 'bda_rot')),
]
eval_pipeline = test_pipeline

# dataset settings
data = dict(
    samples_per_gpu=3,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'TJ4D_infos_train.pkl',
            split='training',
            pts_prefix='velodyne_reduced',
            pipeline=train_pipeline,
            modality=input_modality,
            classes=class_names,
            test_mode=False,
            box_type_3d='LiDAR',
            seq_len=3)),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'TJ4D_infos_val.pkl',
        split='training',
        pts_prefix='velodyne_reduced',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR',
        seq_len=3),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'TJ4D_infos_val.pkl',
        split='training',
        pts_prefix='velodyne_reduced',
        pipeline=test_pipeline,
        modality=input_modality,
        classes=class_names,
        test_mode=True,
        box_type_3d='LiDAR',
        seq_len=3))

# Training settings
lr = 0.00015
max_epochs = 18
optimizer = dict(type='AdamW', lr=lr, betas=(0.95, 0.99), weight_decay=0.01)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
runner = dict(type='EpochBasedRunner', max_epochs=max_epochs)
lr_config = dict(
    policy='CosineAnnealing',
    warmup=None,
    warmup_iters=500,
    warmup_ratio=1.0 / 10,
    min_lr_ratio=1e-5)
momentum_config = None

# KL scale warm-up: kl_scale linearly increases from 0 to 1.0 over epochs 0-5.
# Prior has zero KL gradient when kl_scale=0, letting posterior/detector stabilise
# before the KL term forces prior to match posterior.
custom_hooks = [
    dict(
        type='KLScaleSchedulerHook',
        start_epoch=0,
        end_epoch=5,
        start_value=0.0,
        end_value=1.0,
    ),
]

# log checkpoint & evaluation
evaluation = dict(interval=1, pipeline=eval_pipeline)
checkpoint_config = dict(interval=2)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
device = 'cuda'