# ---------------------------------------------------------------------------
# FG-FULL config: enable ALL foreground supervision used during pretraining
# (msk2d range-view foreground, FRPN BEV proposal mask, 2D instance branch)
# plus the IGDR Foreground-Gated Fusion, on top of the clean mainline.
#
# BASE: me_rssm/configs/TJ4D-R4Det_clean_N4_2x4_24e_pretrained_v2_head_snapshot.py
#   (= Run 10 head-v2 multiseed mainline, 40.42 +- 0.51, shared_stem=False).
#   Every training hyper-parameter (lr, schedule, batch, seed protocol) is
#   inherited unchanged; only foreground/IGDR modules + their data pipeline
#   are switched on, so the run is directly comparable to mainline seed_0.
#
# What is enabled here (all hot-start from checkpoints/pretrained_tj4d.pth):
#   1. img_rpn_head + img_roi_head  -> joint 2D branch training
#      (RPN + bbox + mask losses on 2D GT; IGDR consumes its proposals);
#      building these two also auto-builds `igdr_fusion` (R4Det.py:350),
#      i.e. the paper's IGDR Foreground-Gated Fusion (G_bg) becomes active
#      in train and test forward (R4Det.py:1481 / 1182).
#   2. use_msk2d_supervision=True   -> rangeview_foreground (MRF3Net)
#      range-view foreground supervision.
#   3. use_props_supervision=True   -> proposal_layer (FRPN) BEV foreground
#      mask supervision (former + latter).
#   4. use_depth_supervision=True   -> brings back depth_net.get_depth_loss;
#      with my_gt_depth + segmentation restored in the pipeline the
#      foreground-biased Structural Ranking loss activates
#      (GeometryDepth_Net.py:198-243, L_edge 2000 pairs + L_global 1000
#      pairs, w 1.5/0.5). loss_abs_weight/loss_sam2_weight are zeroed here
#      so the ONLY added depth loss is the foreground-biased one (flip them
#      back to 0.01/0.02 to restore the full pretrain depth family).
#
# Data notes (verified on this machine):
#   - segmentation npy:            data/TJ4D/segmentation/<idx>.npy   (exists)
#   - VLSAM ann/masks (gt_masks):  /data/yanzexin/TJ4D/{annotations,masks}
#     (the tangyousen default paths baked into LoadVLSAMAnnotations do NOT
#     exist here -> overridden below)
#   - my_gt_depth npy:             /data/TJ4D/training/depth_npy_predict
#     (hardlink-copy of /data/yanzexin/TJ4D/training/depth_npy_predict)
#   - DownsampleDepthMap default target (60, 80) matches the base config's
#     480x640 images / downsample=8 (pretrain used 96,128 for 768x1024).
#
# Launch (identical protocol to run10_headv2_multiseed/seed_0):
#   CUDA_VISIBLE_DEVICES=5,6,7 bash tools/dist_train.sh <this config> 3 \
#     --seed 0 --deterministic --work-dir /data/lurui/work_dirs/fgfull_N4_2x4_24e_seed0
# ---------------------------------------------------------------------------

_base_ = ['../../me_rssm/configs/'
          'TJ4D-R4Det_clean_N4_2x4_24e_pretrained_v2_head_snapshot.py']

data_root = 'data/TJ4D/'
class_names = ['Pedestrian', 'Cyclist', 'Car', 'Truck']
point_cloud_range = [0, -39.68, -4, 69.12, 39.68, 2]
_dim_ = 256

# loss weights for the newly enabled foreground branches (pretrain values)
loss_bev_seg = 1.0
loss_range_seg = 0.1

# img_rpn_head/img_roi_head params receive no gradients (2D forward runs
# under no_grad for IGDR; their losses are the only consumers, which DDP
# would flag as unused without this flag)
find_unused_parameters = True

model = dict(
    use_depth_supervision=True,
    use_props_supervision=True,
    use_msk2d_supervision=True,

    # ---- 2D instance branch (also the IGDR proposal source) --------------
    img_rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
    img_roi_head=dict(
        type='StandardRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=4,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='L1Loss', loss_weight=1.0)),
        mask_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=14, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=4,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0))),

    # ---- BEV / range-view foreground supervision -------------------------
    rangeview_foreground=dict(
        _delete_=True,  # base value is None -> replace, not merge
        type='MRF3Net',
        input_channel=_dim_,
        output_channel=1,
        base_channel=_dim_,
        mask_thre_train=0.95,
        mask_thre_test=0.70,
        loss_box=loss_range_seg,
        loss_seg=loss_range_seg),
    proposal_layer=dict(
        _delete_=True,  # base value is None -> replace, not merge
        type='FRPN',
        in_channels=_dim_,
        scale_factor=1.0,
        mask_thre=0.4,
        topk_rate_test=0.01,
        loss_weight=loss_bev_seg),

    # ---- depth loss family: keep ONLY the foreground-biased part ---------
    depth_net=dict(
        loss_abs_weight=0.0,     # pretrain had 0.01 (L_abs, not fg-specific)
        loss_sam2_weight=0.0,    # pretrain had 0.02 (L_dense, not fg-specific)
        # relative_loss_weight=0.04 inherited from base -> L_relative active
    ),

    # ---- 2D assigner/sampler cfg (verbatim from the pretrain config) -----
    train_cfg=dict(
        img_rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        img_rpn_proposal=dict(
            nms_across_levels=False,
            nms_pre=2000,
            nms_post=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        img_rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            mask_size=28,
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        img_rpn=dict(
            nms_across_levels=False,
            nms_pre=1000,
            nms_post=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        img_rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100,
            mask_thr_binary=0.5)),
)

# ---------------------------------------------------------------------------
# Train pipeline: base pipeline + the 2D/foreground data chain restored from
# the pretrain config (order preserved). Image resolution is 480x640 here
# (not 768x1024 as in pretrain), so DownsampleDepthMap uses its default
# (60, 80) target = 480/8 x 640/8, aligned with the depth-net output size.
# ---------------------------------------------------------------------------
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
bda_aug_conf = dict(
    rot_range=(-0.3925, 0.3925),
    scale_ratio_range=(0.95, 1.05),
    translation_std=(1.0, 0.0, 0.0),
    flip_dx_ratio=0.0,
    flip_dy_ratio=0.5,
)

train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=8, use_dim=[0, 1, 2, 3, 5]),
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='loadSegmentation', data_root=data_root, dataset='TJ4D', seg_type='detectron2'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True,
         with_bbox=True, with_label=True),
    dict(type='LoadVLSAMAnnotations',
         ann_root_path='/data/yanzexin/TJ4D/annotations/',
         mask_root_path='/data/yanzexin/TJ4D/'),
    dict(type='LoadMyDepthFromFile',
         depth_base_path='/data/TJ4D/training/depth_npy_predict'),
    dict(type='ImageAug3D2', data_aug_conf=ida_aug_conf, is_train=True),
    dict(type='DownsampleDepthMap', target_shape=(60, 80)),
    dict(type='GlobalRotScaleTransFlipAll', bda_aug_conf=bda_aug_conf, is_train=True),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='CreateDepthFromLiDAR', data_root=data_root, dataset='TJ4D'),
    dict(type='gen2DMask', use_seg=False, use_softlabel=False, is_train=True),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CustomCollect3D2',
         keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d',
               'gt_bboxes', 'gt_labels', 'gt_masks', 'my_gt_depth'],
         meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                    'depth2img', 'cam2img', 'pad_shape',
                    'scale_factor', 'flip', 'pcd_horizontal_flip',
                    'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                    'img_norm_cfg', 'pcd_trans', 'sample_idx',
                    'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                    'transformation_3d_flow', 'img_aug_matrix', 'lidar_aug_matrix',
                    'lidar2cam', 'gt_depths', 'cam_aware', 'bda_rot',
                    'segmentation', 'bbox_Mask')
         ),
]

data = dict(
    train=dict(dataset=dict(pipeline=train_pipeline)),
    # Validation batch size.  mmdet3d/apis/train.py pops `samples_per_gpu`
    # from cfg.data.val and, when > 1, swaps ImageToTensor -> DefaultFormatBundle
    # so that a whole batch of sequences is collated into one [B, N, C, H, W]
    # tensor.  R4Det.forward_test/simple_test/preprocessing_information and
    # _build_bev_instance_map (IGDR) handle that layout for any B.
    #
    # Verified on the full 2040-sample val split, same checkpoint, same code
    # (AP and peak GPU are stable across the three separate runs):
    #   bs=1: peak 2.36 GiB, 3D mod 38.4516, BEV mod 45.8789
    #   bs=2: peak 3.11 GiB, 3D mod 38.4947, BEV mod 45.9299
    #   bs=4: peak 5.92 GiB, 3D mod 38.5420, BEV mod 45.8670
    # AP moves are within run-to-run noise; no batch cross-talk (a sample's
    # output is bit-identical when its batch partners change).  bs=1 numbers
    # stay bitwise reproducible, so already-recorded val points remain valid.
    # Speed (interleaved, workers=2): bs=1/2/4 = 271.8/238.6/230.9 ms/sample
    # => 1.14x/1.18x.  Val stays dataloader-bound, so this is a modest win.
    # NOTE: bs>4 is not used - bs=6 OOMs and nothing above 4 was faster.
    val=dict(samples_per_gpu=4),
)

# match the run10 multiseed seed_0 checkpoint policy (interval=2; interval
# does not affect training results, halves disk usage: 12 x ~0.6 GB)
checkpoint_config = dict(interval=2)
