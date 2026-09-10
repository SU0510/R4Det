import math

import pytest
import torch
import numpy as np

from mmdet3d.core.evaluation.vod_utils.rotate_iou_cpu import (
    rotate_iou_eval)

# Importing the model package also imports CUDA-only ops which assert that a
# GPU is visible. The unit test itself is CPU-only, so only satisfy that import
# guard and restore CUDA detection immediately afterward.
_cuda_is_available = torch.cuda.is_available
try:
    torch.cuda.is_available = lambda: True
    from mmdet3d.models.builder import build_head
finally:
    torch.cuda.is_available = _cuda_is_available


def _make_head():
    cfg = dict(
        type='CenterHeadkitti',
        in_channels=2,
        size_prior_xy=[0.655454, 0.627535],
        max_log_residual=0.25,
        soft_prior_alpha=0.75,
        delta_log_residual=0.15,
        delta_channels=2,
        delta_num_convs=1,
        delta_kernel_size=1,
        size_l1_weight=0.1,
        share_conv_channel=2,
        num_heatmap_convs=1,
        tasks=[dict(num_class=1, class_names=['Pedestrian'])],
        common_heads=dict(
            reg=(2, 1), height=(1, 1), dim=(3, 1), rot=(2, 1)),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=1),
        norm_cfg=dict(type='BN2d'),
        norm_bbox=True,
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='none', loss_weight=0.25),
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            pc_range=[0, -2, -1, 2, 2, 1],
            post_center_range=[-1, -2, -2, 1, 2, 2],
            max_num=4,
            score_threshold=0.0,
            out_size_factor=1,
            voxel_size=[0.5, 0.5],
            code_size=7),
        train_cfg=dict(
            grid_size=[4, 8, 1],
            voxel_size=[0.5, 0.5],
            point_cloud_range=[0, -2, -1, 2, 2, 1],
            out_size_factor=1,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=4,
            min_radius=1,
            code_weights=[1.0] * 8),
        test_cfg=dict(
            post_center_limit_range=[-1, -2, -2, 1, 2, 2],
            max_per_img=4,
            max_pool_nms=False,
            min_radius=[1],
            nms_pre=4,
            score_threshold=0.05))
    return build_head(cfg)


def test_iou_decode_shape_identity_and_delta_gradient():
    head = _make_head()
    batch, height, width = 1, 8, 4
    center = torch.tensor([[0.5, -1.0]], dtype=torch.float32)
    ind = torch.tensor([[5, 5, 5, 5]], dtype=torch.long)
    size = torch.tensor([[0.5, 0.5, 1.0]], dtype=torch.float32)
    yaw = torch.tensor([[0.0]], dtype=torch.float32)

    z = torch.tensor([[0.0]], dtype=torch.float32)
    target = torch.cat(
        (torch.tensor([[0.1, -0.2]]), z, size.log(),
         torch.sin(yaw), torch.cos(yaw)), dim=-1)
    target_box = target.expand(batch, 4, 8)
    pred = target_box.reshape(1, 4, 8).requires_grad_(True)

    final_log_xy = torch.full(
        (batch, 2, height, width), -math.inf, requires_grad=True)
    final_log_xy = torch.where(
        torch.arange(height * width).view(1, 1, height, width).eq(5),
        size[0, :2].log().view(2, 1, 1), final_log_xy)
    final_log_xy.retain_grad()

    mask = torch.zeros((batch, 4, 8), dtype=torch.uint8)
    mask[0, 0, :] = 1

    pred_boxes, target_boxes = head._positive_boxes_for_iou(
        pred, target_box, mask, ind, final_log_xy, width)

    assert pred_boxes.shape[-1] == 7
    assert target_boxes.shape[-1] == 7
    assert torch.allclose(pred_boxes[..., 6], yaw.flatten(), atol=1e-6)
    bev_boxes = pred_boxes[..., [0, 1, 3, 4, 6]].detach().numpy()
    iou_np = np.diag(rotate_iou_eval(bev_boxes, bev_boxes))
    assert np.allclose(iou_np, 1.0, atol=2e-6)

    final_log_xy = final_log_xy + 0.01
    final_log_xy.retain_grad()
    pred_boxes, target_boxes = head._positive_boxes_for_iou(
        pred, target_box, mask, ind, final_log_xy, width)

    # Differentiable IoU surrogate for CPU gradient checking. The actual
    # training loss uses mmcv's CUDA diff_iou_rotated_3d on the same decoded
    # 7D boxes.
    center_loss = (pred_boxes[..., :3] - target_boxes[..., :3]).square().sum()
    size_loss = (pred_boxes[..., 3:6] - target_boxes[..., 3:6]).square().sum()
    yaw_loss = (pred_boxes[..., 6] - target_boxes[..., 6]).square().sum()
    (center_loss + size_loss + yaw_loss).backward()
    assert torch.isfinite(final_log_xy.grad).all()
    assert final_log_xy.grad.abs().sum() > 0

    pred_shifted = pred_boxes.clone()
    pred_shifted[..., 0] += 0.1
    shifted = pred_shifted[..., [0, 1, 3, 4, 6]].detach().numpy()
    shifted_np = np.diag(rotate_iou_eval(shifted, bev_boxes))
    assert np.all(shifted_np < iou_np)


if __name__ == '__main__':
    pytest.main([__file__])
