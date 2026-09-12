import pytest
import torch
from torch import nn
from mmcv import Config

_cuda_is_available = torch.cuda.is_available
try:
    torch.cuda.is_available = lambda: True
    from mmdet3d.models.fusion_layers.ped_highres_branch import (
        PedHighresBranch)
    from mmdet3d.models.detectors.R4Det import R4Det
finally:
    torch.cuda.is_available = _cuda_is_available


def _branch():
    return PedHighresBranch(
        radar_channels=4,
        fused_channels=8,
        out_channels=8,
        channels=4,
        num_convs=2,
        kernel_size=3)


def test_ped_highres_branch_output_shape_and_gradient():
    branch = _branch()
    radar = torch.randn(2, 4, 12, 8)
    fused = torch.randn(2, 8, 4, 6)
    out = branch(radar, fused)

    assert out.shape == (2, 8, 12, 8)
    out.square().mean().backward()
    assert all(
        param.grad is not None and torch.isfinite(param.grad).all()
        for param in branch.parameters())


def test_ped_highres_branch_residual_gets_nonzero_gradient():
    branch = _branch()
    branch.train()
    radar = torch.randn(2, 4, 12, 8)
    fused = torch.randn(2, 8, 4, 6)

    out = branch(radar, fused)
    out.square().mean().backward()

    assert branch.fusion_conv.weight.grad.abs().sum() > 0

    optimizer = torch.optim.SGD(branch.parameters(), lr=0.1)
    optimizer.step()
    optimizer.zero_grad()
    branch(radar, fused).square().mean().backward()

    assert branch.radar_conv.conv.weight.grad.abs().sum() > 0


def test_ped_highres_branch_identity_initialization():
    branch = _branch()
    branch.eval()
    radar = torch.randn(2, 4, 12, 8)
    fused = torch.randn(2, 8, 4, 6)
    expected = torch.nn.functional.interpolate(
            fused,
            size=(12, 8),
            mode='nearest')
    with torch.no_grad():
        out = branch(radar, fused)

    assert torch.allclose(out, expected, atol=1e-6, rtol=1e-6)
    assert torch.count_nonzero(branch.fusion_conv.weight) == 0
    assert torch.count_nonzero(branch.fusion_conv.bias) == 0


def test_ped_highres_centerhead_grid_contract():
    cfg = Config.fromfile(
        'configs/r4det/TJ4D-R4Det_ped_highres_centerhead_3x2x2_3e_raw.py')
    head_cfg = cfg.model.ped_center_head
    expected_grid = [cfg.bev_h_ * 2, cfg.bev_w_ * 2, 1]

    assert head_cfg.train_cfg.grid_size == expected_grid
    # Scatter and fused BEV are both [H_y, W_x]. CenterPoint stores
    # grid_size as [W, H], so the doubled contract is [432, 496].
    assert head_cfg.train_cfg.voxel_size == [
        cfg.voxel_size[0] / 2, cfg.voxel_size[1] / 2]
    assert head_cfg.train_cfg.out_size_factor == 1
    assert (cfg.bev_h_, cfg.bev_w_) == (216, 248)
    assert cfg.model.pts_middle_encoder.output_shape == [496, 432]


def test_ped_highres_freeze_scope_and_detached_inputs():
    detector = R4Det.__new__(R4Det)
    detector.__dict__['_parameters'] = {}
    detector.__dict__['_buffers'] = {}
    detector.__dict__['_modules'] = {}
    detector.__dict__['_non_persistent_buffers_set'] = set()
    detector.highres_ped_branch = _branch()
    detector.ped_center_head = nn.Linear(8, 8)
    detector.frozen = nn.Linear(8, 8)
    detector._highres_radar_scatter = torch.randn(
        2, 4, 12, 8, requires_grad=True)
    pts_feats = [torch.randn(2, 8, 4, 6, requires_grad=True)]

    R4Det._freeze_for_ped_highres_stage(detector)
    ped_feats = R4Det._make_ped_highres_feats(detector, pts_feats)
    ped_feats[0].square().mean().backward()

    assert detector.frozen.weight.requires_grad is False
    assert detector.ped_center_head.weight.requires_grad is True
    assert all(
        param.requires_grad
        for param in detector.highres_ped_branch.parameters())
    assert detector._highres_radar_scatter.grad is None
    assert pts_feats[0].grad is None
    assert detector._frozen_eval_modules
    assert all(
        module.training is False
        for module in detector._frozen_eval_modules)


if __name__ == '__main__':
    pytest.main([__file__])
