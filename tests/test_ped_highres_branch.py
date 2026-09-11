import pytest
import torch
from torch import nn

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
    fused = torch.randn(2, 8, 6, 4)
    out = branch(radar, fused)

    assert out.shape == (2, 8, 12, 8)
    out.square().mean().backward()
    assert all(
        param.grad is not None and torch.isfinite(param.grad).all()
        for param in branch.parameters())


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
    pts_feats = [torch.randn(2, 8, 6, 4, requires_grad=True)]

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
