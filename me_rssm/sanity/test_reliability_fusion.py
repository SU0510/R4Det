"""S0 unit tests for the reliability-gated concat fusion arm.

Frozen clauses (prereg docs/rssm_reliability_fusion_prereg_20261012.md):
  * zero-init gate => bitwise identical to ConcatConvFusion at step 0
  * gate is live (non-zero grad, leaves identity after one update)
  * alpha in (0, 2)
  * occupancy comes from the input point cloud only (x/y, GT-free) and follows
    the requested layout
  * parameter delta vs the baseline equals the gate's own parameter count
  * only opt-in fusions advertise ``accepts_obs_valid`` (wiring guard)
"""
import copy

import pytest
import torch

import me_rssm  # noqa: F401  (registers the arm)
from me_rssm.reliability_fusion_rssm import ReliabilityGatedConcatFusion
from mmdet3d.models.builder import FUSION_LAYERS
from mmdet3d.models.fusion_layers.concat_conv_fusion import ConcatConvFusion

PC_RANGE = [0, -39.68, -4, 69.12, 39.68, 2]
BASE_CFG = dict(img_channels=256, rad_channels=384, out_channels=256,
                kernel_size=3)
ARM_CFG = dict(BASE_CFG, point_cloud_range=PC_RANGE, valid_layout='yx')
H, W = 24, 20


def _build(kind, cfg):
    return FUSION_LAYERS.build(dict(type=kind, **cfg))


def _points(bsz=2, n=40, seed=0):
    g = torch.Generator().manual_seed(seed)
    return [torch.rand(n, 5, generator=g) *
            torch.tensor([60.0, 70.0, 1.0, 1.0, 1.0]) +
            torch.tensor([5.0, -35.0, -2.0, 0.0, 0.0]) for _ in range(bsz)]


def _feats(bsz=2, seed=1):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(bsz, 256, H, W, generator=g),
            torch.randn(bsz, 384, H, W, generator=g))


def _n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ------------------------------------------------------------------ identity
def test_zero_init_gate_is_bitwise_baseline():
    base = _build('ConcatConvFusion', BASE_CFG)
    arm = _build('ReliabilityGatedConcatFusion', ARM_CFG)
    arm.load_state_dict(
        {k: v for k, v in base.state_dict().items()}, strict=False)
    img, radar = _feats()
    with torch.no_grad():
        ref = base(img, radar)
        out = arm(img, radar, obs_points=_points())
    assert torch.equal(ref, out), 'gate must be exactly identity at init'


def test_no_points_reproduces_baseline():
    base = _build('ConcatConvFusion', BASE_CFG)
    arm = _build('ReliabilityGatedConcatFusion', ARM_CFG)
    arm.load_state_dict(base.state_dict(), strict=False)
    # make the gate non-trivial so the fallback is actually exercised
    torch.nn.init.normal_(arm.gate[-1].weight, std=0.5)
    img, radar = _feats()
    with torch.no_grad():
        assert torch.equal(base(img, radar), arm(img, radar))
        assert torch.equal(base(img, radar), arm(img, radar, obs_points=None))


# ------------------------------------------------------------------- liveness
def test_gate_is_live_and_alpha_is_bounded():
    arm = _build('ReliabilityGatedConcatFusion', ARM_CFG)
    img, radar = _feats()
    pts = _points()
    with torch.no_grad():
        alpha0 = arm.reliability_gate(radar, pts)
    assert torch.allclose(alpha0, torch.ones_like(alpha0))

    opt = torch.optim.SGD([p for p in arm.parameters() if p.requires_grad], lr=0.5)
    out = arm(img, radar, obs_points=pts)
    loss = out.pow(2).mean()
    opt.zero_grad()
    loss.backward()
    gnorm = sum(float(p.grad.abs().sum()) for p in arm.gate.parameters()
                if p.grad is not None)
    assert gnorm > 0, 'gate received no gradient'
    gate_grads = [p.grad for p in arm.gate.parameters()]
    assert all(g is not None for g in gate_grads)
    opt.step()
    with torch.no_grad():
        alpha1 = arm.reliability_gate(radar, pts)
    assert not torch.allclose(alpha1, torch.ones_like(alpha1)), 'gate did not move'
    assert float(alpha1.min()) > 0.0 and float(alpha1.max()) < 2.0


# ---------------------------------------------------------------------- mask
def test_occupancy_uses_points_only_and_respects_layout():
    arm = _build('ReliabilityGatedConcatFusion', ARM_CFG)
    x_min, y_min, _, x_max, y_max, _ = PC_RANGE
    # one point in the far-x / far-y corner, one in the near-x / near-y corner
    pts = [torch.tensor([
        [x_max - 0.5, y_max - 0.5, 0.0, 0.0, 0.0],
        [x_min + 0.5, y_min + 0.5, 0.0, 0.0, 0.0],
    ])]
    radar = torch.zeros(1, 384, H, W)
    occ = arm.build_occupancy(pts, radar)
    assert occ.shape == (1, 1, H, W)
    assert float(occ.sum()) == 2.0
    # layout 'yx': rows are y cells, columns are x cells, both ascending
    assert occ[0, 0, H - 1, W - 1] == 1.0    # y_max, x_max -> last row/col
    assert occ[0, 0, 0, 0] == 1.0            # y_min, x_min -> first row/col


def test_layout_switch_transposes_the_mask():
    yx = _build('ReliabilityGatedConcatFusion', ARM_CFG)
    xy = _build('ReliabilityGatedConcatFusion',
                dict(BASE_CFG, point_cloud_range=PC_RANGE, valid_layout='xy'))
    pts = [torch.tensor([[60.0, 30.0, 0.0, 0.0, 0.0]])]
    # on a SQUARE grid the two layouts are exact transposes of each other
    radar = torch.zeros(1, 384, 16, 16)
    a = yx.build_occupancy(pts, radar)
    b = xy.build_occupancy(pts, radar)
    assert float(a.sum()) == float(b.sum()) == 1.0
    assert torch.equal(a[0, 0], b[0, 0].t())
    assert not torch.equal(a[0, 0], b[0, 0]), 'layouts must actually differ'


def test_bad_layout_raises():
    with pytest.raises(ValueError):
        _build('ReliabilityGatedConcatFusion',
               dict(BASE_CFG, valid_layout='zz'))


# ------------------------------------------------------------------ capacity
def test_parameter_delta_equals_gate_size():
    base = _build('ConcatConvFusion', BASE_CFG)
    arm = _build('ReliabilityGatedConcatFusion', ARM_CFG)
    gate = sum(p.numel() for p in arm.gate.parameters())
    assert gate == 3 * 16 * 9 + 16 + 16 * 1 * 9 + 1 == 593
    assert _n_params(arm) - _n_params(base) == gate


# -------------------------------------------------------------------- wiring
def test_only_opt_in_fusions_accept_obs_valid():
    assert ReliabilityGatedConcatFusion.accepts_obs_valid is True
    assert not getattr(ConcatConvFusion, 'accepts_obs_valid', False)
