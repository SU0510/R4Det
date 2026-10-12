"""S0 regression tests for the C1 arm (pre-registration 2026-10-12).

Covers the frozen invariants:
  * input-only mask construction (no GT path; only x/y matter),
  * zero-init identity contract (bitwise equal to the mainline at init),
  * gate liveness + non-zero gradient (no idle parameters),
  * bypass switches reproduce the baseline exactly,
  * capacity delta of the arm vs the mainline (prereg section 3),
  * wiring guard: only fusions declaring ``accepts_obs_valid`` get points.
"""
import copy

import pytest
import torch

import me_rssm  # noqa: F401  (registers the arms)
from me_rssm.validity_gain_rssm import (
    ValidityGatedDeterministicRSSMFusion, ValidityGatedRSSMFusion,
    build_radar_validity_mask)
from mmdet3d.models.builder import FUSION_LAYERS
from mmdet3d.models.fusion_layers.rssm_fusion import (
    DeterministicMotionAlignedLatentFusion, MotionAlignedRSSMFusion)

PC_RANGE = [0, -39.68, -4, 69.12, 39.68, 2]
SMALL_CFG = dict(
    in_channels=8, out_channels=8, kernel_size=3, latent_dim=8, hidden_dim=8,
    action_dim=0, kl_scale=1.0, free_nats=1.0, min_std=0.1, init_std=0.2,
    align_kernel_size=3, align_deform_groups=1, align_z_state=True,
    norm_cfg=dict(type='BN', requires_grad=True),
    act_cfg=dict(type='ReLU', inplace=True))
FULL_CFG = dict(SMALL_CFG, in_channels=256, out_channels=256, latent_dim=256)
DET_KEYS = ('in_channels', 'out_channels', 'kernel_size', 'latent_dim',
            'hidden_dim', 'action_dim', 'norm_cfg', 'act_cfg',
            'align_kernel_size', 'align_deform_groups', 'align_z_state')


def _det_cfg(cfg):
    return {k: v for k, v in cfg.items() if k in DET_KEYS}


def _build(type_name, cfg=None, **extra):
    return FUSION_LAYERS.build(
        dict(type=type_name, **(cfg or SMALL_CFG), **extra))


def _copy_shared_weights(dst, src):
    sd_src, sd_dst = src.state_dict(), dst.state_dict()
    dst.load_state_dict(
        {k: (sd_src[k] if k in sd_src else v) for k, v in sd_dst.items()},
        strict=True)


def _points():
    return [torch.tensor([[10.0, -5.0, 0.0, 1.0, 3.0],
                          [40.0, 10.0, 0.0, 2.0, 5.0]]),
            torch.zeros(0, 5)]


# ---------------------------------------------------------------- mask builder
def test_mask_is_input_only_and_uses_xy():
    pts_a = _points()
    pts_b = [p.clone() for p in pts_a]
    for p in pts_b:
        if p.numel():
            p[:, 2:] = 123.0  # z / velocity / SNR must not matter
    m_a = build_radar_validity_mask(pts_a, 2, 7, 6, PC_RANGE, layout='yx')
    m_b = build_radar_validity_mask(pts_b, 2, 7, 6, PC_RANGE, layout='yx')
    assert torch.equal(m_a, m_b)
    assert m_a.shape == (2, 1, 7, 6) and m_a.dtype == torch.float32
    assert set(torch.unique(m_a).tolist()) <= {0.0, 1.0}
    # empty sample stays empty; the two points land in two distinct cells
    assert float(m_a[1].sum()) == 0.0 and float(m_a[0].sum()) == 2.0


def test_mask_layout_transpose_relation():
    pts = _points()
    m_yx = build_radar_validity_mask(pts, 2, 7, 6, PC_RANGE, layout='yx')
    m_xy = build_radar_validity_mask(pts, 2, 6, 7, PC_RANGE, layout='xy')
    assert m_xy.shape == (2, 1, 6, 7)
    assert torch.equal(m_xy, m_yx.transpose(-1, -2))


def test_mask_ignores_out_of_range_points():
    pts = [torch.tensor([[1e6, 1e6, 0, 0, 0], [5.0, 0.0, 0, 0, 0]])]
    m = build_radar_validity_mask(pts, 1, 7, 6, PC_RANGE, layout='yx')
    assert float(m.sum()) == 1.0


def test_mask_rejects_bad_layout_and_shape():
    with pytest.raises(ValueError):
        build_radar_validity_mask(_points(), 2, 7, 6, PC_RANGE, layout='nope')
    with pytest.raises(ValueError):
        build_radar_validity_mask([torch.zeros(3)], 1, 7, 6, PC_RANGE)


# ------------------------------------------------------------------- the arms
def test_identity_at_init_is_bitwise():
    torch.manual_seed(0)
    base = _build('MotionAlignedRSSMFusion').eval()
    arm = _build('ValidityGatedRSSMFusion').eval()
    _copy_shared_weights(arm, base)
    feat = torch.randn(2, 8, 6, 7)
    with torch.no_grad():
        base.reset_state()
        arm.reset_state()
        o_base = base(feat.clone(), deterministic=True)[0]
        o_arm = arm(feat.clone(), deterministic=True, obs_points=_points())[0]
    assert torch.equal(o_base, o_arm)


def test_gate_weights_are_zero_at_init_and_then_live():
    arm = _build('ValidityGatedRSSMFusion')
    assert float(arm.validity_gain.weight.abs().sum()) == 0.0
    assert float(arm.validity_gain.bias.abs().sum()) == 0.0
    assert arm.validity_gain.weight.numel() == (8 + 1) * 8
    base = _build('MotionAlignedRSSMFusion').eval()
    _copy_shared_weights(arm, base)
    arm.eval()
    feat = torch.randn(2, 8, 6, 7)
    with torch.no_grad():
        base.reset_state()
        arm.reset_state()
        o_base = base(feat.clone(), deterministic=True)[0]
        arm.validity_gain.weight.normal_(0, 0.5)
        arm.validity_gain.bias.normal_(0, 0.5)
        arm.reset_state()
        o_arm = arm(feat.clone(), deterministic=True, obs_points=_points())[0]
    assert float((o_base - o_arm).abs().max()) > 1e-3


def test_gate_receives_gradient():
    arm = _build('ValidityGatedRSSMFusion').train()
    out = arm(torch.randn(2, 8, 6, 7), deterministic=True,
              obs_points=_points())[0]
    out.sum().backward()
    grad = arm.validity_gain.weight.grad
    assert grad is not None and float(grad.abs().sum()) > 0.0


def test_gate_stats_are_tensors_and_identity_at_init():
    arm = _build('ValidityGatedRSSMFusion').eval()
    out = arm(torch.randn(2, 8, 6, 7), deterministic=True,
              obs_points=_points())
    stats = out[5]
    assert torch.is_tensor(stats['v_gate_mean'])
    assert abs(float(stats['v_gate_mean']) - 1.0) < 1e-6
    assert abs(float(stats['v_gate_valid_mean']) - 1.0) < 1e-6
    assert abs(float(stats['v_gate_invalid_mean']) - 1.0) < 1e-6


def test_bypass_switch_and_missing_mask_equal_baseline():
    torch.manual_seed(1)
    base = _build('MotionAlignedRSSMFusion').eval()
    arm = _build('ValidityGatedRSSMFusion').eval()
    _copy_shared_weights(arm, base)
    with torch.no_grad():  # non-identity gate weights
        arm.validity_gain.weight.normal_(0, 0.5)
        arm.validity_gain.bias.normal_(0, 0.5)
    feat = torch.randn(2, 8, 6, 7)
    with torch.no_grad():
        base.reset_state()
        o_base = base(feat.clone(), deterministic=True)[0]
        arm.reset_state()
        arm.use_validity_gain = False
        o_bypass = arm(feat.clone(), deterministic=True,
                       obs_points=_points())[0]
        arm.reset_state()
        arm.use_validity_gain = True
        o_nomask = arm(feat.clone(), deterministic=True)[0]
    assert torch.equal(o_base, o_bypass)
    assert torch.equal(o_base, o_nomask)


def test_mask_is_resized_when_grid_mismatch():
    arm = _build('ValidityGatedRSSMFusion').eval()
    mask = torch.zeros(2, 1, 3, 3)
    out = arm(torch.randn(2, 8, 6, 7), deterministic=True, obs_valid=mask)
    assert out[0].shape == (2, 8, 6, 7)


# ------------------------------------------------------- deterministic control
def test_deterministic_arm_identity_and_gradient():
    torch.manual_seed(2)
    base = _build('DeterministicMotionAlignedLatentFusion',
                  _det_cfg(SMALL_CFG)).eval()
    arm = _build('ValidityGatedDeterministicRSSMFusion',
                 _det_cfg(SMALL_CFG)).eval()
    _copy_shared_weights(arm, base)
    feat = torch.randn(2, 8, 6, 7)
    with torch.no_grad():
        base.reset_state()
        arm.reset_state()
        o_base = base(feat.clone(), deterministic=True)[0]
        o_arm = arm(feat.clone(), deterministic=True, obs_points=_points())[0]
    assert torch.equal(o_base, o_arm)
    assert out_is_none(o_arm is not None and
                       arm(feat.clone(), deterministic=True,
                           obs_points=_points())[2])
    arm.train()
    arm(torch.randn(2, 8, 6, 7), deterministic=True,
        obs_points=_points())[0].sum().backward()
    assert float(arm.validity_gain.weight.grad.abs().sum()) > 0.0


def out_is_none(value):
    return value is None


# ------------------------------------------------------------------- capacity
def test_capacity_delta_matches_prereg():
    base = _build('MotionAlignedRSSMFusion', FULL_CFG)
    arm = _build('ValidityGatedRSSMFusion', FULL_CFG)
    det = _build('ValidityGatedDeterministicRSSMFusion', _det_cfg(FULL_CFG))
    n = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)
    gain = sum(p.numel() for p in arm.validity_gain.parameters())
    assert gain == (256 + 1) * 256 + 256 == 66048
    assert n(base) == 9191510
    assert n(arm) - n(base) == gain
    # measured 2026-10-12: the deterministic control drops the stochastic
    # branch (2360064 active params) and carries the same gain.
    det_base = _build('DeterministicMotionAlignedLatentFusion',
                      _det_cfg(FULL_CFG))
    assert n(det_base) == 6831446
    assert n(det) - n(det_base) == gain
    assert n(arm) - n(det) == 2360064


# -------------------------------------------------------------------- wiring
def test_only_opt_in_fusions_accept_obs_valid():
    arm = _build('ValidityGatedRSSMFusion')
    det = _build('ValidityGatedDeterministicRSSMFusion', _det_cfg(SMALL_CFG))
    base = _build('MotionAlignedRSSMFusion')
    assert arm.accepts_obs_valid is True
    assert det.accepts_obs_valid is True
    assert not getattr(base, 'accepts_obs_valid', False)


def test_config_builds_the_arm_and_leaves_the_mainline_untouched():
    from mmcv import Config
    arm_cfg = Config.fromfile(
        'me_rssm/configs/'
        'TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_c1valid.py')
    main_cfg = Config.fromfile(
        'configs/r4det/'
        'TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr.py')
    assert arm_cfg.model['temporal_fusion']['type'] == 'ValidityGatedRSSMFusion'
    assert main_cfg.model['temporal_fusion']['type'] == 'MotionAlignedRSSMFusion'
    assert arm_cfg.model['temporal_fusion']['valid_layout'] == 'xy'
    assert list(arm_cfg.model['temporal_fusion']['point_cloud_range']) == \
        list(PC_RANGE)
