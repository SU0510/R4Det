"""Wiring guards for the GP-MSF arm (prereg docs/gpmsf_prereg_20261013.md §2/§5).

These are the cheap, CPU-only checks the queue runs *before* spending any GPU
time on the 450-iter smoke:

  * only the GP-MSF fusion advertises ``accepts_geometry_sources``, so every
    baseline / F1 / C1 arm keeps the exact old two-argument call;
  * the arm config is a single-variable change against the mainline and its
    frozen geometry matches the resolved mainline config;
  * ``require_geo`` makes an unwired training forward fail loudly instead of
    silently degenerating into the concat baseline;
  * ``R4Det._gpmsf_geometry`` produces exactly the tensor contract the module
    asserts, and rejects inconsistent shapes;
  * the arm config *as frozen* builds and is bitwise identical to the same-state
    ConcatConvFusion at step 0 on real-shaped tensors;
  * the chunked forward merges heads *without* interleaving the query and
    channel axes: with the residual projections switched on, both direct
    messages and the full output must be identical for one chunk and for a
    ragged 5-cell chunk (P0 regression, see training_runs_full.md §100).
"""
import os
import sys

import pytest
import torch
from mmcv import Config

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import me_rssm                                              # noqa: E402,F401
from mmdet3d.models.builder import FUSION_LAYERS            # noqa: E402
from mmdet3d.models.detectors.R4Det import R4Det            # noqa: E402
from mmdet3d.models.fusion_layers.concat_conv_fusion import (   # noqa: E402
    ConcatConvFusion)
from mmdet3d.models.fusion_layers.geometry_multiscale_fusion import (  # noqa: E402
    GeometryPairedMultiScaleFusion)

ARM_CFG = ('me_rssm/configs/'
           'TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_gpmsf.py')
MAIN_CFG = ('configs/r4det/'
            'TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr.py')

B = 1                                  # one frame per sample in this dataset
C_IMG, C_RAD, C_OUT = 256, 384, 256
H_BEV, W_BEV = 248, 216
IMG_LEVEL_SHAPES = [(120, 160), (60, 80), (30, 40)]     # P2/P3/P4 of 480x640
RAD_LEVEL_SHAPES = [(248, 216), (124, 108), (62, 54)]   # SECOND 0.32/0.64/1.28
RAD_LEVEL_CH = (64, 128, 256)


def _fake_detector():
    class _Fake:
        _native_pts_levels = [
            torch.zeros(B, c, h, w) for c, (h, w) in zip(RAD_LEVEL_CH, RAD_LEVEL_SHAPES)]
    return _Fake()


def _geo_kwargs():
    return dict(
        img_feats=[torch.randn(B, C_IMG, h, w) for h, w in IMG_LEVEL_SHAPES] + [None, None],
        depth=torch.rand(B, 72, 60, 80),
        lidar2img=torch.eye(4).repeat(B, 1, 1),                 # (B,4,4) single view
        post_rots=torch.eye(3).repeat(B, 1, 1),                 # (B,3,3)
        post_trans=torch.zeros(B, 3),                           # (B,3)
        bda=torch.eye(4).repeat(B, 1, 1),
        img_metas=[dict(ori_shape=(375, 1242)) for _ in range(B)],
        points=[torch.rand(50, 5) for _ in range(B)],
    )


# ---------------------------------------------------------------- opt-in wiring
def test_only_gpmsf_advertises_geometry_sources():
    arm = FUSION_LAYERS.build(dict(
        type='GeometryPairedMultiScaleFusion', img_channels=C_IMG,
        rad_channels=C_RAD, out_channels=C_OUT))
    base = FUSION_LAYERS.build(dict(
        type='ConcatConvFusion', img_channels=C_IMG, rad_channels=C_RAD,
        out_channels=C_OUT))
    assert isinstance(arm, GeometryPairedMultiScaleFusion)
    assert arm.accepts_geometry_sources is True
    assert not getattr(base, 'accepts_geometry_sources', False)
    from me_rssm.reliability_fusion_rssm import ReliabilityGatedConcatFusion
    assert not getattr(ReliabilityGatedConcatFusion, 'accepts_geometry_sources', False)


def test_require_geo_fails_loudly_when_unwired():
    arm = FUSION_LAYERS.build(dict(
        type='GeometryPairedMultiScaleFusion', img_channels=C_IMG,
        rad_channels=C_RAD, out_channels=C_OUT, require_geo=True))
    img = torch.randn(B, C_IMG, 4, 6)
    rad = torch.randn(B, C_RAD, 4, 6)
    arm.train()
    with pytest.raises(RuntimeError, match='geo=None'):
        arm(img, rad)
    arm.eval()
    with pytest.raises(RuntimeError, match="geo=None"):
        arm(img, rad)
    arm.require_geo = False
    with torch.no_grad():
        out = arm(img, rad)
    assert out.shape == (B, C_OUT, 4, 6)


# ------------------------------------------------------------------- arm config
def test_arm_config_is_single_variable_and_frozen():
    arm = Config.fromfile(ARM_CFG)
    main = Config.fromfile(MAIN_CFG)
    diff = sorted(k for k in set(arm.model) | set(main.model)
                  if arm.model.get(k) != main.model.get(k))
    assert diff == ['RCFusion']
    assert arm.model['RCFusion']['type'] == 'GeometryPairedMultiScaleFusion'
    assert main.model['RCFusion']['type'] == 'ConcatConvFusion'
    rc = arm.model['RCFusion']
    assert tuple(rc['bev_shape']) == (H_BEV, W_BEV)
    assert tuple(rc['radar_level_channels']) == RAD_LEVEL_CH
    assert tuple(rc['radar_level_steps']) == (0.32, 0.64, 1.28)
    assert tuple(rc['radar_base_shape']) == (496, 432)
    assert tuple(rc['image_level_strides']) == (4, 8, 16)
    assert tuple(rc['image_size']) == (480, 640)
    assert rc['depth_downsample'] == 8
    assert rc['require_geo'] is True
    # the arm must not drift from the mainline on anything but the fusion
    assert arm.model['bev_h_'] == 216 and arm.model['bev_w_'] == 248


def test_smoke_config_keeps_the_frozen_gates():
    smoke = Config.fromfile('me_rssm/sanity/gpmsf_smoke_450iter.py')
    assert smoke.runner['type'] == 'IterBasedRunner'
    assert smoke.runner['max_iters'] == 450
    assert smoke.model['RCFusion']['type'] == 'GeometryPairedMultiScaleFusion'
    assert smoke.model['RCFusion']['collect_stats'] is True
    assert smoke.model['RCFusion']['require_geo'] is True
    hook_types = [h['type'] for h in smoke.custom_hooks]
    assert 'GPMSFWiringStatsHook' in hook_types
    assert 'KLScaleSchedulerHook' in hook_types          # KL schedule preserved


# ------------------------------------------------------------- detector contract
def test_detector_geometry_matches_module_contract():
    fake = _fake_detector()
    geo = R4Det._gpmsf_geometry(fake, **_geo_kwargs())
    assert set(geo) == {'img_feats', 'num_views', 'radar_levels', 'depth',
                        'lidar2img', 'post_rots', 'post_trans', 'ori_shape',
                        'bda', 'points'}
    assert geo['num_views'] == 1
    assert geo['lidar2img'].shape == (B, 1, 4, 4)
    assert geo['post_rots'].shape == (B, 1, 2, 2)         # 3x3 -> 2x2 block
    assert geo['post_trans'].shape == (B, 1, 2)           # 3   -> 2
    assert [f.shape[-2:] for f in geo['img_feats']] == IMG_LEVEL_SHAPES
    assert [t.shape for t in geo['radar_levels']] == \
        [(B, c, h, w) for c, (h, w) in zip(RAD_LEVEL_CH, RAD_LEVEL_SHAPES)]
    assert geo['ori_shape'] == [(375, 1242)] * B
    assert geo['depth'].shape[0] == B * geo['num_views']

    # a 4-D (B,N,3,3) post_rots stays (B,N,2,2); multi-view is not flattened
    kw = _geo_kwargs()
    kw.update(lidar2img=torch.eye(4).repeat(B, 2, 1, 1),
              post_rots=torch.eye(3).repeat(B, 2, 1, 1),
              post_trans=torch.zeros(B, 2, 3),
              img_feats=[torch.randn(B * 2, C_IMG, h, w)
                         for h, w in IMG_LEVEL_SHAPES] + [None, None],
              depth=torch.rand(B * 2, 72, 60, 80),
              points=[torch.rand(50, 5) for _ in range(B)])
    geo2 = R4Det._gpmsf_geometry(fake, **kw)
    assert geo2['num_views'] == 2 and geo2['post_rots'].shape == (B, 2, 2, 2)


def test_detector_geometry_rejects_inconsistent_shapes():
    fake = _fake_detector()
    kw = _geo_kwargs()
    kw['post_trans'] = torch.zeros(B, 5)                  # wrong view count
    with pytest.raises(AssertionError):
        R4Det._gpmsf_geometry(fake, **kw)
    fake._native_pts_levels = None
    with pytest.raises(RuntimeError, match='native radar levels'):
        R4Det._gpmsf_geometry(fake, **_geo_kwargs())


# ------------------------------------------------------- step-0 identity (real cfg)
def test_frozen_config_is_step0_identical_to_baseline():
    arm_cfg = Config.fromfile(ARM_CFG).model['RCFusion']
    arm_cfg = dict(arm_cfg)
    arm_cfg.update(query_chunk=4096, grad_checkpoint=False)
    arm = FUSION_LAYERS.build(dict(arm_cfg))
    base = ConcatConvFusion(img_channels=C_IMG, rad_channels=C_RAD,
                            out_channels=C_OUT, kernel_size=3)
    base.fusion_conv.load_state_dict(arm.fusion_conv.state_dict())
    geo = R4Det._gpmsf_geometry(_fake_detector(), **_geo_kwargs())
    img = torch.randn(B, C_IMG, H_BEV, W_BEV)
    rad = torch.randn(B, C_RAD, H_BEV, W_BEV)
    with torch.no_grad():
        ref = base(img, rad)
        got = arm(img, rad, geo=geo)
    assert torch.equal(ref, got), 'step-0 output is not bitwise the baseline'
    assert got.shape == (B, C_OUT, H_BEV, W_BEV)


# ---------------------------------------------- message-axis regression (P0)
def test_chunked_residual_messages_are_not_scrambled():
    """P0: the head merge must not interleave the query and channel axes.

    The original chunk check ran with both residual projections still at zero,
    so it compared the shared concat path only and stayed green while
    ``reshape`` (without swapping query/channel back) sent every message to the
    wrong BEV cell, chunk-size dependent.  Here the projections are switched on
    and the *direct* messages are read through a hook, because the zero-init
    residual hides them in the output.
    """
    from me_rssm.sanity import gpmsf_cpu_checks as toy
    img, radar = toy.bev_inputs()
    geo = toy.make_geo()
    full = toy.build(nonzero_out=True, query_chunk=0).eval()
    chunked = toy.build(nonzero_out=True, query_chunk=5).eval()
    chunked.load_state_dict(full.state_dict())
    base = toy.build(query_chunk=0).eval()               # same weights, no residual
    base.load_state_dict(full.state_dict())
    with torch.no_grad():
        for layer in (base.proj_r2c, base.proj_i2r):
            layer.weight.zero_()
            layer.bias.zero_()
        m_full = toy._capture_messages(full, img, radar, geo)
        m_chunk = toy._capture_messages(chunked, img, radar, geo)
        out_base = base(img, radar, geo=geo)
    assert float(m_full[0].abs().max()) > 1e-3, 'vacuous: M_R is identically zero'
    assert float(m_full[1].abs().max()) > 1e-3, 'vacuous: M_I is identically zero'
    assert float((m_full[2] - out_base).abs().max()) > 1e-3, \
        'vacuous: the residual projections do not move the output'
    assert torch.equal(m_full[0], m_chunk[0]), 'M_R depends on the chunk size'
    assert torch.equal(m_full[1], m_chunk[1]), 'M_I depends on the chunk size'
    assert torch.equal(m_full[2], m_chunk[2]), 'the output depends on the chunk size'


def test_stats_do_not_disable_training_checkpoint(monkeypatch):
    from me_rssm.sanity import gpmsf_cpu_checks as toy
    import mmdet3d.models.fusion_layers.geometry_multiscale_fusion as module
    calls = []
    original = module.checkpoint

    def observed(*args, **kwargs):
        calls.append(kwargs.get('use_reentrant'))
        return original(*args, **kwargs)

    monkeypatch.setattr(module, 'checkpoint', observed)
    fusion = toy.build(grad_checkpoint=True, collect_stats=True, query_chunk=5)
    fusion.train()
    img, radar = toy.bev_inputs()
    fusion(img, radar, geo=toy.make_geo()).square().mean().backward()
    assert calls and all(value is False for value in calls)
    assert fusion.last_stats is not None


def test_trained_projection_bias_cannot_break_missing_evidence_fallback():
    from me_rssm.sanity import gpmsf_cpu_checks as toy
    fusion = toy.build(nonzero_out=True)
    fusion.eval()
    with torch.no_grad():
        fusion.proj_r2c.bias.fill_(2.)
        fusion.proj_i2r.bias.fill_(3.)
    img, radar = toy.bev_inputs()
    observed = []
    hook = fusion.fusion_conv.register_forward_pre_hook(
        lambda module, args: observed.append(args[0].detach()))
    try:
        with torch.no_grad():
            fusion(img, radar, geo=toy.make_geo(points=False))
            assert torch.equal(observed[-1][:, :img.shape[1]], img)
            fusion(img, radar, geo=toy.make_geo(invert_depth=True))
            assert torch.equal(observed[-1][:, img.shape[1]:], radar)
    finally:
        hook.remove()


def test_new_parameter_initialization_does_not_reseed_cuda(monkeypatch):
    def forbidden(_seed):
        raise AssertionError('torch.manual_seed would also reseed CUDA')
    monkeypatch.setattr(torch, 'manual_seed', forbidden)
    GeometryPairedMultiScaleFusion(img_channels=16, rad_channels=24, out_channels=16)
