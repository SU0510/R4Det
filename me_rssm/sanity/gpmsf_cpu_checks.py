#!/usr/bin/env python
"""CPU wiring checks for GeometryPairedMultiScaleFusion (design section 10.3).

Runs a toy-shaped but *contract-identical* instance of the module (same
align_corners/origin/stride bookkeeping, same augmentation-inverted projection,
same per-candidate softmax structure) on CPU.  Checks, in order:

  T1  step-0 identity with ConcatConvFusion (with real geometry present)
  T2  isolated-RNG initialisation (downstream modules keep the baseline stream)
  T3  meter->index bookkeeping of every native source level (radar + image)
  T4  end-to-end image path: a delta placed at a projected pixel is what the
      paired address reads, and the height reference actually moves the read
  T5  zero radar support   -> M_R == 0 exactly, camera message still non-zero
  T6  all image projections invalid -> M_I == 0 exactly, radar message survives
  T7  two-step backward: output W first, then offsets/keys/level weights
  T8  chunking equivalence
  T9  bounded address radius
  T10 forward/backward smoke, finite outputs, unchanged output shape

Exit code 0 only if every check passes.  This is a CPU wiring gate, not a
substitute for the real-data GPU S0 (design 10.3 item 7).
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from mmdet3d.models.builder import FUSION_LAYERS            # noqa: E402
import mmdet3d.models                                       # noqa: E402,F401
from mmdet3d.models.fusion_layers.concat_conv_fusion import ConcatConvFusion  # noqa: E402
from mmdet3d.models.fusion_layers.geometry_multiscale_fusion import (        # noqa: E402
    GeometryPairedMultiScaleFusion)

B, N = 2, 2
DEV = torch.device('cpu')
TOY = dict(
    img_channels=16, rad_channels=24, out_channels=16,
    embed_dim=64, num_heads=4, num_offsets=4,
    bev_shape=(6, 4), bev_origin=(10.0, -1.0), bev_cell=0.32,
    radar_origin=(10.0, -1.0), radar_base_cell=0.16, radar_base_shape=(64, 48),
    radar_level_channels=(8, 16, 32),
    image_channels=8, image_size=(48, 64), image_level_strides=(4, 8, 16),
    query_chunk=0, grad_checkpoint=False, new_param_seed=1234,
)


# ----------------------------------------------------------------------
# toy inputs in the real coordinate conventions (lidar x forward, y left,
# z up; camera x right, y down, z forward; KITTI-style 4x4 P with row 3 =
# [0, 0, 0, 1]).
# ----------------------------------------------------------------------
def make_geo(points=True, depth=True, invert_depth=False):
    img_feats = [torch.randn(B * N, TOY['image_channels'],
                             TOY['image_size'][0] // s, TOY['image_size'][1] // s)
                 for s in TOY['image_level_strides']]
    img_feats.append(torch.randn(B * N, TOY['image_channels'], 3, 4))   # P5
    img_feats.append(torch.randn(B * N, TOY['image_channels'], 2, 2))   # P6
    radar_levels = [torch.randn(B, c, TOY['radar_base_shape'][0] >> (lev + 1),
                                TOY['radar_base_shape'][1] >> (lev + 1))
                    for lev, c in enumerate(TOY['radar_level_channels'])]
    bda = torch.eye(4).repeat(B, 1, 1)
    cam2img = torch.tensor([[50.0, 0.0, 32.0, 0.0],
                            [0.0, 50.0, 24.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0]])
    lidar2cam = torch.tensor([[0.0, -1.0, 0.0, 0.0],
                              [0.0, 0.0, -1.0, 0.0],
                              [1.0, 0.0, 0.0, 0.0],
                              [0.0, 0.0, 0.0, 1.0]])
    one = cam2img @ lidar2cam @ torch.inverse(bda[0])
    lidar2img = one.unsqueeze(0).repeat(B, 1, 1).unsqueeze(1).repeat(1, N, 1, 1)
    if invert_depth:
        lidar2img = lidar2img.clone()
        lidar2img[..., 2, :] *= -1.0
    pts = []
    for _ in range(B):
        p = torch.rand(400, 5) * torch.tensor([20.0, 20.0, 5.0, 1.0, 1.0])
        p[:, 0] += 5.0
        pts.append(p)
    if not points:
        pts = [torch.zeros(0, 5) for _ in range(B)]
    return dict(
        img_feats=img_feats, num_views=N, radar_levels=radar_levels,
        depth=torch.rand(B * N, 12, TOY['image_size'][0] // 8,
                         TOY['image_size'][1] // 8) if depth else None,
        lidar2img=lidar2img, post_rots=torch.eye(2).repeat(B, N, 1, 1),
        post_trans=torch.zeros(B, N, 2), ori_shape=[TOY['image_size']] * B,
        bda=bda, points=pts)


def bev_inputs(seed=0):
    g = torch.Generator().manual_seed(seed)
    h, w = TOY['bev_shape']
    return (torch.randn(B, TOY['img_channels'], h, w, generator=g),
            torch.randn(B, TOY['rad_channels'], h, w, generator=g))


def build(nonzero_out=False, **over):
    kw = dict(TOY)
    kw.update(over)
    fz = FUSION_LAYERS.build(dict(type='GeometryPairedMultiScaleFusion', **kw))
    assert isinstance(fz, GeometryPairedMultiScaleFusion), 'registry returned the wrong type'
    if nonzero_out:
        with torch.no_grad():
            for layer in (fz.proj_r2c, fz.proj_i2r):
                layer.weight.normal_(0.0, 0.05)
                layer.bias.zero_()
    return fz


def _queries(fz, img, rad):
    """Mirror forward()'s query tensors without the chunking machinery."""
    pos = fz._pos_enc
    b, _, h, w = img.shape
    p_cam = fz.proj_cam(img) + pos
    p_rad = fz.proj_rad(rad) + pos
    q = h * w
    q_cam = fz.norm_cam(p_cam.permute(0, 2, 3, 1).reshape(b, q, -1)).permute(0, 2, 1)
    q_rad = fz.norm_rad(p_rad.permute(0, 2, 3, 1).reshape(b, q, -1)).permute(0, 2, 1)
    f_cat = torch.cat([p_cam, p_rad, pos.expand(b, -1, h, w)], dim=1).reshape(b, -1, q)
    return q_cam, q_rad, f_cat


def _address_for_pixel(lidar2img, u_t, v_t, z0):
    """Solve for (x, y) on the plane z=z0 that projects to (u_t, v_t)."""
    m = lidar2img.double()
    rows = torch.stack([m[0] - u_t * m[2], m[1] - v_t * m[2]], dim=0)   # 2 x 4
    rhs = -(rows[:, 2] * z0 + rows[:, 3])
    return torch.linalg.solve(rows[:, :2], rhs)                          # (2,) float64


# ----------------------------------------------------------------------
# checks
# ----------------------------------------------------------------------
def t1_step0_identity():
    fz = build()
    base = ConcatConvFusion(img_channels=TOY['img_channels'],
                            rad_channels=TOY['rad_channels'],
                            out_channels=TOY['out_channels'])
    base.fusion_conv.load_state_dict(fz.fusion_conv.state_dict())
    img, rad = bev_inputs()
    with torch.no_grad():
        ref = base(img, rad)
        got = fz(img, rad, geo=make_geo())
        got_none = fz(img, rad)
    assert torch.equal(ref, got), 'step-0 output differs from ConcatConvFusion'
    assert torch.equal(ref, got_none), 'geo=None path differs from ConcatConvFusion'
    return 'bitwise identical to same-state ConcatConv (geo present and geo=None)'


def t2_rng_isolation():
    torch.manual_seed(7)
    ConcatConvFusion(img_channels=TOY['img_channels'], rad_channels=TOY['rad_channels'],
                     out_channels=TOY['out_channels'])
    probe_a = nn.Conv2d(3, 5, 3)
    torch.manual_seed(7)
    build()
    probe_b = nn.Conv2d(3, 5, 3)
    assert torch.equal(probe_a.weight, probe_b.weight), 'GP-MSF leaked into the RNG stream'
    assert torch.equal(probe_a.bias, probe_b.bias)
    fz2, fz3 = build(), build()
    for (n1, p1), (n2, p2) in zip(fz2.named_parameters(), fz3.named_parameters()):
        if n1.startswith('fusion_conv.'):
            continue          # fusion_conv follows the global stream by design
        assert n1 == n2 and torch.equal(p1, p2), f'new params not reproducible: {n1}'
    return 'downstream RNG stream preserved; new params reproducible from new_param_seed'


def t3_index_bookkeeping():
    fz = build()
    for lev, step in enumerate(fz.radar_level_steps):
        h = TOY['radar_base_shape'][0] >> (lev + 1)
        w = TOY['radar_base_shape'][1] >> (lev + 1)
        iy = torch.arange(h).view(-1, 1).expand(h, w)
        ix = torch.arange(w).view(1, -1).expand(h, w)
        feat = (iy * w + ix).float().view(1, 1, h, w)
        # origin is the *centre* of pixel 0 (0.08 / -39.60 in the real config),
        # so level-l pixel j sits at origin + j * step for every level.
        cy = fz.radar_origin[1] + iy.float() * step
        cx = fz.radar_origin[0] + ix.float() * step
        xy = torch.stack([cx, cy], dim=-1).view(1, -1, 2)
        grid, inside = fz._grid_from_meter(xy, fz.radar_origin, step, (h, w))
        got = fz._sample_map(feat, grid, inside).view(h, w)
        assert bool(inside.all()), f'radar level {lev}: cell centres marked outside'
        assert torch.allclose(got, feat.view(h, w), atol=1e-4), (
            f'radar level {lev} metre->index mapping is wrong')
    for s, stride in enumerate(fz.image_level_strides):
        h, w = TOY['image_size'][0] // stride, TOY['image_size'][1] // stride
        feat = (torch.arange(h).view(-1, 1) * w + torch.arange(w).view(1, -1)).float()
        feat = feat.view(1, 1, h, w)
        u = (torch.arange(w).float() + 0.5) * stride
        v = (torch.arange(h).float() + 0.5) * stride
        grid = torch.stack([2.0 * u.view(1, -1).expand(h, w) / (stride * w) - 1.0,
                            2.0 * v.view(-1, 1).expand(h, w) / (stride * h) - 1.0], dim=-1)
        got = fz._sample_map(feat, grid.reshape(1, -1, 2),
                             torch.ones(1, h * w, dtype=torch.bool)).view(h, w)
        assert torch.allclose(got, feat.view(h, w), atol=1e-4), (
            f'image level {s} pixel->index mapping is wrong')
    return '3 native radar levels + 3 image levels sample their cell centres exactly'


def t4_image_path():
    """P3 delta read back through lidar2img, plus analytic height sensitivity.

    Part 1: a delta inside one P3 pixel must be read back exactly (times its
    own level weight) for the height reference whose ray hits that pixel.
    Part 2: with a row ramp in P3 every height reference must read *its own*
    projected row, i.e. ``beta * (v/stride - 0.5)`` -- computed here from the
    raw 4x4 matrix, independently of the module.
    """
    fz = build()
    geo = make_geo()
    s_idx, row, col = 1, 4, 5            # delta in P3; row 4 keeps depth > 0
    stride = fz.image_level_strides[s_idx]
    feats = [torch.zeros_like(f) for f in geo['img_feats']]
    for b in range(B):
        for n in range(N):
            feats[s_idx][b * N + n, :, row, col] = 1.0
    geo_d = dict(geo, img_feats=feats)
    ctx = fz._build_context(geo_d, B, DEV, torch.float32)

    z0 = float(fz.z_refs[1])
    u_t, v_t = (col + 0.5) * stride, (row + 0.5) * stride
    xy = _address_for_pixel(geo['lidar2img'][0, 0], u_t, v_t, z0).float()
    addr = xy.view(1, 1, 1, 2).repeat(B, 1, 1, 1)
    q_ift = torch.zeros(B, 1, fz.head_dim)
    i_h, ok, beta, _ = fz._image_terms(ctx, 0, addr, q_ift)
    assert float(addr[0, 0, 0, 0]) > 0.0, 'test address is behind the toy camera'
    assert bool(ok[0, 0, 0, 1]), 'the projected pixel was not marked valid'
    vec = i_h[0, 0, 0, 1]                      # head-0 channel vector
    w_lvl = float(beta[0, 0, 0, 1, s_idx])
    assert w_lvl > 1e-3, 'P3 got no level weight'
    # The 1x1 projection conv maps the all-ones delta pixel to W.sum + b, so the
    # expected read is *analytic* and independent of the module's sampling code.
    with torch.no_grad():
        w0 = fz.proj_img_levels[s_idx]
        unit_vec = (w0.weight.sum(dim=(1, 2, 3)) + w0.bias)[:fz.head_dim]
        b_vec = w0.bias[:fz.head_dim].clone()
        w_sum = w0.weight.sum(dim=(1, 2, 3))[:fz.head_dim].clone()
        # the other levels read all-zero features, i.e. their 1x1 conv bias
        other_bias = [(s, fz.proj_img_levels[s].bias[:fz.head_dim].clone())
                      for s in range(fz.image_num_levels) if s != s_idx]
    exp = w_lvl * unit_vec
    for s, b_s in other_bias:
        exp = exp + float(beta[0, 0, 0, 1, s]) * b_s
    err = float((vec - exp).abs().max())
    assert err < 2e-3, (
        f'P3 delta read differs from its own level weight x projected delta: {err:.5f}')

    # ---- part 2: a row ramp makes each height read its own projected row
    h_feat, w_feat = geo['img_feats'][s_idx].shape[-2:]
    ramp = torch.arange(h_feat, dtype=torch.float32).view(-1, 1).expand(h_feat, w_feat)
    feats_r = [torch.zeros_like(f) for f in geo['img_feats']]
    for n in range(B * N):
        feats_r[s_idx][n] = ramp
    ctx_r = fz._build_context(dict(geo, img_feats=feats_r), B, DEV, torch.float32)
    x_f, y_f = 6.0, 1.2                  # in front of the toy camera
    addr_r = torch.tensor([x_f, y_f]).float().view(1, 1, 1, 2).repeat(B, 1, 1, 1)
    i_r, ok_r, beta_r, _ = fz._image_terms(ctx_r, 0, addr_r, q_ift)
    reads, worst = [], 0.0
    for n, z_ref in enumerate(fz.z_refs.tolist()):
        p4 = torch.tensor([x_f, y_f, z_ref, 1.0], dtype=torch.float64)
        proj = geo['lidar2img'][0, 0].double() @ p4
        u_n, v_n = float(proj[0] / proj[2]), float(proj[1] / proj[2])
        assert proj[2] > 0.0, f'height {n} projects behind the camera'
        assert 0.0 <= u_n <= 64.0 and 0.0 <= v_n <= 48.0, f'height {n} off image'
        w_n = float(beta_r[0, 0, 0, n, s_idx])
        assert w_n > 1e-3, f'height {n} got no P3 level weight'
        rho = v_n / stride - 0.5                  # fractional row index read
        assert 0.0 <= rho <= h_feat - 1.0, 'ramp read falls outside the feature'
        exp_vec = w_n * (rho * w_sum + b_vec)     # conv of the ramp, then read
        for s, b_s in other_bias:                 # zero levels contribute bias only
            exp_vec = exp_vec + float(beta_r[0, 0, 0, n, s]) * b_s
        got_vec = i_r[0, 0, 0, n]
        assert bool(ok_r[0, 0, 0, n]), f'height {n} marked invalid on the ramp'
        worst = max(worst, float((got_vec - exp_vec).abs().max()))
        assert worst < 3e-3, (
            f'height {n} read differs from its own projected row value: {worst:.5f}')
        reads.append(float(got_vec.mean()))
    assert max(reads) - min(reads) > 0.05, 'height references do not move the read'
    return (f'P3 delta read back within {err:.1e} of level weight {w_lvl:.4f} x conv(delta); '
            f'three height refs read {reads[0]:.3f}/{reads[1]:.3f}/{reads[2]:.3f} '
            f'(max err {worst:.1e}) = their own projected ramp rows')


def t5_zero_radar():
    fz = build(nonzero_out=True)
    img, rad = bev_inputs()
    geo = make_geo(points=False)
    with torch.no_grad():
        ctx = fz._build_context(geo, B, DEV, torch.float32)
        assert float(ctx['support'].sum()) == 0.0
        q_cam, q_rad, f_cat = _queries(fz, img, rad)
        m_r, m_i, _ = fz._chunk(q_cam, q_rad, f_cat, ctx, 0)
        c_new = img + fz.proj_r2c(m_r.view(B, -1, *TOY['bev_shape']))
        r_new = rad + fz.proj_i2r(m_i.view(B, -1, *TOY['bev_shape']))
    assert float(m_r.abs().sum()) == 0.0, 'radar message not exactly zero without support'
    assert float(m_i.abs().sum()) > 0.0, 'no camera message without radar support'
    assert torch.equal(c_new, img), 'camera BEV changed despite zero radar message'
    assert not torch.allclose(r_new, rad), 'radar BEV unchanged despite camera message'
    return 'M_R == 0 exactly (camera path untouched), M_I still non-zero'


def t6_no_image():
    fz = build(nonzero_out=True)
    img, rad = bev_inputs()
    geo = make_geo(invert_depth=True)
    with torch.no_grad():
        ctx = fz._build_context(geo, B, DEV, torch.float32)
        q_cam, q_rad, f_cat = _queries(fz, img, rad)
        m_r, m_i, _ = fz._chunk(q_cam, q_rad, f_cat, ctx, 0)
        r_new = rad + fz.proj_i2r(m_i.view(B, -1, *TOY['bev_shape']))
    assert float(m_i.abs().sum()) == 0.0, 'image message not zero with all-invalid projections'
    assert torch.equal(r_new, rad), 'radar BEV changed despite zero image message'
    assert float(m_r.abs().sum()) > 0.0, 'radar message vanished although support exists'
    return 'M_I == 0 exactly with every projection invalid; radar message survives'


def t7_two_step_grad():
    fz = build()
    img, rad = bev_inputs()
    geo = make_geo()
    opt = torch.optim.SGD([p for p in fz.parameters() if p.requires_grad], lr=0.05)
    w = torch.randn(B, TOY['out_channels'], *TOY['bev_shape'])

    def step():
        opt.zero_grad()
        out = fz(img, rad, geo=geo)
        ((out * w).sum()).backward()

    step()
    missing = [n for n, p in fz.named_parameters() if p.grad is None]
    assert not missing, f'parameters without grad after step 1: {missing}'
    off_grad_1 = float(fz.offset_conv[-1].weight.grad.abs().sum())
    assert float(fz.proj_r2c.weight.grad.abs().sum()) > 0, 'output projection got no gradient'
    opt.step()
    step()
    off_grad_2 = float(fz.offset_conv[-1].weight.grad.abs().sum())
    key_grad = sum(float(m.weight.grad.abs().sum()) for m in fz.key_rad)
    lvl_grad = float(fz.level_rad_bias.grad.abs().sum() + fz.level_img_bias.grad.abs().sum())
    assert off_grad_1 == 0.0, f'offsets should have no gradient at step 0 (got {off_grad_1})'
    assert off_grad_2 > 0.0, 'offsets still have no gradient after the first update'
    assert key_grad > 0.0 and lvl_grad > 0.0, 'keys / level weights never learned'
    return (f'offsets {off_grad_1:.3g} at step0 then {off_grad_2:.3g}; keys {key_grad:.3g}; '
            f'level weights {lvl_grad:.3g}')


def t8_chunking():
    img, rad = bev_inputs()
    geo = make_geo()
    a, b = build(query_chunk=0), build(query_chunk=5)
    b.load_state_dict(a.state_dict())
    a.eval()
    b.eval()
    with torch.no_grad():
        oa, ob = a(img, rad, geo=geo), b(img, rad, geo=geo)
    assert torch.allclose(oa, ob, atol=1e-5), 'chunked and single-chunk outputs differ'
    return 'chunked (5 cells) == single chunk'


def t9_bounds():
    fz = build()
    img, rad = bev_inputs()
    with torch.no_grad():
        q_cam, q_rad, f_cat = _queries(fz, img, rad)
        hd, k = fz.num_heads, fz.num_offsets
        off3 = fz.offset_conv(f_cat.unsqueeze(-1)).squeeze(-1)
        off = off3.view(B, hd, k - 1, 2, -1).permute(0, 1, 4, 2, 3)
        tanh_off = torch.tanh(off) * fz.offset_bounds.view(1, hd, 1, 1, 2)
        disp = (tanh_off ** 2).sum(-1).sqrt()
    hi = float(disp.max())
    cap = float((fz.offset_bounds ** 2).sum(-1).sqrt().max())
    assert hi <= cap + 1e-6, f'address radius {hi} above the per-head cap {cap}'
    return f'max radius {hi:.4f} m <= per-head cap {cap:.4f} m'


def t10_smoke():
    fz = build(query_chunk=7, grad_checkpoint=True)
    fz.train()
    img, rad = bev_inputs()
    out = fz(img, rad, geo=make_geo())
    assert out.shape == (B, TOY['out_channels'], *TOY['bev_shape']), out.shape
    assert torch.isfinite(out).all(), 'non-finite forward output'
    out.sum().backward()
    missing = [n for n, p in fz.named_parameters() if p.grad is None]
    assert not missing, f'no gradient on the checkpointed path: {missing}'
    return f'checkpointed forward/backward finite; {len(list(fz.parameters()))} tensors, all with grad'


def main():
    torch.manual_seed(0)
    tests = [t1_step0_identity, t2_rng_isolation, t3_index_bookkeeping, t4_image_path,
             t5_zero_radar, t6_no_image, t7_two_step_grad, t8_chunking, t9_bounds, t10_smoke]
    ok = 0
    for fn in tests:
        try:
            msg = fn()
            print(f'PASS {fn.__name__}: {msg}')
            ok += 1
        except Exception as exc:                                  # noqa: BLE001
            print(f'FAIL {fn.__name__}: {type(exc).__name__}: {exc}')
    print(f'\n{ok}/{len(tests)} CPU wiring checks passed')
    return 0 if ok == len(tests) else 1


if __name__ == '__main__':
    sys.exit(main())
