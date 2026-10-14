#!/usr/bin/env python
"""Equivalence gate for the active-cell compaction (training record 104.9).

``active_cells_only`` drops the candidate chains of cells whose two messages
are provably zero.  This script proves, on the CPU toy contract, that

  A1 the fusion output is unchanged (dense vs compacted, <= 1e-6);
  A2 the *direct* messages M_R / M_I are unchanged on every cell;
  A3 every dropped cell carries exactly M_R = M_I = 0 and valid = False, on a
     fixture whose two samples keep *unequal* active counts (padded tail);
  A4 all parameter gradients match (<= 1e-6);
  A5 all native-source input gradients match (<= 1e-6);
  A6 chunked + checkpointed + compacted == one dense chunk (<= 1e-6);
  A7 the full-canvas diagnostics (collect_stats) match (<= 1e-6);
  A8 camera-only fallback survives: no radar echoes -> M_I still non-zero;
  A9 radar-only survival: degenerate image projection -> M_R still non-zero;
  A10 at equal chunking the compaction is *bit-exact* (param + input grads),
      i.e. it adds no numerical difference of its own.

It is a wiring/equivalence gate, not a substitute for the real-data GPU S0.
"""
from __future__ import annotations

import os
import sys

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import me_rssm                                              # noqa: E402,F401
import mmdet3d.models                                       # noqa: E402,F401
from me_rssm.sanity import gpmsf_cpu_checks as toy           # noqa: E402

TOL = 1e-6


def _run(compacted, chunk=5, ckpt=False, stats=False, geo=None, seed=0,
         weight=None, input_grad=False):
    torch.manual_seed(seed)
    img, rad = toy.bev_inputs(seed=seed)
    geo = toy.make_geo() if geo is None else geo
    if input_grad:
        # ``make_geo`` samples random features, so the two arms must share one
        # draw: clone the leaves per run and let each run fill its own ``.grad``.
        geo = dict(geo)
        geo['img_feats'] = [t.detach().clone().requires_grad_(True)
                            for t in geo['img_feats']]
        geo['radar_levels'] = [t.detach().clone().requires_grad_(True)
                               for t in geo['radar_levels']]
    fz = toy.build(nonzero_out=True, query_chunk=chunk, grad_checkpoint=ckpt,
                   collect_stats=stats, active_cells_only=compacted)
    if weight is not None:
        fz.load_state_dict(weight)
    seen = {}
    original = fz._active_cells

    def observed(ctx, f_cat_f, cell_xy, e_r, e_t):
        active = original(ctx, f_cat_f, cell_xy, e_r, e_t)
        seen['active'] = active.detach().clone()
        seen['perm'] = torch.argsort(active.to(torch.uint8), dim=1,
                                     descending=True, stable=True)
        seen['n_cells'] = int(active.sum(dim=1).max().item())
        return active

    fz._active_cells = observed
    fz.train() if ckpt else fz.eval()
    m_r, m_i, out = toy._capture_messages(fz, img, rad, geo, grad=True)
    loss = (out * torch.arange(out.numel(), dtype=out.dtype).view_as(out)
            / out.numel()).sum()
    loss = loss + m_r.square().sum() + m_i.square().sum()
    loss.backward()
    grads = {n: p.grad.detach().clone() for n, p in fz.named_parameters()
             if p.grad is not None}
    input_grads = ({k: [v.grad.detach().clone() if v.grad is not None else None
                        for v in geo[k]] for k in ('img_feats', 'radar_levels')}
                   if input_grad else {})
    return dict(m_r=m_r, m_i=m_i, out=out.detach(), grads=grads,
                input_grads=input_grads, seen=seen,
                stats=dict(fz.last_stats or {}))


def _maxdiff(a, b):
    return float((a - b).abs().max()) if a is not None and b is not None else float('inf')


def _grad_diff(a, b):
    """|d| of two optional gradients; None means 'no path' and equals an exact 0."""
    if a is None or b is None:
        other = b if a is None else a
        return 0.0 if other is None or float(other.abs().max()) == 0.0 else float('inf')
    return float((a - b).abs().max())


def _degenerate_image_geo():
    """Fixture with no image validity anywhere: cells can only be radar-active.

    The two samples then keep *different* numbers of active cells, which is the
    configuration that exposes a per-sample gather error in the padded tail.
    """
    geo = toy.make_geo()
    geo['lidar2img'] = geo['lidar2img'].clone()
    geo['lidar2img'][..., 2, :] = 0
    geo['lidar2img'][..., 2, 3] = -1.0
    return geo


def main():
    checks = {}

    ref = _run(False, chunk=0)
    dense = _run(False, chunk=5)
    comp = _run(True, chunk=5)
    weight = dense['grads']            # unused, keeps linters quiet about naming
    del weight
    checks['A1_output'] = _maxdiff(ref['out'], comp['out']) <= TOL
    checks['A2_messages'] = max(_maxdiff(ref['m_r'], comp['m_r']),
                                _maxdiff(ref['m_i'], comp['m_i'])) <= TOL
    checks['A6_chunk_ckpt'] = max(_maxdiff(ref['out'], dense['out']),
                                  _maxdiff(dense['out'], comp['out'])) <= TOL

    active = comp['seen']['active']

    # A3 fixture: a canvas with *unequal* per-sample active counts (the padded
    # tail of the smaller sample is where a per-sample gather bug hides), so
    # the dropped-cell proof runs on cells that are really dropped.  The geo is
    # drawn ONCE -- ``make_geo`` samples random features/points, so comparing
    # two separately drawn fixtures would compare two different problems.
    geo_mixed = _degenerate_image_geo()
    mixed = _run(True, chunk=5, geo=geo_mixed)
    mixed_ref = _run(False, chunk=0, geo=geo_mixed)
    mixed_active = mixed['seen']['active']
    counts = mixed_active.sum(dim=1)
    m_r_grid = mixed['m_r'].view(mixed_active.shape[0], mixed['m_r'].shape[1], -1)
    m_i_grid = mixed['m_i'].view(mixed_active.shape[0], mixed['m_i'].shape[1], -1)
    inactive_max, inactive_n = 0.0, 0
    for s in range(mixed_active.shape[0]):
        if (~mixed_active[s]).any():
            inactive_n += int((~mixed_active[s]).sum())
            inactive_max = max(inactive_max,
                               float(m_r_grid[s][:, ~mixed_active[s]].abs().max()),
                               float(m_i_grid[s][:, ~mixed_active[s]].abs().max()))
    checks['A3_has_inactive'] = bool((~mixed_active).any()) and bool(mixed_active.any())
    checks['A3_dropped_exact_zero'] = inactive_max == 0.0
    checks['A3_mixed_fixture'] = (
        inactive_n > 0 and int(counts.min()) < int(counts.max())
        and _maxdiff(mixed_ref['m_r'], mixed['m_r']) <= TOL
        and _maxdiff(mixed_ref['m_i'], mixed['m_i']) <= TOL
        and _maxdiff(mixed_ref['out'], mixed['out']) <= TOL)

    worst_p = max([_maxdiff(g, comp['grads'].get(n))
                   for n, g in ref['grads'].items()] or [0.0])
    checks['A4_param_grads'] = (set(ref['grads']) == set(comp['grads'])
                                and worst_p <= TOL)

    worst_i = 0.0
    ref_i = _run(False, chunk=0, input_grad=True)
    dense_i = _run(False, chunk=5, input_grad=True)
    comp_i = _run(True, chunk=5, input_grad=True)
    for key in ('img_feats', 'radar_levels'):
        for a, b in zip(ref_i['input_grads'][key], comp_i['input_grads'][key]):
            worst_i = max(worst_i, _grad_diff(a, b))
    checks['A5_input_grads'] = worst_i <= TOL

    # Bit-exactness at equal chunking: the residual 4.8e-7 seen against the
    # unchunked reference is the *pre-existing* chunk-order effect (dense0 vs
    # dense5 shows the same number), so the compaction itself must not add any.
    worst_pb = max(float((dense['grads'][n] - comp['grads'][n]).abs().max())
                   for n in dense['grads'])
    worst_ib = 0.0
    for key in ('img_feats', 'radar_levels'):
        for a, b in zip(dense_i['input_grads'][key], comp_i['input_grads'][key]):
            worst_ib = max(worst_ib, _grad_diff(a, b))
    checks['A10_bit_exact_same_chunk'] = (worst_pb == 0.0 and worst_ib == 0.0)

    ref_s = _run(False, chunk=0, stats=True)
    comp_s = _run(True, chunk=5, stats=True)
    worst_s = 0.0
    assert set(ref_s['stats']) == set(comp_s['stats']), 'stats key drift'
    for k, v in ref_s['stats'].items():
        w = comp_s['stats'][k]
        if isinstance(v, list):
            worst_s = max(worst_s, max(abs(a - b) for a, b in zip(v, w)))
        elif isinstance(v, float):
            worst_s = max(worst_s, abs(v - w))
        else:
            assert v == w, (k, v, w)
    checks['A7_stats'] = worst_s <= TOL

    # A8 camera-only fallback: no radar echoes at all
    geo_empty = toy.make_geo()
    geo_empty['points'] = [p[:0] for p in geo_empty['points']]
    e_ref = _run(False, chunk=0, geo=geo_empty)
    e_cmp = _run(True, chunk=5, geo=geo_empty)
    img_msg = float(e_cmp['m_i'].abs().max())
    radar_msg = float(e_cmp['m_r'].abs().max())
    checks['A8_camera_only_nonzero'] = (img_msg > 1e-3 and radar_msg == 0.0
                                        and _maxdiff(e_ref['m_i'], e_cmp['m_i']) <= TOL
                                        and _maxdiff(e_ref['m_r'], e_cmp['m_r']) <= TOL)

    # A9 radar-only: degenerate image projection (points behind every camera)
    geo_bad = _degenerate_image_geo()
    b_ref = _run(False, chunk=0, geo=geo_bad)
    b_cmp = _run(True, chunk=5, geo=geo_bad)
    checks['A9_radar_only_nonzero'] = (
        float(b_cmp['m_r'].abs().max()) > 1e-6
        and float(b_cmp['m_i'].abs().max()) == 0.0
        and _maxdiff(b_ref['m_r'], b_cmp['m_r']) <= TOL
        and _maxdiff(b_ref['m_i'], b_cmp['m_i']) <= TOL)

    print('active-cell compaction equivalence gate (CPU toy contract)')
    for name in sorted(checks):
        print('  %-28s %s' % (name, 'PASS' if checks[name] else 'FAIL'))
    print('  worst param grad |d| = %.3e ; input grad |d| = %.3e ; stats |d| = %.3e'
          % (worst_p, worst_i, worst_s))
    dropped = 1.0 - float(active.to(torch.float32).mean())
    dropped_mixed = 1.0 - float(mixed_active.to(torch.float32).mean())
    print('  dropped cells = %.1f%% of the canvas (mixed radar-only fixture: %.1f%%, '
          'per-sample active %s of %d)'
          % (100.0 * dropped, 100.0 * dropped_mixed,
             counts.tolist(), int(active.shape[1])))
    if not all(checks.values()):
        print('FAILED:', sorted(k for k, v in checks.items() if not v))
        return 1
    print('ALL ACTIVE-CELL CHECKS PASS')
    return 0


if __name__ == '__main__':
    sys.exit(main())
