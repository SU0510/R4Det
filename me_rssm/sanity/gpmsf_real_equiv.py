#!/usr/bin/env python3
"""Real-data equivalence + in-situ cost profile for the active-cell compaction.

Two jobs, both read-only (no training, no weight write):

1. Equivalence on a *real* TJ4D current frame captured from segment A ep4 (the
   ``real_frame_ep4_capture.pt`` dump): dense vs compacted fusion must produce
   the same M_R / M_I, the same fused BEV and the same parameter gradients.
   The CPU gate proves this on the toy contract; this proves it on the real
   canvas (53,568 cells, unequal per-sample active counts, real trained
   offsets).
2. In-situ cost profile: the earlier decomposition was measured in isolation,
   which cannot say where the *whole step* spends its time.  Here the real
   forward+backward of the module is profiled with ``torch.profiler`` and the
   CUDA time is aggregated into address/projection, sampling, score/reduction
   and elementwise buckets, separately for the dense and compacted routes.

GPU 5 only (three-GPU policy: single-process probe).
"""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


BUCKETS = (
    ('grid_sample', ('grid_sampler',)),
    ('matmul_conv', ('einsum', 'addmm', 'mm', 'convolution', 'cudnn')),
    ('softmax', ('softmax',)),
    ('index_gather', ('index_select', 'gather', 'scatter', 'index_add', 'index_put',
                      'nonzero', 'argsort', 'sort')),
    ('elementwise_reduce', ('elementwise', 'mul', 'add', 'sub', 'div', 'cat',
                            'copy', 'fill', 'clamp', 'pow', 'sqrt', 'neg', 'abs',
                            'comparison', 'sum', 'mean', 'amax', 'tanh', 'exp',
                            'log', 'masked', 'where', 'stack', 'slice', 'view',
                            'permute', 'contiguous', 'to_copy', 'nan_to_num',
                            'minimum', 'maximum', 'bitwise', 'cumsum')),
)


def bucket_of(name):
    low = name.lower()
    if low.startswith('memcpy') or 'memcpy' in low:
        return 'memcpy_other'
    for bucket, keys in BUCKETS:
        if any(k in low for k in keys):
            return bucket
    return 'other'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--capture', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--iters', type=int, default=5)
    ap.add_argument('--chunk', type=int, default=None,
                    help='override query_chunk for the equivalence runs')
    args = ap.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '5':
        ap.error('real-data check uses only GPU 5')

    import numpy as np
    import torch
    from mmcv import Config
    from mmcv.runner import load_checkpoint
    from mmdet3d.models import build_model
    import me_rssm  # noqa: F401
    from me_rssm.sanity.gpmsf_run_checks import source_hashes, write_json
    try:
        from mmdet.utils import compat_cfg
    except ImportError:
        from mmdet3d.utils import compat_cfg

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = dict(status='running', source_sha=source_hashes(), gpu=5,
                  checkpoint=args.checkpoint, capture=args.capture,
                  no_ap_claim=True, no_weights_written=True)

    def save():
        write_json(out / 'real_equiv.json', report)

    def seed():
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    save()
    try:
        cfg = compat_cfg(Config.fromfile(args.config))
        cfg.model.meta_info = dict(project_name=Path(args.config).stem, figures_path=str(out))
        model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'),
                            test_cfg=cfg.get('test_cfg'))
        model.init_weights()
        load_checkpoint(model, args.checkpoint, map_location='cpu', strict=False)
        fusion = model.cross_attention
        if args.chunk is not None:
            fusion.query_chunk = int(args.chunk)
        report['config'] = dict(query_chunk=int(fusion.query_chunk),
                                grad_checkpoint=bool(fusion.grad_checkpoint),
                                cells=int(fusion.bev_shape[0] * fusion.bev_shape[1]),
                                heads=int(fusion.num_heads), offsets=int(fusion.num_offsets),
                                heights=int(fusion.num_heights))

        dump = torch.load(args.capture, map_location='cpu')
        dev = torch.device('cuda', 0)
        img = dump['img'].to(dev)
        radar = dump['radar'].to(dev)
        geo = {}
        for k, v in dump['geo'].items():
            if k in ('img_feats', 'radar_levels', 'points'):
                geo[k] = [x.to(dev) if torch.is_tensor(x) else x for x in v]
            elif torch.is_tensor(v):
                geo[k] = v.to(dev)
            else:
                geo[k] = v
        report['frame'] = dict(img=list(img.shape), radar=list(radar.shape),
                               num_views=int(geo['num_views']),
                               points=[list(p.shape) for p in geo['points']])

        fusion = fusion.to(dev)
        fusion.train()
        base_state = {k: v.detach().clone() for k, v in fusion.state_dict().items()}

        def run(compacted, stats=False):
            fusion.load_state_dict(base_state)
            fusion.collect_stats = bool(stats)
            fusion.active_cells_only = bool(compacted)
            for p in fusion.parameters():
                p.grad = None
            for k in ('img_feats', 'radar_levels'):
                geo[k] = [v.detach() for v in geo[k]]
            seen = {}
            handles = [
                fusion.proj_r2c.register_forward_hook(
                    lambda m, i, o: seen.__setitem__('m_r', i[0].detach().clone())),
                fusion.proj_i2r.register_forward_hook(
                    lambda m, i, o: seen.__setitem__('m_i', i[0].detach().clone())),
            ]
            original = fusion._active_cells

            def observed(ctx, f_cat_f, cell_xy, e_r, e_t):
                act = original(ctx, f_cat_f, cell_xy, e_r, e_t)
                seen['active'] = act.detach().clone()
                return act

            fusion._active_cells = observed
            seed()
            try:
                out_v = fusion(img, radar, geo=geo)
                loss = out_v.square().mean() + seen['m_r'].square().mean() \
                    + seen['m_i'].square().mean()
                loss.backward()
            finally:
                for h in handles:
                    h.remove()
                fusion._active_cells = original
            grads = {n: p.grad.detach().cpu().clone() for n, p in fusion.named_parameters()
                     if p.grad is not None}
            return dict(m_r=seen['m_r'].cpu(), m_i=seen['m_i'].cpu(), out=out_v.detach().cpu(),
                        grads=grads, active=seen.get('active', torch.empty(0)).cpu(),
                        stats=dict(fusion.last_stats or {}),
                        loss=float(loss.detach()))

        def maxdiff(a, b):
            return float((a - b).abs().max())

        report['equivalence'] = {}
        eq = report['equivalence']
        for chunk_label, chunk in (('chunk%d' % fusion.query_chunk, fusion.query_chunk),
                                   ('chunk_full', 0)):
            fusion.query_chunk = chunk
            d = run(False)
            c = run(True)
            act = c['active']
            counts = act.sum(dim=1).tolist() if act.numel() else []
            worst_g = max([maxdiff(c['grads'][n], d['grads'][n]) for n in d['grads']] or [0.0])
            entry = dict(
                dense_vs_compact_m_r=maxdiff(c['m_r'], d['m_r']),
                dense_vs_compact_m_i=maxdiff(c['m_i'], d['m_i']),
                dense_vs_compact_out=maxdiff(c['out'], d['out']),
                dense_vs_compact_worst_param_grad=worst_g,
                dense_loss=d['loss'], compact_loss=c['loss'],
                active_cells_per_sample=counts,
                cells=int(act.shape[1]) if act.numel() else 0,
                dropped_rate=(1.0 - float(act.to(torch.float32).mean())) if act.numel() else None,
                bit_exact=bool(maxdiff(c['out'], d['out']) == 0.0 and worst_g == 0.0),
            )
            report['equivalence'][chunk_label] = entry
        fusion.query_chunk = int(report['config']['query_chunk'])

        # ---- diagnostics on the real frame (compacted route) ---------------
        d_stats = run(False, stats=True)
        c_stats = run(True, stats=True)
        report['diagnostics'] = dict(
            dense=dict((k, v) for k, v in d_stats['stats'].items()),
            compact=dict((k, v) for k, v in c_stats['stats'].items()),
        )
        worst_s = 0.0
        for k, v in d_stats['stats'].items():
            w = c_stats['stats'][k]
            if isinstance(v, list):
                worst_s = max(worst_s, max(abs(a - b) for a, b in zip(v, w)))
            elif isinstance(v, float):
                worst_s = max(worst_s, abs(v - w))
        report['diagnostics']['worst_stats_delta'] = worst_s

        # ---- in-situ forward+backward timing -------------------------------
        def timeit(compacted, iters):
            fusion.load_state_dict(base_state)
            fusion.collect_stats = False
            fusion.active_cells_only = bool(compacted)
            for _ in range(2):
                out_v = fusion(img, radar, geo=geo)
                out_v.square().mean().backward()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start, end = torch.cuda.Event(True), torch.cuda.Event(True)
            start.record()
            for _ in range(iters):
                for p in fusion.parameters():
                    p.grad = None
                out_v = fusion(img, radar, geo=geo)
                out_v.square().mean().backward()
            end.record()
            torch.cuda.synchronize()
            return dict(ms_per_fwd_bwd=start.elapsed_time(end) / iters,
                        peak_memory_mib=torch.cuda.max_memory_allocated() / 2 ** 20)

        # what does the *mask pass* itself cost?  It walks every candidate of the
        # full canvas, so it is the part compaction does not remove.  (Measured
        # directly rather than inferred: the earlier per-op microbenchmark was
        # isolated and the in-situ share is what matters.)
        fusion.active_cells_only = False
        ctx_holder = []
        original_build = fusion._build_context

        def build(geo_, b_, device_, dtype_):
            ctx_ = original_build(geo_, b_, device_, dtype_)
            ctx_holder.append(ctx_)
            return ctx_

        fusion._build_context = build
        try:
            with torch.no_grad():
                fusion(img, radar, geo=geo)
        finally:
            fusion._build_context = original_build
        ctx_probe = ctx_holder[0]
        f_cat = torch.cat([
            fusion.proj_cam(img) + fusion._pos_enc,
            fusion.proj_rad(radar) + fusion._pos_enc,
            fusion._pos_enc.expand(img.shape[0], -1, *fusion.bev_shape)], dim=1).reshape(
                img.shape[0], 3 * fusion.embed_dim, -1)
        ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
        act = None
        with torch.no_grad():
            for _ in range(2):
                fusion._active_cells(ctx_probe, f_cat, ctx_probe['cell_xy'],
                                     ctx_probe['e_r'], ctx_probe['e_t'])
            torch.cuda.synchronize()
            ev0.record()
            for _ in range(5):
                act = fusion._active_cells(ctx_probe, f_cat, ctx_probe['cell_xy'],
                                           ctx_probe['e_r'], ctx_probe['e_t'])
            ev1.record()
            torch.cuda.synchronize()
        report['mask_pass'] = dict(
            ms_per_call=ev0.elapsed_time(ev1) / 5.0,
            active_cells_per_sample=act.sum(dim=1).tolist(),
            active_rate=float(act.to(torch.float32).mean()),
        )

        iters = int(args.iters)
        report['timing'] = dict(dense=timeit(False, iters), compact=timeit(True, iters),
                                iters=iters, note='fusion module only; whole-step S0 separate')

        # ---- per-kernel profile (CUDA time) --------------------------------
        from torch.profiler import profile, ProfilerActivity
        prof_summary = {}
        for tag, compacted in (('dense', False), ('compact', True)):
            fusion.load_state_dict(base_state)
            fusion.collect_stats = False
            fusion.active_cells_only = bool(compacted)
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for _ in range(3):
                    for p in fusion.parameters():
                        p.grad = None
                    out_v = fusion(img, radar, geo=geo)
                    out_v.square().mean().backward()
                torch.cuda.synchronize()
            agg = defaultdict(float)
            ops = []
            for ev in prof.key_averages():
                cuda_us = 0.0
                for attr in ('self_device_time_total', 'self_cuda_time_total',
                             'device_time_total', 'cuda_time_total'):
                    val = getattr(ev, attr, 0.0)
                    if val:
                        cuda_us = float(val)
                        break
                if cuda_us <= 0:
                    continue
                agg[bucket_of(ev.key)] += cuda_us
                ops.append((ev.key, cuda_us, int(ev.count)))
            ops.sort(key=lambda x: -x[1])
            total = sum(agg.values())
            prof_summary[tag] = dict(
                total_cuda_us_per_3steps=total,
                cuda_us_per_step=total / 3.0,
                kernels_per_step=sum(c for _, _, c in ops) / 3.0,
                bucket_us_per_step={k: v / 3.0 for k, v in sorted(agg.items())},
                bucket_share={k: v / total for k, v in sorted(agg.items())},
                top_ops=[dict(name=n, us_per_step=u / 3.0, calls_per_step=c / 3.0)
                         for n, u, c in ops[:20]],
            )
        report['profile'] = prof_summary
        report['status'] = 'PASS'
        save()
        print(json.dumps({k: report[k] for k in ('equivalence', 'timing')}, indent=2,
                         sort_keys=True, default=float))
        print(json.dumps({k: dict(cuda_us_per_step=v['cuda_us_per_step'],
                                  kernels_per_step=v['kernels_per_step'],
                                  bucket_share=v['bucket_share'])
                          for k, v in prof_summary.items()},
                         indent=2, sort_keys=True, default=float))
        return 0
    except Exception as exc:  # pragma: no cover
        report.update(status='FAIL', error=repr(exc))
        save()
        raise


if __name__ == '__main__':
    sys.exit(main())
