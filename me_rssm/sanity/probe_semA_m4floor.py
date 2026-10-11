#!/usr/bin/env python3
"""Read-only: is M4's ``own_hit`` distinguishable from the class-PRIOR floor?

WHY THIS EXISTS -- the section-72 archive records M4 as FAIL on
``donor_transfer`` (0.3818 < 0.45) while ``own_hit`` passes (0.7379 >= 0.45).
That asymmetry is only interpretable against the right control: a z residual
that simply always votes the MOST COMMON GT class scores exactly the majority
rate at foreground cells and would "pass" a 0.45 bar whenever the class prior
is skewed. This is the same error class as section 70.6, where a "93.7%
sample-specific" reading turned out to equal the pure-noise floor -- a number
that cannot be distinguished from the trivial strategy is not evidence.

So this probe recomputes, on the SAME checkpoint, SAME 10 val clips and SAME
rasteriser as the M1-M4 gate:

  * ``per_clip_prior_floor`` -- accuracy of the constant majority-class
    predictor at that clip's fg cells (the trivial strategy);
  * ``pooled_prior_floor``   -- the same for a single constant over the window;
  * ``recomputed_mean_own_hit`` -- re-derived from the residual so it is
    joined to the floor by construction, not quoted from the gate JSON.

Only the MARGIN over that floor is evidence of class-specific information in
z. Read-only: no training, no checkpoint write, no weight change.
"""
import argparse
import importlib.util
import json
import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import torch  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    'ep6', os.path.join(_REPO, 'me_rssm', 'sanity', 'probe_semreadout_ep6.py'))
ep6 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ep6)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--gpu-id', type=int, default=0)
    p.add_argument('--start-index', type=int, default=3)
    p.add_argument('--limit', type=int, default=10)
    p.add_argument('--stride', type=int, default=200)
    p.add_argument('--min-valid-history', type=int, default=2)
    p.add_argument('--output', required=True)
    return p.parse_args()


def main():
    from mmcv import Config
    from mmcv.runner import load_checkpoint
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_detector

    args = parse_args()
    torch.cuda.set_device(args.gpu_id)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model['meta_info'] = {'figures_path': '/tmp/r4det_m4floor_figures',
                              'project_name': 'TJ4D-semA-m4floor'}
    os.makedirs(cfg.model['meta_info']['figures_path'], exist_ok=True)
    num_classes = int(cfg.model.temporal_fusion.sem_num_classes)
    min_radius = int(cfg.model.temporal_fusion.object_min_radius)
    pcr = list(cfg.model.point_cloud_range)
    cfg.data.val.pop('samples_per_gpu', None)
    cfg.data.val.pop('workers_per_gpu', None)

    dataset = build_dataset(cfg.data.val)
    model = build_detector(cfg.model, train_cfg=None, test_cfg=None)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = model.cuda().eval()
    model.temporal_fusion.log_sem_maps = True

    indices = ep6.select_clips(
        dataset, ep6._Sel(args.start_index, args.limit,
                          args.min_valid_history, args.stride))
    runs = [ep6.rollout(model, dataset, i, sem_input='z') for i in indices]

    rows, all_labels = [], []
    for idx, run in zip(indices, runs):
        tgt = ep6.make_target(run, num_classes, pcr, min_radius)
        fg = tgt[0].eq(1.0).any(dim=0).flatten()
        sel = torch.nonzero(fg, as_tuple=False).flatten()
        if sel.numel() == 0 or run['residual'] is None:
            continue
        labels = tgt[0].reshape(num_classes, -1).permute(1, 0)[sel].argmax(dim=1)
        own = ep6.residual_class_map(run, num_classes)[sel].argmax(dim=1)
        counts = torch.bincount(labels, minlength=num_classes).tolist()

        # Decision margin AT FG CELLS. A whole-map conv_cls RMS is dominated by
        # background logits (very negative), so it cannot answer "could this
        # residual flip the class argmax here?". The operative quantity is the
        # top1-top2 gap of conv_cls at the foreground cells the residual
        # competes on, next to the residual's own top1-top2 gap.
        scores = ep6.class_scores_from_head(run, num_classes)[sel]
        cs, _ = scores.topk(2, dim=1)
        cls_gap = (cs[:, 0] - cs[:, 1])
        res = ep6.residual_class_map(run, num_classes)[sel]
        rs, _ = res.topk(2, dim=1)
        res_gap = (rs[:, 0] - rs[:, 1])
        res_abs = res.abs().max(dim=1).values

        # HEAD accuracy at fg cells, and its response to the residual's scale.
        # This is the decision-relevant question the M4 donor swap only probes
        # indirectly: if the head is ALREADY more accurate than the residual
        # (0.7379), amplifying the residual cannot help and the readout is not
        # the binding constraint; if the head is worse, the gap is the ceiling
        # a scale change could reach. k=0 is residual-off, k=1e6 forces the
        # head argmax to equal the residual argmax (internal consistency check
        # against `residual own_hit`).
        res_head = run['residual']
        k_curve = {}
        for k in (0.0, 1.0, 2.0, 4.0, 8.0, 1e6):
            scaled = None if res_head is None else res_head * k
            swapped = ep6.head_scores_with_residual(model, run, scaled)
            hm = ep6.class_scores_from_head({'cls_score': swapped},
                                            num_classes)
            k_curve[k] = float((hm[sel].argmax(dim=1) == labels)
                               .float().mean())

        rows.append({
            'index': int(idx), 'cells': int(sel.numel()),
            'own_hit': float((own == labels).float().mean()),
            'cell_class_counts': counts,
            'conv_cls_gap_fg_mean': float(cls_gap.mean()),
            'conv_cls_gap_fg_median': float(cls_gap.median()),
            'residual_gap_fg_mean': float(res_gap.mean()),
            'residual_absmax_fg_mean': float(res_abs.mean()),
            'residual_absmax_fg_min': float(res_abs.min()),
            'frac_cells_residual_can_flip':
                float((res_abs > cls_gap).float().mean()),
            'head_hit_by_residual_scale': k_curve,
        })
        all_labels.append(labels)

    # Pooled over clips (cell-weighted) for the scale curve, so a 1-cell clip
    # cannot swing the reading.
    scales = sorted({k for r in rows for k in r['head_hit_by_residual_scale']})
    pooled_curve = {}
    for k in scales:
        num = den = 0.0
        for r in rows:
            num += r['head_hit_by_residual_scale'][k] * r['cells']
            den += r['cells']
        pooled_curve[str(k)] = num / den if den else None

    pooled = torch.cat(all_labels)
    counts = torch.bincount(pooled, minlength=num_classes).float()

    def _mean(vals):
        return float(sum(vals)) / len(vals) if vals else None

    # Head-side scale: how large is the residual relative to the conv_cls
    # logits it is added to? M4's head_swap_to_donor (0.1167 < chance 0.25)
    # says the head's argmax barely moves when the residual is swapped; a
    # residual that is an order of magnitude smaller than conv_cls would
    # explain that mechanically instead of attributing it to "the head
    # ignoring z".
    res_rms, cls_rms = [], []
    for run in runs:
        if run['residual'] is None:
            continue
        res_rms.append(float(run['residual'].float().pow(2).mean().sqrt()))
        score = run['cls_score'][0][0]
        cls_rms.append(float(score.float().pow(2).mean().sqrt()))
    head_scale = {
        'residual_rms_mean': _mean(res_rms),
        'conv_cls_rms_mean': _mean(cls_rms),
        'residual_over_conv_cls': (_mean(res_rms) / _mean(cls_rms)
                                   if _mean(cls_rms) else None),
        'note': ('conv_cls is the head logit the residual is ADDED to; a ratio '
                 'far below 1 means the residual cannot flip the head argmax '
                 'at any realistic logit gap, which is a scale statement, not '
                 'an information statement.'),
    }

    report = {
        'probe': 'M4 own_hit vs class-prior floor (read-only)',
        'config': os.path.abspath(args.config),
        'checkpoint': os.path.abspath(args.checkpoint),
        'indices': [int(i) for i in indices],
        'per_clip': rows,
        'pooled_class_counts': [int(c) for c in counts.tolist()],
        'pooled_prior_floor': float(counts.max() / counts.sum()),
        'pooled_majority_class': int(counts.argmax()),
        'per_clip_prior_floor_mean': _mean(
            [max(r['cell_class_counts']) / r['cells'] for r in rows]),
        'per_clip_prior_floor_max': max(
            [max(r['cell_class_counts']) / r['cells'] for r in rows]),
        'recomputed_mean_own_hit': _mean([r['own_hit'] for r in rows]),
        'mean_cells_per_clip': _mean([r['cells'] for r in rows]),
        'conv_cls_gap_fg_mean': _mean(
            [r['conv_cls_gap_fg_mean'] for r in rows]),
        'conv_cls_gap_fg_median': _mean(
            [r['conv_cls_gap_fg_median'] for r in rows]),
        'residual_gap_fg_mean': _mean(
            [r['residual_gap_fg_mean'] for r in rows]),
        'residual_absmax_fg_mean': _mean(
            [r['residual_absmax_fg_mean'] for r in rows]),
        'frac_cells_residual_can_flip': _mean(
            [r['frac_cells_residual_can_flip'] for r in rows]),
        'chance_1_over_c': 1.0 / num_classes,
        'head_hit_by_residual_scale_pooled': pooled_curve,
        'head_hit_residual_off': pooled_curve.get('0.0'),
        'head_hit_residual_as_trained': pooled_curve.get('1.0'),
        'head_hit_residual_forced_max': pooled_curve.get('1000000.0'),
        'head_scale': head_scale,
        'note': ('prior_floor = accuracy of a constant majority-class predictor '
                 'at fg cells. A z residual that only ever votes the majority '
                 'class scores exactly this. Only the MARGIN over this floor is '
                 'evidence of class-specific information in z.'),
    }
    report['margin_over_pooled_prior'] = (
        report['recomputed_mean_own_hit'] - report['pooled_prior_floor'])
    report['input_fingerprints'] = ep6.input_fingerprints(
        args.config, args.checkpoint)
    ep6.write_report(report, args.output)
    print(json.dumps({k: v for k, v in report.items() if k not in
                      ('per_clip', 'input_fingerprints')},
                     ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
