#!/usr/bin/env python3
"""Read-only: is A's sem-head val failure a RANKING or a CALIBRATION failure?

WHY THIS EXISTS -- what section 71.9.19 left on the table

71.9.19 located the train/val gap in the DATA SPLIT (same weights, same val
pipeline, same eval mode: train 0.4739/0.4167 vs val 0.1280/0.0417 at n=20).
M1's two readings are both THRESHOLD readings:

    thr50 -- fraction of GT-centre cells whose sigmoid prob exceeds 0.5
    hit@1 -- whether the single argmax cell of each occupied class channel
             lands on a GT-centre cell

Both are sensitive to the head's probability SCALE: a map that ranks objects
above background perfectly but is uniformly under-confident scores 0.0 on
thr50 while carrying all the information a detector could use. Conversely a map
that is well-scaled but spatially wrong can pass neither.

Those two failure modes imply completely different next experiments (recalibrate
/ reweight the loss vs. change what the head sees), so the re-preregistration
needs them separated BEFORE any new training is designed. This probe measures
scale-free and threshold-free readings on both splits with NO extra weights and
NO extra forward passes beyond the ones 71.9.19 already used:

    auc              -- P(prob at a GT-centre cell > prob at a background cell),
                        aggregated per clip over all (cell, class) pairs. This
                        is the rank reading: it is invariant to any monotone
                        rescaling of the map.
    mean_gt_prob     -- mean sigmoid prob AT GT-centre cells (the scale reading)
    mean_bg_prob     -- mean sigmoid prob at background cells
    max_gt_prob      -- the best GT-centre probability in the clip; if this is
                        below 0.5 the threshold reading CANNOT pass no matter
                        where the peak is
    best_thr_recall  -- the best thr50 achievable by sweeping the threshold,
                        i.e. the ceiling of the fixed-0.5 bar
    argmax_dist_cells-- mean distance (in BEV cells) from each occupied
                        channel's argmax to the nearest GT centre of that class;
                        separates "peaks in the right place but weakly" from
                        "peaks in the wrong place"

Read-only: no training, no weight change, no checkpoint written. The z contract
is pinned to the deploy ``mu_q`` and the model stays in eval.

Usage:
  CUDA_VISIBLE_DEVICES=5 python me_rssm/sanity/probe_semA_rank.py \
    --config me_rssm/configs/..._semreadout.py \
    --checkpoint <work_dir>/epoch_6.pth \
    --start-index 3 --stride 100 --limit 20 \
    --output <work_dir>/semA_rank_val_ep6.json
  # and the same with --split-ann-file data/TJ4D/TJ4D_infos_train.pkl
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
    p.add_argument('--limit', type=int, default=20)
    p.add_argument('--stride', type=int, default=100)
    p.add_argument('--min-valid-history', type=int, default=2)
    p.add_argument('--split-ann-file', default=None,
                   help='read a different split through the same val pipeline '
                        '(e.g. data/TJ4D/TJ4D_infos_train.pkl)')
    p.add_argument('--output', required=True)
    return p.parse_args()


def average_rank_auc(scores, positive):
    """Tie-corrected AUC = P(score at a positive > score at a negative).

    Computed from average ranks so that a saturated map (many exactly equal
    probabilities -- which the sem head does produce for background cells)
    cannot be credited with discrimination it does not have. A naive
    ``argsort`` rank assignment breaks ties arbitrarily and would report an
    arbitrary AUC between 0 and 1 for a constant map.
    """
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return None
    values, inverse, counts = torch.unique(
        scores, return_inverse=True, return_counts=True)
    cumulative = torch.cumsum(counts, dim=0)
    before = cumulative - counts
    # Ranks 1..n; for a tie group spanning [before+1, before+count] the average
    # rank is (before + cumulative + 1) / 2.
    avg_rank = (before + cumulative + 1).to(scores.dtype) / 2.0
    ranks = avg_rank[inverse]
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0)
                 / (n_pos * n_neg))


def rank_readings(sem_map, target, num_classes):
    """Scale-free / threshold-free readings for ONE clip.

    ``sem_map`` is (1, C, H, W) logits, ``target`` the (1, C, H, W) 0/1 GT
    rasterisation on the same grid. Everything is aggregated per clip so a clip
    with many objects cannot dominate a mean over clips.

    The two readings that matter, and why:

      * ``auc`` is invariant to ANY monotone rescaling of the map. High AUC
        with low ``thr50`` means the head ranks objects above background but is
        under-confident, and the fix is the loss/calibration, not the input.
      * ``recall_at_top_k`` feeds the head's own top-``num_pos`` cells (the same
        count as the GT centres) through the target: it is the best recall a
        threshold could buy if the map were rescaled to put its top-k at the
        bar. Low AUC AND low recall_at_top_k means the map does not know where
        the objects are at all -- a different failure and a different fix.
    """
    prob = sem_map.float().sigmoid()[0]          # (C, H, W)
    pos = target[0].eq(1.0)                      # (C, H, W)
    if not bool(pos.any()):
        return None
    c, h, w = prob.shape
    aucs, gt_probs, bg_probs, max_gt = [], [], [], []
    topk, argmax_dists = [], []
    for k in range(c):
        pk, tk = prob[k].flatten(), pos[k].flatten()
        n_pos = int(tk.sum())
        if n_pos == 0:
            continue
        p, b = pk[tk], pk[~tk]
        gt_probs.append(float(p.mean()))
        bg_probs.append(float(b.mean()))
        max_gt.append(float(p.max()))
        auc = average_rank_auc(pk, tk)
        if auc is not None:
            aucs.append(auc)
        # Top-k recall: keep exactly the head's k highest-scoring cells, where
        # k = the number of GT-centre cells in this channel.
        kth = torch.topk(pk, min(n_pos, pk.numel())).indices
        topk.append(float(tk[kth].float().mean()))
        # Localisation of this channel's single argmax vs the nearest GT cell.
        ai = int(pk.argmax())
        ay, ax = ai // w, ai % w
        ys, xs = torch.nonzero(pos[k].view(h, w), as_tuple=True)
        d = ((ys - ay).abs() + (xs - ax).abs()).min()
        argmax_dists.append(float(d))
    if not aucs:
        return None
    return {
        'auc_mean_over_occupied_channels': sum(aucs) / len(aucs),
        'auc_per_channel': aucs,
        'mean_gt_prob': sum(gt_probs) / len(gt_probs),
        'mean_bg_prob': sum(bg_probs) / len(bg_probs),
        'max_gt_prob': sum(max_gt) / len(max_gt),
        'recall_at_top_k_per_channel': topk,
        'recall_at_top_k_mean': sum(topk) / len(topk),
        'argmax_cityblock_dist_cells': (
            sum(argmax_dists) / len(argmax_dists)),
        'occupied_channels': len(aucs),
    }


def mean_of(rows, key):
    vals = [r[key] for r in rows if r.get(key) is not None]
    return float(sum(vals)) / len(vals) if vals else None


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
    cfg.model['meta_info'] = {'figures_path': '/tmp/r4det_semA_rank_figures',
                              'project_name': 'TJ4D-semA-rank'}
    os.makedirs(cfg.model['meta_info']['figures_path'], exist_ok=True)
    num_classes = int(cfg.model.temporal_fusion.sem_num_classes)
    min_radius = int(cfg.model.temporal_fusion.object_min_radius)
    pcr = list(cfg.model.point_cloud_range)
    cfg.data.val.pop('samples_per_gpu', None)
    cfg.data.val.pop('workers_per_gpu', None)
    if args.split_ann_file is not None:
        cfg.data.val.ann_file = args.split_ann_file

    dataset = build_dataset(cfg.data.val)
    model = build_detector(cfg.model, train_cfg=None, test_cfg=None)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = model.cuda().eval()
    model.temporal_fusion.log_sem_maps = True

    indices = ep6.select_clips(
        dataset, ep6._Sel(args.start_index, args.limit,
                          args.min_valid_history, args.stride))

    raw = []
    for idx in indices:
        run = ep6.rollout(model, dataset, idx, sem_input='z',
                          rssm_deterministic=True)
        if run['sem_map'] is None:
            continue
        sm = run['sem_map']
        sm = sm[None] if sm.dim() == 3 else sm
        target = ep6.make_target(run, num_classes, pcr, min_radius).to(sm.device)
        raw.append({'index': int(idx), 'sem_map': sm, 'target': target,
                    'fixed': ep6.head_quality(run, num_classes, pcr, min_radius),
                    'image_index': int(
                        dataset.data_infos[idx]['image']['image_idx'])})

    rows = []
    for pos, row in enumerate(raw):
        q = rank_readings(row['sem_map'], row['target'], num_classes)
        fixed = row['fixed']
        if q is None:
            continue
        # GT-shuffle control: score the SAME map against a DIFFERENT clip's
        # target. AUC answers "does the map rank this clip's GT cells above
        # this clip's background cells"; if it answers equally well against a
        # donor's target, what is being measured is a generic map elevation in
        # the busy part of the BEV, not object alignment. Without this control
        # a high AUC cannot be attributed to object content at all.
        donor = raw[(pos + 1) % len(raw)] if len(raw) >= 2 else None
        shuf = (None if donor is None or donor['index'] == row['index']
                else rank_readings(row['sem_map'], donor['target'],
                                   num_classes))
        rows.append(dict(
            q, index=row['index'], image_index=row['image_index'],
            fixed_thr50=None if fixed is None else fixed['thr50'],
            fixed_hit1=None if fixed is None else fixed['hit1'],
            gt_shuffle_auc=None if shuf is None
            else shuf['auc_mean_over_occupied_channels'],
            gt_shuffle_recall_at_top_k=None if shuf is None
            else shuf['recall_at_top_k_mean'],
            gt_shuffle_donor_index=None if donor is None else donor['index']))

    report = {
        'probe': 'candidate A sem-head rank-vs-calibration decomposition',
        'config': os.path.abspath(args.config),
        'checkpoint': os.path.abspath(args.checkpoint),
        'ann_file': str(dataset.ann_file),
        'indices': [int(i) for i in indices],
        'n_clips': len(rows),
        'reading_note': (
            'these booleans describe which failure mode the readings point at; '
            'they are not gates and not a pass/fail of candidate A.'),
        'per_clip': rows,
        'summary': {
            'auc_mean': mean_of(rows, 'auc_mean_over_occupied_channels'),
            'mean_gt_prob': mean_of(rows, 'mean_gt_prob'),
            'mean_bg_prob': mean_of(rows, 'mean_bg_prob'),
            'max_gt_prob': mean_of(rows, 'max_gt_prob'),
            'recall_at_top_k_mean': mean_of(rows, 'recall_at_top_k_mean'),
            'fixed_thr50_mean': mean_of(rows, 'fixed_thr50'),
            'fixed_hit1_mean': mean_of(rows, 'fixed_hit1'),
            'argmax_cityblock_dist_cells': mean_of(
                rows, 'argmax_cityblock_dist_cells'),
            # The control that decides whether the AUC above is object
            # alignment or a generic spatial prior: 0.5 means the map's ranking
            # is specific to its own clip's objects.
            'gt_shuffle_auc_mean': mean_of(rows, 'gt_shuffle_auc'),
            'gt_shuffle_recall_at_top_k_mean': mean_of(
                rows, 'gt_shuffle_recall_at_top_k'),
        },
    }
    s = report['summary']
    # The two failure modes this probe exists to separate. Recorded as booleans
    # rather than prose so the report cannot be read as a pass/fail gate.
    report['reading'] = {
        'ranks_objects_above_background_auc_ge_0_8': bool(
            s['auc_mean'] is not None and s['auc_mean'] >= 0.8),
        'threshold_is_the_bottleneck': bool(
            s['recall_at_top_k_mean'] is not None
            and s['fixed_thr50_mean'] is not None
            and s['recall_at_top_k_mean'] >= 2.0 * max(s['fixed_thr50_mean'],
                                                       1e-6)),
        'argmax_lands_on_gt_cell': bool(
            s['argmax_cityblock_dist_cells'] is not None
            and s['argmax_cityblock_dist_cells'] < 1.0),
        'auc_is_specific_to_this_clips_objects': bool(
            s['auc_mean'] is not None and s['gt_shuffle_auc_mean'] is not None
            and s['auc_mean'] >= 0.8 and s['gt_shuffle_auc_mean'] <= 0.65),
    }
    report['input_fingerprints'] = ep6.input_fingerprints(
        args.config, args.checkpoint)
    ep6.write_report(report, args.output)
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ('per_clip', 'input_fingerprints')},
                     ensure_ascii=False, indent=2, sort_keys=True))
    for r in rows:
        print(f"  idx={r['index']:5d} img={r['image_index']:6d} "
              f"auc={r['auc_mean_over_occupied_channels']:.4f} "
              f"gtp={r['mean_gt_prob']:.5f} bgp={r['mean_bg_prob']:.5f} "
              f"maxgt={r['max_gt_prob']:.4f} topk={r['recall_at_top_k_mean']:.3f} "
              f"thr50={r['fixed_thr50']:.3f} hit1={r['fixed_hit1']:.3f} "
              f"dist={r['argmax_cityblock_dist_cells']:.1f} "
              f"shufAuc={r['gt_shuffle_auc']}")


if __name__ == '__main__':
    main()
