"""CPU-side Cyclist miss diagnosis for prediction dumps.

This extends tools/cyc_drop_analysis.py with three checks that matter for the
shared-stem regression:

1. same-class Cyclist loose/strict recall;
2. for each Cyclist GT, the best overlapping prediction of *any* class;
3. for GT boxes matched in the clean dump but missed by a new dump, whether
   the new dump still has an overlapping prediction under another label.

The dump stores final post-NMS predictions, so this cannot observe the raw
pre-`nms_pre` candidate set directly. It can still expose final-output
saturation and score loss, and it gives the input needed to decide whether an
inference-only `nms_pre` ablation is warranted.

Usage:
  python tools/cyc_drop_diagnosis.py \
      --base clean_seed0_ep16.pkl \
      --new stem_seed0_ep14.pkl stem_seed0_ep16.pkl
"""
import argparse
import pickle
from collections import Counter

import numpy as np
from shapely.geometry import Polygon


LOOSE_IOU = 0.25
STRICT_IOU = 0.5
AABB_PREFILTER_IOU = 0.05


def bev_corners(b):
    x, y, z, w, l, h, yaw = b
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    local = np.array([
        [-l / 2, -w / 2],
        [l / 2, -w / 2],
        [l / 2, w / 2],
        [-l / 2, w / 2],
    ])
    return local @ rot.T + np.array([x, y])


def poly(corners):
    p = Polygon(corners)
    return p.buffer(0) if not p.is_valid else p


def oriented_3d_iou(b1, b2):
    p1 = poly(bev_corners(b1))
    p2 = poly(bev_corners(b2))
    inter_area = p1.intersection(p2).area
    if inter_area <= 1e-12:
        return 0.0

    z1, h1 = b1[2], b1[5]
    z2, h2 = b2[2], b2[5]
    top = max(z1 - h1 / 2, z2 - h2 / 2)
    bot = min(z1 + h1 / 2, z2 + h2 / 2)
    inter_h = bot - top
    if inter_h <= 0:
        return 0.0

    inter_vol = inter_area * inter_h
    v1 = b1[3] * b1[4] * b1[5]
    v2 = b2[3] * b2[4] * b2[5]
    return inter_vol / (v1 + v2 - inter_vol + 1e-9)


def bev_aabb_iou(boxes, gt):
    """Fast axis-aligned prefilter for oriented BEV IoU."""
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.float32)

    boxes = np.asarray(boxes, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    corners = np.stack([bev_corners(b) for b in boxes], axis=0)
    pred_min = corners.min(axis=1)
    pred_max = corners.max(axis=1)

    gt_corners = bev_corners(gt)
    gt_min = gt_corners.min(axis=0)
    gt_max = gt_corners.max(axis=0)

    inter_min = np.maximum(pred_min, gt_min)
    inter_max = np.minimum(pred_max, gt_max)
    inter_wh = np.clip(inter_max - inter_min, 0.0, None)
    inter_area = inter_wh[:, 0] * inter_wh[:, 1]

    pred_area = np.prod(np.clip(pred_max - pred_min, 0.0, None), axis=1)
    gt_area = np.prod(np.clip(gt_max - gt_min, 0.0, None))
    union = pred_area + gt_area - inter_area + 1e-9
    aabb_iou = inter_area / union

    top = np.maximum(boxes[:, 2] - boxes[:, 5] / 2, gt[2] - gt[5] / 2)
    bot = np.minimum(boxes[:, 2] + boxes[:, 5] / 2, gt[2] + gt[5] / 2)
    no_height_overlap = bot <= top
    aabb_iou[no_height_overlap] = 0.0
    return aabb_iou


def best_match(gt_box, gt_label, pred_boxes, pred_labels, pred_scores,
               same_label_only):
    """Return best (iou, pred_idx, label, score) under the requested filter."""
    if len(pred_boxes) == 0:
        return (0.0, -1, -1, 0.0)

    pred_boxes = np.asarray(pred_boxes, dtype=np.float64)
    pred_labels = np.asarray(pred_labels)
    pred_scores = np.asarray(pred_scores, dtype=np.float64)
    if same_label_only:
        mask = pred_labels == gt_label
    else:
        mask = np.ones_like(pred_labels, dtype=bool)

    if not mask.any():
        return (0.0, -1, -1, 0.0)

    inds = np.flatnonzero(mask)
    boxes = pred_boxes[inds]
    labels = pred_labels[inds]
    scores = pred_scores[inds]

    # Exact oriented IoU is relatively expensive. Restrict it to candidates
    # with plausible axis-aligned overlap, then fall back to all candidates if
    # the prefilter is too tight for any box.
    aabb_iou = bev_aabb_iou(boxes, gt_box)
    candidates = np.flatnonzero(aabb_iou >= AABB_PREFILTER_IOU)
    if len(candidates) == 0:
        candidates = np.arange(len(boxes))

    best_iou = 0.0
    best_local = -1
    for local_idx in candidates:
        iou = oriented_3d_iou(gt_box, boxes[local_idx])
        if iou > best_iou:
            best_iou = iou
            best_local = int(local_idx)

    if best_local < 0:
        return (0.0, -1, -1, 0.0)

    pred_idx = int(inds[best_local])
    return (best_iou, pred_idx, int(labels[best_local]), float(scores[best_local]))


def greedy_same_class_matches(gt_boxes, gt_labels, pred_boxes, pred_labels,
                              pred_scores, label):
    """One-to-one score-descending same-class matching, as in cyc_drop_analysis."""
    gt_inds = [int(i) for i in np.flatnonzero(gt_labels == label)]
    pred_inds = [int(i) for i in np.flatnonzero(pred_labels == label)]
    pred_inds = sorted(pred_inds, key=lambda idx: -float(pred_scores[idx]))

    matches = {}
    used_gt = set()
    for pred_idx in pred_inds:
        best_iou = 0.0
        best_gt = None
        for gt_idx in gt_inds:
            if gt_idx in used_gt:
                continue
            iou = oriented_3d_iou(gt_boxes[gt_idx], pred_boxes[pred_idx])
            if iou > best_iou:
                best_iou = iou
                best_gt = gt_idx
        if best_gt is not None:
            used_gt.add(best_gt)
            matches[best_gt] = (best_iou, pred_idx, float(pred_scores[pred_idx]))
    return matches


def load_dump(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def class_index(class_names, name):
    if name not in class_names:
        raise KeyError(f'class {name!r} not present in {class_names}')
    return class_names.index(name)


def analyze(dump, cyc_idx):
    records = []
    sample_total_preds = []
    sample_cyc_preds = []
    sample_max_score = []
    cyc_scores = []
    all_scores = []

    for sample_idx, (gt, pred) in enumerate(zip(dump['gts'], dump['preds'])):
        gt_boxes = np.asarray(gt['tensor'], dtype=np.float64)
        gt_labels = np.asarray(gt['labels'])
        pred_boxes = np.asarray(pred['tensor'], dtype=np.float64)
        pred_labels = np.asarray(pred['labels'])
        pred_scores = np.asarray(pred['scores'], dtype=np.float64)

        sample_total_preds.append(len(pred_boxes))
        sample_cyc_preds.append(int((pred_labels == cyc_idx).sum()))
        if len(pred_scores) > 0:
            sample_max_score.append(float(pred_scores.max()))
            all_scores.extend(pred_scores.tolist())
        else:
            sample_max_score.append(0.0)
        cyc_scores.extend(pred_scores[pred_labels == cyc_idx].tolist())

        same_matches = greedy_same_class_matches(
            gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores, cyc_idx)

        for gt_idx in np.flatnonzero(gt_labels == cyc_idx):
            gt_box = gt_boxes[gt_idx]
            same_iou, same_pred_idx, same_score = same_matches.get(
                int(gt_idx), (0.0, -1, 0.0))
            any_iou, any_pred_idx, any_label, any_score = best_match(
                gt_box, cyc_idx, pred_boxes, pred_labels, pred_scores,
                same_label_only=False)

            records.append({
                'sample_idx': sample_idx,
                'gt_idx': int(gt_idx),
                'same_iou': same_iou,
                'same_pred_idx': same_pred_idx,
                'same_score': same_score,
                'any_iou': any_iou,
                'any_pred_idx': any_pred_idx,
                'any_label': any_label,
                'any_score': any_score,
            })

    return {
        'records': records,
        'sample_total_preds': np.asarray(sample_total_preds),
        'sample_cyc_preds': np.asarray(sample_cyc_preds),
        'sample_max_score': np.asarray(sample_max_score),
        'cyc_scores': np.asarray(cyc_scores),
        'all_scores': np.asarray(all_scores),
    }


def record_key(record):
    return record['sample_idx'], record['gt_idx']


def pct(value, denom):
    return value / denom if denom else 0.0


def quantiles(values):
    if len(values) == 0:
        return (float('nan'), float('nan'), float('nan'), float('nan'))
    return tuple(float(x) for x in np.quantile(values, [0.1, 0.5, 0.9, 0.99]))


def summarize(name, data, class_names, cyc_idx):
    records = data['records']
    total_gt = len(records)
    same_loose = sum(r['same_iou'] >= LOOSE_IOU for r in records)
    same_strict = sum(r['same_iou'] >= STRICT_IOU for r in records)
    any_loose = sum(r['any_iou'] >= LOOSE_IOU for r in records)
    any_strict = sum(r['any_iou'] >= STRICT_IOU for r in records)

    missed_same_loose = [r for r in records if r['same_iou'] < LOOSE_IOU]
    missed_but_other_loose = [
        r for r in missed_same_loose
        if r['any_iou'] >= LOOSE_IOU and r['any_label'] != cyc_idx
    ]
    missed_no_overlap = [
        r for r in missed_same_loose
        if r['any_iou'] < LOOSE_IOU
    ]
    other_label_counts = Counter(
        class_names[r['any_label']] for r in missed_but_other_loose
    )

    cyc_scores = data['cyc_scores']
    all_scores = data['all_scores']
    score_q = quantiles(cyc_scores)
    all_score_q = quantiles(all_scores)

    print(f'[{name}]')
    print(
        f'  Cyclist GT={total_gt}  same-class loose={same_loose} '
        f'({pct(same_loose, total_gt):.3f})  '
        f'strict={same_strict} ({pct(same_strict, total_gt):.3f})'
    )
    print(
        f'  Any-label overlap: loose={any_loose} ({pct(any_loose, total_gt):.3f})  '
        f'strict={any_strict} ({pct(any_strict, total_gt):.3f})'
    )
    print(
        f'  same-class loose miss: other-label overlap loose={len(missed_but_other_loose)}  '
        f'no overlap loose={len(missed_no_overlap)}'
    )
    if other_label_counts:
        label_text = ', '.join(
            f'{label}={count}' for label, count in other_label_counts.most_common())
        print(f'    other-label best-overlap labels: {label_text}')

    print(
        f'  final preds/sample mean={data["sample_total_preds"].mean():.2f} '
        f'median={np.median(data["sample_total_preds"]):.1f} '
        f'max={data["sample_total_preds"].max()} '
        f'cap300_samples={(data["sample_total_preds"] >= 300).sum()}'
    )
    print(
        f'  Cyc preds/sample mean={data["sample_cyc_preds"].mean():.2f} '
        f'median={np.median(data["sample_cyc_preds"]):.1f} '
        f'zero-cyc samples={(data["sample_cyc_preds"] == 0).sum()}'
    )
    if len(cyc_scores):
        print(
            f'  Cyc scores: n={len(cyc_scores)} mean={cyc_scores.mean():.3f} '
            f'q10/q50/q90/q99={score_q[0]:.3f}/{score_q[1]:.3f}/'
            f'{score_q[2]:.3f}/{score_q[3]:.3f} '
            f'frac>0.3={(cyc_scores > 0.3).mean():.3f} '
            f'frac>0.5={(cyc_scores > 0.5).mean():.3f}'
        )
    if len(all_scores):
        print(
            f'  All scores: n={len(all_scores)} mean={all_scores.mean():.3f} '
            f'q10/q50/q90/q99={all_score_q[0]:.3f}/{all_score_q[1]:.3f}/'
            f'{all_score_q[2]:.3f}/{all_score_q[3]:.3f}'
        )
    return {
        'total_gt': total_gt,
        'same_loose': same_loose,
        'same_strict': same_strict,
        'any_loose': any_loose,
        'any_strict': any_strict,
        'missed_but_other_loose': missed_but_other_loose,
        'missed_no_overlap': missed_no_overlap,
    }


def compare_lost(name, base_data, new_data, class_names, cyc_idx):
    base_records = {record_key(r): r for r in base_data['records']}
    new_records = {record_key(r): r for r in new_data['records']}
    common_keys = sorted(set(base_records) & set(new_records))
    if len(common_keys) != len(base_records) or len(common_keys) != len(new_records):
        print(
            f'[{name}] GT record mismatch: base={len(base_records)} '
            f'new={len(new_records)} common={len(common_keys)}'
        )

    lost = [
        key for key in common_keys
        if base_records[key]['same_iou'] >= LOOSE_IOU
        and new_records[key]['same_iou'] < LOOSE_IOU
    ]
    gained = [
        key for key in common_keys
        if base_records[key]['same_iou'] < LOOSE_IOU
        and new_records[key]['same_iou'] >= LOOSE_IOU
    ]

    lost_other = [
        key for key in lost
        if new_records[key]['any_iou'] >= LOOSE_IOU
        and new_records[key]['any_label'] != cyc_idx
    ]
    lost_empty = [
        key for key in lost
        if new_records[key]['any_iou'] < LOOSE_IOU
    ]
    lost_other_labels = Counter(
        class_names[new_records[key]['any_label']] for key in lost_other
    )
    lost_any_iou = np.asarray([new_records[k]['any_iou'] for k in lost])
    lost_base_scores = np.asarray([base_records[k]['same_score'] for k in lost])
    lost_new_any_scores = np.asarray([new_records[k]['any_score'] for k in lost])

    print(f'[{name}] clean-loose -> new-miss comparison')
    print(f'  lost same-class loose={len(lost)}  gained same-class loose={len(gained)}')
    print(
        f'  among lost: other-label overlap loose={len(lost_other)}  '
        f'no any-label overlap loose={len(lost_empty)}'
    )
    if lost_other_labels:
        label_text = ', '.join(
            f'{label}={count}'
            for label, count in lost_other_labels.most_common())
        print(f'    other-label best-overlap labels: {label_text}')
    if len(lost):
        base_q = quantiles(lost_base_scores)
        any_q = quantiles(lost_new_any_scores)
        iou_q = quantiles(lost_any_iou)
        print(
            f'  lost GT base matched score q10/q50/q90/q99='
            f'{base_q[0]:.3f}/{base_q[1]:.3f}/{base_q[2]:.3f}/{base_q[3]:.3f}'
        )
        print(
            f'  lost GT new any-label score q10/q50/q90/q99='
            f'{any_q[0]:.3f}/{any_q[1]:.3f}/{any_q[2]:.3f}/{any_q[3]:.3f}'
        )
        print(
            f'  lost GT new any-label IoU q10/q50/q90/q99='
            f'{iou_q[0]:.3f}/{iou_q[1]:.3f}/{iou_q[2]:.3f}/{iou_q[3]:.3f}'
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True, help='clean reference dump')
    parser.add_argument('--new', nargs='+', required=True,
                        help='new dump(s) to compare against --base')
    return parser.parse_args()


def main():
    args = parse_args()
    base_dump = load_dump(args.base)
    class_names = list(base_dump['class_names'])
    cyc_idx = class_index(class_names, 'Cyclist')

    print('=' * 78)
    print('Cyclist miss diagnosis')
    print(f'  class_names={class_names}  Cyclist label={cyc_idx}')
    print('=' * 78)
    base_data = analyze(base_dump, cyc_idx)
    summaries = {
        args.base: summarize('base', base_data, class_names, cyc_idx),
    }

    for path in args.new:
        dump = load_dump(path)
        if list(dump['class_names']) != class_names:
            raise ValueError(f'class_names mismatch for {path}: {dump["class_names"]}')
        data = analyze(dump, cyc_idx)
        summaries[path] = summarize(path, data, class_names, cyc_idx)
        compare_lost(path, base_data, data, class_names, cyc_idx)

    for path, summary in summaries.items():
        if path == args.base:
            continue
        base_summary = summaries[args.base]
        print(f'[delta] {path} - base')
        print(
            f'  same-class loose recall delta='
            f'{pct(summary["same_loose"], summary["total_gt"]) - pct(base_summary["same_loose"], base_summary["total_gt"]):+.3f}'
        )
        print(
            f'  same-class strict recall delta='
            f'{pct(summary["same_strict"], summary["total_gt"]) - pct(base_summary["same_strict"], base_summary["total_gt"]):+.3f}'
        )
        print(
            f'  any-label loose recall delta='
            f'{pct(summary["any_loose"], summary["total_gt"]) - pct(base_summary["any_loose"], base_summary["total_gt"]):+.3f}'
        )


if __name__ == '__main__':
    main()
