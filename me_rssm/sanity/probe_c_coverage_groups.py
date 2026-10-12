#!/usr/bin/env python3
"""Candidate-C lien-defence diagnostic, step 1: the radar-coverage grouping contract.

WHY THIS EXISTS (2026-10-12). Candidate C ("why is the deterministic state not
enough when the sensor is missing") is a HYPOTHESIS, not a finding: nothing
measured so far ties the A/A2 failures to radar dropout. Section 74.11 only
showed that the train/val gap is the data split and not the BN mode; it did not
say WHICH property of the split, and the A2 arms failed their M1 gate on all
three arms equally, i.e. the failure may just as well be scene composition,
localisation, object scale or supervision generalisation.

Before any C arm is pre-registered, the grouping has to exist and be checked
for the two ways such a study usually goes wrong:

  1. **Thresholds picked after seeing the answers.** Every cut here is derived
     from the TRAIN split only and then applied verbatim to val. The val pass
     never influences a boundary.
  2. **A "missing-radar" group that is really a distance/size/scene group.**
     Objects with zero radar returns are, in every radar dataset, mostly the
     far and small ones -- so every group is reported with its own median
     range, box volume, image bbox height, occlusion flag and scene
     concentration. If the zero group is also the far group, the enrichment
     study cannot separate "no measurement" from "hard object".

What counts as coverage: ``annos['num_points_in_gt']``, which
``tools/data_converter/kitti_converter_TJ4Dradar.py::_calculate_num_points_in_gt``
computes as the number of points of the SAME 8-feature point cloud the model
loads (``load_dim=8, use_dim=[0,1,2,3,5]``) falling inside each rotated GT box.
So a zero is a genuine "this object produced no return in the tensor the
network sees", not a proxy from another sensor. Ignored objects
(``num_points_in_gt < 0``) and ``DontCare`` are excluded from every count.

Scenes are recovered from the frame numbering itself: ``image_idx`` is
contiguous inside a sequence, so a jump > 1 starts a new one. That needs no
assumption about how TJ4D encodes sequence ids in the filename.

Read-only: two pkl files in, one JSON out. No model, no GPU, no weights.

Usage:
  python me_rssm/sanity/probe_c_coverage_groups.py \
    --train-info data/TJ4D/TJ4D_infos_train.pkl \
    --val-info data/TJ4D/TJ4D_infos_val.pkl \
    --output /tmp/c_coverage_groups.json
"""

import argparse
import collections
import json
import os
import pickle
import sys

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

IGNORED_COVERAGE = -1
DONTCARE = 'DontCare'
# The cut between "zero returns" and "a few returns" is the one boundary that
# is NOT a quantile: 0 is a qualitative state (the object is absent from the
# tensor), and no train statistic should be allowed to move it. Only the
# boundary between weak / moderate / dense is data-derived.
ZERO_GROUP = 'zero'


def load_infos(path):
    """TJ4D infos are a bare list; tolerate the KITTI-style {'infos': ...}."""
    with open(path, 'rb') as handle:
        data = pickle.load(handle)
    if isinstance(data, dict) and 'infos' in data:
        return list(data['infos'])
    return list(data)


def scene_runs(indices):
    """{image_idx: scene_ordinal} from contiguity of the frame numbering.

    A sequence is a maximal run of consecutive ``image_idx`` values: TJ4D jumps
    to a new numbering block at a scene boundary (val: 11 runs, train: 34).
    Any other rule (filename prefix arithmetic) would be an assumption about an
    encoding this probe does not need to know.
    """
    ordered = sorted(int(i) for i in indices)
    runs, current = {}, 0
    previous = None
    for idx in ordered:
        if previous is not None and idx - previous > 1:
            current += 1
        runs[idx] = current
        previous = idx
    return runs


def _image_bbox_height(bbox):
    if bbox is None or len(bbox) < 4:
        return None
    return float(bbox[3]) - float(bbox[1])


def object_rows(infos, split, runs):
    """One row per GT object: coverage + the confound controls.

    Rows are what the later per-group failure analysis will attach detection
    outcomes to, so they carry the frame identity as well as the object
    geometry.
    """
    rows = []
    for info in infos:
        idx = int(info['image']['image_idx'])
        annos = info['annos']
        names = list(annos['name'])
        counts = np.asarray(annos['num_points_in_gt'], dtype=np.int64)
        loc = np.asarray(annos['location'], dtype=np.float64)
        dims = np.asarray(annos['dimensions'], dtype=np.float64)
        boxes = np.asarray(annos['bbox'], dtype=np.float64)
        occ = np.asarray(annos.get('occluded', np.zeros(len(names))),
                         dtype=np.float64)
        diff = np.asarray(annos.get('difficulty', np.full(len(names), -1)),
                          dtype=np.float64)
        truncated = np.asarray(
            annos.get('truncated', np.zeros(len(names))), dtype=np.float64)
        for k, name in enumerate(names):
            if name == DONTCARE:
                continue
            coverage = int(counts[k])
            if coverage <= IGNORED_COVERAGE:
                continue
            # Range in the camera frame: z is forward, x lateral. Reported in
            # metres so it is readable next to the KITTI-ish geometry.
            box_range = float(np.hypot(loc[k][0], loc[k][2]))
            rows.append({
                'split': split,
                'scene': runs.get(idx, -1),
                'image_idx': idx,
                'class': str(name),
                'coverage': coverage,
                'range_m': box_range,
                'volume_m3': float(np.prod(dims[k])),
                'bbox_h_px': _image_bbox_height(boxes[k]),
                'occluded': float(occ[k]),
                'truncated': float(truncated[k]),
                'difficulty': float(diff[k]),
            })
    return rows


def derive_cuts(train_rows):
    """Group boundaries from the TRAIN split only.

    zero: 0 returns (qualitative, never data-derived).
    low / mid / high: terciles of the TRAIN positive-coverage population, so
    each group holds ~a third of the objects that do return something. The cut
    values are written into the artifact and reused verbatim on val.
    """
    positive = np.asarray([r['coverage'] for r in train_rows
                           if r['coverage'] > 0], dtype=np.float64)
    if positive.size == 0:
        return {'zero': [0, 0], 'low': [1, 1], 'mid': [2, 2],
                'high': [3, None], 'q1': None, 'q2': None}
    q1, q2 = (float(np.percentile(positive, 100.0 / 3.0)),
              float(np.percentile(positive, 200.0 / 3.0)))
    low_hi = max(1, int(np.floor(q1)))
    mid_hi = max(low_hi + 1, int(np.floor(q2)))
    return {'zero': [0, 0], 'low': [1, low_hi], 'mid': [low_hi + 1, mid_hi],
            'high': [mid_hi + 1, None], 'q1': q1, 'q2': q2}


def assign_group(coverage, cuts):
    if coverage == 0:
        return ZERO_GROUP
    for name in ('low', 'mid', 'high'):
        lo, hi = cuts[name]
        if coverage >= lo and (hi is None or coverage <= hi):
            return name
    return 'high'


def _median(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    if not values:
        return None
    return float(np.median(values))


def _share_true(values, threshold=1.0):
    values = [v for v in values if v is not None and np.isfinite(v)]
    if not values:
        return None
    return float(np.mean([1.0 if v >= threshold else 0.0 for v in values]))


def summarise(rows):
    if not rows:
        return {'n_objects': 0}
    cov = [r['coverage'] for r in rows]
    scenes = collections.Counter(r['scene'] for r in rows)
    top_scene, top_n = scenes.most_common(1)[0]
    return {
        'n_objects': len(rows),
        'n_frames': len({r['image_idx'] for r in rows}),
        'coverage_median': _median(cov),
        'coverage_mean': float(np.mean(cov)),
        'range_m_median': _median([r['range_m'] for r in rows]),
        'volume_m3_median': _median([r['volume_m3'] for r in rows]),
        'bbox_h_px_median': _median([r['bbox_h_px'] for r in rows]),
        'occluded_share': _share_true([r['occluded'] for r in rows], 1.0),
        'truncated_share': _share_true([r['truncated'] for r in rows], 0.5),
        'difficulty_median': _median([r['difficulty'] for r in rows]),
        'n_scenes': len(scenes),
        'top_scene': int(top_scene),
        'top_scene_share': float(top_n / len(rows)),
    }


def analyse(train_rows, val_rows, classes):
    cuts = derive_cuts(train_rows)
    out = {
        'probe': 'candidate C coverage grouping contract (read-only)',
        'coverage_field': 'annos.num_points_in_gt',
        'coverage_source': ('tools/data_converter/kitti_converter_TJ4Dradar.py::'
                            '_calculate_num_points_in_gt'),
        'cuts': cuts,
        'cuts_derived_from': 'train',
        'groups': {},
        'zero_share_by_class': {},
        'camera_feasibility_in_zero_group': {},
    }
    rows_by_split = {'train': train_rows, 'val': val_rows}
    for split, rows in rows_by_split.items():
        assign = [dict(r, group=assign_group(r['coverage'], cuts))
                  for r in rows]
        out['groups'][split] = {}
        for group in (ZERO_GROUP, 'low', 'mid', 'high'):
            out['groups'][split][group] = {
                'pooled': summarise([r for r in assign if r['group'] == group]),
                'per_class': {
                    cls: summarise([r for r in assign
                                    if r['group'] == group and r['class'] == cls])
                    for cls in classes},
            }
        out['zero_share_by_class'][split] = {
            cls: {
                'n_objects': len([r for r in assign if r['class'] == cls]),
                'zero_share': (
                    float(len([r for r in assign
                               if r['class'] == cls
                               and r['group'] == ZERO_GROUP]))
                    / max(1, len([r for r in assign if r['class'] == cls]))),
            } for cls in classes}
    # "Is the camera still informative for the objects radar missed?" -- the
    # precondition of the whole C hypothesis. If the zero group is ALSO the
    # tiny-bbox group, then no camera correction could be expected either, and
    # the C arm would be untestable from the start.
    for split, rows in rows_by_split.items():
        zeros = [r for r in rows if r['coverage'] == 0]
        pos = [r for r in rows if r['coverage'] > 0]
        out['camera_feasibility_in_zero_group'][split] = {
            'zero_bbox_h_px_median': _median([r['bbox_h_px'] for r in zeros]),
            'positive_bbox_h_px_median': _median([r['bbox_h_px'] for r in pos]),
            # KITTI's easy bar is 25 px: objects below it are not detectable in
            # the image either, so they cannot carry the C argument.
            'zero_bbox_h_px_ge_25_share': _share_true(
                [r['bbox_h_px'] for r in zeros], 25.0),
            'positive_bbox_h_px_ge_25_share': _share_true(
                [r['bbox_h_px'] for r in pos], 25.0),
        }
    return out


def _fmt(value, digits=2):
    if value is None:
        return 'n/a'
    if isinstance(value, float):
        return f'{value:.{digits}f}'
    return str(value)


def print_summary(result, stream):
    cuts = result['cuts']
    print(f'cuts (TRAIN-derived): zero=0 | low=1-{cuts["low"][1]} | '
          f'mid={cuts["mid"][0]}-{cuts["mid"][1]} | '
          f'high>={cuts["high"][0]}  (q1={_fmt(cuts["q1"])}, '
          f'q2={_fmt(cuts["q2"])})', file=stream)
    for split in ('train', 'val'):
        print(f'--- {split}', file=stream)
        print(f'{"group":6s} {"n_obj":>7s} {"n_fr":>6s} {"med_cov":>8s} '
              f'{"med_range":>9s} {"med_bbox_h":>10s} {"occ%":>6s} '
              f'{"top_scene%":>10s}', file=stream)
        for group in (ZERO_GROUP, 'low', 'mid', 'high'):
            s = result['groups'][split][group]['pooled']
            if not s.get('n_objects'):
                print(f'{group:6s} {0:>7d}', file=stream)
                continue
            print(f'{group:6s} {s["n_objects"]:>7d} {s["n_frames"]:>6d} '
                  f'{_fmt(s["coverage_median"], 1):>8s} '
                  f'{_fmt(s["range_m_median"]):>9s} '
                  f'{_fmt(s["bbox_h_px_median"]):>10s} '
                  f'{_fmt(100 * s["occluded_share"]) if s["occluded_share"] is not None else "n/a":>6s} '
                  f'{_fmt(100 * s["top_scene_share"], 1):>10s}', file=stream)
        zero = result['zero_share_by_class'][split]
        print('  zero-coverage share by class: ' + ', '.join(
            f'{cls} {_fmt(100 * zero[cls]["zero_share"], 1)}% '
            f'({int(round(zero[cls]["zero_share"] * zero[cls]["n_objects"]))}'
            f'/{zero[cls]["n_objects"]})' for cls in sorted(zero)), file=stream)
        feas = result['camera_feasibility_in_zero_group'][split]
        print(f'  camera precondition: zero-group bbox_h median '
              f'{_fmt(feas["zero_bbox_h_px_median"])} px vs positive '
              f'{_fmt(feas["positive_bbox_h_px_median"])} px | '
              f'share >=25 px: zero '
              f'{_fmt(100 * feas["zero_bbox_h_px_ge_25_share"], 1)}% vs '
              f'positive '
              f'{_fmt(100 * feas["positive_bbox_h_px_ge_25_share"], 1)}%',
              file=stream)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train-info', default='data/TJ4D/TJ4D_infos_train.pkl')
    ap.add_argument('--val-info', default='data/TJ4D/TJ4D_infos_val.pkl')
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    train_infos, val_infos = load_infos(args.train_info), load_infos(args.val_info)
    runs = scene_runs([i['image']['image_idx']
                       for i in train_infos + val_infos])
    train_rows = object_rows(train_infos, 'train', runs)
    val_rows = object_rows(val_infos, 'val', runs)
    classes = sorted({r['class'] for r in train_rows + val_rows})
    result = analyse(train_rows, val_rows, classes)
    result['inputs'] = {
        'train_info': os.path.abspath(args.train_info),
        'val_info': os.path.abspath(args.val_info),
        'n_train_frames': len(train_infos), 'n_val_frames': len(val_infos),
        'n_train_objects': len(train_rows), 'n_val_objects': len(val_rows),
        'n_scenes': len({r['scene'] for r in train_rows + val_rows}),
        'classes': classes,
    }
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print_summary(result, sys.stdout)
    print(f'[json] {os.path.abspath(args.output)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
