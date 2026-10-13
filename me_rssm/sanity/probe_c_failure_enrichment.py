#!/usr/bin/env python3
"""Candidate-C step 2: is detection failure ENRICHED in the missing-radar groups?

Read-only analysis of an already-dumped prediction file. The GPU half is
``tools/test_vod.py --out <preds.pkl>`` (single forward pass, no training).

Note on "CPU": every statistic here is computed on the CPU, but the overlap
kernel is not -- ``kitti_eval``'s own ``d3_box_overlap`` calls the pre-built CUDA
rotate-IoU extension, and this probe deliberately uses that same code path
instead of a CPU re-implementation. The GPU use is a few seconds of IoU on a
frozen prediction dump: no model, no forward pass, no training.

The scoring contract was frozen BEFORE any run, in
``docs/rssm_c_failure_enrichment_contract_20261012.md``. Everything below is an
implementation of that document, not a set of choices made while looking at the
answers:

  * TP/FN attribution uses the repository's OWN matcher -- ``clean_data`` for the
    ignore bookkeeping and ``calculate_iou_partly`` (metric=2, 3D IoU) for the
    overlaps, both exactly as ``kitti_eval`` calls them -- and every per-frame
    (tp, fp, fn) triple this probe derives is checked against
    ``compute_statistics_jit``'s own counts. A mismatch is a hard failure, so the
    per-GT ``found`` flags can be trusted to mean what the metric means.
  * per-class overlap thresholds are the ones the headline metric uses:
    Pedestrian/Cyclist 0.25 and Car/Truck 0.5 (``lap2``/``lap1`` in
    ``kitti_eval``), NOT a uniform 0.5.
  * the score operating point is 0.25; 0.10 and 0.50 are reported as a
    sensitivity sweep and never decide the verdict.
  * FPs that match no GT go into their OWN bucket. They are never attributed to
    a GT's missingness group.
  * natural missingness only. Controlled radar masking is a different experiment
    and must not be merged into these tables.
  * the ~2.2 point offline-vs-EvalHook difference (§51.5) is a CALIBER difference:
    it is recorded for the reader and never added to or subtracted from anything.

Group labels come from the two frozen artifacts (§76 coverage, §77 motion) by
re-running those probes' own code on val and asserting the resulting per-cell
object counts still equal the frozen JSON. If a boundary ever drifts, this
script fails instead of quietly analysing a different partition.

Usage:
  python me_rssm/sanity/probe_c_failure_enrichment.py \
    --preds <run>/c_failpred_ep14_dump.pkl \
    --config <run>/<cfg snapshot>.py \
    --coverage-groups <run>/c_coverage_groups_20261012.json \
    --motion-axis <run>/c_motion_axis_20261012.json \
    --output <run>/c_failure_enrichment_ep14.json
"""

import argparse
import collections
import copy
import hashlib
import importlib.util
import json
import os
import pickle
import sys

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from mmdet3d.core.evaluation.kitti_utils.eval import (  # noqa: E402
    _prepare_data, calculate_iou_partly, compute_statistics_jit)
from mmcv import Config  # noqa: E402
from mmdet3d.datasets import build_dataset  # noqa: E402

try:  # mmdet3d's compat layer, identical to tools/train_vod.py and eval_dump_c
    from mmdet3d.utils import compat_cfg  # noqa: E402
except ImportError:  # pragma: no cover
    from mmdet.utils import compat_cfg  # noqa: E402

# kitti_eval's internal class order (clean_data's CLASS_NAMES). The dataset's
# CLASSES order is different on purpose; min_overlaps is reordered to match this
# one, so these indices are what _prepare_data expects.
KITTI_CLASS_INDEX = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2, 'Truck': 3}
# Per-class 3D IoU, exactly as kitti_eval's Overall aggregation uses them.
CLASS_OVERLAP = {'Pedestrian': 0.25, 'Cyclist': 0.25, 'Car': 0.5, 'Truck': 0.5}
CLASS_ORDER = ('Pedestrian', 'Cyclist', 'Car', 'Truck')
DIFFICULTY_ALL = 2
SCORE_OPERATING_POINT = 0.25
SENSITIVITY_POINTS = (0.10, 0.50)
RANGE_BINS = ((0.0, 50.0, 'le50'), (50.0, 70.0, '50to70'),
              (70.0, float('inf'), 'gt70'))
MIN_BIN_GT = 30
MIN_DELTA_POINTS = 3.0
BOOTSTRAP_ITERS = 2000
BOOTSTRAP_SEED = 0
NO_DETECTION = -10000000


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_REPO, rel))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


coverage = _load('probe_c_coverage_for_enrichment',
                 'me_rssm/sanity/probe_c_coverage_groups.py')
motion = _load('probe_c_motion_for_enrichment',
               'me_rssm/sanity/probe_c_motion_axis.py')


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_pickle(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle)


def attribute_frame(overlaps, gt_datas, dt_datas, ignored_gt, ignored_det,
                    min_overlap, thresh):
    """Per-GT TP/FN flags under compute_statistics_jit's own rule.

    Mirrors the ``compute_fp=True`` branch of
    ``mmdet3d/core/evaluation/kitti_utils/eval.py::compute_statistics_jit``
    (GT-order greedy, highest overlap wins, low-score detections thresholded).
    The caller verifies the returned counts against that function, so this is a
    re-expression, not a second opinion.
    """
    det_size, gt_size = dt_datas.shape[0], gt_datas.shape[0]
    dt_scores = dt_datas[:, -1]
    assigned = [False] * det_size
    ignored_threshold = [bool(dt_scores[j] < thresh) for j in range(det_size)]
    tp, fp, fn = 0, 0, 0
    flags = []
    for i in range(gt_size):
        if ignored_gt[i] == -1:
            flags.append({'gt': i, 'status': 'ignored_gt', 'dt': None})
            continue
        det_idx, valid, max_overlap, assigned_ignored = -1, NO_DETECTION, 0.0, False
        for j in range(det_size):
            if ignored_det[j] == -1 or assigned[j] or ignored_threshold[j]:
                continue
            overlap, score = overlaps[j, i], dt_scores[j]
            if (overlap > min_overlap and (overlap > max_overlap
                                           or assigned_ignored)
                    and ignored_det[j] == 0):
                max_overlap, det_idx, valid = overlap, j, 1
                assigned_ignored = False
            elif (overlap > min_overlap and valid == NO_DETECTION
                  and ignored_det[j] == 1):
                det_idx, valid, assigned_ignored = j, 1, True
        if valid == NO_DETECTION and ignored_gt[i] == 0:
            fn += 1
            flags.append({'gt': i, 'status': 'fn', 'dt': None})
        elif valid != NO_DETECTION and (ignored_gt[i] == 1
                                        or ignored_det[det_idx] == 1):
            assigned[det_idx] = True
            flags.append({'gt': i, 'status': 'ignored_match', 'dt': int(det_idx)})
        elif valid != NO_DETECTION:
            tp += 1
            assigned[det_idx] = True
            flags.append({'gt': i, 'status': 'tp', 'dt': int(det_idx),
                          'score': float(dt_scores[det_idx]),
                          'overlap': float(overlaps[det_idx, i])})
        else:
            flags.append({'gt': i, 'status': 'ignored_gt', 'dt': None})
    for j in range(det_size):
        if not (assigned[j] or ignored_det[j] == -1 or ignored_det[j] == 1
                or ignored_threshold[j]):
            fp += 1
    return {'tp': tp, 'fp': fp, 'fn': fn, 'flags': flags}


def match_class(gt_annos, dt_annos, overlaps, class_name, thresh):
    """Per-frame attribution for one class; hard-verifies against the repo."""
    kitti_idx = KITTI_CLASS_INDEX[class_name]
    min_overlap = CLASS_OVERLAP[class_name]
    rets = _prepare_data(gt_annos, dt_annos, kitti_idx, DIFFICULTY_ALL)
    (gt_datas_list, dt_datas_list, ignored_gts, ignored_dets, dontcares,
     _, _) = rets
    frames = []
    for i in range(len(gt_annos)):
        got = attribute_frame(overlaps[i], gt_datas_list[i], dt_datas_list[i],
                              ignored_gts[i], ignored_dets[i], min_overlap, thresh)
        ref = compute_statistics_jit(
            overlaps[i], gt_datas_list[i], dt_datas_list[i], ignored_gts[i],
            ignored_dets[i], dontcares[i], 2, min_overlap=min_overlap,
            thresh=thresh, compute_fp=True)
        if (got['tp'], got['fp'], got['fn']) != (int(ref[0]), int(ref[1]),
                                                 int(ref[2])):
            raise AssertionError(
                f'per-GT attribution disagrees with compute_statistics_jit on '
                f'frame {i}/{class_name}: ours='
                f"({got['tp']},{got['fp']},{got['fn']}) theirs="
                f'({int(ref[0])},{int(ref[1])},{int(ref[2])})')
        frames.append(got)
    return frames


def scene_of(image_idx, runs):
    return int(runs.get(int(image_idx), -1))


def gt_range_m(anno, k):
    loc = np.asarray(anno['location'], dtype=np.float64)[k]
    return float(np.sqrt(loc[0] ** 2 + loc[1] ** 2 + loc[2] ** 2))


def range_bin(rng):
    for lo, hi, name in RANGE_BINS:
        if lo < rng <= hi:
            return name
    return RANGE_BINS[-1][2]


def recall_of(records):
    n = len(records)
    if n == 0:
        return {'n_gt': 0, 'n_found': 0, 'recall': None}
    found = sum(1 for r in records if r['found'])
    return {'n_gt': n, 'n_found': found, 'recall': found / n}


def delta_stats(records, group_key, reference=coverage.ZERO_GROUP):
    """Reference-group vs everybody-else recall gap, in points.

    ``reference`` is the group that plays the role of "the missing-radar side".
    For the primary coverage axis that is the frozen ``zero`` group; the frozen
    amplitude axis has its own reference name (``no_return``), and comparing
    against the coverage group there would silently compare against an EMPTY set
    -- the first draft of this table did exactly that and reported a
    meaningless "zero n=0 vs everything n=8917" contrast.
    """
    zero = [r for r in records if r[group_key] == reference]
    pos = [r for r in records if r[group_key] != reference]
    z, p = recall_of(zero), recall_of(pos)
    delta = (None if z['recall'] is None or p['recall'] is None
             else 100.0 * (p['recall'] - z['recall']))
    return {'zero': z, 'covered': p, 'delta_points_covered_minus_zero': delta}


def cluster_bootstrap_delta(records, scene_key='scene', iters=BOOTSTRAP_ITERS,
                            seed=BOOTSTRAP_SEED):
    """Scene-clustered bootstrap CI for delta (covered - zero) recall."""
    scenes = sorted({r[scene_key] for r in records})
    by_scene = {s: [r for r in records if r[scene_key] == s] for s in scenes}
    rng = np.random.RandomState(seed)
    deltas = []
    for _ in range(iters):
        picked = rng.randint(0, len(scenes), len(scenes))
        sample = []
        for k in picked:
            sample.extend(by_scene[scenes[k]])
        d = delta_stats(sample, 'coverage_group')['delta_points_covered_minus_zero']
        if d is not None:
            deltas.append(d)
    if not deltas:
        return {'n_resamples': 0, 'ci95': [None, None]}
    return {'n_resamples': len(deltas),
            'ci95': [float(np.percentile(deltas, 2.5)),
                     float(np.percentile(deltas, 97.5))],
            'median': float(np.median(deltas)),
            'cluster': 'scene (image_idx contiguity runs)'}


def fp_summary(per_class_frames, dt_annos):
    """FP bucket: its own population, never assigned to a GT group."""
    total = 0
    by_class = {}
    scores = []
    for class_name, frames in per_class_frames.items():
        n = sum(f['fp'] for f in frames)
        by_class[class_name] = n
        total += n
    for anno in dt_annos:
        for s in np.asarray(anno['score'], dtype=np.float64):
            scores.append(float(s))
    return {
        'n_fp_unmatched': total,
        'by_class': by_class,
        'note': ('unmatched detections are reported as their own bucket; they '
                 'are never attributed to a GT missingness group'),
        'all_dt_scores': {'n': len(scores),
                          'median': float(np.median(scores)) if scores else None},
    }


def build_records(gt_annos, per_class_frames, labels, split='val'):
    """Attach every GT's found/not-found flag to its frozen group labels."""
    records = []
    for i, anno in enumerate(gt_annos):
        image_idx = labels['image_idx_of_frame'][i]
        names = list(anno['name'])
        for cls in CLASS_ORDER:
            frames = per_class_frames[cls][i]
            for flag in frames['flags']:
                if flag['status'] not in ('tp', 'fn'):
                    continue
                k = flag['gt']
                if names[k] != cls:
                    # clean_data appends exactly one ignored_gt entry per GT in
                    # the frame's own order (and _prepare_data keeps the full,
                    # unfiltered gt array), so a tp/fn flag can only point at a
                    # GT of this class. Anything else means the flag index and
                    # the GT list have drifted apart, which would silently attach
                    # coverage groups to the wrong objects.
                    raise AssertionError(
                        f'GT index {k} of frame {image_idx} is {names[k]!r}, '
                        f'not {cls!r}: flag/GT misalignment')
                key = (int(image_idx), int(k))
                lab = labels['by_object'].get(key)
                if lab is None:
                    raise AssertionError(f'no frozen group label for {key}')
                records.append({
                    'image_idx': int(image_idx),
                    'scene': scene_of(image_idx, labels['scene_runs']),
                    'class': cls,
                    'gt_index': int(k),
                    'found': flag['status'] == 'tp',
                    'coverage_group': lab['coverage_group'],
                    'zero_scope': lab['zero_scope'],
                    'amp_group': lab['amp_group'],
                    'range_m': gt_range_m(anno, k),
                    'range_bin': range_bin(gt_range_m(anno, k)),
                })
    return records


def by_class_table(records, group_key='coverage_group'):
    return {cls: delta_stats([r for r in records if r['class'] == cls], group_key)
            for cls in CLASS_ORDER}


def by_bin_table(records, group_key='coverage_group'):
    return {name: delta_stats([r for r in records if r['range_bin'] == name],
                              group_key)
            for _, _, name in RANGE_BINS}


def per_group_recall(records, key):
    groups = collections.defaultdict(list)
    for r in records:
        groups[r[key]].append(r)
    # ``zero_scope`` is only defined for the zero-coverage group, so a None key
    # is normal there; name it instead of letting the JSON encoder choke on a
    # mixed str/None key set.
    return {('not_applicable (coverage > 0)' if g is None else g):
            recall_of(v)
            for g, v in sorted(groups.items(), key=lambda kv: str(kv[0]))}


def verdict_for(records, bins):
    """Mechanical application of the frozen rule (no judgement calls)."""
    pooled = delta_stats(records, 'coverage_group')
    delta = pooled['delta_points_covered_minus_zero']
    qualifying = []
    for name, table in bins.items():
        z = table['zero']
        d = table['delta_points_covered_minus_zero']
        if z['n_gt'] >= MIN_BIN_GT and d is not None and d >= MIN_DELTA_POINTS:
            qualifying.append(name)
    ci = cluster_bootstrap_delta(records)['ci95']
    ci_excludes_zero = bool(ci[0] is not None and ci[0] > 0)
    enriched = bool(delta is not None and delta >= MIN_DELTA_POINTS
                    and len(qualifying) >= 2 and ci_excludes_zero)
    return {
        'delta_points': delta,
        'delta_threshold': MIN_DELTA_POINTS,
        'range_bins_qualifying': qualifying,
        'range_bins_required': 2,
        'bootstrap_ci95': ci,
        'bootstrap_ci_excludes_zero': ci_excludes_zero,
        'enriched': enriched,
        'rule': ('delta >= 3.0 points AND >= 2 range bins (zero-group n >= 30) '
                 'reach 3.0 AND scene-clustered 95% CI excludes 0'),
    }


def build_frozen_labels(val_infos, coverage_json, motion_json):
    """Per-object frozen group labels, recomputed with the two probes' own code.

    Returns ``{'image_idx_of_frame', 'scene_runs', 'by_object', 'cell_counts',
    'frozen_cell_counts', 'frozen_zero_sub_counts'}``. Every frozen cell count
    is re-derived and
    compared here, so any boundary drift raises instead of silently analysing a
    different partition.

    ``by_object`` is keyed by ``(image_idx, k)`` where ``k`` is the k-th scored
    object of that frame in the probes' emission order (DontCare/env-ignored
    entries skipped). Callers on the val split may rely on that order because
    the val infos contain no DontCare objects at all (checked: 2040/2040 frames);
    the cell-count assertion above is what makes the order trustworthy in
    general.
    """
    runs = coverage.scene_runs([i['image']['image_idx'] for i in val_infos])
    cov_cuts = coverage_json['cuts']
    amp_cuts = motion_json['amplitude_cuts']
    if motion_json.get('coverage_cuts') != cov_cuts:
        raise AssertionError(
            'the two frozen artifacts disagree about the coverage cuts: '
            f"coverage-groups {cov_cuts} vs motion-axis "
            f"{motion_json.get('coverage_cuts')} (stale motion artifact?)")
    label_rows = []
    for info in val_infos:
        idx = int(info['image']['image_idx'])
        path = os.path.join(motion_json['inputs']['data_root'],
                            info['point_cloud']['velodyne_path'])
        points = motion.load_points(path)
        kept = motion.filter_outside(info, points)
        rows, _, _ = motion.object_motion_rows(
            info, points, kept, cov_cuts, idx, runs.get(idx, -1))
        for r in rows:
            r['amp_group'] = motion.assign_amplitude(r, amp_cuts)
            label_rows.append(r)

    # Reproduce the frozen cells KEY BY KEY, including the artifact's one
    # deliberate special case: 'zero|no_return' is the summary of EVERY
    # coverage==0 object, not of the objects whose raw cloud is empty. The
    # strictly-empty subset and the frame-level/object-level split live INSIDE
    # that cell (network_visible_zero / zero_scope), so they are compared too.
    # Deriving a "cleaner" partition here would quietly stop checking the frozen
    # one, which is exactly the failure this assertion exists to prevent.
    cell_counts = collections.Counter()
    for cov_group in (coverage.ZERO_GROUP, 'low', 'mid', 'high'):
        for amp_group in motion.AMPLITUDE_GROUPS:
            cell_counts[f'{cov_group}|{amp_group}'] = sum(
                1 for r in label_rows
                if r['coverage_group'] == cov_group and r['amp_group'] == amp_group)
    cell_counts[f'{coverage.ZERO_GROUP}|{motion.NO_RETURN}'] = sum(
        1 for r in label_rows if r['coverage_group'] == coverage.ZERO_GROUP)

    frozen_counts = {
        key: cell['pooled']['n_objects']
        for key, cell in motion_json['cells']['val'].items()
        if '|' in key and cell.get('pooled', {}).get('n_objects') is not None}
    drift = {k: (frozen_counts.get(k), cell_counts.get(k))
             for k in set(frozen_counts) | set(cell_counts)
             if frozen_counts.get(k) != cell_counts.get(k)}

    zero_cell = motion_json['cells']['val'][
        f'{coverage.ZERO_GROUP}|{motion.NO_RETURN}']
    sub_frozen = {
        'network_visible_zero (n_points_raw == 0)': zero_cell[
            'network_visible_zero']['n_objects'],
        'frame_level (coverage==0 in an all-zero frame)': zero_cell[
            'zero_scope']['frame_level']['pooled']['n_objects'],
        'object_level (coverage==0 but its frame had returns)': zero_cell[
            'zero_scope']['object_level']['pooled']['n_objects'],
    }
    sub_now = {
        'network_visible_zero (n_points_raw == 0)': sum(
            1 for r in label_rows if r['n_points_raw'] == 0),
        'frame_level (coverage==0 in an all-zero frame)': sum(
            1 for r in label_rows if r['coverage'] == 0
            and r['zero_scope'] == 'frame_level'),
        'object_level (coverage==0 but its frame had returns)': sum(
            1 for r in label_rows if r['coverage'] == 0
            and r['zero_scope'] == 'object_level'),
    }
    for key, frozen_value in sub_frozen.items():
        if sub_now[key] != frozen_value:
            drift[f'zero|no_return::{key}'] = (frozen_value, sub_now[key])
    if drift:
        raise AssertionError(f'frozen group drift (frozen, recomputed): {drift}')

    by_object, order = {}, collections.Counter()
    for r in label_rows:
        by_object[(int(r['image_idx']), order[int(r['image_idx'])])] = r
        order[int(r['image_idx'])] += 1
    return {
        'image_idx_of_frame': [int(i['image']['image_idx']) for i in val_infos],
        'scene_runs': runs,
        'by_object': by_object,
        'cell_counts': dict(cell_counts),
        'frozen_cell_counts': frozen_counts,
        'frozen_zero_sub_counts': sub_frozen,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--preds', required=True, help='outputs pkl from test_vod.py')
    ap.add_argument('--config', required=True, help='config snapshot of that arm')
    ap.add_argument('--coverage-groups', required=True)
    ap.add_argument('--motion-axis', required=True)
    ap.add_argument('--harness-ap', default=None,
                    help='json written by test_vod.py --saveoutput (caliber note)')
    ap.add_argument('--output', required=True)
    ap.add_argument('--records-out', default=None,
                    help='optional: per-GT records (image_idx, gt_index, class, '
                         'found, frozen group, range) as JSON, so a masked-vs-'
                         'baseline contrast can be paired on the SAME GTs by the '
                         'same harness. Purely additive: it cannot change any '
                         'number in --output.')
    args = ap.parse_args()

    cfg = Config.fromfile(args.config)
    # The run configs carry `data.val.samples_per_gpu` (the old-style dataloader
    # key).  train_vod.py / eval_dump_c.py / tool_c_masked_inference.py all run
    # compat_cfg first, which moves it to `data.val_dataloader`; without that
    # step build_dataset forwards the stray key to TJ4DDataset.__init__ and dies
    # with "unexpected keyword argument 'samples_per_gpu'" -- exactly what
    # happened to C1's staged ep14 probe on 2026-10-13.
    cfg = compat_cfg(cfg)
    coverage_json = json.load(open(args.coverage_groups, encoding='utf-8'))
    motion_json = json.load(open(args.motion_axis, encoding='utf-8'))
    outputs = load_pickle(args.preds)
    dataset = build_dataset(cfg.data.val)
    val_infos = coverage.load_infos(cfg.data.val.get(
        'ann_file', 'data/TJ4D/TJ4D_infos_val.pkl'))
    if len(outputs) != len(dataset) or len(dataset) != len(val_infos):
        raise AssertionError(f'length mismatch: outputs={len(outputs)} '
                             f'dataset={len(dataset)} infos={len(val_infos)}')
    for i, info in enumerate(val_infos):
        got = int(dataset.data_infos[i]['image']['image_idx'])
        want = int(info['image']['image_idx'])
        if got != want:
            raise AssertionError(f'frame order mismatch at {i}: {got} != {want}')

    result_files, tmp_dir = dataset.format_results(outputs, pklfile_prefix=None)
    # TJ4DDataset.format_results -> bbox2result_kitti returns the KITTI-style
    # det_annos list itself (it additionally dumps a throwaway tmp pkl); use the
    # returned object, not the tmp path, so nothing depends on the tmp dir.
    dt_annos = result_files['pts_bbox']
    if len(dt_annos) != len(val_infos):
        raise AssertionError('format_results length mismatch')

    gt_annos = [info['annos'] for info in coverage.load_infos(
        cfg.data.val.get('ann_file', 'data/TJ4D/TJ4D_infos_val.pkl'))]

    labels = build_frozen_labels(val_infos, coverage_json, motion_json)
    runs = labels['scene_runs']
    frozen_counts = labels['frozen_cell_counts']
    sub_frozen = labels['frozen_zero_sub_counts']

    # --- matching -----------------------------------------------------------
    overlaps = calculate_iou_partly(dt_annos, gt_annos, 2, 200)[0]
    per_class_frames = {}
    for cls in CLASS_ORDER:
        per_class_frames[cls] = match_class(
            copy.deepcopy(gt_annos), dt_annos, overlaps, cls,
            SCORE_OPERATING_POINT)

    records = build_records(gt_annos, per_class_frames, labels)
    bins = by_bin_table(records)
    result = {
        'probe': 'candidate C failure enrichment (read-only CPU, frozen contract)',
        'contract': {
            'document': 'docs/rssm_c_failure_enrichment_contract_20261012.md',
            'sha256': sha256_file(os.path.join(
                _REPO, 'docs/rssm_c_failure_enrichment_contract_20261012.md')),
            'score_operating_point': SCORE_OPERATING_POINT,
            'sensitivity_points': list(SENSITIVITY_POINTS),
            'per_class_overlap_3d_iou': CLASS_OVERLAP,
            'difficulty': DIFFICULTY_ALL,
            'attribute_vs_compute_statistics_jit': 'exact match enforced',
            'fp_bucket': 'own bucket, never assigned to a GT group',
            'natural_vs_controlled_masking': 'natural only (controlled = separate prereg)',
        },
        'inputs': {
            'preds': os.path.abspath(args.preds),
            'preds_sha256': sha256_file(args.preds),
            'config': os.path.abspath(args.config),
            'coverage_groups': os.path.abspath(args.coverage_groups),
            'coverage_groups_sha256': sha256_file(args.coverage_groups),
            'motion_axis': os.path.abspath(args.motion_axis),
            'motion_axis_sha256': sha256_file(args.motion_axis),
            'n_frames': len(gt_annos),
            'n_dt': int(sum(len(a['name']) for a in dt_annos)),
            'n_gt': int(sum(len(a['name']) for a in gt_annos)),
            'frozen_group_consistency': {
                'rule': ('val group labels are recomputed with the two probes\' '
                         'own code and every frozen cell count must match'),
                'cells_compared': len(frozen_counts) + len(sub_frozen),
                'cell_counts': dict(sorted(frozen_counts.items())),
                'zero_cell_special_case': (
                    "'zero|no_return' in the frozen artifact is the summary of "
                    'EVERY coverage==0 object; the strictly-empty subset is its '
                    "'network_visible_zero' sub-cell, and the frame-level / "
                    'object-level split is its zero_scope sub-cells'),
                'zero_sub_counts': sub_frozen,
                'drift': 'none (assertion passed)',
            },
        },
        'verdict': verdict_for(records, bins),
        'primary': {
            'pooled': delta_stats(records, 'coverage_group'),
            'by_class': by_class_table(records),
            'by_range_bin': bins,
            'recall_by_coverage_group': per_group_recall(records, 'coverage_group'),
            'recall_by_zero_scope': per_group_recall(records, 'zero_scope'),
            'object_level_only': delta_stats(
                [r for r in records if r['zero_scope'] != 'frame_level'
                 or r['coverage_group'] != coverage.ZERO_GROUP],
                'coverage_group'),
        },
        'secondary': {
            'by_amplitude_group': delta_stats(records, 'amp_group',
                                             reference=motion.NO_RETURN),
            'amplitude_group_recall': per_group_recall(records, 'amp_group'),
            'fp_bucket': fp_summary(per_class_frames, dt_annos),
            'sensitivity': {},
        },
        # Descriptive only: these tables were added AFTER the primary result was
        # seen, to say WHERE the effect lives. They never feed the verdict rule,
        # and no boundary was moved for them.
        'descriptive': {
            'note': ('added after the primary result was visible; used to '
                     'describe the effect, never to decide it'),
            'by_scene': {
                str(scene): delta_stats(
                    [r for r in records if r['scene'] == scene],
                    'coverage_group')
                for scene in sorted({r['scene'] for r in records})},
            'by_class_by_range_bin': {
                cls: by_bin_table([r for r in records if r['class'] == cls])
                for cls in CLASS_ORDER},
        },
        'sensitivity_score_points': {},
    }
    for point in SENSITIVITY_POINTS:
        frames = {cls: match_class(copy.deepcopy(gt_annos), dt_annos, overlaps,
                                   cls, point) for cls in CLASS_ORDER}
        recs = build_records(gt_annos, frames, labels)
        result['sensitivity_score_points'][str(point)] = {
            'pooled': delta_stats(recs, 'coverage_group'),
            'by_class': by_class_table(recs),
        }
    if args.harness_ap:
        result['harness_ap'] = json.load(open(args.harness_ap, encoding='utf-8'))
        result['harness_ap']['_caliber_note'] = (
            'offline harness (dataset.evaluate, no evaluation.pipeline). §51.5: '
            'about 2.2 3D points below the training-time EvalHook. This is a '
            'CALIBER difference, never an additive correction.')
    result['per_gt_records_summary'] = {
        'n_records': len(records),
        'n_found': int(sum(1 for r in records if r['found'])),
    }
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    if args.records_out:
        with open(args.records_out, 'w', encoding='utf-8') as handle:
            json.dump(records, handle, sort_keys=True)
        print(f'[records] {len(records)} per-GT records -> '
              f'{os.path.abspath(args.records_out)}')

    print(f"n_gt={result['inputs']['n_gt']} n_dt={result['inputs']['n_dt']}")
    print('pooled:', json.dumps(result['primary']['pooled']))
    for name, table in bins.items():
        print(f'  range {name}: zero n={table["zero"]["n_gt"]} '
              f'recall={table["zero"]["recall"]} covered n='
              f'{table["covered"]["n_gt"]} recall={table["covered"]["recall"]} '
              f'delta={table["delta_points_covered_minus_zero"]}')
    for cls, table in result['primary']['by_class'].items():
        print(f'  {cls}: zero n={table["zero"]["n_gt"]} covered n='
              f'{table["covered"]["n_gt"]} delta='
              f'{table["delta_points_covered_minus_zero"]}')
    print('verdict:', json.dumps(result['verdict'], sort_keys=True))
    print('fp bucket:', json.dumps(result['secondary']['fp_bucket']['by_class']))
    print(f'[json] {os.path.abspath(args.output)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
