#!/usr/bin/env python3
"""Candidate-C step 3: CONTROLLED radar masking -- the causal half of step 1.

Inference only: one forward pass over the frozen val split per mode, no
training, no parameter update, no code change to the model or the config.

Why this exists
---------------
§78 measured that naturally radar-missing objects are detected far less often
(Delta recall +34.78 points). That is a CORRELATION: the objects with no returns
could simply be the intrinsically hard ones (farther, smaller, more occluded --
§76.3/§78.6 show the composition really does differ). The causal question the
user's step-1 contract asks is: *if we remove the returns from an object that
currently has them, does the detector lose it?* That is an intervention, so it
cannot be read off the natural data -- it has to be applied.

Pre-registered intervention (frozen before this file's first run)
----------------------------------------------------------------
* Target population: every GT of the val split whose FROZEN coverage group
  (§76 cuts) is not ``zero`` -- i.e. exactly the objects that have radar
  evidence today. No subsetting, no tuning, no per-class selection.
* Mask = drop every loaded point that falls inside one of those GT boxes, using
  the SAME box construction and the SAME ``points_in_rbbox`` convention that
  reproduced ``annos['num_points_in_gt']`` with zero mismatches (§77.3/§78.2).
  Points inside zero-group boxes, inside DontCare boxes, or outside every box
  are untouched.
* No margin, no jitter, no per-frame randomness. The mask is applied to the raw
  loaded cloud, before the (deterministic) test-time BDA pipeline, so boxes and
  points stay consistent downstream.
* After masking, a covered object's per-object point count in the tensor is
  ``0`` -- the same network-visible state that 95.6% of naturally missing
  objects are already in (§77.4). That equivalence is the point of the design.

Modes
-----
``identity``      installs the exact same wrapper with masking disabled. The
                  identity run is the control that proves the harness is
                  transparent: its dump must reproduce the baseline
                  ``c_failpred_ep14_dump.pkl`` bit-for-bit (checked with
                  ``--compare-against``).
``mask_covered``  the intervention.

It does NOT do: any scoring (that stays in ``probe_c_failure_enrichment.py``, so
every arm is scored by one harness), any training, any hyper-parameter choice,
or any change to model/config files.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import random
import sys
import time

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import mmcv  # noqa: E402
import torch  # noqa: E402
from mmcv import Config  # noqa: E402
from mmcv.parallel import MMDataParallel  # noqa: E402
from mmcv.runner import load_checkpoint  # noqa: E402
from mmdet3d.apis import single_gpu_test  # noqa: E402
from mmdet3d.core.bbox import box_np_ops  # noqa: E402
from mmdet3d.datasets import build_dataloader, build_dataset  # noqa: E402
from mmdet3d.models import build_model  # noqa: E402

try:  # mmdet > 2.23.0 moved these
    from mmdet.utils import compat_cfg, setup_multi_processes  # noqa: E402
except ImportError:  # pragma: no cover
    from mmdet3d.utils import compat_cfg, setup_multi_processes  # noqa: E402


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_REPO, rel))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


enrich = _load('probe_c_failure_enrichment_for_masking',
               'me_rssm/sanity/probe_c_failure_enrichment.py')
coverage = enrich.coverage
motion = enrich.motion


def coverage_plan(val_infos, coverage_json, motion_json):
    """{image_idx: {'boxes', 'groups', 'masked'}} from the FROZEN labels.

    ``boxes`` is in the raw lidar frame (the converter's own convention), and
    ``groups[k]`` is the frozen coverage group of the k-th scored object. The
    label computation is the enrichment probe's, so both tools cannot drift.
    """
    labels = enrich.build_frozen_labels(val_infos, coverage_json, motion_json)
    by_object = labels['by_object']
    plan, orders = {}, {}
    for info in val_infos:
        idx = int(info['image']['image_idx'])
        geo = motion.frame_boxes(info)
        n_obj = int(geo['no_obj'])
        names = list(geo['names'][:n_obj])
        if any(n == coverage.DONTCARE for n in names):
            raise AssertionError(f'frame {idx}: DontCare inside the sliced range')
        groups = []
        for k in range(n_obj):
            row = by_object.get((idx, k))
            if row is None:
                raise AssertionError(f'frame {idx}: missing frozen label for {k}')
            groups.append(row['coverage_group'])
        masked = [g != coverage.ZERO_GROUP for g in groups]
        plan[idx] = {
            'boxes': np.asarray(geo['boxes'], dtype=np.float64)[:n_obj],
            'groups': groups,
            'names': names,
            'masked': masked,
        }
        orders[idx] = n_obj
    return plan


def mask_rows(points_xyz, boxes, select):
    """Boolean keep-mask: True = keep. Pure function, unit-tested.

    A point is dropped when it lies inside ANY selected box, using
    ``points_in_rbbox`` -- the same call, on the same box convention, that the
    converter used to produce ``num_points_in_gt``.
    """
    n = int(points_xyz.shape[0])
    if n == 0 or not np.any(select):
        return np.ones(n, dtype=bool)
    inside = box_np_ops.points_in_rbbox(points_xyz, boxes[select])
    return ~inside.any(axis=1)


class MaskingLoader(object):
    """Wraps the pipeline's LoadPointsFromFile and applies the frozen mask."""

    def __init__(self, inner, plan, mode):
        self.inner = inner
        self.plan = plan
        self.mode = mode
        self.n_masked_frames = 0

    def __call__(self, results):
        results = self.inner(results)
        if self.mode != 'mask_covered':
            return results
        idx = int(results['sample_idx'])
        entry = self.plan.get(idx)
        if entry is None:
            raise KeyError(f'no frozen plan for frame {idx}')
        points = results['points']
        arr = np.asarray(points.tensor)
        keep = mask_rows(arr[:, :3], entry['boxes'], entry['masked'])
        if not keep.all():
            results['points'] = points.__class__(
                arr[keep], points_dim=arr.shape[-1],
                attribute_dims=points.attribute_dims)
            self.n_masked_frames += 1
        return results


def install_wrapper(dataset, plan, mode):
    """Replace the pipeline's point loader in place; return the wrapper."""
    pipeline = getattr(dataset, 'pipeline', None)
    if pipeline is None or not hasattr(pipeline, 'transforms'):
        raise TypeError('dataset has no Compose pipeline to patch')
    for i, transform in enumerate(pipeline.transforms):
        if transform.__class__.__name__ == 'LoadPointsFromFile':
            wrapper = MaskingLoader(transform, plan, mode)
            pipeline.transforms[i] = wrapper
            return wrapper
    raise TypeError('LoadPointsFromFile not found in the val pipeline')


def removal_stats(val_infos, plan, data_root):
    """CPU-side count of what the intervention removes (no model needed)."""
    per_frame, per_class = [], {}
    for info in val_infos:
        idx = int(info['image']['image_idx'])
        entry = plan[idx]
        path = os.path.join(data_root, info['point_cloud']['velodyne_path'])
        points = motion.load_points(path)
        keep = mask_rows(points[:, :3], entry['boxes'], entry['masked'])
        n_before, n_after = int(points.shape[0]), int(keep.sum())
        per_frame.append({'image_idx': idx, 'n_points_before': n_before,
                          'n_points_after': n_after,
                          'n_removed_points': n_before - n_after,
                          'n_boxes_masked': int(np.sum(entry['masked'])),
                          'n_objects': len(entry['groups'])})
        for name, sel in zip(entry['names'], entry['masked']):
            if not sel:
                continue
            rec = per_class.setdefault(name, {'n_objects_masked': 0})
            rec['n_objects_masked'] += 1
    return {
        'n_frames': len(per_frame),
        'n_frames_with_removed_points': int(sum(
            1 for f in per_frame if f['n_removed_points'] > 0)),
        'n_points_before': int(sum(f['n_points_before'] for f in per_frame)),
        'n_points_after': int(sum(f['n_points_after'] for f in per_frame)),
        'n_points_removed': int(sum(f['n_removed_points'] for f in per_frame)),
        'objects_masked_by_class': per_class,
        'per_frame': per_frame,
    }


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def compare_dumps(left_path, right_path, limit=None):
    """Bitwise comparison of two prediction dumps (frame by frame).

    ``limit`` truncates the comparison to the first N frames and, unlike the
    default, tolerates a LEFT dump that is a strict prefix of the right one --
    that is what makes a short smoke run comparable against the full baseline.
    Without ``limit`` the two dumps must have the same length.
    """
    left, right = mmcv.load(left_path), mmcv.load(right_path)
    if limit is None:
        if len(left) != len(right):
            raise AssertionError(f'length mismatch: {len(left)} vs {len(right)}')
        n = len(left)
    else:
        if len(left) > len(right):
            raise AssertionError(
                f'limit comparison needs left <= right: {len(left)} vs '
                f'{len(right)}')
        n = min(int(limit), len(left), len(right))
    report = {'n_frames': n, 'frames_differs': [], 'max_abs_diff': 0.0,
              'n_boxes_left': 0, 'n_boxes_right': 0}
    for i in range(n):
        a, b = left[i]['pts_bbox'], right[i]['pts_bbox']
        boxes_a = np.asarray(a['boxes_3d'].tensor)
        boxes_b = np.asarray(b['boxes_3d'].tensor)
        sc_a = np.asarray(a['scores_3d'])
        sc_b = np.asarray(b['scores_3d'])
        la = np.asarray(a['labels_3d'])
        lb = np.asarray(b['labels_3d'])
        report['n_boxes_left'] += int(boxes_a.shape[0])
        report['n_boxes_right'] += int(boxes_b.shape[0])
        same = (boxes_a.shape == boxes_b.shape and sc_a.shape == sc_b.shape
                and la.shape == lb.shape)
        diff = 0.0
        if same and boxes_a.size:
            diff = float(np.max(np.abs(boxes_a - boxes_b)))
            diff = max(diff, float(np.max(np.abs(sc_a - sc_b))))
        if not same or diff != 0.0 or not np.array_equal(la, lb):
            report['frames_differs'].append(i)
        report['max_abs_diff'] = max(report['max_abs_diff'], diff)
    report['identical'] = not report['frames_differs']
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--coverage-groups', required=True)
    ap.add_argument('--motion-axis', required=True)
    ap.add_argument('--mode', choices=['identity', 'mask_covered'],
                    required=True)
    ap.add_argument('--out', required=True, help='prediction dump (.pkl)')
    ap.add_argument('--stats-out', default=None,
                    help='JSON with the removal counts (mask mode)')
    ap.add_argument('--compare-against', default=None,
                    help='another dump: report a bitwise identity comparison')
    ap.add_argument('--compare-out', default=None,
                    help='write the comparison report JSON here')
    ap.add_argument('--require-identical', action='store_true',
                    help='exit non-zero if the comparison is not identical')
    ap.add_argument('--limit', type=int, default=None,
                    help='only the first N val frames (smoke runs)')
    ap.add_argument('--gpu-id', type=int, default=0)
    args = ap.parse_args()

    seed = 0  # identical to tools/test_vod.py
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    cfg = Config.fromfile(args.config)
    cfg = compat_cfg(cfg)
    setup_multi_processes(cfg)
    cfg.model.pretrained = None
    cfg.gpu_ids = [args.gpu_id]

    coverage_json = json.load(open(args.coverage_groups, encoding='utf-8'))
    motion_json = json.load(open(args.motion_axis, encoding='utf-8'))
    val_infos = coverage.load_infos(cfg.data.test.ann_file)
    plan = coverage_plan(val_infos, coverage_json, motion_json)

    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)
    if len(dataset) != len(val_infos):
        raise AssertionError(f'dataset/infos length mismatch: '
                             f'{len(dataset)} vs {len(val_infos)}')
    for i, info in enumerate(val_infos):
        if int(dataset.data_infos[i]['image']['image_idx']) != int(
                info['image']['image_idx']):
            raise AssertionError(f'frame order mismatch at {i}')

    if args.limit is not None:
        dataset.data_infos = dataset.data_infos[:args.limit]

    wrapper = install_wrapper(dataset, plan, args.mode)
    print(f'[mask] mode={args.mode} frames={len(dataset)} '
          f'objects_masked={sum(int(np.sum(v["masked"])) for v in plan.values())}')

    if args.stats_out:
        data_root = motion_json['inputs']['data_root']
        stats = removal_stats(val_infos[:args.limit], plan, data_root)
        stats['mode'] = args.mode
        stats['coverage_cuts'] = coverage_json['cuts']
        stats['rule'] = ('mask every point inside a GT box whose FROZEN coverage '
                         'group is not "zero"; no margin, no tuning')
        with open(args.stats_out, 'w', encoding='utf-8') as handle:
            json.dump(stats, handle, indent=2, sort_keys=True)
        print(f'[mask] removed {stats["n_points_removed"]} of '
              f'{stats["n_points_before"]} points in '
              f'{stats["n_frames_with_removed_points"]}/{stats["n_frames"]} frames')

    test_loader_cfg = dict(samples_per_gpu=1, workers_per_gpu=2, dist=False,
                           shuffle=False, seed=seed)
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.CLASSES = dataset.CLASSES
    model = MMDataParallel(model, device_ids=cfg.gpu_ids)

    started = time.time()
    outputs = single_gpu_test(model, data_loader)
    print(f'[mask] inference done in {time.time() - started:.1f}s '
          f'({len(outputs)} frames, wrapper touched '
          f'{wrapper.n_masked_frames} frames in workers)')
    mmcv.dump(outputs, args.out)
    print(f'[mask] dump -> {args.out}')

    if args.compare_against:
        report = compare_dumps(args.out, args.compare_against,
                               limit=args.limit)
        report['sha256_left'] = _sha256(args.out)
        report['sha256_right'] = _sha256(args.compare_against)
        if args.compare_out:
            with open(args.compare_out, 'w', encoding='utf-8') as handle:
                json.dump(report, handle, indent=2, sort_keys=True)
        print('[identity] ' + json.dumps(
            {k: v for k, v in report.items() if k != 'frames_differs'},
            sort_keys=True))
        if args.require_identical and not report['identical']:
            raise SystemExit(
                f'NOT IDENTICAL: {len(report["frames_differs"])} frames differ '
                f'({report["frames_differs"][:10]} ...)')
        if report['frames_differs']:
            print('[identity] first differing frames: '
                  f"{report['frames_differs'][:10]}")
        if not report['identical']:
            raise SystemExit('[identity] FAIL: dumps are not bit-identical')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
