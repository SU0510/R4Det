"""Audit LoadVLSAMAnnotations matching on real TJ4D samples.

This wraps the live pipeline's matching helper so the audit sees the exact
pre-augmentation GT boxes/labels and VLSAM boxes/labels used by training.

Run:
  python me_rssm/sanity/audit_vlsam_matching.py --num-samples 500
"""

import argparse
import os
import sys

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from mmcv import Config
from mmdet3d.datasets import build_dataset
from mmdet3d.datasets.pipelines.loading_custom import LoadVLSAMAnnotations


CFG = os.path.join(_REPO, 'configs/r4det/'
                   'TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head.py')


def _find_vlsam_loader(pipeline):
    for step in pipeline:
        if isinstance(step, LoadVLSAMAnnotations):
            return step
    raise RuntimeError('LoadVLSAMAnnotations not found in train pipeline')


def _unwrap_dataset(dataset):
    while hasattr(dataset, 'dataset'):
        dataset = dataset.dataset
    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-samples', type=int, default=500)
    args = parser.parse_args()

    cfg = Config.fromfile(CFG)
    ds = build_dataset(cfg.data.train)
    base_ds = _unwrap_dataset(ds)
    loader = _find_vlsam_loader(base_ds.pipeline.transforms)
    n = min(args.num_samples, len(ds))

    records = []
    orig_match = loader._match_official_to_vlsam

    def audited_match(official_bboxes, official_labels,
                      vlsam_bboxes, vlsam_labels):
        matches, ious = orig_match(official_bboxes, official_labels,
                                   vlsam_bboxes, vlsam_labels)
        records.append(dict(
            official_labels=np.asarray(official_labels).copy(),
            vlsam_labels=np.asarray(vlsam_labels).copy(),
            matches=np.asarray(matches).copy(),
            ious=np.asarray(ious).copy(),
        ))
        return matches, ious

    loader._match_official_to_vlsam = audited_match

    for idx in range(n):
        _ = ds[idx]

    total_gt = sum(len(r['matches']) for r in records)
    matched = sum(int((r['matches'] >= 0).sum()) for r in records)
    unmatched = total_gt - matched
    duplicate_assignments = 0
    cross_class = 0
    below_threshold = 0
    ious = []
    match_rates = []

    for rec in records:
        matches = rec['matches']
        valid = matches[matches >= 0]
        if len(valid) != len(set(valid.tolist())):
            duplicate_assignments += 1
        for gt_idx, pred_idx in enumerate(matches):
            if pred_idx < 0:
                continue
            if rec['official_labels'][gt_idx] != rec['vlsam_labels'][pred_idx]:
                cross_class += 1
            iou = float(rec['ious'][gt_idx])
            ious.append(iou)
            if iou < loader.iou_threshold:
                below_threshold += 1
        match_rates.append(float((matches >= 0).mean()) if len(matches) else 0.0)

    print('Audited samples:', n)
    print('Recorded matching calls:', len(records))
    print('GT targets in records:', total_gt)
    print('Matched GT:', matched)
    print('Unmatched GT:', unmatched)
    print('Match rate:', matched / total_gt if total_gt else 0.0)
    print('Mean sample match rate:', float(np.mean(match_rates)) if match_rates else 0.0)
    print('Duplicate assignments:', duplicate_assignments)
    print('Cross-class matches:', cross_class)
    print('Below-threshold matches:', below_threshold)
    print('Mean matched IoU:', float(np.mean(ious)) if ious else 0.0)
    print('Median matched IoU:', float(np.median(ious)) if ious else 0.0)
    print('Min matched IoU:', float(np.min(ious)) if ious else 0.0)
    print('Max matched IoU:', float(np.max(ious)) if ious else 0.0)

    ok = (duplicate_assignments == 0 and cross_class == 0 and below_threshold == 0)
    print('\n==== AUDIT:', 'PASS' if ok else 'FAIL', '====')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
