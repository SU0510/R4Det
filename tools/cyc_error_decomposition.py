"""Offline Cyclist error decomposition with formal KITTI 3D matching.

Reads the self-contained prediction dumps produced by tools/dump_predictions.py
and computes:

1. oracle recall for moderate Cyclist under the same 3D IoU=0.25 criterion;
2. the evaluator's 41-point precision/recall curve for the same class/difficulty;
3. score-bin TP/FP counts to separate missed geometry from ranking loss.

The dump stores boxes in LiDAR frame.  Formal KITTI matching is done in camera
frame after the same yaw normalization and calibration transform used by the
dataset evaluator.
"""
import argparse
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


# Formal TJ4D evaluation runs through tools/test_vod.py, which enables the old
# mmdet3d box-conversion branch before importing mmdet3d.  This diagnostic
# reimplements that old branch locally to avoid importing CUDA-only ops.


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_TOOL_DIR = REPO_ROOT / 'tools' / 'eval_tools'
sys.path.insert(0, str(EVAL_TOOL_DIR))

import importlib.util


spec = importlib.util.spec_from_file_location(
    'tj4d_eval', str(EVAL_TOOL_DIR / 'TJ4D-eval.py'))
tj4d_eval = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tj4d_eval)

rotate_spec = importlib.util.spec_from_file_location(
    'vod_rotate_iou_cpu',
    str(REPO_ROOT / 'mmdet3d' / 'core' / 'evaluation' / 'vod_utils' /
        'rotate_iou_cpu.py'))
vod_rotate_iou_cpu = importlib.util.module_from_spec(rotate_spec)
rotate_spec.loader.exec_module(vod_rotate_iou_cpu)


EVAL_CLASS_TO_NAME = {
    0: 'Car',
    1: 'Pedestrian',
    2: 'Cyclist',
    3: 'Truck',
}
EVAL_NAME_TO_CLASS = {v: k for k, v in EVAL_CLASS_TO_NAME.items()}
CLASSES = ('Pedestrian', 'Cyclist', 'Car', 'Truck')
NAME_TO_MODEL_CLASS = {name: i for i, name in enumerate(CLASSES)}
SCORE_BINS = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.000001]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--dump', nargs='+', required=True)
    parser.add_argument('--class-name', default='Cyclist')
    parser.add_argument('--difficulty', type=int, default=1,
                        help='0 easy, 1 moderate, 2 hard')
    parser.add_argument('--iou-thr', type=float, default=0.25)
    parser.add_argument('--out-json', default='')
    return parser.parse_args()


def load_infos(config_path):
    """Read only the val ann_file path from the config.

    Config.fromfile would import the model's custom ops, which assert CUDA
    availability.  This diagnostic is CPU-only and only needs the val info pkl
    (which always lives at ``data/TJ4D/TJ4D_infos_val.pkl`` for TJ4D).
    """
    ann_file = None
    text = Path(config_path).read_text()
    pattern = "ann_file" + r"\s*=\s*data_root\s*\+\s*" + "['\"]([^'\"]+)['\"]"
    match = re.search(pattern, text)
    if match:
        ann_file = 'data/TJ4D/' + match.group(1)
    if ann_file is None:
        fallback = REPO_ROOT / 'data' / 'TJ4D' / 'TJ4D_infos_val.pkl'
        if fallback.exists():
            ann_file = str(fallback)
    if ann_file is None:
        raise FileNotFoundError(
            'Could not locate TJ4D val ann_file from config or default path')
    if not ann_file.startswith('/'):
        ann_file = str(REPO_ROOT / ann_file)
    with open(ann_file, 'rb') as f:
        infos = pickle.load(f)
    return infos


def lidar_boxes_to_camera(boxes, info):
    """Match TJ4DDataset.convert_valid_bboxes exactly."""
    boxes = np.asarray(boxes, dtype=np.float64).copy()
    if len(boxes) == 0:
        return np.zeros((0, 7), dtype=np.float64), boxes
    boxes[:, -1] = boxes[:, -1] - np.pi
    boxes[:, -1] = boxes[:, -1] - np.floor(
        boxes[:, -1] / (np.pi * 2) + 0.5) * (np.pi * 2)

    rect = np.asarray(info['calib']['R0_rect'], dtype=np.float64)
    tr_velo_to_cam = np.asarray(info['calib']['Tr_velo_to_cam'], dtype=np.float64)
    transform = rect @ tr_velo_to_cam
    xyz_h = np.concatenate([boxes[:, :3], np.ones((len(boxes), 1))], axis=1)
    xyz = (xyz_h @ transform.T)[:, :3]
    x_size, y_size, z_size = boxes[:, 3], boxes[:, 4], boxes[:, 5]
    camera_boxes = np.stack(
        [xyz[:, 0], xyz[:, 1], xyz[:, 2],
         y_size, z_size, x_size, boxes[:, 6]],
        axis=1)
    return camera_boxes, boxes


def d3_box_overlap_cpu(boxes, qboxes, criterion=-1):
    """CPU port of TJ4D-eval.py:d3_box_overlap for camera-frame boxes.

    Camera boxes are [x, y, z, x_size, y_size, z_size, yaw] with bottom-center
    origin.  The evaluator's BEV footprint is (x, z) with (x_size, z_size),
    height overlap is along y, and the rotation uses ``yaw`` directly.
    """
    boxes = np.asarray(boxes, dtype=np.float64)
    qboxes = np.asarray(qboxes, dtype=np.float64)
    if len(boxes) == 0 or len(qboxes) == 0:
        return np.zeros((len(boxes), len(qboxes)), dtype=np.float64)
    rinc = vod_rotate_iou_cpu.rotate_iou_eval(
        boxes[:, [0, 2, 3, 5, 6]], qboxes[:, [0, 2, 3, 5, 6]], 2)
    rinc = rinc.astype(np.float64)
    tj4d_eval.d3_box_overlap_kernel(boxes, qboxes, rinc, criterion)
    return rinc


def _camera_corners_for_projection(boxes):
    """Camera-frame corners for 2D projection (mmdet3d CameraInstance3DBoxes)."""
    boxes = np.asarray(boxes, dtype=np.float64)
    corners_norm = np.stack(
        np.unravel_index(np.arange(8), [2, 2, 2]), axis=1).astype(np.float64)
    corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array([0.5, 1.0, 0.5])
    dims3 = boxes[:, [3, 4, 5]]
    corners = dims3[:, None, :] * corners_norm[None, :, :]
    yaw = boxes[:, 6]
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.zeros((len(boxes), 3, 3), dtype=np.float64)
    rot[:, 0, 0] = c
    rot[:, 1, 1] = 1.0
    rot[:, 2, 2] = c
    rot[:, 0, 2] = s
    rot[:, 2, 0] = -s
    corners = np.einsum('nij,nkj->nki', rot, corners)
    corners += boxes[:, None, :3]
    return corners


def prediction_valid_mask(pred_boxes, pred_boxes_lidar, info):
    if len(pred_boxes) == 0:
        return np.zeros((0,), dtype=bool)
    p2 = np.asarray(info['calib']['P2'], dtype=np.float64)
    img_shape = np.asarray(info['image']['image_shape'], dtype=np.float64)
    corners = _camera_corners_for_projection(pred_boxes)
    corners_h = np.concatenate(
        [corners, np.ones((*corners.shape[:2], 1))], axis=-1)
    corners_img = corners_h @ p2.T
    corners_img = corners_img[..., :2] / corners_img[..., 2:3]
    minxy = corners_img.min(axis=1)
    maxxy = corners_img.max(axis=1)
    box_2d = np.concatenate([minxy, maxxy], axis=1)
    valid_cam = ((box_2d[:, 0] < img_shape[1]) &
                 (box_2d[:, 1] < img_shape[0]) &
                 (box_2d[:, 2] > 0) & (box_2d[:, 3] > 0))
    limit_range = np.asarray([0, -40, -3, 70.4, 40, 0.0], dtype=np.float64)
    valid_pcd = ((pred_boxes_lidar[:, :3] > limit_range[:3]) &
                 (pred_boxes_lidar[:, :3] < limit_range[3:])).all(axis=-1)
    return valid_cam & valid_pcd


def base_anno_from_info(info):
    annos = info['annos']
    return {
        'name': np.asarray(annos['name'], dtype=object),
        'truncated': np.asarray(annos['truncated'], dtype=np.float64),
        'occluded': np.asarray(annos['occluded'], dtype=np.int64),
        'alpha': np.asarray(annos['alpha'], dtype=np.float64),
        'bbox': np.asarray(annos['bbox'], dtype=np.float64),
        'dimensions': np.asarray(annos['dimensions'], dtype=np.float64),
        'location': np.asarray(annos['location'], dtype=np.float64),
        'rotation_y': np.asarray(annos['rotation_y'], dtype=np.float64),
        'score': np.asarray(annos['score'], dtype=np.float64),
        'difficulty': np.asarray(annos['difficulty'], dtype=np.int64),
    }


def gt_boxes_from_anno(anno, class_id, difficulty):
    """Camera-frame bottom-center 3D boxes for one class/difficulty.

    The info pkl already stores KITTI camera bottom-center ``location`` and the
    camera-format dimensions, so no coordinate shift is applied here.  The
    evaluator reassigns difficulty from range during ``clean_data`` and treats
    difficulty as cumulative; mirror that rule so oracle recall uses the same
    GT denominator as the formal PR curve.
    """
    name = np.asarray(anno['name'], dtype=object)
    pkl_diff = np.asarray(anno['difficulty'], dtype=np.int64)
    locations = np.asarray(anno['location'], dtype=np.float64)
    ranges = np.linalg.norm(locations, axis=1)
    if difficulty == 0:
        range_mask = (ranges > 0) & (ranges <= 50)
    elif difficulty == 1:
        range_mask = (ranges > 0) & (ranges <= 70)
    elif difficulty == 2:
        range_mask = np.ones_like(ranges, dtype=bool)
    else:
        raise ValueError(f'unsupported difficulty {difficulty}')
    mask = (
        (name == EVAL_CLASS_TO_NAME[class_id]) &
        (pkl_diff >= 0) & range_mask)
    locations = np.asarray(anno['location'][mask], dtype=np.float64)
    dims = np.asarray(anno['dimensions'][mask], dtype=np.float64)
    rots = np.asarray(anno['rotation_y'][mask], dtype=np.float64)
    return np.concatenate([locations, dims, rots[..., None]], axis=1)


def build_eval_annos(infos, dump, class_name, difficulty=1):
    """Build camera-frame GT/pred annots for one class/difficulty."""
    gt_annos = []
    dt_annos = []
    sample_meta = []
    model_class_id = NAME_TO_MODEL_CLASS[class_name]
    eval_class_id = EVAL_NAME_TO_CLASS[class_name]
    for idx, info in enumerate(infos):
        pred = dump['preds'][idx]
        pred_boxes, pred_boxes_lidar = lidar_boxes_to_camera(
            pred['tensor'], info)
        pred_scores = np.asarray(pred['scores'], dtype=np.float64)
        pred_labels = np.asarray(pred['labels']).astype(int)

        gt_annos.append(base_anno_from_info(info))

        pred_mask = pred_labels == model_class_id
        valid_pcd = prediction_valid_mask(
            pred_boxes, pred_boxes_lidar, info)
        pred_mask = pred_mask & valid_pcd
        pred_boxes = pred_boxes[pred_mask]
        pred_boxes_lidar = pred_boxes_lidar[pred_mask]
        pred_scores = pred_scores[pred_mask]

        dt_anno = empty_anno(len(pred_boxes))
        if len(pred_boxes):
            dt_anno['name'] = np.asarray(
                [class_name] * len(pred_boxes), dtype=object)
            dt_anno['bbox'] = np.zeros((len(pred_boxes), 4), dtype=np.float64)
            dt_anno['dimensions'] = pred_boxes[:, 3:6]
            dt_anno['location'] = pred_boxes[:, :3]
            dt_anno['rotation_y'] = pred_boxes[:, 6]
            dt_anno['alpha'] = np.full(
                (len(pred_boxes),), -10.0, dtype=np.float64)
            dt_anno['score'] = pred_scores

        dt_annos.append(dt_anno)
        sample_meta.append({
            'gt_boxes': gt_boxes_from_anno(
                gt_annos[-1], eval_class_id, difficulty),
            'pred_boxes': pred_boxes,
            'pred_boxes_lidar': pred_boxes_lidar,
            'pred_scores': pred_scores,
            'num_gt': int((np.asarray(gt_annos[-1]['name']) == class_name).sum()),
            'num_pred': int(len(pred_boxes)),
        })
    return gt_annos, dt_annos, sample_meta


def empty_anno(n):
    return {
        'name': np.asarray([], dtype=object),
        'truncated': np.zeros((n,), dtype=np.float64),
        'occluded': np.zeros((n,), dtype=np.int64),
        'alpha': np.full((n,), -10.0, dtype=np.float64),
        'bbox': np.zeros((n, 4), dtype=np.float64),
        'dimensions': np.zeros((n, 3), dtype=np.float64),
        'location': np.zeros((n, 3), dtype=np.float64),
        'rotation_y': np.zeros((n,), dtype=np.float64),
        'score': np.zeros((n,), dtype=np.float64),
        'difficulty': np.full((n,), -1, dtype=np.int64),
    }


def eval_moderate_cyclist(gt_annos, dt_annos, class_id, difficulty, iou_thr):
    metric = 2
    original_d3 = tj4d_eval.d3_box_overlap
    tj4d_eval.d3_box_overlap = d3_box_overlap_cpu
    try:
        min_overlaps = np.zeros((1, 3, 1), dtype=np.float64)
        min_overlaps[0, metric, 0] = iou_thr
        ret = tj4d_eval.eval_class(
            gt_annos,
            dt_annos,
            [class_id],
            [difficulty],
            metric,
            min_overlaps,
            num_parts=200,
        )
    finally:
        tj4d_eval.d3_box_overlap = original_d3
    prec = ret['precision'][0, 0, 0]
    rec = ret['recall'][0, 0, 0]
    ap = float(tj4d_eval.get_mAP(ret['precision'])[0, 0, 0])
    return {
        'ap40': ap,
        'precision': prec.tolist(),
        'recall': rec.tolist(),
    }


def oracle_recall(sample_meta, iou_thr):
    """One-to-one oracle matching within each sample, ignoring scores.

    This is a maximum-cardinality greedy matching: each prediction can match at
    most one GT and each GT at most one prediction when IoU > threshold.
    """
    total_gt = 0
    matched = 0
    matched_iou = []
    for sample in sample_meta:
        gt_boxes = sample['gt_boxes']
        pred_boxes = sample['pred_boxes']
        total_gt += len(gt_boxes)
        if len(gt_boxes) == 0 or len(pred_boxes) == 0:
            continue
        overlaps = d3_box_overlap_cpu(gt_boxes, pred_boxes)
        overlaps = np.asarray(overlaps, dtype=np.float64)
        # Greedy by highest IoU is sufficient for the diagnostic signal here.
        pair = []
        for gi in range(overlaps.shape[0]):
            for pi in range(overlaps.shape[1]):
                iou = overlaps[gi, pi]
                if iou >= iou_thr:
                    pair.append((iou, gi, pi))
        pair.sort(reverse=True)
        used_gt = set()
        used_pred = set()
        for iou, gi, pi in pair:
            if gi in used_gt or pi in used_pred:
                continue
            used_gt.add(gi)
            used_pred.add(pi)
            matched += 1
            matched_iou.append(float(iou))
    return {
        'total_gt': int(total_gt),
        'matched_gt': int(matched),
        'recall': float(matched / total_gt) if total_gt else 0.0,
        'matched_iou': matched_iou,
    }


def score_bin_stats(sample_meta, iou_thr):
    """FP/TP counts over score bins for same-class predictions.

    The match is one-to-one per sample at the requested IoU threshold.  A false
    positive is a same-class prediction not matched to any GT at that threshold.
    """
    bins = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0})
    unmatched_gt_scores = []
    for sample in sample_meta:
        gt_boxes = sample['gt_boxes']
        pred_boxes = sample['pred_boxes']
        pred_scores = sample['pred_scores']
        overlaps = np.zeros((len(gt_boxes), len(pred_boxes)), dtype=np.float64)
        if len(gt_boxes) and len(pred_boxes):
            overlaps = d3_box_overlap_cpu(gt_boxes, pred_boxes)
        pairs = []
        for gi in range(overlaps.shape[0]):
            for pi in range(overlaps.shape[1]):
                if overlaps[gi, pi] >= iou_thr:
                    pairs.append((overlaps[gi, pi], gi, pi))
        pairs.sort(reverse=True)
        used_gt = set()
        used_pred = set()
        for _, gi, pi in pairs:
            if gi in used_gt or pi in used_pred:
                continue
            used_gt.add(gi)
            used_pred.add(pi)
            score = float(pred_scores[pi])
            bins[score_bin(score)]['tp'] += 1
        for pi, score in enumerate(pred_scores):
            if pi not in used_pred:
                bins[score_bin(float(score))]['fp'] += 1
        for gi in range(len(gt_boxes)):
            if gi not in used_gt:
                bins['fn']['fn'] += 1
                unmatched_gt_scores.append(0.0)
    return {
        'bins': {k: v for k, v in sorted(bins.items())},
        'unmatched_gt_scores': unmatched_gt_scores,
    }


def score_bin(score):
    for lo, hi in zip(SCORE_BINS[:-1], SCORE_BINS[1:]):
        if lo <= score < hi:
            return f'{lo:.2f}-{hi:.2f}'
    return 'invalid'


def summarize(name, dump, infos, class_name, class_id, difficulty, iou_thr):
    gt_annos, dt_annos, sample_meta = build_eval_annos(infos, dump, class_name)
    oracle = oracle_recall(sample_meta, iou_thr)
    formal = eval_moderate_cyclist(gt_annos, dt_annos, class_id, difficulty, iou_thr)
    bins = score_bin_stats(sample_meta, iou_thr)
    out = {
        'name': name,
        'total_gt': oracle['total_gt'],
        'oracle_recall': oracle['recall'],
        'oracle_matched_gt': oracle['matched_gt'],
        'ap40': formal['ap40'],
        'precision': formal['precision'],
        'recall': formal['recall'],
        'score_bins': bins['bins'],
        'num_pred': int(sum(s['num_pred'] for s in sample_meta)),
    }
    print(f'[{name}]')
    print(f'  moderate Cyc GT={out["total_gt"]} oracle recall={out["oracle_recall"]:.4f} '
          f'({out["oracle_matched_gt"]}/{out["total_gt"]})')
    print(f'  formal 3D AP40 IoU={iou_thr:.2f}: {out["ap40"]:.4f}')
    print('  41-point PR:')
    for i, (p, r) in enumerate(zip(out['precision'], out['recall'])):
        print(f'    {i:02d}  P={p:.4f}  R={r:.4f}')
    print('  score bins (same-class one-to-one IoU >= threshold):')
    for bin_name, vals in out['score_bins'].items():
        print(f'    {bin_name}: TP={vals["tp"]} FP={vals["fp"]} FN={vals["fn"]}')
    return out


EXPECTED_AP = {
    'clean_seed0_ep16': 50.4709,
    'stem_seed0_ep14': 45.3136,
    'stem_seed0_ep16': 41.9200,
}
AP_TOL = 0.05


def main():
    args = parse_args()
    infos = load_infos(args.config)
    class_id = EVAL_NAME_TO_CLASS[args.class_name]
    results = {}
    for dump_path in args.dump:
        with open(dump_path, 'rb') as f:
            dump = pickle.load(f)
        name = Path(dump_path).stem
        results[name] = summarize(
            name, dump, infos, args.class_name, class_id, args.difficulty,
            args.iou_thr)

    checks = []
    for name, out in results.items():
        expected = EXPECTED_AP.get(name)
        if expected is None:
            continue
        ok = abs(out['ap40'] - expected) <= AP_TOL
        checks.append((name, out['ap40'], expected, ok))
        tag = 'OK' if ok else 'MISMATCH'
        print(f'[reproduce] {name}: ap40={out["ap40"]:.4f} '
              f'expected={expected:.4f} {tag}')
    failed = [c for c in checks if not c[3]]
    if failed:
        raise SystemExit(
            'Formal AP reproduction failed; oracle recall and PR output are '
            'not trustworthy until this matches. Failing entries: '
            + ', '.join(f'{n} (got {g:.4f}, want {e:.4f})'
                        for n, g, e, _ in failed))

    if args.out_json:
        import json
        try:
            with open(args.out_json, 'w') as f:
                json.dump(results, f, indent=2)
        except OSError as exc:
            print(f'[warning] could not write {args.out_json}: {exc}')


if __name__ == '__main__':
    main()
