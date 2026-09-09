#!/usr/bin/env python3
"""CPU-only NMS sweep for the Pedestrian CenterHead from a RAW dump.

The raw dump (dump_predictions.py --raw-nms) keeps ALL 100 decoded Ped boxes
per sample (bbox_coder max_num=100, circle NMS radius=0).  This script
re-applies different NMS configs to the Ped subset and reports Pedestrian
recall (greedy, LiDAR frame) at loose@0.25 / strict@0.5, plus box counts,
so we can decide whether NMS is the recall bottleneck.

NMS variants:
  - baseline          : post-NMS dump as saved by the original config (circle r=1)
  - circle r=1.0 / 0.5 / 0.25
  - rotate IoU 0.2

Usage:
  python tools/ped_nms_sweep.py --raw RAW.pkl --post POST.pkl
"""
import argparse
import pickle
import numpy as np
from shapely.geometry import Polygon

PED = 0
STRICT = 0.5
LOOSE = 0.25


def bev_corners(b):
    x, y, z, w, l, h, yaw = b
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    local = np.array([[-l/2, -w/2], [l/2, -w/2], [l/2, w/2], [-l/2, w/2]])
    return local @ R.T + np.array([x, y])


def poly(b):
    p = Polygon(bev_corners(b))
    return p.buffer(0) if not p.is_valid else p


def bev_iou(b1, b2):
    p1, p2 = poly(b1), poly(b2)
    inter = p1.intersection(p2).area
    union = p1.area + p2.area - inter
    return inter / (union + 1e-9)


def oriented_3d_iou(b1, b2):
    p1, p2 = poly(b1), poly(b2)
    inter = p1.intersection(p2).area
    if inter <= 1e-12:
        return 0.0
    z1, h1 = b1[2], b1[5]
    z2, h2 = b2[2], b2[5]
    top = max(z1 - h1/2, z2 - h2/2)
    bot = min(z1 + h1/2, z2 + h2/2)
    oh = bot - top
    if oh <= 0:
        return 0.0
    v1 = b1[3]*b1[4]*b1[5]
    v2 = b2[3]*b2[4]*b2[5]
    return inter*oh / (v1 + v2 - inter*oh + 1e-9)


def circle_nms_idx(boxes, scores, radius):
    order = np.argsort(-scores)
    keep = []
    suppressed = np.zeros(len(scores), dtype=bool)
    for i in order:
        if suppressed[i]:
            continue
        keep.append(i)
        for j in order:
            if j == i or suppressed[j]:
                continue
            d = np.hypot(boxes[i,0]-boxes[j,0], boxes[i,1]-boxes[j,1])
            if d <= radius:
                suppressed[j] = True
    return np.array(keep, dtype=np.int64)


def rotate_nms_idx(boxes, scores, iou_thr):
    order = np.argsort(-scores)
    keep = []
    suppressed = np.zeros(len(scores), dtype=bool)
    for i in order:
        if suppressed[i]:
            continue
        keep.append(i)
        for j in order:
            if j == i or suppressed[j]:
                continue
            if bev_iou(boxes[i], boxes[j]) >= iou_thr:
                suppressed[j] = True
    return np.array(keep, dtype=np.int64)


def greedy_ped_recall(gts, preds):
    """Returns (ped_gt, loose, strict)."""
    gt_total = 0
    l = s = 0
    for i in range(len(gts)):
        gt = gts[i]; pr = preds[i]
        gids = np.where(gt['labels'] == PED)[0]
        dids = np.where(pr['labels'] == PED)[0]
        gt_n = len(gids)
        gt_total += gt_n
        if gt_n == 0 or len(dids) == 0:
            continue
        gb = gt['tensor']; pb = pr['tensor']
        order = sorted(dids, key=lambda j: -pr['scores'][j])
        assigned = set()
        for j in order:
            best, bi = -1, 0.0
            for g in gids:
                if g in assigned:
                    continue
                iu = oriented_3d_iou(gb[g], pb[j])
                if iu > bi:
                    bi, best = iu, g
            if best >= 0:
                assigned.add(best)
                if bi >= LOOSE:
                    l += 1
                if bi >= STRICT:
                    s += 1
    return gt_total, l, s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', required=True)
    ap.add_argument('--post', required=True)
    a = ap.parse_args()

    with open(a.raw, 'rb') as f: raw = pickle.load(f)
    with open(a.post, 'rb') as f: post = pickle.load(f)
    gts = raw['gts']
    N = len(gts)

    # Ped boxes from raw + non-ped boxes from raw (anchor head, frozen, identical)
    def apply(raw_boxes, raw_scores, raw_labels, mode):
        """mode: None -> keep all raw; else (kind,param)."""
        out_boxes, out_scores, out_labels = [], [], []
        for i in range(N):
            boxes = raw_boxes[i]; scores = raw_scores[i]; labels = raw_labels[i]
            ped = labels == PED
            pboxes, pscores = boxes[ped], scores[ped]
            nped = labels != PED
            # non-ped always kept
            ob = boxes[nped]; os = scores[nped]; ol = labels[nped]
            if mode is None:
                pb, ps = pboxes, pscores
            else:
                kind, param = mode
                if kind == 'none':
                    pb, ps = pboxes, pscores
                elif kind == 'circle':
                    k = circle_nms_idx(pboxes, pscores, param)
                    pb, ps = pboxes[k], pscores[k]
                elif kind == 'rotate':
                    k = rotate_nms_idx(pboxes, pscores, param)
                    pb, ps = pboxes[k], pscores[k]
                else:
                    raise ValueError(kind)
            out_boxes.append(np.concatenate([ob, pb], 0))
            out_scores.append(np.concatenate([os, ps], 0))
            out_labels.append(np.concatenate([ol, np.zeros(len(pb), dtype=ol.dtype)], 0))
        return out_boxes, out_scores, out_labels

    rb = [raw['preds'][i]['tensor'] for i in range(N)]
    rs = [raw['preds'][i]['scores'] for i in range(N)]
    rl = [raw['preds'][i]['labels'] for i in range(N)]

    variants = [
        ('post-baseline(circle r=1)', None),
        ('circle r=1.0', ('circle', 1.0)),
        ('circle r=0.5', ('circle', 0.5)),
        ('circle r=0.25', ('circle', 0.25)),
        ('rotate IoU=0.2', ('rotate', 0.2)),
        ('no NMS (top100 raw)', ('none', 0)),
    ]

    # baseline uses the POST dump as-is
    info = {}

    for name, mode in variants:
        if name == 'post-baseline(circle r=1)':
            pb = [post['preds'][i]['tensor'] for i in range(N)]
            ps = [post['preds'][i]['scores'] for i in range(N)]
            pl = [post['preds'][i]['labels'] for i in range(N)]
        else:
            pb, ps, pl = apply(rb, rs, rl, mode)
        preds = [{'tensor': pb[i], 'scores': ps[i], 'labels': pl[i]} for i in range(N)]
        ped_counts = [int((pl[i] == PED).sum()) for i in range(N)]
        gt_total, loose, strict = greedy_ped_recall(gts, preds)
        info[name] = dict(
            ped_boxes_mean=np.mean(ped_counts),
            loose_recall=loose / max(gt_total, 1),
            strict_recall=strict / max(gt_total, 1),
            loose=loose, strict=strict, gt=gt_total)

    print('=' * 72)
    print('Pedestrian NMS sweep on ep7 (LiDAR-frame greedy recall)')
    print('=' * 72)
    hdr = f'{"variant":>24} {"ped/box":>8} {"GT":>5} {"loose@.25":>10} {"strict@.5":>11}'
    print(hdr)
    print('-' * 72)
    for name, _ in variants:
        d = info[name]
        print(f'{name:>24} {d["ped_boxes_mean"]:8.2f} {d["gt"]:5d} '
              f'{d["loose_recall"]:10.4f} {d["strict_recall"]:11.4f}')
    print()
    print('DELTA vs post-baseline (circle r=1):')
    bl = info['post-baseline(circle r=1)']
    for name, _ in variants[1:]:
        d = info[name]
        print(f'  {name:>24}: loose {d["loose_recall"]-bl["loose_recall"]:+.4f} '
              f' strict {d["strict_recall"]-bl["strict_recall"]:+.4f}')

if __name__ == '__main__':
    main()
