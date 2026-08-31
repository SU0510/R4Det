"""CPU-only error diagnosis from dump_predictions.py output.

Produces 5 diagnostics aggregated over the provided checkpoint dumps:
  1. Car/Truck (and full 4-class) classification confusion matrix
  2. Correct-classification strict IoU pass rate
  3. center / size / yaw / z errors (for correctly-classified matches)
  4. Recall bucketed by distance and object size (strict 0.5 & loose 0.25)
  5. Loose-success-but-strict-fail cause decomposition

Box format: LiDAR frame [x, y, z, w, l, h, yaw] (radians).
Class order: dataset.CLASSES = ['Pedestrian','Cyclist','Car','Truck']
             -> index 0=Ped, 1=Cyc, 2=Car, 3=Truck.

Matching mirrors KITTI eval: greedy one-to-one per class, detections sorted by
descending score.

Usage:
  python tools/diagnose_errors.py --dumps a.pkl b.pkl c.pkl
"""
import argparse
import pickle
import numpy as np
from shapely.geometry import Polygon

STRICT = 0.5
LOOSE = 0.25
CLASS_NAMES = ['Pedestrian', 'Cyclist', 'Car', 'Truck']


def bev_corners(b):
    x, y, z, w, l, h, yaw = b
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    local = np.array([[-l/2, -w/2], [l/2, -w/2], [l/2, w/2], [-l/2, w/2]])
    return local @ R.T + np.array([x, y])


def bev_iou(b1, b2):
    p1 = Polygon(bev_corners(b1))
    p2 = Polygon(bev_corners(b2))
    if not p1.is_valid:
        p1 = p1.buffer(0)
    if not p2.is_valid:
        p2 = p2.buffer(0)
    inter = p1.intersection(p2).area
    union = p1.area + p2.area - inter
    return inter / (union + 1e-9)


def _poly(c):
    p = Polygon(c)
    return p.buffer(0) if not p.is_valid else p


def nearest_center_dists(gt_xy, pred_xy):
    """Vectorized L2 dist from each GT center to each pred center -> (ngt, npred)."""
    d = gt_xy[:, None, :] - pred_xy[None, :, :]
    return np.sqrt((d ** 2).sum(-1))


def oriented_3d_iou(b1, b2):
    p1 = _poly(bev_corners(b1))
    p2 = _poly(bev_corners(b2))
    inter_area = p1.intersection(p2).area
    if inter_area <= 1e-12:
        return 0.0
    z1, h1 = b1[2], b1[5]
    z2, h2 = b2[2], b2[5]
    top = max(z1 - h1/2, z2 - h2/2)
    bot = min(z1 + h1/2, z2 + h2/2)
    oh = bot - top
    if oh <= 0:
        return 0.0
    inter_vol = inter_area * oh
    vol1 = b1[3] * b1[4] * b1[5]
    vol2 = b2[3] * b2[4] * b2[5]
    union = vol1 + vol2 - inter_vol
    return inter_vol / (union + 1e-9)


def yaw_diff(a, b):
    d = (a - b + np.pi) % (2*np.pi) - np.pi
    return abs(d)


def long_axis_heading(b):
    """Canonical heading of the long axis (rad, mod pi). w/l convention robust."""
    w, l, yaw = b[3], b[4], b[6]
    if l >= w:
        heading = yaw
    else:
        heading = yaw + np.pi / 2
    return heading % np.pi


def orientation_error(b1, b2):
    """Minimal long-axis angular error in [0, pi/2], robust to w/l + 180deg."""
    h1 = long_axis_heading(b1)
    h2 = long_axis_heading(b2)
    d = abs(h1 - h2)
    d = min(d, np.pi - d)
    return d


def long_extent(b):
    return float(max(b[3], b[4]))


def short_extent(b):
    return float(min(b[3], b[4]))


def center_bev_err(b1, b2):
    return float(np.hypot(b1[0] - b2[0], b1[1] - b2[1]))


def greedy_match(gt_parts, pred_parts):
    """gt_parts/pred_parts: list of (class, idx). Returns dict gt_idx -> (pred_idx, iou, pred_cls)."""
    gt_boxes = gt_parts[0]
    pr_boxes = pred_parts[0]
    gt_lab = gt_parts[1]
    pr_lab = pred_parts[1]
    pr_sco = pred_parts[2]
    m = {}
    for c in range(4):
        gids = np.where(gt_lab == c)[0]
        dids = np.where(pr_lab == c)[0]
        if len(gids) == 0 or len(dids) == 0:
            continue
        order = sorted(dids, key=lambda j: -pr_sco[j])
        assigned_gt = set()
        for j in order:
            best = None
            best_iou = 1e-9
            for g in gids:
                if g in assigned_gt:
                    continue
                iou = oriented_3d_iou(gt_boxes[g], pr_boxes[j])
                if iou > best_iou:
                    best_iou = iou
                    best = g
            if best is not None:
                assigned_gt.add(best)
                m[best] = (j, best_iou, c)
    return m


def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dumps', nargs='+', required=True)
    args = ap.parse_args()

    dumps = [load(p) for p in args.dumps]
    for d in dumps:
        assert list(d['class_names']) == CLASS_NAMES, d['class_names']
    N = len(dumps[0]['gts'])
    for d in dumps:
        assert len(d['gts']) == N and len(d['preds']) == N

    # accumulators
    conf = np.zeros((4, 4), dtype=np.int64)     # gt class -> matched pred class (all-class, IoU>=0.01)
    gt_tot = np.zeros(4, dtype=np.int64)
    # strict pass for correct class
    strict_correct = np.zeros(4, dtype=np.int64)
    strict_denom = np.zeros(4, dtype=np.int64)
    # errors (correct class matches)
    err = {'center_bev': [], 'dz': [], 'long': [], 'short': [], 'orient': []}
    # recall buckets
    dbuck = [0, 20, 40, 60, 80, 100, 1e9]
    dbuck_name = ['0-20', '20-40', '40-60', '60-80', '80-100', '100+']
    sbuck = [0, 1.5, 2.5, 4.0, 1e9]   # max(w,l) meters
    sbuck_name = ['<1.5', '1.5-2.5', '2.5-4.0', '4.0+']
    rec = {}   # (class, dbucket) -> [gt_count, loose_hit, strict_hit]
        # size-bucket recall tracked for Car/Truck only

    def bucket_idx(v, buckets):
        for k in range(len(buckets) - 1):
            if buckets[k] <= v < buckets[k+1]:
                return k
        return len(buckets) - 2

    # loose-strict decomposition
    causes = {'center': 0, 'orient': 0, 'size': 0, 'z': 0, 'comb': 0}

    for i in range(N):
        for d in dumps:
            gt = d['gts'][i]
            pr = d['preds'][i]
            gt_np = np.asarray(gt['tensor'], dtype=np.float64)
            gt_lab = np.asarray(gt['labels'], dtype=np.int64)
            pr_np = np.asarray(pr['tensor'], dtype=np.float64)
            pr_sco = np.asarray(pr['scores'], dtype=np.float64)
            pr_lab = np.asarray(pr['labels'], dtype=np.int64)
            ngt = len(gt_lab)
            npr = len(pr_lab)

            # count all GT (for unmatched accounting), regardless of preds
            for g in range(ngt):
                gt_tot[gt_lab[g]] += 1

            # --- all-class confusion matrix (with center-distance prefilter) ---
            if npr == 0:
                continue
            gt_xy = gt_np[:, :2]
            pr_xy = pr_np[:, :2]
            d = nearest_center_dists(gt_xy, pr_xy)
            # max plausible car/truck half-diagonal ~6m; filter conservatively
            max_diag = np.sqrt(gt_np[:, 3]**2 + gt_np[:, 4]**2) / 2
            for g in range(ngt):
                cand = np.where(d[g] <= max_diag[g] + 3.0)[0]
                best_j, best_iou = -1, 0.0
                for j in cand:
                    iou = oriented_3d_iou(gt_np[g], pr_np[j])
                    if iou > best_iou:
                        best_iou = iou
                        best_j = j
                if best_j >= 0 and best_iou >= 0.01:
                    conf[gt_lab[g], pr_lab[best_j]] += 1

            # --- same-class greedy matching for diagnostics 2-5
            m = greedy_match((gt_np, gt_lab), (pr_np, pr_lab, pr_sco))
            for g, (j, iou, pc) in m.items():
                gc = gt_lab[g]
                gb = gt_np[g]
                pb = pr_np[j]
                dist = float(np.hypot(gb[0], gb[1]))
                size = float(max(gb[3], gb[4]))

                # recall buckets (per class)
                key = (gc, bucket_idx(dist, dbuck))
                rec.setdefault(key, [0, 0, 0])
                rec[key][0] += 1
                if iou >= LOOSE:
                    rec[key][1] += 1
                if iou >= STRICT:
                    rec[key][2] += 1

                # correct-class diagnostics
                if pc == gc:
                    strict_denom[gc] += 1
                    if iou >= STRICT:
                        strict_correct[gc] += 1
                    err['center_bev'].append(center_bev_err(pb, gb))
                    err['dz'].append(abs(pb[2] - gb[2]))
                    err['long'].append(abs(long_extent(pb) - long_extent(gb)))
                    err['short'].append(abs(short_extent(pb) - short_extent(gb)))
                    err['orient'].append(orientation_error(pb, gb))

                # loose ok strict fail decomposition (convention-robust)
                if LOOSE <= iou < STRICT:
                    c = center_bev_err(pb, gb)
                    o = orientation_error(pb, gb)
                    dz = abs(pb[2] - gb[2])
                    # width/length (convention-free) size error normalized by GT short axis
                    s = max(abs(long_extent(pb) - long_extent(gb)),
                            abs(short_extent(pb) - short_extent(gb))) / max(short_extent(gb), 1e-3)
                    r = []
                    if c > 1.0: r.append('center')
                    if o > 0.35: r.append('orient')
                    if s > 0.35: r.append('size')
                    if dz > 0.5: r.append('z')
                    causes[(r[0] if len(r) == 1 else 'comb')] += 1

    print('=' * 72)
    print('1. Classification confusion matrix (rows=GT, cols=pred, IoU>=0.01)')
    print('=' * 72)
    print('            ' + ''.join(f'{n:>12}' for n in CLASS_NAMES) + f'  {"unmatched":>12}')
    for r in range(4):
        row = ''.join(f'{int(conf[r,c]):>12}' for c in range(4))
        unmatched = int(gt_tot[r] - conf[r].sum())
        print(f'{CLASS_NAMES[r]:>12}' + row + f'{unmatched:>12}')

    print()
    print('Car/Truck mutual confusion ratio:')
    for a, b, a_i, b_i in [('Car', 'Truck', 2, 3)]:
        cross = int(conf[a_i, b_i] + conf[b_i, a_i])
        diag = int(conf[a_i, a_i] + conf[b_i, b_i])
        print(f'  {a}<->{b} cross={cross}, diag={diag}, cross_ratio={cross/max(1,(cross+diag)):.3f}')

    print()
    print('=' * 72)
    print('2. Correct-classification strict IoU pass rate (IoU>=0.5 among same-class matches)')
    print('=' * 72)
    for c in range(4):
        den = int(strict_denom[c])
        num = int(strict_correct[c])
        rate = num/den if den else 0.0
        print(f'  {CLASS_NAMES[c]:>12}: {num}/{den} = {rate:.3f}')

    print()
    print('=' * 72)
    print('3. Errors on correctly-classified matches (mean / median)')
    print('=' * 72)
    for k, name in [('center_bev','BEV center(m)'),('dz','z(m)'),
                    ('long','long-axis(m)'),('short','short-axis(m)'),('orient','orient(rad)')]:
        v = np.array(err[k])
        if len(v):
            print(f'  {name:>16}: mean={v.mean():.3f}  median={np.median(v):.3f}')
        else:
            print(f'  {name:>16}: (no matches)')

    print()
    print('=' * 72)
    print('4a. Recall by distance bucket (all classes; loose@0.25 / strict@0.5)')
    print('=' * 72)
    for c in range(4):
        parts = [f'{CLASS_NAMES[c]:>12}:']
        for b in range(len(dbuck_name)):
            key = (c, b)
            gt, l, s = rec.get(key, (0, 0, 0))
            if gt:
                parts.append(f' {dbuck_name[b]}[{gt:>4}] L={l/gt:.2f} S={s/gt:.2f}')
            else:
                parts.append(f' {dbuck_name[b]}[   0]')
        print(''.join(parts))

    print()
    print('4b. Recall by size bucket (Car/Truck only; loose@0.25 / strict@0.5)')
    print('=' * 72)
    # recompute size-bucket recall by re-scanning matches
    srec = {}
    for i in range(N):
        for d in dumps:
            gt = d['gts'][i]
            pr = d['preds'][i]
            gt_np = np.asarray(gt['tensor'], dtype=np.float64)
            gt_lab = np.asarray(gt['labels'], dtype=np.int64)
            pr_np = np.asarray(pr['tensor'], dtype=np.float64)
            pr_sco = np.asarray(pr['scores'], dtype=np.float64)
            pr_lab = np.asarray(pr['labels'], dtype=np.int64)
            m = greedy_match((gt_np, gt_lab), (pr_np, pr_lab, pr_sco))
            for g, (j, iou, pc) in m.items():
                gc = gt_lab[g]
                if gc not in (2, 3):
                    continue
                size = float(max(gt_np[g][3], gt_np[g][4]))
                key = (gc, bucket_idx(size, sbuck))
                srec.setdefault(key, [0, 0, 0])
                srec[key][0] += 1
                if iou >= LOOSE: srec[key][1] += 1
                if iou >= STRICT: srec[key][2] += 1
    for c in (2, 3):
        parts = [f'{CLASS_NAMES[c]:>12}:']
        for b in range(len(sbuck_name)):
            gt, l, s = srec.get((c, b), (0, 0, 0))
            parts.append(f' {sbuck_name[b]}[{gt:>3}] L={l/max(gt,1):.2f} S={s/max(gt,1):.2f}')
        print(''.join(parts))

    print()
    print('=' * 72)
    print('5. Loose-success (0.25<=IoU<0.5) but strict-fail cause decomposition')
    print('=' * 72)
    total = sum(causes.values())
    for k in ['center', 'orient', 'size', 'z', 'comb']:
        print(f'  {k:>8}: {causes[k]}  ({causes[k]/max(total,1):.3f})')
    print(f'  {"total":>8}: {total}')


if __name__ == '__main__':
    main()
