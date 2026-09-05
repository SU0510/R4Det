"""Pedestrian-focused error decomposition from dump_predictions.py output.

Complements tools/diagnose_errors.py. Answers, for Pedestrian only:
  A. Global / per-anchor-max conf (sigmoid) → "is Ped score itself low?"
  B. Recalled-as-Ped λ-recall@0.1..0.85 (anchor capacity of top-dir headed box)
  C. Score margin of every correctly-recalled Ped match + binned error deltas
  D. Per-component 7D breakdown: center_bev / w / l / h / yaw, tolerance-binned
  E. Loose(0.25)/strict(0.5) + distance buckets + Ped size buckets
  F. Mutually-exclusive root-cause flags for the 0.25<=IoU<0.5 band

Frame: LiDAR [x, y, z, w, l, h, yaw] (rad). w=short(across), l=long(along).
CLASSES = ['Pedestrian','Cyclist','Car','Truck']; Ped=0, Cyc=1, Car=2, Truck=3.
Anchor scheme: 3 Ped anchors (r0=0, r1=pi/2) mapped class 0; Cyc/Car/Truck one each.

Matching mirrors KITTI eval: greedy one-to-one per class, dets sorted by desc score.

Usage:
  python tools/ped_error_breakdown.py --dumps a.pkl b.pkl ...
"""
import argparse
import pickle
import numpy as np
from shapely.geometry import Polygon

CLASS_NAMES = ['Pedestrian', 'Cyclist', 'Car', 'Truck']
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
    inter_vol = inter * oh
    vol1 = b1[3] * b1[4] * b1[5]
    vol2 = b2[3] * b2[4] * b2[5]
    return inter_vol / (vol1 + vol2 - inter_vol + 1e-9)


def heading(b):
    """Long-axis heading mod pi (w/l convention-robust)."""
    w, l, yaw = b[3], b[4], b[6]
    return (yaw if l >= w else yaw + np.pi/2) % np.pi


def orient_err(b1, b2):
    d = abs(heading(b1) - heading(b2))
    return min(d, np.pi - d)


def center_err(b1, b2):
    return float(np.hypot(b1[0]-b2[0], b1[1]-b2[1]))


def greedy_match_per_class(gt, pr, c):
    """Return dict gt_idx -> {'j': pred_idx, 'iou': float} (one-to-one, desc score)."""
    gids = np.where(gt['labels'] == c)[0]
    dids = np.where(pr['labels'] == c)[0]
    if len(gids) == 0 or len(dids) == 0:
        return {}
    order = sorted(dids, key=lambda j: -pr['scores'][j])
    gb = gt['tensor']; pb = pr['tensor']
    assigned = set()
    m = {}
    for j in order:
        best, best_iou = -1, 0.0
        for g in gids:
            if g in assigned:
                continue
            iou = oriented_3d_iou(gb[g], pb[j])
            if iou > best_iou:
                best_iou, best = iou, g
        if best >= 0:
            assigned.add(best)
            m[best] = {'j': int(j), 'iou': float(best_iou)}
    return m


def load(p):
    with open(p, 'rb') as f:
        return pickle.load(f)


def binned(vals, name, edges, dump_percentiles=False):
    h, _ = np.histogram(vals, bins=edges)
    tot = len(vals) or 1
    parts = [f'{name}: n={len(vals)}']
    for i in range(len(edges) - 1):
        parts.append(f' [{edges[i]:.2f},{edges[i+1]:.2f})={h[i]} ({h[i]/tot:.3f})')
    print('  ' + '  '.join(parts))


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

    # ---- A. global Ped confidence statistics (all anchors, across all samples)
    ped_conf_all = []
    max_ped_conf = []     # per sample
    for d in dumps:
        for i in range(N):
            pr = d['preds'][i]
            is_ped = pr['labels'] == PED
            if is_ped.sum():
                sv = pr['scores'][is_ped]
                ped_conf_all.extend(sv.tolist())
                max_ped_conf.append(float(sv.max()))
            else:
                max_ped_conf.append(0.0)
    ped_conf_all = np.array(ped_conf_all)
    max_ped_conf = np.array(max_ped_conf)
    print('=' * 80)
    print('A. Pedestrian class-score distribution (sigmoid; all decoded Ped anchors)')
    print('=' * 80)
    print(f'  count={len(ped_conf_all)}, mean={ped_conf_all.mean():.4f}, '
          f'p50={np.median(ped_conf_all):.4f}, p90={np.percentile(ped_conf_all,90):.4f}, '
          f'max={ped_conf_all.max():.4f}')
    print('  per-sample max Ped score: '
          f'mean={max_ped_conf.mean():.4f}, p50={np.median(max_ped_conf):.4f}, '
          f'p90={np.percentile(max_ped_conf,90):.4f}, max={max_ped_conf.max():.4f}')
    print('  fraction of samples with max Ped score < 0.1 / <0.25 / <0.5: '
          f'{float((max_ped_conf<0.1).mean()):.3f} / '
          f'{float((max_ped_conf<0.25).mean()):.3f} / '
          f'{float((max_ped_conf<0.5).mean()):.3f}')

    # ---- B/C/D/E/F: per GT Pedestrian
    # B: lambda-recall (top-dir/heading-agnostic 'does any Ped anchor overlap GT')
    lambdas = [0.1, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.85]
    lam_hits = {lam: 0 for lam in lambdas}
    ped_gt_count = 0
    # C: matched-score margin + binned deltas (correct-recalled Ped)
    matched_margins = []
    matched_err = {'center': [], 'w': [], 'l': [], 'h': [], 'yaw': []}
    # D: component tolerance pass
    comp = {k: 0 for k in ['center<=0.15', 'w<=0.15', 'l<=0.3', 'h<=0.25', 'yaw<=0.25']}
    # E: recall by distance / ped size
    drec = {}
    srec = {}
    # F: root causes for 0.25<=iou<0.5
    cause = {'center': 0, 'w': 0, 'l': 0, 'h': 0, 'yaw': 0, 'comb': 0}
    loose_ok_strict_fail = 0

    d_buck = [0, 20, 40, 60, 80, 100, 1e9]
    d_name = ['0-20', '20-40', '40-60', '60-80', '80-100', '100+']
    # Ped long-axis (l, GT) buckets; Ped l realistically 0.4..~1.2m
    s_buck = [0, 0.55, 0.7, 0.85, 1.0, 1.4, 1e9]
    s_name = ['<0.55', '0.55-0.7', '0.7-0.85', '0.85-1.0', '1.0-1.4', '1.4+']

    def bidx(v, buck):
        for k in range(len(buck) - 1):
            if buck[k] <= v < buck[k+1]:
                return k
        return len(buck) - 2

    for i in range(N):
        for d in dumps:
            gt = d['gts'][i]
            pr = d['preds'][i]
            gids = np.where(gt['labels'] == PED)[0]
            if len(gids) == 0:
                continue
            gb = gt['tensor']; pb = pr['tensor']

            # B: for each GT, max IoU vs ALL Ped anchors (heading-agnostic)
            dids = np.where(pr['labels'] == PED)[0]
            if len(dids):
                for g in gids:
                    ped_gt_count += 1
                    ious = np.array([oriented_3d_iou(gb[g], pb[j]) for j in dids])
                    for lam in lambdas:
                        if ious.max() >= lam:
                            lam_hits[lam] += 1
            else:
                ped_gt_count += len(gids)

            # greedy same-class match for C/D/E/F
            match = greedy_match_per_class(gt, pr, PED)
            for g, mm in match.items():
                j, iou = mm['j'], mm['iou']
                gt_box, pr_box = gb[g], pb[j]
                pr_boxes_ped = pb[pr['labels'] == PED]
                pr_scores_ped = pr['scores'][pr['labels'] == PED]
                j_local = int(np.where(pr['labels'] == PED)[0].tolist().index(j))
                own = pr_scores_ped[j_local]
                margin = 1.0
                if len(pr_scores_ped) > 1:
                    others = pr_scores_ped.copy()
                    others = np.delete(others, j_local)
                    margin = float(own - others.max())
                matched_margins.append((own, margin))

                ce = center_err(pr_box, gt_box)
                dw = abs(pr_box[3] - gt_box[3])
                dl = abs(pr_box[4] - gt_box[4])
                dh = abs(pr_box[5] - gt_box[5])
                ye = orient_err(pr_box, gt_box)
                matched_err['center'].append(ce)
                matched_err['w'].append(dw)
                matched_err['l'].append(dl)
                matched_err['h'].append(dh)
                matched_err['yaw'].append(ye)

                # D: component tolerance (long/short made explicit via w/l)
                if ce <= 0.15: comp['center<=0.15'] += 1
                if dw <= 0.15: comp['w<=0.15'] += 1
                if dl <= 0.3:  comp['l<=0.3'] += 1
                if dh <= 0.25: comp['h<=0.25'] += 1
                if ye <= 0.25: comp['yaw<=0.25'] += 1

                # E buckets
                dk = (bidx(float(np.hypot(gt_box[0], gt_box[1])), d_buck))
                sk = (bidx(float(max(gt_box[3], gt_box[4])), s_buck))
                for table, key in [(drec, dk), (srec, sk)]:
                    table.setdefault(key, [0, 0, 0])
                    table[key][0] += 1
                    if iou >= LOOSE: table[key][1] += 1
                    if iou >= STRICT: table[key][2] += 1

                # F decomposition
                if LOOSE <= iou < STRICT:
                    loose_ok_strict_fail += 1
                    flags = []
                    if ce > 0.15: flags.append('center')
                    if dw > 0.15: flags.append('w')
                    if dl > 0.3: flags.append('l')
                    if dh > 0.25: flags.append('h')
                    if ye > 0.25: flags.append('yaw')
                    cause[flags[0] if len(flags) == 1 else 'comb'] += 1

    print()
    print('=' * 80)
    print(f'B. Pedestrian lambda-recall@t (top-dir GT Penetration; GT count={ped_gt_count})')
    print('    "any Ped anchor (all 6 dirs) overlaps GT BEV at IoU>=t"; '
          'isolates anchor/dir capacity, NOT score')
    print('=' * 80)
    line = '  '
    for lam in lambdas:
        line += f' @{lam:.2f}={lam_hits[lam]/max(ped_gt_count,1):.3f}'
    print(line)
    print('  (no-classifier-cap: same values would be 1.0 if anchor geometry were perfect)')

    print()
    print('=' * 80)
    print('C. Correct-recalled Ped matches: score OWN vs MARGIN over next Ped anchor')
    print('=' * 80)
    if matched_margins:
        mm = np.array(matched_margins)
        print(f'  n={len(mm)}')
        print(f'  own    mean={mm[:,0].mean():.4f} p50={np.median(mm[:,0]):.4f} '
              f'p90={np.percentile(mm[:,0],90):.4f}')
        print(f'  margin mean={mm[:,1].mean():.4f} p50={np.median(mm[:,1]):.4f} '
              f'p90={np.percentile(mm[:,1],90):.4f}  (neg=anchor dump confuses class/dir)')
        print(f'  own<0.5: {float((mm[:,0]<0.5).mean()):.3f}   own<0.25: {float((mm[:,0]<0.25).mean()):.3f}')
        print(f'  margin<0.05: {float((mm[:,1]<0.05).mean()):.3f}')
    else:
        print('  (no correct Ped matches)')

    print()
    print('=' * 80)
    print('D. Per-component 7D error on correct-recalled Ped (binned, meters / rad)')
    print('=' * 80)
    label = {'center': 'center_bev', 'w': 'w(across)', 'l': 'l(along)',
             'h': 'h(height)', 'yaw': 'yaw'}
    edges = {
        'center': [0, 0.1, 0.2, 0.35, 0.5, 0.8, 1.2, 100],
        'w':      [0, 0.1, 0.15, 0.25, 0.4, 0.6, 1.0, 100],
        'l':      [0, 0.15, 0.3, 0.5, 0.75, 1.1, 1.6, 100],
        'h':      [0, 0.15, 0.25, 0.4, 0.6, 0.9, 1.3, 100],
        'yaw':    [0, 0.1, 0.25, 0.4, 0.6, 0.9, 1.4, np.pi/2 + 1e-6],
    }
    for k in ['center', 'w', 'l', 'h', 'yaw']:
        v = np.array(matched_err[k])
        if len(v):
            print(f'  {label[k]:>12}: mean={v.mean():.3f} median={np.median(v):.3f}')
            binned(v, ' ' * 12, edges[k])
        else:
            print(f'  {label[k]:>12}: (no matches)')

    print()
    print('=' * 80)
    print('D2. Component tolerance pass rate (correct-recalled Ped)')
    print('=' * 80)
    den = max(len(matched_err['center']), 1)
    for k, name in [('center', 'center<=0.15m'), ('w', 'w<=0.15m'), ('l', 'l<=0.30m'),
                    ('h', 'h<=0.25m'), ('yaw', 'yaw<=0.25rad')]:
        key = {'center': 'center<=0.15', 'w': 'w<=0.15', 'l': 'l<=0.3',
               'h': 'h<=0.25', 'yaw': 'yaw<=0.25'}[k]
        print(f'  {name}: {comp[key]}/{den-1 if den>1 and len(matched_err[k])==0 else len(matched_err[k])} = {comp[key]/max(len(matched_err[k]),1):.3f}')

    print()
    print('=' * 80)
    print('E. Pedestrian recall: loose@0.25 / strict@0.5, by distance & Ped long-axis size')
    print('=' * 80)
    for dk in range(len(d_name)):
        gt, l, s = drec.get(dk, (0, 0, 0))
        print(f'  dist {d_name[dk]:>7}: gt={gt:>4}  loose={l/max(gt,1):.3f}  strict={s/max(gt,1):.3f}')
    print()
    for sk in range(len(s_name)):
        gt, l, s = srec.get(sk, (0, 0, 0))
        print(f'  size {s_name[sk]:>10}: gt={gt:>4}  loose={l/max(gt,1):.3f}  strict={s/max(gt,1):.3f}')

    print()
    print('=' * 80)
    print('F. Root-cause decomposition of Ped 0.25<=IoU<0.5 (loose-ok, strict-fail)')
    print('=' * 80)
    tot = max(loose_ok_strict_fail, 1)
    for k in ['center', 'w', 'l', 'h', 'yaw', 'comb']:
        print(f'  {k:>8}: {cause[k]:>4}  ({cause[k]/tot:.3f})')
    print(f'  {"total":>8}: {loose_ok_strict_fail}')


if __name__ == '__main__':
    main()
