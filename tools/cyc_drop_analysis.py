"""Diagnose Cyclist loose-recall drop between baseline and truck_refine.

Answers: did the refine run drop Cyclist detections (count/score) or
localization? Compares per-sample Cyc pred count / score distribution and
loose(0.25)/strict(0.5) GT recall via same-class greedy matching.

Usage:
  python tools/cyc_drop_analysis.py --base base.pkl --new new.pkl [--new2 ...]
"""
import argparse, pickle
import numpy as np
from shapely.geometry import Polygon

SCORE_THR = 0.1  # approximate test score_thr from config (verify)

def bev_corners(b):
    x,y,z,w,l,h,yaw = b
    c,s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c,-s],[s,c]])
    local = np.array([[-l/2,-w/2],[l/2,-w/2],[l/2,w/2],[-l/2,w/2]])
    return local @ R.T + np.array([x,y])

def _poly(c):
    p = Polygon(c); return p.buffer(0) if not p.is_valid else p

def oriented_3d_iou(b1,b2):
    p1=_poly(bev_corners(b1)); p2=_poly(bev_corners(b2))
    ia=p1.intersection(p2).area
    if ia<=1e-12: return 0.0
    z1,h1=b1[2],b1[5]; z2,h2=b2[2],b2[5]
    top=max(z1-h1/2,z2-h2/2); bot=min(z1+h1/2,z2+h2/2)
    oh=bot-top
    if oh<=0: return 0.0
    iv=ia*oh
    v1=b1[3]*b1[4]*b1[5]; v2=b2[3]*b2[4]*b2[5]
    return iv/(v1+v2-iv+1e-9)

def greedy_cyc(gt_np, gt_lab, pr_np, pr_lab, pr_sco):
    # same-class match for Cyc (label 1)
    gids=[i for i in range(len(gt_lab)) if gt_lab[i]==1]
    dids=[j for j in range(len(pr_lab)) if pr_lab[j]==1]
    dids=sorted(dids, key=lambda j:-pr_sco[j])
    m={}; used=set()
    for j in dids:
        best=-1;bestg=None
        for g in gids:
            if g in used: continue
            iou=oriented_3d_iou(gt_np[g], pr_np[j])
            if iou>best: best=iou; bestg=g
        if bestg is not None and best>0:
            used.add(bestg); m[bestg]=(j,best)
    return m, len(gids)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--base',required=True)
    ap.add_argument('--new',nargs='+',required=True)
    args=ap.parse_args()
    base=pickle.load(open(args.base,'rb'))
    news=[pickle.load(open(p,'rb')) for p in args.new]

    def agg(d):
        tot_cyc_gt=0; matched_loose=0; matched_strict=0
        n_pred=[]; scos=[]
        cyc_pred_per_sample=[]
        N=len(d['gts'])
        for i in range(N):
            gt=d['gts'][i]; pr=d['preds'][i]
            gtn=np.asarray(gt['tensor'],float); gl=np.asarray(gt['labels'])
            pn=np.asarray(pr['tensor'],float); pl=np.asarray(pr['labels']); ps=np.asarray(pr['scores'])
            cyc_pred=(pl==1).sum(); cyc_pred_per_sample.append(cyc_pred)
            scos.extend(ps[pl==1].tolist())
            m, ngt = greedy_cyc(gtn,gl,pn,pl,ps)
            tot_cyc_gt+=ngt
            for g,(j,iou) in m.items():
                if iou>=0.25: matched_loose+=1
                if iou>=0.5: matched_strict+=1
        return dict(
            tot_cyc_gt=tot_cyc_gt, matched_loose=matched_loose,
            matched_strict=matched_strict, recall_loose=matched_loose/tot_cyc_gt,
            recall_strict=matched_strict/tot_cyc_gt,
            cyc_pred_per_sample=np.asarray(cyc_pred_per_sample),
            scos=np.asarray(scos))

    def show(name, a):
        print(f'[{name}] Cyc GT={a["tot_cyc_gt"]} '
              f'loose_recall={a["recall_loose"]:.3f} strict_recall={a["recall_strict"]:.3f}')
        print(f'   Cyc pred/sample mean={a["cyc_pred_per_sample"].mean():.2f} '
              f'median={np.median(a["cyc_pred_per_sample"]):.1f} '
              f'#samples_with_0_cyc_pred={(a["cyc_pred_per_sample"]==0).sum()}')
        if len(a['scos'])>0:
            print(f'   Cyc score: mean={a["scos"].mean():.3f} median={np.median(a["scos"]):.3f} '
                  f'frac>0.3={(a["scos"]>0.3).mean():.3f} frac>0.5={(a["scos"]>0.5).mean():.3f}')
        return a

    print('='*70)
    print('Cyclist loose-recall 诊断 (baseline vs truck_refine)')
    print('='*70)
    b=show('base      ', agg(base))
    for p,nm in zip(args.new, ['new(ep20)', 'new(ep24)'][:len(args.new)]):
        x=agg(news[args.new.index(p)])
        show(nm, x)
        print(f'  -> Δloose_recall vs base: {x["recall_loose"]-b["recall_loose"]:+.3f}')
        print(f'  -> Δstrict_recall vs base: {x["recall_strict"]-b["recall_strict"]:+.3f}')

if __name__=='__main__':
    main()
