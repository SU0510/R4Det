#!/usr/bin/env python3
import argparse, pickle
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
    z1, h1 = b1[2], b1[5]; z2, h2 = b2[2], b2[5]
    top = max(z1-h1/2, z2-h2/2); bot = min(z1+h1/2, z2+h2/2)
    oh = bot - top
    if oh <= 0: return 0.0
    v1 = b1[3]*b1[4]*b1[5]; v2 = b2[3]*b2[4]*b2[5]
    return inter*oh/(v1+v2-inter*oh+1e-9)

def heading(b):
    w,l,yaw = b[3],b[4],b[6]
    return (yaw if l>=w else yaw+np.pi/2)%np.pi

def orient_err(b1,b2):
    d = abs(heading(b1)-heading(b2)); return min(d, np.pi-d)

def center_err(b1,b2):
    return float(np.hypot(b1[0]-b2[0], b1[1]-b2[1]))

def greedy_match(gt, pr, c):
    gids = np.where(gt[1]==c)[0]; dids = np.where(pr[1]==c)[0]
    if len(gids)==0 or len(dids)==0: return {}
    order = sorted(dids, key=lambda j: -pr[2][j])
    gb, pb = gt[0], pr[0]
    assigned=set(); m={}
    for j in order:
        best, best_iou = -1, 0.0
        for g in gids:
            if g in assigned: continue
            iou = oriented_3d_iou(gb[g], pb[j])
            if iou > best_iou: best_iou, best = iou, g
        if best>=0: assigned.add(best); m[best]={'j':int(j),'iou':float(best_iou)}
    return m

def rep(name, vals, edges):
    h,_ = np.histogram(vals, bins=edges); n=len(vals) or 1
    parts=[f'  {name}: n={len(vals)}']
    for i in range(len(edges)-1):
        parts.append(f'  [{edges[i]:.2f},{edges[i+1]:.2f})={h[i]}({h[i]/n:.3f})')
    print('  '.join(parts))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dump', required=True); ap.add_argument('--label', required=True)
    a = ap.parse_args()
    with open(a.dump,'rb') as f: d = pickle.load(f)
    assert list(d['class_names'])==CLASS_NAMES
    preds,gts=d['preds'],d['gts']; N=len(gts)
    ped_gt_total=0; hit={LOOSE:0,STRICT:0}
    err={'center':[],'w':[],'l':[],'h':[],'yaw':[]}
    pred_whl=[]; gt_whl=[]; sc=[]
    db=[(0,20),(20,40),(40,60),(60,80),(80,100),(100,999)]
    dr={k:[0,0,0] for k in db}
    sb=[(0,0.55),(0.55,0.7),(0.7,0.85),(0.85,1.0),(1.0,1.4),(1.4,99)]
    sr={k:[0,0,0] for k in sb}
    cause={'center':0,'w':0,'l':0,'h':0,'yaw':0,'comb':0}; lsf=0
    for i in range(N):
        gt,pr=gts[i],preds[i]
        gn=np.asarray(gt['tensor'],float); gl=np.asarray(gt['labels'],int)
        pn=np.asarray(pr['tensor'],float); pl=np.asarray(pr['labels'],int)
        ps=np.asarray(pr['scores'],float)
        ped_gt_total += int((gl==PED).sum())
        pp = pl==PED
        sc.extend(ps[pp].tolist())
        gt_whl.extend(gn[gl==PED][:,[3,4,5]].tolist())
        pred_whl.extend(pn[pp][:,[3,4,5]].tolist())
        m = greedy_match((gn,gl),(pn,pl,ps),PED)
        for g,info in m.items():
            iou=info['iou']; j=info['j']; pb,gb=pn[j],gn[g]
            if iou>=LOOSE: hit[LOOSE]+=1
            if iou>=STRICT: hit[STRICT]+=1
            ce=center_err(pb,gb); we=abs(pb[3]-gb[3]); le=abs(pb[4]-gb[4]); he=abs(pb[5]-gb[5]); ye=orient_err(pb,gb)
            err['center'].append(ce); err['w'].append(we); err['l'].append(le); err['h'].append(he); err['yaw'].append(ye)
            dist=float(np.hypot(gb[0],gb[1])); size=float(max(gb[3],gb[4]))
            dk=next((kk for kk in db if kk[0]<=dist<kk[1]),db[-1]); sk=next((kk for kk in sb if kk[0]<=size<kk[1]),sb[-1])
            dr[dk][0]+=1; sr[sk][0]+=1
            if iou>=LOOSE: dr[dk][1]+=1; sr[sk][1]+=1
            if iou>=STRICT: dr[dk][2]+=1; sr[sk][2]+=1
            if LOOSE<=iou<STRICT:
                lsf+=1; f=[]
                if ce>0.15:f.append('center')
                if we>0.15:f.append('w')
                if le>0.3:f.append('l')
                if he>0.25:f.append('h')
                if ye>0.25:f.append('yaw')
                cause[f[0] if len(f)==1 else 'comb']+=1
    def q(z):
        z=np.array(z); return f'mean={z.mean():.3f} median={np.median(z):.3f}'
    print('='*78); print(f'[{a.label}] Pedestrain diagnosis samples={N} PedGT={ped_gt_total}'); print('='*78)
    print(f' GT recall@0.25 = {hit[LOOSE]/max(ped_gt_total,1):.4f}')
    print(f' GT recall@0.5  = {hit[STRICT]/max(ped_gt_total,1):.4f}')
    print()
    print('Errors on matched pairs (m / rad):')
    for k,nm in [('center','center_bev'),('w','w'),('l','l'),('h','h'),('yaw','yaw')]:
        print(f'  {nm:>10}: {q(err[k])}')
    print()
    ed={'center':[0,0.1,0.2,0.35,0.5,0.8,1.2,100],'w':[0,0.1,0.15,0.25,0.4,0.6,1.0,100],'l':[0,0.15,0.3,0.5,0.75,1.1,1.6,100],'h':[0,0.15,0.25,0.4,0.6,0.9,1.3,100],'yaw':[0,0.1,0.25,0.4,0.6,0.9,1.4,np.pi/2+1e-6]}
    for k in ['center','w','l','h','yaw']: rep(k,err[k],ed[k])
    print()
    print('Pred vs GT w/l/h distribution (mean/median):')
    for ax,nm in [(0,'w'),(1,'l'),(2,'h')]:
        pv=np.array([r[ax] for r in pred_whl]).reshape(-1); gv=np.array([r[ax] for r in gt_whl]).reshape(-1)
        print(f'  {nm}: pred {q(pv)} | gt {q(gv)}')
    print()
    sc=np.array(sc)
    if len(sc):
        print(f'Ped score distribution: n={len(sc)} mean={sc.mean():.4f} p50={np.median(sc):.4f} p90={np.percentile(sc,90):.4f} max={sc.max():.4f}')
        print(f'  frac<0.1/<0.25/<0.5 = {(sc<0.1).mean():.3f}/{(sc<0.25).mean():.3f}/{(sc<0.5).mean():.3f}')
    print()
    print('Recall L@0.25/S@0.5 by distance:')
    for k in db: print(f'  dist {k[0]:>3}-{k[1]:<3}: gt={dr[k][0]:>4} L={dr[k][1]/max(dr[k][0],1):.3f} S={dr[k][2]/max(dr[k][0],1):.3f}')
    print('Recall L@0.25/S@0.5 by size:')
    for k in sb: print(f'  size {k[0]:>5}-{k[1]:<4}: gt={sr[k][0]:>4} L={sr[k][1]/max(sr[k][0],1):.3f} S={sr[k][2]/max(sr[k][0],1):.3f}')
    print()
    tot=max(lsf,1)
    print(f'Loose-ok strict-fail cause split (total={lsf}):')
    for k in ['center','w','l','h','yaw','comb']: print(f'  {k:>7}: {cause[k]:>4} ({cause[k]/tot:.3f})')
if __name__=='__main__': main()
