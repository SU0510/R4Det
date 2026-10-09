"""Ego-warp sanity probe (read-only, zero training).

Verifies DC-RSSM-2's L1 geometry wiring by measuring the closed form next to
a Doppler-free cross-check on the same val frames:

  1. Doppler closed form (the module's own solve, mirrored here for the
     diagnostics; robust path by default). The mirror is asserted equal to
     the module's stashed solution, so any divergence below is real.
  2. Occupancy cross-correlation shift between consecutive frames.
     KEPT AS A NEGATIVE CONTROL, not as ground truth: the val canvas is
     only 0.3-1% occupied (~600 cells), so the correlation is dominated by
     a handful of strong clusters (oncoming traffic) instead of the static
     background -- measured peak/background ratios of 1.9-2.1 and shifts
     implying 9-16 m/s at dt=0.1 s while the robust Doppler solve says
     1.4-3.0 m/s and the raw |v_r| percentiles agree with the latter. The
     peak_ratio column is printed so this unreliability is visible; do NOT
     read the occ columns as ego speed on this dataset.

The reliable geometry evidence for this layer lives in
probe_ego_geometry_test.py: the LMedS robust solve over the whole measured
set explains 58-80% of cells with a 0.013 m/s median residual at a single
rigid velocity, which is only possible if canvas<->world mapping, cell
size and sign conventions are all right (8 mapping hypotheses were tested:
all within 10% residual RMS, and the assumed one is the best).

Also reports the static-set health that the 450-iter gate needs:
  measured_frac (count > 0 over the whole canvas),
  sel_over_measured = |quantile pre-filter| / |measured| (~0.8 by
  construction at static_q = 0.9), and
  trim_frac = |MAD-trimmed set| / |pre-filtered set| (the robust stage).

Usage:
  python me_rssm/sanity/probe_ego_warp_sanity.py \
    --config me_rssm/configs/..._dcrssm2.py --checkpoint <iter_N.pth> \
    --start-index 3 --limit 4 --stride 700 \
    --output <work_dir>/ego_warp_sanity_ep16.json
"""

import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings('ignore')
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from mmcv import Config  # noqa: E402
from mmcv.runner import load_checkpoint  # noqa: E402
from mmdet3d.datasets import build_dataset  # noqa: E402
from mmdet3d.models import build_detector  # noqa: E402

from me_rssm.ego_warp_rssm import (  # noqa: E402
    _CH_COUNT, _CH_SNR, _CH_SPEED, _CH_V_MEAN, _CH_V_STD)
from me_rssm.modality_bus import ModalityContext  # noqa: E402
from tools.diagnose_z_utilization import (  # noqa: E402
    build_batch, history_validity, run_sequence)

SHIFT_RANGE = 40          # +/- cells at 0.32 m -> +/- 12.8 m per frame
X_BAND_M = (8.0, 60.0)    # strong static structure, away from the ego pole
Y_BAND_M = (-20.0, 20.0)
NOMINAL_DT = 0.1          # TJ4D frame interval assumption (no timestamps)


def mirror_solve(vel_feat, pc_range, static_q, min_static_cells,
                 robust=True):
    """Mirror of the module's closed form, for diagnostics only.

    Same math as EgoWarpEvidenceFieldRSSMFusion._solve_ego (robust path
    by default: quantile pre-filter -> LMedS seed -> MAD trim -> weighted
    closed form); the caller asserts agreement with the module's stashed
    v_ego, so any divergence in the diagnostics below is real rather than
    a re-implementation artifact. With robust=False it reproduces the
    pre-fix plain weighted fit, which the geometry probe showed inflated
    by +24%..+96% on val frames.
    """
    B, _, H, W = vel_feat.shape
    x_min, y_min, _, x_max, y_max, _ = pc_range
    cell_x = (x_max - x_min) / H
    cell_y = (y_max - y_min) / W
    xs = x_min + (torch.arange(H, device=vel_feat.device) + 0.5) * cell_x
    ys = y_min + (torch.arange(W, device=vel_feat.device) + 0.5) * cell_y
    X = xs[:, None].expand(H, W)
    Y = ys[None, :].expand(H, W)
    r = torch.sqrt(X * X + Y * Y).clamp(min=1e-6)
    ux, uy = X / r, Y / r

    count = vel_feat[:, _CH_COUNT]
    v_mean = vel_feat[:, _CH_V_MEAN]
    speed = vel_feat[:, _CH_SPEED]
    v_std = vel_feat[:, _CH_V_STD]
    snr = vel_feat[:, _CH_SNR]
    measured = count > 0

    def wls(u_x, u_y, b_r, w):
        w = w / w.sum().clamp(min=1e-6)
        m_xx = (w * u_x * u_x).sum()
        m_xy = (w * u_x * u_y).sum()
        m_yy = (w * u_y * u_y).sum()
        r_x = (w * b_r * u_x).sum()
        r_y = (w * b_r * u_y).sum()
        det = m_xx * m_yy - m_xy * m_xy
        if float(det.abs()) < 1e-6:
            return None
        return [float((-r_x * m_yy + m_xy * r_y) / det),
                float((-m_xx * r_y + m_xy * r_x) / det)]

    out = []
    for b in range(B):
        m = measured[b]
        rec = dict(measured=int(m.sum().item()), sel=0, solved=False,
                   v_ego=[0.0, 0.0], trim_frac=0.0, tau_r=0.0,
                   static_q_used=static_q)
        if rec['measured'] >= min_static_cells:
            tau_s = torch.quantile(v_std[b][m], static_q)
            tau_v = torch.quantile(speed[b][m], static_q)
            sel0 = m & (v_std[b] <= tau_s) & (speed[b] <= tau_v)
            rec['sel'] = int(sel0.sum().item())
            if rec['sel'] >= min_static_cells:
                u_x, u_y = ux[sel0], uy[sel0]
                b_r = v_mean[b][sel0]
                w = (count[b] * snr[b])[sel0]
                if robust:
                    seed = torch.arange(-25.0, 25.001, 0.25)
                    r1 = (b_r[:, None] + seed[None, :] * u_x[:, None]).abs()
                    vx_seed = float(seed[r1.median(dim=0).values.argmin()])
                    lvx = torch.arange(vx_seed - 0.6, vx_seed + 0.601, 0.05)
                    lvy = torch.arange(-3.0, 3.001, 0.15)
                    r2 = (b_r[None, None, :]
                          + lvx[:, None, None] * u_x[None, None, :]
                          + lvy[None, :, None] * u_y[None, None, :]).abs()
                    med2 = r2.median(dim=-1).values
                    ix, iy = divmod(int(med2.argmin().item()),
                                    med2.shape[1])
                    resid = (b_r + float(lvx[ix]) * u_x
                             + float(lvy[iy]) * u_y).abs()
                    mad = float(resid.median())
                    tau_r = min(max(3.0 * 1.4826 * mad, 0.2), 1.0)
                    keep = resid <= tau_r
                    rec['tau_r'] = tau_r
                    if int(keep.sum().item()) >= min_static_cells:
                        rec['trim_frac'] = float(keep.float().mean())
                        u_x, u_y, b_r, w = (u_x[keep], u_y[keep],
                                            b_r[keep], w[keep])
                v = wls(u_x, u_y, b_r, w)
                if v is not None:
                    rec['v_ego'] = v
                    rec['solved'] = True
        out.append(rec)
    return out


def occupancy_profiles(pc_range, H, W):
    """Cell index bands for the correlation (x along dim2, y along dim3)."""
    x_min, y_min, _, x_max, y_max, _ = pc_range
    cell_x = (x_max - x_min) / H
    cell_y = (y_max - y_min) / W
    x0 = max(0, int((X_BAND_M[0] - x_min) / cell_x))
    x1 = min(H, int((X_BAND_M[1] - x_min) / cell_x))
    y0 = max(0, int((Y_BAND_M[0] - y_min) / cell_y))
    y1 = min(W, int((Y_BAND_M[1] - y_min) / cell_y))
    return x0, x1, y0, y1, cell_x


def occ_shift(occ_a, occ_b, x0, x1, y0, y1):
    """Sub-cell x shift maximising sum occ_a[x + s, y] * occ_b[x, y].

    Returns (shift_cells, peak_value, peak_ratio) with a parabola fit;
    shift > 0 means content travelled to LARGER x indices in the ego frame
    (i.e. s = v_ego * dt / cell, forward motion positive).
    """
    a = occ_a[x0:x1, y0:y1]
    scores = {}
    for s in range(-SHIFT_RANGE, SHIFT_RANGE + 1):
        lo_a, hi_a = x0 + s, x1 + s
        if lo_a < 0 or hi_a > occ_a.shape[0]:
            continue
        scores[s] = float((occ_a[lo_a:hi_a, y0:y1] * occ_b[x0:x1, y0:y1]).sum())
    if not scores:
        return None
    s_star = max(scores, key=scores.get)
    peak = scores[s_star]
    other = np.mean([v for k, v in scores.items() if abs(k - s_star) > 2]) \
        if len(scores) > 5 else 0.0
    if (s_star - 1) in scores and (s_star + 1) in scores:
        y0v, y1v, y2v = scores[s_star - 1], scores[s_star], scores[s_star + 1]
        denom = (y0v - 2 * y1v + y2v)
        frac = 0.5 * (y0v - y2v) / denom if abs(denom) > 1e-12 else 0.0
        frac = float(np.clip(frac, -0.5, 0.5))
        s_refined = s_star + frac
    else:
        s_refined = float(s_star)
    return s_refined, peak, (peak / other if other > 1e-9 else float('inf'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--start-index', type=int, default=3)
    ap.add_argument('--limit', type=int, default=4)
    ap.add_argument('--stride', type=int, default=700)
    ap.add_argument('--min-valid-history', type=int, default=2)
    ap.add_argument('--output', required=True)
    a = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cfg = Config.fromfile(a.config)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model['meta_info'] = {
        'figures_path': '/tmp/r4det_ego_warp_probe_figs',
        'project_name': 'ego-warp-sanity-probe',
    }
    os.makedirs(cfg.model['meta_info']['figures_path'], exist_ok=True)
    cfg.data.val.pop('samples_per_gpu', None)
    cfg.data.val.pop('workers_per_gpu', None)
    dataset = build_dataset(cfg.data.val)
    model = build_detector(cfg.model, train_cfg=None, test_cfg=None)
    load_checkpoint(model, a.checkpoint, map_location='cpu')
    model = model.to(device).eval()
    fusion = model.temporal_fusion
    assert getattr(fusion, 'use_ego_warp', False), 'not an ego-warp variant'
    pc_range = list(map(float, cfg.model.temporal_fusion['pc_range']))
    static_q = float(cfg.model.temporal_fusion['static_q'])
    min_cells = int(cfg.model.temporal_fusion['min_static_cells'])
    print(f'config temporal_fusion: {cfg.model.temporal_fusion["type"]} '
          f'pc_range={pc_range} static_q={static_q} '
          f'min_static_cells={min_cells}')
    print(f'learned scalars: warp_scale={float(fusion.warp_scale):.6f} '
          f'dt={float(fusion.dt):.6f}')

    captures = []

    def hook(module, inputs, output):
        canvas = ModalityContext.get('velocity_bev')
        solve = module._ego_solve
        captures.append(dict(
            canvas=None if canvas is None else canvas.detach().float().cpu(),
            v_ego=None if solve is None else solve['v_ego'].detach().cpu(),
            solved=None if solve is None else solve['solved'].detach().cpu()))

    handle = fusion.register_forward_hook(hook)

    def select_indices():
        idx, cursor = [], a.start_index
        while cursor < len(dataset) and len(idx) < a.limit:
            mask = history_validity(dataset, cursor)
            if mask is not None and sum(mask[:-1]) >= a.min_valid_history:
                idx.append(cursor)
                cursor += a.stride
            else:
                cursor += 1
        if not idx:
            raise ValueError('no valid sample selected')
        return idx

    sequences = []
    for index in select_indices():
        batch = build_batch(dataset, [index], 0)
        captures.clear()
        with torch.no_grad():
            run_sequence(model, batch, z_mode='normal')
        seq = []
        for t, cap in enumerate(captures):
            if cap['canvas'] is None:
                seq.append(dict(frame=t, bus_missing=True))
                continue
            raw = cap['canvas']  # (B, 7, ny=2W, nx=2H)
            B, C, ny, nx = raw.shape
            H, W = nx // 2, ny // 2
            vel_feat = torch.nn.functional.avg_pool2d(raw, 2)
            vel_feat = vel_feat.permute(0, 1, 3, 2).contiguous()  # (B,C,H,W)
            mir = mirror_solve(vel_feat, pc_range, static_q, min_cells)
            b = 0
            if cap['v_ego'] is not None:
                dv = (torch.tensor(mir[b]['v_ego']) - cap['v_ego'][b]).abs().max()
                assert float(dv) < 1e-4, (
                    f'mirror solve diverged from module: {float(dv):.3e}')
                assert bool(cap['solved'][b]) == mir[b]['solved']
            occ = (vel_feat[b, _CH_COUNT] > 0).numpy()
            rec = dict(
                frame=t,
                canvas_shape=[int(C), int(ny), int(nx)],
                measured_frac=mir[b]['measured'] / float(H * W),
                sel_over_measured=(mir[b]['sel'] / mir[b]['measured']
                                   if mir[b]['measured'] else 0.0),
                v_ego=mir[b]['v_ego'],
                v_ego_speed=float(np.linalg.norm(mir[b]['v_ego'])),
                solved=mir[b]['solved'],
                measured_cells=mir[b]['measured'],
                sel_cells=mir[b]['sel'],
                _occ=occ)
            seq.append(rec)
        # independent occupancy-shift speed between consecutive frames
        x0, x1, y0, y1, cell_x = occupancy_profiles(
            pc_range, H, W)
        for t in range(1, len(seq)):
            prev, cur = seq[t - 1], seq[t]
            if '_occ' not in prev or '_occ' not in cur:
                continue
            res = occ_shift(prev['_occ'], cur['_occ'], x0, x1, y0, y1)
            if res is not None:
                s_ref, peak, ratio = res
                cur['occ_shift_cells'] = s_ref
                cur['occ_shift_m_per_frame'] = s_ref * cell_x
                cur['occ_shift_speed_mps'] = s_ref * cell_x / NOMINAL_DT
                cur['occ_peak_ratio'] = ratio
        for rec in seq:
            rec.pop('_occ', None)
        sequences.append(dict(index=index, frames=seq))

    handle.remove()
    print()
    hdr = (f'{"idx":>6} {"f":>2} {"meas%":>6} {"sel/meas":>8} '
           f'{"|v_ego|":>8} {"vx":>8} {"vy":>8} {"occ_m/fr":>9} '
           f'{"occ_mps":>8} {"peakR":>7}')
    print(hdr)
    for s in sequences:
        for rec in s['frames']:
            if rec.get('bus_missing'):
                print(f'{s["index"]:>6} {rec["frame"]:>2}  bus missing')
                continue
            occ_m = rec.get('occ_shift_m_per_frame')
            occ_v = rec.get('occ_shift_speed_mps')
            pr = rec.get('occ_peak_ratio')
            print(f'{s["index"]:>6} {rec["frame"]:>2} '
                  f'{100 * rec["measured_frac"]:>6.2f} '
                  f'{rec["sel_over_measured"]:>8.3f} '
                  f'{rec["v_ego_speed"]:>8.3f} '
                  f'{rec["v_ego"][0]:>8.3f} {rec["v_ego"][1]:>8.3f} '
                  f'{("%9.3f" % occ_m) if occ_m is not None else "        -"} '
                  f'{("%8.3f" % occ_v) if occ_v is not None else "       -"} '
                  f'{("%7.1f" % pr) if pr is not None else "      -"}')

    # aggregate agreement: frames where both estimates are usable
    pairs = [(r['v_ego_speed'], r['occ_shift_speed_mps'])
             for s in sequences for r in s['frames']
             if r.get('occ_shift_speed_mps') is not None
             and r.get('solved')]
    summary = dict(n_pairs=len(pairs))
    if pairs:
        dop = np.array([p[0] for p in pairs])
        occ = np.array([p[1] for p in pairs])
        ratio = occ / np.clip(dop, 1e-3, None)
        summary.update(
            mean_v_ego=float(dop.mean()),
            mean_occ_speed=float(occ.mean()),
            mean_ratio=float(ratio.mean()),
            ratio_std=float(ratio.std()),
            corr=float(np.corrcoef(dop, occ)[0, 1])
            if len(pairs) > 2 and dop.std() > 1e-6 and occ.std() > 1e-6 else None,
            max_abs_v_ego=float(dop.max()),
        )
        print()
        print(f'frames with both estimates: {len(pairs)}')
        print(f'mean |v_ego| (Doppler)  : {dop.mean():.3f} m/s')
        print(f'mean occ-shift speed    : {occ.mean():.3f} m/s')
        print(f'ratio occ/doppler       : {ratio.mean():.3f} '
              f'+/- {ratio.std():.3f}')
        print(f'max |v_ego|             : {dop.max():.3f} m/s')

    payload = dict(config=a.config, checkpoint=a.checkpoint,
                   pc_range=pc_range, static_q=static_q,
                   min_static_cells=min_cells,
                   warp_scale=float(fusion.warp_scale),
                   dt=float(fusion.dt),
                   nominal_dt=NOMINAL_DT,
                   summary=summary, sequences=sequences)
    with open(a.output, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=1)
    print(f'\nwrote {a.output}')


if __name__ == '__main__':
    main()
