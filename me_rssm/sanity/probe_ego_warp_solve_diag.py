"""Root-cause diagnostic for the L1 closed-form ego solve (read-only).

Triggered by probe_ego_warp_sanity.py's first run (smoke iter_450): the
occupancy cross-correlation shift implied ~1.6 m/frame (~16 m/s at the
nominal 0.1 s), while the Doppler closed form returned only 3-4 m/s; and
docs/08 F1 directly measured mean |radial velocity| of 7-13 m/s on the
SAME val frame (idx 3), which is only compatible with |v_ego| ~ 11-20 m/s.
So the closed form is the suspect.

This script prints, per val frame:
  1. raw channel percentiles over measured cells (v_mean, |v_mean|, v_std,
     snr, count) -- settles channel semantics/scale in one look;
  2. the solve under several SELECTIONS/WEIGHTINGS, all the same closed
     form, to separate the estimator from the static-set rule:
       A: pre-registered rule (tau_s/tau_v at static_q over measured,
          w = count*snr)                         <- the shipped rule
       B: same set, unweighted (w = 1)
       C: high-|v_r| band only (|v_r| >= q50 of measured), weighted
       D: RANSAC-lite: fit on |v_r| >= q90, then refit including cells
          whose residual against that fit is <= 2 m/s
       E: 1-param regression of -v_r on u_x over |v_r| >= q75
  3. the occupancy cross-correlation shift (imported from
     probe_ego_warp_sanity) for the same frame pair.

Reading guide: if A << C/D/E and mean |v_mean| sits at 7-13 m/s (docs/08
F1), the estimator is fine and the STATIC-SET RULE is the bug: the
low-|v_r| population it keeps is a mix of off-axis static cells (small
|cos theta|, uninformative but harmless) and moving objects with small
relative radial speed (v_r ~ 0, actively wrong), and the count*SNR weight
lets close high-RCS movers pull the fit toward zero.

Usage:
  python me_rssm/sanity/probe_ego_warp_solve_diag.py \
    --config <dcrssm2 config> --checkpoint <ckpt> --index 3 \
    --output <work_dir>/ego_solve_diag_idx3.json
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
from me_rssm.sanity.probe_ego_warp_sanity import (  # noqa: E402
    occupancy_profiles, occ_shift)
from tools.diagnose_z_utilization import build_batch, run_sequence  # noqa: E402

NOMINAL_DT = 0.1
RESID_TOL = 2.0  # m/s, RANSAC-lite inlier band (docs/08 F1 flow scale)


def geom(pc_range, H, W):
    x_min, y_min, _, x_max, y_max, _ = pc_range
    cell_x = (x_max - x_min) / H
    cell_y = (y_max - y_min) / W
    xs = x_min + (torch.arange(H) + 0.5) * cell_x
    ys = y_min + (torch.arange(W) + 0.5) * cell_y
    X = xs[:, None].expand(H, W)
    Y = ys[None, :].expand(H, W)
    r = torch.sqrt(X * X + Y * Y).clamp(min=1e-6)
    return X / r, Y / r


def wls(ux, uy, b_r, w, min_cells):
    """Weighted closed form (identical math to the module)."""
    if int(w.numel()) < min_cells:
        return None
    w = w / w.sum().clamp(min=1e-6)
    m_xx = (w * ux * ux).sum()
    m_xy = (w * ux * uy).sum()
    m_yy = (w * uy * uy).sum()
    r_x = (w * b_r * ux).sum()
    r_y = (w * b_r * uy).sum()
    det = m_xx * m_yy - m_xy * m_xy
    if float(det.abs()) < 1e-6:
        return None
    vx = (-r_x * m_yy + m_xy * r_y) / det
    vy = (-m_xx * r_y + m_xy * r_x) / det
    return float(vx), float(vy), int(w.numel())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--index', type=int, default=3)
    ap.add_argument('--static-q', type=float, default=None)
    ap.add_argument('--output', required=True)
    a = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cfg = Config.fromfile(a.config)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model['meta_info'] = {
        'figures_path': '/tmp/r4det_ego_solve_diag_figs',
        'project_name': 'ego-solve-diag',
    }
    os.makedirs(cfg.model['meta_info']['figures_path'], exist_ok=True)
    cfg.data.val.pop('samples_per_gpu', None)
    cfg.data.val.pop('workers_per_gpu', None)
    dataset = build_dataset(cfg.data.val)
    model = build_detector(cfg.model, train_cfg=None, test_cfg=None)
    load_checkpoint(model, a.checkpoint, map_location='cpu')
    model = model.to(device).eval()
    fusion = model.temporal_fusion
    pc_range = list(map(float, cfg.model.temporal_fusion['pc_range']))
    static_q = (float(cfg.model.temporal_fusion['static_q'])
                if a.static_q is None else a.static_q)
    min_cells = int(cfg.model.temporal_fusion['min_static_cells'])
    print(f'pc_range={pc_range} static_q={static_q} '
          f'min_static_cells={min_cells}')

    canvases = []
    handle = fusion.register_forward_hook(
        lambda m, i, o: canvases.append(
            None if ModalityContext.get('velocity_bev') is None
            else ModalityContext.get('velocity_bev').detach().float().cpu()))
    batch = build_batch(dataset, [a.index], 0)
    with torch.no_grad():
        run_sequence(model, batch, z_mode='normal')
    handle.remove()
    print(f'captured {len(canvases)} frame canvases for val index {a.index}')

    report = dict(index=a.index, frames=[])
    occ_per_frame = []
    for t, raw in enumerate(canvases):
        if raw is None:
            continue
        B, C, ny, nx = raw.shape
        H, W = nx // 2, ny // 2
        vel = torch.nn.functional.avg_pool2d(raw, 2)
        vel = vel.permute(0, 1, 3, 2).contiguous()[0].double()
        ux, uy = geom(pc_range, H, W)
        count = vel[_CH_COUNT]
        v_mean = vel[_CH_V_MEAN]
        speed = vel[_CH_SPEED]
        v_std = vel[_CH_V_STD]
        snr = vel[_CH_SNR]
        m = count > 0
        occ_per_frame.append((m.numpy(), H, W))

        def pct(x):
            x = x.detach().cpu().numpy()
            return {f'p{p}': float(np.percentile(x, p))
                    for p in (10, 50, 90, 98, 100)}

        chans = dict(
            v_mean_over_measured=pct(v_mean[m]),
            abs_v_mean_over_measured=pct(speed[m]),
            abs_v_mean_all=pct(speed),
            v_std_over_measured=pct(v_std[m]),
            snr_over_measured=pct(snr[m]),
            count_over_measured=pct(count[m]),
            measured_cells=int(m.sum().item()),
            measured_frac=float(m.float().mean()),
        )
        print()
        print(f'--- frame {t} (idx {a.index}) measured={chans["measured_cells"]}'
              f' ({100 * chans["measured_frac"]:.2f}% of canvas)')
        print(f'  |v_mean| over measured : {chans["abs_v_mean_over_measured"]}')
        print(f'  v_mean  over measured  : {chans["v_mean_over_measured"]}')
        print(f'  v_std   over measured  : {chans["v_std_over_measured"]}')
        print(f'  snr     over measured  : {chans["snr_over_measured"]}')
        print(f'  count   over measured  : {chans["count_over_measured"]}')

        n_meas = int(m.sum().item())
        variants = {}
        if n_meas >= min_cells:
            tau_s = torch.quantile(v_std[m], static_q)
            tau_v = torch.quantile(speed[m], static_q)
            sel_A = m & (v_std <= tau_s) & (speed <= tau_v)
            if int(sel_A.sum().item()) >= min_cells:
                uxs, uys, brs, ws = ux[sel_A], uy[sel_A], v_mean[sel_A], \
                    (count * snr)[sel_A]
                variants['A_prereg_weighted'] = wls(uxs, uys, brs, ws, min_cells)
                variants['B_prereg_unweighted'] = wls(
                    uxs, uys, brs, torch.ones_like(ws), min_cells)
            for q in (0.5, 0.75, 0.9):
                thr = torch.quantile(speed[m], q)
                sel_q = m & (speed >= thr)
                if int(sel_q.sum().item()) >= min_cells:
                    variants[f'C_hi_absvr_q{int(q * 100)}_weighted'] = wls(
                        ux[sel_q], uy[sel_q], v_mean[sel_q],
                        (count * snr)[sel_q], min_cells)
            # D: RANSAC-lite seeded on the top band
            thr90 = torch.quantile(speed[m], 0.9)
            seed = m & (speed >= thr90)
            v_seed = wls(ux[seed], uy[seed], v_mean[seed],
                         (count * snr)[seed], min_cells)
            if v_seed is not None:
                vx, vy = v_seed[0], v_seed[1]
                resid = (v_mean + vx * ux + vy * uy).abs()
                inl = m & (resid <= RESID_TOL)
                v_ref = wls(ux[inl], uy[inl], v_mean[inl],
                            (count * snr)[inl], min_cells)
                if v_ref is not None:
                    variants['D_ransac_lite'] = v_ref
                    variants['D_seed'] = v_seed
                    variants['D_inliers'] = int(inl.sum().item())
            # E: 1-param regression on the top band
            thr75 = torch.quantile(speed[m], 0.75)
            sel_e = m & (speed >= thr75)
            if int(sel_e.sum().item()) >= min_cells:
                w_e = (count * snr)[sel_e]
                w_e = w_e / w_e.sum().clamp(min=1e-6)
                uxx, brr = ux[sel_e], v_mean[sel_e]
                variants['E_1param_topband'] = (
                    float((w_e * (-brr) * uxx).sum()
                          / (w_e * uxx * uxx).sum().clamp(min=1e-6)), 0.0,
                    int(sel_e.sum().item()))
        for k, v in variants.items():
            if v is None:
                print(f'  {k:26s}: n/a')
            elif isinstance(v, int):
                print(f'  {k:26s}: {v} cells')
            else:
                sp = (v[0] ** 2 + v[1] ** 2) ** 0.5
                print(f'  {k:26s}: vx={v[0]:+7.3f} vy={v[1]:+7.3f} '
                      f'|v|={sp:6.3f} m/s  (n={v[2]})')
        rec = dict(frame=t, channels=chans,
                   variants={k: (None if v is None or isinstance(v, int) else v)
                             for k, v in variants.items()})
        report['frames'].append(rec)

    x0, x1, y0, y1, cell_x = occupancy_profiles(pc_range, H, W)
    print(f'\noccupancy x band cells [{x0},{x1}) y band [{y0},{y1}) '
          f'cell_x={cell_x:.3f} m')
    for t in range(1, len(occ_per_frame)):
        occ_a, _, _ = occ_per_frame[t - 1]
        occ_b, _, _ = occ_per_frame[t]
        res = occ_shift(occ_a, occ_b, x0, x1, y0, y1)
        if res is None:
            continue
        s_ref, peak, ratio = res
        print(f'  frames {t - 1}->{t}: shift {s_ref:+.2f} cells = '
              f'{s_ref * cell_x:+.3f} m/frame -> '
              f'{s_ref * cell_x / NOMINAL_DT:+.2f} m/s at dt={NOMINAL_DT}; '
              f'peak ratio {ratio:.1f}')
        report['frames'][t]['occ_shift'] = dict(
            cells=s_ref, m_per_frame=s_ref * cell_x,
            speed_mps=s_ref * cell_x / NOMINAL_DT, peak_ratio=ratio)

    with open(a.output, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=1)
    print(f'\nwrote {a.output}')


if __name__ == '__main__':
    main()
