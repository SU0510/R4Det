"""Canvas dump + geometry-hypothesis test for the L1 ego solve (read-only).

Why this exists: the closed form with the pre-registered static rule returns
4.0-4.3 m/s at val idx 3 (frames 1-3) while (a) the high-|v_r| bands and a
RANSAC-lite refit give 8.3-10.7, (b) raw |v_mean| percentiles over measured
cells are p50 ~ 2.9 / p90 ~ 5.9 / p100 ~ 12 -- impossible for a static field
if the true |v_ego| were 4, and (c) docs/08 F1 independently measured mean
|radial velocity| 7-13 m/s on the same frame. Either the static-set rule
keeps a contaminating low-|v_r| blob (mostly movers with ~0 relative radial
speed + high count*SNR weight) or the CANVAS <-> WORLD MAPPING assumed by
the solve is wrong, which would make U_r(assumed) disagree with each cell's
real bearing and bias the fit.

This script settles both questions offline:
  MODE dump   : run the model on the given val indices and save each
                frame's raw canvas ('velocity_bev') to <output>.pt.
  MODE analyse: for every dumped frame, fit the same closed form under a
                list of geometry hypotheses and report weighted residual
                RMS + fitted |v_ego|. The correct mapping explains the
                measured radial field with a small residual and a |v_ego|
                consistent with the F1 band; a transposed/flipped mapping
                does neither. Also prints |v_r| vs assumed bearing bins for
                the assumed mapping, and the per-cell implied speed
                V_cell = -v_r / cos(theta) median/IQR (a static field gives
                a tight cluster at the true V).

Usage:
  python me_rssm/sanity/probe_ego_geometry_test.py dump \
    --config <dcrssm2 cfg> --checkpoint <ckpt> --indices 3 703 1403 \
    --output /data/lurui/work_dirs/dcrssm2_smoke_450iter/canvases.pt
  python me_rssm/sanity/probe_ego_geometry_test.py analyse \
    --dump /data/lurui/work_dirs/dcrssm2_smoke_450iter/canvases.pt \
    --pc-range 0 -39.68 -4 69.12 39.68 2 \
    --output /data/lurui/work_dirs/dcrssm2_smoke_450iter/geometry_test.json
"""

import argparse
import json
import math
import os
import sys
import warnings

warnings.filterwarnings('ignore')
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from me_rssm.ego_warp_rssm import (  # noqa: E402
    _CH_COUNT, _CH_SNR, _CH_SPEED, _CH_V_MEAN, _CH_V_STD)


# ----------------------------------------------------------------------
# dump mode
# ----------------------------------------------------------------------
def do_dump(a):
    from mmcv import Config  # noqa: PLC0415
    from mmcv.runner import load_checkpoint  # noqa: PLC0415
    from mmdet3d.datasets import build_dataset  # noqa: PLC0415
    from mmdet3d.models import build_detector  # noqa: PLC0415
    from me_rssm.modality_bus import ModalityContext  # noqa: PLC0415
    from tools.diagnose_z_utilization import (  # noqa: PLC0415
        build_batch, run_sequence)

    torch.manual_seed(0)
    np.random.seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = Config.fromfile(a.config)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model['meta_info'] = {
        'figures_path': '/tmp/r4det_geom_test_figs',
        'project_name': 'ego-geometry-test',
    }
    os.makedirs(cfg.model['meta_info']['figures_path'], exist_ok=True)
    cfg.data.val.pop('samples_per_gpu', None)
    cfg.data.val.pop('workers_per_gpu', None)
    dataset = build_dataset(cfg.data.val)
    model = build_detector(cfg.model, train_cfg=None, test_cfg=None)
    load_checkpoint(model, a.checkpoint, map_location='cpu')
    model = model.to(device).eval()
    fusion = model.temporal_fusion

    dumps = {}
    for index in a.indices:
        canvases = []
        handle = fusion.register_forward_hook(
            lambda m, i, o: canvases.append(
                None if ModalityContext.get('velocity_bev') is None
                else ModalityContext.get('velocity_bev').detach()
                .float().cpu()))
        batch = build_batch(dataset, [index], 0)
        with torch.no_grad():
            run_sequence(model, batch, z_mode='normal')
        handle.remove()
        dumps[str(index)] = canvases
        shapes = [None if c is None else list(c.shape) for c in canvases]
        print(f'index {index}: captured {len(canvases)} frames, shapes {shapes}')
    torch.save(dict(
        pc_range=list(map(float, cfg.model.temporal_fusion['pc_range'])),
        static_q=float(cfg.model.temporal_fusion['static_q']),
        min_static_cells=int(cfg.model.temporal_fusion['min_static_cells']),
        dump=dumps), a.output)
    print(f'wrote {a.output}')


# ----------------------------------------------------------------------
# analyse mode
# ----------------------------------------------------------------------
def canvas_to_world(canvas, pc_range, mapping):
    """Return (v_mean, abs_v_mean, v_std, snr, count, X, Y) on the 0.32 m
    feature grid under a geometry hypothesis.

    Canvas layout: (C, ny, nx) where ny = 2*W_feat, nx = 2*H_feat at
    0.16 m. ``mapping`` decides which canvas dim is world x and whether it
    is flipped:
      'yx'  : dim2 = world y (ascending), dim3 = world x (ascending)
              -- the assumption shipped in ego_warp_rssm.py
      'xy'  : dim2 = world x (ascending), dim3 = world y (ascending)
      'yx_fy': dim2 = y descending ... etc.
    Returns tensors on the 0.32 m grid in (x=dim0, y=dim1) order.
    """
    import torch.nn.functional as F  # noqa: PLC0415

    if canvas.dim() == 4:      # (B=1, C, ny, nx)
        canvas = canvas[0]
    C, ny, nx = canvas.shape
    pooled = F.avg_pool2d(canvas[None], 2)[0]  # (C, ny/2, nx/2)
    flip_y = mapping.endswith('_fy')
    flip_x = '_fx' in mapping
    axes = 'yx' if mapping.startswith('yx') else 'xy'
    if axes == 'yx':          # dim2 = y, dim3 = x -> want (C, x, y)
        g = pooled.permute(0, 2, 1)
    else:                     # dim2 = x, dim3 = y -> already (C, x, y)
        g = pooled
    if flip_x and axes == 'yx':
        g = g.flip(1)
    if flip_y and axes == 'yx':
        g = g.flip(2)
    if flip_y and axes == 'xy':
        g = g.flip(2)
    if flip_x and axes == 'xy':
        g = g.flip(1)
    H, W = g.shape[1], g.shape[2]
    x_min, y_min, _, x_max, y_max, _ = pc_range
    cell_x = (x_max - x_min) / H
    cell_y = (y_max - y_min) / W
    xs = x_min + (torch.arange(H) + 0.5) * cell_x
    ys = y_min + (torch.arange(W) + 0.5) * cell_y
    X = xs[:, None].expand(H, W)
    Y = ys[None, :].expand(H, W)
    return g, X, Y


def fit_wls(ux, uy, b_r, w):
    w = w / w.sum().clamp(min=1e-9)
    m_xx = (w * ux * ux).sum()
    m_xy = (w * ux * uy).sum()
    m_yy = (w * uy * uy).sum()
    r_x = (w * b_r * ux).sum()
    r_y = (w * b_r * uy).sum()
    det = m_xx * m_yy - m_xy * m_xy
    if float(det.abs()) < 1e-9:
        return None
    vx = (-r_x * m_yy + m_xy * r_y) / det
    vy = (-m_xx * r_y + m_xy * r_x) / det
    resid = (b_r + vx * ux + vy * uy)
    rms = float(torch.sqrt((w * resid * resid).sum()))
    return float(vx), float(vy), rms


MAPPINGS = ['yx', 'yx_fx', 'yx_fy', 'yx_fxfy', 'xy', 'xy_fx', 'xy_fy',
            'xy_fxfy']


def lmeds_solve(ux, uy, b_r, vx_grid, vy_grid):
    """Robust 1+1-param fit: minimise the MEDIAN absolute residual of
    b_r + vx*ux + vy*uy over all measured cells.

    L2 fits are pulled by the mover tail: a handful of strong-return moving
    cells (oncoming traffic) can carry |v_r| up to 3-4x the ego field, and
    count*SNR weights make them heavy. LMedS uses only the majority
    population, so it stays on the static field as long as static cells are
    the majority. Also returns the inlier fraction, which IS the static
    fraction the L1 set is trying to isolate.
    """
    import numpy as np  # noqa: PLC0415

    uxn = ux.numpy().astype(np.float64)
    uyn = uy.numpy().astype(np.float64)
    bn = b_r.numpy().astype(np.float64)
    best = None
    for vx in vx_grid:
        resid = np.abs(bn + vx * uxn)
        med = np.median(resid)
        if best is None or med < best[0]:
            best = (med, vx, 0.0)
    med0, vx0, _ = best
    for vy in vy_grid:
        resid = np.abs(bn + vx0 * uxn + vy * uyn)
        med = np.median(resid)
        if med < med0:
            med0, vy0 = med, vy
        else:
            vy0 = best[2]
    # local refinement around (vx0, vy0)
    for vx in np.arange(vx0 - 0.15, vx0 + 0.151, 0.01):
        for vy in np.arange(vy0 - 0.6, vy0 + 0.601, 0.05):
            med = np.median(np.abs(bn + vx * uxn + vy * uyn))
            if med < med0:
                med0, vx0, vy0 = med, float(vx), float(vy)
    resid = np.abs(bn + vx0 * uxn + vy0 * uyn)
    return dict(vx=vx0, vy=vy0, speed=math.hypot(vx0, vy0),
                med_abs_resid=float(med0),
                inlier_frac_05=float((resid <= 0.5).mean()),
                inlier_frac_10=float((resid <= 1.0).mean()),
                n=int(resid.size))


def irls_huber(ux, uy, b_r, w, v0, n_iter=5, c=1.345):
    """IRLS-Huber refinement of a closed-form velocity estimate.

    The L2 fit over the pre-registered static set is inflated by the mover
    tail: measured +45% (idx 3) to +96% (idx 703) against the LMedS truth,
    because a few strong-return moving cells carry count*SNR weights and
    large |v_r|. IRLS with a Huber loss and a MAD-derived scale is the
    textbook fix: residual scale sigma = 1.4826 * MAD (data-driven, no
    sweep) and the standard c = 1.345 (95% efficiency) constant; assign
    w <- w * min(1, c*sigma/|r|) and re-solve. Starting from the shipped
    estimate keeps the initial static population (which already excludes
    the extreme |v_r| > q90 movers).

    Returns (vx, vy, rms_huber, inlier_frac_05, sigma_final, n, v_init).
    """
    vx, vy = float(v0[0]), float(v0[1])
    n = int(b_r.numel())
    for _ in range(n_iter):
        resid = (b_r + vx * ux + vy * uy).abs()
        mad = float(resid.median())
        sigma = max(1.4826 * mad, 1e-3)
        huber = torch.clamp(c * sigma / resid.clamp(min=1e-6), max=1.0)
        ww = w * huber
        res = fit_wls(ux, uy, b_r, ww)
        if res is None:
            break
        vx, vy = res[0], res[1]
    resid = (b_r + vx * ux + vy * uy).abs()
    mad = float(resid.median())
    sigma = max(1.4826 * mad, 1e-3)
    rms = float(torch.sqrt((resid * resid).mean()))
    return dict(vx=vx, vy=vy, speed=math.hypot(vx, vy), rms=rms,
                inlier_frac_05=float((resid <= 0.5).float().mean()),
                sigma_final=sigma, n=n, v_init=list(v0))


def do_irls(a):
    """Offline test of the IRLS-Huber refinement against the LMedS truth."""
    import numpy as np  # noqa: PLC0415

    blob = torch.load(a.dump, map_location='cpu')
    pc_range = a.pc_range if a.pc_range else blob['pc_range']
    static_q = blob['static_q']
    min_cells = blob['min_static_cells']
    vx_grid = np.arange(-25.0, 25.01, 0.1)
    vy_grid = np.arange(-8.0, 8.01, 0.25)
    out = dict(dump=a.dump, pc_range=pc_range, frames={})
    print(f'{"idx":>5} {"f":>2} {"V_true":>7} {"V_rule":>7} {"V_irls":>7} '
          f'{"medres":>7} {"in.5":>6} {"sigma":>7} {"err_rule%":>9} '
          f'{"err_irls%":>9}')
    for index, canvases in blob['dump'].items():
        recs = []
        for t, canvas in enumerate(canvases):
            if canvas is None:
                continue
            g, X, Y = canvas_to_world(canvas, pc_range, 'yx')
            r = torch.sqrt(X * X + Y * Y).clamp(min=1e-6)
            ux, uy = X / r, Y / r
            count = g[_CH_COUNT]
            v_mean = g[_CH_V_MEAN]
            speed = g[_CH_SPEED]
            v_std = g[_CH_V_STD]
            snr = g[_CH_SNR]
            m = count > 0
            if int(m.sum().item()) < min_cells:
                continue
            true = lmeds_solve(ux[m], uy[m], v_mean[m], vx_grid, vy_grid)
            tau_s = torch.quantile(v_std[m], static_q)
            tau_v = torch.quantile(speed[m], static_q)
            sel = m & (v_std <= tau_s) & (speed <= tau_v)
            wgt = (count * snr)[sel]
            rule = fit_wls(ux[sel], uy[sel], v_mean[sel], wgt)
            irls = irls_huber(ux[sel], uy[sel], v_mean[sel], wgt,
                              (rule[0], rule[1])) if rule else None
            e_rule = (100 * (rule[2] and math.hypot(rule[0], rule[1])
                             - true['speed']) / true['speed']) \
                if rule and true['speed'] > 0.2 else float('nan')
            e_irls = (100 * (irls['speed'] - true['speed']) / true['speed']) \
                if irls and true['speed'] > 0.2 else float('nan')
            rec = dict(frame=t, V_true=true['speed'], med_abs_resid_true=
                       true['med_abs_resid'], inlier_frac_true=
                       true['inlier_frac_05'],
                       V_rule=(None if not rule else math.hypot(rule[0], rule[1])),
                       V_irls=(None if not irls else irls['speed']),
                       irls=(None if not irls else irls),
                       err_rule_pct=e_rule, err_irls_pct=e_irls)
            recs.append(rec)
            print(f'{index:>5} {t:>2} {true["speed"]:>7.3f} '
                  f'{(math.hypot(rule[0], rule[1]) if rule else float("nan")):>7.3f} '
                  f'{(irls["speed"] if irls else float("nan")):>7.3f} '
                  f'{true["med_abs_resid"]:>7.3f} '
                  f'{(irls["inlier_frac_05"] if irls else float("nan")):>6.3f} '
                  f'{(irls["sigma_final"] if irls else float("nan")):>7.3f} '
                  f'{e_rule:>9.1f} {e_irls:>9.1f}')
        out['frames'][index] = recs
    with open(a.output, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1)
    print(f'\nwrote {a.output}')


def do_robustfix(a):
    """Offline validation of the robust pipeline before touching the module.

    Stages (all detached, no learned params):
      1. shipped quantile pre-filter (tau_s / tau_v at static_q);
      2. robust seed: coarse 1-D LMedS over vx, then a local 2-D
         median-residual refine over (vx, vy);
      3. hard trim: keep cells with |v_r + v* . u_r| <= clamp(3*1.4826*MAD,
         0.2, 1.0) m/s (MAD at the seed; constants physics-derived: the
         static residual floor measured by LMedS is ~0.013 m/s, the trim
         band is generous by ~2 orders of magnitude, and the 1.0 cap keeps
         slow movers out);
      4. shipped closed form (count*SNR weighted 2x2 normal equations) on
         the trimmed set.
    The reference is the LMedS truth over ALL measured cells.
    """
    import numpy as np  # noqa: PLC0415

    blob = torch.load(a.dump, map_location='cpu')
    pc_range = a.pc_range if a.pc_range else blob['pc_range']
    static_q = blob['static_q']
    min_cells = blob['min_static_cells']
    vx_grid = np.arange(-25.0, 25.01, 0.1)
    vy_grid = np.arange(-8.0, 8.01, 0.25)
    out = dict(pipeline='quantile + LMedS seed + MAD trim + WLS',
               pc_range=pc_range, frames={})
    print(f'{"idx":>5} {"f":>2} {"V_true":>7} {"V_rule":>7} {"V_fix":>7} '
          f'{"vx_fix":>7} {"vy_fix":>7} {"kept":>5} {"tau_r":>6} '
          f'{"err_rule%":>9} {"err_fix%":>9}')
    for index, canvases in blob['dump'].items():
        recs = []
        for t, canvas in enumerate(canvases):
            if canvas is None:
                continue
            g, X, Y = canvas_to_world(canvas, pc_range, 'yx')
            r = torch.sqrt(X * X + Y * Y).clamp(min=1e-6)
            ux, uy = X / r, Y / r
            count = g[_CH_COUNT]
            v_mean = g[_CH_V_MEAN]
            speed = g[_CH_SPEED]
            v_std = g[_CH_V_STD]
            snr = g[_CH_SNR]
            m = count > 0
            if int(m.sum().item()) < min_cells:
                continue
            true = lmeds_solve(ux[m], uy[m], v_mean[m], vx_grid, vy_grid)
            tau_s = torch.quantile(v_std[m], static_q)
            tau_v = torch.quantile(speed[m], static_q)
            sel0 = m & (v_std <= tau_s) & (speed <= tau_v)
            ux0, uy0, br0 = ux[sel0], uy[sel0], v_mean[sel0]
            rule = fit_wls(ux0, uy0, br0, (count * snr)[sel0])
            # stage 2: LMedS seed (1-D over vx, then local 2-D)
            gvx = torch.arange(-25.0, 25.001, 0.25)
            R = (br0[:, None] + gvx[None, :] * ux0[:, None]).abs()
            vx_seed = float(gvx[R.median(dim=0).values.argmin()])
            lvx = torch.arange(vx_seed - 0.6, vx_seed + 0.601, 0.05)
            lvy = torch.arange(-3.0, 3.001, 0.15)
            R2 = (br0[None, None, :] + lvx[:, None, None] * ux0[None, None, :]
                  + lvy[None, :, None] * uy0[None, None, :]).abs()
            med2 = R2.median(dim=-1).values
            i_flat = int(med2.argmin().item())
            ix, iy = divmod(i_flat, med2.shape[1])
            vx_s, vy_s = float(lvx[ix]), float(lvy[iy])
            # stage 3: MAD trim
            resid = (br0 + vx_s * ux0 + vy_s * uy0).abs()
            mad = float(resid.median())
            tau_r = min(max(3.0 * 1.4826 * mad, 0.2), 1.0)
            keep = resid <= tau_r
            fix = fit_wls(ux0[keep], uy0[keep], br0[keep],
                          (count * snr)[sel0][keep]) if int(keep.sum()) >= min_cells \
                else None
            e_rule = (100 * (math.hypot(rule[0], rule[1]) - true['speed'])
                      / true['speed']) if rule and true['speed'] > 0.2 else float('nan')
            e_fix = (100 * (math.hypot(fix[0], fix[1]) - true['speed'])
                     / true['speed']) if fix and true['speed'] > 0.2 else float('nan')
            rec = dict(frame=t, V_true=true['speed'], V_rule=(
                None if not rule else math.hypot(rule[0], rule[1])),
                V_fix=(None if not fix else math.hypot(fix[0], fix[1])),
                vx_fix=(None if not fix else fix[0]),
                vy_fix=(None if not fix else fix[1]),
                kept=int(keep.sum().item()), n_sel=int(sel0.sum().item()),
                tau_r=tau_r, vx_seed=vx_seed, vy_seed=vy_s,
                err_rule_pct=e_rule, err_fix_pct=e_fix)
            recs.append(rec)
            print(f'{index:>5} {t:>2} {true["speed"]:>7.3f} '
                  f'{(math.hypot(rule[0], rule[1]) if rule else float("nan")):>7.3f} '
                  f'{(math.hypot(fix[0], fix[1]) if fix else float("nan")):>7.3f} '
                  f'{(fix[0] if fix else float("nan")):>7.3f} '
                  f'{(fix[1] if fix else float("nan")):>7.3f} '
                  f'{int(keep.sum().item()):>5} {tau_r:>6.2f} '
                  f'{e_rule:>9.1f} {e_fix:>9.1f}')
        out['frames'][index] = recs
    with open(a.output, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1)
    print(f'\nwrote {a.output}')


def do_robust(a):
    """LMedS solve + tail diagnostics on dumped canvases (offline)."""
    import numpy as np  # noqa: PLC0415

    blob = torch.load(a.dump, map_location='cpu')
    pc_range = a.pc_range if a.pc_range else blob['pc_range']
    vx_grid = np.arange(-25.0, 25.01, 0.1)
    vy_grid = np.arange(-8.0, 8.01, 0.25)
    out = dict(dump=a.dump, pc_range=pc_range, robust={})
    print(f'LMedS robust solve on dumped canvases (mapping yx, '
          f'all measured cells)')
    print(f'{"idx":>5} {"f":>2} {"n":>5} {"V_lmeds":>8} {"vx":>7} {"vy":>7} '
          f'{"medres":>7} {"in.5":>6} {"in1.0":>6} {"|vr|p50":>8} '
          f'{"|vr|p90":>8} {"tail>V+2":>9}')
    for index, canvases in blob['dump'].items():
        frames = []
        for t, canvas in enumerate(canvases):
            if canvas is None:
                continue
            g, X, Y = canvas_to_world(canvas, pc_range, 'yx')
            r = torch.sqrt(X * X + Y * Y).clamp(min=1e-6)
            ux, uy = X / r, Y / r
            m = g[_CH_COUNT] > 0
            speed = g[_CH_SPEED]
            v_mean = g[_CH_V_MEAN]
            if int(m.sum().item()) < 32:
                continue
            rob = lmeds_solve(ux[m], uy[m], v_mean[m], vx_grid, vy_grid)
            vr = speed[m].numpy()
            tail = float((vr > rob['speed'] + 2.0).mean())
            rec = dict(frame=t, **rob, abs_vr_p50=float(np.median(vr)),
                       abs_vr_p90=float(np.percentile(vr, 90)),
                       tail_frac_gt_V_plus_2=tail,
                       speed_resid_m=rob['speed'] * 0.1)
            frames.append(rec)
            print(f'{index:>5} {t:>2} {rob["n"]:>5} {rob["speed"]:>8.3f} '
                  f'{rob["vx"]:>7.3f} {rob["vy"]:>7.3f} '
                  f'{rob["med_abs_resid"]:>7.3f} '
                  f'{rob["inlier_frac_05"]:>6.3f} '
                  f'{rob["inlier_frac_10"]:>6.3f} '
                  f'{np.median(vr):>8.3f} {np.percentile(vr, 90):>8.3f} '
                  f'{tail:>9.3f}')
        out['robust'][index] = frames
    with open(a.output, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1)
    print(f'\nwrote {a.output}')


def do_analyse(a):
    blob = torch.load(a.dump, map_location='cpu')
    pc_range = a.pc_range if a.pc_range else blob['pc_range']
    static_q = blob['static_q']
    min_cells = blob['min_static_cells']
    out = dict(dump=a.dump, pc_range=pc_range, static_q=static_q,
               min_static_cells=min_cells, indices={})
    print(f'dump {a.dump}: indices {list(blob["dump"].keys())}')
    print(f'pc_range={pc_range} static_q={static_q} '
          f'min_cells={min_cells}')

    for index, canvases in blob['dump'].items():
        per_frame = []
        for t, canvas in enumerate(canvases):
            if canvas is None:
                continue
            print()
            print(f'=== idx {index} frame {t} ===')
            frame_rec = dict(frame=t, mappings={})
            for mapping in MAPPINGS:
                g, X, Y = canvas_to_world(canvas, pc_range, mapping)
                r = torch.sqrt(X * X + Y * Y).clamp(min=1e-6)
                ux, uy = X / r, Y / r
                count = g[_CH_COUNT]
                v_mean = g[_CH_V_MEAN]
                speed = g[_CH_SPEED]
                v_std = g[_CH_V_STD]
                snr = g[_CH_SNR]
                m = count > 0
                n_meas = int(m.sum().item())
                if n_meas < min_cells:
                    continue
                res = {}
                # (1) full measured set, count*snr weighted
                f_full = fit_wls(ux[m], uy[m], v_mean[m], (count * snr)[m])
                # (2) pre-registered rule
                tau_s = torch.quantile(v_std[m], static_q)
                tau_v = torch.quantile(speed[m], static_q)
                sel = m & (v_std <= tau_s) & (speed <= tau_v)
                f_rule = (fit_wls(ux[sel], uy[sel], v_mean[sel],
                                  (count * snr)[sel])
                          if int(sel.sum().item()) >= min_cells else None)
                # (3) high-|v_r| band q50
                thr = torch.quantile(speed[m], 0.5)
                sel_h = m & (speed >= thr)
                f_hi = (fit_wls(ux[sel_h], uy[sel_h], v_mean[sel_h],
                                (count * snr)[sel_h])
                        if int(sel_h.sum().item()) >= min_cells else None)
                if f_full:
                    res['full'] = dict(vx=f_full[0], vy=f_full[1],
                                       speed=math.hypot(f_full[0], f_full[1]),
                                       rms=f_full[2])
                if f_rule:
                    res['prereg'] = dict(vx=f_rule[0], vy=f_rule[1],
                                         speed=math.hypot(f_rule[0], f_rule[1]),
                                         rms=f_rule[2])
                if f_hi:
                    res['hi_band'] = dict(vx=f_hi[0], vy=f_hi[1],
                                          speed=math.hypot(f_hi[0], f_hi[1]),
                                          rms=f_hi[2])
                # per-cell implied speed under the rigid field
                cos_t = ux
                static_like = m & (cos_t.abs() > 0.4) & (speed > 0.5)
                if int(static_like.sum().item()) > 5:
                    v_cell = (-v_mean[static_like]
                              / cos_t[static_like].abs().clamp(min=0.4))
                    v_cell = v_cell.numpy()
                    res['v_cell'] = dict(
                        n=int(static_like.sum().item()),
                        p25=float(np.percentile(v_cell, 25)),
                        p50=float(np.percentile(v_cell, 50)),
                        p75=float(np.percentile(v_cell, 75)))
                frame_rec['mappings'][mapping] = res
            # table for the assumed mapping only
            print(f'{"mapping":>9} {"n":>5} {"full|v|":>8} {"rms":>7} '
                  f'{"prereg":>8} {"rms":>7} {"hi|v|":>7} {"rms":>7} '
                  f'{"Vcell p50":>10}')
            base = blob['dump'][index][t]
            g0, X0, Y0 = canvas_to_world(base, pc_range, 'yx')
            n_meas0 = int((g0[_CH_COUNT] > 0).sum().item())
            for mapping, res in frame_rec['mappings'].items():
                full = res.get('full', {})
                pre = res.get('prereg', {})
                hi = res.get('hi_band', {})
                vc = res.get('v_cell', {})
                print(f'{mapping:>9} {n_meas0:>5} '
                      f'{full.get("speed", float("nan")):>8.2f} '
                      f'{full.get("rms", float("nan")):>7.3f} '
                      f'{pre.get("speed", float("nan")):>8.2f} '
                      f'{pre.get("rms", float("nan")):>7.3f} '
                      f'{hi.get("speed", float("nan")):>7.2f} '
                      f'{hi.get("rms", float("nan")):>7.3f} '
                      f'{vc.get("p50", float("nan")):>10.2f}')
            # bearing-binned |v_r| for the assumed mapping
            speed0 = g0[_CH_SPEED]
            m0 = g0[_CH_COUNT] > 0
            r0 = torch.sqrt(X0 * X0 + Y0 * Y0).clamp(min=1e-6)
            ux0 = X0 / r0
            bins = [(0.95, 1.01), (0.8, 0.95), (0.6, 0.8), (0.3, 0.6),
                    (0.0, 0.3)]
            print('  assumed mapping yx: mean|v_r| by |cos theta| bin')
            binrec = {}
            for lo, hi_ in bins:
                s = m0 & (ux0.abs() >= lo) & (ux0.abs() < hi_)
                if int(s.sum().item()) > 0:
                    val = float(speed0[s].mean())
                    binrec[f'{lo}-{hi_}'] = dict(
                        n=int(s.sum().item()), mean_abs_vr=val)
                    print(f'    |cos| in [{lo},{hi_}): n={int(s.sum().item()):>4} '
                          f'mean |v_r| = {val:6.2f} m/s')
            frame_rec['bearing_bins_yx'] = binrec
            per_frame.append(frame_rec)
        out['indices'][index] = per_frame
    with open(a.output, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=1)
    print(f'\nwrote {a.output}')


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='mode', required=True)
    d = sub.add_parser('dump')
    d.add_argument('--config', required=True)
    d.add_argument('--checkpoint', required=True)
    d.add_argument('--indices', type=int, nargs='+', default=[3, 703, 1403])
    d.add_argument('--output', required=True)
    an = sub.add_parser('analyse')
    an.add_argument('--dump', required=True)
    an.add_argument('--pc-range', type=float, nargs=6, default=None)
    an.add_argument('--output', required=True)
    rb = sub.add_parser('robust')
    rb.add_argument('--dump', required=True)
    rb.add_argument('--pc-range', type=float, nargs=6, default=None)
    rb.add_argument('--output', required=True)
    rf = sub.add_parser('robustfix')
    rf.add_argument('--dump', required=True)
    rf.add_argument('--pc-range', type=float, nargs=6, default=None)
    rf.add_argument('--output', required=True)
    ir = sub.add_parser('irls')
    ir.add_argument('--dump', required=True)
    ir.add_argument('--pc-range', type=float, nargs=6, default=None)
    ir.add_argument('--output', required=True)
    a = ap.parse_args()
    if a.mode == 'dump':
        do_dump(a)
    elif a.mode == 'analyse':
        do_analyse(a)
    elif a.mode == 'robustfix':
        do_robustfix(a)
    elif a.mode == 'irls':
        do_irls(a)
    else:
        do_robust(a)


if __name__ == '__main__':
    main()
