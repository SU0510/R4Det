"""Pre-registered mechanism gate for DC-RSSM-2's L1 warp (read-only).

Gate text (written in the DC-RSSM-2 config header BEFORE any 24e data):

  (a) INERTNESS: mean |warp_scale| * dt * |v_ego| must exceed 1e-3 m. At
      iter 450 of the smoke the applied displacement is millimetres against
      a needed ~0.26 m/frame, so if it is still inertial at ep16 the warp
      was never exercised and the AP verdict says nothing about the
      mechanism -> "unexercised", NOT falsified.
  (b) If non-inert: PAIRED TOGGLE. Same ep16 weights, same val clips, two
      passes (warp ON vs warp_scale forced to 0); the static-region
      cross-frame alignment residual mean |h_warped(prev) - h_curr| over
      static cells, normalised by the mean |h_curr| on the same cells, must
      be LOWER with the warp active in >= 8 of 10 clips (sign test).

AMENDMENT (added 2026-10-09, before ep16 and before any AP data exists):
the scalar learning rate was accelerated 5x pre-run (config header), which
extrapolates to a LEARNED warp_scale well above the analytic optimum 1.0
(observed 1.6e-4/iter in the first 100 iterations under AdamW). A sign test
at an over-shooting learned value would test "the layer as optimised", not
the analytic mechanism. So the probe now runs THREE passes and reports two
sign tests:

  learned vs 0  -- what the model actually learned (the pre-registered
                   formulation, unchanged);
  forced 1.0 vs 0 -- the analytic mechanism at its principled magnitude,
                   independent of where the loss settled the scalar.

Verdicts use the pre-registered "learned vs 0" test; the forced-1.0 test is
reported alongside as the mechanism-magnitude control, and the exercised
fraction (applied / needed displacement) is printed for each pass.

Implementation notes:
  - The warped previous state is reproduced with the module's own formula
    and its own cached grid (identity-anchored grid_sample, exactly as
    EgoWarpEvidenceFieldRSSMFusion._align_with_motion computes it), so the
    ON measurement is the state the deformable actually consumed. At
    warp_scale == 0 the grid collapses to the identity grid, so the OFF
    pass needs no sampling (warped == carried state, bitwise).
  - Static cells per frame come from the canvas through the probe's mirror
    of the module's robust pipeline (asserted equal to the module's own
    solve inside probe_ego_warp_sanity; here the mask is recomputed with
    the same rule), so the residual is only read where the mechanism is
    supposed to act.
  - Eval mode + deterministic posterior: both passes are deterministic.

Usage:
  python me_rssm/sanity/probe_ego_alignment_toggle.py \
    --config <dcrssm2 cfg> --checkpoint <epoch_16.pth> \
    --start-index 3 --limit 10 --stride 300 \
    --output <work_dir>/ego_alignment_toggle_ep16.json
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
import torch.nn.functional as F  # noqa: E402

from me_rssm.ego_warp_rssm import (  # noqa: E402
    _CH_COUNT, _CH_SPEED, _CH_V_STD)
from me_rssm.modality_bus import ModalityContext  # noqa: E402

INERT_MIN_DISP_M = 1e-3   # pre-registered inertness line (metres/frame)
SIGN_TEST_MIN = 8         # of 10 clips


class _Thresholds(object):
    """Fallback constants mirroring the module class attributes.

    Used only when static_mask_from_canvas is called with fusion=None (e.g.
    a pure-canvas debug session); with a real module instance the values are
    read off the instance so there is a single source of truth.
    """
    static_q = 0.9
    min_static_cells = 32
    SEED_VX_MAX = 25.0
    SEED_VX_STEP = 0.25
    REFINE_VX_HALF = 0.6
    REFINE_VX_STEP = 0.05
    REFINE_VY_MAX = 3.0
    REFINE_VY_STEP = 0.15
    TRIM_MAD_K = 3.0
    TRIM_MIN = 0.2
    TRIM_MAX = 1.0


_THRESHOLDS = _Thresholds()


def static_mask_from_canvas(canvas, pc_range, static_q, min_cells,
                            fusion):
    """Recompute the module's static-cell mask on the feature grid.

    Mirrors stages 1-3 of _solve_ego (quantile pre-filter, LMedS seed, MAD
    trim) and returns (mask, v_ego, tau_r) for the batch element 0. Every
    threshold is read off the module instance, never duplicated here, so
    the mirror cannot drift from the code that trained; test W12 asserts
    mask and v_ego equal the module's own stash on synthetic canvases.
    """
    obj = fusion if fusion is not None else _THRESHOLDS
    static_q = float(getattr(obj, 'static_q', static_q))
    min_cells = int(getattr(obj, 'min_static_cells', min_cells))
    vx_max = float(obj.SEED_VX_MAX)
    vx_step = float(obj.SEED_VX_STEP)
    rvx_half = float(obj.REFINE_VX_HALF)
    rvx_step = float(obj.REFINE_VX_STEP)
    rvy_max = float(obj.REFINE_VY_MAX)
    rvy_step = float(obj.REFINE_VY_STEP)
    trim_k = float(obj.TRIM_MAD_K)
    trim_lo = float(obj.TRIM_MIN)
    trim_hi = float(obj.TRIM_MAX)
    det_eps = float(getattr(obj, 'det_eps', 1e-6))

    if canvas.dim() == 4:
        canvas = canvas[0]
    C, ny, nx = canvas.shape
    H, W = nx // 2, ny // 2
    vel = F.avg_pool2d(canvas[None].float(), 2).permute(0, 1, 3, 2)[0]
    x_min, y_min, _, x_max, y_max, _ = pc_range
    cell_x = (x_max - x_min) / H
    cell_y = (y_max - y_min) / W
    xs = x_min + (torch.arange(H) + 0.5) * cell_x
    ys = y_min + (torch.arange(W) + 0.5) * cell_y
    X = xs[:, None].expand(H, W)
    Y = ys[None, :].expand(H, W)
    r = torch.sqrt(X * X + Y * Y).clamp(min=1e-6)
    ux, uy = X / r, Y / r
    count = vel[_CH_COUNT]
    v_mean = vel[2]
    speed = vel[_CH_SPEED]
    v_std = vel[_CH_V_STD]
    snr = vel[5]
    m = count > 0
    if int(m.sum().item()) < min_cells:
        return None, None, None
    tau_s = torch.quantile(v_std[m], static_q)
    tau_v = torch.quantile(speed[m], static_q)
    sel0 = m & (v_std <= tau_s) & (speed <= tau_v)
    if int(sel0.sum().item()) < min_cells:
        return None, None, None
    u_x, u_y = ux[sel0], uy[sel0]
    b_r = v_mean[sel0]
    w = (count * snr)[sel0]
    seed = torch.arange(-vx_max, vx_max + 1e-6, vx_step)
    r1 = (b_r[:, None] + seed[None, :] * u_x[:, None]).abs()
    vx_seed = float(seed[r1.median(dim=0).values.argmin()])
    lvx = torch.arange(vx_seed - rvx_half, vx_seed + rvx_half + 1e-6, rvx_step)
    lvy = torch.arange(-rvy_max, rvy_max + 1e-6, rvy_step)
    r2 = (b_r[None, None, :] + lvx[:, None, None] * u_x[None, None, :]
          + lvy[None, :, None] * u_y[None, None, :]).abs()
    med2 = r2.median(dim=-1).values
    ix, iy = divmod(int(med2.argmin().item()), med2.shape[1])
    resid = (b_r + float(lvx[ix]) * u_x + float(lvy[iy]) * u_y).abs()
    mad = float(resid.median())
    tau_r = min(max(trim_k * 1.4826 * mad, trim_lo), trim_hi)
    keep = resid <= tau_r
    if int(keep.sum().item()) >= min_cells:
        u_x, u_y, b_r, w = u_x[keep], u_y[keep], b_r[keep], w[keep]
    else:
        keep = torch.ones_like(resid, dtype=torch.bool)
    # stage 4: weighted closed form on the trimmed set (same normal
    # equations as the module; degeneracy -> v_ego stays zero)
    w = w / w.sum().clamp(min=1e-6)
    m_xx = (w * u_x * u_x).sum()
    m_xy = (w * u_x * u_y).sum()
    m_yy = (w * u_y * u_y).sum()
    r_x = (w * b_r * u_x).sum()
    r_y = (w * b_r * u_y).sum()
    det = m_xx * m_yy - m_xy * m_xy
    if float(det.abs()) < det_eps:
        v_ego = torch.zeros(2)
    else:
        v_ego = torch.stack([(-r_x * m_yy + m_xy * r_y) / det,
                             (-m_xx * r_y + m_xy * r_x) / det])
    mask = torch.zeros_like(sel0)
    mask[sel0] = keep
    return mask, v_ego, tau_r


def apply_module_warp(state, grid, id_grid):
    """Exactly the module's identity-anchored warp, on a detached state."""
    if grid is None or id_grid is None:
        return state
    s = state.float()
    g = grid.to(s.device)
    ig = id_grid.to(s.device)
    warped = F.grid_sample(s, g, mode='bilinear', padding_mode='zeros',
                           align_corners=True)
    ident = F.grid_sample(s, ig, mode='bilinear', padding_mode='zeros',
                          align_corners=True)
    return s + (warped - ident)


def run_pass(model, fusion, dataset, index, force_warp_scale):
    """One sequence pass; returns per-frame records.

    force_warp_scale: None = as trained (ON), 0.0 = forced off (OFF).
    """



    saved = float(fusion.warp_scale)
    if force_warp_scale is not None:
        fusion.warp_scale.data.fill_(float(force_warp_scale))
    pending = []
    frames = []

    orig_align = fusion._align_with_motion

    def wrapped(state, feat, *args, **kwargs):
        pending.append(dict(
            state_in=state.detach().clone(),
            grid=(None if fusion._ego_grid is None
                  else fusion._ego_grid.detach().clone()),
            id_grid=(None if fusion._ego_id_grid is None
                     else fusion._ego_id_grid.detach().clone()),
            canvas=(None if ModalityContext.get('velocity_bev') is None
                    else ModalityContext.get('velocity_bev').detach()
                    .float().clone())))
        return orig_align(state, feat, *args, **kwargs)

    fusion._align_with_motion = wrapped

    from tools.diagnose_z_utilization import build_batch, run_sequence

    def pre_hook(module, inputs):
        # group the alignment calls of one forward (h first, then z when
        # align_z_state is on) so the main loop never relies on a fixed
        # call count per forward
        pending.clear()

    def post_hook(module, inputs, output):
        frames.append(dict(h_state=module.h_state.detach().clone(),
                           calls=list(pending)))

    h_pre = fusion.register_forward_pre_hook(pre_hook)
    h_post = fusion.register_forward_hook(post_hook)
    try:
        batch = build_batch(dataset, [index], 0)
        with torch.no_grad():
            run_sequence(model, batch, z_mode='normal')
    finally:
        fusion._align_with_motion = orig_align
        h_pre.remove()
        h_post.remove()
        fusion.warp_scale.data.fill_(saved)
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--start-index', type=int, default=3)
    ap.add_argument('--limit', type=int, default=10)
    ap.add_argument('--stride', type=int, default=300)
    ap.add_argument('--min-valid-history', type=int, default=2)
    ap.add_argument('--output', required=True)
    a = ap.parse_args()

    from mmcv import Config  # noqa: PLC0415
    from mmcv.runner import load_checkpoint  # noqa: PLC0415
    from mmdet3d.datasets import build_dataset  # noqa: PLC0415
    from mmdet3d.models import build_detector  # noqa: PLC0415
    from tools.diagnose_z_utilization import history_validity  # noqa: PLC0415

    torch.manual_seed(0)
    np.random.seed(0)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = Config.fromfile(a.config)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model['meta_info'] = {
        'figures_path': '/tmp/r4det_ego_toggle_figs',
        'project_name': 'ego-alignment-toggle',
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
    static_q = float(cfg.model.temporal_fusion['static_q'])
    min_cells = int(cfg.model.temporal_fusion['min_static_cells'])
    print(f'warp_scale (trained) = {float(fusion.warp_scale):.6f}  '
          f'dt = {float(fusion.dt):.6f}')

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

    indices = select_indices()
    print(f'clips: {indices}')
    print(f'{"idx":>6} {"res_off":>9} {"res_lrn":>9} {"res_f1":>9} '
          f'{"d_lrn%":>7} {"d_f1%":>7} {"nstatic":>8} {"disp_lrn_m":>10} '
          f'{"disp_f1_m":>10} {"need_m":>8} {"v_ego":>7}')
    clips = []
    for index in indices:
        passes = {}
        for tag, forced in (('off', 0.0), ('learned', None), ('forced1', 1.0)):
            passes[tag] = run_pass(model, fusion, dataset, index, forced)
        n_frames = min(len(frames_p) for frames_p in passes.values())
        rec = dict(index=index, frames=[], res_off=None, res_learned=None,
                   res_forced1=None)
        num = {t: 0.0 for t in passes}
        den = 0.0
        for t in range(1, n_frames):
            # Pairing: the alignment call INSIDE forward t receives the
            # state carried over from t-1 and warps it before the frame-t
            # update, so its partner is forward t's output h_state -- both
            # are frame-t quantities (carried-then-warped vs updated).
            # Indexing the call at t-1 would compare the state carried out
            # of t-2 against frame t (off by one frame).
            # calls are grouped per forward; call[0] is the h-state
            # alignment (call[1] is the z-state one).
            r_off, r_lrn = passes['off'][t], passes['learned'][t]
            r_f1 = passes['forced1'][t]
            if not (r_off['calls'] and r_lrn['calls'] and r_f1['calls']):
                continue
            call_off, call_lrn = r_off['calls'][0], r_lrn['calls'][0]
            call_f1 = r_f1['calls'][0]
            canvas_t = call_off['canvas']
            if canvas_t is None:
                continue
            mask, v_ego, tau_r = static_mask_from_canvas(
                canvas_t, pc_range, static_q, min_cells, fusion)
            if mask is None or int(mask.sum().item()) < 8:
                continue
            m = mask.to(device)
            h_curr_off = r_off['h_state'].float()
            h_curr_lrn = r_lrn['h_state'].float()
            h_curr_f1 = r_f1['h_state'].float()
            warped_lrn = apply_module_warp(
                call_lrn['state_in'].to(device).float(), call_lrn['grid'],
                call_lrn['id_grid'])
            warped_f1 = apply_module_warp(
                call_f1['state_in'].to(device).float(), call_f1['grid'],
                call_f1['id_grid'])
            num['off'] += float(
                (call_off['state_in'].to(device).float() - h_curr_off)
                [:, :, m].abs().mean())
            num['learned'] += float((warped_lrn - h_curr_lrn)
                                    [:, :, m].abs().mean())
            num['forced1'] += float((warped_f1 - h_curr_f1)
                                    [:, :, m].abs().mean())
            den += float(h_curr_lrn[:, :, m].abs().mean())
            v_norm = float(torch.as_tensor(v_ego).norm())
            need = float(fusion.dt) * v_norm
            rec['frames'].append(dict(
                frame=t, n_static=int(mask.sum().item()),
                v_ego=[float(v_ego[0]), float(v_ego[1])], tau_r=tau_r,
                need_m=need,
                disp_learned_m=float(fusion.warp_scale) * need,
                disp_forced1_m=need))
        if den > 0:
            rec['res_off'] = num['off'] / den
            rec['res_learned'] = num['learned'] / den
            rec['res_forced1'] = num['forced1'] / den
        clips.append(rec)

        def _pct(res_x):
            if rec['res_off'] is None or res_x is None:
                return float('nan')
            return 100 * (res_x - rec['res_off']) / rec['res_off']

        print(f'{index:>6} '
              f'{(rec["res_off"] if rec["res_off"] is not None else float("nan")):>9.5f} '
              f'{(rec["res_learned"] if rec["res_learned"] is not None else float("nan")):>9.5f} '
              f'{(rec["res_forced1"] if rec["res_forced1"] is not None else float("nan")):>9.5f} '
              f'{_pct(rec["res_learned"]):>7.2f} '
              f'{_pct(rec["res_forced1"]):>7.2f} '
              f'{(sum(f["n_static"] for f in rec["frames"])):>8} '
              f'{(np.mean([f["disp_learned_m"] for f in rec["frames"]]) if rec["frames"] else float("nan")):>10.5f} '
              f'{(np.mean([f["disp_forced1_m"] for f in rec["frames"]]) if rec["frames"] else float("nan")):>10.5f} '
              f'{(np.mean([f["need_m"] for f in rec["frames"]]) if rec["frames"] else float("nan")):>8.3f} '
              f'{(np.mean([np.hypot(*f["v_ego"]) for f in rec["frames"]]) if rec["frames"] else float("nan")):>7.3f}')

    def sign_test(key):
        paired_ = [c for c in clips if c['res_off'] is not None
                   and c[key] is not None]
        wins_ = sum(1 for c in paired_ if c[key] < c['res_off'])
        return wins_, len(paired_)

    wins_lrn, n_lrn = sign_test('res_learned')
    wins_f1, n_f1 = sign_test('res_forced1')
    disps_lrn = [f['disp_learned_m'] for c in clips for f in c['frames']]
    disps_f1 = [f['disp_forced1_m'] for c in clips for f in c['frames']]
    needs = [f['need_m'] for c in clips for f in c['frames']]
    mean_lrn = float(np.mean(disps_lrn)) if disps_lrn else float('nan')
    mean_need = float(np.mean(needs)) if needs else float('nan')
    inert = bool(disps_lrn) and (mean_lrn < INERT_MIN_DISP_M)
    print()
    print(f'clips with all residuals  : {n_lrn}/{len(clips)}')
    print(f'learned vs 0  (wins)      : {wins_lrn}/{n_lrn}')
    print(f'forced 1.0 vs 0 (wins)    : {wins_f1}/{n_f1}')
    print(f'learned warp_scale        : {float(fusion.warp_scale):.6f}')
    print(f'mean applied displacement : {mean_lrn:.6f} m/frame '
          f'({100 * mean_lrn / mean_need if mean_need else float("nan"):.1f}% '
          f'of needed; forced-1.0 = {float(np.mean(disps_f1)) if disps_f1 else float("nan"):.4f} m)')
    print(f'mean needed displacement  : {mean_need:.4f} m/frame')
    print(f'INERT (applied < {INERT_MIN_DISP_M} m): {inert}')
    if inert:
        verdict = 'UNEXERCISED: warp displacement is still millimetres; ' \
                  'the toggle test cannot attribute anything to the mechanism'
    elif n_lrn and wins_lrn >= SIGN_TEST_MIN:
        verdict = f'MECHANISM PASS: the learned warp lowers the static-cell ' \
                  f'residual in {wins_lrn}/{n_lrn} clips (>= {SIGN_TEST_MIN})'
    else:
        verdict = (f'MECHANISM FAIL: the learned warp lowers the static-cell '
                   f'residual in only {wins_lrn}/{n_lrn} clips (< '
                   f'{SIGN_TEST_MIN}) while being non-inert')
    verdict_f1 = (f'forced-1.0 control: {wins_f1}/{n_f1} clips improve '
                  f'(>= {SIGN_TEST_MIN} needed)')
    print(f'VERDICT: {verdict}')
    print(f'CONTROL: {verdict_f1}')

    with open(a.output, 'w', encoding='utf-8') as f:
        json.dump(dict(config=a.config, checkpoint=a.checkpoint,
                       warp_scale=float(fusion.warp_scale),
                       dt=float(fusion.dt), clips=clips,
                       wins_learned=wins_lrn, n_learned=n_lrn,
                       wins_forced1=wins_f1, n_forced1=n_f1,
                       mean_disp_learned_m=mean_lrn,
                       mean_disp_forced1_m=(float(np.mean(disps_f1))
                                            if disps_f1 else None),
                       mean_need_m=mean_need,
                       inert=inert, verdict=verdict,
                       verdict_forced1=verdict_f1), f, indent=1)
    print(f'wrote {a.output}')


if __name__ == '__main__':
    main()
