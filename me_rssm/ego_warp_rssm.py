"""DC-RSSM-2: analytic pose-free ego-warp on the carried state (docs/13 L1).

Design: me_rssm/docs/13_dc_rssm.md L1 (step 2 of the DC-RSSM chain). This
file only ADDS a new class to the me_rssm research package; no existing
file is modified.

Why this class exists
---------------------
The DC-RSSM-1 verdict (training_runs_full.md §68.9h) falsified L3 as a
standalone improvement (AP gate FAIL) but kept it as a Ped-repair
component (Ped 23.10 -> 27.07, +3.97). L1 is the chain's flagship layer:
the Doppler canvas is EGO-DOMINATED (docs/08 F1 -- ego-moving frames show
|v| ~ 7-13 m/s with ~100% of filled cells above 1 m/s, and the ego flow
field is a deterministic function of position), so the carried state
expressed in the PREVIOUS ego frame is misaligned with the current frame
by exactly the ego displacement. The deformable alignment currently has
to learn this global rigid shift from detection gradients; L1 removes it
analytically so the deformable sees only the residual (docs/13 L1:
"deformable 对齐只处理残余").

Mechanism (pose-free -- TJ4D has no ego pose and no timestamps, docs/01;
CRT-Fusion depends on nuScenes poses, this setting makes self-calibration
a necessity, not a choice):

    static set  S = {cell: count > 0, v_std <= tau_s, |v_r_mean| <= tau_v}
                (per-frame quantiles at a FIXED level -- no hyperparameter
                sweep; v_std cuts mixed-target pillars, |v_r_mean| cuts
                the most extreme approach/recede returns)
    solve       v_ego* = argmin_{v in R^2} sum_{p in S} w_p (v_r(p) + v.u_r(p))^2
                with w_p = count*SNR (normalization cancels in the solve),
                closed-form 2x2 normal equations, per frame, DETACHED --
                the solve is a measurement, no detection gradients reach
                the canvas math or v_ego.
    displacement  delta_p = warp_scale * (v_ego * dt)
                dt is a LEARNABLE scalar initialized at the nominal frame
                interval (0.1 s, honest design -- TJ4D publishes no frame
                timestamps); warp_scale is a zero-init scalar (identity
                start, docs/13 §4).
    warp        h_warped(p) = h_state(p + delta_p)
                bilinear grid_sample, zeros padding, applied to the
                carried state BEFORE the deformable alignment.

Sign conventions (pinned by docs/08 F1): approaching returns are NEGATIVE
(static content ahead of a forward-moving ego reads v_r ~ -v_ego), the
unit radial direction u_r points from the ego (BEV origin) to the cell.
The fit v_r ~ -v_ego.u_r then returns v = the ego velocity vector in the
BEV/world frame; a static pole 10 m ahead is now at 9.8 m after forward
motion, and h_warped(9.8) samples h_state(10) = h_state(p + delta_p).

Geometry (pinned by R4Det.py:871-873 assert + the scatter's
output_shape): the temporal feature is (B, C, H, W) with dim2 = world x
(forward, pc_range[0..3] over H cells) and dim3 = world y (lateral,
pc_range[1..4] over W cells). The velocity_bev canvas is (B, 7, ny, nx)
with ny along dim2 = world y and nx along dim3 = world x at HALF the BEV
resolution; the same avg_pool2d + permute as _read_context puts it back
on the feature grid. Cell sizes are DERIVED from pc_range and the feature
shape (self-consistent -- no separate cell-size argument to drift).

Deadlock discipline (docs/13 §4, 5th instance, test-asserted): warp_scale
is zero-init for the bitwise identity start, but -- unlike the 68.8i
zero-init-conv-feeds-zero-init-conv pattern -- its input is the state's
own spatial structure: grid_sample's grid gradient at an identity grid is
the local input difference (nonzero wherever the state varies), so
warp_scale receives gradient from step 1. dt's gradient is chained
through warp_scale and is exactly zero UNTIL warp_scale moves (the same
"arrives at step 2+" pattern as the L3 refine conv).

The warp is a VIEW transform applied at consumption time: h_state itself
stays in the previous frame's coordinates (each frame's warp is computed
fresh from the current frame's solve); reset_for_samples zeroing is
naturally preserved (grid_sample of a zero state is zero).

Identity start: at init warp_scale == 0 exactly -> delta_p == 0 -> the
displaced grid equals the identity grid tensor-wise, and the warp is
applied identity-anchored, state + (sample(delta) - sample(identity)):
the two samples are then the same computation and the corrector is
exactly zero, so h_warped == h_state BITWISE (grid_sample's normalized
round trip alone loses ~1 ulp even at align_corners=True -- measured --
which is exactly why the anchor exists). Switch use_ego_warp=False
reproduces DC-RSSM-1 bitwise (same class hierarchy, no new parameters
constructed, checkpoints load unchanged) -- the generational comparison
is pre-registered in the config, not a bitwise gate.

Interface contract (unchanged from MCI/DC-RSSM-1):
    forward(feat, velocity=None, use_posterior=True, deterministic=True,
            detach_state=True)
        -> (output, reconstruction, latent_loss, h_t, z_t, stats)
reset_state() / reset_for_samples(mask) / probe hooks all inherited
unchanged; the warp hooks in via _read_context (solve once per forward,
before alignment) and _align_with_motion (apply to the state).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.models.builder import FUSION_LAYERS

from .evidence_field_rssm import (
    _CH_COUNT, _CH_SNR, _CH_SPEED, _CH_V_MEAN, _CH_V_STD,
    EvidenceFieldInnovationRSSMFusion)
from .modality_bus import ModalityContext

__all__ = ['EgoWarpEvidenceFieldRSSMFusion']


@FUSION_LAYERS.register_module()
class EgoWarpEvidenceFieldRSSMFusion(EvidenceFieldInnovationRSSMFusion):
    """DC-RSSM-1 + L1 analytic ego-warp (docs/13 step 2).

    New args (all EvidenceField/MCI args are inherited unchanged):

    Args:
        use_ego_warp (bool): build the L1 solve + warp. False reproduces
            DC-RSSM-1 bitwise (no new parameters).
        static_q (float): fixed quantile level of the per-frame static
            thresholds tau_s (v_std) and tau_v (|v_r_mean|) over measured
            cells. A design constant, never swept.
        min_static_cells (int): below this many measured cells (or static
            cells) the solve is skipped and v_ego = 0 (identity warp) --
            "no measurement -> no warp", the honest fallback for the
            val set's empty frames.
        dt_init (float): initial value of the learnable frame interval
            (seconds). TJ4D publishes no timestamps; 0.1 s is the nominal
            automotive radar interval (documented assumption).
        det_eps (float): |det(M)| guard of the 2x2 normal equations; below
            it the geometry is degenerate (all cells on one ray) and the
            solve falls back to v_ego = 0.
        pc_range (list[float]): [x_min, y_min, z_min, x_max, y_max,
            z_max] of the BEV feature grid (the dataset's
            point_cloud_range). Cell sizes are derived from it and the
            feature shape.
        warp_z_state (bool): also warp z_state before its alignment
            (align_z_state=True configurations carry z across frames in
            the previous ego frame too). Default True.
        robust_solve (bool): use the robust static-set pipeline (LMedS
            seed + MAD residual trim) instead of the raw weighted
            least-squares fit. Default True. NOT a tuning knob: the probe
            (me_rssm/sanity/probe_ego_geometry_test.py, modes ``robust`` /
            ``robustfix``) showed the plain weighted fit is inflated by
            +24%..+96% on val frames because a handful of strong-return
            moving cells carry count*SNR weights up to ~100x a static
            cell's; the robust path lands within 0.2% of the LMedS truth
            on all 12 probed frames. False is kept only so the unit tests
            can characterise the bias of the shipped rule.
    """

    # Robust-solve constants. Fixed, physics/robust-statistics derived --
    # NOT tuning knobs, and validated offline against the LMedS truth on
    # 12 val frames (me_rssm/sanity/probe_ego_geometry_test.py mode
    # 'robustfix'): 1.4826*MAD is the consistent Gaussian scale estimator;
    # the coarse seed grid spans every plausible ego speed at 0.25 m/s;
    # the trim band [0.2, 1.0] m/s sits two orders of magnitude above the
    # measured static residual floor (0.013 m/s) and below relative mover
    # speeds.
    SEED_VX_MAX = 25.0
    SEED_VX_STEP = 0.25
    REFINE_VX_HALF = 0.6
    REFINE_VX_STEP = 0.05
    REFINE_VY_MAX = 3.0
    REFINE_VY_STEP = 0.15
    TRIM_MAD_K = 3.0
    TRIM_MIN = 0.2
    TRIM_MAX = 1.0

    def __init__(
        self,
        use_ego_warp=True,
        static_q=0.9,
        min_static_cells=32,
        dt_init=0.1,
        det_eps=1e-6,
        pc_range=None,
        warp_z_state=True,
        robust_solve=True,
        **kwargs,
    ):
        if static_q <= 0 or static_q >= 1:
            raise ValueError('static_q must be in (0, 1)')
        if dt_init <= 0:
            raise ValueError('dt_init must be positive')
        self.use_ego_warp = use_ego_warp
        self.static_q = static_q
        self.min_static_cells = min_static_cells
        self.dt_init = dt_init
        self.det_eps = det_eps
        self.warp_z_state = warp_z_state
        self.robust_solve = robust_solve
        if use_ego_warp and pc_range is None:
            raise ValueError('use_ego_warp=True requires pc_range')
        self._pc_range = list(pc_range) if pc_range is not None else None
        super().__init__(**kwargs)
        if use_ego_warp:
            # +2 params total (docs/13's "+1 (dt)" predates the identity-
            # start scale). Both learnable; grid gradient reaches
            # warp_scale from step 1 (see module docstring, 5th deadlock
            # instance).
            self.dt = nn.Parameter(torch.tensor(float(dt_init)))
            self.warp_scale = nn.Parameter(torch.tensor(0.0))
        # persistent=False: geometry is architecture, not learned state --
        # keeps switch-off checkpoints byte-compatible.
        if use_ego_warp and pc_range is not None:
            self.register_buffer(
                '_pc_range_t',
                torch.tensor(pc_range, dtype=torch.float32),
                persistent=False)
        self._ego_grid = None
        self._ego_id_grid = None
        self._ego_solve = None

    # ------------------------------------------------------------------
    # L1 solve: closed-form v_ego from the Doppler canvas (measurement)
    # ------------------------------------------------------------------
    def _solve_ego(self, vel, feat):
        """Per-frame closed-form ego-velocity solve on the static set.

        vel: (B, 7, H, W) Doppler canvas ON THE FEATURE GRID (post
        avg_pool + permute, as in _read_context): dim2 = world x cells
        (forward), dim3 = world y cells (lateral). All outputs detached.

        Returns (v_ego, info): v_ego (B, 2) in m/s (world x, y); info a
        dict of detached diagnostics (static fraction, solved mask).
        """
        B, _, H, W = feat.shape
        x_min, y_min = self._pc_range_t[0], self._pc_range_t[1]
        x_max, y_max = self._pc_range_t[3], self._pc_range_t[4]
        cell_x = (x_max - x_min) / H
        cell_y = (y_max - y_min) / W
        # cell-center world coordinates on the feature grid
        xs = x_min + (torch.arange(H, device=vel.device,
                                    dtype=torch.float32) + 0.5) * cell_x
        ys = y_min + (torch.arange(W, device=vel.device,
                                    dtype=torch.float32) + 0.5) * cell_y
        X = xs[:, None].expand(H, W)
        Y = ys[None, :].expand(H, W)
        r = torch.sqrt(X * X + Y * Y).clamp(min=1e-6)
        ux, uy = X / r, Y / r

        count = vel[:, _CH_COUNT]
        v_mean = vel[:, _CH_V_MEAN]
        speed = vel[:, _CH_SPEED]
        v_std = vel[:, _CH_V_STD]
        snr = vel[:, _CH_SNR]
        measured = count > 0

        v_egos = []
        solved = []
        trim_fracs = []
        tau_rs = []
        for b in range(B):
            m = measured[b]
            v0 = vel.new_zeros(2)
            trim_fracs.append(vel.new_zeros(()))
            tau_rs.append(vel.new_zeros(()))
            if int(m.sum().item()) < self.min_static_cells:
                v_egos.append(v0)
                solved.append(False)
                continue
            # --- stage 1: quantile pre-filter over measured cells -------
            # keeps the static majority and already drops the extreme
            # |v_r| movers; the residual trim below fixes what is left.
            tau_s = torch.quantile(v_std[b][m], self.static_q)
            tau_v = torch.quantile(speed[b][m], self.static_q)
            sel0 = m & (v_std[b] <= tau_s) & (speed[b] <= tau_v)
            if int(sel0.sum().item()) < self.min_static_cells:
                v_egos.append(v0)
                solved.append(False)
                continue
            u_x, u_y = ux[sel0], uy[sel0]
            b_r = v_mean[b][sel0]
            w = (count[b] * snr[b])[sel0]
            if self.robust_solve:
                # --- stage 2: robust seed ------------------------------
                # The plain weighted fit is pulled by the mover tail:
                # a strong-return moving cell carries count*SNR up to
                # ~100x a static cell's, and val probes measured +24%
                # ..+96% inflation (probe_ego_geometry_test.py).
                # LMedS (median absolute residual) is not influenced by
                # that tail: coarse 1-D over vx, then a local 2-D
                # median-residual refine.
                seed = torch.arange(-self.SEED_VX_MAX,
                                    self.SEED_VX_MAX + 1e-6,
                                    self.SEED_VX_STEP, device=vel.device,
                                    dtype=torch.float32)
                r1 = (b_r[:, None] + seed[None, :] * u_x[:, None]).abs()
                vx_seed = float(seed[r1.median(dim=0).values.argmin()])
                lvx = torch.arange(vx_seed - self.REFINE_VX_HALF,
                                   vx_seed + self.REFINE_VX_HALF + 1e-6,
                                   self.REFINE_VX_STEP, device=vel.device,
                                   dtype=torch.float32)
                lvy = torch.arange(-self.REFINE_VY_MAX,
                                   self.REFINE_VY_MAX + 1e-6,
                                   self.REFINE_VY_STEP, device=vel.device,
                                   dtype=torch.float32)
                r2 = (b_r[None, None, :]
                      + lvx[:, None, None] * u_x[None, None, :]
                      + lvy[None, :, None] * u_y[None, None, :]).abs()
                med2 = r2.median(dim=-1).values
                i_flat = int(med2.argmin().item())
                ix, iy = divmod(i_flat, med2.shape[1])
                # --- stage 3: MAD residual trim ------------------------
                # residual scale from the median absolute residual at the
                # seed (1.4826*MAD is the consistent sigma estimator);
                # the band is clamped to [0.2, 1.0] m/s: the static
                # residual floor measured on val is 0.013 m/s (2 orders
                # below), and 1.0 m/s keeps slow relative movers out.
                resid = (b_r + float(lvx[ix]) * u_x
                         + float(lvy[iy]) * u_y).abs()
                mad = float(resid.median())
                tau_r = min(max(self.TRIM_MAD_K * 1.4826 * mad,
                                self.TRIM_MIN), self.TRIM_MAX)
                keep = resid <= tau_r
                if int(keep.sum().item()) >= self.min_static_cells:
                    u_x, u_y, b_r, w = (u_x[keep], u_y[keep], b_r[keep],
                                        w[keep])
                    trim_fracs[-1] = keep.float().mean().detach()
                tau_rs[-1] = vel.new_tensor(tau_r)
            # --- stage 4: weighted closed form on the static set --------
            w = w / w.sum().clamp(min=1e-6)
            # normal equations M v = -rhs  (f(v) = sum w (b + v.u)^2)
            m_xx = (w * u_x * u_x).sum()
            m_xy = (w * u_x * u_y).sum()
            m_yy = (w * u_y * u_y).sum()
            r_x = (w * b_r * u_x).sum()
            r_y = (w * b_r * u_y).sum()
            det = m_xx * m_yy - m_xy * m_xy
            if det.abs() < self.det_eps:
                v_egos.append(v0)
                solved.append(False)
                continue
            # Cramer: v = M^-1 (-r)
            vx = (-r_x * m_yy + m_xy * r_y) / det
            vy = (-m_xx * r_y + m_xy * r_x) / det
            v_egos.append(torch.stack([vx, vy]))
            solved.append(True)
        v_ego = torch.stack(v_egos).detach()
        info = dict(
            static_frac=measured.float().mean().detach(),
            trim_frac=torch.stack(trim_fracs).detach(),
            tau_r=torch.stack(tau_rs).detach(),
            solved=torch.tensor(solved, dtype=torch.bool,
                                device=vel.device))
        return v_ego, info

    # ------------------------------------------------------------------
    # integration hooks: solve once per forward (before alignment), warp
    # the carried state before the deformable sees it
    # ------------------------------------------------------------------
    def _read_context(self, feat):
        """Compute the L1 warp grid alongside the MCI/Evidence context.

        The grid is computed ONCE per forward here (before any alignment
        call) and cached for _align_with_motion; a dead bus or a
        canvas/feature mismatch leaves it None (identity warp).
        """
        ctx = super()._read_context(feat)
        self._ego_grid = None
        self._ego_id_grid = None
        self._ego_solve = None
        if not self.use_ego_warp:
            return ctx
        B, _, H, W = feat.shape
        vel_raw = ModalityContext.get('velocity_bev')
        # canvas (B, 7, ny, nx): ny along dim2 = 2*W (world y), nx along
        # dim3 = 2*H (world x) -- the pinned cross mapping (see module
        # docstring geometry note)
        have_bus = (
            vel_raw is not None
            and vel_raw.shape[0] == B
            and vel_raw.shape[1] == self.motion_channels
            and vel_raw.shape[-2] == 2 * W
            and vel_raw.shape[-1] == 2 * H)
        if not have_bus:
            return ctx
        vel = F.avg_pool2d(vel_raw.float(), 2)
        vel = vel.permute(0, 1, 3, 2).contiguous()
        v_ego, info = self._solve_ego(vel, feat)
        # displacement in meters -> cell units; warp_scale/dt keep the
        # graph (learned), the solve is a detached measurement
        cell_x = (self._pc_range_t[3] - self._pc_range_t[0]) / H
        cell_y = (self._pc_range_t[4] - self._pc_range_t[1]) / W
        dx = (self.warp_scale * (v_ego[:, 0] * self.dt)) / cell_x
        dy = (self.warp_scale * (v_ego[:, 1] * self.dt)) / cell_y
        # h_warped(p) = h_state(p + delta_p): sample source in CELL units
        # added DIRECTLY to the integer index -- identity at delta 0 is
        # bitwise (no world-coordinate round trip). Axis mapping (pinned
        # by grid_sample's convention: grid[..., 0] = source coord along
        # input dim3, grid[..., 1] = along input dim2):
        #   world x (the dx cell shift) -> dim2 (H) -> grid[..., 1]
        #   world y (the dy cell shift) -> dim3 (W) -> grid[..., 0]
        ix = (torch.arange(H, device=feat.device, dtype=dx.dtype)
              [None, :, None] + dx[:, None, None])
        iy = (torch.arange(W, device=feat.device, dtype=dy.dtype)
              [None, None, :] + dy[:, None, None])
        jx = torch.arange(W, device=feat.device, dtype=dx.dtype)
        jy = torch.arange(H, device=feat.device, dtype=dx.dtype)
        gx0 = (2.0 * jx / max(W - 1, 1) - 1.0)[None, None, :].expand(B, H, W)
        gy0 = (2.0 * jy / max(H - 1, 1) - 1.0)[None, :, None].expand(B, H, W)
        gx = (2.0 * iy / max(W - 1, 1) - 1.0).expand(B, H, W)
        gy = (2.0 * ix / max(H - 1, 1) - 1.0).expand(B, H, W)
        self._ego_grid = torch.stack([gx, gy], dim=-1)
        self._ego_id_grid = torch.stack([gx0, gy0], dim=-1)
        self._ego_solve = dict(
            v_ego=v_ego,
            static_frac=info['static_frac'],
            trim_frac=info['trim_frac'],
            tau_r=info['tau_r'],
            solved=info['solved'])
        return ctx

    def _align_with_motion(self, state, feat, offset_mask_conv, deform_conv,
                           motion_conv=None, e_mot=None):
        """MCI alignment, preceded by the L1 ego-warp on the state.

        Warping here (not in forward) injects the warp exactly between
        state init and the deformable alignment for BOTH carried states,
        so the deformable sees only the residual (docs/13 L1). h_state is
        always warped; z_state follows warp_z_state (align_z_state=True
        configurations carry z across frames in the previous ego frame
        too).
        """
        if (self.use_ego_warp and self._ego_grid is not None
                and (self.warp_z_state or state is not self.z_state)):
            state_f = state.float()
            grid = self._ego_grid.to(state_f.device)
            id_grid = self._ego_id_grid.to(state_f.device)
            # Identity-anchored warp: state + (sampled(displaced) -
            # sampled(identity)). grid_sample's normalized round trip
            # loses ~1 ulp even at align_corners=True (measured: 4e-7
            # CPU / 2e-6 CUDA), so a plain grid_sample at zero shift is
            # NOT bitwise the identity; anchoring both samples makes the
            # corrector exactly zero at warp_scale == 0 (identical grid
            # tensors -> identical outputs -> exact subtraction), which
            # is what keeps the identity start bitwise. Gradient flows to
            # warp_scale/dt through the displaced sample only.
            warped = F.grid_sample(state_f, grid, mode='bilinear',
                                   padding_mode='zeros', align_corners=True)
            ident = F.grid_sample(state_f, id_grid, mode='bilinear',
                                  padding_mode='zeros', align_corners=True)
            state = (state_f + (warped - ident)).to(state.dtype)
        return super()._align_with_motion(
            state, feat, offset_mask_conv, deform_conv, motion_conv, e_mot)

    # ------------------------------------------------------------------
    def forward(self, feat, velocity=None, use_posterior=True,
                deterministic=True, detach_state=True):
        out = super().forward(
            feat, velocity=velocity, use_posterior=use_posterior,
            deterministic=deterministic, detach_state=detach_state)
        stats = out[5]
        solve = self._ego_solve
        if solve is not None:
            stats['stat_ego_vx'] = solve['v_ego'][:, 0].mean().detach()
            stats['stat_ego_vy'] = solve['v_ego'][:, 1].mean().detach()
            stats['stat_ego_speed'] = solve['v_ego'].norm(dim=-1).mean() \
                .detach()
            stats['stat_ego_static_frac'] = solve['static_frac']
            stats['stat_ego_trim_frac'] = solve['trim_frac'].mean().detach()
            stats['stat_ego_tau_r'] = solve['tau_r'].mean().detach()
            stats['stat_ego_solved'] = solve['solved'].float().mean() \
                .detach()
            stats['stat_ego_warp_scale'] = self.warp_scale.detach()
            stats['stat_ego_dt'] = self.dt.detach()
        return out
