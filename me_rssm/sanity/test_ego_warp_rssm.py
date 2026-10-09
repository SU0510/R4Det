"""Unit tests for the DC-RSSM-2 analytic ego-warp (docs/13 L1).

Run (a user decision, same as the other sanity suites):
    python me_rssm/sanity/test_ego_warp_rssm.py

Covers:
  W1  switch-off: use_ego_warp=False reproduces DC-RSSM-1 bitwise
      (same class hierarchy, no new parameters constructed);
  W2  identity start: at init (warp_scale == 0) the warp is a BITWISE
      no-op -- the full forward equals the switch-off variant bitwise on
      a carried state (frame 2, not just the zero-state frame 1);
  W3  closed-form solve correctness: a synthetic pure ego-flow canvas
      (v_r(p) = -(v_ego . u_r(p)) on every measured cell) is solved back
      to v_ego, per frame independently (batch independence);
  W4  static-set selection: cells outside the rigid flow (moving
      object blob) and mixed pillars (high v_std) are excluded by the
      per-frame quantile gates -- v_ego is unaffected;
  W5  warp direction: the grid semantics h_warped(p) = h_state(p + d)
      (content moves by -d cells for a positive d), verified on a spike;
  W6  deadlock discipline (5th instance of the docs/13 §4 rule): zero-
      init warp_scale still receives gradient from the first NON-ZERO
      step (zero state on frame 1 gives exactly zero gradient), and dt's
      gradient is chained through warp_scale -- exactly zero until
      warp_scale moves off 0, then arrives;
  W7  degeneracy guards: rank-deficient geometry (no lateral spread)
      and too-few-measured-cells both fall back to v_ego = 0 (identity);
  W8  checkpoint compatibility: DC-RSSM-1 checkpoints load with exactly
      the two new keys (dt, warp_scale) as the only delta;
  W9  stats: stat_ego_* emitted, warp_scale/dt reported, solved fraction;
  W10 warp_z_state switch: z_state skips the warp when False (bitwise
      equals the switch-off twin) and is warped when True.

Conventions follow test_evidence_field_rssm.py (unittest, script-runnable,
small dims, non-square feature H=8 W=10 against transposition bugs). The
test canvas is built in the REAL pipeline's orientation convention:
canvas (B, 7, 2W, 2H) with dim2 = world y (lateral) and dim3 = world x
(forward), i.e. feat cell (i, j) [i = x index, j = y index] lives in
canvas[:, :, 2j:2j+2, 2i:2i+2] -- the same cross mapping the scatter
produces (R4Det.py:871-873 assert).
"""

import os
import sys
import unittest

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from me_rssm.modality_bus import ModalityContext  # noqa: E402

PC_RANGE = [0.0, -4.0, 0.0, 8.0, 4.0, 2.0]  # x_min, y_min, ..., x_max, y_max
H_FEAT, W_FEAT = 8, 10
CELL_X = (PC_RANGE[3] - PC_RANGE[0]) / H_FEAT  # 1.0
CELL_Y = (PC_RANGE[4] - PC_RANGE[1]) / W_FEAT  # 0.8


def _common_kwargs(**overrides):
    kwargs = dict(
        in_channels=32,
        out_channels=32,
        latent_dim=32,
        hidden_dim=16,
        action_dim=0,
        kl_scale=0.1,
        free_nats=1.0,
        min_std=0.1,
        init_std=0.2,
        obs_pred_loss_weight=0.05,
        align_kernel_size=3,
        align_deform_groups=1,
        align_z_state=True,
        posterior_mode='innovation',
        innov_in_gain=True,
        use_motion_offsets=True,
        use_dynamic_gate=True,
        use_reliability_gain=True,
        use_gated_readout=True,
        use_modality_obs=True,
        motion_channels=7,
        camera_bev_channels=32,
        motion_hidden=16,
        motion_latent=16,
        camera_hidden=16,
        camera_latent=16,
        motion_grad=False,
        modality_dropout=0.0,
        state_init='zero',
        use_physical_evidence=True,
        evidence_refine=True,
    )
    kwargs.update(overrides)
    return kwargs


def _build_ego(**overrides):
    from me_rssm.ego_warp_rssm import EgoWarpEvidenceFieldRSSMFusion

    kwargs = _common_kwargs(
        use_ego_warp=True, pc_range=list(PC_RANGE),
        min_static_cells=16, dt_init=0.1)
    kwargs.update(overrides)
    fusion = EgoWarpEvidenceFieldRSSMFusion(**kwargs)
    fusion.eval()
    return fusion


def _build_dcrssm1(**overrides):
    from me_rssm.evidence_field_rssm import (
        EvidenceFieldInnovationRSSMFusion)

    kwargs = _common_kwargs(**overrides)
    fusion = EvidenceFieldInnovationRSSMFusion(**kwargs)
    fusion.eval()
    return fusion


def _cell_centers(pc_range=PC_RANGE, H=H_FEAT, W=W_FEAT, device='cpu'):
    cell_x = (pc_range[3] - pc_range[0]) / H
    cell_y = (pc_range[4] - pc_range[1]) / W
    xs = pc_range[0] + (torch.arange(H, device=device).float() + 0.5) * cell_x
    ys = pc_range[1] + (torch.arange(W, device=device).float() + 0.5) * cell_y
    return xs, ys, cell_x, cell_y


def flow_canvas(v0, B=1, count=1.0, snr=1.0, v_std=0.0, pc_range=PC_RANGE):
    """Raw Doppler canvas with the exact static-scene flow of ego v0.

    v_r(p) = -(v0 . u_r(p)) on every cell (approaching negative, docs/08
    F1), written into channels in the real scatter's orientation:
    canvas (B, 7, 2W, 2H), feat cell (i=x, j=y) <- block
    [:, :, 2j:2j+2, 2i:2i+2].
    """
    H, W = H_FEAT, W_FEAT
    xs, ys, _, _ = _cell_centers(pc_range, H, W)
    X = xs[:, None].expand(H, W)
    Y = ys[None, :].expand(H, W)
    r = torch.sqrt(X * X + Y * Y).clamp(min=1e-6)
    ux, uy = X / r, Y / r
    v_r = -(v0[0] * ux + v0[1] * uy)
    canvas = torch.zeros(B, 7, 2 * W, 2 * H)
    for i in range(H):
        for j in range(W):
            canvas[:, 2, 2 * j:2 * j + 2, 2 * i:2 * i + 2] = v_r[i, j]
            canvas[:, 3, 2 * j:2 * j + 2, 2 * i:2 * i + 2] = v_r[i, j].abs()
            canvas[:, 4, 2 * j:2 * j + 2, 2 * i:2 * i + 2] = v_std
            canvas[:, 5, 2 * j:2 * j + 2, 2 * i:2 * i + 2] = snr
            canvas[:, 6, 2 * j:2 * j + 2, 2 * i:2 * i + 2] = count
    return canvas


def set_bus(canvas, C_cam=32):
    ModalityContext.set('velocity_bev', canvas)
    ModalityContext.set(
        'camera_bev', torch.randn(canvas.shape[0], C_cam, W_FEAT, H_FEAT))


def copy_weights(src, dst):
    return dst.load_state_dict(src.state_dict(), strict=False)


class TestEgoWarpSwitchOff(unittest.TestCase):
    """W1: use_ego_warp=False reproduces DC-RSSM-1 exactly."""

    def test_switch_off_bitwise_equals_dcrssm1(self):
        torch.manual_seed(7)
        base = _build_dcrssm1()
        torch.manual_seed(7)
        ego = _build_ego(use_ego_warp=False)
        copy_weights(base, ego)
        for _ in range(2):
            feat = torch.randn(2, 32, H_FEAT, W_FEAT)
            set_bus(flow_canvas((2.0, 0.5), B=2))
            base.reset_state()
            ego.reset_state()
            with torch.no_grad():
                out_base = base(feat, use_posterior=True, deterministic=True)
                out_ego = ego(feat, use_posterior=True, deterministic=True)
            for j in (0, 1, 3, 4):
                self.assertTrue(torch.equal(out_base[j], out_ego[j]))
            self.assertNotIn('stat_ego_vx', out_ego[5])
        ModalityContext.clear()

    def test_switch_off_constructs_no_warp_params(self):
        ego = _build_ego(use_ego_warp=False)
        self.assertFalse(hasattr(ego, 'dt'))
        self.assertFalse(hasattr(ego, 'warp_scale'))

    def test_warp_requires_pc_range(self):
        with self.assertRaises(ValueError):
            _build_ego(pc_range=None)


class TestEgoWarpIdentityStart(unittest.TestCase):
    """W2: warp_scale == 0 -> bitwise no-op in the full forward."""

    def test_identity_start_bitwise_on_carried_state(self):
        """Two frames, state carried from a NON-zero frame 1: the warp of
        frame 2 must still be bitwise identity at warp_scale == 0."""
        torch.manual_seed(11)
        base = _build_dcrssm1()
        torch.manual_seed(11)
        ego = _build_ego()
        copy_weights(base, ego)
        self.assertEqual(float(ego.warp_scale.detach()), 0.0)
        base.reset_state()
        ego.reset_state()
        for frame in range(2):
            feat = torch.randn(2, 32, H_FEAT, W_FEAT) + 0.5
            set_bus(flow_canvas((3.0, -1.0), B=2))
            with torch.no_grad():
                out_base = base(feat, use_posterior=True, deterministic=True)
                out_ego = ego(feat, use_posterior=True, deterministic=True)
            for j in (0, 1, 3, 4):
                self.assertTrue(
                    torch.equal(out_base[j], out_ego[j]),
                    msg='frame %d output %d not bitwise identical' % (frame, j))
            # the solve ran (moving canvas) and the grid was built
            self.assertIsNotNone(ego._ego_grid)
            self.assertGreater(
                float(ego._ego_solve['v_ego'].norm(dim=-1).min()), 1.0)
        ModalityContext.clear()

    def test_warp_becomes_active_off_identity(self):
        """Sanity complement of W2: warp_scale != 0 must CHANGE the
        output (the warp is not silently dead)."""
        torch.manual_seed(13)
        base = _build_dcrssm1()
        torch.manual_seed(13)
        ego = _build_ego()
        copy_weights(base, ego)
        ego.warp_scale.data.fill_(1.0)
        base.reset_state()
        ego.reset_state()
        for _ in range(2):
            feat = torch.randn(2, 32, H_FEAT, W_FEAT) + 0.5
            set_bus(flow_canvas((3.0, -1.0), B=2))
            with torch.no_grad():
                out_base = base(feat, use_posterior=True, deterministic=True)
                out_ego = ego(feat, use_posterior=True, deterministic=True)
        self.assertFalse(torch.allclose(out_base[0], out_ego[0]))
        ModalityContext.clear()


class TestEgoWarpSolve(unittest.TestCase):
    """W3/W4: closed-form solve correctness and static-set selection."""

    def test_solve_recovers_known_ego_velocity(self):
        ego = _build_ego()
        v0 = (3.0, -2.0)
        set_bus(flow_canvas(v0, B=1))
        feat = torch.randn(1, 32, H_FEAT, W_FEAT)
        ego._read_context(feat)
        v = ego._ego_solve['v_ego']
        self.assertTrue(bool(ego._ego_solve['solved'][0]))
        self.assertAlmostEqual(v[0, 0].item(), v0[0], places=3)
        self.assertAlmostEqual(v[0, 1].item(), v0[1], places=3)
        ModalityContext.clear()

    def test_solve_is_per_frame_independent(self):
        ego = _build_ego()
        c0 = flow_canvas((3.0, -2.0), B=1)
        c1 = flow_canvas((0.5, 1.0), B=1)
        set_bus(torch.cat([c0, c1], dim=0))
        feat = torch.randn(2, 32, H_FEAT, W_FEAT)
        ego._read_context(feat)
        v = ego._ego_solve['v_ego']
        self.assertAlmostEqual(v[0, 0].item(), 3.0, places=3)
        self.assertAlmostEqual(v[0, 1].item(), -2.0, places=3)
        self.assertAlmostEqual(v[1, 0].item(), 0.5, places=3)
        self.assertAlmostEqual(v[1, 1].item(), 1.0, places=3)
        ModalityContext.clear()

    def test_moving_blob_excluded_by_speed_quantile(self):
        """A 10% blob of fast off-flow cells is cut by tau_v."""
        ego = _build_ego()
        v0 = (2.0, 0.0)
        canvas = flow_canvas(v0, B=1)
        # blob: the 8 highest-|speed| cells (top 10% of 80) at ~50 m/s
        H, W = H_FEAT, W_FEAT
        blob = [(3, 3), (3, 4), (4, 3), (4, 4), (5, 5), (5, 6), (6, 5), (6, 6)]
        for (i, j) in blob:
            for th in (2, 3):
                canvas[:, th, 2 * j:2 * j + 2, 2 * i:2 * i + 2] = 50.0
        set_bus(canvas)
        feat = torch.randn(1, 32, H_FEAT, W_FEAT)
        ego._read_context(feat)
        v = ego._ego_solve['v_ego']
        self.assertAlmostEqual(v[0, 0].item(), v0[0], places=3)
        self.assertAlmostEqual(v[0, 1].item(), v0[1], places=3)
        ModalityContext.clear()

    def test_mixed_pillar_blob_excluded_by_vstd_quantile(self):
        """A 10% blob of high-dispersion (mixed) pillars carrying an
        off-flow velocity is cut by tau_s even though its speed gate
        would pass."""
        ego = _build_ego()
        v0 = (2.0, 0.0)
        canvas = flow_canvas(v0, B=1)
        blob = [(3, 3), (3, 4), (4, 3), (4, 4), (5, 5), (5, 6), (6, 5), (6, 6)]
        for (i, j) in blob:
            canvas[:, 2, 2 * j:2 * j + 2, 2 * i:2 * i + 2] = 1.0  # small speed
            canvas[:, 3, 2 * j:2 * j + 2, 2 * i:2 * i + 2] = 1.0
            canvas[:, 4, 2 * j:2 * j + 2, 2 * i:2 * i + 2] = 5.0  # mixed
        set_bus(canvas)
        feat = torch.randn(1, 32, H_FEAT, W_FEAT)
        ego._read_context(feat)
        v = ego._ego_solve['v_ego']
        self.assertAlmostEqual(v[0, 0].item(), v0[0], places=3)
        self.assertAlmostEqual(v[0, 1].item(), v0[1], places=3)
        ModalityContext.clear()


class TestEgoWarpRobustSolve(unittest.TestCase):
    """W11: the robust static-set pipeline vs the plain weighted fit.

    Mirrors the val-set finding (probe_ego_geometry_test.py: plain fit
    inflated +24%..+96% by strong-return moving cells whose count*SNR
    weight reaches ~30x a static cell's). The synthetic scene below puts
    a coherent one-sided mover group INSIDE the quantile pre-filter band
    (so the pre-filter alone cannot remove it) and checks that the
    LMedS-seed + MAD-trim pipeline recovers the true velocity while the
    plain path is measurably biased -- i.e. the test would fail if the
    fix were reverted.
    """

    def _scene(self, v0=(3.0, 0.0)):
        canvas = flow_canvas(v0, B=1, count=0.1, snr=3.0)
        # coherent mover group on one side, moderate residual (-1.5 m/s
        # along the ray) but heavy returns: w = 0.7*12 = 8.4 vs static
        # 0.1*3 = 0.3 (28x). Their |v_r| stays under tau_v so the
        # quantile pre-filter keeps them by construction.
        xs, ys, _, _ = _cell_centers()
        movers = [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1),
                  (3, 0), (3, 1)]
        for (i, j) in movers:
            static_vr = canvas[0, 2, 2 * j, 2 * i].item()
            for th in (2, 3):
                canvas[:, th, 2 * j:2 * j + 2, 2 * i:2 * i + 2] = \
                    static_vr - 1.5
            canvas[:, 5, 2 * j:2 * j + 2, 2 * i:2 * i + 2] = 12.0
            canvas[:, 6, 2 * j:2 * j + 2, 2 * i:2 * i + 2] = 0.7
        return canvas

    def test_robust_path_recovers_and_plain_path_is_biased(self):
        torch.manual_seed(11)
        v0 = (3.0, 0.0)
        canvas = self._scene(v0)
        feat = torch.randn(1, 32, H_FEAT, W_FEAT)

        ego_plain = _build_ego(robust_solve=False)
        set_bus(canvas)
        ego_plain._read_context(feat)
        v_plain = ego_plain._ego_solve['v_ego'][0]
        ModalityContext.clear()

        ego_robust = _build_ego(robust_solve=True)
        set_bus(canvas)
        ego_robust._read_context(feat)
        v_rob = ego_robust._ego_solve['v_ego'][0]
        trim_frac = ego_robust._ego_solve['trim_frac'][0].item()
        ModalityContext.clear()

        err_plain = (v_plain - torch.tensor(v0)).abs().max().item()
        err_rob = (v_rob - torch.tensor(v0)).abs().max().item()
        # the plain fit is measurably dragged by the mover group ...
        self.assertGreater(
            err_plain, 0.15,
            f'plain path was not exercised (err {err_plain:.4f})')
        # ... the robust path recovers the true velocity ...
        self.assertLess(err_rob, 0.05,
                        f'robust solve error {err_rob:.4f}')
        # ... by trimming exactly the mover group: the tau_v quantile
        # pre-filter cuts the 8 highest-|v_r| cells (mostly the axial
        # static ones plus the fastest movers), leaving sel0 = 72 cells
        # with all 8 movers still inside; the MAD trim then removes
        # exactly those 8, i.e. trim_frac = 64/72 (trim_frac is
        # normalised over sel0, not over the 80 measured cells).
        self.assertAlmostEqual(trim_frac, 64.0 / 72.0, places=3)


class TestEgoWarpDirection(unittest.TestCase):
    """W5: h_warped(p) = h_state(p + d) -- content moves by -d cells."""
    def _spike_test(self, v0, spike_ij, expected_ij):
        ego = _build_ego()
        ego.warp_scale.data.fill_(1.0)
        ego.dt.data.fill_(1.0)
        set_bus(flow_canvas(v0, B=1))
        feat = torch.randn(1, 32, H_FEAT, W_FEAT)
        ego._read_context(feat)
        self.assertIsNotNone(ego._ego_grid)
        state = torch.zeros(1, 4, H_FEAT, W_FEAT)
        state[0, :, spike_ij[0], spike_ij[1]] = 1.0
        warped = F.grid_sample(
            state, ego._ego_grid, mode='bilinear',
            padding_mode='zeros', align_corners=True)
        peak = warped[0, 0]
        ii, jj = torch.meshgrid(torch.arange(H_FEAT), torch.arange(W_FEAT))
        flat = int(peak.argmax())
        got = (flat // W_FEAT, flat % W_FEAT)
        self.assertEqual(got, expected_ij)
        self.assertGreater(float(peak.max()), 0.5)
        ModalityContext.clear()

    def test_forward_motion_shifts_content_backward(self):
        # ego +x at 2 m/s, dt = 1 s -> 2 cells; spike at (4, 5) -> (2, 5)
        self._spike_test((2.0, 0.0), (4, 5), (2, 5))

    def test_backward_motion_shifts_content_forward(self):
        # ego -x at 2 m/s -> spike at (4, 5) -> (6, 5)
        self._spike_test((-2.0, 0.0), (4, 5), (6, 5))


class TestEgoWarpGradients(unittest.TestCase):
    """W6: deadlock discipline (5th instance of the docs/13 §4 rule)."""

    def test_warp_scale_grad_arrives_and_dt_grad_follows(self):
        ego = _build_ego()
        ego.train()
        opt = torch.optim.SGD(ego.parameters(), lr=0.5)
        ws_grads, dt_grads = [], []
        for _ in range(3):
            feat = torch.randn(2, 32, H_FEAT, W_FEAT) + 0.5
            set_bus(flow_canvas((3.0, -1.0), B=2))
            out, _, latent_loss, _, _, _ = ego(
                feat, use_posterior=True, deterministic=True)
            opt.zero_grad()
            (out.mean() + latent_loss).backward()
            ws_grads.append(
                None if ego.warp_scale.grad is None
                else float(ego.warp_scale.grad.abs()))
            dt_grads.append(
                None if ego.dt.grad is None
                else float(ego.dt.grad.abs()))
            opt.step()
        ModalityContext.clear()
        # step 1: h_state is all-zero (state_init='zero') -> warping zeros
        # gives exactly zero grid gradient ...
        self.assertIsNotNone(ws_grads[0])
        self.assertAlmostEqual(ws_grads[0], 0.0, places=8)
        # ... but from the first carried (non-zero) step the grid gradient
        # is the state's local spatial difference: nonzero -> no deadlock
        self.assertGreater(ws_grads[1], 0.0)
        self.assertGreater(ws_grads[2], 0.0)
        # dt is chained through warp_scale: exactly zero until warp_scale
        # moves off 0, then arrives (the L3-refine-conv pattern)
        self.assertIsNotNone(dt_grads[0])
        self.assertAlmostEqual(dt_grads[0], 0.0, places=8)
        self.assertAlmostEqual(dt_grads[1], 0.0, places=8)
        self.assertGreater(dt_grads[2], 0.0)


class TestEgoWarpDegeneracy(unittest.TestCase):
    """W7: degenerate solves fall back to the identity warp."""

    def test_rank_deficient_geometry_falls_back(self):
        # razor-thin y range -> no lateral spread -> v_y unidentifiable
        # -> det(M) ~ 0 -> fallback v = 0
        pc = [5.0, -0.01, 0.0, 13.0, 0.01, 2.0]
        ego = _build_ego(pc_range=pc)
        set_bus(flow_canvas((2.0, 0.0), B=1, pc_range=pc))
        feat = torch.randn(1, 32, H_FEAT, W_FEAT)
        ego._read_context(feat)
        self.assertFalse(bool(ego._ego_solve['solved'][0]))
        self.assertTrue(torch.allclose(
            ego._ego_solve['v_ego'], torch.zeros(1, 2)))
        ModalityContext.clear()

    def test_too_few_measured_cells_falls_back(self):
        ego = _build_ego()
        canvas = torch.zeros(1, 7, 2 * W_FEAT, 2 * H_FEAT)
        canvas[:, 2, 0:2, 0:2] = 5.0
        canvas[:, 3, 0:2, 0:2] = 5.0
        canvas[:, 6, 0:2, 0:2] = 1.0  # only one measured cell
        set_bus(canvas)
        feat = torch.randn(1, 32, H_FEAT, W_FEAT)
        ego._read_context(feat)
        self.assertFalse(bool(ego._ego_solve['solved'][0]))
        self.assertTrue(torch.allclose(
            ego._ego_solve['v_ego'], torch.zeros(1, 2)))
        ModalityContext.clear()

    def test_dead_bus_is_identity(self):
        ego = _build_ego()
        ModalityContext.clear()
        feat = torch.randn(1, 32, H_FEAT, W_FEAT)
        ego._read_context(feat)
        self.assertIsNone(ego._ego_grid)
        self.assertIsNone(ego._ego_solve)
        ego.train()
        out = ego(feat, use_posterior=True, deterministic=True)
        self.assertNotIn('stat_ego_vx', out[5])


class TestEgoWarpCheckpointCompatibility(unittest.TestCase):
    """W8: DC-RSSM-1 checkpoints load with exactly two new keys."""

    def test_dcrssm1_checkpoint_loads_into_warp_variant(self):
        base = _build_dcrssm1()
        ego = _build_ego()
        missing, unexpected = copy_weights(base, ego)
        self.assertEqual(list(unexpected), [])
        self.assertEqual(sorted(missing), ['dt', 'warp_scale'])

    def test_switch_off_state_dict_has_no_warp_keys(self):
        base = _build_dcrssm1()
        ego = _build_ego(use_ego_warp=False)
        missing, unexpected = copy_weights(base, ego)
        self.assertEqual(list(unexpected), [])
        self.assertEqual(list(missing), [])
        self.assertEqual(sorted(ego.state_dict().keys()),
                         sorted(base.state_dict().keys()))


class TestEgoWarpStats(unittest.TestCase):
    """W9: stat_ego_* diagnostics."""

    def test_stats_emitted_with_sane_values(self):
        ego = _build_ego()
        ego.train()
        feat = torch.randn(1, 32, H_FEAT, W_FEAT)
        set_bus(flow_canvas((2.0, -1.0), B=1))
        out = ego(feat, use_posterior=True, deterministic=True)
        stats = out[5]
        for k in ('stat_ego_vx', 'stat_ego_vy', 'stat_ego_speed',
                  'stat_ego_static_frac', 'stat_ego_solved',
                  'stat_ego_warp_scale', 'stat_ego_dt'):
            self.assertIn(k, stats)
        self.assertAlmostEqual(stats['stat_ego_vx'].item(), 2.0, places=3)
        self.assertAlmostEqual(stats['stat_ego_vy'].item(), -1.0, places=3)
        self.assertAlmostEqual(
            stats['stat_ego_speed'].item(), (5.0 ** 0.5), places=3)
        self.assertAlmostEqual(stats['stat_ego_solved'].item(), 1.0)
        self.assertAlmostEqual(stats['stat_ego_warp_scale'].item(), 0.0)
        self.assertAlmostEqual(stats['stat_ego_dt'].item(), 0.1, places=6)
        ModalityContext.clear()


class TestEgoWarpZState(unittest.TestCase):
    """W10: warp_z_state switch on the z alignment path."""

    def _align_z(self, fusion, feat):
        return fusion._align_with_motion(
            fusion.z_state, feat,
            fusion.align_z_offset_mask, fusion.align_z_deform_conv,
            fusion.motion_offset_z if fusion.use_motion_offsets else None,
            None)

    def test_z_warp_follows_switch(self):
        torch.manual_seed(23)
        base = _build_dcrssm1()
        torch.manual_seed(23)
        on = _build_ego(warp_z_state=True)
        torch.manual_seed(23)
        off = _build_ego(warp_z_state=False)
        copy_weights(base, on)
        copy_weights(base, off)
        for m in (on, off):
            m.warp_scale.data.fill_(1.0)
            m.dt.data.fill_(1.0)
        feat = torch.randn(1, 32, H_FEAT, W_FEAT) + 0.5
        set_bus(flow_canvas((2.0, 0.0), B=1))
        z = torch.randn(1, 32, H_FEAT, W_FEAT)
        base.reset_state()
        base.z_state = z.clone()
        on.z_state = z.clone()
        off.z_state = z.clone()
        for m in (base, on, off):
            m._read_context(feat)
        with torch.no_grad():
            out_base = self._align_z(base, feat)
            out_on = self._align_z(on, feat)
            out_off = self._align_z(off, feat)
        # switch off: the z path is bitwise the DC-RSSM-1 path
        self.assertTrue(torch.equal(out_base, out_off))
        # switch on: the warp is active -> different result
        self.assertFalse(torch.allclose(out_base, out_on))
        ModalityContext.clear()

    def test_h_state_always_warped_even_when_warp_z_off(self):
        torch.manual_seed(29)
        base = _build_dcrssm1()
        torch.manual_seed(29)
        off = _build_ego(warp_z_state=False)
        copy_weights(base, off)
        off.warp_scale.data.fill_(1.0)
        off.dt.data.fill_(1.0)
        feat = torch.randn(1, 32, H_FEAT, W_FEAT) + 0.5
        set_bus(flow_canvas((2.0, 0.0), B=1))
        h = torch.randn(1, 32, H_FEAT, W_FEAT)
        base.reset_state()
        off.reset_state()
        base.h_state = h.clone()
        off.h_state = h.clone()
        for m in (base, off):
            m._read_context(feat)
        with torch.no_grad():
            out_base = base._align_with_motion(
                base.h_state, feat, base.align_h_offset_mask,
                base.align_h_deform_conv,
                base.motion_offset_h if base.use_motion_offsets else None,
                None)
            out_off = off._align_with_motion(
                off.h_state, feat, off.align_h_offset_mask,
                off.align_h_deform_conv,
                off.motion_offset_h if off.use_motion_offsets else None,
                None)
        self.assertFalse(torch.allclose(out_base, out_off))
        ModalityContext.clear()


class TestEgoWarpToggleHarness(unittest.TestCase):
    """W12: the ep16 toggle probe's helpers must agree with the module.

    probe_ego_alignment_toggle re-derives the static-cell solve and the
    identity-anchored warp outside the module; if either mirror drifts,
    the pre-registered ep16 mechanism gate would measure the wrong thing.
    The GPU-heavy three-pass harness cannot be self-tested while the 24e
    run owns GPUs 5/6/7, so its two mirror functions and its per-forward
    call-grouping assumption are pinned here instead:
      (a) mirror solve == module's own stash (v_ego, tau_r, solved);
      (b) mirror warp == the analytic displacement at warp_scale=1;
      (c) one forward issues exactly 2 alignment calls (h then z) and
          call[0] carries the h state -- the probe indexes calls[0].
    """

    def test_mirror_solve_equals_module_stash(self):
        from me_rssm.sanity.probe_ego_alignment_toggle import (
            static_mask_from_canvas)

        for v0 in ((3.0, -1.5), (1.2, 0.4)):
            ego = _build_ego()
            canvas = flow_canvas(v0, B=1)
            set_bus(canvas)
            feat = torch.randn(1, 32, H_FEAT, W_FEAT)
            ego._read_context(feat)
            stash = ego._ego_solve
            self.assertIsNotNone(stash)
            self.assertTrue(bool(stash['solved'].item()))
            mask, v_ego, tau_r = static_mask_from_canvas(
                canvas, PC_RANGE, 0.9, 16, ego)
            self.assertIsNotNone(mask)
            self.assertGreater(int(mask.sum().item()), 0)
            # the mirror runs the same four stages, so it must reproduce
            # the module's final weighted solve, not merely the grid seed
            self.assertAlmostEqual(float(v_ego[0]),
                                   float(stash['v_ego'][0, 0]), places=5)
            self.assertAlmostEqual(float(v_ego[1]),
                                   float(stash['v_ego'][0, 1]), places=5)
            self.assertAlmostEqual(tau_r, float(stash['tau_r'][0]),
                                   places=6)
            # and the mirror is anchored to the analytic truth
            self.assertAlmostEqual(float(v_ego[0]), v0[0], delta=0.25)
            self.assertAlmostEqual(float(v_ego[1]), v0[1], delta=0.25)
            ModalityContext.clear()
            _ = mask

    def test_mirror_warp_matches_analytic_shift(self):
        from me_rssm.sanity.probe_ego_alignment_toggle import (
            apply_module_warp)

        ego = _build_ego()
        ego.warp_scale.data.fill_(1.0)
        ego.dt.data.fill_(0.1)
        set_bus(flow_canvas((2.0, 0.0), B=1))
        feat = torch.randn(1, 32, H_FEAT, W_FEAT)
        ego._read_context(feat)
        # identity when the scalar is zero: grid == id_grid bitwise, so
        # the anchored form returns the state untouched bitwise
        ego.warp_scale.data.fill_(0.0)
        ego._read_context(feat)
        state = torch.randn(1, 8, H_FEAT, W_FEAT)
        ident = apply_module_warp(state, ego._ego_grid, ego._ego_id_grid)
        self.assertTrue(torch.equal(ident, state))
        ego.warp_scale.data.fill_(1.0)
        ego._read_context(feat)
        i0 = 4
        state = torch.zeros(1, 8, H_FEAT, W_FEAT)
        state[:, :, i0, :] = 1.0
        warped = apply_module_warp(state, ego._ego_grid, ego._ego_id_grid)
        # dx = 1.0 * (2.0 * 0.1) / 1.0 cell = 0.2 cells: the sampled
        # source index moves ahead, content moves back by 0.2
        self.assertAlmostEqual(float(warped[:, :, i0, :].mean()), 0.8,
                               places=5)
        self.assertAlmostEqual(float(warped[:, :, i0 - 1, :].mean()), 0.2,
                               places=5)
        other = [i for i in range(H_FEAT) if i not in (i0, i0 - 1)]
        self.assertAlmostEqual(
            float(warped[:, :, other, :].abs().max()), 0.0, places=6)
        ModalityContext.clear()

    def test_align_calls_grouped_per_forward(self):
        """One forward -> 2 calls (h then z); call[0] carries h_state."""
        ego = _build_ego()
        calls = []

        def pre_hook(module, inputs):
            calls.append([])

        def post_hook(module, inputs, output):
            pass

        orig = ego._align_with_motion

        def wrapped(state, feat, *args, **kwargs):
            if calls:
                calls[-1].append(state.detach().clone())
            return orig(state, feat, *args, **kwargs)

        ego._align_with_motion = wrapped
        h_pre = ego.register_forward_pre_hook(pre_hook)
        h_post = ego.register_forward_hook(post_hook)
        try:
            feat = torch.randn(1, 32, H_FEAT, W_FEAT)
            set_bus(flow_canvas((2.0, 0.0), B=1))
            ego.reset_state()
            with torch.no_grad():
                ego(feat, use_posterior=True, deterministic=True)
            feat2 = torch.randn(1, 32, H_FEAT, W_FEAT)
            set_bus(flow_canvas((2.0, 0.0), B=1))
            # mark the two states distinctly: on frame 2 the states the
            # module carries into the alignment are exactly these
            ego.h_state = torch.full_like(ego.h_state, 1.0)
            ego.z_state = torch.full_like(ego.z_state, -1.0)
            with torch.no_grad():
                ego(feat2, use_posterior=True, deterministic=True)
        finally:
            ego._align_with_motion = orig
            h_pre.remove()
            h_post.remove()
        self.assertEqual(len(calls), 2)
        self.assertEqual([len(c) for c in calls], [2, 2])
        first, second = calls[1]
        self.assertAlmostEqual(float(first.mean()), 1.0, places=6)
        self.assertAlmostEqual(float(second.mean()), -1.0, places=6)
        ModalityContext.clear()


if __name__ == '__main__':
    unittest.main(verbosity=2)
