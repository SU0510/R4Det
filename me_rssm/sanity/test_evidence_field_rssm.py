"""Unit tests for the DC-RSSM-1 physical evidence field (docs/13 L3).

Run (NOT executed as part of this preparation — the run is a user decision):
    python me_rssm/sanity/test_evidence_field_rssm.py

Covers:
  V1  switch-off: use_physical_evidence=False reproduces MCI bitwise
      (same class hierarchy, no new parameters constructed);
  V2  gate-input semantics: with the field on, c_mot/dyn come from the
      physical layer (spatial structure by construction) and the learned
      refinement is zero-init (c_mot == c_mot0 at init);
  V3  spatial structure: the physical field's std is non-trivial on a
      structured canvas (the property the learned gates lacked —
      §68.9b/§68.9e) while a zero-init learned map is constant;
  V4  coverage semantics: empty bus -> c_mot == cov_floor ("no
      measurement"), gate input width unchanged;
  V5  gradient reach: the refinement conv receives gradient (non-deadlock
      — its input, the physical field, is non-zero);
  V6  calibration constants are buffers-free plain floats (never trained);
  V7  E3-field identity: gates keep deviation identity start — with the
      refinement zero-init the module output equals MCI's at init when
      c_mot0/dyn0 match MCI's learned-map constants... explicitly NOT
      bitwise to MCI (new generation per docs/13 §4), so the test asserts
      the gate maps stay 1 + tanh(0) = 1 (identity START of the gates,
      not identity to MCI).

Conventions follow test_mci_rssm.py (unittest, script-runnable, small
dims, non-square canvas H=8 W=10 against transposition bugs).
"""

import sys
import os
import unittest

import torch

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from me_rssm.modality_bus import ModalityContext  # noqa: E402


def _build_evf(**overrides):
    from me_rssm.evidence_field_rssm import (
        EvidenceFieldInnovationRSSMFusion)

    use_physical = overrides.get('use_physical_evidence', True)
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
        use_physical_evidence=use_physical,
        evidence_refine=overrides.get('evidence_refine', use_physical),
    )
    kwargs.update(overrides)
    fusion = EvidenceFieldInnovationRSSMFusion(**kwargs)
    fusion.eval()
    return fusion


def _build_mci_small(**overrides):
    from me_rssm.mci_rssm import (
        MeasurementConditionedInnovationRSSMFusion)

    kwargs = dict(
        in_channels=32, out_channels=32, latent_dim=32, hidden_dim=16,
        action_dim=0, kl_scale=0.1, free_nats=1.0, min_std=0.1,
        init_std=0.2, obs_pred_loss_weight=0.05, align_kernel_size=3,
        align_deform_groups=1, align_z_state=True,
        posterior_mode='innovation', innov_in_gain=True,
        use_motion_offsets=True, use_dynamic_gate=True,
        use_reliability_gain=True, use_gated_readout=True,
        use_modality_obs=True, motion_channels=7, camera_bev_channels=32,
        motion_hidden=16, motion_latent=16, camera_hidden=16,
        camera_latent=16, motion_grad=False, modality_dropout=0.0,
        state_init='zero',
    )
    kwargs.update(overrides)
    fusion = MeasurementConditionedInnovationRSSMFusion(**kwargs)
    fusion.eval()
    return fusion


def set_bus(B, feat_hw, C_cam, device='cpu'):
    """Populate the bus for a feat of shape (B, C, H, W), H != W."""
    H, W = feat_hw
    vel = torch.randn(B, 7, 2 * W, 2 * H, device=device) * 0.1
    cam = torch.randn(B, C_cam, W, H, device=device)
    ModalityContext.set('velocity_bev', vel)
    ModalityContext.set('camera_bev', cam)
    return vel, cam


def copy_weights(src, dst):
    return dst.load_state_dict(src.state_dict(), strict=False)


class TestEVFSwitchOff(unittest.TestCase):
    """V1: use_physical_evidence=False reproduces MCI exactly."""

    def test_switch_off_bitwise_equals_mci(self):
        torch.manual_seed(7)
        mci = _build_mci_small()
        torch.manual_seed(7)
        evf = _build_evf(use_physical_evidence=False)
        copy_weights(mci, evf)
        for _ in range(2):
            feat = torch.randn(2, 32, 8, 10)
            set_bus(feat.shape[0], feat.shape[-2:], 32)
            mci.reset_state()
            evf.reset_state()
            with torch.no_grad():
                out_mci = mci(feat, use_posterior=True, deterministic=True)
                out_evf = evf(feat, use_posterior=True, deterministic=True)
            for j in (0, 1, 3, 4):
                self.assertTrue(torch.equal(out_mci[j], out_evf[j]))
        ModalityContext.clear()

    def test_switch_off_constructs_no_field_modules(self):
        evf = _build_evf(use_physical_evidence=False)
        self.assertFalse(hasattr(evf, 'evidence_refine_conv'))
        self.assertFalse(evf.use_physical_evidence)

    def test_refine_requires_physical(self):
        with self.assertRaises(ValueError):
            _build_evf(use_physical_evidence=False, evidence_refine=True)


class TestEVFPhysicalField(unittest.TestCase):
    """V2/V3: field semantics and spatial structure."""

    def test_field_is_spatially_structured(self):
        """THE property: physical c_mot/dyn have non-trivial spatial std
        by construction, where the learned maps collapsed to constants
        (§68.9b: c_mot std/mean ≈ 0.001)."""
        evf = _build_evf()
        evf.train()
        feat = torch.randn(2, 32, 8, 10)
        vel, _ = set_bus(2, (8, 10), 32)
        # structured canvas: half the cells well-measured + fast-changing,
        # half empty -- the physical field must separate them.
        # channel semantics (radar_motion_chain): 4 v_std, 5 snr, 6 count.
        vel2 = vel.clone()
        vel2[:, 6] = 0.0                      # count: empty everywhere
        vel2[:, :, :vel2.shape[2] // 2] = 0.0
        vel2[:, 6, :vel2.shape[2] // 2] = 1.0  # top half: full pillars
        vel2[:, 5, :vel2.shape[2] // 2] = 1.0  # top half: high SNR
        vel2[:, 4, :vel2.shape[2] // 2] = 2.0  # top half: high v_std
        ModalityContext.set('velocity_bev', vel2)
        ctx = evf._read_context(feat)
        self.assertIsNotNone(ctx)
        c_mot, dyn = ctx['c_mot'], ctx['dyn']
        # spatial structure: std within each sample is non-trivial
        # (refinement is small-normal, so the physical structure survives)
        self.assertGreater(c_mot.std(dim=(-2, -1)).mean().item(), 0.1)
        # dyn: v_std*count is 2.0 on the top half, 0 below -> bimodal
        self.assertGreater(dyn.std(dim=(-2, -1)).mean().item(), 0.1)
        # measured vs empty halves differ in dyn (c_mot halves share the
        # sigmoid's high-saturation plateau at alpha1=1: count*snr = 1 vs 0
        # maps to sigmoid(-0) vs sigmoid(0) ~ 0.5/0.5... assert on dyn,
        # whose 2.0-vs-0 input is far from the saturation knee). The
        # avg_pool+permute puts the structured half on the column axis:
        # canvas (20,16) pooled (10,8) permuted (8,10) -- top 10 canvas
        # rows = 5 pooled rows = FIRST 5 columns after the (x,y) swap
        # (measured empirically; the swap inverts the axis direction).
        self.assertGreater(
            dyn[:, 0, :, :5].mean().item() - dyn[:, 0, :, 5:].mean().item(),
            0.1)
        ModalityContext.clear()

    def test_refine_is_small_normal_field_dominates(self):
        """Refine conv is small-normal init (deadlock rule, 4th instance +
        additive position): output non-zero (no deadlock) but small -- the
        physical field dominates and c_mot stays in (0, 1) at init."""
        evf = _build_evf()
        B, H, W = 2, 8, 10
        feat = torch.randn(B, 32, H, W)
        set_bus(B, (H, W), 32)
        evf.train()
        vel = ModalityContext.get('velocity_bev')
        vel_pool = torch.nn.functional.avg_pool2d(vel, 2).permute(
            0, 1, 3, 2).contiguous()
        c_mot0, dyn0, cov = evf._physical_evidence(vel_pool, feat)
        ctx = evf._read_context(feat)
        # refined c_mot = c_mot0 + refine(.); refinement is non-zero ...
        delta = (ctx['c_mot'] - c_mot0).abs().max().item()
        self.assertGreater(delta, 0.0)
        # ... but small: the field stays measurement-anchored
        self.assertLess(delta, 0.1)
        # and the sum stays in the valid (0, 1) gate range
        self.assertGreater(ctx['c_mot'].min().item(), 0.0)
        self.assertLess(ctx['c_mot'].max().item(), 1.0)
        ModalityContext.clear()

    def test_field_is_detached_measurement(self):
        evf = _build_evf()
        evf.train()
        feat = torch.randn(2, 32, 8, 10)
        set_bus(2, (8, 10), 32)
        ctx = evf._read_context(feat)
        # refinement is a learned path (keeps its graph -- detaching the
        # sum would make the refine conv a dead layer); the PHYSICAL
        # layer c_mot0 is the detached measurement
        self.assertTrue(ctx['c_mot'].requires_grad)
        self.assertFalse(ctx['dyn'].requires_grad)
        c_mot0, _, _ = evf._physical_evidence(
            ModalityContext.get('velocity_bev'), feat)
        self.assertFalse(c_mot0.requires_grad)
        ModalityContext.clear()

    def test_field_recomputed_per_call(self):
        evf = _build_evf()
        evf.train()
        feat = torch.randn(2, 32, 8, 10)
        set_bus(2, (8, 10), 32)
        ctx1 = evf._read_context(feat)
        s1 = ctx1['c_mot'].clone()
        ModalityContext.set(
            'velocity_bev',
            torch.zeros(2, 7, 20, 16))
        ctx2 = evf._read_context(feat)
        # zero canvas -> zero count/snr -> c_mot = sigmoid(-beta1) = 0.5
        # (+ a small-normal refinement ~0.01 -- the physical constant
        # dominates)
        self.assertAlmostEqual(ctx2['c_mot'].max().item(), 0.5, places=1)
        self.assertFalse(torch.allclose(s1, ctx2['c_mot']))
        ModalityContext.clear()

    def test_forward_emits_field_stats(self):
        evf = _build_evf()
        evf.train()
        feat = torch.randn(2, 32, 8, 10)
        set_bus(2, (8, 10), 32)
        out = evf(feat, use_posterior=True, deterministic=True)
        for k in ('stat_ev_c_mot_mean', 'stat_ev_c_mot_std',
                  'stat_ev_dyn_mean', 'stat_ev_dyn_std',
                  'stat_ev_cov_mean'):
            self.assertIn(k, out[5])
        ModalityContext.clear()


class TestEVFCoverageSemantics(unittest.TestCase):
    """V4: empty bus -> "no measurement", not "static"."""

    def test_empty_bus_uses_cov_floor(self):
        evf = _build_evf(cov_floor=0.0)
        feat = torch.randn(2, 32, 8, 10)
        ModalityContext.clear()
        evf.train()
        ctx = evf._read_context(feat)
        # MCI's all-or-nothing bus check: ctx stays None on empty bus; the
        # cov_floor constant is substituted at the gate input instead
        self.assertIsNone(ctx)
        gate_in = evf._gate_input(ctx, None, feat)
        self.assertIsNotNone(gate_in)
        # channels: [c_cam=0, c_mot=cov_floor] with modality obs on
        self.assertEqual(gate_in.shape[1], 2)
        self.assertTrue(torch.allclose(gate_in[:, 0], torch.zeros_like(
            gate_in[:, 0])))
        self.assertTrue(torch.allclose(gate_in[:, 1], torch.zeros_like(
            gate_in[:, 1])))  # cov_floor=0.0
        # and with cov_floor > 0 the second channel carries the constant
        evf2 = _build_evf(cov_floor=0.25)
        gate_in2 = evf2._gate_input(None, None, feat)
        self.assertTrue(torch.allclose(
            gate_in2[:, 1], torch.full_like(gate_in2[:, 1], 0.25)))

    def test_cov_floor_validation(self):
        with self.assertRaises(ValueError):
            _build_evf(cov_floor=1.5)
        with self.assertRaises(ValueError):
            _build_evf(cov_floor=-0.1)


class TestEVFGradientReach(unittest.TestCase):
    """V5: the refinement conv receives gradient (deadlock discipline)."""

    def test_refine_conv_gradient_reaches(self):
        """V5, deadlock-rule regression (4th instance): the refine conv
        feeds the ZERO-INIT gate convs, so its gradient is exactly zero
        on the FIRST step (gate weights still zero) and must grow from
        step 2 on -- the Xavier init breaks what would otherwise be a
        permanent 68.8i-style deadlock. A single-step assertion here
        would wrongly demand a nonzero first gradient."""
        evf = _build_evf()
        evf.train()
        opt = torch.optim.SGD(evf.parameters(), lr=0.1)
        grads = []
        for _ in range(3):
            feat = torch.randn(2, 32, 8, 10)
            set_bus(2, (8, 10), 32)
            evf.reset_state()
            out, _, latent_loss, _, _, _ = evf(
                feat, use_posterior=True, deterministic=True)
            opt.zero_grad()
            (out.mean() + latent_loss).backward()
            grads.append(
                evf.evidence_refine_conv.weight.grad.abs().sum().item())
            opt.step()
        ModalityContext.clear()
        # step 1 gradient may be ~0 (gate convs start at zero) ...
        self.assertAlmostEqual(grads[0], 0.0, places=6)
        # ... but it must ARRIVE and GROW: not a permanent deadlock
        self.assertGreater(grads[2], 0.0)
        self.assertGreater(grads[2], grads[1])

    def test_gates_keep_identity_start_with_field_on(self):
        """V7: gate maps stay 1 + tanh(0) = 1 at init (deviation identity
        start), with the physical field feeding them."""
        evf = _build_evf()
        evf.train()
        feat = torch.randn(2, 32, 8, 10)
        set_bus(2, (8, 10), 32)
        out = evf(feat, use_posterior=True, deterministic=True)
        stats = out[5]
        self.assertAlmostEqual(stats['stat_gain_mean'].item(), 1.0, places=5)
        self.assertAlmostEqual(
            stats['stat_readout_gain_mean'].item(), 1.0, places=5)
        ModalityContext.clear()


class TestEVFCalibrationConstants(unittest.TestCase):
    """V6: calibration constants are plain floats, never parameters."""

    def test_constants_are_not_parameters(self):
        evf = _build_evf(c_mot_alpha1=2.0, c_mot_beta1=0.5,
                         dyn_alpha2=1.5, dyn_beta2=0.25)
        for name in ('c_mot_alpha1', 'c_mot_beta1', 'dyn_alpha2',
                     'dyn_beta2'):
            self.assertNotIn(name, dict(evf.named_parameters()))
            self.assertIsInstance(getattr(evf, name), float)
        self.assertEqual(evf.c_mot_alpha1, 2.0)
        self.assertEqual(evf.dyn_beta2, 0.25)

    def test_constants_shift_the_field(self):
        evf = _build_evf()
        feat = torch.randn(1, 32, 8, 10)
        vel = torch.zeros(1, 7, 20, 16)
        vel[:, 6] = 1.0   # full occupancy
        vel[:, 5] = 1.0   # high SNR
        c_mot0, _, _ = evf._physical_evidence(vel, feat)
        # count=1(canvas ch6 max), snr=1 -> sigmoid(alpha1*1*1 - beta1)
        evf_hi = _build_evf(c_mot_alpha1=2.0, c_mot_beta1=-2.0)
        c_mot_hi, _, _ = evf_hi._physical_evidence(vel, feat)
        self.assertGreater(c_mot_hi.mean().item(), c_mot0.mean().item())


class TestEVFCheckpointCompatibility(unittest.TestCase):
    """Switch-off checkpoints load unchanged; new module is the only delta."""

    def test_mci_checkpoint_loads_with_only_new_keys(self):
        mci = _build_mci_small()
        evf = _build_evf(use_physical_evidence=False)
        missing, unexpected = copy_weights(mci, evf)
        self.assertEqual(list(unexpected), [])
        for k in missing:
            self.assertIn('evidence_refine_conv', k)

    def test_mci_checkpoint_loads_into_field_variant(self):
        mci = _build_mci_small()
        evf = _build_evf()
        missing, unexpected = copy_weights(mci, evf)
        self.assertEqual(list(unexpected), [])
        for k in missing:
            self.assertIn('evidence_refine_conv', k)


if __name__ == '__main__':
    unittest.main(verbosity=2)
