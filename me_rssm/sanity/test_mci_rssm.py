"""Unit tests for the Measurement-Conditioned Innovation RSSM (MCI-RSSM).

Run (NOT executed as part of this preparation — the run is a user decision):
    python me_rssm/sanity/test_mci_rssm.py

Covers the four equivalence regimes from me_rssm/docs/12 (E1 empty bus,
E2 switches off, E3 identity start with everything on), the MAI checkpoint
compatibility contract, gate semantics (1 + tanh identity start, detached
innovation magnitude, gradient reach), the dynamic-gate probability
invariant, state_init='obs' pending-reset semantics, and the validation
rules that keep DDP free of unused parameters.

Conventions follow mmdet3d/models/fusion_layers/test_rssm_fusion.py
(unittest, script-runnable, small dims for speed). Non-square canvases
(H=8, W=10) are used deliberately so a (x,y)/(y,x) transposition bug in
the bus transform cannot pass silently.
"""

import sys
import os
import unittest

import torch

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from me_rssm.modality_bus import ModalityContext  # noqa: E402

VEL_RAW = None  # populated per-test via set_bus


def _build_mai(**overrides):
    from mmdet3d.models.fusion_layers.rssm_fusion import (
        MotionAlignedInnovationRSSMFusion)

    kwargs = dict(
        in_channels=256,
        out_channels=256,
        latent_dim=256,
        hidden_dim=64,
        action_dim=0,
        kl_scale=0.1,
        free_nats=1.0,
        min_std=0.1,
        init_std=0.2,
        obs_pred_loss_weight=0.05,
        align_kernel_size=3,
        align_deform_groups=1,
        align_z_state=True,
    )
    kwargs.update(overrides)
    fusion = MotionAlignedInnovationRSSMFusion(**kwargs)
    fusion.eval()
    return fusion


def _build_mci(**overrides):
    from me_rssm.mci_rssm import (
        MeasurementConditionedInnovationRSSMFusion)

    kwargs = dict(
        in_channels=256,
        out_channels=256,
        latent_dim=256,
        hidden_dim=64,
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
        camera_bev_channels=256,
        motion_hidden=16,
        motion_latent=16,
        camera_hidden=16,
        camera_latent=16,
        motion_grad=False,
        modality_dropout=0.0,
        state_init='zero',
    )
    kwargs.update(overrides)
    fusion = MeasurementConditionedInnovationRSSMFusion(**kwargs)
    fusion.eval()
    return fusion


def _build_small_mai(**overrides):
    kwargs = dict(in_channels=32, out_channels=32, latent_dim=32,
                  hidden_dim=16)
    kwargs.update(overrides)
    return _build_mai(**kwargs)


def _build_small_mci(**overrides):
    kwargs = dict(in_channels=32, out_channels=32, latent_dim=32,
                  hidden_dim=16, camera_bev_channels=32)
    kwargs.update(overrides)
    return _build_mci(**kwargs)


def set_bus(B, feat_hw, C_cam, device='cpu', requires_grad=False):
    """Populate the bus for a feat of shape (B, C, H, W) with H != W.

    feat (B,C,8,10)  <-  vel_raw (B,7,20,16): avg_pool2 -> (B,7,10,8)
                        then permute -> (B,7,8,10)
                        cam_raw (B,C,10,8): permute -> (B,C,8,10)
    """
    H, W = feat_hw
    vel = torch.randn(B, 7, 2 * W, 2 * H, device=device) * 0.1
    cam = torch.randn(B, C_cam, W, H, device=device)
    if requires_grad:
        vel.requires_grad_(True)
        cam.requires_grad_(True)
    ModalityContext.set('velocity_bev', vel)
    ModalityContext.set('camera_bev', cam)
    return vel, cam


def copy_weights(src, dst):
    return dst.load_state_dict(src.state_dict(), strict=False)


class TestMCIValidationRules(unittest.TestCase):
    """Construction-time contract enforcement (DDP-safety rules)."""

    def test_motion_offsets_require_bus(self):
        with self.assertRaises(ValueError):
            _build_small_mci(use_motion_offsets=True, use_modality_obs=False)

    def test_dynamic_gate_requires_bus(self):
        with self.assertRaises(ValueError):
            _build_small_mci(use_dynamic_gate=True, use_modality_obs=False)

    def test_gates_require_an_input_source(self):
        with self.assertRaises(ValueError):
            _build_small_mci(
                use_reliability_gain=True, use_gated_readout=True,
                use_modality_obs=False, innov_in_gain=False)

    def test_innov_in_gain_requires_innovation_posterior(self):
        with self.assertRaises(ValueError):
            _build_small_mci(posterior_mode='standard', innov_in_gain=True)

    def test_invalid_posterior_mode_rejected(self):
        with self.assertRaises(ValueError):
            _build_small_mci(posterior_mode='categorical')

    def test_invalid_state_init_rejected(self):
        with self.assertRaises(ValueError):
            _build_small_mci(state_init='gru')

    def test_negative_obs_pred_weight_rejected(self):
        with self.assertRaises(ValueError):
            _build_small_mci(obs_pred_loss_weight=-0.1)

    def test_switch_off_modules_are_not_constructed(self):
        fusion = _build_small_mci(
            use_motion_offsets=False, use_dynamic_gate=False,
            use_reliability_gain=False, use_gated_readout=False,
            use_modality_obs=False, innov_in_gain=True)
        for name in ('motion_encoder', 'camera_encoder', 'obs_proj',
                     'conf_cam', 'conf_mot', 'motion_offset_h',
                     'motion_offset_z', 'dyn_conv', 'dyn_beta',
                     'correct_gate_conv', 'readout_gate_conv',
                     'innovation_mag_conv'):
            self.assertFalse(hasattr(fusion, name), name)
        self.assertEqual(fusion.posterior_struct, 'innovation')


class TestMCICheckpointCompatibility(unittest.TestCase):
    """MAI checkpoints load; inherited RNG stream is preserved."""

    def test_missing_keys_are_only_new_modules(self):
        torch.manual_seed(7)
        mai = _build_small_mai()
        torch.manual_seed(7)
        mci = _build_small_mci()
        missing, unexpected = copy_weights(mai, mci)
        allowed_prefixes = (
            'motion_encoder.', 'camera_encoder.', 'obs_proj.', 'conf_cam.',
            'conf_mot.', 'motion_offset_h.', 'motion_offset_z.', 'dyn_conv.',
            'dyn_beta', 'correct_gate_conv.', 'readout_gate_conv.',
            'innovation_mag_conv.')
        self.assertTrue(all(k.startswith(allowed_prefixes) for k in missing),
                        msg=f'unexpected missing keys: {missing}')
        self.assertEqual(unexpected, [])

    def test_added_modules_preserve_parent_rng_stream(self):
        common = dict(in_channels=32, out_channels=32, latent_dim=32,
                      hidden_dim=16)
        torch.manual_seed(1234)
        mai = _build_small_mai(**common)
        mai_state = {k: v.detach().clone()
                     for k, v in mai.state_dict().items()}
        torch.manual_seed(1234)
        mci = _build_small_mci(**common)
        for key, value in mai_state.items():
            self.assertTrue(
                torch.equal(mci.state_dict()[key], value),
                msg=f'inherited parameter {key} drifted from parent init')

    def test_shared_checkpoint_keys_load_from_trained_mai(self):
        # Simulate a trained MAI: perturb every shared weight, then load
        # into MCI and verify the values landed bit-exactly.
        torch.manual_seed(11)
        mai = _build_small_mai()
        with torch.no_grad():
            for p in mai.parameters():
                p.add_(torch.randn_like(p) * 0.01)
        torch.manual_seed(11)
        mci = _build_small_mci()
        missing, unexpected = copy_weights(mai, mci)
        self.assertTrue(all(k.startswith(
            ('motion_encoder.', 'camera_encoder.', 'obs_proj.', 'conf_cam.',
             'conf_mot.', 'motion_offset_h.', 'motion_offset_z.',
             'dyn_conv.', 'dyn_beta', 'correct_gate_conv.',
             'readout_gate_conv.', 'innovation_mag_conv.'))
            for k in missing))
        self.assertEqual(unexpected, [])
        for key, value in mai.state_dict().items():
            self.assertTrue(torch.equal(mci.state_dict()[key], value),
                            msg=f'key {key} did not load bit-exactly')


class TestMCIBitwiseEquivalence(unittest.TestCase):
    """E1/E2/E3: MCI == MAI bitwise in the three identity regimes."""

    def _run_seq(self, fusion, feats):
        outs = []
        with torch.no_grad():
            for feat in feats:
                fusion.reset_state()
                out = fusion(feat, use_posterior=True, deterministic=True)
                outs.append(out[0])
        return outs

    def test_e1_empty_bus_bitwise_equals_mai(self):
        torch.manual_seed(3)
        mai = _build_small_mai()
        torch.manual_seed(3)
        mci = _build_small_mci()
        copy_weights(mai, mci)
        ModalityContext.clear()
        feats = [torch.randn(2, 32, 8, 10) for _ in range(2)]
        for a, b in zip(self._run_seq(mai, feats),
                        self._run_seq(mci, feats)):
            self.assertTrue(torch.equal(a, b))

    def test_e2_switches_off_with_bus_bitwise_equals_mai(self):
        torch.manual_seed(5)
        mai = _build_small_mai()
        torch.manual_seed(5)
        mci = _build_small_mci(
            use_motion_offsets=False, use_dynamic_gate=False,
            use_reliability_gain=False, use_gated_readout=False,
            use_modality_obs=False)
        copy_weights(mai, mci)
        feats = [torch.randn(2, 32, 8, 10) for _ in range(2)]
        for feat in feats:
            set_bus(feat.shape[0], feat.shape[-2:], 32)
            for a, b in zip(self._run_seq(mai, [feat]),
                            self._run_seq(mci, [feat])):
                self.assertTrue(torch.equal(a, b))
        ModalityContext.clear()

    def test_e3_all_on_fresh_init_with_bus_bitwise_equals_mai(self):
        # The identity-start core: every added pathway contributes exactly
        # zero at init (zero-init convs; 1 + tanh(0) = 1 gates; beta = 0).
        torch.manual_seed(9)
        mai = _build_small_mai()
        torch.manual_seed(9)
        mci = _build_small_mci()
        copy_weights(mai, mci)
        for i in range(2):
            feat = torch.randn(2, 32, 8, 10)
            set_bus(feat.shape[0], feat.shape[-2:], 32)
            mai.reset_state()
            mci.reset_state()
            with torch.no_grad():
                out_mai = mai(feat, use_posterior=True, deterministic=True)
                out_mci = mci(feat, use_posterior=True, deterministic=True)
            for j in (0, 1, 3, 4):  # output, recon, h_t, z_t
                self.assertTrue(
                    torch.equal(out_mai[j], out_mci[j]),
                    msg=f'output[{j}] differs with bus + all switches on')
            # and the gates really exist and really read the bus
            self.assertIn('stat_gain_mean', out_mci[5])
            self.assertIn('stat_readout_gain_mean', out_mci[5])
        ModalityContext.clear()

    def test_e3b_standard_mode_switches_off_bitwise_equals_standard(self):
        from mmdet3d.models.fusion_layers.rssm_fusion import (
            MotionAlignedRSSMFusion)
        torch.manual_seed(13)
        base = MotionAlignedRSSMFusion(
            in_channels=32, out_channels=32, latent_dim=32, hidden_dim=16,
            action_dim=0)
        base.eval()
        torch.manual_seed(13)
        mci = _build_small_mci(
            posterior_mode='standard', innov_in_gain=False,
            use_motion_offsets=False, use_dynamic_gate=False,
            use_reliability_gain=False, use_gated_readout=False,
            use_modality_obs=False)
        copy_weights(base, mci)
        ModalityContext.clear()
        feat = torch.randn(1, 32, 8, 10)
        base.reset_state()
        mci.reset_state()
        with torch.no_grad():
            out_base = base(feat, use_posterior=True, deterministic=True)
            out_mci = mci(feat, use_posterior=True, deterministic=True)
        for j in (0, 1, 3, 4):
            self.assertTrue(torch.equal(out_base[j], out_mci[j]))
        self.assertEqual(mci.posterior_struct, 'standard')


class TestMCIGateSemantics(unittest.TestCase):
    """Correct/readout gate behaviour and gradient reach."""

    def test_gate_helpers_identity_and_range(self):
        fusion = _build_small_mci()
        B, H, W = 2, 8, 10
        gate_in = torch.randn(B, fusion.gate_in_channels, H, W)
        g = fusion._correct_gain_map(gate_in)
        self.assertTrue(torch.allclose(
            g, torch.ones_like(g), atol=1e-6))
        with torch.no_grad():
            fusion.correct_gate_conv.weight.normal_(0, 1.0)
            fusion.correct_gate_conv.bias.normal_(0, 1.0)
            g = fusion._correct_gain_map(gate_in)
        # tanh saturates to exactly +-1 in float32 for |x| > ~9, so the
        # range is the CLOSED [0, 2] at float precision.
        self.assertTrue((g >= 0).all() and (g <= 2).all())
        self.assertFalse(torch.allclose(g, torch.ones_like(g)))

    def test_gate_gradients_reach_gate_convs(self):
        fusion = _build_small_mci()
        fusion.train()
        feat = torch.randn(2, 32, 8, 10, requires_grad=True)
        set_bus(2, (8, 10), 32)
        out, _, latent_loss, _, _, _ = fusion(
            feat, use_posterior=True, deterministic=True)
        (out.mean() + latent_loss).backward()
        self.assertIsNotNone(fusion.correct_gate_conv.weight.grad)
        self.assertGreater(
            fusion.correct_gate_conv.weight.grad.abs().sum().item(), 0.0)
        self.assertIsNotNone(fusion.readout_gate_conv.weight.grad)
        self.assertGreater(
            fusion.readout_gate_conv.weight.grad.abs().sum().item(), 0.0)
        ModalityContext.clear()

    def test_innovation_mag_conv_is_not_deadlocked(self):
        """Regression for the 68.8i deadlock (product of two zero-inits).

        innovation_mag_conv must start NON-zero: with the old zero-init,
        its output was identically zero, the gate conv's innov-channel
        slice received exactly zero gradient forever, and the reduction
        never learned (found by weight inspection on the seed0-2 ep16
        checkpoints: weight exactly 0 after 16 epochs). With Xavier init
        the deadlock breaks one optimizer step after the gate conv moves;
        the identity-start contract still holds because the GATE convs
        remain zero-init (dev == tanh(0) == 0).
        """
        fusion = _build_small_mci()
        self.assertGreater(
            fusion.innovation_mag_conv.weight.abs().mean().item(), 0.0)
        # E3 identity start unaffected: the gate map is still exactly 1.
        gate_in = torch.randn(1, fusion.gate_in_channels, 8, 10)
        self.assertTrue(torch.allclose(
            fusion._correct_gain_map(gate_in), torch.ones_like(gate_in),
            atol=1e-6))

    def test_innovation_magnitude_is_detached_for_gates(self):
        """The gate observes |innovation| but must not backprop into it.

        Verified on the helper directly: grad of the gate map w.r.t. the
        innovation tensor is None, while grad w.r.t. the confidence
        encoder reaches the map once the gate conv is non-zero (at init
        the zero-init gate conv blocks gradient into ALL of its inputs --
        the standard identity-start bottleneck, documented in docs/12).
        """
        fusion = _build_small_mci()
        B, H, W = 2, 8, 10
        innovation = torch.randn(B, 32, H, W, requires_grad=True)
        set_bus(B, (H, W), 32)
        ctx = fusion._read_context(torch.randn(B, 32, H, W))
        gate_in = fusion._gate_input(ctx, innovation,
                                     torch.randn(B, 32, H, W))
        g = fusion._correct_gain_map(gate_in)
        grad = torch.autograd.grad(g.sum(), innovation,
                                   allow_unused=True, retain_graph=True)[0]
        self.assertIsNone(
            grad, 'gate input must be built from innovation.detach()')
        with torch.no_grad():  # unblock the zero-init bottleneck
            fusion.correct_gate_conv.weight.normal_(0, 0.5)
        g = fusion._correct_gain_map(gate_in)
        grad_conf = torch.autograd.grad(
            g.sum(), fusion.conf_mot.weight, allow_unused=True)[0]
        self.assertIsNotNone(grad_conf)
        self.assertGreater(grad_conf.abs().sum().item(), 0.0)
        ModalityContext.clear()

    def test_readout_gate_changes_output_after_training_signal(self):
        torch.manual_seed(17)
        mai = _build_small_mai()
        torch.manual_seed(17)
        mci = _build_small_mci(use_motion_offsets=False,
                               use_dynamic_gate=False,
                               use_reliability_gain=False)
        copy_weights(mai, mci)
        feat = torch.randn(1, 32, 8, 10)
        set_bus(1, (8, 10), 32)
        mai.reset_state()
        mci.reset_state()
        with torch.no_grad():
            out_mai = mai(feat, use_posterior=True, deterministic=True)[0]
            out_id = mci(feat, use_posterior=True, deterministic=True)[0]
        self.assertTrue(torch.equal(out_mai, out_id))
        with torch.no_grad():  # simulate one training step on the gate
            mci.readout_gate_conv.weight.normal_(0, 0.5)
            mci.readout_gate_conv.bias.fill_(0.5)
        mci.reset_state()
        with torch.no_grad():
            out_gated = mci(feat, use_posterior=True,
                            deterministic=True)[0]
        self.assertFalse(torch.allclose(out_id, out_gated))
        ModalityContext.clear()

    def test_dynamic_gate_beta_zero_is_identity_and_effect_exists(self):
        fusion = _build_small_mci()
        B, C, H, W = 1, 32, 8, 10
        x = torch.randn(B, C, H, W)
        h = torch.randn(B, C, H, W)
        dyn_zero = torch.zeros(B, 1, H, W)
        dyn_one = torch.ones(B, 1, H, W)
        h0 = fusion._gru_with_dynamic_gate(x, h, None)
        h1 = fusion._gru_with_dynamic_gate(x, h, dyn_zero)
        self.assertTrue(torch.equal(h0, h1))  # beta=0 -> d irrelevant
        with torch.no_grad():
            fusion.dyn_beta.fill_(0.9)
        h2 = fusion._gru_with_dynamic_gate(x, h, dyn_one)
        self.assertFalse(torch.allclose(h0, h2))
        self.assertTrue(torch.isfinite(h2).all())

    def test_motion_offsets_zero_init_identity_and_gradient(self):
        fusion = _build_small_mci(use_motion_offsets=True,
                                  use_dynamic_gate=False,
                                  use_reliability_gain=False,
                                  use_gated_readout=False)
        fusion.train()
        feat1 = torch.randn(1, 32, 8, 10)
        set_bus(1, (8, 10), 32)
        # Frame 1 only fills the (zero) state: the deformable warp of a
        # zero state has identically zero offset gradient, so the gradient
        # assertion must run on frame 2 where h_state is non-zero.
        with torch.no_grad():
            fusion(feat1, use_posterior=True, deterministic=True)
        feat2 = torch.randn(1, 32, 8, 10, requires_grad=True)
        out, _, _, _, _, _ = fusion(feat2, use_posterior=True,
                                    deterministic=True)
        out.mean().backward()
        self.assertIsNotNone(fusion.motion_offset_h.weight.grad)
        self.assertGreater(
            fusion.motion_offset_h.weight.grad.abs().sum().item(), 0.0)
        ModalityContext.clear()

    def test_motion_grad_false_keeps_bus_detached(self):
        fusion = _build_small_mci(use_motion_offsets=True)
        fusion.train()
        feat = torch.randn(1, 32, 8, 10)
        vel, cam = set_bus(1, (8, 10), 32, requires_grad=True)
        out, _, latent_loss, _, _, _ = fusion(
            feat, use_posterior=True, deterministic=True)
        (out.mean() + latent_loss).backward()
        self.assertIsNone(vel.grad)
        self.assertIsNone(cam.grad)
        ModalityContext.clear()

    def test_motion_grad_true_allows_bus_gradient(self):
        fusion = _build_small_mci(use_motion_offsets=True, motion_grad=True)
        fusion.train()
        feat = torch.randn(1, 32, 8, 10)
        vel, _ = set_bus(1, (8, 10), 32, requires_grad=True)
        out, _, _, _, _, _ = fusion(feat, use_posterior=True,
                                    deterministic=True)
        out.mean().backward()
        self.assertIsNotNone(vel.grad)
        ModalityContext.clear()


class TestMCIObservationAndStats(unittest.TestCase):
    """Observation pipeline, stats contract, modality dropout, kl_scale."""

    def test_interface_matches_mai(self):
        fusion = _build_small_mci()
        B, C, H, W = 2, 32, 8, 10
        set_bus(B, (H, W), 32)
        feat = torch.randn(B, C, H, W)
        with torch.no_grad():
            out = fusion(feat, use_posterior=True, deterministic=True)
        self.assertEqual(len(out), 6)
        output, recon, latent_loss, h_t, z_t, stats = out
        self.assertEqual(tuple(output.shape), (B, C, H, W))
        self.assertEqual(tuple(recon.shape), (B, C, H, W))
        self.assertEqual(tuple(h_t.shape), (B, 32, H, W))
        self.assertEqual(tuple(z_t.shape), (B, 32, H, W))
        self.assertTrue(torch.isfinite(latent_loss).all())
        for key in ('stat_obs_pred', 'stat_innovation_sq',
                    'stat_gain_mean', 'stat_gain_std', 'stat_gain_min',
                    'stat_gain_max', 'stat_readout_gain_mean',
                    'stat_readout_gain_std', 'stat_conf_cam_mean',
                    'stat_conf_mot_mean', 'stat_dyn_ratio',
                    'stat_speed_mean'):
            self.assertIn(key, stats)
            self.assertNotIn('loss', key)
        ModalityContext.clear()

    def test_obs_prior_trained_only_by_observation_prediction(self):
        fusion = _build_small_mci(obs_pred_loss_weight=0.05,
                                  use_motion_offsets=False,
                                  use_dynamic_gate=False,
                                  use_reliability_gain=False,
                                  use_gated_readout=False,
                                  use_modality_obs=False)
        fusion.train()
        feat = torch.randn(1, 32, 8, 10)
        out, _, latent_loss, _, _, _ = fusion(
            feat, use_posterior=True, deterministic=True)
        latent_loss.backward()
        self.assertIsNotNone(fusion.obs_prior.weight.grad)
        self.assertGreater(
            fusion.obs_prior.weight.grad.abs().sum().item(), 0.0)

    def test_standard_mode_has_no_innovation_stats(self):
        fusion = _build_small_mci(posterior_mode='standard',
                                  innov_in_gain=False)
        set_bus(1, (8, 10), 32)
        feat = torch.randn(1, 32, 8, 10)
        with torch.no_grad():
            *_, stats = fusion(feat, use_posterior=True,
                               deterministic=True)
        self.assertNotIn('stat_obs_pred', stats)
        self.assertNotIn('stat_innovation_sq', stats)
        self.assertIn('stat_gain_mean', stats)
        ModalityContext.clear()

    def test_modality_dropout_zeroes_confidence_in_train(self):
        fusion = _build_small_mci(modality_dropout=1.0)
        fusion.train()
        set_bus(1, (8, 10), 32)
        feat = torch.randn(1, 32, 8, 10)
        with torch.no_grad():
            *_, stats = fusion(feat, use_posterior=True,
                               deterministic=True)
        # dropout p=1 -> confidences zeroed for every sample
        self.assertEqual(stats['stat_conf_cam_mean'].item(), 0.0)
        self.assertEqual(stats['stat_conf_mot_mean'].item(), 0.0)
        fusion.eval()
        with torch.no_grad():
            *_, stats_eval = fusion(feat, use_posterior=True,
                                    deterministic=True)
        self.assertGreater(stats_eval['stat_conf_cam_mean'].item(), 0.0)
        ModalityContext.clear()

    def test_kl_scale_attribute_respected(self):
        # obs_pred_loss_weight=0 so latent_loss == kl_effective * kl_scale;
        # built with kl_scale=1.0 so the ratio equals the new scale exactly.
        fusion = _build_small_mci(kl_scale=1.0, obs_pred_loss_weight=0.0,
                                  use_motion_offsets=False,
                                  use_dynamic_gate=False,
                                  use_reliability_gain=False,
                                  use_gated_readout=False,
                                  use_modality_obs=False)
        feat = torch.randn(1, 32, 8, 10)
        fusion.reset_state()
        with torch.no_grad():
            _, _, ll1, _, _, _ = fusion(feat, use_posterior=True,
                                        deterministic=True)
        fusion.kl_scale = 0.123
        fusion.reset_state()
        with torch.no_grad():
            _, _, ll2, _, _, _ = fusion(feat, use_posterior=True,
                                        deterministic=True)
        self.assertAlmostEqual(
            ll2.item() / ll1.item(), 0.123, places=4)


class TestMCIStateSemantics(unittest.TestCase):
    """state_init='obs' and reset_for_samples pending semantics."""

    def test_obs_init_zero_weights_bitwise_equal_zero_init(self):
        torch.manual_seed(21)
        m_zero = _build_small_mci(state_init='zero',
                                  use_motion_offsets=False,
                                  use_dynamic_gate=False,
                                  use_reliability_gain=False,
                                  use_gated_readout=False,
                                  use_modality_obs=False)
        torch.manual_seed(21)
        m_obs = _build_small_mci(state_init='obs',
                                 use_motion_offsets=False,
                                 use_dynamic_gate=False,
                                 use_reliability_gain=False,
                                 use_gated_readout=False,
                                 use_modality_obs=False)
        # zero-init h/z bootstrap convs must not be in the 'zero' model
        self.assertIsNone(m_zero.h_init_conv)
        self.assertIsNotNone(m_obs.h_init_conv)
        feat = torch.randn(1, 32, 8, 10)
        m_zero.reset_state()
        m_obs.reset_state()
        with torch.no_grad():
            a = m_zero(feat, use_posterior=True, deterministic=True)
            b = m_obs(feat, use_posterior=True, deterministic=True)
        for j in (0, 1, 3, 4):
            self.assertTrue(torch.equal(a[j], b[j]))

    def test_obs_init_learns_bootstrap_and_pending_reset(self):
        fusion = _build_small_mci(state_init='obs',
                                  use_motion_offsets=False,
                                  use_dynamic_gate=False,
                                  use_reliability_gain=False,
                                  use_gated_readout=False,
                                  use_modality_obs=False)
        with torch.no_grad():
            fusion.h_init_conv.weight.normal_(0, 0.05)
            fusion.h_init_conv.bias.normal_(0, 0.05)
            fusion.z_init_conv.weight.normal_(0, 0.05)
            fusion.z_init_conv.bias.normal_(0, 0.05)
        feat = torch.randn(1, 32, 8, 10)
        fusion.reset_state()
        with torch.no_grad():
            fusion(feat, use_posterior=True, deterministic=True)
        self.assertFalse(torch.allclose(
            fusion.h_state, torch.zeros_like(fusion.h_state)))
        # reset_for_samples zeroes now but re-bootstraps on the next step
        mask = torch.tensor([True])
        fusion.reset_for_samples(mask)
        self.assertTrue(torch.allclose(
            fusion.h_state, torch.zeros_like(fusion.h_state)))
        with torch.no_grad():
            fusion(feat, use_posterior=True, deterministic=True)
        self.assertFalse(torch.allclose(
            fusion.h_state, torch.zeros_like(fusion.h_state)))

    def test_state_carries_across_calls_and_resets(self):
        fusion = _build_small_mci(use_motion_offsets=False,
                                  use_dynamic_gate=False,
                                  use_reliability_gain=False,
                                  use_gated_readout=False,
                                  use_modality_obs=False)
        f1 = torch.randn(1, 32, 8, 10)
        f2 = torch.randn(1, 32, 8, 10)
        fusion.reset_state()
        with torch.no_grad():
            fusion(f1, use_posterior=True, deterministic=True)
            h_after_first = fusion.h_state.detach().clone()
            out_second, _, _, _, _, _ = fusion(
                f2, use_posterior=True, deterministic=True)
            self.assertFalse(torch.allclose(
                h_after_first, torch.zeros_like(h_after_first)))
            fusion.reset_state()
            out_fresh, _, _, _, _, _ = fusion(
                f2, use_posterior=True, deterministic=True)
        self.assertFalse(torch.allclose(out_second, out_fresh))

    def test_deterministic_replay_is_bitwise_identical(self):
        fusion = _build_small_mci()
        set_bus(1, (8, 10), 32)
        feat = torch.randn(1, 32, 8, 10)
        fusion.reset_state()
        with torch.no_grad():
            out_a, _, _, _, _, _ = fusion(feat, use_posterior=True,
                                          deterministic=True)
        fusion.reset_state()
        with torch.no_grad():
            out_b, _, _, _, _, _ = fusion(feat, use_posterior=True,
                                          deterministic=True)
        self.assertTrue(torch.equal(out_a, out_b))
        ModalityContext.clear()

    def test_standard_mode_drops_obs_prior_for_ddp(self):
        """In standard mode obs_prior has no consumer -> must not exist.

        A dead module would abort DDP training on unused parameters; MAI
        checkpoints still load (obs_prior.* becomes an unexpected key).
        """
        mci = _build_small_mci(posterior_mode='standard',
                               innov_in_gain=False,
                               use_motion_offsets=False,
                               use_dynamic_gate=False,
                               use_reliability_gain=False,
                               use_gated_readout=False,
                               use_modality_obs=False)
        self.assertFalse(hasattr(mci, 'obs_prior'))
        self.assertNotIn('obs_prior.weight', mci.state_dict())
        mai = _build_small_mai()
        missing, unexpected = copy_weights(mai, mci)
        self.assertTrue(all(k.startswith('obs_prior.')
                            for k in unexpected),
                        msg=f'unexpected keys: {unexpected}')
        self.assertEqual(missing, [])

    def test_posterior_observation_probe_hook_is_exact(self):
        """tools/probe_posterior_source.py reconstructs e_obs via this hook.

        After an eval-mode forward with a trained (non-zero) obs_proj,
        ``_posterior_observation(e_t)`` must equal the e_obs the posterior
        actually consumed; before any forward it falls back to e_t.
        """
        fusion = _build_small_mci(use_motion_offsets=False,
                                  use_dynamic_gate=False,
                                  use_reliability_gain=False,
                                  use_gated_readout=False)
        with torch.no_grad():  # simulate a trained obs_proj
            fusion.obs_proj.weight.normal_(0, 0.05)
            fusion.obs_proj.bias.normal_(0, 0.05)
        feat = torch.randn(1, 32, 8, 10)
        set_bus(1, (8, 10), 32)
        # before any forward: exact fallback to e_t
        e_probe = torch.randn(1, 32, 8, 10)
        self.assertTrue(torch.equal(
            fusion._posterior_observation(e_probe), e_probe))
        fusion.reset_state()
        with torch.no_grad():
            fusion(feat, use_posterior=True, deterministic=True)
        self.assertIsNotNone(fusion._last_obs_residual)
        e_t_last = fusion.encoder(feat)
        reconstructed = fusion._posterior_observation(e_t_last)
        expected = e_t_last + fusion._last_obs_residual
        self.assertTrue(torch.equal(reconstructed, expected))
        ModalityContext.clear()

    def test_z_intervention_hook_is_exact_and_readout_follows(self):
        """The probe counterfactual must replace the FINAL z_t exactly.

        With _z_intervention set, out[4] is the intervened tensor bitwise
        and the readout (out[0]) is recomputed from it; with it None the
        forward is untouched.
        """
        torch.manual_seed(23)
        mai = _build_small_mai()
        torch.manual_seed(23)
        mci = _build_small_mci(use_motion_offsets=False,
                               use_dynamic_gate=False,
                               use_reliability_gain=False)
        copy_weights(mai, mci)
        feat = torch.randn(1, 32, 8, 10)
        set_bus(1, (8, 10), 32)
        mai.reset_state()
        mci.reset_state()
        with torch.no_grad():
            out_id = mci(feat, use_posterior=True, deterministic=True)
        mci._z_intervention = lambda z: torch.zeros_like(z)
        mci.reset_state()
        with torch.no_grad():
            out_zero = mci(feat, use_posterior=True, deterministic=True)
        mci._z_intervention = None
        self.assertTrue(torch.equal(
            out_zero[4], torch.zeros_like(out_zero[4])))
        self.assertFalse(torch.allclose(out_zero[0], out_id[0]))
        mci.reset_state()
        with torch.no_grad():  # hook removed -> back to identity start
            out_again = mci(feat, use_posterior=True, deterministic=True)
        self.assertTrue(torch.equal(out_again[0], out_id[0]))
        ModalityContext.clear()

    def test_eval_stashes_last_h_t_for_source_probe(self):
        fusion = _build_small_mci(use_motion_offsets=False,
                                  use_dynamic_gate=False,
                                  use_reliability_gain=False,
                                  use_gated_readout=False,
                                  use_modality_obs=False)
        feat = torch.randn(1, 32, 8, 10)
        fusion.reset_state()
        with torch.no_grad():
            _, _, _, h_t, _, _ = fusion(feat, use_posterior=True,
                                        deterministic=True)
        self.assertIsNotNone(fusion._last_h_t)
        self.assertTrue(torch.equal(fusion._last_h_t, h_t))
        fusion.train()  # training forwards must not hold eval stashes
        with torch.no_grad():
            fusion(feat, use_posterior=True, deterministic=True)
        self.assertIsNone(fusion._last_h_t)

    def test_use_prior_only_falls_back_to_prior_mean(self):
        fusion = _build_small_mci(use_motion_offsets=False,
                                  use_dynamic_gate=False,
                                  use_reliability_gain=False,
                                  use_gated_readout=False,
                                  use_modality_obs=False)
        feat = torch.randn(1, 32, 8, 10)
        fusion.reset_state()
        with torch.no_grad():
            _, _, _, _, z_post, _ = fusion(feat, use_posterior=True,
                                           deterministic=True)
        fusion.reset_state()
        with torch.no_grad():
            _, _, _, _, z_prior, _ = fusion(feat, use_posterior=False,
                                            deterministic=True)
        self.assertFalse(torch.allclose(z_post, z_prior))


if __name__ == '__main__':
    unittest.main()
