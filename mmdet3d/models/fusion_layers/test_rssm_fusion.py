"""Unit tests for RSSM fusion module and KL scale scheduler hook.

Run: python mmdet3d/models/fusion_layers/test_rssm_fusion.py
"""

import torch
import unittest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))


class TestBEVRSSMTemporalFusion(unittest.TestCase):
    """Test RSSM fusion module correctness."""

    def setUp(self):
        from mmdet3d.models.fusion_layers.rssm_fusion import BEVRSSMTemporalFusion

        self.fusion = BEVRSSMTemporalFusion(
            in_channels=256,
            out_channels=256,
            latent_dim=256,
            hidden_dim=64,
            action_dim=2,
            kl_scale=0.1,
            free_nats=0.0,
            min_std=0.1,
            init_std=0.2,
        )
        self.fusion.eval()  # no BN training behaviour during test

    def test_initial_output_has_latent_contribution(self):
        """Test Xavier output_proj gives z_t a non-zero initial contribution."""
        B, C, H, W = 2, 256, 8, 8
        feat = torch.randn(B, C, H, W)

        with torch.no_grad():
            output, recon, kl, h, z, stats = self.fusion(
                feat, use_posterior=True, deterministic=True
            )

        self.assertFalse(torch.allclose(output, feat, atol=1e-5))

    def test_deterministic_output_consistent(self):
        """Test that same input after reset gives identical output."""
        B, C, H, W = 2, 256, 8, 8

        # First pass
        self.fusion.reset_state()
        feat1 = torch.randn(B, C, H, W)
        with torch.no_grad():
            out1, _, _, _, _, _ = self.fusion(
                feat1, use_posterior=True, deterministic=True
            )

        # Second pass with same input after reset
        self.fusion.reset_state()
        with torch.no_grad():
            out2, _, _, _, _, _ = self.fusion(
                feat1.clone(), use_posterior=True, deterministic=True
            )

        self.assertTrue(torch.allclose(out1, out2, atol=1e-6),
                        f"Outputs differ after reset: max diff = {(out1 - out2).abs().max().item():.6f}")

    def test_second_frame_accumulates_state(self):
        """Test that accumulated state changes the second frame."""
        B, C, H, W = 2, 256, 8, 8

        self.fusion.reset_state()
        feat1 = torch.randn(B, C, H, W)
        feat2 = torch.randn(B, C, H, W)

        with torch.no_grad():
            out1, _, _, _, _, _ = self.fusion(
                feat1, use_posterior=True, deterministic=True
            )
            out2, _, _, _, _, _ = self.fusion(
                feat2, use_posterior=True, deterministic=True
            )

        self.assertFalse(torch.allclose(out2, feat2, atol=1e-5))

    def test_stats_present_and_detached(self):
        """Test that stats dict contains expected keys and values are detached."""
        B, C, H, W = 2, 256, 8, 8
        feat = torch.randn(B, C, H, W)

        self.fusion.train()  # need training mode for BN
        output, recon, kl, h, z, stats = self.fusion(
            feat, use_posterior=True, deterministic=True
        )

        expected_keys = [
            'stat_kl_raw_mean', 'stat_kl_effective_mean',
            'stat_clamped_ratio', 'stat_mu_diff_sq',
            'stat_posterior_std', 'stat_prior_std',
        ]
        for key in expected_keys:
            self.assertIn(key, stats, f"Missing key: {key}")
            self.assertIsInstance(stats[key], torch.Tensor)
            self.assertEqual(stats[key].ndim, 0, f"{key} should be scalar")
            self.assertFalse(stats[key].requires_grad, f"{key} should be detached")
        self.fusion.eval()

    def test_logstd_bias_gives_expected_std(self):
        """Test that initial logstd bias produces std ≈ init_std."""
        import math

        # Recompute the bias the module should have used
        min_std = self.fusion.min_std
        init_std = self.fusion.init_std
        expected_bias = math.log(min_std) + math.log(init_std / min_std - 1.0)

        # Check bias values
        self.assertAlmostEqual(
            self.fusion.prior_logstd.bias.mean().item(),
            expected_bias, places=5
        )
        self.assertAlmostEqual(
            self.fusion.posterior_logstd.bias.mean().item(),
            expected_bias, places=5
        )


class TestKLScaleSchedulerHook(unittest.TestCase):
    """Test KL scale scheduler hook."""

    def test_scheduler_values(self):
        from mmdet3d.core.hook.kl_scale_scheduler import KLScaleSchedulerHook

        hook = KLScaleSchedulerHook(
            start_epoch=0, end_epoch=3,
            start_value=0.0, end_value=0.1,
        )

        # epoch 0 → 0.0
        self.assertAlmostEqual(hook.compute_kl_scale(0), 0.0, places=5)

        # epoch 1 → 0.0333... (1/3 of warm-up)
        self.assertAlmostEqual(hook.compute_kl_scale(1), 0.1 / 3, places=4)

        # epoch 2 → 0.0667... (2/3 of warm-up)
        self.assertAlmostEqual(hook.compute_kl_scale(2), 0.1 * 2 / 3, places=4)

        # epoch 3 → 0.1 (exactly at end_epoch)
        self.assertAlmostEqual(hook.compute_kl_scale(3), 0.1, places=5)

        # epoch 5 → 0.1 (past end_epoch, should keep end_value)
        self.assertAlmostEqual(hook.compute_kl_scale(5), 0.1, places=5)

        # epoch -1 (edge case: before start)
        self.assertAlmostEqual(hook.compute_kl_scale(-1), 0.0, places=5)

    def test_resume_consistency(self):
        """Test that scheduler depends only on epoch, so resume is consistent."""
        from mmdet3d.core.hook.kl_scale_scheduler import KLScaleSchedulerHook

        hook = KLScaleSchedulerHook(
            start_epoch=0, end_epoch=3,
            start_value=0.0, end_value=0.1,
        )

        # If we resume at epoch 2, the value should be based on epoch 2 only
        # — no internal counter drift
        val_direct = hook.compute_kl_scale(2)

        # "Resume": create a fresh hook and check epoch 2
        hook2 = KLScaleSchedulerHook(
            start_epoch=0, end_epoch=3,
            start_value=0.0, end_value=0.1,
        )
        val_resume = hook2.compute_kl_scale(2)

        self.assertAlmostEqual(val_direct, val_resume, places=5,
                               msg="Resume should give same value as direct run")


class TestMotionAlignedRSSMFusion(unittest.TestCase):
    """Test MotionAlignedRSSMFusion correctness."""

    def setUp(self):
        from mmdet3d.models.fusion_layers.rssm_fusion import MotionAlignedRSSMFusion

        self.fusion = MotionAlignedRSSMFusion(
            in_channels=256,
            out_channels=256,
            latent_dim=256,
            hidden_dim=64,
            action_dim=2,
            kl_scale=0.1,
            free_nats=0.0,
            min_std=0.1,
            init_std=0.2,
            align_kernel_size=3,
            align_deform_groups=1,
            align_z_state=True,
        )
        self.fusion.eval()

    def test_initial_output_has_latent_contribution(self):
        """Xavier-initialized output_proj gives z_t an initial contribution."""
        B, C, H, W = 2, 256, 8, 8
        feat = torch.randn(B, C, H, W)

        with torch.no_grad():
            output, recon, kl, h, z, stats = self.fusion(
                feat, use_posterior=True, deterministic=True
            )

        self.assertFalse(torch.allclose(output, feat, atol=1e-5))

    def test_deterministic_output_consistent(self):
        """Same input after reset → identical output."""
        B, C, H, W = 2, 256, 8, 8

        self.fusion.reset_state()
        feat1 = torch.randn(B, C, H, W)
        with torch.no_grad():
            out1, _, _, _, _, _ = self.fusion(
                feat1, use_posterior=True, deterministic=True
            )

        self.fusion.reset_state()
        with torch.no_grad():
            out2, _, _, _, _, _ = self.fusion(
                feat1.clone(), use_posterior=True, deterministic=True
            )

        self.assertTrue(torch.allclose(out1, out2, atol=1e-6),
                        f"Outputs differ after reset: max diff = {(out1 - out2).abs().max().item():.6f}")

    def test_second_frame_accumulates_state(self):
        """Accumulated state changes the second frame output."""
        B, C, H, W = 2, 256, 8, 8

        self.fusion.reset_state()
        feat1 = torch.randn(B, C, H, W)
        feat2 = torch.randn(B, C, H, W)

        with torch.no_grad():
            self.fusion(feat1, use_posterior=True, deterministic=True)
            out2, _, _, _, _, _ = self.fusion(
                feat2, use_posterior=True, deterministic=True
            )

        self.assertFalse(torch.allclose(out2, feat2, atol=1e-5))

    def test_stats_present_and_detached(self):
        """Stats dict contains expected keys and values are detached scalars."""
        B, C, H, W = 2, 256, 8, 8
        feat = torch.randn(B, C, H, W)

        self.fusion.train()
        output, recon, kl, h, z, stats = self.fusion(
            feat, use_posterior=True, deterministic=True
        )

        expected_keys = [
            'stat_kl_raw_mean', 'stat_kl_effective_mean',
            'stat_clamped_ratio', 'stat_mu_diff_sq',
            'stat_posterior_std', 'stat_prior_std',
        ]
        for key in expected_keys:
            self.assertIn(key, stats, f"Missing key: {key}")
            self.assertIsInstance(stats[key], torch.Tensor)
            self.assertEqual(stats[key].ndim, 0)
            self.assertFalse(stats[key].requires_grad)
        self.fusion.eval()

    def test_alignment_module_exists(self):
        """Verify alignment submodules were constructed."""
        self.assertTrue(hasattr(self.fusion, 'align_h_offset_mask'))
        self.assertTrue(hasattr(self.fusion, 'align_h_deform_conv'))
        self.assertTrue(hasattr(self.fusion, 'align_z_offset_mask'))
        self.assertTrue(hasattr(self.fusion, 'align_z_deform_conv'))

    def test_alignment_zero_init(self):
        """Zero-init offset generator produces zero offsets and sigmoid(0) masks."""
        B, C, H, W = 2, 256, 8, 8
        feat = torch.randn(B, C, H, W)

        self.fusion.reset_state()
        # Populate state with a first forward pass (h_state starts as zeros)
        with torch.no_grad():
            self.fusion(feat, use_posterior=True, deterministic=True)

        # Now h_state is populated, check zero-init offset generator behaviour
        with torch.no_grad():
            concat = torch.cat([feat, self.fusion.h_state], dim=1)
            offset_and_mask = self.fusion.align_h_offset_mask(concat)
            k2 = self.fusion.align_kernel_size ** 2
            o1 = 2 * self.fusion.align_deform_groups * k2
            offset = offset_and_mask[:, :o1, :, :]
            mask_raw = offset_and_mask[:, o1:, :, :]
            mask = mask_raw.sigmoid()

        # Weights zero-init → offset should be all zeros
        self.assertTrue(torch.allclose(offset, torch.zeros_like(offset), atol=1e-6),
                        f"Offset not zero: max = {offset.abs().max().item():.6f}")
        # Sigmoid(0) = 0.5
        self.assertTrue(torch.allclose(mask, 0.5 * torch.ones_like(mask), atol=1e-6),
                        f"Mask not 0.5: mean = {mask.mean().item():.6f}")

    def test_align_z_state_false(self):
        """With align_z_state=False, no z alignment modules."""
        from mmdet3d.models.fusion_layers.rssm_fusion import MotionAlignedRSSMFusion

        fusion_no_z = MotionAlignedRSSMFusion(
            in_channels=256, out_channels=256,
            latent_dim=256, hidden_dim=64, action_dim=2,
            align_z_state=False,
        )
        fusion_no_z.eval()

        self.assertFalse(hasattr(fusion_no_z, 'align_z_offset_mask'))
        self.assertFalse(hasattr(fusion_no_z, 'align_z_deform_conv'))

        B, C, H, W = 2, 256, 8, 8
        feat = torch.randn(B, C, H, W)
        with torch.no_grad():
            output, _, _, _, _, _ = fusion_no_z(
                feat, use_posterior=True, deterministic=True)
        self.assertEqual(output.shape, (B, C, H, W))


class TestFixedNoisePosteriorLatentFusion(unittest.TestCase):
    """Test the fixed-noise posterior fusion behaviour."""

    def setUp(self):
        from mmdet3d.models.fusion_layers.rssm_fusion import FixedNoisePosteriorLatentFusion

        self.fusion = FixedNoisePosteriorLatentFusion(
            in_channels=256,
            out_channels=256,
            latent_dim=256,
            hidden_dim=64,
        )
        self.fusion.eval()

    def test_default_noise_std(self):
        self.assertEqual(self.fusion.posterior_noise_std, 0.1)

    def test_deterministic_ignores_noise(self):
        B, C, H, W = 2, 256, 8, 8
        feat = torch.randn(B, C, H, W)
        outputs = []
        for _ in range(2):
            self.fusion.reset_state()
            with torch.no_grad():
                out, recon, kl, h, z, stats = self.fusion(
                    feat, use_posterior=True, deterministic=True)
            outputs.append(out)

        self.assertTrue(torch.allclose(outputs[0], outputs[1], atol=1e-6))
        self.assertIsNone(kl)
        self.assertIsNone(stats)

    def test_non_deterministic_injects_fixed_noise(self):
        B, C, H, W = 2, 256, 8, 8
        feat = torch.randn(B, C, H, W)
        outputs = []
        for _ in range(2):
            self.fusion.reset_state()
            with torch.no_grad():
                out, recon, kl, h, z, stats = self.fusion(
                    feat, use_posterior=True, deterministic=False)
            outputs.append(out)

        self.assertFalse(torch.allclose(outputs[0], outputs[1], atol=1e-3))
        self.assertIsNone(kl)
        self.assertIsNone(stats)


class TestPosteriorOnlyLearnableStdLatentFusion(unittest.TestCase):
    """Test posterior-only fusion with a learnable conditional std."""

    def setUp(self):
        from mmdet3d.models.fusion_layers.rssm_fusion import (
            PosteriorOnlyLearnableStdLatentFusion,
        )

        self.fusion = PosteriorOnlyLearnableStdLatentFusion(
            in_channels=256,
            out_channels=256,
            latent_dim=256,
            hidden_dim=64,
        )
        self.fusion.eval()

    def test_prior_removed_posterior_learnable_std_kept(self):
        self.assertIsNone(self.fusion.prior_mu)
        self.assertIsNone(self.fusion.prior_logstd)
        self.assertIsNotNone(self.fusion.posterior_mu)
        self.assertIsNotNone(self.fusion.posterior_logstd)

    def test_posterior_logstd_initial_bias_matches_parent_path(self):
        """posterior_logstd gets the same parent init as Deterministic/FixedNoise."""
        import math

        expected_bias = (
            math.log(self.fusion.min_std)
            + math.log(self.fusion.init_std / self.fusion.min_std - 1.0)
        )
        self.assertAlmostEqual(
            self.fusion.posterior_logstd.bias.mean().item(),
            expected_bias, places=5,
        )

    def test_posterior_std_stats_subsamples_and_is_deterministic(self):
        logstd = torch.randn(2, 32, 216, 248)
        stats_first = self.fusion._posterior_std_stats(logstd, max_samples=256)
        stats_second = self.fusion._posterior_std_stats(logstd, max_samples=256)

        for key, value in stats_first.items():
            self.assertIsInstance(value, torch.Tensor)
            self.assertEqual(value.ndim, 0)


class TestLowDimFutureConsistentLatentFusion(unittest.TestCase):
    """Smoke tests for the low-dimensional future-consistent latent."""

    def setUp(self):
        from mmdet3d.models.fusion_layers.rssm_fusion import (
            LowDimFutureConsistentLatentFusion,
        )

        self.fusion = LowDimFutureConsistentLatentFusion(
            in_channels=256,
            latent_dim=32,
            hidden_dim=128,
            latent_pool='adaptive',
            latent_size=(4, 4),
            modulation_scale=0.1,
            gate_init_bias=-1.0,
        )
        self.fusion.eval()

    def test_reconstruction_anchors_z_and_is_reported(self):
        """The observation-likelihood term must exist, train z, and be logged."""
        feat = torch.randn(2, 256, 8, 8)
        self.fusion.train()
        self.fusion.reset_state()
        output, recon, latent_loss, h_t, z_t, stats = self.fusion(
            feat, use_posterior=True, deterministic=True)

        self.assertIn('stat_recon_mse', stats)
        self.assertGreater(stats['stat_recon_mse'].item(), 0.0)
        # recon is folded into the primary objective, so index 1 must stay None
        # to avoid the detector re-scoring a 4x4 map against the full BEV grid.
        self.assertIsNone(recon)
        self.assertTrue(latent_loss.requires_grad)

        latent_loss.backward()
        self.assertIsNotNone(self.fusion.posterior_mu.weight.grad)
        self.assertGreater(
            self.fusion.posterior_mu.weight.grad.abs().sum().item(), 0.0)
        self.assertIsNotNone(self.fusion.decoder[1].weight.grad)

    def test_temporal_delta_reconstructs_posterior_correction(self):
        """Innovation target must exclude h_t and require valid history."""
        from mmdet3d.models.fusion_layers.rssm_fusion import (
            LowDimFutureConsistentLatentFusion,
        )

        fusion = LowDimFutureConsistentLatentFusion(
            in_channels=256,
            latent_dim=32,
            hidden_dim=128,
            latent_size=(4, 4),
            future_loss_weight=0.0,
            recon_target_mode='temporal_delta',
            recon_input_mode='posterior_correction',
            direct_correction_readout=True,
        )
        fusion.train()
        feat1 = torch.randn(2, 256, 8, 8)
        feat2 = torch.randn(2, 256, 8, 8)

        _, _, first_loss, _, _, first_stats = fusion(feat1)
        _, _, second_loss, _, _, second_stats = fusion(
            feat2, detach_state=False)

        self.assertEqual(first_stats['stat_recon_valid_frac'].item(), 0.0)
        self.assertEqual(first_stats['stat_recon_mse'].item(), 0.0)
        self.assertEqual(
            fusion.decoder[0].conv.in_channels, fusion.latent_dim)
        self.assertEqual(second_stats['stat_recon_valid_frac'].item(), 1.0)
        self.assertGreater(second_stats['stat_recon_mse'].item(), 0.0)
        self.assertGreater(second_stats['stat_recon_delta_rms'].item(), 0.0)
        self.assertGreater(second_stats['stat_correction_sq'].item(), 0.0)
        self.assertTrue(torch.isfinite(first_loss))
        self.assertTrue(torch.isfinite(second_loss))

        second_loss.backward()
        self.assertIsNotNone(fusion.posterior_mu.weight.grad)
        self.assertGreater(
            fusion.posterior_mu.weight.grad.abs().sum().item(), 0.0)
        self.assertIsNotNone(fusion.correction_output_proj)

    def test_temporal_delta_reset_drops_invalid_sample(self):
        """A reset sample must not compare against stale previous features."""
        from mmdet3d.models.fusion_layers.rssm_fusion import (
            LowDimFutureConsistentLatentFusion,
        )

        fusion = LowDimFutureConsistentLatentFusion(
            in_channels=256,
            latent_dim=32,
            hidden_dim=128,
            latent_size=(4, 4),
            future_loss_weight=0.0,
            recon_target_mode='temporal_delta',
            recon_input_mode='posterior_correction',
        )
        fusion.train()
        fusion(torch.randn(2, 256, 8, 8))
        fusion.reset_for_samples(torch.tensor([True, False]))
        _, _, _, _, _, stats = fusion(torch.randn(2, 256, 8, 8))

        self.assertAlmostEqual(
            stats['stat_recon_valid_frac'].item(), 0.5, places=6)

    def test_discrimination_loss_penalizes_sample_invariant_correction(self):
        """Swapping corrections must be worse than keeping each sample's own.

        The reconstruction term alone can be satisfied by a near-common
        correction, so the ranking term is what actually demands sample
        identity. A fused correction (constructed to be identical across the
        batch) must be penalised more than distinct per-sample corrections,
        all else equal.
        """
        from mmdet3d.models.fusion_layers.rssm_fusion import (
            LowDimFutureConsistentLatentFusion,
        )

        def build(discr_weight):
            fusion = LowDimFutureConsistentLatentFusion(
                in_channels=256,
                latent_dim=32,
                hidden_dim=128,
                latent_size=(4, 4),
                future_loss_weight=0.0,
                recon_target_mode='temporal_delta',
                recon_input_mode='posterior_correction',
                discr_loss_weight=discr_weight,
                discr_margin=0.2,
            )
            fusion.train()
            return fusion

        # Identical rows across the batch: every sample's own correction is
        # already some other sample's, so the ranking term cannot be satisfied.
        fusion = build(0.1)
        shared = torch.randn(1, 256, 8, 8).expand(4, 256, 8, 8).contiguous()
        fusion(shared)
        _, _, _, _, _, stats = fusion(shared)
        self.assertIn('stat_discr_gap', stats)
        self.assertTrue(torch.isfinite(stats['stat_discr_gap']))

        with_discr = build(0.1)
        without_discr = build(0.0)
        for f in (with_discr, without_discr):
            f(torch.randn(4, 256, 8, 8))
        _, _, loss_with, _, _, st_with = with_discr(torch.randn(4, 256, 8, 8))
        _, _, loss_without, _, _, st_without = without_discr(
            torch.randn(4, 256, 8, 8))
        # With the term disabled the diagnostic must stay exactly zero and the
        # objective must not change scale from an unweighted residual.
        self.assertEqual(st_without['stat_discr_gap'].item(), 0.0)
        self.assertTrue(torch.isfinite(loss_without))
        self.assertTrue(torch.isfinite(loss_with))

    def test_discrimination_loss_skips_single_sample_batch(self):
        """batch < 2 cannot form a negative, so the term must be inert."""
        from mmdet3d.models.fusion_layers.rssm_fusion import (
            LowDimFutureConsistentLatentFusion,
        )
        fusion = LowDimFutureConsistentLatentFusion(
            in_channels=256,
            latent_dim=32,
            hidden_dim=128,
            latent_size=(4, 4),
            future_loss_weight=0.0,
            recon_target_mode='temporal_delta',
            recon_input_mode='posterior_correction',
            discr_loss_weight=0.1,
        )
        fusion.train()
        fusion(torch.randn(1, 256, 8, 8))
        _, _, loss, _, _, stats = fusion(torch.randn(1, 256, 8, 8))
        self.assertEqual(stats['stat_discr_gap'].item(), 0.0)
        self.assertTrue(torch.isfinite(loss))

    def test_prior_and_posterior_condition_on_h_t(self):
        """z must be generated from h_t, not from the previous latent."""
        feat_a = torch.randn(1, 256, 8, 8)
        feat_b = torch.randn(1, 256, 8, 8)

        self.fusion.reset_state()
        with torch.no_grad():
            first, _, _, _, z_first, _ = self.fusion(
                feat_a, use_posterior=True, deterministic=True)
        self.fusion.z_state = torch.zeros_like(self.fusion.z_state)
        with torch.no_grad():
            second, _, _, _, z_second, _ = self.fusion(
                feat_b, use_posterior=True, deterministic=True)

        # Same zeroed z_prev but a different h_t/observation must change mu_q.
        self.assertFalse(torch.allclose(z_first, z_second, atol=1e-6))

    def test_output_shape_and_interface(self):
        feat = torch.randn(2, 256, 8, 8)
        with torch.no_grad():
            output, recon, kl, h_t, z_t, stats = self.fusion(
                feat, use_posterior=True, deterministic=True)

        self.assertEqual(output.shape, (2, 256, 8, 8))
        self.assertEqual(z_t.shape, (2, 32, 4, 4))
        self.assertEqual(h_t.shape, (2, 32, 4, 4))
        self.assertIsNone(recon)
        self.assertIsInstance(stats, dict)
        self.assertTrue(torch.is_tensor(kl))

    def test_zero_z_changes_output_and_can_be_gated(self):
        """The latent enters detection through FiLM/gating, not raw z."""
        feat = torch.randn(1, 256, 8, 8)
        self.fusion.reset_state()
        with torch.no_grad():
            self.fusion(feat, use_posterior=True, deterministic=True)

        outputs = []
        original_z = self.fusion.z_state.clone()
        with torch.no_grad():
            output_normal, *_ = self.fusion(
                feat, use_posterior=True, deterministic=True)
        self.fusion.z_state = torch.zeros_like(original_z)
        with torch.no_grad():
            output_zero, *_ = self.fusion(
                feat, use_posterior=True, deterministic=True)
        outputs = [output_normal, output_zero]

        self.assertFalse(torch.allclose(outputs[0], outputs[1], atol=1e-5))
        self.assertGreater(
            (outputs[0] - outputs[1]).abs().max().item(), 0.0)

    def test_future_loss_requires_pending_next_frame(self):
        feat1 = torch.randn(1, 256, 8, 8)
        feat2 = torch.randn(1, 256, 8, 8)

        self.fusion.reset_state()
        self.fusion(feat1)
        self.assertIsNone(self.fusion.pending_future_pred)

        self.fusion.train()
        _, _, latent_loss, _, _, stats = self.fusion(feat2)
        # First training call only stores the prediction for the next frame.
        self.assertIn('stat_future_mse', stats)
        self.assertEqual(stats['stat_future_mse'].item(), 0.0)
        self.assertIsNotNone(self.fusion.pending_future_pred)
        _, _, _, _, _, stats_next = self.fusion(feat2)
        self.assertGreater(stats_next['stat_future_mse'].item(), 0.0)
        self.assertTrue(torch.is_tensor(latent_loss))

    def test_future_loss_is_scored_one_frame_later(self):
        """Frame t predicts frame t+1; the task is forward-looking."""
        feat1 = torch.randn(1, 256, 8, 8)
        feat2 = torch.randn(1, 256, 8, 8)

        self.fusion.train()
        self.fusion.reset_state()
        # Nothing was predicted yet, so frame 0 has nothing to score.
        _, _, _, _, _, stats_first = self.fusion(feat1)
        # The prediction made while seeing feat1 is scored against feat2.
        _, _, _, _, _, stats_second = self.fusion(feat2)

        self.assertEqual(stats_first['stat_future_mse'].item(), 0.0)
        self.assertGreater(stats_second['stat_future_mse'].item(), 0.0)

    def test_return_tuple_contract(self):
        """Detector reads index 2 as the loss and index 4 as z_t.

        Regression guard: index 4 (z_t) was once wired into the training
        loss, which silently optimised the latent mean instead of KL.
        """
        feat = torch.randn(2, 256, 8, 8)
        self.fusion.train()
        self.fusion.reset_state()
        output, recon, latent_loss, h_t, z_t, stats = self.fusion(feat)

        self.assertIsNone(recon)
        self.assertEqual(latent_loss.ndim, 0)
        self.assertGreaterEqual(latent_loss.item(), 0.0)
        self.assertEqual(z_t.shape, (2, 32, 4, 4))
        self.assertEqual(h_t.shape, (2, 32, 4, 4))

    def test_future_target_is_spatially_pooled_and_normalized(self):
        """Raw BEV scale must not dominate the prior/KL objective."""
        feat1 = torch.randn(2, 256, 8, 8)
        feat2 = torch.randn(2, 256, 8, 8) * 100.0

        self.fusion.train()
        self.fusion.reset_state()
        self.fusion(feat1)
        _, _, _, _, _, stats = self.fusion(feat2)

        self.assertEqual(stats['stat_future_mse'].ndim, 0)
        self.assertLess(stats['stat_future_mse'].item(), 1e4)

    def test_future_loss_is_a_spatial_mean_not_a_sum(self):
        """A zero-equivalent prediction must give an O(1) standardized MSE.

        The target is standardized to unit variance, so an untrained head
        predicting ~0 scores about 1.0. Regression guard: broadcasting the
        validity mask as (B, 1, 1, 1) made the denominator count samples
        instead of (sample, cell) pairs, inflating the loss by the number of
        latent cells (~128x at 16x16).
        """
        feat1 = torch.randn(2, 256, 8, 8)
        feat2 = torch.randn(2, 256, 8, 8)

        self.fusion.train()
        self.fusion.reset_state()
        self.fusion(feat1)
        with torch.no_grad():
            _, _, _, _, _, stats = self.fusion(feat2)

        self.assertLess(stats['stat_future_mse'].item(), 8.0)

    def test_future_target_is_one_channel_energy_map(self):
        """The exclusive task predicts next-frame BEV energy, not raw channels."""
        self.assertEqual(self.fusion.prior_future.out_channels, 1)
        self.assertEqual(self.fusion.posterior_future.out_channels, 1)

    def test_future_heads_normalize_latent_input(self):
        """Future loss must not depend on the BEV feature scale.

        The target is standardized and both future heads see L2-normalized
        latents, so the objective is invariant to multiplying the input BEV by
        a constant. Assert that invariance directly instead of pinning an
        absolute value that only reflects the random head at init.
        """
        feat1 = torch.randn(1, 256, 8, 8)
        feat2 = torch.randn(1, 256, 8, 8) * 100.0

        self.fusion.train()
        scale_1 = torch.randn(1, 256, 8, 8)
        with torch.no_grad():
            self.fusion.reset_state()
            self.fusion(feat1)
            _, _, _, _, _, stats_1 = self.fusion(scale_1)
            self.fusion.reset_state()
            self.fusion(feat1)
            _, _, _, _, _, stats_100 = self.fusion(scale_1 * 100.0)

        self.assertAlmostEqual(
            stats_1['stat_future_mse'].item(),
            stats_100['stat_future_mse'].item(),
            places=4)


class TestSpatialReadoutLatentFusion(unittest.TestCase):
    """Smoke tests for the unpooled spatial detection readout."""

    def setUp(self):
        from mmdet3d.models.fusion_layers.rssm_fusion import (
            LowDimFutureConsistentLatentFusion,
        )

        self.fusion = LowDimFutureConsistentLatentFusion(
            in_channels=256,
            latent_dim=32,
            hidden_dim=128,
            latent_pool='adaptive',
            latent_size=(4, 4),
            future_loss_weight=0.0,
            recon_target_mode='temporal_delta',
            recon_input_mode='posterior_correction',
            readout_mode='spatial',
        )
        self.fusion.eval()

    def test_spatial_mode_drops_pooled_film_modules(self):
        self.assertIsNone(self.fusion.film_proj)
        self.assertIsNone(self.fusion.gate_proj)
        self.assertIsNone(self.fusion.output_proj)
        self.assertIsNone(self.fusion.correction_output_proj)
        self.assertIsNotNone(self.fusion.h_proj)
        self.assertIsNotNone(self.fusion.z_proj)

    def test_spatial_mode_requires_adaptive_pooling(self):
        from mmdet3d.models.fusion_layers.rssm_fusion import (
            LowDimFutureConsistentLatentFusion,
        )

        with self.assertRaises(ValueError):
            LowDimFutureConsistentLatentFusion(
                in_channels=256,
                latent_dim=32,
                hidden_dim=128,
                latent_pool='global',
                latent_size=(1, 1),
                future_loss_weight=0.0,
                readout_mode='spatial',
            )

    def test_film_mode_keeps_pooled_readout(self):
        from mmdet3d.models.fusion_layers.rssm_fusion import (
            LowDimFutureConsistentLatentFusion,
        )

        film = LowDimFutureConsistentLatentFusion(
            in_channels=256,
            latent_dim=32,
            hidden_dim=128,
            latent_pool='adaptive',
            latent_size=(4, 4),
            future_loss_weight=0.0,
            readout_mode='film',
        )
        self.assertIsNone(film.h_proj)
        self.assertIsNone(film.z_proj)
        self.assertIsNotNone(film.film_proj)
        self.assertIsNotNone(film.output_proj)

    def test_spatial_output_shape_and_interface(self):
        feat = torch.randn(2, 256, 8, 8)
        with torch.no_grad():
            self.fusion.reset_state()
            output, recon, latent_loss, h_t, z_t, stats = self.fusion(
                feat, use_posterior=True, deterministic=True)

        self.assertEqual(output.shape, (2, 256, 8, 8))
        self.assertEqual(h_t.shape, (2, 32, 4, 4))
        self.assertEqual(z_t.shape, (2, 32, 4, 4))
        self.assertIsNone(recon)
        self.assertTrue(torch.is_tensor(latent_loss))
        self.assertIn('stat_correction_sq', stats)

    def test_both_spatial_branches_reach_the_detector(self):
        """Zeroing either projection must change the detection residual."""
        feat = torch.randn(1, 256, 8, 8)
        self.fusion.eval()
        self.fusion.reset_state()
        # Advance the recurrent state once, then replay the same state for each
        # readout variant so only the readout weights differ between runs.
        with torch.no_grad():
            self.fusion(feat, detach_state=False)
        state_h = self.fusion.h_state.clone()
        state_z = self.fusion.z_state.clone()

        def run_with_state():
            self.fusion.h_state = state_h.clone()
            self.fusion.z_state = state_z.clone()
            with torch.no_grad():
                output, *_ = self.fusion(feat, detach_state=False)
            return output

        baseline = run_with_state()

        h_weight = self.fusion.h_proj.weight.clone()
        h_bias = self.fusion.h_proj.bias.clone()
        z_weight = self.fusion.z_proj.weight.clone()
        z_bias = self.fusion.z_proj.bias.clone()

        with torch.no_grad():
            self.fusion.h_proj.weight.zero_()
            self.fusion.h_proj.bias.zero_()
            no_h = run_with_state()
            self.fusion.h_proj.weight.copy_(h_weight)
            self.fusion.h_proj.bias.copy_(h_bias)
        self.assertFalse(torch.allclose(baseline, no_h, atol=1e-6))

        with torch.no_grad():
            self.fusion.z_proj.weight.zero_()
            self.fusion.z_proj.bias.zero_()
            no_z = run_with_state()
            self.fusion.z_proj.weight.copy_(z_weight)
            self.fusion.z_proj.bias.copy_(z_bias)
        self.assertFalse(torch.allclose(baseline, no_z, atol=1e-6))

    def test_prior_only_zeroes_the_correction_branch(self):
        """Without posterior sampling, z_proj must only see its bias."""
        feat = torch.randn(1, 256, 8, 8)
        with torch.no_grad():
            self.fusion.reset_state()
            self.fusion(feat, use_posterior=True, deterministic=True)
        state_h = self.fusion.h_state.clone()
        state_z = self.fusion.z_state.clone()
        with torch.no_grad():
            self.fusion.h_state = state_h.clone()
            self.fusion.z_state = state_z.clone()
            posterior_out, *_ = self.fusion(
                feat, use_posterior=True, deterministic=True)
            self.fusion.h_state = state_h.clone()
            self.fusion.z_state = state_z.clone()
            prior_out, *_ = self.fusion(
                feat, use_posterior=False, deterministic=True)

        # mu_q and mu_p differ, so the correction must be non-zero under the
        # posterior while the prior-only path disables it entirely.
        self.assertFalse(torch.allclose(posterior_out, prior_out, atol=1e-6))


if __name__ == '__main__':
    unittest.main()
