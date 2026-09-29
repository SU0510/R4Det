"""Unit tests for RSSM fusion module and KL scale scheduler hook.

Run: python mmdet3d/models/fusion_layers/test_rssm_fusion.py
"""

import torch
import torch.nn.functional as F
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


    def test_posterior_obs_mode_switches_the_normalization(self):
        """scaled_raw must keep sample-to-sample amplitude differences."""
        from mmdet3d.models.fusion_layers.rssm_fusion import (
            LowDimFutureConsistentLatentFusion,
        )

        common = dict(
            in_channels=256,
            latent_dim=32,
            hidden_dim=128,
            latent_size=(4, 4),
            future_loss_weight=0.0,
            readout_mode='spatial',
        )
        l2 = LowDimFutureConsistentLatentFusion(
            posterior_obs_mode='l2', **common)
        scaled = LowDimFutureConsistentLatentFusion(
            posterior_obs_mode='scaled_raw', posterior_obs_scale=0.1, **common)

        # A batch-common component plus a per-sample offset: L2 normalization
        # divides by a norm dominated by the common part, while the shared
        # scalar preserves the relative offsets.
        base = torch.full((2, 4, 1, 1), 10.0)
        base[1] += 1.0
        l2_out = l2._posterior_observation(base)
        raw_out = scaled._posterior_observation(base)

        self.assertGreater(
            (raw_out[0] - raw_out[1]).abs().mean().item(), 0.0)
        # L2 collapses the pair toward the same unit vector.
        self.assertLess(
            (l2_out[0] - l2_out[1]).abs().mean().item(),
            (raw_out[0] - raw_out[1]).abs().mean().item())
        self.assertAlmostEqual(
            raw_out[1, 0, 0, 0].item(), 1.1, places=5)

    def test_scaled_raw_scale_must_be_positive(self):
        from mmdet3d.models.fusion_layers.rssm_fusion import (
            LowDimFutureConsistentLatentFusion,
        )

        with self.assertRaises(ValueError):
            LowDimFutureConsistentLatentFusion(
                in_channels=256,
                latent_dim=32,
                hidden_dim=128,
                latent_size=(4, 4),
                future_loss_weight=0.0,
                posterior_obs_mode='scaled_raw',
                posterior_obs_scale=0.0,
            )

    def test_z_proj_xavier_is_larger_than_small(self):
        from mmdet3d.models.fusion_layers.rssm_fusion import (
            LowDimFutureConsistentLatentFusion,
        )

        def rms(init):
            fusion = LowDimFutureConsistentLatentFusion(
                in_channels=256,
                latent_dim=32,
                hidden_dim=128,
                latent_size=(4, 4),
                future_loss_weight=0.0,
                readout_mode='spatial',
                z_proj_init=init,
            )
            return fusion.z_proj.weight.detach().pow(2).mean().sqrt().item()

        small = rms('small')
        xavier = rms('xavier')
        self.assertAlmostEqual(small, 0.01, places=3)
        # xavier_init for 32->256 1x1 gives ~8-10x the small-init scale.
        self.assertGreater(xavier / small, 5.0)
        self.assertLess(xavier / small, 20.0)

    def test_z_proj_xavier_requires_spatial_readout(self):
        from mmdet3d.models.fusion_layers.rssm_fusion import (
            LowDimFutureConsistentLatentFusion,
        )

        with self.assertRaises(ValueError):
            LowDimFutureConsistentLatentFusion(
                in_channels=256,
                latent_dim=32,
                hidden_dim=128,
                latent_size=(4, 4),
                future_loss_weight=0.0,
                readout_mode='film',
                z_proj_init='xavier',
            )


class TestInnovationPosteriorLatentFusion(unittest.TestCase):
    """Route B: Kalman-style predict/correct posterior."""

    def _build(self, **overrides):
        from mmdet3d.models.fusion_layers.rssm_fusion import (
            LowDimFutureConsistentLatentFusion,
        )

        kwargs = dict(
            in_channels=256,
            latent_dim=32,
            hidden_dim=128,
            latent_pool='adaptive',
            latent_size=(4, 4),
            future_loss_weight=0.0,
            recon_target_mode='temporal_delta',
            recon_input_mode='posterior_correction',
            readout_mode='spatial',
            posterior_obs_mode='scaled_raw',
            posterior_obs_scale=0.1,
            posterior_struct='innovation',
        )
        kwargs.update(overrides)
        return LowDimFutureConsistentLatentFusion(**kwargs)

    def test_innovation_mode_builds_obs_prior(self):
        fusion = self._build()
        self.assertIsNotNone(fusion.obs_prior)
        self.assertEqual(fusion.obs_prior.in_channels, fusion.latent_dim)
        self.assertEqual(fusion.obs_prior.out_channels, fusion.latent_dim)

    def test_standard_mode_has_no_obs_prior(self):
        from mmdet3d.models.fusion_layers.rssm_fusion import (
            LowDimFutureConsistentLatentFusion,
        )

        fusion = LowDimFutureConsistentLatentFusion(
            in_channels=256,
            latent_dim=32,
            hidden_dim=128,
            latent_size=(4, 4),
            future_loss_weight=0.0,
        )
        self.assertIsNone(fusion.obs_prior)

    def test_invalid_posterior_struct_rejected(self):
        with self.assertRaises(ValueError):
            self._build(posterior_struct='bogus')

    def test_negative_obs_pred_weight_rejected(self):
        with self.assertRaises(ValueError):
            self._build(obs_pred_loss_weight=-0.1)

    def test_correction_equals_delta_not_mu_difference(self):
        """In innovation mode the correction must be delta_mu itself.

        The readout consumes `posterior_correction`; if it were still
        `mu_q - mu_p` the whole point of the route (correction = the model's
        own predicted delta) would be lost even though mu_q is unchanged.
        """
        fusion = self._build()
        fusion.train()
        feat = torch.randn(2, 256, 8, 8)
        captured = {}

        def hook(module, inputs, output):
            captured['delta'] = output.detach().clone()

        handle = fusion.posterior_mu.register_forward_hook(hook)
        _, _, _, _, _, stats = fusion(feat)
        handle.remove()

        self.assertIn('delta', captured)
        self.assertAlmostEqual(
            stats['stat_correction_sq'].item(),
            captured['delta'].square().mean().item(),
            places=8)

    def test_obs_prior_only_receives_observation_prediction_gradient(self):
        """obs_prior must not be reachable from the reconstruction path."""
        fusion = self._build(recon_loss_weight=0.0, kl_scale=0.0)
        fusion.train()
        feat = torch.randn(2, 256, 8, 8)
        _, _, loss, _, _, _ = fusion(feat)
        loss.backward()

        self.assertIsNotNone(fusion.obs_prior.weight.grad)
        self.assertGreater(
            fusion.obs_prior.weight.grad.abs().sum().item(), 0.0)

    def test_obs_prediction_term_is_reported(self):
        fusion = self._build()
        fusion.train()
        _, _, loss, _, _, stats = fusion(torch.randn(2, 256, 8, 8))

        self.assertIn('stat_obs_pred', stats)
        self.assertIn('stat_innovation_sq', stats)
        self.assertGreaterEqual(stats['stat_obs_pred'].item(), 0.0)
        self.assertGreater(stats['stat_innovation_sq'].item(), 0.0)
        self.assertTrue(torch.isfinite(loss))

    def test_innovation_mode_runs_two_frames_and_backprops(self):
        fusion = self._build()
        fusion.train()
        fusion(torch.randn(2, 256, 8, 8))
        _, _, loss, _, _, stats = fusion(
            torch.randn(2, 256, 8, 8), detach_state=False)

        self.assertGreater(stats['stat_recon_valid_frac'].item(), 0.0)
        loss.backward()
        self.assertIsNotNone(fusion.posterior_mu.weight.grad)
        self.assertGreater(
            fusion.posterior_mu.weight.grad.abs().sum().item(), 0.0)

    def test_innovation_mode_keeps_six_tuple_contract(self):
        fusion = self._build()
        with torch.no_grad():
            out = fusion(torch.randn(2, 256, 8, 8))
        self.assertEqual(len(out), 6)
        self.assertEqual(out[0].shape, (2, 256, 8, 8))
        self.assertIsNone(out[1])
        self.assertEqual(out[3].shape, (2, 32, 4, 4))
        self.assertEqual(out[4].shape, (2, 32, 4, 4))
        self.assertIsInstance(out[5], dict)

    def test_reset_state_clears_obs_prior_free_state(self):
        fusion = self._build()
        fusion.train()
        fusion(torch.randn(2, 256, 8, 8))
        fusion.reset_state()
        self.assertIsNone(fusion.h_state)
        self.assertIsNone(fusion.z_state)


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


class TestZPrevNormalizeAblation(unittest.TestCase):
    """Route-B follow-up: the `z_prev_normalize` single-variable switch."""

    def _build(self, **overrides):
        from mmdet3d.models.fusion_layers.rssm_fusion import (
            LowDimFutureConsistentLatentFusion,
        )

        kwargs = dict(
            in_channels=256,
            latent_dim=32,
            hidden_dim=128,
            latent_pool='adaptive',
            latent_size=(4, 4),
            future_loss_weight=0.0,
            recon_target_mode='temporal_delta',
            recon_input_mode='posterior_correction',
            readout_mode='spatial',
            posterior_obs_mode='scaled_raw',
            posterior_obs_scale=0.1,
            posterior_struct='innovation',
        )
        kwargs.update(overrides)
        return LowDimFutureConsistentLatentFusion(**kwargs)

    def test_default_is_normalize(self):
        """The switch must default to the historical behaviour."""
        fusion = self._build()
        self.assertTrue(fusion.z_prev_normalize)

    def test_non_bool_rejected(self):
        with self.assertRaises(ValueError):
            self._build(z_prev_normalize='yes')

    def test_raw_changes_gru_input_magnitude(self):
        """`raw` must feed the unnormalized pooled z into the GRU.

        With a deliberately inflated z_state the normalized branch produces a
        unit-norm GRU input while the raw branch keeps the original scale, so
        the two branches cannot produce identical downstream latents.
        """
        fusion = self._build(z_prev_normalize=True)
        fusion.eval()
        feat = torch.randn(2, 256, 8, 8)
        with torch.no_grad():
            for _ in range(3):
                fusion(feat, use_posterior=True, deterministic=True)
            state_h = fusion.h_state.clone()
            fusion.z_state.mul_(10.0)
            state_z = fusion.z_state.clone()
            norm_before = fusion._pool_latent(
                F.normalize(fusion._pool_latent(fusion.z_state), dim=1)
            ).norm(dim=1).mean().item()
            out_norm, *_ = fusion(
                feat, use_posterior=True, deterministic=True)

        fusion_raw = self._build(z_prev_normalize=False)
        fusion_raw.load_state_dict(fusion.state_dict(), strict=False)
        fusion_raw.eval()
        with torch.no_grad():
            fusion_raw.h_state = state_h.clone()
            fusion_raw.z_state = state_z.clone()
            norm_after = fusion_raw._pool_latent(
                fusion_raw.z_state).norm(dim=1).mean().item()
            out_raw, *_ = fusion_raw(
                feat, use_posterior=True, deterministic=True)

        # Normalizing collapses every channel vector to unit norm; the raw
        # branch keeps the inflated per-channel magnitude.
        self.assertAlmostEqual(norm_before, 1.0, places=3)
        self.assertGreater(norm_after, 5.0)
        self.assertFalse(torch.allclose(out_norm, out_raw, atol=1e-6))


class TestObjectCentricLatentSupervision(unittest.TestCase):
    """Route C: auxiliary class-heatmap head on the posterior correction."""

    def _build(self, **overrides):
        from mmdet3d.models.fusion_layers.rssm_fusion import (
            LowDimFutureConsistentLatentFusion,
        )

        kwargs = dict(
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
        kwargs.update(overrides)
        return LowDimFutureConsistentLatentFusion(**kwargs)

    @staticmethod
    def _target():
        target = torch.zeros(2, 4, 4, 4)
        target[0, 0, 1, 2] = 1.0
        target[1, 2, 3, 0] = 1.0
        return target

    def test_default_weight_zero_is_bitwise_inert(self):
        """An un-configured run must not see the auxiliary branch at all."""
        import copy

        fusion = self._build()
        baseline = copy.deepcopy(fusion)
        feat = torch.randn(2, 256, 8, 8)

        fusion.train()
        fusion.set_current_object_target(self._target())
        _, _, loss_with_target, _, _, stats = fusion(feat)

        baseline.train()
        baseline.reset_state()
        _, _, loss_without, _, _, baseline_stats = baseline(feat)

        self.assertEqual(stats['stat_object_hm'].item(), 0.0)
        self.assertEqual(stats['stat_object_hm_recall'].item(), 0.0)
        self.assertEqual(stats['stat_object_hm_pos'].item(), 0.0)
        self.assertEqual(
            loss_with_target.item(), loss_without.item())
        self.assertEqual(
            baseline_stats['stat_object_hm'].item(), 0.0)

    def test_head_consumes_correction_not_h_t(self):
        """The auxiliary head must see exactly the posterior correction."""
        fusion = self._build(latent_object_loss_weight=0.05)
        fusion.train()
        captured = {}

        def pre_hook(module, inputs):
            captured['head_in'] = inputs[0].detach()

        handle = fusion.object_head.register_forward_pre_hook(pre_hook)
        fusion.set_current_object_target(self._target())
        _, _, _, _, _, stats = fusion(torch.randn(2, 256, 8, 8))
        handle.remove()

        self.assertAlmostEqual(
            captured['head_in'].square().mean().item(),
            stats['stat_correction_sq'].item(), places=6)

    def test_zero_weight_never_runs_head(self):
        fusion = self._build()
        fusion.train()
        called = []
        handle = fusion.object_head.register_forward_hook(
            lambda *args: called.append(1))
        fusion.set_current_object_target(self._target())
        fusion(torch.randn(2, 256, 8, 8))
        handle.remove()
        self.assertEqual(called, [])

    def test_detection_output_is_independent_of_the_object_head(self):
        """Adding the auxiliary target must not perturb the fused features.

        The head is a training-only side branch: `forward` never reads its
        output, so the detector-facing tensor and the returned h_t/z_t must be
        bit-identical whether or not a target is attached. This also pins the
        "zero inference cost" claim: with the branch removed at inference the
        numbers cannot change.
        """
        import copy

        fusion = self._build(latent_object_loss_weight=0.1)
        baseline = copy.deepcopy(fusion)
        feat = torch.randn(2, 256, 8, 8)

        fusion.train()
        fusion.set_current_object_target(self._target())
        out_with, _, loss_with, h_with, z_with, stats = fusion(
            feat, deterministic=True)

        baseline.train()
        out_without, _, loss_without, h_without, z_without, _ = baseline(
            feat, deterministic=True)

        self.assertTrue(torch.equal(out_with, out_without))
        self.assertTrue(torch.equal(h_with, h_without))
        self.assertTrue(torch.equal(z_with, z_without))
        self.assertGreater(stats['stat_object_hm'].item(), 0.0)
        self.assertNotEqual(loss_with.item(), loss_without.item())

    def test_target_shape_is_validated(self):
        fusion = self._build(latent_object_loss_weight=0.05)
        with self.assertRaises(ValueError):
            fusion.set_current_object_target(torch.zeros(2, 3, 4, 4))
        with self.assertRaises(ValueError):
            fusion.set_current_object_target(torch.zeros(2, 4, 4, 4, 4))

    def test_batch_mismatch_is_rejected_at_loss_time(self):
        fusion = self._build(latent_object_loss_weight=0.05)
        fusion.train()
        fusion.set_current_object_target(torch.zeros(3, 4, 4, 4))
        with self.assertRaises(ValueError):
            fusion(torch.randn(2, 256, 8, 8))

    def test_object_loss_is_sole_gradient_source_when_others_disabled(self):
        """The term must actually reach posterior_correction/mu_q."""
        fusion = self._build(
            latent_object_loss_weight=0.05,
            kl_scale=0.0,
            recon_loss_weight=0.0,
            obs_pred_loss_weight=0.0,
        )
        fusion.train()
        fusion.set_current_object_target(self._target())
        _, _, loss, _, _, stats = fusion(torch.randn(2, 256, 8, 8))
        self.assertGreater(stats['stat_object_hm'].item(), 0.0)
        loss.backward()

        self.assertIsNotNone(fusion.object_head[0].weight.grad)
        self.assertGreater(
            fusion.object_head[0].weight.grad.abs().sum().item(), 0.0)
        self.assertIsNotNone(fusion.posterior_mu.weight.grad)
        self.assertGreater(
            fusion.posterior_mu.weight.grad.abs().sum().item(), 0.0)
        # h_proj leads to h_t only; it must stay untouched by this term.
        self.assertTrue(
            fusion.h_proj.weight.grad is None
            or fusion.h_proj.weight.grad.abs().sum().item() == 0.0)

    def test_empty_target_contributes_exactly_zero(self):
        import copy

        fusion = self._build(latent_object_loss_weight=0.05)
        baseline = copy.deepcopy(fusion)
        feat = torch.randn(2, 256, 8, 8)

        fusion.train()
        fusion.set_current_object_target(torch.zeros(2, 4, 4, 4))
        _, _, loss, _, _, stats = fusion(feat)
        baseline.train()
        _, _, baseline_loss, _, _, _ = baseline(feat)

        self.assertEqual(stats['stat_object_hm'].item(), 0.0)
        self.assertEqual(stats['stat_object_hm_pos'].item(), 0.0)
        self.assertEqual(loss.item(), baseline_loss.item())

    def test_diagnostics_do_not_contain_loss_substring(self):
        fusion = self._build(latent_object_loss_weight=0.05)
        fusion.train()
        fusion.set_current_object_target(self._target())
        _, _, _, _, _, stats = fusion(torch.randn(2, 256, 8, 8))
        for key in stats:
            self.assertNotIn('loss', key)
        self.assertIn('stat_object_hm', stats)
        self.assertIn('stat_object_hm_recall', stats)
        self.assertIn('stat_object_hm_pos', stats)

    def _stats_with_fixed_logits(self, logits):
        """Run one forward with the auxiliary head replaced by `logits`."""
        import torch.nn as nn

        fusion = self._build(latent_object_loss_weight=0.05)
        fusion.train()

        class _FixedHead(nn.Module):
            def __init__(self, value):
                super().__init__()
                self.register_buffer('value', value)

            def forward(self, _x):
                return self.value

        fusion.object_head = _FixedHead(logits)
        fusion.set_current_object_target(self._target())
        _, _, _, _, _, stats = fusion(torch.randn(2, 256, 8, 8))
        return stats

    def test_hit1_is_one_when_head_peaks_at_centers(self):
        """Hit@1 counts a GT centre when its cell wins the class argmax."""
        logits = torch.full((2, 4, 4, 4), -10.0)
        logits[0, 0, 1, 2] = 10.0
        logits[1, 2, 3, 0] = 10.0
        stats = self._stats_with_fixed_logits(logits)

        self.assertAlmostEqual(
            stats['stat_object_hm_recall'].item(), 1.0, places=5)
        self.assertAlmostEqual(
            stats['stat_object_hm_thr50'].item(), 1.0, places=5)
        self.assertEqual(stats['stat_object_hm_pos'].item(), 2.0)

    def test_hit1_is_zero_when_head_peaks_elsewhere(self):
        """A class peak on a background cell must not count as a hit."""
        logits = torch.full((2, 4, 4, 4), -10.0)
        logits[0, 0, 0, 0] = 10.0
        logits[1, 2, 0, 0] = 10.0
        stats = self._stats_with_fixed_logits(logits)

        self.assertEqual(stats['stat_object_hm_recall'].item(), 0.0)
        self.assertEqual(stats['stat_object_hm_thr50'].item(), 0.0)

    def test_hit1_counts_once_per_occupied_channel(self):
        """Extra GT centres in one channel must not inflate hit@1 above 1."""
        logits = torch.full((2, 4, 4, 4), -10.0)
        logits[0, 0, 1, 2] = 10.0
        logits[1, 2, 3, 0] = 10.0
        import torch.nn as nn

        fusion = self._build(latent_object_loss_weight=0.05)
        fusion.train()

        class _FixedHead(nn.Module):
            def __init__(self, value):
                super().__init__()
                self.register_buffer('value', value)

            def forward(self, _x):
                return self.value

        fusion.object_head = _FixedHead(logits)
        target = self._target()
        target[0, 0, 3, 3] = 1.0
        fusion.set_current_object_target(target)
        _, _, _, _, _, stats = fusion(torch.randn(2, 256, 8, 8))

        self.assertLessEqual(stats['stat_object_hm_recall'].item(), 1.0)
        self.assertEqual(stats['stat_object_hm_pos'].item(), 3.0)


class TestBevCentersToHeatmap(unittest.TestCase):
    """Geometry of the route-C auxiliary targets."""

    class _Boxes:
        def __init__(self, centers, dims):
            self.gravity_center = centers
            self.tensor = torch.cat(
                [centers, dims, torch.zeros(centers.shape[0], 1)], dim=1)

    def setUp(self):
        from mmdet3d.core.utils.gaussian import bev_centers_to_heatmap

        self.build = bev_centers_to_heatmap
        self.pc_range = [0.0, -39.68, -4.0, 69.12, 39.68, 2.0]

    def test_center_lands_in_expected_cell(self):
        # x = half the range -> row 8, y = 0 -> col 8 of a 16x16 grid.
        boxes = [self._Boxes(
            torch.tensor([[34.56, 0.0, 0.0]]),
            torch.tensor([[4.32, 4.96, 1.5]]))]
        labels = [torch.tensor([0])]
        heatmap = self.build(
            boxes, labels, num_classes=4, out_size=(16, 16),
            point_cloud_range=self.pc_range, min_radius=1)

        self.assertEqual(heatmap.shape, (1, 4, 16, 16))
        self.assertAlmostEqual(
            heatmap[0, 0, 8, 8].item(), 1.0, places=5)
        self.assertEqual(heatmap[0, 1:].sum().item(), 0.0)
        # Gaussian neighbourhood must spread to adjacent cells.
        self.assertGreater(heatmap[0, 0, 8, 9].item(), 0.0)

    def test_label_selects_channel(self):
        boxes = [self._Boxes(
            torch.tensor([[34.56, 0.0, 0.0]]),
            torch.tensor([[4.32, 4.96, 1.5]]))]
        labels = [torch.tensor([2])]
        heatmap = self.build(
            boxes, labels, num_classes=4, out_size=(16, 16),
            point_cloud_range=self.pc_range)
        self.assertAlmostEqual(
            heatmap[0, 2, 8, 8].item(), 1.0, places=5)
        self.assertEqual(heatmap[0, 0].sum().item(), 0.0)

    def test_out_of_range_and_empty_boxes_are_dropped(self):
        boxes = [
            self._Boxes(
                torch.tensor([[-10.0, 0.0, 0.0], [200.0, 0.0, 0.0]]),
                torch.tensor([[4.32, 4.96, 1.5]]).repeat(2, 1)),
            self._Boxes(torch.zeros(0, 3), torch.zeros(0, 3)),
        ]
        labels = [torch.tensor([0, 1]), torch.zeros(0, dtype=torch.long)]
        heatmap = self.build(
            boxes, labels, num_classes=4, out_size=(16, 16),
            point_cloud_range=self.pc_range)
        self.assertEqual(heatmap.sum().item(), 0.0)

    def test_invalid_label_is_skipped(self):
        boxes = [self._Boxes(
            torch.tensor([[34.56, 0.0, 0.0]]),
            torch.tensor([[4.32, 4.96, 1.5]]))]
        labels = [torch.tensor([7])]
        heatmap = self.build(
            boxes, labels, num_classes=4, out_size=(16, 16),
            point_cloud_range=self.pc_range)
        self.assertEqual(heatmap.sum().item(), 0.0)

    def test_vectorized_radius_matches_scalar_reference(self):
        """The batched radius helper must agree with CenterHead's scalar one."""
        from mmdet3d.core.utils.gaussian import (
            gaussian_radius, gaussian_radius_batch)

        torch.manual_seed(0)
        height = torch.rand(64) * 20.0 + 0.1
        width = torch.rand(64) * 20.0 + 0.1
        batch_radii = gaussian_radius_batch(height, width)
        scalar_radii = torch.stack([
            gaussian_radius((height[i], width[i]))
            for i in range(height.shape[0])
        ])
        self.assertTrue(torch.allclose(
            batch_radii, scalar_radii, atol=1e-6))


class TestMotionAlignedInnovationRSSMFusion(unittest.TestCase):
    """Regression tests for the full-resolution motion-aligned innovation RSSM."""

    def _build(self, **overrides):
        from mmdet3d.models.fusion_layers.rssm_fusion import (
            MotionAlignedInnovationRSSMFusion)

        kwargs = dict(
            in_channels=256,
            out_channels=256,
            latent_dim=256,
            hidden_dim=64,
            action_dim=2,
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

    def _build_small(self, **overrides):
        kwargs = dict(
            in_channels=32, out_channels=32, latent_dim=32, hidden_dim=16,
        )
        kwargs.update(overrides)
        return self._build(**kwargs)

    def test_negative_obs_pred_weight_rejected(self):
        with self.assertRaises(ValueError):
            self._build(obs_pred_loss_weight=-0.1)

    def test_interface_matches_motion_aligned_rssm(self):
        """Six-tuple return with the same shapes as the standard main line."""
        fusion = self._build()
        B, C, H, W = 2, 256, 8, 8
        logstd_bias = fusion.obs_prior.bias.detach().clone()
        feat = torch.randn(B, C, H, W)

        with torch.no_grad():
            out = fusion(feat, use_posterior=True, deterministic=True)

        self.assertEqual(len(out), 6)
        output, recon, latent_loss, h_t, z_t, stats = out
        self.assertEqual(tuple(output.shape), (B, C, H, W))
        self.assertEqual(tuple(recon.shape), (B, C, H, W))
        self.assertEqual(tuple(h_t.shape), (B, 256, H, W))
        self.assertEqual(tuple(z_t.shape), (B, 256, H, W))
        self.assertTrue(torch.isfinite(latent_loss).all())
        # obs_prior must not have been re-initialized away from the parent's
        # RNG stream: its own init is separate and its forward ran.
        self.assertEqual(tuple(fusion.obs_prior.bias.shape),
                         tuple(logstd_bias.shape))
        # supervised diagnostics present, and named without the 'loss'
        # substring so mmdet's _parse_losses does not fold them in.
        self.assertIn('stat_obs_pred', stats)
        self.assertIn('stat_innovation_sq', stats)
        self.assertNotIn('loss', 'stat_obs_pred')
        self.assertNotIn('loss', 'stat_innovation_sq')

    def test_observation_path_survives_zero_obs_prior_weight(self):
        """With weight 0 the innovation term still must not detach the graph.

        `innovation = e_t - stopgrad(e_hat)`; even at weight 0 the posterior
        must remain differentiable w.r.t. the encoder output so the detector
        loss can still train the observation pathway.
        """
        fusion = self._build(obs_pred_loss_weight=0.0)
        fusion.train()
        B, C, H, W = 2, 256, 8, 8
        feat = torch.randn(B, C, H, W, requires_grad=True)

        out, _, latent_loss, _, z_t, _ = fusion(
            feat, use_posterior=True, deterministic=True)
        (out.mean() + latent_loss).backward()
        self.assertIsNotNone(fusion.encoder[0].conv.weight.grad)
        self.assertGreater(
            fusion.encoder[0].conv.weight.grad.abs().sum().item(), 0.0)

    def test_obs_prior_trained_only_by_observation_prediction(self):
        """obs_prior must receive gradient from its own term, not from z_t.

        The prediction term is the only path into `obs_prior`: the innovation
        uses `e_hat.detach()`, and no other branch reads `e_hat`. So a
        backward pass on the latent loss alone must leave that term's
        contribution nonzero while `delta_mu` contributes through the
        posterior instead.
        """
        fusion = self._build_small(obs_pred_loss_weight=0.05)
        B, C, H, W = 2, 32, 8, 8
        feat = torch.randn(B, 32, H, W)
        out, _, latent_loss, _, _, _ = fusion(
            feat, use_posterior=True, deterministic=True)
        latent_loss.backward()
        self.assertIsNotNone(fusion.obs_prior.weight.grad)
        self.assertGreater(
            fusion.obs_prior.weight.grad.abs().sum().item(), 0.0)

    def test_stats_do_not_leak_into_total_loss(self):
        """Diagnostics must avoid the 'loss' substring entirely."""
        fusion = self._build_small()
        B, C, H, W = 1, 32, 8, 8
        feat = torch.randn(B, 32, H, W)
        with torch.no_grad():
            *_, stats = fusion(feat, use_posterior=True, deterministic=True)
        for key in stats:
            self.assertNotIn('loss', key)

    def test_use_prior_only_falls_back_to_prior_mean(self):
        fusion = self._build_small()
        B, C, H, W = 1, 32, 8, 8
        feat = torch.randn(B, 32, H, W)
        fusion.reset_state()
        with torch.no_grad():
            _, _, _, _, z_post, _ = fusion(
                feat, use_posterior=True, deterministic=True)
        fusion.reset_state()
        with torch.no_grad():
            _, _, _, _, z_prior, _ = fusion(
                feat, use_posterior=False, deterministic=True)
        self.assertFalse(torch.allclose(z_post, z_prior))

    def test_deterministic_replay_is_bitwise_identical(self):
        fusion = self._build_small()
        B, C, H, W = 1, 32, 8, 8
        feat = torch.randn(B, 32, H, W)

        fusion.reset_state()
        with torch.no_grad():
            out_a, _, _, _, _, _ = fusion(
                feat, use_posterior=True, deterministic=True)
        fusion.reset_state()
        with torch.no_grad():
            out_b, _, _, _, _, _ = fusion(
                feat, use_posterior=True, deterministic=True)
        self.assertTrue(torch.equal(out_a, out_b))

    def test_state_carries_across_calls_and_resets(self):
        fusion = self._build_small()
        B, C, H, W = 1, 32, 8, 8
        f1 = torch.randn(B, 32, H, W)
        f2 = torch.randn(B, 32, H, W)

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
        # Second frame differs with vs without history.
        self.assertFalse(torch.allclose(out_second, out_fresh))

    def test_added_modules_preserve_parent_rng_stream(self):
        """Constructing the subclass must not shift the parent's init RNG.

        Existing checkpoints only carry the parent's keys; if the added
        obs_prior consumed RNG before the parent finished initializing, every
        inherited parameter would silently land on different values and the
        loaded baseline weights would no longer be reproducible.
        """
        from mmdet3d.models.fusion_layers.rssm_fusion import (
            MotionAlignedRSSMFusion)

        common = dict(
            in_channels=32, out_channels=32, latent_dim=32, hidden_dim=16,
            action_dim=2, kl_scale=0.1, free_nats=1.0, min_std=0.1,
            init_std=0.2, align_kernel_size=3, align_deform_groups=1,
            align_z_state=True,
        )
        torch.manual_seed(1234)
        parent = MotionAlignedRSSMFusion(**common)
        parent_state = {
            k: v.detach().clone() for k, v in parent.state_dict().items()}

        torch.manual_seed(1234)
        child = self._build(**common)
        for key, value in parent_state.items():
            self.assertTrue(
                torch.equal(child.state_dict()[key], value),
                msg=f'inherited parameter {key} drifted from parent init')

    def test_shared_checkpoint_keys_load_without_obs_prior(self):
        """A plain main-line checkpoint must load into this variant."""
        from mmdet3d.models.fusion_layers.rssm_fusion import (
            MotionAlignedRSSMFusion)

        common = dict(
            in_channels=32, out_channels=32, latent_dim=32, hidden_dim=16,
            action_dim=0, kl_scale=1.0, free_nats=1.0, min_std=0.1,
            init_std=0.2, align_kernel_size=3, align_deform_groups=1,
            align_z_state=True,
        )
        torch.manual_seed(7)
        parent = MotionAlignedRSSMFusion(**common)
        torch.manual_seed(7)
        child = self._build(**common)
        # only obs_prior.* is absent from the parent checkpoint.
        missing, unexpected = child.load_state_dict(
            parent.state_dict(), strict=False)
        self.assertTrue(all(k.startswith('obs_prior.') for k in missing))
        self.assertEqual(unexpected, [])


if __name__ == '__main__':
    unittest.main()
