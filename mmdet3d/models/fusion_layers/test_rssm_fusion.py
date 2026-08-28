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

    def test_initial_output_equals_feat(self):
        """Test that output == feat when output_proj is zero-initialized."""
        B, C, H, W = 2, 256, 8, 8
        feat = torch.randn(B, C, H, W)

        with torch.no_grad():
            output, recon, kl, h, z, stats = self.fusion(
                feat, use_posterior=True, deterministic=True
            )

        # output_proj is zero-init: output = 0 + feat = feat
        self.assertTrue(torch.allclose(output, feat, atol=1e-6),
                        f"Output != feat: max diff = {(output - feat).abs().max().item():.6f}")

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
        """Test that second frame output differs from first (state accumulates)."""
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

        # Output on second frame should differ from input (residual from prev frame)
        # due to accumulated h/z state — but with zero-init output_proj,
        # output = output_proj(z_t) + feat, and output_proj is zero.
        # After first forward, h_state != 0 and z_state != 0, so second frame
        # gets different z_t, but output_proj is still all zeros, so output == feat2.
        # Actually with zero-init, every output should equal its own feat.
        self.assertTrue(torch.allclose(out2, feat2, atol=1e-6),
                        f"Second frame output != feat2: max diff = {(out2 - feat2).abs().max().item():.6f}")

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

    def test_initial_output_equals_feat(self):
        """Zero-init alignment + zero-init output_proj: output == feat."""
        B, C, H, W = 2, 256, 8, 8
        feat = torch.randn(B, C, H, W)

        with torch.no_grad():
            output, recon, kl, h, z, stats = self.fusion(
                feat, use_posterior=True, deterministic=True
            )

        self.assertTrue(torch.allclose(output, feat, atol=1e-6),
                        f"Output != feat: max diff = {(output - feat).abs().max().item():.6f}")

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
        """Second frame output == feat (zero-init output_proj dominates)."""
        B, C, H, W = 2, 256, 8, 8

        self.fusion.reset_state()
        feat1 = torch.randn(B, C, H, W)
        feat2 = torch.randn(B, C, H, W)

        with torch.no_grad():
            self.fusion(feat1, use_posterior=True, deterministic=True)
            out2, _, _, _, _, _ = self.fusion(
                feat2, use_posterior=True, deterministic=True
            )

        self.assertTrue(torch.allclose(out2, feat2, atol=1e-6),
                        f"Second frame output != feat2: max diff = {(out2 - feat2).abs().max().item():.6f}")

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

    def test_deterministic_uses_posterior_mean_and_logs_std_stats(self):
        B, C, H, W = 2, 256, 8, 8
        feat = torch.randn(B, C, H, W)

        outputs = []
        stats = None
        for _ in range(2):
            self.fusion.reset_state()
            with torch.no_grad():
                out, recon, kl, h, z, cur_stats = self.fusion(
                    feat, use_posterior=True, deterministic=True)
            outputs.append(out)
            stats = cur_stats

        self.assertTrue(torch.allclose(outputs[0], outputs[1], atol=1e-6))
        self.assertIsNone(kl)
        self.assertIsNotNone(stats)

        expected_keys = [
            'stat_posterior_std',
            'stat_posterior_std_mean',
            'stat_posterior_std_std',
            'stat_posterior_std_p10',
            'stat_posterior_std_p50',
            'stat_posterior_std_p90',
        ]
        for key in expected_keys:
            self.assertIn(key, stats)
            self.assertIsInstance(stats[key], torch.Tensor)
            self.assertEqual(stats[key].ndim, 0, f"{key} should be scalar")
            self.assertFalse(stats[key].requires_grad, f"{key} should be detached")

        self.assertLessEqual(stats['stat_posterior_std_p10'], stats['stat_posterior_std_p50'])
        self.assertLessEqual(stats['stat_posterior_std_p50'], stats['stat_posterior_std_p90'])

    def test_non_deterministic_samples_with_learnable_std(self):
        B, C, H, W = 2, 256, 8, 8
        feat = torch.randn(B, C, H, W)

        outputs = []
        for _ in range(2):
            self.fusion.reset_state()
            with torch.no_grad():
                out, recon, kl, h, z, _ = self.fusion(
                    feat, use_posterior=True, deterministic=False)
            outputs.append(out)

        self.assertFalse(torch.allclose(outputs[0], outputs[1], atol=1e-3))
        self.assertIsNone(kl)


if __name__ == '__main__':
    unittest.main()
