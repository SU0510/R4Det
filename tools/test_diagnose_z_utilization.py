"""Unit tests for the z_t counterfactual intervention logic.

These exercise the intervention contract only (hooking, replacement, tuple
consistency, shuffle source selection); they deliberately do not build a
dataset or checkpoint, so they run without the CUDA-only mmdet3d ops.

Run: python tools/test_diagnose_z_utilization.py
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn

from tools.diagnose_z_utilization import (
    ZIntervention,
    resolve_shuffle_sources,
)


class _FakeFusion(nn.Module):
    """Minimal stand-in exposing the forward contract the tool hooks into.

    Mirrors the real modules the intervention relies on: ``posterior_mu``
    producing the current-frame latent upstream of every consumer, plus two
    downstream branches (output_proj and decoder) and the recurrent
    ``z_state`` handed to the next frame.
    """

    def __init__(self, latent_dim=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.posterior_mu = nn.Conv2d(2 * latent_dim, latent_dim, 3, padding=1)
        self.output_proj = nn.Conv2d(latent_dim, 1, 1)
        self.decoder = nn.Conv2d(2 * latent_dim, 1, 1)
        self.h_state = None
        self.z_state = None
        self.calls = 0

    def reset_state(self):
        self.h_state = None
        self.z_state = None
        self.calls = 0

    def forward(self, feat, velocity=None, use_posterior=True,
                deterministic=True, detach_state=True):
        self.calls += 1
        b, c, h, w = feat.shape
        h_t = torch.zeros(b, self.latent_dim, h, w)
        if self.z_state is None or self.z_state.shape[0] != b:
            self.z_state = torch.zeros(b, self.latent_dim, h, w)
        e_t = feat.mean(dim=1, keepdim=True).expand(
            b, self.latent_dim, h, w)
        mu_q = self.posterior_mu(torch.cat([h_t, e_t], dim=1))
        mu_p = torch.full_like(mu_q, 0.25)
        z_t = mu_q if use_posterior else mu_p
        latent_loss = z_t.square().mean()
        output = feat + self.output_proj(z_t)
        reconstruction = self.decoder(torch.cat([h_t, z_t], dim=1))
        self.z_state = z_t.detach()
        return output, reconstruction, latent_loss, h_t, z_t, latent_loss


class TestZIntervention(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)

    def _feats(self, seq_len=3, batch=2, size=6):
        return [torch.randn(batch, 1, size, size) for _ in range(seq_len)]

    def test_zero_z_reaches_out4_and_all_downstream_branches(self):
        fusion = _FakeFusion()
        feats = self._feats()
        fusion.reset_state()
        with torch.no_grad():
            normal = fusion(feats[-1], deterministic=True)

        fusion.reset_state()
        with ZIntervention(fusion, 'zero_z', len(feats)) as intervention:
            with torch.no_grad():
                for feat in feats[:-1]:
                    fusion(feat, deterministic=True)
                out = fusion(feats[-1], deterministic=True)

        self.assertEqual(out[4].abs().max().item(), 0.0)
        self.assertEqual(intervention.last_z.abs().max().item(), 0.0)
        # Both downstream consumers must change, which proves the hook sat
        # upstream of the branches rather than on one of them.
        self.assertFalse(torch.allclose(out[0], normal[0], atol=1e-6))
        self.assertFalse(torch.allclose(out[1], normal[1], atol=1e-6))

    def test_shuffle_z_installs_replacement_and_reaches_out4(self):
        fusion = _FakeFusion()
        feats = self._feats()
        replacement = torch.full((2, fusion.latent_dim, 6, 6), -3.0)

        fusion.reset_state()
        with ZIntervention(fusion, 'replace_z', len(feats),
                           replacement_z=replacement):
            with torch.no_grad():
                for feat in feats[:-1]:
                    fusion(feat, deterministic=True)
                out = fusion(feats[-1], deterministic=True)

        self.assertTrue(torch.equal(out[4].detach(), replacement))
        self.assertTrue(
            torch.equal(fusion.z_state.detach(), replacement))

    def test_prior_only_uses_prior_without_hooking(self):
        fusion = _FakeFusion()
        feats = self._feats()

        fusion.reset_state()
        with ZIntervention(fusion, 'prior_only', len(feats)):
            with torch.no_grad():
                for feat in feats[:-1]:
                    fusion(feat, deterministic=True)
                out = fusion(feats[-1], deterministic=True)

        self.assertTrue(torch.allclose(out[4], torch.full_like(out[4], 0.25)))

    def test_forward_is_restored_after_context(self):
        fusion = _FakeFusion()
        with ZIntervention(fusion, 'zero_z', 3):
            # The override shadows the class method during the context.
            self.assertIn('forward', fusion.__dict__)
        # ...and is removed afterwards so normal dispatch resumes.
        self.assertNotIn('forward', fusion.__dict__)
        self.assertEqual(len(fusion._forward_hooks), 0)

    def test_normal_replay_is_bit_identical(self):
        fusion = _FakeFusion()
        feats = self._feats()

        fusion.reset_state()
        with torch.no_grad():
            first = fusion(feats[-1], deterministic=True)
        fusion.reset_state()
        with torch.no_grad():
            second = fusion(feats[-1], deterministic=True)

        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertTrue(torch.equal(first[4], second[4]))

    def test_hook_works_for_lowdim_and_standard_module_shapes(self):
        """Both architectures expose posterior_mu; the probe must fit either."""
        try:
            from mmdet3d.models.fusion_layers.rssm_fusion import (
                LowDimFutureConsistentLatentFusion,
                MotionAlignedRSSMFusion,
            )
        except AssertionError as exc:
            # mmdet3d.ops asserts on missing CUDA at import time; the real
            # module check is covered by the GPU-enabled suite.
            self.skipTest(f'mmdet3d ops need CUDA: {exc}')
        for fusion in (
            LowDimFutureConsistentLatentFusion(
                in_channels=8, latent_dim=4, hidden_dim=8,
                latent_size=(2, 2), recon_loss_weight=0.1),
            MotionAlignedRSSMFusion(
                in_channels=8, out_channels=8, latent_dim=4, hidden_dim=8,
                align_deform_groups=1),
        ):
            fusion.eval()
            fusion.reset_state()
            feat = torch.randn(1, 8, 6, 6)
            with torch.no_grad():
                fusion(feat, deterministic=True)
            fusion.reset_state()
            with ZIntervention(fusion, 'zero_z', 2) as intervention:
                with torch.no_grad():
                    fusion(feat, deterministic=True)
                    out = fusion(feat, deterministic=True)
            self.assertEqual(out[4].abs().max().item(), 0.0)
            self.assertEqual(intervention.last_z.abs().max().item(), 0.0)
            self.assertEqual(len(fusion._forward_hooks), 0)


class TestShuffleSources(unittest.TestCase):

    def test_source_is_never_self(self):
        for offset in (0, 1, -1, 2, -2, 3):
            sources = resolve_shuffle_sources([3, 4, 5, 6], offset)
            for index, source in sources.items():
                self.assertNotEqual(index, source)

    def test_offset_zero_uses_next_sample(self):
        sources = resolve_shuffle_sources([3, 4, 5, 6], 0)
        self.assertEqual(sources[3], 4)
        self.assertEqual(sources[6], 3)

    def test_far_window_offsets(self):
        sources = resolve_shuffle_sources([100, 101, 102, 103], 2)
        self.assertEqual(sources[100], 102)
        self.assertEqual(sources[103], 101)

    def test_single_sample_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_shuffle_sources([7], 0)


if __name__ == '__main__':
    unittest.main()
