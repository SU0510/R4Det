"""CPU unit tests for the standard-family z-utilization probe.

Two things in ``probe_z_utilization_standard.py`` differ from the gated probe
and from ``tools/diagnose_z_utilization.py``, and both are silent-failure
hazards, so they are pinned here before the ep6 checkpoint exists:

1. family dispatch — the standard probe must refuse a fusion without a
   ``posterior_mu`` head with a clear TypeError (the gated probe's inverse
   failure, verified 2026-10-10, was an AssertionError deep inside a 3-GPU
   battery);
2. the deployed class residual must reach the head, and must be absent (not
   zero-filled) when the fusion does not produce one — otherwise the probe
   scores a detector that is never run and understates z's effect.

Run: python3 me_rssm/sanity/test_probe_z_utilization_standard.py
"""

import importlib.util
import os
import sys
import unittest

import torch

_REPO = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, _REPO)

_spec = importlib.util.spec_from_file_location(
    'probe_std', os.path.join(_REPO, 'me_rssm', 'sanity',
                              'probe_z_utilization_standard.py'))
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)
dz = probe.dz

C, H, W = 4, 8, 10


class _Fusion:
    """Minimal standard-family fusion: forward + posterior_mu head."""

    def __init__(self, with_posterior_mu=True):
        if with_posterior_mu:
            self.posterior_mu = torch.nn.Conv2d(C, C, 1)
        self.z_state = None
        self.reset_calls = 0

    def reset_state(self):
        self.reset_calls += 1

    def reset_for_samples(self, mask):
        pass


class _Model:
    def __init__(self, residual, with_posterior_mu=True):
        self.seq_len = 2
        self.temporal_fusion = _Fusion(with_posterior_mu)
        self._residual = residual
        self.head_calls = []
        self.pts_bbox_head = self._head
        self.extract_feat = self._extract_feat

    def _head(self, feats, cls_logit_residual=None):
        self.head_calls.append(cls_logit_residual)
        return [torch.zeros(1, 2 * C, W, H)], None, None

    def _extract_feat(self, points, img, metas, is_valid_mask=None,
                      feat_or_dict=0, curr_frame_supervision=False):
        if feat_or_dict == 0:
            self.temporal_fusion.z_state = torch.ones(1, C, H, W)
            return None
        return {
            'pts_feats': [torch.zeros(1, C, W, H)],
            'rssm_cls_logit_residual': self._residual,
            'rssm_stats': {},
        }


def _batch():
    return {
        'points': [None, None],
        'img': torch.zeros(1, 2, 3, H, W),
        'img_metas': [[{'k': 0}], [{'k': 0}]],
        'frame_valid': [torch.tensor([True]), torch.tensor([True])],
    }


class TestFamilyDispatch(unittest.TestCase):
    def test_refuses_a_fusion_without_posterior_mu(self):
        model = _Model(residual=None, with_posterior_mu=False)
        with self.assertRaises(TypeError) as ctx:
            probe.run_sequence_standard(model, _batch())
        self.assertIn('posterior_mu', str(ctx.exception))

    def test_accepts_a_standard_fusion(self):
        model = _Model(residual=None)
        out = probe.run_sequence_standard(model, _batch())
        self.assertIn('head_vector', out)
        self.assertEqual(model.temporal_fusion.reset_calls, 1)


class TestDeployedResidualReachesHead(unittest.TestCase):
    def test_residual_is_forwarded(self):
        residual = torch.full((1, C, W, H), 0.5)
        model = _Model(residual=residual)
        probe.run_sequence_standard(model, _batch())
        self.assertEqual(len(model.head_calls), 1)
        self.assertIsNotNone(
            model.head_calls[0],
            'candidate A deploys the class residual; dropping it would score '
            'a detector that is never run')
        self.assertTrue(torch.equal(model.head_calls[0], residual))

    def test_missing_residual_stays_none(self):
        model = _Model(residual=None)
        probe.run_sequence_standard(model, _batch())
        self.assertIsNone(
            model.head_calls[0],
            'legacy configs must keep the head call bit-identical')


if __name__ == '__main__':
    unittest.main()
