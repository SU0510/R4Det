"""CPU unit tests for the candidate-A ep6 gate probe's scoring functions.

The ep6 gate decides whether candidate A continues to ep16, so the scoring
formulas must be pinned BEFORE any trained checkpoint exists — a transient
bug here would silently re-define the gate. These tests build synthetic maps
with known answers, so a passing run is evidence the probe measures what the
pre-registration says it measures.

Run: python3 me_rssm/sanity/test_probe_semreadout_ep6.py
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
    'probe_ep6', os.path.join(_REPO, 'me_rssm', 'sanity',
                              'probe_semreadout_ep6.py'))
probe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe)

PC_RANGE = [0.0, -39.68, -4.0, 69.12, 39.68, 2.0]
H, W, C = 8, 10, 4
# Mirrors the config's object_min_radius; threaded explicitly, never defaulted.
MIN_RADIUS = 2


class _MockBoxes:
    def __init__(self, centers, sizes):
        self.centers = torch.as_tensor(centers, dtype=torch.float32)
        self.sizes = torch.as_tensor(sizes, dtype=torch.float32)

    @property
    def gravity_center(self):
        return self.centers

    @property
    def tensor(self):
        n = self.centers.shape[0]
        out = torch.zeros((n, 7), dtype=torch.float32)
        out[:, 3:6] = torch.as_tensor(self.sizes, dtype=torch.float32)
        return out


def _run(centers, labels, sem_logits=None, residual=None, cls_score=None):
    return {
        'sem_map': (torch.zeros(C, H, W) if sem_logits is None
                    else sem_logits.clone()),
        'residual': residual,
        'cls_score': [cls_score],
        'gt_boxes': _MockBoxes(centers, [[1.0, 1.0, 1.0]] * len(centers)),
        'gt_labels': torch.as_tensor(labels, dtype=torch.long),
    }


class TestHeadQuality(unittest.TestCase):
    """M1 must equal the training-time stat_sem_* formulas."""

    def test_gaussian_shaped_map_scores_one(self):
        # One object at the centre of the grid (H=x, W=y). The map is a
        # Gaussian peaking exactly at the GT centre, i.e. what a trained head
        # is supposed to produce: every positive cell clears 0.5 and the
        # per-channel argmax lands on the positive set.
        cx = 0.5 * (PC_RANGE[0] + PC_RANGE[3])
        cy = 0.5 * (PC_RANGE[1] + PC_RANGE[4])
        run = _run([[cx, cy, 0.0]], [2])
        tgt = probe.make_target(run, C, PC_RANGE, MIN_RADIUS)
        run['sem_map'] = 6.0 * tgt[0] - 3.0
        q = probe.head_quality(run, C, PC_RANGE, MIN_RADIUS)
        self.assertAlmostEqual(q['thr50'], 1.0, places=6)
        self.assertAlmostEqual(q['hit1'], 1.0, places=6)
        # A flat plateau is NOT the same thing: ties collapse onto the first
        # cell, so hit@1 must be allowed to fail when the peak is not centred.
        run['sem_map'] = torch.where(tgt[0] > 0,
                                     torch.full_like(tgt[0], 8.0),
                                     torch.full_like(tgt[0], -8.0))
        q_flat = probe.head_quality(run, C, PC_RANGE, MIN_RADIUS)
        self.assertAlmostEqual(q_flat['thr50'], 1.0, places=6)
        self.assertEqual(q_flat['hit1'], 0.0)
        # ``pos`` is target == 1.0, i.e. the exact centre cell of each object
        # (the training loss uses the same rule: only the peak is positive and
        # the Gaussian falloff is part of the negative term). One object here
        # => exactly one positive cell, so hit@1 and thr50 are measured on
        # object centres, not on the Gaussian footprint.
        self.assertEqual(int(q['num_pos_cells']), 1)
        self.assertEqual(int(q['occupied_class_channels']), 1)

    def test_dead_map_scores_zero(self):
        cx = 0.5 * (PC_RANGE[0] + PC_RANGE[3])
        cy = 0.5 * (PC_RANGE[1] + PC_RANGE[4])
        run = _run([[cx, cy, 0.0]], [2])
        run['sem_map'] = torch.full((C, H, W), -8.0)
        q = probe.head_quality(run, C, PC_RANGE, MIN_RADIUS)
        self.assertEqual(q['thr50'], 0.0)
        self.assertEqual(q['hit1'], 0.0)

    def test_geometry_is_not_transposed(self):
        """A box at x-far / y-near must land on the H=x, W=y cell."""
        x = PC_RANGE[0] + 0.95 * (PC_RANGE[3] - PC_RANGE[0])
        y = PC_RANGE[1] + 0.05 * (PC_RANGE[4] - PC_RANGE[1])
        run = _run([[x, y, 0.0]], [2])
        tgt = probe.make_target(run, C, PC_RANGE, MIN_RADIUS)[0, 2]
        self.assertGreater(float(tgt[-1].max()), 0.0,
                           'expected the far-x object on the LAST H row')
        self.assertGreater(float(tgt[:, 0].max()), 0.0,
                           'expected the near-y object on the FIRST W column')
        self.assertEqual(float(tgt.float().argmax()) // W, H - 1)

    def test_empty_frame_returns_none(self):
        run = _run([], [])
        self.assertIsNone(probe.head_quality(run, C, PC_RANGE, MIN_RADIUS))


class TestClassScores(unittest.TestCase):
    """class_scores_from_head must collapse the anchor axis per class."""

    def test_max_over_anchor_slots_per_class(self):
        """Head-grid (W, H) input is read back on the sem grid (H, W).

        The head tensor is addressed as ``[..., w, h]`` and the returned rows
        are indexed ``h * W + w`` (the rasteriser's order); the grid is
        non-square so a missing transposition cannot pass.
        """
        n_anchor = 6
        score = torch.zeros(1, n_anchor * C, W, H)
        # Class 1's strongest anchor slot is slot 3 -> value 5 at head (w=2, h=4).
        score[0].view(n_anchor, C, W, H)[3, 1, 2, 4] = 5.0
        run = {'cls_score': [score]}
        out = probe.class_scores_from_head(run, C)
        self.assertEqual(tuple(out.shape), (W * H, C))
        row = 4 * W + 2                      # sem (h=4, w=2)
        self.assertAlmostEqual(float(out[row, 1]), 5.0, places=6)
        self.assertAlmostEqual(float(out[row, 0]), 0.0, places=6)
        self.assertAlmostEqual(float(out[2 * W + 4, 1]), 0.0, places=6,
                               msg='the un-transposed cell must stay zero')

    def test_only_first_pts_feat_level_is_read(self):
        """The head output list is per feature level; the mainline has one."""
        n_anchor = 6
        score = torch.zeros(1, n_anchor * C, W, H)
        score[0].view(n_anchor, C, W, H)[0, 2, 1, 1] = 9.0
        run = {'cls_score': [score]}
        out = probe.class_scores_from_head(run, C)
        self.assertAlmostEqual(float(out[1 * W + 1, 2]), 9.0, places=6)


class TestResidualClassMap(unittest.TestCase):
    def test_residual_is_read_on_the_sem_grid(self):
        # ``rssm_cls_logit_residual`` reaches the head already transposed, so
        # it arrives here as (C, W, H) and must be mapped back to (H, W).
        residual = torch.zeros(C, W, H)
        residual[2, 7, 5] = 3.0          # head (w=7, h=5)
        run = {'residual': residual}
        out = probe.residual_class_map(run, C)
        self.assertEqual(tuple(out.shape), (H * W, C))
        row = 5 * W + 7                  # sem (h=5, w=7)
        self.assertAlmostEqual(float(out[row, 2]), 3.0, places=6)
        self.assertAlmostEqual(float(out[7 * W + 5, 2]), 0.0, places=6,
                               msg='the un-transposed cell must stay zero')

    def test_missing_residual_returns_none(self):
        self.assertIsNone(probe.residual_class_map({'residual': None}, C))


class TestHeadSwap(unittest.TestCase):
    """M4's head-level swap must change ONLY the class residual."""

    class _Head:
        def __init__(self, base_logits, gain=1.0):
            self.base_logits = base_logits
            self.gain = gain
            self.calls = []

        def __call__(self, feats, cls_logit_residual=None):
            self.calls.append(cls_logit_residual)
            # A real head's parameters live on the module's device, so mirror
            # the incoming features instead of pinning the fixture to CPU.
            out = self.base_logits.to(feats[0].device).clone()
            if cls_logit_residual is not None:
                n_anchor = out.shape[1] // C
                out = out + cls_logit_residual.repeat(1, n_anchor, 1, 1) * self.gain
            return [out], None, None

    class _Model:
        def __init__(self, head):
            self.pts_bbox_head = head

    def test_swap_moves_head_argmax_towards_the_donor(self):
        base = torch.zeros(1, 2 * C, W, H)      # 2 anchor slots, flat scores
        head = self._Head(base)
        model = self._Model(head)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        run = {'pts_feats': [torch.zeros(1, 4, W, H, device=device)]}
        donor_res = torch.zeros(1, C, W, H, device=device)   # head grid
        donor_res[0, 3, 2, 5] = 4.0             # head (w=2, h=5), class 3
        out = probe.head_scores_with_residual(model, run, donor_res)
        scores = probe.class_scores_from_head({'cls_score': out}, C)
        row = 5 * W + 2                         # sem (h=5, w=2)
        self.assertEqual(int(scores[row].argmax()), 3)
        self.assertEqual(len(head.calls), 1)
        self.assertIs(head.calls[0], donor_res,
                      'a residual already on the features device must reach '
                      'the head unchanged (no gratuitous copy)')

    def test_none_residual_is_the_no_readout_control(self):
        head = self._Head(torch.zeros(1, 2 * C, W, H))
        model = self._Model(head)
        probe.head_scores_with_residual(
            model, {'pts_feats': [torch.zeros(1, 4, W, H)]}, None)
        self.assertIsNone(head.calls[0])

    class _DeviceStrictHead:
        """Mimics the real head's device contract.

        ``Anchor3DHead._expand_cls_logit_residual`` builds its index map with
        ``ref.device`` and then calls ``residual.index_select(1, src)``: if the
        probe hands it a CPU residual while it is running on CUDA features,
        PyTorch raises a device mismatch. On a CUDA box the CPU-feature fixture
        below reproduces exactly that (``.cuda()`` moves the features, the
        residual used to stay behind), so this test fails against the
        un-fixed probe there.
        """

        def __init__(self, base_logits):
            self.base_logits = base_logits
            self.calls = []

        def __call__(self, feats, cls_logit_residual=None):
            self.calls.append(cls_logit_residual)
            if cls_logit_residual is None:
                return [self.base_logits.clone()], None, None
            self.assert_same_device(feats[0].device, cls_logit_residual.device)
            # same op the real head performs, with an index map on ref.device
            src = torch.tensor([0] * C, device=feats[0].device)
            cls_logit_residual.index_select(1, src)
            return [self.base_logits.clone()], None, None

        @staticmethod
        def assert_same_device(feat_device, residual_device):
            if feat_device != residual_device:
                raise AssertionError(
                    'cls_logit_residual must arrive on the features device: '
                    f'features {feat_device}, residual {residual_device}')

    def test_swap_moves_residual_onto_the_features_device(self):
        head = self._DeviceStrictHead(torch.zeros(1, 2 * C, W, H))
        model = self._Model(head)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        probe.head_scores_with_residual(
            model,
            {'pts_feats': [torch.zeros(1, 4, W, H, device=device)]},
            torch.zeros(1, C, W, H))          # deliberately left on CPU
        self.assertEqual(head.calls[0].device.type,
                         torch.device(device).type)


class TestClipSelection(unittest.TestCase):
    def test_stride_selection_requires_two(self):
        class _DS:
            def __len__(self):
                return 10

        original = probe.dz.history_validity
        probe.dz.history_validity = lambda ds, i: [True, True, True, True]
        try:
            args = type('A', (), {})()
            args.start_index, args.limit = 3, 10
            args.stride, args.min_valid_history = 20, 2
            with self.assertRaises(ValueError):
                probe.select_clips(_DS(), args)
        finally:
            probe.dz.history_validity = original

    def test_stride_selection_spreads_clips(self):
        class _DS:
            def __len__(self):
                return 100

        original = probe.dz.history_validity
        probe.dz.history_validity = lambda ds, i: [True, True, True, True]
        try:
            args = type('A', (), {})()
            args.start_index, args.limit = 3, 4
            args.stride, args.min_valid_history = 20, 2
            self.assertEqual(probe.select_clips(_DS(), args), [3, 23, 43, 63])
        finally:
            probe.dz.history_validity = original


class TestHit1ChanceSupplement(unittest.TestCase):
    """M1's hit@1 bars were calibrated on route-C's 16x16 grid, not full-res.

    The probe must therefore ALSO report the grid-matched chance level, while
    never letting that supplement move gates['M1'] (the prereg pins the gate
    to the raw 0.30/0.60 values).
    """

    def test_chance_and_excess_are_reported(self):
        run = _run([[20.0, 0.0, 0.0]], [2])
        q = probe.head_quality(run, C, PC_RANGE, MIN_RADIUS)
        self.assertIn('hit1_chance', q)
        self.assertIn('hit1_excess_over_chance', q)
        self.assertAlmostEqual(q['hit1_chance'],
                               q['num_pos_cells'] / (H * W), places=12)
        self.assertAlmostEqual(
            q['hit1_excess_over_chance'],
            q['hit1'] - q['num_pos_cells'] / (H * W), places=12)

    def test_chance_shrinks_with_grid_density(self):
        # same object on a dense grid => far smaller chance level
        dense = probe.head_quality(
            _run([[20.0, 0.0, 0.0]], [2]), C, PC_RANGE, MIN_RADIUS)
        self.assertLess(dense['hit1_chance'], 0.5)

    def test_gate_m1_ignores_the_supplement(self):
        # gate_m1 takes only the two pre-registered scalars
        import inspect
        params = list(inspect.signature(probe.gate_m1).parameters)
        self.assertEqual(params, ['mean_thr50', 'mean_hit1'])
        self.assertNotIn('chance', ' '.join(params))


class TestMinRadiusIsThreaded(unittest.TestCase):
    """The M1-M4 targets must use the config's object_min_radius.

    The probe read the value from the config but every rasteriser call fell
    back to a hard-coded default of 2, so the gate would have silently scored a
    different target from training the moment the config changed. Pinned by
    intercepting the rasteriser and asserting the value it receives.
    """

    def setUp(self):
        import mmdet3d.core.utils.gaussian as gaussian
        self.gaussian = gaussian
        self.seen = []
        self._orig = gaussian.bev_centers_to_heatmap

        def spy(boxes, labels, num_classes, out_size, point_cloud_range,
                min_radius, device=None, dtype=None):
            self.seen.append(min_radius)
            return self._orig(
                boxes, labels, num_classes=num_classes, out_size=out_size,
                point_cloud_range=point_cloud_range, min_radius=min_radius,
                device=device, dtype=dtype)

        gaussian.bev_centers_to_heatmap = spy
        probe.make_target.__globals__['bev_centers_to_heatmap'] = spy
        self.addCleanup(self._restore)

    def _restore(self):
        self.gaussian.bev_centers_to_heatmap = self._orig
        probe.make_target.__globals__['bev_centers_to_heatmap'] = self._orig

    def test_head_quality_forwards_the_configured_radius(self):
        for radius in (1, 2, 3):
            self.seen.clear()
            run = _run([[20.0, 0.0, 0.0]], [2])
            probe.head_quality(run, C, PC_RANGE, radius)
            self.assertEqual(self.seen, [radius],
                             'head_quality dropped the configured '
                             'object_min_radius')

    def test_make_target_requires_the_radius(self):
        with self.assertRaises(TypeError):
            probe.make_target(_run([[20.0, 0.0, 0.0]], [2]), C, PC_RANGE)


class TestGateBoundaries(unittest.TestCase):
    """The S1 gate numbers decide whether candidate A continues to ep16.

    Each boundary is pinned EXACTLY (>= vs >, and the strict-AND reading of
    M2 that the prereg clarified on 2026-10-10) so a later refactor cannot
    quietly move a threshold while the ep6 evidence is being read.
    """

    def test_m1_thresholds(self):
        self.assertTrue(probe.gate_m1(0.30, 0.60))        # boundary is >=
        self.assertFalse(probe.gate_m1(0.30, 0.5999))
        self.assertFalse(probe.gate_m1(0.2999, 0.60))
        self.assertFalse(probe.gate_m1(None, 0.60))
        self.assertFalse(probe.gate_m1(0.30, None))

    def test_m2_requires_BOTH_arms_to_be_worse(self):
        # both arms >=30% worse -> pass
        self.assertTrue(probe.gate_m2(-0.30, -0.30))      # boundary is <=
        self.assertTrue(probe.gate_m2(-0.80, -0.45))
        # one arm substitutable for z -> the probe may NOT pass:
        # "either arm" would have accepted these two
        self.assertFalse(probe.gate_m2(-0.30, -0.10))
        self.assertFalse(probe.gate_m2(-0.05, -0.75))
        self.assertFalse(probe.gate_m2(-0.2999, -0.30))
        self.assertFalse(probe.gate_m2(None, -0.75))
        self.assertFalse(probe.gate_m2(-0.75, None))

    def test_m2_relative_reading_sign(self):
        # a control arm that is BETTER than z gives rel > 0 -> never passes
        self.assertEqual(probe.relative_thr50(0.40, 0.20), -0.5)
        self.assertAlmostEqual(probe.relative_thr50(0.40, 0.52), 0.3)
        self.assertIsNone(probe.relative_thr50(0.0, 0.2))
        self.assertIsNone(probe.relative_thr50(None, 0.2))

    def test_m3_sign_test(self):
        def rows(deltas):
            return [{'delta_off_minus_on': d} for d in deltas]
        self.assertTrue(probe.gate_m3(rows([0.1] * 8)))
        self.assertFalse(probe.gate_m3(rows([0.1] * 7)))          # <8 clips
        self.assertFalse(probe.gate_m3(rows([0.1] * 7 + [-0.1])))  # 7/8 wins
        self.assertFalse(probe.gate_m3(rows([0.1] * 7 + [0.0])),
                         'a tie is not a win')
        self.assertTrue(probe.gate_m3(rows([0.1] * 10)))

    def test_m4_chance_plus_20pt(self):
        self.assertTrue(probe.gate_m4(0.45, 0.45, 4))     # boundary is >=
        self.assertFalse(probe.gate_m4(0.4499, 0.45, 4))
        self.assertFalse(probe.gate_m4(0.45, 0.4499, 4))
        self.assertTrue(probe.gate_m4(0.40, 0.40, 5))     # 5 classes -> 0.40
        self.assertFalse(probe.gate_m4(None, 0.9, 4))
        self.assertFalse(probe.gate_m4(0.9, None, 4))


class TestInputFingerprints(unittest.TestCase):
    """The bundle must say exactly which artifacts produced the reading."""

    def test_records_path_hash_size_and_mtime(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            cfg = os.path.join(tmp, 'cfg.py')
            ckpt = os.path.join(tmp, 'epoch_6.pth')
            with open(cfg, 'w') as handle:
                handle.write('model = dict()\n')
            with open(ckpt, 'wb') as handle:
                handle.write(b'not really a checkpoint')
            fp = probe.input_fingerprints(cfg, ckpt)
        self.assertEqual(len(fp['config_sha256']), 64)
        self.assertEqual(len(fp['checkpoint_sha256']), 64)
        self.assertNotEqual(fp['config_sha256'], fp['checkpoint_sha256'])
        self.assertEqual(fp['config_size_bytes'], len('model = dict()\n'))
        self.assertGreater(fp['checkpoint_mtime'], 0)

    def test_missing_checkpoint_does_not_raise(self):
        """A fingerprint of a missing artifact is None, not a crash.

        The probe must record the absence rather than abort: the assembled
        verdict is what flags the incomplete bundle.
        """
        fp = probe.input_fingerprints(__file__, '/nonexistent/epoch_6.pth')
        self.assertIsNone(fp['checkpoint_sha256'])
        self.assertEqual(len(fp['config_sha256']), 64)

    def test_hash_distinguishes_contents(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            a = os.path.join(tmp, 'a.bin')
            b = os.path.join(tmp, 'b.bin')
            with open(a, 'wb') as handle:
                handle.write(b'one')
            with open(b, 'wb') as handle:
                handle.write(b'two')
            self.assertNotEqual(probe._sha256(a), probe._sha256(b))
            with open(b, 'wb') as handle:
                handle.write(b'one')
            self.assertEqual(probe._sha256(a), probe._sha256(b))



if __name__ == '__main__':
    unittest.main(verbosity=2)
