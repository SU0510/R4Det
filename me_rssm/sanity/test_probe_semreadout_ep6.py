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
        tgt = probe.make_target(run, C, PC_RANGE)
        run['sem_map'] = 6.0 * tgt[0] - 3.0
        q = probe.head_quality(run, C, PC_RANGE)
        self.assertAlmostEqual(q['thr50'], 1.0, places=6)
        self.assertAlmostEqual(q['hit1'], 1.0, places=6)
        # A flat plateau is NOT the same thing: ties collapse onto the first
        # cell, so hit@1 must be allowed to fail when the peak is not centred.
        run['sem_map'] = torch.where(tgt[0] > 0,
                                     torch.full_like(tgt[0], 8.0),
                                     torch.full_like(tgt[0], -8.0))
        q_flat = probe.head_quality(run, C, PC_RANGE)
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
        q = probe.head_quality(run, C, PC_RANGE)
        self.assertEqual(q['thr50'], 0.0)
        self.assertEqual(q['hit1'], 0.0)

    def test_geometry_is_not_transposed(self):
        """A box at x-far / y-near must land on the H=x, W=y cell."""
        x = PC_RANGE[0] + 0.95 * (PC_RANGE[3] - PC_RANGE[0])
        y = PC_RANGE[1] + 0.05 * (PC_RANGE[4] - PC_RANGE[1])
        run = _run([[x, y, 0.0]], [2])
        tgt = probe.make_target(run, C, PC_RANGE)[0, 2]
        self.assertGreater(float(tgt[-1].max()), 0.0,
                           'expected the far-x object on the LAST H row')
        self.assertGreater(float(tgt[:, 0].max()), 0.0,
                           'expected the near-y object on the FIRST W column')
        self.assertEqual(float(tgt.float().argmax()) // W, H - 1)

    def test_empty_frame_returns_none(self):
        run = _run([], [])
        self.assertIsNone(probe.head_quality(run, C, PC_RANGE))


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
            out = self.base_logits.clone()
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
        run = {'pts_feats': [torch.zeros(1, 4, W, H)]}
        donor_res = torch.zeros(1, C, W, H)     # head grid
        donor_res[0, 3, 2, 5] = 4.0             # head (w=2, h=5), class 3
        out = probe.head_scores_with_residual(model, run, donor_res)
        scores = probe.class_scores_from_head({'cls_score': out}, C)
        row = 5 * W + 2                         # sem (h=5, w=2)
        self.assertEqual(int(scores[row].argmax()), 3)
        self.assertEqual(len(head.calls), 1)
        self.assertIs(head.calls[0], donor_res,
                      'the donor residual must reach the head unchanged')

    def test_none_residual_is_the_no_readout_control(self):
        head = self._Head(torch.zeros(1, 2 * C, W, H))
        model = self._Model(head)
        probe.head_scores_with_residual(
            model, {'pts_feats': [torch.zeros(1, 4, W, H)]}, None)
        self.assertIsNone(head.calls[0])


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


if __name__ == '__main__':
    unittest.main(verbosity=2)
