"""CPU tests for the sem-head rank-vs-calibration decomposition probe.

The probe exists because 71.9.19 located the train/val gap in the split but
left the MECHANISM unidentified: M1's ``thr50`` and ``hit@1`` are both
threshold readings, so "the head cannot rank objects" and "the head ranks them
but is under-confident" produce the same FAIL and imply opposite next
experiments. These tests pin the properties that make the distinction real:

  * the AUC must be tie-corrected (a saturated map is common for this head, and
    naive rank assignment would hand a constant map an arbitrary AUC);
  * the AUC must come with a GT-shuffle control, because a map that is merely
    elevated in the busy part of the BEV would score well against ANY target;
  * the top-k recall must use the clip's own positive count as k, so it is the
    head's own peak budget and not an arbitrary cut;
  * the report must carry the fixed-threshold readings alongside, so the
    scale-free reading can never be presented in place of the gate reading.

Run: python3 me_rssm/sanity/test_probe_semA_rank.py
"""

import importlib.util
import inspect
import os
import sys
import unittest

_REPO = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, _REPO)

import torch  # noqa: E402


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_REPO, rel))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


rank = _load('probe_rank', 'me_rssm/sanity/probe_semA_rank.py')


def _read(rel):
    with open(os.path.join(_REPO, rel), encoding='utf-8') as handle:
        return handle.read()


class TestAucIsTieCorrected(unittest.TestCase):
    def test_constant_scores_give_exactly_one_half(self):
        """The head produces many exactly-equal background probabilities; a
        constant map carries no information and must not be credited with any.
        """
        scores = torch.full((100,), 0.25)
        positive = torch.zeros(100, dtype=torch.bool)
        positive[:7] = True
        self.assertAlmostEqual(rank.average_rank_auc(scores, positive), 0.5)

    def test_perfect_separation_gives_one(self):
        scores = torch.cat([torch.ones(5), torch.zeros(95)])
        positive = torch.zeros(100, dtype=torch.bool)
        positive[:5] = True
        self.assertAlmostEqual(rank.average_rank_auc(scores, positive), 1.0)

    def test_inverted_separation_gives_zero(self):
        scores = torch.cat([torch.zeros(5), torch.ones(95)])
        positive = torch.zeros(100, dtype=torch.bool)
        positive[:5] = True
        self.assertAlmostEqual(rank.average_rank_auc(scores, positive), 0.0)

    def test_returns_none_when_a_side_is_empty(self):
        scores = torch.zeros(4)
        self.assertIsNone(rank.average_rank_auc(scores, torch.zeros(4, dtype=torch.bool)))
        self.assertIsNone(rank.average_rank_auc(scores, torch.ones(4, dtype=torch.bool)))


class TestTopKUsesTheClipsOwnBudget(unittest.TestCase):
    def test_recall_at_top_k_equals_one_when_positives_are_the_peaks(self):
        """k is the clip's positive count, so a map whose k highest cells are
        exactly the k GT cells must read 1.0 -- not a value that depends on an
        arbitrary constant threshold."""
        prob = torch.zeros(1, 1, 4, 4)
        target = torch.zeros(1, 1, 4, 4)
        peaks = [(0, 0), (1, 1), (3, 2)]
        for (y, x) in peaks:
            prob[0, 0, y, x] = 4.0
            target[0, 0, y, x] = 1.0
        out = rank.rank_readings(prob, target, 1)
        self.assertAlmostEqual(out['recall_at_top_k_mean'], 1.0)

    def test_k_is_the_clip_positive_count(self):
        src = inspect.getsource(rank.rank_readings)
        self.assertIn('torch.topk(pk, min(n_pos, pk.numel()))', src)


class TestShuffleControlIsPresent(unittest.TestCase):
    def test_gt_shuffle_uses_a_different_clip(self):
        src = inspect.getsource(rank.main)
        self.assertIn("raw[(pos + 1) % len(raw)]", src)
        self.assertIn("donor['index'] == row['index']", src)

    def test_report_separates_object_specificity_from_high_auc(self):
        src = _read('me_rssm/sanity/probe_semA_rank.py')
        self.assertIn("'auc_is_specific_to_this_clips_objects'", src)
        self.assertIn('gt_shuffle_auc', src)


class TestFixedThresholdReadingIsKept(unittest.TestCase):
    def test_report_keeps_thr50_and_hit1_next_to_the_scale_free_ones(self):
        src = inspect.getsource(rank.main)
        self.assertIn("'fixed_thr50_mean'", src)
        self.assertIn("'fixed_hit1_mean'", src)

    def test_fixed_reading_comes_from_the_gate_scoring_function(self):
        """The fixed readings must be the gate's own numbers, not a
        re-implementation that could drift from it."""
        src = inspect.getsource(rank.main)
        self.assertIn('ep6.head_quality', src)


class TestProbeIsReadOnly(unittest.TestCase):
    def test_no_optimiser_or_checkpoint_write(self):
        src = _read('me_rssm/sanity/probe_semA_rank.py')
        for forbidden in ('torch.save', 'optimizer', 'backward()',
                          'load_state_dict'):
            self.assertNotIn(forbidden, src)

    def test_split_switch_uses_the_same_pipeline(self):
        """Only the info file may change; the pipeline must stay the val
        pipeline or the split reading is confounded with augmentation."""
        src = inspect.getsource(rank.main)
        self.assertIn('cfg.data.val.ann_file = args.split_ann_file', src)
        self.assertNotIn('pipeline =', src)


if __name__ == '__main__':
    unittest.main()
