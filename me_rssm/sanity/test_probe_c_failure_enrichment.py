"""CPU tests for the candidate-C failure-enrichment probe.

This probe is the one that DECIDES whether candidate C has a立项 basis, so the
things worth pinning are the ways it could decide for the wrong reason:

  * an attribution that is not actually the repository's own (the TP/FN flags
    must agree with ``compute_statistics_jit``'s ``compute_fp=True`` counts on
    every frame, including the odd branches: ignored-GT matches, other-class
    detections that swallow a GT, and detections under the score operating
    point);
  * a per-class overlap threshold that is not the one the headline metric uses
    (uniform 0.5 would silently change which classes count as found);
  * FPs leaking into a GT group (the whole point of the ``fp_unmatched`` bucket);
  * a verdict rule that fires on one lucky range bin, or on a delta below the
    frozen 3.0-point threshold;
  * a GT-label misalignment that attaches coverage/motion groups to the wrong
    object -- the probe must fail loudly instead.

Run: python3 me_rssm/sanity/test_probe_c_failure_enrichment.py
"""

import copy
import importlib.util
import os
import sys
import unittest

import numpy as np

_REPO = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, _REPO)


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_REPO, rel))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


probe = _load('probe_c_failure_enrichment_mod',
              'me_rssm/sanity/probe_c_failure_enrichment.py')


def _gt_annos(names, locs, dims=(1.6, 1.6, 4.0), difficulty=1,
              bbox=(0.0, 0.0, 20.0, 40.0)):
    n = len(names)
    return {
        'name': names,
        'bbox': np.tile(np.array(bbox, dtype=np.float64), (n, 1)),
        'alpha': np.zeros(n, dtype=np.float64),
        'location': np.array(locs, dtype=np.float64).reshape(n, 3),
        'dimensions': np.tile(np.array(dims, dtype=np.float64), (n, 1)),
        'rotation_y': np.zeros(n, dtype=np.float64),
        'difficulty': np.full(n, difficulty, dtype=np.int64),
        'occluded': np.zeros(n, dtype=np.int64),
        'truncated': np.zeros(n, dtype=np.float64),
        'num_points_in_gt': np.zeros(n, dtype=np.int64),
    }


def _dt_annos(names, scores, bbox=(0.0, 0.0, 20.0, 40.0)):
    n = len(names)
    return {
        'name': names,
        'bbox': np.tile(np.array(bbox, dtype=np.float64), (n, 1)),
        'alpha': np.zeros(n, dtype=np.float64),
        'score': np.array(scores, dtype=np.float64),
    }


def _match_all(gt_annos, dt_annos, overlaps, thresh=0.25):
    """Per-class frames for ALL four classes, as the probe's main() builds them.

    ``clean_data`` rewrites ``gt_annos[i]['difficulty']`` in place while it runs,
    so every class must get its own deep copy -- sharing one would make class N
    depend on the class the previous call mutated.
    """
    return {cls: probe.match_class(copy.deepcopy(gt_annos), dt_annos, overlaps,
                                   cls, thresh)
            for cls in probe.CLASS_ORDER}


def _flag_case(overlaps, ignored_gt, ignored_det, dt_scores, min_overlap=0.5,
               thresh=0.25):
    """Hand-built frame: (ours, theirs) tp/fp/fn for one class."""
    gt_datas = np.zeros((len(ignored_gt), 5), dtype=np.float64)
    dt_datas = np.zeros((len(ignored_det), 7), dtype=np.float64)
    dt_datas[:, -1] = np.asarray(dt_scores, dtype=np.float64)
    ours = probe.attribute_frame(
        np.asarray(overlaps, dtype=np.float64), gt_datas, dt_datas,
        list(ignored_gt), list(ignored_det), min_overlap, thresh)
    theirs = probe.compute_statistics_jit(
        np.asarray(overlaps, dtype=np.float64), gt_datas, dt_datas,
        list(ignored_gt), list(ignored_det), np.zeros((0, 4)), 2,
        min_overlap=min_overlap, thresh=thresh, compute_fp=True)
    return ours, (int(theirs[0]), int(theirs[1]), int(theirs[2]))


class AttributionMatchesRepoTest(unittest.TestCase):
    def test_plain_tp_fn_fp(self):
        cases = [
            # overlaps, ignored_gt, ignored_det, dt_scores
            ([[0.6]], [0], [0], [0.9]),           # matched -> tp
            ([[0.4]], [0], [0], [0.9]),           # below IoU -> fn (dt empty)
            ([[0.0]], [0], [0], [0.9]),           # no overlap -> fn + fp
            ([[0.6]], [-1], [0], [0.9]),          # other-class GT, free dt -> fp
        ]
        for overlaps, igt, idt, scores in cases:
            ours, theirs = _flag_case(overlaps, igt, idt, scores)
            self.assertEqual(
                (ours['tp'], ours['fp'], ours['fn']), theirs,
                msg=f'overlaps={overlaps} ignored_gt={igt} ignored_det={idt}')

    def test_score_below_operating_point_is_not_a_tp(self):
        # dt is a valid detection but sits under 0.25: it can neither match nor
        # become an FP, and the GT stays an FN.
        ours, theirs = _flag_case([[0.9]], [0], [0], [0.10])
        self.assertEqual((ours['tp'], ours['fp'], ours['fn']), (0, 0, 1))
        self.assertEqual((ours['tp'], ours['fp'], ours['fn']), theirs)

    def test_highest_overlap_wins(self):
        ours, theirs = _flag_case([[0.6], [0.8]], [0], [0, 0], [0.9, 0.9])
        self.assertEqual((ours['tp'], ours['fp'], ours['fn']), (1, 1, 0))
        self.assertEqual((ours['tp'], ours['fp'], ours['fn']), theirs)
        matched = [f for f in ours['flags'] if f['status'] == 'tp']
        self.assertEqual(matched[0]['dt'], 1)  # the 0.8-overlap detection

    def test_other_class_detection_swallows_the_gt(self):
        # A detection of another class (ignored_det == 1) that overlaps a valid
        # GT is CONSUMED, and that GT is neither tp nor fn. This is the repo's
        # behaviour, not a bug in the probe: the probe only has to reproduce it.
        ours, theirs = _flag_case([[0.9]], [0], [1], [0.9])
        self.assertEqual((ours['tp'], ours['fp'], ours['fn']), (0, 0, 0))
        self.assertEqual((ours['tp'], ours['fp'], ours['fn']), theirs)

    def test_ignored_gt_match_consumes_the_detection(self):
        # ignored_gt == 1 (this class but out of the difficulty band): the match
        # is made and the detection must not be counted as an FP.
        ours, theirs = _flag_case([[0.9]], [1], [0], [0.9])
        self.assertEqual((ours['tp'], ours['fp'], ours['fn']), (0, 0, 0))
        self.assertEqual((ours['tp'], ours['fp'], ours['fn']), theirs)
        self.assertEqual(ours['flags'][0]['status'], 'ignored_match')

    def test_ignored_class_gt_is_skipped_entirely(self):
        ours, theirs = _flag_case([[0.9]], [-1], [1], [0.9])
        self.assertEqual((ours['tp'], ours['fp'], ours['fn']), theirs)
        self.assertEqual(ours['flags'][0]['status'], 'ignored_gt')


class MatchClassUsesRepoPreparedDataTest(unittest.TestCase):
    def test_end_to_end_per_frame_counts_match_the_repo(self):
        gt_annos = [
            _gt_annos(['Car', 'Pedestrian'], [(10.0, 0.0, 1.0),
                                              (5.0, 0.0, 0.5)]),
            _gt_annos(['Car'], [(80.0, 0.0, 1.0)], difficulty=2),
        ]
        # dt0 matches the Car GT, dt1 is a Pedestrian detection (ignored while
        # scoring the Car class, so it is neither tp nor fp), dt2 is a Car that
        # overlaps nothing -> the frame's one FP.
        dt_annos = [
            _dt_annos(['Car', 'Pedestrian', 'Car'], [0.9, 0.8, 0.8]),
            _dt_annos(['Car'], [0.7]),
        ]
        overlaps = [np.array([[0.7, 0.0], [0.0, 0.6], [0.0, 0.0]]),
                    np.array([[0.9]])]
        frames = probe.match_class(copy.deepcopy(gt_annos), dt_annos,
                                   overlaps, 'Car', 0.25)
        self.assertEqual(len(frames), 2)
        self.assertEqual(
            [(f['tp'], f['fp'], f['fn']) for f in frames], [(1, 1, 0), (1, 0, 0)])
        # The Pedestrian GT is not part of the Car bookkeeping.
        self.assertTrue(all(f['status'] != 'fn' for f in frames[0]['flags']
                            if f['gt'] == 1))
        pedestrian = probe.match_class(copy.deepcopy(gt_annos), dt_annos,
                                       overlaps, 'Pedestrian', 0.25)
        self.assertEqual(
            [(f['tp'], f['fp'], f['fn']) for f in pedestrian], [(1, 0, 0), (0, 0, 0)])


class FrozenLabelPlumbingTest(unittest.TestCase):
    def _records(self):
        gt_annos = [_gt_annos(['Car', 'Car'], [(10.0, 0.0, 1.0),
                                               (60.0, 0.0, 1.0)])]
        dt_annos = [_dt_annos(['Car'], [0.9])]
        overlaps = [np.array([[0.9, 0.0]])]
        frames = _match_all(gt_annos, dt_annos, overlaps)
        labels = {
            'image_idx_of_frame': [1000],
            'scene_runs': {1000: 0},
            'by_object': {
                (1000, 0): {'coverage_group': 'zero', 'zero_scope': 'object_level',
                            'amp_group': 'no_return'},
                (1000, 1): {'coverage_group': 'high', 'zero_scope': None,
                            'amp_group': 'v_amp_high'},
            },
        }
        return probe.build_records(gt_annos, frames, labels), labels

    def test_found_flags_are_attached_to_the_frozen_labels(self):
        records, _ = self._records()
        self.assertEqual(len(records), 2)
        by_gt = {r['gt_index']: r for r in records}
        self.assertTrue(by_gt[0]['found'])
        self.assertFalse(by_gt[1]['found'])
        self.assertEqual(by_gt[0]['coverage_group'], 'zero')
        self.assertEqual(by_gt[1]['coverage_group'], 'high')
        self.assertEqual(by_gt[1]['amp_group'], 'v_amp_high')
        # 60 m is clean_data's own difficulty-1 band, so the bins must agree.
        self.assertEqual(by_gt[1]['range_bin'], '50to70')

    def test_misaligned_flag_index_fails_loudly(self):
        # This is what a flag/GT drift looks like: the Car bookkeeping points at
        # index 1, which is the Pedestrian. The probe must refuse rather than
        # attach the Pedestrian's frozen group to a Car detection.
        gt_annos = [_gt_annos(['Car', 'Pedestrian'], [(10.0, 0.0, 1.0),
                                                     (5.0, 0.0, 0.5)])]
        flags = [{'gt': 1, 'status': 'tp', 'dt': 0, 'score': 0.9, 'overlap': 0.9}]
        frames = {cls: [{'tp': 1, 'fp': 0, 'fn': 0, 'flags': list(flags)}]
                  for cls in probe.CLASS_ORDER}
        labels = {'image_idx_of_frame': [1000], 'scene_runs': {1000: 0},
                  'by_object': {(1000, 0): {'coverage_group': 'zero',
                                            'zero_scope': 'object_level',
                                            'amp_group': 'no_return'},
                                (1000, 1): {'coverage_group': 'high',
                                            'zero_scope': None,
                                            'amp_group': 'v_amp_high'}}}
        with self.assertRaises(AssertionError):
            probe.build_records(gt_annos, frames, labels)

    def test_missing_frozen_label_fails_loudly(self):
        gt_annos = [_gt_annos(['Car', 'Pedestrian'], [(10.0, 0.0, 1.0),
                                                     (5.0, 0.0, 0.5)])]
        flags = [{'gt': 0, 'status': 'tp', 'dt': 0, 'score': 0.9, 'overlap': 0.9}]
        frames = {cls: [{'tp': 1, 'fp': 0, 'fn': 0, 'flags': list(flags)}]
                  for cls in probe.CLASS_ORDER}
        labels = {'image_idx_of_frame': [1000], 'scene_runs': {1000: 0},
                  'by_object': {(1000, 1): {'coverage_group': 'high',
                                            'zero_scope': None,
                                            'amp_group': 'v_amp_high'}}}
        with self.assertRaises(AssertionError):
            probe.build_records(gt_annos, frames, labels)


class FpBucketIsolationTest(unittest.TestCase):
    def test_unmatched_detections_get_their_own_bucket(self):
        gt_annos = [_gt_annos(['Car'], [(10.0, 0.0, 1.0)])]
        dt_annos = [_dt_annos(['Car', 'Car'], [0.9, 0.8])]
        # shape is (num_dt, num_gt) -- dt1 matches, dt0 is the FP
        overlaps = [np.array([[0.0], [0.9]])]
        per_class = _match_all(gt_annos, dt_annos, overlaps)
        summary = probe.fp_summary(per_class, dt_annos)
        self.assertEqual(summary['n_fp_unmatched'], 1)
        self.assertEqual(summary['by_class']['Car'], 1)
        # Every FP stays in the bucket: no GT record may carry it.
        labels = {'image_idx_of_frame': [1000], 'scene_runs': {1000: 0},
                  'by_object': {(1000, 0): {'coverage_group': 'zero',
                                            'zero_scope': 'object_level',
                                            'amp_group': 'no_return'}}}
        records = probe.build_records(gt_annos, per_class, labels)
        self.assertEqual(len(records), 1)
        self.assertTrue(all('fp' not in r for r in records))


class RangeBinTest(unittest.TestCase):
    def test_bins_are_half_open_on_the_left_like_clean_data(self):
        self.assertEqual(probe.range_bin(12.0), 'le50')
        self.assertEqual(probe.range_bin(50.0), 'le50')
        self.assertEqual(probe.range_bin(50.0001), '50to70')
        self.assertEqual(probe.range_bin(70.0), '50to70')
        self.assertEqual(probe.range_bin(70.0001), 'gt70')

    def test_gt_range_matches_clean_data_distance_rule(self):
        anno = _gt_annos(['Car'], [(30.0, 40.0, 0.0)])
        self.assertAlmostEqual(probe.gt_range_m(anno, 0), 50.0)
        self.assertEqual(probe.range_bin(probe.gt_range_m(anno, 0)), 'le50')


class DeltaArithmeticTest(unittest.TestCase):
    def _rec(self, group, found, cls='Car', scene=0, rng='le50'):
        return {'coverage_group': group, 'found': found, 'class': cls,
                'scene': scene, 'range_bin': rng, 'zero_scope': 'object_level',
                'amp_group': 'no_return'}

    def test_delta_is_in_percentage_points(self):
        records = ([self._rec('zero', False)] * 3
                   + [self._rec('zero', True)]
                   + [self._rec('high', True)] * 3
                   + [self._rec('high', False)])
        out = probe.delta_stats(records, 'coverage_group')
        self.assertEqual(out['zero']['recall'], 0.25)
        self.assertEqual(out['covered']['recall'], 0.75)
        self.assertAlmostEqual(out['delta_points_covered_minus_zero'], 50.0)

    def test_empty_side_yields_none_not_zero(self):
        out = probe.delta_stats([self._rec('zero', False)], 'coverage_group')
        self.assertIsNone(out['delta_points_covered_minus_zero'])
        self.assertIsNone(out['covered']['recall'])

    def test_recall_of_empty_is_none(self):
        self.assertIsNone(probe.recall_of([])['recall'])


class VerdictRuleTest(unittest.TestCase):
    def _records(self, zero_found, covered_found, scenes=6, per_bin=10,
                 bins=('le50', '50to70')):
        """Synthetic records sized so the rule's clauses can be isolated."""
        records = []
        for s in range(scenes):
            for i in range(per_bin):
                for rng in bins:
                    records.append({'coverage_group': 'zero',
                                    'found': i < zero_found, 'class': 'Car',
                                    'scene': s, 'range_bin': rng})
                    records.append({'coverage_group': 'high',
                                    'found': i < covered_found, 'class': 'Car',
                                    'scene': s, 'range_bin': rng})
        return records

    def test_strong_effect_on_two_bins_passes(self):
        records = self._records(0, 6)  # 0% -> 60% in both bins
        verdict = probe.verdict_for(records, probe.by_bin_table(records))
        self.assertEqual(sorted(verdict['range_bins_qualifying']),
                         ['50to70', 'le50'])
        self.assertTrue(verdict['bootstrap_ci_excludes_zero'])
        self.assertTrue(verdict['enriched'])

    def test_single_lucky_bin_does_not_pass_even_when_pooled_is_large(self):
        records = self._records(0, 6, bins=('gt70',))
        verdict = probe.verdict_for(records, probe.by_bin_table(records))
        self.assertGreaterEqual(verdict['delta_points'], 3.0)
        self.assertEqual(verdict['range_bins_qualifying'], ['gt70'])
        self.assertFalse(verdict['enriched'])

    def test_below_threshold_delta_does_not_pass(self):
        # 0% -> 2.0 points pooled: under the frozen 3.0-point threshold.
        records = self._records(0, 2, per_bin=100)
        verdict = probe.verdict_for(records, probe.by_bin_table(records))
        self.assertAlmostEqual(verdict['delta_points'], 2.0)
        self.assertFalse(verdict['enriched'])

    def test_delta_exactly_at_the_threshold_is_not_disqualified_by_the_delta_clause(
            self):
        # 0% -> 3.0 points: the rule is >=, so this must clear the delta clause
        # (whether the study then passes still depends on the bins and the CI).
        records = self._records(0, 3, per_bin=100)
        verdict = probe.verdict_for(records, probe.by_bin_table(records))
        self.assertAlmostEqual(verdict['delta_points'], 3.0)
        self.assertEqual(sorted(verdict['range_bins_qualifying']),
                         ['50to70', 'le50'])

    def test_small_zero_group_bins_do_not_qualify(self):
        # Fewer than MIN_BIN_GT zero-group GTs per bin: no bin can qualify,
        # however large the delta is.
        records = self._records(0, 6, scenes=1, per_bin=probe.MIN_BIN_GT - 2)
        verdict = probe.verdict_for(records, probe.by_bin_table(records))
        self.assertEqual(verdict['range_bins_qualifying'], [])
        self.assertFalse(verdict['enriched'])

    def test_rule_thresholds_are_the_frozen_ones(self):
        self.assertEqual(probe.MIN_DELTA_POINTS, 3.0)
        self.assertEqual(probe.MIN_BIN_GT, 30)
        self.assertEqual(probe.BOOTSTRAP_ITERS, 2000)
        self.assertEqual(probe.RANGE_BINS,
                         ((0.0, 50.0, 'le50'), (50.0, 70.0, '50to70'),
                          (70.0, float('inf'), 'gt70')))


class ContrastReferenceTest(unittest.TestCase):
    def _records(self):
        return [{'coverage_group': 'zero', 'amp_group': 'no_return',
                 'found': False, 'class': 'Car', 'scene': 0,
                 'range_bin': 'le50'},
                {'coverage_group': 'high', 'amp_group': 'v_amp_low',
                 'found': True, 'class': 'Car', 'scene': 0,
                 'range_bin': 'le50'}]

    def test_reference_group_is_configurable(self):
        out = probe.delta_stats(self._records(), 'amp_group',
                                reference='no_return')
        self.assertEqual(out['zero']['n_gt'], 1)
        self.assertEqual(out['covered']['n_gt'], 1)
        self.assertAlmostEqual(out['delta_points_covered_minus_zero'], 100.0)

    def test_default_reference_stays_the_frozen_zero_coverage_group(self):
        # The coverage constant is the default so the primary tables and the
        # verdict rule cannot be changed by accident...
        self.assertEqual(probe.delta_stats(self._records(), 'coverage_group')[
            'zero']['n_gt'], 1)
        # ...and reusing it on the amplitude axis is exactly the trap that made
        # the first draft of 'by_amplitude_group' compare against an empty set.
        self.assertEqual(probe.delta_stats(self._records(), 'amp_group')[
            'zero']['n_gt'], 0)


class RecallGroupingTest(unittest.TestCase):
    def test_none_scope_is_named_not_dropped(self):
        # zero_scope is only defined for coverage == 0, so covered GTs carry
        # None. It has to survive into the JSON under a name (and it must not
        # silently merge with a real group).
        records = [{'coverage_group': 'zero', 'found': True, 'class': 'Car',
                    'scene': 0, 'range_bin': 'le50', 'zero_scope': 'object_level',
                    'amp_group': 'no_return'},
                   {'coverage_group': 'high', 'found': False, 'class': 'Car',
                    'scene': 0, 'range_bin': 'le50', 'zero_scope': None,
                    'amp_group': 'v_amp_high'}]
        out = probe.per_group_recall(records, 'zero_scope')
        self.assertEqual(out['object_level']['n_gt'], 1)
        self.assertEqual(out['not_applicable (coverage > 0)']['n_gt'], 1)


class BootstrapTest(unittest.TestCase):
    def test_scene_clustered_bootstrap_is_deterministic(self):
        records = []
        for s in range(4):
            for i in range(20):
                records.append({'coverage_group': 'zero', 'found': i % 3 == 0,
                                'class': 'Car', 'scene': s,
                                'range_bin': 'le50'})
                records.append({'coverage_group': 'high', 'found': i % 3 != 0,
                                'class': 'Car', 'scene': s,
                                'range_bin': 'le50'})
        first = probe.cluster_bootstrap_delta(records)
        second = probe.cluster_bootstrap_delta(records)
        self.assertEqual(first['ci95'], second['ci95'])
        self.assertEqual(first['n_resamples'], probe.BOOTSTRAP_ITERS)

    def test_resampling_units_are_scenes_not_objects(self):
        # One scene only -> every resample is the same sample -> zero-width CI.
        records = [{'coverage_group': 'zero', 'found': False, 'class': 'Car',
                    'scene': 0, 'range_bin': 'le50'},
                   {'coverage_group': 'high', 'found': True, 'class': 'Car',
                    'scene': 0, 'range_bin': 'le50'}]
        out = probe.cluster_bootstrap_delta(records)
        self.assertEqual(out['ci95'], [100.0, 100.0])
        self.assertIn('scene', out['cluster'])


class ContractConstantsTest(unittest.TestCase):
    def test_scoring_constants_match_the_frozen_contract(self):
        self.assertEqual(probe.SCORE_OPERATING_POINT, 0.25)
        self.assertEqual(tuple(probe.SENSITIVITY_POINTS), (0.10, 0.50))
        self.assertEqual(probe.DIFFICULTY_ALL, 2)
        self.assertEqual(probe.CLASS_OVERLAP, {
            'Pedestrian': 0.25, 'Cyclist': 0.25, 'Car': 0.5, 'Truck': 0.5})
        self.assertEqual(set(probe.CLASS_ORDER), set(probe.CLASS_OVERLAP))
        self.assertEqual(probe.KITTI_CLASS_INDEX,
                         {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2, 'Truck': 3})

    def test_contract_document_freezes_the_same_numbers(self):
        path = os.path.join(_REPO,
                            'docs/rssm_c_failure_enrichment_contract_20261012.md')
        with open(path, encoding='utf-8') as handle:
            text = handle.read()
        self.assertIn('0.25', text)
        self.assertIn('lap1', text)
        self.assertIn('compute_statistics_jit', text)
        self.assertIn('fp_unmatched', text)


if __name__ == '__main__':
    unittest.main(verbosity=2)
