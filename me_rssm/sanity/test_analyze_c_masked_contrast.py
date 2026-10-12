#!/usr/bin/env python3
"""CPU tests for the paired masked-vs-baseline contrast analysis."""

import importlib.util
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load():
    spec = importlib.util.spec_from_file_location(
        'analyze_c_masked_contrast', os.path.join(_HERE,
                                                  'analyze_c_masked_contrast.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tool = _load()


def record(image_idx, gt_index, cls='Car', found=True, group='high',
           range_bin='le50', scene=0):
    return {'image_idx': image_idx, 'gt_index': gt_index, 'class': cls,
            'found': found, 'coverage_group': group, 'zero_scope': None,
            'amp_group': 'mid', 'range_bin': range_bin, 'range_m': 20.0,
            'scene': scene}


class JoinRecordsTest(unittest.TestCase):
    def test_pairs_by_key(self):
        base = [record(1, 0, found=True), record(1, 1, found=False)]
        mask = [record(1, 0, found=False), record(1, 1, found=False)]
        rows = tool.join_records(base, mask)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]['found_baseline'], True)
        self.assertEqual(rows[0]['found_masked'], False)

    def test_same_object_different_class_is_a_different_key(self):
        base = [record(1, 0, cls='Car')]
        mask = [record(1, 0, cls='Pedestrian')]
        with self.assertRaises(AssertionError):
            tool.join_records(base, mask)

    def test_missing_gt_in_one_arm_raises(self):
        base = [record(1, 0), record(1, 1)]
        mask = [record(1, 0)]
        with self.assertRaises(AssertionError):
            tool.join_records(base, mask)

    def test_frozen_label_drift_raises(self):
        base = [record(1, 0, group='high')]
        mask = [record(1, 0, group='low')]
        with self.assertRaises(AssertionError):
            tool.join_records(base, mask)

    def test_is_covered_flag_follows_the_reference_group(self):
        rows = tool.join_records([record(1, 0, group='zero')], 
                                 [record(1, 0, group='zero')])
        self.assertFalse(rows[0]['is_covered'])


class PairedTableTest(unittest.TestCase):
    def test_delta_and_flips(self):
        rows = [
            {'found_baseline': True, 'found_masked': False},   # lost
            {'found_baseline': True, 'found_masked': True},    # kept
            {'found_baseline': False, 'found_masked': True},   # gained
            {'found_baseline': False, 'found_masked': False},  # never
        ]
        table = tool.paired_table(rows)
        self.assertAlmostEqual(table['delta_mask_points'], 0.0)  # 2 -> 2 found
        self.assertEqual(table['flips']['tp_to_fn (lost by masking)'], 1)
        self.assertEqual(table['flips']['fn_to_tp (gained by masking)'], 1)

    def test_pure_loss_is_minus_25_points(self):
        rows = [{'found_baseline': True, 'found_masked': False}] * 1 + \
               [{'found_baseline': True, 'found_masked': True}] * 3
        table = tool.paired_table(rows)
        self.assertAlmostEqual(table['delta_mask_points'], -25.0)

    def test_empty_population_is_none_not_zero(self):
        table = tool.paired_table([])
        self.assertIsNone(table['delta_mask_points'])
        self.assertEqual(table['n_gt'], 0)


class BootstrapTest(unittest.TestCase):
    def _rows(self, n_scenes, per_scene, lost_fraction):
        rows = []
        for scene in range(n_scenes):
            lost = int(per_scene * lost_fraction)
            for k in range(per_scene):
                rows.append({
                    'scene': scene, 'found_baseline': True,
                    'found_masked': k >= lost,
                    'coverage_group': 'high', 'range_bin': 'le50',
                    'class': 'Car', 'image_idx': scene, 'gt_index': k,
                })
        return rows

    def test_consistent_loss_gives_negative_ci(self):
        rows = self._rows(11, 100, 0.5)
        ci = tool.scene_clustered_bootstrap(rows, iters=200)
        self.assertTrue(ci['excludes_zero'])
        self.assertEqual(ci['direction'], 'negative')
        self.assertLess(ci['ci95'][1], 0.0)

    def test_mixed_scenes_span_zero(self):
        rows = self._rows(10, 100, 0.5) + self._rows(1, 100, 0.0)
        for row in rows[-100:]:
            row['scene'] = 99
            row['found_masked'] = True
        ci = tool.scene_clustered_bootstrap(rows, iters=200)
        self.assertIn(ci['direction'], ('spans zero', 'negative'))

    def test_no_population_is_handled(self):
        ci = tool.scene_clustered_bootstrap([], iters=10)
        self.assertEqual(ci['n_resamples'], 0)
        self.assertFalse(ci['excludes_zero'])


class VerdictTest(unittest.TestCase):
    def setUp(self):
        self.pooled = {'delta_mask_points': -10.0}
        self.bins = {
            'le50': {'n_gt': 100, 'delta_mask_points': -12.0},
            '50to70': {'n_gt': 50, 'delta_mask_points': -8.0},
            'gt70': {'n_gt': 10, 'delta_mask_points': -1.0},
        }
        self.ci = {'excludes_zero': True, 'ci95': [-15.0, -4.0]}

    def test_supported_when_every_clause_holds(self):
        verdict = tool.verdict(self.pooled, self.bins, self.ci, True)
        self.assertTrue(verdict['causal_supported'])
        self.assertEqual(verdict['range_bins_qualifying'], ['le50', '50to70'])

    def test_identity_failure_blocks_everything(self):
        verdict = tool.verdict(self.pooled, self.bins, self.ci, False)
        self.assertFalse(verdict['causal_supported'])
        self.assertFalse(verdict['clauses']['identity_control_bit_identical'])

    def test_positive_delta_is_not_support(self):
        verdict = tool.verdict({'delta_mask_points': 1.0}, self.bins,
                               {'excludes_zero': True, 'ci95': [0.5, 2.0]}, True)
        self.assertFalse(verdict['causal_supported'])

    def test_single_qualifying_bin_is_not_support(self):
        bins = dict(self.bins)
        bins['50to70'] = {'n_gt': 50, 'delta_mask_points': -1.0}
        verdict = tool.verdict(self.pooled, bins, self.ci, True)
        self.assertFalse(verdict['causal_supported'])

    def test_low_n_bin_does_not_qualify(self):
        bins = dict(self.bins)
        bins['50to70'] = {'n_gt': 5, 'delta_mask_points': -30.0}
        verdict = tool.verdict(self.pooled, bins, self.ci, True)
        self.assertEqual(verdict['range_bins_qualifying'], ['le50'])

    def test_ci_spanning_zero_blocks(self):
        verdict = tool.verdict(self.pooled, self.bins,
                               {'excludes_zero': False,
                                'ci95': [-15.0, 3.0]}, True)
        self.assertFalse(verdict['causal_supported'])

    def test_ci_positive_direction_does_not_count_even_if_excludes_zero(self):
        verdict = tool.verdict(self.pooled, self.bins,
                               {'excludes_zero': True, 'ci95': [1.0, 9.0]}, True)
        self.assertFalse(verdict['clauses']['ci95_upper_bound_below_zero'])
        self.assertFalse(verdict['causal_supported'])


class AnalyseTest(unittest.TestCase):
    def test_population_split_and_tables(self):
        base, mask = [], []
        for i in range(160):
            group = 'zero' if i % 4 == 0 else 'high'
            bin_name = 'le50' if i % 2 == 0 else '50to70'
            base.append(record(i, 0, group=group, found=True, scene=i % 11,
                               range_bin=bin_name))
            mask.append(record(i, 0, group=group, range_bin=bin_name,
                               found=group == 'zero', scene=i % 11))
        rows = tool.join_records(base, mask)
        result = tool.analyse(rows, identity_identical=True)
        self.assertEqual(result['population_sizes']['masked_covered'], 120)
        self.assertEqual(result['population_sizes']['zero'], 40)
        self.assertAlmostEqual(result['pooled']['delta_mask_points'], -100.0)
        self.assertEqual(result['verdict']['range_bins_qualifying'],
                         ['le50', '50to70'])
        self.assertTrue(result['verdict']['causal_supported'])

    def test_single_range_bin_is_not_enough_for_the_frozen_rule(self):
        base, mask = [], []
        for i in range(80):
            group = 'zero' if i % 4 == 0 else 'high'
            base.append(record(i, 0, group=group, found=True, scene=i % 11))
            mask.append(record(i, 0, group=group, found=group == 'zero',
                               scene=i % 11))
        rows = tool.join_records(base, mask)
        result = tool.analyse(rows, identity_identical=True)
        self.assertAlmostEqual(result['pooled']['delta_mask_points'], -100.0)
        self.assertFalse(result['verdict']['causal_supported'])
        self.assertFalse(
            result['verdict']['clauses']['at_least_two_qualifying_range_bins'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
