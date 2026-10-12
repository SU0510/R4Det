"""CPU tests for the candidate-C coverage-grouping probe.

The probe decides the grouping that every later C measurement will be
conditioned on, so its two failure modes are the ones worth pinning:

  * a cut that was chosen after seeing the val answers (the prereg rule is
    "thresholds from TRAIN only"), and
  * a "missing-radar" group that silently absorbs ignored annotations or
    ``DontCare``, which would inflate it with objects that were never scored.

Run: python3 me_rssm/sanity/test_probe_c_coverage_groups.py
"""

import importlib.util
import os
import sys
import unittest

_REPO = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, _REPO)


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_REPO, rel))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


groups = _load('probe_c_groups', 'me_rssm/sanity/probe_c_coverage_groups.py')


def _info(idx, coverages, names=None, dontcare_at=None, ignored_at=None,
          bbox_h=50.0, loc=(0.0, 0.0, 40.0), dims=(1.5, 1.5, 4.0)):
    n = len(coverages)
    names = names or ['Car'] * n
    counts = list(coverages)
    for k in (ignored_at or []):
        counts[k] = -1
    if dontcare_at:
        for k in dontcare_at:
            names[k] = groups.DONTCARE
    return {
        'image': {'image_idx': idx, 'image_path': f'x/{idx:06d}.png'},
        'point_cloud': {'velodyne_path': f'x/{idx:06d}.bin'},
        'annos': {
            'name': names,
            'num_points_in_gt': counts,
            'location': [list(loc) for _ in range(n)],
            'dimensions': [list(dims) for _ in range(n)],
            'bbox': [[0.0, 0.0, 20.0, bbox_h] for _ in range(n)],
            'occluded': [0] * n,
            'difficulty': [1] * n,
            'truncated': [0.0] * n,
        },
    }


class TestCutsAreTrainOnly(unittest.TestCase):
    def test_val_never_moves_a_boundary(self):
        train = groups.object_rows([_info(0, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])],
                                   'train', {0: 0})
        val_a = groups.object_rows([_info(10, [0, 1, 2])], 'val', {10: 1})
        val_b = groups.object_rows(
            [_info(10, [0, 500, 900000])], 'val', {10: 1})
        cuts_a = groups.derive_cuts(train)
        self.assertEqual(cuts_a, groups.derive_cuts(train))
        # Same train rows => same cuts, no matter how extreme val becomes.
        groups.analyse(train, val_a, ['Car'])
        groups.analyse(train, val_b, ['Car'])
        self.assertEqual(cuts_a, groups.derive_cuts(train))

    def test_zero_is_qualitative_and_never_a_quantile(self):
        # Even when a third of the train objects return nothing, the 0/1
        # boundary must stay where it is: "absent from the tensor" is a state,
        # not a density.
        train = groups.object_rows(
            [_info(0, [0] * 10 + [1, 2, 3] * 10)], 'train', {0: 0})
        cuts = groups.derive_cuts(train)
        self.assertEqual(cuts['zero'], [0, 0])
        self.assertEqual(cuts['low'][0], 1)
        self.assertEqual(groups.assign_group(0, cuts), groups.ZERO_GROUP)
        self.assertEqual(groups.assign_group(1, cuts), 'low')

    def test_the_high_tail_has_no_ceiling(self):
        train = groups.object_rows([_info(0, [1, 2, 3, 4, 5, 6])], 'train',
                                   {0: 0})
        cuts = groups.derive_cuts(train)
        self.assertIsNone(cuts['high'][1])
        self.assertEqual(groups.assign_group(10 ** 9, cuts), 'high')


class TestGroupPopulationIsHonest(unittest.TestCase):
    def test_ignored_and_dontcare_never_enter_a_group(self):
        rows = groups.object_rows(
            [_info(0, [0, 5, 0], dontcare_at=[1], ignored_at=[2])],
            'train', {0: 0})
        self.assertEqual([r['coverage'] for r in rows], [0])
        self.assertEqual(len(rows), 1)

    def test_scenes_come_from_contiguity(self):
        runs = groups.scene_runs([0, 1, 2, 500, 501, 900])
        self.assertEqual([runs[i] for i in (0, 1, 2)], [0, 0, 0])
        self.assertEqual([runs[i] for i in (500, 501)], [1, 1])
        self.assertEqual(runs[900], 2)

    def test_a_single_frame_has_one_scene(self):
        self.assertEqual(len(set(groups.scene_runs([7]).values())), 1)


class TestConfoundsAreReported(unittest.TestCase):
    def test_camera_precondition_is_measured_for_the_zero_group(self):
        rows = groups.object_rows(
            [_info(0, [0], bbox_h=30.0), _info(1, [12], bbox_h=90.0)],
            'val', {0: 0, 1: 0})
        out = groups.analyse(rows, rows, ['Car'])
        feas = out['camera_feasibility_in_zero_group']['val']
        # 30 px is above KITTI's 25 px easy bar => the camera is still
        # informative for the object radar missed, which is C's precondition.
        self.assertAlmostEqual(feas['zero_bbox_h_px_ge_25_share'], 1.0)
        self.assertEqual(feas['zero_bbox_h_px_median'], 30.0)
        self.assertEqual(feas['positive_bbox_h_px_median'], 90.0)

    def test_range_and_size_are_kept_next_to_coverage(self):
        rows = groups.object_rows([_info(0, [0], loc=(30.0, 0.0, 40.0))],
                                  'train', {0: 0})
        self.assertAlmostEqual(rows[0]['range_m'], 50.0)
        self.assertAlmostEqual(rows[0]['volume_m3'], 1.5 * 1.5 * 4.0)
        summary = groups.summarise(rows)
        self.assertEqual(summary['range_m_median'], 50.0)
        self.assertEqual(summary['n_frames'], 1)

    def test_zero_share_by_class_is_reported(self):
        rows = groups.object_rows(
            [_info(0, [0, 0, 3], names=['Pedestrian'] * 3)], 'val', {0: 0})
        out = groups.analyse([], rows, ['Pedestrian'])
        got = out['zero_share_by_class']['val']['Pedestrian']
        self.assertEqual(got['n_objects'], 3)
        self.assertAlmostEqual(got['zero_share'], 2.0 / 3.0)


if __name__ == '__main__':
    unittest.main()
