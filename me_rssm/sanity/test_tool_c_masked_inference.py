"""CPU tests for the controlled-masking tool (no GPU, no model, no data files).

The intervention is only meaningful if it removes exactly what it claims to
remove and nothing else, so the tests pin:

  * which objects get masked (FROZEN coverage group != zero, from the labels the
    enrichment probe computes -- not a re-derived or tuned selection);
  * that masking drops the points inside those boxes and keeps every other
    point, including points inside zero-group boxes and points outside all
    boxes;
  * that the loader wrapper leaves the pipeline untouched in ``identity`` mode
    (the control run has to be a no-op);
  * that the identity comparison is strict (bitwise, not "close enough").

Run: python3 me_rssm/sanity/test_tool_c_masked_inference.py
"""

import importlib.util
import os
import sys
import unittest

import numpy as np
import torch

_REPO = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..'))
sys.path.insert(0, _REPO)

from mmdet3d.core.bbox import LiDARInstance3DBoxes  # noqa: E402
from mmdet3d.core.points import LiDARPoints  # noqa: E402


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_REPO, rel))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tool = _load('tool_c_masked_inference_mod',
             'me_rssm/sanity/tool_c_masked_inference.py')


def _info(image_idx, names, locs, dims=(2.0, 2.0, 4.0)):
    n = len(names)
    ident = np.eye(4, dtype=np.float64)
    return {
        'image': {'image_idx': image_idx, 'image_shape': np.array([375, 1242])},
        'point_cloud': {'velodyne_path': f'x/{image_idx:06d}.bin'},
        'calib': {'R0_rect': ident.copy(), 'Tr_velo_to_cam': ident.copy(),
                  'P2': ident.copy()},
        'annos': {
            'name': names,
            'num_points_in_gt': [0] * n,
            'location': [list(l) for l in locs],
            'dimensions': [list(dims) for _ in range(n)],
            'rotation_y': [0.0] * n,
            'bbox': [[0.0, 0.0, 20.0, 40.0] for _ in range(n)],
            'occluded': [0] * n,
            'difficulty': [1] * n,
            'truncated': [0.0] * n,
        },
    }


class FrozenSelectionTest(unittest.TestCase):
    """The target population comes from the frozen labels, verbatim."""

    def test_only_non_zero_groups_are_masked(self):
        info = _info(1000, ['Car', 'Pedestrian', 'Cyclist'],
                     [(10.0, 0.0, 40.0), (20.0, 0.0, 40.0), (30.0, 0.0, 40.0)])
        rows = {
            (1000, 0): {'coverage_group': 'zero'},
            (1000, 1): {'coverage_group': 'low'},
            (1000, 2): {'coverage_group': 'high'},
        }
        original = tool.enrich.build_frozen_labels
        tool.enrich.build_frozen_labels = lambda *a, **k: {'by_object': rows}
        try:
            plan = tool.coverage_plan([info], {}, {})
        finally:
            tool.enrich.build_frozen_labels = original
        entry = plan[1000]
        self.assertEqual(entry['groups'], ['zero', 'low', 'high'])
        self.assertEqual(entry['masked'], [False, True, True])
        self.assertEqual(entry['boxes'].shape[0], 3)

    def test_missing_label_raises(self):
        info = _info(1000, ['Car'], [(10.0, 0.0, 40.0)])
        original = tool.enrich.build_frozen_labels
        tool.enrich.build_frozen_labels = lambda *a, **k: {'by_object': {}}
        try:
            with self.assertRaises(AssertionError):
                tool.coverage_plan([info], {}, {})
        finally:
            tool.enrich.build_frozen_labels = original


class MaskRowsTest(unittest.TestCase):
    def setUp(self):
        # One box at (10, 0) 4x2x2 (h,w,l -> l=2, w=2, h=4) in lidar frame.
        self.boxes = np.array([[10.0, 0.0, 40.0, 2.0, 2.0, 4.0, 0.0]])
        # NB points_in_rbbox excludes the bottom plane: inside points sit
        # strictly above the box centre height (the §77 test convention).
        self.points = np.array([
            [10.0, 0.0, 40.5],   # inside
            [10.5, 0.5, 41.0],   # inside
            [30.0, 0.0, 40.5],   # outside (another object)
            [10.0, 0.0, 30.0],   # outside (below the box)
        ])

    def test_points_inside_a_selected_box_are_dropped(self):
        keep = tool.mask_rows(self.points, self.boxes, [True])
        self.assertEqual(keep.tolist(), [False, False, True, True])

    def test_unselected_boxes_do_not_remove_anything(self):
        keep = tool.mask_rows(self.points, self.boxes, [False])
        self.assertTrue(keep.all())

    def test_empty_cloud_is_handled(self):
        keep = tool.mask_rows(np.zeros((0, 3)), self.boxes, [True])
        self.assertEqual(keep.shape, (0,))

    def test_two_boxes_are_masked_independently(self):
        boxes = np.concatenate([self.boxes,
                                [[30.0, 0.0, 40.0, 2.0, 2.0, 4.0, 0.0]]])
        keep = tool.mask_rows(self.points, boxes, [True, True])
        self.assertEqual(keep.tolist(), [False, False, False, True])


class LoaderWrapperTest(unittest.TestCase):
    class LoadPointsFromFile(object):
        """Stands in for the pipeline transform (matched by class name)."""

        def __init__(self, points):
            self.points = points
            self.calls = 0

        def __call__(self, results):
            self.calls += 1
            results['points'] = LiDARPoints(
                self.points.copy(), points_dim=self.points.shape[-1])
            return results

    def _points(self):
        arr = np.array([[10.0, 0.0, 40.5, -1.0, 10.0],
                        [30.0, 0.0, 40.5, -1.0, 10.0]], dtype=np.float32)
        return arr

    def test_wrapper_is_found_and_installed(self):
        class Dataset(object):
            pass

        class Pipeline(object):
            def __init__(self, transforms):
                self.transforms = transforms

        loader = self.LoadPointsFromFile(self._points())
        dataset = Dataset()
        dataset.pipeline = Pipeline([loader])
        plan = {1000: {'boxes': np.array([[10.0, 0.0, 40.0, 2.0, 2.0, 4.0, 0.0]]),
                       'groups': ['low'], 'names': ['Car'], 'masked': [True]}}
        wrapper = tool.install_wrapper(dataset, plan, 'mask_covered')
        results = wrapper({'sample_idx': 1000})
        self.assertEqual(results['points'].tensor.shape[0], 1)
        self.assertAlmostEqual(float(results['points'].tensor[0, 0]), 30.0)

    def test_identity_mode_keeps_every_point(self):
        class Dataset(object):
            pass

        class Pipeline(object):
            def __init__(self, transforms):
                self.transforms = transforms

        loader = self.LoadPointsFromFile(self._points())
        dataset = Dataset()
        dataset.pipeline = Pipeline([loader])
        plan = {1000: {'boxes': np.array([[10.0, 0.0, 40.0, 2.0, 2.0, 4.0, 0.0]]),
                       'groups': ['low'], 'names': ['Car'], 'masked': [True]}}
        wrapper = tool.install_wrapper(dataset, plan, 'identity')
        results = wrapper({'sample_idx': 1000})
        self.assertEqual(results['points'].tensor.shape[0], 2)
        self.assertEqual(wrapper.n_masked_frames, 0)


class CompareDumpsTest(unittest.TestCase):
    def _dump(self, tensor, score):
        boxes = LiDARInstance3DBoxes(torch.tensor(tensor, dtype=torch.float32))
        return [{'pts_bbox': {'boxes_3d': boxes,
                              'scores_3d': torch.tensor(score, dtype=torch.float32),
                              'labels_3d': torch.tensor([0] * len(score),
                                                        dtype=torch.long)}}]

    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp()
        self.box = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0]])

    def test_identical_dumps_pass(self):
        import mmcv
        left = os.path.join(self.tmp, 'l.pkl')
        right = os.path.join(self.tmp, 'r.pkl')
        mmcv.dump(self._dump(self.box, [0.9]), left)
        mmcv.dump(self._dump(self.box, [0.9]), right)
        report = tool.compare_dumps(left, right)
        self.assertTrue(report['identical'])
        self.assertEqual(report['max_abs_diff'], 0.0)

    def test_a_changed_score_is_caught(self):
        import mmcv
        left = os.path.join(self.tmp, 'l2.pkl')
        right = os.path.join(self.tmp, 'r2.pkl')
        mmcv.dump(self._dump(self.box, [0.9]), left)
        mmcv.dump(self._dump(self.box, [0.8]), right)
        report = tool.compare_dumps(left, right)
        self.assertFalse(report['identical'])
        self.assertEqual(report['frames_differs'], [0])

    def test_a_missing_box_is_caught(self):
        import mmcv
        left = os.path.join(self.tmp, 'l3.pkl')
        right = os.path.join(self.tmp, 'r3.pkl')
        mmcv.dump(self._dump(self.box, [0.9]), left)
        mmcv.dump([{'pts_bbox': {
            'boxes_3d': LiDARInstance3DBoxes(torch.zeros((0, 7))),
            'scores_3d': torch.zeros(0), 'labels_3d': torch.zeros(0,
                                                             dtype=torch.long)}}],
            right)
        report = tool.compare_dumps(left, right)
        self.assertFalse(report['identical'])

    def test_length_mismatch_without_limit_raises(self):
        import mmcv
        left = os.path.join(self.tmp, 'l4.pkl')
        right = os.path.join(self.tmp, 'r4.pkl')
        mmcv.dump(self._dump(self.box, [0.9]), left)
        mmcv.dump(self._dump(self.box, [0.9]) * 2, right)
        with self.assertRaises(AssertionError):
            tool.compare_dumps(left, right)

    def test_short_left_prefix_compares_under_limit(self):
        """A smoke run must be comparable against the full baseline dump."""
        import mmcv
        left = os.path.join(self.tmp, 'l5.pkl')
        right = os.path.join(self.tmp, 'r5.pkl')
        mmcv.dump(self._dump(self.box, [0.9]), left)          # 1 frame
        mmcv.dump(self._dump(self.box, [0.9]) * 3, right)     # 3 frames
        report = tool.compare_dumps(left, right, limit=1)
        self.assertTrue(report['identical'])
        self.assertEqual(report['n_frames'], 1)

    def test_prefix_comparison_still_catches_a_difference(self):
        import mmcv
        left = os.path.join(self.tmp, 'l6.pkl')
        right = os.path.join(self.tmp, 'r6.pkl')
        mmcv.dump(self._dump(self.box, [0.9]), left)
        mmcv.dump(self._dump(self.box, [0.7]) * 3, right)
        report = tool.compare_dumps(left, right, limit=1)
        self.assertFalse(report['identical'])
        self.assertEqual(report['frames_differs'], [0])

    def test_left_longer_than_right_under_limit_raises(self):
        import mmcv
        left = os.path.join(self.tmp, 'l7.pkl')
        right = os.path.join(self.tmp, 'r7.pkl')
        mmcv.dump(self._dump(self.box, [0.9]) * 3, left)
        mmcv.dump(self._dump(self.box, [0.9]), right)
        with self.assertRaises(AssertionError):
            tool.compare_dumps(left, right, limit=1)

    def test_sha256_matches_hashlib_on_a_real_file(self):
        path = os.path.join(self.tmp, 'sha.bin')
        with open(path, 'wb') as handle:
            handle.write(b'mask-identity-control')
        expected = __import__('hashlib').sha256(
            b'mask-identity-control').hexdigest()
        self.assertEqual(tool._sha256(path), expected)


if __name__ == '__main__':
    unittest.main(verbosity=2)
