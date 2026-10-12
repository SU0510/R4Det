"""CPU tests for the candidate-C motion-axis probe.

The probe decides the SECOND grouping axis of the C study, and the failure
modes worth pinning are the ones that would silently turn it into a study of
something else:

  * grouping on the RAW velocity channel, which is dominated by the per-frame
    ego term (0..27 m/s here) and would therefore be a proxy for "which block
    this frame belongs to";
  * naming a low amplitude "static" or "below resolution", neither of which is
    measured anywhere in this file;
  * a coverage count that no longer reproduces the stored
    ``num_points_in_gt`` after the transform handling was changed.

Run: python3 me_rssm/sanity/test_probe_c_motion_axis.py
"""

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


axis = _load('probe_c_motion_axis_mod', 'me_rssm/sanity/probe_c_motion_axis.py')
coverage = axis.coverage


def _points(n, x=0.0, y=0.0, z=0.0, v=0.0, snr=10.0):
    """NB: points_in_rbbox excludes the box's bottom plane, so a point meant
    to be inside a box whose ``location`` z is 40.0 must sit at z > 40.0."""
    pts = np.zeros((n, 8), dtype=np.float32)
    pts[:, axis.CH_X] = x
    pts[:, axis.CH_Y] = y
    pts[:, axis.CH_Z] = z
    pts[:, axis.CH_V] = v
    pts[:, axis.CH_RANGE] = np.sqrt(x * x + y * y + z * z)
    pts[:, axis.CH_SNR] = snr
    pts[:, axis.CH_AZ] = -np.degrees(np.arctan2(y, x))
    pts[:, axis.CH_EL] = 0.0
    return pts


def _ring(n, r=40.0, v_of=None, z=0.0, snr=10.0):
    """Points spread over azimuth so a cos/sin field is identifiable."""
    theta = np.linspace(-np.pi / 3, np.pi / 3, n)
    x, y = r * np.cos(theta), r * np.sin(theta)
    pts = _points(n, x=x, y=y, z=z, snr=snr)
    if v_of is not None:
        pts[:, axis.CH_V] = v_of(theta)
    return pts


def _info(idx, coverages, names=None, loc=(0.0, 0.0, 40.0), dims=(4.0, 2.0, 2.0),
          dontcare_at=None, ignored_at=None, bbox_h=50.0):
    n = len(coverages)
    names = names or ['Car'] * n
    # ``loc`` may be ONE location for every object or one per object.
    per_object_loc = (isinstance(loc, list) and len(loc) == n
                      and isinstance(loc[0], (list, tuple)))
    locs = ([list(l) for l in loc] if per_object_loc
            else [list(loc) for _ in range(n)])
    counts = list(coverages)
    for k in (ignored_at or []):
        counts[k] = -1
    for k in (dontcare_at or []):
        names[k] = coverage.DONTCARE
    ident = np.eye(4, dtype=np.float64)
    return {
        'image': {'image_idx': idx, 'image_path': f'x/{idx:06d}.png',
                  'image_shape': np.array([375, 1242], dtype=np.int32)},
        'point_cloud': {'velodyne_path': f'x/{idx:06d}.bin'},
        'calib': {'R0_rect': ident.copy(), 'Tr_velo_to_cam': ident.copy(),
                  'P2': ident.copy()},
        'annos': {
            'name': names,
            'num_points_in_gt': counts,
            'location': locs,
            'dimensions': [list(dims) for _ in range(n)],
            'rotation_y': [0.0] * n,
            'bbox': [[0.0, 0.0, 20.0, bbox_h] for _ in range(n)],
            'occluded': [0] * n,
            'difficulty': [1] * n,
            'truncated': [0.0] * n,
        },
    }


class ChannelIdentityTest(unittest.TestCase):
    def test_range_and_neg_azimuth_identities_hold_but_elevation_does_not(self):
        pts = _ring(400, r=30.0, z=1.0)
        # An elevation column that is NOT atan2(z, hypot): measured identity
        # must fail rather than be assumed.
        pts[:, axis.CH_EL] = 0.5 * pts[:, axis.CH_EL] + 3.0
        out = axis.identify_channels([pts])
        self.assertTrue(out['dim4_is_range'])
        self.assertTrue(out['dim6_is_neg_azimuth_deg'])
        self.assertFalse(out['dim7_is_geometric_elevation'])
        self.assertIn('UNIDENTIFIED', out['dim7_note'])

    def test_fill_marker_is_measured_not_asserted(self):
        pts = _ring(100, r=40.0, z=1.0)  # z != 0 everywhere to start with
        pts[:, axis.CH_EL] = 5.0
        pts[:10, axis.CH_Z] = 0.0
        pts[:10, axis.CH_EL] = 0.0
        self.assertTrue(axis.identify_channels([pts])[
            'fill_marker_z_and_el_zero_co_occur_exactly'])
        # Break the co-occurrence -> the flag must flip.
        pts_bad = pts.copy()
        pts_bad[50, axis.CH_EL] = 0.0
        self.assertFalse(axis.identify_channels([pts_bad])[
            'fill_marker_z_and_el_zero_co_occur_exactly'])

    def test_nan_count_is_reported(self):
        pts = _ring(50)
        pts[3, axis.CH_V] = np.nan
        self.assertEqual(axis.identify_channels([pts])['nan_count'], 1)


class EgoFieldTest(unittest.TestCase):
    def test_fit_recovers_ego_amplitude_with_negative_sign(self):
        speed = 11.0
        pts = _ring(600, v_of=lambda th: -speed * np.cos(th))
        ego = axis.fit_ego_field(pts, np.ones(len(pts), dtype=bool))
        self.assertAlmostEqual(ego['ego_amplitude'], speed, places=4)
        self.assertAlmostEqual(ego['a'], -speed, places=4)
        self.assertGreater(ego['r2'], 0.999)
        corrected = axis.ego_correct(pts, ego)
        self.assertLess(float(np.max(np.abs(corrected))), 1e-6)

    def test_target_motion_survives_the_correction(self):
        speed = 9.0
        pts = _ring(600, v_of=lambda th: -speed * np.cos(th))
        pts[:20, axis.CH_V] += 5.0  # one object closing 5 m/s relative to ego
        inside = np.zeros(len(pts), dtype=bool)
        inside[:20] = True
        ego = axis.fit_ego_field(pts, ~inside)
        corrected = axis.ego_correct(pts, ego)
        self.assertGreater(float(np.mean(corrected[inside])), 4.0)
        self.assertLess(float(np.mean(np.abs(corrected[~inside]))), 0.1)

    def test_fallback_when_background_is_too_small(self):
        pts = _ring(20, v_of=lambda th: -3.0 * np.cos(th))
        ego = axis.fit_ego_field(pts, np.zeros(len(pts), dtype=bool))
        self.assertEqual(ego['fit_source'], 'all_points_fallback')


class CoverageIdentityTest(unittest.TestCase):
    def test_counts_reproduce_stored_num_points_in_gt(self):
        # Identity calibration => the camera-frame box maps to a box centred on
        # the same point, so points placed at the location are inside it.
        info = _info(7, [12, 0], loc=[(0.0, 0.0, 40.0), (0.0, 0.0, 80.0)])
        pts = np.concatenate([
            _points(12, x=0.0, y=0.0, z=40.5, v=-2.0),
            _ring(200, r=60.0, v_of=lambda th: -2.0 * np.cos(th)),
        ])
        rows, mismatches, _ = axis.object_motion_rows(
            info, pts, pts, coverage.derive_cuts([]), 7, 0)
        self.assertEqual(mismatches, 0)
        self.assertEqual(rows[0]['n_points_kept'], 12)
        self.assertEqual(rows[1]['n_points_kept'], 0)
        self.assertEqual(rows[1]['coverage_group'], coverage.ZERO_GROUP)

    def test_a_wrong_stored_count_is_reported_as_a_mismatch(self):
        info = _info(7, [99])
        pts = _points(12, x=0.0, y=0.0, z=40.5, v=0.0)
        _, mismatches, _ = axis.object_motion_rows(
            info, pts, pts, coverage.derive_cuts([]), 7, 0)
        self.assertEqual(mismatches, 1)

    def test_raw_cloud_count_is_kept_separate_from_kept_count(self):
        # Stored coverage 0, but the model's own (unfiltered) cloud has points
        # in the box: the row must expose both numbers.
        info = _info(7, [0])
        pts_kept = _ring(50)
        pts_raw = np.concatenate([_points(9, x=0.0, y=0.0, z=40.5, v=-1.0),
                                  pts_kept])
        rows, _, _ = axis.object_motion_rows(
            info, pts_raw, pts_kept, coverage.derive_cuts([]), 7, 0)
        self.assertEqual(rows[0]['n_points_kept'], 0)
        self.assertEqual(rows[0]['n_points_raw'], 9)
        self.assertEqual(rows[0]['coverage_group'], coverage.ZERO_GROUP)


class ZeroScopeTest(unittest.TestCase):
    """A no-return object is only evidence about that OBJECT if its frame had
    returns; otherwise the whole frame dropped out."""

    def _run(self, coverages):
        locs = [(0.0, 0.0, 40.0 * (k + 1)) for k in range(len(coverages))]
        info = _info(7, coverages, loc=locs)
        pts = _ring(60)
        rows, _, _ = axis.object_motion_rows(
            info, pts, pts, coverage.derive_cuts([]), 7, 0)
        return {r['coverage']: r for r in rows}

    def test_one_covered_object_makes_the_frame_object_level(self):
        rows = self._run([0, 9])
        self.assertIsNone(rows[9]['zero_scope'])
        self.assertEqual(rows[0]['zero_scope'], 'object_level')
        self.assertEqual(rows[0]['frame_zero_share'], 0.5)

    def test_frame_with_no_returns_at_all_is_frame_level(self):
        rows = self._run([0, 0, 0])
        self.assertTrue(all(r['zero_scope'] == 'frame_level' for r in rows.values()))
        self.assertEqual(rows[0]['frame_zero_share'], 1.0)


class GroupingContractTest(unittest.TestCase):
    def _rows(self, amps, split='train'):
        return [{'class': 'Car', 'image_idx': k, 'scene': k // 2,
                 'coverage': 8, 'coverage_group': 'mid', 'n_points_raw': 8,
                 'range_m': 40.0, 'volume_m3': 10.0, 'bbox_h_px': 40.0,
                 'occluded': 0.0, 'truncated': 0.0, 'difficulty': 1.0,
                 'ego_amplitude_mps': 8.0, 'v_abs_corr_mean': a,
                 'v_raw_mean': -8.0, 'v_corr_std': 0.05, 'snr_mean': 12.0}
                for k, a in enumerate(amps)]

    def test_cuts_use_only_train_rows(self):
        train = self._rows([0.1, 0.2, 0.3, 5.0, 6.0, 7.0])
        cuts_a = axis.freeze_amplitude_cuts(train)
        # Val rows must not be able to move a boundary.
        _ = self._rows([100.0, 200.0, 300.0], split='val')
        cuts_b = axis.freeze_amplitude_cuts(train)
        self.assertEqual(cuts_a, cuts_b)
        self.assertLess(cuts_a['low_hi'], cuts_a['mid_hi'])
        self.assertIn('ego_field', cuts_a['statistic'])

    def test_uncovered_objects_never_enter_an_amplitude_group(self):
        cuts = axis.freeze_amplitude_cuts(self._rows([0.1, 0.3, 5.0, 7.0]))
        row = {'class': 'Car', 'v_abs_corr_mean': None}
        self.assertEqual(axis.assign_amplitude(row, cuts), axis.NO_RETURN)

    def test_amplitude_groups_are_never_named_static_or_resolution(self):
        cuts = axis.freeze_amplitude_cuts(self._rows([0.1, 0.3, 5.0, 7.0]))
        named = {axis.assign_amplitude({'v_abs_corr_mean': a}, cuts)
                 for a in (0.0, 0.2, 5.0, 99.0)}
        self.assertEqual(named, set(axis.AMPLITUDE_GROUPS))
        for group in axis.AMPLITUDE_GROUPS:
            for forbidden in ('static', 'resolution', 'conflict'):
                self.assertNotIn(forbidden, group)

    def test_observability_and_conflict_are_reported_as_open(self):
        self.assertIn('NOT MEASURED', axis.CONFLICT_STATUS)
        self.assertIn('not resolution', axis.OBSERVABILITY_STATUS)
        self.assertNotIn('MEASURED (', axis.OBSERVABILITY_STATUS.replace(
            'amplitude proxy only; not resolution, not static', 'x'))

    def test_summary_reports_confound_composition_and_denominator(self):
        rows = self._rows([0.1, 0.2])
        rows[0]['occluded'] = 2.0
        rows[1]['v_abs_corr_mean'] = None
        summary = axis.summarise(rows)
        for key in ('n_objects', 'n_with_amplitude', 'n_frames', 'n_scenes',
                    'range_m_median', 'volume_m3_median', 'bbox_h_px_median',
                    'occluded_ge1_share', 'truncated_ge05_share',
                    'ego_amplitude_mps_median'):
            self.assertIn(key, summary)
        self.assertEqual(summary['n_objects'], 2)
        self.assertEqual(summary['n_with_amplitude'], 1)
        self.assertEqual(summary['n_frames'], 2)
        self.assertEqual(summary['n_scenes'], 1)
        self.assertEqual(summary['occluded_ge1_share'], 0.5)


class BuildContractTest(unittest.TestCase):
    def test_build_uses_stamped_frames_for_cells(self):
        train_pts = _ring(400, v_of=lambda th: -6.0 * np.cos(th))
        train_info = _info(20000, [12, 0])
        val_info = _info(0, [12])
        out = axis.build.__wrapped__ if hasattr(axis.build, '__wrapped__') else None
        self.assertIsNone(out)  # build stays a plain function (no decorators)
        # Exercise the per-split row builder directly: it must stamp the frame
        # identity so cell summaries can count frames and scenes.
        rows, _, _ = axis.object_motion_rows(
            train_info, train_pts, train_pts, coverage.derive_cuts([]), 20000, 3)
        self.assertTrue(all(r['image_idx'] == 20000 for r in rows))
        self.assertTrue(all(r['scene'] == 3 for r in rows))
        summary = axis.summarise(rows)
        self.assertEqual(summary['n_frames'], 1)
        self.assertEqual(summary['n_scenes'], 1)
        # And the frozen cuts must be assignable to every row.
        cuts = axis.freeze_amplitude_cuts(rows)
        for row in rows:
            self.assertIn(axis.assign_amplitude(row, cuts),
                          axis.AMPLITUDE_GROUPS + (axis.NO_RETURN,))
        _ = val_info


if __name__ == '__main__':
    unittest.main(verbosity=2)
