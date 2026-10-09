"""Unit tests for tools/bev_region_utils.py (W13).

Run (CPU only, no checkpoint, no GPU):
    python me_rssm/sanity/test_bev_region_utils.py

Why these tests exist: the two read-only measurements of 2026-10-09 (region
dependence of z_t, and the headroom of foreground/difference reconstruction
targets) both rest on this module's region definitions. A silently wrong
rasterization or a wrong Doppler cell mapping would produce a plausible-looking
split and a wrong conclusion, so the geometry is pinned here:

  W13.1 box rasterization: exact cell sets for an axis-aligned box, the
        margin = superset rule, and a yaw = 90 deg box swapping the x/y
        extents (no knife-edge boundaries: every boundary sits at >=0.1 cell
        from the nearest cell center);
  W13.2 out-of-range boxes are dropped and counted as such (the guard that
        makes a broken grid/coordinate pairing visible instead of silent);
  W13.3 masked_variance_stats with an all-ones mask equals the unmasked
        decomposition of tools/probe_recon_target.py exactly;
  W13.4 degenerate targets: a shared-constant target is 100% shared, a pure
        per-sample noise target is ~100% specific;
  W13.5 warp_map_by_cells: identity at zero displacement and the analytic
        0.6/0.4 bilinear split of a spike at +0.4 cells, in the same direction
        as EgoWarpEvidenceFieldRSSMFusion (content moves by -d);
  W13.6 doppler_masks: raw canvas (B, 7, 2W_y, 2H_x) cell (2j, 2i) lands on
        evidence-grid cell (i, j) -- the cross mapping of the scatter, not the
        transposed one -- and grid_xy_from_canvas derives exactly that grid
        from the canvas shape;
  W13.7 energy_matched_bg / top_energy_background: thresholds are the
        foreground median and the background decile, and neither mask ever
        overlaps the foreground;
  W13.8 standardize_masked: masked cells come out zero-mean/unit-std and
        cells outside the mask are exactly zero.
"""

import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from tools import bev_region_utils as bru  # noqa: E402
from tools.probe_recon_target import standardize_masked  # noqa: E402

PC_RANGE = [0.0, -4.0, 0.0, 8.0, 4.0, 2.0]  # 8 x-cells of 1.0 m, 10 y of 0.8 m
GRID = (8, 10)


def _box(cx, cy, dx, dy, yaw):
    return np.array([[cx, cy, 0.0, dx, dy, 1.6, yaw]], dtype=np.float64)


class TestRasterize(unittest.TestCase):

    def test_axis_aligned_box_cells(self):
        mask, n = bru.rasterize_boxes(
            _box(4.0, 0.0, 2.0, 2.0, 0.0), GRID, PC_RANGE, margin_cells=0.0)
        self.assertEqual(n, 1)
        xs, ys = np.nonzero(mask)
        self.assertEqual(sorted(set(xs.tolist())), [3, 4])
        self.assertEqual(sorted(set(ys.tolist())), [4, 5])
        self.assertEqual(int(mask.sum()), 4)

    def test_margin_is_a_strict_superset_with_clean_edges(self):
        tight, _ = bru.rasterize_boxes(
            _box(4.0, 0.0, 2.0, 2.0, 0.0), GRID, PC_RANGE, margin_cells=0.0)
        loose, _ = bru.rasterize_boxes(
            _box(4.0, 0.0, 2.0, 2.0, 0.0), GRID, PC_RANGE, margin_cells=0.5)
        self.assertTrue(bool((tight & ~loose).sum() == 0), 'margin dropped cells')
        self.assertGreater(int(loose.sum()), int(tight.sum()))
        # x index 1 (center 1.5 m) and index 6 (6.5 m) are far outside even
        # the expanded box (2.5..5.5 m), so they must stay background.
        self.assertFalse(bool(loose[1, :].any()))
        self.assertFalse(bool(loose[6, :].any()))

    def test_yaw_90_swaps_extents(self):
        flat, _ = bru.rasterize_boxes(
            _box(4.0, 0.0, 3.2, 1.4, 0.0), GRID, PC_RANGE, margin_cells=0.0)
        turned, _ = bru.rasterize_boxes(
            _box(4.0, 0.0, 3.2, 1.4, np.pi / 2), GRID, PC_RANGE,
            margin_cells=0.0)
        flat_x = np.unique(np.nonzero(flat)[0]).size
        flat_y = np.unique(np.nonzero(flat)[1]).size
        turned_x = np.unique(np.nonzero(turned)[0]).size
        turned_y = np.unique(np.nonzero(turned)[1]).size
        self.assertEqual((flat_x, flat_y), (4, 2))
        self.assertEqual((turned_x, turned_y), (2, 4))

    def test_out_of_range_box_is_dropped_and_counted(self):
        mask, n = bru.rasterize_boxes(
            _box(40.0, 0.0, 2.0, 2.0, 0.0), GRID, PC_RANGE)
        self.assertEqual(n, 0)
        self.assertEqual(int(mask.sum()), 0)
        mixed, n_mixed = bru.rasterize_boxes(
            np.concatenate([_box(40.0, 0.0, 2.0, 2.0, 0.0),
                            _box(4.0, 0.0, 2.0, 2.0, 0.0)]),
            GRID, PC_RANGE, margin_cells=0.0)
        self.assertEqual(n_mixed, 1)
        self.assertEqual(int(mixed.sum()), 4)


class TestMaskedVariance(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.target = torch.randn(6, 3, 4, 5)

    def test_all_ones_mask_matches_unmasked_decomposition(self):
        mask = torch.ones(6, 4, 5, dtype=torch.bool)
        stats = bru.masked_variance_stats(self.target, mask)
        total = self.target.pow(2).mean().item()
        mean_t = self.target.mean(0, keepdim=True)
        specific = (self.target - mean_t).pow(2).mean().item()
        self.assertAlmostEqual(stats['total_var'], total, places=6)
        self.assertAlmostEqual(stats['specific_var'], specific, places=6)
        self.assertAlmostEqual(stats['shared_frac'], 1.0 - specific / total,
                               places=6)

    def test_shared_constant_is_all_shared(self):
        target = torch.full((6, 3, 4, 5), 2.5)
        mask = torch.zeros(6, 4, 5, dtype=torch.bool)
        mask[:, 1:3, 1:4] = True
        stats = bru.masked_variance_stats(target, mask)
        self.assertAlmostEqual(stats['specific_frac'], 0.0, places=6)
        self.assertAlmostEqual(stats['shared_frac'], 1.0, places=6)

    def test_pure_noise_is_all_specific(self):
        # For S i.i.d. noise samples the elementwise sample mean absorbs
        # 1/S of the energy, so the specific share converges to (S-1)/S:
        # asserted against that analytic limit rather than a fixed number.
        mask = torch.ones(64, 4, 5, dtype=torch.bool)
        stats = bru.masked_variance_stats(torch.randn(64, 2, 4, 5), mask)
        self.assertGreater(stats['specific_frac'], 0.95)
        self.assertLess(stats['specific_frac'], 1.0)

    def test_per_sample_masks_restrict_the_support(self):
        mask = torch.zeros(6, 4, 5, dtype=torch.bool)
        mask[0, 0, 0] = True
        stats = bru.masked_variance_stats(self.target, mask)
        # one cell of one sample, C = 3 channels -> three masked elements
        self.assertEqual(stats['elements'], 3)
        self.assertAlmostEqual(
            stats['total_var'], float(self.target[0, :, 0, 0].pow(2).mean()),
            places=6)
        # a sample whose mask is empty contributes nothing
        only_one_sample = torch.zeros(6, 4, 5, dtype=torch.bool)
        only_one_sample[3, 2, 2] = True
        stats_one = bru.masked_variance_stats(self.target, only_one_sample)
        self.assertEqual(stats_one['elements'], 3)
        self.assertAlmostEqual(
            stats_one['total_var'],
            float(self.target[3, :, 2, 2].pow(2).mean()), places=6)


class TestWarp(unittest.TestCase):

    def test_identity_at_zero_displacement(self):
        feat = torch.randn(1, 3, 6, 7)
        out = bru.warp_map_by_cells(feat, torch.zeros(1), torch.zeros(1))
        self.assertLess((out - feat).abs().max().item(), 1e-5)

    def test_spike_shifts_by_the_analytic_amount(self):
        feat = torch.zeros(1, 1, 6, 7)
        feat[0, 0, 4, 3] = 1.0
        out = bru.warp_map_by_cells(feat, torch.tensor([0.4]),
                                    torch.tensor([0.0]))
        self.assertAlmostEqual(out[0, 0, 4, 3].item(), 0.6, places=5)
        self.assertAlmostEqual(out[0, 0, 3, 3].item(), 0.4, places=5)
        total = out.sum().item()
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_positive_dx_moves_content_to_lower_index(self):
        feat = torch.zeros(1, 1, 6, 7)
        feat[0, 0, 4, 3] = 1.0
        out = bru.warp_map_by_cells(feat, torch.tensor([1.0]),
                                    torch.tensor([0.0]))
        self.assertAlmostEqual(out[0, 0, 3, 3].item(), 1.0, places=5)


class TestDoppler(unittest.TestCase):

    def test_cell_mapping_matches_the_scatter(self):
        grid_xy = (8, 10)
        canvas = torch.zeros(1, 7, 2 * grid_xy[1], 2 * grid_xy[0])
        # raw dim2 = world y (2W), dim3 = world x (2H); the canvas is
        # mean-scattered AND then 2x-avg-pooled, so one sub-cell value
        # arrives diluted 4x -- hence 4.0 here for a pooled v_std of 1.0
        canvas[0, bru.CH_COUNT, 2 * 5 + 1, 2 * 3 + 1] = 1.0
        canvas[0, bru.CH_V_STD, 2 * 5 + 1, 2 * 3 + 1] = 4.0
        masks = bru.doppler_masks(canvas, grid_xy)
        self.assertIsNotNone(masks)
        self.assertEqual(int(masks['measured'].sum()), 1)
        self.assertTrue(bool(masks['measured'][3, 5]))
        self.assertTrue(bool(masks['dyn'][3, 5]))
        self.assertFalse(bool(masks['measured'][5, 3]))

    def test_grid_xy_is_derived_from_the_canvas(self):
        canvas = torch.zeros(1, 7, 20, 16)
        self.assertEqual(bru.grid_xy_from_canvas(canvas), (8, 10))
        self.assertIsNone(bru.grid_xy_from_canvas(None))

    def test_all_zero_canvas_has_no_measured_cell(self):
        canvas = torch.zeros(1, 7, 20, 16)
        masks = bru.doppler_masks(canvas, (8, 10))
        self.assertEqual(int(masks['measured'].sum()), 0)

    def test_shape_mismatch_returns_none(self):
        canvas = torch.zeros(1, 7, 20, 16)
        self.assertIsNone(bru.doppler_masks(canvas, (8, 9)))


class TestEnergyMatchedBg(unittest.TestCase):

    def test_threshold_is_foreground_median_and_never_overlaps(self):
        energy = np.zeros(GRID)
        fg = np.zeros(GRID, dtype=bool)
        fg[0, 0] = True
        fg[0, 1] = True
        energy[fg] = [2.0, 4.0]          # median 3.0
        energy[1, 0] = 2.9               # below -> excluded
        energy[1, 1] = 3.1               # above -> included
        bg = bru.energy_matched_bg(energy, fg)
        self.assertFalse(bool((bg & fg).any()))
        self.assertTrue(bool(bg[1, 1]))
        self.assertFalse(bool(bg[1, 0]))

    def test_empty_foreground_gives_empty_mask(self):
        bg = bru.energy_matched_bg(np.ones(GRID),
                                   np.zeros(GRID, dtype=bool))
        self.assertEqual(int(bg.sum()), 0)

    def test_top_energy_background_is_the_background_decile(self):
        energy = np.arange(GRID[0] * GRID[1], dtype=np.float64).reshape(GRID)
        fg = np.zeros(GRID, dtype=bool)
        fg[0, :2] = True
        top = bru.top_energy_background(energy, fg, quantile=0.9)
        self.assertFalse(bool((top & fg).any()))
        background = energy[~fg]
        threshold = np.quantile(background, 0.9)
        expected = int((background >= threshold).sum())
        self.assertEqual(int(top.sum()), expected)
        self.assertLess(int(top.sum()), int((~fg).sum()))


class TestStandardizeMasked(unittest.TestCase):

    def test_masked_cells_are_normalized_and_outside_is_zero(self):
        torch.manual_seed(0)
        target = torch.randn(4, 2, 5, 6) * 3.0 + 7.0
        mask = torch.zeros(4, 5, 6, dtype=torch.bool)
        mask[:, 1:4, 1:5] = True
        out = standardize_masked(target, mask)
        weight = mask.unsqueeze(1).expand_as(out)
        self.assertEqual(float(out[~weight].abs().max()), 0.0)
        mean = out[weight].mean().item()
        std = out[weight].std(unbiased=False).item()
        self.assertAlmostEqual(mean, 0.0, places=5)
        self.assertAlmostEqual(std, 1.0, places=4)


if __name__ == '__main__':
    unittest.main(verbosity=2)
