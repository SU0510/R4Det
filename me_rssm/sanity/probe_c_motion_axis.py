#!/usr/bin/env python3
"""Candidate-C diagnostic, step 1b: the motion-observability axis (read-only CPU).

Coverage alone cannot carry the C hypothesis; the agreed contract is
class x coverage x motion-observability. This probe builds the second axis and
refuses to over-name it.

WHICH CHANNEL IS WHICH (three independent pieces of evidence)
-------------------------------------------------------------
1. Code convention. The baseline pipeline loads the radar file with
   ``LoadPointsFromFile(load_dim=8, use_dim=[0, 1, 2, 3, 5])``
   (configs/r4det/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head.py:232), so the
   network sees raw dims x, y, z, <raw 3>, <raw 5>. ``me_rssm/
   radar_motion_chain.py:81`` names dims 0..4 ``x, y, z, radial velocity, SNR``
   and :93 reads ``snr = features[:, :, 4]`` -- therefore selected index 3 =
   RAW index 3 = radial velocity, and selected index 4 = RAW index 5 = SNR.
   The 5th selected channel is NOT a velocity.
2. Geometric identity, measured (not assumed):
     dim 4 == sqrt(x^2 + y^2 + z^2)      (range, metres)
     dim 6 == -degrees(atan2(y, x))      (exact, residual 0.0)
   dim 7 does NOT reproduce the elevation implied by (x, y, z); the identity is
   REPORTED AS FAILED and the channel is left UNIDENTIFIED.
3. Fill markers, measured: dim 2 == 0 and dim 7 == 0 co-occur exactly
   (same set, same share) and should be read as one missing-elevation marker.

WHAT IS MEASURED ABOUT THE VELOCITY PHYSICS, AND WHAT IS NOT
------------------------------------------------------------
The exporter that wrote the 8-feature file is NOT in this repository (the
converters only read the .bin), so unit, sign and ego-compensation cannot be
settled by reading code. They are settled here by measurement instead:

  * fitting ``v_raw = a*cos(theta) + b*sin(theta) + c`` per frame on the points
    outside every GT box recovers the ego-induced range-rate field. On clean
    frames 99.5% of the velocity variance is explained by that three-term
    model, and objects with no motion relative to the ego come out at
    |v_corr| ~ 0.05 m/s. So: the channel is RAW range-rate dominated by ego
    motion, the ego term enters with a NEGATIVE sign (-(vx*cos + vy*sin)), and
    frames differ in ego speed by 0..27 m/s;
  * ``v_corr = v_raw - (a*cos + b*sin + c)`` is therefore the only version of
    the amplitude axis that is not a proxy for "which block this frame is in";
  * the unit is CONSISTENT WITH m/s (metric geometry + plausible vehicle
    speeds) but is NOT PROVEN, and no compensation is guaranteed to have been
    applied by the exporter -- what is proven is that the fitted common-mode
    field is there and can be removed.

A THIRD DISCRIMINANT, AND WHY IT IS NOT A TUNED THRESHOLD
---------------------------------------------------------
Some of this dataset's frames return almost nothing at all: the whole scan is
nearly empty (the parked-ego blocks carry a v == 0 share above 0.97 and far
fewer points). A zero-coverage object in such a frame says nothing about that
OBJECT, so every zero row is labelled by the frame it came from:

  * ``frame_level`` -- no scored object in the frame has any return;
  * ``object_level`` -- the frame does have returns, this object does not.

The rule is qualitative (share == 1.0 vs < 1.0), so there is no boundary to fit
after seeing the answers -- which is the point.

WHAT IT REFUSES TO CLAIM (user instruction, 2026-10-12)
-------------------------------------------------------
  * a LOW |v_corr| is NOT "static" and NOT "below sensor resolution": the axis
    is named for the proxy it is, ``v_amp_low/mid/high``;
  * a large velocity VARIANCE is NOT "conflicting measurements": a conflict
    needs a prediction, a shared frame and a residual rule, none of which exist
    here, so the artifact carries ``conflict: NOT MEASURED``.

All cuts are frozen from TRAIN only and applied verbatim to val (section 76
convention). Read-only: two pkl files in, one JSON out. No model, no GPU.

Usage:
  python me_rssm/sanity/probe_c_motion_axis.py \
    --train-info data/TJ4D/TJ4D_infos_train.pkl \
    --val-info data/TJ4D/TJ4D_infos_val.pkl --data-root data/TJ4D \
    --output /tmp/c_motion_axis.json
"""

import argparse
import collections
import importlib.util
import json
import os
import pickle
import sys

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from mmdet3d.core.bbox import box_np_ops  # noqa: E402

# Channel indices in the RAW 8-feature TJ4D radar point (before selection).
CH_X, CH_Y, CH_Z, CH_V, CH_RANGE, CH_SNR, CH_AZ, CH_EL = range(8)
AMPLITUDE_GROUPS = ('v_amp_low', 'v_amp_mid', 'v_amp_high')
NO_RETURN = 'no_return'
OBSERVABILITY_STATUS = 'PARTIAL (amplitude proxy only; not resolution, not static)'
CONFLICT_STATUS = 'NOT MEASURED (no prediction, no shared frame, no residual rule)'


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_REPO, rel))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


coverage = _load('probe_c_groups_for_motion',
                 'me_rssm/sanity/probe_c_coverage_groups.py')


def load_points(path):
    pts = np.fromfile(path, dtype=np.float32)
    if pts.size % 8:
        raise ValueError(f'{path}: {pts.size} floats is not a multiple of 8')
    return pts.reshape(-1, 8)


def azimuth_rad(points):
    """Geometric azimuth, derived from x/y -- not from the unproven dim 6."""
    return np.arctan2(points[:, CH_Y].astype(np.float64),
                      points[:, CH_X].astype(np.float64))


def identify_channels(samples):
    """Identity check for dims 4 / 6 / 7, aggregated over sampled frames."""
    err_range, err_az, err_el, zeros, nan_total = [], [], [], [], 0
    fill_cooccur = []
    snr_min, snr_max = np.inf, -np.inf
    slope_el, resid_el = [], []
    n_points = 0
    for points in samples:
        x, y, z = (points[:, c].astype(np.float64) for c in (CH_X, CH_Y, CH_Z))
        rng = np.sqrt(x * x + y * y + z * z)
        az = np.degrees(np.arctan2(y, x))
        el = np.degrees(np.arctan2(z, np.maximum(np.hypot(x, y), 1e-6)))
        err_range.append(np.max(np.abs(points[:, CH_RANGE] - rng)))
        err_az.append(np.max(np.abs(points[:, CH_AZ] + az)))
        err_el.append(np.max(np.abs(points[:, CH_EL] - el)))
        design = np.vstack([el, np.ones_like(el)]).T
        coef, *_ = np.linalg.lstsq(design, points[:, CH_EL].astype(np.float64),
                                   rcond=None)
        slope_el.append(float(coef[0]))
        resid_el.append(float(np.std(points[:, CH_EL].astype(np.float64)
                                     - design @ coef)))
        zeros.append({str(k): float(np.mean(points[:, k] == 0.0))
                      for k in range(8)})
        # Fill-marker test: are the z == 0 and dim7 == 0 sets the SAME set?
        fill_cooccur.append({
            'dim2_zero_share': float(np.mean(points[:, CH_Z] == 0.0)),
            'dim7_zero_share': float(np.mean(points[:, CH_EL] == 0.0)),
            'same_set': bool(np.array_equal(points[:, CH_Z] == 0.0,
                                            points[:, CH_EL] == 0.0)),
        })
        nan_total += int(np.isnan(points).sum())
        snr_min = min(snr_min, float(points[:, CH_SNR].min()))
        snr_max = max(snr_max, float(points[:, CH_SNR].max()))
        n_points += int(points.shape[0])
    groups = range(8)
    zero_share = {str(k): float(np.median([z[str(k)] for z in zeros]))
                  for k in groups}
    return {
        'n_frames_sampled': len(samples),
        'n_points': n_points,
        'dim4_is_range_max_abs_err': float(max(err_range)),
        'dim4_is_range': bool(max(err_range) < 0.05),
        'dim6_is_neg_azimuth_deg_max_abs_err': float(max(err_az)),
        'dim6_form': 'dim6 == -degrees(atan2(y, x))',
        'dim6_is_neg_azimuth_deg': bool(max(err_az) < 0.05),
        'dim7_vs_geometric_elevation_max_abs_err': float(max(err_el)),
        'dim7_is_geometric_elevation': bool(max(err_el) < 0.05),
        'dim7_best_linear_fit_slope_vs_elevation': float(np.median(slope_el)),
        'dim7_best_linear_fit_resid_std': float(np.median(resid_el)),
        'dim7_note': ('identity FAILED: dim7 is not the elevation implied by '
                      '(x, y, z) and stays UNIDENTIFIED; it co-occurs exactly '
                      'with the z == 0 fill marker, so the two are reported as '
                      'one missing-elevation marker rather than given a name'),
        'fill_marker_z_and_el_zero_co_occur_exactly': bool(
            all(f['same_set'] for f in fill_cooccur)),
        'fill_marker_per_frame': fill_cooccur,
        'exact_zero_share_per_dim_median': zero_share,
        'zero_share_range_m_units': 'metres (dim 4)',
        'snr_min': snr_min,
        'snr_max': snr_max,
        'nan_count': nan_total,
    }


def frame_boxes(info):
    """GT boxes in lidar frame + stored coverage, ignoring DontCare/ignored."""
    annos = info['annos']
    names = [str(n) for n in annos['name']]
    stored = np.asarray(annos['num_points_in_gt'], dtype=np.int64)
    dims = np.asarray(annos['dimensions'], dtype=np.float64)
    loc = np.asarray(annos['location'], dtype=np.float64)
    rots = np.asarray(annos['rotation_y'], dtype=np.float64)
    boxes_px = np.asarray(annos['bbox'], dtype=np.float64)
    occ = np.asarray(annos.get('occluded', np.zeros(len(names))), dtype=np.float64)
    trunc = np.asarray(annos.get('truncated', np.zeros(len(names))),
                       dtype=np.float64)
    diff = np.asarray(annos.get('difficulty', np.full(len(names), -1)),
                      dtype=np.float64)
    no_obj = len([n for n in names if n != coverage.DONTCARE])
    cam = np.concatenate([loc[:no_obj], dims[:no_obj],
                          rots[:no_obj, np.newaxis]], axis=1)
    # R0_rect and Tr_velo_to_cam are stored as 4x4 in these infos, exactly as
    # the converter passes them to box_camera_to_lidar.
    rect = np.asarray(info['calib']['R0_rect'], dtype=np.float64).reshape(4, 4)
    trv2c = np.asarray(info['calib']['Tr_velo_to_cam'],
                       dtype=np.float64).reshape(4, 4)
    boxes = box_np_ops.box_camera_to_lidar(cam, rect, trv2c)
    return {'names': names, 'no_obj': no_obj, 'stored': stored, 'boxes': boxes,
            'loc': loc[:no_obj], 'dims': dims[:no_obj], 'bbox_px': boxes_px,
            'occluded': occ, 'truncated': trunc, 'difficulty': diff}


def filter_outside(info, points):
    """The frustum filter the converter applied BEFORE counting coverage."""
    calib = info['calib']
    return box_np_ops.remove_outside_points(
        points, np.asarray(calib['R0_rect'], dtype=np.float64).reshape(4, 4),
        np.asarray(calib['Tr_velo_to_cam'], dtype=np.float64).reshape(4, 4),
        np.asarray(calib['P2'], dtype=np.float64).reshape(4, 4),
        info['image']['image_shape'])


def fit_ego_field(points, mask, trim_sigma=3.0):
    """Fit v_raw = a*cos(theta) + b*sin(theta) + c on the chosen points.

    ``mask`` selects the fit population (background = outside every GT box).
    One 3-sigma trimming pass removes moving targets and multipath before the
    reported coefficients. Under the static-clutter model the fitted vector
    (-a, -b) IS the ego velocity in sensor axes, with the NEGATIVE sign: a
    forward-driving ego makes static returns systematically negative.
    """
    sel = mask & np.isfinite(points[:, CH_V])
    if int(sel.sum()) < 30:
        sel = np.isfinite(points[:, CH_V])
        source = 'all_points_fallback'
    else:
        source = 'background_outside_gt_boxes'
    theta = azimuth_rad(points)
    design = np.vstack([np.cos(theta[sel]), np.sin(theta[sel]),
                        np.ones(int(sel.sum()))]).T
    v = points[sel, CH_V].astype(np.float64)
    coef, *_ = np.linalg.lstsq(design, v, rcond=None)
    resid = v - design @ coef
    keep = np.abs(resid) <= trim_sigma * (resid.std() + 1e-9)
    n_inlier = int(keep.sum())
    if n_inlier >= 30:
        coef, *_ = np.linalg.lstsq(design[keep], v[keep], rcond=None)
        resid = v - design @ coef
    total = v - v.mean()
    return {
        'a': float(coef[0]), 'b': float(coef[1]), 'c': float(coef[2]),
        'ego_amplitude': float(np.hypot(coef[0], coef[1])),
        'resid_std': float(resid.std()),
        'v_raw_std': float(total.std()),
        'r2': float(1.0 - np.sum(resid ** 2) / max(np.sum(total ** 2), 1e-12)),
        'n_fit_points': int(sel.sum()), 'n_inlier': n_inlier,
        'fit_source': source,
    }


def ego_correct(points, ego):
    theta = azimuth_rad(points)
    return points[:, CH_V].astype(np.float64) - (
        ego['a'] * np.cos(theta) + ego['b'] * np.sin(theta) + ego['c'])


def object_motion_rows(info, points_raw, points_kept, cuts, image_idx, scene):
    """One row per scored GT object: coverage + amplitude + confound controls.

    Counting uses ``points_kept`` (the frustum-filtered cloud) so the counts
    must reproduce ``annos['num_points_in_gt']`` -- mismatches are reported.
    ``coverage_raw`` is the same count on the cloud the NETWORK actually loads,
    because the pipeline has no frustum filter: a stored zero can still be a
    non-zero number of points as far as the model is concerned.
    """
    geo = frame_boxes(info)
    inside_kept = box_np_ops.points_in_rbbox(points_kept[:, :3], geo['boxes'])
    inside_raw = box_np_ops.points_in_rbbox(points_raw[:, :3], geo['boxes'])
    background = ~inside_raw.any(axis=1)
    ego = fit_ego_field(points_raw, background)
    v_corr_raw = ego_correct(points_raw, ego)
    v_corr_kept = ego_correct(points_kept, ego)

    rows, mismatches = [], 0
    for k, name in enumerate(geo['names'][:geo['no_obj']]):
        if name == coverage.DONTCARE:
            continue
        sel_kept, sel_raw = inside_kept[:, k], inside_raw[:, k]
        n_kept, n_raw = int(sel_kept.sum()), int(sel_raw.sum())
        if n_kept != int(geo['stored'][k]):
            mismatches += 1
        loc, dims = geo['loc'][k], geo['dims'][k]
        pbox = geo['bbox_px'][k]
        row = {
            'class': name,
            'image_idx': int(image_idx), 'scene': int(scene),
            'coverage': int(geo['stored'][k]),
            'coverage_group': coverage.assign_group(int(geo['stored'][k]), cuts),
            'n_points_kept': n_kept, 'n_points_raw': n_raw,
            'range_m': float(np.hypot(loc[0], loc[2])),
            'volume_m3': float(np.prod(dims)),
            'bbox_h_px': (float(pbox[3] - pbox[1]) if pbox.size >= 4 else None),
            'occluded': float(geo['occluded'][k]),
            'truncated': float(geo['truncated'][k]),
            'difficulty': float(geo['difficulty'][k]),
            'ego_amplitude_mps': ego['ego_amplitude'],
            'ego_fit_r2': ego['r2'],
            # Frame-level context, filled in below: an object with no return is
            # only evidence about that OBJECT if its frame had returns.
            'frame_n_points': int(points_raw.shape[0]),
            'zero_scope': None,
        }
        if n_raw:
            v = points_raw[sel_raw, CH_V].astype(np.float64)
            vc = v_corr_raw[sel_raw]
            row.update({
                'v_raw_mean': float(np.mean(v)),
                'v_abs_corr_mean': float(np.mean(np.abs(vc))),
                'v_corr_mean': float(np.mean(vc)),
                'v_corr_std': float(np.std(vc)),
                'snr_mean': float(np.mean(points_raw[sel_raw, CH_SNR])),
            })
        if n_kept:
            row['v_abs_corr_mean_kept'] = float(np.mean(np.abs(v_corr_kept[sel_kept])))
        rows.append(row)
    # Qualitative scope label: no tuned boundary, no quantile.
    n_zero = sum(1 for r in rows if r['coverage'] == 0)
    scope = 'frame_level' if (rows and n_zero == len(rows)) else 'object_level'
    frame_zero_share = (float(n_zero) / len(rows)) if rows else None
    for row in rows:
        row['frame_zero_share'] = frame_zero_share
        if row['coverage'] == 0:
            row['zero_scope'] = scope
    return rows, int(mismatches), ego


def freeze_amplitude_cuts(train_rows):
    """Terciles of the per-object mean |v_corr| over TRAIN covered objects.

    Computed on the ego-corrected velocity, because the raw channel's spread is
    dominated by the per-frame ego term (0..27 m/s), which would make any raw
    cut a proxy for "which block/ego speed this frame has".
    """
    vals = np.asarray([r['v_abs_corr_mean'] for r in train_rows
                       if r.get('v_abs_corr_mean') is not None],
                      dtype=np.float64)
    if vals.size == 0:
        return {'low_hi': None, 'mid_hi': None, 'q1': None, 'q2': None}
    q1, q2 = (float(np.percentile(vals, 100.0 / 3.0)),
              float(np.percentile(vals, 200.0 / 3.0)))
    return {'low_hi': q1, 'mid_hi': q2, 'q1': q1, 'q2': q2,
            'statistic': 'per-object mean |v_raw - ego_field| over covered objects',
            'derived_from': 'train'}


def assign_amplitude(row, cuts):
    if row.get('v_abs_corr_mean') is None:
        return NO_RETURN
    if cuts['low_hi'] is None:
        return AMPLITUDE_GROUPS[0]
    if row['v_abs_corr_mean'] <= cuts['low_hi']:
        return AMPLITUDE_GROUPS[0]
    if row['v_abs_corr_mean'] <= cuts['mid_hi']:
        return AMPLITUDE_GROUPS[1]
    return AMPLITUDE_GROUPS[2]


def _median(values):
    values = [v for v in values if v is not None and np.isfinite(v)]
    return float(np.median(values)) if values else None


def _share_true(values, threshold=1.0):
    values = [v for v in values if v is not None and np.isfinite(v)]
    if not values:
        return None
    return float(np.mean([1.0 if v >= threshold else 0.0 for v in values]))


def summarise(rows):
    """Group composition: counts AND the range/size/occlusion confounds."""
    if not rows:
        return {'n_objects': 0}
    scenes = collections.Counter(r['scene'] for r in rows)
    top_scene, top_n = scenes.most_common(1)[0]
    return {
        'n_objects': len(rows),
        # Denominator for every v_* median below: objects with zero coverage
        # can still carry raw-cloud returns (the model's cloud has no frustum
        # filter), so the two counts are NOT the same population.
        'n_with_amplitude': sum(1 for r in rows
                                if r.get('v_abs_corr_mean') is not None),
        'n_frames': len({r['image_idx'] for r in rows}),
        'n_scenes': len(scenes),
        'top_scene_share': float(top_n / len(rows)),
        'coverage_median': _median([r['coverage'] for r in rows]),
        'range_m_median': _median([r['range_m'] for r in rows]),
        'range_m_p10': (float(np.percentile([r['range_m'] for r in rows], 10))
                        if rows else None),
        'range_m_p90': (float(np.percentile([r['range_m'] for r in rows], 90))
                        if rows else None),
        'volume_m3_median': _median([r['volume_m3'] for r in rows]),
        'bbox_h_px_median': _median([r['bbox_h_px'] for r in rows]),
        'occluded_ge1_share': _share_true([r['occluded'] for r in rows], 1.0),
        'truncated_ge05_share': _share_true([r['truncated'] for r in rows], 0.5),
        'difficulty_median': _median([r['difficulty'] for r in rows]),
        'v_abs_corr_mean_median': _median(
            [r.get('v_abs_corr_mean') for r in rows]),
        'v_raw_mean_median': _median([r.get('v_raw_mean') for r in rows]),
        'v_corr_std_median': _median([r.get('v_corr_std') for r in rows]),
        'snr_mean_median': _median([r.get('snr_mean') for r in rows]),
        'ego_amplitude_mps_median': _median([r['ego_amplitude_mps'] for r in rows]),
        'frame_n_points_median': _median([r.get('frame_n_points') for r in rows]),
        'frame_zero_share_median': _median([r.get('frame_zero_share') for r in rows]),
    }


def _columnar(frames):
    keys = ['image_idx', 'block', 'n_points', 'n_fit_points', 'n_inlier', 'a',
            'b', 'c', 'ego_amplitude', 'resid_std', 'v_raw_std', 'r2',
            'v_raw_zero_share']
    return {k: [f[k] for f in frames] for k in keys}


def _block_summary(frames):
    by_block = collections.defaultdict(list)
    for f in frames:
        by_block[f['block']].append(f)
    out = {}
    for block, group in sorted(by_block.items()):
        out[str(block)] = {
            'n_frames': len(group),
            'ego_amplitude_mps_median': _median([g['ego_amplitude'] for g in group]),
            'ego_amplitude_mps_p10': float(np.percentile(
                [g['ego_amplitude'] for g in group], 10)),
            'ego_amplitude_mps_p90': float(np.percentile(
                [g['ego_amplitude'] for g in group], 90)),
            'v_raw_median': _median([g['v_raw_median'] for g in group]),
            'resid_std_median': _median([g['resid_std'] for g in group]),
            'r2_median': _median([g['r2'] for g in group]),
            'v_raw_zero_share_median': _median(
                [g['v_raw_zero_share'] for g in group]),
        }
    return out


def _scan_split(infos, split, data_root, stride, cuts, identity_frames,
                scene_runs):
    rows, frames, mismatches, samples = [], [], 0, []
    for pos, info in enumerate(infos):
        if pos % stride:
            continue
        idx = int(info['image']['image_idx'])
        scene = scene_runs.get(idx, -1)
        path = os.path.join(data_root, info['point_cloud']['velodyne_path'])
        points = load_points(path)
        kept = filter_outside(info, points)
        obj_rows, mm, ego = object_motion_rows(
            info, points, kept, cuts, idx, scene)
        rows.extend(obj_rows)
        mismatches += mm
        if len(samples) < identity_frames:
            samples.append(points)
        frames.append({
            'image_idx': idx, 'block': idx // 10000, 'n_points': int(len(points)),
            'n_fit_points': ego['n_fit_points'], 'n_inlier': ego['n_inlier'],
            'a': ego['a'], 'b': ego['b'], 'c': ego['c'],
            'ego_amplitude': ego['ego_amplitude'],
            'resid_std': ego['resid_std'], 'v_raw_std': ego['v_raw_std'],
            'r2': ego['r2'],
            'v_raw_zero_share': float(np.mean(points[:, CH_V] == 0.0)),
            'v_raw_median': float(np.median(points[:, CH_V])),
        })
    return {'split': split, 'rows': rows, 'frames': frames,
            'mismatches': mismatches, 'samples': samples}


def build(train_infos, val_infos, data_root, stride=1, identity_frames=3):
    runs = coverage.scene_runs([i['image']['image_idx']
                                for i in train_infos + val_infos])
    # Coverage cuts come from the section-76 probe so both axes share ONE
    # frozen partition of the same population.
    coverage_cuts = coverage.derive_cuts(
        coverage.object_rows(train_infos, 'train', runs))

    scans = {split: _scan_split(infos, split, data_root, stride,
                                coverage_cuts, identity_frames, runs)
             for split, infos in (('train', train_infos), ('val', val_infos))}
    cuts = freeze_amplitude_cuts(scans['train']['rows'])
    cuts.update({
        'caveat': ('amplitude proxy only: a low |v_corr| is NOT proven static '
                   'and NOT proven below sensor resolution'),
        'conflict': CONFLICT_STATUS,
    })

    out = {
        'probe': 'candidate C motion axis (ego-corrected Doppler amplitude, read-only)',
        'status': {
            'amplitude_axis': 'MEASURED (ego-corrected amplitude proxy)',
            'observability': OBSERVABILITY_STATUS,
            'conflict': CONFLICT_STATUS,
            'unit': 'consistent with m/s but NOT PROVEN (exporter absent from repo)',
            'ego_compensation': ('MEASURED as a fitted per-frame common-mode '
                                 'field; whether the exporter already applied a '
                                 'compensation cannot be proven from this file'),
        },
        'channel_semantics': {
            'raw_layout': ('8 float32 dims: 0 x, 1 y, 2 z, 3 radial velocity, '
                           '4 range, 5 SNR, 6 -azimuth_deg, 7 UNIDENTIFIED'),
            'consumer_code': 'me_rssm/radar_motion_chain.py:81,93',
            'loader_config': ('configs/r4det/TJ4D-'
                              'R4Det_fgfull_N4_2x4_24e_pretrained_v2_head.py:232 '
                              '(load_dim=8, use_dim=[0, 1, 2, 3, 5])'),
            'selected_index_3': {'raw_index': CH_V, 'name': 'radial velocity'},
            'selected_index_4': {'raw_index': CH_SNR, 'name': 'SNR',
                                 'why_not_velocity': ('the 5th SELECTED channel '
                                                      'is SNR, not a velocity')},
            'raw_generator_in_repo': False,
            'raw_generator_note': (
                'the 8-feature file is produced outside this repository; unit, '
                'sign and ego-compensation are therefore measured here and '
                'otherwise marked NOT MEASURED'),
            'identity_check': identify_channels(scans['train']['samples']),
        },
        'coverage_cuts_source': 'probe_c_coverage_groups.derive_cuts (train)',
        'coverage_cuts': coverage_cuts,
        'amplitude_cuts': cuts,
        'ego_model': {
            'model': 'v_raw = a*cos(theta) + b*sin(theta) + c, per frame',
            'theta': 'atan2(y, x) from geometry (equals -dim6)',
            'fit_population': ('points outside every GT box, one 3-sigma '
                               'trimming pass; fallback = all points'),
            'interpretation': ('(-a, -b) is the ego velocity in sensor axes and '
                               'static clutter sits at v_corr = 0; NEGATIVE sign '
                               'w.r.t. forward motion'),
            'per_frame': {}, 'per_block': {}, 'split_summary': {},
        },
        'coverage_identity': {}, 'cells': {}, 'sampling': {
            'stride': stride,
            'n_train_frames_scanned': len(scans['train']['frames']),
            'n_val_frames_scanned': len(scans['val']['frames']),
        },
    }
    for split in ('train', 'val'):
        scan = scans[split]
        out['ego_model']['per_frame'][split] = _columnar(scan['frames'])
        out['ego_model']['per_block'][split] = _block_summary(scan['frames'])
        amps = [f['ego_amplitude'] for f in scan['frames']]
        r2 = [f['r2'] for f in scan['frames'] if f['v_raw_std'] > 0.2]
        zero_share = np.mean([f['v_raw_zero_share'] for f in scan['frames']])
        out['ego_model']['split_summary'][split] = {
            'n_frames': len(scan['frames']),
            'ego_amplitude_mps_median': _median(amps),
            'ego_amplitude_mps_p10': float(np.percentile(amps, 10)),
            'ego_amplitude_mps_p90': float(np.percentile(amps, 90)),
            'share_frames_ego_below_1mps': float(np.mean(
                [1.0 if a < 1.0 else 0.0 for a in amps])),
            'r2_median_frames_with_v_std_gt_0.2': _median(r2),
            'mean_v_raw_exact_zero_share': float(zero_share),
            'n_blocks': len({f['block'] for f in scan['frames']}),
        }
        out['coverage_identity'][split] = {
            'n_mismatch_frames_with_stored_num_points_in_gt':
                int(scan['mismatches']),
            'note': ('counts are computed on the frustum-filtered cloud, the '
                     'same convention as the converter; the raw-cloud counts '
                     'in coverage_raw differ by construction'),
        }
        rows = scan['rows']
        out['cells'][split] = {}
        classes = sorted({r['class'] for r in rows})
        for cov_group in (coverage.ZERO_GROUP, 'low', 'mid', 'high'):
            cov_rows = [r for r in rows if r['coverage_group'] == cov_group]
            for amp_group in AMPLITUDE_GROUPS:
                picked = [r for r in cov_rows
                          if assign_amplitude(r, cuts) == amp_group]
                out['cells'][split][f'{cov_group}|{amp_group}'] = {
                    'pooled': summarise(picked),
                    'per_class': {cls: summarise(
                        [r for r in picked if r['class'] == cls])
                        for cls in classes},
                }
        zero_rows = [r for r in rows if r['coverage'] == 0]
        zero_raw_rows = [r for r in rows if r['n_points_raw'] == 0]
        out['cells'][split][f'{coverage.ZERO_GROUP}|{NO_RETURN}'] = {
            'pooled': summarise(zero_rows),
            'per_class': {cls: summarise(
                [r for r in zero_rows if r['class'] == cls]) for cls in classes},
            'zero_scope': {
                scope: {
                    'pooled': summarise(
                        [r for r in zero_rows if r['zero_scope'] == scope]),
                    'per_class': {cls: summarise(
                        [r for r in zero_rows
                         if r['zero_scope'] == scope and r['class'] == cls])
                        for cls in classes},
                } for scope in ('object_level', 'frame_level')},
            'network_visible_zero': {
                'definition': 'n_points_raw == 0 (nothing in the model input)',
                'n_objects': len(zero_raw_rows),
                'share_of_stored_zero': (float(len(zero_raw_rows))
                                         / max(1, len(zero_rows))),
            },
        }
        out['cells'][split]['coverage_axis'] = {
            g: summarise([r for r in rows if r['coverage_group'] == g])
            for g in (coverage.ZERO_GROUP, 'low', 'mid', 'high')}
        out['cells'][split]['amplitude_axis'] = {
            g: summarise([r for r in rows if assign_amplitude(r, cuts) == g])
            for g in AMPLITUDE_GROUPS}
    return out


def _fmt(value, digits=2):
    if value is None:
        return 'n/a'
    if isinstance(value, float):
        return f'{value:.{digits}f}'
    return str(value)


def print_summary(result, stream=sys.stdout):
    ch = result['channel_semantics']['identity_check']
    print(f"dim4==range: {ch['dim4_is_range']} "
          f"(max err {ch['dim4_is_range_max_abs_err']:.2e}) | "
          f"dim6==-azimuth: {ch['dim6_is_neg_azimuth_deg']} "
          f"(max err {ch['dim6_is_neg_azimuth_deg_max_abs_err']:.2e}) | "
          f"dim7==elevation: {ch['dim7_is_geometric_elevation']} "
          f"(max err {ch['dim7_vs_geometric_elevation_max_abs_err']:.2f})", file=stream)
    print(f"coverage mismatches: {result['coverage_identity']}", file=stream)
    for split in ('train', 'val'):
        s = result['ego_model']['split_summary'][split]
        print(f'{split}: {s["n_frames"]} frames, ego |v| median '
              f'{_fmt(s["ego_amplitude_mps_median"])} m/s '
              f'[p10 {_fmt(s["ego_amplitude_mps_p10"])}, '
              f'p90 {_fmt(s["ego_amplitude_mps_p90"])}], '
              f'R2 median {_fmt(s["r2_median_frames_with_v_std_gt_0.2"], 4)}, '
              f'share frames with ego<1 m/s '
              f'{_fmt(100 * s["share_frames_ego_below_1mps"], 1)}%', file=stream)
    cuts = result['amplitude_cuts']
    print(f'amplitude cuts (TRAIN, |v_corr|): low<={_fmt(cuts["low_hi"], 3)} '
          f'mid<={_fmt(cuts["mid_hi"], 3)}', file=stream)
    for split in ('train', 'val'):
        print(f'--- {split} cells (coverage|amplitude)', file=stream)
        for key in sorted(result['cells'][split]):
            if '|' not in key:
                continue
            s = result['cells'][split][key]['pooled']
            if not s.get('n_objects'):
                continue
            print(f'  {key:24s} n={s["n_objects"]:6d} fr={s["n_frames"]:5d} '
                  f'sc={s["n_scenes"]:3d} range={_fmt(s["range_m_median"], 1):>6s} '
                  f'|v_corr|={_fmt(s["v_abs_corr_mean_median"], 3):>7s} '
                  f'occ={_fmt(s["occluded_ge1_share"]):>5s} '
                  f'ego={_fmt(s["ego_amplitude_mps_median"], 1):>5s}', file=stream)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--train-info', default='data/TJ4D/TJ4D_infos_train.pkl')
    ap.add_argument('--val-info', default='data/TJ4D/TJ4D_infos_val.pkl')
    ap.add_argument('--data-root', default='data/TJ4D')
    ap.add_argument('--stride', type=int, default=1,
                    help='frame stride per split (1 = every frame)')
    ap.add_argument('--identity-frames', type=int, default=3)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    train_infos = coverage.load_infos(args.train_info)
    val_infos = coverage.load_infos(args.val_info)
    result = build(train_infos, val_infos, args.data_root, args.stride,
                   args.identity_frames)
    result['inputs'] = {
        'train_info': os.path.abspath(args.train_info),
        'val_info': os.path.abspath(args.val_info),
        'data_root': os.path.abspath(args.data_root),
        'n_train_frames': len(train_infos), 'n_val_frames': len(val_infos),
    }
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print_summary(result)
    print(f'[json] {os.path.abspath(args.output)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
