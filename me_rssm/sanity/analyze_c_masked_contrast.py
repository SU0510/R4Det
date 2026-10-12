#!/usr/bin/env python3
"""Candidate C, causal half: PAIRED masked-vs-baseline recall contrast.

Reads the per-GT records produced by ``probe_c_failure_enrichment.py
--records-out`` for two arms -- the frozen baseline dump and the controlled-
masking dump -- and computes the pre-registered contrast on the SAME ground-
truth objects (paired), never on two different populations.

The rule it applies is frozen in
``docs/rssm_c_failure_enrichment_contract_20261012.md`` §8:

  population : every GT whose FROZEN coverage group is not ``zero``
  Δ_mask     : recall(masked covered GTs | intervention) - recall(same GTs | baseline)
  Δ_vs_zero  : recall(masked covered GTs | intervention) - recall(zero group | baseline)
  supported  : identity control bit-identical AND Δ_mask <= -3.0 points AND
               >= 2 range bins with Δ_mask <= -3.0 (n_gt >= 30) AND
               scene-clustered 95% CI of Δ_mask upper bound < 0.

Description only (never a decision input): per-class and per-range-bin tables,
the TP->FN / FN->TP flip counts, and Δ_vs_zero.
"""

import argparse
import json
import os
import sys

import numpy as np

BOOTSTRAP_ITERS = 2000
BOOTSTRAP_SEED = 0
DELTA_THRESHOLD = -3.0
MIN_BIN_GT = 30
REQUIRED_BINS = 2
BIN_ORDER = ('le50', '50to70', 'gt70')
CLASS_ORDER = ('Pedestrian', 'Cyclist', 'Car', 'Truck')
KEY_FIELDS = ('image_idx', 'gt_index', 'class')


def key_of(record):
    return tuple(record[f] for f in KEY_FIELDS)


def join_records(baseline, masked, population_group='zero'):
    """Pair the two arms key-for-key; refuse to analyse anything else."""
    base_by_key = {key_of(r): r for r in baseline}
    mask_by_key = {key_of(r): r for r in masked}
    if len(base_by_key) != len(baseline):
        raise AssertionError('duplicate keys in the baseline records')
    if len(mask_by_key) != len(masked):
        raise AssertionError('duplicate keys in the masked records')
    missing = sorted(set(base_by_key) - set(mask_by_key))
    extra = sorted(set(mask_by_key) - set(base_by_key))
    if missing or extra:
        raise AssertionError(
            f'arms do not cover the same GTs: {len(missing)} missing, '
            f'{len(extra)} extra (first: {missing[:3]} / {extra[:3]})')
    rows = []
    for key in sorted(base_by_key):
        b, m = base_by_key[key], mask_by_key[key]
        for field in ('coverage_group', 'range_bin', 'class', 'scene'):
            if b[field] != m[field]:
                raise AssertionError(f'frozen label drift at {key} on {field}: '
                                     f'{b[field]!r} vs {m[field]!r}')
        rows.append({
            'image_idx': b['image_idx'], 'gt_index': b['gt_index'],
            'class': b['class'], 'scene': b['scene'],
            'coverage_group': b['coverage_group'],
            'zero_scope': b['zero_scope'], 'amp_group': b['amp_group'],
            'range_bin': b['range_bin'], 'range_m': b['range_m'],
            'found_baseline': bool(b['found']),
            'found_masked': bool(m['found']),
            'is_covered': b['coverage_group'] != population_group,
        })
    return rows


def recall(rows, field):
    n = len(rows)
    if n == 0:
        return {'n_gt': 0, 'n_found': 0, 'recall': None}
    found = sum(1 for r in rows if r[field])
    return {'n_gt': n, 'n_found': found, 'recall': found / n}


def paired_table(rows):
    """baseline vs intervention on the same rows, in points of recall."""
    base = recall(rows, 'found_baseline')
    mask = recall(rows, 'found_masked')
    delta = (None if base['recall'] is None or mask['recall'] is None
             else 100.0 * (mask['recall'] - base['recall']))
    lost = sum(1 for r in rows if r['found_baseline'] and not r['found_masked'])
    gained = sum(1 for r in rows if not r['found_baseline'] and r['found_masked'])
    return {
        'n_gt': len(rows),
        'baseline': base,
        'masked': mask,
        'delta_mask_points': delta,
        'flips': {
            'tp_to_fn (lost by masking)': lost,
            'fn_to_tp (gained by masking)': gained,
            'tp_to_tp': sum(1 for r in rows
                            if r['found_baseline'] and r['found_masked']),
            'fn_to_fn': sum(1 for r in rows
                            if not r['found_baseline'] and not r['found_masked']),
        },
    }


def unpaired_table(rows_a, rows_b):
    a, b = recall(rows_a, 'found_baseline'), recall(rows_b, 'found_masked')
    delta = (None if a['recall'] is None or b['recall'] is None
             else 100.0 * (b['recall'] - a['recall']))
    return {'left': a, 'right': b, 'delta_points_right_minus_left': delta}


def scene_clustered_bootstrap(rows, iters=BOOTSTRAP_ITERS, seed=BOOTSTRAP_SEED):
    """Resample the 11 val scene blocks, recompute the PAIRED delta each time."""
    scenes = sorted({r['scene'] for r in rows})
    by_scene = {s: [r for r in rows if r['scene'] == s] for s in scenes}
    rng = np.random.RandomState(seed)
    deltas = []
    for _ in range(iters):
        picked = rng.randint(0, len(scenes), len(scenes))
        sample = []
        for k in picked:
            sample.extend(by_scene[scenes[k]])
        delta = paired_table(sample)['delta_mask_points']
        if delta is not None:
            deltas.append(delta)
    if not deltas:
        return {'n_resamples': 0, 'ci95': [None, None],
                'excludes_zero': False, 'cluster': 'scene'}
    lo, hi = float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))
    return {
        'n_resamples': len(deltas),
        'ci95': [lo, hi],
        'median': float(np.median(deltas)),
        'excludes_zero': bool(hi < 0 or lo > 0),
        'direction': ('negative' if hi < 0 else ('positive' if lo > 0
                                                 else 'spans zero')),
        'cluster': f'scene ({len(scenes)} val blocks)',
    }


def verdict(pooled, bins, ci, identity_identical):
    qualifying = [name for name in BIN_ORDER
                  if name in bins
                  and bins[name]['n_gt'] >= MIN_BIN_GT
                  and bins[name]['delta_mask_points'] is not None
                  and bins[name]['delta_mask_points'] <= DELTA_THRESHOLD]
    clauses = {
        'identity_control_bit_identical': bool(identity_identical),
        'delta_mask_at_most_threshold': bool(
            pooled['delta_mask_points'] is not None
            and pooled['delta_mask_points'] <= DELTA_THRESHOLD),
        'at_least_two_qualifying_range_bins': len(qualifying) >= REQUIRED_BINS,
        'ci95_upper_bound_below_zero': bool(ci.get('excludes_zero')
                                            and ci['ci95'][1] is not None
                                            and ci['ci95'][1] < 0),
    }
    return {
        'causal_supported': all(clauses.values()),
        'clauses': clauses,
        'range_bins_qualifying': qualifying,
        'range_bins_required': REQUIRED_BINS,
        'delta_threshold': DELTA_THRESHOLD,
        'min_bin_gt': MIN_BIN_GT,
        'rule': ('identity bit-identical AND Δ_mask <= -3.0 points AND >= 2 range '
                 'bins (n_gt >= 30) with Δ_mask <= -3.0 AND scene-clustered 95% '
                 'CI upper bound < 0'),
    }


def analyse(rows, identity_identical):
    covered = [r for r in rows if r['is_covered']]
    zero = [r for r in rows if not r['is_covered']]
    pooled = paired_table(covered)
    bins = {name: paired_table([r for r in covered if r['range_bin'] == name])
            for name in BIN_ORDER}
    classes = {cls: paired_table([r for r in covered if r['class'] == cls])
               for cls in CLASS_ORDER}
    ci = scene_clustered_bootstrap(covered)
    return {
        'primary_population': 'frozen coverage group != zero (the objects that '
                              'currently have radar returns)',
        'pooled': pooled,
        'by_range_bin': bins,
        'by_class': classes,
        'delta_vs_zero_group': unpaired_table(zero, covered),
        'bootstrap_ci95_delta_mask': ci,
        'population_sizes': {
            'masked_covered': len(covered),
            'zero': len(zero),
            'all': len(rows),
        },
        'verdict': verdict(pooled, bins, ci, identity_identical),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-records', required=True)
    parser.add_argument('--masked-records', required=True)
    parser.add_argument('--identity-compare', required=True,
                        help='compare json written by tool_c_masked_inference.py '
                             '--compare-out (the transparency control)')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    baseline = json.load(open(args.baseline_records, encoding='utf-8'))
    masked = json.load(open(args.masked_records, encoding='utf-8'))
    identity = json.load(open(args.identity_compare, encoding='utf-8'))
    if not identity.get('identical'):
        raise SystemExit('identity control is NOT bit-identical: the masking '
                         'numbers are inadmissible (contract §8)')

    rows = join_records(baseline, masked)
    result = analyse(rows, identity_identical=bool(identity['identical']))
    result['inputs'] = {
        'baseline_records': os.path.abspath(args.baseline_records),
        'masked_records': os.path.abspath(args.masked_records),
        'identity_compare': os.path.abspath(args.identity_compare),
        'identity_compare_sha256_left': identity.get('sha256_left'),
        'identity_compare_sha256_right': identity.get('sha256_right'),
        'n_frames_compared': identity.get('n_frames'),
    }
    with open(args.output, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)

    pooled = result['pooled']
    print(f"population: covered {result['population_sizes']['masked_covered']} "
          f"vs zero {result['population_sizes']['zero']}")
    print(f"Δ_mask = {pooled['delta_mask_points']:.2f} points "
          f"(baseline recall {pooled['baseline']['recall']:.4f} -> masked "
          f"{pooled['masked']['recall']:.4f}); flips {pooled['flips']}")
    for name in BIN_ORDER:
        table = result['by_range_bin'][name]
        print(f"  bin {name}: n={table['n_gt']} Δ={table['delta_mask_points']}")
    for cls in CLASS_ORDER:
        print(f"  {cls}: n={result['by_class'][cls]['n_gt']} "
              f"Δ={result['by_class'][cls]['delta_mask_points']}")
    print(f"bootstrap 95% CI: {result['bootstrap_ci95_delta_mask']['ci95']}")
    print(f"verdict: {json.dumps(result['verdict'], sort_keys=True)}")
    print(f'[json] {os.path.abspath(args.output)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
