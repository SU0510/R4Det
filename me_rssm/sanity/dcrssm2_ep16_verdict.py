#!/usr/bin/env python3
"""DC-RSSM-2 ep16 verdict: AP gate + mechanism three-way classification.

READ-ONLY, CPU-only. Run it after the ep16 watchdog has finished
("probes done" in watch_seed0_ep16.log) and after the official eval has
produced the ep12-16 val rows. It does the arithmetic the pre-registered
gates require and prints one consolidated verdict block; the decision
itself stays a manual step (this script changes nothing).

Pre-registered gates (written BEFORE any 24e data; config header + the
docs/training_runs_full.md §68.9i record):

  AP gate (all four must hold for GO):
    A1  ep12-16 window mean of pts_bbox/KITTI/Overall_3D_moderate
        >= 39.2809   (same-seed no2d_igdr baseline, §68.8d/§68.9h)
    A2  >= 38.8786   (MCI-v2 seed0 39.3786 - 0.5: must at least tie the
        previous generation)
    A3  Pedestrian loose paired delta >= -1.5 vs the same-seed baseline
        window (the L3 asset must not regress the class it repaired)
    A4  Car strict and Truck strict no regression vs the baseline window

  Gate keys are pinned here on purpose (decided 2026-10-09, with only
  ep1-2 of DC-RSSM-2 on disk): Car_3D_moderate_STRICT and
  Truck_3D_moderate_STRICT, because that is the pair used by the family's
  comparison table and by §68.8d's per-class window means -- the numbers
  §68.9h quotes as Car 73.75 / Truck 52.02 are the LOOSE variants of the
  same run (Car loose ~73, Truck loose ~52), so mixing the families would
  compare different keys. Cyclist is reported as context only: it was not
  part of the pre-registered DC-RSSM-2 gate (the family lost 4-5 points
  there in both MCI-v2 and DC-RSSM-1, which is why it is documented, but
  adding it to the gate now would be a post-hoc criterion).

  Mechanism gate (three-way, from ego_alignment_toggle_ep16.json, written
  BEFORE any 24e data):
    UNEXERCISED   mean applied displacement |warp_scale|*dt*|v_ego|
                  < INERT_MIN_DISP_M -> the warp was never exercised, so
                  the AP verdict says nothing about the mechanism;
    MECHANISM PASS / FAIL  the learned warp lowers the static-cell
                  cross-frame residual in >= 8/10 clips (sign test);
                  forced-1.0 reported alongside as the magnitude control.

Baseline reference window values (docs/training_runs_full.md):
  ep12-16 (n=5) Overall_3D 39.2809 | Car strict 47.8399 |
  Cyclist loose 48.6313 | Ped loose 30.9277 | Truck strict 29.7246
  (line 5999 table, column "no2d_igdr（RSSM）"; Ped/Car/Truck cross-checked
  against §68.8d's per-class window means).

Usage:
  python3 me_rssm/sanity/dcrssm2_ep16_verdict.py \
    [--run /data/lurui/work_dirs/fgfull_N4_no2d_igdr_dcrssm2_2x4_24e_seed0]
"""

import argparse
import glob
import json
import os
import statistics
import sys

METRIC = 'pts_bbox/KITTI/Overall_3D_moderate'
WINDOW = (12, 16)
BASELINE_OVERALL = 39.2809
V2_MINUS_05 = 38.8786
PED_MIN_DELTA = -1.5
# Gate keys (see docstring): the family table's strict pair for Car/Truck
# plus Ped loose (the class L3 repaired).
GATE_CLASSES = {
    # key: (baseline window mean, coarser rule, human name)
    'pts_bbox/KITTI/Car_3D_moderate_strict': (47.8399, '>=0', 'Car strict'),
    'pts_bbox/KITTI/Pedestrian_3D_moderate_loose': (
        30.9277, '>=-1.5', 'Ped loose'),
    'pts_bbox/KITTI/Truck_3D_moderate_strict': (
        29.7246, '>=0', 'Truck strict'),
}
# Reported for context only, never gated (see docstring).
INFO_CLASSES = {
    'pts_bbox/KITTI/Cyclist_3D_moderate_loose': (48.6313, 'Cyc loose'),
    'pts_bbox/KITTI/Car_3D_moderate_loose': (None, 'Car loose'),
    'pts_bbox/KITTI/Truck_3D_moderate_loose': (None, 'Truck loose'),
    'pts_bbox/KITTI/Pedestrian_3D_moderate_strict': (None, 'Ped strict'),
}
INERT_MIN_DISP_M = 1e-3
SIGN_TEST_MIN = 8
DEFAULT_RUN = ('/data/lurui/work_dirs/'
               'fgfull_N4_no2d_igdr_dcrssm2_2x4_24e_seed0')


def val_by_epoch(run_dir):
    """epoch -> val record (last one wins), across all log.json files."""
    out = {}
    for path in sorted(glob.glob(os.path.join(run_dir, '*.log.json'))):
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get('mode') != 'val':
                    continue
                ep = rec.get('epoch')
                if ep is None:
                    continue
                out[int(ep)] = rec
    return out


def window(by_epoch, key, lo=None, hi=None):
    lo = WINDOW[0] if lo is None else lo
    hi = WINDOW[1] if hi is None else hi
    vals = [by_epoch[e][key] for e in range(lo, hi + 1)
            if e in by_epoch and key in by_epoch[e]]
    return (statistics.mean(vals), len(vals)) if vals else (None, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default=DEFAULT_RUN)
    args = ap.parse_args()
    run = args.run

    print(f'run: {run}')
    by_epoch = val_by_epoch(run)
    if not by_epoch:
        print('no val records yet -> NOT EVALUABLE')
        return 2
    eps = sorted(by_epoch)
    print(f'val epochs present: {eps[0]}..{eps[-1]} ({len(eps)} rows)')

    # ---- full curve (uniform reporting: every epoch, never a summary) --
    print('\ncurve (Overall_3D_moderate), full run:')
    for e in eps:
        v = by_epoch[e].get(METRIC)
        print(f'  ep{e:>2}: {v:.4f}' if v is not None else f'  ep{e:>2}: -')

    peaks = [(by_epoch[e][METRIC], e) for e in eps if METRIC in by_epoch[e]]
    if peaks:
        peak, peak_ep = max(peaks)
        print(f'peak: {peak:.4f} @ ep{peak_ep}')
        saved = [(by_epoch[e][METRIC], e) for e in eps
                 if METRIC in by_epoch[e] and e % 2 == 0]
        if saved:
            bs, bs_ep = max(saved)
            print(f'best saved (even epochs): {bs:.4f} @ ep{bs_ep}')

    wmean, n = window(by_epoch, METRIC)
    if wmean is None:
        print(f'\nep{WINDOW[0]}-{WINDOW[1]} window incomplete '
              f'({n}/5 rows) -> AP gate NOT EVALUABLE yet')
        return 2
    print(f'\nAP gate window ep{WINDOW[0]}-{WINDOW[1]} (n={n}): '
          f'{wmean:.4f}')
    ok = True
    ok &= wmean >= BASELINE_OVERALL
    print(f'  A1 >= {BASELINE_OVERALL:.4f} (baseline seed0): '
          f'{wmean - BASELINE_OVERALL:+.4f} -> '
          f'{"PASS" if wmean >= BASELINE_OVERALL else "FAIL"}')
    ok &= wmean >= V2_MINUS_05
    print(f'  A2 >= {V2_MINUS_05:.4f} (MCI-v2 - 0.5): '
          f'{wmean - V2_MINUS_05:+.4f} -> '
          f'{"PASS" if wmean >= V2_MINUS_05 else "FAIL"}')
    for key, (base, rule, name) in GATE_CLASSES.items():
        cmean, cn = window(by_epoch, key)
        if cmean is None:
            print(f'  {name}: MISSING -> not evaluable')
            ok = False
            continue
        delta = cmean - base
        good = delta >= PED_MIN_DELTA if rule == '>=-1.5' else delta >= 0.0
        ok &= good
        print(f'  {rule:>7} {name:10s} (n={cn}): {cmean:.4f} vs {base:.4f}'
              f' -> {delta:+.4f} {"PASS" if good else "FAIL"}')
    print('  context (not gated):')
    for key, (base, name) in INFO_CLASSES.items():
        cmean, cn = window(by_epoch, key)
        if cmean is None:
            print(f'    {name:11s}: MISSING')
            continue
        if base is None:
            print(f'    {name:11s} (n={cn}): {cmean:.4f}')
        else:
            print(f'    {name:11s} (n={cn}): {cmean:.4f} vs {base:.4f}'
                  f' -> {cmean - base:+.4f}')

    # ---- mechanism gate ------------------------------------------------
    print('\nmechanism gate (pre-registered, from '
          'ego_alignment_toggle_ep16.json):')
    tpath = os.path.join(run, 'ego_alignment_toggle_ep16.json')
    if not os.path.exists(tpath):
        print(f'  MISSING {tpath} -> mechanism verdict NOT AVAILABLE')
        mech = 'MISSING'
    else:
        with open(tpath, encoding='utf-8') as f:
            tog = json.load(f)
        ws = tog.get('warp_scale')
        print(f'  warp_scale (learned) = {ws!r}  dt = {tog.get("dt")!r}')
        print(f'  clips with all residuals = {tog.get("n_learned")}')
        print(f'  learned vs 0 wins   = {tog.get("wins_learned")}'
              f'/{tog.get("n_learned")} (need >= {SIGN_TEST_MIN})')
        print(f'  forced-1.0 vs 0 wins= {tog.get("wins_forced1")}'
              f'/{tog.get("n_forced1")}  (magnitude control)')
        print(f'  applied displacement = {tog.get("mean_disp_learned_m")!r}'
              f' m/frame; needed = {tog.get("mean_need_m")!r} m/frame')
        if tog.get('mean_need_m'):
            frac = (tog['mean_disp_learned_m'] / tog['mean_need_m']
                    if tog.get('mean_disp_learned_m') is not None else None)
            print(f'  exercised fraction  = {100 * frac:.2f}%' if frac
                  else '  exercised fraction  = -')
        print(f'  inert = {tog.get("inert")!r} '
              f'(line {INERT_MIN_DISP_M:.0e} m)')
        mech = tog.get('verdict', 'MISSING')
        print(f'  VERDICT: {mech}')
        print(f'  CONTROL: {tog.get("verdict_forced1")}')
        if mech.startswith('UNEXERCISED'):
            print('  -> the AP verdict cannot be attributed to the warp; '
                  'record as unexercised, not falsified')

    print('\n=== CONSOLIDATED ===')
    print(f'AP gate : {"GO" if ok else "REJECT"} (window {wmean:.4f})')
    print(f'MECH    : {mech}')
    if ok:
        print('next: write GO into <run>/gate_decision.txt and run '
              'me_rssm/sanity/continue_dcrssm2_ep16.sh (ep16 -> ep24)')
    else:
        print('next: stop the DC-RSSM direction, record the mechanism '
              'attribution and pivot to the mechanism-analysis paper '
              '(docs/rssm_paper_directions_20261008.md)')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
