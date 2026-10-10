#!/usr/bin/env python3
"""Candidate A ep6 verdict assembler: S1 mechanism gates + early window.

READ-ONLY, CPU-only. Run it after the ep6 watchdog has finished ("probes
done" in watch_semA_ep6.log). It collects the pre-registered S1 gate readings
from the probe JSONs, recomputes nothing that the probes already decided,
prints the early AP window for context, and states the fork the pre-registration
demands. The decision itself stays a manual step; this script changes nothing
and launches nothing.

Pre-registration: docs/rssm_candidate_A_prereg_20261010.md section 5 (S1).

  M1  val thr50 >= 0.30 and hit@1 >= 0.60
  M2  z-exclusivity: BOTH control arms (h and feat) >= 30% worse than z
  M3  residual toggle worsens positive-cell class CE on >= 8/10 clips
  M4  own_hit and donor_transfer >= chance + 20 pt

Two readings this script deliberately does NOT turn into a pass/fail:

  * hit1_excess_over_chance / hit1_chance -- recorded because M1's 0.60 bar
    was calibrated on route-C's 16x16 grid while A supervises 216x248 (209x
    the cells). If M1 misses only on the raw full-res rate while thr50 and
    the excess-over-chance reading look healthy, the pre-registered outcome is
    INSUFFICIENT EVIDENCE, not a mechanism refutation: add a full-res matched
    control before any verdict.
  * the ep6 AP window -- reported with its true span and n, explicitly not the
    S2 judgement (S2 is ep12-16).

Usage:
  python3 me_rssm/sanity/semA_ep6_verdict.py \
    [--run /data/lurui/work_dirs/fgfull_N4_no2d_igdr_2x4_24e_semA_seed0]
"""

import argparse
import glob
import json
import os
import statistics

METRIC = 'pts_bbox/KITTI/Overall_3D_moderate'
CLASS_KEYS = {
    'Car_strict': 'pts_bbox/KITTI/Car_3D_moderate_strict',
    'Truck_strict': 'pts_bbox/KITTI/Truck_3D_moderate_strict',
    'Ped_loose': 'pts_bbox/KITTI/Pedestrian_3D_moderate_loose',
    'Cyc_loose': 'pts_bbox/KITTI/Cyclist_3D_moderate_loose',
}
BASELINE = {'Overall': 39.2809, 'Car_strict': 47.8399, 'Truck_strict': 29.7246,
            'Ped_loose': 30.9277, 'Cyc_loose': 48.6313}
DEFAULT_RUN = ('/data/lurui/work_dirs/'
               'fgfull_N4_no2d_igdr_2x4_24e_semA_seed0')
GATE_JSON = 'semreadout_ep6_gate.json'


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def val_by_epoch(run_dir):
    out = {}
    for path in sorted(glob.glob(os.path.join(run_dir, '*.log.json'))):
        with open(path, encoding='utf-8') as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get('mode') == 'val' and rec.get('epoch') is not None:
                    out[int(rec['epoch'])] = rec
    return out


def window(by_epoch, key, lo, hi):
    vals = [by_epoch[e][key] for e in range(lo, hi + 1)
            if e in by_epoch and key in by_epoch[e]]
    return (statistics.mean(vals), len(vals)) if vals else (None, 0)


def window_labelled(by_epoch, key, lo, hi):
    """(mean, n, actual_span) -- span must be reported, never implied."""
    eps = [e for e in range(lo, hi + 1)
           if e in by_epoch and key in by_epoch[e]]
    if not eps:
        return None, 0, None
    return statistics.mean([by_epoch[e][key] for e in eps]), len(eps), \
        f'ep{eps[0]}-{eps[-1]}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default=DEFAULT_RUN)
    ap.add_argument('--window', default='1-6',
                    help='early context window LO-HI (default 1-6)')
    args = ap.parse_args()
    run = args.run
    lo, hi = (int(x) for x in args.window.split('-'))

    print(f'run: {run}')
    by_epoch = val_by_epoch(run)
    eps = sorted(by_epoch)
    if eps:
        print(f'val epochs present: {eps[0]}..{eps[-1]} ({len(eps)} rows)')
    else:
        print('no val records yet -> S1 NOT EVALUABLE')

    gate = load_json(os.path.join(run, GATE_JSON))
    print()
    print('=' * 72)
    print('S1 MECHANISM GATES (pre-registered section 5, read from '
          f'{GATE_JSON})')
    print('=' * 72)
    if gate is None:
        print(f'  MISSING {GATE_JSON} -- the ep6 watchdog has not finished '
              'the battery; S1 NOT EVALUABLE')
    else:
        gates = gate.get('gates', {})
        m1 = gate.get('M1_sem_head', {})
        m2 = gate.get('M2_exclusivity', {})
        m3 = gate.get('M3_readout_toggle', {})
        m4 = gate.get('M4_class_information', {})

        # --- M1 ---------------------------------------------------------
        print(f"M1 semantic head : {'PASS' if gates.get('M1') else 'FAIL'}"
              f"  thr50={m1.get('mean_thr50')} (bar >=0.30)"
              f"  hit@1={m1.get('mean_hit1')} (bar >=0.60)")
        clips = m1.get('per_clip') or []
        if clips:
            ch = [c.get('hit1_chance') for c in clips
                  if c.get('hit1_chance') is not None]
            ex = [c.get('hit1_excess_over_chance') for c in clips
                  if c.get('hit1_excess_over_chance') is not None]
            if ch:
                print(f'   [supplement, NOT gated] hit1_chance mean='
                      f'{statistics.mean(ch):.6f}  '
                      f'excess_over_chance mean={statistics.mean(ex):.4f}')
                print('   NOTE: bars were calibrated on route-C 16x16; A runs '
                      '216x248. If M1 misses on the raw full-res rate while '
                      'thr50/the excess reading are healthy, the '
                      'pre-registered outcome is INSUFFICIENT EVIDENCE, not a '
                      'mechanism refutation -- add a full-res matched control.')

        # --- M2 ---------------------------------------------------------
        print(f"M2 z-exclusivity: {'PASS' if gates.get('M2') else 'FAIL'}"
              f"  h_relative={m2.get('h_relative')}"
              f"  feat_relative={m2.get('feat_relative')}  (both must be <=-0.30)")
        print(f'   z thr50={m2.get("z_mean_thr50")} '
              f'h={m2.get("h_mean_thr50")} feat={m2.get("feat_mean_thr50")}')
        if not gates.get('M2'):
            print('   -> M2 failing means the "probabilistic state" claim is '
                  'void (the semantics came from h/observation), per section 7.')

        # --- M3 ---------------------------------------------------------
        print(f"M3 readout      : {'PASS' if gates.get('M3') else 'FAIL'}"
              f"  wins={m3.get('wins')}/{m3.get('n')} (need >=8/10)")

        # --- M4 ---------------------------------------------------------
        print(f"M4 class info   : {'PASS' if gates.get('M4') else 'FAIL'}"
              f"  own_hit={m4.get('mean_own_hit')} "
              f"donor_transfer={m4.get('mean_donor_transfer')} "
              f"(bar {m4.get('bar')})")
        print(f"   head_swap_to_donor={m4.get('mean_head_swap_to_donor')} "
              f"changed_frac={m4.get('mean_changed_frac')} (context)")

        all_pass = all(gates.get(k) is True for k in ('M1', 'M2', 'M3', 'M4'))
        print()
        print(f'S1 AGGREGATE: {"ALL PASS" if all_pass else "NOT ALL PASS"}')

    print()
    print('=' * 72)
    print(f'EARLY AP WINDOW ep{lo}-{hi} (context only -- NOT the S2 judgement)')
    print('=' * 72)
    if not eps:
        print('  no val rows')
    else:
        mean, n, span = window_labelled(by_epoch, METRIC, lo, hi)
        print(f'  Overall_3D_moderate: mean={mean} (span {span}, n={n}) '
              f'| baseline {BASELINE["Overall"]}')
        peak_ep = max((e for e in eps if METRIC in by_epoch[e]),
                      key=lambda e: by_epoch[e][METRIC], default=None)
        if peak_ep is not None:
            print(f'  all-epoch peak: {by_epoch[peak_ep][METRIC]:.4f} '
                  f'@ ep{peak_ep} (weight-independent, per AGENTS rule 5)')
        for name, key in CLASS_KEYS.items():
            mean_c, n_c, _ = window_labelled(by_epoch, key, lo, hi)
            d = None if mean_c is None else mean_c - BASELINE[name]
            print(f'  {name:13s} mean={mean_c} n={n_c} '
                  f'baseline={BASELINE[name]} delta={d}')
        print('  Reminder: S2 is the ep12-16 gate (Overall >= 40.00; non-'
              'inferiority Car >= 47.34 / Truck >= 29.22 / Ped >= 29.93 / '
              'Cyc >= 47.63; hard fail < 38.78) and needs user approval '
              'before any ep6->ep16 continuation.')

    print()
    print('=' * 72)
    print('FORK (pre-registered section 7, manual step)')
    print('=' * 72)
    print('  S1 all pass -> request user approval of the section 5 S2 numbers '
          'BEFORE continuing ep6->ep16.')
    print('  S1 any fail -> archive as a negative result with the measured '
          'reason; if M2 failed, the probabilistic-state claim is void.')
    print('  Any Ped/Cyc <= non-inferiority line -> stop, record as the '
          "family side-effect's sixth reproduction.")
    print('  Never: weight/KL sweeps or an ep24 rescue.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
