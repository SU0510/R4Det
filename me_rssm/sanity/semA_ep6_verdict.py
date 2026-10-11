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
import datetime
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

# Every artifact the ep6 battery is supposed to leave behind, with the schema
# keys that must be present. The watchdog is a shell `&&`-free sequence of
# probe calls, so a probe that dies (OOM, CUDA error, wrong family) writes no
# JSON at all while the log still ends with "probes done". Checking the bundle
# is therefore part of the verdict: an incomplete bundle is NOT EVALUABLE, and
# must never be read as a failed gate.
EVIDENCE = (
    ('S1 gate (M1-M4)', GATE_JSON,
     ('gates', 'M1_sem_head', 'M2_exclusivity', 'M3_readout_toggle',
      'M4_class_information', 'indices', 'checkpoint',
      'input_fingerprints')),
    ('z-utilization w3', 'z_util_std_ep6_w3.json',
     ('indices', 'per_sample')),
    ('z-utilization w100', 'z_util_std_ep6_w100.json',
     ('indices', 'per_sample')),
    ('z-utilization w1800', 'z_util_std_ep6_w1800.json',
     ('indices', 'per_sample')),
    ('posterior source', 'post_source_std_ep6_stride700.json',
     ('indices', 'fusion_class', 'mu_q', 'h_t', 'e_pooled')),
)

# The M1-M4 gates must all be present for an aggregate verdict to mean
# anything; a probe that reports 3 of 4 gates is not a pass.
REQUIRED_GATES = ('M1', 'M2', 'M3', 'M4')

# --- A2 pre-registered gates (docs/rssm_candidate_A2_prereg_20261011.md §5) --
# A2 re-uses A's M1-M4, but adds scale-free / localisation gates that A's probe
# never produced. They live in the rank probe's own JSON, so this assembler has
# to evaluate them; otherwise an A2 run whose M1'/M1-loc failed would print
# "ALL PASS" off M1-M4 alone and the arm would be promoted on an incomplete
# gate set. Detected by the presence of the rank artifact, so A's archived
# bundle (which has no rank JSON) keeps its exact original verdict.
A2_RANK_VAL = 'semA_rank_val_ep6.json'
A2_RANK_TRAIN = 'semA_rank_train_ep6.json'
A2_M4FLOOR = 'semA_m4floor_ep6.json'
A2_FROZEN_N = 20
A2_BAR_AUC = 0.80
A2_BAR_AUC_MARGIN = 0.25
A2_BAR_MAXGT_CLIPS = 12          # of A2_FROZEN_N
A2_BAR_RECALL_TOPK_MEDIAN = 0.25


def _fmt(value, digits=4):
    """Fixed-width float for the printed gate lines; 'None' stays explicit."""
    return 'None' if value is None else f'{value:.{digits}f}'


def _finite(value):
    return isinstance(value, (int, float)) and value == value


def _mean(rows, key):
    vals = [r[key] for r in rows if _finite(r.get(key))]
    return statistics.mean(vals) if vals else None


def _median(rows, key):
    vals = [r[key] for r in rows if _finite(r.get(key))]
    return statistics.median(vals) if vals else None


def arm_of_bundle(run_dir):
    """'A2S' / 'A2R' / None -- which generation produced this ep6 bundle.

    The discriminator is the CONFIG PATH, not the mere presence of an A2-only
    artifact: A's archived bundle also carries a ``semA_rank_val_ep6.json``
    (the rank probe was run against it after the fact, with a different clip
    contract), so keying on the file would re-open A's closed verdict as
    "NOT EVALUABLE". Only the A2 configs turn the added gates on.
    """
    gate = load_json(os.path.join(run_dir, GATE_JSON)) or {}
    cfg = str((gate.get('input_fingerprints') or {}).get('config_path')
              or gate.get('config') or '')
    if not cfg:
        rank = load_json(os.path.join(run_dir, A2_RANK_VAL)) or {}
        cfg = str(rank.get('config') or '')
    if 'semA2S' in cfg:
        return 'A2S'
    if 'semA2R' in cfg:
        return 'A2R'
    return None


def a2_readings(run_dir, arm=None):
    """Evaluate A2's added gates from the rank artifact. None when not A2.

    Returns {'applicable', 'gates', 'readings', 'blockers'} or None when this
    bundle is not an A2 arm (see ``arm_of_bundle``). ``blockers`` are reasons
    the readings cannot be judged at all (wrong frozen n, index mismatch);
    those must surface as NOT EVALUABLE, never as a silent pass.
    """
    if arm is None:
        arm = arm_of_bundle(run_dir)
    if arm is None:
        return None
    val = load_json(os.path.join(run_dir, A2_RANK_VAL))
    if val is None:
        return {'applicable': True, 'arm': arm, 'gates': {'M1p': None},
                'readings': {'n_clips': 0},
                'blockers': [f'{A2_RANK_VAL} is missing; the watchdog never '
                             f'produced the M1\'/M1-loc readings']}
    rows = val.get('per_clip') or []
    blockers = []
    if len(rows) != A2_FROZEN_N:
        blockers.append(f'rank val n={len(rows)}, frozen contract is '
                        f'{A2_FROZEN_N}')
    auc = _mean(rows, 'auc_mean_over_occupied_channels')
    shift = _mean(rows, 'gt_shuffle_auc')
    auc_margin = (None if auc is None or shift is None else auc - shift)
    n_maxgt = sum(1 for r in rows
                  if _finite(r.get('max_gt_prob')) and r['max_gt_prob'] >= 0.5)
    recall_median = _median(rows, 'recall_at_top_k_mean')
    readings = {
        'n_clips': len(rows),
        'auc': auc, 'auc_gt_shuffle': shift, 'auc_margin': auc_margin,
        'n_max_gt_prob_ge_0.5': n_maxgt,
        'recall_at_top_k_median': recall_median,
    }
    gates = {
        'M1p': (auc is not None and auc_margin is not None
                and auc >= A2_BAR_AUC and auc_margin >= A2_BAR_AUC_MARGIN),
    }
    # M1-loc is A2-S only (pre-registration section 5), so a stray rank
    # artifact on the A2-R arm cannot quietly import a supervision-side gate.
    if arm == 'A2S':
        gates['M1loc'] = bool(
            n_maxgt >= A2_BAR_MAXGT_CLIPS
            and recall_median is not None
            and recall_median >= A2_BAR_RECALL_TOPK_MEDIAN)
    if blockers:
        for key in gates:
            gates[key] = None          # NOT MEASURED, never pass/fail
    return {'applicable': True, 'arm': arm, 'gates': gates,
            'readings': readings, 'blockers': blockers}

# Pre-registered M1 grid-extrapolation rule (training record section
# 71.9.3(2)). The hit@1 >= 0.60 bar was calibrated on route-C's 16x16 pooled
# grid (chance ~1.6e-2); candidate A supervises the full-res 216x248 grid
# (chance ~7.5e-5, 209x the cells), where a correct centre in one argmax cell
# is a far stricter event. The pre-registration therefore fixes the outcome in
# advance: a miss on the RAW full-res rate while thr50 AND the
# excess-over-chance reading are healthy is INSUFFICIENT EVIDENCE -- add a
# full-res matched control -- and must NOT be recorded as a mechanism
# refutation. The prereg fixed no numeric excess bar, so the check below uses
# the excess the calibrated bar implies on its own grid and prints it.
M1_BAR_THR50 = 0.30
M1_BAR_HIT1 = 0.60
M1_CALIBRATION_CHANCE = 1.6e-2
M1_EXCESS_BAR = M1_BAR_HIT1 - M1_CALIBRATION_CHANCE


def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def check_evidence(run_dir):
    """(rows, ok) -- does the battery bundle exist and carry its schema?

    Returns one row per expected artifact: (label, filename, status, detail).
    """
    rows = []
    ok = True
    for label, name, keys in EVIDENCE:
        path = os.path.join(run_dir, name)
        if not os.path.exists(path):
            rows.append((label, name, 'MISSING', 'probe wrote no JSON'))
            ok = False
            continue
        try:
            payload = load_json(path)
        except (ValueError, OSError) as exc:
            rows.append((label, name, 'UNREADABLE', str(exc)[:80]))
            ok = False
            continue
        if not isinstance(payload, dict):
            rows.append((label, name, 'MALFORMED', 'top level is not an object'))
            ok = False
            continue
        missing = [k for k in keys if k not in payload]
        if missing:
            rows.append((label, name, 'INCOMPLETE',
                         'missing keys: ' + ', '.join(missing)))
            ok = False
            continue
        if name == GATE_JSON:
            gates = payload.get('gates') or {}
            absent = [g for g in REQUIRED_GATES if g not in gates]
            if absent:
                rows.append((label, name, 'INCOMPLETE',
                             'missing gates: ' + ', '.join(absent)))
                ok = False
                continue
            if not payload.get('indices'):
                rows.append((label, name, 'INCOMPLETE', 'no clips selected'))
                ok = False
                continue
        if name.startswith('z_util'):
            count = len(payload.get('per_sample') or {})
            if count < 2:
                rows.append((label, name, 'INCOMPLETE',
                             f'only {count} per-sample reading(s)'))
                ok = False
                continue
            bad = [k for k, v in (payload.get('per_sample') or {}).items()
                   if not isinstance(v, dict)
                   or 'variants' not in v]
            if bad:
                rows.append((label, name, 'INCOMPLETE',
                             f'{len(bad)} reading(s) lack variants'))
                ok = False
                continue
        rows.append((label, name, 'OK', ''))
    return rows, ok


def m1_excess_mean(m1):
    """Mean hit1_excess_over_chance over the M1 clips, or None."""
    vals = [c.get('hit1_excess_over_chance') for c in (m1.get('per_clip') or [])
            if c.get('hit1_excess_over_chance') is not None]
    return statistics.mean(vals) if vals else None


def m1_grid_limited(mean_thr50, mean_hit1, mean_excess):
    """Is M1's failure attributable solely to the full-res hit@1 scale?

    True only when thr50 cleared its bar and the excess-over-chance reading
    reaches the excess the calibrated bar implies: the head does find object
    centres, but the raw one-cell-argmax rate is not comparable across a grid
    209x denser than the one the bar came from. Per section 71.9.3(2) that
    outcome is INSUFFICIENT EVIDENCE, not a refutation.
    """
    if mean_thr50 is None or mean_hit1 is None or mean_excess is None:
        return False
    if mean_thr50 < M1_BAR_THR50:
        return False
    if mean_hit1 >= M1_BAR_HIT1:
        return False  # it passed on the raw rate; nothing to explain away
    return mean_excess >= M1_EXCESS_BAR


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

    print()
    print('=' * 72)
    print('EVIDENCE BUNDLE COMPLETENESS (a dead probe writes no JSON)')
    print('=' * 72)
    rows, evidence_ok = check_evidence(run)
    for label, name, status, detail in rows:
        line = f'  [{status:10s}] {label:22s} {name}'
        if detail:
            line += f'  <- {detail}'
        print(line)
    if evidence_ok:
        print('  -> all expected artifacts present')
    else:
        print('  -> INCOMPLETE BUNDLE: rerun the missing probe(s) before '
              'reading any gate number as a failure')

    decision = {}
    a2 = None                      # A2 added-gate readings; set below if A2
    gate = load_json(os.path.join(run, GATE_JSON))
    print()
    print('=' * 72)
    print('S1 MECHANISM GATES (pre-registered section 5, read from '
          f'{GATE_JSON})')
    print('=' * 72)
    if not evidence_ok and gate is not None:
        print('  WARNING: the bundle is incomplete; the gate reading below '
              'is provisional until the missing artifacts exist.')
    if gate is None:
        print(f'  MISSING {GATE_JSON} -- the ep6 watchdog has not finished '
              'the battery; S1 NOT EVALUABLE')
    else:
        gates = gate.get('gates', {})
        # A gate key that is absent (partial report) must not be rendered as a
        # plain FAIL: "not measured" and "measured and failed" are different
        # findings, and only one of them is a mechanism refutation.
        def verdict_of(name):
            if name not in gates:
                return 'NOT MEASURED'
            return 'PASS' if gates.get(name) else 'FAIL'

        m1 = gate.get('M1_sem_head', {})
        m2 = gate.get('M2_exclusivity', {})
        m3 = gate.get('M3_readout_toggle', {})
        m4 = gate.get('M4_class_information', {})

        # --- M1 ---------------------------------------------------------
        print(f"M1 semantic head : {verdict_of('M1')}"
              f"  thr50={m1.get('mean_thr50')} (bar >=0.30)"
              f"  hit@1={m1.get('mean_hit1')} (bar >=0.60)")
        clips = m1.get('per_clip') or []
        if clips:
            ch = [c.get('hit1_chance') for c in clips
                  if c.get('hit1_chance') is not None]
            ex = [c.get('hit1_excess_over_chance') for c in clips
                  if c.get('hit1_excess_over_chance') is not None]
            if ch:
                mean_excess = statistics.mean(ex) if ex else None
                print(f'   [supplement, NOT gated] hit1_chance mean='
                      f'{statistics.mean(ch):.6f}  '
                      f'excess_over_chance mean={mean_excess}')
                decision["m1_grid_limited"] = m1_grid_limited(
                    m1.get('mean_thr50'), m1.get('mean_hit1'), mean_excess)
                if decision["m1_grid_limited"]:
                    print(f'   -> PRE-REGISTERED RULE APPLIES (section '
                          f'71.9.3(2)): thr50 cleared {M1_BAR_THR50} and the '
                          f'excess reading {mean_excess:.4f} reaches the '
                          f'{M1_EXCESS_BAR:.4f} the calibrated bar implies. '
                          f'M1\'s miss is on the FULL-RES raw rate only '
                          f'-> INSUFFICIENT EVIDENCE, not a mechanism '
                          f'refutation. Add a full-res matched control '
                          f'before any verdict.')
                else:
                    print('   NOTE: bars were calibrated on route-C 16x16; A '
                          'runs 216x248. If M1 misses on the raw full-res '
                          'rate while thr50/the excess reading are healthy, '
                          'the pre-registered outcome is INSUFFICIENT '
                          'EVIDENCE, not a mechanism refutation -- add a '
                          'full-res matched control.')

        # --- M2 ---------------------------------------------------------
        print(f"M2 z-exclusivity: {verdict_of('M2')}"
              f"  h_relative={m2.get('h_relative')}"
              f"  feat_relative={m2.get('feat_relative')}  (both must be <=-0.30)")
        print(f'   z thr50={m2.get("z_mean_thr50")} '
              f'h={m2.get("h_mean_thr50")} feat={m2.get("feat_mean_thr50")}')
        if not gates.get('M2'):
            print('   -> M2 failing means the "probabilistic state" claim is '
                  'void (the semantics came from h/observation), per section 7.')

        # --- M3 ---------------------------------------------------------
        # The bar is a RATE of the FROZEN window, so the required count scales
        # with n (A: 8/10; A2: 16/20). Printing a hard-coded "/10" here made
        # the A2 verdict look four times stricter than it was.
        m3_need = m3.get('required_wins')
        m3_need_txt = (f'>={m3_need}/{m3.get("frozen_n")}'
                       if m3_need is not None else '>=8/10')
        print(f"M3 readout      : {verdict_of('M3')}"
              f"  wins={m3.get('wins')}/{m3.get('n')} (need {m3_need_txt})")

        # --- M4 ---------------------------------------------------------
        print(f"M4 class info   : {verdict_of('M4')}"
              f"  own_hit={m4.get('mean_own_hit')} "
              f"donor_transfer={m4.get('mean_donor_transfer')} "
              f"(own bar {m4.get('own_hit_bar')}, "
              f"donor bar {m4.get('bar')})")
        if m4.get('pooled_prior_floor') is not None:
            print(f'   pooled prior floor={m4.get("pooled_prior_floor")} '
                  f'(the own_hit bar is floor + 0.20, NOT 1/C + 0.20)')
        print(f"   head_swap_to_donor={m4.get('mean_head_swap_to_donor')} "
              f"changed_frac={m4.get('mean_changed_frac')} (context)")

        fp = gate.get('input_fingerprints') or {}
        if fp.get('checkpoint_sha256'):
            print(f"   checkpoint: {gate.get('checkpoint')}")
            print(f"     sha256={fp['checkpoint_sha256'][:16]}... "
                  f"size={fp.get('checkpoint_size_bytes')} "
                  f"mtime={fp.get('checkpoint_mtime')} "
                  f"({datetime.datetime.utcfromtimestamp(fp['checkpoint_mtime']).strftime('%Y-%m-%d %H:%M:%S')} UTC)")
            print(f"   config sha256={str(fp.get('config_sha256'))[:16]}...")

        all_pass = all(gates.get(k) is True for k in REQUIRED_GATES)
        # The only non-gate exception the pre-registration allows: an M1 miss
        # that is explained by the cross-grid hit@1 extrapolation. All other
        # gates must still pass for this branch to apply.
        other_pass = all(gates.get(k) is True
                         for k in REQUIRED_GATES if k != 'M1')

        # --- A2 added gates (M1' / M1-loc) -------------------------------
        # These are pre-registered mechanism gates for the A2 arms and are NOT
        # part of A's M1-M4 set, so they are evaluated separately and folded
        # into the aggregate below instead of being forced into REQUIRED_GATES
        # (which would rewrite A's archived verdict).
        a2 = a2_readings(run)
        if a2 is not None:
            r2 = a2['readings']
            g2 = a2['gates']
            # The frozen clip list cannot be re-chosen after the fact: the rank
            # probe and the gate probe must have selected the same clips.
            rank_idx = load_json(os.path.join(run, A2_RANK_VAL)).get('indices')
            gate_idx = gate.get('indices')
            if rank_idx != gate_idx:
                a2['blockers'].append('rank and gate probes selected '
                                      'different clips')
                for key in g2:
                    g2[key] = None
            print()
            print('=' * 72)
            print(f'A2 ADDED GATES (pre-registered section 5, arm '
                  f'{a2["arm"]}, read from {A2_RANK_VAL})')
            print('=' * 72)
            sym = {True: 'PASS', False: 'FAIL', None: 'NOT MEASURED'}

            def verdict2(name):
                return sym.get(g2.get(name), 'NOT MEASURED')

            print(f"M1' rank vs shuffle : {verdict2('M1p')}"
                  f"  auc={_fmt(r2['auc'])} (bar >={A2_BAR_AUC})"
                  f"  auc-gt_shuffle={_fmt(r2['auc_margin'])}"
                  f" (bar >={A2_BAR_AUC_MARGIN})")
            if 'M1loc' in g2:
                print(f"M1-loc peak budget  : {verdict2('M1loc')}"
                      f"  clips with max_gt_prob>=0.5 = "
                      f"{r2['n_max_gt_prob_ge_0.5']}/{A2_FROZEN_N}"
                      f" (bar >={A2_BAR_MAXGT_CLIPS})"
                      f"  recall@top-k median="
                      f"{_fmt(r2['recall_at_top_k_median'])}"
                      f" (bar >={A2_BAR_RECALL_TOPK_MEDIAN})")
            elif a2['arm'] == 'A2R':
                print('M1-loc              : n/a for A2-R (pre-registered '
                      'A2-S only)')
            for note in a2['blockers']:
                print(f'   BLOCKER: {note} -> A2 gates read NOT MEASURED')
            decision['a2_gates'] = g2
            a2_unmeasured = any(v is None for v in g2.values())
            a2_all_pass = (not a2_unmeasured
                           and all(v is True for v in g2.values()))

        grid_limited = (decision.get('m1_grid_limited') and other_pass
                        and gates.get('M1') is False)
        print()
        if all_pass and not evidence_ok:
            print('S1 AGGREGATE: ALL GATES PASS but the evidence bundle is '
                  'INCOMPLETE -- treat as NOT EVALUABLE until it is rerun')
        elif grid_limited:
            print('S1 AGGREGATE: M1 failed on the full-res raw rate only, '
                  'with a healthy thr50/excess reading and all other gates '
                  'passing -> INSUFFICIENT EVIDENCE (pre-registered section '
                  '71.9.3(2)), NOT a mechanism refutation. Do not archive as '
                  'negative; add a full-res matched control first.')
        else:
            print(f'S1 AGGREGATE: '
                  f'{"ALL PASS" if all_pass else "NOT ALL PASS"}')

        # --- A2 aggregate -------------------------------------------------
        # An A2 arm passes S1 only when A's M1-M4 AND the A2-added gates pass.
        # Reporting the M1-M4 line alone would let a run whose M1'/M1-loc
        # failed read as a pass, which is the defect this section closes.
        if a2 is not None:
            a2_gates = a2['gates']
            a2_measured = all(v is not None for v in a2_gates.values())
            a2_pass = a2_measured and all(v is True for v in a2_gates.values())
            labels = ', '.join(f'{k}={sym.get(v, "NOT MEASURED")}'
                               for k, v in a2_gates.items())
            print()
            if not a2_measured:
                print(f'A2 AGGREGATE: {labels} -> NOT EVALUABLE for the A2 '
                      f'added gates; fix the blockers above and re-read')
            elif all_pass and a2_pass:
                print(f'A2 AGGREGATE: ALL PASS (M1-M4 + {labels}) -- the '
                      f'section-5 S2 numbers may be requested')
            elif all_pass and not a2_pass:
                print(f'A2 AGGREGATE: NOT ALL PASS -- M1-M4 all passed but the '
                      f'A2 added gates failed ({labels}). The arm does NOT '
                      f'enter ep16.')
            else:
                print(f'A2 AGGREGATE: NOT ALL PASS (M1-M4 not all passing; '
                      f'{labels})')

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
    print('  EXCEPTION (section 71.9.3(2)): M1 failing ONLY on the full-res '
          'raw hit@1, with thr50/the excess reading healthy and every other '
          'gate passing, is INSUFFICIENT EVIDENCE -- add a full-res matched '
          'control before deciding; do not archive as a refutation.')
    print('  Any Ped/Cyc <= non-inferiority line -> stop, record as the '
          "family side-effect's sixth reproduction.")
    print('  Never: weight/KL sweeps or an ep24 rescue.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
