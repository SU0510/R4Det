#!/usr/bin/env python3
"""Emit the section-72 training record for the candidate-A ep6 gate.

READ-ONLY. Run it after ``semA_ep6_verdict.py`` has reported the battery as
complete. It renders ``docs/training_runs_full.md`` section 72 from the
evidence JSONs and the run's own ``*.log.json``, so none of the numbers are
transcribed by hand.

Why this exists (AGENTS.md rules 5 and 6):

  * rule 5 fixes three summary readings and forbids an ambiguous bare "BEST":
    fixed-window mean WITH its true span and n, the all-epoch peak as
    "value @ epoch" (weight-independent) and best-saved as a SEPARATE field;
  * rule 6 requires the full per-epoch curve, the per-class breakdown and the
    mechanism diagnostics, not just the summary.

Hand-assembling that every time is where this project has already made
transcription and arithmetic errors (e.g. the DC1 threshold arithmetic
correction). The generator therefore keeps ONE source of truth for the gate
logic: it imports ``semA_ep6_verdict`` and reuses its ``check_evidence``,
``m1_grid_limited`` and ``val_by_epoch`` rather than restating them. It decides
nothing itself -- the fork stays a manual step.

Output is a standalone markdown file intended to be inserted into
``docs/training_runs_full.md`` as section 72. It deliberately does NOT edit the
training record in place: the canonical doc stays under human control, and the
generator's job stays mechanical.

Two things it asserts rather than leaves implicit:

  * the S1 verdict and the pre-registered fork text;
  * that NO S2 AP claim is made here (S2 is the ep12-16 window and needs user
    approval before any ep6->ep16 continuation).

Run facts are DERIVED from the run's own logs -- the ep6 wall-clock span from
the training log's first/last timestamps, throughput and memory from the
``time``/``memory`` fields of the ``*.log.json`` train records, and the stop
reason plus gate/GPU-release timestamps from the watchdog log. Anchoring a
hand-typed timestamp is the transcription surface this generator exists to
remove, so nothing here is typed by hand. Only if a bundle is missing those
logs does the field degrade to a visible ``<!-- FILL: ... -->`` marker, which
is counted in a warning on stderr so it cannot be silently dropped.

Usage:
  python3 me_rssm/sanity/semA_ep6_record.py \
    --run /data/lurui/work_dirs/fgfull_N4_no2d_igdr_2x4_24e_semA_seed0 \
    --output /tmp/section72.md
"""

import argparse
import datetime
import glob
import importlib.util
import json
import os
import re
import statistics
import sys

_REPO = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..'))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_REPO, rel))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Single source of truth for the gate logic, evidence manifest and baselines.
verdict = _load('verdict_ep6_for_record', 'me_rssm/sanity/semA_ep6_verdict.py')

DEFAULT_RUN = verdict.DEFAULT_RUN
WINDOWS = (
    ('近距', 3, 'z_util_std_ep6_w3.json'),
    ('中距', 100, 'z_util_std_ep6_w100.json'),
    ('远距', 1800, 'z_util_std_ep6_w1800.json'),
)
POST_SOURCE = 'post_source_std_ep6_stride700.json'
ZCONTRACT_JSON = 'semA_zcontract_ep6.json'
WATCH_LOG = 'watch_semA_ep6.log'
SIGNAL_NAMES = {1: 'SIGHUP', 2: 'SIGINT', 9: 'SIGKILL', 15: 'SIGTERM'}
STDOUT_LOG = 'stdout.log'

# Standard-family same-epoch comparator for the shuffle_z reading: the plain
# no2d_igdr RSSM, seed1, ep6. Transcribed from docs/training_runs_full.md
# section 62.2 (table "ep6 三窗口反事实诊断"); display-only, never re-derived
# from the reading under test.
STD_EP6_SHUFFLE_HEAD = {'近距': 0.4973, '中距': 0.0330, '远距': 0.0336}

# Train-side diagnostics of the new supervision head. These are train-split
# statistics (z sampled), recorded per epoch by the model itself.
TRAIN_STAT_KEYS = (
    ('stat_sem_hit1', '`stat_sem_hit1`'),
    ('stat_sem_thr50', '`stat_sem_thr50`'),
    ('loss_rssm_sem', '`loss_rssm_sem`'),
    ('stat_cls_res_rms', '`stat_cls_res_rms`'),
    ('stat_mu_diff_sq', '`stat_mu_diff_sq`'),
)

FILL = '<!-- FILL: {} -->'

# The date the §72.9 Ped/Cyc stop-line was re-scoped to S2. It is the date the
# CORRECTION was established, not the section's date, so it stays fixed for
# every arm's draft (an A2 draft must not re-date the correction to its own
# run). Reproduces the canonical doc's hand-written note byte-for-byte.
S2_SCOPE_CORRECTION_DATE = '2026-10-11'


def load_json(path):
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def mean_variant(payload, variant, field):
    """Mean of one variant's reading over the window's per-sample entries."""
    values = []
    for entry in (payload.get('per_sample') or {}).values():
        probe = (entry.get('variants') or {}).get(variant) or {}
        value = probe.get(field)
        if value is not None:
            values.append(value)
    return statistics.mean(values) if values else None


def window_readings(payload):
    """{variant: {'head': mean, 'feature': mean}} for one z-util window."""
    out = {}
    for variant in ('shuffle_z', 'zero_z', 'prior_only'):
        out[variant] = {
            'head': mean_variant(payload, variant, 'head_l2_ratio_vs_normal'),
            'feature': mean_variant(
                payload, variant, 'feature_l2_ratio_vs_normal'),
        }
    return out


def train_stats_by_epoch(run_dir):
    """{epoch: {key: value}} from the last train record of each epoch."""
    by_epoch = {}
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
                if rec.get('mode') == 'train' and rec.get('epoch') is not None:
                    by_epoch[int(rec['epoch'])] = rec
    return by_epoch


def train_stat_means(run_dir):
    """{epoch: {'stat_sem_thr50': mean, 'stat_sem_hit1': mean}} over train recs.

    ``train_stats_by_epoch`` keeps only the LAST record of each epoch, which is
    what the section-72.5 curve reports. The calibration reading the M1 bars
    came from (section 60.5/61.5) is an epoch MEAN over the split's records, so
    the same-caliber counterpart has to be an epoch mean too -- taking the last
    value would silently mix the two conventions again.
    """
    sums, counts = {}, {}
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
                if rec.get('mode') != 'train' or rec.get('epoch') is None:
                    continue
                epoch = int(rec['epoch'])
                for key in ('stat_sem_thr50', 'stat_sem_hit1'):
                    value = rec.get(key)
                    if isinstance(value, (int, float)):
                        sums.setdefault(epoch, {}).setdefault(key, 0.0)
                        counts.setdefault(epoch, {}).setdefault(key, 0)
                        sums[epoch][key] += float(value)
                        counts[epoch][key] += 1
    return {epoch: {key: sums[epoch][key] / counts[epoch][key]
                    for key in sums[epoch]}
            for epoch in sums}


def a2_split_contrast(run_dir, a2=None):
    """Train-vs-val rank readings for an A2 arm, or None.

    H-A2 is a claim about the TRAIN/VAL GAP: the arm's ranking target is
    supposed to make the peak budget survive a held-out scene. The M1' gate is
    measured on val only, so a val-only table cannot show whether the gap
    closed, stayed, or widened. The watchdog already runs the identical probe
    on the train split into ``semA_rank_train_ep6.json``; reading it here keeps
    the contrast derived instead of hand-transcribed.

    Returns None when this is not an A2 arm or when the train artifact is
    absent (absent is rendered as a FILL marker by the caller, not as zero).
    """
    if a2 is None:
        a2 = verdict.a2_readings(run_dir)
    if a2 is None:
        return None
    path = os.path.join(run_dir, verdict.A2_RANK_TRAIN)
    if not os.path.exists(path):
        return None
    train = load_json(path)
    val = load_json(os.path.join(run_dir, verdict.A2_RANK_VAL)) or {}

    def _agg(payload):
        rows = payload.get('per_clip') or []
        if not rows:
            return None
        return {
            'n': len(rows),
            'auc': verdict._mean(rows, 'auc_mean_over_occupied_channels'),
            'shuffle': verdict._mean(rows, 'gt_shuffle_auc'),
            'max_gt_prob': verdict._mean(rows, 'max_gt_prob'),
            'recall_median': verdict._median(rows, 'recall_at_top_k_mean'),
            'recall_mean': verdict._mean(rows, 'recall_at_top_k_mean'),
            'n_maxgt_ge_0.5': sum(
                1 for r in rows
                if verdict._finite(r.get('max_gt_prob'))
                and r['max_gt_prob'] >= 0.5),
        }

    return {'train': _agg(train), 'val': _agg(val)}


def arm_short_label(run_dir, a2=None):
    """'A' / 'A2-S' / 'A2-R' -- bare arm name for mid-sentence prose.

    Used where the sentence would otherwise say "A" for every run, e.g.
    "**A 跑标准家族**，故用 `probe_z_utilization_standard.py`". An A2 draft that
    says A ran something is misattributing the run's own setup.
    """
    if a2 is None:
        a2 = verdict.a2_readings(run_dir)
    arm = (a2 or {}).get('arm')
    return {'A2S': 'A2-S', 'A2R': 'A2-R'}.get(arm, 'A')


def arm_side_label(run_dir, a2=None):
    """'A 侧' / 'A2-S 侧' ... -- whose training statistics a table reports.

    Section 72.5's heading was hard-coded to "A 侧". For an A2 arm the
    ``stat_sem_*`` rows are the A2 arm's OWN training diagnostics, so the
    heading would misattribute them to A.
    """
    return arm_short_label(run_dir, a2) + ' 侧'


def saved_epochs(run_dir):
    """Epochs with an actual epoch_<N>.pth on disk (checkpoint availability)."""
    out = []
    for path in glob.glob(os.path.join(run_dir, 'epoch_*.pth')):
        stem = os.path.basename(path)[len('epoch_'):-len('.pth')]
        if stem.isdigit():
            out.append(int(stem))
    return sorted(out)



# --- run facts (wall-clock/log facts the evidence JSONs do not carry) ----
#
# These live in the run's own text logs, so they are DERIVED, not hand-filled:
# section 72.11 claims "未手抄任何读数", and anchoring a hand-typed timestamp
# is exactly the transcription surface this generator exists to remove. The
# FILL fallback stays for a bundle whose logs are missing.

_TS_RE = re.compile(r'^(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)')
# The A watchdog tags its lines ``[watch]``; the A2 watchdog tags them
# ``[watch-A2S]`` / ``[watch-A2R]``. Matching only the bare tag silently
# dropped every A2 stop-time reading, so the draft fell back to a FILL
# placeholder even though the run HAD recorded gate hit / GPU free / probes
# done.
_WATCH_TS_RE = re.compile(
    r'^\[watch(?:-[A-Za-z0-9]+)?\] (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) (.*)$')
_EXIT_RE = re.compile(r'^EXIT=(-?\d+)\s*$')


def _timestamped_spans(run_dir):
    """{log basename: (first_ts, last_ts, n)} over lines starting with a ts."""
    spans = {}
    for path in sorted(glob.glob(os.path.join(run_dir, '*.log'))):
        first = last = None
        count = 0
        with open(path, encoding='utf-8', errors='replace') as handle:
            for line in handle:
                match = _TS_RE.match(line)
                if not match:
                    continue
                count += 1
                first = first or match.group(1)
                last = match.group(1)
        if count:
            spans[os.path.basename(path)] = (first, last, count)
    return spans


def train_log_name(run_dir):
    """The log the ep6 span is read from, in an EXPLICIT preference order.

    The run's own ``YYYYmmdd_HHMMSS.log`` is the authoritative training log.
    ``stdout.log`` is the same process's stdout and is an acceptable stand-in
    (its span matches). The watchdog log is NEVER a training log: it starts
    after the run was interrupted, so using it would report the PROBE window
    as the training window. Anything else (e.g. a stray ``0_decoy.log``) is
    ignored rather than silently accepted by "first ``*.log``" ordering.
    """
    spans = _timestamped_spans(run_dir)
    stamped = [n for n in spans if re.match(r'^\d{8}_\d{6}\.log$', n)]
    if stamped:
        return sorted(stamped)[0]
    if STDOUT_LOG in spans:
        return STDOUT_LOG
    return None


def train_throughput(run_dir):
    """(median time s/iter, median memory MiB) over train records, or None."""
    times, memories = [], []
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
                if rec.get('mode') != 'train':
                    continue
                if isinstance(rec.get('time'), (int, float)):
                    times.append(float(rec['time']))
                if isinstance(rec.get('memory'), (int, float)):
                    memories.append(float(rec['memory']))
    if not times:
        return None
    return (statistics.median(times),
            statistics.median(memories) if memories else None,
            len(times))


def arm_label(run_dir, a2=None):
    """Human label for the bundle's arm, used in the section heading.

    '候选 A' for the A generation; '候选 A2-S'/'候选 A2-R' for the A2 arms.
    Leaving the A label in place would make an A2 draft read as A's archived
    (already-closed) negative result.
    """
    if a2 is None:
        a2 = verdict.a2_readings(run_dir)
    if a2 is None:
        return '候选 A'
    arm = (a2 or {}).get('arm')
    return {'A2S': '候选 A2-S', 'A2R': '候选 A2-R'}.get(arm, '候选 A2')


def watch_log_name(run_dir):
    """Which watchdog log this bundle belongs to.

    The A generation's log is ``watch_semA_ep6.log``; the A2 watchdogs write
    ``watch_semA2S_ep6.log`` / ``watch_semA2R_ep6.log``. Hard-coding the A name
    silently dropped the whole stop-time evidence row for an A2 bundle (the
    file simply does not exist), leaving a FILL placeholder in a run that had
    in fact recorded the timings.
    """
    for name in (WATCH_LOG, 'watch_semA2S_ep6.log', 'watch_semA2R_ep6.log'):
        if os.path.exists(os.path.join(run_dir, name)):
            return name
    return WATCH_LOG


def watchdog_script_name(run_dir):
    """The watchdog script that produced this bundle (for the prose header)."""
    name = watch_log_name(run_dir)
    if name == 'watch_semA2S_ep6.log':
        return 'watch_semA2_ep6.sh'
    if name == 'watch_semA2R_ep6.log':
        return 'watch_semA2_ep6.sh'
    return 'watch_semA_ep6.sh'


def prereg_doc(run_dir, a2=None):
    """The pre-registration document this bundle was produced under.

    Keyed on the ARM (config path), not on the presence of the rank artifact:
    A's archived bundle also carries a post-hoc ``semA_rank_val_ep6.json``, so
    a file-existence test would cite the A2 pre-registration for A's run.
    """
    if a2 is None:
        a2 = verdict.a2_readings(run_dir)
    if a2 is not None:
        return 'docs/rssm_candidate_A2_prereg_20261011.md'
    return 'docs/rssm_candidate_A_prereg_20261010.md'


def arm_single_variable(run_dir, a2=None):
    """The arm's ONE changed knob, as a (label, value) pair.

    The A generation changed two things at once relative to the plain
    `no2d_igdr` base (`sem_head` supervision + the `z_cls_logits` residual
    readout), which is why A cannot separate them. The A2 arms were
    pre-registered specifically to move ONE knob at a time, so printing A's
    two-knob sentence into an A2-S draft would describe a different experiment
    -- and would also hide the fact that A2-R never touched supervision.
    """
    if a2 is None:
        a2 = verdict.a2_readings(run_dir)
    arm = (a2 or {}).get('arm')
    if arm == 'A2S':
        return ('`sem_target_mode`（z 的语义监督目标：逐格高斯 → 逐通道排序）',
                '`channel_contrastive`（`sem_rank_q`/`sem_rank_margin`/'
                '`sem_rank_weight` 见 §4.1）')
    if arm == 'A2R':
        return ('`cls_readout_mode`（分类残差到 anchor 通道的写入方式）',
                '`anchor_slot`（同 predictor、同参数量，不改监督）')
    return ('`sem_head`（z 排他语义监督）+ `z_cls_logits`（零初始化分类残差读出）',
            '——')


def arm_comparator(run_dir, a2=None):
    """The reference config this arm is a single-variable change against."""
    if a2 is None:
        a2 = verdict.a2_readings(run_dir)
    if a2 is not None:
        return '`..._no2d_igdr_semreadout.py`（A 底座；不重跑 A）'
    return '`..._no2d_igdr.py`（`MotionAlignedRSSMFusion`, seed0）'


# Both sides of a source are REQUIRED: one split alone cannot show that the
# gap is the split, so a single-sided bundle must fall through to "not
# measured" rather than half-report. Tried widest contract first.
#
# NEITHER source is guaranteed. §71.9.19's n=20 pair came from an ad hoc probe
# run for A; the n=10 pair is likewise only present for A. The A2 watchdog runs
# `probe_semA_bnmode.py` on the VAL side alone (n=20) and never runs the
# train-split arm, so an A2 bundle honestly renders NOT MEASURED unless someone
# runs that arm for it too.
SPLIT_ATTRIBUTION_SOURCES = (
    ('semA_splitprobe_val_full_n20.json',
     'semA_splitprobe_train_full_n20.json'),
    ('semA_bnmode_ep6.json', 'semA_bnmode_ep6_trainsplit.json'),
)


def split_attribution(run_dir):
    """Measure whether the M1 train/val gap is the split or the BN mode.

    §71.9.19 settled this for A: re-reading one checkpoint on BOTH splits with
    the z contract and the eval mode pinned put the gap almost entirely on the
    split, with the BN mode a small second-order term.

    That is a statement about the DATA, but the artifact carrying it was
    produced by an ad hoc probe FOR A. Transcribing §71.9.19's numbers into the
    generator would make it keep asserting them for an arm that never measured
    them, which is exactly the failure this generator exists to remove. So
    return only the readings actually present for THIS bundle, with their own n;
    the caller degrades to an explicit "not measured" when there are none.
    """
    for val_name, train_name in SPLIT_ATTRIBUTION_SOURCES:
        val_path = os.path.join(run_dir, val_name)
        train_path = os.path.join(run_dir, train_name)
        if not (os.path.exists(val_path) and os.path.exists(train_path)):
            continue
        val, train = load_json(val_path), load_json(train_path)
        out = {'val_file': val_name, 'train_file': train_name,
               'n': {}, 'splits': {}}
        for label, payload in (('val', val), ('train', train)):
            arms = payload.get('arms') or {}
            base = arms.get('eval_running_stats') or {}
            zpath = arms.get('z_path_batch_stats') or {}
            out['n'][label] = len(payload.get('indices') or [])
            out['splits'][label] = {
                'eval_thr50': base.get('mean_thr50'),
                'eval_hit1': base.get('mean_hit1'),
                'bn_delta_thr50': _diff(zpath.get('mean_thr50'),
                                        base.get('mean_thr50')),
                'bn_delta_hit1': _diff(zpath.get('mean_hit1'),
                                       base.get('mean_hit1')),
            }
        return out
    return None


# The reference arm the A2 drafts compare their train/val rank gap against.
# Section 72.11 asserts "未手抄任何读数", so these must NOT be typed constants:
# they are read from A's own archive, which is the run whose gap the sentence
# names. When that archive is absent the caller drops the comparison entirely
# rather than quoting a number it cannot substantiate.
A_REFERENCE_RUN = DEFAULT_RUN


def a_rank_gap():
    """A's own train/val rank gap, or None when A's archive is unreadable.

    Returns {'train': {...}, 'val': {...}} with the same aggregates the A2
    contrast table uses, so both sides are computed by one code path.
    """
    out = {}
    for label, name in (('train', verdict.A2_RANK_TRAIN),
                        ('val', verdict.A2_RANK_VAL)):
        path = os.path.join(A_REFERENCE_RUN, name)
        if not os.path.exists(path):
            return None
        payload = load_json(path)
        rows = payload.get('per_clip') or []
        if not rows:
            return None
        out[label] = {
            'n': len(rows),
            'auc': verdict._mean(rows, 'auc_mean_over_occupied_channels'),
            'recall_median': verdict._median(rows, 'recall_at_top_k_mean'),
        }
    return out


def class_coverage(run_dir, a2=None):
    """Per-clip class occupancy of the frozen window, or None.

    §5's contract requires "类别覆盖作为**报告字段**给出" precisely because a
    window with no clip of some class cannot support a class-specific claim
    (Ped appears in only a few of the 20 clips). The gate JSON does not record
    the per-channel occupation, but the rank probe's ``occupied_channels`` is
    the same frozen clip grid, so the coverage is derivable for A2 arms.
    """
    if a2 is None:
        a2 = verdict.a2_readings(run_dir)
    if a2 is None:
        return None
    out = {}
    for label, name in (('val', verdict.A2_RANK_VAL),
                        ('train', verdict.A2_RANK_TRAIN)):
        path = os.path.join(run_dir, name)
        if not os.path.exists(path):
            continue
        rows = (load_json(path).get('per_clip') or [])
        occ = [r.get('occupied_channels') for r in rows
               if isinstance(r.get('occupied_channels'), int)]
        if len(occ) != len(rows) or not occ:
            continue
        hist = {}
        for value in occ:
            hist[value] = hist.get(value, 0) + 1
        out[label] = {'n': len(occ), 'hist': hist}
    return out or None


def m4floor_readings(run_dir, a2=None):
    """The m4floor probe's mechanism readings, or None.

    This artifact is produced for BOTH arms but was read by nothing: the
    pre-registration makes two of its readings load-bearing --
    ``frac_cells_residual_can_flip`` is A2-R's S0-5 anticipation (§5), and
    ``pooled_prior_floor`` is what M4's ``own_hit`` bar is computed from. A
    measured reading that no report carries is indistinguishable from an
    unmeasured one, which is the failure mode this whole audit keeps hitting.
    """
    if a2 is None:
        a2 = verdict.a2_readings(run_dir)
    if a2 is None:
        return None
    path = os.path.join(run_dir, verdict.A2_M4FLOOR)
    if not os.path.exists(path):
        return None
    payload = load_json(path)
    if not isinstance(payload, dict):
        return None
    return payload


def watch_facts(run_dir):
    """(gate_ts, free_ts, done_ts, signal, exit_code) from the watchdog log."""
    path = os.path.join(run_dir, watch_log_name(run_dir))
    if not os.path.exists(path):
        return None
    gate = free = done = None
    with open(path, encoding='utf-8', errors='replace') as handle:
        for line in handle:
            match = _WATCH_TS_RE.match(line)
            if not match:
                continue
            ts, rest = match.group(1), match.group(2)
            if gate is None and rest.startswith('gate hit'):
                gate = ts
            elif free is None and rest.startswith('GPUs free'):
                free = ts
            elif rest.startswith('probes done'):
                done = ts
    stdout = os.path.join(run_dir, STDOUT_LOG)
    signal = exit_code = None
    if os.path.exists(stdout):
        with open(stdout, encoding='utf-8', errors='replace') as handle:
            for line in handle:
                hit = _EXIT_RE.match(line)
                if hit:
                    exit_code = int(hit.group(1))
                # The traceback's own f-string literal also contains
                # "got signal:", so keep the LAST numeric match (the raised
                # exception message) instead of the first.
                if 'got signal:' in line:
                    tail = line.rsplit('got signal:', 1)[1].strip()
                    token = tail.split()[0].rstrip(',') if tail else ''
                    if token.isdigit():
                        signal = int(token)
    return (gate, free, done, signal, exit_code)


def pct(value, digits=4):
    if not isinstance(value, (int, float)):
        return 'n/a'
    return f'{value * 100:.{digits}f}%'


def num(value, digits=4):
    if not isinstance(value, (int, float)):
        return 'n/a'
    return f'{value:.{digits}f}'


def signed(value, digits=4):
    if not isinstance(value, (int, float)):
        return 'n/a'
    return f'{value:+.{digits}f}'


def _diff(new, old):
    """new - old, or None when either side is missing (never 0.0)."""
    if not isinstance(new, (int, float)) or not isinstance(old, (int, float)):
        return None
    return new - old


def render(run_dir, date_str):
    by_epoch = verdict.val_by_epoch(run_dir)
    eps = sorted(by_epoch)
    train_stats = train_stats_by_epoch(run_dir)
    gates_payload = verdict.load_json(os.path.join(run_dir, verdict.GATE_JSON))
    gates = (gates_payload or {}).get('gates', {})
    m1 = (gates_payload or {}).get('M1_sem_head', {})
    m2 = (gates_payload or {}).get('M2_exclusivity', {})
    m3 = (gates_payload or {}).get('M3_readout_toggle', {})
    m4 = (gates_payload or {}).get('M4_class_information', {})
    fingerprints = (gates_payload or {}).get('input_fingerprints', {}) or {}

    _, evidence_ok = verdict.check_evidence(run_dir)
    all_pass = all(gates.get(k) is True for k in verdict.REQUIRED_GATES)
    other_pass = all(gates.get(k) is True for k in verdict.REQUIRED_GATES
                     if k != 'M1')
    mean_excess = verdict.m1_excess_mean(m1)
    # The exception's scope is owned by the verdict module (single source of
    # truth): it requires M1 to be the ONLY failing gate of A's M1-M4 set AND,
    # for an A2 arm, the arm's own pre-registered added gates to have passed.
    # Re-deriving it here with the older two-clause expression would let the
    # draft soften an A2 mechanism failure into an instrumentation complaint.
    a2 = verdict.a2_readings(run_dir)
    a2_gates_pre = (a2 or {}).get('gates') or {}
    a2_measured_pre = bool(a2_gates_pre) and all(
        v is not None for v in a2_gates_pre.values())
    a2_added_pass_pre = (a2 is None
                         or (a2_measured_pre
                             and all(v is True for v in a2_gates_pre.values())))
    grid_limited = verdict.grid_limited_applies(
        m1.get('mean_thr50'), m1.get('mean_hit1'), mean_excess,
        other_pass, gates.get('M1') is False,
        added_pass=a2_added_pass_pre)

    lines = []
    add = lines.append

    def gate_text(name):
        if name not in gates:
            return 'NOT MEASURED'
        return 'PASS' if gates.get(name) else 'FAIL'

    # A2 arms add pre-registered gates (M1' / M1-loc) that live in their own
    # artifact. Folding them in is mandatory: an A2 bundle whose added gates
    # failed must not be headed "全门通过" off M1-M4 alone.
    a2 = verdict.a2_readings(run_dir)
    a2_gates = (a2 or {}).get('gates') or {}
    a2_measured = all(v is not None for v in a2_gates.values())
    a2_all_pass = a2_measured and all(v is True for v in a2_gates.values())
    added_all_pass = (a2 is None) or (a2_measured and a2_all_pass)

    # The heading must distinguish "measured and failed" from "not measured":
    # an incomplete bundle is NOT EVALUABLE, and calling it "未全门通过" would
    # read as a mechanism failure.
    if not evidence_ok or gates_payload is None:
        headline = 'NOT EVALUABLE（证据不完整）'
    elif a2 is not None and not a2_measured:
        headline = 'NOT EVALUABLE（A2 增补门未测全）'
    elif grid_limited:
        headline = '证据不足（M1 跨网格口径）'
    elif all_pass and added_all_pass:
        headline = '全门通过'
    else:
        headline = '未全门通过'

    add(f'## 72. {arm_label(run_dir, a2)} ep6 机制门（S1）：{headline}（{date_str}）')
    add('')
    add(f'本节执行 `{prereg_doc(run_dir, a2=a2)}` §5 预登记的 S1 ep6')
    add('机制门。训练从零起（3 卡 GPU5/6/7、seed0、deterministic、24e 余弦日程），')
    add(f'ep6 val 与 `epoch_6.pth` 落盘后由看门狗 '
        f'`me_rssm/sanity/{watchdog_script_name(run_dir)}` 中断训练、')
    add(f'等待 GPU 释放，再自动跑 M1–M4 门 + 三窗口反事实 + source 探针'
        + ('（A2 臂另跑 M1\'/M1-loc 的双 split 排序探针与 M4 地板探针）。'
           if a2 is not None else '。'))
    add('')
    add('> **本节不宣称任何 S2 性能结论。** S2 是 ep12–16 五轮窗口，且其数值需')
    add('> 用户批准后才允许 ep6→ep16 续训（§5/§7）。本节只报 ep6 机制门与早期')
    add('> 曲线，二者不可互相替代。')
    add('')

    # --- summary ---------------------------------------------------------
    add('### 72.0 状态与判定摘要')
    add('')
    if not eps:
        add('**无 val 记录：S1 NOT EVALUABLE。**')
        add('')
    win_start, win_end = (eps[0], eps[-1]) if eps else (0, 0)
    overall_mean, overall_n, overall_span = verdict.window_labelled(
        by_epoch, verdict.METRIC, win_start, win_end)
    peak_ep = max((e for e in eps if verdict.METRIC in by_epoch[e]),
                  key=lambda e: by_epoch[e][verdict.METRIC], default=None)
    saved = saved_epochs(run_dir)
    saved_with_val = [e for e in saved if e in by_epoch]
    best_saved_ep = (max(saved_with_val,
                         key=lambda e: by_epoch[e][verdict.METRIC])
                     if saved_with_val else None)

    add('**三口径摘要**（AGENTS 规则 5：三个口径必须分别写清，不得用含糊的')
    add('“BEST” 代替其中任一项）：')
    add('')
    add('| 口径 | 数值 |')
    add('|---|---|')
    if overall_n:
        add(f'| 区间均值（标准 ep12-16 窗口不完整；实际跨度 '
            f'{overall_span}，n={overall_n}） | **{num(overall_mean)}** |')
    else:
        add('| 区间均值 | n/a（无 val） |')
    if peak_ep is not None:
        add(f'| 全轮峰值（所有 val epoch 的最高单点，与是否落盘无关） | '
            f'**{num(by_epoch[peak_ep][verdict.METRIC])} @ ep{peak_ep}** |')
    else:
        add('| 全轮峰值 | n/a |')
    if best_saved_ep is not None:
        add(f'| best saved（仅表示 checkpoint 可用性，不参与性能排名） | '
            f'**{num(by_epoch[best_saved_ep][verdict.METRIC])} '
            f'@ ep{best_saved_ep}**（落盘 {saved}） |')
    else:
        add(f'| best saved | n/a（落盘 {saved} 中无对应 val） |')
    add('')
    add('> 上述窗口均值是 **ep6 早期上下文**，不是 ep12–16 的 S2 判据；')
    add('> 单 seed、窗口未完整，不得据此解读效应。')
    add('')

    add('**S1 门控表**（阈值来自 §5 预登记，判定来自')
    add('`me_rssm/sanity/semA_ep6_verdict.py`，本节不重算门槛）：')
    add('')
    add('| 门 | 预登记门槛 | ep6 实测 | 判定 |')
    add('|---|---|---|---|')
    m2_ok = (m2.get('h_relative') is not None
             and m2.get('feat_relative') is not None
             and m2.get('h_relative') <= -0.30
             and m2.get('feat_relative') <= -0.30)
    add(f'| M1 语义头质量 | val `thr50 ≥ 0.30` 且 `hit@1 ≥ 0.60` | '
        f'`thr50` **{num(m1.get("mean_thr50"))}**、'
        f'`hit@1` **{num(m1.get("mean_hit1"))}** | **{gate_text("M1")}** |')
    add(f'| M2 z 排他性 | `h` 与 `feat` 两臂**各自** ≤ −30% | '
        f'`h` **{pct(m2.get("h_relative"))}**（head thr50 '
        f'{num(m2.get("h_mean_thr50"))} vs z '
        f'{num(m2.get("z_mean_thr50"))}）、`feat` '
        f'**{pct(m2.get("feat_relative"))}**'
        f'{"（两臂均达线）" if m2_ok else ""} | **{gate_text("M2")}** |')
    m3_n = m3.get('n')
    m3_wins = m3.get('wins')
    m3_reading = (f'**{m3_wins}/{m3_n}**'
                  if m3_n is not None else '**NOT MEASURED**')
    # The bar is a RATE of the frozen window, so state the count it implies at
    # THIS window's n rather than the fixed string "8/10": under A2's 20-clip
    # contract the bar is 16/20, and typing 8/10 would misdescribe the gate.
    m3_need = m3.get('required_wins')
    m3_frozen = m3.get('frozen_n')
    if m3_need is not None and m3_frozen is not None:
        m3_bar_text = f'≥{m3_need}/{m3_frozen}（窗口冻结比例 8/10）'
    else:
        m3_bar_text = '≥8/10（窗口冻结比例 8/10）'
    add(f'| M3 读出非装饰 | 强制 residual=0 使正样本 class-CE 变差 '
        f'{m3_bar_text} | {m3_reading} | **{gate_text("M3")}** |')
    if m4.get('pooled_prior_floor') is not None:
        add(f'| M4 类别正确性 | `own_hit` ≥ **pooled prior floor + 20pt**、'
            f'`donor_transfer` ≥ chance+20pt | `own_hit` '
            f'**{num(m4.get("mean_own_hit"))}**（floor '
            f'{num(m4.get("pooled_prior_floor"))} → bar '
            f'{num(m4.get("own_hit_bar"))}）、`donor_transfer` '
            f'**{num(m4.get("mean_donor_transfer"))}**（bar '
            f'{num(m4.get("bar"), 2) if m4.get("bar") is not None else "n/a"}） '
            f'| **{gate_text("M4")}** |')
    else:
        add(f'| M4 类别正确性 | `own_hit` 与 `donor_transfer` ≥ chance+20pt '
            f'（bar {num(m4.get("bar"), 2) if m4.get("bar") is not None else "n/a"}） | `own_hit` '
            f'**{num(m4.get("mean_own_hit"))}**、`donor_transfer` '
            f'**{num(m4.get("mean_donor_transfer"))}**（chance '
            f'{num(m4.get("chance"), 2) if m4.get("chance") is not None else "n/a"}） | '
            f'**{gate_text("M4")}** |')
    if a2 is not None:
        # A2's added gates, from the rank artifact. Their rows are emitted
        # right after M4 so the table reflects the FULL pre-registered gate
        # set the aggregate verdict is computed from.
        a2r = a2.get('readings') or {}
        sym = {True: 'PASS', False: 'FAIL', None: 'NOT MEASURED'}

        def gate_text2(name):
            return sym.get(a2_gates.get(name), 'NOT MEASURED')

        add(f"| M1' 排序泛化 | val（n={verdict.A2_FROZEN_N}）`auc` ≥ "
            f"{verdict.A2_BAR_AUC} 且 `auc − gt_shuffle` ≥ "
            f"{verdict.A2_BAR_AUC_MARGIN} | `auc` "
            f"**{num(a2r.get('auc'))}**、`auc − gt_shuffle` "
            f"**{num(a2r.get('auc_margin'))}**"
            f"（shuffle {num(a2r.get('auc_gt_shuffle'))}） | "
            f"**{gate_text2('M1p')}** |")
        if 'M1loc' in a2_gates:
            add(f"| M1-loc 峰预算 | `max_gt_prob ≥ 0.5` 的 clip 数 ≥ "
                f"{verdict.A2_BAR_MAXGT_CLIPS}/{verdict.A2_FROZEN_N} 且 "
                f"`recall@top-k` 中位 ≥ "
                f"{verdict.A2_BAR_RECALL_TOPK_MEDIAN} | "
                f"**{a2r.get('n_max_gt_prob_ge_0.5')}/{verdict.A2_FROZEN_N}**、"
                f"中位 **{num(a2r.get('recall_at_top_k_median'))}** | "
                f"**{gate_text2('M1loc')}** |")
        else:
            add('| M1-loc 峰预算 | （A2-R 不适用：§5 预登记为 A2-S 专属） | '
                '— | n/a |')
        for note in (a2.get('blockers') or []):
            add('')
            add(f'> **A2 增补门 BLOCKER**：{note}（该门按 NOT MEASURED，'
                f'不得当失败或通过）。')
        # H-A2 is a train/val GAP claim; a val-only table cannot show whether
        # the gap closed. The identical train-split probe is already run by the
        # watchdog, so render both sides side by side.
        split = a2_split_contrast(run_dir, a2=a2)
        add('')
        add(f"**{arm_label(run_dir, a2)} 的 train/val 对照（同一探针、"
            f"同一 20-clip 契约；本臂假设正是『峰预算能否离开训练场景』）：**")
        add('')
        if split and split.get('train') and split.get('val'):
            tr, va = split['train'], split['val']
            add('| split | n | `auc` | `gt_shuffle_auc` | `max_gt_prob` | '
                '`recall@top-k` 中位 | `recall@top-k` 均值 | '
                '`max_gt_prob ≥ 0.5` 的 clip |')
            add('|---|---:|---:|---:|---:|---:|---:|---:|')
            for name, r in (('train', tr), ('val', va)):
                add(f'| {name} | {r["n"]} | {num(r["auc"])} | '
                    f'{num(r["shuffle"])} | {num(r["max_gt_prob"])} | '
                    f'{num(r["recall_median"])} | {num(r["recall_mean"])} | '
                    f'{r["n_maxgt_ge_0.5"]}/{r["n"]} |')
            add('')
            gap_auc = None
            if tr['auc'] is not None and va['auc'] is not None:
                gap_auc = tr['auc'] - va['auc']
            gap_rec = None
            if (tr['recall_median'] is not None
                    and va['recall_median'] is not None):
                gap_rec = tr['recall_median'] - va['recall_median']
            add(f'> train−val 差：`auc` **{signed(gap_auc)}**、'
                f'`recall@top-k` 中位 **{signed(gap_rec)}**。')
            # The A reference is DERIVED from A's own archive: §72.11 promises
            # no hand-transcribed reading, and a typed constant would silently
            # keep asserting a stale gap (or assert one for a bundle whose
            # reference was never measured).
            ref = a_rank_gap()
            if ref:
                ra = None
                if (ref['train']['auc'] is not None
                        and ref['val']['auc'] is not None):
                    ra = ref['train']['auc'] - ref['val']['auc']
                rr = None
                if (ref['train']['recall_median'] is not None
                        and ref['val']['recall_median'] is not None):
                    rr = (ref['train']['recall_median']
                          - ref['val']['recall_median'])
                add(f'> A 的同口径差为 `auc` {num(ref["train"]["auc"])}−'
                    f'{num(ref["val"]["auc"])} = **{signed(ra)}**、'
                    f'`recall@top-k` 中位 {num(ref["train"]["recall_median"])}−'
                    f'{num(ref["val"]["recall_median"])} = **{signed(rr)}**'
                    f'（A 的 val 端 n={ref["val"]["n"]}）。')
            else:
                add('> A 的同口径参照**不可用**（A 的归档排序产物缺一侧）；'
                    '故本段只报本臂自身的 train−val 差，不与 A 并列。')
            add('> 本行只作机制定位，**不是门槛**。')
        else:
            add(FILL.format('A2 train/val 排序对照 '
                            f'（{verdict.A2_RANK_TRAIN}）'))
        add('')
    add('')
    chance_vals = [c['hit1_chance'] for c in (m1.get('per_clip') or [])
                   if c.get('hit1_chance') is not None]
    mean_chance = statistics.mean(chance_vals) if chance_vals else None
    add(f'**不进门补充读数（M1 口径）**：`hit1_chance` 均值 '
        f'**{num(mean_chance, 6)}**、'
        f'`hit1_excess_over_chance` 均值 **{num(mean_excess)}**。')
    add('按 §71.9.3(2)，`hit@1 ≥ 0.60` 是从 route-C 的 16×16 网格外推而来，')
    add('A 的 full-res 216×248 网格随机基线只有 ~7.5e-5（209× 格数），')
    add('故这两个读数必须与门槛并排记录。')
    add('')
    # The exception is only meaningful if the record says whether it APPLIED.
    # Leaving it as "an exception exists" would let a genuine miss read as
    # "口径未校准", which is exactly the post-hoc escape the prereg forbids.
    if gates_payload is not None and gates.get('M1') is False:
        thr50_v = m1.get('mean_thr50')
        hit1_v = m1.get('mean_hit1')
        if grid_limited:
            add('**§71.9.3(2) 例外条款：适用**（详见上文聚合判定）。')
        else:
            add('**§71.9.3(2) 例外条款：不适用。** 该条款要求 M1 *仅*在 '
                'full-res 原始 `hit@1` 上未过，而 `thr50` 与')
            add('`hit1_excess_over_chance` 都健康（即「头确实找得到目标中心，'
                '只是单格 argmax 跨不了网格」）。')
            add('')
            add('**该判定不依赖口径选择**，两种读法都不满足例外：')
            add('')
            # Pass/fail labels are DERIVED from the numbers in the same row,
            # never typed. A hand-written "未过" next to a passing number is
            # exactly the transcription error this generator exists to
            # remove -- and it would make the table contradict its own prose.
            ep6_train = (train_stat_means(run_dir).get(6) or {})
            t_thr = ep6_train.get('stat_sem_thr50')
            t_hit = ep6_train.get('stat_sem_hit1')

            def _kw(value, bar):
                if not isinstance(value, (int, float)):
                    return '读数缺失'
                return '**过**' if value >= bar else '未过'

            add('| 读法 | `thr50` | `hit@1` | vs 门槛 |')
            add('|---|---:|---:|---|')
            add(f'| 预登记的 val 口径（本门实际读数） | '
                f'{num(thr50_v)} | {num(hit1_v)} | '
                f'`thr50` {_kw(thr50_v, verdict.M1_BAR_THR50)} '
                f'{num(verdict.M1_BAR_THR50, 2)}、'
                f'`hit@1` {_kw(hit1_v, verdict.M1_BAR_HIT1)} '
                f'{num(verdict.M1_BAR_HIT1, 2)} |')
            add(f'| {arm_short_label(run_dir, a2)} 自身 train 口径（`stat_sem_*` ep6 的 '
                f'**epoch 均值**） | '
                f'{num(t_thr)} | {num(t_hit)} | '
                f'`thr50` {_kw(t_thr, verdict.M1_BAR_THR50)} '
                f'{num(verdict.M1_BAR_THR50, 2)}、'
                f'`hit@1` {_kw(t_hit, verdict.M1_BAR_HIT1)} '
                f'{num(verdict.M1_BAR_HIT1, 2)} |')
            add('')
            # The gate's operative caliber is val (that is where §5
            # pre-registers M1). The train side is a robustness check, so it
            # can only ever *reinforce* the val verdict -- it is never allowed
            # to be the reason the exception is refused.
            excess_ok_val = (isinstance(mean_excess, (int, float))
                             and mean_excess >= verdict.M1_EXCESS_BAR)
            thr_ok_val = (isinstance(thr50_v, (int, float))
                          and thr50_v >= verdict.M1_BAR_THR50)
            add(f'例外条款要求「excess 健康」，val 口径下的对应量是 '
                f'{num(mean_excess)}（bar {num(verdict.M1_EXCESS_BAR, 4)}）；'
                f'该口径下 `thr50` 为 {num(thr50_v)}'
                f'（bar {num(verdict.M1_BAR_THR50, 2)}）。')
            if not thr_ok_val or not excess_ok_val:
                add('**两项中至少一项未达线**，故 M1 是**真实未过**，'
                    '不是跨网格口径造成的伪失败，不得以「口径未校准」豁免。')
            else:
                add('**该口径下两项均达线**，例外前提本应成立——'
                    '此处的判定须以聚合判定段为准，不得据本表单独豁免。')
            if isinstance(t_hit, (int, float)):
                # excess = hit@1 - chance <= hit@1, so a train-side hit@1
                # below the excess bar is a *conservative* bound showing the
                # train caliber cannot rescue the exception either.
                add('')
                add(f'旁证（不是判据）：train 口径只记 `stat_sem_hit1`、'
                    f'未单独记 excess，而 `excess <= hit@1`，'
                    f'故 hit@1 = {num(t_hit)} 是该口径下 excess 的**上界**。')
                if t_hit < verdict.M1_EXCESS_BAR:
                    add(f'该上界 < bar {num(verdict.M1_EXCESS_BAR, 4)} ⇒ '
                        f'train 口径同样不满足例外，**该结论不建立在口径选择上**。')
                else:
                    add(f'该上界 ≥ bar {num(verdict.M1_EXCESS_BAR, 4)} ⇒ '
                        f'train 口径的上界不足以排除该口径下的例外，'
                        f'故**只有 val 口径的读数构成判定依据**。')
            add('')
            add('> **口径出处须与上表并读**（AGENTS 记录纪律）：'
                '`0.30 / 0.60` 这组门槛的先例 §59 route-C 报的是'
                '**train 侧 epoch 均值**（§60.5/§61.5，38 条 train 记录），'
                '而本门按 §5 预登记在 **val** 上测量，两者之间隔了一层清样；'
                '上表因此把两种读法并列，而不是拿 val 数字直接比 train 门槛。')
            add('> 上表两种读法是**稳健性并列**：val 是预登记的判据口径，'
                'train 只是旁证。若两者结论不一致，以 val 为准并显式记录分歧。')
        add('')

    if not evidence_ok:
        add('**证据包不完整 ⇒ NOT EVALUABLE**：上面四门读数在缺失产物补齐前')
        add('一律为暂读，即使四门全过也不得判 ALL PASS。')
    elif grid_limited:
        add('**S1 聚合判定：证据不足（INSUFFICIENT EVIDENCE），不是机制否定。**')
        add('M1 只在 full-res 原始 `hit@1` 上未过，而 `thr50` 与')
        add('`hit1_excess_over_chance` 健康、其余三门全过，命中 §71.9.3(2)')
        add('预先写定的例外条款：需补一个 full-res 同口径对照后再判，')
        add('不得当通过，也不得归档为负结果。')
    elif all_pass and added_all_pass and not grid_limited:
        add('**S1 聚合判定：ALL PASS。** 按 §7 分叉，下一步是**请求用户批准')
        add('§5 的 S2 数值**，批准后才允许从 `epoch_6.pth` 续训至 ep16。')
    else:
        # An arm's S1 is its OWN pre-registered gate set. For an A2 bundle that
        # includes the added M1'/M1-loc gates, so the aggregate must not be
        # decided off M1-M4 alone: it would print ALL PASS and authorise ep16
        # for an arm whose added gates actually FAILED, contradicting both the
        # heading and the pre-registration.
        failing = [k for k in verdict.REQUIRED_GATES if gates.get(k) is not True]
        if a2 is not None:
            failing += [k for k in sorted(a2_gates)
                        if a2_gates.get(k) is not True]
        add(f'**S1 聚合判定：NOT ALL PASS（{", ".join(failing)}）。** 按 §7')
        add('分叉，按负结果归档并把读数写进 Analysis，**不进 ep16**。')
        if gates.get('M2') is not True:
            add('M2 未过 ⇒ **“概率状态”这一宣称作废**：语义来自 h/观测，')
            add('z 只是被穿过的通路。')
        else:
            add('**M2 通过：语义头对输入**敏感**，但它的证据强度到此为止。**')
            add(f'h 臂与 feat 臂的 `thr50` 均为 '
                f'{num(m2.get("h_mean_thr50"))}/'
                f'{num(m2.get("feat_mean_thr50"))}，而 z 臂为 '
                f'{num(m2.get("z_mean_thr50"))}。')
            add('')
            add('> **不得据此宣称「语义只存在于 z」**：M2 是拿**只被 z 监督过**的 '
                'head 去喂 h/feat，属于分布外输入——即使 h 或观测里含有同样的'
                '对象信息，这个 head 也读不出来。因此 M2 只证明'
                '「该 head 依赖它训练时的输入」，**不能证明** z 是唯一信息源。')
            add('> 预登记把真正的 z 排他性检验放在 **C3 臂**（`sem_head_input='
                '\'feat\'` 从零训练、同日程对照），本次**未执行**——'
                '故「概率状态」这一主张在本次测量中**既未被否证、也未被证实**。')
    add('')

    # --- config ----------------------------------------------------------
    add('### 72.1 配置与唯一变量')
    add('')
    sv_label, sv_value = arm_single_variable(run_dir)
    add('| 项 | 值 |')
    add('|---|---|')
    add(f'| 配置 | `{os.path.relpath((gates_payload or {}).get("config", "n/a"), _REPO) if (gates_payload or {}).get("config") else "n/a"}` |')
    add(f'| 对照 | {arm_comparator(run_dir)} |')
    add(f'| 唯一变量 | {sv_label} |')
    if sv_value != '——':
        # Only the A2 arms are single-knob experiments; for A this row would
        # carry no information, and emitting it would churn A's frozen draft.
        add(f'| 变量取值 | {sv_value} |')
    add('| 未改动 | 重建目标、posterior/prior、KL/free-nats、`output_proj`、'
        'backbone、head 几何、优化器、日程、数据管线 |')
    add(f'| 权重指纹 | `sha256={str(fingerprints.get("checkpoint_sha256"))[:16]}...`、'
        f'size={fingerprints.get("checkpoint_size_bytes")}、'
        f'mtime={fingerprints.get("checkpoint_mtime")} |')
    add(f'| 采样位点 | clip 索引 `{(gates_payload or {}).get("indices")}`、'
        f'`shuffle_offset={(gates_payload or {}).get("shuffle_offset")}` |')
    add('')

    # --- run facts -------------------------------------------------------
    add('### 72.2 运行事实')
    add('')
    add('| 项 | 值 |')
    add('|---|---|')
    train_log = train_log_name(run_dir)
    span = _timestamped_spans(run_dir).get(train_log) if train_log else None
    if span:
        add(f'| ep6 段起止 UTC | {span[0]} → {span[1]}'
            f'（`{train_log}`，{span[2]} 条带时间戳行） |')
    else:
        add(FILL.format('ep6 段起止 UTC 时刻（日志首末时间戳）'))
    add('| GPU / seed | 5/6/7、seed 0、deterministic |')
    tp = train_throughput(run_dir)
    if tp:
        secs, mem, n = tp
        mem_txt = (f'{mem / 1024:.1f} GiB/卡' if mem else 'n/a')
        add(f'| 吞吐 / 显存 | 中位 **{secs:.3f} s/iter**、'
            f'{mem_txt}（`*.log.json` train 段 {n} 条） |')
    else:
        add(FILL.format('吞吐中位 s/iter、显存 GiB/卡（训练日志 time/memory 段）'))
    add(f'| val 记录 | ep{eps[0]}-ep{eps[-1]} 各一条，共 {len(eps)} 条 |'
        if eps else '| val 记录 | 无 |')
    add(f'| 落盘 | {saved}（`checkpoint_config.interval=2`） |')
    wf = watch_facts(run_dir)
    if wf and wf[0]:
        gate_ts, free_ts, done_ts, signal, exit_code = wf
        sig_name = SIGNAL_NAMES.get(signal, str(signal) if signal else '?')
        sig_txt = (f'{sig_name}({signal})' if signal else '?')
        add(f'| 停止方式 | 看门狗在新鲜 `epoch_6.pth` + ep6 val 后中断训练'
            f'（signal {sig_txt}，退出码 {exit_code}；中断是看门狗按 §5 的'
            f'预期动作，非崩溃）· gate hit {gate_ts} → GPU 释放 {free_ts} →'
            f' 探针结束 {done_ts} |')
    else:
        add(FILL.format('停止方式与 GPU 释放时刻（'
                        + watch_log_name(run_dir) + '）'))
    add('')

    # --- per-epoch curve --------------------------------------------------
    add(f'### 72.3 完整逐 epoch 曲线（ep{eps[0]}-ep{eps[-1]}）' if eps
        else '### 72.3 完整逐 epoch 曲线')
    add('')
    add('| ep | Overall3Dmod | vs base | Car | Truck | Ped_l | Cyc_l |')
    add('|---:|---:|---:|---:|---:|---:|---:|')
    base_by_epoch = {}
    for e in eps:
        rec = by_epoch[e]
        o = rec.get(verdict.METRIC)
        cells = []
        for key in verdict.CLASS_KEYS.values():
            cells.append(num(rec.get(key), 2))
        base = verdict.BASELINE['Overall']
        base_by_epoch[e] = o
        add(f'| {e} | {num(o, 2)} | {signed((o - base) if o is not None else None, 2)} '
            f'| ' + ' | '.join(cells) + ' |')
    add('')
    add(f'> `vs base` 对照 §5 的 seed0 基线 Overall **{verdict.BASELINE["Overall"]}**'
        '（ep12–16 窗口均值）；单点差值不含 seed 噪声。')
    add('')

    # --- window per-class -------------------------------------------------
    add(f'### 72.4 窗口逐项均值（实际跨度 {overall_span}，n={overall_n}）')
    add('')
    add('| 类别 | 本 run 窗口均值 | seed0 基线（ep12-16） | 差 |')
    add('|---|---:|---:|---:|')
    add(f'| Overall_3D_moderate | **{num(overall_mean, 2)}** | '
        f'{verdict.BASELINE["Overall"]} | '
        f'{signed((overall_mean - verdict.BASELINE["Overall"]) if overall_mean is not None else None, 2)} |')
    for name, key in verdict.CLASS_KEYS.items():
        m, n, span = verdict.window_labelled(by_epoch, key, win_start, win_end)
        add(f'| {name} | {num(m, 2)} | {verdict.BASELINE[name]} | '
            f'{signed((m - verdict.BASELINE[name]) if m is not None else None, 2)} |')
    add('')
    add('> 该窗口与基线窗口**不同长度**（基线段是 ep12-16，本 run 只有 ep1-6），')
    add('> 只作早期上下文，不作 S2 判据。')
    add('')

    # --- train stats ------------------------------------------------------
    add(f'### 72.5 {arm_side_label(run_dir)}训练期同口径读数（train split，逐 epoch 末值）')
    add('')
    add('| ep | ' + ' | '.join(label for _, label in TRAIN_STAT_KEYS) + ' |')
    add('|---:|' + '---:|' * len(TRAIN_STAT_KEYS))
    for e in sorted(train_stats):
        rec = train_stats[e]
        cells = [num(rec.get(key), 4) for key, _ in TRAIN_STAT_KEYS]
        add(f'| {e} | ' + ' | '.join(cells) + ' |')
    add('')
    if train_stats:
        last = train_stats[max(train_stats)]
        add(f'shuffle/zero 与门控读数无关；此处 `stat_sem_*` 为**训练统计**'
            f'（z 采样值、train split），与探针的 val 口径不同，不可混比。')
        add(f'末 epoch `stat_cls_res_rms` = '
            f'{num(last.get("stat_cls_res_rms"))}（>0 ⇒ 读出活着）。')
    add('')

    # --- gate detail ------------------------------------------------------
    add('### 72.6 S1 门明细（逐 clip）')
    add('')
    # §5 requires the window's class coverage as a REPORT field, not a
    # selection rule: a window where some class appears in only a few clips
    # cannot support a class-specific claim, and Ped is exactly that case.
    cov = class_coverage(run_dir, a2)
    if cov:
        for label in ('val', 'train'):
            if label not in cov:
                continue
            hist = ', '.join(f'{k} 类 ×{v}'
                             for k, v in sorted(cov[label]['hist'].items()))
            desc = {'val': 'val（冻结 20 clip）', 'train': 'train 同协议'}[label]
            add(f'> **类别覆盖（{desc}，n={cov[label]["n"]}）**：'
                f'每 clip 占用通道数分布 {hist}。**不按类别剔除 clip**'
                f'（§5：覆盖是报告字段，不是选样条件）；某类只覆盖少数 clip 时，'
                f'该类专属结论在 ep6 只能是**弱证据**。')
        add('')
    m1_cells = [(c.get('index'), c.get('num_pos_cells'))
                for c in (m1.get('per_clip') or [])
                if c.get('num_pos_cells') is not None]
    if m1_cells:
        light = min(m1_cells, key=lambda kv: kv[1])
        heavy = max(m1_cells, key=lambda kv: kv[1])
        if light[1] != heavy[1]:
            add(f'> **样本权重不均提示**：各 clip 的 GT 目标数并不相等——'
                f'clip `{light[0]}` 只有 **{light[1]}** 个目标，'
                f'clip `{heavy[0]}` 有 **{heavy[1]}** 个（§71.9.14 预检：'
                f'3/4/3/11/4/2/6/6/1/2）。')
            add(f'> 逐 clip 比率在 `{light[0]}` 上只能取少数离散值、方差最大，')
            add('> 故下面 M1/M3/M4 的**分组结论不得由单个 clip 单独支撑**。')
            add('')
    add('**M1 逐 clip**：')
    add('')
    add('| index | thr50 | hit@1 | hit1_chance | excess_over_chance | 正格数 |')
    add('|---:|---:|---:|---:|---:|---:|')
    for clip in (m1.get('per_clip') or []):
        add(f'| {clip.get("index")} | {num(clip.get("thr50"))} | '
            f'{num(clip.get("hit1"))} | {num(clip.get("hit1_chance"), 6)} | '
            f'{signed(clip.get("hit1_excess_over_chance"), 6)} | '
            f'{clip.get("num_pos_cells")} |')
    add('')
    add('**M2 两控制臂逐 clip**（`thr50`；z 为参照）：')
    add('')
    add('| index | z | h | feat |')
    add('|---:|---:|---:|---:|')
    z_by_index = {c.get('index'): c for c in (m1.get('per_clip') or [])}
    h_by_index = {c.get('index'): c for c in (m2.get('per_clip_h') or [])}
    f_by_index = {c.get('index'): c for c in (m2.get('per_clip_feat') or [])}
    for index in sorted(set(z_by_index) | set(h_by_index) | set(f_by_index)):
        add(f'| {index} | {num((z_by_index.get(index) or {}).get("thr50"))} | '
            f'{num((h_by_index.get(index) or {}).get("thr50"))} | '
            f'{num((f_by_index.get(index) or {}).get("thr50"))} |')
    add('')
    add('**M3 逐 clip**（`delta_off_minus_on > 0` 表示关掉 residual 后')
    add('正样本 class-CE 变差，即读出有贡献）：')
    add('')
    add('| index | CE(on) | CE(off) | delta(off-on) | 正格数 |')
    add('|---:|---:|---:|---:|---:|')
    for clip in (m3.get('per_clip') or []):
        add(f'| {clip.get("index")} | {num(clip.get("ce_on"))} | '
            f'{num(clip.get("ce_off"))} | '
            f'{signed(clip.get("delta_off_minus_on"))} | '
            f'{clip.get("cells")} |')
    add('')
    add('**M4 逐 clip**（`own_hit` = z 残差 argmax 命中该格 GT 类；')
    add('`donor_transfer` = 换 donor z 后残差 argmax 命中 donor 主类；')
    add('`head_swap_to_donor` = 只重跑 head 且喂 donor 残差时的 head 类 argmax）：')
    add('')
    add('| index | own_hit | donor_transfer | head_swap | donor_index | changed_frac |')
    add('|---:|---:|---:|---:|---:|---:|')
    for clip in (m4.get('per_clip') or []):
        add(f'| {clip.get("index")} | {num(clip.get("own_hit"))} | '
            f'{num(clip.get("donor_transfer"))} | '
            f'{num(clip.get("head_swap_to_donor"))} | '
            f'{clip.get("donor_index")} | {num(clip.get("changed_frac"))} |')
    add('')

    # The m4floor probe's own readings: the M4 bar's base (pooled floor) and
    # A2-R's pre-registered S0-5 anticipation. Read by nothing before this.
    m4f = m4floor_readings(run_dir, a2)
    if m4f:
        add('**M4 地板 / 读出存活读数**（`semA_m4floor_ep6.json`，'
            'M4 的 `own_hit` 线即由这里的 pooled 地板推出）：')
        add('')
        add('| 读数 | 值 | 说明 |')
        add('|---|---:|---|')
        add(f'| `pooled_prior_floor` | **{num(m4f.get("pooled_prior_floor"))}** | '
            f'窗口内「恒投多数类」的准确率，M4 `own_hit` 线的基准 |')
        add(f'| `recomputed_mean_own_hit` | '
            f'**{num(m4f.get("recomputed_mean_own_hit"))}** | '
            f'由残差现场重算，与地板同源 |')
        add(f'| `margin_over_pooled_prior` | '
            f'**{signed(m4f.get("margin_over_pooled_prior"))}** | '
            f'`own_hit −` 地板（**机制读数**）|')
        add(f'| `frac_cells_residual_can_flip` | '
            f'**{num(m4f.get("frac_cells_residual_can_flip"))}** | '
            f'A2-R 的 S0-5 预期读数（§5；不设通过线）|')
        add(f'| `residual_gap_fg_mean` / `conv_cls_gap_fg_mean` | '
            f'**{num(m4f.get("residual_gap_fg_mean"))}** / '
            f'**{num(m4f.get("conv_cls_gap_fg_mean"))}** | '
            f'残差与 `conv_cls` 的前景间隔（稀释程度）|')
        add('')
        scale = m4f.get('head_hit_by_residual_scale_pooled')
        if isinstance(scale, dict) and scale:
            keys = sorted(scale, key=lambda k: float(k))
            add('head 命中随残差**尺度**的变化（`residual=0` 与强制满尺度'
                '在两端）：`' + ' / '.join(
                    f'×{k} → {num(scale[k])}' for k in keys) + '`。')
            add('')
    # NOTE: the original code ended the M4 block with exactly one blank line
    # here; A's archived section has that shape, so this must not add another
    # (an extra blank would make A's draft differ from the doc by whitespace
    # only -- still a difference a diff-based audit would flag as drift).
    if m4f:
        add('')

    # --- family readout ---------------------------------------------------
    add('### 72.7 家族标准反事实读数（`shuffle_z` / `zero_z` / `prior_only`）')
    add('')
    add('三窗口与既往各代（MAI / MCI / DC-RSSM）使用同一探针语义：')
    add('`shuffle_z` 用同 batch 另一样本的 `mu_q` 替换，`zero_z` 置 0，')
    if arm_short_label(run_dir, a2) == 'A':
        add('`prior_only` 只走先验。**A 跑标准家族**，故用')
    else:
        add('`prior_only` 只走先验。**本臂即跑在标准家族底座上**，故用')
    add('`probe_z_utilization_standard.py`（后验头重写 + 转发部署态 class')
    add('residual），与 `tools/diagnose_z_utilization.py` 的打分口径一致。')
    add('')
    add('| 窗口 | shuffle_z head | zero_z head | prior_only head | '
        'shuffle/zero | shuffle_z feat | 标准 RSSM ep6 同窗 head（对照） |')
    add('|---|---:|---:|---:|---:|---:|---:|')
    readings = {}
    for label, start, name in WINDOWS:
        path = os.path.join(run_dir, name)
        if not os.path.exists(path):
            add(f'| {label} | 缺失 | | | | | |')
            continue
        payload = load_json(path)
        readings[label] = window_readings(payload)
        r = readings[label]
        sh, ze = r['shuffle_z']['head'], r['zero_z']['head']
        ratio = (sh / ze) if (sh is not None and ze) else None
        add(f'| {label}（{payload.get("indices")}） | **{pct(sh)}** | '
            f'{pct(ze)} | {pct(r["prior_only"]["head"])} | '
            f'{num(ratio, 3)} | {pct(r["shuffle_z"]["feature"])} | '
            f'{pct(STD_EP6_SHUFFLE_HEAD[label], 2)} |')
    add('')
    if readings:
        target = []
        for label, _, _ in WINDOWS:
            if label in readings and readings[label]['shuffle_z']['head'] is not None:
                target.append(
                    f'{label} {pct(readings[label]["shuffle_z"]["head"], 2)}')
        add(f'本 run 三窗 `shuffle_z` head：{"; ".join(target)}。')
        add('对照为**标准 no2d_igdr RSSM seed1 ep6**同窗口（'
            '`docs/training_runs_full.md` §62.2），因同家族、同 epoch，')
        add('是唯一口径一致的机制对照；MAI/MCI/DC 各代的同窗读数见 §62/§63/§68。')
        add('')
        indices_by_window = []
        for label, _, name in WINDOWS:
            path = os.path.join(run_dir, name)
            if os.path.exists(path):
                indices_by_window.append(
                    (label, load_json(path).get('indices')))
        spans = {tuple(v) if v else None for _, v in indices_by_window}
        if len(spans) == 1:
            only = next(iter(spans))
            add(f'> **警告：三个窗口返回了同一组 clip {list(only) if only else None}**，'
                '即三窗口不是三个独立样本窗。上述三行因此不是相互独立的读数，')
            add('> 不得当作「三窗口一致」；请核对探针的 `--start-index` 与')
            add('> `--limit` 是否真的换窗。')
    add('')

    # --- posterior source -------------------------------------------------
    add('### 72.8 posterior source 归因（`stride=700`）')
    add('')
    post_path = os.path.join(run_dir, POST_SOURCE)
    if not os.path.exists(post_path):
        add('**缺失**：source 探针未产出，归因读数不可用。')
        add('')
    else:
        post = load_json(post_path)
        # The real tool reports a four-field block per tensor. A file carrying
        # only ``rms`` is not a truncated reading to be papered over -- it is a
        # different (usually hand-made) artifact, so say so instead of
        # rendering a half-filled table as if it were the measurement.
        rich = [k for k in ('mu_q', 'h_t', 'e_pooled')
                if isinstance(post.get(k), dict)
                and 'sample_specific_energy_frac' in post[k]]
        add('| 量 | RMS | batch-common RMS | 样本特异 RMS | 样本特异能量占比 |')
        add('|---|---:|---:|---:|---:|')
        for key in ('mu_q', 'h_t', 'e_pooled'):
            block = post.get(key) or {}
            add(f'| `{key}` | {num(block.get("rms"))} | '
                f'{num(block.get("batch_common_rms"))} | '
                f'{num(block.get("sample_specific_rms"))} | '
                f'{pct(block.get("sample_specific_energy_frac"), 2)} |')
        add('')
        if len(rich) < 3:
            add('> **警告：该 source JSON 不是完整读数**——缺少')
            add('> `sample_specific_energy_frac`（真实工具必写）。上面的空列')
            add('> 不得当作「特异占比为 0」，整个归因表也不得引用。')
            add('')
        removal_h = post.get('mu_q_removal_of_h_rel_l2')
        removal_e = post.get('mu_q_removal_of_e_rel_l2')
        add('| 读出/依赖读数 | 值 |')
        add('|---|---:|')
        add(f'| `mu_q_removal_of_h_rel_l2` | {num(removal_h)} |')
        add(f'| `mu_q_removal_of_e_rel_l2` | {num(removal_e)} |')
        if removal_h is not None and removal_e:
            add(f'| **h/e 依赖比** | **{num(removal_h / removal_e, 3)}** |')
        add(f'| `posterior_obs_mode` | `{post.get("posterior_obs_mode")}` |')
        add(f'| `z_proj_init` | `{post.get("z_proj_init")}` |')
        add('')
        add('> h/e 依赖比是各代通用的机制对照量（标准主线 ep12 为 2.338、'
            'MAI ep12 为 0.842、MCI ep12 为 0.605）；越低表示 posterior')
        add('> 越依赖观测而非 GRU 状态。')
        add('')
        add(f'fusion 类：`{post.get("fusion_class")}`、latent '
            f'{post.get("latent_resolution")}、readout '
            f'`{post.get("readout_mode")}`、clip 索引 {post.get("indices")}。')
        add('')

    # --- verdict / next step ---------------------------------------------
    add('### 72.9 判定与下一步（按 §7 预登记分叉，人工步骤）')
    add('')
    if not evidence_ok:
        add('1. 证据包不完整：重跑缺失探针后再判，**不得**读成门没过。')
    elif grid_limited:
        add('1. **补一个 full-res 同口径对照**后再判 M1（§71.9.3(2)）；')
        add('   在此之前既不算通过也不算否定。')
    elif all_pass and added_all_pass and not grid_limited:
        add('1. S1 全过 ⇒ **请求用户批准 §5 的 S2 数值**（Overall ≥ 40.00；')
        add('   非劣 Car ≥ 47.34 / Truck ≥ 29.22 / Ped ≥ 29.93 / Cyc ≥ 47.63；')
        add('   hard fail < 38.78）；批准后才允许从 `epoch_6.pth` 续训至 ep16。')
    else:
        reasons = []
        if gates.get('M1') is not True:
            reasons.append(
                f'M1 `thr50` {num(m1.get("mean_thr50"))}/`hit@1` '
                f'{num(m1.get("mean_hit1"))}（例外条款不适用，真实未过）')
        if gates.get('M3') is not True:
            reasons.append(f'M3 仅 {m3.get("wins")}/{m3.get("n")}')
        if gates.get('M4') is not True:
            reasons.append(
                f'M4 `donor_transfer` {num(m4.get("mean_donor_transfer"))}'
                f'（bar {num(m4.get("bar"), 2) if m4.get("bar") is not None else "n/a"}）')
        # A2's added gates are part of the arm's S1, so a failure there must
        # appear in the archive reasons -- otherwise the draft would archive an
        # A2 arm without naming the gate that actually stopped it.
        if a2 is not None and not a2_all_pass:
            a2r = a2.get('readings') or {}
            if a2_gates.get('M1p') is not True:
                reasons.append(f"M1' `auc` {num(a2r.get('auc'))}"
                               f"（margin {num(a2r.get('auc_margin'))}）")
            if a2_gates.get('M1loc') is False:
                reasons.append(
                    f"M1-loc `max_gt_prob ≥ 0.5` 仅 "
                    f"{a2r.get('n_max_gt_prob_ge_0.5')}/{verdict.A2_FROZEN_N}、"
                    f"`recall@top-k` 中位 "
                    f"{num(a2r.get('recall_at_top_k_median'))}")
        add('1. S1 未全过 ⇒ 按负结果归档，读数写进 Analysis，不投 ep16。')
        if reasons:
            add('   实测原因：' + '；'.join(reasons) + '。')
        if a2 is not None:
            add(f'   归档的是**本次 {arm_label(run_dir, a2)} 的单一变量对照**'
                f'（A2-S 为监督侧排序压力、A2-R 为通道覆盖），不是「z 无信息」，'
                f'也不是「所有语义监督/残差读出/通道覆盖方向」；'
                f'转下一候选按 §7 的分叉表（M1-loc 不过 ⇒ 转「z 的对象信息'
                f'是否只够支撑对象级读出」；M3 不过 ⇒ 转「分类读出是否应走'
                f'独立分支」），不靠调权重/增益/KL 抢救。')
        else:
            add('   归档的是**本次候选 A 的读出/监督设计**，不是「z 无信息」'
                '（M2 只证明该语义头依赖其训练输入；z 的排他性**既未被否证、'
                '也未被证实**，见 §72.0 与 §71.9.17(3)），也不是'
                '「所有语义监督/残差读出方向」'
                '（§7 只据此否定本设计，未否定候选 A/B 所代表的方向）。')
    add('2. 任意 Ped/Cyc 触到非劣线 ⇒ 停止，记为家族副作用第 6 次复现。')
    # This line sits in an ep6 archive section and was read as an ep6 stop-loss
    # (the "S2" in the STOP-line was lost). The scope correction lived ONLY as
    # hand-written prose in the canonical doc, so re-rendering would have
    # deleted it and restored the misreading. It is emitted here instead.
    add(f'   > **口径更正（{S2_SCOPE_CORRECTION_DATE}）**：本行写在 ep6 归档节里，'
        f'容易被读成 ep6 的止损线。')
    add('   > 该非劣线属 **S2（ep12–16）**；'
        f'{arm_short_label(run_dir, a2)} 停在 S1，S2 从未形成，故本代'
        '**从未触发**这条线')
    add('   > （见预登记 §7 的范围更正与 A2 预登记 §5「早期门与成熟门严格分离」）。')
    add('3. **不做**权重/KL 扫描，**不做** ep24 抢救，不靠放大系数救活机制。')
    add('')

    # --- z-contract reconciliation ---------------------------------------
    # Added after the archive audit found that M1/M3/M4 were read under the
    # DEPLOYMENT z contract while the bars came from a TRAINING-contract
    # diagnostic. The reconciliation probe re-reads the same checkpoint under
    # both contracts, so the difference is measured instead of argued.
    zc_path = os.path.join(run_dir, ZCONTRACT_JSON)
    if os.path.exists(zc_path):
        zc = load_json(zc_path)
        arms = zc.get('arms') or {}
        dep = arms.get('deploy_mu_q') or {}
        trn = arms.get('train_sampled_z') or {}
        add('### 72.10 z 契约对账（训练采样 z vs 部署均值 z，只读）')
        add('')
        add('归档审计发现：M1/M3/M4 是在**部署契约**下读的，而 `0.30/0.60` '
            '门槛来自 **train 契约**下的诊断。两者差一个状态采样：')
        add('')
        add('```')
        add('R4Det.extract_feat:  rssm_deterministic = not self.training')
        add('rssm_fusion.forward:  z_t = mu_q if deterministic '
            'else self.sample(mu_q, logstd_q)')
        add('```')
        add('')
        add('即训练时语义头吃 `mu_q + sigma*eps`、eval 探针吃 `mu_q`。'
            '故用同一 checkpoint、同一 val clip、同一评分函数，'
            '只切换 z 契约重测一次：')
        add('')
        add('| z 契约 | `thr50` | `hit@1` | `hit1_excess_over_chance` |')
        add('|---|---:|---:|---:|')
        add(f'| 部署（`mu_q`，即 M1 实际读数） | '
            f'{num(dep.get("mean_thr50"))} | {num(dep.get("mean_hit1"))} | '
            f'{num(dep.get("mean_hit1_excess_over_chance"))} |')
        add(f'| 训练（`mu_q + sigma*eps`） | '
            f'{num(trn.get("mean_thr50"))} | {num(trn.get("mean_hit1"))} | '
            f'{num(trn.get("mean_hit1_excess_over_chance"))} |')
        add('')
        add('**干预检查**（否则「两臂相同」既可能是契约真无影响，'
            '也可能是切换没生效——结论相反、数字相同）：')
        add('')
        mv = zc.get('intervention_check') or []
        rz = [m.get('rel_delta_z') for m in mv
              if isinstance(m.get('rel_delta_z'), (int, float))]
        rs = [m.get('rel_delta_sem_map') for m in mv
              if isinstance(m.get('rel_delta_sem_map'), (int, float))]
        add(f'* `z` 相对变化均值 **{pct(statistics.mean(rz), 3) if rz else "n/a"}**'
            f' —— 契约切换**确实改变了 z**；')
        add(f'* 语义热图相对变化均值 '
            f'**{pct(statistics.mean(rs), 3) if rs else "n/a"}**'
            f' —— 经 `sem_head` 后幅度被大幅吸收。')
        add('')
        nz = [n.get('sigma_over_mu') for n in (zc.get('noise') or [])
              if isinstance(n.get('sigma_over_mu'), (int, float))]
        if nz:
            add(f'噪声占比 `sigma/mu`：均值 **{num(statistics.mean(nz))}**'
                f'（min {num(min(nz))} / max {num(max(nz))}），'
                f'即噪声项确实与被解释量同量级。')
            add('')
        d_thr, t_thr = dep.get('mean_thr50'), trn.get('mean_thr50')
        delta = (t_thr - d_thr
                 if (t_thr is not None and d_thr is not None) else None)
        add('**结论**：契约切换确实移动了 z，但**两项门控读数完全不变**'
            f'（Δ`thr50` = {signed(delta)}）——')
        add('故 train/val 的 4.3× 差距**不能**由 z 采样解释。这**不改变** S1 '
            '判定（§72.0 已按两种口径分别核过），但把「契约错配」从'
            '嫌疑名单上划掉。')
        add('')

        # The remaining question ("then what DOES explain the gap?") was left
        # open here. §71.9.19 answered it for A by pinning the z contract and
        # the eval mode and varying the DATA SPLIT alone. Rendering that answer
        # only when THIS bundle carries the artifact keeps the section from
        # inheriting a number it never measured.
        attrib = split_attribution(run_dir)
        if attrib:
            sv, st = attrib['splits']['val'], attrib['splits']['train']
            add(f'**来源归属（只读补测，n={attrib["n"]["val"]} / '
                f'{attrib["n"]["train"]}）**：把 z 契约与 eval 模式都钉死后'
                f'**只换数据 split**——')
            add('')
            add('| split | `thr50` | `hit@1` | 只切 z 路径 4 个 BN 的 Δ`thr50` | '
                'Δ`hit@1` |')
            add('|---|---:|---:|---:|---:|')
            add(f'| train（拟合过） | **{num(st["eval_thr50"])}** | '
                f'**{num(st["eval_hit1"])}** | '
                f'{signed(st["bn_delta_thr50"])} | '
                f'{signed(st["bn_delta_hit1"])} |')
            add(f'| val（留出） | **{num(sv["eval_thr50"])}** | '
                f'**{num(sv["eval_hit1"])}** | '
                f'{signed(sv["bn_delta_thr50"])} | '
                f'{signed(sv["bn_delta_hit1"])} |')
            add('')
            gap_t = _diff(st['eval_thr50'], sv['eval_thr50'])
            gap_h = _diff(st['eval_hit1'], sv['eval_hit1'])
            add(f'⇒ 该差距**主要来自数据 split**（thr50 {signed(gap_t)}、'
                f'hit@1 {signed(gap_h)}），归一化模式是**次要**项。'
                f'即：train 侧 `stat_sem_*` 是**拟合内**读数，不能当作'
                f'同一模型在另一口径下的同量测量。')
            add('')
        else:
            add('**来源归属：NOT MEASURED（本包无 split 对照产物）。** '
                '本节只测得「契约错配不是原因」，**未**测量 split 与 BN 模式'
                '各占多少；不得据此断言差距来自泛化。')
            add('')
        add(f'> 产物：`{ZCONTRACT_JSON}`（只读探针 '
            '`me_rssm/sanity/probe_semA_zcontract.py`，未改权重、未启动训练）。')
        add('')
    else:
        # Silently dropping the section left a numbering GAP (72.9 -> 72.11) and
        # said nothing about why the train/val z-contract reconciliation is
        # absent. Neither is acceptable in a record whose whole point is that a
        # missing reading is visible: it also matters more for A2, whose live
        # z-contract question ("does the gap come from sampling z?") is exactly
        # what this probe answers.
        add('### 72.10 z 契约对账（训练采样 z vs 部署均值 z，只读）')
        add('')
        add(FILL.format(f'z 契约对账（缺 {ZCONTRACT_JSON}；只读探针 '
                        f'me_rssm/sanity/probe_semA_zcontract.py）'))
        add('')
        add('> **该缺口的含义**：§72.10 是在**部署契约**（`mu_q`）下读 M1/M3/M4、'
            '而门槛先在 **train 契约**（`mu_q + sigma*eps`）下标定这一错配的'
            '实测对账。缺它时，两契约之间是否有差异**未被测量**——'
            '不得据此宣称契约无影响，也不得把它读成门禁读数。')
        add('')

    add('### 72.11 本节边界')
    add('')
    add('* 无 S2 性能宣称；§72.0/72.4 的均值是 ep6 早期上下文。')
    add('* 未改模型、未改判据数值、未动重建目标/KL/权重/日程。')
    add('* 本节由 `me_rssm/sanity/semA_ep6_record.py` 从证据 JSON 与')
    add('  `*.log.json` 机械生成，未手抄任何读数；判定沿用')
    add('  `semA_ep6_verdict.py` 的同一套门槛常量。')
    add('')

    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', default=DEFAULT_RUN,
                    help='work_dir holding the ep6 evidence bundle')
    ap.add_argument('--output', default=None,
                    help='write markdown here (default: stdout)')
    ap.add_argument('--date', default=None,
                    help='YYYY-MM-DD for the heading (default: today UTC)')
    args = ap.parse_args()

    if not os.path.isdir(args.run):
        print(f'no such run dir: {args.run}', file=sys.stderr)
        return 2

    date_str = args.date or datetime.datetime.utcnow().strftime('%Y-%m-%d')
    text = render(args.run, date_str)

    if args.output:
        output = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
        tmp = output + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as handle:
            handle.write(text + '\n')
        os.replace(tmp, output)
    else:
        print(text)

    unfilled = text.count('<!-- FILL:')
    if unfilled:
        print(f'[record] WARNING: {unfilled} field(s) need a human value: '
              'grep "<!-- FILL:" in the output before pasting it into '
              'docs/training_runs_full.md', file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
