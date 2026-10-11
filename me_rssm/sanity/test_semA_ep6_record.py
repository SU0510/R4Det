"""CPU unit tests for the section-72 training-record generator.

The generator exists because AGENTS rules 5 and 6 require the three summary
readings (fixed-window mean with true span and n, all-epoch peak as
"value @ epoch", best saved listed separately) PLUS the full per-epoch curve
and per-class breakdown -- and hand-assembling that is where this project has
already mis-transcribed numbers. These tests pin the properties that make the
output safe to paste into docs/training_runs_full.md:

  * the three readings are always present and are never a bare "BEST";
  * the fixed-window mean always shows its actual span and n;
  * peak is weight-independent and best-saved is read from disk, separately;
  * the S1 verdict follows the shared gate module, including the
    pre-registered M1 grid-extrapolation exception;
  * an incomplete evidence bundle can never render as ALL PASS;
  * every field the artifacts cannot supply is marked as a visible FILL;
  * the three z-util windows are labelled with their own clip indices, and
    an identical-index bundle raises a warning instead of reading as
    "three consistent windows".

Run: python3 me_rssm/sanity/test_semA_ep6_record.py
"""

import importlib.util
import json
import os
import shutil
import sys
import tempfile
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


def _read(rel):
    """Read a repo file as text (used to pin *structural* properties).

    Some invariants cannot be observed from a rendered section -- e.g. "the
    train-caliber reading is derived from the log instead of pasted as a
    constant". Those are pinned by reading the generator source itself.
    """
    with open(os.path.join(_REPO, rel), encoding='utf-8') as handle:
        return handle.read()


record = _load('record_ep6', 'me_rssm/sanity/semA_ep6_record.py')
verdict = record.verdict


def _clip(index, thr50=0.5, hit1=0.7, num_pos_cells=4):
    return {'index': index, 'thr50': thr50, 'hit1': hit1,
            'num_pos_cells': num_pos_cells, 'occupied_class_channels': 2,
            'hit1_chance': 0.0001, 'hit1_excess_over_chance': hit1 - 0.0001}


def _gate_payload(**gates):
    resolved = {'M1': True, 'M2': True, 'M3': True, 'M4': True}
    for name in ('M1', 'M2'):
        if name in gates:
            resolved[name] = gates.pop(name)
    resolved.update(gates)
    return {
        'gates': resolved,
        'M1_sem_head': {'mean_thr50': 0.5, 'mean_hit1': 0.7,
                        'per_clip': [_clip(3), _clip(203)]},
        'M2_exclusivity': {'z_mean_thr50': 0.5, 'h_mean_thr50': 0.1,
                           'feat_mean_thr50': 0.1,
                           'h_relative': -0.8, 'feat_relative': -0.8,
                           'per_clip_h': [_clip(3, 0.1, 0.1)],
                           'per_clip_feat': [_clip(3, 0.1, 0.1)]},
        'M3_readout_toggle': {'wins': 9, 'n': 10,
                              'per_clip': [{'index': 3, 'ce_on': 1.0,
                                            'ce_off': 1.2, 'cells': 4,
                                            'delta_off_minus_on': 0.2}]},
        'M4_class_information': {'chance': 0.25, 'bar': 0.45,
                                 'mean_own_hit': 0.8,
                                 'mean_donor_transfer': 0.7,
                                 'mean_head_swap_to_donor': 0.7,
                                 'mean_changed_frac': 1.0,
                                 'per_clip': [{'index': 3, 'own_hit': 0.8,
                                               'donor_transfer': 0.7,
                                               'head_swap_to_donor': 0.7,
                                               'donor_index': 203,
                                               'changed_frac': 1.0}]},
        'checkpoint': '/data/x/epoch_6.pth',
        'config': '/repo/me_rssm/configs/x.py',
        'indices': [3, 203],
        'shuffle_offset': 1,
        'input_fingerprints': {'checkpoint_sha256': 'a' * 64,
                               'checkpoint_size_bytes': 1,
                               'checkpoint_mtime': 2,
                               'config_sha256': 'b' * 64},
    }


def _z_util(indices, shuffle=0.05, zero=0.2):
    return {
        'indices': list(indices),
        'per_sample': {
            str(i): {'variants': {
                'shuffle_z': {'head_l2_ratio_vs_normal': shuffle,
                              'feature_l2_ratio_vs_normal': 0.01},
                'zero_z': {'head_l2_ratio_vs_normal': zero,
                           'feature_l2_ratio_vs_normal': 0.9},
                'prior_only': {'head_l2_ratio_vs_normal': 0.03,
                               'feature_l2_ratio_vs_normal': 0.3},
            }} for i in indices},
    }


class _Fixture:
    """A run dir with a complete battery and a 6-epoch val curve."""

    def __init__(self, epochs=6, start_index=2):
        self.dir = tempfile.mkdtemp(prefix='rec72_')
        self.start_index = start_index
        self._write(epochs)

    def _write(self, epochs):
        gate = _gate_payload()
        self.dump(verdict.GATE_JSON, gate)
        for (_, start, name) in record.WINDOWS:
            self.dump(name, _z_util(range(start, start + 2)))
        self.dump(record.ZCONTRACT_JSON, {
            'probe': 'candidate A z-contract reconciliation (read-only)',
            'indices': [3, 203],
            'arms': {
                'deploy_mu_q': {
                    'mean_thr50': 0.1167, 'mean_hit1': 0.0833,
                    'mean_hit1_excess_over_chance': 0.0833, 'per_clip': []},
                'train_sampled_z': {
                    'mean_thr50': 0.1167, 'mean_hit1': 0.0833,
                    'mean_hit1_excess_over_chance': 0.0833, 'per_clip': []},
            },
            'intervention_check': [
                {'index': 3, 'rel_delta_z': 0.3535,
                 'rel_delta_sem_map': 0.0132}],
            'noise': [{'index': 3, 'sigma_over_mu': 0.3538}],
        })
        self.dump(record.POST_SOURCE, {
            'indices': [3, 703, 1403],
            'fusion_class': 'SemanticReadoutRSSMFusion',
            'latent_resolution': [216, 248],
            'readout_mode': 'film',
            'mu_q': {'rms': 0.5}, 'h_t': {'rms': 0.4},
            'e_pooled': {'rms': 0.8}})
        with open(os.path.join(self.dir, 'run.log.json'), 'w') as handle:
            for e in range(1, epochs + 1):
                overall = 30.0 + e
                handle.write(json.dumps({
                    'mode': 'val', 'epoch': e, 'iter': 10,
                    'pts_bbox/KITTI/Overall_3D_moderate': overall,
                    'pts_bbox/KITTI/Car_3D_moderate_strict': 40.0 + e,
                    'pts_bbox/KITTI/Truck_3D_moderate_strict': 20.0 + e,
                    'pts_bbox/KITTI/Pedestrian_3D_moderate_loose': 25.0 + e,
                    'pts_bbox/KITTI/Cyclist_3D_moderate_loose': 45.0 + e,
                }) + '\n')
                handle.write(json.dumps({
                    'mode': 'train', 'epoch': e, 'iter': 1900,
                    'time': 1.5 + 0.05 * e, 'memory': 17106,
                    'stat_sem_hit1': 0.1 * e, 'stat_sem_thr50': 0.05 * e,
                    'loss_rssm_sem': 1.0 / e, 'stat_cls_res_rms': 0.5,
                    'stat_mu_diff_sq': 0.2,
                }) + '\n')

    def dump(self, name, payload):
        with open(os.path.join(self.dir, name), 'w') as handle:
            json.dump(payload, handle)

    def load(self, name):
        with open(os.path.join(self.dir, name)) as handle:
            return json.load(handle)

    def add_checkpoint(self, epoch):
        path = os.path.join(self.dir, f'epoch_{epoch}.pth')
        with open(path, 'w') as handle:
            handle.write('x')
        return path

    def render(self):
        return record.render(self.dir, '2026-10-10')

    def close(self):
        shutil.rmtree(self.dir, ignore_errors=True)


class TestSummaryReadings(unittest.TestCase):
    def setUp(self):
        self.fx = _Fixture(epochs=6)

    def tearDown(self):
        self.fx.close()

    def test_window_mean_shows_true_span_and_n(self):
        text = self.fx.render()
        # Span ep1-6 (the default 12-16 window does not exist yet), n=6, and
        # the mean of 31..36 is 33.5.
        self.assertIn('实际跨度 ep1-6，n=6', text)
        self.assertIn('**33.5000**', text)

    def test_peak_is_labelled_with_epoch_and_is_weight_independent(self):
        text = self.fx.render()
        self.assertIn('**36.0000 @ ep6**', text)
        # No checkpoint exists; peak must still be reported.
        self.assertIn('全轮峰值', text)

    def test_best_saved_is_separate_and_comes_from_disk(self):
        self.fx.add_checkpoint(2)
        self.fx.add_checkpoint(4)
        text = self.fx.render()
        # ep4 is the highest SAVED epoch (34.0), NOT the global peak (36.0).
        self.assertIn('best saved', text)
        self.assertIn('**34.0000 @ ep4**', text)
        self.assertIn('不参与性能排名', text)

    def test_no_bare_best_label(self):
        text = self.fx.render()
        for line in text.splitlines():
            if 'BEST' not in line:
                continue
            # The generator legitimately quotes the word while forbidding it.
            # What must never appear is BEST *as a reading label*, i.e. a table
            # row or bullet that assigns it a number.
            stripped = line.strip()
            if stripped.startswith('|') or stripped.startswith('*'):
                self.fail(f'ambiguous bare BEST label: {line!r}')
            self.assertIn('代替其中任一项', line)

    def test_per_epoch_curve_lists_every_epoch(self):
        text = self.fx.render()
        for e in range(1, 7):
            self.assertRegex(text, rf'\|\s*{e}\s*\|\s*{30 + e}\.00\s*\|')

    def test_per_class_window_breakdown_present(self):
        text = self.fx.render()
        for name in ('Car_strict', 'Truck_strict', 'Ped_loose', 'Cyc_loose'):
            self.assertIn(name, text)

    def test_train_stats_table_lists_every_epoch(self):
        text = self.fx.render()
        section = text.split('### 72.5')[1].split('### 72.6')[0]
        for e in range(1, 7):
            self.assertIn(f'| {e} |', section)


class TestVerdictFollowsTheSharedGate(unittest.TestCase):
    def setUp(self):
        self.fx = _Fixture()

    def tearDown(self):
        self.fx.close()

    def test_all_pass_requests_s2_approval_and_forbids_training(self):
        text = self.fx.render()
        self.assertIn('ALL PASS', text)
        self.assertIn('请求用户批准 §5 的 S2 数值', text)
        self.assertIn('批准后才允许', text)

    def test_failing_gate_archives_negative(self):
        gate = self.fx.load(verdict.GATE_JSON)
        gate['gates']['M3'] = False
        self.fx.dump(verdict.GATE_JSON, gate)
        text = self.fx.render()
        self.assertIn('NOT ALL PASS（M3）', text)
        self.assertIn('不投 ep16', text)

    def test_m2_failure_voids_the_probabilistic_state_claim(self):
        gate = self.fx.load(verdict.GATE_JSON)
        gate['gates']['M2'] = False
        self.fx.dump(verdict.GATE_JSON, gate)
        text = self.fx.render()
        self.assertIn('概率状态', text)
        self.assertIn('作废', text)

    def test_grid_limited_m1_is_insufficient_evidence_not_refutation(self):
        gate = self.fx.load(verdict.GATE_JSON)
        gate['gates']['M1'] = False
        gate['M1_sem_head']['mean_thr50'] = 0.4   # clears the 0.30 bar
        gate['M1_sem_head']['mean_hit1'] = 0.05   # misses the 0.60 bar
        gate['M1_sem_head']['per_clip'] = [
            _clip(3, 0.4, 0.05), _clip(203, 0.4, 0.05)]
        for clip in gate['M1_sem_head']['per_clip']:
            clip['hit1_excess_over_chance'] = 0.99  # healthy excess
        self.fx.dump(verdict.GATE_JSON, gate)
        text = self.fx.render()
        self.assertIn('证据不足', text)
        self.assertIn('不是机制否定', text)
        self.assertIn('full-res 同口径对照', text)

    def test_grid_limited_branch_needs_every_other_gate_to_pass(self):
        gate = self.fx.load(verdict.GATE_JSON)
        gate['gates']['M1'] = False
        gate['gates']['M4'] = False
        gate['M1_sem_head']['mean_thr50'] = 0.4
        gate['M1_sem_head']['mean_hit1'] = 0.05
        for clip in gate['M1_sem_head']['per_clip']:
            clip['hit1_excess_over_chance'] = 0.99
        self.fx.dump(verdict.GATE_JSON, gate)
        text = self.fx.render()
        self.assertIn('NOT ALL PASS（M1, M4）', text)
        self.assertNotIn('不是机制否定', text)


class TestEvidenceIntegrity(unittest.TestCase):
    def setUp(self):
        self.fx = _Fixture()

    def tearDown(self):
        self.fx.close()

    def test_incomplete_bundle_is_never_all_pass(self):
        os.remove(os.path.join(self.fx.dir, 'z_util_std_ep6_w100.json'))
        text = self.fx.render()
        self.assertIn('NOT EVALUABLE', text)
        self.assertNotIn('S1 聚合判定：ALL PASS', text)

    def test_missing_bundle_also_forbids_the_negative_archive(self):
        os.remove(os.path.join(self.fx.dir, 'z_util_std_ep6_w100.json'))
        text = self.fx.render()
        self.assertIn('不得**读成门没过', text)

    def test_unfillable_run_facts_are_marked(self):
        text = self.fx.render()
        self.assertIn('<!-- FILL:', text)

    def test_fill_markers_are_counted_on_stderr(self):
        """main() must warn; otherwise FILL markers get pasted verbatim."""
        with open(os.path.join(
                _REPO, 'me_rssm', 'sanity', 'semA_ep6_record.py'),
                encoding='utf-8') as handle:
            source = handle.read()
        self.assertIn("text.count('<!-- FILL:')", source)


class TestRunFactDerivation(unittest.TestCase):
    """The §72.2 run facts are DERIVED from the run's own logs.

    They used to be ``<!-- FILL -->`` markers, i.e. hand-typed timestamps.
    Section 72.11 promises "未手抄任何读数", and hand transcription is the
    error class this generator exists to remove -- so the values come from
    the artifacts whenever the logs are present.
    """

    def setUp(self):
        self.fx = _Fixture()
        self.addCleanup(self.fx.close)

    def _write_logs(self):
        d = self.fx.dir
        with open(os.path.join(d, '20261010_145003.log'), 'w') as h:
            h.write('2026-10-10 14:50:03,170 - mmdet - INFO - start\n')
            h.write('noise line\n')
            h.write('2026-10-10 20:56:08,141 - mmdet - INFO - end\n')
        with open(os.path.join(d, 'stdout.log'), 'w') as h:
            h.write('2026-10-10 14:50:03,170 - mmdet - INFO - start\n')
            h.write("    raise SignalException(f\"Process {os.getpid()}"
                    " got signal: {sigval}\", sigval=sigval)\n")
            h.write('SignalException: Process 1573258 got signal: 2\n')
            h.write('EXIT=1\n')
        with open(os.path.join(d, record.WATCH_LOG), 'w') as h:
            h.write('[watch] 2026-10-10 20:56:29 gate hit (fresh ckpt); '
                    'stopping\n')
            h.write('[watch] 2026-10-10 20:57:15 GPUs free after 45s; '
                    'running probes\n')
            # The real watchdog log also carries the probes' own timestamped
            # mmcv output, i.e. it IS a timestamped span -- so "first *.log"
            # selection would silently report the PROBE window as training.
            h.write('2026-10-10 20:57:25,983 - mmcv - INFO - probe output\n')
            h.write('2026-10-10 20:58:54,100 - mmcv - INFO - probe output\n')
            h.write('[watch] 2026-10-10 20:58:57 probes done; manual step\n')

    def test_span_is_derived_and_no_fill_remains(self):
        self._write_logs()
        text = self.fx.render()
        self.assertIn('2026-10-10 14:50:03 → 2026-10-10 20:56:08', text)
        self.assertNotIn('<!-- FILL:', text)

    def test_train_log_is_the_runs_own_not_stdout_or_watch(self):
        self._write_logs()
        self.assertEqual(record.train_log_name(self.fx.dir),
                         '20261010_145003.log')

    def test_train_log_ignores_a_log_that_sorts_earlier(self):
        """A decoy that sorts BEFORE the run log (and a log with no
        timestamps) must not be picked -- taking the first *.log would."""
        self._write_logs()
        with open(os.path.join(self.fx.dir, '0_decoy.log'), 'w') as h:
            h.write('2020-01-01 00:00:00 - decoy - start\n')
            h.write('2020-01-01 00:00:01 - decoy - end\n')
        self.assertEqual(record.train_log_name(self.fx.dir),
                         '20261010_145003.log')
        text = self.fx.render()
        self.assertIn('2026-10-10 14:50:03 → 2026-10-10 20:56:08', text)
        self.assertNotIn('2020-01-01 00:00:00 →', text)

    def test_train_log_never_falls_back_to_the_watch_log(self):
        """The watchdog log starts AFTER the interruption; using it would
        report the PROBE window as the training window."""
        self._write_logs()
        os.remove(os.path.join(self.fx.dir, '20261010_145003.log'))
        os.remove(os.path.join(self.fx.dir, 'stdout.log'))
        self.assertIsNone(record.train_log_name(self.fx.dir))

    def test_stdout_is_an_acceptable_stand_in(self):
        """stdout.log is the same process's stdout, so its span matches."""
        self._write_logs()
        os.remove(os.path.join(self.fx.dir, '20261010_145003.log'))
        self.assertEqual(record.train_log_name(self.fx.dir), 'stdout.log')


    def test_throughput_is_the_median_of_train_records(self):
        secs, mem, n = record.train_throughput(self.fx.dir)
        self.assertEqual(n, 6)          # one per fixture epoch
        self.assertAlmostEqual(secs, 1.675)     # median of 1.55..1.80
        self.assertAlmostEqual(mem, 17106.0)

    def test_throughput_ignores_val_records(self):
        """Only train records carry time/memory; a val-only bundle must not
        silently produce a throughput reading."""
        with open(os.path.join(self.fx.dir, 'run.log.json')) as h:
            lines = [json.loads(x) for x in h if x.strip()]
        with open(os.path.join(self.fx.dir, 'run.log.json'), 'w') as h:
            for rec in lines:
                if rec.get('mode') == 'train':
                    continue
                h.write(json.dumps(rec) + '\n')
        self.assertIsNone(record.train_throughput(self.fx.dir))

    def test_watch_facts_take_the_real_signal_not_the_traceback_literal(self):
        """The raising frame's own source line also contains "got signal:".

        Taking the FIRST match yields the literal ``{sigval}`` and renders
        garbage into the record.
        """
        self._write_logs()
        gate, free, done, signal, exit_code = record.watch_facts(self.fx.dir)
        self.assertEqual(gate, '2026-10-10 20:56:29')
        self.assertEqual(free, '2026-10-10 20:57:15')
        self.assertEqual(done, '2026-10-10 20:58:57')
        self.assertEqual(signal, 2)
        self.assertEqual(exit_code, 1)

    def test_signal_is_named_and_never_renders_the_raw_literal(self):
        self._write_logs()
        text = self.fx.render()
        self.assertIn('SIGINT(2)', text)
        self.assertNotIn('sigval', text)

    def test_missing_logs_still_fall_back_to_visible_fill(self):
        text = self.fx.render()          # no log files written at all
        self.assertIn('<!-- FILL:', text)


class TestM1ExceptionIsAdjudicated(unittest.TestCase):
    """The prereg's cross-grid exception must be recorded as APPLIED or NOT.

    "An exception exists" is not a finding. If the record only ever describes
    the exception, a genuine miss can be read as "口径未校准" -- the post-hoc
    escape section 7 forbids. The generator therefore states the adjudication
    from the shared threshold constants in semA_ep6_verdict.
    """

    def setUp(self):
        self.fx = _Fixture()
        self.addCleanup(self.fx.close)

    def _fail_m1_flatly(self):
        gate = self.fx.load(verdict.GATE_JSON)
        gate['gates']['M1'] = False
        gate['M1_sem_head']['mean_thr50'] = 0.1167
        gate['M1_sem_head']['mean_hit1'] = 0.0833
        gate['M1_sem_head']['per_clip'] = [
            _clip(3, 0.1, 0.05), _clip(203, 0.12, 0.1)]
        self.fx.dump(verdict.GATE_JSON, gate)

    def test_not_applicable_is_stated_with_both_missed_bars(self):
        self._fail_m1_flatly()
        text = self.fx.render()
        self.assertIn('例外条款：不适用', text)
        self.assertIn('真实未过', text)
        # both bars quoted, so the reader can check the adjudication
        self.assertIn('0.3000', text)
        self.assertIn('0.5840', text)

    def test_applicable_exception_is_not_called_a_genuine_miss(self):
        """The exception is a NARROW band: thr50 clears 0.30 and hit@1 sits
        just under the 0.60 bar, at/above the 0.584 the calibrated bar
        implies. A miss well below the band is a genuine miss."""
        gate = self.fx.load(verdict.GATE_JSON)
        gate['gates']['M1'] = False
        gate['M1_sem_head']['mean_thr50'] = 0.40       # cleared 0.30
        gate['M1_sem_head']['mean_hit1'] = 0.59        # missed 0.60, in band
        gate['M1_sem_head']['per_clip'] = [
            _clip(3, 0.4, 0.59), _clip(203, 0.4, 0.59)]
        self.fx.dump(verdict.GATE_JSON, gate)
        text = self.fx.render()
        self.assertIn('例外条款：适用', text)
        self.assertNotIn('真实未过', text)

    def test_exception_band_is_bounded_below_by_the_excess_bar(self):
        """Regression guard on the adjudication itself: drop hit@1 to 0.50
        (thr50 still 0.40) and the SAME fixture must flip to 'not
        applicable'. Without this, the exception could silently widen into
        'any M1 miss with ok thr50 is inconclusive'."""
        gate = self.fx.load(verdict.GATE_JSON)
        gate['gates']['M1'] = False
        gate['M1_sem_head']['mean_thr50'] = 0.40
        gate['M1_sem_head']['mean_hit1'] = 0.50
        gate['M1_sem_head']['per_clip'] = [
            _clip(3, 0.4, 0.50), _clip(203, 0.4, 0.50)]
        self.fx.dump(verdict.GATE_JSON, gate)
        self.assertIn('例外条款：不适用', self.fx.render())

    def test_passing_m1_gets_no_exception_paragraph(self):
        text = self.fx.render()          # fixture passes every gate
        self.assertNotIn('例外条款：适用', text)
        self.assertNotIn('真实未过', text)


class TestM2PassIsStatedPositively(unittest.TestCase):
    """M2 passing must record what SURVIVES, not just what failed."""

    def setUp(self):
        self.fx = _Fixture()
        self.addCleanup(self.fx.close)

    def test_m2_pass_keeps_the_probabilistic_claim_open(self):
        """M2 passing must be stated SCOPED, not as an endorsement.

        It used to read "未被本次测量否证", which invites the reader to treat
        an untested claim as a surviving one. The correct statement is that
        the claim is neither refuted nor confirmed: M2 only shows the head
        depends on its trained input.
        """
        gate = self.fx.load(verdict.GATE_JSON)
        gate['gates']['M3'] = False       # force NOT ALL PASS
        self.fx.dump(verdict.GATE_JSON, gate)
        text = self.fx.render()
        self.assertIn('M2 通过', text)
        self.assertIn('既未被否证、也未被证实', text)
        self.assertNotIn('这一宣称作废', text)

    def test_m2_fail_voids_the_probabilistic_claim(self):
        gate = self.fx.load(verdict.GATE_JSON)
        gate['gates']['M2'] = False
        gate['M2_exclusivity']['h_relative'] = -0.1
        self.fx.dump(verdict.GATE_JSON, gate)
        text = self.fx.render()
        # Scope to §72.0: the scoped wording legitimately appears in §72.9's
        # archive-scope sentence in every variant, so a whole-document
        # assertNotIn would be asserting the wrong thing.
        section = text.split('### 72.0')[1].split('### 72.1')[0]
        self.assertIn('这一宣称作废', section)
        self.assertNotIn('既未被否证、也未被证实', section)


class TestClipWeightCaveat(unittest.TestCase):
    """A 1-target clip must not be able to carry a grouped conclusion."""

    def setUp(self):
        self.fx = _Fixture()
        self.addCleanup(self.fx.close)

    def test_unequal_clip_weights_are_flagged(self):
        gate = self.fx.load(verdict.GATE_JSON)
        gate['M1_sem_head']['per_clip'] = [
            _clip(3, num_pos_cells=11), _clip(203, num_pos_cells=1)]
        self.fx.dump(verdict.GATE_JSON, gate)
        text = self.fx.render()
        self.assertIn('样本权重不均提示', text)
        self.assertIn('不得由单个 clip 单独支撑', text)
        self.assertIn('`203`', text)      # the 1-target clip is named

    def test_uniform_clip_weights_raise_no_note(self):
        gate = self.fx.load(verdict.GATE_JSON)
        gate['M1_sem_head']['per_clip'] = [
            _clip(3, num_pos_cells=4), _clip(203, num_pos_cells=4)]
        self.fx.dump(verdict.GATE_JSON, gate)
        self.assertNotIn('样本权重不均提示', self.fx.render())


class TestNegativeArchiveNamesItsReasons(unittest.TestCase):
    """Section 7 says archive "把读数写进 Analysis" -- so the fork line must
    carry the measured reasons, not just the word 负结果. It must also scope
    what is being rejected, since M2 passing means z DOES carry the signal.
    """

    def setUp(self):
        self.fx = _Fixture()
        self.addCleanup(self.fx.close)

    def test_reasons_are_named_per_failing_gate(self):
        gate = self.fx.load(verdict.GATE_JSON)
        gate['gates'].update({'M1': False, 'M3': False, 'M4': False})
        gate['M1_sem_head']['mean_thr50'] = 0.1167
        gate['M1_sem_head']['mean_hit1'] = 0.0833
        gate['M1_sem_head']['per_clip'] = [_clip(3, 0.12, 0.08)]
        gate['M3_readout_toggle'] = {'wins': 7, 'n': 10, 'per_clip': []}
        gate['M4_class_information']['mean_donor_transfer'] = 0.3818
        self.fx.dump(verdict.GATE_JSON, gate)
        text = self.fx.render()
        self.assertIn('实测原因', text)
        self.assertIn('0.1167', text)
        self.assertIn('7/10', text)
        self.assertIn('0.3818', text)

    def test_archive_scope_excludes_what_M2_proved(self):
        gate = self.fx.load(verdict.GATE_JSON)
        gate['gates']['M1'] = False
        gate['M1_sem_head']['mean_thr50'] = 0.1
        self.fx.dump(verdict.GATE_JSON, gate)
        text = self.fx.render()
        self.assertIn('不是「z 无信息」', text)


class TestZContractBlock(unittest.TestCase):
    """Section 72.10 records the train-vs-deploy z contract reconciliation.

    The archive audit found the gate compared a deployment-contract reading
    against training-calibrated bars. The block must show BOTH contracts AND
    must carry the intervention check -- two identical arms are otherwise
    ambiguous between "the contract makes no difference" and "the switch did
    nothing", which are opposite conclusions from the same numbers.
    """

    def setUp(self):
        self.fx = _Fixture()
        self.addCleanup(self.fx.close)

    def test_block_renders_both_arms(self):
        text = self.fx.render()
        self.assertIn('### 72.10 z 契约对账', text)
        self.assertIn('部署（`mu_q`', text)
        self.assertIn('训练（`mu_q + sigma*eps`）', text)

    def test_intervention_check_is_reported(self):
        text = self.fx.render()
        self.assertIn('干预检查', text)
        self.assertIn('确实改变了 z', text)

    def test_noise_ratio_is_reported(self):
        self.assertIn('噪声占比', self.fx.render())

    def test_missing_artifact_omits_the_block_without_crashing(self):
        os.remove(os.path.join(self.fx.dir, record.ZCONTRACT_JSON))
        text = self.fx.render()
        self.assertNotIn('### 72.10 z 契约对账', text)


class TestM2ClaimIsScoped(unittest.TestCase):
    """M2 feeds an out-of-distribution input to a head trained on z only.

    That makes M2 evidence the head depends on its trained input, NOT that z
    is the sole carrier of the semantics. The prereg puts the real
    z-exclusivity test in the C3 arm, which was never run.
    """

    def setUp(self):
        self.fx = _Fixture()
        self.addCleanup(self.fx.close)

    def test_overclaim_is_refused(self):
        gate = self.fx.load(verdict.GATE_JSON)
        gate['gates']['M1'] = False
        self.fx.dump(verdict.GATE_JSON, gate)
        text = self.fx.render()
        self.assertIn('不得据此宣称', text)
        self.assertIn('C3', text)
        self.assertIn('既未被否证、也未被证实', text)

    def test_does_not_claim_z_is_the_only_source(self):
        gate = self.fx.load(verdict.GATE_JSON)
        gate['gates']['M1'] = False
        self.fx.dump(verdict.GATE_JSON, gate)
        text = self.fx.render()
        self.assertNotIn('语义信号确实由 z 承载', text)


class TestM1ExceptionIsCaliberIndependent(unittest.TestCase):
    """The exception must be adjudicated under both calibers.

    The 0.30/0.60 bars came from a TRAIN-side diagnostic (route-C) while the
    gate reads VAL, so quoting only one side leaves the conclusion resting on
    the caliber choice.
    """

    def setUp(self):
        self.fx = _Fixture()
        self.addCleanup(self.fx.close)

    def test_train_caliber_is_derived_from_the_log_not_hardcoded(self):
        src = _read('me_rssm/sanity/semA_ep6_record.py')
        self.assertIn('def train_stat_means(', src)
        self.assertIn('train_stat_means(run_dir).get(6)', src)
        self.assertNotIn('TRAIN_STAT_EP6 = {', src)

    def test_train_caliber_is_an_epoch_mean_not_the_last_value(self):
        """§60.5/§61.5 report epoch means; using the last record would mix
        conventions again. Pinned behaviourally, not just by the docstring.
        """
        self.assertIn('epoch MEAN', record.train_stat_means.__doc__)
        gate = self.fx.load(verdict.GATE_JSON)
        gate['gates']['M1'] = False       # the caliber table is M1-failure-only
        self.fx.dump(verdict.GATE_JSON, gate)
        with open(os.path.join(self.fx.dir, 'run.log.json'), 'a') as handle:
            # A SECOND train record for epoch 6. The 72.5 curve (last-value
            # convention) moves to 0.9, while the caliber table must stay at
            # the epoch mean of {0.3, 0.9} = 0.6 for thr50.
            handle.write(json.dumps({
                'mode': 'train', 'epoch': 6, 'iter': 1900,
                'time': 1.5, 'memory': 17106,
                'stat_sem_hit1': 0.9, 'stat_sem_thr50': 0.9,
                'loss_rssm_sem': 0.1, 'stat_cls_res_rms': 0.5,
                'stat_mu_diff_sq': 0.2,
            }) + '\n')
        text = self.fx.render()
        table = text.split('不依赖口径选择')[1].split('### 72.1')[0]
        self.assertIn('| 0.6000 | 0.7500 |', table)   # epoch means, not last
        self.assertNotIn('| 0.9000 |', table)        # not the last record

    def test_both_calibers_are_shown(self):
        gate = self.fx.load(verdict.GATE_JSON)
        gate['gates']['M1'] = False
        gate['M1_sem_head']['mean_thr50'] = 0.1167
        gate['M1_sem_head']['mean_hit1'] = 0.0833
        self.fx.dump(verdict.GATE_JSON, gate)
        text = self.fx.render()
        self.assertIn('不依赖口径选择', text)
        self.assertIn('val 口径（本门实际读数）', text)
        self.assertIn('train 口径', text)


class TestNoS2Claim(unittest.TestCase):
    def setUp(self):
        self.fx = _Fixture()

    def tearDown(self):
        self.fx.close()

    def test_record_states_no_s2_claim(self):
        text = self.fx.render()
        self.assertIn('本节不宣称任何 S2 性能结论', text)

    def test_early_window_is_never_the_s2_judgement(self):
        text = self.fx.render()
        self.assertIn('不是 ep12–16 的 S2 判据', text)


class TestThreeWindowGuard(unittest.TestCase):
    def setUp(self):
        self.fx = _Fixture()

    def tearDown(self):
        self.fx.close()

    def test_windows_are_labelled_with_their_own_indices(self):
        text = self.fx.render()
        section = text.split('### 72.7')[1].split('### 72.8')[0]
        self.assertIn('（[3, 4]）', section)       # near
        self.assertIn('（[100, 101]）', section)   # mid
        self.assertIn('（[1800, 1801]）', section)  # far

    def test_identical_indices_raise_a_warning(self):
        same = list(range(3, 5))
        for (_, _, name) in record.WINDOWS:
            self.fx.dump(name, _z_util(same))
        text = self.fx.render()
        self.assertIn('同一组 clip', text)
        self.assertIn('不是相互独立的读数', text)

    def test_distinct_indices_do_not_warn(self):
        text = self.fx.render()
        self.assertNotIn('同一组 clip', text)


class TestPosteriorSourceSection(unittest.TestCase):
    """Section 72.8 must not render a truncated source file as a reading."""

    def setUp(self):
        self.fx = _Fixture()

    def tearDown(self):
        self.fx.close()

    def _rich_source(self):
        # Shape of a real tools/probe_posterior_source.py output.
        return {
            'indices': [3, 703, 1403],
            'fusion_class': 'SemanticReadoutRSSMFusion',
            'latent_resolution': [216, 248],
            'readout_mode': 'film',
            'mu_q': {'rms': 0.4583, 'batch_common_rms': 0.4030,
                     'sample_specific_rms': 0.2181,
                     'sample_specific_energy_frac': 0.2265},
            'h_t': {'rms': 0.1061, 'batch_common_rms': 0.0956,
                    'sample_specific_rms': 0.0460,
                    'sample_specific_energy_frac': 0.1879},
            'e_pooled': {'rms': 1.0451, 'batch_common_rms': 0.6246,
                         'sample_specific_rms': 0.8379,
                         'sample_specific_energy_frac': 0.6428},
            'mu_q_removal_of_h_rel_l2': 0.2795,
            'mu_q_removal_of_e_rel_l2': 0.3320,
            'posterior_obs_mode': 'l2',
        }

    def test_full_source_file_renders_every_diagnostic(self):
        self.fx.dump(record.POST_SOURCE, self._rich_source())
        text = self.fx.render()
        section = text.split('### 72.8')[1].split('### 72.9')[0]
        self.assertIn('样本特异能量占比', section)
        self.assertIn('64.28%', section)         # e_pooled specific
        self.assertIn('**0.842**', section)       # h/e removal ratio
        self.assertNotIn('不是完整读数', section)

    def test_rms_only_source_file_is_flagged_not_reported(self):
        """A hand-made stencil must not masquerade as the mechanism reading."""
        self.fx.dump(record.POST_SOURCE, {
            'indices': [3, 703, 1403],
            'fusion_class': 'SemanticReadoutRSSMFusion',
            'latent_resolution': [216, 248],
            'readout_mode': 'film',
            'mu_q': {'rms': 0.5}, 'h_t': {'rms': 0.4},
            'e_pooled': {'rms': 0.8}})
        text = self.fx.render()
        section = text.split('### 72.8')[1].split('### 72.9')[0]
        self.assertIn('不是完整读数', section)
        self.assertIn('不得当作「特异占比为 0」', section)
        # The h/e ratio must not be invented from missing inputs.
        self.assertNotIn('h/e 依赖比** | **0', section)

    def test_source_field_names_match_the_real_tool(self):
        with open(os.path.join(_REPO, 'tools', 'probe_posterior_source.py'),
                  encoding='utf-8') as handle:
            source = handle.read()
        for key in ('sample_specific_energy_frac', 'batch_common_rms',
                    'sample_specific_rms', 'mu_q_removal_of_h_rel_l2',
                    'mu_q_removal_of_e_rel_l2'):
            self.assertIn(key, source,
                          f'probe_posterior_source no longer writes {key}')


class TestDegenerateBundles(unittest.TestCase):
    """The draft is rendered unattended; a broken bundle must not crash it.

    If render() raises, the watchdog logs a warning and there is no draft at
    all -- exactly when a human most needs to see what is missing. These
    bundles must all produce text.
    """

    def _empty_run(self, gate=True, log=True, epochs=()):
        d = tempfile.mkdtemp(prefix='rec72_empty_')
        if gate:
            with open(os.path.join(d, verdict.GATE_JSON), 'w') as handle:
                json.dump({'gates': {}, 'M1_sem_head': {}, 'M2_exclusivity': {},
                           'M3_readout_toggle': {}, 'M4_class_information': {},
                           'indices': [], 'checkpoint': 'x',
                           'input_fingerprints': {}}, handle)
        if log:
            with open(os.path.join(d, 'run.log.json'), 'w') as handle:
                for e in epochs:
                    handle.write(json.dumps({
                        'mode': 'val', 'epoch': e, 'iter': 1,
                        'pts_bbox/KITTI/Overall_3D_moderate': 30.0 + e}) + '\n')
        self.addCleanup(shutil.rmtree, d, ignore_errors=True)
        return d

    def test_entirely_empty_directory_renders(self):
        text = record.render(tempfile.mkdtemp(prefix='rec72_none_'),
                             '2026-10-10')
        self.assertIn('NOT EVALUABLE', text)

    def test_headline_never_calls_an_unmeasured_bundle_failed(self):
        """The H1 must say NOT EVALUABLE, not '未全门通过'.

        '未全门通过' means gates were measured and some failed, which is a
        mechanism finding. An absent bundle is not that.
        """
        for name, run in (('empty', tempfile.mkdtemp(prefix='rec72_h_')),
                          ('no val', self._empty_run(log=False))):
            headline = record.render(run, '2026-10-10').splitlines()[0]
            self.assertIn('NOT EVALUABLE', headline, name)
            self.assertNotIn('未全门通过', headline, name)

    def test_no_val_records_renders(self):
        text = record.render(self._empty_run(log=False), '2026-10-10')
        self.assertIn('无 val 记录', text)
        self.assertIn('NOT MEASURED', text)

    def test_partial_metrics_do_not_crash_the_class_table(self):
        text = record.render(self._empty_run(epochs=(1, 2)), '2026-10-10')
        self.assertIn('ep1-ep2', text)

    def test_missing_gate_readings_are_not_rendered_as_failures(self):
        """'not measured' must never read as 'measured and failed'."""
        text = record.render(self._empty_run(log=False), '2026-10-10')
        self.assertNotIn('NOT ALL PASS', text)
        self.assertIn('NOT MEASURED', text)

    def test_m3_absent_counts_do_not_render_as_none_over_none(self):
        text = record.render(self._empty_run(log=False), '2026-10-10')
        self.assertNotIn('None/None', text)

    def test_non_numeric_reading_renders_as_na(self):
        self.assertEqual(record.num(None), 'n/a')
        self.assertEqual(record.pct(None), 'n/a')
        self.assertEqual(record.signed(None), 'n/a')
        self.assertEqual(record.num('x'), 'n/a')
        self.assertEqual(record.pct('x'), 'n/a')


class TestProducerConsumerContract(unittest.TestCase):
    """The generator reads keys the real probes must actually write."""

    def test_probe_attaches_index_to_every_m1_and_m2_clip(self):
        with open(os.path.join(
                _REPO, 'me_rssm', 'sanity', 'probe_semreadout_ep6.py'),
                encoding='utf-8') as handle:
            source = handle.read()
        # All three head_quality comprehensions must carry the clip index.
        self.assertEqual(source.count('dict(q, index=int(idx))'), 3)

    def test_head_quality_does_not_itself_return_a_conflicting_index(self):
        # head_quality has no index of its own, so attaching it is a pure
        # addition and cannot silently overwrite a probe reading.
        with open(os.path.join(
                _REPO, 'me_rssm', 'sanity', 'probe_semreadout_ep6.py'),
                encoding='utf-8') as handle:
            source = handle.read()
        body = source.split('def head_quality')[1].split('\ndef ')[0]
        self.assertNotIn("'index'", body)

    def test_record_reads_only_documented_probe_keys(self):
        text = record.render.__doc__ or ''
        # The three z-util windows and the source tool are the only files the
        # generator opens besides the gate JSON and the run's log.json.
        self.assertEqual(
            [name for _, _, name in record.WINDOWS],
            ['z_util_std_ep6_w3.json', 'z_util_std_ep6_w100.json',
             'z_util_std_ep6_w1800.json'])
        self.assertEqual(record.POST_SOURCE,
                         'post_source_std_ep6_stride700.json')

    def test_evidence_manifest_is_the_verdict_s_own(self):
        """A renamed artifact must not be silently skipped by the record."""
        labels = {name for _, name, _ in verdict.EVIDENCE}
        expected = {verdict.GATE_JSON, record.POST_SOURCE} | {
            name for _, _, name in record.WINDOWS}
        self.assertEqual(labels, expected)

    def test_std_comparator_window_labels_match(self):
        self.assertEqual(set(record.STD_EP6_SHUFFLE_HEAD),
                         {label for label, _, _ in record.WINDOWS})


if __name__ == '__main__':
    unittest.main(verbosity=2)
