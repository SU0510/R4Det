"""CPU unit tests for the ep6 verdict assembler's evidence-integrity check.

The ep6 battery is a plain shell sequence: if a probe dies (OOM, CUDA error,
wrong fusion family) it writes no JSON while the watchdog log still ends with
"probes done". These tests pin the two halves of the fix:

  * a complete bundle is reported as complete;
  * every way a bundle can be incomplete (missing file, unreadable JSON,
    non-object top level, missing schema key, missing gate, too few clips,
    per-sample entries without variants) makes ``check_evidence`` return
    ``ok=False`` instead of silently degrading to a gate failure.

The last test is the contract guard: the keys ``check_evidence`` requires must
be exactly the keys the probes actually write, so a probe-side rename cannot
turn every future verdict into a false "INCOMPLETE".

Run: python3 me_rssm/sanity/test_semA_ep6_verdict.py
"""

import importlib.util
import json
import os
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


verdict = _load('verdict_ep6', 'me_rssm/sanity/semA_ep6_verdict.py')
probe = _load('probe_ep6_for_contract',
              'me_rssm/sanity/probe_semreadout_ep6.py')

PS_KEYS = ('variants',)


def _read(rel):
    with open(os.path.join(_REPO, rel), encoding='utf-8') as handle:
        return handle.read()


def _gate(**overrides):
    gate = {
        'gates': {'M1': True, 'M2': True, 'M3': False, 'M4': True},
        'M1_sem_head': {'mean_thr50': 0.4, 'mean_hit1': 0.05, 'per_clip': []},
        'M2_exclusivity': {'h_relative': -0.4, 'feat_relative': -0.5},
        'M3_readout_toggle': {'wins': 7, 'n': 10},
        'M4_class_information': {'mean_own_hit': 0.5, 'bar': 0.45},
        'indices': [3, 203],
        'checkpoint': '/data/x/epoch_6.pth',
        'input_fingerprints': {
            'checkpoint_sha256': 'a' * 64,
            'checkpoint_size_bytes': 688860076,
            'checkpoint_mtime': 1791650561,
            'config_sha256': 'b' * 64,
        },
    }
    for key, value in overrides.items():
        if key in gate['gates']:
            gate['gates'][key] = value
    return gate


def _write(run_dir, name, payload, raw=None):
    path = os.path.join(run_dir, name)
    os.makedirs(os.path.dirname(path) or run_dir, exist_ok=True)
    with open(path, 'w') as handle:
        handle.write(raw if raw is not None else json.dumps(payload))
    return path


def build_bundle(run_dir, gate=None):
    _write(run_dir, verdict.GATE_JSON, _gate() if gate is None else gate)
    for name in ('z_util_std_ep6_w3.json', 'z_util_std_ep6_w100.json',
                 'z_util_std_ep6_w1800.json'):
        _write(run_dir, name,
               {'indices': [3, 4], 'per_sample': {
                   '3': {'variants': {'normal': {}}},
                   '4': {'variants': {'normal': {}}}}})
    _write(run_dir, 'post_source_std_ep6_stride700.json',
           {'indices': [3], 'fusion_class': 'MotionAlignedRSSMFusion',
            'mu_q': {}, 'h_t': {}, 'e_pooled': {}})


class TestEvidenceComplete(unittest.TestCase):

    def test_full_bundle_is_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            build_bundle(tmp)
            rows, ok = verdict.check_evidence(tmp)
        self.assertTrue(ok)
        self.assertTrue(all(r[2] == 'OK' for r in rows))

    def test_every_expected_artifact_is_listed(self):
        with tempfile.TemporaryDirectory() as tmp:
            build_bundle(tmp)
            rows, _ = verdict.check_evidence(tmp)
        self.assertEqual(len(rows), len(verdict.EVIDENCE))
        self.assertEqual([r[1] for r in rows],
                         [name for _, name, _ in verdict.EVIDENCE])


class TestEvidenceIncomplete(unittest.TestCase):

    def test_missing_file_fails_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            build_bundle(tmp)
            os.remove(os.path.join(tmp, 'post_source_std_ep6_stride700.json'))
            rows, ok = verdict.check_evidence(tmp)
        self.assertFalse(ok)
        missing = [r for r in rows if r[2] == 'MISSING']
        self.assertEqual(len(missing), 1)
        self.assertIn('post_source', missing[0][1])

    def test_unreadable_json_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            build_bundle(tmp)
            _write(tmp, 'z_util_std_ep6_w3.json', None, raw='{not json')
            rows, ok = verdict.check_evidence(tmp)
        self.assertFalse(ok)
        self.assertTrue(any(r[2] == 'UNREADABLE' for r in rows))

    def test_non_object_top_level_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            build_bundle(tmp)
            _write(tmp, 'z_util_std_ep6_w100.json', [1, 2, 3])
            rows, ok = verdict.check_evidence(tmp)
        self.assertFalse(ok)
        self.assertTrue(any(r[2] == 'MALFORMED' for r in rows))

    def test_missing_schema_key_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            build_bundle(tmp)
            _write(tmp, 'post_source_std_ep6_stride700.json',
                   {'indices': [3], 'fusion_class': 'X'})  # no mu_q/h_t/e_pooled
            rows, ok = verdict.check_evidence(tmp)
        self.assertFalse(ok)
        bad = [r for r in rows if r[2] == 'INCOMPLETE']
        self.assertEqual(len(bad), 1)
        self.assertIn('mu_q', bad[0][3])

    def test_missing_fingerprint_fails(self):
        """A reading with no artifact hash cannot be tied to a checkpoint."""
        gate = _gate()
        del gate['input_fingerprints']
        with tempfile.TemporaryDirectory() as tmp:
            build_bundle(tmp, gate=gate)
            rows, ok = verdict.check_evidence(tmp)
        self.assertFalse(ok)
        bad = [r for r in rows if r[1] == verdict.GATE_JSON]
        self.assertEqual(bad[0][2], 'INCOMPLETE')
        self.assertIn('input_fingerprints', bad[0][3])

    def test_missing_gate_fails(self):
        """3 of 4 gates present is not a pass."""
        gate = _gate()
        del gate['gates']['M2']
        with tempfile.TemporaryDirectory() as tmp:
            build_bundle(tmp, gate=gate)
            rows, ok = verdict.check_evidence(tmp)
        self.assertFalse(ok)
        bad = [r for r in rows if r[1] == verdict.GATE_JSON]
        self.assertEqual(bad[0][2], 'INCOMPLETE')
        self.assertIn('M2', bad[0][3])

    def test_no_clips_selected_fails(self):
        gate = _gate()
        gate['indices'] = []
        with tempfile.TemporaryDirectory() as tmp:
            build_bundle(tmp, gate=gate)
            _, ok = verdict.check_evidence(tmp)
        self.assertFalse(ok)

    def test_single_per_sample_reading_fails(self):
        """z-util's paired shuffle needs >= 2 clips; 1 is a silent no-op."""
        with tempfile.TemporaryDirectory() as tmp:
            build_bundle(tmp)
            _write(tmp, 'z_util_std_ep6_w1800.json',
                   {'indices': [3], 'per_sample': {'3': {'variants': {}}}})
            rows, ok = verdict.check_evidence(tmp)
        self.assertFalse(ok)
        bad = [r for r in rows if r[1] == 'z_util_std_ep6_w1800.json']
        self.assertEqual(bad[0][2], 'INCOMPLETE')
        self.assertIn('1 per-sample', bad[0][3])

    def test_per_sample_without_variants_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            build_bundle(tmp)
            _write(tmp, 'z_util_std_ep6_w3.json',
                   {'indices': [3, 4],
                    'per_sample': {'3': {'variants': {}}, '4': {}}})
            rows, ok = verdict.check_evidence(tmp)
        self.assertFalse(ok)
        bad = [r for r in rows if r[1] == 'z_util_std_ep6_w3.json']
        self.assertEqual(bad[0][2], 'INCOMPLETE')


class TestProducerConsumerContract(unittest.TestCase):
    """The required keys must be the keys the probes actually emit.

    Renaming a probe-side key would otherwise make every future bundle read as
    INCOMPLETE -- a false negative that would block the gate without any
    visible cause in the probe itself.
    """

    def test_required_gate_names_match_the_probe(self):
        self.assertEqual(set(verdict.REQUIRED_GATES),
                         {'M1', 'M2', 'M3', 'M4'})
        source = _read(os.path.join('me_rssm', 'sanity',
                                    'probe_semreadout_ep6.py'))
        for gate in verdict.REQUIRED_GATES:
            self.assertIn(f"report['gates']['{gate}']", source)

    def test_gate_schema_keys_are_written_by_the_probe(self):
        source = open(os.path.join(
            _REPO, 'me_rssm', 'sanity',
            'probe_semreadout_ep6.py'), encoding='utf-8').read()
        for key in ('M1_sem_head', 'M2_exclusivity', 'M3_readout_toggle',
                    'M4_class_information', 'indices'):
            self.assertIn(f"'{key}'", source)
        self.assertIn("'checkpoint'", source)

    def test_z_util_schema_keys_are_written_by_the_probe(self):
        source = _read(os.path.join('me_rssm', 'sanity',
                                    'probe_z_utilization_standard.py'))
        for key in ('per_sample', 'variants'):
            self.assertIn(f"'{key}'", source)

    def test_posterior_source_keys_are_written_by_the_tool(self):
        source = _read(os.path.join('tools', 'probe_posterior_source.py'))
        for key in ('indices', 'fusion_class', 'mu_q', 'h_t', 'e_pooled'):
            self.assertIn(f"'{key}'", source)

    def test_evidence_covers_the_watchdog_outputs(self):
        """Every JSON the watchdog writes must be checked by the verdict."""
        watchdog = _read(os.path.join('me_rssm', 'sanity',
                                      'watch_semA_ep6.sh'))
        checked = {name for _, name, _ in verdict.EVIDENCE}
        self.assertIn(verdict.GATE_JSON, watchdog)
        for name in checked:
            self.assertIn(name, watchdog,
                          f'{name} is watched by the verdict but the watchdog '
                          'never writes it')


if __name__ == '__main__':
    unittest.main(verbosity=2)


class TestM1GridLimitedRule(unittest.TestCase):
    """Pre-registered section 71.9.3(2): an M1 miss caused by the cross-grid
    hit@1 extrapolation is INSUFFICIENT EVIDENCE, not a refutation.

    The arms must be mutually exclusive, or the rule would swallow real M1
    failures.
    """

    def test_healthy_excess_with_cleared_thr50_is_grid_limited(self):
        # thr50 well past its bar, hit@1 far below 0.60, excess at the
        # calibrated bar -> the head does find centres; only the raw rate fails.
        self.assertTrue(verdict.m1_grid_limited(0.45, 0.08, 0.60))
        self.assertTrue(verdict.m1_grid_limited(0.30, 0.01, 0.5841))

    def test_thr50_also_failing_is_not_excused(self):
        """If thr50 missed its bar, the miss is NOT attributable to the grid."""
        self.assertFalse(verdict.m1_grid_limited(0.29, 0.08, 0.60))
        self.assertFalse(verdict.m1_grid_limited(0.0, 0.0, 0.99))

    def test_healthy_excess_but_hit1_passing_is_not_excused(self):
        """Nothing to explain away when M1 actually passed on the raw rate."""
        self.assertFalse(verdict.m1_grid_limited(0.45, 0.60, 0.60))
        self.assertFalse(verdict.m1_grid_limited(0.45, 0.99, 0.99))

    def test_low_excess_is_not_excused(self):
        self.assertFalse(verdict.m1_grid_limited(0.45, 0.02, 0.019))
        self.assertFalse(verdict.m1_grid_limited(0.45, 0.02, 0.0))

    def test_missing_readings_never_excuse(self):
        for args in ((None, 0.08, 0.60), (0.45, None, 0.60),
                     (0.45, 0.08, None)):
            self.assertFalse(verdict.m1_grid_limited(*args))

    def test_thresholds_match_the_preregistered_bars(self):
        self.assertEqual(verdict.M1_BAR_THR50, 0.30)
        self.assertEqual(verdict.M1_BAR_HIT1, 0.60)
        # the excess bar is derived from the calibrated chance, not invented
        self.assertAlmostEqual(
            verdict.M1_EXCESS_BAR,
            verdict.M1_BAR_HIT1 - verdict.M1_CALIBRATION_CHANCE)
        self.assertLess(verdict.M1_EXCESS_BAR, verdict.M1_BAR_HIT1)
        # and the full-res chance must be far below the calibrated one
        self.assertLess(verdict.M1_CALIBRATION_CHANCE, 0.05)

    def test_excess_mean_reads_the_probe_supplement(self):
        m1 = {'per_clip': [{'hit1_excess_over_chance': 0.2},
                           {'hit1_excess_over_chance': 0.4},
                           {'hit1_excess_over_chance': None}]}
        self.assertAlmostEqual(verdict.m1_excess_mean(m1), 0.3)
        self.assertIsNone(verdict.m1_excess_mean({'per_clip': []}))
        self.assertIsNone(verdict.m1_excess_mean({}))

    def test_rule_is_only_reachable_for_a_failing_m1(self):
        """Source guard: the aggregate branch requires M1 to be False."""
        source = _read(os.path.join('me_rssm', 'sanity',
                                    'semA_ep6_verdict.py'))
        self.assertIn("gates.get('M1') is False", source)
        self.assertIn("k != 'M1'", source)


class TestA2AddedGates(unittest.TestCase):
    """A2's pre-registered M1'/M1-loc gates must actually be evaluated.

    These live in the rank artifact, not the M1-M4 gate JSON. Before this
    path existed nothing read them: an A2 arm whose added gates failed still
    printed "ALL PASS" off M1-M4 alone. The tests below pin both halves --
    that the gates ARE folded in for an A2 config, and that the A generation
    is left byte-for-byte alone (A's bundle also carries a rank JSON, produced
    after the fact under a different clip contract).
    """

    def _rank(self, cfg, n=20, auc=0.90, shift=0.50, maxgt_hi=15,
              recall_med=0.40):
        rows = []
        for i in range(n):
            rows.append({
                'index': 3 + 100 * i,
                'auc_mean_over_occupied_channels': auc,
                'gt_shuffle_auc': shift,
                'max_gt_prob': 0.9 if i < maxgt_hi else 0.1,
                'recall_at_top_k_mean': recall_med,
            })
        return {'probe': 'rank', 'config': cfg, 'n_clips': n,
                'indices': [r['index'] for r in rows], 'per_clip': rows}

    def test_a_generation_is_not_treated_as_a2(self):
        with tempfile.TemporaryDirectory() as tmp:
            build_bundle(tmp)                      # A config path
            _write(tmp, verdict.A2_RANK_VAL,
                   self._rank('/x/..._semA_seed0.py'))
            self.assertIsNone(verdict.arm_of_bundle(tmp))
            self.assertIsNone(verdict.a2_readings(tmp))

    def test_a2s_arm_is_detected_from_the_config_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            gate = _gate()
            gate['input_fingerprints']['config_path'] = (
                '/x/TJ4D-..._no2d_igdr_semA2S_24e.py')
            build_bundle(tmp, gate)
            self.assertEqual(verdict.arm_of_bundle(tmp), 'A2S')

    def test_a2s_gets_both_added_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            gate = _gate()
            gate['input_fingerprints']['config_path'] = '/x/..._semA2S_24e.py'
            build_bundle(tmp, gate)
            _write(tmp, verdict.A2_RANK_VAL, self._rank('/x/..._semA2S_24e.py'))
            got = verdict.a2_readings(tmp)
            self.assertEqual(set(got['gates']), {'M1p', 'M1loc'})
            self.assertTrue(got['gates']['M1p'])
            self.assertTrue(got['gates']['M1loc'])

    def test_a2r_does_not_import_the_supervision_side_gate(self):
        """M1-loc is pre-registered A2-S only (section 5)."""
        with tempfile.TemporaryDirectory() as tmp:
            gate = _gate()
            gate['input_fingerprints']['config_path'] = '/x/..._semA2R_24e.py'
            build_bundle(tmp, gate)
            _write(tmp, verdict.A2_RANK_VAL, self._rank('/x/..._semA2R_24e.py'))
            got = verdict.a2_readings(tmp)
            self.assertEqual(set(got['gates']), {'M1p'})

    def test_m1loc_fails_on_either_leg(self):
        with tempfile.TemporaryDirectory() as tmp:
            gate = _gate()
            gate['input_fingerprints']['config_path'] = '/x/..._semA2S_24e.py'
            build_bundle(tmp, gate)
            # too few clips reach max_gt_prob >= 0.5
            _write(tmp, verdict.A2_RANK_VAL,
                   self._rank('/x/..._semA2S_24e.py', maxgt_hi=11))
            self.assertFalse(verdict.a2_readings(tmp)['gates']['M1loc'])
            # peak budget recovered, but the top-k budget is still lost
            _write(tmp, verdict.A2_RANK_VAL,
                   self._rank('/x/..._semA2S_24e.py', recall_med=0.24))
            self.assertFalse(verdict.a2_readings(tmp)['gates']['M1loc'])

    def test_m1p_requires_both_the_level_and_the_margin(self):
        with tempfile.TemporaryDirectory() as tmp:
            gate = _gate()
            gate['input_fingerprints']['config_path'] = '/x/..._semA2S_24e.py'
            build_bundle(tmp, gate)
            # high auc but no shuffle margin -> the ranking is not specific
            _write(tmp, verdict.A2_RANK_VAL,
                   self._rank('/x/..._semA2S_24e.py', auc=0.95, shift=0.80))
            self.assertFalse(verdict.a2_readings(tmp)['gates']['M1p'])
            # healthy margin but the level itself slipped below the bar
            _write(tmp, verdict.A2_RANK_VAL,
                   self._rank('/x/..._semA2S_24e.py', auc=0.79, shift=0.40))
            self.assertFalse(verdict.a2_readings(tmp)['gates']['M1p'])

    def test_wrong_frozen_n_is_not_measured_not_passed(self):
        with tempfile.TemporaryDirectory() as tmp:
            gate = _gate()
            gate['input_fingerprints']['config_path'] = '/x/..._semA2S_24e.py'
            build_bundle(tmp, gate)
            # 10 clips, all perfect: must NOT read as a pass
            _write(tmp, verdict.A2_RANK_VAL,
                   self._rank('/x/..._semA2S_24e.py', n=10))
            got = verdict.a2_readings(tmp)
            self.assertIsNone(got['gates']['M1p'])
            self.assertTrue(got['blockers'])

    def test_missing_rank_artifact_blocks_rather_than_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            gate = _gate()
            gate['input_fingerprints']['config_path'] = '/x/..._semA2S_24e.py'
            build_bundle(tmp, gate)
            got = verdict.a2_readings(tmp)
            self.assertIsNone(got['gates']['M1p'])
            self.assertTrue(got['blockers'])

    def test_clip_contract_mismatch_blocks_the_gates(self):
        """Source guard: the aggregate voids the added gates on index drift."""
        source = _read(os.path.join('me_rssm', 'sanity',
                                    'semA_ep6_verdict.py'))
        self.assertIn('rank and gate probes selected ', source)
        self.assertIn('different clips', source)

    def test_record_renderer_consults_the_a2_gates(self):
        """The draft generator must not head an A2 bundle 'all gates pass'
        off M1-M4 alone."""
        source = _read(os.path.join('me_rssm', 'sanity',
                                    'semA_ep6_record.py'))
        self.assertIn('verdict.a2_readings(run_dir)', source)
        self.assertIn('added_all_pass', source)
        self.assertIn('A2 增补门未测全', source)
