"""CPU tests for the candidate-A z-contract reconciliation probe.

The probe exists because section 72 compared a val-contract gate reading
against train-contract bars. These tests pin the two properties that make its
output trustworthy: (1) it must actually vary the z contract between arms --
a probe that silently reads the same contract twice would "prove" the gap was
zero; (2) its deployment arm must reproduce the frozen S1 probe, so the two
artifacts stay comparable.

Run: python3 me_rssm/sanity/test_probe_semA_zcontract.py
"""

import importlib.util
import inspect
import os
import sys
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


zcontract = _load('probe_zcontract',
                  'me_rssm/sanity/probe_semA_zcontract.py')
ep6 = zcontract.ep6


def _read(rel):
    with open(os.path.join(_REPO, rel), encoding='utf-8') as handle:
        return handle.read()


class TestContractsActuallyDiffer(unittest.TestCase):
    def test_rollout_threads_the_contract_to_every_extract_feat(self):
        """Both the history frames and the current frame must honour it.

        If only the current frame were switched, the recurrent state feeding
        it would still come from the other contract and the two arms would
        differ for the wrong reason.
        """
        src = inspect.getsource(ep6.rollout)
        # Calls span multiple lines, so join continuations before counting.
        flat = ' '.join(line.strip() for line in src.splitlines())
        calls = [c for c in flat.split('model.extract_feat(')[1:]]
        self.assertEqual(len(calls), 2, len(calls))
        for call in calls:
            self.assertIn('rssm_deterministic', call[:400])

    def test_rollout_default_preserves_the_detector_rule(self):
        """Default None -> detector decides; a default of True would silently
        change every historical probe reading."""
        sig = inspect.signature(ep6.rollout)
        self.assertIsNone(sig.parameters['rssm_deterministic'].default)

    def test_probe_switches_the_contract_between_arms(self):
        src = _read('me_rssm/sanity/probe_semA_zcontract.py')
        # Deploy arm must NOT pass the flag (detector rule = mean in eval),
        # sampled arm must pass False; if both were the same the two arms
        # would be one measurement printed twice.
        deploy = src.count("ep6.rollout(model, dataset, i)")
        sampled = src.count("rssm_deterministic=False) for i in indices")
        self.assertEqual(deploy, 1, deploy)
        self.assertEqual(sampled, 1, sampled)


class TestDeployArmMatchesTheFrozenGate(unittest.TestCase):
    def test_deploy_arm_uses_the_same_scoring_entry_point(self):
        """Both artifacts must score through head_quality with the config's
        min_radius, or the two JSONs are not comparable."""
        src = _read('me_rssm/sanity/probe_semA_zcontract.py')
        self.assertIn('ep6.head_quality(', src)
        self.assertIn('min_radius', src)
        self.assertNotIn('MIN_RADIUS', src)

    def test_reports_the_chance_aware_supplement(self):
        """hit@1 is grid-dependent; the probe must carry the same
        excess-over-chance reading the gate records, so the two can be joined
        without re-deriving it."""
        src = _read('me_rssm/sanity/probe_semA_zcontract.py')
        self.assertIn('mean_hit1_excess_over_chance', src)
        self.assertIn('mean_hit1_chance', src)


class TestInterventionIsVerified(unittest.TestCase):
    """Identical arms are ambiguous: either the contract does not move these
    statistics, or the switch silently did nothing. The probe must measure the
    movement instead of letting the reader assume it."""

    def setUp(self):
        self.src = _read('me_rssm/sanity/probe_semA_zcontract.py')

    def test_movement_between_arms_is_reported(self):
        self.assertIn('intervention_check', self.src)
        self.assertIn('rel_delta_z', self.src)
        self.assertIn('rel_delta_sem_map', self.src)

    def test_probe_keeps_the_raw_runs_for_the_delta(self):
        """The delta needs the un-scored rollouts (z / sem_map), so scoring
        must not be the only thing kept."""
        self.assertIn('deploy_runs', self.src)
        self.assertIn('sampled_runs', self.src)


class TestNoiseIsMeasuredNotAssumed(unittest.TestCase):
    def test_sigma_over_mu_is_reported(self):
        src = _read('me_rssm/sanity/probe_semA_zcontract.py')
        self.assertIn('sigma_over_mu', src)
        self.assertIn("'noise'", src)

    def test_missing_noise_cannot_crash_the_report(self):
        """sigma_over_mu returns None when no sample was captured; the report
        builder must tolerate that instead of dict(None)."""
        src = _read('me_rssm/sanity/probe_semA_zcontract.py')
        self.assertIn('sigma_over_mu(model, dataset, i) or {}', src)


class TestReadOnly(unittest.TestCase):
    def test_no_checkpoint_or_weight_write(self):
        src = _read('me_rssm/sanity/probe_semA_zcontract.py')
        for forbidden in ('torch.save', 'load_state_dict', 'train()',
                          'optimizer'):
            self.assertNotIn(forbidden, src)


if __name__ == '__main__':
    unittest.main()
