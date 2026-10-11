"""CPU tests for the sem-head caliber probe (split/appearance vs BN mode).

The probe exists because section 71.9.17 left one number unexplained: the same
checkpoint reads 0.4923 thr50 / 0.3586 hit1 on the train split (model in
train()) and 0.1167 / 0.0833 on val (model in eval()). Those are different
functions of the same weights, and the record must not present one as
"unlocated" while a cheap, mechanical explanation is still untested -- nor may
a failed intervention be read as evidence.

These tests pin the four properties that make the probe's output usable:

  * the BN flip must be scoped to the z path (the other 166 BatchNorms are not
    what the training logger's z path normalised with), and the scope control
    must exercise the complement -- otherwise a movement could be attributed to
    z when it came from anywhere;
  * a train()-statistics arm must run a real batch (batch size 1 makes the
    batch statistic zero-variance, i.e. a strawman that would "explain" the gap
    by construction);
  * arms must be PAIRED clip-by-clip, so a delta cannot be produced by an arm
    silently keeping a different set of clips;
  * the eval arm must be checked against the frozen gate reading, because a
    batched rollout that does not reproduce the gate is not the gate's
    measurement.

Run: python3 me_rssm/sanity/test_probe_semA_bnmode.py
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


bnmode = _load('probe_bnmode', 'me_rssm/sanity/probe_semA_bnmode.py')


def _read(rel):
    with open(os.path.join(_REPO, rel), encoding='utf-8') as handle:
        return handle.read()


class TestBnScopeIsExplicit(unittest.TestCase):
    def test_bn_selector_takes_a_scope_flag(self):
        sig = inspect.signature(bnmode._bn_modules)
        self.assertEqual(list(sig.parameters), ['model', 'under_temporal_fusion'])

    def test_selector_splits_the_z_path_from_the_rest(self):
        """The z path is ``temporal_fusion.*``; the probe must not flip all 170
        BatchNorms and call the result "the z path"."""
        src = inspect.getsource(bnmode._bn_modules)
        self.assertIn("startswith('temporal_fusion.')", src)
        self.assertIn('if under == under_temporal_fusion', src)

    def test_all_three_arms_are_read(self):
        src = inspect.getsource(bnmode.main)
        for arm in ("'eval'", "'z_path'", "'out_of_scope'"):
            self.assertIn(arm, src)


class TestTrainStatsArmNeedsARealBatch(unittest.TestCase):
    def test_batch_sizes_below_two_are_refused_for_train_stats(self):
        src = inspect.getsource(bnmode.read_arm)
        self.assertIn('batch_size < 2', src)
        self.assertIn('raise ValueError', src)

    def test_rollout_is_batched(self):
        """The BatchNorm statistic is estimated over the batch dimension, so
        the arm has to feed one."""
        src = inspect.getsource(bnmode._batched_rollout)
        self.assertIn('build_batch(dataset, list(indices)', src)

    def test_limit_must_divide_the_batch(self):
        """Arms that split the last chunk differently would not even be
        compared at the same batch composition."""
        src = inspect.getsource(bnmode.main)
        self.assertIn('--limit', src)
        self.assertIn('multiple of --batch-size', src)


class TestPairingAndReproduction(unittest.TestCase):
    def test_delta_is_paired_by_index(self):
        src = inspect.getsource(bnmode.main)
        self.assertIn('base_map', src)
        self.assertIn("r['index'] in base_map", src)

    def test_reports_how_many_readings_moved(self):
        src = _read('me_rssm/sanity/probe_semA_bnmode.py')
        self.assertIn("'n_thr50_moved'", src)
        self.assertIn("'n_hit1_moved'", src)

    def test_eval_arm_is_checked_against_the_frozen_gate(self):
        src = _read('me_rssm/sanity/probe_semA_bnmode.py')
        self.assertIn("'eval_reproduces_frozen_gate'", src)
        self.assertIn("'abs_delta_thr50'", src)

    def test_intervention_check_reports_a_dead_arm_as_dead(self):
        src = _read('me_rssm/sanity/probe_semA_bnmode.py')
        self.assertIn("'all_arms_bitwise_identical_to_eval'", src)


class TestSplitArmAndFloors(unittest.TestCase):
    def test_probe_can_read_another_split_through_the_same_pipeline(self):
        """The train logger's own numbers change BOTH the split and the
        normalisation mode, so separating them needs a same-pipeline,
        same-mode read of the other split."""
        src = inspect.getsource(bnmode.main)
        self.assertIn('--split-ann-file', _read('me_rssm/sanity/probe_semA_bnmode.py'))
        self.assertIn('args.split_ann_file', src)

    def test_report_records_which_split_and_pipeline_was_read(self):
        src = inspect.getsource(bnmode.main)
        self.assertIn("'ann_file': str(dataset.ann_file)", src)
        self.assertIn("'pipeline':", src)

    def test_val_gate_reproduction_only_claims_to_apply_on_val(self):
        """On the train split the frozen gate value is a reference, not a
        correctness descriptor; the report must say which it is."""
        src = _read('me_rssm/sanity/probe_semA_bnmode.py')
        self.assertIn("'check_applies':", src)
        self.assertIn('TJ4D_infos_val.pkl', src)

    def test_gt_shuffle_floor_is_measured(self):
        """A map that always peaks in the populous part of the BEV would score
        against ANY clip's target; the floor must be measured, not assumed."""
        src = inspect.getsource(bnmode.read_arm)
        self.assertIn('donor = raw[(pos + 1) % len(raw)]', src)
        self.assertIn("'reading_survives_gt_shuffle'", src)


class TestScopeControlIsCountMatched(unittest.TestCase):
    def test_placebo_takes_the_same_number_of_layers_as_the_z_path(self):
        src = inspect.getsource(bnmode._arm_scope)
        self.assertIn('[:len(z_bn)]', src)

    def test_arms_can_be_selected_for_a_single_arm_control(self):
        """Batching neutrality is checked by reading ONLY the eval arm at two
        batch sizes and requiring the same number."""
        src = _read('me_rssm/sanity/probe_semA_bnmode.py')
        self.assertIn("'--arms'", src)
        self.assertIn('_selected_arms', src)


class TestProbeIsReadOnly(unittest.TestCase):
    def test_no_optimiser_or_checkpoint_write(self):
        src = _read('me_rssm/sanity/probe_semA_bnmode.py')
        for forbidden in ('torch.save', 'optimizer', 'backward()',
                          'load_state_dict', 'DataLoader'):
            self.assertNotIn(forbidden, src)

    def test_running_stats_are_snapshotted_and_restored(self):
        src = inspect.getsource(bnmode.read_arm)
        self.assertIn('_snapshot(all_bn)', src)
        self.assertIn('_restore(all_bn, snapshot)', src)
        self.assertIn('model.eval()', src)

    def test_only_norm_layers_are_toggled(self):
        """The arms must not put the model into train() as a whole: dropout and
        training-only branches would then leak into the reading."""
        src = _read('me_rssm/sanity/probe_semA_bnmode.py')
        self.assertIn('mod.train(True)', src)
        self.assertNotIn('model.train()', src)


if __name__ == '__main__':
    unittest.main()
