"""CPU tests for the M4 class-prior-floor / readout-scale probe.

The probe exists because section 72 recorded M4 as PASS on ``own_hit``
(0.7379 >= 0.45) and FAIL on ``donor_transfer``, without either reading being
compared against the trivial strategy. A residual that always votes the
majority GT class scores exactly the majority rate, so a 0.45 bar is not a
test of class-specific information when the class prior is skewed -- the same
error class as section 70.6, where "93.7% sample specific" equalled the
pure-noise floor.

These tests pin the properties that make the probe's output trustworthy:

  * it must recompute ``own_hit`` from the residual rather than quote the gate
    JSON (otherwise floor and reading are not joined by construction);
  * it must derive the floor from the SAME rasteriser/config the gate used
    (a hard-coded 1/C "chance" would understate a 0.51 prior floor, and a
    hard-coded min_radius would score a different target);
  * the head-scale curve must carry its internal consistency reading, since a
    force-to-maximum arm that does NOT reproduce the residual's own argmax
    means the substitution path is not what the probe claims it is.

Run: python3 me_rssm/sanity/test_probe_semA_m4floor.py
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


m4floor = _load('probe_m4floor', 'me_rssm/sanity/probe_semA_m4floor.py')
ep6 = m4floor.ep6


def _read(rel):
    with open(os.path.join(_REPO, rel), encoding='utf-8') as handle:
        return handle.read()


class TestFloorIsDerivedNotAssumed(unittest.TestCase):
    def test_own_hit_is_recomputed_from_the_residual(self):
        src = inspect.getsource(m4floor.main)
        self.assertIn('residual_class_map', src)
        self.assertIn("own == labels", src)
        self.assertNotIn("'reported_mean_own_hit': 0.7378", src)

    def test_floor_uses_the_same_rasteriser_and_min_radius(self):
        """The floor is a property of the GT rasterisation, so it must come
        from the config the gate read -- not a hard-coded 1/C or radius."""
        src = inspect.getsource(m4floor.main)
        self.assertIn('object_min_radius', src)
        self.assertIn('make_target', src)

    def test_report_distinguishes_prior_floor_from_chance(self):
        """1/4 and the majority-class rate are different bars; conflating them
        is what made the archived M4 pass look meaningful."""
        src = _read('me_rssm/sanity/probe_semA_m4floor.py')
        self.assertIn("'chance_1_over_c'", src)
        self.assertIn("'pooled_prior_floor'", src)


class TestHeadScaleCurve(unittest.TestCase):
    def test_curve_includes_switch_off_and_forced_maximum(self):
        src = inspect.getsource(m4floor.main)
        self.assertIn('head_scores_with_residual', src)
        self.assertIn('1e6', src)

    def test_curve_is_pooled_cell_weighted(self):
        """A 1-cell clip must not swing a mean over 10 clips."""
        src = inspect.getsource(m4floor.main)
        self.assertIn("pooled_curve[str(k)]", src)
        self.assertIn("r['cells']", src)

    def test_scale_note_does_not_claim_information(self):
        src = _read('me_rssm/sanity/probe_semA_m4floor.py')
        self.assertIn('a scale statement, not', src)
        self.assertIn('an information statement', src)


class TestGuardAgainstSilentTriviality(unittest.TestCase):
    def test_report_carries_the_margin_field(self):
        src = _read('me_rssm/sanity/probe_semA_m4floor.py')
        self.assertIn("'margin_over_pooled_prior'", src)

    def test_sel_stride_default_keeps_legacy_callers_working(self):
        """``_Sel`` gained a stride argument for this probe; the gate probe's
        own call site passes only three arguments and must keep working."""
        sig = inspect.signature(ep6._Sel.__init__)
        self.assertEqual(list(sig.parameters), [
            'self', 'start_index', 'limit', 'min_valid_history', 'stride'])
        self.assertEqual(sig.parameters['stride'].default, 1)


if __name__ == '__main__':
    unittest.main()
