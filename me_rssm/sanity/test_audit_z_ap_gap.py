"""Standard-library tests for the read-only z/AP audit; no model imports."""
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "audit_z_ap_gap", Path(__file__).with_name("audit_z_ap_gap.py"))
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


class AuditZAPGapTest(unittest.TestCase):
    def test_analytical_checks_are_not_ap_measurements(self):
        result = audit.analytical_checks()
        self.assertGreater(result["synthetic_one_positive_auc"], .92)
        self.assertEqual(result["synthetic_top1_recall"], 0)
        self.assertEqual(result["radius2_gaussian_positive_support"], 25)
        self.assertEqual(result["radius2_gaussian_exact_one_count"], 1)
        self.assertTrue(result["shared_cell_class_offset_preserves_anchor_logit_differences"])

    def test_empty_mature_window_is_not_zero_ap(self):
        result = audit.window({1: {audit.OVERALL: 25}}, range(12, 17))
        self.assertEqual(result["n"], 0)
        self.assertEqual(result["means"], {})

    def test_partial_window_records_actual_epochs(self):
        result = audit.window({12: {audit.OVERALL: 30}, 14: {audit.OVERALL: 40}}, range(12, 17))
        self.assertEqual(result["actual_epochs"], [12, 14])
        self.assertEqual(result["means"][audit.OVERALL], 35)

    def test_peak_ignores_saving_and_last_duplicate_val_wins(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [{"mode": "val", "epoch": e, audit.OVERALL: x}
                    for e, x in ((1, 30), (2, 40), (3, 35))]
            (root / "20260101.log.json").write_text("\n".join(map(json.dumps, rows)) + "\n")
            (root / "20260102.log.json").write_text(json.dumps(
                {"mode": "val", "epoch": 2, audit.OVERALL: 45}) + "\n")
            (root / "epoch_2.pth").touch()
            (root / "epoch_3.pth").touch()
            result = audit.read_run(root, 3, Path.read_bytes)
            self.assertEqual(result["all_observed_peak"], {"value": 45, "epoch": 2})
            self.assertEqual(result["val_sources"][2]["file"], str(root / "20260102.log.json"))
            (root / "epoch_2.pth").unlink()
            result = audit.read_run(root, 3, Path.read_bytes)
            self.assertEqual(result["all_observed_peak"]["epoch"], 2)
            self.assertEqual(result["best_saved"], {"value": 35, "epoch": 3})

    def test_cutoff_excludes_future_val_and_partial_training_epoch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [{"mode": "val", "epoch": 11, audit.OVERALL: 39},
                    {"mode": "val", "epoch": 12, audit.OVERALL: 99},
                    {"mode": "train", "epoch": 12, "stat_sem_thr50": .8}]
            (root / "run.log.json").write_text("\n".join(map(json.dumps, rows)) + "\n")
            result = audit.read_run(root, 11, Path.read_bytes)
            self.assertEqual(list(result["val_records"]), [11])
            self.assertEqual(result["train_logger_means"], {})
            self.assertEqual(result["windows"]["fixed_ep12_16"]["n"], 0)

    def test_live_partial_tail_is_explicitly_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "run.log.json").write_text(json.dumps(
                {"mode": "val", "epoch": 1, audit.OVERALL: 10}) + "\n{" )
            result = audit.read_run(root, 1, Path.read_bytes)
            self.assertEqual(len(result["ignored_partial_tail_records"]), 1)

    def test_malformed_complete_line_is_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "run.log.json").write_text("not-json\n")
            with self.assertRaises(json.JSONDecodeError):
                audit.read_run(root, 1, Path.read_bytes)


if __name__ == "__main__":
    unittest.main()
