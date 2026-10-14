"""CPU-only regression proof for the explicit training debug-mode entry."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("train_vod_runtime", ROOT / "tools/train_vod_runtime.py")
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


def fake_entry(tmp_path, body):
    entry = tmp_path / "entry.py"
    entry.write_text("import torch, sys, json\n"
                     "torch.autograd.set_detect_anomaly(True)\n"
                     "def main():\n" + body)
    config = tmp_path / "config.py"
    config.write_text("model = {}\n")
    work = tmp_path / "work"
    args = ["--config", str(config), "--work-dir", str(work), "--seed", "17"]
    return entry, work, args


def receipt(work):
    paths = list((work / "runtime_receipts").glob("*.json"))
    assert len(paths) == 1
    return json.loads(paths[0].read_text())


@pytest.mark.parametrize("mode", ["on", "off"])
def test_mode_applied_after_import_and_args_unchanged(tmp_path, mode):
    entry, work, args = fake_entry(tmp_path,
        "    assert torch.is_anomaly_enabled() == " + str(mode == "on") + "\n")
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"checkpoint hash sentinel")
    args += ["--resume-from", str(checkpoint), "--cfg-options", "x=[1,2]", "--local-rank=2"]
    old_argv = sys.argv
    old_mode = torch.is_anomaly_enabled()
    runtime.run_entry(args + ["--detect-anomaly", mode], entry=entry)
    result = receipt(work)
    assert result["argv_forwarded"] == args
    assert result["detect_anomaly_effective"] == (mode == "on")
    assert result["resume_sha256"] == runtime.sha256(checkpoint)
    assert result["entry_sha256"] == runtime.sha256(entry)
    assert result["wrapper_sha256"] == runtime.sha256(runtime.__file__)
    assert result["status"] == "completed" and result["exit_code"] == 0
    assert sys.argv is old_argv
    assert torch.is_anomaly_enabled() == old_mode
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("body,exception,code,status", [
    ("    raise RuntimeError('sentinel failure')\n", RuntimeError, 1, "failed"),
    ("    raise SystemExit(7)\n", SystemExit, 7, "failed"),
    ("    raise SystemExit('bad cli')\n", SystemExit, 1, "failed"),
    ("    raise SystemExit(0)\n", SystemExit, 0, "completed"),
    ("    raise KeyboardInterrupt()\n", KeyboardInterrupt, 130, "failed"),
])
def test_exit_receipt_never_labels_failure_success(tmp_path, body, exception, code, status):
    entry, work, args = fake_entry(tmp_path, body)
    with pytest.raises(exception):
        runtime.run_entry(args + ["--detect-anomaly", "off"], entry=entry)
    result = receipt(work)
    assert result["status"] == status and result["exit_code"] == code


def test_import_failure_recorded(tmp_path):
    entry, work, args = fake_entry(tmp_path, "    pass\n")
    entry.write_text("raise RuntimeError('import failure')\n")
    with pytest.raises(RuntimeError, match="import failure"):
        runtime.run_entry(args + ["--detect-anomaly", "off"], entry=entry)
    assert receipt(work)["status"] == "failed"
    assert "detect_anomaly_effective" not in receipt(work)


def test_mode_must_be_explicit(tmp_path):
    entry, work, args = fake_entry(tmp_path, "    pass\n")
    with pytest.raises(SystemExit) as error:
        runtime.run_entry(args, entry=entry)
    assert error.value.code == 2
    assert not work.exists()


def test_receipt_cannot_overwrite_previous_invocation(tmp_path):
    entry, work, args = fake_entry(tmp_path, "    pass\n")
    args += ["--detect-anomaly", "off"]
    runtime.run_entry(args, entry=entry)
    before = receipt(work)
    with pytest.raises(FileExistsError):
        runtime.run_entry(args, entry=entry)
    assert receipt(work) == before


@pytest.mark.parametrize("visible,gpus", [("0,1,2", "3"), ("5,5,6", "3"),
                                           ("5,6", "3"), ("5,6,7", "4")])
def test_launcher_rejects_invalid_gpu_selection(monkeypatch, visible, gpus):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)
    result = subprocess.run(["bash", str(ROOT / "tools/dist_train_runtime.sh"),
                             "unused.py", gpus], capture_output=True, text=True)
    assert result.returncode == 2


def test_launcher_preserves_environment_and_argument_boundaries(tmp_path, monkeypatch):
    capture = tmp_path / "captured.json"
    executable = tmp_path / "python"
    executable.write_text(f"#!{sys.executable}\nimport os,sys,json\n"
        f"json.dump(dict(argv=sys.argv[1:], env=dict(os.environ)), open({str(capture)!r}, 'w'))\n")
    executable.chmod(0o755)
    monkeypatch.setenv("PATH", str(tmp_path) + ":" + __import__("os").environ["PATH"])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "5,6,7")
    result = subprocess.run(["bash", str(ROOT / "tools/dist_train_runtime.sh"),
        "config with spaces.py", "3", "--detect-anomaly", "off",
        "--work-dir", "new work dir", "--resume-from", "epoch_4.pth"],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    captured = json.loads(capture.read_text())
    assert captured["argv"][-6:] == ["--detect-anomaly", "off", "--work-dir",
                                    "new work dir", "--resume-from", "epoch_4.pth"]
    assert "config with spaces.py" in captured["argv"]
    for key, value in dict(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
            NCCL_P2P_DISABLE="1", NCCL_IB_DISABLE="1", CUDA_VISIBLE_DEVICES="5,6,7").items():
        assert captured["env"][key] == value


def test_gpmsf_anomaly_modes_equivalent_for_n4_burn_in_and_bptt():
    # Same geometry/learned nonzero projections, chunking and checkpoint path.
    # Not a detector/DDP/GPU benchmark or proof of bitwise resumed RNG streams.
    from me_rssm.sanity.gpmsf_cpu_checks import build, bev_inputs, make_geo
    old_threads = torch.get_num_threads()
    old_mode = torch.is_anomaly_enabled()
    old_check_nan = torch.is_anomaly_check_nan_enabled()
    try:
        torch.set_num_threads(1)
        torch.manual_seed(20261014)
        base = build(nonzero_out=True, query_chunk=5, grad_checkpoint=True,
                     collect_stats=True).train()
        frames = [(bev_inputs(seed=100 + i), make_geo()) for i in range(4)]
        results = []
        for mode in (True, False):
            module = copy.deepcopy(base)
            optimizer = torch.optim.AdamW(module.parameters(), lr=1e-4)
            torch.autograd.set_detect_anomaly(mode)
            outputs, leaves, checkpoint_flags = [], [], []
            carry = 0.0
            for i, ((img, rad), original_geo) in enumerate(frames):
                img, rad = img.clone().requires_grad_(), rad.clone().requires_grad_()
                geo = copy.deepcopy(original_geo)
                for key in ("img_feats", "radar_levels"):
                    for value in geo[key]:
                        value.requires_grad_()
                geo["depth"].requires_grad_()
                leaves.extend([img, rad, geo["depth"]] + geo["img_feats"] + geo["radar_levels"])
                with torch.set_grad_enabled(i >= 2):
                    output = module(img + carry * 0.01, rad, geo=geo)
                    carry = output
                checkpoint_flags.append(module.last_stats["checkpoint_active"])
                outputs.append(output.detach().clone())
            output.square().mean().backward()
            grads = {name: None if p.grad is None else p.grad.clone()
                     for name, p in module.named_parameters()}
            assert len(grads) == 60 and all(g is not None for g in grads.values())
            leaf_grads = [None if v.grad is None else v.grad.clone() for v in leaves]
            assert checkpoint_flags == [False, False, True, True]
            optimizer.step()
            results.append((outputs, grads, leaf_grads,
                            {k: v.clone() for k, v in module.state_dict().items()}))
        for group_on, group_off in zip(results[0], results[1]):
            if isinstance(group_on, dict):
                group_on, group_off = list(group_on.values()), list(group_off.values())
            for on, off in zip(group_on, group_off):
                if on is None:
                    assert off is None
                else:
                    assert torch.isfinite(on).all() and torch.equal(on, off)
        assert not torch.cuda.is_initialized()
    finally:
        torch.set_num_threads(old_threads)
        torch.autograd.set_detect_anomaly(old_mode, check_nan=old_check_nan)
