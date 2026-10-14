#!/usr/bin/env python
"""Explicit debug-mode entrypoint; the frozen train_vod.py stays untouched.

All original arguments are forwarded verbatim except --detect-anomaly on/off.
Use a NEW work directory and --resume-from (not load_from) when continuing a
run. Per-rank receipts record this protocol change; they are not an AP gate or
a replacement for the experiment source freeze. A killed rank has no successful
completion receipt. Finite-gradient checks must be verified in the GPU smoke.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys

import torch

ROOT = Path(__file__).resolve().parent.parent
ENTRY = ROOT / "tools/train_vod.py"


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_receipt(path, receipt):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def run_entry(argv=None, *, entry=ENTRY):
    """Load the original entry, then override its import-time debug setting."""
    parser = argparse.ArgumentParser(
        description=__doc__, add_help=False, allow_abbrev=False)
    parser.add_argument("--detect-anomaly", choices=("on", "off"), required=True)
    options, forwarded = parser.parse_known_args(argv)
    # Inspect but DO NOT reconstruct the original command line.
    audit_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    audit_parser.add_argument("--work-dir", required=True)
    audit_parser.add_argument("--config", required=True)
    audit_parser.add_argument("--resume-from")
    audit_parser.add_argument("--local-rank", "--local_rank", default="0")
    audit, _ = audit_parser.parse_known_args(forwarded)
    entry = Path(entry).resolve()
    rank = os.environ.get("RANK", os.environ.get("LOCAL_RANK", audit.local_rank))
    receipt_dir = Path(audit.work_dir).resolve() / "runtime_receipts"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = receipt_dir / f"rank_{rank}_pid_{os.getpid()}.json"
    # Refuse to overwrite an earlier invocation, including failed ones.
    with receipt_path.open("x"):
        pass
    receipt = dict(
        status="loading", started_at=utc_now(), rank=rank, pid=os.getpid(),
        detect_anomaly_requested=options.detect_anomaly,
        wrapper=str(Path(__file__).resolve()), entry=str(entry),
        argv_forwarded=forwarded, work_dir=str(Path(audit.work_dir).resolve()),
        exit_code=None,
    )
    old_argv = sys.argv
    old_mode = torch.is_anomaly_enabled()
    old_check_nan = torch.is_anomaly_check_nan_enabled()
    try:
        write_receipt(receipt_path, receipt)
        receipt.update(wrapper_sha256=sha256(__file__), entry_sha256=sha256(entry),
                       config=str(Path(audit.config).resolve()),
                       config_sha256=sha256(audit.config))
        if audit.resume_from:
            receipt.update(resume_from=str(Path(audit.resume_from).resolve()),
                           resume_sha256=sha256(audit.resume_from))
        sys.argv = [str(entry)] + forwarded
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        namespace = runpy.run_path(str(entry), run_name="__r4det_runtime_entry__")
        # train_vod.py enables anomaly at import time. Setting it BEFORE loading
        # that file would silently be undone. This must be AFTER run_path.
        torch.autograd.set_detect_anomaly(options.detect_anomaly == "on")
        receipt.update(status="running", main_started_at=utc_now(),
                       detect_anomaly_effective=torch.is_anomaly_enabled(),
                       torch_version=torch.__version__)
        write_receipt(receipt_path, receipt)
        print(f"RUNTIME rank={rank} anomaly={options.detect_anomaly} "
              f"receipt={receipt_path}", flush=True)
        namespace["main"]()
    except SystemExit as error:
        code = error.code if isinstance(error.code, int) else (0 if error.code is None else 1)
        receipt.update(status="completed" if code == 0 else "failed", exit_code=code)
        raise
    except BaseException as error:
        receipt.update(status="failed", exit_code=130 if isinstance(error, KeyboardInterrupt) else 1,
                       error=f"{type(error).__name__}: {error}")
        raise
    else:
        receipt.update(status="completed", exit_code=0)
    finally:
        receipt["finished_at"] = utc_now()
        try:
            write_receipt(receipt_path, receipt)
        finally:
            sys.argv = old_argv
            torch.autograd.set_detect_anomaly(old_mode, check_nan=old_check_nan)


if __name__ == "__main__":
    run_entry()
