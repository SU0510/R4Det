#!/usr/bin/env python3
"""Average checkpoint state dictionaries with explicit integer-buffer handling.

Only floating-point tensors are averaged. Non-floating tensors (for example,
BatchNorm ``num_batches_tracked``) are copied from a designated source
checkpoint, which defaults to the last ``--checkpoints`` entry. The output
contains only ``state_dict`` and ``meta``; optimizer state is intentionally
discarded.

Example:
  python tools/average_checkpoints.py \
    --checkpoints seed/epoch_12.pth seed/epoch_14.pth seed/epoch_16.pth \
    --out seed/epoch_avg_12_14_16.pth
"""

import argparse
import os
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="checkpoint paths to average, normally in epoch order",
    )
    parser.add_argument(
        "--out", required=True, type=Path, help="output checkpoint path"
    )
    parser.add_argument(
        "--integer-from",
        help=(
            "checkpoint path whose non-floating tensors are copied; must be "
            "one of --checkpoints (default: last checkpoint)"
        ),
    )
    return parser.parse_args()


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(f"{path}: expected a checkpoint dict with state_dict")
    return checkpoint


def average_checkpoints(checkpoint_paths, integer_source):
    integer_source = str(Path(integer_source).resolve())
    resolved_paths = [str(Path(path).resolve()) for path in checkpoint_paths]
    if integer_source not in resolved_paths:
        raise ValueError("--integer-from must be one of --checkpoints")

    reference_path = resolved_paths[-1]
    reference = load_checkpoint(reference_path)
    reference_state = reference["state_dict"]
    averaged = {}

    for index, path in enumerate(resolved_paths):
        checkpoint = reference if path == reference_path else load_checkpoint(path)
        state = checkpoint["state_dict"]
        if set(state) != set(reference_state):
            missing = sorted(set(reference_state) - set(state))
            unexpected = sorted(set(state) - set(reference_state))
            raise ValueError(
                f"{path}: state_dict keys differ from reference; "
                f"missing={missing[:5]}, unexpected={unexpected[:5]}"
            )

        for key, value in state.items():
            reference_value = reference_state[key]
            if value.shape != reference_value.shape:
                raise ValueError(
                    f"{path}: shape mismatch for {key}: "
                    f"{tuple(value.shape)} != {tuple(reference_value.shape)}"
                )
            if value.dtype != reference_value.dtype:
                raise ValueError(
                    f"{path}: dtype mismatch for {key}: "
                    f"{value.dtype} != {reference_value.dtype}"
                )
            if not torch.is_floating_point(value):
                continue
            if index == 0:
                averaged[key] = value.clone()
            else:
                averaged[key].add_(value)

    divisor = float(len(resolved_paths))
    for key in averaged:
        averaged[key].div_(divisor)

    for key, value in reference_state.items():
        if not torch.is_floating_point(value):
            averaged[key] = value.clone()

    if len(averaged) != len(reference_state):
        raise RuntimeError("internal error: averaged state_dict size mismatch")

    meta = dict(reference.get("meta", {}))
    meta["checkpoint_averaging"] = {
        "checkpoints": resolved_paths,
        "integer_from": integer_source,
        "floating": "arithmetic_mean",
    }
    return {"state_dict": averaged, "meta": meta}


def main():
    args = parse_args()
    integer_source = args.integer_from or args.checkpoints[-1]
    output = average_checkpoints(args.checkpoints, integer_source)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_name(args.out.name + ".tmp")
    torch.save(output, temporary)
    os.replace(temporary, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
