#!/usr/bin/env python3
"""Score an EXISTING prediction dump with the repository's own evaluation call.

Why this exists
---------------
Every arm of the candidate-C work must be scored by ONE harness (frozen in
``docs/rssm_c_failure_enrichment_contract_20261012.md``). The baseline arm was
scored by ``tools/test_vod.py --eval bbox`` -- i.e. by calling
``dataset.evaluate(outputs, **eval_kwargs)`` on the freshly produced dump. The
controlled-masking arm produces its dump with a different tool
(``tool_c_masked_inference.py``), so it needs the SAME call applied to an
already-written dump. This script is exactly that call and nothing else: no
inference, no thresholding, no re-matching, no re-implementation.

Self-check: run it on the baseline dump and it must reproduce the frozen
``c_failpred_ep14_harness_ap.json`` key-for-key.

Usage
-----
    python me_rssm/sanity/eval_dump_c.py \
        --config <run>/TJ4D-...py --dump <dump.pkl> --out <ap.json> \
        [--metric bbox] [--expected <frozen ap.json>]

With ``--expected`` the tool compares against a frozen AP json and exits
non-zero on a mismatch ("identical" is reported both key-wise and as the
``pts_bbox/KITTI/Overall_3D_moderate`` value).
"""

import argparse
import json
import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import mmcv  # noqa: E402
from mmcv import Config  # noqa: E402
from mmdet3d.datasets import build_dataset  # noqa: E402

try:  # mmdet > 2.23.0 moved these
    from mmdet.utils import compat_cfg  # noqa: E402
except ImportError:  # pragma: no cover
    from mmdet3d.utils import compat_cfg  # noqa: E402

# Exactly the keys tools/test_vod.py strips from cfg.evaluation before calling
# dataset.evaluate (EvalHook-only arguments).
EVAL_HOOK_KEYS = ('interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                  'rule')

# The single number the training log quotes for this metric family.
PRIMARY_KEY = 'pts_bbox/KITTI/Overall_3D_moderate'


def eval_kwargs_from_cfg(cfg, metric, extra=None):
    """Reproduce tools/test_vod.py's eval kwargs construction, verbatim."""
    kwargs = cfg.get('evaluation', {}).copy()
    for key in EVAL_HOOK_KEYS:
        kwargs.pop(key, None)
    kwargs.update(dict(metric=metric))
    if extra:
        kwargs.update(extra)
    return kwargs


def compare_ap(new, expected):
    """Key-wise comparison of two evaluation results (floats: exact)."""
    diff = {}
    for key in set(new) | set(expected):
        left, right = new.get(key), expected.get(key)
        if isinstance(left, dict) and isinstance(right, dict):
            nested = compare_ap(left, right)
            if nested:
                diff[key] = nested
        elif left != right:
            diff[key] = (left, right)
    return diff


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--dump', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--metric', default='bbox')
    parser.add_argument('--expected', default=None,
                        help='frozen AP json to compare against')
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    cfg = compat_cfg(cfg)
    cfg.data.test.test_mode = True
    dataset = build_dataset(cfg.data.test)

    outputs = mmcv.load(args.dump)
    if len(outputs) != len(dataset):
        raise AssertionError(f'dump/dataset length mismatch: {len(outputs)} '
                             f'vs {len(dataset)}')

    kwargs = eval_kwargs_from_cfg(cfg, args.metric)
    result = dataset.evaluate(outputs, **kwargs)
    with open(args.out, 'w', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, sort_keys=True)

    print(f'[eval] {args.metric}: {PRIMARY_KEY} = {result.get(PRIMARY_KEY)}')
    print(f'[eval] ap json -> {os.path.abspath(args.out)}')

    if args.expected:
        expected = json.load(open(args.expected, encoding='utf-8'))
        diff = compare_ap(result, expected)
        print(f'[eval] vs frozen {os.path.basename(args.expected)}: '
              f'{"IDENTICAL" if not diff else "DIFFERS"}')
        if diff:
            print(json.dumps(diff, indent=2, sort_keys=True)[:4000])
            raise SystemExit(2)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
