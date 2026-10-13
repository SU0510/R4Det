#!/usr/bin/env python3
"""Fail-closed S0: real preflight + two completed, recipe-matched DDP smokes."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from me_rssm.sanity.gpmsf_run_checks import (  # noqa: E402
    completed_phase, numeric_leaves, performance, read_rows, source_hashes, write_json)

ARM_CFG = 'me_rssm/configs/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_gpmsf.py'
MAIN_CFG = 'configs/r4det/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr.py'
FROZEN = dict(max_iters=450, new_params=177726, fusion_conv_params=1475072,
              height_refs=(-1.1230436265, -0.3003779650, 0.6601446867), perf_ratio_cap=1.2)
REQUIRED_REAL = ('common_initial_state', 'initial_rng', 'zero_projections',
                 'train_step0_outputs', 'train_step0_head', 'train_input_unchanged',
                 'eval_step0_outputs', 'eval_step0_head', 'eval_input_unchanged',
                 'real_aug_projection', 'no_gt_geometry', 'two_step_offsets',
                 'native_source_gradients', 'zero_radar_identity',
                 'mixed_batch_identity', 'invalid_image_identity')


def projection_evidence(state):
    result = {}
    for name in ('proj_r2c', 'proj_i2r'):
        for key in ('weight', 'bias'):
            tensor = state.get(f'cross_attention.{name}.{key}')
            if tensor is None:
                raise ValueError(f'missing {name}.{key}')
            import torch
            if not torch.isfinite(tensor).all():
                raise ValueError(f'non-finite {name}.{key}')
            result[f'{name}.{key}'] = dict(abs_mean=float(tensor.abs().mean()),
                                          abs_max=float(tensor.abs().max()))
    if not all(result[f'{name}.weight']['abs_mean'] > 0
               for name in ('proj_r2c', 'proj_i2r')):
        raise ValueError('both projection weights must leave zero')
    return result


def smoke_artifacts(directory, expected_config):
    import torch
    phase = completed_phase(directory)
    if phase.get('config') != expected_config:
        raise ValueError('smoke manifest points to a different arm/config')
    rows = read_rows(directory, since=phase['started_at'])
    train = [r for r in rows if r.get('mode') == 'train']
    if not train or max(r.get('iter', 0) for r in train) != 450:
        raise ValueError('smoke did not log iter 450')
    if not any('loss' in key for r in train for key in r):
        raise ValueError('no logged training losses')
    path = Path(directory) / 'iter_450.pth'
    if path.stat().st_mtime < phase['started_at']:
        raise ValueError('stale smoke checkpoint')
    checkpoint = torch.load(str(path), map_location='cpu')
    meta = checkpoint.get('meta', {})
    # Installed MMCV saves runner.iter BEFORE its post-step increment.
    # iter_450.pth therefore has meta.iter=449; log/hook completion is 450.
    if meta.get('iter') != 449:
        raise ValueError('checkpoint meta.iter is not zero-based 449 for iter_450')
    from mmcv import Config
    try:
        from mmdet.utils import compat_cfg
    except ImportError:
        from mmdet3d.utils import compat_cfg
    actual = compat_cfg(Config.fromstring(meta['config'], '.py'))
    expected = compat_cfg(Config.fromfile(expected_config))
    actual_model = dict(actual.model)
    actual_model.pop('meta_info', None)
    expected_model = dict(expected.model)
    expected_model.pop('meta_info', None)
    if actual_model != expected_model:
        raise ValueError('actual smoke model differs from registered arm')
    for key in ('data', 'optimizer', 'optimizer_config', 'runner', 'lr_config', 'custom_hooks'):
        if actual.get(key) != expected.get(key):
            raise ValueError('actual smoke recipe differs: ' + key)
    if meta.get('seed') != 0 or len(actual.get('gpu_ids', [])) != 3:
        raise ValueError('actual smoke seed/world differs')
    if any(not torch.isfinite(v).all() for v in checkpoint['state_dict'].values()
           if torch.is_tensor(v)):
        raise ValueError('non-finite checkpoint state')
    return rows, checkpoint


def single_variable_contract():
    from mmcv import Config
    import me_rssm  # noqa: F401
    from mmdet3d.models.builder import FUSION_LAYERS
    import torch
    arm, main = Config.fromfile(ARM_CFG), Config.fromfile(MAIN_CFG)
    diff = sorted(k for k in set(arm.model) | set(main.model)
                  if arm.model.get(k) != main.model.get(k))
    # Bootstrap-only values are not research/solver changes.
    ignored = {'model', '_cfg_dir', 'custom_imports'}
    recipe_diff = sorted(k for k in set(arm) | set(main) if k not in ignored
                         and arm.get(k) != main.get(k))
    if diff != ['RCFusion'] or recipe_diff:
        raise ValueError(f'not a single-variable arm: model={diff}, recipe={recipe_diff}')
    fresh = FUSION_LAYERS.build(dict(arm.model.RCFusion))
    base = FUSION_LAYERS.build(dict(main.model.RCFusion))
    if not fresh.require_geo or tuple(fresh.height_refs) != FROZEN['height_refs']:
        raise ValueError('geometry contract changed')
    if not all(torch.count_nonzero(p) == 0 for layer in (fresh.proj_r2c, fresh.proj_i2r)
               for p in layer.parameters()):
        raise ValueError('residual initialization is not zero')
    if fresh.fusion_conv.conv.weight.shape != base.fusion_conv.conv.weight.shape:
        raise ValueError('shared fusion conv shape changed')
    shared = sum(p.numel() for p in fresh.fusion_conv.parameters())
    new = sum(p.numel() for p in fresh.parameters()) - shared
    if (new, shared) != (FROZEN['new_params'], FROZEN['fusion_conv_params']):
        raise ValueError('frozen capacity changed')
    a = Config.fromfile('me_rssm/sanity/gpmsf_smoke_450iter.py')
    b = Config.fromfile('me_rssm/sanity/gpmsf_baseline_smoke_450iter.py')
    if sorted(k for k in set(a.model) | set(b.model) if a.model.get(k) != b.model.get(k)) != ['RCFusion']:
        raise ValueError('resource baseline model is not B0')
    for key in set(a) | set(b):
        if key not in ignored | {'custom_hooks'} and a.get(key) != b.get(key):
            raise ValueError('resource smoke recipe differs: ' + key)
    hooks = [dict(h) for h in a.custom_hooks if h['type'] != 'GPMSFWiringStatsHook']
    if hooks != [dict(h) for h in b.custom_hooks]:
        raise ValueError('non-diagnostic smoke hooks differ')
    return dict(model_diff=diff, recipe_diff=recipe_diff, new_params=new, shared_params=shared)


def verdict(work_dir, baseline_dir, real_path):
    result = dict(S0_ALL_PASS=False, frozen=FROZEN)
    try:
        result['config_contract'] = single_variable_contract()
        real = json.loads(Path(real_path).read_text())
        if (real.get('status') != 'PASS' or real.get('source_sha') != source_hashes()
                or any(real.get('checks', {}).get(k) is not True for k in REQUIRED_REAL)):
            raise ValueError('missing/failed/stale real-batch preflight evidence')
        result['real_preflight'] = real
        rows, checkpoint = smoke_artifacts(work_dir, 'me_rssm/sanity/gpmsf_smoke_450iter.py')
        baseline_rows, _ = smoke_artifacts(baseline_dir, 'me_rssm/sanity/gpmsf_baseline_smoke_450iter.py')
        result['checkpoint_projection_stats'] = projection_evidence(checkpoint['state_dict'])
        wiring = json.loads((Path(work_dir) / 'gpmsf_wiring_stats.json').read_text())
        digests = wiring.get('ddp_parameter_sha', [])
        if (wiring.get('collect_stats') is not True or wiring.get('require_geo') is not True
                or wiring.get('ddp_world_size') != 3 or len(digests) != 3
                or len(set(digests)) != 1 or not digests[0]
                or wiring.get('completed_iters') != 450):
            raise ValueError('missing wiring/DDP sync completion proof')
        records = wiring.get('records', [])
        if [r.get('iter') for r in records] != list(range(50, 451, 50)):
            raise ValueError('incomplete wiring probe records')
        for record in records:
            stats = record.get('stats') or {}
            if (stats.get('checkpoint_active') is not True or
                    stats.get('native_radar_requires_grad') != [True] * 3 or
                    stats.get('native_image_requires_grad') != [True] * 3 or
                    stats.get('native_radar_has_grad_fn') != [True] * 3 or
                    stats.get('native_image_has_grad_fn') != [True] * 3):
                raise ValueError('native current-frame evidence detached or checkpoint disabled')
            if any(not math.isfinite(v) for _, v in numeric_leaves(stats)):
                raise ValueError('non-finite wiring stats')
        result['wiring_probe'] = wiring
        arm_perf, base_perf = performance(rows), performance(baseline_rows)
        result.update(smoke_perf=arm_perf, matched_b0_perf=base_perf,
                      perf_ratio_time=arm_perf['mean_iter_time_s'] / base_perf['mean_iter_time_s'],
                      perf_ratio_memory=arm_perf['max_memory_mib'] / base_perf['max_memory_mib'])
        if max(result['perf_ratio_time'], result['perf_ratio_memory']) > FROZEN['perf_ratio_cap']:
            raise ValueError('matched resource budget exceeded 1.20x; manual review required')
        result['S0_ALL_PASS'] = True
    except Exception as exc:
        result['failure'] = repr(exc)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--baseline-dir', required=True)
    parser.add_argument('--real-preflight', required=True)
    parser.add_argument('--out')
    args = parser.parse_args()
    result = verdict(args.work_dir, args.baseline_dir, args.real_preflight)
    write_json(args.out or str(Path(args.work_dir) / 's0_verdict.json'), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result['S0_ALL_PASS'] else 1


if __name__ == '__main__':
    sys.exit(main())
