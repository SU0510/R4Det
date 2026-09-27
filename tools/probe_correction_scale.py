"""Is the sample-specific signal missing from the correction, or just tiny?

Read-only probe. For one validation window it measures:
  * per-sample RMS of mu_q, mu_p and their difference (the innovation)
  * how much of the innovation is batch-common vs sample-specific
  * head_ratio for a batch-permuted correction as the z_proj readout gain is
    scaled up, which separates "the correction carries no sample identity"
    from "the identity is present but the readout is too weak to show it".
"""
import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/lurui/workspace/R4Det')

import numpy as np
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_detector
from tools.diagnose_z_utilization import (
    build_batch, run_sequence, select_indices, vector_l2_ratio)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--start-index', type=int, default=3)
    ap.add_argument('--limit', type=int, default=4)
    ap.add_argument('--shuffle-offset', type=int, default=2)
    ap.add_argument('--output', required=True)
    a = ap.parse_args()

    torch.cuda.set_device(0)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    cfg = Config.fromfile(a.config)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model['meta_info'] = {
        'figures_path': '/tmp/r4det_scale_probe_figs',
        'project_name': 'scale-probe',
    }
    os.makedirs(cfg.model['meta_info']['figures_path'], exist_ok=True)
    cfg.data.val.pop('samples_per_gpu', None)
    cfg.data.val.pop('workers_per_gpu', None)
    dataset = build_dataset(cfg.data.val)
    model = build_detector(cfg.model, train_cfg=None, test_cfg=None)
    load_checkpoint(model, a.checkpoint, map_location='cpu')
    model = model.cuda().eval()
    fusion = model.temporal_fusion

    class A:
        pass

    args = A()
    args.require_valid_history = True
    args.start_index = a.start_index
    args.limit = a.limit
    args.min_valid_history = 2
    idx = select_indices(dataset, args)

    captured = {}

    def make_hook(name):
        def hook(module, inputs, output):
            captured[name] = output.detach().float()
        return hook

    handles = [
        fusion.posterior_mu.register_forward_hook(make_hook('mu_q')),
        fusion.prior_mu.register_forward_hook(make_hook('mu_p')),
    ]

    per_sample_z = {}
    for i in idx:
        batch = build_batch(dataset, [i], 0)
        out = run_sequence(model, batch, z_mode='normal')
        z = out['z_t']
        z = z.reshape(-1, *z.shape[-3:])
        assert z.shape[0] == 1, z.shape
        per_sample_z[i] = z[0]

    sources = {}
    for pos, i in enumerate(idx):
        sources[i] = idx[(pos + a.shuffle_offset) % len(idx)]
    replacement = torch.stack(
        [per_sample_z[sources[i]] for i in idx], 0).cuda()

    batch_all = build_batch(dataset, idx, 0)
    base = run_sequence(model, batch_all, z_mode='normal')
    mu_q = captured['mu_q']
    mu_p = captured['mu_p']
    for handle in handles:
        handle.remove()

    diff = mu_q - mu_p

    def rms(t):
        return float(t.pow(2).mean().sqrt().item())

    common_q = mu_q.mean(dim=0, keepdim=True)
    innovation_common = diff.mean(dim=0, keepdim=True)
    stats = {
        'mu_q_rms': rms(mu_q),
        'mu_p_rms': rms(mu_p),
        'innovation_rms': rms(diff),
        'mu_q_batch_common_rms': rms(common_q.expand_as(mu_q)),
        'mu_q_sample_specific_rms': rms(mu_q - common_q),
        'innovation_batch_common_rms': rms(
            innovation_common.expand_as(diff)),
        'innovation_sample_specific_rms': rms(diff - innovation_common),
        'shuffle_map': {int(i): int(sources[i]) for i in idx},
    }

    has_spatial = getattr(fusion, 'z_proj', None) is not None
    stats['readout_mode'] = getattr(fusion, 'readout_mode', 'film')
    base_head = base['head_vector']
    gains = {}

    if has_spatial:
        z_proj = fusion.z_proj
        base_weight = z_proj.weight.detach().clone()
        base_bias = z_proj.bias.detach().clone()
        for gain in (1.0, 10.0, 100.0, 1000.0):
            with torch.no_grad():
                z_proj.weight.copy_(base_weight * gain)
                z_proj.bias.copy_(base_bias * gain)
            out = run_sequence(
                model, batch_all, z_mode='replace_z',
                replacement_z=replacement)
            gains[str(gain)] = {
                'feature_ratio': vector_l2_ratio(
                    out['features'], base['features']),
                'head_ratio': vector_l2_ratio(out['head_vector'], base_head),
            }
        with torch.no_grad():
            z_proj.weight.copy_(base_weight)
            z_proj.bias.copy_(base_bias)
    stats['shuffle_ratio_by_z_proj_gain'] = gains

    json.dump({'indices': idx, 'stats': stats}, open(a.output, 'w'), indent=2)
    print(json.dumps({'indices': idx, **stats}, indent=2))
    print('[json]', a.output)


if __name__ == '__main__':
    main()
