"""Which input actually drives the posterior mean: h_t or the observation?

Read-only probe. On one validation window it captures h_t, e_pooled and mu_q,
then re-runs the posterior head with one half of its input zeroed. The
batch-common / sample-specific split of each variant answers whether the
collapse originates in the posterior ignoring e_t, or in h_t already being
sample-invariant.
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
import torch.nn.functional as F
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_detector
from tools.diagnose_z_utilization import (
    build_batch, run_sequence, select_indices)


def rms(t):
    return float(t.pow(2).mean().sqrt().item())


def split(t):
    common = t.mean(dim=0, keepdim=True).expand_as(t)
    return {
        'rms': rms(t),
        'batch_common_rms': rms(common),
        'sample_specific_rms': rms(t - common),
        'sample_specific_energy_frac': float(
            (t - common).pow(2).sum() / t.pow(2).sum().clamp_min(1e-12)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--start-index', type=int, default=3)
    ap.add_argument('--limit', type=int, default=4)
    ap.add_argument('--stride', type=int, default=1,
                    help='pick every Nth valid index; >1 measures true '
                         'cross-sample variance instead of within-window')
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
        'figures_path': '/tmp/r4det_posterior_source_figs',
        'project_name': 'posterior-source-probe',
    }
    os.makedirs(cfg.model['meta_info']['figures_path'], exist_ok=True)
    cfg.data.val.pop('samples_per_gpu', None)
    cfg.data.val.pop('workers_per_gpu', None)
    dataset = build_dataset(cfg.data.val)
    model = build_detector(cfg.model, train_cfg=None, test_cfg=None)
    load_checkpoint(model, a.checkpoint, map_location='cpu')
    model = model.cuda().eval()
    fusion = model.temporal_fusion
    latent_dim = fusion.latent_dim

    class A:
        pass

    args = A()
    args.require_valid_history = True
    args.start_index = a.start_index
    args.limit = a.limit
    args.min_valid_history = 2
    if a.stride <= 1:
        idx = select_indices(dataset, args)
    else:
        from tools.diagnose_z_utilization import history_validity
        idx = []
        cursor = a.start_index
        while cursor < len(dataset) and len(idx) < a.limit:
            mask = history_validity(dataset, cursor)
            if mask is not None and sum(mask[:-1]) >= args.min_valid_history:
                idx.append(cursor)
                cursor += a.stride
            else:
                cursor += 1
        if not idx:
            raise ValueError('stride selection found no valid sample')

    captured = {}

    def make_hook(name, pool=None):
        def hook(module, inputs, output):
            value = output.detach().float()
            if pool is not None and tuple(value.shape[-2:]) != tuple(pool):
                value = F.adaptive_avg_pool2d(value, pool)
            captured[name] = value
        return hook

    def make_input_hook(name, pool=(16, 16)):
        def hook(module, inputs, output):
            value = inputs[0].detach().float()
            if tuple(value.shape[-2:]) != tuple(pool):
                value = F.adaptive_avg_pool2d(value, pool)
            captured[name] = value
        return hook

    handles = [
        fusion.encoder.register_forward_hook(make_input_hook('feat_pooled')),
        fusion.latent_gru.register_forward_hook(make_hook('h_t')),
        fusion.encoder.register_forward_hook(
            make_hook('e_t', pool=(16, 16))),
        fusion.posterior_mu.register_forward_hook(make_hook('mu_q')),
        fusion.prior_mu.register_forward_hook(make_hook('mu_p')),
    ]
    batch_all = build_batch(dataset, idx, 0)
    run_sequence(model, batch_all, z_mode='normal')
    for handle in handles:
        handle.remove()

    h_t = captured['h_t']
    # Honor the fusion's own observation mode: route-A's scaled_raw cells do
    # not L2-normalize, so hard-coding F.normalize here would report the
    # pre-change input for a changed model.
    e_pooled = fusion._posterior_observation(captured['e_t'])
    mu_q = captured['mu_q']
    mu_p = captured['mu_p']
    correction = mu_q - mu_p

    with torch.no_grad():
        mu_q_h_only = fusion.posterior_mu(
            torch.cat([h_t, torch.zeros_like(e_pooled)], dim=1))
        mu_q_e_only = fusion.posterior_mu(
            torch.cat([torch.zeros_like(h_t), e_pooled], dim=1))

    w = fusion.posterior_mu.weight.detach().float()
    stats = {
        'indices': idx,
        'readout_mode': getattr(fusion, 'readout_mode', 'film'),
        'feat_pooled_raw': split(captured['feat_pooled']),
        'e_t_pooled_raw': split(captured['e_t']),
        'e_t_pooled_normalized': split(e_pooled),
        'h_t': split(h_t),
        'e_pooled': split(e_pooled),
        'mu_p': split(mu_p),
        'mu_q': split(mu_q),
        'correction': split(correction),
        'mu_q_h_only': split(mu_q_h_only),
        'mu_q_e_only': split(mu_q_e_only),
        'posterior_weight_rms_h_half': rms(w[:, :latent_dim]),
        'posterior_weight_rms_e_half': rms(w[:, latent_dim:]),
        'mu_q_removal_of_e_rel_l2': float(
            (mu_q - mu_q_h_only).norm() / mu_q.norm()),
        'mu_q_removal_of_h_rel_l2': float(
            (mu_q - mu_q_e_only).norm() / mu_q.norm()),
    }
    stats['posterior_obs_mode'] = getattr(
        fusion, 'posterior_obs_mode', 'l2')
    stats['z_proj_init'] = getattr(fusion, 'z_proj_init', None)
    if getattr(fusion, 'z_proj', None) is not None:
        weight = fusion.z_proj.weight.detach().float()
        stats['z_proj_weight_rms'] = rms(weight)

    json.dump(stats, open(a.output, 'w'), indent=2)
    print(json.dumps(stats, indent=2))
    print('[json]', a.output)


if __name__ == '__main__':
    main()
