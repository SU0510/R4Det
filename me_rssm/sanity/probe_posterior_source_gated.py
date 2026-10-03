"""Posterior-source probe for GATED RSSM variants (MCI).

Read-only. Same decomposition as ``tools/probe_posterior_source.py``
(batch-common vs sample-specific energy of h_t / e_pooled / mu_p / mu_q /
correction / innovation, plus the h_only / e_only counterfactuals), with
ONE difference: ``h_t`` is read from the class's eval-mode stash
``fusion._last_h_t`` instead of a forward hook on the ``transition`` module
— the dynamic-gated GRU calls the cell's convs directly, so a module hook
on ``transition`` never fires for MCI (the original probe crashes with
KeyError 'h_t').

The observation side honors ``fusion._posterior_observation(e_t)`` (the
class implements it), so ``e_pooled`` is the e_obs the posterior actually
conditions on, not the raw encoder output.
"""

import argparse
import json
import os
import sys
import warnings

warnings.filterwarnings('ignore')
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from mmcv import Config  # noqa: E402
from mmcv.runner import load_checkpoint  # noqa: E402
from mmdet3d.datasets import build_dataset  # noqa: E402
from mmdet3d.models import build_detector  # noqa: E402

from tools.diagnose_z_utilization import (  # noqa: E402
    build_batch, history_validity, run_sequence, select_indices)


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
    ap.add_argument('--stride', type=int, default=1)
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
        'figures_path': '/tmp/r4det_posterior_source_gated_figs',
        'project_name': 'posterior-source-probe-gated',
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
        fusion.encoder.register_forward_hook(
            make_input_hook('feat_pooled', pool=(16, 16))),
        fusion.encoder.register_forward_hook(make_hook('e_t')),
        fusion.posterior_mu.register_forward_hook(
            make_hook('posterior_head')),
        fusion.prior_mu.register_forward_hook(make_hook('mu_p')),
    ]
    batch_all = build_batch(dataset, idx, 0)
    run_sequence(model, batch_all, z_mode='normal')
    for handle in handles:
        handle.remove()

    # MCI: h_t does not flow through a module whose hook could capture it
    # (the dynamic-gated GRU calls the cell convs directly); read the
    # eval-mode stash of the LAST (current) frame instead.
    assert hasattr(fusion, '_last_h_t') and fusion._last_h_t is not None, \
        'eval run did not stash _last_h_t (not an MCI variant or no forward)'
    h_t = fusion._last_h_t.detach().float()
    h_t = h_t.detach().float()
    if hasattr(fusion, '_posterior_observation'):
        e_pooled = fusion._posterior_observation(captured['e_t'])
    else:
        e_pooled = captured['e_t']
    mu_p = captured['mu_p']
    is_innovation = getattr(
        fusion, 'posterior_struct', 'standard') == 'innovation'
    delta_mu = captured['posterior_head']
    if is_innovation:
        mu_q = mu_p + delta_mu
        correction = delta_mu
    else:
        mu_q = delta_mu
        correction = mu_q - mu_p

    w = fusion.posterior_mu.weight.detach().float()
    innovation_stats = {}
    if is_innovation:
        with torch.no_grad():
            e_hat = fusion.obs_prior(h_t)
            innovation_map = e_pooled - e_hat
            obs_pred = F.smooth_l1_loss(e_hat, e_pooled)
            mu_q_h_only = mu_p + fusion.posterior_mu(
                torch.cat([h_t, torch.zeros_like(innovation_map)], dim=1))
            mu_q_e_only = mu_p + fusion.posterior_mu(
                torch.cat([torch.zeros_like(h_t), innovation_map], dim=1))
        innovation_stats = {
            'obs_prior_e_hat': split(e_hat),
            'innovation': split(innovation_map),
            'stat_obs_pred': float(obs_pred.item()),
        }
    else:
        with torch.no_grad():
            mu_q_h_only = fusion.posterior_mu(
                torch.cat([h_t, torch.zeros_like(e_pooled)], dim=1))
            mu_q_e_only = fusion.posterior_mu(
                torch.cat([torch.zeros_like(h_t), e_pooled], dim=1))

    # Gate-input magnitudes (MCI-specific): the gates observe the detached
    # |innovation| through innovation_mag_conv.
    gate_stats = {}
    if getattr(fusion, 'innov_in_gain', False) and is_innovation and \
            hasattr(fusion, 'innovation_mag_conv'):
        with torch.no_grad():
            innov_mag = fusion.innovation_mag_conv(
                innovation_map.detach().abs())
        gate_stats['innovation_mag_gate_input'] = split(innov_mag)
    if hasattr(fusion, 'correct_gate_conv'):
        gate_stats['correct_gate_conv_weight_rms'] = rms(
            fusion.correct_gate_conv.weight.detach().float())
        gate_stats['correct_gate_conv_bias_abs_mean'] = float(
            fusion.correct_gate_conv.bias.detach().float().abs().mean())
    if hasattr(fusion, 'readout_gate_conv'):
        gate_stats['readout_gate_conv_weight_rms'] = rms(
            fusion.readout_gate_conv.weight.detach().float())

    stats = {
        'indices': idx,
        'fusion_class': type(fusion).__name__,
        'probe': 'posterior_source_gated (h_t from _last_h_t stash)',
        'latent_resolution': list(e_pooled.shape[-2:]),
        'feat_pooled_raw': split(captured['feat_pooled']),
        'e_t_pooled_raw': split(captured['e_t']),
        'e_t_pooled_normalized': split(e_pooled),
        'h_t': split(h_t),
        'e_pooled': split(e_pooled),
        'mu_p': split(mu_p),
        'mu_q': split(mu_q),
        'correction': split(correction),
        'posterior_head_output': split(delta_mu),
        'mu_q_h_only': split(mu_q_h_only),
        'mu_q_e_only': split(mu_q_e_only),
        'posterior_weight_rms_h_half': rms(w[:, :latent_dim]),
        'posterior_weight_rms_e_half': rms(w[:, latent_dim:]),
        'mu_q_removal_of_e_rel_l2': float(
            (mu_q - mu_q_h_only).norm() / mu_q.norm()),
        'mu_q_removal_of_h_rel_l2': float(
            (mu_q - mu_q_e_only).norm() / mu_q.norm()),
    }
    stats.update(innovation_stats)
    stats.update(gate_stats)
    stats['posterior_obs_mode'] = getattr(
        fusion, 'posterior_obs_mode', 'l2')

    json.dump(stats, open(a.output, 'w'), indent=2)
    print(json.dumps(stats, indent=2))
    print('[json]', a.output)


if __name__ == '__main__':
    main()
