#!/usr/bin/env python3
"""Candidate-A z-contract reconciliation (read-only, no training).

WHY THIS EXISTS -- a defect found while auditing the section-72 archive:

Section 72 recorded M1/M3/M4 as read under the DEPLOYMENT contract and then
compared them to bars calibrated on a TRAINING-contract diagnostic, without
accounting for the difference. The two contracts differ in the state itself:

    R4Det.extract_feat (mmdet3d/models/detectors/R4Det.py:891)
        rssm_deterministic = not self.training
    MotionAlignedRSSMFusion.forward (rssm_fusion.py:1637)
        z_t = mu_q if deterministic else self.sample(mu_q, logstd_q)

so TRAINING feeds the semantic head ``mu_q + sigma * eps`` while the eval-mode
probe feeds it ``mu_q``. At ep6 the logged ``stat_posterior_std`` is ~0.23
against a ``mu_q`` RMS of ~0.58, i.e. the noise is a large fraction of the
signal -- so the two readings (stat_sem_thr50 0.5022 train / 0.1167 val) are
not two measurements of one quantity, and the gate's attribution is unsafe
until the gap is decomposed.

This probe decomposes nothing else: SAME checkpoint, SAME val clips, SAME
rasteriser, SAME formulas. It only varies the z contract, and reports

    deterministic=True   (deployment: mu_q)                 -- the S1 reading
    deterministic=False  (training: mu_q + sigma * eps)      -- the stat_* one
    deterministic=prior  (use_posterior=False: mu_p)         -- prior-only ref

Reported per clip and aggregated: ``thr50``, ``hit@1``, ``hit1_chance``,
``hit1_excess_over_chance`` (the same quantities ``stat_sem_*`` logs, so the
train/val gap can be read directly), plus ``sigma/mu`` RMS ratio so the noise
fraction behind any gap is visible rather than assumed.

Read-only: no training, no weight change, no checkpoint written.

Usage:
  CUDA_VISIBLE_DEVICES=5 python me_rssm/sanity/probe_semA_zcontract.py \
    --config me_rssm/configs/..._semreadout.py \
    --checkpoint <work_dir>/epoch_6.pth \
    --start-index 3 --stride 200 --limit 10 \
    --output <work_dir>/semA_zcontract_ep6.json
"""

import argparse
import importlib.util
import json
import os
import statistics
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import torch  # noqa: E402


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_REPO, rel))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ep6 = _load('probe_ep6_for_zcontract',
            'me_rssm/sanity/probe_semreadout_ep6.py')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--gpu-id', type=int, default=0)
    p.add_argument('--start-index', type=int, default=3)
    p.add_argument('--limit', type=int, default=10)
    p.add_argument('--stride', type=int, default=200)
    p.add_argument('--min-valid-history', type=int, default=2)
    p.add_argument('--output', required=True)
    return p.parse_args()


def sigma_over_mu(model, dataset, index, chunks=8):
    """RMS(sigma) / RMS(mu) at the current frame, under the sampling path.

    Reported so a reader can see how much of the train/val gap COULD be
    explained by the noise term instead of assuming it. Uses the same rollout
    and seed as the readings above.
    """
    fusion = model.temporal_fusion
    batch = ep6.dz.build_batch(dataset, [index], torch.cuda.current_device())
    points, img, metas, valid = ep6.dz.build_sequence_frames(batch)
    n = model.seq_len
    captured = {}
    original = fusion.sample

    def spy(mu, logstd):
        captured['mu'] = mu.detach().float()
        captured['sigma'] = torch.exp(
            fusion._constrained_logstd(logstd)).detach().float()
        return original(mu, logstd)

    fusion.sample = spy
    try:
        with torch.no_grad():
            fusion.reset_state()
            for t in range(n - 1):
                if valid[t].any():
                    model.extract_feat(
                        points[t], img[t], metas[t], is_valid_mask=valid[t],
                        feat_or_dict=0, curr_frame_supervision=False,
                        rssm_deterministic=False)
                if (~valid[t]).any():
                    fusion.reset_for_samples(~valid[t])
            last_valid = valid[n - 2] if n >= 2 else torch.ones(
                1, dtype=torch.bool, device=img[n - 1].device)
            model.extract_feat(
                points[n - 1], img[n - 1], metas[n - 1],
                is_valid_mask=last_valid, feat_or_dict=1,
                rssm_deterministic=False)
    finally:
        fusion.sample = original
    if 'mu' not in captured:
        return None
    mu_rms = float(captured['mu'].pow(2).mean().sqrt().item())
    sg_rms = float(captured['sigma'].pow(2).mean().sqrt().item())
    return {'mu_rms': mu_rms, 'sigma_rms': sg_rms,
            'sigma_over_mu': float(sg_rms / mu_rms) if mu_rms else None}


def main():
    from mmcv import Config
    from mmcv.runner import load_checkpoint
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_detector

    import numpy as np

    args = parse_args()
    torch.cuda.set_device(args.gpu_id)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model['meta_info'] = {
        'figures_path': '/tmp/r4det_semA_zcontract_figures',
        'project_name': 'TJ4D-semA-zcontract',
    }
    os.makedirs(cfg.model['meta_info']['figures_path'], exist_ok=True)
    num_classes = int(cfg.model.temporal_fusion.sem_num_classes)
    min_radius = int(cfg.model.temporal_fusion.object_min_radius)
    pcr = list(cfg.model.point_cloud_range)
    cfg.data.val.pop('samples_per_gpu', None)
    cfg.data.val.pop('workers_per_gpu', None)
    dataset = build_dataset(cfg.data.val)
    model = build_detector(cfg.model, train_cfg=None, test_cfg=None)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = model.cuda().eval()
    fusion = model.temporal_fusion
    if getattr(fusion, 'sem_head', None) is None:
        raise RuntimeError('this checkpoint/config has no sem_head')
    fusion.log_sem_maps = True

    indices = ep6.select_clips(dataset, args)

    def read(**kw):
        runs = [ep6.rollout(model, dataset, i, **kw) for i in indices]
        rows = []
        for idx, r in zip(indices, runs):
            q = ep6.head_quality(r, num_classes, pcr, min_radius)
            if q:
                rows.append(dict(q, index=int(idx)))
        return rows

    # The S1 reading, reproduced (must match semreadout_ep6_gate.json).
    deploy_runs = [ep6.rollout(model, dataset, i) for i in indices]
    sampled_runs = [ep6.rollout(model, dataset, i,
                                rssm_deterministic=False) for i in indices]

    def score(runs):
        rows = []
        for idx, r in zip(indices, runs):
            q = ep6.head_quality(r, num_classes, pcr, min_radius)
            if q:
                rows.append(dict(q, index=int(idx)))
        return rows

    deploy, sampled = score(deploy_runs), score(sampled_runs)

    # INTERVENTION CHECK. Two arms can agree because the contract genuinely
    # does not move these statistics -- or because the switch silently did
    # nothing. Those are opposite conclusions from identical numbers, so the
    # movement is measured rather than assumed.
    movement = []
    for idx, a, b in zip(indices, deploy_runs, sampled_runs):
        if a['z'] is None or b['z'] is None:
            continue
        dz = (a['z'] - b['z']).abs().mean().item()
        zr = a['z'].abs().mean().item()
        entry = {'index': int(idx), 'mean_abs_delta_z': dz, 'z_rms': zr,
                 'rel_delta_z': (dz / zr) if zr else None}
        if a['sem_map'] is not None and b['sem_map'] is not None:
            ds = (a['sem_map'] - b['sem_map']).abs().mean().item()
            sr = a['sem_map'].abs().mean().item()
            entry['mean_abs_delta_sem_map'] = ds
            entry['rel_delta_sem_map'] = (ds / sr) if sr else None
        movement.append(entry)

    def mean(rows, key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return float(statistics.mean(vals)) if vals else None

    report = {
        'probe': 'candidate A z-contract reconciliation (read-only)',
        'config': os.path.abspath(args.config),
        'checkpoint': os.path.abspath(args.checkpoint),
        'indices': [int(i) for i in indices],
        'num_classes': num_classes,
        'object_min_radius': min_radius,
        'point_cloud_range': pcr,
        'arms': {
            'deploy_mu_q': {
                'deterministic': True,
                'mean_thr50': mean(deploy, 'thr50'),
                'mean_hit1': mean(deploy, 'hit1'),
                'mean_hit1_chance': mean(deploy, 'hit1_chance'),
                'mean_hit1_excess_over_chance': mean(
                    deploy, 'hit1_excess_over_chance'),
                'per_clip': deploy,
            },
            'train_sampled_z': {
                'deterministic': False,
                'mean_thr50': mean(sampled, 'thr50'),
                'mean_hit1': mean(sampled, 'hit1'),
                'mean_hit1_chance': mean(sampled, 'hit1_chance'),
                'mean_hit1_excess_over_chance': mean(
                    sampled, 'hit1_excess_over_chance'),
                'per_clip': sampled,
            },
        },
        'noise': [dict(sigma_over_mu(model, dataset, i) or {},
                       index=int(i))
                  for i in indices],
        'intervention_check': movement,
        'input_fingerprints': ep6.input_fingerprints(args.config,
                                                     args.checkpoint),
    }
    out = ep6.write_report(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
    print('[json]', out)


if __name__ == '__main__':
    main()
