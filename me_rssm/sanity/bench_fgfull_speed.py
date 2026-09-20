"""Controlled single-GPU speed benchmark: N3 vs N4 fgfull configs.

Measures, with identical batch/dataset/loader settings:
  T1. train step (forward + full loss + backward + optimizer)
  T2. inference forward (no_grad, test path)

Warmup iterations are excluded, and CUDA is synchronized around every timed
region so the numbers are real device time, not async-launch noise.

Run:
  CUDA_VISIBLE_DEVICES=3 python me_rssm/sanity/bench_fgfull_speed.py \
      [warmup] [timed] [config path ...]
"""

import os
import sys
import tempfile
import time

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import torch
from mmcv import Config
from mmcv.runner import load_checkpoint, build_optimizer
from mmcv.parallel import MMDataParallel, collate

from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset
from torch.utils.data import DataLoader

CKPT = os.path.join(_REPO, 'checkpoints/pretrained_tj4d.pth')
DEFAULT_CFGS = [
    os.path.join(_REPO, 'configs/r4det/'
                 'TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head.py'),
    os.path.join(_REPO, 'configs/r4det/'
                 'TJ4D-R4Det_fgfull_N3_2x4_24e_pretrained_v2_head.py'),
]


def build(cfg_path):
    cfg = Config.fromfile(cfg_path)
    cfg.model.update(meta_info=dict(
        figures_path=tempfile.mkdtemp(prefix='bench_fig_'),
        project_name='tj4d_bench'))
    model = build_model(cfg.model)
    load_checkpoint(model, CKPT, map_location='cpu', strict=False, logger=None)
    optimizer = build_optimizer(model, cfg.optimizer)
    net = MMDataParallel(model.cuda().train(), device_ids=[0])
    return cfg, net, optimizer


def train_batches(cfg, n):
    ds = build_dataset(cfg.data.train)
    dl = DataLoader(ds, batch_size=cfg.data.samples_per_gpu, shuffle=False,
                    num_workers=2,
                    collate_fn=lambda x: collate(x, samples_per_gpu=2))
    it = iter(dl)
    return [next(it) for _ in range(n)]


def val_batches(cfg, n):
    ds = build_dataset(cfg.data.val)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2,
                    collate_fn=lambda x: collate(x, samples_per_gpu=1))
    it = iter(dl)
    return [next(it) for _ in range(n)]


def main():
    warmup = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    timed = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    cfgs = sys.argv[3:] or DEFAULT_CFGS

    results = {}
    for cfg_path in cfgs:
        label = os.path.basename(cfg_path)
        cfg, net, optimizer = build(cfg_path)
        key = 'N%d_h%d' % (cfg.model.seq_len,
                           cfg.model.temporal_fusion.hidden_dim)
        print('\n=== %s  seq_len=%d hidden_dim=%d ===' % (
            key, cfg.model.seq_len, cfg.model.temporal_fusion.hidden_dim))
        print('config: %s' % label)

        batches = train_batches(cfg, warmup + timed)
        net.train()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.time()
        for batch in batches:
            losses = net(return_loss=True, **batch)
            loss = sum(v for k, v in losses.items()
                       if isinstance(v, torch.Tensor) and v.numel() == 1
                       and k.startswith('loss'))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        train_total = time.time() - t0
        train_per = train_total / (warmup + timed)
        train_peak = torch.cuda.max_memory_allocated() / 2**30
        print('T1 train: %.3f s/iter (%d iters)  peak=%.2f GiB' % (
            train_per, warmup + timed, train_peak))

        vbatches = val_batches(cfg, warmup + timed)
        net.eval()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            for batch in vbatches:
                net(return_loss=False, rescale=True, **batch)
        torch.cuda.synchronize()
        infer_total = time.time() - t0
        infer_per = infer_total / (warmup + timed)
        infer_peak = torch.cuda.max_memory_allocated() / 2**30
        print('T2 inference: %.3f s/sample (batch=1, %d samples)  peak=%.2f GiB' % (
            infer_per, warmup + timed, infer_peak))

        results[key] = dict(train=train_per, infer=infer_per,
                            train_peak=train_peak, infer_peak=infer_peak)
        del net, optimizer
        torch.cuda.empty_cache()

    print('\n==== SUMMARY ====')
    base = results.get('N4_h128')
    if base is not None:
        for tag, k in (('T1 train s/iter', 'train'),
                       ('T2 inference s/sample', 'infer')):
            b = base[k]
            for name in sorted(results):
                if name == 'N4_h128':
                    continue
                a = results[name][k]
                print('%s: %s=%.3f  N4_h128=%.3f  ratio=%.3fx  saves %.1f%%'
                      % (tag, name, a, b, b / a, (b - a) / b * 100))
    for k in sorted(results):
        print(k, results[k])
    return 0


if __name__ == '__main__':
    sys.exit(main())
