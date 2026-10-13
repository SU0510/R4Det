#!/usr/bin/env python
"""Bounded single-GPU cost decomposition for the GP-MSF fusion block.

Read-only w.r.t. training: it builds the *frozen arm config*'s ``RCFusion`` in
isolation (real shapes, real ``_gpmsf_geometry`` contract, no dataset, no DDP,
no optimizer step) and times forward+backward over a small variant matrix.

Why it exists (user execution order step 2, training record section 99/100):
the first S0 measured ``6.7145x`` step time versus B0, but that number comes
from the mis-merged implementation and mixes several effects -- global
``torch.autograd.set_detect_anomaly`` (tools/train_vod.py:12, on for every run,
smoke and 24e alike), gradient-checkpoint recompute, per-chunk statistics
syncs and thousands of tiny per-head kernels.  This script separates them so
the mechanism-preserving speedup work (head vectorisation, compute reuse,
chunk sweep) targets the dominant term instead of guessing.

It writes aggregate JSON only and never touches the queue or a work_dir.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from mmcv import Config                                     # noqa: E402

import me_rssm                                              # noqa: E402,F401
from mmdet3d.models.builder import FUSION_LAYERS            # noqa: E402
from mmdet3d.models.detectors.R4Det import R4Det            # noqa: E402
from mmdet3d.models.fusion_layers.concat_conv_fusion import (   # noqa: E402
    ConcatConvFusion)

ARM_CFG = ('me_rssm/configs/'
           'TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_gpmsf.py')

B = 2                                   # bs2/rank, as in the real smoke
C_IMG, C_RAD, C_OUT = 256, 384, 256
H_BEV, W_BEV = 248, 216
IMG_LEVEL_SHAPES = [(120, 160), (60, 80), (30, 40)]     # P2/P3/P4 of 480x640
RAD_LEVEL_SHAPES = [(248, 216), (124, 108), (62, 54)]   # SECOND 0.32/0.64/1.28
RAD_LEVEL_CH = (64, 128, 256)


def _fake_detector():
    class _Fake:
        _native_pts_levels = [
            torch.zeros(B, c, h, w)
            for c, (h, w) in zip(RAD_LEVEL_CH, RAD_LEVEL_SHAPES)]
    return _Fake()


def _geo_kwargs(seed=0):
    g = torch.Generator().manual_seed(seed)
    return dict(
        img_feats=[torch.randn(B, C_IMG, h, w, generator=g)
                   for h, w in IMG_LEVEL_SHAPES] + [None, None],
        depth=torch.rand(B, 72, 60, 80, generator=g),
        lidar2img=torch.eye(4).repeat(B, 1, 1),             # (B,4,4) single view
        post_rots=torch.eye(3).repeat(B, 1, 1),             # (B,3,3)
        post_trans=torch.zeros(B, 3),                       # (B,3)
        bda=torch.eye(4).repeat(B, 1, 1),
        img_metas=[dict(ori_shape=(375, 1242)) for _ in range(B)],
        points=[torch.rand(20000, 5, generator=g) * 40.0 for _ in range(B)],
    )


def _move(geo, device):
    out = {}
    for key, value in geo.items():
        if torch.is_tensor(value):
            out[key] = value.to(device)
        elif isinstance(value, list) and value and torch.is_tensor(value[0]):
            out[key] = [v.to(device) for v in value]
        else:
            out[key] = value
    return out


def build_arm(**over):
    arm_cfg = dict(Config.fromfile(ARM_CFG).model['RCFusion'])
    arm_cfg.update(over)
    return FUSION_LAYERS.build(dict(arm_cfg))


def build_base():
    return ConcatConvFusion(
        img_channels=C_IMG, rad_channels=C_RAD, out_channels=C_OUT, kernel_size=3,
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        act_cfg=dict(type='ReLU', inplace=True))


def _call(module, img, rad, geo):
    """Only the GP-MSF arm accepts the geometry sources (see wiring test)."""
    if getattr(module, 'accepts_geometry_sources', False):
        return module(img, rad, geo=geo)
    return module(img, rad)


def bench(module, img, rad, geo, anomaly, iters, warmup):
    """Mean/min forward+backward wall time (ms) and peak allocated memory."""
    torch.autograd.set_detect_anomaly(anomaly)
    module.train()
    module.zero_grad(set_to_none=True)
    for _ in range(warmup):
        out = _call(module, img, rad, geo)
        out.sum().backward()
        module.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        start = time.perf_counter()
        out = _call(module, img, rad, geo)
        out.sum().backward()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
        module.zero_grad(set_to_none=True)
    torch.autograd.set_detect_anomaly(False)
    return dict(
        time_ms=1000.0 * sum(times) / len(times),
        time_min_ms=1000.0 * min(times),
        peak_mib=torch.cuda.max_memory_allocated() / 2 ** 20,
        n=len(times))


def kernel_count(module, img, rad, geo, anomaly=False, top=12):
    """CUDA kernel launches for one forward+backward (op-count explanation)."""
    torch.autograd.set_detect_anomaly(anomaly)
    module.train()
    module.zero_grad(set_to_none=True)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        out = _call(module, img, rad, geo)
        out.sum().backward()
    module.zero_grad(set_to_none=True)
    torch.autograd.set_detect_anomaly(False)
    events = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    counts = {}
    for event in events:
        name = event.name.split('(')[0]
        counts[name] = counts.get(name, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])[:top]
    return dict(total=len(events), top=ranked)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='docs/gpmsf_cost_decomposition_20261013.json')
    ap.add_argument('--iters', type=int, default=7)
    ap.add_argument('--warmup', type=int, default=3)
    ap.add_argument('--anomaly-iters', type=int, default=3)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()

    device = torch.device(args.device)
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = False
    img = torch.randn(B, C_IMG, H_BEV, W_BEV, device=device)
    rad = torch.randn(B, C_RAD, H_BEV, W_BEV, device=device)
    geo = _move(R4Det._gpmsf_geometry(_fake_detector(), **_geo_kwargs()), device)

    rows = []
    variants = [
        ('B0_concat', build_base, {}, False),
        ('B0_concat', build_base, {}, True),
        ('GPMSF_frozen', build_arm, {}, False),                      # 24e cfg: ckpt+chunk4096
        ('GPMSF_frozen', build_arm, {}, True),
        ('GPMSF_smoke_stats1', build_arm, dict(collect_stats=True), False),   # what S0 ran
        ('GPMSF_smoke_stats1', build_arm, dict(collect_stats=True), True),
        ('GPMSF_ckpt_c0', build_arm, dict(query_chunk=0), False),
        ('GPMSF_nockpt', build_arm, dict(grad_checkpoint=False), False),
        ('GPMSF_nockpt_c0', build_arm, dict(grad_checkpoint=False, query_chunk=0), False),
        ('GPMSF_nockpt_c2048', build_arm, dict(grad_checkpoint=False, query_chunk=2048), False),
    ]
    for tag, factory, over, anomaly in variants:
        module = factory(**over).to(device)
        iters = args.anomaly_iters if anomaly else args.iters
        warmup = 1 if anomaly else args.warmup
        stat = bench(module, img, rad, geo, anomaly, iters, warmup)
        stat.update(tag=tag, anomaly=anomaly, over={k: v for k, v in over.items()},
                    params_m=sum(p.numel() for p in module.parameters()) / 1e6)
        rows.append(stat)
        print(f"{tag:24s} anomaly={int(anomaly)} "
              f"time={stat['time_ms']:9.2f} ms  peak={stat['peak_mib']:8.1f} MiB "
              f"({stat['params_m']:.3f} M params)", flush=True)
        del module
        torch.cuda.empty_cache()

    kernels = {}
    for tag, factory, over in (('B0_concat', build_base, {}),
                               ('GPMSF_frozen', build_arm, {}),
                               ('GPMSF_smoke_stats1', build_arm, dict(collect_stats=True)),
                               ('GPMSF_ckpt_c0', build_arm, dict(query_chunk=0)),
                               ('GPMSF_nockpt', build_arm, dict(grad_checkpoint=False)),
                               ('GPMSF_nockpt_c0',
                                build_arm, dict(grad_checkpoint=False, query_chunk=0))):
        module = factory(**over).to(device)
        kernels[tag] = kernel_count(module, img, rad, geo)
        print(f'kernels {tag}: {kernels[tag]["total"]}', flush=True)
        del module
        torch.cuda.empty_cache()

    payload = dict(
        batch=B, bev_shape=(H_BEV, W_BEV), arm_cfg=ARM_CFG,
        device=torch.cuda.get_device_name(device), torch=torch.__version__,
        rows=rows, cuda_kernels_per_step=kernels,
        note=('fusion block only: forward+backward of RCFusion with real shapes; '
              'no dataset/DDP/optimizer; anomaly detection mirrors '
              'tools/train_vod.py:12.'),
    )
    with open(args.out, 'w', encoding='utf-8') as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    print('wrote', args.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
