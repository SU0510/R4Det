"""GPU smoke test for the fgfull config family: real train step + val forward.

Run:  CUDA_VISIBLE_DEVICES=4 python me_rssm/sanity/smoke_fgfull_gpu.py \
          [n_iters] [config path]

Exercises, with the real batch size (samples_per_gpu=2) and the configured
temporal window:
  S1. train forward + full loss assembly + backward + optimizer step
      (2D branch training, IGDR fusion, msk2d/FRPN losses, fg-biased
      relative depth loss, RSSM temporal)
  S2. val forward on real val batches (IGDR test path, RPN inference)
  S3. peak memory report -> decides whether 3-GPU training is safe
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

CFG = os.path.join(_REPO, 'configs/r4det/'
                   'TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head.py')
CKPT = os.path.join(_REPO, 'checkpoints/pretrained_tj4d.pth')


def main():
    n_iters = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    cfg_path = os.path.abspath(sys.argv[2]) if len(sys.argv) > 2 else CFG
    print(f'SMOKE CONFIG: {cfg_path}')
    cfg = Config.fromfile(cfg_path)
    cfg.model.update(meta_info=dict(
        figures_path=tempfile.mkdtemp(prefix='fgfull_smoke_fig_'),
        project_name='tj4d_fgfull_smoke'))

    model = build_model(cfg.model)
    print(f'SMOKE seq_len={cfg.model.seq_len} '
          f'hidden_dim={cfg.model.temporal_fusion.hidden_dim}')
    load_checkpoint(model, CKPT, map_location='cpu', strict=False, logger=None)
    optimizer = build_optimizer(model, cfg.optimizer)
    model = MMDataParallel(model.cuda().train(), device_ids=[0])

    ds = build_dataset(cfg.data.train)
    dl = DataLoader(ds, batch_size=cfg.data.samples_per_gpu, shuffle=False,
                    num_workers=2, collate_fn=lambda x: collate(x, samples_per_gpu=2))
    it = iter(dl)

    # ---- S1: real train steps -------------------------------------------
    ok = True
    for i in range(n_iters):
        batch = next(it)
        t0 = time.time()
        losses = model(return_loss=True, **batch)
        loss = sum(v for k, v in losses.items()
                   if isinstance(v, torch.Tensor) and v.numel() == 1
                   and k.startswith('loss'))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(f'S1 iter {i}: total_loss={float(loss):.3f}  '
              f'({time.time()-t0:.1f}s)  peak={peak:.2f} GiB')
        for k in sorted(losses):
            v = losses[k]
            if isinstance(v, torch.Tensor) and v.numel() == 1:
                print(f'    {k}: {float(v):.4f}')
        if float(loss) != float(loss):  # nan guard
            print('    !! NaN loss detected')
            ok = False
            break

    # ---- S2: val forward (IGDR test path) -------------------------------
    model.eval()
    val_ds = build_dataset(cfg.data.val)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2,
                        collate_fn=lambda x: collate(x, samples_per_gpu=1))
    n_done = 0
    for batch in val_dl:
        if n_done >= 2:
            break
        with torch.no_grad():
            results = model(return_loss=False, rescale=True, **batch)
        n = [len(r['pts_bbox']['boxes_3d']) for r in results]
        print(f'S2 val sample {n_done}: detected boxes per sample: {n}')
        n_done += 1
    ok &= n_done == 2

    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f'S3 PEAK MEMORY: {peak:.2f} GiB '
          f'(3-GPU DDP adds ~0.3-0.5 GiB buckets on top)')
    print('==== SMOKE:', 'ALL OK' if ok else 'FAILURES PRESENT', '====')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
