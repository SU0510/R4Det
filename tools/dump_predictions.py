"""Dump raw predictions + GT for error diagnosis (no eval, no AP computation).

Runs single_gpu_test on the val split and saves, for every sample:
  - pred boxes (LiDAR frame), scores, labels (raw model output, before KITTI filter)
  - GT boxes (LiDAR frame), labels, names
so that a separate CPU-only analysis can do confusion-matrix / error / recall
diagnosis without touching CUDA again.

Usage:
  CUDA_VISIBLE_DEVICES=<gpu> python tools/dump_predictions.py \
      --config configs/r4det/<cfg>.py \
      --checkpoint <ckpt.pth> \
      --out <save.pkl>
"""
import argparse
import pickle
import warnings
warnings.filterwarnings('ignore')

import torch
import numpy as np
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model
from mmdet3d.apis import single_gpu_test


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--gpu-id', type=int, default=0)
    p.add_argument('--deterministic', action='store_true', default=True)
    p.add_argument('--limit', type=int, default=0, help='debug: only first N samples')
    return p.parse_args()


def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu_id)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    cfg = Config.fromfile(args.config)
    if args.limit > 0:
        cfg.data.val.ann_file = cfg.data.val.ann_file  # unchanged
    cfg.model.pretrained = None
    cfg.model.train_cfg = None

    # R4Det.__init__ requires meta_info['figures_path'] / ['project_name'].
    import os
    figures_path = os.path.join('/tmp', 'r4det_diag_figures')
    os.makedirs(figures_path, exist_ok=True)
    cfg.model['meta_info'] = {'figures_path': figures_path, 'project_name': 'TJ4D'}

    dataset = build_dataset(cfg.data.val)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=2,
        dist=False,
        shuffle=False)

    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    model.CLASSES = dataset.CLASSES
    _ = load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = MMDataParallel(model, device_ids=[args.gpu_id])

    outputs = single_gpu_test(model, data_loader)

    # Save predictions + GT (GT is already available from dataset info pkl,
    # but we serialize it here so analysis is self-contained and in LiDAR frame).
    assert len(outputs) == len(dataset), f'{len(outputs)} vs {len(dataset)}'
    n = args.limit if args.limit > 0 else len(dataset)

    sample_preds = []
    sample_gts = []
    for i in range(n):
        pd = outputs[i]
        ppts = pd['pts_bbox']
        boxes = ppts['boxes_3d']            # LiDARInstance3DBoxes
        scores = ppts['scores_3d']
        labels = ppts['labels_3d']
        sample_preds.append({
            'tensor': boxes.tensor.cpu().numpy(),   # [x,y,z,w,l,h,yaw]
            'scores': scores.cpu().numpy(),
            'labels': labels.cpu().numpy(),
        })

        ai = dataset.get_ann_info(i)
        gt_boxes = ai['gt_bboxes_3d']       # LiDARInstance3DBoxes
        sample_gts.append({
            'tensor': gt_boxes.tensor.cpu().numpy(),
            'labels': ai['gt_labels_3d'],
            'names': list(ai['gt_names']),
        })

    with open(args.out, 'wb') as f:
        pickle.dump({
            'class_names': list(dataset.CLASSES),
            'preds': sample_preds,
            'gts': sample_gts,
        }, f)
    print(f'[dump] saved {n} samples -> {args.out}')
    print(f'[dump] class_names = {dataset.CLASSES}')


if __name__ == '__main__':
    main()
