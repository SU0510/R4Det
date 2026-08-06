"""One-iter smoke test for seq_len=3 RSSM. Builds model + one batch, runs
forward_train, prints loss + rssm stats. Does NOT backprop or step.
Usage: python tools/smoke_seq3.py --seq_len 3
"""
import argparse, os, warnings
warnings.filterwarnings('ignore')
os.environ.setdefault('PYTHONWARNINGS', 'ignore')
import torch
from mmcv import Config
from mmcv.parallel import collate, scatter
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_detector


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default='configs/r4det/TJ4D-R4Det_motion_align_rssm_det3d_2x4_12e.py')
    ap.add_argument('--seq_len', type=int, default=3)
    ap.add_argument('--n', type=int, default=2, help='batch size')
    args = ap.parse_args()

    cfg = Config.fromfile(args.config)
    # override seq_len on model + datasets for the test
    cfg.model.seq_len = args.seq_len
    for k in ['train', 'val', 'test']:
        d = getattr(cfg.data, k)
        if 'dataset' in d:  # RepeatDataset
            d.dataset.seq_len = args.seq_len
        else:
            d.seq_len = args.seq_len

    device = 'cuda:0'
    cfg.work_dir = './work_dirs/_smoke_seq3'
    os.makedirs(os.path.join(cfg.work_dir, 'figures_path'), exist_ok=True)
    cfg.model.update(meta_info={'figures_path': os.path.join(cfg.work_dir, 'figures_path'),
                                'project_name': 'TJ4D'})
    model = build_detector(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model = model.to(device).train()

    ds = build_dataset(cfg.data.val)  # val has GT; use for forward_train smoke
    # pick contiguous indices so history is valid
    indices = [200 + i for i in range(args.n)]
    samples = [ds[i] for i in indices]
    batch = collate(samples, samples_per_gpu=args.n)
    # scatter to device (mmcv scatter wants gpu index int under CUDA_VISIBLE_DEVICES)
    batch = scatter(batch, [0])[0]

    # Build the kwargs forward_train expects. The collate/scatter path the
    # real runner uses goes through the dataset's collate -> DataContainer
    # unwrap. Replicate minimal: img tensor [B,N,C,H,W], points list, etc.
    img = batch['img']
    if hasattr(img, 'data'): img = img.data
    if isinstance(img, list): img = img[0]
    img = img.to(device)
    print('img shape:', tuple(img.shape), 'seq_len param:', model.seq_len)

    points = batch['points']
    if hasattr(points, 'data'): points = points.data
    if isinstance(points, list) and len(points) == 1: points = points[0]
    # points: list per sample, each a list of N frame tensors
    print('n samples (points):', len(points), 'frames per sample:', len(points[0]))

    img_metas = batch['img_metas']
    if hasattr(img_metas, 'data'): img_metas = img_metas.data
    if isinstance(img_metas, list) and len(img_metas) == 1: img_metas = img_metas[0]

    # GT: forward_train now expects the full per-sample list of N frames
    # (it extracts frame N-1 internally). Pass the raw scattered structure.
    def to_dev(gt):
        # gt is a per-sample list of N frame box-objects; move each to device
        return [g.to(device) for g in gt]
    gt3d = batch['gt_bboxes_3d']
    if hasattr(gt3d, 'data'): gt3d = gt3d.data
    gtl3d = batch['gt_labels_3d']
    if hasattr(gtl3d, 'data'): gtl3d = gtl3d.data
    # scattered: list per sample, each a list of N frames
    gt3d = [ [g.to(device) for g in sample_frames] for sample_frames in gt3d ]
    gtl3d = [ [g.to(device) for g in sample_frames] for sample_frames in gtl3d ]

    with torch.cuda.amp.autocast(enabled=False):
        losses = model.forward_train(
            points=points,
            img_metas=img_metas,
            gt_bboxes_3d=gt3d,
            gt_labels_3d=gtl3d,
            gt_labels=None,
            gt_bboxes=None,
            img=img,
        )
    print('=== losses keys ===')
    for k in sorted(losses):
        v = losses[k]
        if torch.is_tensor(v):
            print(f'  {k}: {v.item():.5f}' if v.numel() == 1 else f'  {k}: shape {tuple(v.shape)}')
        else:
            print(f'  {k}: {v}')
    print('SMOKE OK')


if __name__ == '__main__':
    main()
