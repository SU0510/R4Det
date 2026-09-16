"""Measure per-class loss gradient directions on the shared BEV feature.

This is a read-only diagnostic for a single training batch. It runs the normal
sequence forward far enough to obtain the shared fused BEV feature, then
replays the anchor-head loss once per class while keeping the same shared
feature graph. For each class it computes the gradient of that class's
classification loss (and optionally regression loss) w.r.t. the shared feature.

Gradients are never stepped into model parameters.

Usage:
  CUDA_VISIBLE_DEVICES=5 python tools/class_grad_cosine.py \
      --config <cfg.py> --checkpoint <ckpt.pth> --gpu-id 0 --batch-size 2
"""
import argparse
import os
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmdet.core import multi_apply

from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_detector


CLASS_NAMES = ('Pedestrian', 'Cyclist', 'Car', 'Truck')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--start-index', type=int, default=200)
    parser.add_argument('--grad-kind', choices=['cls', 'bbox', 'cls_bbox'],
                        default='cls')
    return parser.parse_args()


def unwrap_dc(value):
    if hasattr(value, 'data'):
        return value.data
    return value


def first_if_singleton(value):
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    return value


def collate_batch(dataset, indices, device):
    samples = [dataset[i] for i in indices]
    batch = collate(samples, samples_per_gpu=len(indices))
    batch = scatter(batch, [device])[0]
    device = torch.device('cuda', device)

    img = unwrap_dc(batch['img'])
    img = first_if_singleton(img).to(device)

    points = unwrap_dc(batch['points'])
    points = first_if_singleton(points)

    img_metas = unwrap_dc(batch['img_metas'])
    img_metas = first_if_singleton(img_metas)

    gt3d = unwrap_dc(batch['gt_bboxes_3d'])
    gtl3d = unwrap_dc(batch['gt_labels_3d'])
    gt3d = [[g.to(device) for g in sample_frames] for sample_frames in gt3d]
    gtl3d = [[g.to(device) for g in sample_frames] for sample_frames in gtl3d]

    return dict(
        points=points,
        img=img,
        img_metas=img_metas,
        gt_bboxes_3d=gt3d,
        gt_labels_3d=gtl3d,
    )


def build_shared_feature(model, batch):
    """Run the same sequence/RSSM setup as forward_train, return current BEV."""
    points = batch['points']
    img = batch['img']
    img_metas = batch['img_metas']
    gt_bboxes_3d = batch['gt_bboxes_3d']
    gt_labels_3d = batch['gt_labels_3d']

    N = model.seq_len
    frame_points = [[p[t] for p in points] for t in range(N)]
    frame_img = [img[:, t, ...] for t in range(N)]
    frame_img_metas = [[meta[t] for meta in img_metas] for t in range(N)]
    frame_valid = [
        torch.tensor([meta[t]['is_prev_frame_valid'] for meta in img_metas],
                     device=img.device)
        for t in range(N)
    ]
    frame_gt3d = [[gt[t] for gt in gt_bboxes_3d] for t in range(N)]
    frame_gtl3d = [[gt[t] for gt in gt_labels_3d] for t in range(N)]

    for t in range(N):
        for i in range(len(frame_img_metas[t])):
            frame_img_metas[t][i]['gt_bboxes_3d'] = frame_gt3d[t][i].to(
                frame_gtl3d[t][i].device)
            frame_img_metas[t][i]['gt_labels_3d'] = frame_gtl3d[t][i]

    if model.temporal_fusion is not None:
        model.temporal_fusion.reset_state()
        history_steps = N - 1
        if model.rssm_bptt_steps is None or model.rssm_bptt_steps < 0:
            grad_steps = history_steps
        else:
            grad_steps = min(int(model.rssm_bptt_steps), history_steps)
        burn_in = history_steps - grad_steps
        for t in range(burn_in):
            valid_t = frame_valid[t]
            if valid_t.any():
                with torch.no_grad():
                    model.extract_feat(frame_points[t], frame_img[t],
                                       frame_img_metas[t],
                                       is_valid_mask=valid_t, feat_or_dict=0)
            if (~valid_t).any():
                model.temporal_fusion.reset_for_samples(~valid_t)
        for t in range(burn_in, N - 1):
            valid_t = frame_valid[t]
            if valid_t.any():
                model.extract_feat(frame_points[t], frame_img[t],
                                   frame_img_metas[t],
                                   is_valid_mask=valid_t, feat_or_dict=0,
                                   rssm_detach_state=False)
            if (~valid_t).any():
                model.temporal_fusion.reset_for_samples(~valid_t)

    last_hist_valid = frame_valid[N - 2] if N >= 2 else torch.ones(
        len(frame_img_metas[N - 1]), dtype=torch.bool, device=img.device)
    feature_dict = model.extract_feat(
        frame_points[N - 1], frame_img[N - 1], frame_img_metas[N - 1],
        is_valid_mask=last_hist_valid, feat_or_dict=1)
    return feature_dict, frame_gt3d[N - 1], frame_gtl3d[N - 1]


def class_loss_from_head(head, outs, gt_bboxes, gt_labels, img_metas,
                         class_idx, grad_kind):
    """Replay Anchor3DHead targets/loss with a class mask."""
    cls_scores, bbox_preds, dir_cls_preds = outs
    featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
    device = cls_scores[0].device
    anchor_list = head.get_anchors(featmap_sizes, img_metas, device=device)
    label_channels = head.cls_out_channels if head.use_sigmoid_cls else 1
    gt_bboxes = [b.to(device) for b in gt_bboxes]
    # Keep only this class's GT. anchor_target_3d then generates labels only
    # for this class; positive anchors of other classes are treated as bg for
    # the per-class loss. This isolates the loss signal of one class.
    class_gt_bboxes = []
    class_gt_labels = []
    for boxes, labels in zip(gt_bboxes, gt_labels):
        mask = labels == class_idx
        class_gt_bboxes.append(boxes[mask])
        class_gt_labels.append(labels[mask])

    cls_reg_targets = head.anchor_target_3d(
        anchor_list,
        class_gt_bboxes,
        img_metas,
        gt_bboxes_ignore_list=None,
        gt_labels_list=class_gt_labels,
        num_classes=head.num_classes,
        label_channels=label_channels,
        sampling=head.sampling,
        bev_semantic_mask_list=None)
    if cls_reg_targets is None:
        return None
    (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
     dir_targets_list, dir_weights_list, num_total_pos,
     num_total_neg) = cls_reg_targets
    num_total_samples = (
        num_total_pos + num_total_neg if head.sampling else num_total_pos)
    iou_preds = getattr(head, '_iou_preds', None)
    if iou_preds is None:
        iou_preds = [None] * len(cls_scores)
    anchor_levels = [
        a.reshape(-1, head.box_code_size) for a in anchor_list[0]
    ]
    losses_cls, losses_bbox, _, _ = multi_apply(
        head.loss_single,
        cls_scores,
        bbox_preds,
        dir_cls_preds,
        labels_list,
        label_weights_list,
        bbox_targets_list,
        bbox_weights_list,
        dir_targets_list,
        dir_weights_list,
        iou_preds,
        anchor_levels,
        num_total_samples=num_total_samples)
    loss_cls = sum(losses_cls)
    loss_bbox = sum(losses_bbox)
    if grad_kind == 'cls':
        return loss_cls
    if grad_kind == 'bbox':
        return loss_bbox
    return loss_cls + loss_bbox


def flatten_grad(grad):
    return torch.cat([g.reshape(-1) for g in grad if g is not None])


def compute_grads(feature_dict, head, gt3d, gtl3d, img_metas, grad_kind):
    shared = feature_dict['pts_feats'][0]
    grads = []
    names = []
    for class_idx, class_name in enumerate(CLASS_NAMES):
        loss = class_loss_from_head(head, head([shared]), gt3d, gtl3d,
                                    img_metas, class_idx, grad_kind)
        if loss is None:
            print(f'[skip] {class_name}: no targets')
            grads.append(torch.zeros(shared.numel()))
            names.append(class_name)
            continue
        shared_grad = torch.autograd.grad(
            loss,
            shared,
            retain_graph=True,
            allow_unused=True)[0]
        if shared_grad is None:
            shared_grad = torch.zeros_like(shared)
        grads.append(shared_grad.detach().cpu().reshape(-1))
        names.append(class_name)
        print(f'[loss] {class_name:10s} {loss.item():.6f}')
        del loss, shared_grad
    return grads, names


def cosine(a, b):
    denom = a.norm() * b.norm()
    if denom.item() == 0:
        return float('nan')
    return float(torch.dot(a, b) / denom)


def main():
    args = parse_args()
    torch.cuda.set_device(args.gpu_id)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.model.train_cfg = cfg.model.get('train_cfg') or cfg.get('train_cfg')
    cfg.model['meta_info'] = {
        'figures_path': '/tmp/r4det_class_grad_figures',
        'project_name': 'TJ4D',
    }
    os.makedirs(cfg.model['meta_info']['figures_path'], exist_ok=True)

    dataset = build_dataset(cfg.data.val)
    model = build_detector(cfg.model, train_cfg=None, test_cfg=None)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = model.cuda().train()

    indices = [args.start_index + i for i in range(args.batch_size)]
    batch = collate_batch(dataset, indices, args.gpu_id)
    feature_dict, gt3d, gtl3d = build_shared_feature(model, batch)
    grads, names = compute_grads(
        feature_dict, model.pts_bbox_head, gt3d, gtl3d,
        batch['img_metas'], args.grad_kind)

    print(f'grad_kind={args.grad_kind} samples={indices}')
    print('Shared feature shape:', tuple(feature_dict['pts_feats'][0].shape))
    for name, grad in zip(names, grads):
        print(f'{name:10s} grad_norm={grad.norm().item():.6e}')
    print('Pairwise cosine:')
    for i, name_i in enumerate(names):
        row = []
        for j, name_j in enumerate(names):
            row.append(f'{cosine(grads[i], grads[j]):+.3f}')
        print(f'  {name_i:10s} ' + '  '.join(row))

    mean_grad = torch.stack(grads, dim=0).mean(dim=0)
    print(f'mean_grad_norm={mean_grad.norm().item():.6e}')
    for name, grad in zip(names, grads):
        print(f'{name:10s} cosine_to_mean={cosine(grad, mean_grad):+.3f}')


if __name__ == '__main__':
    main()
