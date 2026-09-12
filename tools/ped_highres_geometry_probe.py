#!/usr/bin/env python3
"""Probe Pedestrian geometry decodability from high-resolution BEV crops.

The detector's full high-resolution CenterHead path is intentionally not used
here.  A frozen R4Det checkpoint produces the same current-frame fused BEV and
0.16 m radar scatter used by the high-resolution branch, then this tool caches
small crops around GT Pedestrian centers and trains a tiny geometry head on
those local features only.
"""
import argparse
import json
import os
import random
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv import Config
from mmdet.apis import set_random_seed
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_detector


def _unwrap(value):
    seen = set()
    while (hasattr(value, "data")
           and not isinstance(value, (list, tuple))
           and id(value) not in seen):
        seen.add(id(value))
        value = value.data
    return value


def _to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, list):
        return [_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(_to_device(v, device) for v in value)
    return value


def _unpack_sample(sample):
    """Return per-frame points/images/metas/boxes/labels from a seq_len item."""
    points = _unwrap(sample["points"])
    img = _unwrap(sample["img"])
    img_metas = _unwrap(sample["img_metas"])
    gt_boxes = _unwrap(sample["gt_bboxes_3d"])
    gt_labels = _unwrap(sample["gt_labels_3d"])

    if not isinstance(points, list):
        points = [points]
    if not isinstance(gt_boxes, list):
        gt_boxes = [gt_boxes]
    if not isinstance(gt_labels, list):
        gt_labels = [gt_labels]
    if isinstance(img_metas, list) and img_metas and isinstance(
            img_metas[0], list):
        assert len(img_metas) == 1, 'geometry probe expects one sample'
        img_metas = img_metas[0]
    if not isinstance(img_metas, list):
        img_metas = [img_metas]
    if img.dim() == 5:
        assert img.shape[0] == 1, 'geometry probe expects one sample'
        img = img[0]
    elif img.dim() == 3 and len(points) == 1:
        img = img.unsqueeze(0)
    assert img.shape[0] == len(points)
    assert len(img_metas) == len(points)
    return points, img, img_metas, gt_boxes, gt_labels


def _index_xy(xy, pc_range, voxel_size):
    index = (xy - xy.new_tensor(pc_range[:2])) / xy.new_tensor(voxel_size[:2])
    return index.floor().long()


def _crop_feature(feature, index_x, index_y, crop_size):
    _, channels, height, width = feature.shape
    half = crop_size // 2
    x0, x1 = int(index_x) - half, int(index_x) + half + 1
    y0, y1 = int(index_y) - half, int(index_y) + half + 1
    if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
        return None
    return feature[0, :, y0:y1, x0:x1].detach().cpu()


def _crop_patch(highres, radar, box, highres_voxel,
                pc_range, crop_size):
    high_index = _index_xy(box[:2], pc_range, highres_voxel)
    highres_patch = _crop_feature(
        highres, high_index[0], high_index[1], crop_size)
    radar_patch = _crop_feature(
        radar, high_index[0], high_index[1], crop_size)
    if highres_patch is None or radar_patch is None:
        return None
    return highres_patch, radar_patch


class GeometryHead(nn.Module):
    def __init__(self, input_channels, hidden_channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(hidden_channels, 4)

    def forward(self, x):
        x = self.net(x)
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)


def _build_model(cfg, checkpoint, device):
    cfg.model.update(meta_info={
        "figures_path": os.path.join(cfg.work_dir, "figures_path"),
        "project_name": os.path.basename(cfg.filename).split(".")[0],
    })
    model = build_detector(cfg.model, test_cfg=cfg.get("test_cfg"))
    state = torch.load(checkpoint, map_location="cpu")
    state = state.get("state_dict", state)
    model.load_state_dict(state, strict=False)
    return model.to(device).eval()


@torch.no_grad()
def _extract_sample(model, sample, device):
    points, img, img_metas, boxes, labels = _unpack_sample(sample)
    points = _to_device(points, device)
    img = img.to(device)
    img_metas = _to_device(img_metas, device)

    # Match simple_test(): burn in valid history frames, then evaluate the
    # current frame with the RSSM state produced by that rollout.
    if model.temporal_fusion is not None:
        model.temporal_fusion.reset_state()
        for frame_index in range(len(points) - 1):
            valid = bool(img_metas[frame_index].get("is_prev_frame_valid", True))
            if valid:
                meta = img_metas[frame_index]
                frame_points = points[frame_index]
                if not isinstance(frame_points, list):
                    frame_points = [frame_points]
                model.extract_feat(
                    frame_points,
                    img=img[frame_index],
                    img_metas=[meta],
                    feat_or_dict=0,
                )
            else:
                model.temporal_fusion.reset_for_samples(
                    torch.tensor([True], device=device))

    current = len(points) - 1
    meta = img_metas[current]
    meta["gt_bboxes_3d"] = boxes[current].to(device)
    meta["gt_labels_3d"] = labels[current].to(device)
    current_points = points[current]
    if not isinstance(current_points, list):
        current_points = [current_points]
    feat_dict = model.extract_feat(
        current_points,
        img=img[current],
        img_metas=[meta],
        feat_or_dict=1,
    )
    fused = feat_dict["pts_feats"][0]
    highres = model._make_ped_highres_feats([fused])[0]
    radar = model._highres_radar_scatter
    return fused, highres, radar, boxes[current], labels[current]


def _cache_split(model, dataset, indices, cfg, device, crop_size, cache_path):
    highres_patches, radar_patches = [], []
    targets, source_indices = [], []
    highres_voxel = [cfg.voxel_size[0] / 2, cfg.voxel_size[1] / 2]

    for count, index in enumerate(indices):
        print(f"extracting frame {count + 1}/{len(indices)} (idx={index})",
              flush=True)
        sample = dataset[index]
        fused, highres, radar, boxes, labels = _extract_sample(
            model, sample, device)
        ped_boxes = boxes.tensor[labels == 0].detach().cpu().float()
        for box in ped_boxes:
            patch = _crop_patch(
                highres, radar, box, highres_voxel,
                cfg.point_cloud_range, crop_size)
            if patch is None:
                continue
            hp, rp = patch
            highres_patches.append(hp)
            radar_patches.append(rp)
            source_indices.append(index)
            targets.append([torch.log(box[3]), torch.log(box[4]),
                            torch.sin(box[6]), torch.cos(box[6])])

        if (count + 1) % 25 == 0:
            print(f"cached {count + 1}/{len(indices)} frames", flush=True)

    if not highres_patches:
        raise RuntimeError(f"no Pedestrian crops cached for {cache_path}")
    payload = {
        "highres": torch.stack(highres_patches).half(),
        "radar": torch.stack(radar_patches).half(),
        "targets": torch.tensor(targets, dtype=torch.float32),
        "source_indices": torch.tensor(source_indices, dtype=torch.long),
    }
    torch.save(payload, cache_path)
    print(f"saved {cache_path}: {len(targets)} Ped GT crops", flush=True)
    return payload


def _evaluate(head, data, batch_size, device):
    head.eval()
    preds, targets = [], []
    with torch.no_grad():
        count = data["targets"].shape[0]
        for start in range(0, count, batch_size):
            highres = data["highres"][start:start + batch_size].float().to(device)
            radar = data["radar"][start:start + batch_size].float().to(device)
            inputs = torch.cat((highres, radar), dim=1)
            preds.append(head(inputs).cpu())
            targets.append(data["targets"][start:start + batch_size])

    pred = torch.cat(preds)
    target = torch.cat(targets)
    pred_w = pred[:, 0].exp()
    pred_l = pred[:, 1].exp()
    pred_yaw = torch.atan2(pred[:, 2], pred[:, 3])
    gt_w = target[:, 0].exp()
    gt_l = target[:, 1].exp()
    gt_yaw = torch.atan2(target[:, 2], target[:, 3])
    yaw_err = torch.abs(torch.atan2(torch.sin(pred_yaw - gt_yaw),
                                    torch.cos(pred_yaw - gt_yaw)))
    corr = torch.corrcoef(torch.stack([pred_w, gt_w]))[0, 1].item()
    return {
        "n": int(target.shape[0]),
        "w_mae": float((pred_w - gt_w).abs().mean()),
        "w_corr": float(corr),
        "l_mae": float((pred_l - gt_l).abs().mean()),
        "yaw_mae": float(yaw_err.mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--crop-size", type=int, default=9)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit-train", type=int, default=0)
    parser.add_argument("--limit-val", type=int, default=0)
    args = parser.parse_args()

    set_random_seed(args.seed, deterministic=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cfg = Config.fromfile(args.config)
    cfg.work_dir = args.work_dir
    os.makedirs(cfg.work_dir, exist_ok=True)

    model = _build_model(cfg, args.checkpoint, device)
    train_dataset = build_dataset(cfg.data.train.dataset
                                  if "dataset" in cfg.data.train
                                  else cfg.data.train)
    val_dataset = build_dataset(cfg.data.val)
    train_indices = list(range(len(train_dataset)))
    val_indices = list(range(len(val_dataset)))
    if args.limit_train:
        train_indices = train_indices[:args.limit_train]
    if args.limit_val:
        val_indices = val_indices[:args.limit_val]

    train_cache = os.path.join(cfg.work_dir, f"train_crop{args.crop_size}.pt")
    val_cache = os.path.join(cfg.work_dir, f"val_crop{args.crop_size}.pt")
    if os.path.exists(train_cache):
        train_data = torch.load(train_cache, map_location="cpu")
    else:
        train_data = _cache_split(
            model, train_dataset, train_indices, cfg, device,
            args.crop_size, train_cache)
    if os.path.exists(val_cache):
        val_data = torch.load(val_cache, map_location="cpu")
    else:
        val_data = _cache_split(
            model, val_dataset, val_indices, cfg, device,
            args.crop_size, val_cache)

    input_channels = (train_data["highres"].shape[1]
                      + train_data["radar"].shape[1])
    head = GeometryHead(input_channels).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr)
    criterion = nn.SmoothL1Loss()
    train_count = train_data["targets"].shape[0]
    generator = torch.Generator().manual_seed(args.seed)
    history = []

    for epoch in range(args.epochs):
        head.train()
        order = torch.randperm(train_count, generator=generator)
        total_loss = 0.0
        for start in range(0, train_count, args.batch_size):
            batch = order[start:start + args.batch_size]
            highres = train_data["highres"][batch].float().to(device)
            radar = train_data["radar"][batch].float().to(device)
            target = train_data["targets"][batch].to(device)
            pred = head(torch.cat((highres, radar), dim=1))
            loss = criterion(pred, target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
        metrics = _evaluate(head, val_data, args.batch_size, device)
        metrics["epoch"] = epoch + 1
        metrics["train_loss"] = total_loss / max(train_count, 1)
        history.append(metrics)
        print(json.dumps(metrics), flush=True)

    passed = (
        history[-1]["w_mae"] <= 0.24
        and history[-1]["w_corr"] >= 0.20
        and history[-1]["yaw_mae"] <= 0.45
    )
    result = {
        "config": args.config,
        "checkpoint": args.checkpoint,
        "crop_size": args.crop_size,
        "train_crops": int(train_data["targets"].shape[0]),
        "val_crops": int(val_data["targets"].shape[0]),
        "history": history,
        "passed": passed,
    }
    with open(os.path.join(cfg.work_dir, "probe_result.json"), "w") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({"passed": passed, "final": history[-1]}), flush=True)


if __name__ == "__main__":
    main()
