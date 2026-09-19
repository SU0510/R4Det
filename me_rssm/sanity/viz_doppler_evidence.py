"""Doppler evidence visualization (zero-training data inspection).

Renders, for real TJ4D radar frames, the raw velocity-evidence canvas that
ME-RSSM feeds to the temporal module (`velocity_bev`, scattered by
PointPillarsScatterMotion), next to the raw radar point cloud and the GT
boxes. This is 06_second_layer item 3: paper-figure material + qualitative
data-level sanity that the motion-evidence chain behaves as designed --
moving objects light up in speed (ch3) / velocity dispersion (ch4), static
background stays dark, occupancy (ch6) shows where the sensor is confident.

No training, no metrics, no checkpoints: the canvas is produced by the
parameter-free statistics path (RadarPillarFeatureNetMotion._motion_stats
-> PointPillarsScatterMotion._scatter_motion_batch), so the figures are
valid for ANY checkpoint generation and reproducible from data alone.

Usage:
  python me_rssm/sanity/viz_doppler_evidence.py \
      --config me_rssm/configs/TJ4D-R4Det_me_rssm_N4_24e_pretrained_v2_head.py \
      --indices 0 400 1200 \
      --out me_rssm/figures

Canvas channel map (see radar_motion_chain.py MOTION_STAT docstring):
  0/1: vx/vy pseudo-2D velocity, 2: signed mean radial v, 3: |v_mean|,
  4: v_std (dynamic evidence), 5: SNR mean, 6: pillar occupancy.
"""

import argparse
import os
import sys

import numpy as np
import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

import me_rssm  # noqa: F401,E501  (registers the Motion classes)
from mmcv import Config  # noqa: E402
from mmcv.ops import Voxelization  # noqa: E402
from mmdet3d.datasets import build_dataset  # noqa: E402
from me_rssm.radar_motion_chain import (  # noqa: E402
    PointPillarsScatterMotion, RadarPillarFeatureNetMotion)

CANVAS_CH = {0: r'$v_x$ (m/s)', 1: r'$v_y$ (m/s)', 2: r'$v_r$ mean (m/s)',
             3: r'$|v_r|$ speed', 4: r'$v_r$ $\sigma$ (dynamic)',
             5: 'SNR mean', 6: 'occupancy'}


def build_val_dataset(cfg_path):
    cfg = Config.fromfile(cfg_path)
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    voxel_layer = Voxelization(**cfg.model.pts_voxel_layer)
    pfn_cfg = {k: v for k, v in cfg.model.pts_voxel_encoder.items()
               if k != 'type'}
    pfn_cfg['type'] = 'RadarPillarFeatureNetMotion'
    from mmcv.utils import build_from_cfg
    from mmdet3d.models.builder import VOXEL_ENCODERS
    pfn = build_from_cfg(pfn_cfg, VOXEL_ENCODERS)
    scatter = PointPillarsScatterMotion(**{
        k: v for k, v in cfg.model.pts_middle_encoder.items() if k != 'type'})
    return dataset, voxel_layer, pfn, scatter


def motion_canvas(voxel_layer, pfn, scatter, points):
    """(N,5) points -> ((7, ny, nx) canvas, voxels meta). CPU, no grad."""
    points = points.cpu().float()
    voxels, coors, num_points = voxel_layer(points)
    coors = torch.nn.functional.pad(coors, (1, 0), value=0)  # batch idx
    with torch.no_grad():
        _, motion = pfn(voxels, num_points, coors)
        canvas = scatter._scatter_motion_batch(
            motion.unsqueeze(0) if motion.dim() == 1 else motion,
            coors, batch_size=1)
    return canvas[0], voxels, coors, num_points


def draw_gt(ax, gt_boxes, color='k'):
    """Draw BEV rotated rectangles from (x, y, z, l, w, h, yaw) boxes."""
    if gt_boxes is None:
        return
    try:
        tensor = gt_boxes.tensor
        tensor = tensor.cpu() if hasattr(tensor, 'cpu') else tensor
        boxes = np.asarray(tensor)
    except (AttributeError, NotImplementedError):
        return
    if boxes.size == 0:
        return
    for b in boxes:
        x, y, l, w, yaw = b[0], b[1], b[3], b[4], b[6]
        cos, sin = np.cos(yaw), np.sin(yaw)
        # local corners (length along heading axis, width lateral), rotated
        local = np.array([[l / 2, w / 2], [l / 2, -w / 2],
                          [-l / 2, -w / 2], [-l / 2, w / 2]])
        rot = np.array([[cos, -sin], [sin, cos]])
        corners = local @ rot.T + np.array([x, y])
        ax.add_patch(Polygon(corners, closed=True, fill=False,
                             edgecolor=color, linewidth=1.0))


def _unwrap_dc(field):
    """DataContainer -> inner object (handles the list/tuple wrappers)."""
    obj = field.data if hasattr(field, 'data') else field
    if isinstance(obj, (list, tuple)) and len(obj) == 1:
        obj = obj[0]
    return obj


def render_sample(idx, dataset, voxel_layer, pfn, scatter, out_dir,
                  pc_range):
    data = dataset[idx]
    points_list = _unwrap_dc(data['points'])
    # The TJ4D pipeline returns N consecutive radar frames (the N in
    # "N=4 RSSM"); each frame yields its own Doppler evidence canvas.
    if not isinstance(points_list, (list, tuple)):
        points_list = [points_list]
    gt = _unwrap_dc(data.get('gt_bboxes_3d'))
    # per-frame GT (list of LiDARInstance3DBoxes, one per radar frame)
    if isinstance(gt, (list, tuple)) and len(gt) == len(points_list):
        gts = list(gt)
    elif gt is not None:
        gts = [gt] * len(points_list)
    else:
        gts = [None] * len(points_list)

    x_min, y_min = pc_range[0], pc_range[1]
    x_max, y_max = pc_range[3], pc_range[4]
    extent = [x_min, x_max, y_min, y_max]

    n_frames = len(points_list)
    fig, axes = plt.subplots(n_frames, 4, figsize=(19, 3.2 * n_frames),
                             squeeze=False)

    stats_lines = []
    # adaptive velocity color scale: the Doppler is ego-motion dominated
    # (|v| can reach ~15 m/s), a fixed +-4 m/s scale would saturate.
    all_v = np.concatenate([q.numpy()[:, 3] for q in points_list])
    v_lim = max(2.0, float(np.percentile(np.abs(all_v), 98)))
    for t, points in enumerate(points_list):
        canvas, _, _, num_points = motion_canvas(
            voxel_layer, pfn, scatter, points)
        p = points.numpy()
        n_moving = int((canvas[3] > 1.0).sum())
        stats_lines.append(
            f'frame {t}: pts={p.shape[0]} pillars={int((num_points > 0).sum())} '
            f'cells(|v|>1)={n_moving}')

        ax = axes[t][0]
        sc = ax.scatter(p[:, 0], p[:, 1], c=p[:, 3], s=0.5, cmap='bwr',
                        vmin=-v_lim, vmax=v_lim)
        if t == 0:
            fig.colorbar(sc, ax=ax, fraction=0.04,
                         label=f'v_r (±{v_lim:.0f} m/s)')
        if gts[t] is not None:
            draw_gt(ax, gts[t])
        ax.set_ylabel(f'frame {t}\ny (m)')
        ax.set_xlabel('x (m)')
        ax.set_aspect('equal')
        if t == 0:
            ax.set_title('raw points (radial velocity)')

        for ch, col in [(3, 1), (4, 2), (6, 3)]:
            ax = axes[t][col]
            im = ax.imshow(canvas[ch].numpy(), origin='lower', extent=extent,
                           aspect='equal', cmap='viridis')
            ax.set_xlabel('x (m)')
            if t == 0:
                fig.colorbar(im, ax=ax, fraction=0.04)
                ax.set_title(f'ch{ch}: {CANVAS_CH[ch]}')

    fig.suptitle(f'Doppler evidence canvas · sample idx={idx} '
                 f'({n_frames} frames)\n' + ' | '.join(stats_lines),
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path = os.path.join(out_dir, f'doppler_evidence_idx{idx}.png')
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f'[viz] idx={idx}: {stats_lines[-1]} ({n_frames} frames) -> '
          f'{out_path}')
    return stats_lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=os.path.join(
        _REPO, 'me_rssm/configs',
        'TJ4D-R4Det_me_rssm_N4_24e_pretrained_v2_head.py'))
    ap.add_argument('--indices', type=int, nargs='+', default=[0, 400, 1200])
    ap.add_argument('--out', default=os.path.join(_REPO, 'me_rssm/figures'))
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dataset, voxel_layer, pfn, scatter = build_val_dataset(args.config)
    pc_range = list(voxel_layer.point_cloud_range)

    for idx in args.indices:
        try:
            render_sample(idx, dataset, voxel_layer, pfn, scatter,
                          args.out, pc_range)
        except Exception as e:
            print(f'[viz] idx={idx} FAILED: {type(e).__name__}: {e}')
    print(f'[viz] done -> {args.out}')


if __name__ == '__main__':
    main()
