"""Spatial gate probe: what do the MCI gates do at GT box centers per class?

Read-only diagnostic for the Ped-loose regression question. For a set of
validation samples it runs one normal sequence per sample (eval mode, bus
stashed by the class's ``_stash_gate_maps`` flag), reads the per-pixel gate
maps of the current frame, and samples them in a 3x3 window around each GT
BEV box center, aggregated per class; a background grid (>8 cells from any
GT center) is aggregated as reference.

Interpretation keys (Ped-loose hypothesis test):
  - c_mot / dyn low at Ped centers relative to Car  -> Doppler evidence
    cannot see Pedestrians (weak radial returns), so the reliability/dynamic
    gates classify Ped regions as "low-evidence";
  - g_c / g_r low at Ped centers -> the gates then shrink the temporal
    correction AND the readout there, i.e. Ped loses the temporal benefit
    the ungated baseline enjoys;
  - innov_mag high at Ped centers -> the innovation is large/noisy for Ped
    (state predicts Ped poorly), feeding the gate a misleading signal.

Usage:
  python me_rssm/sanity/probe_gate_spatial.py \
    --config me_rssm/configs/..._mci.py --checkpoint <ckpt> \
    --start-index 3 --limit 6 --stride 700 --output <work_dir>/gate_spatial.json
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
from mmcv import Config  # noqa: E402
from mmcv.runner import load_checkpoint  # noqa: E402
from mmdet3d.datasets import build_dataset  # noqa: E402
from mmdet3d.models import build_detector  # noqa: E402

from tools.diagnose_z_utilization import (  # noqa: E402
    build_batch, history_validity, run_sequence)

MAP_KEYS = ('g_c', 'g_r', 'c_cam', 'c_mot', 'dyn', 'innov_mag')
NEIGH = 1  # 3x3 window around the projected center cell
BG_RADIUS = 8  # cells


def sample_window(map2d, row, col):
    """Mean of a (2*NEIGH+1)^2 window, clamped to the canvas."""
    H, W = map2d.shape[-2:]
    r0, r1 = max(0, row - NEIGH), min(H, row + NEIGH + 1)
    c0, c1 = max(0, col - NEIGH), min(W, col + NEIGH + 1)
    if r1 <= r0 or c1 <= c0:
        return None
    return float(map2d[..., r0:r1, c0:c1].mean().item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--start-index', type=int, default=3)
    ap.add_argument('--limit', type=int, default=6)
    ap.add_argument('--stride', type=int, default=700)
    ap.add_argument('--min-valid-history', type=int, default=2)
    ap.add_argument('--output', required=True)
    a = ap.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    cfg = Config.fromfile(a.config)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model['meta_info'] = {
        'figures_path': '/tmp/r4det_gate_spatial_figs',
        'project_name': 'gate-spatial-probe',
    }
    os.makedirs(cfg.model['meta_info']['figures_path'], exist_ok=True)
    cfg.data.val.pop('samples_per_gpu', None)
    cfg.data.val.pop('workers_per_gpu', None)
    dataset = build_dataset(cfg.data.val)
    model = build_detector(cfg.model, train_cfg=None, test_cfg=None)
    load_checkpoint(model, a.checkpoint, map_location='cpu')
    model = model.to(device).eval()
    fusion = model.temporal_fusion
    assert hasattr(fusion, '_z_intervention'), 'not an MCI variant'
    fusion._stash_gate_maps = True

    # BEV canvas geometry: rows = x (216), cols = y (248).
    scatter = model.pts_middle_encoder
    pc_range = np.array(cfg.point_cloud_range, dtype=np.float64)
    nx = scatter.nx  # x cells (canvas rows after permute)
    ny = scatter.ny  # y cells (canvas cols)
    res_x = (pc_range[3] - pc_range[0]) / nx
    res_y = (pc_range[4] - pc_range[1]) / ny
    print(f'canvas: nx={nx} ny={ny} res=({res_x:.3f},{res_y:.3f}) '
          f'pc_range={pc_range.tolist()}')

    classes = list(dataset.CLASSES)

    def select_indices():
        idx, cursor = [], a.start_index
        while cursor < len(dataset) and len(idx) < a.limit:
            mask = history_validity(dataset, cursor)
            if mask is not None and sum(mask[:-1]) >= a.min_valid_history:
                idx.append(cursor)
                cursor += a.stride
            else:
                cursor += 1
        if not idx:
            raise ValueError('no valid sample selected')
        return idx

    per_class = {c: {k: [] for k in MAP_KEYS} for c in classes}
    per_class['background'] = {k: [] for k in MAP_KEYS}
    counts = {c: 0 for c in classes}

    for index in select_indices():
        batch = build_batch(dataset, [index], 0)
        run_sequence(model, batch, z_mode='normal')
        maps = getattr(fusion, '_last_gate_maps', None)
        assert maps is not None, 'gate maps were not stashed'
        ann = dataset.get_ann_info(index)
        boxes = ann['gt_bboxes_3d']
        labels = ann['gt_labels_3d']
        # gravity center (x, y) in LiDAR frame
        centers = boxes.gravity_center[:, :2].detach().cpu().numpy()
        tensor_maps = {}
        for k in MAP_KEYS:
            tensor_maps[k] = maps[k].detach().float().cpu() if \
                maps.get(k) is not None else None
        ref = tensor_maps['g_r'] if tensor_maps['g_r'] is not None \
            else tensor_maps['g_c']
        H, W = ref.shape[-2:]

        occupied = np.zeros((H, W), dtype=bool)
        for (x, y), label in zip(centers, labels):
            row = int((x - pc_range[0]) / res_x)
            col = int((y - pc_range[1]) / res_y)
            if not (0 <= row < H and 0 <= col < W):
                continue
            name = classes[int(label)]
            counts[name] += 1
            for k in MAP_KEYS:
                v = sample_window(tensor_maps[k], row, col) \
                    if tensor_maps[k] is not None else None
                if v is not None:
                    per_class[name][k].append(v)
            occupied[max(0, row - BG_RADIUS):min(H, row + BG_RADIUS + 1),
                     max(0, col - BG_RADIUS):min(W, col + BG_RADIUS + 1)] = True

        # background grid: every 16th cell not near any GT center
        for r in range(8, H - 8, 16):
            for c in range(8, W - 8, 16):
                if occupied[r, c]:
                    continue
                for k in MAP_KEYS:
                    if tensor_maps[k] is not None:
                        per_class['background'][k].append(
                            float(tensor_maps[k][0, 0, r, c].item()))

    result = {
        'config': a.config,
        'checkpoint': a.checkpoint,
        'gt_counts': counts,
        'per_class': {
            cls: {k: (round(float(np.mean(v)), 5) if v else None)
                  for k, v in maps.items()}
            for cls, maps in per_class.items()
        },
    }
    json.dump(result, open(a.output, 'w'), indent=2)
    print(json.dumps(result, indent=2))
    print('[json]', a.output)


if __name__ == '__main__':
    main()
