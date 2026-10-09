"""Shared BEV region definitions for the read-only latent/target diagnostics.

Two probes must agree on what "foreground" and "measured" mean, or their two
numbers ("how much does z_t move the head inside the foreground" and "how much
of the foreground target's variance is sample-specific") would describe
different regions and could not be read against each other:

  me_rssm/sanity/probe_z_util_region_split.py   region split of the
                                                zero_z/shuffle_z/prior_only
                                                counterfactuals
  tools/probe_recon_target.py                   region split of candidate
                                                reconstruction targets

Grid convention (pinned 2026-10-09 from three independent code sites, after
an orientation bug in the first smoke run of the region-split probe): the
BEV EVIDENCE grid -- the fusion's input feature and everything the Doppler
canvas lives on -- is (H_x, W_y) with dim2 = world x and dim3 = world y,
0.32 m cells, pc_range x 0..69.12 over 216 cells and y -39.68..39.68 over 248:

  * the raw Doppler canvas is (B, 7, 2*W_y, 2*H_x) = (B, 7, 496, 432) and the
    scatter's index is coors_y * nx + coors_x, so canvas dim3 is x;
  * EgoWarpEvidenceFieldRSSMFusion._read_context asserts
    vel_raw.shape[-2] == 2*W and vel_raw.shape[-1] == 2*H against the fusion's
    own feature, and _solve_ego derives cell_x = 69.12/H and cell_y = 79.36/W,
    which is exactly 0.32 m only when H = 216 (dim2) and W = 248 (dim3);
  * Anchor3DRangeGenerator.anchors_single_range puts x_centers on feature
    dim3 and y_centers on dim2, i.e. the HEAD's tensor is image-like (y, x).

R4Det permutes the refined feature to the head layout before handing it to
pts_bbox_head, so `feature_dict['pts_feats'][0]` is the TRANSPOSE of the
evidence grid: callers must transpose the last two dims before comparing it
against these masks (see grid_xy_from_canvas + the transposes in the probes).

Region definitions and their limits:
  fg        GT 3D boxes rasterized onto the feature grid. Ideal upper bound of
            "where the objects are" -- available at val time, NOT at test time,
            so it measures headroom, not a deployable mask.
  bg_hi     background cells whose feature energy reaches the foreground
            median. This is the fair control: a raw fg-vs-bg comparison is
            confounded by feature magnitude, which is much larger on objects.
  measured  Doppler canvas occupancy > 0 (a radar return exists in the cell).
  dyn       measured & v_std > threshold -- a Doppler-defined dynamic proxy.
            Known-broken for Pedestrian (docs §68.9b: Ped dynamic evidence
            0.145), reported for contrast only and never used to define fg.
"""

import math

import numpy as np
import torch
import torch.nn.functional as F

CLASS_NAMES = ('Pedestrian', 'Cyclist', 'Car', 'Truck')

# Doppler canvas channels (radar_motion_chain.MOTION_STAT_CHANNELS layout).
CH_V_STD = 4
CH_COUNT = 6
# Threshold on the FEATURE-grid canvas (post 2x-avg-pool): pooling averages a
# 2x2 block, so a single strong mover sub-cell arrives diluted ~4x. 0.2 m/s
# pooled corresponds to movers of ~0.8 m/s and up in the raw canvas -- the
# same domain the fusion's own dynamic gate consumes. `dyn` is a contrast
# region only (see the module docstring): it is not a deployable foreground.
DYN_V_STD_THRESH = 0.2  # m/s, on the pooled canvas


def unwrap(value):
    """Recursively unwrap mmcv DataContainer objects without collating."""
    from mmcv.parallel import DataContainer

    if isinstance(value, DataContainer):
        return unwrap(value.data)
    if isinstance(value, list):
        return [unwrap(item) for item in value]
    if isinstance(value, tuple):
        return tuple(unwrap(item) for item in value)
    return value


def gt_boxes_of_frame(dataset, index, frame_idx=-1):
    """(N, 7) lidar-frame GT boxes and (N,) labels for one frame.

    The val/test pipeline keeps `LoadAnnotations3D(with_bbox_3d=True)` and
    `CustomCollect3D2` collects gt_bboxes_3d, so the boxes are already
    converted to the LiDAR frame the BEV grid lives in (no bda augmentation
    in the test pipeline). Returns (None, None) when the item carries no GT.
    """
    sample = dataset[index]
    boxes = unwrap(sample.get('gt_bboxes_3d'))
    labels = unwrap(sample.get('gt_labels_3d'))
    if boxes is None or len(boxes) == 0:
        return None, None
    entry = boxes[frame_idx]
    tensor = getattr(entry, 'tensor', None)
    if tensor is None:
        return None, None
    if torch.is_tensor(tensor):
        tensor = tensor.detach().cpu().double().numpy()
    else:
        tensor = np.asarray(tensor, dtype=np.float64)
    if labels is None:
        return tensor, None
    lab = labels[frame_idx]
    if torch.is_tensor(lab):
        lab = lab.detach().cpu().numpy()
    lab = np.asarray(lab).reshape(-1)
    return tensor, lab


def cell_centers(grid_hw, pc_range):
    """Cell-center world coordinates and cell sizes for the (H, W) grid."""
    height, width = grid_hw
    x_min, y_min = float(pc_range[0]), float(pc_range[1])
    x_max, y_max = float(pc_range[3]), float(pc_range[4])
    cell_x = (x_max - x_min) / height
    cell_y = (y_max - y_min) / width
    xs = x_min + (np.arange(height) + 0.5) * cell_x
    ys = y_min + (np.arange(width) + 0.5) * cell_y
    return xs, ys, cell_x, cell_y


def rasterize_boxes(boxes, grid_hw, pc_range, margin_cells=0.5):
    """Rasterize (N, 7) lidar boxes (x, y, z, dx, dy, dz, yaw) onto a mask.

    A cell is foreground when its CENTER falls inside the rotated rectangle
    of any box, with `margin_cells` cells of slack on every side. The margin
    exists because GT boxes are used here as a region upper bound; a knife-
    edge mask on 0.32 m cells would cut partly-covered cells off. Returns
    (mask, n_boxes_in_range).
    """
    mask = np.zeros(grid_hw, dtype=bool)
    if boxes is None or len(boxes) == 0:
        return mask, 0
    xs, ys, cell_x, cell_y = cell_centers(grid_hw, pc_range)
    x_min, y_min = float(pc_range[0]), float(pc_range[1])
    x_max, y_max = float(pc_range[3]), float(pc_range[4])
    grid_x = xs[:, None]
    grid_y = ys[None, :]
    n_boxes = 0
    for box in boxes:
        cx, cy, dx, dy, yaw = box[0], box[1], box[3], box[4], box[6]
        if not (dx > 0 and dy > 0):
            continue
        half_l = dx / 2.0 + margin_cells * cell_x
        half_w = dy / 2.0 + margin_cells * cell_y
        if (cx + half_l < x_min or cx - half_l > x_max
                or cy + half_w < y_min or cy - half_w > y_max):
            continue
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        rel_x = grid_x - cx
        rel_y = grid_y - cy
        local_l = cos_y * rel_x + sin_y * rel_y
        local_w = -sin_y * rel_x + cos_y * rel_y
        mask |= (np.abs(local_l) <= half_l) & (np.abs(local_w) <= half_w)
        n_boxes += 1
    return mask, n_boxes


def fg_masks(boxes, labels, grid_hw, pc_range, margin_cells=0.5,
             class_names=CLASS_NAMES):
    """{'fg': mask} plus one 'fg:<class>' mask per present class."""
    masks = {}
    fg, n_boxes = rasterize_boxes(boxes, grid_hw, pc_range, margin_cells)
    masks['fg'] = fg
    masks['n_boxes'] = n_boxes
    if labels is not None and boxes is not None and len(boxes) > 0:
        for class_idx, name in enumerate(class_names):
            subset = boxes[labels == class_idx]
            if len(subset):
                subset_mask, _ = rasterize_boxes(
                    subset, grid_hw, pc_range, margin_cells)
                masks['fg:' + name] = subset_mask
    return masks


def grid_xy_from_canvas(vel_raw):
    """(H_x, W_y) of the evidence grid, derived from the raw Doppler canvas.

    Canvas dim3 is world x and dim2 is world y (see the module docstring), so
    the evidence grid is (dim3 // 2, dim2 // 2). Deriving it from the canvas
    rather than from a captured feature makes the orientation deterministic
    instead of assumed: a transposed grid fails this and the canvas shape
    check in doppler_masks() at the same time.
    """
    if vel_raw is None:
        return None
    return (int(vel_raw.shape[-1]) // 2, int(vel_raw.shape[-2]) // 2)


def doppler_masks(vel_raw, grid_xy, v_std_thresh=DYN_V_STD_THRESH):
    """{'measured', 'dyn'} in the (x, y) layout, or None on shape mismatch."""
    if vel_raw is None:
        return None
    height_x, width_y = grid_xy
    if tuple(vel_raw.shape[-2:]) != (2 * width_y, 2 * height_x):
        return None
    vel = F.avg_pool2d(vel_raw.detach().float(), 2)
    vel = vel.permute(0, 1, 3, 2)  # (B, 7, H_x, W_y): dim2 = x, dim3 = y
    v_std = vel[0, CH_V_STD].cpu().numpy()
    measured = vel[0, CH_COUNT].cpu().numpy() > 0
    return {'measured': measured, 'dyn': measured & (v_std > v_std_thresh)}


def energy_matched_bg(cell_energy, fg_mask):
    """Background cells whose energy reaches the foreground's median."""
    if not bool(fg_mask.any()):
        return np.zeros_like(fg_mask)
    threshold = float(np.median(cell_energy[fg_mask]))
    return (~fg_mask) & (cell_energy >= threshold)


def top_energy_background(cell_energy, fg_mask, quantile=0.9):
    """Background cells in the top (1 - quantile) of background energy.

    A sharper magnitude control than the fg-median rule when the foreground
    median sits low: it holds the background to its brightest decile, so a
    "z matters more on objects" claim survives only if it beats even the
    brightest background cells.
    """
    background = ~fg_mask
    if not bool(background.any()):
        return np.zeros_like(fg_mask)
    threshold = float(np.quantile(cell_energy[background], quantile))
    return background & (cell_energy >= threshold)


def cell_energy(feat_map):
    """Mean |activation| per cell of a (B, C, H, W) or (C, H, W) feature."""
    if feat_map.dim() == 4:
        feat_map = feat_map[0]
    return feat_map.abs().mean(dim=0).detach().float().cpu().numpy()


def masked_mean(value_map, mask):
    """Mean of value_map over masked cells; NaN when the mask is empty."""
    if value_map.dim() == 3:
        value_map = value_map.mean(dim=0)
    if not bool(mask.any()):
        return float('nan')
    return float(value_map[mask].mean().item())


def masked_stats(value_map, base_map, mask):
    """Absolute and relative change of `value_map` vs `base_map` in a region.

    Both maps are per-cell (H, W) or (C, H, W) tensors of the SAME shape and
    carry signed values; `base` is the region's mean |base_map| and `delta`
    the region's mean |value_map - base_map|. The relative number divides one
    by the other so regions with different activation magnitudes (objects are
    much brighter than empty road) can be compared directly.
    """
    if value_map.dim() == 3:
        value_map = value_map.mean(dim=0)
    if base_map.dim() == 3:
        base_map = base_map.mean(dim=0)
    if not bool(mask.any()):
        return {'cells': 0, 'base': float('nan'), 'delta': float('nan'),
                'rel': float('nan')}
    base = float(base_map.abs()[mask].mean().item())
    delta = float((value_map - base_map).abs()[mask].mean().item())
    return {'cells': int(mask.sum()), 'base': base, 'delta': delta,
            'rel': delta / base if base > 0 else float('nan')}


def masked_variance_stats(target, mask):
    """Variance split of a (S, C, H, W) target over masked elements.

    `mask` is (S, H, W) (per-sample regions such as GT boxes) or (H, W).
    total is the mean squared target value over masked elements; specific is
    the same quantity after subtracting the elementwise across-sample mean
    (the sample-invariant template); shared = total - specific, which folds
    the cross term in when the masks differ between samples (documented, not
    hidden). `elements` counts masked target ELEMENTS (S x C x H x W), not
    cells. Matches the unmasked decomposition of tools/probe_recon_target.py
    exactly when the mask is all-ones.
    """
    if not torch.is_tensor(target):
        target = torch.as_tensor(target)
    if not torch.is_tensor(mask):
        mask = torch.as_tensor(mask)
    if mask.dim() == 2:
        mask = mask.unsqueeze(0).expand(target.shape[0], -1, -1)
    elif mask.dim() == 4 and mask.shape[1] == 1:
        mask = mask[:, 0]  # (S, 1, H, W) from a stacked per-sample mask
    weight = mask.unsqueeze(1).expand_as(target).to(target.dtype)
    count = float(weight.sum().item())
    if count == 0:
        return None
    total = float((target.pow(2) * weight).sum().item() / count)
    per_element = weight.sum(dim=0).clamp(min=1.0)
    template = (target * weight).sum(dim=0) / per_element
    specific = float(
        ((target - template).pow(2) * weight).sum().item() / count)
    return {
        'total_var': total,
        'specific_var': specific,
        'shared_var': total - specific,
        'specific_frac': specific / total if total > 0 else float('nan'),
        'shared_frac': (total - specific) / total if total > 0 else float('nan'),
        'elements': int(count),
    }


def _grid_cell_sizes(fusion, feat):
    pc_range = [float(v) for v in fusion._pc_range_t.tolist()]
    height, width = feat.shape[-2:]
    cell_x = (pc_range[3] - pc_range[0]) / height
    cell_y = (pc_range[4] - pc_range[1]) / width
    return cell_x, cell_y


def analytic_ego_velocity(fusion, feat):
    """Per-frame L1 analytic v_ego from the bus canvas, or None.

    Read-only: replicates the three canvas-reading lines of
    EgoWarpEvidenceFieldRSSMFusion._read_context (it does not mutate any
    module state) and returns (v_ego (B, 2), cell_x, cell_y).
    """
    if not getattr(fusion, 'use_ego_warp', False) or \
            not hasattr(fusion, '_solve_ego'):
        return None
    from me_rssm.modality_bus import ModalityContext

    vel_raw = ModalityContext.get('velocity_bev')
    height, width = feat.shape[-2:]
    if vel_raw is None or vel_raw.shape[0] != feat.shape[0] or \
            vel_raw.shape[1] != getattr(fusion, 'motion_channels', 7) or \
            vel_raw.shape[-2] != 2 * width or vel_raw.shape[-1] != 2 * height:
        return None
    vel = F.avg_pool2d(vel_raw.detach().float(), 2).permute(0, 1, 3, 2)
    vel = vel.contiguous()
    v_ego, _ = fusion._solve_ego(vel, feat)
    cell_x, cell_y = _grid_cell_sizes(fusion, feat)
    return v_ego, cell_x, cell_y


def warp_map_by_cells(feat, dx, dy):
    """Sample `feat` at (index + displacement) in cell units.

    Same convention as EgoWarpEvidenceFieldRSSMFusion._read_context (grid
    channel 0 carries the dim3/y index, channel 1 the dim2/x index,
    bilinear + align_corners + zero padding), WITHOUT the module's identity
    anchor: the anchor exists to keep the module bitwise-identity at zero
    shift, while a measurement of the aligned residual wants the plain
    displaced sample.
    """
    batch, _, height, width = feat.shape
    # displacements may come from a CUDA-side solve while the cached maps are
    # CPU tensors; the shift itself is device-agnostic, so coerce to the map
    dx = torch.as_tensor(dx, device=feat.device, dtype=feat.dtype)
    dy = torch.as_tensor(dy, device=feat.device, dtype=feat.dtype)
    index_x = torch.arange(height, device=feat.device,
                           dtype=feat.dtype)[None, :, None] + dx[:, None, None]
    index_y = torch.arange(width, device=feat.device,
                           dtype=feat.dtype)[None, None, :] + dy[:, None, None]
    grid_x = (2.0 * index_y / max(width - 1, 1) - 1.0).expand(batch, height,
                                                             width)
    grid_y = (2.0 * index_x / max(height - 1, 1) - 1.0).expand(batch, height,
                                                              width)
    grid = torch.stack([grid_x, grid_y], dim=-1)
    return F.grid_sample(feat, grid, mode='bilinear', padding_mode='zeros',
                         align_corners=True)
