"""Sanity: radar velocity-evidence chain correctness (CPU, no CUDA).

Builds RadarPillarFeatureNetMotion + PointPillarsScatterMotion with the
mainline parameters and feeds synthetic pillars with known velocities.

Checks:
  1. The baseline pillar-feature canvas returned by the subclass equals
     the one produced by the original PointPillarsScatter (bitwise).
  2. Motion statistics land on the same (row, col) cells as the baseline
     scatter activation for the same pillar (orientation consistency).
  3. Per-channel values are correct: v_mean, |v_mean|, v_std, count_norm.
  4. After avg_pool(2)+permute, the cell index matches the physical (x, y)
     position of the synthetic points in the temporal-fusion canvas
     convention (rows = x, cols = y).
  5. Multi-sample batching keeps canvases independent.
"""

import os
import sys

import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import me_rssm  # noqa: F401  (registers modules)
from me_rssm.radar_motion_chain import (MOTION_STAT_CHANNELS,
                                        PointPillarsScatterMotion,
                                        RadarPillarFeatureNetMotion)
from mmdet3d.models.middle_encoders.pillar_scatter import PointPillarsScatter
from mmdet3d.models.voxel_encoders.pillar_encoder import RadarPillarFeatureNet

torch.manual_seed(0)

PC_RANGE = (0, -39.68, -4, 69.12, 39.68, 2)
VOXEL = (0.16, 0.16, 6.0)
MAX_POINTS = 10


def build_pfn(cls):
    return cls(in_channels=5, feat_channels=[64], with_distance=False,
               voxel_size=VOXEL, point_cloud_range=PC_RANGE, legacy=False,
               with_velocity_snr_center=True)


def make_pillars():
    """Three pillars with known velocity content.

    Pillar A: batch 0, voxel (y=10, x=20), 3 points, v = +5.0, snr = 20
    Pillar B: batch 0, voxel (y=30, x=40), 1 point,  v = -2.5, snr = 10
    Pillar C: batch 1, voxel (y=11, x=21), 2 points, v = 0.0,  snr = 15
    """
    voxels, num_points, coors = [], [], []

    def add(batch, y_idx, x_idx, vs, snr):
        n = len(vs)
        v = torch.tensor(vs, dtype=torch.float32).view(n, 1)
        s = torch.full((n, 1), float(snr))
        # xyz: points inside the pillar, non-zero so decorations are sane
        base_x = x_idx * VOXEL[0] + VOXEL[0] / 2 + PC_RANGE[0]
        base_y = y_idx * VOXEL[1] + VOXEL[1] / 2 + PC_RANGE[1]
        xy = torch.tensor([[base_x, base_y, 0.0]]).repeat(n, 1)
        pts = torch.cat([xy, v, s], dim=1)
        pad = torch.zeros(MAX_POINTS - n, 5)
        voxels.append(torch.cat([pts, pad], dim=0))
        num_points.append(torch.tensor(n))
        coors.append(torch.tensor([batch, 0, y_idx, x_idx]))

    add(0, 10, 20, [5.0, 5.0, 5.0], 20)
    add(0, 30, 40, [-2.5], 10)
    add(1, 11, 21, [0.0, 0.0], 15)
    return (torch.stack(voxels, 0), torch.stack(num_points, 0),
            torch.stack(coors, 0))


def main():
    voxels, num_points, coors = make_pillars()

    pfn_base = build_pfn(RadarPillarFeatureNet)
    pfn_motion = build_pfn(RadarPillarFeatureNetMotion)
    pfn_motion.load_state_dict(pfn_base.state_dict())  # identical weights

    main_base = pfn_base(voxels.clone(), num_points, coors)
    main_new, motion = pfn_motion(voxels.clone(), num_points, coors)
    assert torch.allclose(main_base, main_new, atol=1e-6), \
        'pillar features differ from baseline'
    assert motion.shape == (3, MOTION_STAT_CHANNELS), motion.shape

    scatter_base = PointPillarsScatter(64, output_shape=[496, 432])
    scatter_new = PointPillarsScatterMotion(64, output_shape=[496, 432])
    canvas_base = scatter_base(main_base, coors, batch_size=2)
    canvas_new = scatter_new((main_new, motion), coors, batch_size=2)
    assert torch.allclose(canvas_base, canvas_new, atol=1e-6), \
        'scatter canvas differs from baseline'
    assert canvas_base.shape == (2, 64, 496, 432), canvas_base.shape

    vel = me_rssm.ModalityContext.get('velocity_bev')
    assert vel is not None and vel.shape == (2, 7, 496, 432), \
        f'velocity canvas {None if vel is None else vel.shape}'

    # channel layout: 0 vx, 1 vy, 2 v_mean, 3 |v_mean|, 4 v_std, 5 snr, 6 count
    # pillar A: batch 0, row 10, col 20
    a = vel[0, :, 10, 20]
    assert abs(a[2].item() - 5.0) < 1e-4, f'v_mean A {a[2]}'
    assert abs(a[3].item() - 5.0) < 1e-4, f'|v| A {a[3]}'
    assert abs(a[4].item()) < 1e-4, f'v_std A {a[4]}'
    assert abs(a[5].item() - 20.0) < 1e-3, f'snr A {a[5]}'
    assert abs(a[6].item() - (torch.log1p(torch.tensor(0.3)) /
                              math_log2())) < 1e-5, f'count A {a[6]}'
    # radial direction at (x~3.28m, y~-38.0m): x/r small positive,
    # y/r ~ -1  =>  vx ~ +0.43, vy ~ -4.98
    assert a[0].item() > 0 and abs(a[1].item() + 5.0) < 0.2, \
        f'pseudo velocity A {a[0]}, {a[1]}'

    # pillar B: v = -2.5
    b = vel[0, :, 30, 40]
    assert abs(b[2].item() + 2.5) < 1e-4, f'v_mean B {b[2]}'

    # pillar C: batch independence
    c = vel[1, :, 11, 21]
    assert abs(c[2].item()) < 1e-5, f'v_mean C {c[2]}'
    assert vel[0, :, 11, 21].abs().sum().item() == 0.0, \
        'batch-1 pillar leaked into batch-0 canvas'

    # orientation: after pool+permute, rows = x cells, cols = y cells.
    # NOTE: avg_pool(2) dilutes a single hot sub-cell by 4x (5.0 -> 1.25);
    # the count channel lets the network compensate for this dilution.
    vel_ds = torch.nn.functional.avg_pool2d(vel, 2).permute(0, 1, 3, 2)
    # x=20 (0.16m cells) -> x-cell 10 at 0.32m ; y=10 -> y-cell 5
    assert abs(vel_ds[0, 2, 10, 5].item() - 1.25) < 1e-4, \
        f'orientation {vel_ds[0, 2, 10, 5]}'
    assert abs(vel_ds[0, 2, 20, 15].item() + 0.625) < 1e-4, \
        f'orientation B {vel_ds[0, 2, 20, 15]}'

    print('[PASS] radar motion chain: baseline parity, cell placement, '
          'channel values, batch independence, orientation')


def math_log2():
    import math
    return math.log(2.0)


if __name__ == '__main__':
    main()
