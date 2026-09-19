"""Radar velocity-evidence chain: pillar stats + scatter subclasses.

Design goal
-----------
In the baseline, the per-point radial velocity (points dim 3) and SNR
(dim 4) are consumed by ``RadarPillarFeatureNet`` as ordinary decorations
(``with_velocity_snr_center`` adds a velocity/SNR cluster-center term,
pillar_encoder.py:598-603) and are then *entangled* into the learned
64-channel pillar features by a linear+BN+ReLU+max. Downstream (SECOND
backbone, ConcatConvFusion, RSSM encoder) nobody can tell appearance from
Doppler any more, and the RSSM transition receives no explicit motion
signal at all.

This module gives the raw Doppler evidence a *parallel, unlearned* path:

    RadarPillarFeatureNetMotion  (VOXEL_ENCODERS, subclass of
                                  RadarPillarFeatureNet)
        returns (main_pillar_feats, motion_stats) where motion_stats is a
        per-pillar (P, 7) tensor of *raw* statistics:

            0: vx = v_mean * x/r          pseudo 2D velocity (longitudinal)
            1: vy = v_mean * y/r          pseudo 2D velocity (lateral)
            2: v_mean                     signed mean radial velocity
            3: |v_mean|                   speed magnitude
            4: v_std                      within-pillar velocity dispersion
                                          (mixed-target / dynamic evidence)
            5: snr_mean                   raw SNR mean
            6: log1p(count)/log1p(M)      normalized pillar occupancy

        The radial direction (x, y)/r is computed from the RAW point
        coordinates *before* the parent forward normalizes xyz in place.
        No new parameters are introduced, so the pretrained TJ4D checkpoint
        keys ``pts_voxel_encoder.*`` keep loading unchanged.

    PointPillarsScatterMotion  (MIDDLE_ENCODERS, subclass of
                                PointPillarsScatter)
        unpacks the (main, motion) tuple (same pattern as the existing
        ``PointPillarsScatterRCS``), scatters the main features exactly as
        the baseline and additionally mean-scatters the motion statistics
        onto a (B, 7, ny, nx) canvas that is stashed on ``ModalityContext``
        under ``'velocity_bev'``. The baseline return value and behaviour
        are bit-identical.

All tensors flow through the existing ``extract_pts_feat`` call sites
(R4Det.py:720-733) without any modification to the detector.
"""

import math

import torch
from mmcv.runner import auto_fp16, force_fp32

from mmdet3d.models.builder import MIDDLE_ENCODERS, VOXEL_ENCODERS
from mmdet3d.models.voxel_encoders.pillar_encoder import RadarPillarFeatureNet
from mmdet3d.models.voxel_encoders.utils import get_paddings_indicator
from mmdet3d.models.middle_encoders.pillar_scatter import PointPillarsScatter

from .modality_bus import ModalityContext

__all__ = ['RadarPillarFeatureNetMotion', 'PointPillarsScatterMotion',
           'MOTION_STAT_CHANNELS']

MOTION_STAT_CHANNELS = 7


@VOXEL_ENCODERS.register_module()
class RadarPillarFeatureNetMotion(RadarPillarFeatureNet):
    """RadarPillarFeatureNet + raw per-pillar Doppler statistics.

    All constructor arguments are identical to ``RadarPillarFeatureNet``;
    this class adds no parameters. ``forward`` returns a tuple
    ``(main_features, motion_stats)`` instead of a single tensor; the
    companion ``PointPillarsScatterMotion`` is the only intended consumer.
    """

    def _motion_stats(self, features, num_points):
        """Per-pillar raw velocity statistics (computed pre-normalization).

        Args:
            features: padded voxel point features (P, M, C>=5) with
                x, y, z, radial velocity, SNR in dims 0..4 (dim 5 may hold
                the RadarStaticDynamicScore; it is ignored here).
            num_points: (P,) true point count per pillar.

        Returns:
            (P, 7) float tensor of the statistics documented above.
        """
        mask = get_paddings_indicator(num_points, features.shape[1], axis=0)
        mask = mask.unsqueeze(-1).type_as(features)          # (P, M, 1)
        denom = mask.sum(dim=1).clamp(min=1.0)               # (P, 1)

        vel = features[:, :, 3]                              # (P, M)
        snr = features[:, :, 4]
        # Raw xyz -- the parent forward normalizes them in place later.
        px = features[:, :, 0]
        py = features[:, :, 1]

        v_mean = (mask * vel.unsqueeze(-1)).sum(dim=1) / denom        # (P,1)
        snr_mean = (mask * snr.unsqueeze(-1)).sum(dim=1) / denom
        v_sq_mean = (mask * vel.unsqueeze(-1) ** 2).sum(dim=1) / denom
        v_std = (v_sq_mean - v_mean ** 2).clamp(min=0.0).sqrt()

        x_mean = (mask * px.unsqueeze(-1)).sum(dim=1) / denom
        y_mean = (mask * py.unsqueeze(-1)).sum(dim=1) / denom
        r = torch.sqrt(x_mean ** 2 + y_mean ** 2).clamp(min=1e-6)
        vx = v_mean * x_mean / r
        vy = v_mean * y_mean / r

        count = mask.sum(dim=1) / float(features.shape[1])            # (P,1)
        # log-squash [0, 1] -> [0, log(2)] so full pillars saturate gently.
        count = torch.log1p(count) / math.log(2.0)

        stats = torch.cat([
            vx, vy, v_mean, v_mean.abs(), v_std, snr_mean, count
        ], dim=-1)                                                    # (P,7)
        return stats

    @force_fp32(out_fp16=True)
    def forward(self, features, num_points, coors):
        # Motion statistics must be computed BEFORE super().forward, which
        # normalizes xyz in place (pillar_encoder.py:221-223 / 576-591).
        motion_stats = self._motion_stats(features, num_points)
        main = super().forward(features, num_points, coors)
        return main, motion_stats.squeeze()


@MIDDLE_ENCODERS.register_module()
class PointPillarsScatterMotion(PointPillarsScatter):
    """Scatter baseline pillar features + stash a velocity-evidence canvas.

    Accepts either the baseline single tensor (behaviour identical to
    ``PointPillarsScatter``) or the ``(main, motion_stats)`` tuple produced
    by ``RadarPillarFeatureNetMotion``. The motion statistics are
    mean-scattered (sum + count divide) so the canvas holds per-cell
    averages of the per-pillar statistics. No parameters are added.
    """

    @auto_fp16(apply_to=('voxel_features', ))
    def forward(self, voxel_features, coors, batch_size=None):
        if isinstance(voxel_features, (tuple, list)):
            main, motion = voxel_features
        else:
            main, motion = voxel_features, None

        canvas_main = super().forward(main, coors, batch_size)

        if motion is not None:
            if batch_size is not None:
                canvas_motion = self._scatter_motion_batch(
                    motion, coors, batch_size)
            else:
                canvas_motion = self._scatter_motion_single(motion, coors)
            ModalityContext.set('velocity_bev', canvas_motion)

        return canvas_main

    def _scatter_motion_single(self, motion, coors):
        return self._scatter_motion_batch(motion, coors, batch_size=1)

    def _scatter_motion_batch(self, motion, coors, batch_size):
        """Mean-scatter (P, 7) stats onto (B, 7, ny, nx)."""
        num_ch = motion.shape[1]
        canvases = []
        for b in range(batch_size):
            canvas = motion.new_zeros(num_ch, self.ny * self.nx)
            counts = motion.new_zeros(1, self.ny * self.nx)
            batch_mask = coors[:, 0] == b
            if batch_mask.any():
                this_coors = coors[batch_mask, :]
                indices = (this_coors[:, 2] * self.nx + this_coors[:, 3])
                indices = indices.type(torch.long)
                # index_put with accumulate: mean over duplicate pillars
                canvas.index_add_(
                    1, indices, motion[batch_mask, :].t().to(canvas.dtype))
                counts.index_add_(
                    1, indices,
                    torch.ones_like(indices, dtype=canvas.dtype).unsqueeze(0))
            canvas = canvas / counts.clamp(min=1.0)
            canvases.append(canvas)
        return torch.stack(canvases, 0).view(
            batch_size, num_ch, self.ny, self.nx)
