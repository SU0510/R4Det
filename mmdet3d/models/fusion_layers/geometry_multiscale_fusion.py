"""Geometry-Paired Multi-Scale Fusion (GP-MSF) -- first single block.

Design (authoritative): ``docs/geometry_multiscale_fusion_design_20261013.md``
CPU audit: ``docs/geometry_multiscale_fusion_audit_20261013.json``
Training record: ``docs/training_runs_full.md`` sections 91 (design), 92 (arm).

Single-variable arm vs ``ConcatConvFusion``.  The baseline path is untouched::

    C_new = C + W_RtoC(M_R)          # (B,64) -> (B,256), zero-initialised
    R_new = R + W_ItoR(M_I)          # (B,64) -> (B,384), zero-initialised
    O_t   = fusion_conv(cat([C_new, R_new], dim=1))   # same conv, same weights

so at step 0 (W == 0) the arm is bitwise identical to the same-seed ConcatConv
baseline.  What the new branch adds is a second, geometry-grounded evidence path
rather than another weight on the existing one:

* bounded metric addresses ``a_hk(p) = p + b_par*tanh(u)*e_r + b_tan*tanh(v)*e_theta``
  in the **augmented** BEV frame, relative to the augmented sensor origin;
* the *same* address reads the three native SECOND levels through their own
  index bookkeeping (0.32/0.64/1.28 m, origin 0.08/-39.60) and the three native
  image levels (P2/P3/P4) through calibration + three train-only height refs;
* a per-candidate masked softmax over source levels ("which resolution explains
  this address") and then a separate masked softmax over candidates ("where is
  the evidence"); the two directions are normalised independently and each keeps
  a value=0 **null** entry;
* new direct radar evidence requires local input-echo support (5x5 window on the
  0.16 m scatter grid); the image->fusion message stays available without it.

No GT/label/segmentation/future-frame input is read, no new loss is added, and
every source tensor is passed per call (no shared global bus), so multi-frame
calls cannot reuse another frame's native features or geometry.

Each head only consumes its own 16-d slice of a source, so this implementation
samples per head instead of materialising the full 64-d candidate tensor (4x
cheaper and mathematically identical).  ``query_chunk`` bounds the forward
working set; ``grad_checkpoint`` recomputes a chunk in backward, because
chunking alone does *not* free the backward-saved activations (design 10.2).

New parameters are built under an isolated RNG so a same-seed run keeps the
identical downstream (RSSM/head) initial state as the baseline arm: the same
seed alone does not guarantee it (design 6.5).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from mmdet3d.models.builder import FUSION_LAYERS
from mmdet3d.models.fusion_layers.concat_conv_fusion import ConcatConvFusion

# --------------------------------------------------------------------------
# Frozen geometry constants (design 5.1/5.3, CPU-verified)
# --------------------------------------------------------------------------
# Train-split-only height coverage prior (weighted q10/q50/q90 of the surface
# CDF).  NOT a height predictor and NOT a per-sample GT value.
DEFAULT_HEIGHT_REFS = (-1.1230436265, -0.3003779650, 0.6601446867)
# native SECOND levels: k3/p1/s2 on the 0.16 m scatter canvas
DEFAULT_RADAR_LEVEL_STEPS = (0.32, 0.64, 1.28)
# physical centre of scatter cell 0 on both axes: (0 + .16/2, -39.68 + .16/2)
DEFAULT_RADAR_ORIGIN = (0.08, -39.60)
DEFAULT_RADAR_BASE_CELL = 0.16
DEFAULT_RADAR_BASE_SHAPE = (496, 432)      # (y, x) of the 0.16 m scatter canvas
DEFAULT_IMG_LEVEL_STRIDES = (4, 8, 16)     # P2/P3/P4 of the 480x640 canvas
# initial per-head (b_parallel, b_tangent) proposals, metres
DEFAULT_OFFSET_BOUNDS = (
    (0.64, 0.32),
    (1.28, 0.32),
    (0.32, 0.64),
    (0.32, 1.28),
)
DEFAULT_BEV_SHAPE = (248, 216)             # (y, x) of the fusion-input layout
DEFAULT_BEV_ORIGIN = (0.0, -39.68)
DEFAULT_BEV_CELL = 0.32
_ALPHA_INIT = 0.5493061443340549           # atanh(0.5)


def _masked_softmax(scores, mask, dim):
    """Softmax over ``dim`` that is exactly zero where ``mask`` is False.

    An all-masked slice yields a finite uniform weight which the mask then
    zeroes, so no NaN can leak into a value aggregation.
    """
    mask = mask.to(torch.bool)
    filled = scores.masked_fill(~mask, -1e4)
    weights = torch.softmax(filled, dim=dim)
    return weights * mask.to(weights.dtype)


def _head_linear(layers, x):
    """Apply the per-head ``Linear``s in ``layers`` to ``x`` (B,Hd,...,dk).

    Exactly the same parameters and the same math as a plain head loop, but
    ``F.linear`` keeps the (M,dk)x(dk,dk) GEMM shape cuBLAS is fast at.  A
    single ``torch.einsum('bhmk,hok->bhmo')`` over the head axis was measured
    at ~20x the GEMM time of these calls (record section 102), so the batched
    head axis is used for the *geometry and sampling* only.
    """
    return torch.stack([layers[h](x[:, h]) for h in range(len(layers))], dim=1)


@FUSION_LAYERS.register_module()
class GeometryPairedMultiScaleFusion(ConcatConvFusion):
    """Camera/radar BEV fusion with paired, geometry-grounded multi-scale evidence.

    ``ConcatConvFusion``'s arguments and ``fusion_conv`` are kept intact, so
    pretrained ``cross_attention.fusion_conv.*`` weights load unchanged; every
    new parameter lives in a new sub-module.

    Args:
        img_channels (int): camera BEV channels (default 256).
        rad_channels (int): radar BEV channels (default 384).
        out_channels (int): fused channels (default 256).
        kernel_size (int): fusion conv kernel (default 3) -- unchanged.
        embed_dim (int): total width of the new evidence path (64 = 4 x 16).
        num_heads (int): heads; ``embed_dim // num_heads`` must be 16.
        num_offsets (int): K addresses per head (k=0 is the fixed centre).
        offset_bounds (tuple): per-head ``(b_parallel, b_tangent)`` metres.
        height_refs (tuple): train-only height coverage prior, metres.
        radar_level_channels (tuple): channels of the native SECOND levels.
        radar_level_steps (tuple): native radar strides, metres.
        radar_origin (tuple): physical centre of scatter cell 0, (x, y).
        radar_base_cell (float): scatter canvas cell size (0.16 m).
        radar_base_shape (tuple): scatter canvas (y, x) for the support mask.
        support_kernel (int): local echo support window (odd, default 5).
        image_num_levels (int): native image levels read (P2/P3/P4).
        image_level_strides (tuple): their strides in pixels.
        image_channels (int): image FPN channels.
        image_size (tuple): (H, W) of the network input canvas.
        depth_downsample (int): stride of the predicted depth map.
        bev_shape (tuple): expected (y, x) of the fusion inputs (asserted).
        bev_origin (tuple): (x, y) of the first BEV cell's lower corner.
        bev_cell (float): BEV cell size, metres.
        min_camera_depth (float): minimum camera-frame depth of a valid projection.
        query_chunk (int): cells per chunk (<=0 means one chunk).
        grad_checkpoint (bool): recompute each chunk in backward.
        gamma_init (float): initial value of the non-negative matching coefficient.
        null_score_init (float): initial learnable null score per head.
        new_param_seed (int): seed of the isolated new-parameter RNG.
        collect_stats (bool): record cheap diagnostics under ``self.last_stats``.
        active_cells_only (bool): skip the candidate chains of cells whose both
            messages are provably zero (implementation switch, section 104.9).
            Diagnostics are python scalars and therefore force a sync; with
            ``grad_checkpoint`` the chunk statistics are counted twice.
    """

    accepts_geometry_sources = True

    def __init__(self,
                 img_channels=256,
                 rad_channels=384,
                 out_channels=256,
                 kernel_size=3,
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                 act_cfg=dict(type='ReLU', inplace=True),
                 embed_dim=64,
                 num_heads=4,
                 num_offsets=4,
                 offset_bounds=DEFAULT_OFFSET_BOUNDS,
                 height_refs=DEFAULT_HEIGHT_REFS,
                 radar_level_channels=(64, 128, 256),
                 radar_level_steps=DEFAULT_RADAR_LEVEL_STEPS,
                 radar_origin=DEFAULT_RADAR_ORIGIN,
                 radar_base_cell=DEFAULT_RADAR_BASE_CELL,
                 radar_base_shape=DEFAULT_RADAR_BASE_SHAPE,
                 support_kernel=5,
                 image_num_levels=3,
                 image_level_strides=DEFAULT_IMG_LEVEL_STRIDES,
                 image_channels=256,
                 image_size=(480, 640),
                 depth_downsample=8,
                 bev_shape=DEFAULT_BEV_SHAPE,
                 bev_origin=DEFAULT_BEV_ORIGIN,
                 bev_cell=DEFAULT_BEV_CELL,
                 min_camera_depth=1e-3,
                 query_chunk=4096,
                 grad_checkpoint=True,
                 gamma_init=1.0,
                 null_score_init=0.0,
                 new_param_seed=20261013,
                 require_geo=False,
                 collect_stats=False,
                 active_cells_only=False):
        super(GeometryPairedMultiScaleFusion, self).__init__(
            img_channels=img_channels,
            rad_channels=rad_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        # kept for the new sub-modules' shapes (ConcatConvFusion stores neither)
        self.img_channels = int(img_channels)
        self.rad_channels = int(rad_channels)
        self.out_channels = int(out_channels)

        assert embed_dim % num_heads == 0, 'embed_dim must divide by num_heads'
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.num_offsets = int(num_offsets)
        assert self.num_offsets >= 1

        self.height_refs = tuple(float(z) for z in height_refs)
        self.num_heights = len(self.height_refs)
        assert self.num_heights >= 1

        self.radar_level_channels = tuple(int(c) for c in radar_level_channels)
        self.num_rad_levels = len(self.radar_level_channels)
        self.radar_level_steps = tuple(float(s) for s in radar_level_steps)
        assert len(self.radar_level_steps) == self.num_rad_levels

        self.radar_origin = (float(radar_origin[0]), float(radar_origin[1]))
        self.radar_base_cell = float(radar_base_cell)
        self.radar_base_shape = (int(radar_base_shape[0]), int(radar_base_shape[1]))
        self.support_kernel = int(support_kernel)
        assert self.support_kernel % 2 == 1, 'support_kernel must be odd'

        self.image_num_levels = int(image_num_levels)
        self.image_level_strides = tuple(int(s) for s in image_level_strides)
        assert len(self.image_level_strides) == self.image_num_levels
        self.image_channels = int(image_channels)
        self.image_size = (int(image_size[0]), int(image_size[1]))
        self.depth_downsample = int(depth_downsample)

        self.bev_shape = (int(bev_shape[0]), int(bev_shape[1]))
        self.bev_origin = (float(bev_origin[0]), float(bev_origin[1]))
        self.bev_cell = float(bev_cell)
        self.min_camera_depth = float(min_camera_depth)

        self.query_chunk = int(query_chunk)
        self.grad_checkpoint = bool(grad_checkpoint)
        self.collect_stats = bool(collect_stats)
        # Resource-only switch (training record section 104.9): run the dense
        # candidate chains only on cells whose two messages can be non-zero.
        # Exact by construction (see ``_active_cells``); no new mechanism.
        self.active_cells_only = bool(active_cells_only)
        self.new_param_seed = int(new_param_seed)
        # Train/eval hard guard against a silently unwired arm (the detector
        # must pass `geo`; otherwise this module would be the plain baseline).
        self.require_geo = bool(require_geo)
        self.last_stats = None

        self.register_buffer('offset_bounds',
                             torch.tensor(offset_bounds, dtype=torch.float32),
                             persistent=False)
        self.register_buffer('z_refs',
                             torch.tensor(self.height_refs, dtype=torch.float32),
                             persistent=False)
        self.register_buffer('_pos_enc', self._build_pos_enc(), persistent=False)

        # Isolated RNG: shared modules built *after* the fusion (RSSM, head,
        # proposal layers) must see the same RNG stream as the baseline arm.
        rng_state = torch.random.get_rng_state()
        try:
            torch.random.default_generator.manual_seed(self.new_param_seed)
            self._build_new_params(gamma_init=float(gamma_init),
                                   null_score_init=float(null_score_init))
        finally:
            torch.random.set_rng_state(rng_state)

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def _build_pos_enc(self):
        """Fixed metric sine/cosine position code of the BEV query grid."""
        h, w = self.bev_shape
        d = self.embed_dim
        n_freq = max(int(math.ceil(d / 4.0)), 1)
        y = self.bev_origin[1] + (torch.arange(h, dtype=torch.float32) + 0.5) * self.bev_cell
        x = self.bev_origin[0] + (torch.arange(w, dtype=torch.float32) + 0.5) * self.bev_cell
        pe = []
        for k in range(n_freq):
            lam = 0.5 * (2.0 ** k)
            pe.append(torch.sin(x[None, :] / lam).expand(h, w))
            pe.append(torch.cos(x[None, :] / lam).expand(h, w))
            pe.append(torch.sin(y[:, None] / lam).expand(h, w))
            pe.append(torch.cos(y[:, None] / lam).expand(h, w))
        pe = torch.stack(pe, dim=0)[:d]
        return pe.unsqueeze(0)                     # (1, d, h, w)

    def _build_new_params(self, gamma_init=1.0, null_score_init=0.0):
        d, hd, dk = self.embed_dim, self.num_heads, self.head_dim

        self.proj_cam = nn.Conv2d(self.img_channels, d, 1)
        self.proj_rad = nn.Conv2d(self.rad_channels, d, 1)
        self.proj_rad_levels = nn.ModuleList(
            [nn.Conv2d(c, d, 1) for c in self.radar_level_channels])
        self.proj_img_levels = nn.ModuleList(
            [nn.Conv2d(self.image_channels, d, 1) for _ in range(self.image_num_levels)])

        self.norm_cam = nn.LayerNorm(d)
        self.norm_rad = nn.LayerNorm(d)

        # Bounded address generator: random weights, structured bias so that the
        # addresses start as centre / radial- / radial+ / tangential+ (design 6.1).
        self.offset_conv = nn.Sequential(
            nn.Conv2d(3 * d, d, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d, hd * (self.num_offsets - 1) * 2, 1))
        with torch.no_grad():
            bias = self.offset_conv[-1].bias.view(hd, self.num_offsets - 1, 2)
            bias.zero_()
            bias[:, 0, 0] = -_ALPHA_INIT
            if self.num_offsets > 2:
                bias[:, 1, 0] = _ALPHA_INIT
            if self.num_offsets > 3:
                bias[:, 2, 1] = _ALPHA_INIT
            if self.num_offsets > 4:
                bias[:, 4:, 0] = -_ALPHA_INIT

        self.key_rad = nn.ModuleList([nn.Linear(dk, dk) for _ in range(hd)])
        self.key_img = nn.ModuleList([nn.Linear(dk, dk) for _ in range(hd)])
        self.level_rad_bias = nn.Parameter(torch.zeros(hd, self.num_rad_levels))
        self.level_img_bias = nn.Parameter(torch.zeros(hd, self.image_num_levels))

        # Non-negative learned matching coefficient (softplus(raw)); not a gate.
        self.gamma_raw = nn.Parameter(
            torch.tensor(math.log(math.expm1(max(gamma_init, 1e-6))), dtype=torch.float32))
        self.null_rad = nn.Parameter(torch.full((hd,), float(null_score_init)))
        self.null_img = nn.Parameter(torch.full((hd,), float(null_score_init)))

        self.geom_score = nn.Sequential(
            nn.Linear(6, 32), nn.ReLU(inplace=True), nn.Linear(32, hd))
        self.depth_score = nn.Sequential(
            nn.Linear(4, 16), nn.ReLU(inplace=True), nn.Linear(16, 1))

        # Zero-initialised residual projections: step 0 == baseline.
        self.proj_r2c = nn.Conv2d(d, self.out_channels, 1)
        self.proj_i2r = nn.Conv2d(d, self.rad_channels, 1)
        for layer in (self.proj_r2c, self.proj_i2r):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    # ------------------------------------------------------------------
    # sampling helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _grid_from_meter(xy, origin, step, size_hw):
        """Metric address -> normalised grid (align_corners=False) + in-bounds.

        ``xy`` is ``(..., 2)`` metres in the augmented frame, ``origin`` the
        physical centre of pixel 0 of the source map and ``step`` its stride.
        """
        h, w = int(size_hw[0]), int(size_hw[1])
        xi = (xy[..., 0] - origin[0]) / step
        yi = (xy[..., 1] - origin[1]) / step
        grid_x = 2.0 * (xi + 0.5) / w - 1.0
        grid_y = 2.0 * (yi + 0.5) / h - 1.0
        # cell centres of the outermost ring must not fall out on float noise
        eps = 1e-4
        inside = ((xi >= -0.5 - eps) & (xi <= w - 0.5 + eps) &
                  (yi >= -0.5 - eps) & (yi <= h - 0.5 + eps))
        return torch.stack([grid_x, grid_y], dim=-1), inside

    # ------------------------------------------------------------------
    # shared geometry helpers (the dense chunk path and the active-cell
    # selection must see *bit-identical* masks, so both call these)
    # ------------------------------------------------------------------
    def _addresses(self, f_cat_c, cell_xy, e_r, e_t):
        """Bounded metric addresses ``a_hk(p)`` for one chunk of cells.

        ``f_cat_c`` is ``(B, 3*D, qc)``; ``cell_xy``/``e_r``/``e_t`` are the
        already sliced (or gathered) cell geometry: ``(1|B, qc, 2)``.  Returns
        ``addr`` and ``disp`` (both ``(B,Hd,qc,K,2)``).

        The ``cell_xy`` row axis is *kept*: a dense slice carries one row (the
        canvas grid is shared by the batch and broadcasts), while the gathered
        active-cell subset carries B rows whose *columns* hold different cells
        per sample -- taking row 0 there would silently address sample 0's
        cells for every sample (proved by the degenerate-image A9 fixture).
        """
        b = f_cat_c.shape[0]
        qc = f_cat_c.shape[-1]
        hd, k = self.num_heads, self.num_offsets
        dtype = f_cat_c.dtype
        off = self.offset_conv(f_cat_c.unsqueeze(-1)).squeeze(-1)
        off = off.view(b, hd, k - 1, 2, qc).permute(0, 1, 4, 2, 3)
        bounds = self.offset_bounds.view(1, hd, 1, 1, 2).to(dtype)
        tanh_off = torch.tanh(off) * bounds                             # (B,Hd,qc,K-1,2)
        zeros = torch.zeros_like(tanh_off[..., :1, :])
        tanh_off = torch.cat([zeros, tanh_off], dim=3)                  # (B,Hd,qc,K,2)

        # (1|B, qc, 2) -> (1|B, 1, qc, 1, 2); broadcast over (B, Hd, qc, K, 2)
        rows = int(cell_xy.shape[0])
        assert rows in (1, b) and int(cell_xy.shape[1]) == qc, (
            f'cell geometry {tuple(cell_xy.shape)} does not match chunk (B={b}, qc={qc})')
        assert int(e_r.shape[0]) in (1, b) and int(e_t.shape[0]) in (1, b), (
            'radial/tangential basis must be single-row or per-sample')
        p_e = cell_xy.unsqueeze(1).unsqueeze(3)
        e_r = e_r.unsqueeze(1).unsqueeze(3)
        e_t = e_t.unsqueeze(1).unsqueeze(3)
        addr = (p_e + tanh_off[..., 0:1] * e_r + tanh_off[..., 1:2] * e_t)
        disp = addr - p_e                                               # (B,Hd,qc,K,2)
        return addr, disp

    def _project_candidates(self, ctx, addr):
        """Camera-frame projection of every candidate address in a chunk.

        ``addr`` ``(B,Hd,qc,K,2)`` -> ``cam_mask``/``grid_uv`` in the
        ``(B*Hd*N, qc*K*n_h(,2))`` layout the sampling uses, plus the raw
        camera-frame ``depth`` (B,Hd,N,M).  Both the dense chunk path and the
        active-cell selection call this exact function so the validity masks
        can never drift apart.
        """
        b = ctx['b']
        hd, dk = self.num_heads, self.head_dim
        qc, k = int(addr.shape[2]), int(addr.shape[3])
        n_h, n_views = self.num_heights, ctx['num_views']
        device, dtype = ctx['device'], ctx['dtype']
        h_img, w_img = self.image_size
        mc = qc * k * n_h

        z_refs = self.z_refs.to(device=device, dtype=dtype).view(1, 1, 1, 1, n_h)
        z_aug = (ctx['z_scale'].view(b, 1, 1, 1, 1) * z_refs
                 + ctx['z_shift'].view(b, 1, 1, 1, 1))
        pts = torch.cat([
            addr.unsqueeze(4).expand(b, hd, qc, k, n_h, 2),
            z_aug.view(b, 1, 1, 1, n_h, 1).expand(b, hd, qc, k, n_h, 1),
        ], dim=-1).reshape(b, hd, mc, 3)
        pts4 = torch.cat([pts, torch.ones(b, hd, mc, 1, device=device, dtype=dtype)], dim=-1)

        proj = torch.einsum('bnij,bhmj->bhnmi', ctx['lidar2img'], pts4)  # (B,Hd,N,M,4)
        # KITTI-style 4x4 P: row 3 is [0,0,0,1]; the camera-frame depth is the
        # *third* homogeneous component (R4Det.py:1072 does the same).
        depth = proj[..., 2]
        depth_safe = depth.clamp_min(1e-3)
        u = proj[..., 0] / depth_safe
        v = proj[..., 1] / depth_safe
        valid = ((depth > self.min_camera_depth) &
                 (u >= 0) & (u <= w_img - 1) & (v >= 0) & (v <= h_img - 1))

        # Real (un-padded) image support: invert the pixel-space augmentation and
        # require the point to fall inside the *resized* image rectangle.
        rel = torch.stack([u, v], dim=-1) - ctx['post_trans'].view(b, 1, n_views, 1, 2)
        p_res = torch.einsum('bnij,bhnmj->bhnmi', ctx['inv_post_rot'], rel)
        new_w = ctx['new_hw'][..., 1].view(b, 1, n_views, 1)
        new_h = ctx['new_hw'][..., 0].view(b, 1, n_views, 1)
        valid = (valid & (p_res[..., 0] >= 0) & (p_res[..., 0] <= new_w - 1) &
                 (p_res[..., 1] >= 0) & (p_res[..., 1] <= new_h - 1))

        cam_mask = valid.reshape(b * hd * n_views, mc)
        grid_uv = torch.stack([u, v], dim=-1).reshape(b * hd * n_views, mc, 2)
        return cam_mask, grid_uv, depth

    def _radar_candidate_geometry(self, ctx, addr_flat):
        """Echo support and in-canvas masks of every candidate address.

        ``addr_flat`` ``(B*Hd, qc, K, 2)`` -> ``m_hk`` ``(B*Hd, qc, K)`` bool
        and, per native level, the normalised grid and in-bounds mask in the
        flattened ``(B*Hd, qc*K(,2))`` layout that ``_sample_map`` consumes.
        """
        m_hk = self._sample_head(ctx['support_hd'], addr_flat, self.radar_origin,
                                 self.radar_base_cell)[0][..., 0] > 0.5
        rows = addr_flat.shape[0]
        grids, insides = [], []
        for lev in range(self.num_rad_levels):
            shape = ctx['rad_src'][lev].shape[-2:]
            grid, inside = self._grid_from_meter(addr_flat, self.radar_origin,
                                                 self.radar_level_steps[lev], shape)
            grids.append(grid.reshape(rows, -1, 2))
            insides.append(inside.reshape(rows, -1))
        return m_hk, grids, insides

    def _active_cells(self, ctx, f_cat_f, cell_xy, e_r, e_t):
        """Cells whose two messages cannot both be provably zero.

        A cell is *inactive* only when no candidate address of any head has
        image validity (``ok_h``) and none has radar echo support
        (``radar_valid``).  For such a cell the module provably returns
        ``M_R = M_I = 0`` and ``valid_r = valid_i = False`` (every candidate is
        excluded from both candidate softmaxes, so each softmax puts weight 1 on
        its null entry) -- the residual projections are then multiplied by zero.
        Skipping the cell therefore cannot change the fusion output.

        Runs under ``no_grad`` on purpose: masks are comparisons (zero
        subgradient a.e.) and the differentiable geometry is recomputed inside
        each processed chunk, so the sampling-grid gradient is untouched.
        """
        b = ctx['b']
        q = f_cat_f.shape[-1]
        hd = self.num_heads
        n_views = ctx['num_views']
        with torch.no_grad():
            addr, _ = self._addresses(f_cat_f, cell_xy, e_r, e_t)
            cam_mask, _, _ = self._project_candidates(ctx, addr)
            mc = cam_mask.shape[1]
            image_ok = cam_mask.view(b, hd, n_views, mc).any(dim=2)     # (B,Hd,M)
            image_ok = image_ok.view(b, hd, q, -1).any(dim=3)
            addr_flat = addr.reshape(b * hd, q, self.num_offsets, 2)
            m_hk, _, rad_inside = self._radar_candidate_geometry(ctx, addr_flat)
            inside_any = torch.stack(rad_inside, dim=-1)
            inside_any = inside_any.view(b * hd, q, self.num_offsets, -1).any(dim=-1)
            radar_sup = (m_hk & inside_any).view(b, hd, q, self.num_offsets).any(dim=3)
            return (image_ok | radar_sup).any(dim=1)                    # (B,q) bool

    @staticmethod
    def _sample_map(feat, grid, mask):
        """Sample ``feat`` (N,C,h,w) at ``grid`` (N,Q,2); zero where masked."""
        n, _, _, _ = feat.shape
        q = grid.shape[1]
        safe = torch.nan_to_num(grid, nan=0.0, posinf=1.0, neginf=-1.0).clamp_(-1.0, 1.0)
        out = F.grid_sample(feat, safe.view(n, 1, q, 2), mode='bilinear',
                            padding_mode='border', align_corners=False)
        out = out.squeeze(2).transpose(1, 2)                 # (N, Q, C)
        return out * mask.unsqueeze(-1).to(out.dtype)

    def _sample_head(self, feat, addr, origin, step):
        """feat (B,C,h,w), addr (B,qc,K,2) -> (B,qc,K,C) and inside (B,qc,K)."""
        b = feat.shape[0]
        qc, k = int(addr.shape[1]), int(addr.shape[2])
        grid, inside = self._grid_from_meter(addr, origin, step, feat.shape[-2:])
        val = self._sample_map(feat, grid.reshape(b, qc * k, 2), inside.reshape(b, qc * k))
        return val.view(b, qc, k, -1), inside

    def _radar_support(self, points, device, dtype):
        """Local input-echo support on the 0.16 m scatter grid (5x5 window).

        GT-free: derived only from the point cloud the radar branch itself
        consumes.  A zero-support address carries no new *direct* radar
        evidence; it never disables the image path.
        """
        hb, wb = self.radar_base_shape
        cell = self.radar_base_cell
        min_x = self.radar_origin[0] - 0.5 * cell
        min_y = self.radar_origin[1] - 0.5 * cell
        occ = torch.zeros((len(points), 1, hb, wb), device=device, dtype=dtype)
        for b, pts in enumerate(points):
            if pts is None or pts.numel() == 0:
                continue
            xy = pts[:, :2].to(device=device, dtype=dtype)
            ix = torch.floor((xy[:, 0] - min_x) / cell).long()
            iy = torch.floor((xy[:, 1] - min_y) / cell).long()
            keep = (ix >= 0) & (ix < wb) & (iy >= 0) & (iy < hb)
            if bool(keep.any()):
                occ[b, 0, iy[keep], ix[keep]] = 1.0
        return F.max_pool2d(occ, self.support_kernel, stride=1,
                            padding=self.support_kernel // 2)

    # ------------------------------------------------------------------
    # per-forward geometry context
    # ------------------------------------------------------------------
    def _build_context(self, geo, b, device, dtype):
        h, w = self.bev_shape
        num_views = int(geo['num_views'])
        hd, dk = self.num_heads, self.head_dim

        iy = torch.arange(h, device=device, dtype=dtype)
        ix = torch.arange(w, device=device, dtype=dtype)
        yy = self.bev_origin[1] + (iy + 0.5) * self.bev_cell
        xx = self.bev_origin[0] + (ix + 0.5) * self.bev_cell
        cell_y, cell_x = torch.meshgrid(yy, xx)
        cell_xy = torch.stack([cell_x, cell_y], dim=-1).view(1, h * w, 2)

        # Augmented sensor origin (BDA translation): the radial/tangential basis
        # must be built about *this* point, not about (0, 0).
        bda = geo['bda'].to(device=device, dtype=dtype)
        origin = bda.new_zeros((b, 4))
        origin[:, 3] = 1.0
        origin_aug = torch.einsum('bij,bj->bi', bda, origin)[:, :2]
        radial = cell_xy - origin_aug.view(b, 1, 2)
        e_r = radial / radial.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        e_t = torch.stack([-e_r[..., 1], e_r[..., 0]], dim=-1)

        radar_levels = geo['radar_levels']
        assert len(radar_levels) == self.num_rad_levels, (
            f'expected {self.num_rad_levels} native radar levels, got {len(radar_levels)}')
        rad_src = []
        for lev, feat in enumerate(radar_levels):
            feat = feat.to(device=device, dtype=dtype)
            expect = (self.radar_base_shape[0] >> (lev + 1),
                      self.radar_base_shape[1] >> (lev + 1))
            assert tuple(feat.shape[-2:]) == expect, (
                f'native radar level {lev} has shape {tuple(feat.shape[-2:])}, expected '
                f'{expect} for stride {self.radar_level_steps[lev]} m')
            assert feat.shape[0] == b
            rad_src.append(self.proj_rad_levels[lev](feat))

        img_feats = geo['img_feats']
        assert len(img_feats) >= self.image_num_levels, (
            f'need {self.image_num_levels} image levels, got {len(img_feats)}')
        img_src = []
        for s in range(self.image_num_levels):
            feat = img_feats[s].to(device=device, dtype=dtype)
            stride = self.image_level_strides[s]
            assert feat.shape[0] == b * num_views, (
                f'image level {s} batch {feat.shape[0]} != B*N {b * num_views}')
            assert (feat.shape[-2] * stride, feat.shape[-1] * stride) == self.image_size, (
                f'image level {s} has shape {tuple(feat.shape[-2:])}, inconsistent with '
                f'stride {stride} and canvas {self.image_size}')
            proj = self.proj_img_levels[s](feat)                # (B*N, d, h, w)
            # head-major batch (B, Hd, N): the per-head loop is gone, so the
            # sampling grid and the features must share one head order
            img_src.append(proj.view(b, num_views, hd, dk, *proj.shape[-2:])
                               .permute(0, 2, 1, 3, 4, 5)
                               .reshape(b * hd * num_views, dk, *proj.shape[-2:]))

        post_rots = geo['post_rots'].to(device=device, dtype=dtype)
        post_trans = geo['post_trans'].to(device=device, dtype=dtype)
        resize = torch.det(post_rots).abs().sqrt()                    # (B,N)
        ori_hw = torch.tensor([[float(o[0]), float(o[1])] for o in geo['ori_shape']],
                              device=device, dtype=dtype)
        new_hw = (ori_hw.view(b, 1, 2) * resize.unsqueeze(-1)).floor()  # (B,N)=(H,W)

        depth_stats = None
        depth = geo.get('depth', None)
        if depth is not None:
            depth = depth.to(device=device, dtype=dtype)
            assert depth.shape[0] == b * num_views
            assert depth.shape[-2] * self.depth_downsample == self.image_size[0]
            assert depth.shape[-1] * self.depth_downsample == self.image_size[1]
            prob = depth.clamp_min(0)
            bins = torch.arange(prob.shape[1], device=device, dtype=dtype).view(1, -1, 1, 1) + 1.0
            exp_depth = (bins * prob).sum(dim=1, keepdim=True)
            ent = -(prob * (prob + 1e-8).log()).sum(dim=1, keepdim=True)
            depth_stats = torch.cat([exp_depth, ent], dim=1)           # (B*N,2,hd,wd)

        support = self._radar_support(geo['points'], device, dtype)    # (B,1,HB,WB)
        # one echo map per head-batch slot, built once per forward instead of
        # once per chunk (the gate itself is still evaluated per address)
        support_hd = support.repeat_interleave(hd, dim=0)              # (B*Hd,1,HB,WB)
        grid, inside = self._grid_from_meter(cell_xy.expand(b, -1, -1),
                                             self.radar_origin, self.radar_base_cell,
                                             self.radar_base_shape)
        m_parent = self._sample_map(support, grid, inside)             # (B,Q,1)

        return dict(
            b=b, h=h, w=w, device=device, dtype=dtype, num_views=num_views,
            cell_xy=cell_xy, e_r=e_r, e_t=e_t, support=support,
            support_hd=support_hd, m_parent=m_parent,
            rad_src=rad_src, img_src=img_src, depth_stats=depth_stats,
            lidar2img=geo['lidar2img'].to(device=device, dtype=dtype),
            inv_post_rot=torch.inverse(post_rots), post_trans=post_trans,
            new_hw=new_hw,
            z_scale=bda[:, 2, 2].view(b, 1), z_shift=bda[:, 2, 3].view(b, 1))

    # ------------------------------------------------------------------
    # camera side
    # ------------------------------------------------------------------
    def _image_terms(self, ctx, addr, q_ift_h):
        """Camera evidence for one chunk -- every head at once.

        ``addr`` is ``(B,Hd,qc,K,2)`` (each head has its own bounded offsets)
        and ``q_ift_h`` is ``(B,Hd,qc,dk)``.  Returns ``i`` (B,Hd,qc,K,nH,dk),
        validity (B,Hd,qc,K,nH), level weights (B,Hd,qc,K,nH,L) and the
        predicted-depth descriptor score (B,Hd,qc,K,nH).  The head axis is
        explicit so the projection, the level sampling and the descriptor are
        one broadcast op each instead of ``Hd`` Python-loop copies (training
        record section 101); the geometry and the numbers are unchanged.
        """
        b = ctx['b']
        hd, dk = self.num_heads, self.head_dim
        qc, k = int(addr.shape[2]), int(addr.shape[3])
        n_h, n_views = self.num_heights, ctx['num_views']
        device, dtype = ctx['device'], ctx['dtype']
        h_img, w_img = self.image_size
        mc = qc * k * n_h

        # geometry lives in one shared helper: the dense chunk path and the
        # active-cell selection call the *same* code, so their masks cannot drift
        cam_mask, grid_uv, depth = self._project_candidates(ctx, addr)
        count = cam_mask.view(b, hd, n_views, mc).sum(dim=2).to(dtype)    # (B,Hd,M)
        denom = count.clamp_min(1.0).unsqueeze(-1)                      # (B,Hd,M,1)
        ok = count > 0                                                  # (B,Hd,M)

        lv_val, lv_score = [], []
        for s in range(self.image_num_levels):
            feat = ctx['img_src'][s]                     # (B*Hd*N,dk,h,w)
            stride = self.image_level_strides[s]
            # u normalises by the feature width, v by the *height* (480x640
            # levels are not square, so a single divisor would skew every read)
            grid = torch.stack([
                2.0 * grid_uv[..., 0] / float(stride * feat.shape[-1]) - 1.0,
                2.0 * grid_uv[..., 1] / float(stride * feat.shape[-2]) - 1.0,
            ], dim=-1)
            sampled = self._sample_map(feat, grid, cam_mask).view(b, hd, n_views, mc, dk)
            agg = (sampled * cam_mask.view(b, hd, n_views, mc, 1).to(dtype)).sum(dim=2) / denom
            keyed = _head_linear(self.key_img, agg).view(b, hd, qc, k, n_h, dk)
            score = (keyed * q_ift_h.unsqueeze(3).unsqueeze(4)).sum(-1) / math.sqrt(dk)
            lv_val.append(agg)
            lv_score.append(score.reshape(b, hd, mc) +
                            self.level_img_bias.view(1, hd, 1, -1)[..., s])

        lv_score = torch.stack(lv_score, dim=-1)                         # (B,Hd,M,L)
        lv_mask = ok.unsqueeze(-1).expand(-1, -1, -1, self.image_num_levels)
        beta = _masked_softmax(lv_score, lv_mask, dim=-1)
        i_h = sum(beta[..., s:s + 1] * lv_val[s] for s in range(self.image_num_levels))
        i_h = i_h.view(b, hd, qc, k, n_h, dk)
        beta = beta.view(b, hd, qc, k, n_h, self.image_num_levels)
        ok_h = ok.view(b, hd, qc, k, n_h)

        # predicted-depth descriptor: projected depth vs expected depth / entropy
        if ctx['depth_stats'] is not None:
            stats = ctx['depth_stats']                                  # (B*N,2,hd,wd)
            gu = 2.0 * grid_uv[..., 0] / (stats.shape[-1] * self.depth_downsample) - 1.0
            gv = 2.0 * grid_uv[..., 1] / (stats.shape[-2] * self.depth_downsample) - 1.0
            gd = torch.stack([gu, gv], dim=-1)
            # the same depth map serves every head: tile it into the (B,Hd,N)
            # order of ``cam_mask``/``grid_uv`` instead of reordering the grid
            stats_h = (stats.view(b, n_views, *stats.shape[1:])
                            .unsqueeze(1)
                            .expand(b, hd, n_views, *stats.shape[1:])
                            .reshape(b * hd * n_views, *stats.shape[1:]))
            exp_d = self._sample_map(stats_h[:, 0:1], gd, cam_mask).squeeze(-1)
            ent_d = self._sample_map(stats_h[:, 1:2], gd, cam_mask).squeeze(-1)
            d_feat = torch.stack([
                depth.reshape(-1, mc) / 50.0,
                exp_d / 50.0,
                (depth.reshape(-1, mc) - exp_d).abs() / 10.0,
                ent_d / 5.0,
            ], dim=-1)                                                  # (B*Hd*N,M,4)
            d_score = self.depth_score(d_feat).squeeze(-1)              # (B*Hd*N,M)
            d_score = d_score.view(b, hd, n_views, mc)
            mask_f = cam_mask.view(b, hd, n_views, mc).to(dtype)
            f_depth = (d_score * mask_f).sum(dim=2) / mask_f.sum(dim=2).clamp_min(1.0)
        else:
            # no predicted depth in this call: descriptor contributes exactly 0
            # but stays in the graph so every parameter is used under DDP
            zero = self.depth_score(torch.zeros((1, 1, 4), device=device, dtype=dtype)).sum()
            f_depth = torch.zeros((b, hd, mc), device=device, dtype=dtype) + 0.0 * zero

        return i_h, ok_h, beta, f_depth.view(b, hd, qc, k, n_h)

    # ------------------------------------------------------------------
    # one chunk of cells
    # ------------------------------------------------------------------
    def _chunk(self, q_cam_c, q_rad_c, f_cat_c, cell_xy_c, e_r_c, e_t_c, m_par_c, ctx):
        """Chunk of cells -> (M_R, M_I), each (B, D, qc).

        The head axis is explicit everywhere (``Hd`` sits right after the
        batch) so a chunk costs a constant number of launches instead of
        ``chunk * head * level`` copies of them.  Addresses, offsets, level
        mixing, null candidates and validity gates are unchanged.

        The cell geometry (``cell_xy_c``/``e_r_c``/``e_t_c``/``m_par_c``) is
        passed in already sliced -- or gathered, when the caller runs only the
        active-cell subset (``active_cells_only``).  The cell axis carries no
        identity of its own, so both routes are the same computation.
        """
        b = q_cam_c.shape[0]
        qc = q_cam_c.shape[-1]
        hd, dk, k, n_h = self.num_heads, self.head_dim, self.num_offsets, self.num_heights
        dtype = ctx['dtype']

        # ---- bounded metric addresses -----------------------------------
        addr, disp = self._addresses(f_cat_c, cell_xy_c, e_r_c, e_t_c)

        e_r = e_r_c.unsqueeze(1).unsqueeze(3)
        e_t = e_t_c.unsqueeze(1).unsqueeze(3)
        geo_feat = torch.cat([
            (disp[..., 0:1]) / 1.28, (disp[..., 1:2]) / 1.28,
            disp.norm(dim=-1, keepdim=True) / 1.28,
            (disp * e_r).sum(-1, keepdim=True) / 1.28,
            (disp * e_t).sum(-1, keepdim=True) / 1.28,
            addr.norm(dim=-1, keepdim=True) / 40.0,
        ], dim=-1)                                                      # (B,Hd,qc,K,6)
        b_geo = self.geom_score(geo_feat)                # (B,Hd,qc,K,Hd)
        # head h's own bias is the one head h's candidates use
        b_geo = torch.diagonal(b_geo, dim1=1, dim2=4).permute(0, 3, 1, 2)  # (B,Hd,qc,K)

        q_cam_h = q_cam_c.view(b, hd, dk, qc).permute(0, 1, 3, 2)       # (B,Hd,qc,dk)
        q_rad_h = q_rad_c.view(b, hd, dk, qc).permute(0, 1, 3, 2)
        m_par = m_par_c.view(b, 1, qc, 1)
        q_ift_h = q_cam_h + m_par * q_rad_h                             # (B,Hd,qc,dk)

        gamma = F.softplus(self.gamma_raw)
        addr_flat = addr.reshape(b * hd, qc, k, 2)
        m_hk_flat, rad_grids, rad_insides = self._radar_candidate_geometry(ctx, addr_flat)
        m_hk = m_hk_flat.view(b, hd, qc, k)

        # ---- radar: native levels ---------------------------------------
        rad_val, rad_mask, rad_score = [], [], []
        for lev in range(self.num_rad_levels):
            src = ctx['rad_src'][lev]
            feat = src.view(b * hd, dk, *src.shape[-2:])
            val_l = self._sample_map(feat, rad_grids[lev], rad_insides[lev])
            val_l = val_l.view(b, hd, qc, k, dk)
            keyed = _head_linear(self.key_rad, val_l)
            sc = (keyed * q_cam_h.unsqueeze(3)).sum(-1) / math.sqrt(dk)
            rad_val.append(val_l)
            rad_mask.append(rad_insides[lev].view(b, hd, qc, k))
            rad_score.append(sc + self.level_rad_bias.view(1, hd, 1, 1, -1)[..., lev])
        rad_score = torch.stack(rad_score, dim=-1)                      # (B,Hd,qc,K,L)
        rad_mask = torch.stack(rad_mask, dim=-1)
        beta_rad = _masked_softmax(rad_score, rad_mask, dim=-1)
        r_h = sum(beta_rad[..., lev:lev + 1] * rad_val[lev]
                  for lev in range(self.num_rad_levels))

        # ---- camera ------------------------------------------------------
        i_h, ok_h, beta_img, f_depth = self._image_terms(ctx, addr, q_ift_h)

        # ---- paired compatibility ----------------------------------------
        compat = (r_h.unsqueeze(4) * i_h).sum(-1) / math.sqrt(dk)       # (B,Hd,qc,K,nH)
        cm = compat.masked_fill(~ok_h, -1e4)
        n_ok = ok_h.to(dtype).sum(dim=-1)                               # (B,Hd,qc,K)
        c_mean = torch.where(
            n_ok > 0,
            torch.logsumexp(cm, dim=-1) - n_ok.clamp_min(1.0).log(),
            torch.zeros_like(n_ok))
        keyed_r = _head_linear(self.key_rad, r_h)
        s_rad = (keyed_r * q_cam_h.unsqueeze(3)).sum(-1) / math.sqrt(dk)
        s_rad = s_rad + gamma * c_mean + b_geo
        radar_valid = m_hk & rad_mask.any(dim=-1)
        sup = radar_valid.to(dtype)
        s_rad = torch.cat([s_rad, self.null_rad.view(1, hd, 1, 1).expand(b, hd, qc, 1)],
                          dim=-1)
        mask_rad = torch.cat([sup, torch.ones((b, hd, qc, 1), device=sup.device,
                                              dtype=torch.bool)], dim=-1)
        w_rad = _masked_softmax(s_rad, mask_rad, dim=-1)
        m_r = (w_rad[..., :k].unsqueeze(-1) * r_h).sum(dim=3)           # (B,Hd,qc,dk)

        keyed_i = _head_linear(self.key_img, i_h)
        s_img = (keyed_i * q_ift_h.unsqueeze(3).unsqueeze(4)).sum(-1) / math.sqrt(dk)
        s_img = s_img + m_hk.unsqueeze(-1).to(dtype) * gamma * compat \
                      + f_depth + b_geo.unsqueeze(-1)
        s_img = s_img.reshape(b, hd, qc, k * n_h)
        s_img = torch.cat([s_img, self.null_img.view(1, hd, 1, 1).expand(b, hd, qc, 1)],
                          dim=-1)
        mask_img = torch.cat([ok_h.reshape(b, hd, qc, k * n_h),
                              torch.ones((b, hd, qc, 1), device=ok_h.device,
                                         dtype=torch.bool)], dim=-1)
        w_img = _masked_softmax(s_img, mask_img, dim=-1)
        m_i = (w_img[..., :k * n_h].view(b, hd, qc, k, n_h, 1) * i_h).sum(dim=(3, 4))

        stats = {'sup': 0.0, 'n_rad': 0, 'n_img': 0, 'null_rad': 0.0, 'null_img': 0.0,
                 'lv_rad': [0.0] * self.num_rad_levels,
                 'lv_img': [0.0] * self.image_num_levels}
        if self.collect_stats:
            count = b * qc * hd          # every head's cells are counted once
            stats['sup'] += float(sup.mean()) * count
            stats['n_rad'] += count
            stats['n_img'] += count
            stats['null_rad'] += float(w_rad[..., -1].mean()) * count
            stats['null_img'] += float(w_img[..., -1].mean()) * count
            for lev in range(self.num_rad_levels):
                stats['lv_rad'][lev] += float(beta_rad[..., lev].mean()) * count
            for s in range(self.image_num_levels):
                stats['lv_img'][s] += float(beta_img[..., s].mean()) * count

        # each head's message is (B, qc, dk): swap back to (B, dk, qc) *before*
        # merging heads, otherwise the query and channel axes are interleaved
        # by the reshape and every message lands at the wrong BEV cell
        # (chunk-size dependent; proven by the chunked-vs-single-chunk
        # counterexample in gpmsf_cpu_checks t11/t12).
        m_r = m_r.permute(0, 1, 3, 2).reshape(b, self.embed_dim, qc)
        m_i = m_i.permute(0, 1, 3, 2).reshape(b, self.embed_dim, qc)
        # Gate the complete projection, including learned bias.
        # Empty evidence stays an exact identity after training too.
        return (m_r, m_i, (stats if self.collect_stats else None),
                radar_valid.any(dim=3).any(dim=1).unsqueeze(1),
                ok_h.flatten(3).any(dim=3).any(dim=1).unsqueeze(1))

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, img_bev, radar_bev, geo=None):
        """Fuse camera/radar BEV features.

        Args:
            img_bev (Tensor): camera LSS BEV ``(B, 256, 248, 216)`` (y, x).
            radar_bev (Tensor): radar SECONDFPN BEV ``(B, 384, 248, 216)``.
            geo (dict, optional): current-frame geometry sources.  ``None``
                reproduces the plain ConcatConv path exactly only when
                ``require_geo=False`` (in train and eval). Keys:
                ``img_feats`` (>=3 native image levels, batch B*N),
                ``num_views``, ``radar_levels`` (3 native SECOND levels),
                ``depth`` (predicted depth distribution, B*N x D x h x w or
                None), ``lidar2img`` (B,N,4,4), ``post_rots`` (B,N,2,2),
                ``post_trans`` (B,N,2), ``ori_shape`` (list of (H,W)),
                ``bda`` (B,4,4 lidar augmentation), ``points`` (list of input
                xyz, augmented frame).

        Returns:
            Tensor: fused BEV ``(B, out_channels, 248, 216)``.
        """
        # reset *before* any early return, so a probe can never read the previous
        # forward's statistics -- the geo=None fallback included
        self.last_stats = None
        if geo is None:
            if self.require_geo:
                raise RuntimeError(
                    'GeometryPairedMultiScaleFusion got geo=None: the '
                    'detector wiring did not deliver the geometry sources, so this arm '
                    'would silently degenerate into ConcatConvFusion. Pass `geo` or set '
                    'require_geo=False when the plain fallback is intended.')
            return self.fusion_conv(torch.cat([img_bev, radar_bev], dim=1))

        b, _, h, w = img_bev.shape
        assert (h, w) == self.bev_shape, (
            f'fusion input is {(h, w)}, configured bev_shape is {self.bev_shape}')
        device, dtype = img_bev.device, img_bev.dtype

        ctx = self._build_context(geo, b, device, dtype)

        pos = self._pos_enc.to(device=device, dtype=dtype)
        p_cam = self.proj_cam(img_bev) + pos
        p_rad = self.proj_rad(radar_bev) + pos
        # LayerNorm over the channel dim of a (B, D, H, W) map == channels-last
        q_cam_f = self.norm_cam(p_cam.permute(0, 2, 3, 1).reshape(b, h * w, self.embed_dim))
        q_rad_f = self.norm_rad(p_rad.permute(0, 2, 3, 1).reshape(b, h * w, self.embed_dim))
        q = h * w
        q_cam_f = q_cam_f.permute(0, 2, 1).contiguous()
        q_rad_f = q_rad_f.permute(0, 2, 1).contiguous()
        # the address generator sees C, R *and* the metric position (design 6.1)
        f_cat_f = torch.cat([p_cam, p_rad, pos.expand(b, -1, h, w)], dim=1)
        f_cat_f = f_cat_f.reshape(b, 3 * self.embed_dim, q)

        chunk = self.query_chunk if self.query_chunk > 0 else q
        m_r_parts, m_i_parts = [], []
        valid_r_parts, valid_i_parts = [], []
        agg = {'sup': 0.0, 'n_rad': 0, 'n_img': 0, 'null_rad': 0.0, 'null_img': 0.0,
               'lv_rad': [0.0] * self.num_rad_levels,
               'lv_img': [0.0] * self.image_num_levels}
        use_ckpt = self.grad_checkpoint and self.training and torch.is_grad_enabled()

        # ---- active-cell subset (training record section 104.9) ----------
        # Only cells whose two messages can be non-zero are computed.  The
        # permutation is active-first and the canvases are zero-initialised, so
        # every dropped cell keeps exactly the values the dense path produces
        # for it: M_R = M_I = 0 and valid_r = valid_i = False (each candidate
        # softmax puts weight 1 on its null entry), hence a zero residual.
        order, n_cells = None, q
        if self.active_cells_only:
            active = self._active_cells(ctx, f_cat_f, ctx['cell_xy'],
                                        ctx['e_r'], ctx['e_t'])
            n_max = int(active.sum(dim=1).max().item()) if b else 0
            if n_max > 0:
                order = torch.argsort(active.to(torch.uint8), dim=1,
                                      descending=True, stable=True)
                n_cells = n_max
        compact = order is not None
        if compact:
            m_r = torch.zeros((b, self.embed_dim, q), device=device, dtype=dtype)
            m_i = torch.zeros_like(m_r)
            valid_r = torch.zeros((b, 1, q), device=device, dtype=torch.bool)
            valid_i = torch.zeros_like(valid_r)

        for start in range(0, n_cells, chunk):
            end = min(start + chunk, n_cells)
            if compact:
                idx = order[:, start:end]
                idx_c = idx.unsqueeze(1)
                q_cam_c = q_cam_f.gather(2, idx_c.expand(-1, self.embed_dim, -1))
                q_rad_c = q_rad_f.gather(2, idx_c.expand(-1, self.embed_dim, -1))
                f_cat_c = f_cat_f.gather(2, idx_c.expand(-1, 3 * self.embed_dim, -1))
                idx_xy = idx.unsqueeze(-1).expand(-1, -1, 2)
                cell_xy_c = ctx['cell_xy'].expand(b, -1, -1).gather(1, idx_xy)
                e_r_c = ctx['e_r'].gather(1, idx_xy)
                e_t_c = ctx['e_t'].gather(1, idx_xy)
                m_par_c = ctx['m_parent'][:, :, 0].gather(1, idx).unsqueeze(-1)
            else:
                q_cam_c = q_cam_f[..., start:end]
                q_rad_c = q_rad_f[..., start:end]
                f_cat_c = f_cat_f[..., start:end]
                cell_xy_c = ctx['cell_xy'][:, start:end]
                e_r_c = ctx['e_r'][:, start:end]
                e_t_c = ctx['e_t'][:, start:end]
                m_par_c = ctx['m_parent'][:, start:end]
            args = (q_cam_c, q_rad_c, f_cat_c, cell_xy_c, e_r_c, e_t_c, m_par_c)
            if use_ckpt:
                out = checkpoint(self._chunk, *args, ctx, use_reentrant=False)
            else:
                out = self._chunk(*args, ctx)
            if compact:
                m_r.scatter_(2, idx_c.expand(-1, self.embed_dim, -1), out[0])
                m_i.scatter_(2, idx_c.expand(-1, self.embed_dim, -1), out[1])
                valid_r.scatter_(2, idx_c, out[3])
                valid_i.scatter_(2, idx_c, out[4])
            else:
                m_r_parts.append(out[0])
                m_i_parts.append(out[1])
                valid_r_parts.append(out[3])
                valid_i_parts.append(out[4])
            if self.collect_stats:
                for key in ('sup', 'null_rad', 'null_img'):
                    agg[key] += out[2][key]
                agg['n_rad'] += out[2]['n_rad']
                agg['n_img'] += out[2]['n_img']
                for lev in range(self.num_rad_levels):
                    agg['lv_rad'][lev] += out[2]['lv_rad'][lev]
                for s in range(self.image_num_levels):
                    agg['lv_img'][s] += out[2]['lv_img'][s]

        if compact:
            # Cells at permuted positions >= every sample's active count are
            # inactive for every sample (no echo support, no image validity):
            # sup = 0, null weights = 1, level weights = 0.  Adding that
            # contribution analytically keeps the diagnostics means over the
            # *whole* canvas, exactly as in the dense path.
            extra = (q - n_cells) * self.num_heads
            if extra:
                agg['n_rad'] += extra * b
                agg['n_img'] += extra * b
                agg['null_rad'] += float(extra * b)
                agg['null_img'] += float(extra * b)
        else:
            m_r = torch.cat(m_r_parts, dim=-1)
            m_i = torch.cat(m_i_parts, dim=-1)
            valid_r = torch.cat(valid_r_parts, dim=-1)
            valid_i = torch.cat(valid_i_parts, dim=-1)
        m_r = m_r.view(b, self.embed_dim, h, w)
        m_i = m_i.view(b, self.embed_dim, h, w)
        valid_r = valid_r.view(b, 1, h, w)
        valid_i = valid_i.view(b, 1, h, w)
        c_new = img_bev + self.proj_r2c(m_r) * valid_r.to(dtype)
        r_new = radar_bev + self.proj_i2r(m_i) * valid_i.to(dtype)

        if self.collect_stats:
            nr = max(agg['n_rad'], 1)
            ni = max(agg['n_img'], 1)
            self.last_stats = {
                'radar_support_rate': agg['sup'] / nr,
                'native_radar_requires_grad': [x.requires_grad for x in geo['radar_levels']],
                'native_image_requires_grad': [x.requires_grad for x in geo['img_feats'][:3]],
                'native_radar_has_grad_fn': [x.grad_fn is not None for x in geo['radar_levels']],
                'native_image_has_grad_fn': [x.grad_fn is not None for x in geo['img_feats'][:3]],
                'checkpoint_active': bool(use_ckpt),
                'null_rad_weight': agg['null_rad'] / nr,
                'null_img_weight': agg['null_img'] / ni,
                'level_rad_weight': [v / nr for v in agg['lv_rad']],
                'level_img_weight': [v / ni for v in agg['lv_img']],
            }
        return self.fusion_conv(torch.cat([c_new, r_new], dim=1))
