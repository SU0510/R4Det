"""Candidate A: object-semantic supervision of ``z`` + detection cls residual.

Design + pre-registration: ``docs/rssm_candidate_A_prereg_20261010.md``.
This file only ADDS a class to the ``me_rssm`` research package; the baseline
``MotionAlignedRSSMFusion`` and every existing run are untouched (the two
detector/head hooks it needs are flag-gated and default to off).

Why this class exists
---------------------
Three measured facts (training_runs_full.md §49, §70) point at a broken
supervision->readout chain rather than at an uninformative reconstruction
target:

* the reconstruction term is only 0.18% of the total loss at ep24 (§70.2) while
  the stage-0.2 remeasurement showed the target itself is 56.5% sample specific
  (§70.6) -- the target is not a shared template, the *pressure* on ``z`` is
  simply absent;
* KL is fully absorbed by free-bits and the posterior collapses onto the prior
  (mu_diff^2 1.527 -> 0.0072, §70.2), i.e. ``z`` carries no observation;
* ``zero_z`` moves the head output by 38.02% while ``shuffle_z`` moves it by
  only 4.67% (§49) -- ``z`` acts as a sample-common bias, not as sample-level
  discriminative evidence; and the standard RSSM's ``shuffle_z`` effect is
  ALREADY foreground-concentrated (fg/bg_hi = 4.3x, §69), so the readout's
  spatial attention is not the missing piece either.

Candidate A changes exactly two things, both on the supervision/readout axis,
and nothing on the state/measurement axis (reconstruction target, KL, free
nats, gates, alignment, schedule all stay exactly as in ``no2d_igdr``):

1. ``sem_head`` -- a **z-exclusive** centre-heatmap head (3x3 conv -> ReLU ->
   1x1 conv on ``z_t``) trained with the CenterNet focal loss against the
   current frame's GT box centres. ``h_t`` is deliberately NOT an input: the
   only way to lower this loss is to make ``z`` itself carry per-cell class
   evidence (unit test T3 asserts zero gradient into ``h_t``).
2. ``z_cls_logits`` -- a **zero-initialised** 1x1 conv on ``z_t`` emitting
   ``sem_num_classes`` class logits, broadcast over the anchor slots of each
   class and added to ``conv_cls``'s output inside ``Anchor3DHead``::

       logits[a, c](cell) = conv_cls(x)[a, c](cell) + (W_cls z_t)(cell)|_c

   ``conv_reg`` / ``conv_dir_cls`` are untouched, so the class residual stays
   separate from the geometric residual.

Zero-init means the arm starts bit-identical to the baseline: the residual is
exactly 0 before the first optimiser step, and ``sem_head`` is not in the
detection forward path at all. Ablations are single flags:
``sem_supervision_weight=0`` (readout only), ``cls_residual=False``
(supervision only), ``sem_head_input='h'|'feat'`` (the same supervision applied
to the recurrent state or to the current observation instead of ``z`` -- the
control that decides whether a gain has to come from semantics in the
stochastic state at all).

Interface contract (unchanged from the baseline):

    forward(feat, velocity=None, use_posterior=True, deterministic=True,
            detach_state=True)
        -> (output, reconstruction, kl, h_t, z_t, stats)

The tuple is returned EXACTLY as the parent produced it. The new tensors travel
as attributes -- ``last_sem_map``, ``last_cls_logit_residual``,
``pending_semantic_loss`` -- and extra ``stat_*`` diagnostics are added to the
existing ``stats`` dict. ``reset_state()`` also clears the auxiliary GT and the
pending loss, so a history frame can never be supervised against the wrong
frame's targets (the route-C discipline, §59).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.core.utils.gaussian import bev_centers_to_heatmap
from mmdet3d.models.builder import FUSION_LAYERS
from mmdet3d.models.fusion_layers.rssm_fusion import MotionAlignedRSSMFusion

__all__ = ['SemanticReadoutRSSMFusion']

_SEM_INPUTS = ('z', 'h', 'feat')
# A2-S: 'gaussian_full' is candidate A's full-resolution Gaussian target;
# 'channel_contrastive' keeps the *same* focal term and adds one ranking term
# per occupied class channel (prereg section 4.1).
_SEM_TARGET_MODES = ('gaussian_full', 'channel_contrastive')
# A2-R: 'per_class_slot' is candidate A (4 logits -> the class's own anchor
# slots, 12/48 channels); 'anchor_slot' broadcasts each class logit to all
# ``num_anchors`` slots of that class (48/48) using the SAME 4-channel
# predictor and the same parameter count (prereg section 4.2).
_CLS_READOUT_MODES = ('per_class_slot', 'anchor_slot')


@FUSION_LAYERS.register_module()
class SemanticReadoutRSSMFusion(MotionAlignedRSSMFusion):
    """Standard motion-aligned RSSM + z-semantic supervision + cls residual.

    Args:
        sem_num_classes (int): class channels of the auxiliary map and of the
            cls residual (TJ4D mainline: 4).
        sem_head_input (str): 'z' (the arm), 'h' or 'feat' (controls). Only
            'z' supports the candidate-A claim; the others exist so the
            arm can be falsified.
        sem_head_hidden (int): hidden channels of the auxiliary head.
        sem_supervision_weight (float): weight of ``loss_rssm_sem``. 0 turns
            the supervision off (readout-only ablation) without building the
            loss.
        object_min_radius (int): minimum Gaussian radius, in BEV cells, of the
            centre-heatmap target (0.32 m/cell on the mainline grid).
        cls_residual (bool): build ``z_cls_logits`` and emit
            ``last_cls_logit_residual`` for ``Anchor3DHead``.
        cls_residual_gate (float): runtime multiplier of the residual. 1.0 in
            training and validation; probes set it to 0.0 for the paired
            toggle that proves the readout is not decorative.
        log_sem_maps (bool): keep ``last_sem_map`` / ``last_cls_logit_residual``
            alive outside training (probes). Cheap, off by default.
    """

    def __init__(self,
                 sem_num_classes=4,
                 sem_head_input='z',
                 sem_head_hidden=64,
                 sem_supervision_weight=1.0,
                 object_min_radius=2,
                 sem_target_mode='gaussian_full',
                 sem_rank_q=0.999,
                 sem_rank_margin=1.0,
                 sem_rank_weight=0.1,
                 cls_residual=True,
                 cls_residual_gate=1.0,
                 cls_readout_mode='per_class_slot',
                 cls_readout_anchor_slots=None,
                 log_sem_maps=False,
                 init_cfg=None,
                 **kwargs):
        if sem_head_input not in _SEM_INPUTS:
            raise ValueError(
                f'sem_head_input must be one of {_SEM_INPUTS}, got '
                f'{sem_head_input!r}')
        if sem_num_classes < 1:
            raise ValueError('sem_num_classes must be >= 1')
        if sem_supervision_weight < 0:
            raise ValueError('sem_supervision_weight must be >= 0')
        if object_min_radius < 0:
            raise ValueError('object_min_radius must be >= 0')
        # A2 arm variables (prereg docs/rssm_candidate_A2_prereg_20261011.md
        # sections 4.1 / 4.2). Both defaults reproduce candidate A exactly, so
        # an unmodified config keeps the archived A behaviour bit for bit.
        if sem_target_mode not in _SEM_TARGET_MODES:
            raise ValueError(
                f'sem_target_mode must be one of {_SEM_TARGET_MODES}, got '
                f'{sem_target_mode!r}')
        if not 0.0 < sem_rank_q <= 1.0:
            raise ValueError(
                f'sem_rank_q must be in (0, 1], got {sem_rank_q!r}')
        if sem_rank_margin < 0:
            raise ValueError('sem_rank_margin must be >= 0')
        if sem_rank_weight < 0:
            raise ValueError('sem_rank_weight must be >= 0')
        if cls_readout_mode not in _CLS_READOUT_MODES:
            raise ValueError(
                f'cls_readout_mode must be one of {_CLS_READOUT_MODES}, got '
                f'{cls_readout_mode!r}')
        # Only consulted by ``anchor_slot``; it is the detector head's
        # ``num_anchors`` (6 sizes x 2 rotations = 12 on the TJ4D mainline). It
        # is a broadcast constant, NOT a parameter: both readout arms keep the
        # same single 4-channel predictor, so no capacity is added. None on
        # ``per_class_slot`` is fine -- the head's own mapping table decides
        # the destination channels there, exactly as in candidate A.
        if cls_readout_anchor_slots is not None:
            if int(cls_readout_anchor_slots) < 1:
                raise ValueError(
                    'cls_readout_anchor_slots must be >= 1, got '
                    f'{cls_readout_anchor_slots!r}')
            self.cls_readout_anchor_slots = int(cls_readout_anchor_slots)
        else:
            # ``R4Det`` fills this in on the first forward by reading the
            # detection head's own ``num_anchors`` -- duplicating the constant
            # in a config would let it drift away from ``conv_cls``, and a
            # silent shape mismatch there would corrupt the readout arm.
            self.cls_readout_anchor_slots = None

        # Build the new modules without perturbing the global RNG stream: the
        # baseline weights of every pre-existing module must come out bit
        # identical to a plain MotionAlignedRSSMFusion built with the same
        # seed (unit test T1), otherwise "off == baseline" is untestable.
        self.sem_num_classes = int(sem_num_classes)
        self.sem_head_input = sem_head_input
        self.sem_head_hidden = int(sem_head_hidden)
        self.sem_supervision_weight = float(sem_supervision_weight)
        self.object_min_radius = int(object_min_radius)
        self.sem_target_mode = sem_target_mode
        self.sem_rank_q = float(sem_rank_q)
        self.sem_rank_margin = float(sem_rank_margin)
        self.sem_rank_weight = float(sem_rank_weight)
        self.cls_residual_enabled = bool(cls_residual)
        self.cls_residual_gate = float(cls_residual_gate)
        self.cls_readout_mode = cls_readout_mode
        self.log_sem_maps = bool(log_sem_maps)

        # Deterministic super().__init__(): the parent calls init_weights(),
        # which consumes RNG for its own convs; those draws must be identical
        # to the baseline, so nothing new is constructed before it.
        super().__init__(init_cfg=init_cfg, **kwargs)

        # Capture the stream only around the NEW modules, then rewind: after
        # construction the RNG sits exactly where a plain
        # MotionAlignedRSSMFusion would leave it (unit test T1). The new
        # modules themselves get a deterministic init in init_weights().
        rng_cpu = torch.get_rng_state()
        rng_cuda = (torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available() else None)

        latent = self.latent_dim
        # Auxiliary supervision head — NOT in the detection forward path, so it
        # is only constructed when the supervision term is actually wanted
        # (a readout-only ablation must not carry dead parameters).
        if self.sem_supervision_weight > 0:
            self.sem_head = nn.Sequential(
                nn.Conv2d(latent, self.sem_head_hidden, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.sem_head_hidden, self.sem_num_classes, 1),
            )
            # CenterNet-style low prior so the focal term does not open with a
            # huge negative-gradient spike over the (99%+) background cells.
            nn.init.constant_(self.sem_head[2].bias, -2.19)
        else:
            self.sem_head = None

        # Detection readout: z-only, zero-init => exact identity at step 0.
        if self.cls_residual_enabled:
            self.z_cls_logits = nn.Conv2d(latent, self.sem_num_classes, 1)
            nn.init.zeros_(self.z_cls_logits.weight)
            nn.init.zeros_(self.z_cls_logits.bias)
        else:
            self.z_cls_logits = None

        torch.set_rng_state(rng_cpu)
        if rng_cuda is not None:
            torch.cuda.set_rng_state_all(rng_cuda)

        # Per-forward state (always reset at the top of forward()).
        self.current_object_gt = None
        self.current_object_pcr = None
        self.pending_semantic_loss = None
        self.last_sem_map = None
        self.last_cls_logit_residual = None
        self._last_h_t = None
        self._last_feat = None
        self._last_z_t = None

    def init_weights(self):
        """Re-assert the new modules' init AFTER the generic sweep.

        ``R4Det.init_weights()`` walks the whole detector, so anything set in
        ``__init__`` can be overwritten by mmcv's default conv init. The
        zero-init of the readout is not cosmetic -- it is what makes the arm
        start bit-identical to the baseline (prereg §2), so it is re-applied
        here, exactly like route-C re-inits its object head
        (``rssm_fusion.py:1381``).
        """
        # The parent __init__ already calls init_weights() once, i.e. before
        # these attributes exist -> getattr, not self.X.
        super().init_weights()
        sem_head = getattr(self, 'sem_head', None)
        if sem_head is not None:
            nn.init.constant_(sem_head[2].bias, -2.19)
        readout = getattr(self, 'z_cls_logits', None)
        if readout is not None:
            nn.init.zeros_(readout.weight)
            nn.init.zeros_(readout.bias)

    # ------------------------------------------------------------------
    # auxiliary GT plumbing
    # ------------------------------------------------------------------
    def set_object_gt(self, gt_bboxes_3d, gt_labels_3d, point_cloud_range):
        """Attach the CURRENT frame's GT for the semantic head.

        Rasterisation is deferred to ``forward`` because the heatmap grid is
        the runtime latent grid (216 x 248 on the mainline). The detector must
        call this immediately before the current-frame forward only;
        ``reset_state()`` clears it again.
        """
        if gt_bboxes_3d is None or gt_labels_3d is None:
            self.current_object_gt = None
            return
        if len(gt_bboxes_3d) != len(gt_labels_3d):
            raise ValueError(
                'gt_bboxes_3d / gt_labels_3d length mismatch: '
                f'{len(gt_bboxes_3d)} vs {len(gt_labels_3d)}')
        self.current_object_gt = (gt_bboxes_3d, gt_labels_3d)
        self.current_object_pcr = list(point_cloud_range)

    def reset_state(self):
        super().reset_state()
        self.current_object_gt = None
        self.current_object_pcr = None
        self.pending_semantic_loss = None
        self.last_sem_map = None
        self.last_cls_logit_residual = None
        self._last_z_t = None

    # ------------------------------------------------------------------
    # detection-path plumbing
    # ------------------------------------------------------------------
    def detection_cls_logit_residual(self, is_valid_mask=None):
        """The readout map on the DETECTION grid, ready for ``Anchor3DHead``.

        ``R4Det`` hands the detection head the TRANSPOSED BEV grid -- the
        latent works on (H, W) = (bev_h_, bev_w_) = (216, 248) while
        ``pts_feats`` is (248, 216) (``extract_feat`` permutes it), so the map
        has to follow the same convention (unit test T7). A sample whose
        recurrent state was unreliable falls back to the un-fused features in
        ``extract_feat``; its class residual must be identically zero there
        (unit test T6).

        Returns ``None`` when the readout is disabled, so the detector passes
        nothing to the head and the baseline path stays bit-identical.
        """
        residual = self.last_cls_logit_residual
        if residual is None:
            return None
        if residual.dim() != 4:  # pragma: no cover - guarded by construction
            raise ValueError(
                'cls logit residual must be (B, C, H, W), got '
                f'{tuple(residual.shape)}')
        out = residual.permute(0, 1, 3, 2).contiguous()
        if is_valid_mask is not None:
            keep = is_valid_mask.to(device=out.device, dtype=torch.bool)
            if keep.shape[0] != out.shape[0]:
                raise ValueError(
                    'is_valid_mask length mismatch: residual '
                    f'{out.shape[0]}, mask {keep.shape[0]}')
            out = torch.where(
                keep[:, None, None, None], out, torch.zeros_like(out))
        return out

    # ------------------------------------------------------------------
    # semantic map + loss
    # ------------------------------------------------------------------
    def _sem_source(self, h_t, z_t, feat):
        """Pick the semantic head's input.

        The three arms are experiment variables (the M2 z-exclusivity gate
        compares them), so an unrecognised value must NOT silently fall through
        to one of them: a typo would quietly turn the 'h' arm into a duplicate
        of the 'feat' arm, and M2 would then report a two-way comparison of the
        same tensor as if it were three. Validate loudly instead -- the
        constructor already range-checks this argument, and this is the
        runtime path the probes actually exercise.
        """
        arm = self.sem_head_input
        if arm == 'z':
            return z_t
        if arm == 'h':
            return h_t
        if arm == 'feat':
            return feat
        raise ValueError(
            f"sem_head_input must be one of 'z'/'h'/'feat', got {arm!r}")

    def _channel_rank_loss(self, logits, pos):
        """A2-S ranking term: the GT peak must beat its own background quantile.

        Exact contract from the A2 prereg (section 4.1); the four numbers are
        frozen at registration and must not be tuned during training:

            g_c   = max_{i in P_c} s_c[i]
            q_c   = Quantile_{alpha=q}( { s_c[j] : j in B_c } )
            L_rank = (1/|C_occ|) * sum_c relu(margin + q_c - g_c)

        with ``q = sem_rank_q`` (0.999), ``margin = sem_rank_margin`` (1.0)
        and a per-occupied-channel arithmetic mean. Channels with no GT
        (``P_c = empty``) are skipped entirely -- they are not in ``C_occ``
        and contribute no gradient; a frame with no occupied channel at all
        returns a zero tensor, matching the focal branch's ``num_pos == 0``
        handling.
        """
        dtype = logits.dtype
        zero = torch.zeros((), device=logits.device, dtype=dtype)
        per_channel = []
        # Per SAMPLE first: the batch is the data-loader batch, and mixing
        # frames here would let a busy frame's background quantile set the
        # threshold for a sparse frame in the same tensor.
        for b in range(logits.shape[0]):
            logits_b = logits[b].to(torch.float32).float()
            pos_b = pos[b]
            terms = []
            for c in range(logits_b.shape[0]):
                mask = pos_b[c]
                if not bool(mask.any()):
                    continue
                flat = logits_b[c].reshape(-1)
                flat_mask = mask.reshape(-1)
                gt_peak = flat[flat_mask].max()
                background = flat[~flat_mask]
                if background.numel() == 0:
                    # Every cell of this channel is GT-marked: there is no
                    # background distribution to contrast against, so the
                    # channel cannot be ranked. Skip rather than fabricate a
                    # quantile from an empty tensor.
                    continue
                # torch.quantile is the frozen tie/interpolation rule; it is
                # deliberately not hand-rolled with a sort.
                bg_q = torch.quantile(background, self.sem_rank_q)
                terms.append(F.relu(self.sem_rank_margin + bg_q - gt_peak))
            if terms:
                per_channel.append(torch.stack(terms).mean())
        if not per_channel:
            return zero
        return torch.stack(per_channel).mean().to(dtype)

    def _semantic_heatmap_loss(self, sem_map, target):
        """CenterNet focal loss + objectness diagnostics.

        Same formulation as the route-C head (§59) so the two generations stay
        comparable: logits in, ``logsigmoid`` for stability, negative weights
        ``neg^4``, normalised by ``num_pos``, and an image without objects
        contributes no gradient instead of a ~200x-inflated term.
        """
        zero = torch.zeros((), device=sem_map.device, dtype=sem_map.dtype)
        logits = sem_map.to(torch.float32).float()
        prob = logits.sigmoid()
        pos = target.eq(1.0).float()
        neg = 1.0 - target
        pos_loss = -F.logsigmoid(logits) * (1.0 - prob).pow(2) * pos
        neg_loss = -F.logsigmoid(-logits) * prob.pow(2) * neg.pow(4) * neg
        num_pos = pos.sum()
        if num_pos.item() == 0:
            return zero, zero, zero, num_pos.detach()
        loss = (pos_loss.sum() + neg_loss.sum()) / num_pos
        # A2-S: same focal term, PLUS one ranking term. The focal branch and
        # its weight are untouched, so the only difference between the two
        # arms is the presence of this term (prereg section 4.1).
        if self.sem_target_mode == 'channel_contrastive':
            loss = loss + self.sem_rank_weight * self._channel_rank_loss(
                logits, pos.bool())
        # Hit@1 over occupied channels + the fixed 0.5-threshold recall: a
        # Gaussian soft target does not drive every centre past 0.5 early on,
        # so hit@1 is the reading that says whether the map is object-aligned.
        flat = prob.flatten(2)
        peak_idx = flat.argmax(dim=2)
        peak = torch.zeros_like(flat).scatter_(
            2, peak_idx[:, :, None], 1.0).view_as(prob)
        hits = (peak * pos).sum()
        occupied = (pos.sum(dim=(2, 3)) > 0).sum().clamp_min(1.0)
        hit1 = hits / occupied
        thr50 = ((prob > 0.5) & pos.bool()).sum().float() / num_pos
        return (loss.to(sem_map.dtype), hit1.detach(), thr50.detach(),
                num_pos.detach())

    def _rasterize_target(self, device, out_size):
        if self.current_object_gt is None:
            return None
        boxes, labels = self.current_object_gt
        return bev_centers_to_heatmap(
            boxes,
            labels,
            num_classes=self.sem_num_classes,
            out_size=out_size,
            point_cloud_range=self.current_object_pcr,
            min_radius=self.object_min_radius,
            device=device,
            dtype=torch.float32)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, feat, velocity=None, use_posterior=True,
                deterministic=True, detach_state=True):
        out = super().forward(
            feat, velocity=velocity, use_posterior=use_posterior,
            deterministic=deterministic, detach_state=detach_state)
        output, reconstruction, kl, h_t, z_t, stats = out

        self._last_h_t = h_t
        self._last_feat = feat
        if self.training:
            # Kept for the gradient-pressure probe and for T3; the tensor is
            # already in the graph, so this adds no extra memory.
            self._last_z_t = z_t

        self.pending_semantic_loss = None
        self.last_sem_map = None
        self.last_cls_logit_residual = None

        # The readout is part of the DETECTION path: it must be emitted in
        # training AND validation, otherwise train/test contracts differ and a
        # validation AP number would not reflect the trained model at all.
        if self.cls_residual_enabled:
            residual = self.z_cls_logits(z_t.to(torch.float32)).float()
            if self.cls_readout_mode == 'anchor_slot':
                # A2-R: broadcast each class logit to ALL anchor slots of that
                # class, i.e. write every one of the ``num_anchors *
                # num_classes`` conv_cls channels instead of only the class's
                # own anchor slot. The predictor and its parameter count are
                # unchanged -- only the destination set grows (prereg 4.2).
                #
                # conv_cls is laid out ``[anchor * num_classes + class]``, so
                # writing class c into every anchor is exactly
                # ``(B, C, H, W) -> (B, A, C, H, W) -> (B, A*C, H, W)``: no
                # mapping table and no dependence on ``anchor_class_mapping``
                # are involved, unlike the ``per_class_slot`` arm where the
                # head's table picks one slot per class.
                if self.cls_readout_anchor_slots is None:
                    raise ValueError(
                        "cls_readout_mode='anchor_slot' needs "
                        'cls_readout_anchor_slots (the head num_anchors); it '
                        'cannot be guessed here without risking a silent '
                        'shape mismatch against conv_cls. R4Det binds it on '
                        'the first forward, so a None here means the fusion '
                        'is being driven outside the detector.')
                b, c = residual.shape[0], residual.shape[1]
                residual = residual[:, None].expand(
                    b, self.cls_readout_anchor_slots, c,
                    *residual.shape[2:]).reshape(b, -1, *residual.shape[2:])
            if self.cls_residual_gate != 1.0:
                residual = residual * float(self.cls_residual_gate)
            self.last_cls_logit_residual = residual
            if stats is not None:
                stats['stat_cls_res_rms'] = (
                    residual.detach().pow(2).mean().sqrt().float())

        # The auxiliary map is supervision/probe material: always on in
        # training, opt-in elsewhere (probes set `log_sem_maps`). Not built at
        # all when the supervision weight is 0 (readout-only ablation).
        if not ((self.training or self.log_sem_maps)
                and self.sem_head is not None):
            return out

        src = self._sem_source(h_t, z_t, feat)
        sem_map = self.sem_head(src.to(torch.float32)).float()
        self.last_sem_map = sem_map

        if (self.training and self.sem_supervision_weight > 0
                and self.current_object_gt is not None):
            target = self._rasterize_target(
                sem_map.device, tuple(sem_map.shape[-2:]))
            if target.shape[0] != sem_map.shape[0]:
                raise ValueError(
                    'semantic GT batch mismatch: expected '
                    f'{sem_map.shape[0]}, got {target.shape[0]}')
            if target.shape[1] != sem_map.shape[1]:
                raise ValueError(
                    'semantic GT class mismatch: expected '
                    f'{sem_map.shape[1]}, got {target.shape[1]}')
            loss, hit1, thr50, num_pos = self._semantic_heatmap_loss(
                sem_map, target)
            self.pending_semantic_loss = loss * self.sem_supervision_weight
            # NOTE on the log: every stat_* entry is summed into the reported
            # total loss by mmcv's parse_losses (it contributes no gradient).
            # Only O(1) diagnostics go in here, and `stat_sem_loss` /
            # `stat_sem_num_pos` deliberately do NOT, so this run's loss
            # decomposition stays comparable with §70.2's percentages. num_pos
            # is read directly by the probes (it is a dataset constant, not a
            # training signal).
            if stats is not None:
                stats['stat_sem_hit1'] = hit1.float()
                stats['stat_sem_thr50'] = thr50.float()
                stats['stat_sem_pos_per_img'] = (
                    num_pos.float() / max(1, sem_map.shape[0]) / 100.0)
        return out
