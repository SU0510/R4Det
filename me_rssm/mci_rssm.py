"""Measurement-Conditioned Innovation RSSM (MCI-RSSM).

Merged model, prepared 2026-10-02 (design: docs/rssm_review_20261002.md §5A/5B,
me_rssm/docs/12). This file only ADDS a new class to the me_rssm research
package; no existing file is modified.

Why this class exists
---------------------
Two partially-validated lines were never combined:

* ``MotionAlignedInnovationRSSMFusion`` (MAI, mainline
  ``rssm_fusion.py:1865``) fixed the *information flow* of the posterior:
  ``mu_q = mu_p + posterior_mu([h_t, innovation])`` with
  ``innovation = e_t - stopgrad(obs_prior(h_t))``. Measured effects
  (training_runs_full.md §62-64): h/e dependence ratio 2.23 -> <1,
  innovation specific energy 47-67%. But seed0 +0.84 / seed1 -2.51 and the
  z -> head causal effect stayed bias-dominated (zero_z 26-56% vs
  shuffle_z 2.7-8.4%).

* ``MotionEvidenceRSSMFusion`` (ME-RSSM, ``motion_evidence_rssm.py``)
  supplies the three *measurement* mechanisms that were designed, sanity
  tested (13/13) and never trained: Doppler-conditioned deformable
  alignment offsets, dynamic-evidence-modulated GRU update gate, and
  modality observations with per-pixel reliability.

MCI-RSSM combines them and closes the remaining loop at the *readout*
(the missing link identified in the review, P1):

    Predict   h_t = GRU( align(h_{t-1}; offset + M(e_mot)), z_{t-1} )
              u'  = u + tanh(beta) * d * u * (1-u)          (dynamic gate)
    Predict   e_hat_t = obs_prior(h_t)                        (from MAI)
    Correct   innovation = e_obs - stopgrad(e_hat_t)
              delta  = posterior_mu([h_t, innovation])       (from MAI)
              z_t    = mu_p + g_c * (z_sel - mu_p)
              g_c    = 1 + tanh(gate_conv([c_cam, c_mot, |innovation|]))
    Readout   output  = feat + g_r * output_proj(z_t)
              g_r     = 1 + tanh(readout_gate_conv(same inputs))

Identity-start contract (stronger than ME-RSSM's sigmoid(4) ~ 0.982 gain):
every added pathway is zero-init, and both gates use ``1 + tanh(.)`` with
zero-init convs, so g_c = g_r = 1 EXACTLY at init. With equal weights the
module is bitwise-identical to MAI in all three regimes:

    E1  bus empty                     -> exactly MAI
    E2  bus present, switches off     -> exactly MAI (switch-off modules
                                         are not even constructed)
    E3  bus present, all switches on,
        fresh init                    -> exactly MAI

so any MAI checkpoint fine-tunes MCI with bitwise-identical starting
behaviour, and every switch is a clean single-variable ablation.

Gate input note (06_second_layer P2, pre-registered): the gain convs see
the *magnitude* of the innovation through a zero-init 1x1 reduction,
``innovation.detach().abs() -> innovation_mag_conv``. The detach is a
design constraint: the gate must OBSERVE disagreement but must not be a
gradient shortcut that could shrink the innovation itself. c_cam / c_mot
keep gradients (they are learned confidence estimates, as in ME-RSSM).

DDP safety: modules are constructed if and only if their switch is on AND
the information they consume can exist. Two hard validation rules follow
from this (both enforced in __init__):

* ``use_motion_offsets`` / ``use_dynamic_gate`` require
  ``use_modality_obs=True`` (their inputs come from the Doppler bus);
* the reliability/readout gates require at least one input source
  (modality confidences or innovation magnitude).

Training with ``use_modality_obs=True`` still requires the full producer
chain (RadarPillarFeatureNetMotion + PointPillarsScatterMotion +
ConcatConvFusionStash); with an empty bus the bus-consuming modules would
not run and DDP would abort on unused parameters. Degraded-bus runs are
for eval/sanity only (same rule as me_rssm/docs/06).

Interface contract with the unmodified detector (R4Det.extract_feat):
    forward(feat, velocity=None, use_posterior=True, deterministic=True,
            detach_state=True)
        -> (output, reconstruction, latent_loss, h_t, z_t, stats)
    reset_state() / reset_for_samples(mask)
    attribute kl_scale mutated by KLScaleSchedulerHook
``posterior_struct`` is 'innovation' or 'standard' so the existing
diagnostic tools (tools/diagnose_z_utilization.py,
tools/probe_posterior_source.py) read the right tensors.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import auto_fp16
from mmcv.cnn import ConvModule

from mmdet3d.models.builder import FUSION_LAYERS
from mmdet3d.models.fusion_layers.rssm_fusion import (
    MotionAlignedInnovationRSSMFusion,
)

from .modality_bus import ModalityContext

__all__ = ['MeasurementConditionedInnovationRSSMFusion']


@FUSION_LAYERS.register_module()
class MeasurementConditionedInnovationRSSMFusion(
        MotionAlignedInnovationRSSMFusion):
    # Probe-only intervention hook (me_rssm/sanity/probe_z_utilization_gated.py):
    # when not None, a callable z_t -> z_t' applied to the FINAL latent
    # (after the correct gate, before decoder/readout/state write) so a
    # counterfactual can replace exactly what the head consumes. None in
    # training/eval -> a no-op branch, never a parameter.
    _z_intervention = None

    """MAI backbone + Doppler-conditioned predict + reliability-gated
    correct and readout. See the module docstring for the full design.

    New args (all MAI args are inherited unchanged):

    Args:
        posterior_mode (str): 'innovation' (default, MAI semantics) or
            'standard' (``mu_q = posterior_mu([h_t, e_obs])``; ablation
            switch that removes obs_prior/innovation entirely).
        innov_in_gain (bool): feed the detached innovation magnitude into
            the gate convs (06-P2). Requires posterior_mode='innovation'.
        use_motion_offsets (bool): ME-RSSM C1, zero-init additive Doppler
            term on the deformable alignment offsets. Requires bus.
        use_dynamic_gate (bool): ME-RSSM C3, dynamic-evidence-modulated
            GRU update gate, ``tanh(dyn_beta)`` init 0. Requires bus.
        use_reliability_gain (bool): correct-step gate
            ``z_t = mu_p + g_c * (z_sel - mu_p)``, ``g_c = 1 + tanh(.)``
            zero-init (identity start, unlike ME-RSSM's sigmoid(4)).
        use_gated_readout (bool): readout gate
            ``output = feat + g_r * output_proj(z_t)`` with the same
            identity-start parameterisation. This is the P1 fix: the
            head's reliance on z becomes per-pixel and reliability-driven
            instead of a global additive bias.
        use_modality_obs (bool): master switch for consuming the bus
            (modality encoders, zero-init ``obs_proj`` residual into the
            observation, confidence maps).
        motion_channels (int): channels of the stashed 'velocity_bev'
            canvas (7, see radar_motion_chain).
        camera_bev_channels (int): channels of 'camera_bev' (256).
        motion_hidden / motion_latent / camera_hidden / camera_latent:
            widths of the modality observation encoders.
        motion_grad (bool): allow detection gradients into the Doppler
            canvas. Default False (measurement, detached).
        modality_dropout (float): training-time probability of zeroing
            per-sample confidence maps (fallback practice). 0 disables.
        conf_init_bias (float): bias of the confidence convs (sigmoid(2)
            ~ 0.88 at init); zero weights make the map constant.
        dyn_init_bias (float): bias of the dynamic-evidence conv
            (sigmoid(-2) ~ 0.12, 'mostly static' start).
        state_init (str): 'zero' (baseline h_0 = z_0 = 0) or 'obs'
            (me_rssm/docs/07 P1: zero-init 1x1 convs bootstrap h_0/z_0
            from the current observation; bitwise-identical until trained).
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        latent_dim=None,
        hidden_dim=64,
        action_dim=2,
        kl_scale=1.0,
        free_nats=1.0,
        min_std=0.1,
        init_std=0.2,
        obs_pred_loss_weight=0.05,
        posterior_mode='innovation',
        innov_in_gain=True,
        use_motion_offsets=True,
        use_dynamic_gate=True,
        use_reliability_gain=True,
        use_gated_readout=True,
        use_modality_obs=True,
        motion_channels=7,
        camera_bev_channels=256,
        motion_hidden=32,
        motion_latent=32,
        camera_hidden=128,
        camera_latent=64,
        motion_grad=False,
        modality_dropout=0.0,
        conf_init_bias=2.0,
        dyn_init_bias=-2.0,
        state_init='zero',
        norm_cfg=dict(type='BN', requires_grad=True),
        act_cfg=dict(type='ReLU', inplace=True),
        align_kernel_size=3,
        align_deform_groups=1,
        align_z_state=True,
        init_cfg=None,
    ):
        if posterior_mode not in ('innovation', 'standard'):
            raise ValueError(
                f"posterior_mode must be 'innovation' or 'standard', "
                f'got {posterior_mode!r}')
        if state_init not in ('zero', 'obs'):
            raise ValueError(
                f"state_init must be 'zero' or 'obs', got {state_init!r}")
        if posterior_mode != 'innovation' and innov_in_gain:
            raise ValueError(
                'innov_in_gain requires posterior_mode="innovation" '
                '(there is no innovation to observe in standard mode)')
        if use_motion_offsets and not use_modality_obs:
            raise ValueError(
                'use_motion_offsets requires use_modality_obs=True: the '
                'Doppler evidence canvas only exists on the bus')
        if use_dynamic_gate and not use_modality_obs:
            raise ValueError(
                'use_dynamic_gate requires use_modality_obs=True: the '
                'dynamic-evidence canvas only exists on the bus')
        gate_in_channels = (
            (2 if use_modality_obs else 0)
            + (1 if (innov_in_gain and posterior_mode == 'innovation')
               else 0))
        if (use_reliability_gain or use_gated_readout) and \
                gate_in_channels == 0:
            raise ValueError(
                'the reliability/readout gates need at least one input '
                'source: enable use_modality_obs or innov_in_gain')
        if modality_dropout and not use_modality_obs:
            raise ValueError(
                'modality_dropout requires use_modality_obs=True')
        if obs_pred_loss_weight < 0:
            raise ValueError('obs_pred_loss_weight must be >= 0')

        # Parent chain (MAI -> MotionAlignedRSSMFusion -> BEVRSSMTemporal-
        # Fusion) builds encoder/transition/prior/posterior/decoder/
        # output_proj/alignment/obs_prior and runs init_weights() BEFORE
        # any module below exists: the inherited RNG stream and the
        # checkpoint key set of MAI are untouched.
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            kl_scale=kl_scale,
            free_nats=free_nats,
            min_std=min_std,
            init_std=init_std,
            obs_pred_loss_weight=obs_pred_loss_weight,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            align_kernel_size=align_kernel_size,
            align_deform_groups=align_deform_groups,
            align_z_state=align_z_state,
            init_cfg=init_cfg,
        )
        if posterior_mode != 'innovation':
            # Diagnostic tools key off this attribute; MAI's __init__ set
            # 'innovation' unconditionally.
            self.posterior_struct = 'standard'
            # MAI's __init__ always creates obs_prior; in standard mode it
            # has no consumer (innovation/obs_pred do not exist), which
            # would surface as a DDP unused-parameter abort. Remove it:
            # MAI checkpoints still load with strict=False (obs_prior.*
            # becomes an unexpected key, ignored by design).
            del self.obs_prior

        self.posterior_mode = posterior_mode
        self.innov_in_gain = innov_in_gain
        self.use_motion_offsets = use_motion_offsets
        self.use_dynamic_gate = use_dynamic_gate
        self.use_reliability_gain = use_reliability_gain
        self.use_gated_readout = use_gated_readout
        self.use_modality_obs = use_modality_obs
        self.motion_channels = motion_channels
        self.camera_bev_channels = camera_bev_channels
        self.motion_grad = motion_grad
        self.modality_dropout = modality_dropout
        self.conf_init_bias = conf_init_bias
        self.dyn_init_bias = dyn_init_bias
        self.state_init = state_init
        self.gate_in_channels = gate_in_channels

        # ---- modality observations + confidence (bus consumers) --------
        if use_modality_obs:
            self.motion_encoder = nn.Sequential(
                ConvModule(motion_channels, motion_hidden, 3, padding=1,
                           norm_cfg=norm_cfg, act_cfg=act_cfg),
                ConvModule(motion_hidden, motion_latent, 3, padding=1,
                           norm_cfg=norm_cfg, act_cfg=act_cfg),
            )
            self.camera_encoder = nn.Sequential(
                ConvModule(camera_bev_channels, camera_hidden, 3, padding=1,
                           norm_cfg=norm_cfg, act_cfg=act_cfg),
                ConvModule(camera_hidden, camera_latent, 3, padding=1,
                           norm_cfg=norm_cfg, act_cfg=act_cfg),
            )
            # Zero-init residual into the observation: e_obs == e_t until
            # trained, which is what keeps E3 (identity start) bitwise.
            self.obs_proj = nn.Conv2d(motion_latent + camera_latent,
                                      in_channels, 1)
            self.conf_cam = nn.Conv2d(camera_latent, 1, 1)
            self.conf_mot = nn.Conv2d(motion_latent, 1, 1)

        # ---- Doppler-conditioned alignment offsets (ME-RSSM C1) --------
        if use_motion_offsets:
            k2 = align_kernel_size * align_kernel_size
            self.motion_offset_channels = 2 * align_deform_groups * k2
            self.motion_offset_h = nn.Conv2d(
                motion_latent, self.motion_offset_channels,
                align_kernel_size, padding=align_kernel_size // 2)
            if align_z_state:
                self.motion_offset_z = nn.Conv2d(
                    motion_latent, self.motion_offset_channels,
                    align_kernel_size, padding=align_kernel_size // 2)
            else:
                self.motion_offset_z = None

        # ---- dynamic-evidence update gate (ME-RSSM C3) ------------------
        if use_dynamic_gate:
            self.dyn_conv = nn.Conv2d(motion_channels, 1, 1)
            self.dyn_beta = nn.Parameter(torch.tensor(0.0))

        # ---- gates (correct step + readout) -----------------------------
        # 1 + tanh(.) with zero-init convs: g == 1 EXACTLY at init, range
        # (0, 2). g < 1 shrinks the correction toward the motion-propagated
        # prediction (fallback), g > 1 amplifies it where the observation
        # is trustworthy and the innovation is large.
        if use_reliability_gain:
            self.correct_gate_conv = nn.Conv2d(
                gate_in_channels, 1, 3, padding=1)
        if use_gated_readout:
            self.readout_gate_conv = nn.Conv2d(
                gate_in_channels, 1, 3, padding=1)
        if innov_in_gain and posterior_mode == 'innovation' and \
                (use_reliability_gain or use_gated_readout):
            self.innovation_mag_conv = nn.Conv2d(self.latent_dim, 1, 1)

        # ---- observation-bootstrapped initial state (docs/07 P1) --------
        if state_init == 'obs':
            self.h_init_conv = nn.Conv2d(in_channels, self.channels, 1)
            self.z_init_conv = nn.Conv2d(in_channels, self.latent_dim, 1)
        else:
            self.h_init_conv = None
            self.z_init_conv = None
        self._pending_obs_init = None
        # init_weights() already ran inside super().__init__() (before any
        # MCI module existed); this flag lets the override re-apply our
        # zero-init idempotently on later (mmcv-recursive) calls.
        self._mci_modules_built = True

        self._init_mci_weights()

    # ------------------------------------------------------------------
    # init
    # ------------------------------------------------------------------
    def _init_mci_weights(self):
        """Zero-init every added pathway (identity start, idempotent).

        Biases that must start non-zero are set explicitly: confidence
        maps start moderately positive, the dynamic-evidence map starts
        'mostly static', dyn_beta is exactly 0. The gate convs get zero
        weight AND zero bias so 1 + tanh(0) = 1 exactly.

        Deadlock note (2026-10-05, found by weight inspection after the
        three-seed runs): ``innovation_mag_conv`` must NOT be zero-init.
        Its output feeds the gate convs; if both it and the gate convs
        start at zero, the innov channel's gradient is identically zero
        forever (output == 0 -> downstream weight slice never grows ->
        upstream never receives gradient) -- the section-38.10 dead-branch
        pattern in its product-of-zeros form. Xavier-init breaks the
        deadlock while the gate convs stay zero-init, so the identity-
        start contract (dev == tanh(0) == 0) is unaffected. All MCI runs
        before this fix trained with an inert innov channel (documented
        in 68.8i); the fix lands with the pdrop10 follow-up generation.
        """
        zero_modules = [
            self.obs_proj if self.use_modality_obs else None,
            self.motion_offset_h if self.use_motion_offsets else None,
            self.motion_offset_z if self.use_motion_offsets else None,
            self.correct_gate_conv if self.use_reliability_gain else None,
            self.readout_gate_conv if self.use_gated_readout else None,
            self.h_init_conv, self.z_init_conv,
        ]
        for m in zero_modules:
            if m is None:
                continue
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        if getattr(self, 'innovation_mag_conv', None) is not None:
            # Non-zero init: see the deadlock note above. The identity
            # start is carried by the DOWNSTREAM gate convs (zero-init),
            # not by this reduction.
            nn.init.xavier_uniform_(self.innovation_mag_conv.weight)
            nn.init.zeros_(self.innovation_mag_conv.bias)
        if self.use_modality_obs:
            nn.init.zeros_(self.conf_cam.weight)
            nn.init.zeros_(self.conf_mot.weight)
            with torch.no_grad():
                self.conf_cam.bias.fill_(self.conf_init_bias)
                self.conf_mot.bias.fill_(self.conf_init_bias)
        if self.use_dynamic_gate:
            nn.init.zeros_(self.dyn_conv.weight)
            with torch.no_grad():
                self.dyn_conv.bias.fill_(self.dyn_init_bias)
            self.dyn_beta.data.zero_()

    def init_weights(self):
        # Parent chain xavier-inits encoder/GRU/mu/logstd/output_proj and
        # zero-inits the alignment generators; obs_prior is xavier'd by
        # the MAI __init__. The parent init_weights() runs before our
        # modules exist (flag guard), so re-apply ours idempotently on
        # every later call.
        super().init_weights()
        if getattr(self, '_mci_modules_built', False):
            self._init_mci_weights()

    # ------------------------------------------------------------------
    # state resets (state_init='obs' pending semantics, from docs/07)
    # ------------------------------------------------------------------
    def reset_state(self):
        super().reset_state()
        self._pending_obs_init = None

    def reset_for_samples(self, mask):
        if self.h_init_conv is None:
            super().reset_for_samples(mask)
            return
        if self.h_state is None or not mask.any():
            return
        mask = mask.to(device=self.h_state.device, dtype=torch.bool)
        keep = ~mask
        self.h_state = torch.where(
            keep[:, None, None, None], self.h_state,
            torch.zeros_like(self.h_state))
        self.z_state = torch.where(
            keep[:, None, None, None], self.z_state,
            torch.zeros_like(self.z_state))
        if (self._pending_obs_init is None
                or self._pending_obs_init.shape[0] != self.h_state.shape[0]
                or self._pending_obs_init.device != self.h_state.device):
            self._pending_obs_init = torch.zeros(
                self.h_state.shape[0], dtype=torch.bool,
                device=self.h_state.device)
        self._pending_obs_init = self._pending_obs_init | mask

    # ------------------------------------------------------------------
    # context (bus)
    # ------------------------------------------------------------------
    def _read_context(self, feat):
        """Fetch/normalize the per-frame modality context (ME-RSSM contract).

        Returns a dict or None when the bus is empty / inconsistent /
        switched off. Canvas transforms: 'velocity_bev' (B,Cm,496,432)
        avg_pool(2) -> (B,Cm,248,216) -> permute -> (B,Cm,216,248);
        'camera_bev' (B,C,248,216) -> permute -> (B,C,216,248).
        """
        if not self.use_modality_obs:
            return None
        vel_raw = ModalityContext.get('velocity_bev')
        cam_raw = ModalityContext.get('camera_bev')
        B = feat.shape[0]
        if vel_raw is None or cam_raw is None:
            return None
        if vel_raw.shape[0] != B or cam_raw.shape[0] != B:
            return None
        if vel_raw.shape[1] != self.motion_channels:
            return None

        vel = F.avg_pool2d(vel_raw, 2)
        vel = vel.permute(0, 1, 3, 2).contiguous()
        cam = cam_raw.permute(0, 1, 3, 2).contiguous()
        if vel.shape[-2:] != feat.shape[-2:] or \
                cam.shape[-2:] != feat.shape[-2:]:
            return None

        if not self.motion_grad:
            vel = vel.detach()
            cam = cam.detach()

        e_mot = self.motion_encoder(vel)
        e_cam = self.camera_encoder(cam)
        c_cam = torch.sigmoid(self.conf_cam(e_cam))
        c_mot = torch.sigmoid(self.conf_mot(e_mot))

        if self.training and self.modality_dropout > 0:
            drop_cam = (torch.rand(B, 1, 1, 1, device=feat.device) <
                        self.modality_dropout).type_as(c_cam)
            drop_mot = (torch.rand(B, 1, 1, 1, device=feat.device) <
                        self.modality_dropout).type_as(c_mot)
            c_cam = c_cam * (1.0 - drop_cam)
            c_mot = c_mot * (1.0 - drop_mot)

        dyn = torch.sigmoid(self.dyn_conv(vel)) \
            if self.use_dynamic_gate else None

        return dict(vel=vel, e_mot=e_mot, e_cam=e_cam, c_cam=c_cam,
                    c_mot=c_mot, dyn=dyn)

    # ------------------------------------------------------------------
    # building blocks
    # ------------------------------------------------------------------
    def _align_with_motion(self, state, feat, offset_mask_conv, deform_conv,
                           motion_conv=None, e_mot=None):
        """Baseline deformable alignment + zero-init additive motion term."""
        concat = torch.cat([feat, state], dim=1)
        offset_and_mask = offset_mask_conv(concat)
        k2 = self.align_kernel_size * self.align_kernel_size
        o1 = 2 * self.align_deform_groups * k2
        if motion_conv is not None and e_mot is not None:
            offset = offset_and_mask[:, :o1] + motion_conv(e_mot)
            offset_and_mask = torch.cat(
                [offset, offset_and_mask[:, o1:]], dim=1)
        offset = offset_and_mask[:, :o1]
        mask = offset_and_mask[:, o1:].sigmoid()
        return deform_conv(state, offset, mask)

    def _gru_with_dynamic_gate(self, x, h, dyn_map=None):
        """Baseline ConvGRUCell math, optionally with the dynamic gate.

        u' = u + beta * d * u * (1 - u), beta = tanh(dyn_beta) in (-1, 1):
        for |beta| <= 1 the map sends (0, 1) into (u^2, 1), so the update
        gate stays a valid probability. beta=0 or dyn_map=None reproduces
        the parent cell exactly.
        """
        cell = self.transition
        combined = torch.cat([x, h], dim=1)
        r = torch.sigmoid(cell.reset_conv(combined))
        u = torch.sigmoid(cell.update_conv(combined))
        if dyn_map is not None and self.use_dynamic_gate:
            beta = torch.tanh(self.dyn_beta)
            u = u + beta * dyn_map * u * (1.0 - u)
        combined_c = torch.cat([x, r * h], dim=1)
        h_candidate = torch.tanh(cell.candidate_conv(combined_c))
        return (1.0 - u) * h + u * h_candidate

    def _gate_input(self, ctx, innovation, feat):
        """Assemble the shared gate input map.

        Channels: [c_cam, c_mot] when use_modality_obs (zeros when the
        bus is empty so the conv always sees its full input width), plus
        the detached innovation magnitude reduction when enabled. Returns
        None when the gates have no input source (cannot happen for
        validated configurations that reach this code path with gates on,
        but kept explicit for safety).
        """
        parts = []
        if self.use_modality_obs:
            B, _, H, W = feat.shape
            if ctx is not None:
                parts.append(ctx['c_cam'])
                parts.append(ctx['c_mot'])
            else:
                zeros = feat.new_zeros(B, 1, H, W)
                parts.append(zeros)
                parts.append(zeros.clone())
        if getattr(self, 'innovation_mag_conv', None) is not None and \
                innovation is not None:
            parts.append(self.innovation_mag_conv(innovation.detach().abs()))
        if not parts:
            return None
        if len(parts) == 1:
            return parts[0]
        return torch.cat(parts, dim=1)

    def _correct_gain_dev(self, gate_in):
        """Deviation form of the correct gate: g_c = 1 + tanh(conv(.)).

        The forward applies ``z_sel + dev * (z_sel - mu_p)`` rather than
        ``mu_p + (1 + dev) * (z_sel - mu_p)``: at init dev == 0 exactly,
        so z_t == z_sel BITWISE (no float rounding through
        ``mu_p + 1.0 * (z_sel - mu_p)``).
        """
        return torch.tanh(self.correct_gate_conv(gate_in))

    def _readout_gain_dev(self, gate_in):
        return torch.tanh(self.readout_gate_conv(gate_in))

    def _correct_gain_map(self, gate_in):
        """Full gain map (stats/tests): 1 + deviation, range (0, 2)."""
        return 1.0 + self._correct_gain_dev(gate_in)

    def _readout_gain_map(self, gate_in):
        return 1.0 + self._readout_gain_dev(gate_in)

    def _posterior_observation(self, e_t):
        """Probe-compatibility hook (tools/probe_posterior_source.py).

        The posterior conditions on ``e_obs = e_t + obs_proj([c_cam*e_cam,
        c_mot*e_mot])``, not on the raw encoder output. The probe recomposes
        the innovation outside a forward and honors this method when present
        (it captures ``e_t`` from a hook). Eval-mode forwards stash the
        detached residual so the reconstruction is exact; without a stash
        (no forward yet, or obs_proj still zero) it falls back to ``e_t``,
        which is exact at init. Training forwards skip the stash (probes
        run in eval mode; the module must not hold graph references).
        """
        residual = getattr(self, '_last_obs_residual', None)
        if residual is None:
            return e_t
        return e_t + residual

    # ------------------------------------------------------------------
    # forward -- same signature and 6-tuple return as MAI
    # ------------------------------------------------------------------
    @auto_fp16(apply_to=['feat', 'velocity'])
    def forward(self, feat, velocity=None, use_posterior=True,
                deterministic=True, detach_state=True):
        B, C, H, W = feat.shape
        ctx = self._read_context(feat)
        have_ctx = ctx is not None

        # ---- velocity / ego-action (inherited, disabled by action_dim=0)
        if self.use_action:
            if velocity is None:
                velocity = torch.zeros(B, self.action_dim,
                                       device=feat.device)
            velocity_map = velocity[:, :, None, None].expand(
                B, self.action_dim, H, W)
        else:
            velocity_map = None

        # ---- observation encoding (before state init: 'obs' bootstrap
        # reads it; encoder(feat) has no state dependency, so computing
        # it early is output-equivalent to the MAI ordering) ------------
        e_t = self.encoder(feat)

        # ---- state init -------------------------------------------------
        if self.h_state is None or self.h_state.shape[0] != B:
            if self.h_init_conv is not None:
                self.h_state = self.h_init_conv(e_t).to(dtype=feat.dtype)
                self.z_state = self.z_init_conv(e_t).to(dtype=feat.dtype)
            else:
                self.h_state = torch.zeros(
                    B, self.channels, H, W, device=feat.device,
                    dtype=feat.dtype)
                self.z_state = torch.zeros(
                    B, self.latent_dim, H, W, device=feat.device,
                    dtype=feat.dtype)
            self._pending_obs_init = None
        elif (self._pending_obs_init is not None
                and self.h_init_conv is not None
                and self._pending_obs_init.shape[0] == B
                and self._pending_obs_init.device == feat.device
                and self._pending_obs_init.any()):
            h_boot = self.h_init_conv(e_t).to(dtype=feat.dtype)
            z_boot = self.z_init_conv(e_t).to(dtype=feat.dtype)
            self.h_state = torch.where(
                self._pending_obs_init[:, None, None, None], h_boot,
                self.h_state)
            self.z_state = torch.where(
                self._pending_obs_init[:, None, None, None], z_boot,
                self.z_state)
        self._pending_obs_init = None

        # ---- 1. motion-aware alignment (C1: zero-init Doppler offsets) --
        e_mot = ctx['e_mot'] if have_ctx else None
        h_aligned = self._align_with_motion(
            self.h_state, feat,
            self.align_h_offset_mask, self.align_h_deform_conv,
            self.motion_offset_h if self.use_motion_offsets else None,
            e_mot)
        if self.align_z_state:
            z_aligned = self._align_with_motion(
                self.z_state, feat,
                self.align_z_offset_mask, self.align_z_deform_conv,
                self.motion_offset_z
                if (self.use_motion_offsets and self.motion_offset_z
                    is not None) else None,
                e_mot)
        else:
            z_aligned = self.z_state

        # ---- 2. deterministic transition (C3: dynamic-gated update) -----
        dyn_map = ctx['dyn'] if have_ctx else None
        if self.use_action:
            x = torch.cat([z_aligned, velocity_map], dim=1)
        else:
            x = z_aligned
        h_t = self._gru_with_dynamic_gate(x, h_aligned, dyn_map)
        if not self.training:
            # Probe stash (probe_posterior_source reads h_t for its
            # decomposition); the dynamic-gated GRU calls the cell's convs
            # directly, so a module hook on `transition` never fires.
            self._last_h_t = h_t.detach()
        else:
            self._last_h_t = None

        # ---- 3. prior: p(z_t | h_t) -------------------------------------
        mu_p = self.prior_mu(h_t)
        logstd_p = self.prior_logstd(h_t)

        # ---- 4. observation (modality-augmented, zero-init residual) ----
        if have_ctx:
            gated = torch.cat(
                [ctx['c_cam'] * ctx['e_cam'], ctx['c_mot'] * ctx['e_mot']],
                dim=1)
            e_obs = e_t + self.obs_proj(gated)
        else:
            e_obs = e_t
        if not self.training:
            # Exact e_obs reconstruction for out-of-forward probes
            # (see _posterior_observation); detached, no graph retained.
            self._last_obs_residual = (e_obs - e_t).detach()
        else:
            self._last_obs_residual = None

        # ---- 5. posterior ----------------------------------------------
        if self.posterior_mode == 'innovation':
            e_hat_t = self.obs_prior(h_t)
            obs_pred = F.smooth_l1_loss(e_hat_t, e_obs.detach())
            innovation = e_obs - e_hat_t.detach()
            posterior_input = torch.cat([h_t, innovation], dim=1)
            delta_mu = self.posterior_mu(posterior_input)
            mu_q = mu_p + delta_mu
            logstd_q = self.posterior_logstd(posterior_input)
        else:
            innovation = None
            obs_pred = None
            posterior_input = torch.cat([h_t, e_obs], dim=1)
            mu_q = self.posterior_mu(posterior_input)
            logstd_q = self.posterior_logstd(posterior_input)

        # ---- 6. latent selection + correct gate (C2, identity start) ----
        if use_posterior:
            z_sel = mu_q if deterministic else self.sample(mu_q, logstd_q)
        else:
            z_sel = mu_p if deterministic else self.sample(mu_p, logstd_p)

        gate_in = None
        if self.use_reliability_gain or self.use_gated_readout:
            gate_in = self._gate_input(ctx, innovation, feat)

        g_c = None
        if use_posterior and self.use_reliability_gain and \
                gate_in is not None:
            g_c_dev = self._correct_gain_dev(gate_in)
            g_c = 1.0 + g_c_dev
            # Deviation form: dev == 0 at init -> z_t == z_sel bitwise.
            z_t = z_sel + g_c_dev * (z_sel - mu_p)
        else:
            z_t = z_sel

        # Probe-only counterfactual point: everything below (KL inputs are
        # mu_q/mu_p, unaffected; decoder, readout, state write) consumes the
        # intervened latent, i.e. exactly what the head would see.
        if self._z_intervention is not None:
            z_t = self._z_intervention(z_t)

        # ---- 7. KL (inherited, incl. free-bits + stats) ------------------
        kl, stats = self.kl_loss(mu_q, logstd_q, mu_p, logstd_p)
        if obs_pred is not None:
            stats['stat_obs_pred'] = obs_pred.detach()
            stats['stat_innovation_sq'] = \
                innovation.square().mean().detach()
        if g_c is not None:
            stats['stat_gain_mean'] = g_c.mean().detach()
            stats['stat_gain_std'] = g_c.std().detach()
            stats['stat_gain_min'] = g_c.min().detach()
            stats['stat_gain_max'] = g_c.max().detach()
        if have_ctx:
            with torch.no_grad():
                stats['stat_conf_cam_mean'] = ctx['c_cam'].mean().detach()
                stats['stat_conf_mot_mean'] = ctx['c_mot'].mean().detach()
                if ctx['dyn'] is not None:
                    stats['stat_dyn_ratio'] = \
                        (ctx['dyn'] > 0.5).float().mean().detach()
                stats['stat_speed_mean'] = \
                    ctx['vel'][:, 3:4].mean().detach()

        # ---- 8. reconstruction + gated residual readout (P1 fix) --------
        reconstruction = self.decoder(torch.cat([h_t, z_t], dim=1))
        residual = self.output_proj(z_t)
        g_r = None
        if self.use_gated_readout and gate_in is not None:
            g_r_dev = self._readout_gain_dev(gate_in)
            g_r = 1.0 + g_r_dev
            # Deviation form: dev == 0 at init -> output == feat + residual
            # bitwise, i.e. exactly the MAI readout.
            output = feat + residual + g_r_dev * residual
        else:
            output = feat + residual
        if g_r is not None:
            stats['stat_readout_gain_mean'] = g_r.mean().detach()
            stats['stat_readout_gain_std'] = g_r.std().detach()
        if not self.training and getattr(self, '_stash_gate_maps', False):
            # Probe stash (probe_gate_spatial.py): per-pixel gate inputs and
            # gate maps of the current frame, all detached. Enabled only by
            # the probe; normal training/eval never touches this branch.
            self._last_gate_maps = dict(
                g_c=g_c.detach() if g_c is not None else None,
                g_r=g_r.detach() if g_r is not None else None,
                c_cam=ctx['c_cam'].detach() if have_ctx else None,
                c_mot=ctx['c_mot'].detach() if have_ctx else None,
                dyn=(ctx['dyn'].detach() if have_ctx
                     and ctx['dyn'] is not None else None),
                innov_mag=(self.innovation_mag_conv(
                    innovation.detach().abs()).detach()
                    if getattr(self, 'innovation_mag_conv', None) is not None
                    and innovation is not None else None),
            )

        # ---- 9. state storage (inherited detach semantics) ---------------
        if detach_state:
            self.h_state = h_t.detach()
            self.z_state = z_t.detach()
        else:
            self.h_state = h_t
            self.z_state = z_t

        latent_loss = kl * self.kl_scale
        if obs_pred is not None and self.obs_pred_loss_weight > 0:
            latent_loss = latent_loss + self.obs_pred_loss_weight * obs_pred

        return output, reconstruction, latent_loss, h_t, z_t, stats
