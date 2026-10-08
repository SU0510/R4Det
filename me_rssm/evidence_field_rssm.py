"""DC-RSSM-1: physical evidence field replacing the learned gate inputs.

Design: me_rssm/docs/13_dc_rssm.md L3 (step 1 of the DC-RSSM chain). This
file only ADDS a new class to the me_rssm research package; no existing
file is modified.

Why this class exists
---------------------
The MCI-v2 ep16 verdict (training_runs_full.md §68.9e) closed the loop on
where the gates' spatial blindness comes from: every gate input the
current family feeds the gate convs is either LEARNED FROM DETECTION
GRADIENTS (c_cam/c_mot collapse to spatial constants, §68.9b) or derived
from a collapse-prone path (innovation magnitude: 94.6% batch-common
energy at ep16). Detection gradients demonstrably do not produce spatial
structure -- so the structure must come from the measurement, by
construction.

L3 replaces the learned confidence maps c_mot (and optionally dyn) with a
PARAMETER-FREE physical evidence field computed directly from the raw
Doppler scatter canvas ('velocity_bev', channels documented in
radar_motion_chain: ch2 v_mean, ch3 |v_mean|, ch4 v_std, ch5 SNR mean,
ch6 normalized occupancy):

    c_mot0 = sigmoid(alpha1 * log1p(count) * SNR - beta1)
    dyn0   = sigmoid(alpha2 * v_std * count   - beta2)
    cov    = count / log1p(M)

        count here is the canvas's already-normalized occupancy (ch6, in
        [0, 1] -- log1p(fill_ratio)/log(2), see radar_motion_chain); the
        docs/13 formula's log1p(count) reads that normalized value, so no
        second squashing is applied. SNR and v_std are used raw.
        alpha1/beta1/alpha2/beta2 are
FIXED calibration constants (constructor args, calibrated once from val
quantiles -- never trained), so the field is a pure function of the
measurement: its spatial structure is guaranteed by construction, which
is exactly the property the learned gates lacked.

The learned refinement is zero-init: ``c_mot = c_mot0 + refine_conv(
[c_mot0, dyn0, cov])`` -- learning may only correct the physical layer,
never replace it (docs/13 §0 unified principle). The refinement conv and
its zero-init follow the deadlock discipline (§4): a zero-init conv whose
input is non-zero (the physical field) receives gradient fine; the
identity start (c_mot == c_mot0 at init) is exact.

coverage semantics (Ped dead-zone answer, pre-registered): cov ~ 0 means
"no measurement here", NOT "static / low-reliability". cov enters the
gate input as its own channel so the gate convs can learn camera-led
routing for unmeasured pixels; the learned refinement also sees it.

Gates keep their MCI-v2 forms (deviation identity start, 1 + tanh(.)).
The bus-empty fallback replaces the physical c_mot with a cov=0 constant
(the "no measurement" semantic) instead of the learned-map zero, keeping
the gate input width constant.

Switch ``use_physical_evidence``: False reproduces MCI-v2 exactly (same
class hierarchy, no new parameters constructed, checkpoints load
unchanged). True changes the gate-input semantics -- this is a NEW DESIGN
GENERATION per docs/13 §4: the identity baseline is the physical field,
and comparisons are pre-registered as a generational gate (>= baseline
39.2809 AND >= v2 - 0.5 AND Ped paired delta >= -1.5), not a bitwise
equivalence.

Interface contract (unchanged from MCI):
    forward(feat, velocity=None, use_posterior=True, deterministic=True,
            detach_state=True)
        -> (output, reconstruction, latent_loss, h_t, z_t, stats)
reset_state() / reset_for_samples(mask) / kl_scale mutation / probe hooks
(_z_intervention, _last_h_t, _last_obs_residual, _stash_gate_maps) are
all inherited unchanged.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet3d.models.builder import FUSION_LAYERS

from .mci_rssm import MeasurementConditionedInnovationRSSMFusion
from .modality_bus import ModalityContext

__all__ = ['EvidenceFieldInnovationRSSMFusion']

# velocity_bev canvas channels (radar_motion_chain MOTION_STAT_CHANNELS):
# 0 vx, 1 vy, 2 v_mean, 3 |v_mean|, 4 v_std, 5 snr_mean, 6 norm occupancy.
_CH_V_MEAN = 2
_CH_SPEED = 3
_CH_V_STD = 4
_CH_SNR = 5
_CH_COUNT = 6


@FUSION_LAYERS.register_module()
class EvidenceFieldInnovationRSSMFusion(
        MeasurementConditionedInnovationRSSMFusion):
    """MCI-v2 + L3 physical evidence field (DC-RSSM step 1).

    New args (all MCI args are inherited unchanged):

    Args:
        use_physical_evidence (bool): build the L3 field and feed it to
            the gates (c_mot -> physical, dyn -> physical) plus a
            zero-init learned refinement. False reproduces MCI-v2.
        evidence_refine (bool): zero-init 1x1 refinement
            ``c_mot = c_mot0 + refine_conv([c_mot0, dyn0, cov])``.
            Requires use_physical_evidence.
        c_mot_alpha1 / c_mot_beta1 / dyn_alpha2 / dyn_beta2 (float):
            fixed calibration constants of the physical field (sigmoid
            argument scale and shift; calibrated once from val quantiles,
            never trained).
        cov_floor (float): value the bus-empty fallback uses for cov
            (0 = "no measurement", the Ped dead-zone semantic).
    """

    def __init__(
        self,
        use_physical_evidence=True,
        evidence_refine=True,
        c_mot_alpha1=1.0,
        c_mot_beta1=0.0,
        dyn_alpha2=1.0,
        dyn_beta2=0.0,
        cov_floor=0.0,
        **kwargs,
    ):
        if not use_physical_evidence and evidence_refine:
            raise ValueError(
                'evidence_refine requires use_physical_evidence=True')
        if cov_floor < 0 or cov_floor > 1:
            raise ValueError('cov_floor must be in [0, 1]')
        self.use_physical_evidence = use_physical_evidence
        self.evidence_refine = evidence_refine
        self.c_mot_alpha1 = c_mot_alpha1
        self.c_mot_beta1 = c_mot_beta1
        self.dyn_alpha2 = dyn_alpha2
        self.dyn_beta2 = dyn_beta2
        self.cov_floor = cov_floor
        super().__init__(**kwargs)
        if use_physical_evidence and evidence_refine:
            self.evidence_refine_conv = nn.Conv2d(3, 1, 1)
            # Init per the section-4 deadlock rule + the ADDITIVE position
            # of this conv (4th instance of the rule, caught by the unit
            # tests): its output feeds the ZERO-INIT gate convs, so a
            # zero-init refine conv would have identically zero gradient
            # forever (the 68.8i innovation_mag_conv pattern) -- output
            # must be non-zero. But unlike innovation_mag_conv (whose
            # output is re-mapped through the gates' tanh), this residual
            # ADDS DIRECTLY to c_mot0 in (0, 1), so Xavier's ~1.2-scale
            # weights would swamp the physical field at init (measured in
            # tests: c_mot left (0, 1), spatial structure destroyed).
            # Small-normal std=0.01 keeps the output non-zero (~0.008
            # mean) AND the field measurement-anchored; the gates' identity
            # start (dev == tanh(0) == 0) is carried by the gate convs.
            nn.init.normal_(self.evidence_refine_conv.weight, std=0.01)
            nn.init.zeros_(self.evidence_refine_conv.bias)
        self._evidence_field = None

    # ------------------------------------------------------------------
    # L3 physical evidence field (parameter-free)
    # ------------------------------------------------------------------
    def _physical_evidence(self, vel_canvas, feat):
        """Compute (c_mot0, dyn0, cov) from the raw Doppler canvas.

        vel_canvas: (B, 7, H, W) post-avgpool/post-permute Doppler canvas
        (the same tensor MCI's _read_context produces). All outputs are
        detached -- the field is a measurement, not a learned path.
        """
        count = vel_canvas[:, _CH_COUNT:_CH_COUNT + 1]
        snr = vel_canvas[:, _CH_SNR:_CH_SNR + 1]
        v_std = vel_canvas[:, _CH_V_STD:_CH_V_STD + 1]
        c_mot0 = torch.sigmoid(
            self.c_mot_alpha1 * count * snr - self.c_mot_beta1)
        dyn0 = torch.sigmoid(
            self.dyn_alpha2 * v_std * count - self.dyn_beta2)
        cov = count / math.log(2.0)
        return c_mot0.detach(), dyn0.detach(), cov.detach()

    def _read_context(self, feat):
        """MCI context + physical evidence override.

        When use_physical_evidence is on and the bus is alive, c_mot and
        dyn come from the physical field with a zero-init learned
        refinement. Gradient semantics split the two layers: the physical
        field is a measurement (detached -- no detection gradients into
        the canvas math), the refinement is a learned path (KEEP its
        graph; detaching the sum would make the refinement conv a dead
        layer -- the exact section-38.10 pattern). The learned conf_mot /
        dyn_conv still run so their parameters stay used for DDP and the
        switch-off rollback keeps their weights.

        When the bus is dead, this override keeps MCI's ctx=None fallback
        flow untouched; the cov_floor "no measurement" constant is
        substituted at the gate input (see _gate_input), which is where
        the semantic belongs.
        """
        ctx = super()._read_context(feat)
        if not self.use_physical_evidence:
            return ctx
        B = feat.shape[0]
        vel_raw = ModalityContext.get('velocity_bev')
        have_bus = (vel_raw is not None
                    and vel_raw.shape[0] == B
                    and vel_raw.shape[1] == self.motion_channels)
        if not have_bus:
            self._evidence_field = None
            return ctx

        vel = F.avg_pool2d(vel_raw, 2)
        vel = vel.permute(0, 1, 3, 2).contiguous()
        c_mot0, dyn0, cov = self._physical_evidence(vel, feat)
        if self.evidence_refine:
            # refinement keeps its graph: learned correction of the
            # measurement. Xavier-init (see __init__ deadlock note); the
            # physical layer itself stays detached (measurement).
            c_mot = c_mot0 + self.evidence_refine_conv(
                torch.cat([c_mot0, dyn0, cov], dim=1))
        else:
            c_mot = c_mot0
        dyn = dyn0
        if ctx is None:
            # bus alive for velocity but cam missing/inconsistent: MCI
            # returned None (its own all-or-nothing check). Keep that
            # contract; the field is still recorded for stats.
            self._evidence_field = dict(c_mot0=c_mot0, dyn0=dyn0, cov=cov)
            return ctx
        ctx = dict(ctx)
        ctx['c_mot'] = c_mot
        ctx['dyn'] = dyn
        self._evidence_field = dict(c_mot0=c_mot0, dyn0=dyn0, cov=cov)
        return ctx

    def _gate_input(self, ctx, innovation, feat):
        """MCI gate input + cov_floor fallback for the physical channel.

        The bus-empty fallback replaces the physical c_mot with the
        cov_floor constant ("no measurement", the Ped dead-zone semantic)
        instead of a plain zero, keeping the gate input width constant.
        """
        if not self.use_physical_evidence:
            return super()._gate_input(ctx, innovation, feat)
        if ctx is not None:
            return super()._gate_input(ctx, innovation, feat)
        if not (self.use_reliability_gain or self.use_gated_readout):
            return None
        B, _, H, W = feat.shape
        parts = []
        if self.use_modality_obs:
            zeros = feat.new_zeros(B, 1, H, W)
            parts.append(zeros)
            parts.append(
                feat.new_full((B, 1, H, W), self.cov_floor))
        if getattr(self, 'innovation_mag_conv', None) is not None and \
                innovation is not None:
            parts.append(self.innovation_mag_conv(innovation.detach().abs()))
        if not parts:
            return None
        return torch.cat(parts, dim=1)

    # ------------------------------------------------------------------
    # forward: same signature and 6-tuple return as MCI, + field stats
    # ------------------------------------------------------------------
    def forward(self, feat, velocity=None, use_posterior=True,
                deterministic=True, detach_state=True):
        out = super().forward(
            feat, velocity=velocity, use_posterior=use_posterior,
            deterministic=deterministic, detach_state=detach_state)
        stats = out[5]
        field = self._evidence_field
        if field is not None:
            stats['stat_ev_c_mot_mean'] = field['c_mot0'].mean().detach()
            stats['stat_ev_c_mot_std'] = field['c_mot0'].std().detach()
            stats['stat_ev_dyn_mean'] = field['dyn0'].mean().detach()
            stats['stat_ev_dyn_std'] = field['dyn0'].std().detach()
            stats['stat_ev_cov_mean'] = field['cov'].mean().detach()
        return out
