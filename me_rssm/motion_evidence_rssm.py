"""Motion-Evidence RSSM (ME-RSSM) for Camera + 4D Radar BEV perception.

Motivation (short form; see me_rssm/docs/04_design_me_rssm.md)
-------------------------------------------------------------
The baseline ``MotionAlignedRSSMFusion`` keeps a probabilistic state
(h, z) but receives a single *fused appearance* observation: the radar
Doppler signal -- the only direct motion measurement available (TJ4D has
no ego pose, no timestamps) -- never reaches the temporal dynamics, the
prior p(z|h) has no predictive content (posterior collapse is the
measured steady state), and no mechanism decides how much the filter
should trust the current observation.

ME-RSSM turns the RSSM into a radar-informed predict-correct filter
while keeping every baseline tensor interface and loss contract:

  Predict   the deterministic transition h_t = GRU(align(z), align(h))
            is conditioned on a raw Doppler *motion-evidence* map
            (pseudo-2D velocity field from radial velocity + beam
            geometry): (a) the deformable state alignment offsets get a
            zero-init additive motion term; (b) the GRU update gate is
            modulated by a dynamic-evidence map so fast-changing regions
            update faster and static background persists longer
            (u' = u + beta * d * u * (1-u), beta init 0).

  Correct   modality observations e_cam (camera BEV) and e_mot (Doppler
            evidence) with learned per-pixel reliability maps c_cam /
            c_mot enter the posterior through a *zero-init residual*
            proj into e_fused -- posterior conv shapes stay identical to
            the baseline, so baseline weights load unchanged. A learned
            Kalman-like gain g (init bias high => g ~ 1) mixes the
            selected posterior latent with the prior mean:

                z_t = mu_p + g * (z_sel - mu_p)

            so pixels with unreliable observations fall back to the
            motion-propagated prediction instead of trusting garbage.

Empty-bus guarantee: when the side-channel context is absent (or every
switch is disabled) the module reduces EXACTLY to
``MotionAlignedRSSMFusion`` -- identical parameter names, identical math,
bitwise-equal outputs given equal weights. This makes it simultaneously
(a) a drop-in replacement, (b) its own ablation baseline, and (c)
fine-tunable from any trained baseline RSSM checkpoint.

Interface contract with the unmodified detector (R4Det.extract_feat):
    forward(feat, velocity=None, use_posterior=True, deterministic=True,
            detach_state=True)
        -> (output, reconstruction, kl*kl_scale, h_t, z_t, stats)
    reset_state() / reset_for_samples(mask)
    attribute kl_scale mutated by KLScaleSchedulerHook
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import auto_fp16
from mmcv.cnn import ConvModule, xavier_init

from mmdet3d.models.builder import FUSION_LAYERS
from mmdet3d.models.fusion_layers.rssm_fusion import MotionAlignedRSSMFusion

from .modality_bus import ModalityContext

__all__ = ['MotionEvidenceRSSMFusion']


@FUSION_LAYERS.register_module()
class MotionEvidenceRSSMFusion(MotionAlignedRSSMFusion):
    """RSSM with Doppler motion evidence in the transition and a
    reliability-gated observation model (see module docstring).

    New args (everything else inherited from MotionAlignedRSSMFusion):

    Args:
        motion_channels (int): channels of the stashed velocity-evidence
            canvas ('velocity_bev'), default 7 (see radar_motion_chain).
        camera_bev_channels (int): channels of 'camera_bev' (default 256).
        motion_hidden (int): hidden width of the motion-evidence encoder.
        motion_latent (int): output width of the motion-evidence encoder;
            also the input width of the motion-offset convolutions.
        camera_hidden (int): hidden width of the camera-observation encoder.
        camera_latent (int): output width of the camera-observation encoder.
        use_motion_offsets (bool): add motion-conditioned term to the
            deformable state-alignment offsets (ablation switch A).
        use_reliability_gain (bool): enable the Kalman-like gain mixing
            between posterior latent and prior mean (ablation switch C).
        use_dynamic_gate (bool): enable dynamic-evidence modulation of the
            GRU update gate (ablation switch D).
        use_modality_obs (bool): master switch for consuming the side
            channel (modality observations + confidences). False makes the
            module behave exactly like the baseline regardless of the bus.
        motion_grad (bool): if True, gradients may flow from the detection
            loss through the motion offsets into the velocity-evidence
            canvas. Default False: the Doppler map is a *measurement*,
            treated as a detached control signal.
        modality_dropout (float): training-time probability of zeroing the
            camera (resp. motion) confidence maps per sample, forcing the
            filter to practise prior fallback. 0 disables.
        gain_init_bias (float): bias of the gain conv; sigmoid(4)~0.982 so
            training starts near the posterior-only baseline behaviour.
        conf_init_bias (float): bias of the confidence convs; sigmoid(2)
            ~0.88 at init.
        dyn_init_bias (float): bias of the dynamic-evidence conv; sigmoid
            (-2) ~0.12, i.e. most cells start "mostly static".
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        latent_dim=None,
        hidden_dim=128,
        action_dim=0,
        kl_scale=1.0,
        free_nats=1.0,
        min_std=0.1,
        init_std=0.2,
        norm_cfg=dict(type='BN', requires_grad=True),
        act_cfg=dict(type='ReLU', inplace=True),
        align_kernel_size=3,
        align_deform_groups=1,
        align_z_state=True,
        motion_channels=7,
        camera_bev_channels=256,
        motion_hidden=32,
        motion_latent=32,
        camera_hidden=128,
        camera_latent=64,
        use_motion_offsets=True,
        use_reliability_gain=True,
        use_dynamic_gate=True,
        use_modality_obs=True,
        motion_grad=False,
        modality_dropout=0.0,
        gain_init_bias=4.0,
        conf_init_bias=2.0,
        dyn_init_bias=-2.0,
        init_cfg=None,
    ):
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
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            align_kernel_size=align_kernel_size,
            align_deform_groups=align_deform_groups,
            align_z_state=align_z_state,
            init_cfg=init_cfg,
        )

        self.motion_channels = motion_channels
        self.camera_bev_channels = camera_bev_channels
        self.use_motion_offsets = use_motion_offsets
        self.use_reliability_gain = use_reliability_gain
        self.use_dynamic_gate = use_dynamic_gate
        self.use_modality_obs = use_modality_obs
        self.motion_grad = motion_grad
        self.modality_dropout = modality_dropout
        self.gain_init_bias = gain_init_bias
        self.conf_init_bias = conf_init_bias
        self.dyn_init_bias = dyn_init_bias

        # ---- observation encoders (modality-specific) -------------------
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
        # Zero-init residual into e_fused: posterior conv shapes and the
        # whole downstream graph stay baseline-identical until trained.
        self.obs_proj = nn.Conv2d(motion_latent + camera_latent, in_channels,
                                  1)

        # ---- per-modality reliability ----------------------------------
        self.conf_cam = nn.Conv2d(camera_latent, 1, 1)
        self.conf_mot = nn.Conv2d(motion_latent, 1, 1)

        # ---- Kalman-like gain (posterior vs prior trust) ----------------
        self.gain_conv = nn.Conv2d(2, 1, 3, padding=1)

        # ---- dynamic-evidence map for the update gate -------------------
        self.dyn_conv = nn.Conv2d(motion_channels, 1, 1)
        self.dyn_beta = nn.Parameter(torch.tensor(0.0))

        # ---- motion-conditioned alignment offsets -----------------------
        k2 = align_kernel_size * align_kernel_size
        self.motion_offset_channels = 2 * align_deform_groups * k2
        self.motion_offset_h = nn.Conv2d(
            motion_latent, self.motion_offset_channels, align_kernel_size,
            padding=align_kernel_size // 2)
        if align_z_state:
            self.motion_offset_z = nn.Conv2d(
                motion_latent, self.motion_offset_channels, align_kernel_size,
                padding=align_kernel_size // 2)
        else:
            self.motion_offset_z = None

        self._init_me_weights()

    # ------------------------------------------------------------------
    # init
    # ------------------------------------------------------------------
    def _init_me_weights(self):
        """Identity-start init for every new pathway.

        Zero-init: obs_proj, gain_conv, dyn_conv, motion_offset_{h,z} and
        the confidence convs. Biases: gain bias high (trust observation,
        baseline behaviour), confidence biases moderately positive,
        dynamic bias negative (mostly static), dyn_beta exactly 0.
        Re-running is idempotent (constants, not distributions).
        """
        for m in [self.obs_proj, self.gain_conv, self.dyn_conv,
                  self.motion_offset_h, self.conf_cam, self.conf_mot]:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        if self.motion_offset_z is not None:
            nn.init.zeros_(self.motion_offset_z.weight)
            nn.init.zeros_(self.motion_offset_z.bias)
        with torch.no_grad():
            self.gain_conv.bias.fill_(self.gain_init_bias)
            self.conf_cam.bias.fill_(self.conf_init_bias)
            self.conf_mot.bias.fill_(self.conf_init_bias)
            self.dyn_conv.bias.fill_(self.dyn_init_bias)
        self.dyn_beta.data.zero_()

    def init_weights(self):
        # Parent chain (BEVRSSMTemporalFusion) xavier-inits encoder/GRU/
        # mu/logstd/output_proj and zero-inits the alignment generators.
        # The parent __init__ calls init_weights() BEFORE our extra layers
        # exist, so guard; _init_me_weights() re-runs (idempotently) from
        # our own __init__ tail.
        super().init_weights()
        if hasattr(self, 'obs_proj'):
            self._init_me_weights()

    # ------------------------------------------------------------------
    # context
    # ------------------------------------------------------------------
    def _read_context(self, feat):
        """Fetch and normalize the per-frame modality context.

        Returns (vel_map, cam_map, e_mot, e_cam, c_cam, c_mot, dyn_map) or
        (None, ...) when the bus is empty / inconsistent / switched off.
        All returned tensors are detached unless ``motion_grad`` is set
        (the Doppler map is a measurement, not a trainable pathway).
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

        # scatter canvas rows = y (496), cols = x (432) at 0.16 m;
        # temporal-fusion canvas rows = x (216), cols = y (248).
        vel = F.avg_pool2d(vel_raw, 2)                    # (B,Cm,248,216)
        vel = vel.permute(0, 1, 3, 2).contiguous()        # (B,Cm,216,248)
        cam = cam_raw.permute(0, 1, 3, 2).contiguous()    # (B,C,216,248)
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

        dyn_map = torch.sigmoid(self.dyn_conv(vel))

        return dict(vel=vel, e_mot=e_mot, e_cam=e_cam, c_cam=c_cam,
                    c_mot=c_mot, dyn=dyn_map)

    # ------------------------------------------------------------------
    # building blocks
    # ------------------------------------------------------------------
    def _align_with_motion(self, state, feat, offset_mask_conv, deform_conv,
                           motion_conv=None, e_mot=None):
        """Baseline deformable alignment + additive motion offsets."""
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

        u' = u + beta * d * u * (1 - u), with beta = tanh(dyn_beta) in
        (-1, 1). For |beta| <= 1 the map u -> u + beta*d*u*(1-u) sends
        (0, 1) into (u^2, 1) -- the gate stays a valid probability for any
        learned beta, and beta=0 or dyn_map=None reproduces the parent
        cell exactly.
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

    # ------------------------------------------------------------------
    # forward -- same signature and 6-tuple return as the baseline
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
                velocity = torch.zeros(B, self.action_dim, device=feat.device)
            velocity_map = velocity[:, :, None, None].expand(
                B, self.action_dim, H, W)
        else:
            velocity_map = None

        # ---- state init (inherited semantics: zeros at sequence start) --
        if self.h_state is None or self.h_state.shape[0] != B:
            self.h_state = torch.zeros(
                B, self.channels, H, W, device=feat.device, dtype=feat.dtype)
            self.z_state = torch.zeros(
                B, self.latent_dim, H, W, device=feat.device,
                dtype=feat.dtype)

        # ---- 1. motion-aware alignment of the historical state ----------
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

        # ---- 2. deterministic transition with dynamic-gated update ------
        dyn_map = ctx['dyn'] if have_ctx else None
        if self.use_action:
            x = torch.cat([z_aligned, velocity_map], dim=1)
        else:
            x = z_aligned
        h_t = self._gru_with_dynamic_gate(x, h_aligned, dyn_map)

        # ---- 3. prior: p(z_t | h_t) -------------------------------------
        mu_p = self.prior_mu(h_t)
        logstd_p = self.prior_logstd(h_t)

        # ---- 4. observation: fused appearance + gated modality streams --
        e_fused = self.encoder(feat)
        if have_ctx:
            gated = torch.cat(
                [ctx['c_cam'] * ctx['e_cam'], ctx['c_mot'] * ctx['e_mot']],
                dim=1)
            e_fused = e_fused + self.obs_proj(gated)

        # ---- 5. posterior: q(z_t | h_t, o_t) ----------------------------
        mu_q = self.posterior_mu(torch.cat([h_t, e_fused], dim=1))
        logstd_q = self.posterior_logstd(torch.cat([h_t, e_fused], dim=1))

        # ---- 6. latent selection + Kalman-like gain ---------------------
        if use_posterior:
            z_sel = mu_q if deterministic else self.sample(mu_q, logstd_q)
        else:
            z_sel = mu_p if deterministic else self.sample(mu_p, logstd_p)
        if use_posterior and have_ctx and self.use_reliability_gain:
            g = torch.sigmoid(self.gain_conv(
                torch.cat([ctx['c_cam'], ctx['c_mot']], dim=1)))
            z_t = mu_p + g * (z_sel - mu_p)
        else:
            g = None
            z_t = z_sel

        # ---- 7. KL (inherited, incl. free-bits + stats) ------------------
        kl, stats = self.kl_loss(mu_q, logstd_q, mu_p, logstd_p)
        if have_ctx:
            with torch.no_grad():
                stats.update(
                    stat_gain_mean=g.mean().detach() if g is not None
                    else torch.ones((), device=feat.device),
                    stat_conf_cam_mean=ctx['c_cam'].mean().detach(),
                    stat_conf_mot_mean=ctx['c_mot'].mean().detach(),
                    stat_dyn_ratio=(ctx['dyn'] > 0.5).float().mean().detach(),
                    stat_speed_mean=ctx['vel'][:, 3:4].mean().detach(),
                )

        # ---- 8. reconstruction + residual output (inherited) ------------
        reconstruction = self.decoder(torch.cat([h_t, z_t], dim=1))
        output = self.output_proj(z_t)
        output = output + feat

        # ---- 9. state storage (inherited detach semantics) --------------
        if detach_state:
            self.h_state = h_t.detach()
            self.z_state = z_t.detach()
        else:
            self.h_state = h_t
            self.z_state = z_t

        return output, reconstruction, kl * self.kl_scale, h_t, z_t, stats
