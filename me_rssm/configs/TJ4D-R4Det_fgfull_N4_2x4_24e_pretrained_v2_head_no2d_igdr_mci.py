# Measurement-Conditioned Innovation RSSM (MCI-RSSM) on the no2d_igdr/MAI
# main line. Prepared 2026-10-02; design + runbook: me_rssm/docs/12_mci_rssm.md
# and docs/rssm_review_20261002.md §5A/5B. NOT yet trained — every launch is
# a user-approved operation (see docs/12 §runbook).
#
# Why this config exists
# ----------------------
# MAI fixed the posterior's information flow (h/e dependence 2.23 -> <1,
# innovation specific energy 47-67%) but seed0 +0.84 / seed1 -2.51, and the
# z -> head causal effect stayed bias-dominated (zero_z 26-56% vs shuffle_z
# 2.7-8.4%). MCI-RSSM merges the MAI innovation posterior with the ME-RSSM
# Doppler mechanisms (motion-conditioned offsets, dynamic update gate,
# modality observations + reliability) and adds the missing link: a
# reliability-gated READOUT (output = feat + g_r * output_proj(z_t)).
#
# Identity-start contract (me_rssm/docs/12 §equivalence): every added
# pathway is zero-init and both gates use the deviation form
# (z = z_sel + tanh(.) * (z_sel - mu_p), output = feat + res + tanh(.) * res),
# so with MAI weights loaded the model is BITWISE identical to MAI in all
# three regimes: empty bus / switches off / all on at init. MAI is therefore
# both the fine-tuning source and the paired baseline; the seven *_abl_*
# configs isolate each switch as a single variable.
#
# Changes relative to ..._no2d_igdr_mai (the direct base):
#   pts_voxel_encoder : -> RadarPillarFeatureNetMotion   (adds raw Doppler
#                        pillar stats; pretrained keys load unchanged)
#   pts_middle_encoder: -> PointPillarsScatterMotion     (same scatter +
#                        publishes the velocity-evidence canvas on the bus)
#   RCFusion          : -> ConcatConvFusionStash         (identical fusion +
#                        stashes cam/radar BEVs on the bus)
#   temporal_fusion   : -> MeasurementConditionedInnovationRSSMFusion
#
# DDP constraint (me_rssm/docs/06, carried over): training with
# use_modality_obs=True REQUIRES the full producer chain above; with an
# empty bus the bus-consuming modules never run and DDP aborts on unused
# parameters. The producer chain is part of this config.

import os
import sys

# Bootstrap: make the repo root importable so `me_rssm` resolves no matter
# which working directory mmcv's temp-dir config import uses.
try:
    _cfg_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:  # pragma: no cover
    _cfg_dir = os.path.join(os.getcwd(), 'me_rssm', 'configs')
_repo_root = os.path.abspath(os.path.join(_cfg_dir, '..', '..'))
if not os.path.isdir(os.path.join(_repo_root, 'me_rssm')):
    _p = os.getcwd()
    for _ in range(5):
        if os.path.isdir(os.path.join(_p, 'me_rssm')):
            _repo_root = _p
            break
        _p = os.path.dirname(_p)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

# Register me_rssm modules at config-exec time: tools/test_vod.py (the eval
# entry) does not process custom_imports.
import me_rssm  # noqa: E402,F401

_base_ = (
    '../../configs/r4det/'
    'TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_mai.py'
)

custom_imports = dict(imports=[
    'mmdet3d',
    'mmdet3d.core.hook.kl_scale_scheduler',
    'me_rssm',
])

model = dict(
    pts_voxel_encoder=dict(
        type='RadarPillarFeatureNetMotion',
    ),
    pts_middle_encoder=dict(
        type='PointPillarsScatterMotion',
    ),
    RCFusion=dict(
        type='ConcatConvFusionStash',
    ),
    temporal_fusion=dict(
        type='MeasurementConditionedInnovationRSSMFusion',
        # --- MAI-inherited args (identical to ..._mai.py) ----------------
        in_channels=256,
        out_channels=256,
        kernel_size=3,
        latent_dim=256,
        hidden_dim=128,
        action_dim=0,
        kl_scale=1.0,
        free_nats=1.0,
        min_std=0.1,
        init_std=0.2,
        obs_pred_loss_weight=0.05,
        align_kernel_size=3,
        align_deform_groups=1,
        align_z_state=True,
        norm_cfg=dict(type='BN', requires_grad=True),
        act_cfg=dict(type='ReLU', inplace=True),
        # --- MCI switches (all identity-start; each has an *_abl_* config)
        posterior_mode='innovation',
        innov_in_gain=True,
        use_motion_offsets=True,
        use_dynamic_gate=True,
        use_reliability_gain=True,
        use_gated_readout=True,
        use_modality_obs=True,
        # --- bus consumer hyper-parameters -------------------------------
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
    ),
)
