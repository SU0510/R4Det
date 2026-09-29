# Motion-Aligned Innovation RSSM (MAI) on the no2d_igdr main line.
#
# Base: ..._no2d_igdr (the full-resolution MotionAlignedRSSMFusion baseline,
# best saved ep14 = 40.3853, ep8-12 mean 38.8906 over its three seeds).
#
# Why this config exists
# ----------------------
# Route C (object-centric supervision on a 16x16x32 latent) stopped at ep12
# with AP 34.9091 vs baseline 39.2953 (-4.3862). Section 61 of
# docs/training_runs_full.md established that this gap cannot be attributed to
# the posterior alone: the low-dim replacement had simultaneously removed
# deformable motion alignment, native 216x248 resolution, and the
# full-resolution readout. The measured mechanism, however, did improve:
# shuffle_z head reached 2.4290% on the near window and the h/e dependence
# ratio turned toward the observation for the first time (1.456 -> 1.275).
#
# This config therefore re-runs the innovation posterior on the *unmodified*
# main line. The ONLY change relative to ..._no2d_igdr is the fusion class and
# its observation-prediction weight:
#
#   temporal_fusion.type: MotionAlignedRSSMFusion
#                     -> MotionAlignedInnovationRSSMFusion
#   obs_pred_loss_weight = 0.05
#
# Everything else is inherited unchanged: latent_dim=256 (full resolution,
# 216x248), ConvGRU transition, deformable h/z alignment, free_nats=1.0,
# kl_scale warm-up 0.0 -> 1.0 over ep0-10, reconstruction + KL losses,
# output_proj residual readout, and the 24-epoch cosine schedule.
#
# Explicitly NOT changed in this first round (see section 61.8):
#   - no object heatmap auxiliary loss
#   - no KL balancing / free-bits rework
#   - no readout changes
#   - no latent down-dimensioning
#
# That keeps the experiment a single-variable test of whether the innovation
# posterior helps while the main-line AP is preserved.

_base_ = (
    './TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr.py'
)

model = dict(
    temporal_fusion=dict(
        type='MotionAlignedInnovationRSSMFusion',
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
    ),
)
