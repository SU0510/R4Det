# Route C: object-centric latent supervision on top of the rawnorm route-B
# control.
#
# Base: ..._lowdim_z_innovation_rawnorm (route-B innovation posterior with the
# `z_prev` L2 normalization removed). That config passed the ep6 AP gate at
# -1.13 vs no2d_igdr seed0 (inside the pre-registered 1-2 band) but its
# correction still collapsed monotonically (0.2218 -> 0.0685 over ep1-6), so
# the pre-registered branch is route C rather than a straight ep12 extension.
#
# This config changes exactly one thing relative to the rawnorm control:
#   latent_object_loss_weight: 0.0 -> 0.1
#
# The auxiliary head consumes only `posterior_correction` (16x16x32), never
# h_t or the pooled observation, and is trained with a CenterNet-style
# Gaussian focal loss on 4-class box-center heatmaps rasterized from the
# CURRENT frame's GT. History frames contribute no object term. The head is
# absent from inference: `_object_heatmap_loss` returns early unless
# `self.training`, and the detection path never reads the head.
#
# Weight history: the pre-registered first setting was 0.05. The 450-iter
# smoke (docs/training_runs_full.md section 59) showed the auxiliary head was
# barely moving: `object_head.0.weight` ended at RMS 0.04817 versus a Xavier
# init of 0.04811, the class bias only moved from -2.19 to -2.20, and
# `stat_object_hm` fell just 12.8% (2.2243 -> 1.9393, with a rebound over the
# last 50 iters). That is exactly the pre-registered "gradient too weak"
# condition, so this config uses the documented escalation 0.05 -> 0.1
# before any other mechanism is added.
#
# Note on effective weight: training averages the current frame's latent loss
# with the 1 BPTT history frame, so the object term (current frame only) is
# halved to ~0.05 per optimizer step.

_base_ = (
    './TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_'
    'no2d_igdr_lowdim_z_innovation_rawnorm.py'
)

model = dict(
    temporal_fusion=dict(
        latent_object_loss_weight=0.1,
        num_object_classes=4,
        object_head_hidden_dim=64,
        object_min_radius=1,
    ),
)
