# Route-B follow-up: `z_prev` L2-normalize ablation (pre-registered branch).
#
# Route-B ep6 gate result (docs/training_runs_full.md section 56):
#   * AP gap -2.53 vs no2d_igdr seed0 ep6  -> stop-loss tripped (> 2.0)
#   * pre-registered branch: before layering route C, run a 450-iter
#     single-variable ablation of the `z_prev` L2 normalization.
#
# Rationale from the ep6 probes: `mu_q` depended on h_t (removal rel-L2 0.259)
# more than on the current observation (0.165), and h_t itself had already
# become sample-specific (12.19%). Feeding a unit-norm `z_prev` into the GRU
# discards the recurrent state's magnitude, which is one candidate reason the
# iterate keeps collapsing toward a common bias.
#
# This config changes exactly one thing relative to the route-B config:
#   z_prev_normalize: True -> False
# Everything else (posterior_struct=innovation, obs_pred_loss_weight=0.05,
# scaled_raw posterior input, xavier spatial readout, KL schedule, latent size)
# is inherited unchanged.

_base_ = (
    './TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_'
    'no2d_igdr_lowdim_z_innovation.py'
)

model = dict(
    temporal_fusion=dict(
        z_prev_normalize=False,
    ),
)
