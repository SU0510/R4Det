# Route B: Innovation-Conditioned RSSM (Kalman-style predict/correct).
#
# Base: ..._lowdim_z_delta_spatial_raw_xavier (route-A cell 4/4, the best cell
# of the 2x2 matrix: e_pooled sample-specific 14.19%, correction 12.11%).
#
# Route A separated the two bottlenecks but showed that neither the input
# normalization nor the readout gain changes the shuffle/zero ratio (0.036-0.046
# in all four cells). The correction was still `mu_q - mu_p`, a difference of
# two batch-common vectors with a small absolute magnitude and an unconstrained
# direction. This config replaces that construction with an explicit
# prediction/correction split:
#
#   e_hat_t      = obs_prior(h_t)                    # trained by L_obs_pred only
#   innovation_t = e_pooled - stopgrad(e_hat_t)
#   delta_mu     = posterior_delta(cat([h_t, innovation_t]))
#   mu_q         = mu_p + delta_mu
#   correction_t = delta_mu                          # not a difference
#
# `z_prev` keeps its L2 normalization: per the route plan that ablation is
# deliberately deferred so this config changes one thing (the posterior
# construction) rather than two.

_base_ = (
    './TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_'
    'no2d_igdr_lowdim_z_delta_spatial_raw_xavier.py'
)

model = dict(
    temporal_fusion=dict(
        posterior_struct='innovation',
        obs_pred_loss_weight=0.05,
    ),
)
