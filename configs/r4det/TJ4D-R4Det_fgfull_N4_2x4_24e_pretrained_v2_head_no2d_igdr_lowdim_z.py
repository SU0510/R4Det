# ---------------------------------------------------------------------------
# Low-dimensional future-consistent z ablation.
#
# Base: no2d_igdr. Replaces the full-resolution pixel-wise RSSM with a
# 16x16x32 recurrent latent, residual FiLM/gating, and a pending next-frame
# BEV consistency task. Same backbone, detector head, data, schedule, and
# no2d_igdr ablation scope.
#
# Standard-RSSM structure restoration (2026-09-26): the latent now conditions
# on h_t (p(z_t|h_t), q(z_t|h_t,e_t)) and carries an observation-likelihood
# term (decoder(h_t,z_t) -> pooled normalized BEV, weight 0.1). The previous
# run without either anchor collapsed z into a near-constant common bias.
# ---------------------------------------------------------------------------

_base_ = './TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr.py'

model = dict(
    temporal_fusion=dict(
        _delete_=True,
        type='LowDimFutureConsistentLatentFusion',
        in_channels=256,
        latent_dim=32,
        hidden_dim=128,
        action_dim=0,
        kl_scale=1.0,
        free_nats=0.1,
        min_std=0.1,
        init_std=0.2,
        latent_pool='adaptive',
        latent_size=(16, 16),
        predict_future_channels=256,
        future_loss_weight=0.1,
        recon_loss_weight=0.1,
        modulation_scale=0.1,
        gate_init_bias=-1.0,
    ),
)

custom_hooks = [
    dict(
        type='KLScaleSchedulerHook',
        start_epoch=0,
        end_epoch=12,
        start_value=0.01,
        end_value=1.0,
    ),
]
