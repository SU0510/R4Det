# Ablation C-off: disable the Kalman-like reliability gain. The latent is
# always the selected posterior value (z_t = z_sel), i.e. the filter always
# trusts the observation and the prior never acts as a fallback. This is
# the direct test of the "prior as prediction" contribution.
_base_ = ['./TJ4D-R4Det_me_rssm_N4_24e_pretrained_v2_head.py']

model = dict(
    temporal_fusion=dict(
        use_reliability_gain=False,
    ),
)
