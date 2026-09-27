# Posterior-innovation repair with a batch-shuffle discrimination term.
#
# Base: lowdim_z_delta. The 450-iter gate showed the delta reconstruction target
# alone still admits a near-common correction: recon fell only 11.1% and
# correction^2 slid to 0.0039 with no plateau. The reconstruction term does not
# require the correction to identify its own sample, so this variant adds a
# margin ranking loss that scores a sample's own correction better than a
# batch-rolled one. KL, free-nats, latent size and the schedule are unchanged.

_base_ = (
    './TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_'
    'no2d_igdr_lowdim_z_delta.py'
)

model = dict(
    temporal_fusion=dict(
        discr_loss_weight=0.1,
        discr_margin=0.2,
    ),
)
