# Low-dimensional posterior-innovation repair.
#
# The previous reconstruction target was 92.1% shared template and could be
# solved by a sample-invariant z. This variant reconstructs the standardized
# frame-to-frame BEV delta from mu_q - stopgrad(mu_p), then exposes that
# spatial correction directly to the detector. The ineffective one-channel
# future-energy task is disabled so the mechanism test has one coherent goal.

_base_ = (
    './TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_'
    'no2d_igdr_lowdim_z.py'
)

model = dict(
    temporal_fusion=dict(
        recon_target_mode='temporal_delta',
        recon_input_mode='posterior_correction',
        direct_correction_readout=True,
        future_loss_weight=0.0,
    ),
)
