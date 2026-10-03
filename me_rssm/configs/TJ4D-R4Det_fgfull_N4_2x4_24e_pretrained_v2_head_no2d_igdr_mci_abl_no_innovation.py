# MCI ablation I-off: revert the posterior to the standard form
# (mu_q = posterior_mu([h_t, e_obs]), no obs_prior/innovation). Coupled
# change: innov_in_gain goes off with it (there is no innovation for the
# gates to observe; enforced by the class validation). Isolates the
# innovation posterior's contribution WITHIN the merged model (the
# MAI-vs-standard comparison at full resolution, now with Doppler + gates
# active).
_base_ = ['./TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_mci.py']

model = dict(
    temporal_fusion=dict(
        posterior_mode='standard',
        innov_in_gain=False,
    ),
)
