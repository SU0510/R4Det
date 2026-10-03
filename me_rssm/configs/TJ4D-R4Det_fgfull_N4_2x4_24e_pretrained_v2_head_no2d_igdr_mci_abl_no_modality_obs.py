# MCI ablation B-off: consume no side-channel context at all. Single
# GROUP-variable vs ..._mci: the three coupled bus-consumer switches are
# turned off together (use_modality_obs, and with it use_motion_offsets and
# use_dynamic_gate, whose inputs only exist on the bus). The reliability/
# readout gates survive through the innovation-magnitude input
# (innov_in_gain=True), so this isolates the value of the modality-specific
# observation model as a whole. The producer chain upstream still runs but
# nothing reads the bus.
_base_ = ['./TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_mci.py']

model = dict(
    temporal_fusion=dict(
        use_modality_obs=False,
        use_motion_offsets=False,
        use_dynamic_gate=False,
    ),
)
