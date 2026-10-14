# Versioned BPTT0 reference recipe for the GP-MSF resource gate.
#
# Keep the historical BPTT1 mainline config unchanged.  This file makes the
# temporal training protocol explicit and shared by the B0 reference and the
# GP-MSF arm: all four frames still run forward, but only the current frame
# carries detection/reconstruction/KL gradients.

_base_ = './TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr.py'

model = dict(
    rssm_bptt_steps=0,
)
