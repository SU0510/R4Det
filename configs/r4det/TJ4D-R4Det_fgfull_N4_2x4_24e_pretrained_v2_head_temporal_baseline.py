# ---------------------------------------------------------------------------
# FG-FULL N=4 with ONLY the temporal fusion module replaced.
#
# Base: TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head.py
# So everything else is kept identical to the current FG-FULL mainline:
#   - pretrained backbone / detector input recipe
#   - N=4 data window and batch/schedule
#   - head-v2
#   - 2D RPN/RoI, MRF3Net, FRPN, foreground-biased relative depth supervision
#   - IGDR fusion
#
# The only model difference is:
#   MotionAlignedRSSMFusion -> TemporalDeformableFusionBaseline
#
# The adapter exposes the six-item temporal-fusion interface expected by the
# current detector while internally running the original
# TemporalDeformableFusion on (current frame, previous raw BEV). It produces no
# reconstruction or KL loss.
#
# rssm_bptt_steps=0 is intentional: the original Temporal baseline consumed a
# detached previous-frame BEV feature and fused only the current frame. Setting
# this to 0 keeps all history frames under no_grad, matching that semantics
# instead of importing RSSM's truncated-BPTT training protocol into the GRU
# baseline.
# ---------------------------------------------------------------------------

_base_ = './TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head.py'

model = dict(
    temporal_fusion=dict(
        _delete_=True,
        type='TemporalDeformableFusionBaseline',
        in_channels=256,
        out_channels=256,
        kernel_size=3,
        deform_groups=1,
        gate_kernel_size=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        act_cfg=dict(type='ReLU', inplace=True),
    ),
    rssm_bptt_steps=0,
)

custom_hooks = []
