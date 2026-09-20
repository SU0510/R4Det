# FG-FULL N=3 variant of the N=4 foreground-supervision run.
#
# Historical frame-length ablation:
#   N=3 + hidden_dim=64  -> 34.27 best Overall-3D-moderate
#   N=3 + hidden_dim=128 -> 32.87
#   N=4 + hidden_dim=128 -> 34.05
#
# The capacity/frame match matters: hidden_dim=128 only helped once the RSSM
# saw four frames. For the shorter 3-frame sequence, hidden_dim=64 is the
# better and faster configuration. Everything else is inherited unchanged
# from the FG-FULL N=4 run, so this is a clean N=3 ablation minus the known
# N=3+128 capacity mismatch.

_base_ = './TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head.py'

model = dict(
    seq_len=3,
    temporal_fusion=dict(hidden_dim=64),
)

data = dict(
    train=dict(dataset=dict(seq_len=3)),
    val=dict(seq_len=3),
    test=dict(seq_len=3),
)
