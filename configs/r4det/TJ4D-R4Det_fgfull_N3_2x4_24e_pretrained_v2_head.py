# FG-FULL N=3 variant of the N=4 foreground-supervision run.
#
# This is a strict frame-count ablation: seq_len 4 -> 3 and nothing else.
# hidden_dim stays at the N=4 value (128) so the comparison isolates the
# number of temporal frames; the historical N=3/hdim64 result (section 8) was
# a separate capacity-matched ablation, not this one.

_base_ = './TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head.py'

model = dict(
    seq_len=3,
)

data = dict(
    train=dict(dataset=dict(seq_len=3)),
    val=dict(seq_len=3),
    test=dict(seq_len=3),
)
