# Benchmark-only override: FG-FULL N=4 with hidden_dim=64.
#
# Used to isolate the N-frame effect from the hidden_dim effect when comparing
# N3 (seq_len=3, hidden_dim=64) against N4 (seq_len=4, hidden_dim=128).
# Training hyper-parameters are otherwise identical to the N4 fgfull config.

_base_ = ['/home/lurui/workspace/R4Det/configs/r4det/'
          'TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head.py']

model = dict(
    temporal_fusion=dict(hidden_dim=64),
)
