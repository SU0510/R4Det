# MCI-RSSM + observation-bootstrapped initial state (docs/07 P1, review §5C).
#
# Single variable vs ..._mci: state_init='zero' -> 'obs'. Everything else,
# including all seven MCI switches and the producer chain, is inherited
# unchanged. The h/z bootstrap convs are zero-init, so at load time the
# behaviour is bitwise identical to the mainline MCI; training turns them
# into an observation-bootstrapped filter start.
#
# Motivation (docs/rssm_review_20261002.md P5/§8.4): t0 transition output is
# a per-sample constant (dead state, me_rssm/docs/10) and MAI-family runs
# start ep1 well below the baseline (e.g. 13.66 vs 15.26 in the route-C ep6
# gate). This variant is the prepared low-cost fix; gate it on the ep1
# reading as well as the standard mechanism gates.
#
# DDP note: state_init='obs' adds h_init_conv/z_init_conv, which ARE used in
# every forward (the sequence-start branch), so no unused-parameter risk.

_base_ = [
    './TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_mci.py'
]

model = dict(
    temporal_fusion=dict(
        state_init='obs',
    ),
)
