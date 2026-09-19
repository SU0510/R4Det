# ---------------------------------------------------------------------------
# ME-RSSM + P1 observation-guided state initialization (state_init='obs').
#
# Second-wave variant: inherits the ME-RSSM mainline config and flips ONLY
# `state_init` from 'zero' (baseline semantics, no extra parameters) to
# 'obs' (06_second_layer P1, learnable initial state).
#
# Mechanism: two zero-init 1x1 convs map the current observation encoding
# e_fused = encoder(feat) to h_0 (out_channels) and z_0 (latent_dim) at
# every sequence start, and re-initialize samples reset by
# reset_for_samples from their own current frame at the next step.
# Zero weights make the initial behaviour bitwise-identical to the
# mainline; the mechanism only activates as the convs learn. Do NOT stack
# any other variable on this config.
#
# Status: implemented + sanity-tested (me_rssm/sanity/test_state_init.py);
# NOT yet trained. Queue strictly behind the ME-RSSM mainline seed0 verdict.
# ---------------------------------------------------------------------------

_base_ = './TJ4D-R4Det_me_rssm_N4_24e_pretrained_v2_head.py'

model = dict(
    temporal_fusion=dict(
        state_init='obs',
    ),
)
