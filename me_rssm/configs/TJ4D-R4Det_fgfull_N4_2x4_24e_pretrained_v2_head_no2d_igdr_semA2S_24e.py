# A2S arm (24e mainline run, same protocol as candidate A).
# Pre-registration: docs/rssm_candidate_A2_prereg_20261011.md  (section 4.1)
#
# Single variable vs ..._semreadout.py: sem_target_mode
#   gaussian_full        -> candidate A (the base config's default)
#   channel_contrastive  -> A2-S   (this file)
# The focal term, its weight, the head, the readout, the reconstruction
# target, the posterior/prior/KL, the schedule and the seed protocol are
# untouched. The four ranking numbers below are FROZEN at pre-registration
# time; tuning any of them during training is forbidden (prereg section 2).
#
# Launch (identical protocol to candidate A):
#   CUDA_VISIBLE_DEVICES=5,6,7 bash tools/dist_train.sh <this config> 3 \
#     --seed 0 --deterministic \
#     --work-dir /data/lurui/work_dirs/fgfull_N4_no2d_igdr_2x4_24e_semA2S_seed0
_base_ = [
    '../configs/'
    'TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_semreadout.py'
]

model = dict(
    temporal_fusion=dict(
        sem_target_mode='channel_contrastive',
        sem_rank_q=0.999,
        sem_rank_margin=1.0,
        sem_rank_weight=0.1,
    ),
)
