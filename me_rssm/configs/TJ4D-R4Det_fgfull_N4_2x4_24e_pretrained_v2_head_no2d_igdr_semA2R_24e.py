# A2-R arm, 24e mainline run: SAME candidate-A supervision, SAME 4-channel
# predictor and the SAME parameter count -- only the residual's DESTINATION
# channels change.
# Pre-registration: docs/rssm_candidate_A2_prereg_20261011.md  (section 4.2)
#
# Single variable vs ..._semreadout.py: cls_readout_mode
#   per_class_slot -> candidate A: class c lands on c's own anchor slot only,
#                     i.e. 12 of the 48 conv_cls channels.
#   anchor_slot    -> A2-R (this file): class c is written into every anchor
#                     slot of class c, i.e. all 48 channels.
# conv_cls layout is [anchor * num_classes + class]; ``num_anchors`` is taken
# from the detection head at runtime (R4Det._bind_cls_readout_anchor_slots),
# NOT retyped here, so it cannot drift away from cls_out_channels.
#
# Launch (identical protocol to candidate A):
#   CUDA_VISIBLE_DEVICES=5,6,7 bash tools/dist_train.sh <this config> 3 \
#     --seed 0 --deterministic \
#     --work-dir /data/lurui/work_dirs/fgfull_N4_no2d_igdr_2x4_24e_semA2R_seed0
_base_ = [
    '../configs/'
    'TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_semreadout.py'
]

model = dict(
    temporal_fusion=dict(
        cls_readout_mode='anchor_slot',
    ),
)
