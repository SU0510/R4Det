# ---------------------------------------------------------------------------
# FG-FULL N=4 without the 2D instance branch and without IGDR.
#
# Base: TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head.py
#
# This is a narrow ablation of the current FG-FULL recipe. The only change is
# that img_rpn_head and img_roi_head are disabled:
#   - no 2D RPN / RoI / mask heads are built or trained;
#   - no instance features are produced for instance-gated BEV refinement;
#   - igdr_fusion is therefore not built by R4Det, because its constructor
#     gates it on (with_rpn and with_roi_head).
#
# MRF3Net range-view foreground supervision, FRPN BEV proposal supervision,
# the foreground-biased relative depth loss, and the RSSM temporal branch are
# inherited unchanged from the FG-FULL base config.
# ---------------------------------------------------------------------------

_base_ = './TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head.py'

model = dict(
    img_rpn_head=None,
    img_roi_head=None,
)
