"""Pre-flight validation for the fgfull config family (CPU only).

Run:  python me_rssm/sanity/test_fgfull_config.py [config path]

Checks
------
F1. Config loads through mmcv.Config.fromfile (3-level inheritance chain)
    and the merged switches are correct.
F2. The full R4Det model builds on CPU with the expected new modules.
F3. Pretrained-key compatibility: loading checkpoints/pretrained_tj4d.pth
    with strict=False. Expected: ONLY igdr_fusion.* missing (fresh init,
    identity-start by design); everything else (2D heads, MRF3Net, FRPN)
    hot-starts from the pretrained weights.
F4. The train dataset builds and ONE real sample passes through the full
    extended pipeline (segmentation / VLSAM masks / my_gt_depth /
    bbox_Mask / gt_depths all present with sane shapes).
"""

import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import numpy as np
import torch

from mmcv import Config
from mmdet3d.models import build_model
from mmdet3d.datasets import build_dataset

CFG = os.path.join(_REPO, 'configs/r4det/'
                   'TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head.py')
if len(sys.argv) > 1:
    CFG = os.path.abspath(sys.argv[1])
CKPT = os.path.join(_REPO, 'checkpoints/pretrained_tj4d.pth')


def main():
    ok = True

    # F1: config load + merged switch check
    cfg = Config.fromfile(CFG)
    model_cfg = cfg.model
    for key, expect in [
            ('use_depth_supervision', True),
            ('use_props_supervision', True),
            ('use_msk2d_supervision', True)]:
        got = model_cfg.get(key)
        status = 'OK' if got is expect else 'FAIL'
        ok &= got is expect
        print(f'F1 model.{key} = {got} (expect {expect}) [{status}]')
    print('F1 model.img_rpn_head type =', model_cfg.get('img_rpn_head', {}).get('type'))
    print('F1 model.img_roi_head type =', model_cfg.get('img_roi_head', {}).get('type'))
    print('F1 model.rangeview_foreground type =',
          model_cfg.get('rangeview_foreground', {}).get('type'))
    print('F1 model.proposal_layer type =', model_cfg.get('proposal_layer', {}).get('type'))
    print('F1 pts_bbox_head.shared_stem =',
          model_cfg.pts_bbox_head.get('shared_stem'), '(expect False)')
    ok &= model_cfg.pts_bbox_head.get('shared_stem') is False
    print('F1 depth_net.loss_abs_weight =', model_cfg.depth_net.loss_abs_weight,
          '(expect 0.0)  relative_loss_weight =',
          model_cfg.depth_net.relative_loss_weight, '(expect 0.04)')
    ok &= model_cfg.depth_net.loss_abs_weight == 0.0
    ok &= model_cfg.depth_net.relative_loss_weight == 0.04
    print('F1 temporal_fusion type =', model_cfg.temporal_fusion.type,
          '(expect MotionAlignedRSSMFusion)')
    ok &= model_cfg.temporal_fusion.type == 'MotionAlignedRSSMFusion'
    print('F1 model.seq_len =', model_cfg.seq_len)
    print('F1 temporal_fusion.hidden_dim =', model_cfg.temporal_fusion.hidden_dim)

    # F2: model build on CPU (inject meta_info exactly like tools/train_vod.py)
    import tempfile
    cfg.model.update(meta_info=dict(
        figures_path=tempfile.mkdtemp(prefix='fgfull_sanity_fig_'),
        project_name='tj4d_fgfull_sanity'))
    model = build_model(model_cfg)
    has = {}
    for name in ['img_rpn_head', 'img_roi_head', 'igdr_fusion',
                 'rangeview_foreground', 'proposal_layer_former',
                 'temporal_fusion']:
        has[name] = getattr(model, name, None) is not None
        print(f'F2 model.{name} built: {has[name]}')
    # proposal_layer_latter is tied to use_backward_projection (False in the
    # mainline) -> its absence is expected; FRPN supervision runs via former.
    latter = getattr(model, 'proposal_layer_latter', None)
    print(f'F2 model.proposal_layer_latter built: {latter is not None} '
          '(False is expected with use_backward_projection=False)')
    ok &= all([has['img_rpn_head'], has['img_roi_head'], has['igdr_fusion'],
               has['rangeview_foreground'], has['proposal_layer_former'],
               has['temporal_fusion']])
    n_igdr = sum(p.numel() for n, p in model.igdr_fusion.named_parameters())
    print(f'F2 igdr_fusion params: {n_igdr} (fresh init, identity-start)')

    # F3: pretrained key coverage (mmcv semantics: shape mismatches are
    # skipped with a warning, e.g. the pretrain 3D head vs head-v2 — the
    # mainline training has the same behaviour).
    ckpt = torch.load(CKPT, map_location='cpu')
    sd = ckpt.get('state_dict', ckpt)
    msd = model.state_dict()
    loaded_new = ['img_rpn_head', 'img_roi_head', 'rangeview_foreground',
                  'proposal_layer_former']
    for p in loaded_new:
        ck_keys = [k for k in sd if k.startswith(p + '.')]
        hit = [k for k in ck_keys
               if k in msd and tuple(msd[k].shape) == tuple(sd[k].shape)]
        print(f'F3 hot-start {p}: {len(hit)}/{len(ck_keys)} ckpt keys '
              f'shape-matched into model')
        ok &= len(ck_keys) > 0 and len(hit) == len(ck_keys)
    n_igdr_hit = len([k for k in sd if k.startswith('igdr_fusion.')])
    print(f'F3 igdr_fusion keys in ckpt: {n_igdr_hit} (expect 0 -> fresh)')
    ok &= n_igdr_hit == 0

    # F4: one real sample through the extended pipeline
    ds = build_dataset(cfg.data.train)
    print('F4 train dataset len:', len(ds))
    sample = ds[0]
    assert sample is not None, 'F4 dataset returned None'
    keys = list(sample.keys())
    print('F4 sample keys:', keys)
    for need in ['points', 'img', 'gt_masks', 'my_gt_depth',
                 'gt_bboxes', 'gt_labels']:
        present = need in keys
        print(f'F4 key {need}: {"OK" if present else "MISSING"}')
        ok &= present
    metas = sample['img_metas'].data
    metas_last = metas[-1] if isinstance(metas, list) else metas
    for need in ['segmentation', 'bbox_Mask', 'gt_depths', 'sample_idx']:
        present = need in metas_last
        shape = getattr(metas_last.get(need), 'shape', None)
        print(f'F4 meta {need}: {"OK" if present else "MISSING"} shape={shape}')
        ok &= present
    d = metas_last['segmentation']
    print('F4 segmentation unique vals:', np.unique(d)[:5],
          'fg ratio:', float((d > 0).mean()))
    m = metas_last['bbox_Mask']
    print('F4 bbox_Mask fg ratio:', float((m > 0).mean()))
    md = sample['my_gt_depth']
    mdv = md[-1] if isinstance(md, list) else md
    mdv = mdv.data if hasattr(mdv, 'data') else mdv
    print('F4 my_gt_depth shape:', np.shape(mdv), 'range:',
          float(np.min(mdv)), float(np.max(mdv)))
    gm = sample['gt_masks']
    gmv = gm[-1] if isinstance(gm, list) else gm
    gmv = gmv.data if hasattr(gmv, 'data') else gmv
    print('F4 gt_masks:', type(gmv).__name__,
          'n masks:', len(gmv.masks) if hasattr(gmv, 'masks') else '?')

    print('\n==== RESULT:', 'ALL OK' if ok else 'FAILURES PRESENT', '====')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
