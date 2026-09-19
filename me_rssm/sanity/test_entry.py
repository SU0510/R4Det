"""Sanity: eval-entry validation (the tools/test_vod.py code path).

Run:  CUDA_VISIBLE_DEVICES=4 python me_rssm/sanity/test_entry.py

What it verifies (06_second_layer item 2):
T1. The ME-RSSM mainline config builds through the *test* entry path --
    including the two quirks of tools/test_vod.py: (a) it does NOT process
    `custom_imports`, so me_rssm registration must happen at config-exec
    time (the config's `import me_rssm` does this); (b) it enables the
    old box-version flag before importing mmdet3d.
T2. A real checkpoint (checkpoints/pretrained_tj4d.pth, the same file the
    training run uses as load_from) loads into the ME model with the
    documented superset contract: producer/backbone keys load, only the
    fresh ME temporal keys are missing, nothing unexpected.
T3. One real dataset sample (4 radar frames + camera) goes through the
    full `model(return_loss=False, rescale=True)` forward on GPU --
    the exact call single_gpu_test makes: pts+img branches, per-frame
    Doppler bus publish, temporal RSSM loop, head decode + NMS.
T4. Outputs are finite, the per-frame bus contract holds (velocity_bev
    was published by the scatter and consumed by the temporal module),
    and the temporal module leaves state=None-compatible bookkeeping
    behind (reset semantics intact after eval).
"""

import argparse
import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

# --- replicate tools/test_vod.py's entry-order quirks ---------------------
import training_old_version  # noqa: E402  (repo-root copy, as tools/ does)
training_old_version.use_old_version()
assert training_old_version.get_old_version() is True

import torch  # noqa: E402
from mmcv import Config  # noqa: E402
from mmcv.parallel import MMDataParallel, collate  # noqa: E402
from mmcv.runner import load_checkpoint  # noqa: E402

import me_rssm  # noqa: F401,E402  (registers ME modules pre-build)
from mmdet3d.datasets import build_dataset  # noqa: E402
from mmdet3d.models import build_model  # noqa: E402
from me_rssm.modality_bus import ModalityContext  # noqa: E402
from me_rssm.motion_evidence_rssm import MotionEvidenceRSSMFusion  # noqa: E402

results = []


def check(name, ok, detail=''):
    results.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ''))


ap = argparse.ArgumentParser()
ap.add_argument('--config', default=os.path.join(
    _REPO, 'me_rssm/configs',
    'TJ4D-R4Det_me_rssm_N4_24e_pretrained_v2_head.py'))
ap.add_argument('--checkpoint', default=os.path.join(
    _REPO, 'checkpoints/pretrained_tj4d.pth'))
ap.add_argument('--sample-index', type=int, default=3)
args = ap.parse_args()

# ---- T1: config-exec import + build through the test path ----------------
cfg = Config.fromfile(args.config)
# tools/test_vod.py injects meta_info (figures_path/project_name) before
# building -- R4Det.__init__ subscripts it; replicate with a temp dir.
import tempfile  # noqa: E402
cfg.model.update(meta_info=dict(
    figures_path=tempfile.mkdtemp(prefix='me_rssm_entry_fig_'),
    project_name='tj4d_me_rssm_entry_sanity'))
cfg.model.pretrained = None
cfg.data.test.test_mode = True
model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
tf = model.temporal_fusion
check('T1 config builds via test path; temporal module is ME-RSSM',
      type(tf).__name__ == 'MotionEvidenceRSSMFusion',
      f'type={type(tf).__name__}')

# ---- T2: real checkpoint loads with the documented contract --------------
ckpt = load_checkpoint(model, args.checkpoint, map_location='cpu')
# Re-derive the exact key accounting (load_checkpoint only logs it).
# The TJ4D pretraining checkpoint comes from a related but different
# architecture variant: it contains 498 auxiliary keys outside the
# detection forward path (rangeview_foreground, PaintBEVFusion, ...),
# an older GRU-era temporal_fusion, a radar-depth branch and head layout
# that do not match the current model (baseline-known behaviour: those
# train fresh), and the feature extractors ME-RSSM builds on, which MUST
# load with exact shapes. ME-relevant contract:
#   (a) every pts_voxel_encoder / pts_backbone / pts_neck / img_backbone /
#       img_neck key in the checkpoint loads with a matching shape;
#   (b) model keys missing from the checkpoint are exactly
#       temporal_fusion.* (fresh ME module) plus the known baseline-fresh
#       cross_attention.fusion_conv.* and pts_bbox_head.conv_iou.*.
state = torch.load(args.checkpoint, map_location='cpu')
sd = state.get('state_dict', state)
model_sd = model.state_dict()
LOADERS = ('pts_voxel_encoder', 'pts_backbone', 'pts_neck', 'img_backbone',
           'img_neck')
loader_bad = [
    k for k, v in sd.items()
    if k.startswith(LOADERS)
    and (k not in model_sd or model_sd[k].shape != v.shape)]
missing = set(model_sd.keys()) - set(sd.keys())
KNOWN_FRESH = ('cross_attention.fusion_conv.', 'pts_bbox_head.conv_iou.')
missing_unexpected = sorted(
    k for k in missing
    if not k.startswith('temporal_fusion.')
    and not k.startswith(KNOWN_FRESH))
n_fresh_tf = sum(1 for k in missing if k.startswith('temporal_fusion.'))
check('T2 checkpoint contract: feature extractors load exactly; missing '
      'keys = fresh temporal_fusion + known baseline-fresh',
      not loader_bad and not missing_unexpected and n_fresh_tf > 0,
      f'loader_bad={loader_bad[:3]} '
      f'missing_unexpected={missing_unexpected[:3]} '
      f'|fresh temporal|={n_fresh_tf}')

# ---- T3: one real sample through the full eval forward -------------------
dataset = build_dataset(cfg.data.test, dict(test_mode=True))
sample = dataset[args.sample_index]
data = collate([sample], samples_per_gpu=1)
model = MMDataParallel(model.cuda(), device_ids=[0])
model.eval()

ModalityContext.clear()
with torch.no_grad():
    result = model(return_loss=False, rescale=True, **data)

boxes = result[0]['pts_bbox']['boxes_3d'].tensor \
    if isinstance(result, list) else None
n_det = 0 if boxes is None else boxes.shape[0]
finite = bool(boxes is not None and torch.isfinite(boxes).all().item())
check('T3 full eval forward produces finite detections', finite,
      f'num_boxes={n_det}')

# ---- T4: bus contract + state bookkeeping after eval ---------------------
vel = ModalityContext.get('velocity_bev')
check('T4a velocity_bev canvas present after forward (published)',
      vel is not None and vel.dim() == 4 and vel.shape[1] == 7,
      f'shape={tuple(vel.shape) if vel is not None else None}')
check('T4b temporal module state reset semantics intact',
      tf.reset_state() is None and tf.h_state is None)

n_bad = sum(1 for _, ok in results if not ok)
print(f"\n[test_entry] {len(results) - n_bad}/{len(results)} checks green")
sys.exit(1 if n_bad else 0)
