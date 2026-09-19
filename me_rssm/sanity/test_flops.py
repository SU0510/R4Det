"""Sanity: static FLOPs estimate of the temporal stage (GPU, random input).

Run:  CUDA_VISIBLE_DEVICES=4 python me_rssm/sanity/test_flops.py

Hook-based MAC (multiply-accumulate) counter over nn.Conv2d modules --
a *static* estimate on random inputs (no training, no metric selection).
Deformable conv / deform-offset ops are identical in both modules and are
therefore excluded from the comparison (they cancel in the delta).

Measures one forward of the temporal module at the real operating size
(B=2, 216x248 canvas, full-size 432x496 velocity canvas) for:
  - MotionAlignedRSSMFusion   (baseline)
  - MotionEvidenceRSSMFusion  (full ME-RSSM)
  - MotionEvidenceRSSMFusion  with use_modality_obs=False
"""

import os
import sys

import torch
import torch.nn as nn

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import me_rssm  # noqa: F401
from me_rssm import ModalityContext
from me_rssm.motion_evidence_rssm import MotionEvidenceRSSMFusion
from mmdet3d.models.fusion_layers.rssm_fusion import MotionAlignedRSSMFusion

DEV = 'cuda'
B, C, H, W = 2, 256, 216, 248
COMMON = dict(in_channels=C, out_channels=C, kernel_size=3, latent_dim=C,
              hidden_dim=128, action_dim=0, kl_scale=1.0, free_nats=1.0)


class MACCounter:
    def __init__(self, module):
        self.macfs = 0.0
        self.handles = []
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                self.handles.append(
                    m.register_forward_hook(self._hook))

    def _hook(self, m, inp, out):
        k = m.kernel_size[0] * m.kernel_size[1]
        self.macfs += out.numel() * (m.in_channels // m.groups) * k

    def __enter__(self):
        return self

    def __exit__(self, *a):
        for h in self.handles:
            h.remove()


def run(module, use_bus):
    module.reset_state()
    module.eval()
    with torch.no_grad():
        with MACCounter(module) as cnt:
            for t in range(4):
                if use_bus:
                    ModalityContext.set(
                        'velocity_bev',
                        torch.randn(B, 7, 496, 432, device=DEV) * 0.5)
                    ModalityContext.set(
                        'camera_bev', torch.randn(B, 256, 248, 216,
                                                  device=DEV))
                module(torch.randn(B, C, H, W, device=DEV),
                       use_posterior=True, deterministic=(t < 3),
                       detach_state=(t < 3))
    ModalityContext.clear()
    return cnt.macfs / 4 / 1e9  # mean GMAC per frame


base = MotionAlignedRSSMFusion(**COMMON).to(DEV)
full = MotionEvidenceRSSMFusion(**COMMON).to(DEV)
full.load_state_dict(base.state_dict(), strict=False)
noobs = MotionEvidenceRSSMFusion(**COMMON, use_modality_obs=False).to(DEV)
noobs.load_state_dict(base.state_dict(), strict=False)

g_base = run(base, use_bus=False)
g_full = run(full, use_bus=True)
g_noobs = run(noobs, use_bus=False)

print(f'[INFO] temporal-stage MACs per frame (conv-only, deform ops '
      f'excluded):')
print(f'[INFO]   baseline MotionAlignedRSSMFusion : {g_base:8.2f} GMAC')
print(f'[INFO]   ME-RSSM use_modality_obs=False  : {g_noobs:8.2f} GMAC')
print(f'[INFO]   ME-RSSM full                    : {g_full:8.2f} GMAC '
      f'({100 * (g_full - g_base) / g_base:+.1f}%)')
assert g_full > g_base, 'expected extra conv cost'
assert abs(g_noobs - g_base) / g_base < 0.02, \
    'switches-off runtime should match baseline'
print('[PASS] test_flops: cost accounting consistent with design')
