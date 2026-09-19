"""Sanity: ME-RSSM equivalence with the baseline + forward behaviour (GPU).

Requires CUDA (ModulatedDeformConv2d). Run with a free GPU, e.g.
    CUDA_VISIBLE_DEVICES=4 python me_rssm/sanity/test_equivalence.py

Checks
------
E1. Weight compatibility: every baseline MotionAlignedRSSMFusion parameter
    loads into MotionEvidenceRSSMFusion with matching names/shapes; the
    only missing keys are the new identity-start extras.
E2. Empty-bus equivalence: a 4-frame sequence (3 deterministic burn-in
    frames + 1 stochastic current frame) produces bitwise-close output /
    reconstruction / KL in both modules given equal weights.
E3. Populated-bus + all-switches-off equivalence: with the velocity canvas
    and camera BEV present but use_motion_offsets / use_reliability_gain /
    use_dynamic_gate disabled, output is still baseline-identical
    (zero-init residual + gain bypass + beta=0 are exact identities).
E4. Populated-bus + all switches on: finite outputs, extended stat_*
    keys present, gain in (0, 1).
"""

import os
import sys

import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import me_rssm  # noqa: F401
from me_rssm import ModalityContext
from me_rssm.motion_evidence_rssm import MotionEvidenceRSSMFusion
from mmdet3d.models.fusion_layers.rssm_fusion import MotionAlignedRSSMFusion

torch.manual_seed(0)
DEV = 'cuda'
B, C, H, W = 2, 256, 54, 62
COMMON = dict(in_channels=C, out_channels=C, kernel_size=3, latent_dim=C,
              hidden_dim=128, action_dim=0, kl_scale=1.0, free_nats=1.0,
              min_std=0.1, init_std=0.2, align_kernel_size=3,
              align_deform_groups=1, align_z_state=True)


def build_pair(switches=None):
    base = MotionAlignedRSSMFusion(**COMMON).to(DEV)
    new = MotionEvidenceRSSMFusion(
        **COMMON,
        motion_channels=7,
        camera_bev_channels=256,
        use_motion_offsets=True if switches is None else switches.get(
            'use_motion_offsets', True),
        use_reliability_gain=True if switches is None else switches.get(
            'use_reliability_gain', True),
        use_dynamic_gate=True if switches is None else switches.get(
            'use_dynamic_gate', True),
        use_modality_obs=True if switches is None else switches.get(
            'use_modality_obs', True),
    ).to(DEV)
    missing, unexpected = new.load_state_dict(base.state_dict(),
                                              strict=False)
    return base, new, list(missing), list(unexpected)


def expected_extra_prefixes():
    return ('motion_encoder.', 'camera_encoder.', 'obs_proj.', 'conf_cam.',
            'conf_mot.', 'gain_conv.', 'dyn_conv.', 'motion_offset_h.',
            'motion_offset_z.', 'dyn_beta')


def populate_bus():
    # Canvas sizes scaled down from (432, 496)/(248, 216) to match the
    # small H, W=54, 62 test canvas: pool(2) -> (54, 62) after permute.
    vel = torch.randn(B, 7, 124, 108, device=DEV) * 0.5
    cam = torch.randn(B, 256, 62, 54, device=DEV)
    ModalityContext.set('velocity_bev', vel)
    ModalityContext.set('camera_bev', cam)


def run_sequence(module, use_bus):
    module.reset_state()
    module.eval()
    outs = []
    with torch.no_grad():
        for t in range(4):
            if use_bus:
                populate_bus()
            torch.manual_seed(1234 + t)
            outs.append(module(
                feats[t], use_posterior=True,
                deterministic=(t < 3), detach_state=(t < 3)))
    return outs


def make_feats():
    return [torch.randn(B, C, H, W, device=DEV) for _ in range(4)]


def assert_close(a, b, name, atol=1e-5):
    diff = (a - b).abs().max().item()
    assert diff < atol, f'{name} differs: max|Δ|={diff}'


feats = make_feats()

# ---- E1: weight compatibility ------------------------------------------
base, new, missing, unexpected = build_pair()
assert unexpected == [], f'unexpected keys: {unexpected}'
bad = [k for k in missing if not k.startswith(expected_extra_prefixes())]
assert bad == [], f'unexpected missing keys: {bad}'
print(f'[PASS] E1 weight compatibility: {len(missing)} new keys, '
      f'0 unexpected')

# ---- E2: empty-bus equivalence ------------------------------------------
base_outs = run_sequence(base, use_bus=False)
new_outs = run_sequence(new, use_bus=False)
ModalityContext.clear()
for t in range(4):
    assert_close(base_outs[t][0], new_outs[t][0], f'E2 output t={t}')
    assert_close(base_outs[t][1], new_outs[t][1], f'E2 recon t={t}')
    if base_outs[t][2] is not None:
        assert_close(base_outs[t][2], new_outs[t][2], f'E2 kl t={t}')
    assert_close(base_outs[t][3], new_outs[t][3], f'E2 h_t t={t}')
print('[PASS] E2 empty-bus equivalence (max|Δ| < 1e-5 on all frames)')

# ---- E3: populated bus, all switches off --------------------------------
switches = dict(use_motion_offsets=False, use_reliability_gain=False,
                use_dynamic_gate=False, use_modality_obs=True)
base2, new2, _, _ = build_pair(switches)
base_outs = run_sequence(base2, use_bus=False)
new_outs = run_sequence(new2, use_bus=True)
ModalityContext.clear()
for t in range(4):
    assert_close(base_outs[t][0], new_outs[t][0], f'E3 output t={t}')
    assert_close(base_outs[t][1], new_outs[t][1], f'E3 recon t={t}')
print('[PASS] E3 populated-bus + switches-off equivalence')

# ---- E4: populated bus, full model --------------------------------------
base4, new4, _, _ = build_pair()
new4.train()
new4.reset_state()
for t in range(4):
    populate_bus()
    out, recon, kl, h_t, z_t, stats = new4(
        feats[t], use_posterior=True, deterministic=(t < 3),
        detach_state=(t < 3))
    for name, tensor in [('output', out), ('recon', recon), ('h_t', h_t),
                         ('z_t', z_t)]:
        assert torch.isfinite(tensor).all(), f'E4 NaN in {name} t={t}'
assert kl is not None and torch.isfinite(kl)
extra_stats = ['stat_gain_mean', 'stat_conf_cam_mean', 'stat_conf_mot_mean',
               'stat_dyn_ratio', 'stat_speed_mean']
for k in ['stat_kl_raw_mean', 'stat_mu_diff_sq'] + extra_stats:
    assert k in stats, f'E4 missing stat {k}'
g = stats['stat_gain_mean'].item()
assert 0.0 < g < 1.0, f'gain mean {g} outside (0,1)'
ModalityContext.clear()
print(f'[PASS] E4 full model forward: finite, stats complete, '
      f'gain_mean={g:.3f}')

print('[PASS] test_equivalence: all checks green')
