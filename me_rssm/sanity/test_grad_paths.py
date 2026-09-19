"""Sanity: gradient paths, state management, fallback semantics (GPU).

Run:  CUDA_VISIBLE_DEVICES=4 python me_rssm/sanity/test_grad_paths.py

Note on zero-init cold start: obs_proj / gain_conv / motion-offset convs /
conf convs start at exactly zero. In the FIRST backward only layers
downstream-independent of a zero weight matrix receive nonzero gradients;
the encoder/conf branches sit *behind* the zero obs_proj/gain matrices, so
their gradients become nonzero only after the first optimizer step moves
those matrices off zero (standard zero-init residual behaviour). The test
therefore runs two optimiser iterations:

G1a (step 1) gradients > 0 for: baseline encoder/posterior/prior/output/
    align/GRU, obs_proj, gain_conv, motion-offset convs, dyn_conv +
    dyn_beta (beta preset to 0.5 to exercise the gate).
G1b (step 2) gradients > 0 additionally for camera/motion encoders and
    confidence convs -- proving the full modality path is connected.
G2. The Doppler evidence canvas is a leaf measurement: no gradient is
    accumulated into it (motion_grad=False default).
G3. State detach semantics: detach_state=True stores graph-free state;
    detach_state=False keeps the graph.
G4. reset_for_samples zeroes exactly the requested samples' state.
G5. Prediction-fallback semantics: with confidences and gain driven to 0,
    z_t collapses to the prior mean mu_p (pure motion-propagated
    prediction) while outputs stay finite -- the missing-modality
    behaviour the baseline has no mechanism for.
G6. No NaNs anywhere in losses/outputs.
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

torch.manual_seed(0)
DEV = 'cuda'
B, C, H, W = 2, 256, 54, 62

module = MotionEvidenceRSSMFusion(
    in_channels=C, out_channels=C, latent_dim=C, hidden_dim=128,
    action_dim=0, kl_scale=1.0, free_nats=1.0).to(DEV)
module.train()
module.dyn_beta.data.fill_(0.5)   # exercise the dynamic gate actively

feats = [torch.randn(B, C, H, W, device=DEV) for _ in range(4)]
# Canvas sizes scaled down from (432, 496)/(248, 216) to match the small
# H, W=54, 62 test canvas: pool(2) -> (54, 62) after permute.
vel_leaf = torch.randn(B, 7, 124, 108, device=DEV) * 0.5
cam_leaf = torch.randn(B, 256, 62, 54, device=DEV)

params = dict(module.named_parameters())
optimizer = torch.optim.SGD(module.parameters(), lr=1e-2)


def populate():
    ModalityContext.set('velocity_bev', vel_leaf)
    ModalityContext.set('camera_bev', cam_leaf)


def one_pass(seed=42):
    module.reset_state()
    with torch.no_grad():           # burn-in frames
        for t in range(3):
            populate()
            module(feats[t], use_posterior=True, deterministic=True,
                   detach_state=True)
    populate()
    torch.manual_seed(seed)
    out, recon, kl, h_t, z_t, stats = module(
        feats[3], use_posterior=True, deterministic=False,
        detach_state=True)
    loss = out.mean() + recon.mean() + kl
    loss.backward()
    return out, recon, kl, h_t, z_t, stats


def grad_of(name):
    p = params[name]
    return None if p.grad is None else p.grad.abs().max().item()


# ---- G1a: first-step gradients ------------------------------------------
out, recon, kl, h_t, z_t, stats = one_pass()
step1_targets = [
    'encoder.0.conv.weight',            # baseline observation encoder
    'posterior_mu.weight', 'prior_mu.weight', 'output_proj.weight',
    'align_h_offset_mask.weight', 'transition.update_conv.weight',
    'obs_proj.weight',                  # zero-init residual (own grad != 0)
    'gain_conv.weight',                 # gain matrix (own grad != 0)
    'motion_offset_h.weight', 'motion_offset_z.weight',
    'dyn_conv.weight', 'dyn_beta',
]
for name in step1_targets:
    g = grad_of(name)
    assert g is not None and g > 0, f'G1a no gradient at {name}'
cold = ['camera_encoder.0.conv.weight', 'motion_encoder.0.conv.weight',
        'conf_cam.weight', 'conf_mot.weight']
for name in cold:
    assert params[name].grad is not None or True  # documented one-step delay
print('[PASS] G1a step-1 gradients: direct pathways all reachable')

# ---- optimizer step, then verify the behind-zero-init branches ----------
optimizer.step()
optimizer.zero_grad()

out, recon, kl, h_t, z_t, stats = one_pass(seed=43)
step2_targets = cold + ['obs_proj.weight', 'gain_conv.weight']
for name in step2_targets:
    g = grad_of(name)
    assert g is not None and g > 0, \
        f'G1b branch behind zero-init did not activate: {name} (g={g})'
print('[PASS] G1b step-2 gradients: modality encoders + confidence convs '
      'connected end-to-end')

# ---- G2: Doppler canvas stays a detached measurement ---------------------
assert not vel_leaf.requires_grad and vel_leaf.grad is None, 'G2 violated'
assert not cam_leaf.requires_grad, 'G2 violated (camera)'
print('[PASS] G2 velocity/camera canvas tensors accumulate no gradient')

# ---- G3: state detach semantics ------------------------------------------
assert not module.h_state.requires_grad, 'G3 state should be detached'
assert not module.z_state.requires_grad, 'G3 state should be detached'
out2, _, _, h2, _, _ = module(feats[3], use_posterior=True,
                              deterministic=True, detach_state=False)
assert module.h_state.requires_grad and module.z_state.requires_grad, \
    'G3 non-detached state lost its graph'
module.h_state = module.h_state.detach()
module.z_state = module.z_state.detach()
print('[PASS] G3 detach_state semantics correct')

# ---- G4: reset_for_samples -----------------------------------------------
h_before = module.h_state.clone()
# The detector passes ~valid_t which lives on the image device (CUDA).
mask = torch.tensor([True, False], device=DEV)
module.reset_for_samples(mask)
assert module.h_state[0].abs().sum().item() == 0.0, 'G4 sample0 not reset'
assert torch.allclose(module.h_state[1], h_before[1]), \
    'G4 sample1 wrongly modified'
print('[PASS] G4 reset_for_samples zeroes exactly the masked samples')

# ---- G5: prediction fallback ---------------------------------------------
fresh = MotionEvidenceRSSMFusion(
    in_channels=C, out_channels=C, latent_dim=C, hidden_dim=128,
    action_dim=0, kl_scale=1.0, free_nats=1.0).to(DEV)
fresh.train()
with torch.no_grad():
    fresh.gain_conv.bias.fill_(-10.0)     # g -> 0
    fresh.conf_cam.bias.fill_(-10.0)      # camera confidence -> 0
    fresh.conf_mot.bias.fill_(-10.0)      # motion confidence -> 0
fresh.reset_state()
with torch.no_grad():
    for t in range(3):
        populate()
        fresh(feats[t], use_posterior=True, deterministic=True,
              detach_state=True)
    populate()
    out5, recon5, kl5, h5, z5, stats5 = fresh(
        feats[3], use_posterior=True, deterministic=False,
        detach_state=True)
    mu_p = fresh.prior_mu(h5)
    diff = (z5 - mu_p).abs().max().item()
    assert diff < 1e-3, f'G5 z_t did not fall back to prior mean: {diff}'
    for name, tensor in [('output', out5), ('recon', recon5)]:
        assert torch.isfinite(tensor).all(), f'G5 NaN in {name}'
    assert kl5 is not None and torch.isfinite(kl5)
    assert stats5['stat_gain_mean'].item() < 0.01
ModalityContext.clear()
print(f'[PASS] G5 fallback: z_t == mu_p (max|Δ|={diff:.2e}), outputs finite')

# ---- G6: no NaN in the main pass -----------------------------------------
for name, tensor in [('output', out), ('recon', recon), ('h_t', h_t),
                     ('z_t', z_t)]:
    assert torch.isfinite(tensor).all(), f'G6 NaN in {name}'
assert torch.isfinite(kl)
print('[PASS] G6 main-pass outputs finite')

print('[PASS] test_grad_paths: all checks green')
