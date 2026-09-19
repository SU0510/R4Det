"""Sanity: P1 observation-guided state initialization (state_init='obs').

Run:  CUDA_VISIBLE_DEVICES=4 python me_rssm/sanity/test_state_init.py

Device split
------------
S1-S5 are BITWISE equality checks and therefore run on **CPU**: on this
environment's mmcv build, repeated in-process CUDA forwards of the deform
alignment make bitwise cross-run comparisons allocator-layout sensitive
(an environmental property shared by the untouched baseline
MotionAlignedRSSMFusion -- verified 2026-09-19, see docs/07; not a P1
regression). CPU execution is deterministic and layout-stable, which is
what identity/equality semantics need. S6 (gradient connectivity) and the
smoke forward run on GPU where the real training happens.

Checks:

S1. Regression guard after the forward reorder (e_fused computed before
    state init): with the default state_init='zero' and an empty bus the
    ME module is still bitwise-equal to the untouched baseline
    MotionAlignedRSSMFusion under identical weights.
S2. Identity start: a state_init='obs' module whose extra convs are at
    their zero init produces bitwise-equal outputs to the state_init='zero'
    module (the only extra params are the 4 init-conv tensors).
S3. Weight-key contract: loading a 'zero' checkpoint into an 'obs' module
    reports exactly {h_init_conv.*, z_init_conv.*} as missing and nothing
    unexpected; loading 'obs' into 'zero' reports exactly those keys as
    unexpected. Baseline RSSM checkpoints therefore stay loadable.
S4. Bootstrapping activates: perturbing the init convs changes the
    sequence-start behaviour (outputs and stored state differ from S2).
S5. reset_for_samples pending semantics (single-sample batch B=1, so no
    cross-sample dimension is involved at all): after resetting the only
    sample between two steps, its state is zeroed immediately (baseline
    semantics kept) and re-bootstrapped from its own current frame at the
    next step, while an unreset control module evolves identically up to
    the reset and diverges after it. The pending flag is consumed by
    exactly one step.
S6. Gradients reach the bootstrap convs on the first backward on GPU
    (h_init_conv/z_init_conv get nonzero grads through the alignment ->
    GRU -> posterior path).
S7. Parameter accounting: the 'obs' variant adds exactly
    2 * (in_channels * width + width) parameters over 'zero'.
"""

import os
import sys
import tempfile

import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import me_rssm  # noqa: F401
from me_rssm.motion_evidence_rssm import MotionEvidenceRSSMFusion
from mmdet3d.models.fusion_layers.rssm_fusion import MotionAlignedRSSMFusion

torch.manual_seed(0)
DEV = 'cpu'          # bitwise checks: deterministic, allocator-stable
DEV_G = 'cuda'       # gradient check on the real training device
B, C, H, W = 1, 32, 24, 28

results = []


def check(name, ok, detail=''):
    results.append((name, ok, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ''))


# ---- S0: the stateinit config loads and flips only state_init ------------
try:
    from mmcv import Config
    from mmdet3d.models import build_model
    _cfg = Config.fromfile(os.path.join(
        _REPO, 'me_rssm/configs',
        'TJ4D-R4Det_me_rssm_N4_24e_pretrained_v2_head_stateinit.py'))
    _cfg.model.update(meta_info=dict(
        figures_path=tempfile.mkdtemp(prefix='me_rssm_fig_'),
        project_name='tj4d_me_rssm_sanity'))
    _model = build_model(_cfg.model)
    _tf = _model.temporal_fusion
    _n_init = sum(p.numel() for p in _tf.h_init_conv.parameters()) + \
        sum(p.numel() for p in _tf.z_init_conv.parameters())
    _expected_init = 2 * (256 * 256 + 256)  # 256->256 h and z, plus biases
    check('S0 stateinit config builds; only state_init flipped',
          type(_tf).__name__ == 'MotionEvidenceRSSMFusion'
          and _tf.state_init == 'obs'
          and _tf.h_init_conv.weight.abs().sum().item() == 0.0
          and _n_init == _expected_init,
          f'state_init={_tf.state_init}, init conv params={_n_init} '
          f'(expected {_expected_init})')
except Exception as e:  # pragma: no cover
    check('S0 stateinit config builds; only state_init flipped', False, repr(e))


def build(state_init, dev=DEV):
    m = MotionEvidenceRSSMFusion(
        in_channels=C, out_channels=C, latent_dim=C, hidden_dim=64,
        action_dim=0, kl_scale=1.0, free_nats=1.0, state_init=state_init)
    return m.to(dev)


def copy_shared(src, dst):
    """Copy all shared weights; return (missing, unexpected) key sets."""
    res = dst.load_state_dict(src.state_dict(), strict=False)
    return sorted(res.missing_keys), sorted(res.unexpected_keys)


def run_steps(module, feats):
    """Deterministic multi-step forward; returns per-step (out, z_t)."""
    module.reset_state()
    outs = []
    with torch.no_grad():
        for f in feats:
            out, _, _, _, z_t, _ = module(
                f, use_posterior=True, deterministic=True, detach_state=True)
            outs.append((out.clone(), z_t.clone()))
    return outs


torch.manual_seed(7)
me_zero = build('zero')
torch.manual_seed(123)
feats = [torch.randn(B, C, H, W, device=DEV) for _ in range(3)]

# ---- S1: empty-bus bitwise equivalence to the baseline (after reorder) ---
base = MotionAlignedRSSMFusion(
    in_channels=C, out_channels=C, latent_dim=C, hidden_dim=64,
    action_dim=0, kl_scale=1.0, free_nats=1.0).to(DEV)
missing, unexpected = copy_shared(me_zero, base)
outs_me = run_steps(me_zero, feats)
outs_base = run_steps(base, feats)
same = all(
    torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    for a, b in zip(outs_me, outs_base))
check('S1 zero-init == baseline (empty bus, bitwise, CPU)', same,
      f'baseline missing={len(missing)} (all ME-only), unexpected={len(unexpected)}')

# ---- S2: identity start of the 'obs' variant ------------------------------
me_obs = build('obs')
copy_shared(me_zero, me_obs)
outs_obs = run_steps(me_obs, feats)
same = all(
    torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    for a, b in zip(outs_me, outs_obs))
check('S2 obs variant bitwise == zero variant at zero init', same)

# ---- S3: state-dict key contract ------------------------------------------
missing, unexpected = copy_shared(me_zero, me_obs)
ok = missing == ['h_init_conv.bias', 'h_init_conv.weight',
                 'z_init_conv.bias', 'z_init_conv.weight'] and not unexpected
check('S3 zero->obs load keys', ok, f'missing={missing} unexpected={unexpected}')
missing2, unexpected2 = copy_shared(me_obs, me_zero)
ok2 = unexpected2 == ['h_init_conv.bias', 'h_init_conv.weight',
                      'z_init_conv.bias', 'z_init_conv.weight'] and not missing2
check('S3b obs->zero load keys', ok2,
      f'missing={missing2} unexpected={unexpected2}')

# ---- S4: perturbed bootstrap convs activate -------------------------------
me_obs_p = build('obs')
copy_shared(me_zero, me_obs_p)
with torch.no_grad():
    me_obs_p.h_init_conv.weight.normal_(0, 0.05)
    me_obs_p.z_init_conv.weight.normal_(0, 0.05)
outs_p = run_steps(me_obs_p, feats)
differs = any(
    (not torch.equal(a[0], b[0])) or (not torch.equal(a[1], b[1]))
    for a, b in zip(outs_me, outs_p))
check('S4 perturbed init convs change sequence-start outputs', differs,
      f'|dh_state| after last step = '
      f'{(me_obs_p.h_state - me_zero.h_state).abs().max().item():.3e}')

# ---- S5: reset_for_samples pending semantics (B=1, CPU) -------------------
# control: same perturbed 'obs' module, never reset mid-sequence;
# treat: identical module, reset_for_samples([True]) before the 2nd frame.
# Up to the reset the two evolve bitwise identically; right after the
# reset the treat state is zeroed (baseline semantics kept); at the next
# step it is re-bootstrapped from its own current frame and must diverge
# from the control.
control = build('obs')
copy_shared(me_zero, control)
with torch.no_grad():
    control.h_init_conv.weight.normal_(0, 0.05)
    control.z_init_conv.weight.normal_(0, 0.05)
treat = build('obs')
treat.load_state_dict(control.state_dict())  # exact same bootstrap convs

control.reset_state()
treat.reset_state()
outs_ctrl, outs_trt = [], []
reset_zeroed = True
with torch.no_grad():
    for t, f in enumerate(feats):
        if t == 1:
            treat.reset_for_samples(torch.ones(1, dtype=torch.bool,
                                               device=DEV))
            reset_zeroed = bool(treat.h_state.abs().sum().item() == 0.0) \
                and bool(treat._pending_obs_init[0].item())
        o1, _, _, _, z1, _ = control(f, use_posterior=True,
                                     deterministic=True, detach_state=True)
        o2, _, _, _, z2, _ = treat(f, use_posterior=True,
                                   deterministic=True, detach_state=True)
        outs_ctrl.append((o1.clone(), z1.clone()))
        outs_trt.append((o2.clone(), z2.clone()))

pre_same = all(
    torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
    for a, b in zip(outs_ctrl[:1], outs_trt[:1]))
post_dev = (outs_trt[1][1] - outs_ctrl[1][1]).abs().max().item()
pending_consumed = treat._pending_obs_init is None
check('S5a pre-reset steps bitwise identical', pre_same)
check('S5b reset zeroed the state immediately (baseline semantics kept)',
      reset_zeroed)
check('S5c reset sample re-bootstrapped from its own frame',
      post_dev > 1e-4, f'|dz| @step2 = {post_dev:.3e}')
check('S5d pending flag consumed by one step', pending_consumed)

# ---- S6: gradients reach the bootstrap convs (GPU) ------------------------
g = build('obs', DEV_G)
copy_shared(me_zero.to(DEV_G), g)
g.train()
g.reset_state()
gf = [f.to(DEV_G) for f in feats]
out, recon, kl, _, _, _ = g(gf[0], use_posterior=True,
                            deterministic=False, detach_state=True)
(out.mean() + recon.mean() + kl).backward()
gh = g.h_init_conv.weight.grad
gz = g.z_init_conv.weight.grad
check('S6 h_init_conv grad nonzero on first backward',
      gh is not None and gh.abs().max().item() > 0,
      f'max|grad|={0.0 if gh is None else gh.abs().max().item():.3e}')
check('S6b z_init_conv grad nonzero on first backward',
      gz is not None and gz.abs().max().item() > 0,
      f'max|grad|={0.0 if gz is None else gz.abs().max().item():.3e}')

# ---- S7: parameter accounting ---------------------------------------------
n_zero = sum(p.numel() for p in me_zero.parameters())
n_obs = sum(p.numel() for p in me_obs.parameters())
expected = 2 * (C * C + C)  # in_channels->width and ->latent, plus biases
check('S7 param delta matches formula', n_obs - n_zero == expected,
      f'delta={n_obs - n_zero} expected={expected}')

n_bad = sum(1 for _, ok, _ in results if not ok)
print(f"\n[test_state_init] {len(results) - n_bad}/{len(results)} checks green")
sys.exit(1 if n_bad else 0)
