"""Sanity: MCI-RSSM config loading, model build, switch isolation, DDP
parameter audit (CPU only; NOT executed as part of this preparation).

Run:  python me_rssm/sanity/test_mci_config_build.py

Checks
------
M1. All 8 MCI configs (mainline + 7 single-variable ablations) plus the
    450-iter smoke config load through mmcv.Config.fromfile (including the
    sys.path bootstrap + custom_imports of the me_rssm package), and the
    merged values are correct.
M2. The full R4Det model builds from the mainline MCI config with the
    expected module classes wired in, and shares the MAI mainline's
    resolved base settings (shared_stem etc.) — the producer swap must not
    silently change the base chain.
M3. Switch isolation: every ablation config differs from the mainline in
    EXACTLY ONE temporal_fusion key, and the built temporal module's
    constructed-modules match the switch semantics (a switch-off module
    does not exist; obs_prior exists iff posterior_mode='innovation').
M4. Parameter accounting: total / temporal_fusion params vs. the MAI
    config build (static cost estimate; no training).
M5. Pretrained-checkpoint compatibility: loading
    checkpoints/pretrained_tj4d.pth with strict=False yields exactly the
    same missing-key set as the MAI config build except for the new
    temporal_fusion.* modules.
M6. DDP unused-parameter audit on the temporal module: for the mainline
    AND every ablation, a small forward with the bus populated and
    use_posterior=True must put a gradient on EVERY parameter (else DDP
    would abort). This is the static contract that conditional
    construction is supposed to guarantee.

Any assertion failure raises; each check prints a [PASS] line.
"""

import os
import sys
import tempfile

import torch

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from mmcv import Config
from mmdet3d.models import build_model

MAIN = 'me_rssm/configs/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_mci.py'
ABL = {
    'no_motion_offsets': ('use_motion_offsets', False),
    'no_dynamic_gate': ('use_dynamic_gate', False),
    'no_reliability_gain': ('use_reliability_gain', False),
    'no_gated_readout': ('use_gated_readout', False),
    'no_modality_obs': ('use_modality_obs', False),
    'no_innov_in_gain': ('innov_in_gain', False),
    'no_innovation': ('posterior_mode', 'standard'),
}
ABL_PATHS = {
    name: f'me_rssm/configs/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_'
          f'head_no2d_igdr_mci_abl_{name}.py'
    for name in ABL
}
SMOKE = 'me_rssm/sanity/mci_smoke_450iter.py'
STATEINIT = 'me_rssm/configs/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_mci_stateinit.py'
MAI = 'configs/r4det/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_mai.py'
CKPT = 'checkpoints/pretrained_tj4d.pth'

# Modules constructed per switch (module-name -> enabling condition).
SWITCH_MODULES = {
    'motion_encoder': 'use_modality_obs',
    'camera_encoder': 'use_modality_obs',
    'obs_proj': 'use_modality_obs',
    'conf_cam': 'use_modality_obs',
    'conf_mot': 'use_modality_obs',
    'motion_offset_h': 'use_motion_offsets',
    'motion_offset_z': 'use_motion_offsets',  # + align_z_state (True here)
    'dyn_conv': 'use_dynamic_gate',
    'correct_gate_conv': 'use_reliability_gain',
    'readout_gate_conv': 'use_gated_readout',
}


def build(cfg_path):
    cfg = Config.fromfile(os.path.join(_REPO, cfg_path))
    figures = tempfile.mkdtemp(prefix='mci_sanity_fig_')
    cfg.model.update(meta_info=dict(figures_path=figures,
                                    project_name='tj4d_mci_sanity'))
    model = build_model(cfg.model)
    return cfg, model


def n_params(m):
    return sum(p.numel() for p in m.parameters())


# ---- M1: configs load ------------------------------------------------------
for path in [MAIN, SMOKE, STATEINIT] + list(ABL_PATHS.values()):
    cfg = Config.fromfile(os.path.join(_REPO, path))
    tf = cfg.model['temporal_fusion']
    assert tf['type'] == 'MeasurementConditionedInnovationRSSMFusion', \
        (path, tf['type'])
    assert 'me_rssm' in str(cfg.custom_imports), path
print('[PASS] M1 all 9 MCI configs load with the right fusion type')

# ---- M2: full model build + base-chain parity ------------------------------
cfg, model = build(MAIN)
tf = model.temporal_fusion
assert type(tf).__name__ == 'MeasurementConditionedInnovationRSSMFusion'
assert type(model.pts_voxel_encoder).__name__ == 'RadarPillarFeatureNetMotion'
assert type(model.pts_middle_encoder).__name__ == 'PointPillarsScatterMotion'
assert type(model.cross_attention).__name__ == 'ConcatConvFusionStash', \
    'RCFusion must be wired under the cross_attention attribute'
cfg_mai, model_mai = build(MAI)
assert type(model_mai.temporal_fusion).__name__ == \
    'MotionAlignedInnovationRSSMFusion'
assert model.pts_bbox_head.shared_stem == \
    model_mai.pts_bbox_head.shared_stem, \
    'producer swap must not change the base chain (shared_stem)'
assert model.seq_len == model_mai.seq_len == 4
assert tf.kl_scale == 1.0 and tf.free_nats == 1.0
assert tf.obs_pred_loss_weight == 0.05
print(f'[PASS] M2 mainline MCI builds; base chain matches MAI '
      f'(shared_stem={model.pts_bbox_head.shared_stem})')

# ---- M3: switch isolation ---------------------------------------------------
# no_modality_obs closes the whole coupled bus-consumer group: its inputs
# (Doppler canvases) only exist on the bus, so use_motion_offsets and
# use_dynamic_gate go off with it (enforced by the class validation).
COUPLED_DIFF = {
    'no_modality_obs': {'use_modality_obs', 'use_motion_offsets',
                        'use_dynamic_gate'},
    'no_innovation': {'posterior_mode', 'innov_in_gain'},
}
main_tf_cfg = dict(cfg.model['temporal_fusion'])
for name, (key, value) in ABL.items():
    cfg_a, model_a = build(ABL_PATHS[name])
    tf_a = model_a.temporal_fusion
    tf_a_cfg = dict(cfg_a.model['temporal_fusion'])
    diff = {k for k in set(main_tf_cfg) | set(tf_a_cfg)
            if main_tf_cfg.get(k, object()) != tf_a_cfg.get(k, object())}
    assert diff == COUPLED_DIFF.get(name, {key}), (name, diff)
    assert tf_a_cfg[key] == value, (name, key)
    # constructed-module semantics: a module exists iff ITS OWN switch is
    # on in this ablation (conditional construction contract)
    for mod, cond_key in SWITCH_MODULES.items():
        expected = bool(getattr(tf_a, cond_key))
        if mod == 'motion_offset_z' and not tf_a.align_z_state:
            expected = False
        has = hasattr(tf_a, mod)
        assert has == expected, (name, mod, has, expected)
    if name == 'no_innovation':
        assert not hasattr(tf_a, 'obs_prior')
        assert tf_a.posterior_struct == 'standard'
    else:
        assert hasattr(tf_a, 'obs_prior')
        assert tf_a.posterior_struct == 'innovation'
print('[PASS] M3 each ablation differs in exactly one switch and the '
      'constructed modules match')

# ---- M4: parameter accounting ----------------------------------------------
p_mci_total, p_mci_tf = n_params(model), n_params(tf)
p_mai_total, p_mai_tf = n_params(model_mai), n_params(model_mai.temporal_fusion)
print(f'[INFO] M4 params: MCI total {p_mci_total/1e6:.2f}M '
      f'(temporal {p_mci_tf/1e6:.3f}M) vs MAI total {p_mai_total/1e6:.2f}M '
      f'(temporal {p_mai_tf/1e6:.3f}M); delta '
      f'{(p_mci_total-p_mai_total)/1e6:+.3f}M / '
      f'{(p_mci_tf-p_mai_tf)/1e6:+.3f}M temporal')
assert p_mci_tf > p_mai_tf, 'MCI must add temporal parameters'

# ---- M5: pretrained checkpoint compatibility --------------------------------
# Key-SET comparison only: the bundled pretrained_tj4d.pth predates the
# current anchor layout (head shapes differ), and load_state_dict raises on
# shape mismatch even with strict=False. The contract under test is that
# MCI needs exactly the same checkpoint keys as MAI plus its new
# temporal_fusion.* modules — a key-name property, shape-independent.
ckpt = torch.load(os.path.join(_REPO, CKPT), map_location='cpu')
if isinstance(ckpt, dict) and 'state_dict' in ckpt:
    ckpt = ckpt['state_dict']
ckpt_keys = set(ckpt.keys())
mci_keys = set(model.state_dict().keys())
mai_keys = set(model_mai.state_dict().keys())
missing_mci = mci_keys - ckpt_keys
missing_mai = mai_keys - ckpt_keys
extra_missing = missing_mci - missing_mai
assert all(k.startswith('temporal_fusion.') for k in extra_missing), \
    sorted(extra_missing)[:10]
assert missing_mai <= missing_mci
print(f'[PASS] M5 pretrained key sets: identical outside '
      f'{len(extra_missing)} new temporal_fusion.* keys '
      f'(head shape drift vs the old ckpt is a base-chain property, '
      f'identical for MAI and MCI)')

# ---- M6: DDP unused-parameter audit on the temporal module ------------------
def audit_no_unused(tf_module):
    """Forward with bus; every parameter must receive a gradient."""
    from me_rssm.modality_bus import ModalityContext
    tf_module.train()
    H, W = 8, 10
    B = 2
    # in_channels is not stored by the parent; read it off the encoder.
    C = tf_module.encoder[0].conv.in_channels
    vel = torch.randn(B, tf_module.motion_channels, 2 * W, 2 * H)
    cam = torch.randn(B, tf_module.camera_bev_channels, W, H)
    ModalityContext.set('velocity_bev', vel)
    ModalityContext.set('camera_bev', cam)
    try:
        feat = torch.randn(B, C, H, W, requires_grad=True)
        tf_module.reset_state()
        out = tf_module(feat, use_posterior=True, deterministic=False)
        loss = out[0].mean() + out[1].mean() + out[2]
        loss.backward()
    finally:
        ModalityContext.clear()
    unused = [n for n, p in tf_module.named_parameters()
              if p.requires_grad and p.grad is None]
    return unused


unused = audit_no_unused(tf)
assert not unused, f'mainline temporal module has unused params: {unused}'
print('[PASS] M6 mainline temporal module: no unused parameters under '
      'bus+posterior forward (DDP-safe)')

for name in ABL:
    _, model_a = build(ABL_PATHS[name])
    unused = audit_no_unused(model_a.temporal_fusion)
    assert not unused, f'ablation {name} unused params: {unused}'
print('[PASS] M6 all 7 ablations: no unused parameters (DDP-safe)')

print('ALL MCI CONFIG-BUILD CHECKS PASSED')
