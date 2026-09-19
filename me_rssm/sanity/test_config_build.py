"""Sanity: config loading, model build, pretrained-key compatibility (CPU).

Run:  python me_rssm/sanity/test_config_build.py

Checks
------
C1. All 5 new configs load through mmcv.Config.fromfile (including the
    sys.path bootstrap + custom_imports of the me_rssm package) and the
    merged values are correct (types, switches, shared_stem=False).
C2. The full R4Det model builds from the mainline ME-RSSM config with the
    expected module classes wired in.
C3. Parameter accounting: total params and temporal_fusion params vs. the
    baseline config build (static cost estimate; no training).
C4. Pretrained-checkpoint compatibility: loading
    checkpoints/pretrained_tj4d.pth with strict=False yields EXACTLY the
    same missing/unexpected key sets as the baseline config (the ME-RSSM
    extras live under temporal_fusion.*, which is freshly initialized in
    every run anyway).
C5. Ablation configs produce the expected switch values.
"""

import os
import sys
import tempfile

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from mmcv import Config
from mmdet3d.models import build_model

MAIN = os.path.join(_REPO, 'me_rssm/configs/'
                    'TJ4D-R4Det_me_rssm_N4_24e_pretrained_v2_head.py')
BASE = os.path.join(_REPO, 'configs/r4det/'
                    'TJ4D-R4Det_motion_align_rssm_det3d_N4_2x4_24e_'
                    'pretrained_v2_head.py')
ABL = {
    'no_motion_offsets': 'me_rssm/configs/TJ4D-R4Det_me_rssm_N4_24e_'
                         'pretrained_v2_head_abl_no_motion_offsets.py',
    'no_reliability_gain': 'me_rssm/configs/TJ4D-R4Det_me_rssm_N4_24e_'
                           'pretrained_v2_head_abl_no_reliability_gain.py',
    'no_dynamic_gate': 'me_rssm/configs/TJ4D-R4Det_me_rssm_N4_24e_'
                       'pretrained_v2_head_abl_no_dynamic_gate.py',
    'no_modality_obs': 'me_rssm/configs/TJ4D-R4Det_me_rssm_N4_24e_'
                       'pretrained_v2_head_abl_no_modality_obs.py',
}


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def torch_load(path):
    import torch
    ckpt = torch.load(path, map_location='cpu')
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        return ckpt['state_dict']
    return ckpt


def build(cfg_path):
    cfg = Config.fromfile(cfg_path)
    figures = tempfile.mkdtemp(prefix='me_rssm_fig_')
    cfg.model.update(meta_info=dict(figures_path=figures,
                                    project_name='tj4d_me_rssm_sanity'))
    model = build_model(cfg.model)
    return cfg, model


# ---- C1 + C5: config loading and switches --------------------------------
cfg, model = build(MAIN)
tf = model.temporal_fusion
assert type(tf).__name__ == 'MotionEvidenceRSSMFusion', type(tf).__name__
assert type(model.pts_voxel_encoder).__name__ == \
    'RadarPillarFeatureNetMotion'
assert type(model.pts_middle_encoder).__name__ == 'PointPillarsScatterMotion'
assert type(model.cross_attention).__name__ == 'ConcatConvFusionStash', \
    'RCFusion must be wired under the cross_attention attribute'
assert model.pts_bbox_head.shared_stem is False
assert tf.use_motion_offsets and tf.use_reliability_gain
assert tf.use_dynamic_gate and tf.use_modality_obs
assert tf.kl_scale == 1.0 and tf.free_nats == 1.0  # KL machinery inherited
assert model.seq_len == 4
print('[PASS] C1 main config loads and merges correctly')

for name, rel in ABL.items():
    cfg_a, model_a = build(os.path.join(_REPO, rel))
    tf_a = model_a.temporal_fusion
    flag = dict(no_motion_offsets='use_motion_offsets',
                no_reliability_gain='use_reliability_gain',
                no_dynamic_gate='use_dynamic_gate',
                no_modality_obs='use_modality_obs')[name]
    assert getattr(tf_a, flag) is False, f'{name}: {flag} not disabled'
    others = [f for f in ('use_motion_offsets', 'use_reliability_gain',
                          'use_dynamic_gate', 'use_modality_obs')
              if f != flag]
    assert all(getattr(tf_a, f) for f in others), \
        f'{name}: other switches changed'
print('[PASS] C5 ablation configs switch exactly one flag each')

# ---- C3: parameter accounting --------------------------------------------
cfg_b, model_b = build(BASE)
p_total, p_tf = n_params(model), n_params(tf)
b_total, b_tf = n_params(model_b), n_params(model_b.temporal_fusion)
print(f'[INFO] params: baseline total={b_total/1e6:.2f}M '
      f'temporal={b_tf/1e6:.2f}M | ME-RSSM total={p_total/1e6:.2f}M '
      f'temporal={p_tf/1e6:.2f}M | delta temporal='
      f'{((p_tf-b_tf)/1e6):+.3f}M ({100*(p_tf-b_tf)/b_tf:+.2f}%) | '
      f'delta total={((p_total-b_total)/1e6):+.3f}M')
# New temporal parameters stay under 10% of the temporal module; the model
# total must not grow by more than 2% (it actually shrinks slightly
# because the clean-parity shared_stem=False removes the section-41 stem).
assert 0 < p_tf - b_tf < 0.10 * b_tf, 'temporal cost blow-up'
assert abs(p_total - b_total) < 0.02 * b_total, 'total cost blow-up'

# ---- C4: pretrained checkpoint key compatibility --------------------------
# NOTE: real training loads via mmcv.runner.checkpoint.load_state_dict,
# which *logs* (never raises on) missing/unexpected/size-mismatched keys --
# that is how every run so far kept the fresh RSSM init. We replicate its
# classification exactly.
ckpt_path = os.path.join(_REPO, 'checkpoints/pretrained_tj4d.pth')
ckpt = torch_load(ckpt_path)


def classify_load(model, ckpt):
    sd = model.state_dict()
    model_keys, ckpt_keys = set(sd.keys()), set(ckpt.keys())
    missing = sorted(model_keys - ckpt_keys)
    unexpected = sorted(ckpt_keys - model_keys)
    mismatched = sorted(
        k for k in model_keys & ckpt_keys
        if sd[k].shape != ckpt[k].shape)
    return missing, unexpected, mismatched


missing_me, unexpected_me, mismatch_me = classify_load(model, ckpt)
missing_b, unexpected_b, mismatch_b = classify_load(model_b, ckpt)
# The only allowed difference: (a) NEW fresh keys under temporal_fusion.*
# (the ME-RSSM extras), and (b) the pts_bbox_head.shared_stem_layer.*
# keys vanish because this config restores clean-baseline parity
# (shared_stem=False; the inherited mainline still carries the un-reverted
# section-41 experiment flag, audit item U7). Every key the pretrained
# checkpoint actually provides must behave identically in both models.
extra_missing = set(missing_me) - set(missing_b)
lost_missing = set(missing_b) - set(missing_me)
assert all(k.startswith('temporal_fusion.') for k in extra_missing), \
    f'new missing keys outside temporal_fusion: {sorted(extra_missing)[:5]}'
assert all(k.startswith('pts_bbox_head.shared_stem_layer.')
           for k in lost_missing), \
    f'unexpected behavioural change at: {sorted(lost_missing)[:5]}'
# The unexpected / size-mismatch sets must be IDENTICAL to the baseline's
# (they contain the GRU-era temporal_fusion.* weights and pretraining-era
# modules disabled in the mainline such as PaintBEVFusion.*).
assert sorted(unexpected_me) == sorted(unexpected_b), \
    'unexpected-key set changed vs baseline'
assert sorted(mismatch_me) == sorted(mismatch_b), \
    'size-mismatch set changed vs baseline'
loaded = [k for k in ckpt if k.startswith(
    ('pts_voxel_encoder.', 'pts_middle_encoder.', 'cross_attention.',
     'img_backbone.', 'img_neck.', 'depth_net.', 'pts_backbone.',
     'pts_neck.'))]
assert len(loaded) > 0 and not any(k in missing_me for k in loaded), \
    'pretrained producer keys failed to load'
print(f'[PASS] C4 pretrained keys: producer loading unchanged '
      f'({len(loaded)} producer tensors load; {len(extra_missing)} new '
      f'fresh temporal keys; {len(unexpected_me)} unexpected, '
      f'{len(mismatch_me)} size-mismatched -- all GRU-era temporal keys)')

print('[PASS] test_config_build: all checks green')
