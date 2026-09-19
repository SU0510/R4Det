"""State-init probe: quantify the dead sequence-start state on a TRAINED
baseline (zero-training diagnostic; error/mechanism analysis).

Run:  CUDA_VISIBLE_DEVICES=4 python me_rssm/sanity/diag_state_probe.py \
          --checkpoint work_dirs/run10_headv2_multiseed/seed_0/epoch_16.pth

Why
---
06_second_layer P1 claims the baseline RSSM starts every sequence from the
dead state h_0 = z_0 = 0, so the first transition's deform alignment acts
on zeros and the first frame's history path contributes nothing. This probe
verifies that claim on a *trained* checkpoint (Run 10 clean seed0, ep16)
by instrumenting the temporal module during real eval forwards and
grouping every temporal-fusion call by its position t within the sequence:

  t=0  entry state is the freshly reset (zero) state
  t=1..N-1  entry state is the carried-over state

Recorded per call:
  h_in_norm / z_in_norm   L2 norm of the carried-in state (before init)
  h_out_norm              L2 norm of the produced h_t
  mu_diff_sq              posterior/prior mean divergence (stats dict)
  post_std / prior_std    stochastic-branch widths (stats dict)
  off_h_norm / off_z_norm L2 norm of the deform offset generator outputs

If P1's motivation holds: t=0 shows h_in_norm == 0 and materially
different mu_diff_sq / offset magnitudes than t>=1. The results feed the
P1 motivation figure/table (docs/10) and the decision on whether the
stateinit variant is worth a training slot.
"""

import argparse
import json
import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import training_old_version  # noqa: E402
training_old_version.use_old_version()

import torch  # noqa: E402
from mmcv import Config  # noqa: E402
from mmcv.parallel import MMDataParallel, collate  # noqa: E402
from mmcv.runner import load_checkpoint  # noqa: E402

import me_rssm  # noqa: F401,E402
import tempfile  # noqa: E402
from mmdet3d.datasets import build_dataset  # noqa: E402
from mmdet3d.models import build_model  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument('--config', default=os.path.join(
    _REPO, 'configs/r4det',
    'TJ4D-R4Det_motion_align_rssm_det3d_N4_2x4_24e_pretrained_v2_head.py'))
ap.add_argument('--checkpoint', default=os.path.join(
    _REPO, 'work_dirs/run10_headv2_multiseed/seed_0/epoch_16.pth'))
ap.add_argument('--indices', type=int, nargs='+',
                default=[3, 10, 50, 100, 200, 400])
ap.add_argument('--out', default=os.path.join(
    _REPO, 'me_rssm/figures/state_probe.json'))
args = ap.parse_args()

cfg = Config.fromfile(args.config)
cfg.model.update(meta_info=dict(
    figures_path=tempfile.mkdtemp(prefix='me_rssm_probe_fig_'),
    project_name='tj4d_state_probe'))
cfg.model.pretrained = None
# The committed mainline config still carries shared_stem=True from the
# closed section-41 experiment (audit U7); Run 10 multiseed trained with
# shared_stem=False, so match it or the head keys won't load.
cfg.model.pts_bbox_head.shared_stem = False
cfg.data.test.test_mode = True
model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
ckpt = load_checkpoint(model, args.checkpoint, map_location='cpu')
model = MMDataParallel(model.cuda(), device_ids=[0])
model.eval()
tf = model.module.temporal_fusion
assert type(tf).__name__ == 'MotionAlignedRSSMFusion', type(tf).__name__

records = []
call_idx = {'t': -1}


def norm(x):
    return None if x is None else float(x.detach().float().pow(2).sum().sqrt())


orig_forward = tf.forward
orig_reset = tf.reset_state


def counting_reset():
    call_idx['t'] = -1
    return orig_reset()


tf.reset_state = counting_reset


def probing_forward(feat, **kw):
    call_idx['t'] += 1
    h_in, z_in = tf.h_state, tf.z_state
    h_in_norm, z_in_norm = norm(h_in), norm(z_in)
    out = orig_forward(feat, **kw)
    rec = {'t': call_idx['t'], 'h_in_norm': h_in_norm, 'z_in_norm': z_in_norm,
           'h_out_norm': norm(out[3]), 'z_out_norm': norm(out[4])}
    stats = out[5] or {}
    for k in ('stat_mu_diff_sq', 'stat_posterior_std', 'stat_prior_std',
              'stat_clamped_ratio'):
        if k in stats:
            v = stats[k]
            rec[k.replace('stat_', '')] = float(v) \
                if torch.is_tensor(v) else float(v)
    rec['off_h_norm'] = norm(getattr(tf, '_last_off_h', None))
    rec['off_z_norm'] = norm(getattr(tf, '_last_off_z', None))
    records.append(rec)
    return out


tf.forward = probing_forward

# capture the deform offset generator outputs at module level
for name, key in [('align_h_offset_mask', '_last_off_h'),
                  ('align_z_offset_mask', '_last_off_z')]:
    mod = getattr(tf, name)
    orig = mod.forward

    def mk(orig, key):
        def f(*a, **k):
            out = orig(*a, **k)
            setattr(tf, key, out.detach())
            return out
        return f
    mod.forward = mk(orig, key)

dataset = build_dataset(cfg.data.test, dict(test_mode=True))
with torch.no_grad():
    for idx in args.indices:
        data = collate([dataset[idx]], samples_per_gpu=1)
        model(return_loss=False, rescale=True, **data)
        parts = []
        for r in records[-4:]:
            h_in = 'None' if r['h_in_norm'] is None \
                else f"{r['h_in_norm']:.2f}"
            mu = r.get('mu_diff_sq')
            mu_s = '--' if mu is None else f'{mu:.4f}'
            parts.append(f"t{r['t']}: h_in={h_in} mu_diff={mu_s}")
        print(f'[probe] sample idx={idx}: ' + ' | '.join(parts))

# aggregate by sequence position
seq_len = max(r['t'] for r in records) + 1
summary = {}
for t in range(seq_len):
    rows = [r for r in records if r['t'] == t]
    agg = {}
    for k in rows[0]:
        if k == 't':
            continue
        vals = [r[k] for r in rows if r[k] is not None]
        agg[k] = sum(vals) / len(vals) if vals else None
    summary[f't{t}'] = {'n': len(rows), **agg}

os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, 'w', encoding='utf-8') as f:
    json.dump({'checkpoint': args.checkpoint, 'n_samples': len(args.indices),
               'per_position_mean': summary, 'raw': records}, f, indent=2)

print('\n==== per-position means (over '
      f'{len(args.indices)} samples) ====')
hdr = ['pos', 'n', 'h_in_norm', 'z_in_norm', 'h_out_norm', 'mu_diff_sq',
       'post_std', 'off_h_norm', 'off_z_norm']
print(' '.join(f'{h:>12}' for h in hdr))
for t in range(seq_len):
    s = summary[f't{t}']
    row = [f't{t}', str(s['n'])]
    for k in hdr[2:]:
        v = s.get(k)
        row.append(f'{v:.4f}' if isinstance(v, float) else '--')
    print(' '.join(f'{c:>12}' for c in row))
print(f'\n[probe] saved -> {args.out}')
