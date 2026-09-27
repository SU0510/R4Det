"""Compare candidate reconstruction targets by how sample-discriminative they are.

A target whose variance is dominated by an across-sample template can be
reconstructed by a sample-invariant z, so it cannot anchor z to the current
observation no matter how well the reconstruction loss itself descends. This
tool measures that split for candidate target definitions on cached BEV
features, without training.

Usage (physical GPU 5):

    CUDA_VISIBLE_DEVICES=5 python tools/probe_recon_target.py \
        --config configs/r4det/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_lowdim_z.py \
        --checkpoint /data/lurui/work_dirs/<run>/epoch_12.pth \
        --start-index 3 --limit 16 --output /tmp/recon_target.json

For each candidate target definition we standardize exactly the way the training
loss would, then decompose per-element variance into:
  shared  = energy of the across-sample mean (a constant predictor can fit it)
  specific= residual that actually differs between samples
A target dominated by `shared` can be reconstructed to a low loss by a
sample-invariant z, so it cannot anchor z to the observation.
"""
import argparse, json, os, sys, types, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/lurui/workspace/R4Det')
import numpy as np, torch
import torch.nn.functional as F
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_detector
from tools.diagnose_z_utilization import build_batch, run_sequence, select_indices

CANDS = ['pool16_globalstd', 'pool16_percellstd', 'pool32_globalstd',
         'fullres_globalstd', 'pool16_channelstd']

def make_target(feat, kind):
    if kind == 'fullres_globalstd':
        t = feat
    elif kind == 'pool16_globalstd':
        t = F.adaptive_avg_pool2d(feat, (16, 16))
    elif kind == 'pool32_globalstd':
        t = F.adaptive_avg_pool2d(feat, (32, 32))
    elif kind == 'pool16_percellstd':
        # Deliberately *no* standardization: the point of this variant is to
        # test whether the shared template is introduced by the normalizer.
        return F.adaptive_avg_pool2d(feat, (16, 16))
    elif kind == 'pool16_channelstd':
        t = F.adaptive_avg_pool2d(feat, (16, 16))
        m = t.mean(dim=(2, 3), keepdim=True)
        s = t.std(dim=(2, 3), keepdim=True).clamp_min(1e-4)
        return (t - m) / s
    m = t.mean(dim=(1, 2, 3), keepdim=True)
    s = t.std(dim=(1, 2, 3), keepdim=True).clamp_min(1e-4)
    return (t - m) / s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--start-index', type=int, default=3)
    ap.add_argument('--limit', type=int, default=16)
    ap.add_argument('--output', required=True)
    a = ap.parse_args()
    torch.cuda.set_device(0); torch.manual_seed(0); np.random.seed(0)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    cfg = Config.fromfile(a.config); cfg.model.pretrained = None; cfg.model.train_cfg = None
    cfg.model['meta_info'] = {'figures_path': '/tmp/r4det_td_figs', 'project_name': 'td'}
    os.makedirs(cfg.model['meta_info']['figures_path'], exist_ok=True)
    cfg.data.val.pop('samples_per_gpu', None); cfg.data.val.pop('workers_per_gpu', None)
    ds = build_dataset(cfg.data.val)
    model = build_detector(cfg.model, train_cfg=None, test_cfg=None)
    load_checkpoint(model, a.checkpoint, map_location='cpu')
    model = model.cuda().eval()
    fusion = model.temporal_fusion
    cap = []
    orig = fusion.forward
    def wrapper(self, feat, *args, **kwargs):
        cap.append(feat.detach().float().cpu())
        return orig(feat, *args, **kwargs)
    fusion.forward = types.MethodType(wrapper, fusion)
    class A: pass
    args = A(); args.require_valid_history = True
    args.start_index = a.start_index; args.limit = a.limit; args.min_valid_history = 2
    idx = select_indices(ds, args)
    feats = []
    for i in idx:
        cap.clear()
        run_sequence(model, build_batch(ds, [i], 0), z_mode='normal')
        feats.append(cap[-1][0].unsqueeze(0))   # (1,C,H,W)
    res = {'indices': idx, 'n': len(idx), 'designs': {}}
    for kind in CANDS:
        T = torch.stack([make_target(f, kind) for f in feats])
        total = T.pow(2).mean().item()
        mean_t = T.mean(0, keepdim=True)
        shared = mean_t.pow(2).mean().item()
        spec = (T - mean_t).pow(2).mean().item()
        # constant-predictor loss on this target
        res['designs'][kind] = {
            'shape': list(T.shape[1:]),
            'total_var': total,
            'shared_template_frac': shared / total if total else float('nan'),
            'sample_specific_frac': spec / total if total else float('nan'),
            'cross_sample_mse_mean': spec,
        }
    json.dump(res, open(a.output, 'w'), indent=2)
    print('%-22s %-16s %8s %10s %10s' % ('design', 'shape', 'total', 'shared%', 'specific%'))
    for k, v in res['designs'].items():
        print('%-22s %-16s %8.3f %9.1f%% %9.1f%%'
              % (k, str(tuple(v['shape'])), v['total_var'],
                 100*v['shared_template_frac'], 100*v['sample_specific_frac']))

if __name__ == '__main__':
    main()
