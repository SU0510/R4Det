"""Does z_t anchor to the observation, or only to a sample-invariant constant?

Replays a frozen checkpoint with z_t left alone, zeroed, and replaced by another
sample's complete posterior mean, reporting the reconstruction loss each time.

The three outcomes separate cleanly:
  zero   high, shuffle high  -> z_t carries sample-specific observation content
  zero   high, shuffle flat  -> z_t is a shared constant the decoder needs but
                                any sample's value will do
  zero  ~flat                -> the decoder ignores z_t entirely

Only the first case means the observation-likelihood term is anchoring z_t to
the current frame. The middle case looks healthy under a zero-ablation and is
still degenerate, so a zero test alone is not sufficient evidence.

Usage (physical GPU 5):

    CUDA_VISIBLE_DEVICES=5 python tools/probe_recon_anchor.py \
        --config configs/r4det/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_lowdim_z.py \
        --checkpoint /data/lurui/work_dirs/<run>/epoch_12.pth \
        --start-index 3 --limit 4 --output /tmp/recon_anchor.json
"""
import argparse, json, os, sys, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '/home/lurui/workspace/R4Det')
import numpy as np, torch
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_detector
from tools.diagnose_z_utilization import (
    build_batch, run_sequence, select_indices)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--start-index', type=int, default=3)
    ap.add_argument('--limit', type=int, default=4)
    ap.add_argument('--output', required=True)
    a = ap.parse_args()
    torch.cuda.set_device(0)
    torch.manual_seed(0); np.random.seed(0)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    cfg = Config.fromfile(a.config)
    cfg.model.pretrained = None; cfg.model.train_cfg = None
    cfg.model['meta_info'] = {'figures_path': '/tmp/r4det_recon_probe_figs',
                              'project_name': 'recon-probe'}
    os.makedirs(cfg.model['meta_info']['figures_path'], exist_ok=True)
    cfg.data.val.pop('samples_per_gpu', None); cfg.data.val.pop('workers_per_gpu', None)
    ds = build_dataset(cfg.data.val)
    model = build_detector(cfg.model, train_cfg=None, test_cfg=None)
    load_checkpoint(model, a.checkpoint, map_location='cpu')
    model = model.cuda().eval()
    class A: pass
    args = A(); args.require_valid_history=True; args.start_index=a.start_index; args.limit=a.limit; args.min_valid_history=2
    idx = select_indices(ds, args)
    out = {}
    for i in idx:
        batch = build_batch(ds, [i], 0)
        n = run_sequence(model, batch, z_mode='normal')
        shuf_ref = idx[(idx.index(i) + 1) % len(idx)]
        z = run_sequence(model, batch, z_mode='zero_z')
        sref = run_sequence(model, build_batch(ds, [shuf_ref], 0), z_mode='normal')
        sh = run_sequence(model, batch, z_mode='replace_z', replacement_z=sref['z_t'])
        out[i] = {
            'normal_recon': n['rssm_stats'].get('stat_recon_mse'),
            'zero_recon': z['rssm_stats'].get('stat_recon_mse'),
            'normal_future': n['rssm_stats'].get('stat_future_mse'),
            'zero_future': z['rssm_stats'].get('stat_future_mse'),
            'normal_kl_raw': n['rssm_stats'].get('stat_kl_raw_mean'),
            'shuffle_ref': shuf_ref,
            'shuffle_recon': sh['rssm_stats'].get('stat_recon_mse'),
        }
    json.dump({'indices': idx, 'per_sample': out}, open(a.output,'w'), indent=2)
    for i in idx:
        r=out[i]
        print('idx %4d  recon normal %.5f  zero %.5f (%+.1f%%)  shuffle %.5f (%+.2f%%)'
              % (i, r['normal_recon'], r['zero_recon'],
                 100.0*(r['zero_recon']-r['normal_recon'])/r['normal_recon'],
                 r['shuffle_recon'],
                 100.0*(r['shuffle_recon']-r['normal_recon'])/r['normal_recon']))
    print('[json]', a.output)

if __name__ == '__main__':
    main()
