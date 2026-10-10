"""Candidate A wiring probe: identity at init + real gradient pressure.

Run (one GPU, read-only, no training, no checkpoint written):
    CUDA_VISIBLE_DEVICES=5 python3 me_rssm/sanity/probe_semreadout_wiring.py \
        --checkpoint /data/lurui/work_dirs/fgfull_N4_no2d_igdr_2x4_24e_seed0/epoch_14.pth \
        --output /data/lurui/work_dirs/semA_wiring_probe.json

Checks (docs/rssm_candidate_A_prereg_20261010.md §5 S0):

  S0-b  IDENTITY. The A-class model loaded with the BASELINE checkpoint must
        produce bit-identical class/regression logits to the baseline model on
        the same val sequence: the readout is zero-initialised and the
        auxiliary head is off the detection path. This is the end-to-end proof
        that the new hooks (fusion -> detector -> conv_cls, including the
        (216,248) -> (248,216) grid contract and the invalid-sample zeroing)
        change nothing until they are trained. It also reports the checkpoint
        load diff, so a silently missing baseline key cannot hide.
  S0-d  PRESSURE. ||d loss_rssm_sem / d z_t|| / ||d total / d z_t|| >= 5%.
        The mainline reconstruction term is 0.18% of the loss and demonstrably
        fails to anchor z (§70.2); a new supervision term of the same effective
        scale is worthless, so the pressure is measured BEFORE any epoch is
        spent, and the answer is not allowed to come from a loss weight.
  S0-e  LIVENESS / SEPARATION. ||d total / d W_cls|| > 0 proves the residual
        reaches the class logits (not merely the log dict), and
        ||d loss_rssm_sem / d W_cls|| == 0 keeps supervision and readout
        separate, which is what makes the C1/C2 ablations meaningful.

No AP is measured and no performance claim is made here.
"""

import argparse
import importlib.util
import json
import os
import sys
import tempfile

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import torch  # noqa: E402

ARM_CFG = os.path.join(
    _REPO, 'me_rssm/configs/'
    'TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_semreadout.py')
BASE_CFG = os.path.join(
    _REPO, 'configs/r4det/'
    'TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr.py')


def _load(name, relative_path):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_REPO, relative_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dz = _load('dz', os.path.join('tools', 'diagnose_z_utilization.py'))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--arm-config', default=ARM_CFG)
    p.add_argument('--base-config', default=BASE_CFG)
    p.add_argument('--checkpoint', required=True,
                   help='baseline checkpoint (e.g. seed0 epoch_14.pth)')
    p.add_argument('--pretrained',
                   default=os.path.join(_REPO,
                                        'checkpoints/pretrained_tj4d.pth'),
                   help='hot-start weights for the pressure test')
    p.add_argument('--gpu-id', type=int, default=0)
    p.add_argument('--val-index', type=int, default=3)
    p.add_argument('--train-index', type=int, default=3)
    p.add_argument('--output', default='', help='optional JSON dump path')
    return p.parse_args()


def _load_cfg(path, figures_prefix):
    from mmcv import Config
    cfg = Config.fromfile(path)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model['meta_info'] = {
        'figures_path': tempfile.mkdtemp(prefix=figures_prefix),
        'project_name': 'TJ4D-semA-probe',
    }
    return cfg


def _build_eval(cfg_path, checkpoint, device):
    from mmcv.runner import load_checkpoint
    from mmdet3d.models import build_detector

    cfg = _load_cfg(cfg_path, 'semA_probe_fig_')
    model = build_detector(cfg.model, train_cfg=None, test_cfg=None)
    load = load_checkpoint(model, checkpoint, map_location='cpu',
                           strict=False, logger=None)
    model = model.to(device).eval()
    return cfg, model, {'missing_keys': sorted(getattr(load, 'missing_keys', [])),
                        'unexpected_keys': sorted(
                            getattr(load, 'unexpected_keys', []))}


def _val_rollout(model, dataset, index):
    """Val-mode rollout of one sequence; returns head logits + residual."""
    n = model.seq_len
    batch = dz.build_batch(dataset, [index], torch.cuda.current_device())
    frame_points, frame_img, frame_img_metas, frame_valid = \
        dz.build_sequence_frames(batch)
    fusion = model.temporal_fusion
    with torch.no_grad():
        fusion.reset_state()
        for t in range(n - 1):
            valid_t = frame_valid[t]
            if valid_t.any():
                model.extract_feat(
                    frame_points[t], frame_img[t], frame_img_metas[t],
                    is_valid_mask=valid_t, feat_or_dict=0,
                    curr_frame_supervision=False)
            if (~valid_t).any():
                fusion.reset_for_samples(~valid_t)
        last_hist_valid = frame_valid[n - 2] if n >= 2 else torch.ones(
            1, dtype=torch.bool, device=frame_img[n - 1].device)
        feature_dict = model.extract_feat(
            frame_points[n - 1], frame_img[n - 1], frame_img_metas[n - 1],
            is_valid_mask=last_hist_valid, feat_or_dict=1)
        residual = feature_dict.get('rssm_cls_logit_residual')
        outs = model.pts_bbox_head(
            feature_dict['pts_feats'], cls_logit_residual=residual)
    cls_maps = [t.detach().float().cpu() for t in outs[0]]
    reg_maps = [t.detach().float().cpu() for t in outs[1]]
    return (cls_maps, reg_maps,
            None if residual is None else residual.detach().float().cpu())


def _worst_diff(a_maps, b_maps):
    worst = 0.0
    for a, b in zip(a_maps, b_maps):
        assert a.shape == b.shape, f'{tuple(a.shape)} vs {tuple(b.shape)}'
        worst = max(worst, float((a - b).abs().max()))
    return worst


def main():
    from mmcv.parallel import collate
    from mmcv.runner import load_checkpoint
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_detector

    args = parse_args()
    torch.cuda.set_device(args.gpu_id)
    torch.manual_seed(0)
    device = torch.device('cuda', args.gpu_id)
    report = {
        'probe': 'candidate A wiring (identity + gradient pressure)',
        'checkpoint': os.path.abspath(args.checkpoint),
        'arm_config': os.path.abspath(args.arm_config),
        'base_config': os.path.abspath(args.base_config),
        'gates': {'S0b_identity': None, 'S0d_pressure_5pct': None,
                  'S0e_readout_live': None, 'S0e_sem_stays_off_readout': None},
    }

    # -------------------- S0-b: identity vs the baseline model --------------
    arm_cfg, arm_model, arm_load = _build_eval(
        args.arm_config, args.checkpoint, device)
    new_prefixes = ('temporal_fusion.sem_head', 'temporal_fusion.z_cls_logits')
    report['arm_checkpoint_load'] = arm_load
    report['arm_new_keys'] = [k for k in arm_load['missing_keys']
                              if k.startswith(new_prefixes)]
    report['arm_other_missing_keys'] = [
        k for k in arm_load['missing_keys'] if k not in report['arm_new_keys']]
    fusion = arm_model.temporal_fusion
    report['readout_zero_at_load'] = bool(
        float(fusion.z_cls_logits.weight.detach().abs().sum()) == 0.0
        and float(fusion.z_cls_logits.bias.detach().abs().sum()) == 0.0)

    val_cfg = dict(arm_cfg.data.val)
    val_cfg.pop('samples_per_gpu', None)
    val_cfg.pop('workers_per_gpu', None)
    val_ds = build_dataset(val_cfg)
    arm_cls, arm_reg, arm_res = _val_rollout(arm_model, val_ds, args.val_index)
    report['identity'] = {
        'val_index': args.val_index,
        'residual_emitted': arm_res is not None,
        'residual_absmax': None if arm_res is None
        else float(arm_res.abs().max()),
        'residual_grid': None if arm_res is None
        else list(arm_res.shape[-2:]),
    }
    del arm_model
    torch.cuda.empty_cache()

    _, base_model, base_load = _build_eval(
        args.base_config, args.checkpoint, device)
    report['base_checkpoint_load'] = base_load
    base_cls, base_reg, _ = _val_rollout(base_model, val_ds, args.val_index)
    report['identity']['max_abs_cls_diff'] = _worst_diff(arm_cls, base_cls)
    report['identity']['max_abs_reg_diff'] = _worst_diff(arm_reg, base_reg)
    report['identity']['same_missing_keys_apart_from_new_modules'] = bool(
        sorted(set(base_load['missing_keys']) - set(report['arm_new_keys']))
        == sorted(set(arm_load['missing_keys']) - set(report['arm_new_keys'])))
    report['gates']['S0b_identity'] = bool(
        report['identity']['max_abs_cls_diff'] == 0.0
        and report['identity']['max_abs_reg_diff'] == 0.0
        and report['readout_zero_at_load']
        and report['identity']['residual_emitted']
        and report['identity']['residual_absmax'] == 0.0
        and report['identity']['same_missing_keys_apart_from_new_modules'])
    del base_model
    torch.cuda.empty_cache()

    # ---------------- S0-d / S0-e: pressure and liveness --------------------
    cfg = _load_cfg(args.arm_config, 'semA_grad_fig_')
    model = build_detector(cfg.model, cfg.train_cfg, cfg.test_cfg)
    load_checkpoint(model, args.pretrained, map_location='cpu', strict=False,
                    logger=None)
    model = model.to(device).train()

    train_cfg = dict(cfg.data.train)
    train_cfg.pop('samples_per_gpu', None)
    train_cfg.pop('workers_per_gpu', None)
    train_ds = build_dataset(train_cfg)
    batch = collate([train_ds[args.train_index]], samples_per_gpu=1)

    fusion = model.temporal_fusion
    losses = model(return_loss=True, **batch)
    sem = losses.get('loss_rssm_sem')
    report['loss_keys'] = sorted(k for k in losses if k.startswith('loss_'))
    report['loss_rssm_sem'] = None if sem is None else float(sem)
    trainable = {k: v for k, v in losses.items()
                 if k.startswith('loss_') and torch.is_tensor(v)
                 and v.numel() == 1 and v.requires_grad}
    report['trainable_loss_keys'] = sorted(trainable)
    total = sum(trainable.values())

    z_t = fusion._last_z_t
    weight = fusion.z_cls_logits.weight
    if z_t is None or not z_t.requires_grad:
        raise RuntimeError('z_t was not captured in the graph')

    def _norm(grads):
        return 0.0 if grads[0] is None else float(grads[0].norm())

    g_sem_z = _norm(torch.autograd.grad(sem, z_t, retain_graph=True,
                                        allow_unused=True))
    g_total_z = _norm(torch.autograd.grad(total, z_t, retain_graph=True,
                                          allow_unused=True))
    g_sem_w = _norm(torch.autograd.grad(sem, weight, retain_graph=True,
                                        allow_unused=True))
    g_total_w = _norm(torch.autograd.grad(total, weight, retain_graph=True,
                                          allow_unused=True))
    ratio = (g_sem_z / g_total_z) if g_total_z > 0 else float('nan')
    report['pressure'] = {
        'grad_sem_on_z': g_sem_z,
        'grad_total_on_z': g_total_z,
        'ratio': ratio,
        'grad_sem_on_W_cls': g_sem_w,
        'grad_total_on_W_cls': g_total_w,
        'loss_rssm_recon': (float(losses['loss_rssm_recon'])
                            if 'loss_rssm_recon' in losses else None),
        'loss_rssm_kl': (float(losses['loss_rssm_kl'])
                         if 'loss_rssm_kl' in losses else None),
        'loss_cls': (float(losses['loss_cls'])
                     if 'loss_cls' in losses else None),
    }
    report['gates']['S0d_pressure_5pct'] = bool(ratio >= 0.05)
    report['gates']['S0e_readout_live'] = bool(g_total_w > 0.0)
    report['gates']['S0e_sem_stays_off_readout'] = bool(g_sem_w == 0.0)
    report['stats'] = {k: float(v) for k, v in losses.items()
                       if k.startswith('stat_') and torch.is_tensor(v)
                       and v.numel() == 1}
    report['peak_memory_gib'] = (
        torch.cuda.max_memory_allocated(device) / 2 ** 30)

    out = json.dumps(report, indent=2, sort_keys=True)
    print(out)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)) or '.',
                    exist_ok=True)
        with open(args.output, 'w') as handle:
            handle.write(out + '\n')
        print(f'[json] {os.path.abspath(args.output)}')
    if not all(v is True for v in report['gates'].values()):
        print('\n[WARN] at least one S0 gate did not pass — see the JSON above')


if __name__ == '__main__':
    main()
