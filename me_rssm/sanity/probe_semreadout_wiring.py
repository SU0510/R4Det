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

The first batch of the train split is a boundary sample whose LAST history
frame is invalid; the detector deliberately zeroes the residual there, so
S0-e must select a batch whose last history frame IS valid (99.25% of the
split). The report records both the mask readout of the used batch and a
400-sample sweep, so a masked batch can never be read as a dead path.

No AP is measured and no performance claim is made here.
"""

import argparse
import importlib.util
import json
import os
import sys
import tempfile

import numpy as np

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
    p.add_argument('--train-index', type=int, default=0,
                   help='fallback batch offset if no valid-history batch '
                        'is found within --max-probe-batches')
    p.add_argument('--max-probe-batches', type=int, default=64,
                   help='how many batches to scan for a valid last '
                        'history frame')
    p.add_argument('--output', default='', help='optional JSON dump path')
    return p.parse_args()


def _load_cfg(path, figures_prefix, keep_train_cfg=False):
    from mmcv import Config
    cfg = Config.fromfile(path)
    cfg.model.pretrained = None
    if not keep_train_cfg:
        # Eval builds do not need the assigner/sampler (probe_z_util precedent).
        # The train-step stage must keep it: this repo puts train_cfg INSIDE
        # model (tools/train.py passes cfg.get('train_cfg'), which is None).
        cfg.model.train_cfg = None
    cfg.model['meta_info'] = {
        'figures_path': tempfile.mkdtemp(prefix=figures_prefix),
        'project_name': 'TJ4D-semA-probe',
    }
    return cfg


def _build_eval(cfg_path, checkpoint, device):
    """Build an eval model and report the checkpoint diff EXPLICITLY.

    mmcv's ``load_checkpoint`` returns a ``missing_keys`` field whose meaning is
    "missing in the SOURCE state dict", and it prints the model-side gaps
    through the logger only. The gate needs the model-side list (which
    parameters of THIS model the checkpoint cannot fill), so it is computed
    from the state dicts directly instead of trusting that tuple.
    """
    from mmcv.runner import load_checkpoint
    from mmdet3d.models import build_detector

    cfg = _load_cfg(cfg_path, 'semA_probe_fig_')
    model = build_detector(cfg.model, train_cfg=None, test_cfg=None)
    load = load_checkpoint(model, checkpoint, map_location='cpu',
                           strict=False, logger=None)
    model = model.to(device).eval()
    state = torch.load(checkpoint, map_location='cpu').get('state_dict', {})
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state.keys())
    return cfg, model, {
        'not_in_checkpoint': sorted(model_keys - ckpt_keys),
        'not_in_model': sorted(ckpt_keys - model_keys),
        'loader_missing_keys': sorted(getattr(load, 'missing_keys', []) or []),
        'loader_unexpected_keys': sorted(
            getattr(load, 'unexpected_keys', []) or []),
    }


def _iter_frames(metas_container):
    """Per-sample meta lists from a collated ``DataContainer``.

    ``collate`` wraps the non-stacked ``img_metas`` container in the batch
    dimension (``[[frames_0, frames_1, ...]]``); ``R4Det`` receives the
    unwrapped ``[frames_0, frames_1, ...]`` from MMDataParallel. Peel that
    wrapper here so the probe inspects exactly what the detector sees.
    """
    data = metas_container.data
    while (isinstance(data, list) and len(data) == 1
           and isinstance(data[0], list) and data[0]
           and isinstance(data[0][0], list)):
        data = data[0]
    return data


def _last_history_coverage(dataset, sample=400, seed=0):
    """Fraction of samples whose LAST history frame is valid.

    The class-logit readout is zeroed for samples whose most recent history
    frame is invalid (the detector falls back to un-fused features there), so
    the S0-e liveness gate is only meaningful on a valid-history batch.
    """
    inner = dataset
    while hasattr(inner, 'dataset'):
        inner = inner.dataset
    length = len(inner)
    count = min(int(sample), length)
    rng = np.random.RandomState(seed)
    indices = rng.choice(length, size=count, replace=False)
    last_valid = 0
    for index in indices:
        item = inner[int(index)]
        metas = item['img_metas'].data
        if bool(metas[-2]['is_prev_frame_valid']):
            last_valid += 1
    return {
        'sampled': int(count),
        'dataset_len': int(length),
        'last_history_valid': int(last_valid),
        'last_history_valid_ratio': float(last_valid) / max(1, count),
    }


def _pick_valid_batch(loader, fallback_index, max_batches=64):
    """Return ``(batch, offset)`` with a valid last history frame.

    Falls back to ``fallback_index`` when no valid-history batch is found in
    the first ``max_batches``; the caller reports the per-sample mask state so
    a fully masked batch cannot be mistaken for a dead readout.
    """
    for offset, batch in enumerate(loader):
        metas = _iter_frames(batch['img_metas'])
        if metas and all(bool(meta[-2]['is_prev_frame_valid'])
                         for meta in metas):
            return batch, offset
        if offset >= max_batches:
            break
    for offset, batch in enumerate(loader):
        if offset == fallback_index:
            return batch, offset
    raise RuntimeError('train split yielded no batch at all')


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


def _dump(report, path):
    if not path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    with open(path, 'w') as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True) + '\n')


def _worst_diff(a_maps, b_maps):
    worst = 0.0
    for a, b in zip(a_maps, b_maps):
        assert a.shape == b.shape, f'{tuple(a.shape)} vs {tuple(b.shape)}'
        worst = max(worst, float((a - b).abs().max()))
    return worst


def main():
    from mmcv.parallel import collate, MMDataParallel
    from mmcv.runner import load_checkpoint
    from torch.utils.data import DataLoader
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
    print('[stage 1] building the candidate-A model ...')
    arm_cfg, arm_model, arm_load = _build_eval(
        args.arm_config, args.checkpoint, device)
    new_prefixes = ('temporal_fusion.sem_head', 'temporal_fusion.z_cls_logits')
    report['arm_checkpoint_load'] = arm_load
    report['arm_new_keys'] = [
        k for k in arm_load['not_in_checkpoint'] if k.startswith(new_prefixes)]
    report['arm_other_missing_keys'] = [
        k for k in arm_load['not_in_checkpoint']
        if k not in report['arm_new_keys']]
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
    print(f'[stage 1a] arm rollout done: {report["identity"]}')

    print('[stage 1b] building the baseline model ...')
    _, base_model, base_load = _build_eval(
        args.base_config, args.checkpoint, device)
    report['base_checkpoint_load'] = base_load
    base_cls, base_reg, _ = _val_rollout(base_model, val_ds, args.val_index)
    report['identity']['max_abs_cls_diff'] = _worst_diff(arm_cls, base_cls)
    report['identity']['max_abs_reg_diff'] = _worst_diff(arm_reg, base_reg)
    report['identity']['same_missing_keys_apart_from_new_modules'] = bool(
        sorted(set(base_load['not_in_checkpoint'])
               - set(report['arm_new_keys']))
        == sorted(set(arm_load['not_in_checkpoint'])
                  - set(report['arm_new_keys'])))
    report['identity']['baseline_load_is_clean'] = bool(
        not base_load['not_in_checkpoint'])
    report['gates']['S0b_identity'] = bool(
        report['identity']['max_abs_cls_diff'] == 0.0
        and report['identity']['max_abs_reg_diff'] == 0.0
        and report['readout_zero_at_load']
        and report['identity']['residual_emitted']
        and report['identity']['residual_absmax'] == 0.0
        and report['identity']['same_missing_keys_apart_from_new_modules']
        and report['identity']['baseline_load_is_clean'])
    del base_model
    torch.cuda.empty_cache()

    # ---------------- S0-d / S0-e: pressure and liveness --------------------
    print('[stage 2] building the train-mode model for the gradient probe...')
    _dump(report, args.output)
    cfg = _load_cfg(args.arm_config, 'semA_grad_fig_', keep_train_cfg=True)
    model = build_detector(cfg.model, cfg.get('train_cfg'),
                           cfg.get('test_cfg'))
    load_checkpoint(model, args.pretrained, map_location='cpu', strict=False,
                    logger=None)
    # The training batch carries mmcv DataContainers, so the forward has to go
    # through MMDataParallel (same as me_rssm/sanity/smoke_fgfull_gpu.py). The
    # raw module keeps the graph handles (z_t) that the wrapper would hide.
    raw = model
    model = MMDataParallel(model.to(device).train(), device_ids=[args.gpu_id])

    train_cfg = dict(cfg.data.train)
    train_cfg.pop('samples_per_gpu', None)
    train_cfg.pop('workers_per_gpu', None)
    train_ds = build_dataset(train_cfg)
    # Real batch size: the pipeline contains BatchNorm layers, so a batch of 1
    # cannot run in train mode ("Expected more than 1 value per channel").
    samples_per_gpu = int(cfg.data.samples_per_gpu)
    report['train_mask_coverage'] = _last_history_coverage(train_ds)
    loader = DataLoader(
        train_ds, batch_size=samples_per_gpu, shuffle=False, num_workers=2,
        collate_fn=lambda x: collate(x, samples_per_gpu=samples_per_gpu))
    batch, batch_offset = _pick_valid_batch(
        loader, args.train_index, args.max_probe_batches)
    report['train_batch_offset'] = batch_offset

    fusion = raw.temporal_fusion
    report['train_batch_last_history_valid'] = [
        bool(meta[-2]['is_prev_frame_valid'])
        for meta in _iter_frames(batch['img_metas'])]
    losses = model(return_loss=True, **batch)
    if isinstance(losses, (list, tuple)):
        losses = losses[0]
    def _scalar(value):
        if isinstance(value, (list, tuple)):
            value = value[0]
        return None if value is None else float(value)

    sem = losses.get('loss_rssm_sem')
    if isinstance(sem, (list, tuple)):
        sem = sem[0]
    report['loss_keys'] = sorted(k for k in losses if k.startswith('loss_'))
    report['loss_rssm_sem'] = _scalar(sem)
    trainable = {k: (v[0] if isinstance(v, (list, tuple)) else v)
                 for k, v in losses.items()
                 if k.startswith('loss_')
                 and torch.is_tensor(v[0] if isinstance(v, (list, tuple)) else v)
                 and (v[0] if isinstance(v, (list, tuple)) else v).numel() == 1
                 and (v[0] if isinstance(v, (list, tuple)) else v).requires_grad}
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
    # A2-S (prereg section 4.1): the ranking term changes z's gradient
    # budget, so it must be MEASURED, not asserted away. The two ratios are
    # RECORDED ONLY -- deliberately with no pass line, so that "the budget
    # moved" can never be used after the fact to explain a result either way.
    rank_term = None
    if getattr(fusion, 'sem_target_mode', None) == 'channel_contrastive':
        sem_map = fusion.last_sem_map
        if sem_map is not None:
            target = fusion._rasterize_target(
                sem_map.device, tuple(sem_map.shape[-2:]))
            if target is not None:
                rank_term = fusion.sem_rank_weight * fusion._channel_rank_loss(
                    sem_map.to(torch.float32).float(), target.eq(1.0))
    if rank_term is not None:
        g_rank_z = _norm(torch.autograd.grad(
            rank_term, z_t, retain_graph=True, allow_unused=True))
        report['rank_gradient_budget'] = {
            'rank_term_value': float(rank_term.detach()),
            'grad_rank_on_z': g_rank_z,
            'grad_sem_on_z': g_sem_z,
            'rank_over_sem_on_z': (g_rank_z / g_sem_z) if g_sem_z > 0
            else float('nan'),
            'sem_over_total_on_z': (g_sem_z / g_total_z) if g_total_z > 0
            else float('nan'),
            'note': ('recorded only, no pass line: see prereg section 4.1. '
                     'These are NOT gates.'),
        }
    else:
        report['rank_gradient_budget'] = None
    # BOTH readings are recorded: a batch whose last history frame is
    # invalid gets an identically-zero residual by design (extract_feat
    # falls back to un-fused features), which is NOT a dead readout.
    residual_last = fusion.last_cls_logit_residual
    residual_masked = getattr(raw, '_rssm_cls_logit_residual', None)
    report['residual_absmax_before_mask'] = (
        None if residual_last is None
        else float(residual_last.detach().abs().max()))
    report['residual_absmax_to_head'] = (
        None if residual_masked is None
        else float(residual_masked.detach().abs().max()))
    g_total_w = _norm(torch.autograd.grad(total, weight, retain_graph=True,
                                          allow_unused=True))
    ratio = (g_sem_z / g_total_z) if g_total_z > 0 else float('nan')
    report['pressure'] = {
        'grad_sem_on_z': g_sem_z,
        'grad_total_on_z': g_total_z,
        'ratio': ratio,
        'grad_sem_on_W_cls': g_sem_w,
        'grad_total_on_W_cls': g_total_w,
        'loss_rssm_recon': (_scalar(losses['loss_rssm_recon'])
                            if 'loss_rssm_recon' in losses else None),
        'loss_rssm_kl': (_scalar(losses['loss_rssm_kl'])
                         if 'loss_rssm_kl' in losses else None),
        'loss_cls': (_scalar(losses['loss_cls'])
                     if 'loss_cls' in losses else None),
        'loss_bbox': (_scalar(losses['loss_bbox'])
                      if 'loss_bbox' in losses else None),
    }
    report['gates']['S0d_pressure_5pct'] = bool(ratio >= 0.05)
    # The forward residual is EXACTLY zero at init -- that is what buys the
    # S0-b bit-identity -- so liveness cannot be read off the zero-init
    # forward pass. Two independent readings are used instead: (a) the class
    # loss must be differentiable w.r.t. W_cls, and (b) after a small
    # perturbation the residual must reach the head un-zeroed on a
    # valid-history batch.
    with torch.no_grad():
        fusion.z_cls_logits.weight.normal_(0.0, 1e-3)
        fusion.z_cls_logits.bias.normal_(0.0, 1e-3)
    recalced = fusion.z_cls_logits(z_t.detach().to(torch.float32)).float()
    valid_mask = torch.tensor(report['train_batch_last_history_valid'],
                              device=device, dtype=torch.bool)
    # Recompute the LIVE residual (the cached one predates the
    # perturbation) and apply the same (H,W)->(W,H) grid contract the
    # detector applies before handing it to the head.
    fusion.last_cls_logit_residual = recalced
    perturbed = fusion.detection_cls_logit_residual(valid_mask)
    report['readout_perturbed_absmax'] = float(
        perturbed.detach().abs().max())
    report['readout_perturbed_grid'] = list(perturbed.shape[-2:])
    report['gates']['S0e_readout_live'] = bool(
        g_total_w > 0.0 and report['readout_perturbed_absmax'] > 0.0)
    report['gates']['S0e_sem_stays_off_readout'] = bool(g_sem_w == 0.0)
    report['stats'] = {}
    for k, v in losses.items():
        if k.startswith('stat_'):
            value = v[0] if isinstance(v, (list, tuple)) else v
            if torch.is_tensor(value) and value.numel() == 1:
                report['stats'][k] = float(value)
    report['peak_memory_gib'] = (
        torch.cuda.max_memory_allocated(device) / 2 ** 30)

    out = json.dumps(report, indent=2, sort_keys=True)
    print(out)
    _dump(report, args.output)
    if args.output:
        print(f'[json] {os.path.abspath(args.output)}')
    if not all(v is True for v in report['gates'].values()):
        print('\n[WARN] at least one S0 gate did not pass — see the JSON above')


if __name__ == '__main__':
    main()
