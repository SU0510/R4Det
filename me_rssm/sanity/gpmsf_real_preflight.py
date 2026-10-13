#!/usr/bin/env python3
"""S0 on real train/val batches; no AP claim and no saved trained weights."""
import argparse
import copy
import gc
import os
from pathlib import Path
from types import SimpleNamespace
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def batch_fingerprint(value):
    """Hash real training annotations too; no object-address/pickle fallback."""
    from me_rssm.sanity.paired_z_prediction import fingerprint

    def normalize(item):
        if type(item).__name__ == 'BitmapMasks':
            return dict(type='BitmapMasks', masks=item.masks,
                        height=item.height, width=item.width)
        if type(item).__name__ == 'DataContainer':
            return dict(type='DataContainer', data=normalize(item.data))
        if isinstance(item, dict):
            return {k: normalize(v) for k, v in item.items()}
        if isinstance(item, (tuple, list)):
            return type(item)(normalize(v) for v in item)
        return item

    return fingerprint(normalize(value))


def projection_errors(geo, metas):
    """Compare augmented projection with raw calibration + pixel augmentation."""
    import torch
    errors = []
    for i, meta in enumerate(metas):
        pts = geo['points'][i][:256, :3]
        if not len(pts):
            continue
        pts4 = torch.cat([pts, torch.ones_like(pts[:, :1])], dim=1)
        bda = geo['bda'][i]
        raw = pts4 @ torch.inverse(bda).T
        p0 = torch.as_tensor(meta['cam2img'] @ meta['lidar2cam'],
                             device=pts.device, dtype=pts.dtype)
        raw_proj = raw @ p0.T
        aug_proj = pts4 @ geo['lidar2img'][i, 0].T
        valid = (raw_proj[:, 2] > 1.) & (aug_proj[:, 2] > 1.)
        if valid.any():
            uv = raw_proj[valid, :2] / raw_proj[valid, 2:3]
            expected = uv @ geo['post_rots'][i, 0].T + geo['post_trans'][i, 0]
            observed = aug_proj[valid, :2] / aug_proj[valid, 2:3]
            errors.append(float((expected - observed).abs().max()))
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--baseline', required=True)
    parser.add_argument('--out-dir', required=True)
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '5':
        parser.error('preflight uses only GPU 5 after queue drain')
    import numpy as np
    import torch
    import training_old_version
    training_old_version.use_old_version()
    from mmcv import Config
    from mmcv.parallel import MMDataParallel
    from mmcv.runner import load_checkpoint
    from mmdet3d.datasets import build_dataset, build_dataloader
    from mmdet3d.models import build_model
    import me_rssm  # noqa: F401
    from me_rssm.sanity.paired_z_prediction import fingerprint
    from me_rssm.sanity.gpmsf_run_checks import source_hashes, write_json
    try:
        from mmdet.utils import compat_cfg
    except ImportError:
        from mmdet3d.utils import compat_cfg

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', checks={}, source_sha=source_hashes(),
                  gpu=5, seed=0, no_ap_claim=True)

    def save():
        write_json(out / 'real_preflight.json', report)

    def seed():
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def build(path):
        seed()
        cfg = compat_cfg(Config.fromfile(path))
        cfg.model.meta_info = dict(project_name=Path(path).stem, figures_path=str(out))
        model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
        model.init_weights()
        if cfg.get('resume_from'):
            raise ValueError('S0 must start fresh, not resume')
        if cfg.get('load_from'):
            load_checkpoint(model, cfg.load_from, map_location='cpu', strict=False)
        return cfg, model

    save()
    try:
        _, base = build(args.baseline)
        base_state = {k: v.detach().cpu().clone() for k, v in base.state_dict().items()}
        base_sha = fingerprint(base_state)
        base_rng = fingerprint((torch.get_rng_state(), torch.cuda.get_rng_state_all()))
        del base
        gc.collect()
        cfg, model = build(args.config)
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        common = {k: state[k] for k in base_state}
        report['checks']['common_initial_state'] = fingerprint(common) == base_sha
        report['checks']['initial_rng'] = fingerprint(
            (torch.get_rng_state(), torch.cuda.get_rng_state_all())) == base_rng
        report['common_state_sha'] = base_sha
        del common, base_state
        fusion = model.cross_attention
        report['checks']['zero_projections'] = all(
            torch.count_nonzero(p).item() == 0
            for layer in (fusion.proj_r2c, fusion.proj_i2r) for p in layer.parameters())
        if not all(report['checks'].values()):
            raise AssertionError('shared initial state/RNG or zero residual contract failed')
        parallel = MMDataParallel(model, device_ids=[0]).cuda(0)
        from mmdet3d.core.hook.kl_scale_scheduler import KLScaleSchedulerHook
        for hook_config in cfg.get('custom_hooks', []):
            if hook_config['type'] == 'KLScaleSchedulerHook':
                hook_args = {k: v for k, v in hook_config.items() if k != 'type'}
                KLScaleSchedulerHook(**hook_args).before_train_epoch(
                    SimpleNamespace(epoch=0, model=parallel))
        report['epoch0_kl_scale'] = float(model.temporal_fusion.kl_scale)
        cfg.dump(str(out / 'resolved_config.py'))
        geometry_records, captured = [], []
        original_geo = model._gpmsf_geometry

        def geometry(**kwargs):
            geo = original_geo(**kwargs)
            errors = projection_errors(geo, kwargs['img_metas'])
            geometry_records.append(dict(errors_px=errors, bda=fingerprint(geo['bda']),
                                         no_gt_keys=not any('gt' in k for k in geo)))
            return geo

        model._gpmsf_geometry = geometry
        original_forward = fusion.forward
        head_hashes = []

        def finite_output(value):
            if torch.is_tensor(value):
                if not torch.isfinite(value).all():
                    raise AssertionError('non-finite real step-zero output')
            elif hasattr(value, 'tensor'):
                finite_output(value.tensor)
            elif isinstance(value, dict):
                for item in value.values():
                    finite_output(item)
            elif isinstance(value, (list, tuple)):
                for item in value:
                    finite_output(item)

        def capture_head(_module, _args, value):
            finite_output(value)
            head_hashes.append(fingerprint(value))

        head_hook = model.pts_bbox_head.register_forward_hook(capture_head)

        def probe_forward(img, radar, geo=None):
            # Last call is the current frame, with its own real geometry.
            captured[:] = [(img.detach().clone(), radar.detach().clone(),
                            {k: ([v.detach().clone() for v in value] if k in
                                  ('img_feats', 'radar_levels', 'points') else
                                  value.detach().clone() if torch.is_tensor(value) else value)
                             for k, value in geo.items()})]
            return original_forward(img, radar, geo=geo)

        for mode in ('train', 'eval'):
            seed()
            data_cfg = copy.deepcopy(cfg.data.train if mode == 'train' else cfg.data.val)
            if mode == 'eval':
                data_cfg.test_mode = True
            dataset = build_dataset(data_cfg)
            batch_size = (cfg.data.train_dataloader.get('samples_per_gpu', 2)
                          if mode == 'train' else 1)
            if mode == 'train' and (batch_size != 2 or model.seq_len != 4):
                raise ValueError('real preflight requires original bs=2 and N4')
            loader = build_dataloader(dataset, samples_per_gpu=batch_size,
                                     workers_per_gpu=0, dist=False, shuffle=False, seed=0)
            data = next(iter(loader))
            input_sha = batch_fingerprint(data)
            hashes, heads = [], []
            for reference in (True, False):
                model.load_state_dict(state)
                seed()
                model.temporal_fusion.reset_state()
                parallel.train(mode == 'train')
                head_hashes.clear()
                fusion.forward = (lambda img, radar, geo=None: fusion.fusion_conv(
                    torch.cat([img, radar], dim=1))) if reference else probe_forward
                with torch.no_grad():
                    result = parallel(return_loss=(mode == 'train'), rescale=True,
                                      **copy.deepcopy(data))
                finite_output(result)
                hashes.append(fingerprint(result))
                heads.append(list(head_hashes))
                del result
            report['checks'][mode + '_step0_outputs'] = hashes[0] == hashes[1]
            report['checks'][mode + '_step0_head'] = bool(heads[0]) and heads[0] == heads[1]
            report[mode + '_batch'] = dict(input_sha=input_sha, samples=batch_size,
                                         output_sha=hashes, head_sha=heads)
            report['checks'][mode + '_input_unchanged'] = batch_fingerprint(data) == input_sha
            if mode == 'train':
                real_train = captured[0]
            del data, loader, dataset
            gc.collect()
        fusion.forward = original_forward
        head_hook.remove()
        model._gpmsf_geometry = original_geo
        errors = [v for r in geometry_records for v in r['errors_px']]
        report['geometry_records'] = geometry_records
        report['checks']['real_aug_projection'] = bool(errors) and max(errors) < 1e-2
        report['checks']['no_gt_geometry'] = all(r['no_gt_keys'] for r in geometry_records)
        # Restore step zero; this isolated two-step backward is a wiring probe,
        # not the formal optimizer/loss experiment. Full DDP proof follows.
        model.load_state_dict(state)
        img, radar, geo = real_train
        del real_train, state
        captured.clear()
        model.temporal_fusion.reset_state()
        fusion.train()
        for key in ('img_feats', 'radar_levels'):
            geo[key] = [v.detach().requires_grad_(True) for v in geo[key]]
        optimizer = torch.optim.SGD(fusion.parameters(), lr=1e-3)
        gradients = []
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            for key in ('img_feats', 'radar_levels'):
                for value in geo[key]:
                    value.grad = None
            fusion(img, radar, geo=geo).square().mean().backward()
            gradient = {n: float(p.grad.abs().sum()) if p.grad is not None else None
                        for n, p in fusion.named_parameters()}
            gradients.append(gradient)
            if any(p.grad is None or not torch.isfinite(p.grad).all()
                   for p in fusion.parameters()):
                raise AssertionError('missing/non-finite GP-MSF parameter gradient')
            optimizer.step()
        report['two_step_gradients'] = gradients
        report['checks']['two_step_offsets'] = (gradients[0]['offset_conv.2.weight'] == 0.
                                                 and gradients[1]['offset_conv.2.weight'] > 0.)
        report['checks']['native_source_gradients'] = all(
            v.grad is not None and torch.isfinite(v.grad).all() and v.grad.abs().sum() > 0
            for key in ('img_feats', 'radar_levels') for v in geo[key])
        # Use learned nonzero weights AND deliberately nonzero biases.
        fusion.eval()
        with torch.no_grad():
            fusion.proj_r2c.bias.fill_(2.)
            fusion.proj_i2r.bias.fill_(3.)
        inputs = []
        hook = fusion.fusion_conv.register_forward_pre_hook(
            lambda _m, values: inputs.append(values[0].detach()))
        with torch.no_grad():
            empty = dict(geo, points=[p[:0] for p in geo['points']])
            out_value = fusion(img, radar, geo=empty)
            report['checks']['zero_radar_identity'] = (torch.equal(inputs[-1][:, :img.shape[1]], img)
                                                       and bool(torch.isfinite(out_value).all()))
            mixed = dict(geo, points=[geo['points'][0][:0]] + geo['points'][1:])
            fusion(img, radar, geo=mixed)
            report['checks']['mixed_batch_identity'] = torch.equal(inputs[-1][0, :img.shape[1]], img[0])
            invalid = dict(geo, lidar2img=geo['lidar2img'].clone())
            invalid['lidar2img'][..., 2, :] = 0
            invalid['lidar2img'][..., 2, 3] = -1
            out_value = fusion(img, radar, geo=invalid)
            report['checks']['invalid_image_identity'] = (torch.equal(inputs[-1][:, img.shape[1]:], radar)
                                                          and bool(torch.isfinite(out_value).all()))
        hook.remove()
        if not all(bool(v) for v in report['checks'].values()):
            raise AssertionError('real S0 checks failed: ' + str(report['checks']))
        if report['source_sha'] != source_hashes():
            raise AssertionError('source changed during real preflight')
        report['status'] = 'PASS'
        save()
        return 0
    except Exception as exc:
        report.update(status='FAIL', error=repr(exc))
        save()
        raise


if __name__ == '__main__':
    sys.exit(main())
