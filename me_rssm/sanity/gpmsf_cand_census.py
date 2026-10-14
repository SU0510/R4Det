#!/usr/bin/env python3
"""Census of *which* GP-MSF candidates can contribute at all (read-only).

Why: the camera evidence chain samples every ``qc*K*n_h`` candidate slot of a
chunk, but a slot whose address misses the (resized) image in *every* view is
already provably zero by construction -- ``ok_h`` masks the level softmax and
``mask_img=ok_h`` drops it from the candidate softmax.  "Effective candidate
compression" (docs/training_runs_full.md section 104.7) is only worth building
if that provably-zero set is large, so this script measures it on **real TJ4D
geometry with the trained ep4 offsets** instead of guessing.

It also dumps the captured current-frame fusion inputs so later optimisation
work can be timed on the real distribution without rebuilding the model.

No training, no optimizer step, no AP claim, no weight write.
"""
import argparse
import copy
import gc
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def census(image_calls, chunk_calls, bev_shape):
    """Aggregate the captured per-call tensors into rates (no GPU needed).

    ``image_calls`` / ``chunk_calls`` are ``{chunk_start: payload}`` so that the
    checkpoint recompute pass (which re-enters the same chunks in backward)
    overwrites instead of double counting.
    """
    import torch

    h, w = int(bev_shape[0]), int(bev_shape[1])
    q_total = h * w
    slots = valid = 0
    cells = cells_any = 0
    k_hist = None
    flat_any_valid = torch.zeros(q_total, dtype=torch.bool)
    per_cell_valid = torch.zeros(q_total, dtype=torch.float32)
    per_cell_slots = torch.zeros(q_total, dtype=torch.float32)
    for start, (ok_h, beta) in sorted(image_calls.items()):
        b, hd, qc, k, n_h = ok_h.shape
        slots += ok_h.numel()
        valid += int(ok_h.sum())
        any_cell = ok_h.flatten(3).any(dim=3)               # (B,Hd,qc)
        cells += any_cell.numel()
        cells_any += int(any_cell.sum())
        n_valid = ok_h.sum(dim=(3, 4))                      # (B,Hd,qc)
        hist = torch.bincount(n_valid.flatten(), minlength=k * n_h + 1)
        k_hist = hist if k_hist is None else k_hist + hist
        chunk_any = any_cell.any(dim=1).any(dim=0)          # (qc,)
        flat_any_valid[start:start + qc] |= chunk_any
        # mean over (b, head) of valid-slot count / coverage per cell
        per_cell_valid[start:start + qc] += n_valid.float().mean(dim=(0, 1))
        per_cell_slots[start:start + qc] += 1.0
    radar_cells = radar_any = 0
    chunks = chunks_any = 0
    img_msg = radar_msg = 0.0
    for start, (valid_r, valid_i, m_r, m_i) in sorted(chunk_calls.items()):
        chunks += valid_i.numel()
        chunks_any += int(valid_i.sum())
        radar_cells += valid_r.numel()
        radar_any += int(valid_r.sum())
        img_msg += float(m_i.abs().mean())
        radar_msg += float(m_r.abs().mean())
    n_img = max(len(image_calls), 1)
    n_chunk = max(len(chunk_calls), 1)
    per_cell_valid = (per_cell_valid / per_cell_slots.clamp_min(1.0)).view(h, w)
    # the per-column/row maps are only meaningful when the keys are *dense*
    # chunk starts (the pre-compaction route); the compact route keys by call
    # index, so report them as unavailable rather than silently wrong.
    keys = sorted(image_calls)
    dense_starts = keys == list(range(0, q_total, qc)) if keys else False
    return dict(
        image_slot_valid_rate=valid / max(slots, 1),
        # per (b, head, cell): at least one of the 12 slots is image-valid
        image_cell_any_valid_rate=cells_any / max(cells, 1),
        # per (b, cell), union over heads -- this is the compaction unit
        image_cell_any_valid_any_head_rate=chunks_any / max(chunks, 1),
        image_slots_total=slots,
        image_slots_valid=valid,
        radar_cell_any_valid_rate=radar_any / max(radar_cells, 1),
        mean_abs_image_message=img_msg / n_img,
        mean_abs_radar_message=radar_msg / n_chunk,
        valid_slots_per_cell_hist=[int(x) for x in k_hist],
        mean_valid_slots_per_cell=float(per_cell_valid.mean()),
        per_bev_column_any_valid=([round(float(x), 6)
                                  for x in flat_any_valid.view(h, w).any(dim=0)]
                                  if dense_starts else None),
        per_bev_row_any_valid=([round(float(x), 6)
                               for x in flat_any_valid.view(h, w).any(dim=1)]
                               if dense_starts else None),
        mean_valid_slots_per_cell_by_row=(
            [round(float(x), 4) for x in per_cell_valid.mean(dim=1)]
            if dense_starts else None),
        mean_valid_slots_per_cell_by_column=(
            [round(float(x), 4) for x in per_cell_valid.mean(dim=0)]
            if dense_starts else None),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--capture', required=True, help='path of the .pt dump')
    args = parser.parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '5':
        parser.error('census uses only GPU 5')

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
    from me_rssm.sanity.gpmsf_run_checks import source_hashes, write_json
    try:
        from mmdet.utils import compat_cfg
    except ImportError:
        from mmdet3d.utils import compat_cfg

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = dict(status='running', source_sha=source_hashes(), gpu=5, seed=0,
                  checkpoint=args.checkpoint, no_ap_claim=True, no_weights_written=True)

    def save():
        write_json(out / 'candidate_census.json', report)

    def seed():
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    save()
    try:
        seed()
        cfg = compat_cfg(Config.fromfile(args.config))
        cfg.model.meta_info = dict(project_name=Path(args.config).stem, figures_path=str(out))
        model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
        model.init_weights()
        missing = load_checkpoint(model, args.checkpoint, map_location='cpu', strict=False)
        report['checkpoint_missing_keys'] = len(getattr(missing, 'missing_keys', []) or [])
        report['checkpoint_unexpected_keys'] = len(getattr(missing, 'unexpected_keys', []) or [])
        fusion = model.cross_attention

        parallel = MMDataParallel(model, device_ids=[0]).cuda(0)
        from mmdet3d.core.hook.kl_scale_scheduler import KLScaleSchedulerHook
        for hook_config in cfg.get('custom_hooks', []):
            if hook_config['type'] == 'KLScaleSchedulerHook':
                hook_args = {k: v for k, v in hook_config.items() if k != 'type'}
                KLScaleSchedulerHook(**hook_args).before_train_epoch(
                    __import__('types').SimpleNamespace(epoch=4, model=parallel))

        captured = []
        original_forward = fusion.forward

        def probe_forward(img, radar, geo=None):
            captured[:] = [(img.detach().clone(), radar.detach().clone(),
                            {k: ([v.detach().clone() for v in value]
                                 if k in ('img_feats', 'radar_levels', 'points')
                                 else value.detach().clone() if torch.is_tensor(value) else value)
                             for k, value in geo.items()})]
            return original_forward(img, radar, geo=geo)

        seed()
        data_cfg = copy.deepcopy(cfg.data.train)
        dataset = build_dataset(data_cfg)
        batch_size = cfg.data.train_dataloader.get('samples_per_gpu', 2)
        if batch_size != 2 or model.seq_len != 4:
            raise ValueError('census requires the original bs=2 and N4')
        loader = build_dataloader(dataset, samples_per_gpu=batch_size, workers_per_gpu=0,
                                  dist=False, shuffle=False, seed=0)
        data = next(iter(loader))
        model.temporal_fusion.reset_state()
        parallel.train()
        fusion.forward = probe_forward
        with torch.no_grad():
            parallel(return_loss=True, rescale=True, **data)
        fusion.forward = original_forward
        del data, loader, dataset, parallel
        gc.collect()
        torch.cuda.empty_cache()
        if not captured:
            raise AssertionError('no fusion call captured')
        img, radar, geo = captured[0]
        report['captured'] = dict(
            img=tuple(img.shape), radar=tuple(radar.shape),
            num_views=int(geo['num_views']),
            image_levels=[tuple(t.shape) for t in geo['img_feats']],
            radar_levels=[tuple(t.shape) for t in geo['radar_levels']],
            points=[tuple(p.shape) for p in geo['points']],
            depth=None if geo['depth'] is None else tuple(geo['depth'].shape),
            image_size=[int(x) for x in fusion.image_size],
            bev_shape=[int(x) for x in fusion.bev_shape],
            query_chunk=int(fusion.query_chunk),
            grad_checkpoint=bool(fusion.grad_checkpoint),
            cells=int(fusion.bev_shape[0] * fusion.bev_shape[1]),
            candidate_slots_per_cell=int(fusion.num_offsets * fusion.num_heights),
        )

        # ---- census: wrap the two producers, never edit the module ----------
        image_calls, chunk_calls = {}, {}
        cell_call = [0]
        original_terms = fusion.__class__._image_terms
        original_chunk = fusion.__class__._chunk
        current_start = [0]

        def terms(self, ctx, addr, q_ift_h):
            out = original_terms(self, ctx, addr, q_ift_h)
            image_calls[current_start[0]] = (out[1].detach().cpu(), out[2].detach().cpu())
            return out

        def chunk(self, q_cam_c, q_rad_c, f_cat_c, cell_xy_c, e_r_c, e_t_c, m_par_c, ctx):
            # the chunk no longer receives a global ``start``: the cell identity
            # travels in the sliced/gathered geometry, so the current counter is
            # the call index (dense route = ascending chunk number, compact route
            # = that sample's own active-cell column block).
            cell_call[0] += 1
            current_start[0] = cell_call[0]
            out = original_chunk(self, q_cam_c, q_rad_c, f_cat_c, cell_xy_c,
                                 e_r_c, e_t_c, m_par_c, ctx)
            chunk_calls[cell_call[0]] = (out[3].detach().cpu(), out[4].detach().cpu(),
                                         out[0].detach().cpu(), out[1].detach().cpu())
            return out

        fusion.__class__._image_terms = terms
        fusion.__class__._chunk = chunk
        fusion.train()
        try:
            for key in ('img_feats', 'radar_levels'):
                geo[key] = [v.detach().requires_grad_(True) for v in geo[key]]
            out_value = fusion(img, radar, geo=geo)
            out_value.square().mean().backward()
            report['finite_output'] = bool(torch.isfinite(out_value).all())
            report['finite_input_grads'] = all(
                v.grad is not None and torch.isfinite(v.grad).all()
                for key in ('img_feats', 'radar_levels') for v in geo[key])
        finally:
            fusion.__class__._image_terms = original_terms
            fusion.__class__._chunk = original_chunk
        report['census'] = census(image_calls, chunk_calls, fusion.bev_shape)
        snapshot = dict(report['census'])
        torch.save(dict(img=img.cpu(), radar=radar.cpu(),
                        geo={k: ([v.cpu() for v in val] if k in ('img_feats', 'radar_levels', 'points')
                                 else val.cpu() if torch.is_tensor(val) else val)
                             for k, val in geo.items()},
                        meta=dict(config=args.config, checkpoint=args.checkpoint,
                                  bev_shape=[int(x) for x in fusion.bev_shape],
                                  query_chunk=int(fusion.query_chunk),
                                  cells=int(fusion.bev_shape[0] * fusion.bev_shape[1]))),
                   args.capture)
        report['capture_path'] = args.capture
        report['status'] = 'PASS'
        save()
        print(json.dumps(snapshot, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        report.update(status='FAIL', error=repr(exc))
        save()
        raise


if __name__ == '__main__':
    sys.exit(main())
