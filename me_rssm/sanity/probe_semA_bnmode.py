#!/usr/bin/env python3
"""Read-only: where does the train/val gap in the sem-head readings come from?

ANSWER (measured 2026-10-11, ep6 seed0, see docs/training_runs_full.md 71.9.19):
the gap is dominated by the DATA SPLIT, not by the normalisation mode. Reading
the SAME eval-mode model through the SAME val pipeline on the train info file
recovers 0.4739 thr50 / 0.4167 hit1 at n=20 -- i.e. essentially the train-mode
logger's own train-split reading (0.4923 / 0.3586) -- while the val split reads
0.1280 / 0.0417. Flipping the z path's BatchNorms to batch statistics moves the
val reading far less (0.1280 -> 0.2452 thr50 at n=20) and does not explain the
gap on its own.

WHY THIS EXISTS -- the one unclosed item from section 72.10

Section 71.9.17 corrected M1's caliber defect by measuring the same quantity on
the train split (0.4923 thr50 / 0.3586 hit1 at ep6, epoch means over 38 logging
windows) and comparing it to the gate's val reading (0.1167 / 0.0833). Same
checkpoint, same formulas, roughly 4.3x apart. Section 72.10 ruled out the z
SAMPLING contract -- switching mu_q <-> mu_q + sigma*eps moved z by 35.26% and
changed thr50/hit1 by exactly 0.0000.

Two calibers were then left confounded, and this probe separates them:

  (a) SPLIT / AUGMENTATION -- the train logger reads augmented training frames
      (18-step pipeline, ImageAug3D2 + GlobalRotScaleTransFlipAll), the gate
      reads the deterministic val pipeline (10 steps). The GT is transformed
      with the points in the shared augmentation step, so a frame mismatch is
      not automatic -- but the *image appearance* under augmentation is not
      what the network sees at test time.
  (b) NORMALISATION MODE -- the logger runs with the model in train(), so every
      ``_BatchNorm`` on the z path normalises with BATCH statistics; the gate
      runs in eval() with the checkpoint's RUNNING statistics. Same weights,
      different function.

Arms (the z contract is pinned to the deploy ``mu_q`` everywhere, so
normalisation/split is the only variable):

    eval_running_stats   -- the M1 reading: whole model eval()               (A)
    tz_batch_stats       -- only the BatchNorms under ``temporal_fusion``
                            (the z path: encoder.0/1 -> e_t -> mu_q, and
                            decoder.0/1) switched to train()               (B)
    out_batch_stats      -- scope CONTROL: every BatchNorm OUTSIDE
                            ``temporal_fusion`` switched to train(), z path
                            left on running stats. If the z-path arm moves a
                            reading and this one moves it by the same amount,
                            the movement is not attributable to the z path.

The BatchNorm train()-mode statistics are estimated from the batch, so an arm
runs with a REAL batch (``--batch-size`` >= 2, default 4). A batch of one gives
a zero-variance batch statistic, which is not the statistic the training logger
used and would make the arm a strawman; ``read_arm`` refuses to run otherwise.

Every arm is paired clip-by-clip against the same clip's ``eval_running_stats``
reading, so the reported delta is a within-clip difference and cannot be
inflated by which clips an arm happened to keep.

BatchNorm running statistics and ``num_batches_tracked`` are SNAPSHOTTED before
and RESTORED after each arm, so the probe cannot mutate the checkpoint state in
memory, and ``model.training`` is restored too. Every other module (dropout,
etc.) stays in eval in every arm -- only norm layers are toggled. Read-only: no
training, no weight change, no checkpoint written.
"""
import argparse
import importlib.util
import json
import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import torch  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    'ep6', os.path.join(_REPO, 'me_rssm', 'sanity', 'probe_semreadout_ep6.py'))
ep6 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ep6)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--gpu-id', type=int, default=0)
    p.add_argument('--start-index', type=int, default=3)
    p.add_argument('--limit', type=int, default=10,
                   help='number of clips; must be a multiple of --batch-size')
    p.add_argument('--stride', type=int, default=200)
    p.add_argument('--min-valid-history', type=int, default=2)
    p.add_argument('--batch-size', type=int, default=4,
                   help='>= 2, otherwise BatchNorm train() statistics are '
                        'degenerate and the arm is not the training statistic')
    p.add_argument('--split-ann-file', default=None,
                   help='optional: read the SAME val pipeline and the SAME '
                        'eval-mode model on a different split (e.g. the train '
                        'info file). This separates "the train split looks '
                        'different" from "train-mode normalisation differs", '
                        'which the train logger readings cannot.')
    p.add_argument('--arms',
                   default='eval,z_path,matched_placebo,out_of_scope',
                   help='comma-separated subset of '
                        'eval/z_path/matched_placebo/out_of_scope. A single '
                        '``eval`` arm is the batching-neutrality control: run '
                        'it at batch size 1 and at the study batch size and '
                        'the reading must not move, since eval-mode BatchNorm '
                        'uses running statistics.')
    p.add_argument('--output', required=True)
    return p.parse_args()


_ALL_ARMS = (('eval_running_stats', 'eval'),
             ('z_path_batch_stats', 'z_path'),
             ('matched_placebo_batch_stats', 'matched_placebo'),
             ('out_of_scope_batch_stats', 'out_of_scope'))


def _selected_arms(spec):
    """Resolve --arms into (report_name, arm_id) pairs, in reading order."""
    wanted = [w.strip() for w in spec.split(',') if w.strip()]
    known = {arm for _, arm in _ALL_ARMS}
    unknown = [w for w in wanted if w not in known]
    if unknown:
        raise ValueError(f'unknown arm(s) {unknown}; known: {sorted(known)}')
    # The eval arm is the reference every delta is taken against, so it is
    # always read (and read first) even when the caller lists only an arm.
    ordered = list(_ALL_ARMS) if 'eval' in wanted else list(_ALL_ARMS)
    return [(n, a) for n, a in ordered if a == 'eval' or a in wanted]


def _bn_modules(model, under_temporal_fusion):
    """BatchNorms of one scope, in a stable (definition) order.

    ``under_temporal_fusion`` selects the z path (``model.temporal_fusion.*``)
    or everything else. The scope matters: 170 BatchNorms exist in the built
    detector, only 4 of them on the z path, and the rest (depth_net,
    rangeview_foreground, pts_*) are not what the training logger's z path
    normalises with.
    """
    out = []
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.modules.batchnorm._BatchNorm):
            continue
        under = name.startswith('temporal_fusion.')
        if under == under_temporal_fusion:
            out.append((name, mod))
    return out


def _snapshot(bns):
    """Running stats + num_batches_tracked, so train mode cannot leak."""
    return {name: (None if mod.running_mean is None else mod.running_mean.clone(),
                   None if mod.running_var is None else mod.running_var.clone(),
                   None if mod.num_batches_tracked is None
                   else mod.num_batches_tracked.clone())
            for name, mod in bns}


def _restore(bns, snap):
    for name, mod in bns:
        mean, var, nbt = snap[name]
        if mean is not None:
            mod.running_mean.copy_(mean)
        if var is not None:
            mod.running_var.copy_(var)
        if nbt is not None:
            mod.num_batches_tracked.copy_(nbt)


def _batched_rollout(model, dataset, indices):
    """One rollout over a REAL batch; returns per-sample sem maps and GT.

    The BatchNorm train() statistic is estimated over the batch dimension, so
    an arm that flips BN to train() is only meaningful at batch size >= 2.
    Everything the reading needs is kept per sample and in dataset order, so
    the paired comparison against the eval arm is unambiguous.
    """
    import torch as _t
    dz = ep6.dz
    fusion = model.temporal_fusion
    batch = dz.build_batch(dataset, list(indices), _t.cuda.current_device())
    points, img, metas, valid = dz.build_sequence_frames(batch)
    n = model.seq_len
    gts = [(ep6.sample_current_gt(dataset, i)) for i in indices]

    fusion.reset_state()
    for t in range(n - 1):
        if valid[t].any():
            model.extract_feat(points[t], img[t], metas[t],
                               is_valid_mask=valid[t], feat_or_dict=0,
                               curr_frame_supervision=False,
                               rssm_deterministic=True)
        if (~valid[t]).any():
            fusion.reset_for_samples(~valid[t])
    last_valid = valid[n - 2] if n >= 2 else _t.ones(
        len(indices), dtype=_t.bool, device=img[n - 1].device)
    model.extract_feat(points[n - 1], img[n - 1], metas[n - 1],
                       is_valid_mask=last_valid, feat_or_dict=1,
                       rssm_deterministic=True)
    sem = fusion.last_sem_map
    rows = []
    for pos, idx in enumerate(indices):
        sm = sem[pos:pos + 1].detach().float().cpu()
        rows.append({
            'index': int(idx),
            'sem_map': sm,
            'gt_boxes': gts[pos][0],
            'gt_labels': gts[pos][1],
        })
    return rows


@torch.no_grad()
def _arm_scope(model, arm):
    """Which BatchNorms an arm flips to train(). Returns a (name, mod) list.

    Only norm layers are ever toggled; the model itself stays in eval so
    dropout and the training-only branches cannot leak into the reading.

    'eval'            -- nothing (the archived M1 reading)
    'z_path'          -- the BatchNorms under ``temporal_fusion``, i.e. the
                         ones the training logger actually normalised with
                         batch statistics on the z path
    'out_of_scope'    -- every other BatchNorm (the whole upstream cascade)
    'matched_placebo' -- the FIRST ``len(z_path)`` out-of-scope BatchNorms, a
                         count-matched control. The upstream modules feed the
                         same ``feat`` the z path encodes, so any movement there
                         cascades into z; without a count-matched control a
                         movement cannot be attributed to the z path rather than
                         to "this reading reacts to any normalisation change".
    """
    z_bn = _bn_modules(model, True)
    if arm == 'eval':
        return []
    if arm == 'z_path':
        return z_bn
    if arm == 'out_of_scope':
        return _bn_modules(model, False)
    if arm == 'matched_placebo':
        return _bn_modules(model, False)[:len(z_bn)]
    raise ValueError(f'unknown arm {arm!r}')


def read_arm(model, dataset, indices, num_classes, pcr, min_radius, arm,
             batch_size):
    """One arm over the SAME clips, in batches of ``batch_size``.

    See ``_arm_scope`` for what each arm flips.
    """
    if arm != 'eval' and batch_size < 2:
        raise ValueError(
            f'arm {arm!r} uses BatchNorm train() statistics, which need a '
            f'real batch; got batch_size={batch_size}')
    flipped = _arm_scope(model, arm)
    all_bn = _bn_modules(model, True) + _bn_modules(model, False)
    snapshot = _snapshot(all_bn)
    try:
        model.eval()
        for _, mod in flipped:
            mod.train(True)
        raw = []
        rows = []
        for start in range(0, len(indices), batch_size):
            chunk = indices[start:start + batch_size]
            for row in _batched_rollout(model, dataset, chunk):
                raw.append(row)
                q = ep6.head_quality(
                    {'sem_map': row['sem_map'], 'gt_boxes': row['gt_boxes'],
                     'gt_labels': row['gt_labels']},
                    num_classes, pcr, min_radius)
                if q:
                    rows.append(dict(q, index=row['index']))
        # GT-shuffle floor: score each clip's sem map against a DIFFERENT
        # clip's target. If the reading survives that swap, it is not evidence
        # that the map is aligned to this clip's objects -- a map that always
        # peaks in the same (frequently occupied) region of the BEV would score
        # just as well. This is the same "does it survive a permutation"
        # discipline the family uses for shuffle_z, applied to the target
        # instead of the input, and it needs no extra forward pass.
        shuffled = []
        if len(raw) >= 2:
            for pos, row in enumerate(raw):
                donor = raw[(pos + 1) % len(raw)]
                q = ep6.head_quality(
                    {'sem_map': row['sem_map'],
                     'gt_boxes': donor['gt_boxes'],
                     'gt_labels': donor['gt_labels']},
                    num_classes, pcr, min_radius)
                if q:
                    shuffled.append(dict(q, index=row['index'],
                                         donor_index=donor['index']))
    finally:
        model.eval()
        _restore(all_bn, snapshot)
        for _, mod in all_bn:
            mod.eval()
    self_thr = ep6.mean_of(rows, 'thr50')
    self_hit = ep6.mean_of(rows, 'hit1')
    shuf_thr = ep6.mean_of(shuffled, 'thr50')
    shuf_hit = ep6.mean_of(shuffled, 'hit1')
    return {
        'arm': arm,
        'batch_size': batch_size,
        'z_contract': 'mu_q (rssm_deterministic=True, pinned in every arm)',
        'n_bn_flipped': len(flipped),
        'flipped_bn_modules': [n for n, _ in flipped],
        'per_clip': rows,
        'mean_thr50': self_thr,
        'mean_hit1': self_hit,
        'mean_hit1_excess_over_chance': ep6.mean_of(
            rows, 'hit1_excess_over_chance'),
        # Same maps, another clip's target: the floor a spatially biased map
        # reaches without carrying this clip's object evidence.
        'self_vs_gt_shuffle': {
            'gt_shuffle_per_clip': shuffled,
            'gt_shuffle_mean_thr50': shuf_thr,
            'gt_shuffle_mean_hit1': shuf_hit,
            'thr50_shuffle_over_self': (
                None if not self_thr or shuf_thr is None
                else shuf_thr / self_thr),
            'hit1_shuffle_over_self': (
                None if not self_hit or shuf_hit is None
                else shuf_hit / self_hit),
            'reading_survives_gt_shuffle': bool(
                shuf_thr is not None and self_thr is not None
                and (shuf_thr >= 0.9 * self_thr
                     or shuf_hit >= 0.9 * self_hit)),
        },
    }


def main():
    from mmcv import Config
    from mmcv.runner import load_checkpoint
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_detector

    args = parse_args()
    arms = _selected_arms(args.arms)
    single_eval_only = [a for _, a in arms] == ['eval']
    if not single_eval_only and args.limit % args.batch_size != 0:
        raise ValueError(
            f'--limit {args.limit} must be a multiple of --batch-size '
            f'{args.batch_size}, or the last arm would mix batch sizes and the '
            'BN statistic would differ between arms')
    torch.cuda.set_device(args.gpu_id)
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model['meta_info'] = {'figures_path': '/tmp/r4det_semA_bnmode_figures',
                              'project_name': 'TJ4D-semA-bnmode'}
    os.makedirs(cfg.model['meta_info']['figures_path'], exist_ok=True)
    num_classes = int(cfg.model.temporal_fusion.sem_num_classes)
    min_radius = int(cfg.model.temporal_fusion.object_min_radius)
    pcr = list(cfg.model.point_cloud_range)
    cfg.data.val.pop('samples_per_gpu', None)
    cfg.data.val.pop('workers_per_gpu', None)

    if args.split_ann_file is not None:
        cfg.data.val.ann_file = args.split_ann_file
    dataset = build_dataset(cfg.data.val)
    model = build_detector(cfg.model, train_cfg=None, test_cfg=None)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = model.cuda().eval()
    model.temporal_fusion.log_sem_maps = True

    indices = ep6.select_clips(
        dataset, ep6._Sel(args.start_index, args.limit,
                          args.min_valid_history, args.stride))
    if not single_eval_only and len(indices) % args.batch_size != 0:
        raise ValueError(
            f'selection returned {len(indices)} clips, not a multiple of '
            f'batch size {args.batch_size}')

    report = {
        'probe': 'candidate A sem-head caliber decomposition: split vs BN mode',
        'config': os.path.abspath(args.config),
        'checkpoint': os.path.abspath(args.checkpoint),
        'indices': [int(i) for i in indices],
        'image_indices': [int(dataset.data_infos[i]['image']['image_idx'])
                          for i in indices],
        'batch_size': int(args.batch_size),
        'ann_file': str(dataset.ann_file),
        'pipeline': [p.get('type') for p in cfg.data.val.pipeline],
        'bn_counts': {
            'z_path': len(_bn_modules(model, True)),
            'out_of_scope': len(_bn_modules(model, False)),
            'z_path_modules': [n for n, _ in _bn_modules(model, True)],
        },
        'arms': {},
    }

    for name, arm in _selected_arms(args.arms):
        report['arms'][name] = read_arm(
            model, dataset, indices, num_classes, pcr, min_radius, arm,
            args.batch_size)
        ep6.write_report(report, args.output)
    report['arms_read'] = [name for name, _ in _selected_arms(args.arms)]

    base = report['arms']['eval_running_stats']
    base_map = {r['index']: r for r in base['per_clip']}
    report['paired_delta_vs_eval'] = {}
    for name in [n for n, _ in arms if n != 'eval_running_stats']:
        arm = report['arms'][name]
        pairs = [(base_map[r['index']], r) for r in arm['per_clip']
                 if r['index'] in base_map]
        report['paired_delta_vs_eval'][name] = {
            'n_paired_clips': len(pairs),
            'mean_delta_thr50': (sum(b['thr50'] - a['thr50']
                                     for a, b in pairs) / len(pairs)
                                 if pairs else None),
            'mean_delta_hit1': (sum(b['hit1'] - a['hit1']
                                    for a, b in pairs) / len(pairs)
                                if pairs else None),
            'n_thr50_moved': sum(1 for a, b in pairs
                                 if a['thr50'] != b['thr50']),
            'n_hit1_moved': sum(1 for a, b in pairs
                                if a['hit1'] != b['hit1']),
        }
    # Reproduction check: the eval arm runs the SAME clips through the SAME
    # scoring formulas as the frozen gate, only batched. Published config notes
    # say a sample's output is bit-identical when its batch partners change
    # (it is BN-free cross-talk in eval, since BN uses running statistics), so
    # the eval arm must land on the archived M1 reading. If it does not, the
    # batched rollout is not the gate's measurement and no delta below it can
    # be attributed to normalisation.
    is_val_split = (
        os.path.basename(str(dataset.ann_file)) == 'TJ4D_infos_val.pkl')
    report['eval_reproduces_frozen_gate'] = {
        'check_applies': bool(is_val_split),
        'check_applies_note': (
            'the frozen gate values belong to the val split; on any other '
            'split this field is a reference match, not a descriptor of '
            'probe correctness'),
        'frozen_thr50': 0.1167,
        'frozen_hit1': 0.0833,
        'in_process_thr50': base['mean_thr50'],
        'in_process_hit1': base['mean_hit1'],
        'abs_delta_thr50': (None if base['mean_thr50'] is None
                            else abs(base['mean_thr50'] - 0.1167)),
        'abs_delta_hit1': (None if base['mean_hit1'] is None
                           else abs(base['mean_hit1'] - 0.0833)),
        'reproduces': bool(base['mean_thr50'] is not None
                           and base['mean_hit1'] is not None
                           and abs(base['mean_thr50'] - 0.1167) <= 1e-3
                           and abs(base['mean_hit1'] - 0.0833) <= 1e-3),
    }

    # Intervention check: an arm that changed nothing cannot be evidence for or
    # against the normalisation-mode explanation.
    moved = any(v['n_thr50_moved'] or v['n_hit1_moved']
                for v in report['paired_delta_vs_eval'].values())
    report['intervention_check'] = {
        'some_arm_moved_a_reading': bool(moved),
        'all_arms_bitwise_identical_to_eval': not moved,
    }
    report['references'] = {
        'val_gate_M1_thr50': 0.1167,
        'val_gate_M1_hit1': 0.0833,
        'train_caliber_stat_sem_thr50_ep6_epoch_mean': 0.4923,
        'train_caliber_stat_sem_hit1_ep6_epoch_mean': 0.3586,
        'note': ('train_caliber readings are ep6 EPOCH MEANS over 38 logging '
                 'windows of the train split (model in train()); they are '
                 'references for orientation, not a gate. The val references '
                 'are the frozen gate values (section 72); the eval arm is the '
                 'in-process reproduction of them.'),
    }
    report['input_fingerprints'] = ep6.input_fingerprints(
        args.config, args.checkpoint)
    ep6.write_report(report, args.output)
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ('arms', 'input_fingerprints')},
                     ensure_ascii=False, indent=2, sort_keys=True))
    for name, arm in report['arms'].items():
        print(f'{name}: thr50={arm["mean_thr50"]} hit1={arm["mean_hit1"]} '
              f'(n={len(arm["per_clip"])})')


if __name__ == '__main__':
    main()
