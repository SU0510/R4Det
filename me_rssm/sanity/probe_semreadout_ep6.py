"""Candidate A ep6 mechanism gate (pre-registered M1-M4, read-only).

Pre-registration: docs/rssm_candidate_A_prereg_20261010.md §5 S1.

    M1  semantic-head quality on val: at GT centres, thr50 >= 0.30 and
        hit@1 >= 0.60. Both are the SAME quantities the training loop logs as
        ``stat_sem_thr50`` / ``stat_sem_hit1``, recomputed here on val with the
        identical formulas (prob > 0.5 on positive cells, and per-occupied-
        class-channel argmax hit).
    M2  z-exclusivity: with the SAME weights, feeding sem_head with h_t or the
        raw feat instead of z_t must be >= 30% WORSE in thr50 -- BOTH arms
        separately (prereg §5 M2 clarification of 2026-10-10: the section is
        titled z-exclusivity, so neither h nor the observation may substitute
        for z), otherwise the semantics came from h/observation and z is just a
        passthrough.
    M3  the readout is not decorative: paired toggle of the residual on the
        trained weights (residual ON vs forced 0) must worsen the positive-
        cell class CE on >= 8/10 clips (sign test).
    M4  the residual carries CLASS information, not just amplitude: the class
        argmax of z's residual must hit the GT class at GT foreground cells
        (own reading) and follow the donor's dominant class when z is swapped
        for another sample's z (donor reading): both >= chance + 20 pt.

No AP is measured, no training, no checkpoint is written. Read-only.

Usage:
  CUDA_VISIBLE_DEVICES=5 python me_rssm/sanity/probe_semreadout_ep6.py \
    --config me_rssm/configs/..._semreadout.py \
    --checkpoint <work_dir>/epoch_6.pth \
    --start-index 3 --limit 10 --stride 200 \
    --output <work_dir>/semreadout_ep6.json
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
import torch.nn.functional as F  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    'dz', os.path.join(_REPO, 'tools', 'diagnose_z_utilization.py'))
dz = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dz)


def parse_args():
    p = argparse.ArgumentParser(
        description='Candidate A ep6 mechanism gate (M1-M4).')
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--gpu-id', type=int, default=0)
    p.add_argument('--start-index', type=int, default=3)
    p.add_argument('--limit', type=int, default=10)
    p.add_argument('--stride', type=int, default=200,
                   help='step between selected val clips')
    p.add_argument('--min-valid-history', type=int, default=2)
    p.add_argument('--shuffle-offset', type=int, default=1,
                   help='donor offset for M4; must not map a clip to itself')
    p.add_argument('--output', required=True)
    return p.parse_args()


class _Sel:
    """Minimal stand-in for the argparse namespace dz.select_indices wants."""

    def __init__(self, start_index, limit, min_valid_history):
        self.start_index = start_index
        self.limit = limit
        self.min_valid_history = min_valid_history
        self.require_valid_history = True


def select_clips(dataset, args):
    """Spread the clips over val instead of taking one contiguous block.

    The family-standard probes take ``limit`` consecutive hits from
    ``start_index``; for a 10-clip sign test that would concentrate every clip
    in one scene block. Striding keeps the M3 sign test from being a
    same-scene repeat.
    """
    if args.stride <= 1:
        return dz.select_indices(
            dataset, _Sel(args.start_index, args.limit, args.min_valid_history))
    selected = []
    index = args.start_index
    while index < len(dataset) and len(selected) < args.limit:
        mask = dz.history_validity(dataset, index)
        if mask is not None and sum(mask[:-1]) >= args.min_valid_history:
            selected.append(index)
            index += args.stride
        else:
            index += 1
    if len(selected) < 2:
        raise ValueError(f'stride {args.stride} selected too few clips: {selected}')
    return selected


@torch.no_grad()
def rollout(model, dataset, index, sem_input=None, use_residual=True):
    """One val rollout of the current frame; returns everything M1-M4 need."""
    fusion = model.temporal_fusion
    batch = dz.build_batch(dataset, [index], torch.cuda.current_device())
    points, img, metas, valid = dz.build_sequence_frames(batch)
    n = model.seq_len

    restore = None
    if sem_input is not None:
        restore = fusion.sem_head_input
        fusion.sem_head_input = sem_input
    try:
        fusion.reset_state()
        for t in range(n - 1):
            if valid[t].any():
                model.extract_feat(
                    points[t], img[t], metas[t], is_valid_mask=valid[t],
                    feat_or_dict=0, curr_frame_supervision=False)
            if (~valid[t]).any():
                fusion.reset_for_samples(~valid[t])
        last_valid = valid[n - 2] if n >= 2 else torch.ones(
            1, dtype=torch.bool, device=img[n - 1].device)
        out = model.extract_feat(
            points[n - 1], img[n - 1], metas[n - 1],
            is_valid_mask=last_valid, feat_or_dict=1)
        residual = out.get('rssm_cls_logit_residual')
        sem_map = fusion.last_sem_map
        z_t = fusion.z_state
        outs = (model.pts_bbox_head(out['pts_feats'],
                                    cls_logit_residual=residual)
                if use_residual
                else model.pts_bbox_head(out['pts_feats'],
                                         cls_logit_residual=None))
        meta = metas[n - 1][0]
        return {
            'sem_map': None if sem_map is None
            else sem_map.detach().float().cpu(),
            'z': None if z_t is None else z_t.detach().float().cpu(),
            'residual': None if residual is None
            else residual.detach().float().cpu(),
            'cls_score': [t.detach().float().cpu() for t in outs[0]],
            'pts_feats': [t.detach().float().cpu() for t in out['pts_feats']],
            'gt_boxes': meta['gt_bboxes_3d'],
            'gt_labels': meta['gt_labels_3d'],
        }
    finally:
        if restore is not None:
            fusion.sem_head_input = restore


def make_target(run, num_classes, pcr, min_radius=2):
    """GT centre heatmap on the SEM grid (H=x, W=y), matching training."""
    from mmdet3d.core.utils.gaussian import bev_centers_to_heatmap
    sm = run['sem_map']
    sm = sm[None] if sm.dim() == 3 else sm
    return bev_centers_to_heatmap(
        [run['gt_boxes']], [run['gt_labels']], num_classes=num_classes,
        out_size=tuple(sm.shape[-2:]), point_cloud_range=pcr,
        min_radius=min_radius, device=sm.device, dtype=torch.float32)


def head_quality(run, num_classes, pcr):
    """The exact quantities the training loop logs as stat_sem_*."""
    sm = run['sem_map']
    if sm is None:
        return None
    sm = sm[None] if sm.dim() == 3 else sm
    tgt = make_target(run, num_classes, pcr).to(sm.device)
    prob = sm.float().sigmoid()
    pos = tgt.eq(1.0)
    num_pos = float(pos.sum())
    if num_pos == 0:
        return None
    flat = prob.flatten(2)
    peak_idx = flat.argmax(dim=2)
    peak = torch.zeros_like(flat).scatter_(
        2, peak_idx[:, :, None], 1.0).view_as(prob)
    occupied = (pos.sum(dim=(2, 3)) > 0).sum().clamp_min(1.0)
    return {
        'thr50': float(((prob > 0.5) & pos).sum()) / num_pos,
        'hit1': float((peak * pos).sum()) / float(occupied),
        'num_pos_cells': int(num_pos),
        'occupied_class_channels': int(occupied),
    }


# ``pts_feats`` / the detection head live on the TRANSPOSED BEV grid: the
# latent and the GT rasteriser use (H, W) = (216, 248) with H indexing x, while
# the head sees (248, 216). Every probe reading is converted back to the
# rasteriser's grid so the cell index means the same thing in M1/M3/M4 and a
# silent transposition cannot pass (the grid is deliberately non-square).
def _to_sem_grid(tensor):
    """(..., Wh, Hh) on the head grid -> (..., Hh, Wh) on the sem grid."""
    return tensor.permute(*range(tensor.dim() - 2), tensor.dim() - 1,
                          tensor.dim() - 2).contiguous()


def class_scores_from_head(run, num_classes):
    """(N_cells, num_classes) max-over-anchor-slot logits, SEM-grid order.

    ``conv_cls`` is (num_anchors * num_classes, W, H) in
    ``[anchor * num_classes + class]`` order, so reshaping the channel axis
    into (num_anchors, num_classes) and taking the max over anchors recovers
    the per-class score the head competes with at that cell; the result is then
    transposed onto the (H=x, W=y) grid shared with ``make_target``.
    """
    score = run['cls_score'][0][0]              # (A*C, Wh, Hh)
    n_anchor = score.shape[0] // num_classes
    w, h = score.shape[-2:]
    pooled = score.reshape(n_anchor, num_classes, w, h).amax(dim=0)
    return _to_sem_grid(pooled).reshape(num_classes, -1).permute(1, 0)


def residual_class_map(run, num_classes):
    """(N_cells, num_classes) z->class residual on the SEM (H=x, W=y) grid.

    ``rssm_cls_logit_residual`` is handed to the head already transposed and
    validity-masked; the probe converts it back to the rasteriser's grid.
    """
    r = run['residual']
    if r is None:
        return None
    r = r[None] if r.dim() == 3 else r
    return _to_sem_grid(r[0]).reshape(num_classes, -1).permute(1, 0)


def head_scores_with_residual(model, run, residual):
    """Re-run ONLY the detection head with a substituted class residual.

    The M4 swap must change exactly one thing: the class evidence produced
    from z. Features, weights, anchors and the GT are held fixed, so any
    movement in the head's class argmax is attributable to the swapped z.

    ``residual`` is on the head grid (the same convention the detector uses);
    pass ``None`` for the no-readout control. The residual is moved to the
    features' device: ``Anchor3DHead._expand_cls_logit_residual`` builds its
    channel map on ``ref.device`` and calls ``residual.index_select`` with it,
    so a CPU residual against CUDA features raises a device mismatch (the
    rollouts deliberately keep every reading on CPU).
    """
    feats = [t.cuda() if torch.cuda.is_available() else t
             for t in run['pts_feats']]
    if residual is not None:
        residual = residual.to(feats[0].device)
    with torch.no_grad():
        outs = model.pts_bbox_head(feats, cls_logit_residual=residual)
    return [t.detach().float().cpu() for t in outs[0]]


def mean_of(rows, key):
    vals = [r[key] for r in rows]
    return float(sum(vals)) / len(vals) if vals else None


def relative_thr50(z_thr, control_thr):
    """(control - z) / z; negative == the control is WORSE than z."""
    if z_thr is None or control_thr is None or not z_thr:
        return None
    return (control_thr - z_thr) / z_thr


def gate_m1(mean_thr50, mean_hit1):
    """Pre-registered S1/M1: val head quality at GT centres."""
    return bool(mean_thr50 is not None and mean_hit1 is not None
                and mean_thr50 >= 0.30 and mean_hit1 >= 0.60)


def gate_m2(rel_h, rel_f):
    """Pre-registered S1/M2: z-exclusivity, STRICT reading.

    Both control arms must be >= 30% worse than z. The prereg text used to
    read "``h_t`` (或 ``feat``)", which can be read as "either arm suffices";
    the section is titled z-exclusivity, so the strict reading (neither h nor
    the observation may substitute for z) was pinned in the prereg on
    2026-10-10 while ep6 was still unread, and this function is that reading.
    """
    return bool(rel_h is not None and rel_f is not None
                and rel_h <= -0.30 and rel_f <= -0.30)


def gate_m3(rows, min_clips=8, min_wins=8):
    """Pre-registered S1/M3: residual toggle worsens positive-cell CE."""
    wins = sum(1 for r in rows if r['delta_off_minus_on'] > 0)
    return bool(len(rows) >= min_clips and wins >= min_wins)


def gate_m4(own_hit, transfer, num_classes):
    """Pre-registered S1/M4: class-correct information, not amplitude."""
    bar = 1.0 / num_classes + 0.20
    return bool(own_hit is not None and transfer is not None
                and own_hit >= bar and transfer >= bar)


def main():
    from mmcv import Config
    from mmcv.runner import load_checkpoint
    from mmdet3d.datasets import build_dataset
    from mmdet3d.models import build_detector

    import numpy as np

    args = parse_args()
    torch.cuda.set_device(args.gpu_id)
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.model['meta_info'] = {
        'figures_path': '/tmp/r4det_semreadout_ep6_figures',
        'project_name': 'TJ4D-semreadout-ep6',
    }
    os.makedirs(cfg.model['meta_info']['figures_path'], exist_ok=True)
    num_classes = int(cfg.model.temporal_fusion.sem_num_classes)
    min_radius = int(cfg.model.temporal_fusion.object_min_radius)
    pcr = list(cfg.model.point_cloud_range)
    cfg.data.val.pop('samples_per_gpu', None)
    cfg.data.val.pop('workers_per_gpu', None)
    dataset = build_dataset(cfg.data.val)
    model = build_detector(cfg.model, train_cfg=None, test_cfg=None)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model = model.cuda().eval()
    fusion = model.temporal_fusion
    if getattr(fusion, 'sem_head', None) is None:
        raise RuntimeError(
            'this checkpoint/config has no sem_head (sem_supervision_weight=0)')
    # Probe-only opt-in: in eval the sem map is otherwise skipped, which would
    # make M1/M2 unmeasurable. It changes no weight and no detection output.
    fusion.log_sem_maps = True

    indices = select_clips(dataset, args)
    if len(indices) < 2:
        raise RuntimeError(f'need >= 2 val clips, got {indices}')

    report = {
        'probe': 'candidate A ep6 mechanism gate (M1-M4)',
        'config': os.path.abspath(args.config),
        'checkpoint': os.path.abspath(args.checkpoint),
        'indices': [int(i) for i in indices],
        'shuffle_offset': args.shuffle_offset,
        'gates': {},
    }

    # ------------------------- M1 / M2 ----------------------------------
    z_runs = [rollout(model, dataset, i, sem_input='z') for i in indices]
    h_runs = [rollout(model, dataset, i, sem_input='h') for i in indices]
    feat_runs = [rollout(model, dataset, i, sem_input='feat') for i in indices]

    m1 = [q for q in (head_quality(r, num_classes, pcr) for r in z_runs) if q]
    m2_h = [q for q in (head_quality(r, num_classes, pcr) for r in h_runs) if q]
    m2_f = [q for q in (head_quality(r, num_classes, pcr) for r in feat_runs) if q]

    m1_thr, m1_hit = mean_of(m1, 'thr50'), mean_of(m1, 'hit1')
    h_thr, f_thr = mean_of(m2_h, 'thr50'), mean_of(m2_f, 'thr50')
    rel_h = relative_thr50(m1_thr, h_thr)
    rel_f = relative_thr50(m1_thr, f_thr)
    report['M1_sem_head'] = {'per_clip': m1, 'mean_thr50': m1_thr,
                             'mean_hit1': m1_hit}
    report['M2_exclusivity'] = {
        'z_mean_thr50': m1_thr, 'h_mean_thr50': h_thr, 'feat_mean_thr50': f_thr,
        'h_relative': rel_h, 'feat_relative': rel_f,
        'per_clip_h': m2_h, 'per_clip_feat': m2_f,
    }
    report['gates']['M1'] = gate_m1(m1_thr, m1_hit)
    report['gates']['M2'] = gate_m2(rel_h, rel_f)

    # ---------------------------- M3 -------------------------------------
    # Paired toggle on the TRAINED weights: residual ON vs forced 0, class CE
    # on GT foreground cells.
    m3 = []
    for pos, idx in enumerate(indices):
        on = z_runs[pos]
        off = rollout(model, dataset, idx, use_residual=False)
        tgt = make_target(on, num_classes, pcr)
        fg = tgt[0].eq(1.0).any(dim=0).flatten()
        sel = torch.nonzero(fg, as_tuple=False).flatten()
        if sel.numel() == 0:
            continue
        labels = tgt[0].reshape(num_classes, -1).permute(1, 0)[sel].argmax(dim=1)
        ce_on = F.cross_entropy(class_scores_from_head(on, num_classes)[sel],
                                labels)
        ce_off = F.cross_entropy(class_scores_from_head(off, num_classes)[sel],
                                 labels)
        m3.append({'index': int(idx), 'cells': int(sel.numel()),
                   'ce_on': float(ce_on), 'ce_off': float(ce_off),
                   'delta_off_minus_on': float(ce_off - ce_on)})
    wins = sum(1 for r in m3 if r['delta_off_minus_on'] > 0)
    report['M3_readout_toggle'] = {'per_clip': m3, 'wins': wins,
                                   'n': len(m3)}
    report['gates']['M3'] = gate_m3(m3)

    # ---------------------------- M4 -------------------------------------
    # Does the class evidence in z actually reach the head's decision, and is
    # it CLASS-specific rather than a global bias? Three readings, all on GT
    # foreground cells:
    #   own_hit          residual argmax vs the cell's GT class
    #   donor_transfer   residual argmax vs the DONOR clip's dominant class
    #                    after z is replaced  (donor chosen to be a different
    #                    dominant class where possible)
    #   head_swap        the head's own class argmax after re-running ONLY the
    #                    head with the donor's residual: does the decision
    #                    follow the donor's class?
    # The residual-only readings say what z carries; head_swap says whether the
    # detection path consumes it. Chance is 1/num_classes = 25%; the
    # pre-registered bar is chance + 20 pt.

    def dominant_class(run):
        labels = run['gt_labels']
        if labels.numel() == 0:
            return None
        return int(torch.bincount(labels.long().cpu(),
                                  minlength=num_classes).argmax())

    donor_of = {}
    for pos, idx in enumerate(indices):
        own_dom = dominant_class(z_runs[pos])
        donor_pos = None
        for step in range(1, len(indices)):
            candidate = (pos + step) % len(indices)
            if candidate == pos:
                continue
            cand_dom = dominant_class(z_runs[candidate])
            if cand_dom is not None and cand_dom != own_dom:
                donor_pos = candidate
                break
        if donor_pos is None:
            donor_pos = (pos + args.shuffle_offset) % len(indices)
        if donor_pos == pos:
            raise ValueError('donor selection mapped a clip onto itself')
        donor_of[pos] = donor_pos

    m4 = []
    for pos, idx in enumerate(indices):
        run = z_runs[pos]
        donor_pos = donor_of[pos]
        donor_run = z_runs[donor_pos]
        tgt = make_target(run, num_classes, pcr)
        fg = tgt[0].eq(1.0).any(dim=0).flatten()
        sel = torch.nonzero(fg, as_tuple=False).flatten()
        if sel.numel() == 0 or run['residual'] is None:
            continue
        labels = tgt[0].reshape(num_classes, -1).permute(1, 0)[sel].argmax(dim=1)
        own = residual_class_map(run, num_classes)[sel].argmax(dim=1)
        donor_res = residual_class_map(donor_run, num_classes)
        donor_dom = dominant_class(donor_run)
        donor_pred = donor_res[sel].argmax(dim=1)

        # Head-level swap: same features, same weights, only the residual
        # changes. The donor residual must be resized/masked onto this clip's
        # grid the same way the detector does.
        donor_res_head = donor_run['residual']
        if donor_res_head is not None:
            swapped = head_scores_with_residual(model, run, donor_res_head)
            head_map = class_scores_from_head({'cls_score': swapped},
                                             num_classes)
            head_arg = head_map[sel].argmax(dim=1)
            head_swap = float((head_arg == donor_dom).float().mean())
        else:
            head_swap = None

        m4.append({
            'index': int(idx),
            'donor_index': int(indices[donor_pos]),
            'own_dominant_class': dominant_class(run),
            'donor_dominant_class': donor_dom,
            'cells': int(sel.numel()),
            'own_hit': float((own == labels).float().mean()),
            'donor_transfer': float((donor_pred == donor_dom).float().mean()),
            'head_swap_to_donor': head_swap,
            'changed_frac': float((donor_pred != own).float().mean()),
        })

    def _m(rows, key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return float(sum(vals)) / len(vals) if vals else None

    chance = 1.0 / num_classes
    own_hit = _m(m4, 'own_hit')
    transfer = _m(m4, 'donor_transfer')
    head_swap = _m(m4, 'head_swap_to_donor')
    report['M4_class_information'] = {
        'per_clip': m4, 'chance': chance, 'bar': chance + 0.20,
        'mean_own_hit': own_hit, 'mean_donor_transfer': transfer,
        'mean_head_swap_to_donor': head_swap,
        'mean_changed_frac': _m(m4, 'changed_frac'),
        'donor_rule': ('first clip (cyclically) whose dominant GT class differs '
                       'from this clip\'s; falls back to --shuffle-offset'),
        'note': ('own_hit = z residual argmax vs the cell GT class; '
                 'donor_transfer = residual argmax vs the donor clip dominant '
                 'class after swapping z; head_swap_to_donor = the HEAD class '
                 'argmax after re-running only the head with the donor '
                 'residual. Pre-registered bar: chance + 20 pt on the readings '
                 'that describe z (own_hit / donor_transfer); head_swap is '
                 'reported with the same bar.'),
    }
    report['gates']['M4'] = gate_m4(own_hit, transfer, num_classes)

    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
    with open(output, 'w') as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    print(json.dumps(report, indent=2, sort_keys=True))
    print('[json]', output)
    if not all(v is True for v in report['gates'].values()):
        print('\n[INFO] at least one S1 gate did not pass — see the JSON above')


if __name__ == '__main__':
    main()
