#!/usr/bin/env python3
"""Read-only prediction-level A2-R audit; no model/config/source rewriting.

Run unwrapped reference, wrapped identity, residual_off, prior_only in order.
Identity gates every intervention. Use deployed detector and repository KITTI
AP. --limit is wiring smoke ONLY; deliberately no AP. Training-stack imports
are lazy: --help needs only Python.
"""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import random
import sys
import time
from collections.abc import Mapping
from contextlib import ExitStack

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ARMS = ('reference', 'identity', 'residual_off', 'prior_only')
PRIMARY = 'pts_bbox/KITTI/Overall_3D_moderate'


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value):
    """Hash values, not pickle storage IDs; reject unknown input types."""
    import numpy as np
    import torch
    digest = hashlib.sha256()

    def visit(x):
        tag = type(x).__module__ + '.' + type(x).__qualname__
        digest.update((tag + '\0').encode())
        if torch.is_tensor(x):
            y = x.detach().cpu().contiguous()
            digest.update(str((tuple(y.shape), y.dtype)).encode())
            digest.update(y.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(x, np.ndarray):
            if x.dtype.hasobject:
                raise TypeError('object arrays have no stable byte fingerprint')
            digest.update(str((x.shape, x.dtype)).encode())
            digest.update(x.tobytes())
        elif isinstance(x, Mapping):
            for k in sorted(x, key=str):
                visit(k)
                visit(x[k])
        elif isinstance(x, (list, tuple)):
            digest.update(str(len(x)).encode())
            for item in x:
                visit(item)
        elif x is None or isinstance(x, (str, int, float, bool, np.generic)):
            digest.update(repr(x).encode())
        elif isinstance(x, type):
            digest.update((x.__module__ + '.' + x.__qualname__).encode())
        elif hasattr(x, 'tensor'):
            visit(x.tensor)
        elif type(x).__name__ == 'DataContainer':
            visit(x.data)
        else:
            raise TypeError('no stable fingerprint for ' + tag)
        digest.update(b'\0END\0')

    visit(value)
    return digest.hexdigest()


def replace_method(stack, obj, name, replacement):
    """Restore instance dictionary exactly, including on inference failure."""
    existed = name in obj.__dict__
    old = obj.__dict__.get(name)
    setattr(obj, name, replacement)

    def restore():
        if existed:
            setattr(obj, name, old)
        else:
            delattr(obj, name)
    stack.callback(restore)


def selected_candidates(head, cls_scores, bbox_preds, dir_preds, anchors, cfg):
    """Mirror get_bboxes_single pre-NMS selection, without changing it.

    Save selected candidates only, not every anchor. Decoded boxes are before
    direction correction, exactly as they enter deployed NMS. Separately hash
    full raw head tensors to prove score-only isolation.
    """
    import torch
    rows = []
    if not (len(cls_scores) == len(bbox_preds) == len(dir_preds) == len(anchors)):
        raise AssertionError('candidate level counts differ')
    for level, (cls, reg, direction, anchor) in enumerate(zip(
            cls_scores, bbox_preds, dir_preds, anchors)):
        logits = cls.permute(1, 2, 0).reshape(-1, head.num_classes)
        scores = logits.sigmoid()
        codes = reg.permute(1, 2, 0).reshape(-1, head.box_code_size)
        dirs = direction.permute(1, 2, 0).reshape(-1, 2).argmax(-1)
        count = len(scores)
        ids = torch.arange(count, device=scores.device)
        nms_pre = cfg.get('nms_pre', -1)
        if nms_pre > 0 and count > nms_pre:
            ids = scores.max(dim=1)[0].topk(nms_pre)[1]
        selected = dict(level=level, n_total=count, anchor_id=ids,
                        scores=scores[ids], box_codes=codes[ids],
                        anchors=anchor[ids], dir_class=dirs[ids],
                        boxes_before_direction=head.bbox_coder.decode(
                            anchor[ids], codes[ids]))
        rows.append({k: v.detach().cpu() if torch.is_tensor(v) else v
                     for k, v in selected.items()})
    return rows


class PredictionAudit:
    """Select current frame by feat_or_dict, NOT number of fusion calls.

    Missing history can skip fusion.forward. Counting N calls would miss the
    current frame on exactly those hard sequences.
    """
    def __init__(self, model, mode, candidate_dir=None):
        if mode not in ARMS[1:]:
            raise ValueError('unsupported audit arm: ' + mode)
        self.model, self.mode = model, mode
        self.candidate_dir = Path(candidate_dir) if candidate_dir else None
        self.current = False
        self.records = []
        self.row = None
        self.stack = ExitStack()

    def begin_frame(self, input_sha):
        self.row = dict(input_sha=input_sha, current_calls=0, history_calls=0,
                        head_calls=0, decode_calls=0)

    def finish_frame(self):
        if tuple(self.row[k] for k in (
                'current_calls', 'head_calls', 'decode_calls')) != (1, 1, 1):
            raise AssertionError('expected one current/head/decode call: ' +
                                 repr(self.row))
        if self.mode == 'prior_only' and self.row['z_sha'] != self.row['prior_sha']:
            raise AssertionError('prior intervention did not reach returned z')
        self.records.append(self.row)
        self.row = None

    def __enter__(self):
        import torch
        model, fusion, head = (self.model, self.model.temporal_fusion,
                               self.model.pts_bbox_head)
        extract, forward, predict, decode = (model.extract_feat, fusion.forward,
                                             head.forward, head.get_bboxes_single)
        extract_sig, forward_sig = inspect.signature(extract), inspect.signature(forward)
        predict_sig, decode_sig = inspect.signature(predict), inspect.signature(decode)

        def extract_wrapped(*args, **kwargs):
            bound = extract_sig.bind(*args, **kwargs)
            bound.apply_defaults()
            was_current = self.current
            self.current = bound.arguments['feat_or_dict'] == 1
            if self.current:
                mask = bound.arguments.get('is_valid_mask')
                self.row['last_history_valid'] = (
                    mask.detach().cpu().bool().tolist() if mask is not None else None)
            try:
                out = extract(*args, **kwargs)
                if self.current:
                    self.row['feature_sha'] = fingerprint(out['pts_feats'])
                return out
            finally:
                self.current = was_current

        def forward_wrapped(*args, **kwargs):
            if not self.current:
                self.row['history_calls'] += 1
                return forward(*args, **kwargs)
            bound = forward_sig.bind(*args, **kwargs)
            bound.apply_defaults()
            if not bound.arguments['deterministic']:
                raise AssertionError('this audit requires deterministic eval')
            if not bound.arguments['use_posterior']:
                raise AssertionError('reference path must use posterior')
            if self.mode == 'prior_only':
                bound.arguments['use_posterior'] = False
            self.row['current_calls'] += 1
            self.row['fusion_input_sha'] = fingerprint(bound.arguments['feat'])
            out = forward(*bound.args, **bound.kwargs)
            self.row.update(h_sha=fingerprint(out[3]), z_sha=fingerprint(out[4]),
                            z_rms=float(out[4].float().square().mean().sqrt()))
            return out

        def prior_hook(module, inputs, output):
            if self.current:
                self.row['prior_sha'] = fingerprint(output)

        def predict_wrapped(*args, **kwargs):
            bound = predict_sig.bind(*args, **kwargs)
            bound.apply_defaults()
            residual = bound.arguments['cls_logit_residual']
            if residual is None:
                raise AssertionError('A2-R deployed residual is missing')
            self.row['residual_sha'] = fingerprint(residual)
            self.row['residual_rms'] = float(residual.float().square().mean().sqrt())
            if self.mode == 'residual_off':
                bound.arguments['cls_logit_residual'] = None
            out = predict(*bound.args, **bound.kwargs)
            if len(out) != 3:
                raise AssertionError('only three-output Anchor3DHead supported')
            self.row.update(head_calls=self.row['head_calls'] + 1,
                            cls_sha=fingerprint(out[0]), bbox_sha=fingerprint(out[1]),
                            dir_sha=fingerprint(out[2]),
                            iou_sha=fingerprint(getattr(head, '_iou_preds', None)))
            return out

        def decode_wrapped(*args, **kwargs):
            bound = decode_sig.bind(*args, **kwargs)
            bound.apply_defaults()
            values = bound.arguments
            self.row['decode_calls'] += 1
            self.row['sample_idx'] = str(values['input_meta']['sample_idx'])
            if self.candidate_dir is not None:
                rows = selected_candidates(
                    head, values['cls_scores'], values['bbox_preds'],
                    values['dir_cls_preds'], values['mlvl_anchors'],
                    values['cfg'] if values['cfg'] is not None else head.test_cfg)
                torch.save(rows, str(self.candidate_dir /
                                     ('%06d.pt' % len(self.records))))
            return decode(*args, **kwargs)

        self.stack.callback(fusion.prior_mu.register_forward_hook(prior_hook).remove)
        replace_method(self.stack, model, 'extract_feat', extract_wrapped)
        replace_method(self.stack, fusion, 'forward', forward_wrapped)
        replace_method(self.stack, head, 'forward', predict_wrapped)
        replace_method(self.stack, head, 'get_bboxes_single', decode_wrapped)
        return self

    def __exit__(self, *exc):
        return self.stack.__exit__(*exc)


def assert_paired_records(identity, other, mode):
    if mode not in ('residual_off', 'prior_only'):
        raise ValueError('unsupported comparison arm: ' + mode)
    if not identity or len(identity) != len(other):
        raise AssertionError('paired record counts differ or are empty')
    keys = ['input_sha', 'sample_idx', 'fusion_input_sha', 'h_sha', 'prior_sha',
            'last_history_valid', 'history_calls']
    if mode == 'residual_off':
        keys += ['z_sha', 'feature_sha', 'bbox_sha', 'dir_sha', 'iou_sha', 'residual_sha']
    for index, (left, right) in enumerate(zip(identity, other)):
        for key in keys:
            if left[key] != right[key]:
                raise AssertionError('%s frame %d changed %s' % (mode, index, key))
    return dict(n_frames=len(identity), invariants=keys,
                z_changed_frames=sum(a['z_sha'] != b['z_sha'] for a, b in zip(identity, other)),
                cls_changed_frames=sum(a['cls_sha'] != b['cls_sha'] for a, b in zip(identity, other)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out-dir', required=True, help='must not already exist')
    parser.add_argument('--limit', type=int, help='prefix wiring smoke; NO AP')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    if args.workers < 0:
        parser.error('--workers must be nonnegative')
    # Match train_vod/test_vod explicitly. Never copy their source-file swaps.
    import training_old_version
    training_old_version.use_old_version()
    import mmcv
    import numpy as np
    import torch
    from mmcv import Config
    from mmcv.parallel import MMDataParallel
    from mmcv.runner import load_checkpoint
    from mmdet3d.datasets import build_dataset, build_dataloader
    from mmdet3d.models import build_model
    from me_rssm.sanity.eval_dump_c import eval_kwargs_from_cfg
    try:
        from mmdet.utils import compat_cfg, setup_multi_processes
    except ImportError:
        from mmdet3d.utils import compat_cfg, setup_multi_processes

    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if visible not in ('5', '6', '7'):
        parser.error('set CUDA_VISIBLE_DEVICES to exactly one of 5,6,7')
    if not torch.cuda.is_available():
        parser.error('CUDA unavailable; --help and CPU tests need no GPU')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)
    cfg = compat_cfg(Config.fromfile(args.config))
    setup_multi_processes(cfg)
    cfg.model.pretrained, cfg.model.train_cfg = None, None
    cfg.data.val.test_mode = True
    dataset = build_dataset(cfg.data.val)
    full_count = len(dataset)
    if args.limit is not None:
        dataset.data_infos = dataset.data_infos[:args.limit]
    if not len(dataset):
        raise ValueError('empty validation dataset')
    (out_dir / 'figures').mkdir()
    cfg.model.meta_info = dict(project_name=Path(args.config).stem,
                               figures_path=str(out_dir / 'figures'))
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu', strict=True)
    epoch = checkpoint.get('meta', {}).get('epoch')
    del checkpoint
    model.CLASSES = dataset.CLASSES
    model.eval()
    fusion, head = model.temporal_fusion, model.pts_bbox_head
    if (not getattr(fusion, 'cls_residual_enabled', False) or
            getattr(fusion, 'posterior_struct', 'standard') != 'standard' or
            model.ped_center_head is not None or not head.use_sigmoid_cls or
            not head.use_direction_classifier):
        raise ValueError('only standard A/A2-R sigmoid Anchor3DHead supported')
    parallel = MMDataParallel(model, device_ids=[0])
    parallel.cuda(0).eval()
    state_sha = fingerprint(model.state_dict())
    sources = [args.config, args.checkpoint, cfg.data.val.ann_file,
               __file__, ROOT / 'mmdet3d/models/detectors/R4Det.py',
               ROOT / 'mmdet3d/models/fusion_layers/rssm_fusion.py',
               ROOT / 'me_rssm/semantic_readout_rssm.py',
               ROOT / 'mmdet3d/models/dense_heads/anchor3d_head.py',
               ROOT / 'mmdet3d/core/evaluation/kitti_utils/eval.py',
               ROOT / 'mmdet3d/datasets/kitti_dataset.py',
               inspect.getfile(type(head.bbox_coder)),
               ROOT / 'me_rssm/sanity/eval_dump_c.py']
    cfg.dump(str(out_dir / 'resolved_config.py'))
    sources.append(out_dir / 'resolved_config.py')
    report = dict(status='running', config=str(Path(args.config).resolve()),
                  checkpoint=str(Path(args.checkpoint).resolve()),
                  sources={str(Path(p).resolve()): sha256_file(p) for p in sources},
                  state_sha=state_sha, cuda_visible_devices=visible,
                  classes=list(dataset.CLASSES), epoch=epoch, split='val', full_val_count=full_count,
                  head_contract=dict(num_anchors=head.num_anchors,
                                     num_classes=head.num_classes,
                                     box_code_size=head.box_code_size,
                                     use_iou_branch=head.use_iou_branch,
                                     dir_offset=head.dir_offset,
                                     dir_limit_offset=head.dir_limit_offset,
                                     test_cfg=dict(head.test_cfg)),
                  n_frames=len(dataset), smoke=args.limit is not None,
                  legacy_training_version=training_old_version.get_old_version(),
                  torch_version=torch.__version__, arms={}, checks={})

    def save_report():
        with open(out_dir / 'report.json', 'w') as handle:
            json.dump(report, handle, indent=2, sort_keys=True)

    reference_sha, input_reference, identity_records = None, None, None
    save_report()
    try:
        for arm in ARMS:
            random.seed(0)
            np.random.seed(0)
            torch.manual_seed(0)
            torch.cuda.manual_seed_all(0)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            model.temporal_fusion.reset_state()
            loader = build_dataloader(dataset, samples_per_gpu=1,
                                     workers_per_gpu=args.workers, dist=False,
                                     shuffle=False, seed=0)
            arm_dir = out_dir / arm
            arm_dir.mkdir()
            candidates = arm_dir / 'candidates'
            candidates.mkdir()
            outputs, input_sha = [], []
            started = time.time()
            with ExitStack() as stack:
                audit = None if arm == 'reference' else stack.enter_context(
                    PredictionAudit(model, arm, candidates))
                progress = mmcv.ProgressBar(len(dataset))
                for data in loader:
                    value_sha = fingerprint(data)
                    input_sha.append(value_sha)
                    if audit:
                        audit.begin_frame(value_sha)
                    with torch.no_grad():
                        results = parallel(return_loss=False, rescale=True, **data)
                    if len(results) != 1:
                        raise AssertionError('audit requires batch size one')
                    if audit:
                        audit.finish_frame()
                    outputs.extend(results)
                    progress.update()
            prediction_sha = fingerprint(outputs)
            if len(outputs) != len(dataset):
                raise AssertionError('prediction/dataset length mismatch')
            if fingerprint(model.state_dict()) != state_sha:
                raise AssertionError('parameters or persistent buffers changed')
            for path, expected_sha in report['sources'].items():
                if sha256_file(path) != expected_sha:
                    raise AssertionError('source changed during audit: ' + path)
            mmcv.dump(outputs, str(arm_dir / 'predictions.pkl'))
            row = dict(seconds=time.time() - started, prediction_sha=prediction_sha,
                       input_sha=input_sha, records=audit.records if audit else [], ap=None)
            report['arms'][arm] = row
            if arm == 'reference':
                reference_sha, input_reference = prediction_sha, input_sha
            else:
                if input_sha != input_reference:
                    raise AssertionError('validation inputs changed between arms')
                if arm == 'identity':
                    if prediction_sha != reference_sha:
                        raise AssertionError('identity changed deployed predictions')
                    identity_records = audit.records
                    report['checks']['identity'] = 'bit-exact predictions and inputs'
                else:
                    report['checks'][arm] = assert_paired_records(
                        identity_records, audit.records, arm)
            if args.limit is None:
                row['ap'] = dataset.evaluate(outputs, **eval_kwargs_from_cfg(cfg, 'bbox'))
                if arm == 'identity' and row['ap'] != report['arms']['reference']['ap']:
                    raise AssertionError('identity evaluator mismatch')
                row['primary_3d_moderate'] = row['ap'][PRIMARY]
                with open(arm_dir / 'ap.json', 'w') as handle:
                    json.dump(row['ap'], handle, indent=2, sort_keys=True)
            save_report()
            del outputs
        if args.limit is None:
            baseline = report['arms']['identity']['ap']
            report['ap_delta_vs_identity'] = {
                arm: {k: v - baseline[k] for k, v in report['arms'][arm]['ap'].items()
                      if isinstance(v, (int, float)) and isinstance(baseline.get(k), (int, float))}
                for arm in ('residual_off', 'prior_only')}
        report['status'] = 'smoke_complete_no_ap' if args.limit is not None else 'full_val_complete'
        save_report()
    except BaseException as error:
        report.update(status='failed', error=type(error).__name__ + ': ' + str(error))
        save_report()
        raise
    print('\n' + report['status'] + ': ' + str(out_dir / 'report.json'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
