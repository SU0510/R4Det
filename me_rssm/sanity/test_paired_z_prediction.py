"""CPU contract tests; deliberately do not import the MMDetection3D stack."""
import copy
import importlib.util
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from torch import nn

_spec = importlib.util.spec_from_file_location(
    'paired_z_prediction', Path(__file__).with_name('paired_z_prediction.py'))
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
PredictionAudit = _module.PredictionAudit
assert_paired_records = _module.assert_paired_records
fingerprint = _module.fingerprint
selected_candidates = _module.selected_candidates


class FakeFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.prior_mu = nn.Identity()
        self.calls = []

    def forward(self, feat, velocity=None, use_posterior=True,
                deterministic=True, detach_state=True):
        self.calls.append(use_posterior)
        h = feat * 0 + 1
        prior = self.prior_mu(h)
        z = feat + h if use_posterior else prior
        return feat + z, None, None, h, z, {}


class FakeHead(nn.Module):
    num_classes = 2
    box_code_size = 1
    test_cfg = {'nms_pre': 2}
    bbox_coder = SimpleNamespace(decode=lambda anchors, codes: anchors + codes)

    def __init__(self):
        super().__init__()
        self._iou_preds = None

    def forward(self, feats, cls_logit_residual=None):
        x = feats[0]
        self._iou_preds = [x.sigmoid()]  # actual A2-R has this auxiliary branch
        cls = x.repeat(1, 2, 1, 1)
        if cls_logit_residual is not None:
            cls = cls + cls_logit_residual
        return [cls], [x], [torch.cat((x, -x), dim=1)]

    def get_bboxes_single(self, cls_scores, bbox_preds, dir_cls_preds,
                          mlvl_anchors, input_meta, cfg=None, rescale=False):
        return cls_scores, bbox_preds, dir_cls_preds


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.temporal_fusion = FakeFusion()
        self.pts_bbox_head = FakeHead()

    def extract_feat(self, points, img, img_metas, is_valid_mask=None,
                     feat_or_dict=0, rssm_deterministic=None):
        out = self.temporal_fusion(img, deterministic=True)
        return {'pts_feats': [out[0]], 'rssm_cls_logit_residual': out[4] * 0.1}


def run(model, mode, history=0, candidate_dir=None):
    with PredictionAudit(model, mode, candidate_dir) as audit:
        audit.begin_frame('fixed-input')
        x = torch.tensor([[[[0.25, -0.5, 0.9]]]])
        for _ in range(history):
            model.extract_feat([], x, [], feat_or_dict=0)
        feature = model.extract_feat([], x, [], feat_or_dict=1)
        outs = model.pts_bbox_head(
            feature['pts_feats'], feature['rssm_cls_logit_residual'])
        model.pts_bbox_head.get_bboxes_single(
            [outs[0][0][0]], [outs[1][0][0]], [outs[2][0][0]],
            [torch.zeros(3, 1)], {'sample_idx': 42})
        audit.finish_frame()
    return outs, audit.records


class PredictionAuditTests(unittest.TestCase):
    def test_identity_preserves_outputs_and_restores_methods(self):
        model = FakeModel()
        before = [set(x.__dict__) for x in
                  (model, model.temporal_fusion, model.pts_bbox_head)]
        first, rows = run(model, 'identity')
        second, again = run(model, 'identity')
        self.assertEqual(fingerprint(first), fingerprint(second))
        self.assertEqual(rows, again)
        self.assertEqual(before, [set(x.__dict__) for x in
                                 (model, model.temporal_fusion, model.pts_bbox_head)])
        self.assertFalse(model.temporal_fusion.prior_mu._forward_hooks)

    def test_residual_off_is_score_only(self):
        model = FakeModel()
        normal, left = run(model, 'identity')
        off, right = run(model, 'residual_off')
        report = assert_paired_records(left, right, 'residual_off')
        self.assertEqual(report['z_changed_frames'], 0)
        self.assertEqual(report['cls_changed_frames'], 1)
        self.assertTrue(torch.equal(normal[1][0], off[1][0]))
        self.assertTrue(torch.equal(normal[2][0], off[2][0]))

    def test_prior_targets_current_despite_missing_history(self):
        for history in (0, 1, 3):
            model = FakeModel()
            _, normal = run(model, 'identity', history)
            model.temporal_fusion.calls = []
            _, prior = run(model, 'prior_only', history)
            self.assertEqual(model.temporal_fusion.calls, [True] * history + [False])
            self.assertEqual(prior[0]['z_sha'], prior[0]['prior_sha'])
            self.assertEqual(assert_paired_records(normal, prior, 'prior_only')
                             ['z_changed_frames'], 1)

    def test_failed_inference_restores_methods(self):
        model = FakeModel()
        original = model.temporal_fusion.forward
        with self.assertRaisesRegex(RuntimeError, 'inference broke'):
            with PredictionAudit(model, 'identity'):
                raise RuntimeError('inference broke')
        self.assertEqual(model.temporal_fusion.forward, original)
        self.assertNotIn('forward', model.temporal_fusion.__dict__)
        self.assertFalse(model.temporal_fusion.prior_mu._forward_hooks)

    def test_no_silent_missing_current_or_head(self):
        model = FakeModel()
        with PredictionAudit(model, 'identity') as audit:
            audit.begin_frame('fixed')
            with self.assertRaisesRegex(AssertionError, 'one current/head/decode'):
                audit.finish_frame()

    def test_isolation_failure_rejected(self):
        _, records = run(FakeModel(), 'identity')
        changed = copy.deepcopy(records)
        changed[0]['bbox_sha'] = 'corrupt'
        with self.assertRaisesRegex(AssertionError, 'changed bbox_sha'):
            assert_paired_records(records, changed, 'residual_off')
        with self.assertRaises(AssertionError):
            assert_paired_records(records, [], 'prior_only')

    def test_prior_must_reach_returned_latent(self):
        model = FakeModel()
        original = model.temporal_fusion.forward

        def broken_prior(*args, **kwargs):
            import inspect
            bound = inspect.signature(original).bind(*args, **kwargs)
            bound.apply_defaults()
            out = original(*args, **kwargs)
            if bound.arguments['use_posterior'] is False:
                out = out[:4] + (out[4] + 10,) + out[5:]
            return out

        # Keep the production signature so the wrapper can bind it.
        import functools
        model.temporal_fusion.forward = functools.wraps(original)(broken_prior)
        with self.assertRaisesRegex(AssertionError, 'did not reach returned z'):
            run(model, 'prior_only')

    def test_unsupported_arms_rejected(self):
        with self.assertRaises(ValueError):
            PredictionAudit(FakeModel(), 'zero_z')
        with self.assertRaises(ValueError):
            assert_paired_records([], [], 'zero_z')

    def test_candidates_follow_deployed_topk(self):
        head = FakeHead()
        cls = torch.tensor([[[3., -4., 2.]], [[-2., -3., 4.]]])
        reg, direction = torch.arange(3.).view(1, 1, 3), torch.zeros(2, 1, 3)
        anchors = torch.arange(3.).view(3, 1) * 10
        row = selected_candidates(head, [cls], [reg], [direction],
                                  [anchors], head.test_cfg)[0]
        self.assertEqual(row['anchor_id'].tolist(), [2, 0])
        self.assertEqual(row['boxes_before_direction'].flatten().tolist(), [22., 0.])
        all_rows = selected_candidates(head, [cls], [reg], [direction],
                                       [anchors], {'nms_pre': -1})
        self.assertEqual(all_rows[0]['anchor_id'].tolist(), [0, 1, 2])
        with self.assertRaises(AssertionError):
            selected_candidates(head, [cls], [], [direction], [anchors], {})

    def test_candidate_artifact_is_written_and_does_not_change_outputs(self):
        model = FakeModel()
        expected, _ = run(model, 'identity')
        with tempfile.TemporaryDirectory() as directory:
            actual, _ = run(model, 'identity', candidate_dir=directory)
            rows = torch.load(str(Path(directory) / '000000.pt'))
            self.assertEqual(len(rows), 1)
            self.assertEqual(len(rows[0]['anchor_id']), 2)
            self.assertEqual(fingerprint(actual), fingerprint(expected))

    def test_fingerprints_include_shape_dtype_content_and_order(self):
        x = torch.tensor([1., 2.])
        self.assertEqual(fingerprint(x), fingerprint(x.clone()))
        self.assertNotEqual(fingerprint(x), fingerprint(x.view(1, 2)))
        self.assertNotEqual(fingerprint(x), fingerprint(x.double()))
        self.assertNotEqual(fingerprint(x), fingerprint(x.flip(0)))
        self.assertNotEqual(fingerprint([x, None]), fingerprint([None, x]))
        with self.assertRaises(TypeError):
            fingerprint(object())


if __name__ == '__main__':
    unittest.main()
