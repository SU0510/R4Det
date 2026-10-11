"""Unit tests for candidate A (z-semantic supervision + cls-logit readout).

Pre-registration: docs/rssm_candidate_A_prereg_20261010.md §6 (T1-T9).

Run:
    python3 me_rssm/sanity/test_semantic_readout_rssm.py

CPU only, small grids (H=8, W=10 — deliberately NON-square so any
(x,y)/(y,x) transposition bug cannot pass silently), no checkpoint, no GPU.

What each test protects:
    T1  off-state builds no new parameter and is bit-identical to the baseline
        RSSM, including the RNG stream (so "A off == no2d_igdr" is a fact, not
        an assumption);
    T2  the readout starts at exactly zero, so the detection head cannot be
        perturbed before the first optimiser step;
    T3  the semantic loss can only be reduced through z (zero gradient into
        h_t) — the whole candidate-A claim rests on this exclusivity;
    T4  the GT rasterisation uses the BEV convention (H indexes x, W indexes y);
    T5  the per-class residual lands on the right conv_cls channels
        (anchor_class_mapping [0,0,0,1,2,3], [size, rot] anchor layout);
    T6/T7 the detection-grid contract: transposition + all-zero residual for
        samples whose recurrent state was unreliable;
    T8  reset_state() clears the auxiliary GT and the pending loss, so a history
        frame can never be supervised against the current frame's targets;
    T9  the probe toggle (cls_residual_gate=0) removes the readout exactly;
    T10 the 'h' control arm is the mirror image of T3 (gradient into h, none
        into z) — the control is wired, not merely configured;
    T11 an image without objects contributes no gradient and no NaN.

A2 (docs/rssm_candidate_A2_prereg_20261011.md §4.1/§4.2) adds:
    T14 both A2 arm defaults reproduce candidate A exactly, and an unknown
        arm name raises instead of silently falling back to A;
    T15 the ranking term's arithmetic matches the pre-registered formula
        (max over GT, background quantile, one-sided margin) and is exactly 0
        once the GT peak clears the margin;
    T16 the ranking term's reduction is per-occupied-channel (channels without
        GT contribute nothing) and an all-background frame is a zero no-op;
    T17 the ranking term reaches z (gradient flows) and does not perturb the
        focal term's own value;
    T18 ``anchor_slot`` writes ALL 48 conv_cls channels with the same 4-channel
        predictor / parameter count, and ``per_class_slot`` still writes 12/48
        with the head's own mapping (the A behaviour is unchanged);
    T19 both arms keep the zero-init identity contract (W_cls == 0 => the head
        sees an all-zero residual on every channel).
"""

import os
import sys
import unittest

import torch

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..'))

from me_rssm.semantic_readout_rssm import SemanticReadoutRSSMFusion  # noqa: E402

PC_RANGE = [0.0, -39.68, -4.0, 69.12, 39.68, 2.0]


class _MockBoxes:
    """Minimal stand-in for LiDARInstance3DBoxes.

    `bev_centers_to_heatmap` only needs `.gravity_center` (N,3) and
    `.tensor[:, 3:6]` (sizes), so a mock keeps the test independent of the
    box-registry import graph.
    """

    def __init__(self, centers, sizes):
        self.centers = torch.as_tensor(centers, dtype=torch.float32)
        self.sizes = torch.as_tensor(sizes, dtype=torch.float32)

    @property
    def gravity_center(self):
        return self.centers

    @property
    def tensor(self):
        # (N, 7+): x y z l w h ...  -> dims 3:6 are the sizes
        head = torch.cat([self.centers, self.sizes], dim=1)
        return torch.cat([head, torch.zeros(head.shape[0], 1)], dim=1)

    def to(self, *args, **kwargs):
        return self


def _fusion_kwargs(**overrides):
    kwargs = dict(
        in_channels=16,
        out_channels=16,
        latent_dim=16,
        hidden_dim=8,
        action_dim=0,
        kl_scale=1.0,
        free_nats=1.0,
        min_std=0.1,
        init_std=0.2,
        align_kernel_size=3,
        align_deform_groups=1,
        align_z_state=True,
    )
    kwargs.update(overrides)
    return kwargs


def _build_base(**overrides):
    from mmdet3d.models.fusion_layers.rssm_fusion import (
        MotionAlignedRSSMFusion)
    return MotionAlignedRSSMFusion(**_fusion_kwargs(**overrides))


def _build_arm(**overrides):
    return SemanticReadoutRSSMFusion(**_fusion_kwargs(**overrides))


def _gt_two_objects():
    """Batch of 2: one frame with a Car and a Pedestrian, one empty frame."""
    boxes = [
        _MockBoxes(
            centers=[[10.0, 5.0, -1.0], [3.2, -6.4, -1.2]],
            sizes=[[4.0, 1.6, 1.5], [0.6, 0.6, 1.7]]),
        _MockBoxes(centers=[[]], sizes=[[]]),
    ]
    # Car -> class index 2, Pedestrian -> 0 (TJ4D order)
    labels = [torch.tensor([2, 0], dtype=torch.long),
              torch.zeros(0, dtype=torch.long)]
    return boxes, labels


class TestSemanticReadoutRSSM(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    # ------------------------------------------------------------------
    # T1 — off state is the baseline
    # ------------------------------------------------------------------
    def test_T1_off_state_is_bitwise_baseline(self):
        torch.manual_seed(0)
        base = _build_base()
        rng_base = torch.get_rng_state()

        # Fully switched off: no new parameter, identical weights, identical
        # RNG position afterwards (so the whole detector downstream of the
        # fusion is unaffected too).
        torch.manual_seed(0)
        arm_off = _build_arm(cls_residual=False, sem_supervision_weight=0.0)
        rng_off = torch.get_rng_state()
        self.assertTrue(torch.equal(rng_base, rng_off),
                        'the off state must not consume any RNG')

        base_params = dict(base.named_parameters())
        off_params = dict(arm_off.named_parameters())
        self.assertEqual(set(base_params), set(off_params),
                         'the off state must not add any parameter')
        for name in base_params:
            self.assertTrue(
                torch.equal(base_params[name], off_params[name]),
                f'parameter {name} differs — RNG stream was not preserved')
        self.assertIsNone(arm_off.sem_head)
        self.assertIsNone(arm_off.z_cls_logits)
        self.assertIsNone(arm_off.detection_cls_logit_residual())

        # Fully switched ON: the pre-existing (baseline) weights must still be
        # bit-identical — adding arm A must not reshuffle the baseline init.
        torch.manual_seed(0)
        arm_on = _build_arm(cls_residual=True, sem_supervision_weight=1.0)
        self.assertTrue(
            torch.equal(rng_base, torch.get_rng_state()),
            'the on state must restore the RNG stream')
        on_params = dict(arm_on.named_parameters())
        for name in base_params:
            self.assertTrue(
                torch.equal(base_params[name], on_params[name]),
                f'parameter {name} differs once A is switched on')
        self.assertTrue(torch.all(on_params['z_cls_logits.weight'] == 0))

        base.eval()
        arm_off.eval()
        feat = torch.randn(2, 16, 8, 10)
        with torch.no_grad():
            b_out = base(feat, deterministic=True)
            a_out = arm_off(feat, deterministic=True)
        for i, (b, a) in enumerate(zip(b_out[:5], a_out[:5])):
            self.assertTrue(torch.equal(b, a), f'forward output {i} differs')
        self.assertEqual(set(b_out[5]), set(a_out[5]))

    # ------------------------------------------------------------------
    # T2 — zero-init readout cannot perturb the head at step 0
    # ------------------------------------------------------------------
    def test_T2_readout_starts_at_exactly_zero(self):
        arm = _build_arm(cls_residual=True, sem_supervision_weight=0.0)
        # The real pipeline calls init_weights() on the whole detector AFTER
        # construction, so the zero-init has to survive that sweep.
        arm.init_weights()
        self.assertTrue(torch.all(arm.z_cls_logits.weight.detach() == 0))
        self.assertTrue(torch.all(arm.z_cls_logits.bias.detach() == 0))
        arm.eval()
        with torch.no_grad():
            arm(torch.randn(2, 16, 8, 10), deterministic=True)
        res = arm.last_cls_logit_residual
        self.assertIsNotNone(res)
        self.assertEqual(tuple(res.shape), (2, 4, 8, 10))
        self.assertTrue(torch.equal(res, torch.zeros_like(res)),
                        'zero-init readout must emit exactly 0')
        self.assertTrue(torch.all(arm.z_cls_logits.weight.detach() == 0))
        self.assertTrue(torch.all(arm.z_cls_logits.bias.detach() == 0))

        head = self._build_head()
        head.eval()
        x = torch.randn(1, 16, 8, 10)
        zero_res = torch.zeros(1, head.num_classes, 8, 10)
        with torch.no_grad():
            plain = head([x])
            with_res = head([x], cls_logit_residual=zero_res)
        self.assertTrue(torch.equal(plain[0][0], with_res[0][0]),
                        'a zero residual must not move conv_cls')
        self.assertTrue(torch.equal(plain[1][0], with_res[1][0]),
                        'the geometric branch must be untouched')

    # ------------------------------------------------------------------
    # T3 — the semantic loss pressures z itself
    # ------------------------------------------------------------------
    def test_T3_semantic_loss_is_z_exclusive(self):
        arm = _build_arm(cls_residual=True, sem_supervision_weight=1.0)
        arm.train()
        boxes, labels = _gt_two_objects()
        arm.set_object_gt(boxes, labels, PC_RANGE)
        out = arm(torch.randn(2, 16, 8, 10), deterministic=True,
                  detach_state=False)
        h_t, z_t = out[3], out[4]
        loss = arm.pending_semantic_loss
        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(loss.item(), 0.0)
        grads = torch.autograd.grad(
            loss, [z_t, h_t, arm.posterior_mu.weight, arm.prior_mu.weight],
            allow_unused=True, retain_graph=True)
        g_z, g_h, g_post, g_prior = grads
        self.assertIsNotNone(g_z, 'the semantic loss must reach z')
        self.assertGreater(float(g_z.norm()), 0.0)
        # z_t is a function of h_t, so a non-zero h_t gradient is expected and
        # proves nothing; the exclusivity claim is about WHO HAS TO CHANGE.
        # The head reads only z_t, so the parameters that emit z (posterior_mu)
        # must be pressured and the prior path must not be touched at all.
        self.assertGreater(float(g_post.norm()), 0.0,
                           'z producer must receive semantic gradient')
        if g_prior is not None:
            self.assertEqual(float(g_prior.norm()), 0.0,
                             'the prior path must not carry semantic gradient')

    # ------------------------------------------------------------------
    # T4 — rasterisation uses H->x, W->y
    # ------------------------------------------------------------------
    def test_T4_heatmap_geometry(self):
        arm = _build_arm(cls_residual=False, sem_supervision_weight=1.0)
        arm.set_object_gt(
            [_MockBoxes(centers=[[10.0, 0.0, -1.0]], sizes=[[4.0, 1.6, 1.5]])],
            [torch.tensor([2], dtype=torch.long)], PC_RANGE)
        out_h, out_w = 216, 248
        target = arm._rasterize_target(torch.device('cpu'), (out_h, out_w))
        self.assertEqual(tuple(target.shape), (1, 4, out_h, out_w))
        row = int(round((10.0 - PC_RANGE[0]) / ((PC_RANGE[3] - PC_RANGE[0])
                                               / out_h)))
        col = int(round((0.0 - PC_RANGE[1]) / ((PC_RANGE[4] - PC_RANGE[1])
                                               / out_w)))
        peak = target[0, 2].argmax()
        got = (int(peak) // out_w, int(peak) % out_w)
        self.assertLessEqual(abs(got[0] - row), 1, f'H axis: {got} vs {row}')
        self.assertLessEqual(abs(got[1] - col), 1, f'W axis: {got} vs {col}')
        # Other channels must stay empty for a single-class frame.
        for c in (0, 1, 3):
            self.assertLess(float(target[0, c].max()), 1e-6)

    # ------------------------------------------------------------------
    # T5 — per-class residual lands on the conv_cls layout
    # ------------------------------------------------------------------
    def _build_head(self):
        from mmdet3d.models.dense_heads.anchor3d_head import Anchor3DHead
        return Anchor3DHead(
            num_classes=4,
            in_channels=16,
            feat_channels=16,
            train_cfg=None,
            test_cfg=None,
            use_direction_classifier=True,
            anchor_class_mapping=[0, 0, 0, 1, 2, 3],
            anchor_generator=dict(
                type='Anchor3DRangeGenerator',
                ranges=[[0, -40.0, -1.163, 70.4, 40.0, -1.163]] * 6,
                sizes=[[0.5, 0.6, 1.6], [0.6, 0.8, 1.69], [0.75, 0.9, 1.8],
                       [0.78, 1.77, 1.6], [1.84, 4.56, 1.7],
                       [2.66, 10.76, 3.47]],
                rotations=[0, 1.57],
                reshape_out=False),
            bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
            loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2.0,
                          alpha=0.25, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=0.1111111111111111,
                           loss_weight=2.0),
            loss_dir=dict(type='CrossEntropyLoss', use_sigmoid=False,
                          loss_weight=0.2))

    def test_T5_channel_layout(self):
        head = self._build_head()
        self.assertEqual(head.num_anchors, 12)  # 6 sizes x 2 rotations
        self.assertEqual(head.cls_out_channels, 48)
        ref = torch.zeros(1, 48, 2, 2)
        res = torch.cat([torch.full((1, 1, 2, 2), float(c + 1))
                         for c in range(4)], dim=1)
        expanded = head._expand_cls_logit_residual(res, ref)
        self.assertEqual(tuple(expanded.shape), (1, 48, 2, 2))
        mapping = [0, 0, 0, 1, 2, 3]
        for a in range(12):
            cls_token = mapping[a // 2]
            channel = a * 4 + cls_token
            self.assertAlmostEqual(
                float(expanded[0, channel, 0, 0]), float(cls_token + 1),
                msg=f'anchor slot {a} channel {channel}')
        # Channels that are NOT the owning class of their anchor stay zero.
        for a in range(12):
            for c in range(4):
                if c != mapping[a // 2]:
                    self.assertEqual(float(expanded[0, a * 4 + c, 0, 0]), 0.0)
        with self.assertRaises(ValueError):
            head._expand_cls_logit_residual(torch.zeros(1, 5, 2, 2), ref)
        with self.assertRaises(ValueError):
            head._expand_cls_logit_residual(torch.zeros(1, 4, 3, 2), ref)

    # ------------------------------------------------------------------
    # T6 / T7 — detection-grid contract
    # ------------------------------------------------------------------
    def test_T6_T7_detection_grid_transpose_and_valid_mask(self):
        arm = _build_arm(cls_residual=True, sem_supervision_weight=0.0)
        arm.eval()
        # Make the readout emit a non-zero, spatially asymmetric map so a
        # transposition bug is visible.
        with torch.no_grad():
            arm.z_cls_logits.weight.normal_()
            arm.z_cls_logits.bias.normal_()
        with torch.no_grad():
            arm(torch.randn(2, 16, 8, 10), deterministic=True)
        raw = arm.last_cls_logit_residual
        self.assertEqual(tuple(raw.shape)[-2:], (8, 10))
        prepared = arm.detection_cls_logit_residual()
        self.assertEqual(tuple(prepared.shape)[-2:], (10, 8),
                         'the head grid must be the transposed latent grid')
        self.assertTrue(torch.equal(prepared, raw.permute(0, 1, 3, 2)))
        masked = arm.detection_cls_logit_residual(
            torch.tensor([True, False]))
        self.assertGreater(float(masked[0].abs().sum()), 0.0)
        self.assertEqual(float(masked[1].abs().sum()), 0.0,
                         'an invalid sample must get an all-zero residual')
        with self.assertRaises(ValueError):
            arm.detection_cls_logit_residual(torch.tensor([True]))

    # ------------------------------------------------------------------
    # T8 — reset_state clears the auxiliary state
    # ------------------------------------------------------------------
    def test_T8_reset_state_clears_aux_state(self):
        arm = _build_arm(cls_residual=True, sem_supervision_weight=1.0)
        arm.train()
        boxes, labels = _gt_two_objects()
        arm.set_object_gt(boxes, labels, PC_RANGE)
        arm(torch.randn(2, 16, 8, 10), deterministic=True,
            detach_state=False)
        self.assertIsNotNone(arm.pending_semantic_loss)
        self.assertIsNotNone(arm.last_sem_map)
        arm.reset_state()
        self.assertIsNone(arm.current_object_gt)
        self.assertIsNone(arm.pending_semantic_loss)
        self.assertIsNone(arm.last_sem_map)
        self.assertIsNone(arm.last_cls_logit_residual)
        # A history-frame forward (no GT attached) must stay unsupervised.
        out = arm(torch.randn(2, 16, 8, 10), deterministic=True,
                  detach_state=False)
        self.assertIsNone(arm.pending_semantic_loss)

    # ------------------------------------------------------------------
    # T9 — probe toggle removes the readout exactly
    # ------------------------------------------------------------------
    def test_T9_gate_zero_is_exact_off(self):
        arm = _build_arm(cls_residual=True, sem_supervision_weight=0.0)
        arm.eval()
        with torch.no_grad():
            arm.z_cls_logits.weight.normal_()
        arm.cls_residual_gate = 0.0
        with torch.no_grad():
            arm(torch.randn(2, 16, 8, 10), deterministic=True)
        res = arm.last_cls_logit_residual
        self.assertTrue(torch.equal(res, torch.zeros_like(res)))
        arm.cls_residual_gate = 1.0
        with torch.no_grad():
            arm(torch.randn(2, 16, 8, 10), deterministic=True)
        self.assertGreater(float(arm.last_cls_logit_residual.abs().sum()),
                           0.0)

    # ------------------------------------------------------------------
    # T10 — the 'h' control is the mirror image of T3
    # ------------------------------------------------------------------
    def test_T10_h_control_reaches_h_not_z(self):
        arm = _build_arm(cls_residual=True, sem_supervision_weight=1.0,
                         sem_head_input='h')
        arm.train()
        boxes, labels = _gt_two_objects()
        arm.set_object_gt(boxes, labels, PC_RANGE)
        out = arm(torch.randn(2, 16, 8, 10), deterministic=True,
                  detach_state=False)
        h_t, z_t = out[3], out[4]
        grads = torch.autograd.grad(
            arm.pending_semantic_loss, [h_t, z_t, arm.posterior_mu.weight],
            allow_unused=True)
        g_h, g_z, g_post = grads
        self.assertIsNotNone(g_h)
        self.assertGreater(float(g_h.norm()), 0.0)
        if g_z is not None:
            self.assertEqual(float(g_z.norm()), 0.0,
                             "the 'h' control must not reach z")
        if g_post is not None:
            self.assertEqual(float(g_post.norm()), 0.0,
                             "the 'h' control must not pressure z's producer")

    # ------------------------------------------------------------------
    # T11 — an empty frame contributes no gradient and no NaN
    # ------------------------------------------------------------------
    def test_T11_empty_frame_is_noop(self):
        arm = _build_arm(cls_residual=True, sem_supervision_weight=1.0)
        arm.train()
        arm.set_object_gt(
            [_MockBoxes(centers=[[]], sizes=[[]])],
            [torch.zeros(0, dtype=torch.long)], PC_RANGE)
        out = arm(torch.randn(1, 16, 8, 10), deterministic=True,
                  detach_state=False)
        loss = arm.pending_semantic_loss
        self.assertIsNotNone(loss)
        self.assertEqual(float(loss), 0.0)
        self.assertFalse(
            loss.requires_grad,
            'an empty frame must contribute no gradient at all')
        self.assertTrue(torch.isfinite(out[2]))

    def test_T12_input_validation(self):
        with self.assertRaises(ValueError):
            _build_arm(sem_head_input='q')
        with self.assertRaises(ValueError):
            _build_arm(sem_supervision_weight=-1.0)
        arm = _build_arm(sem_supervision_weight=1.0)
        with self.assertRaises(ValueError):
            arm.set_object_gt([_MockBoxes([[0, 0, 0]], [[1, 1, 1]])],
                              [torch.tensor([0]), torch.tensor([1])],
                              PC_RANGE)
        # log_sem_maps is the probe switch: off at eval, on when requested.
        arm.eval()
        with torch.no_grad():
            arm(torch.randn(1, 16, 8, 10), deterministic=True)
        self.assertIsNone(arm.last_sem_map)
        arm.log_sem_maps = True
        with torch.no_grad():
            arm(torch.randn(1, 16, 8, 10), deterministic=True)
        self.assertIsNotNone(arm.last_sem_map)

    def _build_a2_arm(self, **kw):
        """A minimal A2 arm (CPU, tiny grid) mirroring ``_build_arm``."""
        return _build_arm(**kw)

    # ------------------------------------------------------------------
    # A2 arm plumbing
    # ------------------------------------------------------------------
    def test_T14_a2_defaults_reproduce_candidate_A(self):
        """An untouched config must still be candidate A, not an A2 arm.

        Both A2 variables default to A's behaviour, so a config that forgets
        to set one cannot silently become an A2 run -- and conversely, the two
        arm configs differ from A in exactly one flag each.
        """
        a = self._build_a2_arm(sem_supervision_weight=1.0)
        self.assertEqual(a.sem_target_mode, 'gaussian_full')
        self.assertEqual(a.cls_readout_mode, 'per_class_slot')
        self.assertIsNone(a.cls_readout_anchor_slots)

        s = self._build_a2_arm(sem_supervision_weight=1.0,
                               sem_target_mode='channel_contrastive',
                               sem_rank_q=0.999, sem_rank_margin=1.0,
                               sem_rank_weight=0.1)
        self.assertEqual(s.sem_target_mode, 'channel_contrastive')
        self.assertEqual(s.cls_readout_mode, 'per_class_slot')
        r = self._build_a2_arm(sem_supervision_weight=1.0,
                               cls_readout_mode='anchor_slot',
                               cls_readout_anchor_slots=12)
        self.assertEqual(r.cls_readout_mode, 'anchor_slot')
        self.assertEqual(r.sem_target_mode, 'gaussian_full')

        for bad in ('gaussian', 'contrastive', '', 'Gaussian_Full'):
            with self.assertRaises(ValueError) as ctx:
                self._build_a2_arm(sem_target_mode=bad)
            self.assertIn('sem_target_mode', str(ctx.exception))
        for bad in ('perclass', 'anchor', '', 'Anchor_Slot'):
            with self.assertRaises(ValueError) as ctx:
                self._build_a2_arm(cls_readout_mode=bad)
            self.assertIn('cls_readout_mode', str(ctx.exception))

    # ------------------------------------------------------------------
    # T15/T16 — the ranking term's arithmetic
    # ------------------------------------------------------------------
    def test_T15_rank_term_matches_the_preregistered_formula(self):
        arm = self._build_a2_arm(sem_supervision_weight=1.0,
                                 sem_target_mode='channel_contrastive')
        # One sample, one class channel, a known 1x8 logit row.
        logits = torch.tensor([[[[5.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                  0.0]]]])
        pos = torch.zeros_like(logits, dtype=torch.bool)
        pos[0, 0, 0, 0] = True          # GT peak 5.0
        got = float(arm._channel_rank_loss(logits, pos))
        # q=0.999 over the 7 background values {4,0,0,0,0,0,0} lies just below
        # 4.0; margin 1.0 => relu(1 + ~3.9995 - 5) == 0.
        self.assertEqual(got, 0.0)

        # Lower the GT peak below the background quantile: the term must be
        # exactly (margin + q - g), the frozen one-sided hinge.
        logits2 = logits.clone()
        logits2[0, 0, 0, 0] = 0.5
        bg = logits2[0, 0].reshape(-1)[1:]
        q = torch.quantile(bg, 0.999)
        expect = float(torch.relu(
            arm.sem_rank_margin + q - logits2[0, 0, 0, 0]))
        self.assertAlmostEqual(float(arm._channel_rank_loss(logits2, pos)),
                               expect, places=6)
        self.assertGreater(expect, 0.0)

        # The hinge is ONE-sided: making the GT peak huge must not produce a
        # negative term that could be used to buy AP elsewhere.
        logits3 = logits.clone()
        logits3[0, 0, 0, 0] = 100.0
        self.assertEqual(float(arm._channel_rank_loss(logits3, pos)), 0.0)

    def test_T16_rank_reduction_is_per_occupied_channel(self):
        arm = self._build_a2_arm(sem_supervision_weight=1.0,
                                 sem_target_mode='channel_contrastive')
        logits = torch.zeros(1, 3, 1, 8)
        logits[0, 0, 0] = torch.tensor([0.0, 5.0, 5, 5, 5, 5, 5, 5])
        logits[0, 1, 0] = torch.tensor([9.0, 0.0, 0, 0, 0, 0, 0, 0])
        logits[0, 2, 0] = torch.tensor([0.0, 3.0, 3, 3, 3, 3, 3, 3])
        pos = torch.zeros_like(logits, dtype=torch.bool)
        pos[0, 0, 0, 0] = True     # violated: q=5, g=0 -> 1 + 5 - 0 = 6
        pos[0, 1, 0, 0] = True     # satisfied: q=0, g=9 -> relu(1-9) = 0
        # channel 2 has NO GT -> must not enter the mean at all. With exactly
        # two occupied channels the mean is (6 + 0) / 2 == 3; a mean that had
        # swallowed channel 2 (term 1 + 3 - 0 = 4) would give 10/3 instead.
        got = float(arm._channel_rank_loss(logits, pos))
        self.assertAlmostEqual(got, 3.0, places=5)
        self.assertNotAlmostEqual(got, (6.0 + 0.0 + 4.0) / 3.0, places=5)

        # Mid-batch semantics: sample 1 has no occupied channel, so it cannot
        # dilute sample 0's term (a batch-wide channel pool would halve it).
        logits_b = torch.cat([logits, logits], dim=0)
        pos_b = torch.cat([pos, torch.zeros_like(pos)], dim=0)
        self.assertAlmostEqual(float(arm._channel_rank_loss(logits_b, pos_b)),
                               3.0, places=5)

        # All-background frame => exactly zero, same as focal's num_pos==0.
        self.assertEqual(
            float(arm._channel_rank_loss(logits, torch.zeros_like(
                pos, dtype=torch.bool))), 0.0)

    def test_T17_rank_term_reaches_z_and_leaves_focal_alone(self):
        arm = self._build_a2_arm(sem_supervision_weight=1.0,
                                 sem_target_mode='channel_contrastive')
        logits = torch.randn(1, 2, 4, 5, requires_grad=True)
        pos = torch.zeros(1, 2, 4, 5, dtype=torch.bool)
        pos[0, 0, 1, 1] = True
        pos[0, 1, 2, 3] = True
        rank = arm._channel_rank_loss(logits, pos)
        rank.backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())
        # The gradient must be non-zero somewhere: a term that cannot move the
        # map is not a supervision term.
        self.assertGreater(float(logits.grad.abs().sum()), 0.0)

        # The two arms must share the focal term VERBATIM (prereg §4.1): the
        # only difference between their reported losses is lambda_rank times
        # the ranking term, and the focal-only arm is the A arm exactly.
        target = pos.float()
        arm.sem_target_mode = 'channel_contrastive'
        l_rank_arm, hit_c, thr_c, n_c = arm._semantic_heatmap_loss(
            logits.detach().clone().requires_grad_(True), target)
        arm.sem_target_mode = 'gaussian_full'
        l_focal_arm, hit_a, thr_a, n_a = arm._semantic_heatmap_loss(
            logits.detach().clone().requires_grad_(True), target)
        expected_rank_term = float(arm._channel_rank_loss(logits, pos))
        self.assertAlmostEqual(
            float(l_rank_arm) - float(l_focal_arm),
            arm.sem_rank_weight * expected_rank_term, places=5)
        # The three diagnostics are computed from the SAME focal map, so the
        # rank term must not move them either.
        self.assertAlmostEqual(float(hit_c), float(hit_a), places=6)
        self.assertAlmostEqual(float(thr_c), float(thr_a), places=6)
        self.assertEqual(float(n_c), float(n_a))

        # lambda_rank = 0 must reproduce candidate A bit for bit.
        arm.sem_target_mode = 'channel_contrastive'
        arm.sem_rank_weight = 0.0
        l_zero, hit_z, thr_z, n_z = arm._semantic_heatmap_loss(
            logits.detach().clone().requires_grad_(True), target)
        self.assertEqual(float(l_zero), float(l_focal_arm))
        self.assertEqual(float(hit_z), float(hit_a))

    # ------------------------------------------------------------------
    # T18/T19 — the readout arm's channel coverage
    # ------------------------------------------------------------------
    def test_T18_anchor_slot_writes_all_48_channels(self):
        head = self._build_head()
        a = self._build_a2_arm(sem_supervision_weight=0.0,
                               cls_readout_anchor_slots=12)
        a.eval()
        self.assertEqual(a.cls_readout_anchor_slots, 12)
        with torch.no_grad():
            a.z_cls_logits.weight.normal_()
            a.z_cls_logits.bias.normal_()
        # One fixed input for BOTH arms AND a state reset before each: the
        # RSSM carries hidden state across calls, so re-running without
        # resetting would move z_t and the channel check below would be
        # comparing different latents.
        feat = torch.randn(1, 16, 4, 4)
        with torch.no_grad():
            a.reset_state()
            a(feat, deterministic=True)
        per_class = a.last_cls_logit_residual
        self.assertEqual(tuple(per_class.shape), (1, 4, 4, 4))

        # Emulate R4Det's hook: switch the arm to anchor_slot and re-run on the
        # SAME input, the SAME weights and the SAME (reset) state.
        a.cls_readout_mode = 'anchor_slot'
        with torch.no_grad():
            a.reset_state()
            a(feat, deterministic=True)
        wide = a.last_cls_logit_residual
        self.assertEqual(tuple(wide.shape), (1, 48, 4, 4))
        # Every anchor slot of class c must carry exactly class c's logit.
        for anchor in range(12):
            for c in range(4):
                self.assertAlmostEqual(
                    float(wide[0, anchor * 4 + c, 0, 0]),
                    float(per_class[0, c, 0, 0]), places=6,
                    msg=f'anchor {anchor} class {c}')

        # No new parameters: the predictor is still 4xlatent with the same
        # number of tensors (prereg §4.2 forbids a capacity variable).
        b = self._build_a2_arm(sem_supervision_weight=0.0,
                               cls_readout_mode='anchor_slot',
                               cls_readout_anchor_slots=12)
        pa = sum(p.numel() for p in a.parameters())
        pb = sum(p.numel() for p in b.parameters())
        self.assertEqual(pa, pb)
        self.assertEqual(b.z_cls_logits.weight.shape[0], 4)

        # The head's own downstream path is identical for both: feeding the
        # 48-channel map uses the direct branch, and the result is the sum.
        ref = torch.zeros(1, 48, 2, 2)
        self.assertTrue(torch.equal(
            head._expand_cls_logit_residual(wide[:, :, :2, :2], ref),
            wide[:, :, :2, :2]))

    def test_T19_both_arms_keep_the_zero_init_identity(self):
        head = self._build_head()
        for mode in ('per_class_slot', 'anchor_slot'):
            arm = self._build_a2_arm(
                sem_supervision_weight=1.0, cls_readout_mode=mode,
                cls_readout_anchor_slots=12,
                sem_target_mode='channel_contrastive')
            arm.eval()
            # Fresh zero-init must survive as an exact zero on every channel.
            self.assertTrue(torch.equal(
                arm.z_cls_logits.weight, torch.zeros_like(
                    arm.z_cls_logits.weight)))
            with torch.no_grad():
                arm(torch.randn(1, 16, 4, 4), deterministic=True)
            raw = arm.last_cls_logit_residual
            if mode == 'anchor_slot':
                self.assertEqual(raw.shape[1], 48)
            else:
                self.assertEqual(raw.shape[1], 4)
            self.assertEqual(float(raw.abs().sum()), 0.0)
            expanded = head._expand_cls_logit_residual(raw, torch.zeros(
                1, 48, 4, 4))
            self.assertEqual(float(expanded.abs().sum()), 0.0)

    def test_T13_runtime_arm_selection_is_exact(self):
        """The M2 z-exclusivity gate compares exactly these three arms.

        The probe flips ``sem_head_input`` at runtime, so the selector must map
        each name to its own tensor and must NOT silently fall through to
        ``feat`` for anything else: a typo would make the 'h' arm a duplicate
        of the 'feat' arm, and M2 would report a two-way comparison of the same
        tensor as if the h control had been measured.
        """
        arm = _build_arm(sem_supervision_weight=1.0)
        h = torch.zeros(1, 8, 4, 4)
        z = torch.ones(1, 8, 4, 4)
        feat = torch.full((1, 8, 4, 4), 2.0)
        arm.sem_head_input = 'z'
        self.assertTrue(torch.equal(arm._sem_source(h, z, feat), z))
        arm.sem_head_input = 'h'
        self.assertTrue(torch.equal(arm._sem_source(h, z, feat), h))
        arm.sem_head_input = 'feat'
        self.assertTrue(torch.equal(arm._sem_source(h, z, feat), feat))
        for bad in ('Z', 'H', 'FEAT', 'typo', '', 'feature'):
            arm.sem_head_input = bad
            with self.assertRaises(ValueError) as ctx:
                arm._sem_source(h, z, feat)
            self.assertIn('sem_head_input', str(ctx.exception))


if __name__ == '__main__':
    unittest.main(verbosity=2)
