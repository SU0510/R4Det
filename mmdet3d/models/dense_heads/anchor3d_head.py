
import numpy as np
import torch
import torch.nn.functional as F
from mmcv.runner import BaseModule, force_fp32
from torch import nn as nn

from mmdet3d.core import (PseudoSampler, box3d_multiclass_nms, limit_period,
                          xywhr2xyxyr)
from mmdet.core import (build_assigner, build_bbox_coder,
                        build_prior_generator, build_sampler, multi_apply)
from ..builder import HEADS, build_loss
from .train_mixins import AnchorTrainMixin


@HEADS.register_module()
class Anchor3DHead(BaseModule, AnchorTrainMixin):
    """Anchor head for SECOND/PointPillars/MVXNet/PartA2.

    Args:
        num_classes (int): Number of classes.
        in_channels (int): Number of channels in the input feature map.
        train_cfg (dict): Train configs.
        test_cfg (dict): Test configs.
        feat_channels (int): Number of channels of the feature map.
        use_direction_classifier (bool): Whether to add a direction classifier.
        anchor_generator(dict): Config dict of anchor generator.
        assigner_per_size (bool): Whether to do assignment for each separate
            anchor size.
        assign_per_class (bool): Whether to do assignment for each class.
        diff_rad_by_sin (bool): Whether to change the difference into sin
            difference for box regression loss.
        dir_offset (float | int): The offset of BEV rotation angles.
            (TODO: may be moved into box coder)
        dir_limit_offset (float | int): The limited range of BEV
            rotation angles. (TODO: may be moved into box coder)
        bbox_coder (dict): Config dict of box coders.
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        loss_dir (dict): Config of direction classifier loss.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 train_cfg,
                 test_cfg,
                 feat_channels=256,
                 use_direction_classifier=True,
                 anchor_generator=dict(
                     type='Anchor3DRangeGenerator',
                     range=[0, -39.68, -1.78, 69.12, 39.68, -1.78],
                     strides=[2],
                     sizes=[[3.9, 1.6, 1.56]],
                     rotations=[0, 1.57],
                     custom_values=[],
                     reshape_out=False),
                 assigner_per_size=False,
                 assign_per_class=False,
                 diff_rad_by_sin=True,
                 dir_offset=0, #-np.pi / 2,
                 dir_limit_offset=1,#原本是0
                 bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
                 loss_dir=dict(type='CrossEntropyLoss', loss_weight=0.2),
                 init_cfg=None,
                 truck_refine=False,
                 truck_refine_channels=64,
                 truck_refine_dims=(0, 1, 4),
                 truck_anchor_class=None,
                 truck_refine_detach=False,
                 truck_tower=False,
                 truck_tower_channels=64,
                 truck_tower_dims=(0, 1, 4),
                 ped_refine=False,
                 ped_refine_channels=64,
                 ped_refine_dims=(0, 1, 2, 3, 4, 5, 6),
                 ped_refine_detach=False,
                 **kwargs):
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.diff_rad_by_sin = diff_rad_by_sin
        self.use_direction_classifier = use_direction_classifier
        self.ignore_dir_classes = kwargs.get('ignore_dir_classes', [])
        self.anchor_class_mapping = kwargs.get('anchor_class_mapping', None)
        self.use_iou_branch = kwargs.get('use_iou_branch', False)
        self.confusion_pairs = kwargs.get('confusion_pairs', None)
        self.confusion_loss_weight = kwargs.get('confusion_loss_weight', 0.0)
        # Truck-specific residual refinement (dx, dy, dl only), to prioritize
        # Truck center + length regression without disturbing other classes.
        self.truck_refine = truck_refine
        self.truck_refine_channels = truck_refine_channels
        self.truck_refine_dims = tuple(truck_refine_dims)
        self.truck_anchor_class = truck_anchor_class
        self.truck_refine_detach = truck_refine_detach
        self.truck_tower = truck_tower
        self.truck_tower_channels = truck_tower_channels
        self.truck_tower_dims = tuple(truck_tower_dims)
        self.ped_refine = ped_refine
        self.ped_refine_channels = ped_refine_channels
        self.ped_refine_dims = tuple(ped_refine_dims)
        self.ped_refine_detach = ped_refine_detach
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.assigner_per_size = assigner_per_size
        self.assign_per_class = assign_per_class
        self.dir_offset = dir_offset
        self.dir_limit_offset = dir_limit_offset
        import warnings
        warnings.warn(
            'dir_offset and dir_limit_offset will be depressed and be '
            'incorporated into box coder in the future')
        self.fp16_enabled = False
        # build anchor generator
        self.anchor_generator = build_prior_generator(anchor_generator)
        # In 3D detection, the anchor stride is connected with anchor size
        self.num_anchors = self.anchor_generator.num_base_anchors
        # build box coder
        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.box_code_size = self.bbox_coder.code_size

        # build loss function
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        self.sampling = loss_cls['type'] not in ['FocalLoss', 'GHMC']
        if not self.use_sigmoid_cls:
            self.num_classes += 1
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_dir = build_loss(loss_dir)
        self.fp16_enabled = False

        self._init_layers()
        self._init_assigner_sampler()

        if init_cfg is None:
            self.init_cfg = dict(
                type='Normal',
                layer='Conv2d',
                std=0.01,
                override=dict(
                    type='Normal', name='conv_cls', std=0.01, bias_prob=0.01))

    def _init_assigner_sampler(self):
        """Initialize the target assigner and sampler of the head."""
        if self.train_cfg is None:
            return

        if self.sampling:
            self.bbox_sampler = build_sampler(self.train_cfg.sampler)
        else:
            self.bbox_sampler = PseudoSampler()
        if isinstance(self.train_cfg.assigner, dict):
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
        elif isinstance(self.train_cfg.assigner, list):
            self.bbox_assigner = [
                build_assigner(res) for res in self.train_cfg.assigner
            ]

    def _anchor_class_channel_inds(self, dims, class_token):
        """Resolve conv_reg channel indices for one class's anchor slots.

        Args:
            dims (tuple[int]): box-code dims to select (e.g. (0, 1, 4)).
            class_token (int): class index whose anchors to touch. Anchor
                slots are [size, rot] (size-major, rot-inner); each class
                owns a contiguous run of sizes given by the size index where
                that class first appears in ``anchor_class_mapping`` through
                the index where it last appears.

        Returns:
            list[int]: flat channel indices in conv_reg's
                [anchor * code_size + dim] layout.
        """
        cls_tokens = self.anchor_class_mapping
        class_token = int(class_token)
        start = cls_tokens.index(class_token)
        end = len(cls_tokens) - 1 - cls_tokens[::-1].index(class_token)
        num_rot = len(self.anchor_generator.rotations)
        anchor_flat_start = start * num_rot
        anchor_flat_end = (end + 1) * num_rot
        anchor_flat = list(range(anchor_flat_start, anchor_flat_end))

        inds = []
        for a in anchor_flat:
            for d in dims:
                inds.append(a * self.box_code_size + d)
        return inds

    def _truck_channel_inds(self, dims):
        """Resolve conv_reg channel indices for the Truck anchors."""
        num_rot = len(self.anchor_generator.rotations)
        if self.truck_anchor_class is not None:
            truck_size_idx = self.truck_anchor_class
        elif self.anchor_class_mapping is not None:
            # last distinct class token = Truck; its anchors are the tail
            truck_size_idx = len(self.anchor_class_mapping) - 1
        else:
            # no mapping: assume Truck is the last anchor size slot
            truck_size_idx = self.anchor_generator.num_base_anchors // num_rot - 1

        # flat anchor indices in [size, rot] (size-major, rot-inner) order
        self.num_truck_rot = num_rot
        anchor_flat_start = truck_size_idx * num_rot
        anchor_flat = list(range(anchor_flat_start, anchor_flat_start + num_rot))

        inds = []
        for a in anchor_flat:
            for d in dims:
                inds.append(a * self.box_code_size + d)
        return inds

    def _init_layers(self):
        """Initialize neural network layers of the head."""
        self.cls_out_channels = self.num_anchors * self.num_classes
        self.conv_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 1)
        self.conv_reg = nn.Conv2d(self.feat_channels,
                                  self.num_anchors * self.box_code_size, 1)
        if self.use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(self.feat_channels,
                                          self.num_anchors * 2, 1)
        if self.use_iou_branch:
            self.conv_iou = nn.Conv2d(self.feat_channels, self.num_anchors, 1)

        # --- Truck-specific regression branches ---
        # Both branches only touch Truck anchor channels in the bbox output.
        # Channel indices map into conv_reg's [anchor * code_size + dim]
        # layout. The final conv of each branch is zero-initialized so that at
        # training start the branch output is zero.
        self._truck_refine_inds = None
        if self.truck_refine:
            # Truck residual refinement (dx, dy, dl): base conv_reg still
            # predicts everything; this branch only adds Truck corrections.
            self._truck_refine_inds = self._truck_channel_inds(
                self.truck_refine_dims)
            n_out = self.num_truck_rot * len(self.truck_refine_dims)

            refine_layers = [
                nn.Conv2d(self.feat_channels, self.truck_refine_channels, 3,
                          padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.truck_refine_channels, n_out, 1),
            ]
            self.truck_refine_conv = nn.Sequential(*refine_layers)
            # zero-init the final 1x1 so refinement starts at zero
            nn.init.constant_(self.truck_refine_conv[-1].weight, 0.0)
            nn.init.constant_(self.truck_refine_conv[-1].bias, 0.0)

        self._truck_tower_inds = None
        if self.truck_tower:
            # Truck independent regression tower: directly predicts (dx, dy, dl)
            # for Truck anchors and REPLACES the shared conv_reg output for
            # those channels. No residual addition — the shared conv_reg no
            # longer receives gradient on these Truck channels.
            self._truck_tower_inds = self._truck_channel_inds(
                self.truck_tower_dims)
            n_out = self.num_truck_rot * len(self.truck_tower_dims)

            tower_layers = [
                nn.Conv2d(self.feat_channels, self.truck_tower_channels, 3,
                          padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.truck_tower_channels, n_out, 1),
            ]
            self.truck_tower_conv = nn.Sequential(*tower_layers)
            # zero-init the final 1x1 so the tower starts outputting zero
            nn.init.constant_(self.truck_tower_conv[-1].weight, 0.0)
            nn.init.constant_(self.truck_tower_conv[-1].bias, 0.0)

        self._ped_refine_inds = None
        if self.ped_refine:
            # Pedestrian residual refinement over the FULL 7D box code
            # (dx, dy, dz, dw, dl, dh, dθ). Adds to shared conv_reg output for
            # the 3 Ped anchor slots only. Input detached by default in v1 to
            # verify head capacity in isolation (no shared-BEV feature drain).
            self._ped_refine_inds = self._anchor_class_channel_inds(
                self.ped_refine_dims, 0)  # class token 0 = Pedestrian
            n_out = len(self._ped_refine_inds)

            layers = [
                nn.Conv2d(self.feat_channels, self.ped_refine_channels, 3,
                          padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.ped_refine_channels, n_out, 1),
            ]
            self.ped_refine_conv = nn.Sequential(*layers)
            # zero-init the final 1x1 so refinement starts at zero
            nn.init.constant_(self.ped_refine_conv[-1].weight, 0.0)
            nn.init.constant_(self.ped_refine_conv[-1].bias, 0.0)

    def init_weights(self):
        super().init_weights()
        # init_cfg applies Normal over every Conv2d in super().init_weights(),
        # which would overwrite the zero-initialization set in _init_layers for
        # the Truck branches. Re-apply zero-init to the final conv of each
        # branch so their outputs are exactly zero at training start. Baseline
        # (all Truck branches disabled) is unaffected.
        if self.truck_refine:
            nn.init.constant_(self.truck_refine_conv[-1].weight, 0.0)
            nn.init.constant_(self.truck_refine_conv[-1].bias, 0.0)
        if self.truck_tower:
            nn.init.constant_(self.truck_tower_conv[-1].weight, 0.0)
            nn.init.constant_(self.truck_tower_conv[-1].bias, 0.0)
        if self.ped_refine:
            nn.init.constant_(self.ped_refine_conv[-1].weight, 0.0)
            nn.init.constant_(self.ped_refine_conv[-1].bias, 0.0)

    def forward_single(self, x):
        """Forward function on a single-scale feature map.

        Args:
            x (torch.Tensor): Input features.

        Returns:
            tuple[torch.Tensor]: Contain score of each class, bbox
                regression and direction classification predictions.
        """

        cls_score = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)

        if self.truck_tower:
            tower = self.truck_tower_conv(x)  # [B, n_out, H, W]
            inds = torch.tensor(self._truck_tower_inds, device=x.device,
                                dtype=torch.long)
            bbox_pred = bbox_pred.clone()
            bbox_pred.index_copy_(1, inds, tower)

        if self.truck_refine:
            refine_input = x.detach() if self.truck_refine_detach else x
            refine = self.truck_refine_conv(refine_input)  # [B, n_out, H, W]
            inds = torch.tensor(self._truck_refine_inds, device=x.device,
                                dtype=torch.long)
            bbox_pred = bbox_pred.clone()
            bbox_pred.index_add_(1, inds, refine)

        if self.ped_refine:
            refine_input = x.detach() if self.ped_refine_detach else x
            refine = self.ped_refine_conv(refine_input)  # [B, n_out, H, W]
            inds = torch.tensor(self._ped_refine_inds, device=x.device,
                                dtype=torch.long)
            bbox_pred = bbox_pred.clone()
            bbox_pred.index_add_(1, inds, refine)

        dir_cls_preds = None
        if self.use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(x)

        return cls_score, bbox_pred, dir_cls_preds

    def forward(self, feats):
        """Forward pass.

        Args:
            feats (list[torch.Tensor]): Multi-level features, e.g.,
                features produced by FPN.

        Returns:
            tuple[list[torch.Tensor]]: Multi-level class score, bbox
                and direction predictions.
        """

        if feats is not None and len(feats) > 0:
            first_feat = feats[0]

        if self.use_iou_branch:
            self._iou_preds = [self.conv_iou(f).sigmoid() for f in feats]

        return multi_apply(self.forward_single, feats)

    def get_anchors(self, featmap_sizes, input_metas, device='cuda'):
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            input_metas (list[dict]): contain pcd and img's meta info.
            device (str): device of current module.

        Returns:
            list[list[torch.Tensor]]: Anchors of each image, valid flags
                of each image.
        """
        num_imgs = len(input_metas)
        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, device=device)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]
        return anchor_list

    def loss_single(self, cls_score, bbox_pred, dir_cls_preds, labels,
                    label_weights, bbox_targets, bbox_weights, dir_targets,
                    dir_weights, iou_pred, anchors, num_total_samples):
        """Calculate loss of Single-level results.

        Args:
            cls_score (torch.Tensor): Class score in single-level.
            bbox_pred (torch.Tensor): Bbox prediction in single-level.
            dir_cls_preds (torch.Tensor): Predictions of direction class
                in single-level.
            labels (torch.Tensor): Labels of class.
            label_weights (torch.Tensor): Weights of class loss.
            bbox_targets (torch.Tensor): Targets of bbox predictions.
            bbox_weights (torch.Tensor): Weights of bbox loss.
            dir_targets (torch.Tensor): Targets of direction predictions.
            dir_weights (torch.Tensor): Weights of direction loss.
            num_total_samples (int): The number of valid samples.

        Returns:
            tuple[torch.Tensor]: Losses of class, bbox
                and direction, respectively.
        """
        # classification loss
        if num_total_samples is None:
            num_total_samples = int(cls_score.shape[0])
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
        assert labels.max().item() <= self.num_classes
        loss_cls = self.loss_cls(
            cls_score, labels, label_weights, avg_factor=num_total_samples)

        # Car<->Truck (or arbitrary class-pair) confusion suppression:
        # on positive samples of class A, explicitly push class B's logit down
        # (and vice versa). This targets hard false positives directly, instead
        # of waiting for the plain per-class focal term to fix them.
        if self.confusion_pairs is not None and self.confusion_loss_weight > 0:
            loss_conf = cls_score.new_zeros(())
            total_pairs = 0
            for cls_a, cls_b in self.confusion_pairs:
                mask_a = labels == cls_a
                mask_b = labels == cls_b
                n_a = int(mask_a.sum())
                n_b = int(mask_b.sum())
                if n_a > 0:
                    loss_conf = loss_conf + F.binary_cross_entropy_with_logits(
                        cls_score[mask_a, cls_b],
                        cls_score.new_zeros(n_a),
                        reduction='sum')
                    total_pairs += n_a
                if n_b > 0:
                    loss_conf = loss_conf + F.binary_cross_entropy_with_logits(
                        cls_score[mask_b, cls_a],
                        cls_score.new_zeros(n_b),
                        reduction='sum')
                    total_pairs += n_b
            if total_pairs > 0:
                loss_conf = loss_conf / total_pairs
            loss_cls = loss_cls + self.confusion_loss_weight * loss_conf

        # regression loss
        bbox_pred = bbox_pred.permute(0, 2, 3,
                                      1).reshape(-1, self.box_code_size)
        bbox_targets = bbox_targets.reshape(-1, self.box_code_size)
        bbox_weights = bbox_weights.reshape(-1, self.box_code_size)

        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero(
                        as_tuple=False).reshape(-1)
        num_pos = len(pos_inds)

        pos_bbox_pred = bbox_pred[pos_inds]
        pos_bbox_targets = bbox_targets[pos_inds]
        pos_bbox_weights = bbox_weights[pos_inds]

        # dir loss
        if self.use_direction_classifier:
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).reshape(-1, 2)
            dir_targets = dir_targets.reshape(-1)
            dir_weights = dir_weights.reshape(-1)
            pos_dir_cls_preds = dir_cls_preds[pos_inds]
            pos_dir_targets = dir_targets[pos_inds]
            pos_dir_weights = dir_weights[pos_inds]

        if num_pos > 0:
            code_weight = self.train_cfg.get('code_weight', None)
            if code_weight:
                pos_bbox_weights = pos_bbox_weights * bbox_weights.new_tensor(
                    code_weight)
            if self.diff_rad_by_sin:
                pos_bbox_pred, pos_bbox_targets = self.add_sin_difference(
                    pos_bbox_pred, pos_bbox_targets)
            loss_bbox = self.loss_bbox(
                pos_bbox_pred,
                pos_bbox_targets,
                pos_bbox_weights,
                avg_factor=num_total_samples)

            # direction classification loss
            loss_dir = None
            if self.use_direction_classifier:
                # Apply ignore_dir_classes: zero out dir weights for excluded classes
                if self.ignore_dir_classes:
                    pos_labels = labels[pos_inds]
                    pos_dir_weights = pos_dir_weights.clone()
                    for ignore_cls in self.ignore_dir_classes:
                        pos_dir_weights[pos_labels == ignore_cls] = 0.0
                loss_dir = self.loss_dir(
                    pos_dir_cls_preds,
                    pos_dir_targets,
                    pos_dir_weights,
                    avg_factor=num_total_samples)
        else:
            loss_bbox = pos_bbox_pred.sum()
            if self.use_direction_classifier:
                loss_dir = pos_dir_cls_preds.sum()

        # IoU-aware quality branch loss
        loss_iou = None
        if iou_pred is not None and num_pos > 0:
            with torch.no_grad():
                # anchors is per-image; pos_inds are batch-level. All images share same anchor layout, so use modulo.
                num_anchors_per_img = anchors.size(0)
                pos_anchors = anchors[pos_inds % num_anchors_per_img]
                # Decode predicted and target boxes
                pos_pred_boxes = self.bbox_coder.decode(pos_anchors, pos_bbox_pred)
                pos_gt_boxes = self.bbox_coder.decode(pos_anchors, pos_bbox_targets)
                # Compute centerness proxy (BEV distance normalized by box diagonal)
                pred_xy = pos_pred_boxes[:, :2]
                gt_xy = pos_gt_boxes[:, :2]
                gt_dims = pos_gt_boxes[:, 3:5]  # w, l
                diag = torch.sqrt(gt_dims[:, 0]**2 + gt_dims[:, 1]**2 + 1e-6)
                dist = torch.sqrt(((pred_xy - gt_xy)**2).sum(dim=1))
                iou_target = torch.exp(-dist / diag).clamp(0, 1)
            iou_pred_flat = iou_pred.permute(0, 2, 3, 1).reshape(-1)
            pos_iou_pred = iou_pred_flat[pos_inds]
            loss_iou = F.l1_loss(pos_iou_pred, iou_target, reduction='none')
            loss_iou = loss_iou.mean()

        return loss_cls, loss_bbox, loss_dir, loss_iou

    @staticmethod
    def add_sin_difference(boxes1, boxes2):
        """Convert the rotation difference to difference in sine function.

        Args:
            boxes1 (torch.Tensor): Original Boxes in shape (NxC), where C>=7
                and the 7th dimension is rotation dimension.
            boxes2 (torch.Tensor): Target boxes in shape (NxC), where C>=7 and
                the 7th dimension is rotation dimension.

        Returns:
            tuple[torch.Tensor]: ``boxes1`` and ``boxes2`` whose 7th
                dimensions are changed.
        """
        rad_pred_encoding = torch.sin(boxes1[..., 6:7]) * torch.cos(
            boxes2[..., 6:7])
        rad_tg_encoding = torch.cos(boxes1[..., 6:7]) * torch.sin(boxes2[...,
                                                                         6:7])
        boxes1 = torch.cat(
            [boxes1[..., :6], rad_pred_encoding, boxes1[..., 7:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :6], rad_tg_encoding, boxes2[..., 7:]],
                           dim=-1)
        return boxes1, boxes2

    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'dir_cls_preds'))
    def loss(self,
             cls_scores,
             bbox_preds,
             dir_cls_preds,
             gt_bboxes,
             gt_labels,
             input_metas,
             bev_semantic_mask=None,
             gt_bboxes_ignore=None):
        """Calculate losses.

        Args:
            cls_scores (list[torch.Tensor]): Multi-level class scores.
            bbox_preds (list[torch.Tensor]): Multi-level bbox predictions.
            dir_cls_preds (list[torch.Tensor]): Multi-level direction
                class predictions.
            gt_bboxes (list[:obj:`BaseInstance3DBoxes`]): Gt bboxes
                of each sample.
            gt_labels (list[torch.Tensor]): Gt labels of each sample.
            input_metas (list[dict]): Contain pcd and img's meta info.
            gt_bboxes_ignore (list[torch.Tensor]): Specify
                which bounding boxes to ignore.

        Returns:
            dict[str, list[torch.Tensor]]: Classification, bbox, and
                direction losses of each level.

                - loss_cls (list[torch.Tensor]): Classification losses.
                - loss_bbox (list[torch.Tensor]): Box regression losses.
                - loss_dir (list[torch.Tensor]): Direction classification
                    losses.
        """



        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels
        device = cls_scores[0].device
        anchor_list = self.get_anchors(
            featmap_sizes, input_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        gt_bboxes = [b.to(device) for b in gt_bboxes]



        cls_reg_targets = self.anchor_target_3d(
            anchor_list,
            gt_bboxes,
            input_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            num_classes=self.num_classes,
            label_channels=label_channels,
            sampling=self.sampling,
            bev_semantic_mask_list=bev_semantic_mask)



        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         dir_targets_list, dir_weights_list, num_total_pos,
         num_total_neg) = cls_reg_targets

        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        # Get IoU predictions from side channel (set during forward)
        iou_preds = getattr(self, '_iou_preds', None)
        if iou_preds is None:
            iou_preds = [None] * len(cls_scores)
        # Prepare per-level anchors for IoU decoding
        anchor_levels = [a.reshape(-1, self.box_code_size) for a in anchor_list[0]] if anchor_list else [None] * len(cls_scores)

        losses_cls, losses_bbox, losses_dir, losses_iou = multi_apply(
            self.loss_single,
            cls_scores,
            bbox_preds,
            dir_cls_preds,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            dir_targets_list,
            dir_weights_list,
            iou_preds,
            anchor_levels,
            num_total_samples=num_total_samples)
        loss_dict = dict(
            loss_cls=losses_cls, loss_bbox=losses_bbox, loss_dir=losses_dir)
        if any(l is not None for l in losses_iou):
            loss_dict['loss_iou'] = losses_iou
        return loss_dict

    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   dir_cls_preds,
                   input_metas,
                   cfg=None,
                   rescale=False):
        """Get bboxes of anchor head.

        Args:
            cls_scores (list[torch.Tensor]): Multi-level class scores.
            bbox_preds (list[torch.Tensor]): Multi-level bbox predictions.
            dir_cls_preds (list[torch.Tensor]): Multi-level direction
                class predictions.
            input_metas (list[dict]): Contain pcd and img's meta info.
            cfg (:obj:`ConfigDict`): Training or testing config.
            rescale (list[torch.Tensor]): Whether th rescale bbox.

        Returns:
            list[tuple]: Prediction resultes of batches.
        """
        assert len(cls_scores) == len(bbox_preds)
        assert len(cls_scores) == len(dir_cls_preds)
        num_levels = len(cls_scores)
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        device = cls_scores[0].device
        mlvl_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, device=device)
        mlvl_anchors = [
            anchor.reshape(-1, self.box_code_size) for anchor in mlvl_anchors
        ]
        result_list = []
        for img_id in range(len(input_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            dir_cls_pred_list = [
                dir_cls_preds[i][img_id].detach() for i in range(num_levels)
            ]
            input_meta = input_metas[img_id]
            proposals = self.get_bboxes_single(cls_score_list, bbox_pred_list,
                                               dir_cls_pred_list, mlvl_anchors,
                                               input_meta, cfg, rescale)
            result_list.append(proposals)
        return result_list

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          dir_cls_preds,
                          mlvl_anchors,
                          input_meta,
                          cfg=None,
                          rescale=False):
        """Get bboxes of single branch.

        Args:
            cls_scores (torch.Tensor): Class score in single batch.
            bbox_preds (torch.Tensor): Bbox prediction in single batch.
            dir_cls_preds (torch.Tensor): Predictions of direction class
                in single batch.
            mlvl_anchors (List[torch.Tensor]): Multi-level anchors
                in single batch.
            input_meta (list[dict]): Contain pcd and img's meta info.
            cfg (:obj:`ConfigDict`): Training or testing config.
            rescale (list[torch.Tensor]): whether th rescale bbox.

        Returns:
            tuple: Contain predictions of single batch.

                - bboxes (:obj:`BaseInstance3DBoxes`): Predicted 3d bboxes.
                - scores (torch.Tensor): Class score of each bbox.
                - labels (torch.Tensor): Label of each bbox.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_dir_scores = []
        for cls_score, bbox_pred, dir_cls_pred, anchors in zip(
                cls_scores, bbox_preds, dir_cls_preds, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            assert cls_score.size()[-2:] == dir_cls_pred.size()[-2:]
            dir_cls_pred = dir_cls_pred.permute(1, 2, 0).reshape(-1, 2)
            dir_cls_score = torch.max(dir_cls_pred, dim=-1)[1]

            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.num_classes)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)

            bbox_pred = bbox_pred.permute(1, 2,
                                          0).reshape(-1, self.box_code_size)



            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    max_scores, _ = scores[:, :-1].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                dir_cls_score = dir_cls_score[topk_inds]

            bboxes = self.bbox_coder.decode(anchors, bbox_pred)

            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_dir_scores.append(dir_cls_score)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_bboxes_for_nms = xywhr2xyxyr(input_meta['box_type_3d'](
            mlvl_bboxes, box_dim=self.box_code_size).bev)
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_dir_scores = torch.cat(mlvl_dir_scores)
        if self.use_sigmoid_cls:
            # Add a dummy background class to the front when using sigmoid
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)

        score_thr = cfg.get('score_thr', 0)
        results = box3d_multiclass_nms(mlvl_bboxes, mlvl_bboxes_for_nms,
                                       mlvl_scores, score_thr, cfg.max_num,
                                       cfg, mlvl_dir_scores)

        bboxes, scores, labels, dir_scores = results

        if bboxes.shape[0] > 0:
            yaw_minus_offset = bboxes[..., 6] - self.dir_offset
            dir_rot = limit_period(bboxes[..., 6] - self.dir_offset,
                                   self.dir_limit_offset, np.pi)

            bboxes[..., 6] = (
                    dir_rot + self.dir_offset +
                    np.pi * dir_scores.to(bboxes.dtype))
        bboxes = input_meta['box_type_3d'](bboxes, box_dim=self.box_code_size)
        return bboxes, scores, labels
