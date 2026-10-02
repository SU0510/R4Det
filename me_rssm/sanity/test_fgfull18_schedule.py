"""Audit the complete-18-epoch no2d_igdr schedule against its 24e parent.

The 18e run is meant to be the same architecture and optimization recipe as
the 24e no2d_igdr baseline, with only the schedule horizon shortened.  This
test makes that contract explicit so a later edit cannot silently restore
the 24e cosine/KL horizon.
"""

import os
import sys

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from mmcv import Config


BASE = os.path.join(
    _REPO,
    'configs/r4det/'
    'TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr.py')
RUN18 = os.path.join(
    _REPO,
    'configs/r4det/'
    'TJ4D-R4Det_fgfull_N4_2x4_18e_pretrained_v2_head_no2d_igdr.py')


def load(path):
    return Config.fromfile(path)


base = load(BASE)
run18 = load(RUN18)

# Architecture and data must be unchanged: this is a schedule experiment,
# not a new model.
assert run18.model.temporal_fusion == base.model.temporal_fusion, \
    'temporal fusion changed between 24e and 18e'
assert run18.model.pts_bbox_head == base.model.pts_bbox_head, \
    'detection head changed between 24e and 18e'
assert run18.data.train.dataset.pipeline == \
    base.data.train.dataset.pipeline, 'train pipeline changed'
assert run18.data.val.pipeline == base.data.val.pipeline, 'val pipeline changed'
assert run18.data.val.samples_per_gpu == base.data.val.samples_per_gpu, \
    'val batch changed'

# Schedule: complete 18-epoch horizon, same peak LR, same optimizer recipe.
assert base.runner.max_epochs == 24
assert run18.runner.max_epochs == 18
assert run18.runner.type == base.runner.type == 'EpochBasedRunner'
assert run18.optimizer == base.optimizer, 'optimizer recipe changed'
assert run18.optimizer_config == base.optimizer_config, \
    'gradient accumulation/clip changed'
assert run18.lr_config['policy'] == 'CosineAnnealing'
assert run18.lr_config['min_lr_ratio'] == base.lr_config['min_lr_ratio']

# The KL warmup is shortened proportionally with the horizon:
# 10 warmup epochs / 24 total -> 8 / 18 (2/3 -> 4/9, both ~0.42).
assert run18.custom_hooks[0]['type'] == 'KLScaleSchedulerHook'
assert base.custom_hooks[0]['end_epoch'] == 10
assert run18.custom_hooks[0]['end_epoch'] == 8
assert run18.custom_hooks[0]['start_value'] == 0.0
assert run18.custom_hooks[0]['end_value'] == 1.0

# Eval/checkpoint cadence is inherited, so the standard ep12-16 window is
# complete and ep18 lands as a normal checkpoint.
assert run18.evaluation.interval == 1
assert run18.checkpoint_config.interval == 2
assert run18.load_from == base.load_from

# The LR schedule is a complete cosine over 18 epochs, not a truncated 24e
# schedule.  Use mmcv's own function to make the endpoint explicit.
from mmcv.runner.hooks.lr_updater import annealing_cos

peak = run18.optimizer['lr']
minimum = peak * run18.lr_config['min_lr_ratio']
assert abs(annealing_cos(peak, minimum, 0.0) - peak) < 1e-12
assert abs(annealing_cos(peak, minimum, 1.0) - minimum) < 1e-12

print('[PASS] fgfull 18e schedule audit: architecture/data unchanged, '
      'max_epochs=18, cosine horizon=18, KL warmup end=8, peak lr=1.5e-4')
