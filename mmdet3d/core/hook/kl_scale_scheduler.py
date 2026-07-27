"""KL scale scheduler hook for RSSM temporal fusion.

Linearly warms up kl_scale from start_value to end_value over
[start_epoch, end_epoch]. Uses runner.epoch directly (not internal
counters), so resume-from-checkpoint is correct by construction.
"""

from mmcv.runner.hooks import HOOKS, Hook
from mmdet3d.core.hook.utils import is_parallel


@HOOKS.register_module()
class KLScaleSchedulerHook(Hook):
    """Linearly increase kl_scale on temporal_fusion over a range of epochs.

    During the warm-up, kl_scale increases linearly from start_value to
    end_value. After end_epoch, kl_scale stays at end_value. Before
    start_epoch, it stays at start_value.

    This allows the detection head and posterior to stabilise before the
    KL term forces the prior to match the posterior.

    Args:
        start_epoch (int): Epoch at which warm-up begins.
        end_epoch (int): Epoch at which warm-up ends (kl_scale reaches end_value).
        start_value (float): kl_scale at start_epoch.
        end_value (float): kl_scale at end_epoch.
    """

    def __init__(self, start_epoch=0, end_epoch=3, start_value=0.0, end_value=0.1):
        super().__init__()
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.start_value = start_value
        self.end_value = end_value

    def before_train_epoch(self, runner):
        kl_scale = self.compute_kl_scale(runner.epoch)

        # Unwrap model: check is_parallel on runner.model (not model.module)
        if is_parallel(runner.model):
            model = runner.model.module
        else:
            model = runner.model

        if hasattr(model, 'temporal_fusion') and model.temporal_fusion is not None:
            model.temporal_fusion.kl_scale = kl_scale

    def compute_kl_scale(self, epoch):
        """Compute kl_scale for a given epoch (public for testing / logging)."""
        if epoch < self.start_epoch:
            return self.start_value
        elif epoch >= self.end_epoch:
            return self.end_value
        else:
            progress = (epoch - self.start_epoch) / (
                self.end_epoch - self.start_epoch
            )
            return self.start_value + progress * (
                self.end_value - self.start_value
            )