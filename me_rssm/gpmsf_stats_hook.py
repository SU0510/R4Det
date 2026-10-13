"""Train-time wiring probe for the GP-MSF arm (prereg §5, real-data S0).

Dumps the fusion's ``last_stats`` (radar-support rate, null weights, per-level
weights) to ``gpmsf_wiring_stats.json`` in the work dir every ``interval``
iterations, so the S0 verdict can show that the geometry path really ran with
plausible source usage instead of silently degenerating into concat.

The probe never raises: a broken diagnostic must not kill a training run.  The
*hard* S0 evidence that the new path executed is the checkpoint (the residual
projections must have left their zero initialisation).
"""
import json
import os

from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class GPMSFWiringStatsHook(Hook):
    """Write ``cross_attention.last_stats`` to JSON every ``interval`` iters."""

    def __init__(self, interval=50, out='gpmsf_wiring_stats.json'):
        self.interval = int(interval)
        self.out = out
        self.records = []

    @staticmethod
    def _fusion(runner):
        model = getattr(runner.model, 'module', runner.model)
        return getattr(model, 'cross_attention', None)

    def after_train_iter(self, runner):
        try:
            if self.interval <= 0 or (runner.iter + 1) % self.interval:
                return
            fusion = self._fusion(runner)
            last = getattr(fusion, 'last_stats', None)
            self.records.append({'iter': int(runner.iter) + 1, 'stats': last})
            payload = {
                'collect_stats': bool(getattr(fusion, 'collect_stats', False)),
                'require_geo': bool(getattr(fusion, 'require_geo', False)),
                'new_param_seed': int(getattr(fusion, 'new_param_seed', -1)),
                'records': self.records,
            }
            path = os.path.join(runner.work_dir, self.out)
            with open(path, 'w') as f:
                json.dump(payload, f, indent=2, sort_keys=True)
        except Exception:  # noqa: BLE001 - diagnostics must never break training
            return
