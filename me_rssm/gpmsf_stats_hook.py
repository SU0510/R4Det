"""Required S0 diagnostics: rank-zero atomic writes, rank-wide sync proof."""
import json
import os

import torch.distributed as dist
from mmcv.runner import HOOKS, Hook, get_dist_info


def atomic_json(path, payload):
    temporary = path + '.tmp'
    with open(temporary, 'w') as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
    os.replace(temporary, path)


@HOOKS.register_module()
class GPMSFWiringStatsHook(Hook):
    def __init__(self, interval=50, out='gpmsf_wiring_stats.json'):
        if int(interval) <= 0:
            raise ValueError('S0 stats interval must be positive')
        self.interval, self.out = int(interval), out
        self.records = []

    @staticmethod
    def _fusion(runner):
        model = getattr(runner.model, 'module', runner.model)
        return model.cross_attention

    def before_train_iter(self, runner):
        # Sample current-frame stats; do not disable checkpointing. Avoid scalar
        # GPU synchronisations on unsampled steps in the throughput comparison.
        self._fusion(runner).collect_stats = (runner.iter + 1) % self.interval == 0

    def _write(self, runner, **extra):
        if get_dist_info()[0] != 0:
            return
        fusion = self._fusion(runner)
        payload = dict(collect_stats=True, require_geo=fusion.require_geo,
                       new_param_seed=fusion.new_param_seed,
                       records=self.records, **extra)
        atomic_json(os.path.join(runner.work_dir, self.out), payload)

    def after_train_iter(self, runner):
        if (runner.iter + 1) % self.interval:
            return
        fusion = self._fusion(runner)
        if fusion.last_stats is None:
            raise RuntimeError('GP-MSF S0 geometry stats missing')
        self.records.append(dict(iter=int(runner.iter) + 1, stats=fusion.last_stats))
        self._write(runner)

    def after_run(self, runner):
        from me_rssm.sanity.paired_z_prediction import fingerprint
        fusion = self._fusion(runner)
        digest = fingerprint(dict(fusion.named_parameters()))
        world = get_dist_info()[1]
        digests = [None] * world
        if dist.is_initialized():
            dist.all_gather_object(digests, digest)
        else:
            digests[0] = digest
        if len(set(digests)) != 1:
            raise RuntimeError('GP-MSF parameters differ across DDP ranks')
        self._write(runner, ddp_world_size=world, ddp_parameter_sha=digests,
                    completed_iters=int(runner.iter))
