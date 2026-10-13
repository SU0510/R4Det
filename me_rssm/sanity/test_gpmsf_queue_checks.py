"""Queue evidence regressions; never train or allocate a CUDA tensor."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from mmcv import Config

from me_rssm.sanity import gpmsf_run_checks as checks
from me_rssm.sanity import gpmsf_s0_verdict as gate
from me_rssm import gpmsf_stats_hook as stats_hook


def write_rows(directory, rows, name='20261013.log.json'):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / name).write_text(''.join(json.dumps(r) + '\n' for r in rows))


def val(epoch, value):
    return dict(mode='val', epoch=epoch, **{checks.PRIMARY: value},
                **{key: value for key in checks.CLASSES.values()})


@pytest.mark.parametrize('value', [float('inf'), float('-inf'), float('nan')])
def test_nonfinite_early_or_disappearing_keys_fail(tmp_path, value):
    write_rows(tmp_path, [dict(mode='train', iter=1, loss_old=value),
                          dict(mode='train', iter=450, loss_new=1.)])
    with pytest.raises(ValueError, match='non-finite'):
        checks.read_rows(tmp_path)


def test_live_only_tolerates_unfinished_last_line(tmp_path):
    path = tmp_path / 'run.log.json'
    path.write_text('{"mode":"train"}\n{"iter":')
    assert len(checks.read_rows(tmp_path, live=True)) == 1
    with pytest.raises(ValueError, match='malformed'):
        checks.read_rows(tmp_path)
    path.write_text('{bad}\n{"iter":')
    with pytest.raises(ValueError, match='malformed'):
        checks.read_rows(tmp_path, live=True)


def test_incomplete_window_is_never_a_pass(tmp_path):
    write_rows(tmp_path, [val(12, 39.), val(14, 40.)])
    with pytest.raises(ValueError, match='incomplete frozen window'):
        checks.curve_report(tmp_path, (12, 16))


def test_full_peak_does_not_depend_on_saved_weights(tmp_path):
    write_rows(tmp_path, [val(e, 50. if e == 13 else 39.) for e in range(1, 25)])
    (tmp_path / 'epoch_14.pth').touch()
    report = checks.curve_report(tmp_path, (12, 16), expected_epochs=24)
    assert report['window']['n'] == 5
    assert report['window']['mean'][checks.PRIMARY] == 41.2
    assert report['all_epoch_peak'][checks.PRIMARY] == dict(value=50., epoch=13)
    assert report['best_saved'][checks.PRIMARY] == dict(value=39., epoch=14)
    assert len(report['full_epoch_curve']) == 24


def test_b0_reads_union_of_resumed_logs_not_only_newest(tmp_path):
    write_rows(tmp_path, [val(e, e) for e in range(1, 7)], 'a.log.json')
    write_rows(tmp_path, [val(e, e) for e in range(7, 17)], 'b.log.json')
    assert len(checks.curve_report(tmp_path, (12, 16), expected_epochs=16)['full_epoch_curve']) == 16


def test_phase_requires_actual_success_and_same_source(tmp_path, monkeypatch):
    monkeypatch.setattr(checks, 'source_hashes', lambda: {'fusion.py': 'a'})
    checks.start_phase(tmp_path, 'arm.py')
    with pytest.raises(ValueError, match='successfully'):
        checks.completed_phase(tmp_path)
    checks.finish_phase(tmp_path)
    assert checks.completed_phase(tmp_path)['exit_code'] == 0
    monkeypatch.setattr(checks, 'source_hashes', lambda: {'fusion.py': 'b'})
    with pytest.raises(ValueError, match='source'):
        checks.completed_phase(tmp_path)
    with pytest.raises(ValueError, match='stale'):
        checks.start_phase(tmp_path, 'arm.py')


@pytest.mark.parametrize('state', [
    {'cross_attention.proj_r2c.bias': torch.ones(3)},
    {'cross_attention.proj_r2c.weight': torch.ones(3)},
    {f'cross_attention.{name}.{key}': torch.full((3,), float('inf'))
     for name in ('proj_r2c', 'proj_i2r') for key in ('weight', 'bias')},
])
def test_missing_or_nonfinite_projection_never_vacuously_passes(state):
    with pytest.raises(ValueError):
        gate.projection_evidence(state)


def test_resource_reference_requires_measured_time_and_memory():
    with pytest.raises(ValueError, match='measurements'):
        checks.performance([dict(mode='train', iter=450, time=1.)])
    rows = [dict(mode='train', iter=i, time=1., memory=100.) for i in range(100, 451, 50)]
    assert checks.performance(rows)['mean_iter_time_s'] == 1.


def test_baseline_smoke_is_recipe_matched():
    report = gate.single_variable_contract()
    assert report['model_diff'] == ['RCFusion']
    assert report['recipe_diff'] == []
    assert report['new_params'] == 177726
    arm = Config.fromfile('me_rssm/sanity/gpmsf_smoke_450iter.py')
    base = Config.fromfile('me_rssm/sanity/gpmsf_baseline_smoke_450iter.py')
    for key in ('data', 'optimizer', 'optimizer_config', 'runner', 'lr_config'):
        assert arm[key] == base[key]


def test_no_real_evidence_prevents_formal_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, 'single_variable_contract', lambda: {})
    result = gate.verdict(str(tmp_path), str(tmp_path), str(tmp_path / 'missing.json'))
    assert result['S0_ALL_PASS'] is False
    assert 'failure' in result


def test_stats_rank_zero_only_atomic_and_required(tmp_path, monkeypatch):
    fusion = SimpleNamespace(collect_stats=True, require_geo=True, new_param_seed=1,
                             last_stats=dict(radar_support_rate=0.5))
    runner = SimpleNamespace(iter=49, work_dir=str(tmp_path),
                             model=SimpleNamespace(cross_attention=fusion))
    hook = stats_hook.GPMSFWiringStatsHook()
    monkeypatch.setattr(stats_hook, 'get_dist_info', lambda: (1, 3))
    hook.after_train_iter(runner)
    assert not (tmp_path / hook.out).exists()
    monkeypatch.setattr(stats_hook, 'get_dist_info', lambda: (0, 3))
    hook.after_train_iter(runner)
    assert json.loads((tmp_path / hook.out).read_text())['records']
    assert not list(tmp_path.glob('*.tmp'))
    fusion.last_stats = None
    with pytest.raises(RuntimeError, match='missing'):
        hook.after_train_iter(runner)


def test_queue_no_c1_gate_or_early_done():
    script = Path('me_rssm/sanity/queue_gpmsf.sh').read_text()
    assert 'c1valid_curve_check' not in script
    assert 'gpmsf_cpu_checks.py' in script
    assert 'gpmsf_real_preflight.py' in script
    assert 'gpmsf_baseline_smoke_450iter.py' in script
    assert '--expected-epochs 24' in script
    assert script.rfind("say 'DONE:") > script.rfind('epoch_24.pth')
    assert 'pkill' not in script


def test_real_batch_fingerprint_handles_training_bitmap_masks():
    import numpy as np
    from mmcv.parallel import DataContainer
    from mmdet3d.datasets.structures.mask.structures import BitmapMasks
    from me_rssm.sanity.gpmsf_real_preflight import batch_fingerprint
    masks = BitmapMasks(np.zeros((1, 4, 5), dtype=np.uint8), 4, 5)
    data = dict(gt_masks=DataContainer([[masks]], cpu_only=True))
    before = batch_fingerprint(data)
    masks.masks[0, 0, 0] = 1
    assert batch_fingerprint(data) != before


@pytest.mark.parametrize('delta,expected', [(-2., 0), (-2.001, 2), (0., 0)])
def test_ep6_operational_stop_uses_complete_paired_mean(tmp_path, monkeypatch, delta, expected):
    base, arm = tmp_path / 'base', tmp_path / 'arm'
    write_rows(base, [val(e, 29.042) for e in range(1, 7)])
    write_rows(arm, [val(e, 29.042 + delta) for e in range(1, 7)])
    monkeypatch.setattr('sys.argv', ['checks', 'curve', '--work-dir', str(arm),
                                   '--b0', str(base), '--window', '1-6', '--early-stop'])
    assert checks.main() == expected


def test_owned_launcher_checks_exit_and_kills_failed_child_group(tmp_path):
    import os
    import subprocess
    script = Path('me_rssm/sanity/queue_gpmsf.sh').read_text()
    stop = script[script.index('stop_owned(){'):script.index('\non_exit(){')]
    run = script[script.index('run_owned(){'):script.index("\nsay 'armed (reviewed gates)")]
    harness = tmp_path / 'owned.sh'
    pid_file = tmp_path / 'child.pid'
    harness.write_text('set -u\nCHILD=0\nsay(){ :; }\nsleep(){ command sleep .05; }\n' +
                       stop + '\n' + run + '\n' +
                       'run_owned 10 "$1/success.log" no bash -c "exit 0" || exit 10\n' +
                       'if run_owned 10 "$1/fail.log" no bash -c "exit 7"; then exit 11; fi\n' +
                       'stop_owned\n' +
                       "if run_owned 0 \"$1/timeout.log\" no bash -c 'echo $$ > \"$1/child.pid\"; exec sleep 1000' bash \"$1\"; then exit 12; fi\n" +
                       'command sleep .2\nstop_owned\n')
    result = subprocess.run(['bash', str(harness), str(tmp_path)], capture_output=True, timeout=10)
    assert result.returncode == 0, result.stderr.decode()
    assert pid_file.exists(), result.stderr.decode()
    pid = int(pid_file.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_projection_roundtrip_keeps_stereo_column_and_bev_affine():
    from me_rssm.sanity.gpmsf_real_preflight import projection_errors
    camera = torch.tensor([[50., 0., 32., -8.], [0., 50., 24., 2.],
                           [0., 0., 1., 0.], [0., 0., 0., 1.]])
    extrinsic = torch.tensor([[0., -1., 0., 0.], [0., 0., -1., 0.],
                             [1., 0., 0., 0.], [0., 0., 0., 1.]])
    bda = torch.tensor([[0., -1.1, 0., 2.], [-1.1, 0., 0., -3.],
                        [0., 0., 1.1, .5], [0., 0., 0., 1.]])
    post = torch.eye(4)
    post[:2, :2] = torch.tensor([[0.8, -.1], [.1, .8]])
    post[:2, 2] = torch.tensor([5., 7.])
    raw = torch.tensor([[10., 0., 0., 1.], [20., 1., 2., 1.]])
    points = (raw @ bda.T)[:, :3]
    geo = dict(points=[points], bda=bda[None],
               post_rots=post[None, None, :2, :2], post_trans=post[None, None, :2, 2],
               lidar2img=(post @ camera @ extrinsic @ torch.inverse(bda))[None, None])
    meta = dict(cam2img=camera.numpy(), lidar2cam=extrinsic.numpy())
    assert max(projection_errors(geo, [meta])) < 1e-4
    # Dropping stereo's fourth column must not pass the independent check.
    broken = camera.clone()
    broken[:2, 3] = 0
    geo['lidar2img'] = (post @ broken @ extrinsic @ torch.inverse(bda))[None, None]
    assert max(projection_errors(geo, [meta])) > .1


def test_smoke_checkpoint_uses_installed_mmcv_zero_based_meta(tmp_path, monkeypatch):
    config_path = 'me_rssm/sanity/gpmsf_smoke_450iter.py'
    cfg = Config.fromfile(config_path)
    cfg.gpu_ids = range(3)
    # train_vod captures runner.meta.config BEFORE cfg.seed is assigned.
    phase = dict(started_at=0., config=config_path)
    monkeypatch.setattr(gate, 'completed_phase', lambda _directory: phase)
    write_rows(tmp_path, [dict(mode='train', iter=i, loss=1., time=1., memory=100.)
                          for i in range(50, 451, 50)])
    from mmcv.runner import IterBasedRunner
    fake_runner = SimpleNamespace(iter=449, epoch=0, model=torch.nn.Linear(1, 1),
                                  meta=dict(seed=0, config=cfg.pretty_text), optimizer=None)
    IterBasedRunner.save_checkpoint(fake_runner, str(tmp_path),
                                    filename_tmpl='iter_{}.pth', save_optimizer=False,
                                    create_symlink=False)
    checkpoint = torch.load(tmp_path / 'iter_450.pth', map_location='cpu')
    assert checkpoint['meta']['iter'] == 449
    rows, _ = gate.smoke_artifacts(str(tmp_path), config_path)
    assert rows[-1]['iter'] == 450
    checkpoint['meta']['iter'] = 448
    torch.save(checkpoint, tmp_path / 'iter_450.pth')
    with pytest.raises(ValueError, match='zero-based'):
        gate.smoke_artifacts(str(tmp_path), config_path)


@pytest.mark.parametrize('time_ratio,memory_ratio,expected', [
    (1.19, 1.19, True), (1.20, 1.20, True), (1.201, 1., False), (1., 1.201, False)])
def test_matched_resource_budget_is_a_hard_gate(tmp_path, monkeypatch, time_ratio, memory_ratio, expected):
    arm, baseline = tmp_path / 'arm', tmp_path / 'baseline'
    arm.mkdir()
    baseline.mkdir()
    real = tmp_path / 'real.json'
    sha = {'fusion.py': 'same'}
    real.write_text(json.dumps(dict(status='PASS', source_sha=sha,
                                   checks={key: True for key in gate.REQUIRED_REAL})))
    monkeypatch.setattr(gate, 'single_variable_contract', lambda: {})
    monkeypatch.setattr(gate, 'source_hashes', lambda: sha)
    monkeypatch.setattr(gate, 'projection_evidence', lambda _state: {})

    def fake_smoke(directory, _config):
        is_arm = directory == str(arm)
        rows = [dict(mode='train', iter=i, time=time_ratio if is_arm else 1.,
                     memory=100. * memory_ratio if is_arm else 100.)
                for i in range(100, 451, 50)]
        return rows, dict(state_dict={})

    monkeypatch.setattr(gate, 'smoke_artifacts', fake_smoke)
    wiring = dict(collect_stats=True, require_geo=True, ddp_world_size=3,
                  ddp_parameter_sha=['same'] * 3, completed_iters=450,
                  records=[dict(iter=i, stats=dict(checkpoint_active=True,
                               native_radar_requires_grad=[True] * 3,
                               native_image_requires_grad=[True] * 3,
                               native_radar_has_grad_fn=[True] * 3,
                               native_image_has_grad_fn=[True] * 3))
                           for i in range(50, 451, 50)])
    path = arm / 'gpmsf_wiring_stats.json'
    path.write_text(json.dumps(wiring))
    result = gate.verdict(str(arm), str(baseline), str(real))
    assert result['S0_ALL_PASS'] is expected
    assert result['perf_ratio_time'] == pytest.approx(time_ratio)
    assert result['perf_ratio_memory'] == pytest.approx(memory_ratio)
    # A missing mandatory probe must fail even with valid checkpoint evidence.
    path.unlink()
    assert gate.verdict(str(arm), str(baseline), str(real))['S0_ALL_PASS'] is False
