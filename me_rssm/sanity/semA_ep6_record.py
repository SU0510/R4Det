#!/usr/bin/env python3
"""Emit the section-72 training record for the candidate-A ep6 gate.

READ-ONLY. Run it after ``semA_ep6_verdict.py`` has reported the battery as
complete. It renders ``docs/training_runs_full.md`` section 72 from the
evidence JSONs and the run's own ``*.log.json``, so none of the numbers are
transcribed by hand.

Why this exists (AGENTS.md rules 5 and 6):

  * rule 5 fixes three summary readings and forbids an ambiguous bare "BEST":
    fixed-window mean WITH its true span and n, the all-epoch peak as
    "value @ epoch" (weight-independent) and best-saved as a SEPARATE field;
  * rule 6 requires the full per-epoch curve, the per-class breakdown and the
    mechanism diagnostics, not just the summary.

Hand-assembling that every time is where this project has already made
transcription and arithmetic errors (e.g. the DC1 threshold arithmetic
correction). The generator therefore keeps ONE source of truth for the gate
logic: it imports ``semA_ep6_verdict`` and reuses its ``check_evidence``,
``m1_grid_limited`` and ``val_by_epoch`` rather than restating them. It decides
nothing itself -- the fork stays a manual step.

Output is a standalone markdown file intended to be inserted into
``docs/training_runs_full.md`` as section 72. It deliberately does NOT edit the
training record in place: the canonical doc stays under human control, and the
generator's job stays mechanical.

Two things it asserts rather than leaves implicit:

  * the S1 verdict and the pre-registered fork text;
  * that NO S2 AP claim is made here (S2 is the ep12-16 window and needs user
    approval before any ep6->ep16 continuation).

Fields that cannot be derived from the artifacts (wall-clock run facts that
live only in the terminal history) are emitted as visible ``<!-- FILL: ... -->``
markers and counted in a warning on stderr, so they cannot be silently dropped.

Usage:
  python3 me_rssm/sanity/semA_ep6_record.py \
    --run /data/lurui/work_dirs/fgfull_N4_no2d_igdr_2x4_24e_semA_seed0 \
    --output /tmp/section72.md
"""

import argparse
import datetime
import glob
import importlib.util
import json
import os
import statistics
import sys

_REPO = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..'))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_REPO, rel))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Single source of truth for the gate logic, evidence manifest and baselines.
verdict = _load('verdict_ep6_for_record', 'me_rssm/sanity/semA_ep6_verdict.py')

DEFAULT_RUN = verdict.DEFAULT_RUN
WINDOWS = (
    ('近距', 3, 'z_util_std_ep6_w3.json'),
    ('中距', 100, 'z_util_std_ep6_w100.json'),
    ('远距', 1800, 'z_util_std_ep6_w1800.json'),
)
POST_SOURCE = 'post_source_std_ep6_stride700.json'

# Standard-family same-epoch comparator for the shuffle_z reading: the plain
# no2d_igdr RSSM, seed1, ep6. Transcribed from docs/training_runs_full.md
# section 62.2 (table "ep6 三窗口反事实诊断"); display-only, never re-derived
# from the reading under test.
STD_EP6_SHUFFLE_HEAD = {'近距': 0.4973, '中距': 0.0330, '远距': 0.0336}

# Train-side diagnostics of the new supervision head. These are train-split
# statistics (z sampled), recorded per epoch by the model itself.
TRAIN_STAT_KEYS = (
    ('stat_sem_hit1', '`stat_sem_hit1`'),
    ('stat_sem_thr50', '`stat_sem_thr50`'),
    ('loss_rssm_sem', '`loss_rssm_sem`'),
    ('stat_cls_res_rms', '`stat_cls_res_rms`'),
    ('stat_mu_diff_sq', '`stat_mu_diff_sq`'),
)

FILL = '<!-- FILL: {} -->'


def load_json(path):
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def mean_variant(payload, variant, field):
    """Mean of one variant's reading over the window's per-sample entries."""
    values = []
    for entry in (payload.get('per_sample') or {}).values():
        probe = (entry.get('variants') or {}).get(variant) or {}
        value = probe.get(field)
        if value is not None:
            values.append(value)
    return statistics.mean(values) if values else None


def window_readings(payload):
    """{variant: {'head': mean, 'feature': mean}} for one z-util window."""
    out = {}
    for variant in ('shuffle_z', 'zero_z', 'prior_only'):
        out[variant] = {
            'head': mean_variant(payload, variant, 'head_l2_ratio_vs_normal'),
            'feature': mean_variant(
                payload, variant, 'feature_l2_ratio_vs_normal'),
        }
    return out


def train_stats_by_epoch(run_dir):
    """{epoch: {key: value}} from the last train record of each epoch."""
    by_epoch = {}
    for path in sorted(glob.glob(os.path.join(run_dir, '*.log.json'))):
        with open(path, encoding='utf-8') as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get('mode') == 'train' and rec.get('epoch') is not None:
                    by_epoch[int(rec['epoch'])] = rec
    return by_epoch


def saved_epochs(run_dir):
    """Epochs with an actual epoch_<N>.pth on disk (checkpoint availability)."""
    out = []
    for path in glob.glob(os.path.join(run_dir, 'epoch_*.pth')):
        stem = os.path.basename(path)[len('epoch_'):-len('.pth')]
        if stem.isdigit():
            out.append(int(stem))
    return sorted(out)


def pct(value, digits=4):
    return 'n/a' if value is None else f'{value * 100:.{digits}f}%'


def num(value, digits=4):
    return 'n/a' if value is None else f'{value:.{digits}f}'


def signed(value, digits=4):
    return 'n/a' if value is None else f'{value:+.{digits}f}'


def render(run_dir, date_str):
    by_epoch = verdict.val_by_epoch(run_dir)
    eps = sorted(by_epoch)
    train_stats = train_stats_by_epoch(run_dir)
    gates_payload = verdict.load_json(os.path.join(run_dir, verdict.GATE_JSON))
    gates = (gates_payload or {}).get('gates', {})
    m1 = (gates_payload or {}).get('M1_sem_head', {})
    m2 = (gates_payload or {}).get('M2_exclusivity', {})
    m3 = (gates_payload or {}).get('M3_readout_toggle', {})
    m4 = (gates_payload or {}).get('M4_class_information', {})
    fingerprints = (gates_payload or {}).get('input_fingerprints', {}) or {}

    _, evidence_ok = verdict.check_evidence(run_dir)
    all_pass = all(gates.get(k) is True for k in verdict.REQUIRED_GATES)
    other_pass = all(gates.get(k) is True for k in verdict.REQUIRED_GATES
                     if k != 'M1')
    mean_excess = verdict.m1_excess_mean(m1)
    grid_limited = (verdict.m1_grid_limited(
        m1.get('mean_thr50'), m1.get('mean_hit1'), mean_excess)
        and other_pass and gates.get('M1') is False)

    lines = []
    add = lines.append

    def gate_text(name):
        if name not in gates:
            return 'NOT MEASURED'
        return 'PASS' if gates.get(name) else 'FAIL'

    add(f'## 72. 候选 A ep6 机制门（S1）：'
        f'{"全门通过" if all_pass else "未全门通过"}（{date_str}）')
    add('')
    add('本节执行 `docs/rssm_candidate_A_prereg_20261010.md` §5 预登记的 S1 ep6')
    add('机制门。训练从零起（3 卡 GPU5/6/7、seed0、deterministic、24e 余弦日程），')
    add('ep6 val 与 `epoch_6.pth` 落盘后由看门狗 `semA_watch`')
    add('（`me_rssm/sanity/watch_semA_ep6.sh`）中断训练、等待 GPU 释放，再自动跑')
    add('M1–M4 门 + 三窗口反事实 + source 探针。')
    add('')
    add('> **本节不宣称任何 S2 性能结论。** S2 是 ep12–16 五轮窗口，且其数值需')
    add('> 用户批准后才允许 ep6→ep16 续训（§5/§7）。本节只报 ep6 机制门与早期')
    add('> 曲线，二者不可互相替代。')
    add('')

    # --- summary ---------------------------------------------------------
    add('### 72.0 状态与判定摘要')
    add('')
    if not eps:
        add('**无 val 记录：S1 NOT EVALUABLE。**')
        add('')
    win_start, win_end = (eps[0], eps[-1]) if eps else (0, 0)
    overall_mean, overall_n, overall_span = verdict.window_labelled(
        by_epoch, verdict.METRIC, win_start, win_end)
    peak_ep = max((e for e in eps if verdict.METRIC in by_epoch[e]),
                  key=lambda e: by_epoch[e][verdict.METRIC], default=None)
    saved = saved_epochs(run_dir)
    saved_with_val = [e for e in saved if e in by_epoch]
    best_saved_ep = (max(saved_with_val,
                         key=lambda e: by_epoch[e][verdict.METRIC])
                     if saved_with_val else None)

    add('**三口径摘要**（AGENTS 规则 5：三个口径必须分别写清，不得用含糊的')
    add('“BEST” 代替其中任一项）：')
    add('')
    add('| 口径 | 数值 |')
    add('|---|---|')
    if overall_n:
        add(f'| 区间均值（标准 ep12-16 窗口不完整；实际跨度 '
            f'{overall_span}，n={overall_n}） | **{num(overall_mean)}** |')
    else:
        add('| 区间均值 | n/a（无 val） |')
    if peak_ep is not None:
        add(f'| 全轮峰值（所有 val epoch 的最高单点，与是否落盘无关） | '
            f'**{num(by_epoch[peak_ep][verdict.METRIC])} @ ep{peak_ep}** |')
    else:
        add('| 全轮峰值 | n/a |')
    if best_saved_ep is not None:
        add(f'| best saved（仅表示 checkpoint 可用性，不参与性能排名） | '
            f'**{num(by_epoch[best_saved_ep][verdict.METRIC])} '
            f'@ ep{best_saved_ep}**（落盘 {saved}） |')
    else:
        add(f'| best saved | n/a（落盘 {saved} 中无对应 val） |')
    add('')
    add('> 上述窗口均值是 **ep6 早期上下文**，不是 ep12–16 的 S2 判据；')
    add('> 单 seed、窗口未完整，不得据此解读效应。')
    add('')

    add('**S1 门控表**（阈值来自 §5 预登记，判定来自')
    add('`me_rssm/sanity/semA_ep6_verdict.py`，本节不重算门槛）：')
    add('')
    add('| 门 | 预登记门槛 | ep6 实测 | 判定 |')
    add('|---|---|---|---|')
    m2_ok = (m2.get('h_relative') is not None
             and m2.get('feat_relative') is not None
             and m2.get('h_relative') <= -0.30
             and m2.get('feat_relative') <= -0.30)
    add(f'| M1 语义头质量 | val `thr50 ≥ 0.30` 且 `hit@1 ≥ 0.60` | '
        f'`thr50` **{num(m1.get("mean_thr50"))}**、'
        f'`hit@1` **{num(m1.get("mean_hit1"))}** | **{gate_text("M1")}** |')
    add(f'| M2 z 排他性 | `h` 与 `feat` 两臂**各自** ≤ −30% | '
        f'`h` **{pct(m2.get("h_relative"))}**（head thr50 '
        f'{num(m2.get("h_mean_thr50"))} vs z '
        f'{num(m2.get("z_mean_thr50"))}）、`feat` '
        f'**{pct(m2.get("feat_relative"))}**'
        f'{"（两臂均达线）" if m2_ok else ""} | **{gate_text("M2")}** |')
    add(f'| M3 读出非装饰 | 强制 residual=0 使正样本 class-CE 变差 ≥8/10 | '
        f'**{m3.get("wins")}/{m3.get("n")}** | **{gate_text("M3")}** |')
    add(f'| M4 类别正确性 | `own_hit` 与 `donor_transfer` ≥ chance+20pt '
        f'（bar {num(m4.get("bar"), 2)}） | `own_hit` '
        f'**{num(m4.get("mean_own_hit"))}**、`donor_transfer` '
        f'**{num(m4.get("mean_donor_transfer"))}**（chance '
        f'{num(m4.get("chance"), 2)}） | **{gate_text("M4")}** |')
    add('')
    chance_vals = [c['hit1_chance'] for c in (m1.get('per_clip') or [])
                   if c.get('hit1_chance') is not None]
    mean_chance = statistics.mean(chance_vals) if chance_vals else None
    add(f'**不进门补充读数（M1 口径）**：`hit1_chance` 均值 '
        f'**{num(mean_chance, 6)}**、'
        f'`hit1_excess_over_chance` 均值 **{num(mean_excess)}**。')
    add('按 §71.9.3(2)，`hit@1 ≥ 0.60` 是从 route-C 的 16×16 网格外推而来，')
    add('A 的 full-res 216×248 网格随机基线只有 ~7.5e-5（209× 格数），')
    add('故这两个读数必须与门槛并排记录。')
    add('')

    if not evidence_ok:
        add('**证据包不完整 ⇒ NOT EVALUABLE**：上面四门读数在缺失产物补齐前')
        add('一律为暂读，即使四门全过也不得判 ALL PASS。')
    elif grid_limited:
        add('**S1 聚合判定：证据不足（INSUFFICIENT EVIDENCE），不是机制否定。**')
        add('M1 只在 full-res 原始 `hit@1` 上未过，而 `thr50` 与')
        add('`hit1_excess_over_chance` 健康、其余三门全过，命中 §71.9.3(2)')
        add('预先写定的例外条款：需补一个 full-res 同口径对照后再判，')
        add('不得当通过，也不得归档为负结果。')
    elif all_pass:
        add('**S1 聚合判定：ALL PASS。** 按 §7 分叉，下一步是**请求用户批准')
        add('§5 的 S2 数值**，批准后才允许从 `epoch_6.pth` 续训至 ep16。')
    else:
        failing = [k for k in verdict.REQUIRED_GATES if gates.get(k) is not True]
        add(f'**S1 聚合判定：NOT ALL PASS（{", ".join(failing)}）。** 按 §7')
        add('分叉，按负结果归档并把读数写进 Analysis，**不进 ep16**。')
        if gates.get('M2') is not True:
            add('M2 未过 ⇒ **“概率状态”这一宣称作废**：语义来自 h/观测，')
            add('z 只是被穿过的通路。')
    add('')

    # --- config ----------------------------------------------------------
    add('### 72.1 配置与唯一变量')
    add('')
    add('| 项 | 值 |')
    add('|---|---|')
    add(f'| 配置 | `{os.path.relpath((gates_payload or {}).get("config", "n/a"), _REPO) if (gates_payload or {}).get("config") else "n/a"}` |')
    add('| 对照 | `..._no2d_igdr.py`（`MotionAlignedRSSMFusion`, seed0） |')
    add('| 唯一变量 | `sem_head`（z 排他语义监督）+ `z_cls_logits`'
        '（零初始化分类残差读出） |')
    add('| 未改动 | 重建目标、posterior/prior、KL/free-nats、`output_proj`、'
        'backbone、head 几何、优化器、日程、数据管线 |')
    add(f'| 权重指纹 | `sha256={str(fingerprints.get("checkpoint_sha256"))[:16]}...`、'
        f'size={fingerprints.get("checkpoint_size_bytes")}、'
        f'mtime={fingerprints.get("checkpoint_mtime")} |')
    add(f'| 采样位点 | clip 索引 `{(gates_payload or {}).get("indices")}`、'
        f'`shuffle_offset={(gates_payload or {}).get("shuffle_offset")}` |')
    add('')

    # --- run facts -------------------------------------------------------
    add('### 72.2 运行事实')
    add('')
    add('| 项 | 值 |')
    add('|---|---|')
    add(FILL.format('ep6 段起止 UTC 时刻（stdout.log 首末时间戳）'))
    add('| GPU / seed | 5/6/7、seed 0、deterministic |')
    add(FILL.format('吞吐中位 s/iter、显存 GiB/卡（训练日志 time/memory 段）'))
    add(f'| val 记录 | ep{eps[0]}-ep{eps[-1]} 各一条，共 {len(eps)} 条 |'
        if eps else '| val 记录 | 无 |')
    add(f'| 落盘 | {saved}（`checkpoint_config.interval=2`） |')
    add(FILL.format('停止方式与 GPU 释放时刻（watch_semA_ep6.log）'))
    add('')

    # --- per-epoch curve --------------------------------------------------
    add(f'### 72.3 完整逐 epoch 曲线（ep{eps[0]}-ep{eps[-1]}）' if eps
        else '### 72.3 完整逐 epoch 曲线')
    add('')
    add('| ep | Overall3Dmod | vs base | Car | Truck | Ped_l | Cyc_l |')
    add('|---:|---:|---:|---:|---:|---:|---:|')
    base_by_epoch = {}
    for e in eps:
        rec = by_epoch[e]
        o = rec.get(verdict.METRIC)
        cells = []
        for key in verdict.CLASS_KEYS.values():
            cells.append(num(rec.get(key), 2))
        base = verdict.BASELINE['Overall']
        base_by_epoch[e] = o
        add(f'| {e} | {num(o, 2)} | {signed((o - base) if o is not None else None, 2)} '
            f'| ' + ' | '.join(cells) + ' |')
    add('')
    add(f'> `vs base` 对照 §5 的 seed0 基线 Overall **{verdict.BASELINE["Overall"]}**'
        '（ep12–16 窗口均值）；单点差值不含 seed 噪声。')
    add('')

    # --- window per-class -------------------------------------------------
    add(f'### 72.4 窗口逐项均值（实际跨度 {overall_span}，n={overall_n}）')
    add('')
    add('| 类别 | 本 run 窗口均值 | seed0 基线（ep12-16） | 差 |')
    add('|---|---:|---:|---:|')
    add(f'| Overall_3D_moderate | **{num(overall_mean, 2)}** | '
        f'{verdict.BASELINE["Overall"]} | '
        f'{signed((overall_mean - verdict.BASELINE["Overall"]) if overall_mean is not None else None, 2)} |')
    for name, key in verdict.CLASS_KEYS.items():
        m, n, span = verdict.window_labelled(by_epoch, key, win_start, win_end)
        add(f'| {name} | {num(m, 2)} | {verdict.BASELINE[name]} | '
            f'{signed((m - verdict.BASELINE[name]) if m is not None else None, 2)} |')
    add('')
    add('> 该窗口与基线窗口**不同长度**（基线段是 ep12-16，本 run 只有 ep1-6），')
    add('> 只作早期上下文，不作 S2 判据。')
    add('')

    # --- train stats ------------------------------------------------------
    add('### 72.5 A 侧训练期同口径读数（train split，逐 epoch 末值）')
    add('')
    add('| ep | ' + ' | '.join(label for _, label in TRAIN_STAT_KEYS) + ' |')
    add('|---:|' + '---:|' * len(TRAIN_STAT_KEYS))
    for e in sorted(train_stats):
        rec = train_stats[e]
        cells = [num(rec.get(key), 4) for key, _ in TRAIN_STAT_KEYS]
        add(f'| {e} | ' + ' | '.join(cells) + ' |')
    add('')
    if train_stats:
        last = train_stats[max(train_stats)]
        add(f'shuffle/zero 与门控读数无关；此处 `stat_sem_*` 为**训练统计**'
            f'（z 采样值、train split），与探针的 val 口径不同，不可混比。')
        add(f'末 epoch `stat_cls_res_rms` = '
            f'{num(last.get("stat_cls_res_rms"))}（>0 ⇒ 读出活着）。')
    add('')

    # --- gate detail ------------------------------------------------------
    add('### 72.6 S1 门明细（逐 clip）')
    add('')
    add('**M1 逐 clip**：')
    add('')
    add('| index | thr50 | hit@1 | hit1_chance | excess_over_chance | 正格数 |')
    add('|---:|---:|---:|---:|---:|---:|')
    for clip in (m1.get('per_clip') or []):
        add(f'| {clip.get("index")} | {num(clip.get("thr50"))} | '
            f'{num(clip.get("hit1"))} | {num(clip.get("hit1_chance"), 6)} | '
            f'{signed(clip.get("hit1_excess_over_chance"), 6)} | '
            f'{clip.get("num_pos_cells")} |')
    add('')
    add('**M2 两控制臂逐 clip**（`thr50`；z 为参照）：')
    add('')
    add('| index | z | h | feat |')
    add('|---:|---:|---:|---:|')
    z_by_index = {c.get('index'): c for c in (m1.get('per_clip') or [])}
    h_by_index = {c.get('index'): c for c in (m2.get('per_clip_h') or [])}
    f_by_index = {c.get('index'): c for c in (m2.get('per_clip_feat') or [])}
    for index in sorted(set(z_by_index) | set(h_by_index) | set(f_by_index)):
        add(f'| {index} | {num((z_by_index.get(index) or {}).get("thr50"))} | '
            f'{num((h_by_index.get(index) or {}).get("thr50"))} | '
            f'{num((f_by_index.get(index) or {}).get("thr50"))} |')
    add('')
    add('**M3 逐 clip**（`delta_off_minus_on > 0` 表示关掉 residual 后')
    add('正样本 class-CE 变差，即读出有贡献）：')
    add('')
    add('| index | CE(on) | CE(off) | delta(off-on) | 正格数 |')
    add('|---:|---:|---:|---:|---:|')
    for clip in (m3.get('per_clip') or []):
        add(f'| {clip.get("index")} | {num(clip.get("ce_on"))} | '
            f'{num(clip.get("ce_off"))} | '
            f'{signed(clip.get("delta_off_minus_on"))} | '
            f'{clip.get("cells")} |')
    add('')
    add('**M4 逐 clip**（`own_hit` = z 残差 argmax 命中该格 GT 类；')
    add('`donor_transfer` = 换 donor z 后残差 argmax 命中 donor 主类；')
    add('`head_swap_to_donor` = 只重跑 head 且喂 donor 残差时的 head 类 argmax）：')
    add('')
    add('| index | own_hit | donor_transfer | head_swap | donor_index | changed_frac |')
    add('|---:|---:|---:|---:|---:|---:|')
    for clip in (m4.get('per_clip') or []):
        add(f'| {clip.get("index")} | {num(clip.get("own_hit"))} | '
            f'{num(clip.get("donor_transfer"))} | '
            f'{num(clip.get("head_swap_to_donor"))} | '
            f'{clip.get("donor_index")} | {num(clip.get("changed_frac"))} |')
    add('')

    # --- family readout ---------------------------------------------------
    add('### 72.7 家族标准反事实读数（`shuffle_z` / `zero_z` / `prior_only`）')
    add('')
    add('三窗口与既往各代（MAI / MCI / DC-RSSM）使用同一探针语义：')
    add('`shuffle_z` 用同 batch 另一样本的 `mu_q` 替换，`zero_z` 置 0，')
    add('`prior_only` 只走先验。**A 跑标准家族**，故用')
    add('`probe_z_utilization_standard.py`（后验头重写 + 转发部署态 class')
    add('residual），与 `tools/diagnose_z_utilization.py` 的打分口径一致。')
    add('')
    add('| 窗口 | shuffle_z head | zero_z head | prior_only head | '
        'shuffle/zero | shuffle_z feat | 标准 RSSM ep6 同窗 head（对照） |')
    add('|---|---:|---:|---:|---:|---:|---:|')
    readings = {}
    for label, start, name in WINDOWS:
        path = os.path.join(run_dir, name)
        if not os.path.exists(path):
            add(f'| {label} | 缺失 | | | | | |')
            continue
        payload = load_json(path)
        readings[label] = window_readings(payload)
        r = readings[label]
        sh, ze = r['shuffle_z']['head'], r['zero_z']['head']
        ratio = (sh / ze) if (sh is not None and ze) else None
        add(f'| {label}（{payload.get("indices")}） | **{pct(sh)}** | '
            f'{pct(ze)} | {pct(r["prior_only"]["head"])} | '
            f'{num(ratio, 3)} | {pct(r["shuffle_z"]["feature"])} | '
            f'{pct(STD_EP6_SHUFFLE_HEAD[label], 2)} |')
    add('')
    if readings:
        target = []
        for label, _, _ in WINDOWS:
            if label in readings and readings[label]['shuffle_z']['head'] is not None:
                target.append(
                    f'{label} {pct(readings[label]["shuffle_z"]["head"], 2)}')
        add(f'本 run 三窗 `shuffle_z` head：{"; ".join(target)}。')
        add('对照为**标准 no2d_igdr RSSM seed1 ep6**同窗口（'
            '`docs/training_runs_full.md` §62.2），因同家族、同 epoch，')
        add('是唯一口径一致的机制对照；MAI/MCI/DC 各代的同窗读数见 §62/§63/§68。')
        add('')
        indices_by_window = []
        for label, _, name in WINDOWS:
            path = os.path.join(run_dir, name)
            if os.path.exists(path):
                indices_by_window.append(
                    (label, load_json(path).get('indices')))
        spans = {tuple(v) if v else None for _, v in indices_by_window}
        if len(spans) == 1:
            only = next(iter(spans))
            add(f'> **警告：三个窗口返回了同一组 clip {list(only) if only else None}**，'
                '即三窗口不是三个独立样本窗。上述三行因此不是相互独立的读数，')
            add('> 不得当作「三窗口一致」；请核对探针的 `--start-index` 与')
            add('> `--limit` 是否真的换窗。')
    add('')

    # --- posterior source -------------------------------------------------
    add('### 72.8 posterior source 归因（`stride=700`）')
    add('')
    post_path = os.path.join(run_dir, POST_SOURCE)
    if not os.path.exists(post_path):
        add('**缺失**：source 探针未产出，归因读数不可用。')
        add('')
    else:
        post = load_json(post_path)
        # The real tool reports a four-field block per tensor. A file carrying
        # only ``rms`` is not a truncated reading to be papered over -- it is a
        # different (usually hand-made) artifact, so say so instead of
        # rendering a half-filled table as if it were the measurement.
        rich = [k for k in ('mu_q', 'h_t', 'e_pooled')
                if isinstance(post.get(k), dict)
                and 'sample_specific_energy_frac' in post[k]]
        add('| 量 | RMS | batch-common RMS | 样本特异 RMS | 样本特异能量占比 |')
        add('|---|---:|---:|---:|---:|')
        for key in ('mu_q', 'h_t', 'e_pooled'):
            block = post.get(key) or {}
            add(f'| `{key}` | {num(block.get("rms"))} | '
                f'{num(block.get("batch_common_rms"))} | '
                f'{num(block.get("sample_specific_rms"))} | '
                f'{pct(block.get("sample_specific_energy_frac"), 2)} |')
        add('')
        if len(rich) < 3:
            add('> **警告：该 source JSON 不是完整读数**——缺少')
            add('> `sample_specific_energy_frac`（真实工具必写）。上面的空列')
            add('> 不得当作「特异占比为 0」，整个归因表也不得引用。')
            add('')
        removal_h = post.get('mu_q_removal_of_h_rel_l2')
        removal_e = post.get('mu_q_removal_of_e_rel_l2')
        add('| 读出/依赖读数 | 值 |')
        add('|---|---:|')
        add(f'| `mu_q_removal_of_h_rel_l2` | {num(removal_h)} |')
        add(f'| `mu_q_removal_of_e_rel_l2` | {num(removal_e)} |')
        if removal_h is not None and removal_e:
            add(f'| **h/e 依赖比** | **{num(removal_h / removal_e, 3)}** |')
        add(f'| `posterior_obs_mode` | `{post.get("posterior_obs_mode")}` |')
        add(f'| `z_proj_init` | `{post.get("z_proj_init")}` |')
        add('')
        add('> h/e 依赖比是各代通用的机制对照量（标准主线 ep12 为 2.338、'
            'MAI ep12 为 0.842、MCI ep12 为 0.605）；越低表示 posterior')
        add('> 越依赖观测而非 GRU 状态。')
        add('')
        add(f'fusion 类：`{post.get("fusion_class")}`、latent '
            f'{post.get("latent_resolution")}、readout '
            f'`{post.get("readout_mode")}`、clip 索引 {post.get("indices")}。')
        add('')

    # --- verdict / next step ---------------------------------------------
    add('### 72.9 判定与下一步（按 §7 预登记分叉，人工步骤）')
    add('')
    if not evidence_ok:
        add('1. 证据包不完整：重跑缺失探针后再判，**不得**读成门没过。')
    elif grid_limited:
        add('1. **补一个 full-res 同口径对照**后再判 M1（§71.9.3(2)）；')
        add('   在此之前既不算通过也不算否定。')
    elif all_pass:
        add('1. S1 全过 ⇒ **请求用户批准 §5 的 S2 数值**（Overall ≥ 40.00；')
        add('   非劣 Car ≥ 47.34 / Truck ≥ 29.22 / Ped ≥ 29.93 / Cyc ≥ 47.63；')
        add('   hard fail < 38.78）；批准后才允许从 `epoch_6.pth` 续训至 ep16。')
    else:
        add('1. S1 未全过 ⇒ 按负结果归档，读数写进 Analysis，不投 ep16。')
    add('2. 任意 Ped/Cyc 触到非劣线 ⇒ 停止，记为家族副作用第 6 次复现。')
    add('3. **不做**权重/KL 扫描，**不做** ep24 抢救，不靠放大系数救活机制。')
    add('')

    add('### 72.10 本节边界')
    add('')
    add('* 无 S2 性能宣称；§72.0/72.4 的均值是 ep6 早期上下文。')
    add('* 未改模型、未改判据数值、未动重建目标/KL/权重/日程。')
    add('* 本节由 `me_rssm/sanity/semA_ep6_record.py` 从证据 JSON 与')
    add('  `*.log.json` 机械生成，未手抄任何读数；判定沿用')
    add('  `semA_ep6_verdict.py` 的同一套门槛常量。')
    add('')

    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run', default=DEFAULT_RUN,
                    help='work_dir holding the ep6 evidence bundle')
    ap.add_argument('--output', default=None,
                    help='write markdown here (default: stdout)')
    ap.add_argument('--date', default=None,
                    help='YYYY-MM-DD for the heading (default: today UTC)')
    args = ap.parse_args()

    if not os.path.isdir(args.run):
        print(f'no such run dir: {args.run}', file=sys.stderr)
        return 2

    date_str = args.date or datetime.datetime.utcnow().strftime('%Y-%m-%d')
    text = render(args.run, date_str)

    if args.output:
        output = os.path.abspath(args.output)
        os.makedirs(os.path.dirname(output) or '.', exist_ok=True)
        tmp = output + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as handle:
            handle.write(text + '\n')
        os.replace(tmp, output)
    else:
        print(text)

    unfilled = text.count('<!-- FILL:')
    if unfilled:
        print(f'[record] WARNING: {unfilled} field(s) need a human value: '
              'grep "<!-- FILL:" in the output before pasting it into '
              'docs/training_runs_full.md', file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
