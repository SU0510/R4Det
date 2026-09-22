# 主模型口径下 GRU 时序基线（baseline_temporal）的准确率预测

> 性质：**实验前的先验估计（prior），不是实测结果，不能写进论文当数字用**。
> 目的：给「要不要花 3 卡 × 24h 重跑同栈 GRU 控制」和「汇报时怎么摆 baseline 那一行」提供决策依据。
> 生成日期：2026-09-22（盘上状态：`fgfull_N4_no2d_igdr` 正在 ep18+ 收尾）

---

## 0. 先明确「baseline」指哪个

| 读法 | 具体对象 | 状态 |
|---|---|---|
| **本文采用的读法** | GRU 时序基线 `TemporalDeformableFusionBaseline`（= 18e 的 `baseline_temporal` 模块）在当前栈上训练 | **无任何实测**，只死过一次（`fgfull_N4_temporal_baseline_seed0`，ep1 iter900 被杀，无 val、无权重） |
| 另一读法 | 单帧对照 `no_temporal`（时序关闭） | **已实测**：35.59 BEST / 34.65 窗口（`no_temporal_N4_2x4_24e_seed0`），不需要预测 |

所以下面预测的是**当前栈口径下的 GRU 时序基线**。

---

## 1. 当前栈的「时序机制阶梯」（全部同栈：pretrained + head-v2 + 24e + 3 卡 batch12 + N4 + seed0）

均为非 FG 栈，BEST 取各 run 最高单点，窗口取 ep12–16 均值：

| # | 时序机制 | run | BEST 3D_mod | 窗口 ep12-16 |
|---|---|---|---:|---:|
| 1 | **关闭**（`temporal_fusion=None`） | `no_temporal_N4_2x4_24e_seed0` | 35.59 @ep20 | 34.65 |
| 2 | 确定性隐变量（`DeterministicMotionAlignedLatentFusion`，有对齐、**无随机 z**） | `deterministic_latent_N4_2x4_24e_seed0` | 38.44 @ep13 | 37.08 |
| 3 | 固定噪声 posterior | `rssm_N4_fixednoise_posterior_seed0` | 38.42 @ep16 | 36.35 |
| 4 | 仅 posterior、可学 std | `posterior_only_learnable_std_N4_2x4_24e_seed0` | 39.66 @ep10 | 37.04 |
| 5 | 满血 RSSM，KL=0 | `rssm_kl0_N4_2x4_24e_seed0` | 40.35 @ep22 | 39.06 |
| 6 | 满血 RSSM（Run 9） | `rssm_N4_2x4_24e_pretrained_v2_head` | 40.59 @ep11 | 39.06 |
| 7 | 满血 RSSM（run10 seed0） | `run10_headv2_multiseed/seed_0` | 39.88 @ep16 | 38.39 |
| 8 | 满血 RSSM（主模型 seed2） | `run10_headv2_multiseed/seed_2` | **40.88** @ep14 | 39.40 |
| 9 | 满血 RSSM + FG（seed0） | `fgfull_N4_2x4_24e_seed0` | 40.69 @ep16 | **40.02** |
| 10 | 满血 RSSM + FG − 2D − IGDR（seed0） | `fgfull_N4_no2d_igdr_2x4_24e_seed0` | 40.39 @ep14 | 39.28 |

**这张阶梯给出的关键差值（当前栈内、单变量）**：

- 「任意时序聚合（含对齐）」相对关闭：**+2.85 BEST / +2.43 窗口**（行 2 − 行 1）
- 「把随机 z 加回来」相对确定性隐变量：**+1.44 BEST / +1.31 窗口**（行 7 − 行 2）
- FG 监督的相对贡献：**+0.81 BEST / +1.63 窗口**（行 9 − 行 7，同 seed、同机制）

---

## 2. 预测

GRU 基线在结构上没有自维护隐状态、没有随机 z、没有对 h/z 的运动对齐（它对齐的是 `feat_prev`），
但在阶梯上**它一定带「任意时序聚合」这一项**（对应行 2 的 +2.85），**不带**随机性那一项（+1.44）。

因此它应当落在**行 1 与行 7 之间、贴近行 2 的位置**：

| 口径 | 预测中位 | 80% 区间 | 与之对照 |
|---|---:|---|---:|
| 当前栈（非 FG），BEST | **38.5** | 37.0 – 40.5 | 满血 RSSM seed0 = 39.88 |
| 当前栈（非 FG），窗口 ep12-16 | **37.3** | 36.2 – 38.6 | 满血 RSSM seed0 = 38.39 |
| FG 栈（= 死掉那个 run 的口径），BEST | **39.3** | 37.8 – 41.3 | FG 满血 RSSM seed0 = 40.69 |
| FG 栈，窗口 ep12-16 | **38.9** | 37.5 – 40.4 | FG 满血 RSSM seed0 = 40.02 |

**由此推出的 RSSM 净收益（同栈、同 seed、同机制，唯一变量 = GRU ↔ Motion-Aligned RSSM）**：

| 口径 | 预测 RSSM − GRU | 对照：现在表里用的 |
|---|---:|---:|
| 当前栈 BEST | **+1.4**（区间 0 ~ +3.4） | +6.38（vs 18e 的 34.50） |
| FG 栈 BEST | **+1.4**（区间 0 ~ +3.4） | 同上 |
| 窗口 ep12-16 | **+1.1**（区间 −0.3 ~ +2.9） | — |

---

## 3. 两个估计量，以及为什么这么加权

**估计量 A（主，权重高）—— 当前栈阶梯插值**：GRU 落在「无时序」与「确定性隐变量」之间偏上。
依据是 §1 的实测阶梯（行 1/2/7），全部同栈单变量。→ 38.5 BEST。

**估计量 B（辅，权重低）—— 18e 栈的 GRU ↔ RSSM 排序迁移**：18e 那次 8 个 run 里
**GRU（34.50）排第 1，压过全部 RSSM 变体**（v2 34.01、v3 33.77、baseline_rssm 33.54、v1 30.89），
相对最好 RSSM +0.49、相对家族均值 +1.45。若直接迁移这个排序，GRU ≈ 40.69 + 0.5 ≈ **41.2，反超主模型**。

**B 被降权的原因**：18e 那批 RSSM 是**冻结前**的状态（§8.1 的 `b64dd1f` Xavier init + `action_dim=0` 在 18e 之后），
其中 `baseline_rssm` 与 v1 明确「KL 被压死」，v2 是「修 KL 超参、追平 baseline」——即对手是被削弱过的 RSSM；
且 Run 1 与 RSSM 系的 config diff 里还混着 `depth2img`/`gt_depths`/`LoadAnnotations3D` 等数据管线字段差异（《训练记录》§4 脚注），
不是干净的单变量对照。

**但 B 不能丢**：它把上行风险撑起来了——预测区间上沿给到 40.5/41.3 正是为了容纳
「GRU 在真实同栈里其实不输 RSSM」这一可能。这也意味着**预测 RSSM − GRU 的符号并不稳固**，
估计「GRU 反超或打平」的概率约 **20–25%**。

---

## 4. 对汇报和论文的三条直接结论

1. **现在表里那行 `baseline_temporal 34.50` 不能用来说明 RSSM 有用**。它是 18e / 2 卡 / 无预训练 / head-v1 的栈，
   跟主模型的 24e / 3 卡 / 预训练 / head-v2 差了三项，量级差 +6.38 里大头是栈差不是方法差。
2. **能安全主张的是「时序开关」而不是「RSSM vs GRU」**：同栈单变量实测 **+4.29 BEST / +3.74 窗口**
   （`no_temporal` 35.59/34.65 ↔ `run10 seed_0` 39.88/38.39）。这条是实测，站得住。
3. **一旦补上同栈 GRU 控制，RSSM 的领先预计从 +6.38 缩到约 +1.4（可能为 0）**。
   如果论文必须有 GRU 这一行，建议提前按这个预期组织叙事（把主张放在时序开关 + 机制阶梯），
   而不是指望 GRU 那一行给出大差距。

---

## 5. 怎么花最小的代价验证

- 完整重跑 FG 栈 GRU 控制（`fgfull_N4_temporal_baseline_seed0`）：24e × 3 卡 ≈ 24 h，能把预测的 ±1.5 收紧到 ±0.5。
- **更便宜的一条**：先跑非 FG 栈的 GRU 控制（`TemporalDeformableFusionBaseline` + 非 FG 24e pretrained head-v2 + seed0），
  与 `run10_headv2_multiseed/seed_0`（39.88）直接配对，同样 24 h 但**对照对象更干净**（主模型口径就是 run10 系列）。
- 判别点看 **Car strict**：主模型相对基线的增量约 73% 来自 Car strict（+18.53/6.38）；
  如果 GRU 控制把 Car strict 也拉到 45+，那 §3 的上行情景成立，RSSM 的净收益会进一步被压缩。
- 复核窗口口径时注意：FG 栈把窗口值抬得比单点口径更多（行 9 vs 行 7：窗口 +1.63、单点 +0.81），
  所以**同一行不要混用 BEST 和窗口**，两张表分开列。

---

## 6. 数据来源

- BEST / 窗口：各 `work_dirs/<run>/*.log.json`，每 epoch 取最后一条 `mode=val`，脚本 `tools/summarize_run.py` 的 `load_eval_series`，
  指标 `pts_bbox/KITTI/Overall_3D_moderate`（混合口径 = mean(Ped_loose, Cyc_loose, Car_strict, Truck_strict)）。
- 18e 数值：`docs/training_runs_full.md` §0–§8。【记录口径】原始日志已删，不可重算。
- 配置差异：各 run 目录下的 `*.py` 快照（如 `deterministic_latent_.../TJ4D-R4Det_motion_align_deterministic_latent_N4_2x4_24e_pretrained_v2_head.py` 的 `temporal_fusion.type`）。
