# 可用于汇报展示对比的 run 清单

> 口径：**BEST**（汇报/论文用 BEST；窗口均值只用于实验室内选型）。
> 指标：`Overall 3D moderate`（AP40，TJ4D val）。带 ★ 的是建议直接上台的主叙事。
> 数据来源：`work_dirs/<run>/*.log.json` 的 `mode=val`，每 epoch 取最后一次 val。
> 方法与实时状态见文末。

---

## 第 1 组：主叙事（A 栈，4 次 run）★

同栈 = 24e / pretrained / head-v2 / **无 fg 监督**。这组是唯一「无时序 vs 有时序」的干净对照。

| # | run | BEST 3D_mod | @ep | 定位 |
|---|---|---:|---:|---|
| 1 | `no_temporal_N4_2x4_24e_seed0` | 35.59 | 20 | 单帧下限（时序融合的对照） |
| 2 | `run10_headv2_multiseed/seed_2` | **40.88** | 14 | 最高单点（已存，released 候选） |
| 3 | `run10_headv2_multiseed/seed_1` | 40.51 | 15 | 复现 seed |
| 4 | `run10_headv2_multiseed/seed_0` | 39.88 | 16 | 复现 seed |

**主结论：时序融合净贡献 +4.83**（40.42 − 35.59），三 seed 报 **40.42 ± 0.51**。
（若只用同 seed 对比 seed0：39.88 − 35.59 = +4.29。）

## 第 2 组：机制消融（A 栈 seed0，4 次 run）

同栈、同 seed0，回答「RSSM 里哪些部件不能删」。

| run | BEST 3D_mod | @ep | Δ vs 完整 RSSM(39.88) |
|---|---:|---:|---:|
| 完整 RSSM（seed0 参考） | 39.88 | 16 | — |
| `deterministic_latent_...` | 38.44 | 13 | −1.44 |
| `rssm_N4_fixednoise_posterior_seed0` | 38.42 | 16 | −1.46 |
| `posterior_only_learnable_std_...` | 39.66 | 10 | −0.22（窗口崩到 37.04） |
| `rssm_kl0_N4_2x4_24e_seed0` | 40.35 | 22 | +0.47（**但多 seed 不稳，见下**） |

**主结论**：确定性 latent、固定噪声、只留 posterior 都会掉 → **KL + prior + posterior sampling 三者都要保留**。

## 第 3 组：贡献链（2 次 run，讲「为什么最后能到 40+」）

| run | BEST 3D_mod | @ep | 增量 |
|---|---:|---:|---:|
| `rssm_N4_2x4_30e`（无预训练） | 34.71 | 22 | 基线 |
| `rssm_N4_2x4_30e_pretrained`（+预训练） | 37.94 | 19 | **+3.23**（Car strict +12.62） |
| → 再加 head-v2（= 第 1 组 RSSM） | 40.42 | — | **+2.48** |

⚠️ 这两次是 **30e**，第 1 组是 **24e**，表注要写清。

## 第 4 组：负消融（A 栈，同 base 只改一处，4 次 run）

「我们试过但没用」的证据，建议一页带过或放附录。

| run | 唯一变量 | BEST 3D_mod | @ep | 代价 |
|---|---|---:|---:|---|
| `..._truck` | Truck×3 anchor | 38.45 | 11 | Car −5.01 |
| `..._truck_car2` | Car-large anchor | 38.11 | 14 | Truck −5.98 |
| `..._dynmask` | Doppler 动静 mask | 36.21 | 8 | 全线回退 |
| `..._bptt` | truncated BPTT | 37.84 | 14 | 负收益 |

---

## 不放进主对比的

- **完整 FG-FULL `fgfull_N4_2x4_24e_seed0` = 40.69 @ep16**：已跑完、分数最高，但它属 **B 栈**（含 2D instance + IGDR 监督），
  与第 1 组不同栈，**不能直接和 40.42 比**；且只有 seed0。要列就单开一行并注明栈不同。
- **KL=0 seed1 `rssm_kl0_N4_2x4_24e` = 41.10 @ep20**：全库单点最高，但 KL=0 已被三 seed 否决
  （seed0 40.35 / seed1 41.10 / **seed2 36.76**）。只能当「单点不可信」的反例讲，不能当成绩。
- **CycCls seed0 40.63 / shared stem 40.30 / crossmodal 39.28**：seed0 单点，且 CycCls seed1 复现失败；
  另有 16ep 截断问题。最多作为「Car strict 有跨 seed 正信号 +2.9」的附带证据。
- **Run 1–8（18e 栈，9 次 run）**：记录齐全可用，但要**单独一页**讲早期栈，不跨栏算 Δ。
  注意它显示「18e 时代没有任何 RSSM 变体超过 GRU 34.50」——单独放会反噬，
  必须配最终栈的 GRU 控制一起讲。
- **被 bug 作废的**（§36 IoU 编码、§38.1-5 转置布局）、smoke/probe/只读诊断、截断未跑完的。

---

## 方法 run 与两个待补项（实时状态，2026-09-21 16:xx）

| 项 | run | 状态 |
|---|---|---|
| **本文方法** = FG-FULL − 2D instance − IGDR | `fgfull_N4_no2d_igdr_2x4_24e_seed0` | **训练中**，BEST 34.08 @ep5（还在爬升，远未到窗口期） |
| GRU 时序控制（补 U3） | `fgfull_N4_temporal_baseline_seed0` | **已死**：14:28 启动 → 14:46 停在 ep1，0 个 val epoch，无报错，疑似抢卡 |
| 完整 FG-FULL（B 栈上限参考） | `fgfull_N4_2x4_24e_seed0` | ✅ 已完成，40.69 @ep16 |

方法 run 只有 seed0 且未跑完 → 第 2 组（B 栈）现在**还不能定稿**。
另外 `docs/*.md` 完全没记录这两个 run（§46 停在 "not launched here"），跑完需补章节。
