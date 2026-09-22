# No-Temporal vs 主模型（N4 Motion-Aligned RSSM）—— 6 项指标对照

**对照对**：`no_temporal_N4_2x4_24e_seed0`（单帧，`temporal_fusion=None`）
↔ 主模型 `run10_headv2_multiseed/seed_{0,1,2}`（N4 Motion-Aligned RSSM）。

**同栈单变量**：两份 config 差异只有 `temporal_fusion`（diff 已核验），其余为记账项。
三个 seed 共用同一份 config（seed0/seed2 仅差 `checkpoint_config interval`）。
**四类均值恰好等于 Overall 3D**（自校验通过）。

## 主模型 = N4 RSSM，三个 seed

| seed | BEST Overall 3D | @ep | BEST Overall BEV | 窗口 3D (ep12-16) | 窗口 BEV |
|---|---:|---:|---:|---:|---:|
| seed0 | 39.88 | ep16 | 47.70 | 38.39 | 46.52 |
| seed1 | 40.51 | ep15 | 48.13 | 39.20 | 46.85 |
| **seed2（最高）** | **40.88** | **ep14** | **49.85** | **39.40** | **48.33** |
| 均值 | **40.42 ± 0.51** | ep14–16 | 48.56 | 39.00 | 47.23 |

## 表 1：BEST 口径（论文 / 汇报用）

| 指标 | No-Temporal<br>(seed0) | 主模型 seed0<br>(种子对齐) | Δ | 主模型 seed2<br>(最高单点) | Δ |
|---|---:|---:|---:|---:|---:|
| **Overall 3D moderate** | 35.59 | **39.88** | **+4.29** | **40.88** | **+5.29** |
| **Overall BEV moderate** | 42.23 | **47.70** | **+5.47** | **49.85** | **+7.62** |
| Car 3D moderate strict | 47.86 | 49.98 | +2.11 | 51.35 | +3.49 |
| Truck 3D moderate strict | 25.53 | 30.43 | +4.90 | 30.76 | +5.23 |
| Pedestrian 3D moderate loose | 27.07 | 28.64 | +1.57 | 31.32 | +4.25 |
| Cyclist 3D moderate loose | 41.89 | 50.47 | +8.58 | 50.09 | +8.20 |

## 表 2：窗口口径（ep12-16 均值，实验室选型用）

| 指标 | No-Temporal<br>(seed0) | 主模型 seed0<br>(种子对齐) | Δ | 主模型 seed2<br>(最高单点) | Δ |
|---|---:|---:|---:|---:|---:|
| **Overall 3D moderate** | 34.65 | **38.39** | **+3.74** | **39.40** | **+4.75** |
| **Overall BEV moderate** | 41.62 | **46.52** | **+4.90** | **48.33** | **+6.71** |
| Car 3D moderate strict | 46.16 | 47.80 | +1.64 | 48.98 | +2.81 |
| Truck 3D moderate strict | 23.48 | 28.21 | +4.73 | 30.04 | +6.56 |
| Pedestrian 3D moderate loose | 26.67 | 28.92 | +2.24 | 30.58 | +3.91 |
| Cyclist 3D moderate loose | 42.28 | 48.63 | +6.34 | 47.99 | +5.71 |

## 说明

- **该引用哪个 Δ**：No-Temporal 只有 seed0 一次，所以
  - 只做"时序融合净收益"的科学对照 → 用 **seed0 ↔ seed0 = +4.29**（种子对齐，可辩护）；
  - 作为"我们的最终成绩"汇报 → 用 **主模型最高 40.88**，但 Δ 应标成 **+5.29**，
    并注明这是"主模型最优 seed vs 单帧 seed0"，不要悄悄混用。
- 六个指标取自同一帧日志记录，无跨栈拼接。
- Overall 3D = (Car strict + Truck strict + Ped loose + Cyc loose) / 4；Overall BEV 同口径。
- 四类两个 seed 下全部为正，无一类下降 → 不是牺牲某一类换 Overall。
- 数据来源：原始 `*.log.json` 重算（`tools/summarize_run.py` 同口径）。
