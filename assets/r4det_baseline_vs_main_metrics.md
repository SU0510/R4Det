# baseline_temporal vs 主模型：Overall 与四类 AP 对比（18 个数）

图：`assets/r4det_baseline_vs_main_metrics.svg` / `.png` / `.pdf`
（渲染脚本 `tools/draw_r4det_metric_comparison.py`）

- **baseline_temporal**：GRU 时序融合基线 `TemporalDeformableFusion`（N=2, 18e），BEST ep12。
- **主模型**：预训练 backbone + N=4 Motion-Aligned RSSM + head-v2（24e），**已存最高单点 seed_2 ep14**。
- 评估协议：KITTI 式 3D/BEV AP40，**moderate** 难度。两行均取**各自 BEST 单点**。

## 主表（6 项 × 2 模型 + 6 个差值）

| 指标 | 口径 | baseline_temporal | 主模型 | Δ（主 − 基线） |
|---|---|---:|---:|---:|
| Car | strict | 32.82 | 51.35 | **+18.53** |
| Truck | strict | 29.91 | 30.76 | +0.85 |
| Pedestrian | loose | 26.64 | 31.32 | +4.68 |
| Cyclist | loose | 48.63 | 50.09 | +1.46 |
| **Overall 3D moderate** | 混合 | **34.50** | **40.88** | **+6.38** |
| **Overall BEV moderate** | 混合 | **41.68** | **49.85** | **+8.17** |

## 主模型该用哪个数（四个候选，别混用）

| 口径 | Overall 3D moderate | 说明 |
|---|---:|---|
| **已存最高单点** | **40.88** | `run10_headv2_multiseed/seed_2` ep14，权重已落盘；**本表采用** |
| 三 seed BEST 均值 ± std | 40.42 ± 0.51 | 文档 §19.9 定的对外报告口径（seed_0 39.88 / seed_1 40.51 / seed_2 40.88） |
| 原 Run 10 单次峰值 | 40.60 | ep11；`checkpoint_interval=2` 导致**未落盘**，只存在于日志 |
| 原 Run 10 已存最优 | 39.65 | ep14；**不是主模型的最高值**，早期误用（见下方「修订记录」） |

⚠️ **口径偏差提示**：主模型列取的是**三个 seed 里的最高点**，baseline 只有单次 run 的单点，
所以这张表对主模型有轻微**向上偏置**。要去掉偏置请看「补充口径 B」的三 seed 均值（Overall 40.42，Δ +5.92）。

## 为什么恰好是这四类

`Overall_3D_moderate` 不是四类全 strict，也不是四类全 loose，而是混合口径
（`mmdet3d/core/evaluation/kitti_utils/eval.py:925-937`，与 `tools/summarize_run.py` 的 `CONSTITUENTS` 一致）：

```
Overall_3D = mean( Pedestrian_loose, Cyclist_loose, Car_strict, Truck_strict )
```

所以表里那四行本身就是 Overall 的构成项，前四行求平均 == Overall 3D 行：

- 基线：(32.82 + 29.91 + 26.64 + 48.63) / 4 = **34.50** ✓
- 主模型：(51.35 + 30.76 + 31.32 + 50.09) / 4 = **40.88** ✓

差值也自洽：ΔOverall = (+18.53 + 0.85 + 4.68 + 1.46) / 4 = **+6.38**。

## 结论要点

1. **Car strict 仍是最大来源**：+18.53，折算到 Overall 贡献 4.63 点，占 Overall 增量（+6.38）约 **73%**。
2. **Pedestrian 这次是正贡献**（+4.68，占约 18%）—— 上一版用 39.65 时它是 −0.91。换成最高单点后
   它转正，但注意这是 seed_2 单点的表现，三 seed 均值下为 +3.48（仍正）。
3. **Truck strict 基本原地**（+0.85），**Cyclist loose 小幅正**（+1.46）。
4. **BEV 口径提升（+8.17）大于 3D 口径（+6.38）**，说明框的中心/朝向改善比整体定位改善更多。

## 补充口径 A：四类全部按 loose 看

主表的四类混了 strict/loose，若统一按 loose（召回口径，IoU 0.25）看：

| 指标 | baseline_temporal | 主模型 | Δ |
|---|---:|---:|---:|
| Car loose | 52.18 | 75.77 | +23.59 |
| Truck loose | 43.02 | 52.01 | +8.99 |
| Pedestrian loose | 26.64 | 31.32 | +4.68 |
| Cyclist loose | 48.63 | 50.09 | +1.46 |

loose 口径下 Truck 的收益（+8.99）远大于 strict（+0.85）—— 召回上去了，定位没跟上。

## 补充口径 B：主模型取三 seed 复现均值（无偏置口径）

主模型的对外报告口径是三 seed BEST 的 mean ± std（`docs/training_runs_full.md` §19）：

| 指标 | baseline_temporal | 主模型 三 seed 均值 | Δ |
|---|---:|---:|---:|
| Car strict | 32.82 | 49.60 ± 1.95 | +16.78 |
| Truck strict | 29.91 | 31.99 ± 2.43 | +2.08 |
| Pedestrian loose | 26.64 | 30.12 ± 1.36 | +3.48 |
| Cyclist loose | 48.63 | 49.97 ± 0.57 | +1.34 |
| Overall 3D moderate | 34.50 | **40.42 ± 0.51** | **+5.92** |
| Overall BEV moderate | 41.68 | 48.56 ± 1.14 | +6.88 |

## 修订记录

- **v1（已废弃）**：主模型列误用 39.65（原 Run 10 的已存最优）→ Overall Δ 只有 +5.15，且 Pedestrian 显示为 −0.91。
  错在把「与 `r4det_n4_run10` 架构图同一个 run」当成选数理由；比较协议既然是「各自 BEST 单点」，
  主模型就该取已存最高的 40.88。
- **v2（当前）**：主模型列改用 seed_2 ep14 = 40.88。

## 数据来源与可复现性备注

- 主模型数值：`work_dirs/run10_headv2_multiseed/seed_2/*.log.json` 中 `mode=val`、epoch=14 记录
  （按每 epoch 最后一条 val 取值）。三 seed 均值同源：`seed_{0,1,2}`（BEST 分别 ep16/ep15/ep14）。
- baseline_temporal 数值取自 `docs/training_runs_full.md` §3 / §7 的「逐类别 BEST 对比 (loose/strict)」，
  该 run 的原始日志已删（`docs/R4Det_RSSM_full_report.md` §记录口径），**无法从日志重算**。
- 口径不对等提示：baseline_temporal 是**单次、非 deterministic、18e、2 卡**的结果，主模型是 **24e、3 卡**（batch 12）。
  两者不是同一 batch/schedule 下的严格对照，本表用来说明「主模型相对时序基线在 Overall 与各类别上的量级差」，
  不构成受控消融。受控对照请用 `no_temporal_N4_2x4_24e_seed0`（无时序）或 `fgfull_N4_2x4_24e_seed0`（FG-FULL 时序控制）。
- 附注：全库单点最高的 run 实际是 `rssm_kl0_N4_2x4_24e`（41.10 @ep20），但那是 **KL=0 消融**，不属于主模型。
