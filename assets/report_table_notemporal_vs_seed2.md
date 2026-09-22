# No-Temporal vs 主模型最高单点（N4 RSSM, seed2）

**对照**：`no_temporal_N4_2x4_24e_seed0`（单帧，`temporal_fusion=None`）
vs 主模型 `run10_headv2_multiseed/seed_2`（N4 Motion-Aligned RSSM，最高单点 40.88 @ep14）。

同栈单变量：两份 config 差异只有 `temporal_fusion`；权重的结构差异也仅 46 个
`temporal_fusion.*` tensor（10,575,930 参数，占模型 22.0%）。

## BEST 口径（论文 / 汇报用）　Δ = 主模型 seed2 − No-Temporal

| 指标 | No-Temporal | N4 RSSM | Δ |
|---|---:|---:|---:|
| **Overall 3D moderate** | 35.59 | **40.88** | **+5.29** |
| **Overall BEV moderate** | 42.23 | **49.85** | **+7.62** |
| Car 3D moderate strict | 47.86 | 51.35 | +3.49 |
| Truck 3D moderate strict | 25.53 | 30.76 | +5.23 |
| Pedestrian 3D moderate loose | 27.07 | 31.32 | +4.25 |
| Cyclist 3D moderate loose | 41.89 | 50.09 | +8.20 |

## 窗口口径（ep12-16 均值，实验室选型用）

| 指标 | No-Temporal | N4 RSSM | Δ |
|---|---:|---:|---:|
| **Overall 3D moderate** | 34.65 | **39.40** | **+4.75** |
| **Overall BEV moderate** | 41.62 | **48.33** | **+6.71** |
| Car 3D moderate strict | 46.16 | 48.98 | +2.82 |
| Truck 3D moderate strict | 23.48 | 30.04 | +6.56 |
| Pedestrian 3D moderate loose | 26.67 | 30.58 | +3.91 |
| Cyclist 3D moderate loose | 42.28 | 47.99 | +5.71 |

## 说明

- No-Temporal BEST 落在 **ep20**，主模型 seed2 BEST 落在 **ep14**。
- No-Temporal 只有 seed0 一次，所以本表是「**单帧 seed0 vs 主模型最优 seed**」。
  若要做种子严格对齐的对照，应改用 seed0 ↔ seed0（+4.29 3D / +5.47 BEV）。
- Δ 由**显示值相减**得出（Car 精确差为 +3.48，显示值 51.35 − 47.86 = +3.49），
  以保证读者能自行验算。
- 四类均值已验证等于 Overall 3D 行；Overall BEV 同口径。
- 数据来源：原始 `*.log.json` 重算。
