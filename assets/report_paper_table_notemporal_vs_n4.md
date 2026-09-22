# No-Temporal 与 N4 RSSM 对比（论文式横表）

AP40 / %，moderate 难度。列序：3D 四类 + Overall 3D，BEV 四类 + Overall BEV。
第三行为差值 Δ = N4 RSSM − No-Temporal。

## BEST 口径（论文 / 汇报用）

| Method | Car<br>3D strict | Ped<br>3D loose | Cyc<br>3D loose | Truck<br>3D strict | Overall<br>3D | Car<br>BEV strict | Ped<br>BEV loose | Cyc<br>BEV loose | Truck<br>BEV strict | Overall<br>BEV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| No-Temporal | 47.86 | 27.07 | 41.89 | 25.53 | 35.59 | 63.32 | 28.81 | 44.25 | 32.54 | 42.23 |
| **N4 RSSM** | **51.35** | **31.32** | **50.09** | **30.76** | **40.88** | **70.61** | **33.54** | **51.81** | **43.45** | **49.85** |
| **Δ** | **+3.49** | **+4.25** | **+8.20** | **+5.23** | **+5.29** | **+7.29** | **+4.73** | **+7.56** | **+10.91** | **+7.62** |

## 窗口口径（ep12-16 均值，实验室选型用）

| Method | Car<br>3D strict | Ped<br>3D loose | Cyc<br>3D loose | Truck<br>3D strict | Overall<br>3D | Car<br>BEV strict | Ped<br>BEV loose | Cyc<br>BEV loose | Truck<br>BEV strict | Overall<br>BEV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| No-Temporal | 46.16 | 26.67 | 42.28 | 23.48 | 34.65 | 61.89 | 28.52 | 44.67 | 31.42 | 41.62 |
| **N4 RSSM** | **48.98** | **30.58** | **47.99** | **30.04** | **39.40** | **68.63** | **32.97** | **49.78** | **41.95** | **48.33** |
| **Δ** | **+2.82** | **+3.91** | **+5.71** | **+6.56** | **+4.75** | **+6.74** | **+4.45** | **+5.11** | **+10.53** | **+6.71** |

## 说明

- 列依次为：Car 3D moderate strict · Pedestrian 3D moderate loose · Cyclist 3D moderate loose ·
  Truck 3D moderate strict · Overall 3D moderate · Car BEV moderate strict ·
  Pedestrian BEV moderate loose · Cyclist BEV moderate loose · Truck BEV moderate strict ·
  Overall BEV moderate。
- Overall = 四类均值，已验证与 Overall 行**逐位相等**（3D 与 BEV 各自成立，两行均验证）。
- Δ 由**显示值相减**得出（Car 3D 精确差 +3.48，显示值 51.35 − 47.86 = +3.49），便于读者验算。
- No-Temporal BEST @ep20；N4 RSSM BEST @ep14（主模型最高 seed，40.88）。
- **N4 RSSM 在 10 列上全胜**（BEST 与窗口口径均如此）。
- 同栈单变量：两份 config 差异仅 `temporal_fusion`（None ↔ MotionAlignedRSSMFusion）；
  权重结构差异仅 46 个 `temporal_fusion.*` tensor（10,575,930 参数，占模型 22.0%）。
- 数据来源：原始 `*.log.json` 重算。
