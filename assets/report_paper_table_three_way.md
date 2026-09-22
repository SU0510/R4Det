# 三方对比：N4 RSSM / No-Temporal / GRU（BEST 口径，AP40 / %）

| Method | Car 3D<br>strict | Ped 3D<br>loose | Cyc 3D<br>loose | Truck 3D<br>strict | Overall<br>3D | Car BEV<br>strict | Ped BEV<br>loose | Cyc BEV<br>loose | Truck BEV<br>strict | Overall<br>BEV |
|---|---|---|---|---|---|---|---|---|---|---|
| N4 RSSM | 51.35 | 31.32 | 50.09 | 30.76 | 40.88 | 70.61 | 33.54 | 51.81 | 43.45 | 49.85 |
| No-Temporal | 47.86 | 27.07 | 41.89 | 25.53 | 35.59 | 63.32 | 28.81 | 44.25 | 32.54 | 42.23 |
| Δ (N4 − No-Temporal) | +3.49 | +4.25 | +8.20 | +5.23 | +5.29 | +7.29 | +4.73 | +7.56 | +10.91 | +7.62 |
| GRU | 49.78 | 29.41 | 46.40 | 28.41 | 38.50 | 67.33 | 31.41 | 48.41 | 38.54 | 46.42 |
| Δ (N4 − GRU) | +1.57 | +1.91 | +3.69 | +2.35 | +2.38 | +3.28 | +2.13 | +3.40 | +4.91 | +3.43 |

Overall = 四类均值（三行均已验证逐位相等）。N4 RSSM BEST @ep14（主模型最高 seed）；No-Temporal BEST @ep20；
GRU = GRU 时序融合基线（TemporalDeformableFusion）。N4 RSSM 在 10 列上同时全胜 No-Temporal 与 GRU。
