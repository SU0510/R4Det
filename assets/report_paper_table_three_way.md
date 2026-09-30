# 三方对比：N4 RSSM / No-Temporal / GRU（BEST 口径，AP40 / %）

| Method | Car 3D<br>strict | Ped 3D<br>loose | Cyc 3D<br>loose | Truck 3D<br>strict | Overall<br>3D | Car BEV<br>strict | Ped BEV<br>loose | Cyc BEV<br>loose | Truck BEV<br>strict | Overall<br>BEV |
|---|---|---|---|---|---|---|---|---|---|---|
| N4 RSSM | 51.35 | 31.32 | 50.09 | 30.76 | 40.88 | 70.61 | 33.54 | 51.81 | 43.45 | 49.85 |
| No-Temporal | 47.86 | 27.07 | 41.89 | 25.53 | 35.59 | 63.32 | 28.81 | 44.25 | 32.54 | 42.23 |
| Δ (N4 − No-Temporal) | +3.49 | +4.25 | +8.20 | +5.23 | +5.29 | +7.29 | +4.73 | +7.56 | +10.91 | +7.62 |
| GRU | 49.69 | 29.31 | 49.91 | 29.27 | 39.54 | 66.13 | 31.77 | 52.10 | 37.10 | 46.77 |
| Δ (N4 − GRU) | +1.66 | +2.01 | +0.18 | +1.49 | +1.34 | +4.48 | +1.77 | -0.29 | +6.35 | +3.08 |

Overall = 四类均值（按未取整值成立；逐位取整后最大偏差 0.005）。N4 RSSM：非 FG 栈 seed2 BEST @ep14（40.88）；
No-Temporal：非 FG 栈 BEST @ep20；GRU（TemporalDeformableFusion）：FG 栈 seed0 `fgfull_N4_temporal_baseline_seed0`，BEST @ep11（val 至 ep20/24）。

N4 RSSM 对 No-Temporal 10/10 全胜；对 GRU 9/10，唯一落后项为 Cyclist BEV loose（−0.29）。数据来源：原始 `*.log.json` 重算。
