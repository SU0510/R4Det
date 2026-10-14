# R4Det 训练完整记录（效果 / 配置 / 模块改动）

> 数据来源：各 `work_dirs/<run>/*.log.json` 的 `mode=val` 记录（每 epoch 取该 run 最后一次 val，避免中途重启污染）+ 各 run 目录下 `*.py` 配置快照 + git log（branches: `motionalignrssm` → `Nframerssm` → `detection_head_v2` → `radar_static_dynamic`）。
> 评估口径：KITTI 3D detection AP。主指标 **Overall 3D moderate**（`pts_bbox/KITTI/Overall_3D_moderate`）。`_loose` 为松评估 per-class。
> **Overall 口径说明**：`Overall_3D_moderate` = `(Car_strict + Truck_strict + Pedestrian_loose + Cyclist_loose) / 4`。即「Car/Truck 看 strict、Ped/Cyc 看 loose」的混合口径，不是四类全 strict（详见 `kitti_utils/eval.py` 的 Overall 聚合）。读任何 Overall 数字时请按此四项理解，否则会用错驱动变量。
> 训练时长按阶段：Run 1–8 为 18 epoch；Run 9 为 30 epoch；Run 10–15 为 24 epoch（`baseline_temporal` 日志从 ep6 起，ep1-5 无记录）。
> 配套诊断见 [rssm_diagnosis.md](rssm_diagnosis.md)。

> **统一记录口径（2026-09-24 定稿，后续所有 run 必须遵守）**
>
> 每次训练只以两个数值口径做横向判断，二者必须同时记录，不能互相替代：
>
> 1. **区间均值（comparison mean）**：用于公平对照和抗 epoch 抖动。默认取固定窗口
>    **ep12-16 五轮等权均值**；截断较早、窗口不完整的 run 必须写明实际跨度和样本数。
>    训练完整、平台已明确下移的 run 可同时附 `last-5` 作为补充，但主对照仍用 ep12-16。
> 2. **全轮峰值（all-epoch peak）**：取该 run **所有有 val 记录的 epoch 中的最大单点**，
>    与是否保存该 epoch 的权重无关。必须写 `指标值 + epoch`；它回答“上限能到多少”。
>
> 另单列第三项，仅用于 checkpoint 可用性和复现实验，不参与比较排名：
> **已存最高点（best saved）** = 磁盘上实际存在权重的 epoch 中 val 最高的那个。若
> 全轮峰值恰好落盘，则二者相同；若不落盘，则全轮峰值和 best saved 必须分别写清楚。
>
> 禁止再用含糊的 `BEST` 单独表示其中任一项。历史章节中的旧 `BEST` 默认指全轮峰值；
> 需要权重的语境一律写 `best saved`。`checkpoint_interval=2` 时奇数 epoch 的峰值仍按
> 口径 2 记录，只是注明“无对应权重”。
>
> **区间均值必须写出 epoch 跨度**。24e 主线的标准主窗口是 ep12-16；18e/30e/12e 等不同
> 长度的 run 若 ep12-16 不存在或不是共同平台，使用实际共同窗口并明确标注，不能把不同
> epoch 区间算出的两个“mean”直接视作同口径。
>
> **记录层级**：上述两个口径只是每节的**必填摘要**，不是完整记录的全部。正式训练仍必须
> 按实际情况保留：配置与唯一变量、运行事实（起止/中断/落盘）、完整逐 epoch 曲线（至少主
> Overall，正式对照建议含 BEV 与四类构成项）、对照所用窗口的逐项均值、关键诊断/归因、
> 结论和下一步。逐 epoch 表、逐类别拆解、动力学或误差诊断等“详细完整指标”不得因为已有
> 摘要而被省略；摘要只是保证不同 run 至少有一组可比数字。

---

## 0. 早期五次 run 一览（Run 1–5，仅时序融合模块对比）

| # | run 目录 | 时序融合模块 | BEST 3D_mod @ep | 起点 | 一句话定位 |
|---|---|---|---:|---|---|
| 1 | `baseline_temporal` | `TemporalDeformableFusion`（GRU+deform） | **34.50** @ep12 | resume latest | 最强 baseline，GRU 直接吃上一帧 BEV |
| 2 | `baseline_rssm` | `BEVRSSMTemporalFusion`（RSSM，无对齐） | 33.54 @ep17 | resume ep4 | RSSM 首次落地，KL 被压死 |
| 3 | `motion_align_rssm` (v1) | `MotionAlignedRSSMFusion` | 30.89 @ep10 | from scratch | 对齐层 + RSSM，KL 仍压死，踩坑跑 |
| 4 | `motion_align_rssm2` (v2) | `MotionAlignedRSSMFusion` | 34.01 @ep14 | resume v1 latest | 修 KL 超参，追平 baseline |
| 5 | `motion_align_rssm3` (v3) | `MotionAlignedRSSMFusion` | 33.77 @ep11 | from scratch | 同 v2 配置 + logstd 防爆炸，从头训 |

> 后续 Run 6–15 见第 7–15 节；本节只覆盖最初五次时序融合模块对比。

**BEST 3D_moderate 排名**：`baseline_temporal` 34.50 > `motion_align_rssm2` 34.01 > `motion_align_rssm3` 33.77 > `baseline_rssm` 33.54 > `motion_align_rssm(v1)` 30.89。

---

## 1. 主指标汇总：BEST vs LAST

| run | BEST @ep | 3D_easy | 3D_mod | 3D_hard | BEV_mod | LAST @ep | 3D_easy | 3D_mod | 3D_hard | BEV_mod |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| baseline_temporal | ep12 | 35.77 | **34.50** | 33.27 | 41.68 | ep16 | 35.92 | 34.46 | 33.24 | 40.91 |
| baseline_rssm | ep17 | 35.00 | 33.54 | 32.37 | 42.48 | ep18 | 34.46 | 33.37 | 32.28 | 42.68 |
| motion_align_rssm (v1) | ep10 | 31.58 | 30.89 | 29.86 | 40.24 | ep18 | 30.98 | 29.96 | 28.91 | 38.67 |
| motion_align_rssm2 | ep14 | 35.26 | 34.01 | 32.74 | 42.25 | ep18 | 34.16 | 33.21 | 32.00 | 41.10 |
| motion_align_rssm3 | ep11 | 35.32 | 33.77 | 32.46 | 42.07 | ep18 | 34.92 | 32.93 | 31.63 | 40.52 |

## 2. 逐 epoch 曲线：Overall 3D_moderate

| epoch | baseline_rssm | baseline_temporal | v1 | v2 | v3 |
|---|---:|---:|---:|---:|---:|
| 1  |  9.36 | — |  7.38 |  7.62 |  9.33 |
| 2  | 16.16 | — | 12.74 | 14.16 | 13.89 |
| 3  | 16.76 | — | 17.68 | 19.51 | 15.82 |
| 4  | 20.12 | — | 19.61 | 20.90 | 21.09 |
| 5  | 23.68 | — | 22.14 | 25.79 | 22.97 |
| 6  | 26.66 | 29.82 | 28.37 | 28.73 | 24.73 |
| 7  | 28.81 | 28.66 | 24.36 | 29.72 | 27.56 |
| 8  | 29.09 | 32.31 | 27.32 | 25.92 | 31.40 |
| 9  | 31.01 | 31.28 | 27.69 | 28.56 | 30.48 |
| 10 | 31.90 | 31.75 | **30.89** | 30.28 | 33.01 |
| 11 | 29.72 | 31.04 | 29.78 | 31.55 | **33.77** |
| 12 | 33.00 | **34.50** | 30.64 | 31.64 | 32.94 |
| 13 | 31.68 | 31.00 | 28.97 | 30.53 | 30.97 |
| 14 | 31.82 | 34.47 | 29.74 | **34.01** | 33.02 |
| 15 | 33.19 | 33.13 | 29.85 | 31.85 | 32.62 |
| 16 | 32.72 | 34.46 | 29.00 | 32.94 | 32.85 |
| 17 | **33.54** | — | 29.14 | 32.87 | 31.90 |
| 18 | 33.37 | — | 29.96 | 33.21 | 32.93 |

加粗为各 run BEST epoch。

## 3. 各类别 3D_moderate_loose @ BEST epoch

| run | BEST @ep | Car | Cyclist | Pedestrian | Truck | Overall 3D_mod | Overall BEV_mod |
|---|---|---:|---:|---:|---:|---:|---:|
| baseline_temporal | ep12 | 52.18 | 48.63 | **26.64** | 43.02 | **34.50** | 41.68 |
| baseline_rssm | ep17 | 50.27 | **50.37** | 24.48 | **47.33** | 33.54 | **42.48** |
| motion_align_rssm (v1) | ep10 | 54.41 | 47.50 | 19.87 | 40.65 | 30.89 | 40.24 |
| motion_align_rssm2 | ep14 | 52.57 | 49.34 | 21.13 | 42.90 | 34.01 | 42.25 |
| motion_align_rssm3 | ep11 | **56.03** | 49.02 | 23.35 | 43.65 | 33.77 | 42.07 |

- v3 的 Car 类全场最高（56.03），但 Pedestrian（23.35）仍低于两个 baseline。
- v1 Pedestrian 只有 19.87，明显拖后腿——KL 压死导致 posterior 没学好。

---

## 4. 网络配置对比

骨干、检测头、训练 schedule 五次 run 完全相同，**唯一差异是 `temporal_fusion` 模块及其超参**。下表只列差异项。

| 配置项 | baseline_temporal | baseline_rssm | motion_align v1 | v2 | v3 |
|---|---|---|---|---|---|
| `temporal_fusion.type` | `TemporalDeformableFusion` | `BEVRSSMTemporalFusion` | `MotionAlignedRSSMFusion` | 同 v1 | 同 v1 |
| `latent_dim` | — | 256 | 256 | 256 | 256 |
| `hidden_dim` | — | 64 | 64 | 64 | 64 |
| `action_dim` | — | 2 | 2 | 2 | 2 |
| `kl_scale` | — | **0.1** | **0.1** | **1.0** | **1.0** |
| `free_nats` | — | **0.0** | **0.0** | **1.0** | **1.0** |
| `min_std` | — | 0.1 | 0.1 | 0.1 | 0.1 |
| `init_std` | — | 0.2 | 0.2 | 0.2 | 0.2 |
| `align_kernel_size` | — | — | 3 | 3 | 3 |
| `align_deform_groups` | — | — | 1 | 1 | 1 |
| `align_z_state` | — | — | True | True | True |
| `deform_groups`(GRU) | 1 | — | — | — | — |
| `gate_kernel_size` | 1 | — | — | — | — |
| KL warmup `end_epoch` | — | 3 | 3 | **5** | 5 |
| KL warmup `end_value` | — | 0.1 | 0.1 | **1.0** | 1.0 |
| `custom_imports` KL hook | 无 | 有 | 有 | 有 | 有 |
| `resume_from` | latest.pth | epoch_4.pth | **None**（从头） | **v1 latest.pth**（续训） | **None**（从头） |
| `checkpoint_config.interval` | 1 | 2 | 2 | 2 | 2 |

**注**：`baseline_temporal` 与 RSSM 系的配置 diff 里还有一批 data pipeline 字段差异（`depth2img`/`gt_depths`/`LoadAnnotations3D` 等），是早期配置分支遗留，与本次时序模块对比无关，不展开。

### 4.1 三个融合模块的结构差异

| 模块 | 历史状态来源 | transition | 输出结构 | 输出初始化 |
|---|---|---|---|---|
| `TemporalDeformableFusion` | `feat_prev`（上一帧真实 BEV 特征） | GRU 门控（z/r/gate），deform 对齐 feat_prev | `output_layer(h_t)`，3×3 Conv | **Xavier**，主动改造 |
| `BEVRSSMTemporalFusion` | `h_state`/`z_state`（自维护隐+随机状态） | `ConvGRUCell(h, z, action)` | `output_proj(z_t) + feat`（残差） | 见下注 |
| `MotionAlignedRSSMFusion` | 同上，但 h/z 先 deform 对齐到当前帧 | 同上 | 同上 | 同上 |

> **输出初始化注**：当前仓库代码（v3 之后）`output_proj` 已是 Xavier 初始化；但 **v1/v2/v3 三次实际训练时** `output_proj` 都是**零初始化**（`constant_(weight,0)`/`constant_(bias,0)`），残差 `+ feat` 保证初始输出≈feat，z_t 起步零贡献。Xavier 改动是 v3 跑完诊断后才上的，**尚未在任何一次已记录的 run 中验证**（见第 6 节"未验证改动"）。

### 4.2 KL hook（`KLScaleSchedulerHook`）

只在 RSSM 系三 run 启用（`baseline_rssm` + v1/v2/v3）。`kl_scale` 从 `start_value` 线性升到 `end_value`，覆盖 `[start_epoch, end_epoch]`。`kl_scale=0` 时 prior 无 KL 梯度，让 posterior/检测器先稳住。

- baseline_rssm / v1：`0→0.1` over ep0-3（KL 几乎没启用）
- v2 / v3：`0→1.0` over ep0-5（KL 真正生效）

---

## 5. 每次分别改了哪些模块（git log + config diff）

### Run 1 — `baseline_temporal`（参考 baseline，不在 motionalignrssm 分支演进线上）

- 模块：`TemporalDeformableFusion`（`temporal_r4det_fusion.py`）
- 无 KL hook、无 RSSM。GRU 直接吃 deform 对齐后的 `feat_prev`。
- resume 自身 latest（中途续训）。

### Run 2 — `baseline_rssm`（commit `256f0c8` rssm train ready + `2168bed` rssm v1 update）

模块新增/改动：
- **新增** `mmdet3d/models/fusion_layers/rssm_fusion.py`（`ConvGRUCell` + `BEVRSSMTemporalFusion`）
- **新增** `mmdet3d/core/hook/kl_scale_scheduler.py`（`KLScaleSchedulerHook`）
- 改 `mmdet3d/models/detectors/R4Det.py`：接入 `temporal_fusion` 返回的 6 元组（output/recon/kl/h/z/stats），加 KL/recon loss
- 改 `fusion_layers/__init__.py`：注册新模块
- 新增 config `TJ4D-R4Det_baseline_rssm_det3d_2x4_12e.py`
- IoU 评估重构：`core/evaluation/kitti_utils/eval.py` + `rotate_iou.py`

配置关键值：`kl_scale=0.1, free_nats=0.0`（KL 被压死），warmup `0→0.1` over ep0-3。resume ep4。

### Run 3 — `motion_align_rssm` v1（commit `57da70f` rssm v2 with motionalign）

模块改动：
- **新增** `MotionAlignedRSSMFusion`（继承 `BEVRSSMTemporalFusion`，rssm_fusion.py +218 行）：在 ConvGRU 之前用 `ModulatedDeformConv2d` 把 `h_{t-1}`/`z_{t-1}` warp 到当前帧，offset/mask 由 `concat(feat_cur, state)` 预测；align 层零初始化（identity 起步）
- 新增 `test_rssm_fusion.py`
- 新增 config `TJ4D-R4Det_motion_align_rssm_det3d_2x4_12e.py`

配置：**与 baseline_rssm 完全相同的 KL 超参**（`kl_scale=0.1, free_nats=0.0`），只多了 align 三参数。`resume_from=None` 从头训。

结果：BEST 30.89，比 baseline_rssm 还低 ~2.6 点。**对齐层没救回被压死的 KL**，且零初始化残差输出起步慢。

### Run 4 — `motion_align_rssm2` v2（commit `c10307f` modify klscale freenats clamp and klwarmup）

改动（**纯 config，模块代码几乎没动**——rssm_fusion.py 只改了 4 行）：
```
kl_scale:        0.1  -> 1.0
free_nats:       0.0  -> 1.0
KL warmup end_epoch: 3 -> 5
KL warmup end_value: 0.1 -> 1.0
resume_from:     None -> v1 latest.pth   (续训 v1)
```
结果：BEST 30.89 → **34.01**（**+3.12**），追平两个 baseline。

解读：v1 把 KL 压得太狠（`free_nats=0` + `kl_scale=0.1`），posterior 没学到东西；放开 KL 后收益巨大。但 +3.12 里有**续训本身**的贡献（v2 续的是已训 18 epoch 的 v1），不能全归给超参。

### Run 5 — `motion_align_rssm3` v3（commit `f39dedd` logstd 上界 + `bc06162` IoU on CUDA + `74283ff` tqdm）

模块改动：
- `f39dedd` **`rssm_fusion.py` logstd 上界 `max=3.0 → max=0.0`**（std ≤ 1.0 防方差爆炸），改在 `sample()` 和 `kl_loss()` 两处
- `bc06162` IoU 计算搬到 CUDA（`rotate_iou.py` + `iou3d_utils.py`）——只影响评估速度，不影响训练指标口径
- `74283ff` eval 工具加 tqdm——不影响训练

配置：**与 v2 完全相同**（仅路径不同），`resume_from=None` 从头训（不续 v2）。

结果：BEST 34.01 → **33.77**（**−0.24**），基本持平略降。Car 类反而最高（56.03），但 Pedestrian/Cyclist/Truck 均未超 baseline。logstd 防爆炸既没帮上也没拖后腿。

### 改动汇总矩阵

| 改动 | baseline_rssm | v1 | v2 | v3 |
|---|:---:|:---:|:---:|:---:|
| RSSM 模块（encoder/GRU/prior/posterior/decoder） | ✓ 新增 | 继承 | 继承 | 继承 |
| KL hook `KLScaleSchedulerHook` | ✓ | ✓ | ✓ | ✓ |
| Motion-align deform 对齐层 | — | ✓ 新增 | 继承 | 继承 |
| `kl_scale 0.1→1.0` | — | — | ✓ | ✓ |
| `free_nats 0.0→1.0` | — | — | ✓ | ✓ |
| KL warmup 拉长 `3→5` / `0.1→1.0` | — | — | ✓ | ✓ |
| logstd 上界 `3.0→0.0` | — | — | — | ✓ |
| IoU on CUDA / tqdm | — | — | — | ✓（eval only） |

---

## 6. 诊断结论与未验证改动

详见 [rssm_diagnosis.md](rssm_diagnosis.md)。要点：

1. **v1 是踩坑跑**：KL/free_nats 太小，BEST 30.89，比两个 baseline 低 ~3 点。
2. **v2 修好 KL 超参**，BEST 34.01，与 baseline 同档；+3.12 来自超参修正 + 续训，不是 motion-align 本身功劳。
3. **v3 相比 v2 无提升**（33.77 vs 34.01，−0.24）；logstd 防爆炸中性。
4. 三次 motion_align **都没超过最强的 `baseline_temporal`（34.50）**。
5. RSSM 打不过 GRU 的真因：**零初始化残差输出导致起步慢**（18 epoch 短训练吃亏）+ **velocity 全程是零且数据集无 ego 位姿**。KL 死反而是中性的（推理用 posterior，prior 不在路径上）。

### v3 之后已改、但**尚未训练验证**的两处（当前仓库代码状态）

- `output_proj` 零初始化 → **Xavier**（`rssm_fusion.py:255`）：让 z_t 从第一步就参与，针对起步慢。
- velocity 分支用 `use_action` 开关移出网络（`action_dim=0`）：卫生清理，代码全保留，不指望涨点。

当前 `configs/r4det/TJ4D-R4Det_motion_align_rssm_det3d_2x4_12e.py` 已是这两改动后的状态（`action_dim=0`、Xavier init）。**下一次训练才会验证**——若 AP 追上/超过 34.5 则确认起步慢是主因。

---

## 附：数据可信度

之前一次未核对的分析里 `baseline_rssm` 出现过 ep15=35.18 等数字，是脚本把多次中途重启的 `*.log.json` 叠加读取的误读。本文件所有数字均按「每个 epoch 取该 run 最后一次 val」重新核对，`baseline_rssm` 真实 BEST 为 ep17=33.54。



---

## 7. N=3 和 N=4 RSSM 训练（多帧消融 + hidden_dim 消融）

> 分支: Nframerssm。基础模块同 motion_align_rssm v3（MotionAlignedRSSMFusion，action_dim=0，kl_scale=1.0，free_nats=1.0）。
> 所有 run 从头训练 18 epoch，samples_per_gpu=4，lr=0.0002。

### Run 6 -- N=3 RSSM (seq_len=3, hidden_dim=64)

- config: configs/r4det/TJ4D-R4Det_motion_align_rssm_det3d_N3_2x3_12e.py
- work_dir: work_dirs/rssm_N3_2x3_12e
- 改动: 仅 seq_len=2->3，其他超参与 v3 完全相同
- BEST: ep14 34.27
- LAST: ep18 33.27

与 N=2 v3 对比:
- Overall: +0.50 (33.77->34.27)
- Truck loose: +3.83 (43.65->47.48)
- Pedestrian loose: -0.47 (23.35->22.88)
- Car loose: -0.33 (56.03->55.70), Car strict: +0.81 (33.20->34.01)

关键观察:
- ep1 起步极慢 (2.51 vs v3 9.33)，但从 ep5 反超 v3
- ep10 出现严重暴跌 (28.05)，可能 KL loss 或数据问题
- 收敛到 ep14 才 peak (v3 在 ep11)
- 总体 +0.50，第3帧贡献有限

### Run 7 -- N=4 RSSM (seq_len=4, hidden_dim=128)

- config: configs/r4det/TJ4D-R4Det_motion_align_rssm_det3d_N4_2x4_12e.py
- work_dir: work_dirs/rssm_N4_2x4_12e
- 改动: seq_len=3->4 + hidden_dim=64->128（两变量同时改，无法隔离）
- BEST: ep17 34.05
- LAST: ep18 32.07（暴跌，可能过拟合）

与 N=3 对比:
- Overall: -0.22 (34.27->34.05)
- Car strict: +4.48 (34.01->38.49) 🔥 历史最高！
- Car loose: -2.03 (55.70->53.67)
- Ped loose: +1.59 (22.88->24.46) RSSM首次行人提升
- Trk loose: +1.49 (47.48->48.97)
- Trk strict: -7.08 (31.72->24.64) 崩塌
- Cyc: 基本持平略降

关键观察:
- Car strict 38.49 是所有 run 最高值（之前最高 v2: 37.79）
- Car strict/loose 比值 61% -> 72%，3D框质量显著改善
- 训练比 N=3 更稳定（无 ep10 暴跌），但收敛更慢（ep17 才 peak）
- ep18 暴跌至 32.07，暗示 hidden_dim=128 需要更长训练
- WARNING: 两个变量同时改变，Car strict 提升可能来自 hidden_dim=128，Car loose 下降可能来自 N=4 平滑

### Run 8 -- N=3 RSSM (seq_len=3, hidden_dim=128)

- config: configs/r4det/TJ4D-R4Det_motion_align_rssm_det3d_N3_hdim128_2x4_12e.py
- work_dir: work_dirs/rssm_N3_hdim128_2x4_12e
- 改动: 仅 hidden_dim=64->128，其他超参与 Run 6 (N=3) 完全相同
- 目的: 隔离 hidden_dim 对 Car strict 的贡献（Run 7 中两变量同时变化无法判断）
- BEST: ep17 32.87
- LAST: ep18 30.98（暴跌）

与 N=3 hdim=64 (Run 6) 对比:
- Overall: **-1.40** (34.27->32.87) ❌ 反效果！
- Car strict: -1.43 (34.01->32.58)
- Car loose: -4.57 (55.70->51.13)
- Cyc loose: -2.83 (48.48->45.65)
- Ped loose: +1.18 (22.88->24.07)
- Trk loose: +0.68 (47.48->48.16)
- Trk strict: -2.55 (31.72->29.17)

关键观察:
- hidden_dim=128 在 N=3 下几乎全面劣化！仅 Ped loose 微涨
- 训练起点更高 (ep1=6.28 vs ep1=2.51)，但收敛上限更低
- 对象 N=4+hdim=128 反而有正向表现，说明 hdim=128 的收益依赖更多帧数
- 假设: hdim=128 增大了模型容量，N=3 时序多样性不足以利用额外容量，导致过拟合
- 结论: **N=4 是 hdim=128 的"解锁条件"**，N=3 下 hdim=64 更优

### N=3 和 N=4 epoch-by-epoch

| epoch | N=3 hdim128 | N=4 (hdim=128) | N=3 (hdim=64) | N=4 (hdim=64) | N=2 v3 | baseline_temporal |
|---|---:|---:|---:|---:|---:|---:|
| 1  |  6.28 |  3.07 |  2.51 |  5.21 |  9.33 | -- |
| 2  | 14.12 | 13.87 | 11.52 | 12.82 | 13.89 | -- |
| 3  | 19.07 | 16.51 | 17.33 | 19.67 | 15.82 | -- |
| 4  | 22.57 | 19.77 | 18.62 | 22.71 | 21.09 | -- |
| 5  | 24.30 | 25.52 | 25.96 | 24.85 | 22.97 | -- |
| 6  | 27.50 | 28.34 | 28.21 | 28.07 | 24.73 | 29.82 |
| 7  | 29.89 | 30.43 | 30.74 | 30.37 | 27.56 | 28.66 |
| 8  | 27.21 | 28.94 | 29.47 | 28.16 | 31.40 | 32.31 |
| 9  | 29.94 | 31.70 | 31.11 | 30.43 | 30.48 | 31.28 |
| 10 | 29.12 | 30.36 | 28.05 | 29.22 | 33.01 | 31.75 |
| 11 | 31.04 | 32.13 | 31.58 | 31.62 | 33.77 | 31.04 |
| 12 | 31.55 | 33.55 | 32.82 | **33.00** | 32.94 | 34.50 |
| 13 | 31.30 | 32.91 | 32.46 | 31.27 | 30.97 | 31.00 |
| 14 | 32.02 | 32.97 | **34.27** | 32.45 | 33.02 | 34.47 |
| 15 | 32.15 | 32.58 | 32.73 | 30.85 | 32.62 | 33.13 |
| 16 | 31.05 | 32.84 | 31.94 | 32.02 | 32.85 | 34.46 |
| 17 | **32.87** | **34.05** | 32.60 | 31.67 | 31.90 | -- |
| 18 | 30.98 | 32.07 | 33.27 | 32.46 | 32.93 | -- |

### 逐类别 BEST 对比 (loose / strict)

| Run | BEST ep | Car_l | Car_s | Cyc_l | Cyc_s | Ped_l | Ped_s | Trk_l | Trk_s | Ovl_3D |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_temporal (N=2) | 12 | 52.18 | 32.82 | 48.63 | 22.52 | **26.64** | 0.04 | 43.02 | 29.91 | **34.50** |
| N=3 (hdim=64) | 14 | **55.70** | 34.01 | 48.48 | 21.73 | 22.88 | 0.05 | 47.48 | **31.72** | 34.27 |
| N=4 (hdim=128) | 17 | 53.67 | **38.49** | 48.59 | 20.76 | 24.46 | 0.02 | **48.97** | 24.64 | 34.05 |
| N=3 (hdim=128) | 17 | 51.13 | 32.58 | 45.65 | 19.67 | 24.07 | 0.02 | 48.16 | 29.17 | 32.87 |
| N=4 (hdim=64) | 12 | 51.23 | 33.77 | 54.01 | 24.36 | 25.60 | 2.52 | 48.59 | 21.81 | 33.00 |
| N=2 v3 (hdim=64) | 11 | 56.03 | 33.20 | **49.02** | **22.49** | 23.35 | 0.01 | 43.65 | 29.51 | 33.77 |
| N=2 v2 (hdim=64) | 14 | 52.57 | 37.79 | 49.34 | 21.26 | 21.13 | 0.01 | 42.90 | 27.79 | 34.01 |

---

## 8. N帧消融总结

### 总体排名（3D moderate BEST）

| # | Run | N | hdim | BEST ep | 3D_mod |
|---|---:|---|---:|---:|
| 1 | baseline_temporal (GRU) | 2 | -- | 12 | **34.50** |
| 2 | N=3 RSSM | 3 | 64 | 14 | 34.27 |
| 3 | N=4 RSSM | 4 | 128 | 17 | 34.05 |
| 4 | v2 RSSM | 2 | 64 | 14 | 34.01 |
| 5 | v3 RSSM | 2 | 64 | 11 | 33.77 |
| 6 | baseline_rssm (no align) | 2 | 64 | 17 | 33.54 |
| 7 | N=3 RSSM (hdim=128) | 3 | 128 | 17 | 32.87 |
| 8 | N=4 RSSM (hdim=64) | 4 | 64 | 12 | 33.00 |
| 9 | v1 RSSM (KL broken) | 2 | 64 | 10 | 30.89 |

### 核心发现

1. N=2->3 有 +0.50 微弱但正向提升，主要来自 Truck (+3.83)
2. N=3->4 总体零收益甚至微负 (34.27->34.05)，边际帧数收益已耗尽
3. hidden_dim=128 在 N=3 下反效果 (-1.40)，但在 N=4 下有正向作用 (Car strict 历史最高 38.49)
4. **关键发现: hdim=128 需要 N=4 解锁** — N=3 时序多样性不足以利用额外容量，导致过拟合/欠收敛
5. 所有RSSM变体都没超过简单的N=2 GRU baseline (34.50) — 差距0.23-1.63，统计噪声级
6. Pedestrian 3D strict 对所有模型都是坏的 (0.01-0.05) — 瓶颈在检测头不在时序融合
7. 18 epoch 对更大模型 (hdim=128) 可能不够，ep17才peak且ep18暴跌
8. hdim=128 对 Ped loose 有微弱正向 (+1.18~+1.59 不等)，可能有助行人检测
9. N=4 + hdim=64 = 33.00 < N=4 + hdim=128 (34.05)：hdim=64 在 N=4 下也不够强，确认「N=4 配 hdim=128、N=3 配 hdim=64」的容量-帧数匹配关系

### 待做消融

- [x] N=3 + hidden_dim=128: 已训练完成，hdim=128 在 N=3 下全面劣化，确认需要 N=4 配合
- [x] N=4 + hidden_dim=64: 已训练完成，BEST 33.00 @ep12，比 N=4(hdim=128) 34.05 和 N=3(hdim=64) 34.27 都低，确认 N=4 必须配 hdim=128
- [ ] extended training (24-30ep): 验证 hdim=128 是否需要更长收敛
- [ ] velocity 分支注入: 所有当前 RSSM 训练都使用 action_dim=0，可尝试注入 velocity 信号
- [ ] deformable alignment 消融: 当前所有 RSSM 变体都使用 MotionAlignedRSSMFusion（含 deformable align），对比纯 flow warp

---

## 9. N=4 RSSM + Pretrained Backbone (30e)

> Branch: Nframerssm. Based on Run 7 config, only added load_from=pretrained_tj4d.pth.
> Pretrained weights from TJ4D author (HuggingFace: Hoeyy/R4Det), covers image backbone, radar backbone, depth net, BEV encoder.
> Auxiliary modules from pretraining (proposal_layer_former/latter, rangeview_foreground) not in detection forward path.
> seq_len=4, hidden_dim=128, samples_per_gpu=4, lr=1.5e-4, CosineAnnealing, 30 epochs。
> 该 run 在 4 卡上训练，**没有梯度累积**，有效 batch 为 4×4=16。

### 9.0 N=4 30e 无 pretrain 对照组（Run 9 的对照基线）

> 第 9.3 节与第 16 节都引用了这条 run 的数值（BEST `34.71`、last-5 `33.67`），但此前没有独立
> 章节。本次审计按磁盘日志补账，它是正式训练而非 smoke。

- 工作目录：`/data/lurui/work_dirs/rssm_N4_2x4_30e`。
- 配置：`configs/r4det/TJ4D-R4Det_motion_align_rssm_det3d_N4_2x4_30e.py`
  （`load_from=None`，其余 30e / seq_len=4 / hidden_dim=128 / samples_per_gpu=4 /
  lr=1.5e-4 / CosineAnnealing 与 Run 9 相同；同为 4 卡、无梯度累积，有效 batch 16）。
- 2026-08-11 02:49 UTC 有一次启动后立即中止（`20260811_024942.log` 只有环境与配置，无训练 iter）；
  实际训练从 02:52:01 UTC 开始（`20260811_025201.log(.json)`），ep30 val 于 20:54:21 UTC 写完。
- 30 个 epoch 全部有 val 行；`checkpoint_config=dict(interval=2)`，磁盘有 `epoch_22.pth`、
  `epoch_30.pth`，`latest.pth -> epoch_30.pth`；全轮峰值 ep22 的权重已落盘。

完整 val 曲线（`pts_bbox/KITTI/*`）：

| epoch | Overall 3D | Overall BEV | Car strict | Cyclist loose | Ped loose | Truck strict |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5.4317 | 8.3427 | 10.2460 | 10.7706 | 0.1991 | 0.5111 |
| 2 | 12.8657 | 16.7716 | 21.1037 | 25.2004 | 3.1672 | 1.9914 |
| 3 | 15.9292 | 20.9450 | 20.6740 | 26.6351 | 8.7386 | 7.6692 |
| 4 | 20.6816 | 26.1808 | 24.4629 | 42.7281 | 11.0507 | 4.4847 |
| 5 | 21.0176 | 28.0571 | 19.0098 | 35.6480 | 16.3832 | 13.0295 |
| 6 | 26.8640 | 33.0792 | 29.0752 | 39.8438 | 18.5251 | 20.0118 |
| 7 | 29.0485 | 35.7267 | 29.5329 | 45.2979 | 20.6814 | 20.6819 |
| 8 | 27.7829 | 35.0507 | 25.6262 | 47.3229 | 22.6327 | 15.5500 |
| 9 | 31.1969 | 38.8825 | 33.1912 | 47.7149 | 23.3121 | 20.5693 |
| 10 | 31.9248 | 39.4064 | 33.7537 | 45.0553 | 24.0706 | 24.8195 |
| 11 | 29.4525 | 36.1264 | 33.9373 | 37.7132 | 21.5164 | 24.6429 |
| 12 | 30.4216 | 36.6946 | 27.4097 | 44.0986 | 25.6003 | 24.5778 |
| 13 | 31.1247 | 39.3288 | 31.1769 | 44.9317 | 22.2686 | 26.1217 |
| 14 | 31.9362 | 39.3138 | 29.8750 | 42.4488 | 26.4193 | 29.0018 |
| 15 | 32.4261 | 41.6730 | 30.7512 | 45.1131 | 27.3925 | 26.4475 |
| 16 | 33.3552 | 41.9668 | 34.1614 | 45.5273 | 26.0722 | 27.6596 |
| 17 | 30.7963 | 39.3221 | 31.1591 | 41.5369 | 24.0539 | 26.4351 |
| 18 | 32.1473 | 41.1823 | 34.2524 | 38.9210 | 28.1125 | 27.3033 |
| 19 | 32.3735 | 39.9132 | 32.3466 | 41.8781 | 26.5473 | 28.7219 |
| 20 | 34.3669 | 43.4006 | 34.2292 | 46.8998 | 27.3312 | 29.0070 |
| 21 | 33.5068 | 41.9425 | 33.0988 | 44.4770 | 27.6070 | 28.8443 |
| 22 | **34.7083** | 43.8497 | 36.3351 | 46.2762 | 27.4171 | 28.8048 |
| 23 | 34.3833 | 43.8906 | 31.9399 | 46.2110 | 30.7037 | 28.6788 |
| 24 | 33.0742 | 42.1690 | 36.0561 | 44.6847 | 25.9899 | 25.5663 |
| 25 | 34.3330 | 43.2907 | 31.4963 | 47.3707 | 26.9394 | 31.5259 |
| 26 | 33.8185 | 43.7186 | 32.1002 | 47.4625 | 28.2404 | 27.4708 |
| 27 | 33.9518 | 43.5700 | 30.9198 | 47.8426 | 27.6156 | 29.4290 |
| 28 | 33.4243 | 42.4675 | 33.5324 | 44.7486 | 27.2772 | 28.1392 |
| 29 | 33.1884 | 43.0669 | 31.3980 | 45.1728 | 28.1170 | 28.0657 |
| 30 | 33.9915 | 43.0370 | 34.7521 | 46.2863 | 27.3774 | 27.5504 |

全轮峰值与平台：

| 口径 | epoch | Overall 3D | Overall BEV | Car strict | Cyclist loose | Ped loose | Truck strict |
|---|---:|---:|---:|---:|---:|---:|---:|
| 全轮峰值 | 22 | **34.7083** | 43.8497 | 36.3351 | 46.2762 | 27.4171 | 28.8048 |
| LAST | 30 | 33.9915 | 43.0370 | 34.7521 | 46.2863 | 27.3774 | 27.5504 |

- last-5（ep26-30）等权 Overall 均值为 `33.6749`、频率加权为 `34.14`（与第 16 节 `33.67 / 34.14`
  一致）。全轮峰值 ep22 的权重已落盘，可直接复评。
- 与 Run 9（同配方 + pretrained，全轮峰值 `37.94`）相比，pretrained 权重带来 `+3.23` Overall，
  主要集中在 Car strict（`36.34 → 48.96`）；这是第 9.3 节结论的数据来源。

### Run 9: N=4 Pretrained RSSM (seq_len=4, hdim=128, 30e)

- config: configs/r4det/TJ4D-R4Det_motion_align_rssm_det3d_N4_2x4_30e_pretrained.py
- work_dir: work_dirs/rssm_N4_2x4_30e_pretrained
- **频率加权 Overall（平台，last-5）**: **40.07 ± 0.47**（幅度提升来自 Car 权重 25%→48.4%；权重见 section 16）
- diff: load_from only, else identical to Run 7 (N4 30e no-pretrain)
- BEST: ep19 37.94 🔥 (ALL-TIME RECORD)
- ⚠️ 盘点注：ep19 权重已不在盘中，当前落盘为 ep20=36.94，见 17.2 节。
- LAST: ep30 35.63

### 9.1 Epoch curve: Overall 3D_moderate

| ep | mod | ep | mod | ep | mod |
|---:|---:|---:|---:|---:|---:|
| 1 | 17.07 | 11 | 34.99 | 21 | 35.86 |
| 2 | 20.82 | 12 | 34.94 | 22 | 37.36 |
| 3 | 26.07 | 13 | 35.84 | 23 | 37.08 |
| 4 | 28.25 | 14 | 36.39 | 24 | 34.98 |
| 5 | 27.15 | 15 | 36.27 | 25 | 36.22 |
| 6 | 32.19 | 16 | 36.09 | 26 | 35.68 |
| 7 | 31.46 | 17 | 35.44 | 27 | 36.16 |
| 8 | 34.05 | 18 | 35.56 | 28 | 36.18 |
| 9 | 35.38 | 19 | 37.94 | 29 | 36.23 |
| 10| 33.53 | 20 | 36.94 | 30 | 35.63 |

ep1 starts 17.07 (no-pretrain: 3.07, +14pt gap). Peaks at ep19 (37.94), then oscillates downward. ep30=35.63 still above no-pretrain BEST (34.71). 30 epochs too long for pretrained init.

### 9.2 BEST vs LAST

| Metric | BEST ep19 | LAST ep30 |
|---|---|---|
| Overall 3D_moderate | 37.94 | 35.63 |
| Overall 3D_easy | 40.80 | 37.74 |
| Overall 3D_hard | 36.41 | 34.15 |
| Overall BEV_moderate | 43.17 | 41.57 |
| Car 3D_mod_loose | 65.25 | 65.02 |
| Car 3D_mod_strict | 48.96 | 46.96 |
| Cyclist 3D_mod_loose | 41.43 | 38.07 |
| Cyclist 3D_mod_strict | 25.05 | 23.58 |
| Pedestrian 3D_mod_loose | 28.07 | 26.73 |
| Pedestrian 3D_mod_strict | 0.07 | 0.11 |
| Truck 3D_mod_loose | 47.94 | 47.46 |
| Truck 3D_mod_strict | 33.31 | 30.77 |

### 9.3 vs N4 30e no-pretrain (core ablation)

| Metric | No-pretrain ep22 | Pretrained ep19 | Delta |
|---|---:|---:|
| Overall 3D_moderate | 34.71 | 37.94 | +3.23 |
| Car 3D_mod_loose | 59.46 | 65.25 | +5.79 |
| Car 3D_mod_strict | 36.34 | 48.96 | +12.62 |
| Cyclist 3D_mod_loose | 46.28 | 41.43 | -4.85 |
| Cyclist 3D_mod_strict | 21.20 | 25.05 | +3.85 |
| Pedestrian 3D_mod_loose | 27.42 | 28.07 | +0.65 |
| Pedestrian 3D_mod_strict | 0.05 | 0.07 | +0.02 |
| Truck 3D_mod_loose | 50.50 | 47.94 | -2.56 |
| Truck 3D_mod_strict | 28.80 | 33.31 | +4.51 |

Pretrained backbone: +3.23 overall, focused on Car strict (+12.62). Cyclist loose -4.85 is only significant regression (precision-recall tradeoff: strict +3.85, loose -4.85).

### 9.4 vs ALL-TIME bests

| Metric | Previous best | Pretrained ep19 | Delta |
|---|---:|---:|
| Overall 3D_moderate | 34.71 (N4 30e) | 37.94 | +3.23 |
| Car 3D_mod_strict | 38.49 (N4 12e) | 48.96 | +10.47 |
| Cyclist 3D_mod_strict | 22.52 (baseline_temp) | 25.05 | +2.53 |
| Pedestrian 3D_mod_loose | 26.64 (baseline_temp) | 28.07 | +1.43 |
| Truck 3D_mod_strict | 31.72 (N3 hdim64) | 33.31 | +1.59 |
| Truck 3D_mod_loose | 50.50 (N4 30e) | 47.94 | -2.56 |
| Cyclist 3D_mod_loose | 50.37 (baseline_rssm) | 41.43 | -8.94 |

Nearly all records broken. Only Cyclist loose and Truck loose slightly below prior bests.

### 9.5 RSSM dynamics

| ep | kl_loss | recon_loss | eff_kl | post_std | prior_std | clamp |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.00 | 0.60 | 14.84 | 0.20 | 0.20 | 84.5% |
| 3 | 0.19 | 0.01 | 1.14 | 0.39 | 0.45 | 99.3% |
| 5 | 0.33 | 0.006 | 1.00 | 0.68 | 0.87 | 99.9% |
| 10 | 0.75 | 0.003 | 1.00 | 0.68 | 0.90 | 99.9% |
| 19 | 1.00 | 0.002 | 1.00 | 0.84 | 0.98 | 100% |
| 30 | 1.00 | 0.002 | 1.00 | 0.91 | 0.99 | 100% |

KL loss warms up ep1-8 then stabilizes at 1.0 (free_nats threshold). Recon loss converges 0.60->0.002. Posterior collapses to prior (post_std=prior_std, clamp>99.9%), stochastic state contributes minimally. Same behavior as all previous RSSM runs, not pretrain-related.

### 9.6 Key findings

1. 🔥 Pretrained backbone is largest single improvement: +3.23 overall, Car strict +12.62
2. ep1 fast start: 17.07 vs no-pretrain 3.07 (14 point gap). Pretrained feature quality extremely high
3. 30 epochs too long: peaked ep19, overfitting ep20-30. Best checkpoint: ep19
4. Cyclist loose -4.85: foreground supervision makes backbone features more conservative on cyclists (precision↑ recall↓). Cyclists are 21.5% of dataset (2790/5706 frames), not a rare class
5. Pedestrian strict near zero (0.07): consistent across all runs, detection head bottleneck

### 9.7 N=4 30e lr2e4 变体（negative run，未跑完）

> 该 run 只在第 16 节频率加权总表里出现过一行（`N4 30e lr2e4 = 28.30 ± 1.22`），此前没有独立记录。
> 本次审计按磁盘日志补账。**它不是正式主表 run，而是一次 lr 放大失败的负向尝试。**

- 工作目录：`/data/lurui/work_dirs/rssm_N4_2x4_30e_lr2e4`。
- 配置快照：同目录下的
  `TJ4D-R4Det_motion_align_rssm_det3d_N4_2x4_30e_lr2e4.py`（仓库内没有提交对应 config）。
- 与 Run 9（`N4_2x4_30e_pretrained`）的差异：`lr` 从 `1.5e-4` 放大到 `2e-4`，
  并把 `load_from` 置为 `None`（不加载 pretrained 权重）；其余 30e / samples_per_gpu=4 /
  CosineAnnealing / checkpoint_interval=2 与 Run 9 相同。
- 2026-08-12 01:30 启动，日志 `20260812_013046.log(.json)`；`checkpoint_config=dict(interval=2)`，
  磁盘只有 `epoch_12.pth` / `epoch_14.pth`，`latest.pth -> epoch_14.pth`。
- 训练在 ep15 iter 100/714 处中断（最后一条日志 2026-08-12 09:59:27 UTC），ep15 没有 val 行；
  属于**未跑完的负向 run**，不能与完整 30e run 直接比较。

完整 val 曲线（`pts_bbox/KITTI/*`，val 写到 ep14）：

| epoch | Overall 3D | Overall BEV | Car strict | Cyclist loose | Ped loose | Truck strict |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5.0308 | 6.4537 | 9.1358 | 7.8443 | 2.6283 | 0.5147 |
| 2 | 12.2053 | 15.5320 | 15.6949 | 24.0696 | 3.4445 | 5.6123 |
| 3 | 14.4329 | 18.3290 | 17.1589 | 35.8316 | 2.4341 | 2.3071 |
| 4 | 18.5879 | 24.5679 | 19.5221 | 40.0354 | 6.0661 | 8.7280 |
| 5 | 23.6179 | 29.7422 | 21.7181 | 43.5485 | 18.1949 | 11.0100 |
| 6 | 26.8642 | 33.9331 | 25.5705 | 45.8394 | 18.4087 | 17.6384 |
| 7 | 28.5554 | 36.2970 | 25.0540 | 46.1537 | 24.8047 | 18.2094 |
| 8 | 28.8649 | 35.5137 | 25.9161 | 48.3771 | 21.9073 | 19.2590 |
| 9 | 27.5870 | 34.2866 | 30.2644 | 41.7532 | 25.9336 | 12.3969 |
| 10 | 28.8201 | 35.6477 | 34.5154 | 39.7876 | 23.6978 | 17.2796 |
| 11 | 27.0096 | 36.6799 | 29.7362 | 46.3569 | 14.3875 | 17.5577 |
| 12 | 30.0844 | 38.8798 | 31.0891 | 43.4304 | 25.2556 | 20.5624 |
| 13 | 28.2174 | 37.2982 | 31.6215 | 38.8671 | 22.0208 | 20.3604 |
| 14 | 27.3748 | 37.1936 | 31.9762 | 37.9077 | 23.2034 | 16.4117 |

全轮峰值与 best saved：

| 口径 | epoch | Overall 3D | Overall BEV | Car strict | Cyclist loose | Ped loose | Truck strict |
|---|---:|---:|---:|---:|---:|---:|---:|
| 全轮峰值 | 12 | 30.0844 | 38.8798 | 31.0891 | 43.4304 | 25.2556 | 20.5624 |
| LAST val | 14 | 27.3748 | 37.1936 | 31.9762 | 37.9077 | 23.2034 | 16.4117 |
| last disk | 14 | 27.3748 | 37.1936 | 31.9762 | 37.9077 | 23.2034 | 16.4117 |

读数：

- 该 run 的全轮峰值 `30.0844` 比 Run 9（pretrained，30e）的全轮峰值 `37.94` 低 `7.86`；两者最先分叉的
  不是 pretrained 权重本身就是 lr 与 pretrained 同时改变，所以只能作为「去掉 pretrained 且放大 lr
  的负向尝试」记录，不能当作纯 lr 消融。
- 后段 ep10-14 在 27-30 平台震荡，没有追赶 Run 9 的 pretrained 曲线；ep15 未跑完，ep15-30 未知。
- 记录目的：避免后续再看到 `28.30 ± 1.22`（第 16 节 last-5 表）时误以为它是一条可比主 run。
6. RSSM stochastic states contribute minimally (clamp 99.9%): posterior collapsed to prior, model relies on deterministic path. Normal for free_nats=1.0, not hurting detection
7. Detection head losses decreasing: bbox 1.35->0.14, cls 1.05->0.08

### 9.7 Conclusions

- **Pretraining is the right direction** — benefit far exceeds any other single change
- Best model: ep19 checkpoint (37.94 3D_moderate)
- Future priorities:
  1. Reduce training epochs to 20-24 to match pretrained init speed
  2. Cyclist recall recovery: longer detection head finetune, lower fg supervision weight, cyclist-specific augmentations
  3. Pedestrian strict: needs detection head / loss function fix (all-run issue)
  4. Velocity signal injection: switch from action_dim=0 to using ego velocity from dataset
  5. Ablate pretrained depth_net vs image backbone contributions separately


---

## 10. N=4 Pretrained RSSM + Detection Head v2 (24e)

> Branch: `detection_head_v2`。基于 Run 9 配置（N4, hdim=128, pretrained backbone），**唯一改动是检测头 Anchor3DHead**。
> 时序模块、backbone、pretrained 权重、评估口径与 Run 9 完全一致，因此 Run 10 vs Run 9 是干净的「检测头消融」。
>
> 检测头改动（4 项，3 项生效）：
> 1. **行人 3 anchors**：anchor 组 4→6，Ped×3 + Cyc/Car/Truck 各 1，`anchor_class_mapping=[0,0,0,1,2,3]`
> 2. **行人关闭方向分类**：`ignore_dir_classes=[0]`
> 3. **IoU-aware quality 分支**：`use_iou_branch=True`，训练期给 L1 回归信号，推理不打分（避免与 N 帧 RSSM 前向冲突）
> 4. ~~per-class FocalLoss alpha~~ 已回退 `alpha=0.25`（CUDA kernel 只支持 float）
>
> 24 epochs，samples_per_gpu=4，lr=1.5e-4（CosineAnnealing），`load_from=pretrained_tj4d.pth`，**`checkpoint_interval=2`（注意：奇数 epoch 不存）**。
> 该 run 在 4 卡上训练，**没有梯度累积**，有效 batch 为 4×4=16。累积 hook 在仓库其他分支
> 早有使用（如 2026-08-19 的 N2 BPTT、08-24 的 no-temporal），但**本 config 直到
> 2026-09-01（commit `6221e6a`）才被加上** `GradientCumulativeOptimizerHook(cumulative_iters=2)`，
> 因此 Run 10 本身与多 seed 复现之间存在 batch 口径差异。

### Run 10: N=4 Pretrained RSSM + head-v2

- config: `configs/r4det/TJ4D-R4Det_motion_align_rssm_det3d_N4_2x4_24e_pretrained_v2_head.py`
- work_dir: `work_dirs/rssm_N4_2x4_24e_pretrained_v2_head`
- **频率加权 Overall（平台，last-5）**: **43.51 ± 0.51**（幅度提升来自 Car 权重 25%→48.4%；权重见 section 16）
- **BEST overall: ep11 = 40.60** ⚠️ 未存（奇数 epoch，interval=2）
- **best saved overall: ep14 = 39.65**
- **⚠️ 盘点注：全轮峰值 ep11 未落盘；原 best saved ep14 权重也已不在盘中，当前落盘为 ep12=39.26，见 17.2 节**
- **Car strict BEST: ep20 = 53.04（已存）**
- LAST: ep24 = 38.76

### 10.1 Epoch curve: Overall 3D_moderate

| ep | mod | ep | mod | ep | mod |
|---:|---:|---:|---:|---:|---:|
| 1 | 17.11 | 9 | 37.97 | 17 | 35.64 ⬇ |
| 2 | 23.29 | 10 | 36.47 | 18 | 39.03 ⬆ |
| 3 | 26.52 | 11 | **40.60** | 19 | 38.77 |
| 4 | 33.17 | 12 | 39.26 | 20 | 38.50 |
| 5 | 34.63 | 13 | 39.03 | 21 | 38.78 |
| 6 | 35.44 | 14 | 39.65 | 22 | 38.14 |
| 7 | 36.92 | 15 | 38.97 | 23 | 38.34 |
| 8 | 35.38 | 16 | 38.40 | 24 | 38.76 |

- **ep11 冲到 40.60 全场峰值**，ep12–16 在 38.4–39.7 高位震荡；ep17 突发断崖到 35.64（Car 52→47，验证集偶发抖动），ep18 立即拉回 39.03，之后 ep18–24 稳定在 38.1–38.8 平台。
- **峰在 ep11（40.60），但 interval=2 导致 ep11 权重未落盘**，这是本轮唯一的实质损失。

### 10.2 BEST vs LAST

| Metric | BEST | epoch | LAST (ep24) |
|---|---:|---:|---:|
| Overall 3D_moderate（全轮峰值） | **40.60** | 11 (未存) | 38.76 |
| Overall 3D_moderate（best saved） | **39.65** | 14 | 38.76 |
| Overall 3D_easy | 42.49 | 15 | 40.73 |
| Overall 3D_hard | 39.06 | 11 | 37.42 |
| Overall BEV_moderate | 48.70 | 13 | 46.66 |
| Car 3D_mod_strict | **53.04** | 20 | 51.80 |
| Cyclist 3D_mod_strict | 25.41 | 8 | 23.62 |
| Pedestrian 3D_mod_strict | 0.42 | 2 | 0.08 |
| Truck 3D_mod_strict | 33.23 | 11 | 26.26 |

### 10.3 vs Run 9（同 base，唯一差异=检测头）

| Metric | Run 9 全轮峰值 (ep19) | Run 10 全轮峰值 | Run 10 已存最高 | Δ (峰值) |
|---|---:|---:|---:|---:|
| Overall 3D_moderate | 37.94 | **40.60** (ep11) | 39.65 (ep14) | **+2.66 / +1.71** |
| Overall 3D_easy | 40.80 | **42.49** (ep15) | 41.58 (ep14) | +1.69 |
| Overall 3D_hard | 36.41 | **39.06** (ep11) | 38.14 (ep14) | +2.65 |
| Overall BEV_moderate | 43.17 | **48.70** (ep13) | 47.50 (ep14) | +5.53 |
| Car 3D_mod_strict | 48.96 | **53.04** (ep20) | 53.04 (ep20) | **+4.08** |
| Cyclist 3D_mod_strict | 25.05 | 25.41 (ep8) | 24.59 (ep21) | +0.36 |
| Pedestrian 3D_mod_strict | 0.07 | 0.42 (ep2) | 0.14 (ep20) | 噪声级 |
| Truck 3D_mod_strict | 33.31 | 33.23 (ep11) | 30.26 (ep14) | −0.08 |

### 10.4 Per-class loose（召回口径，0.25 IoU）

| Metric | Run 9 BEST | Run 10 BEST | Δ |
|---|---:|---:|---:|
| Car 3D_mod_loose | 65.25 | **73.96** (ep12) | **+8.71** |
| Cyclist 3D_mod_loose | 41.43 | **52.75** (ep12) | **+11.32** |
| Pedestrian 3D_mod_loose | 28.07 | 28.74 (ep10) | +0.67 |
| Truck 3D_mod_loose | 47.94 | **53.16** (ep18) | **+5.22** |

### 10.5 Key findings

1. **检测头改动带来明确正向收益**：Overall 3D_moderate +1.71（best saved ep14）/ +2.66（全轮峰值 ep11），BEV 口径 +5.53。Car strict 刷到 **53.04**，全库历史最高，超 Run 9 达 +4.08。
2. **loose（召回）口径全面提升**：Car +8.71、Cyclist +11.32、Truck +5.22。来源是多 anchor + `anchor_class_mapping` + IoU-proxy 训练信号共同改善了共享检测头特征与打分质量；Car/Cyclist/Truck 仍各只有 1 个 anchor，说明提升是「共享头标定变好」，不是简单加 anchor。
3. **Pedestrian strict 依旧 ≈ 0**：3 组行人 anchor 只让 loose 微涨（+0.67）、strict 在 0.08–0.42 的噪声带内横跳。**证实瓶颈是 BEV 0.16m 分辨率 + 点云稀疏（行人仅 ~15 cell），不是 anchor 数量**，与上轮判断一致。
4. **Truck strict 持平（33.2 vs 33.3）但 loose +5.22**：召回上去了、定位没上去，**证实短货车(2.8m)到半挂(25.5m)的巨大尺寸方差 + 单 anchor 才是 Truck 瓶颈**，与尺寸方差分析吻合。
5. **Cyclist strict 几乎没动（25.4 vs 25.1）**：`ignore_dir_classes=[0]` 只关了行人方向分类，未伤及 Cyclist；但也没带来骑行者 strict 提升。
6. ⚠️ **`checkpoint_interval=2` 丢掉了本轮的峰值 checkpoint**（ep11=40.60 只存在于日志）。下一轮必须改成 `interval=1`（或加 `max_keep_ckpts`），否则奇数 epoch 的峰值还会再次丢失。

### 10.6 Conclusions / Next

- **head-v2 是当前最强模型**：best saved ep14（Overall 39.65）+ 新纪录 Car strict ep20（53.04）。按需选点：看 Overall 用 ep14，看 Car 单项用 ep20。
- 检测头这一刀砍对了方向，但它主要救了 Car / 召回，**没解决 Ped strict 和 Truck strict 两个真正的硬骨头**。
- 下一步优先级：
  1. **Truck 多 anchor**（van/标准货/半挂 3 组）——直接打当前最大的 strict 差距（Truck 26~33 vs Car 52），零替代风险；
  2. `checkpoint_interval=1` 必修，避免再丢峰值；
  3. Pedestrian 需跳出 anchor 层——BEV 分辨率 / 点云稠密化 / point-based 头，才是真正杠杆；
  4. Cyclist：loose 已大涨，strict 未跟上，可考虑骑行者专属 size/增强。

### 10.7 复现性评估：Run 10 的 39.65 是「平台顶」，不是「典型值」

- **40.60 从未成为可用 checkpoint**：Run 10 `checkpoint_interval=2`，奇数 epoch 不落盘，ep11=40.60（实际日志 40.59）只存在于日志。当前 best saved 是 **ep14=39.65**。
- 拆开 ep11 看，40.59 是一次性脉冲：Truck strict 22.77→33.23 单类跳涨，下一轮跌回 27.49；Car 只是维持高位。它不可复现。
- **稳定平台 ep12–24 的 Overall = 38.56 ± 0.97**（区间 35.64～39.65）。39.65 是这个平台的上沿，重跑一次大概率落在 38.5～39.5。
- **真正的漂移源不是 Car，而是 Cyclist loose 和 Truck strict**：

| 构成项 | ep2–24 均值 | 波动范围 |
|---|---:|---:|
| Car strict | 49.12 | 38.6～53.0 |
| Truck strict | 25.83 | 14.0～33.2 |
| Ped loose | 25.94 | 19.8～28.7 |
| Cyclist loose | 45.49 | 20.8～52.7 |

- Cyclist loose ±7.77 是最大漂移项（其余几个都在 ±2～4.5），它占 Overall 1/4 权重，会直接带着 Overall 上下漂近 2 点。
- **结论**：Run 10 的复现性应由「平台均值 38.56」或固定末 N 个 epoch 均值来衡量，不要拿单点 BEST（39.65 或 40.60）当作可复现水平；后续与 N2/N4/BPTT 对比时同理，比平台均值而非单点峰值。

---

## 11. N=4 Pretrained RSSM + head-v2 + Truck 多 anchor (24e)

> Branch: `detection_head_v2`。基于 Run 10 配置，**唯一改动是给 Truck 加 3 组 anchor**（其余时序/backbone/pretrain/评估口径与 Run 10 完全一致）。
> Truck anchor 来源：训练集 k-means（`[w,l,h]`）：短货车 `[2.50, 6.72, 2.36]`、标准货车 `[3.81, 9.96, 2.66]`、长半挂 `[4.09, 17.88, 3.08]`。
> 同时把 `checkpoint_interval` 2→1，修复 Run 10 丢峰值的问题。

### Run 11: N=4 Pretrained RSSM + head-v2 + Truck×3 anchor

- config: `configs/r4det/TJ4D-R4Det_motion_align_rssm_det3d_N4_2x4_24e_pretrained_v2_head_truck.py`
- work_dir: `work_dirs/rssm_N4_2x4_24e_pretrained_v2_head_truck`
- **频率加权 Overall（平台，last-5）**: **39.88 ± 0.78**（幅度提升来自 Car 权重 25%→48.4%；权重见 section 16）
- **全轮峰值：ep11 = 38.45（✅ 已落盘，interval=1 生效）**
- **LAST: ep24 = 37.20**
- Truck strict BEST: **35.03 @ep11（历史最高）**
- Cyclist strict BEST: **27.11 @ep20（历史最高）**
- Car strict BEST: 48.03 @ep18

### 11.1 Epoch curve: Overall 3D_moderate

| ep | mod | ep | mod | ep | mod |
|---:|---:|---:|---:|---:|---:|
| 1 | 16.23 | 9 | 37.02 | 17 | 36.11 |
| 2 | 22.68 | 10 | 34.91 | 18 | 37.97 |
| 3 | 23.74 | 11 | **38.45** | 19 | 36.74 |
| 4 | 32.03 | 12 | 36.29 | 20 | 37.28 |
| 5 | 32.36 | 13 | 37.71 | 21 | 35.43 |
| 6 | 33.27 | 14 | 37.35 | 22 | 35.88 |
| 7 | 35.77 | 15 | 36.51 | 23 | 36.53 |
| 8 | 33.65 | 16 | 37.50 | 24 | 37.20 |

- 前期比 Run 10 慢（ep1 16.23 vs 17.11/E，ep2 22.68 vs 23.29），但 ep7 起追平，ep11 冲到峰值 38.45 后一直在 35.4–38.0 震荡，没有出现 Run 10 的 ep17 断崖。
- **ep11 峰值全部落盘**，本轮没有 checkpoint 丢失问题。

### 11.2 BEST vs LAST

| Metric | BEST (ep11) | LAST (ep24) |
|---|---:|---:|
| Overall 3D_moderate | **38.45** | 37.20 |
| Overall 3D_easy | 40.57 | 38.95 |
| Overall 3D_hard | 36.90 | 35.74 |
| Overall BEV_moderate | 45.10 | 43.78 |
| Car 3D_mod_strict | 48.03 (ep18) | 44.91 |
| Cyclist 3D_mod_strict | 27.11 (ep20) | 26.10 |
| Pedestrian 3D_mod_strict | 0.28 (ep2) | 0.08 |
| Truck 3D_mod_strict | 35.03 (ep11) | 30.46 |

### 11.3 vs Run 10（同 base，唯一差异=Truck 多 anchor）

| Metric | Run 10 BEST | Run 11 BEST | Δ |
|---|---:|---:|---:|
| Overall 3D_moderate | **40.60** (ep11) / 39.65 best saved | 38.45 (ep11) | **−2.15 / −1.20** |
| Car 3D_mod_strict | **53.04** (ep20) | 48.03 (ep18) | **−5.01** ❌ |
| Truck 3D_mod_strict | 33.23 (ep11) | **35.03** (ep11) | **+1.80** ✅ 历史最高 |
| Cyclist 3D_mod_strict | 25.41 (ep8) | **27.11** (ep20) | **+1.70** ✅ 历史最高 |
| Pedestrian 3D_mod_strict | 0.42 (ep2) | 0.28 (ep2) | 噪声级 |

### 11.4 Per-class loose（召回口径，0.25 IoU）

| Metric | Run 10 BEST | Run 11 BEST | Δ |
|---|---:|---:|---:|
| Car 3D_mod_loose | **73.96** (ep12) | 70.03 (ep13) | **−3.93** ❌ |
| Cyclist 3D_mod_loose | **52.75** (ep12) | 48.25 (ep20) | **−4.50** ❌ |
| Pedestrian 3D_mod_loose | 28.74 (ep10) | 29.71 (ep9) | +0.97 |
| Truck 3D_mod_loose | 53.16 (ep18) | 52.38 (ep11) | −0.78 |

### 11.5 Key findings

1. **Truck 多 anchor 的目标达成了**：Truck strict 35.03，历史最高（此前 33.31 Run 9 / 33.23 Run 10），+1.80。Cyclist strict 也顺带涨到 27.11（历史最高）。
2. **但代价是 Car 显著回吐**：Car strict −5.01（53.04→48.03），Car loose −3.93、Cyclist loose −4.50。因为 Car 占数据集 48.1%、主导 Overall，**Overall 反降 −1.2~−2.2**。
3. **本质是"抢正样本"**：3 组 Truck 的 MaxIoU 分配器把更多 head 容量和正样本判给 Truck；标准货车 anchor `3.81×9.96` 与 Car 尺寸高度重叠，直接吃掉了 Car 的召回。属于拆东墙补西墙。
4. **Truck 与 Car 的混淆是根子**，不是 anchor 数：Truck 长度谱 2.8–25.5m 且与 Car 重叠区大，加 anchor 只换来 +1.8 strict，付出了 −5 Car 的代价。
5. Pedestrian strict 依旧在 0.03–0.28 噪声带内，与 anchor 无关（同上轮结论）。
6. ✅ `checkpoint_interval=1` 生效，ep1–24 全部落盘，**峰值 ep11 保住了**——Run 10 的坑已补。

### 11.6 Conclusions / Next

- **这一轮是负收益**：以 Overall 指标为标准，Truck 多 anchor 不值得默认采用（−1.2 已存口径）。主模型仍应回退到 **Run 10 head-v2 ep14（39.65）** 或 ep20（Car strict 53.04）。
- **除非指标权重明确偏向 Truck**，否则不要保留 Truck×3 anchor。若要保留 Truck 提升，可考虑：
  1. Car 与 Truck 分离预测分支 / 类专属 head，避免共享 head 正样本互抢；
  2. 先做 Car–Truck 混淆消融（类间 hard example 采样、尺寸先验），而不是继续加 anchor。
- 真正没解决的两个硬骨头仍是 **Car–Truck 混淆**（Truck strict 35 vs Car strict 48）和 **Pedestrian strict ≈ 0**（BEV 分辨率/点云稀疏瓶颈）。

---

## 12. N=4 Pretrained RSSM + head-v2 + Car-large + Truck×3 anchor (24e)

> Branch: `detection_head_v2`。基于 Run 11 配置，**唯一改动是给 Car 增加一组大 anchor**（标准 Car `1.84×4.56×1.70` + Car-large/SUV/van `1.94×5.07×2.02`），即 anchor 构成从 Run 11 的 Ped×3/Cyc×1/Car×1/Truck×3 变为 **Ped×3/Cyc×1/Car×2/Truck×3**。
> 时序模块、pretrained backbone、训练 schedule、评估口径与 Run 11 完全一致，因此 Run 12 vs Run 11 是干净的「Car-large anchor」消融。`checkpoint_interval=1`，所有 epoch 已落盘。

### Run 12: N=4 Pretrained RSSM + head-v2 + Car×2 + Truck×3 anchor

- config: `configs/r4det/TJ4D-R4Det_motion_align_rssm_det3d_N4_2x4_24e_pretrained_v2_head_truck_car2.py`
- work_dir: `work_dirs/rssm_N4_2x4_24e_pretrained_v2_head_truck_car2`
- **频率加权 Overall（平台，last-5）**: **40.23 ± 0.65**（幅度提升来自 Car 权重 25%→48.4%；权重见 section 16）
- **全轮峰值：ep14 = 38.11（✅ 已落盘）**
- **LAST: ep24 = 35.73**
- Car strict BEST: **53.77 @ep14（历史最高）**
- Truck strict BEST: 29.05 @ep20
- Cyclist strict BEST: 23.81 @ep8

### 12.1 Epoch curve: Overall 3D_moderate

| ep | mod | ep | mod | ep | mod |
|---:|---:|---:|---:|---:|---:|
| 1 | 15.20 | 9 | 35.64 | 17 | 34.60 |
| 2 | 19.43 | 10 | 33.77 | 18 | 35.65 |
| 3 | 22.81 | 11 | 36.87 | 19 | 36.35 |
| 4 | 31.31 | 12 | 37.79 | 20 | 35.92 |
| 5 | 31.84 | 13 | 37.04 | 21 | 36.00 |
| 6 | 31.12 | 14 | **38.11** | 22 | 35.15 |
| 7 | 34.99 | 15 | 36.78 | 23 | 35.57 |
| 8 | 34.19 | 16 | 35.52 | 24 | 35.73 |

- 起点明显偏低（ep1=15.20，低于 Run 10 的 17.11 和 Run 11 的 16.23），但 ep4 追到 31.31，ep12–14 升到 37.8–38.1 平台。
- 峰值出现在 ep14（38.11），之后 34.6–36.4 震荡，没有再突破峰值；整体曲线波动比 Run 10 更平缓，没有 ep17 断崖。

### 12.2 BEST vs LAST

| Metric | BEST | epoch | LAST (ep24) |
|---|---:|---:|---:|
| Overall 3D_moderate | **38.11** | 14 | 35.73 |
| Overall 3D_easy | 39.99 | 14 | 37.83 |
| Overall 3D_hard | 36.71 | 14 | 34.37 |
| Overall BEV_moderate | 46.34 | 14 | 43.49 |
| Car 3D_mod_strict | 53.77 | 14 | 49.05 |
| Cyclist 3D_mod_strict | 23.81 | 8 | 20.49 |
| Pedestrian 3D_mod_strict | 0.31 | 17 | 0.16 |
| Truck 3D_mod_strict | 29.05 | 20 | 27.16 |

### 12.3 vs Run 11（同 base，唯一差异=Car-large anchor）

| Metric | Run 11 BEST | Run 12 BEST | Δ |
|---|---:|---:|---:|
| Overall 3D_moderate | **38.45** (ep11) | 38.11 (ep14) | **−0.34** |
| Car 3D_mod_strict | 48.03 (ep18) | **53.77** (ep14) | **+5.74** ✅ 历史最高 |
| Truck 3D_mod_strict | **35.03** (ep11) | 29.05 (ep20) | **−5.98** ❌ |
| Cyclist 3D_mod_strict | **27.11** (ep20) | 23.81 (ep8) | **−3.30** ❌ |
| Pedestrian 3D_mod_strict | 0.28 (ep2) | 0.31 (ep17) | +0.03（噪声级） |

### 12.4 Per-class loose（召回口径，0.25 IoU）

| Metric | Run 11 BEST | Run 12 BEST | Δ |
|---|---:|---:|---:|
| Car 3D_mod_loose | 70.03 (ep13) | **74.33** (ep13) | **+4.30** ✅ 历史最高 |
| Cyclist 3D_mod_loose | **48.25** (ep20) | 47.98 (ep8) | −0.27 |
| Pedestrian 3D_mod_loose | 29.71 (ep9) | 29.44 (ep21) | −0.27 |
| Truck 3D_mod_loose | **52.38** (ep11) | 51.85 (ep20) | −0.53 |

### 12.5 Key findings

1. **Car-large anchor 对 Car 非常有效**：Car strict 53.77（ep14）刷新历史最高，超过 Run 10 的 53.04（ep20）+0.73；Car loose 74.33（ep13）也超过 Run 10 的 73.96 +0.37。Run 11 被 Truck anchor 吃掉的那部分 Car 基本抢回来了。
2. **但共享检测头继续零和博弈**：Truck strict 从 Run 11 峰值 35.03 回落到 29.05（−5.98），Run 11 拿到的 Truck 增益几乎全部还回去；Cyclist strict 也从 27.11 退到 23.81（−3.30）。
3. **Overall 不升反微降**：38.11 vs Run 11 38.45（−0.34），并且仍比 Run 10 已存 ep14 39.65 低 **−1.54**。Car 类占比最大，但对 Overall 的贡献被 Truck/Cyclist 回吐抵消。
4. **Car-Truck 的根因不是 anchor 数量**：三轮 anchor 调整的结果都是拆东墙补西墙；共享 head 中 max-IoU 正样本分配像跷跷板，重划给 Car 的容量会从 Truck 抢回正样本，反之亦然。
5. Pedestrian strict 依然在 0.10–0.31 的噪声带内，anchor 数量继续被证明不是行人 3D strict 的瓶颈。
6. ✅ `checkpoint_interval=1` 继续生效，ep1–24 全部落盘，峰值 ep14 已保存。

### 12.6 Conclusions / Next

- **本轮仍是负收益**：Car-large anchor 单独使用能救 Car，但会牺牲 Truck/Cyclist，Overall 没有超过 Run 11，更低于 Run 10 的 best saved。
- 每个 epoch 的 Overall 最优 checkpoint 已保存为 Run 12 ep14；如果只关注 Car 单项，Run 12 ep14 的 Car strict 53.77 已是新纪录；综合指标仍回退到 **Run 10 head-v2 ep14（39.65）**。
- anchor 床铺已经加到 Ped×3/Cyc×1/Car×2/Truck×3，继续加 anchor 预计收益会更低。下一步应转向解决共享 head 的 Car–Truck 争夺：优先验证仓库中已备好的 `confusion_pairs=[[2,3]]` 抑制 loss 配置（`_head_confuse.py`），或拆分 Car/Truck 专属分支。
- 另一个未落入 Run 12 的改变是 Doppler static/dynamic mask（`_head_dynmask.py`），应在不混入 anchor 变量的前提下单独跑消融。

## 13. Doppler 动静 mask 设计注记 + 训练结果（Run 13）

> Branch: `radar_static_dynamic`。基于 Run 10 最优配置（head-v2，无 anchor 变量），新增 `RadarStaticDynamicScore` 点云变换 + `RadarPillarFeatureNet` 的 soft-mask 门控，利用 4D 雷达自带的径向多普勒速度做「动/静」显式建模。config 为
> `configs/r4det/TJ4D-R4Det_motion_align_rssm_det3d_N4_2x4_24e_pretrained_v2_head_dynmask.py`。

### 13.1 设计与数据流

1. **点云变换 `RadarStaticDynamicScore`**（在 `loading.py`，置于任何旋转/缩放增广**之前**，保证方向向量与径向速度同处于传感器系）：
   - 对每帧点云，用所有点的单位方向向量 `u` 对径向速度 `v_r` 做一次最小二乘拟合估计 ego 速度 `v_ego`（静态点满足 `v_r ≈ u @ v_ego`），再用内点（`|residual| < 0.5 m/s`）refit 一次以剔除运动目标；
   - 目标点残余越大越「动」：`dynamic_score = 1 - exp(-(e / sigma)^2)`，作为第 6 通道拼接（`in_channels=6`）。
2. **PFN 之前的 soft-mask 消费**（`pillar_encoder.py` 的 `RadarPillarFeatureNet.forward`）：
   - 第 6 通道**不进入** 12 维 PFN 布局，而是在构建 PFN 输入前被消耗为门控：`gating = 1 + β · score`；
   - **velocity-only 模式**（默认 `dynamic_mask_snr=False`）：只对第 3 维速度乘门控，SNR（第 4 维）不变。即 `[x, y, z, v·gating, snr]`；
   - 可选 `dynamic_mask_snr=True` 时对 `[v, snr]` 同乘门控（更强但会稀释弱目标 SNR，当前默认关闭）。

### 13.2 为什么必须是 velocity-only + β=0.5（设计取舍备忘）

1. **先发现的硬 bug：6 通道不能直接进 PFN。** `PFNLayer_Radar` 硬编码了 12 维布局的索引 `[0..11]`（中心 x/y/z + 离群 x/y/z + 体素中心 x/y/z + v/snr 各一维）。若把原始 6 通道喂进 `RadarPillarFeatureNet`，会产出 13 维特征，既悄悄打乱 spatial/velocity/SNR 中心特征的顺序，又让 score 被误读成 cluster-x。因此 score 必须像上面一样在进 PFN 前消费掉，PFN 布局与**预训练权重零错位**。
2. **先数据后拍板：对慢速目标（尤其行人）的稀释风险是真实存在的。** 在 80 个训练帧上把点投到 GT 3D 框里统计 dynamic_score：
   - Pedestrian：mean 0.145 / **median 0.000** / 仅 15% 点 > 0.5 —— 行人基本是静止的，速度维没有可提取的运动信号；
   - Car：mean 0.315 / median 0.169；
   - Cyclist：mean 0.488 / median 0.50。

   结论：如果对 SNR 或整体特征做门控，低速行人会在相对意义上被系统性压低，行人 3D 本来就难（strict 长期 0.1–0.31 噪声带），不可接受。
3. **因此收敛到当前的保守版本**：
   - `dynamic_mask_snr=False`：SNR 完全不动，只放大/保留速度维，弱回波行人的 SNR 不被蚕食；
   - `dynamic_weight=0.5`（β），而不是 1.0：温和提升动目标速度显著性，避免大动态目标单点主导；
   - `sigma=0.5`（而非默认 0.3）：让分数更平滑，中段速度不会一上来就饱和到 1。
4. **开关语义**：`dynamic_weight=0.0` 或输入为 5 通道时整个分支是 identity no-op，作为同 config 的反向对照可以直接复用。
5. **`pts_voxel_encoder.in_channels=6` 要保留，不要「顺手优化成 5」**：
   - `RadarPillarFeatureNet.__init__` 用这个值做装饰通道计数，会得到 `self.in_channels=13`；但 `PFNLayer_Radar` 的 forward 用的是**硬编码** `linear1(8→32)/linear2(2→16)/linear3(2→16)` 与 `index_select([0,1,2,5,6,7,8,9] / [3,10] / [4,11])`，完全不读 `self.in_channels`——所以 13 vs 12 无 runtime 差异，预训练权重照常匹配（形状都是 8/2/2）。
   - 而 `R4Det.pts_dim = kwargs['pts_voxel_encoder']['in_channels']` 需要「原始点 = 6 通道」这个真值，所以 `in_channels=6` 对这条路径才是对的；当前 `use_sa_radarnet=False` 使这几处 `pts_dim` 成为死代码，但语义要保留正确。
   - 结论：保持 `in_channels=6`。改成 5 会在将来启用 voxelpainting/`use_sa_radarnet` 时把 `pts_dim` 算错。
6. **稀疏帧 fallback 已修**：`n < min_points(10)` 时原返回 `score=0.5`（整帧 velocity 被统一 ×1.25），改为 `score=0.0`（`gating=1`，identity）——「分不清动/静」应退回 baseline 而不是施加偏置，与 velocity-only 的保守原则一致。

### 13.3 训练结果（Run 13）

- config: `configs/r4det/TJ4D-R4Det_motion_align_rssm_det3d_N4_2x4_24e_pretrained_v2_head_dynmask.py`
- work_dir: `work_dirs/rssm_N4_2x4_24e_pretrained_v2_head_dynmask`
- **频率加权 Overall（平台，last-5）**: **38.62 ± 1.15**（幅度提升来自 Car 权重 25%→48.4%；权重见 section 16）
- **全轮峰值：ep8 = 36.21（✅ 已落盘，interval=1）**
- **LAST: ep24 = 35.10**
- Car strict BEST: 50.26 @ep8
- Cyclist strict BEST: 25.32 @ep7
- Truck strict BEST: 29.07 @ep18
- Pedestrian strict BEST: 0.38 @ep10（噪声带内）

### 13.4 Epoch curve: Overall 3D_moderate

| ep | mod | ep | mod | ep | mod |
|---:|---:|---:|---:|---:|---:|
| 1 | 15.31 | 9 | 31.72 | 17 | 35.51 |
| 2 | 18.46 | 10 | 35.88 | 18 | 35.80 |
| 3 | 26.02 | 11 | 35.11 | 19 | 34.65 |
| 4 | 29.45 | 12 | 35.70 | 20 | 35.90 |
| 5 | 29.89 | 13 | 35.42 | 21 | 34.77 |
| 6 | 32.58 | 14 | 33.98 | 22 | 34.68 |
| 7 | 35.30 | 15 | 34.81 | 23 | 35.34 |
| 8 | **36.21** | 16 | 34.40 | 24 | 35.10 |

- 起点偏低（ep1=15.31，低于 Run 10 的 17.11），ep4 追到 29.45 后增速放缓。
- **峰值在 ep8（36.21），之后 ep9 暴跌到 31.72，再未恢复到 36+**。ep10–24 在 34–36 区间窄幅震荡，整体呈「早峰后稳态下降」。
- 与 Run 10 的 ep11=40.60 / ep14=39.65 相比，**峰值低 3.44–4.39 点，是迄今最大单次回退**。

### 13.5 BEST vs LAST

| Metric | BEST (ep8) | LAST (ep24) |
|---|---:|---:|
| Overall 3D_moderate | **36.21** | 35.10 |
| Overall 3D_easy | 38.43 | 36.52 |
| Overall 3D_hard | 35.04 | 33.83 |
| Overall BEV_moderate | 43.68 | 42.38 |
| Car 3D_mod_strict | 50.26 | 43.54 |
| Cyclist 3D_mod_strict | 19.50 | 21.05 |
| Pedestrian 3D_mod_strict | 0.06 | 0.05 |
| Truck 3D_mod_strict | 21.08 | 26.74 |

### 13.6 vs Run 10（同 base，唯一差异=Doppler 动静 mask）

| Metric | Run 10 BEST | Run 13 BEST | Delta |
|---|---:|---:|---:|
| Overall 3D_moderate | **40.60** (ep11) / 39.65 best saved (ep14) | 36.21 (ep8) | **−4.39 / −3.44** ❌❌ |
| Overall 3D_easy | 42.49 (ep15) | 38.43 | −4.06 |
| Overall 3D_hard | 39.06 (ep11) | 35.04 | −4.02 |
| Overall BEV_moderate | 48.70 (ep13) | 43.68 | −5.02 |
| Car 3D_mod_strict | **53.04** (ep20) | 50.26 (ep8) | **−2.78** |
| Car 3D_mod_loose | **73.96** (ep12) | 70.99 (ep11) | −2.97 |
| Cyclist 3D_mod_strict | 25.41 (ep8) | 25.32 (ep7) | −0.09 |
| Cyclist 3D_mod_loose | **52.75** (ep12) | 46.64 (ep8) | **−6.11** |
| Pedestrian 3D_mod_strict | 0.42 (ep2) | 0.38 (ep10) | 噪声级 |
| Pedestrian 3D_mod_loose | 28.74 (ep10) | 31.48 (ep10) | **+2.74** |
| Truck 3D_mod_strict | 33.23 (ep11) | 29.07 (ep18) | **−4.16** |
| Truck 3D_mod_loose | 53.16 (ep18) | 52.44 (ep11) | −0.72 |

### 13.7 Per-class loose @ BEST epoch (ep8)

| Class | Run 10 BEST | Run 13 @ep8 | Delta |
|---|---:|---:|---:|
| Car | **73.96** | 67.57 | −6.39 |
| Cyclist | **52.75** | 46.64 | −6.11 |
| Pedestrian | 28.74 | 26.84 | −1.90 |
| Truck | **53.16** | 45.23 | −7.93 |

### 13.8 Key findings

1. **Doppler 动静 mask 是显著负收益**：Overall 36.21 vs Run 10 已存 39.65，**−3.44**。这是所有 run 中最大的单次回退（此前最差为 Run 11 Truck-anchor 的 −1.20）。
2. **全线溃退，非零和**：与 anchor 实验不同，本轮不是「拆东墙补西墙」，而是 **BEV −5.02、Car strict −2.78、Cyc loose −6.11、Trk strict −4.16** 同时下降。唯一亮点是 Ped loose +2.74（但 strict 仍 ≈ 0）。
3. **峰值在 ep8 后再未恢复**：ep9 暴跌 4.5 点（36.21→31.72），此后 ep10–24 稳定在 34–36 但始终低于峰值。说明 mask 不是噪声抖动，而是系统性地压低了模型上限。
4. **根因推测**：velocity-only soft-mask（`gating = 1 + 0.5·score`）放大了动态点的速度维。但预训练权重的 PFN 是在原始速度分布上标定的，**velocity distortion 破坏了 pretrained backbone → BEV encoder 的特征对齐**。这解释了为什么 BEV_mod 跌得最狠（−5.02）——BEV 特征质量本身被拉低。
5. **Ped loose 微涨是唯一正面信号**：行人点 85% 是静态（score≈0），velocity 基本不变，但 score 拟合过程可能间接帮助了行人–背景分离（ego-velocity 估计剔除了静态杂波）。但这不足以补偿其他类的损失。
6. ✅ `checkpoint_interval=1` 生效，ep1–24 全部落盘。

### 13.9 Conclusions / Next

- **Doppler 动静 mask 当前版本不应采用**。主模型仍为 **Run 10 head-v2 ep14（39.65）**。
- 预训练 + velocity distortion 不兼容是核心矛盾。若要继续探索 Doppler, 建议：
  1. **`dynamic_weight=0.0`**：score 通道仍在（in_channels=6），但不对 velocity 做任何缩放——纯信息注入，不改变预训练对齐；
  2. **从头训练（不用 pretrained）**：让 PFN/BEV encoder 从零适应 velocity distortion，但会损失 pretrain 的 +3.23 基础增益；
  3. **ego-velocity 接入 RSSM `action_dim`**：不修改点云特征，而是把 `v_ego` 作为 RSSM transition 的 action 输入，与点级 mask 正交。
-- 单变量消融（`dynamic_weight=0.0` vs `0.5`）仍是确认根因的必要实验。

## 14. N=4 Pretrained RSSM + head-v2 + truncated BPTT (24e)

> Branch: `radar_static_dynamic`。基于 Run 10 的最优配置（N=4, hdim=128, pretrained, 24e），
> 唯一改动是把 RSSM 的随机状态接回计算图，以 `rssm_bptt_steps=1` 做 truncated BPTT。
> 为适配 3 卡显存，`samples_per_gpu=2`，并用 `cumulative_iters=2` 恢复有效 batch；
> 训练前还修了 `reset_for_samples` 的 in-place 操作，避免切断新序列首帧的 autograd。
> `checkpoint_interval=1`，epoch 1–24 全部落盘。

### Run 14: N=4 Pretrained RSSM + head-v2 + BPTT

- config: `configs/r4det/TJ4D-R4Det_motion_align_rssm_det3d_N4_2x4_24e_pretrained_v2_head_bptt.py`
- work_dir: `work_dirs/rssm_N4_2x4_24e_pretrained_v2_head_bptt`
- **频率加权 Overall（平台，last-5）**: **38.15 ± 0.68**（幅度提升来自 Car 权重 25%→48.4%；权重见 section 16）
- **全轮峰值：ep14 = 37.84（✅ 已落盘）**
- **LAST: ep24 = 34.90**
- Car strict BEST: 49.09 @ep10
- Ped loose BEST: 31.49 @ep14（历史最高）
- Truck strict BEST: 33.82 @ep14

### 14.1 Epoch curve: Overall 3D_moderate

| ep | mod | ep | mod | ep | mod |
|---:|---:|---:|---:|---:|---:|
| 1  | 16.04 | 9  | 31.56 | 17 | 36.52 |
| 2  | 22.97 | 10 | 36.73 | 18 | 36.80 |
| 3  | 28.95 | 11 | 34.13 | 19 | 35.60 |
| 4  | 30.64 | 12 | 34.63 | 20 | 35.35 |
| 5  | 31.55 | 13 | 36.43 | 21 | 36.00 |
| 6  | 30.92 | 14 | **37.84** | 22 | 35.63 |
| 7  | 34.27 | 15 | 35.43 | 23 | 35.15 |
| 8  | 33.45 | 16 | 36.43 | 24 | 34.90 |

- 起点明显低于 Run 10（ep1=16.04 vs 17.11），但 ep3 后追近并长期在 34–37 震荡。
- 峰值出现在 **ep14=37.84**，之后没有继续抬升，ep19–24 在 35.0–36.0 的平台收束。
- 最后 5 个 epoch 没有出现 Run 10 那样的高位恢复，终点反而下探到 34.90。

### 14.2 BEST vs LAST

| Metric | BEST (ep) | LAST (ep24) |
|---|---:|---:|
| Overall 3D_moderate | **37.84** (14) | 34.90 |
| Overall 3D_easy | 40.05 (14) | 37.53 |
| Overall 3D_hard | 36.30 (14) | 33.50 |
| Overall BEV_moderate | 45.84 (16) | 43.40 |
| Car 3D_mod_strict | 49.09 (10) | 40.48 |
| Cyclist 3D_mod_strict | 22.25 (15) | 19.96 |
| Pedestrian 3D_mod_strict | 0.97 (12) | 0.14 |
| Truck 3D_mod_strict | 33.82 (14) | 31.09 |
| Pedestrian 3D_mod_loose | 31.49 (14) | 28.70 |
| Truck 3D_mod_loose | 48.08 (20) | 45.70 |

### 14.3 vs Run 10（同 base，唯一差异=truncated BPTT）

| Metric | Run 10 BEST | Run 14 BEST | Δ |
|---|---:|---:|---:|
| Overall 3D_moderate | **39.65** (ep14) / 峰值 40.60 (ep11) | 37.84 (ep14) | **−1.81 / −2.76** ❌ |
| Overall 3D_easy | **42.49** (ep15) | 40.05 (ep14) | −2.44 |
| Overall 3D_hard | **39.06** (ep11) | 36.30 (ep14) | −2.76 |
| Overall BEV_moderate | **48.70** (ep13) | 45.84 (ep16) | −2.86 |
| Car 3D_mod_strict | **53.04** (ep20) | 49.09 (ep10) | −3.95 |
| Car 3D_mod_loose | **73.96** (ep12) | 73.27 (ep10) | −0.69 |
| Cyclist 3D_mod_strict | **25.41** (ep8) | 22.25 (ep15) | −3.16 |
| Cyclist 3D_mod_loose | **52.75** (ep12) | 46.82 (ep7) | −5.93 |
| Pedestrian 3D_mod_strict | 0.42 (ep2) | 0.97 (ep12) | 噪声级 |
| Pedestrian 3D_mod_loose | 28.74 (ep10) | **31.49** (ep14) | **+2.75** ✅ |
| Truck 3D_mod_strict | 33.23 (ep11) | **33.82** (ep14) | +0.59 ✅ |
| Truck 3D_mod_loose | **53.16** (ep18) | 48.08 (ep20) | −5.08 |

### 14.4 RSSM dynamics（节选）

| ep | kl_loss | recon_loss | clamped_ratio | post_std | prior_std |
|---:|---:|---:|---:|---:|---:|
| 1  | 0.00   | 0.0821 | 61.7% | 0.137 | 0.200 |
| 5  | 0.4124 | 0.0060 | 99.2% | 0.130 | 0.335 |
| 10 | 0.9092 | 0.0045 | 99.6% | 0.134 | 0.344 |
| 14 | 1.0036 | 0.0033 | 99.8% | 0.124 | 0.321 |
| 20 | 1.0006 | 0.0025 | 99.8% | 0.118 | 0.311 |
| 24 | 1.0004 | 0.0024 | 99.8% | 0.117 | 0.309 |

### 14.5 Key findings

1. **BPTT 当前设置没有带来提升**。Overall 全轮峰值 37.84（ep14），比 Run 10 best saved 39.65 低 **1.81**，比 Run 10 全轮峰值 40.60 低 **2.76**；Run 14 的 LAST 34.90 更比 Run 10 LAST 38.76 低 **3.86**。
2. **主要输在 Car、Cyclist、BEV**：Car strict −3.95、Cyclist strict −3.16、Cyclist loose −5.93、BEV_mod −2.86，和前面几轮 anchor/mask 实验不同，不是「某一类换另一类」，而是核心类别整体低一档。
3. **最晚 5 个 epoch 趋势不乐观**：从 ep14 峰值后回落到 35 平台，低学习率阶段也没有再回到 38+，说明这不是早期抖动，而是本配置的上限大约就在 37.8 附近。
4. **BPTT 没有破坏训练稳定性**：`loss` 从 ep1 的 ~1.79 平稳降到 ep24 的 ~1.27，`grad_norm` 从 15 收敛到 ~4.0，没有 NaN 或梯度爆炸。
5. **posterior collapse 依旧存在**：clamp 在 ep12 后稳定在 99.8%，`post_std` 和 `prior_std` 基本重合，随机状态贡献仍接近 0。`rssm_bptt_steps=1` 的一步 truncated BPTT 不足以改变这一结果。
6. **仅有的正面信号是 Ped loose 31.49**（Run 10 28.74，+2.75，历史最高）以及 Truck strict 33.82（与 Run 10 33.23 基本持平但低于 Run 11 的 35.03）。Ped strict 12 个 epoch 的 0.4 / 0.97 只是噪声，不是实质改善。
7. **结论**：以 Overall 3D_moderate 为口径，Run 14 / BPTT 是 **负收益**，不应替代当前主模型。主模型仍回退到 **Run 10 head-v2 ep14（39.65）**。

### 14.6 Conclusions / Next

- 当前版本的 single-step truncated BPTT 对 RSSM 随机状态的训练没有带来有效增益，且核心类别与 Overall 均弱于 Run 10。
- 如果继续沿 BPTT 方向实验，建议优先改变的是「窗口长度」而不是数据 pipeline：`rssm_bptt_steps=1` 的序列梯度窗口太小，可尝试 2–4 步；同时需要单独监控 posterior/prior 距离，避免把结论只建立在最终 AP 上。
- 另一个未验证且与 BPTT 正交的方向是**`dynamic_weight=0.0`** 的纯 Doppler 信息注入（不缩放 velocity），以及**ego-velocity 接入 RSSM `action_dim`**；这两者应在不混入 BPTT 变量的前提下单独跑。
- 本轮 peak `epoch_14.pth` 已保存，若后续继续 resume，应从这个 37.84 峰值点继续，而不是 LAST。

---

## 15. N2 Pretrained RSSM + head-v2 + truncated BPTT (24e)

> Branch: `radar_static_dynamic`。基于 Run 10 最优配置（head-v2，pretrained，24e），
> 引入 truncated BPTT（`rssm_bptt_steps=1`），并把时序长度从 N4 收窄到 N2、`hidden_dim` 从 128 降到 64。
> 这是 Run 14（N4 BPTT）的姊妹消融：对比 N4+hdim128 与 N2+hdim64 在相同 BPTT 设置下的表现，
> 用来隔离「是否值得为 BPTT 保留更深/更长的 RSSM」。其余配置与 Run 14 完全一致：
> `samples_per_gpu=2` 与 `cumulative_iters=2` 恢复有效 batch，`lr=1.5e-4`，`checkpoint_interval=1`，ep1–24 全部落盘。

### Run 15: N2 (seq_len=2, hdim=64) Pretrained RSSM + head-v2 + BPTT

- config: `configs/r4det/TJ4D-R4Det_motion_align_rssm_det3d_N2_2x4_24e_pretrained_v2_head_bptt.py`
- work_dir: `work_dirs/rssm_N2_2x4_24e_pretrained_v2_head_bptt`
- **频率加权 Overall（平台，last-5）**: **39.87 ± 0.45**（幅度提升来自 Car 权重 25%→48.4%；权重见 section 16）
- **全轮峰值：ep14 = 38.08**（已落盘）
- **LAST: ep24 = 37.08**
- Car strict BEST: 46.22 @ep12
- Cyclist strict BEST: 25.02 @ep10
- Truck strict BEST: 32.65 @ep21
- Pedestrian strict BEST: 2.70 @ep24

### 15.1 Epoch curve: Overall 3D_moderate

| ep | mod | ep | mod | ep | mod |
|---:|---:|---:|---:|---:|---:|
| 1 | 13.95 | 9 | 36.28 | 17 | 35.17 |
| 2 | 24.83 | 10 | 33.89 | 18 | 37.58 |
| 3 | 29.13 | 11 | 34.29 | 19 | 36.88 |
| 4 | 31.24 | 12 | 37.42 | 20 | 36.54 |
| 5 | 33.76 | 13 | 35.68 | 21 | 37.97 |
| 6 | 32.00 | 14 | **38.08** | 22 | 37.21 |
| 7 | 35.97 | 15 | 31.15 | 23 | 37.04 |
| 8 | 32.74 | 16 | 37.36 | 24 | 37.08 |

- 起步明显偏低（ep1=13.95，低于 Run 10 的 17.11 和 Run 14 的 16.04），但 ep2 快速追到 24.83，ep3 后基本与 Run 14 同轨。
- 峰值在 ep14（38.08），ep15 突降 6.9 点（31.15），随后 ep16–24 在 35–38 区间震荡，没有再突破峰值。
- 尾部 8 个 epoch 波动明显大于 Run 14，但最终 LAST（37.08）显著高于 Run 14 的 34.90。

### 15.2 BEST vs LAST

| Metric | BEST | epoch | LAST (ep24) |
|---|---:|---:|---:|
| Overall 3D_moderate | **38.08** | 14 | 37.08 |
| Overall 3D_easy | 40.33 | 14 | 39.45 |
| Overall 3D_hard | 36.67 | 14 | 35.59 |
| Overall BEV_moderate | 45.84 | 14 | 44.81 |
| Car 3D_mod_strict | 46.22 | 12 | 43.89 |
| Cyclist 3D_mod_strict | 25.02 | 10 | 23.26 |
| Pedestrian 3D_mod_strict | 2.70 | 24 | 2.70 |
| Truck 3D_mod_strict | 32.65 | 21 | 31.73 |

### 15.3 vs Run 14（N4 BPTT，同 base，同 BPTT，唯一差异=N/hdim）

| Metric | N4 BPTT BEST | N2 BPTT BEST | Delta |
|---|---:|---:|---:|
| Overall 3D_moderate | 37.84 (ep14) | **38.08** (ep14) | **+0.24** ✅ |
| Overall 3D_easy | 40.05 (ep14) | 40.33 (ep14) | +0.28 |
| Overall 3D_hard | 36.30 (ep14) | 36.67 (ep14) | +0.37 |
| Overall BEV_moderate | 45.84 (ep16) | 45.84 (ep14) | 持平 |
| Car 3D_mod_strict | **49.09** (ep10) | 46.22 (ep12) | −2.87 ❌ |
| Car 3D_mod_loose | 73.27 (ep10) | 69.47 (ep22) | −3.80 ❌ |
| Cyclist 3D_mod_strict | 22.25 (ep15) | 25.02 (ep10) | +2.77 ✅ |
| Cyclist 3D_mod_loose | 46.82 (ep7) | 49.15 (ep6) | +2.33 ✅ |
| Pedestrian 3D_mod_strict | 0.97 (ep12) | 2.70 (ep24) | +1.73 |
| Pedestrian 3D_mod_loose | 31.49 (ep14) | 31.38 (ep14) | −0.11 |
| Truck 3D_mod_strict | **33.82** (ep14) | 32.65 (ep21) | −1.17 ❌ |
| Truck 3D_mod_loose | 48.08 (ep20) | 49.09 (ep13) | +1.01 ✅ |

### 15.4 vs Run 10（head-v2，无 BPTT，主模型基线）

| Metric | Run 10 BEST | N2 BPTT BEST | Delta |
|---|---:|---:|---:|
| Overall 3D_moderate | **39.65** (ep14) / 峰值 40.60 (ep11) | 38.08 (ep14) | **−1.57 / −2.52** ❌ |
| Car 3D_mod_strict | **53.04** (ep20) | 46.22 (ep12) | −6.82 ❌ |
| Cyclist 3D_mod_strict | 25.41 (ep8) | 25.02 (ep10) | −0.39 |
| Pedestrian 3D_mod_loose | 28.74 (ep10) | 31.38 (ep14) | +2.64 ✅ |
| Truck 3D_mod_strict | 33.23 (ep11) | 32.65 (ep21) | −0.58 |

### 15.5 RSSM dynamics（节选，ep 末 iter）

| ep | kl_loss | recon_loss | post_std | prior_std | mu_diff² |
|---:|---:|---:|---:|---:|---:|
| 1  | 0.0000 | 0.0951 | 0.134 | 0.201 | 1.294 |
| 4  | 0.3291 | 0.0072 | 0.131 | 0.326 | 0.187 |
| 8  | 0.7140 | 0.0050 | 0.141 | 0.345 | 0.046 |
| 12 | 1.0098 | 0.0038 | 0.146 | 0.339 | 0.027 |
| 14 | 1.0054 | 0.0033 | 0.142 | 0.334 | 0.017 |
| 20 | 1.0015 | 0.0026 | 0.136 | 0.328 | 0.009 |
| 24 | 1.0010 | 0.0024 | 0.134 | 0.327 | 0.008 |

### 15.6 Key findings

1. **N2 BPTT 略优于 N4 BPTT**：Overall 38.08 vs 37.84（+0.24），且 LAST 37.08 远高于 N4 的 34.90，尾部更稳。就 BPTT 这条线而言，收窄到 N2+hdim64 无害反有小幅增益。
2. **但仍跑不过无 BPTT 的 Run 10**：比已存 39.65 低 1.57，比峰值 40.60 低 2.52；Car strict −6.82 是最大拖累，符合「BPTT 负收益」的整体结论。
3. **N2 与 N4 不是零和**：Cyclist strict +2.77 / loose +2.33，N2 明显更擅长小目标/自行车，而 N4 在 Car strict 更强（49.09 vs 46.22）。这与第 8 节「hdim128 需要 N4 解锁、N3 配 hdim64 更优」的容量-帧数匹配规律一致：低容量短序列对小目标/稀疏数据更友好。
4. **Ped loose 保持高位**：N2 31.38 与 N4 31.49 几乎持平，都是 Run 10（28.74）之上的历史高位，属于 BPTT 这条线的一贯表现，而不是 N/hdim 变量的贡献。
5. **posterior collapse 依旧存在**：`post_std` 与 `prior_std` 基本重合，`mu_diff²` 从 ep1 的 1.29 单调收敛到 ep24 的 0.008，随机状态逐步坍缩，`rssm_bptt_steps=1` 没有改变这一趋势。
6. ✅ `checkpoint_interval=1` 生效，ep1–24 权重全部落盘，latest 指向 ep24，峰值 ep14 已保存。
7. **Ped strict 尾段异动值得留意**：ep1–20 的 Ped 3D strict 基本在 0.03–0.87 的噪声带内，但 ep21=2.58、ep24=2.70 明显抬升，是历次 run 里第一次在训练末期出现非噪声级的行人 strict 信号（此前所有 run 均 ≤0.4）。仍需更多 epoch 确认，不能排除验证集抖动。

### 15.7 Conclusions / Next

- N2 BPTT 定性仍是负收益：综合 Overall 与核心 Car 均弱于 Run 10，不能替代当前主模型（Run 10 head-v2 ep14=39.65）。
- 但 N2 相对 N4 的 BPTT 消融给出明确信号：在 truncated BPTT 下，`seq_len=2 + hidden_dim=64` 的轻配置不输甚至略好于 `seq_len=4 + hidden_dim=128`，且 Cyclist 明显占优。若继续沿 BPTT 方向，轻量 N2 配置是更经济也更稳的起点。
- 与 Run 14 相同的未验证正交方向仍适用：`rssm_bptt_steps=2–4` 的更长 BPTT 窗口；`dynamic_weight=0.0` 纯 Doppler 信息注入；ego-velocity 接入 `action_dim`。
- 本轮 peak `epoch_14.pth` 已保存（38.08），latest 为 ep24（37.08）。

---

## 16. 频率加权 Overall 对比：重新排名的结果

> 前面的 `Overall_3D_moderate` 是**等权 1/4**（`kitti_utils/eval.py` 的默认聚合）。本节按训练集真实类别频率重新加权，权重（四类归一、排除 0.5% Other）：
> Car 48.37% / Cyclist 21.64% / Truck 16.46% / Pedestrian 13.53%。
> 即 `Overall_freq = 0.4837·Car_s + 0.2164·Cyc_l + 0.1646·Truck_s + 0.1353·Ped_l`（沿用同类「Car/Truck strict、Ped/Cyc loose」混合口径）。
> 工具：`python3 tools/summarize_run.py <work_dir> --weighted --tail 5`。

### 16.1 两种权重口径的平台（last-5，历史补充口径）对比

> **口径提示（2026-09-24）**：下表 `last-5` 是第 16 节当初的分析口径，不再作为后续主对照。
> 统一主对照请用 **ep12-16 固定窗口均值**；本表仅保留用于说明“等权 vs 频率加权”会改变排名。

| Run | 等权 mean±std | 频率加权 mean±std | 等权 BEST | 频率加权 BEST |
|---|---:|---:|---:|---:|
| Run 10 (head-v2) | 38.50 ± 0.27 | **43.51 ± 0.51** | 40.59 | **45.15** |
| Run 12 (Car-large) | 35.68 ± 0.34 | 40.23 ± 0.65 | 38.11 | 43.73 |
| Run 9 (pretrained 30e) | 35.98 ± 0.29 | 40.07 ± 0.47 | 37.94 | 41.93 |
| Run 11 (Truck×3) | 36.47 ± 0.81 | 39.88 ± 0.78 | 38.45 | 41.89 |
| Run 15 (N2 BPTT) | 37.17 ± 0.52 | 39.87 ± 0.45 | 38.08 | 40.96 |
| Run 13 (dynmask) | 35.16 ± 0.49 | 38.62 ± 1.15 | 36.21 | 41.51 |
| Run 14 (N4 BPTT) | 35.41 ± 0.43 | 38.15 ± 0.68 | 37.84 | 41.25 |
| N4 30e（无 pretrain） | 33.67 ± 0.35 | 34.14 ± 0.61 | 34.71 | 36.04 |
| N4 30e lr2e4 | 28.30 ± 1.22 | 30.28 ± 0.97 | 30.08 | 31.36 |

（均取各自最后 5 个 epoch；Run 9/N4 系列为 30e 训练，其余 24e。注意这会让不同 run 取到的
epoch 区间不完全一致，因此只适合同配方横向参考，不能替代 ep12-16 固定窗口。）

### 16.2 新结论

1. **排序会因口径改变，这是最关键的发现**：
   - 等权（1/4）：Run 10 > Run 15(N2) > Run 11 > Run 9 > Run 12 > Run 14 > Run 13。
   - 频率加权：Run 10 >> Run 12 > Run 9 > Run 11 ≈ Run 15(N2) > Run 13 > Run 14。
   - **Run 12（Car-large anchor）从第 5 名跳到第 2 名；Run 15（N2 BPTT）从第 2 名掉到第 5 名。** 原因纯粹是权重变化：Car 权重 25%→48.4%，谁的 Car strict 强谁就跃升。

2. **Run 10 在两种口径下都是第 1，但领先幅度含义不同**：等权下领先第 2（N2）约 0.9 个点；频率加权下领先第 2（Run 12）约 **3.3 个点**。Run 10 的 Car strict（约 52）碾压其它 run（43~46），所以频率加权把它跟第二名的差距拉得更大。

3. **Run 12（Car-large anchor）被严重低估了**：等权下它只有 35.68（第 5），会让人误判「Car-large anchor 没用」；频率加权下它是 40.23（第 2）。文档 12.6 原本判它「本轮仍是负收益」，这个结论在等权口径下成立，但在「Car 主导的实际部署场景」口径下是**错的方向**——如果你想优化 Car 召回/严格 AP，Run 12 的 Car-large anchor 才是正确选择。

4. **N2 BPTT 的「略优于 N4 BPTT」结论需要弱化**：等权下 N2 37.17 > N4 35.41（+1.76）；但频率加权下两者只差 1.72 且都被 Run 12 / Run 9 / Run 11 反超。N2 的「小目标友好」优势在 Cyclist/Ped 权重被压低后基本消失。

5. **幅度整体抬升 ~4~5 点是口径效应，不是真实性能变化**：频率加权后所有 run 的 Overall 都涨了约 4~5 点（因为 Car 权重翻倍，Car strict 是所有类里最高的）。这个抬升对任何 run 都一样，**不改变「谁比谁好」除了权重结构带来的重排**。不要把这个抬升误读成「频率加权让模型变强了」。

6. **决策建议**：先明确「你优化的是哪个目标」：
   - 平衡多类（发论文通用口径）：继续用等权 1/4，Run 10 第一。
   - Car 主导的部署场景：换频率加权，Run 12（Car-large anchor）值得重新评估，Run 10 仍最强但 Run 12 紧跟。
   - 涉及跨 run 主对照时，统一用 ep12-16 固定窗口均值；全轮峰值只作为上限单列。

### 16.3 落地

- 工具已支持 `--weighted`，以后每个新 run 跑完跑一条 `--weighted` 即得频率加权口径。
- 权重写死在 `tools/summarize_run.py` 的 `CLASS_PRIORS`，若数据集（train/val）类别分布变化需同步更新。

---

## 17. Checkpoint 保留规则与当前盘点

> 写于 2026-08-22，追加在文档末尾，便于每次实验后照着做。

### 17.1 保留规则（此后所有 run 统一执行）

- 每个 run 只保留 `best saved epoch` + `last epoch` 两个真实 `.pth` 文件，外加 `latest.pth` 符号链接。
- `best saved` 是磁盘上实际存在的权重中 val `Overall_3D_moderate`（或指定主指标）最高的 epoch。
  全轮峰值按本文件顶部统一口径单独记录，即使它没有权重也不得省略。
- 若全轮峰值 epoch 未落盘，回退为 val 最接近该峰值的已存 epoch（例如 Run 10 head-v2 保留 `epoch_12`）。
- `latest.pth` 必须指向 `last.pth`；删除中间 epoch 时**不得删除** `epoch_best`、`epoch_last`、`latest.pth`。
- 清理删除是破坏性操作，只 dry-run 确认无误后执行，default 不自动删。

### 17.2 当前盘点（2026/08/22）

已存在的 9 个历史 run 均已符合上述规则：每个目录只有 2 个 epoch checkpoint，`latest.pth` 均正常指向 last，无悬挂链接、无散落的 `.pth`。

其中两个历史 run 的最优 epoch 已不可恢复：

| Run | 原最优（丢失） | 当前保留 best | 备注 |
|---|---:|---:|---|
| Run 10 head-v2（`rssm_N4_2x4_24e_pretrained_v2_head`） | ep14 39.65 | ep12 39.26 | 当前最接近的已保存权重 |
| Run 9 pretrained 30e（`rssm_N4_2x4_30e_pretrained`） | ep19 37.94 | ep20 36.94 | 当前保留的是 30e best |

这两个丢失项会在多 seed 结束后重新确定新的 best 并补上记录。

### 17.3 不做删除的部分

- `checkpoints/pretrained_tj4d.pth`：所有 pretrained 训练的 `load_from` 输入，不可删。
- `work_dirs/run10_headv2_multiseed`：当前多 seed 实验在跑，中间的 epoch 实时保存，不在清理范围。

### 17.4 删除前的检查点

- 确认每个 run 的 `best_history` 与 `last` 均已写入本文件，并能从日志重新获得精确 val。
- 用 `python3 tools/summarize_run.py <work_dir> --tail 5` 再核一遍 best/last。
- 本次清理未执行删除，因为实际候选删除文件已为 0；后续清理由脚本 dry-run 后人工触发。

### 17.5 盘点（2026-09-24，已执行 6 个 run 的清理）

按 17.1 规则逐目录核对 `best saved`、`last`、`latest.pth` 指向，并扫描 `configs/`、`tools/`、`docs/`
里对 `*.pth` 的显式引用（下游 `load_from` / `resume_from`、诊断脚本 `--checkpoint`、文档固定选点）。
下表第 17.5.1 小节是 dry-run 结果；其中 6 个 run 经确认后已于 2026-09-24 执行删除（见 17.5.2），
其余目录仍未删除，留待后续人工确认。

排除项：

- `fgfull_N4_no2d_igdr_2x4_24e_seed2`：审计当时仍在运行（GPU 5/6/7），因此未纳入该轮清理；
  该 run 已于 2026-09-25 03:25 UTC 自然跑满 24e 退出（见 48.2），其 ep2-24 中间权重
  尚未清理，需在下一轮 checkpoint 清理中单独处理。
- `checkpoints/pretrained_tj4d.pth`：所有 pretrained 训练的 `load_from` 输入。
- `epoch_avg_12_14_16.pth`、`iter_100.pth`、`latest.pth` 符号链接：不在 `epoch_<N>.pth` 清理范围内。

依赖保护：以下中间 epoch 不是所在 run 的 best saved，但被仓库显式引用，必须保留。

| 被引用的文件 | 引用方 |
|---|---|
| `run10_headv2_multiseed/seed_0/epoch_16.pth` | `TJ4D-R4Det_ped_centerhead_stage1_3x2x2_12e.py`、`tools/diagnose_z_utilization.py` |
| `run10_headv2_multiseed/seed_1/epoch_14.pth` | `TJ4D-R4Det_ped_centerhead_stage1_3x2x2_12e_seed1.py`、`..._seed1_restore_ep15.py` |
| `run10_headv2_multiseed/seed_2/epoch_14.pth` | `TJ4D-R4Det_ped_centerhead_stage1_3x2x2_12e_seed2.py` |
| `ped_centerhead_stage1_3x2x2_12e_seed0/epoch_7.pth` | `..._stage2_dim_...py`、`..._highres_centerhead_3x2x2_3e_raw.py`、`..._6e_identity_lr1e-4.py`、`..._gradfix_3x2x2_3e_seed0.py` |
| `fgfull_N4_2x4_24e_seed0/epoch_16.pth` | `tools/diagnose_temporal_shortcut.py` |
| `fgfull_N4_2x4_24e_seed0/epoch_2.pth` | 文档第 47 节的断点续训固定记录（仅叙述引用，已于 17.5.2 释放） |

#### 17.5.1 可删除候选（dry-run，202 个文件，合计 99.2 GB）

| run | 现有 | 保留 | 删除数 | 删除 epoch | 可释放 | 状态 |
|---|---:|---|---:|---|---:|---|
| `run10_headv2_multiseed/seed_2` | 24 | ep14(best+依赖), ep24(last) | 22 | 1-13, 15-23 | 11.8 GB | **已执行** |
| `fgfull_N4_2x4_24e_seed0` | 12 | ep2(文档), ep16(best+依赖), ep24(last) | 9 | 4,6,8,10,12,14,18,20,22 | 7.1 GB | **已执行**（含 ep2） |
| `cyccls_branch_N4_2x4_24e_seed0` | 15 | ep12(best), ep24(last) | 13 | 10,11,13-23 | 7.1 GB | **已执行** |
| `fgfull_N3_h128_2x4_24e_seed0` | 10 | ep14(best), ep20(last) | 8 | 2,4,6,8,10,12,16,18 | 6.3 GB | **已执行** |
| `fgfull_N4_no2d_igdr_2x4_24e_seed0` | 11 | ep14(best), ep22(last) | 9 | 2,4,6,8,10,12,16,18,20 | 5.8 GB | **已执行** |
| `run10_headv2_multiseed/seed_0` | 12 | ep16(best+依赖), ep24(last) | 10 | 2,4,6,8,10,12,14,18,20,22 | 5.3 GB | **已执行** |
| `fgfull_N4_no2d_igdr_2x4_24e_seed1` | 10 | ep16(best), ep20(last) | 8 | 2,4,6,8,10,12,14,18 | 5.1 GB |
| `crossmodal_fusion_N4_2x4_24e_seed0` | 10 | ep16(best+last) | 9 | 7-15 | 4.8 GB |
| `run10_headv2_multiseed/seed_1` | 12 | ep12(best), ep14(依赖), ep24(last) | 9 | 2,4,6,8,10,16,18,20,22 | 4.8 GB |
| `cyccls_branch_N4_2x4_24e_multiseed/seed_2` | 10 | ep14(best), ep18(last) | 8 | 9-13,15-17 | 4.3 GB |
| `cyccls_branch_N4_2x4_24e_multiseed/seed_1` | 10 | ep15(best), ep22(last) | 8 | 13,14,16-21 | 4.3 GB |
| `shared_stem_N4_2x4_24e_seed0` | 10 | ep14(best), ep16(last) | 8 | 7-13,15 | 4.3 GB |
| `fgfull_N4_temporal_baseline_seed0` | 10 | ep18(best), ep20(last) | 8 | 2,4,6,8,10,12,14,16 | 4.3 GB |
| `rssm_kl0_N4_2x4_24e_seed0` | 10 | ep22(best), ep24(last) | 8 | 15-21,23 | 4.3 GB |
| `rssm_kl0_N4_2x4_24e/seed_2` | 10 | ep17(best), ep24(last) | 8 | 15,16,18-23 | 4.3 GB |
| `rssm_kl0_N4_2x4_24e/seed_1` | 10 | ep20(best), ep24(last) | 8 | 15-19,21-23 | 4.3 GB |
| `no_temporal_N4_2x4_24e_seed0` | 10 | ep20(best), ep24(last) | 8 | 15-19,21-23 | 3.3 GB |
| `ped_centerhead_stage1_3x2x2_12e_seed2` | 12 | ep1(best), ep12(last) | 10 | 2-11 | 1.8 GB |
| `ped_centerhead_stage1_3x2x2_12e_seed0` | 12 | ep7(best+依赖), ep12(last) | 10 | 1-6,8-11 | 1.8 GB |
| `ped_centerhead_stage1_3x2x2_12e_seed1` | 10 | ep6(best), ep10(last) | 8 | 1-5,7-9 | 1.5 GB |
| `ped_highres_centerhead_gradfix_3x2x2_6e_seed0` | 6 | ep6(best+last) | 5 | 1-5 | 1.0 GB |
| `ped_highres_centerhead_identity_3x2x2_6e_lr1e-4_seed0` | 6 | ep5(best), ep6(last) | 4 | 1-4 | 0.8 GB |
| `ped_highres_centerhead_gradfix_3x2x2_3e_seed0` | 3 | ep3(best+last) | 2 | 1,2 | 0.4 GB |
| `ped_highres_centerhead_3x2x2_3e_seed0_raw` | 3 | ep2(best), ep3(last) | 1 | 1 | 0.2 GB |
| `ped_centerhead_stage2_delta_ioufix_3x2x2_3e_seed0` | 3 | ep1(best), ep3(last) | 1 | 2 | 0.2 GB |

#### 17.5.2 已执行删除（2026-09-24，6 个 run）

按 17.1 规则保留 `best saved` + `last`，删除中间 epoch。best 分期**由日志逐 epoch 复算两遍**
（`tools/summarize_run.py` 与独立 raw-line 解析，并用四类构成项重算 `Overall_3D_moderate` 交叉验证），
确认每项 top-1 后再删；`latest.pth` 指向的 last 权重均保留。

| run | 保留 best saved | 保留 last | 删除 | 释放 |
|---|---|---|---|---|
| `run10_headv2_multiseed/seed_2` | **ep14 = 40.88** | ep24 = 37.91 | 22 | 11.76 GB |
| `fgfull_N4_2x4_24e_seed0` | **ep16 = 40.69** | ep24 = 38.54 | 10 | 7.93 GB |
| `cyccls_branch_N4_2x4_24e_seed0` | **ep12 = 40.64** | ep24 = 38.38 | 13 | 7.06 GB |
| `fgfull_N3_h128_2x4_24e_seed0` | **ep14 = 39.38** | ep20 = 36.38 | 8 | 6.34 GB |
| `fgfull_N4_no2d_igdr_2x4_24e_seed0` | **ep14 = 40.39** | ep22 = 38.71 | 9 | 5.76 GB |
| `run10_headv2_multiseed/seed_0` | **ep16 = 39.88** | ep24 = 37.71 | 10 | 5.35 GB |

合计删除 72 个文件，释放 44.19 GB；`/data` 可用空间 48 GB -> 92 GB。

best 判定易错点及本次核对结果：

- `fgfull_N3_h128_2x4_24e_seed0`：全轮峰值 40.28 @ ep9 **没有权重**，best saved 是 **ep14 39.38**；
  ep16 的 39.36 只差 0.02，已用两遍独立复算确认 ep14 胜出，未误留。
- `fgfull_N4_2x4_24e_seed0`：ep10 的 40.27 高于 last，但仍低于 ep16 的 40.69，故 best 取 ep16。
- 上述 run 的日志均无同 epoch 重复 val 记录（无重启双计），无需做 last-record-per-epoch 去重歧义。
- `fgfull_N4_2x4_24e_seed0/epoch_2.pth` 本次一并删除。它只出现在第 47 节的断点续训历史叙述里，
  不存在于任何 config 的 `load_from` / `resume_from`，删除后无代码依赖受影响。

删除后复验：

- 6 个目录均只剩 2 个 `.pth`，`latest.pth` 全部解析正常，无悬挂链接。
- 全仓库 `latest.pth` 悬挂检查通过（0 个）。
- 硬依赖复查：`run10_headv2_multiseed/seed_{0,1,2}` 被引用的 ep16/ep14/ep14、
  `ped_centerhead_stage1_3x2x2_12e_seed0/epoch_7.pth`、`fgfull_N4_2x4_24e_seed0/epoch_16.pth` 均仍在位。

说明：

- 上表 `best` 一律指 `best saved`（实际有权重的 val 最高点），不是全轮峰值；全轮峰值另见各 run 正文，可能没有权重。
- `last` 指 val 曲线最后一个 epoch。部分 run 的 `latest.pth` 指向的 checkpoint 序号大于最后一个 val epoch（例如 `fgfull_N3_h128` 的 ep20 对 val ep21、`fgfull_N4_no2d_igdr_2x4_24e_seed1` 的 ep20 对 val ep21、`ped_centerhead_stage1_3x2x2_12e_seed1` 的 ep10 对 val ep9、`cyccls_branch_..._multiseed/seed_2` 的 ep18 对 val ep17），按 17.1 规则保留磁盘上的 last 权重。
- 已合规、无需处理的目录（仅 best saved + last）：`deterministic_latent_N4_2x4_24e_seed0`、`ped_centerhead_stage2_delta_3x2x2_3e_seed0`、`ped_centerhead_stage2_dim_3x2x2_6e_seed0`、`ped_highres_centerhead_3x2x2_3e_prior_smoke`、`ped_highres_centerhead_3x2x2_3e_seed0_smoke`、`ped_refine_N4_2x4_24e_seed0`、`pedrot4_N4_2x4_24e_seed0`、`posterior_only_learnable_std_N4_2x4_24e_seed0`、`rssm_N2_2x4_24e_pretrained_v2_head_bptt`、`rssm_N4_2x4_24e_pretrained_v2_head`、`rssm_N4_2x4_24e_pretrained_v2_head_bptt`、`rssm_N4_2x4_24e_pretrained_v2_head_dynmask`、`rssm_N4_2x4_24e_pretrained_v2_head_truck`、`rssm_N4_2x4_24e_pretrained_v2_head_truck_car2`、`rssm_N4_2x4_30e`、`rssm_N4_2x4_30e_lr2e4`、`rssm_N4_2x4_30e_pretrained`、`rssm_N4_fixednoise_posterior_seed0`、`seed1_ep15_restore`、`truck_tower_N4_2x4_24e_seed0`。
- 非 checkpoint 的可回收项（未计入上表）：`diag_dumps/` 约 545 MB、各 run `figures_path/` 合计约 3.0 GB。这些是诊断/可视化中间产物，需按项目需要单独确认。

#### 17.5.3 第二轮删除（2026-10-03，29 个 run，185 个文件，88.55 GB）

触发原因：`/data` 已用 94%（剩 36 GB）。执行前逐 run 复核：best 判定按日志时间序合并
（每 epoch 取最后一次 val，覆盖 MAI seed0 的三段重启日志）、在落盘权重集合内取 argmax
独立复算两遍，并用四类构成项重算 `Overall_3D_moderate` 交叉验证；保留集合与 17.5.1
待删行逐行比对一致，另覆盖 17.5.1 明确留给下一轮的 `no2d_igdr seed2` 与 2026-09-24
之后完成的 run（MAI seed0/1/2、18e 主线、`lowdim_z_*` 五个 24e run）。全仓库
`configs/ tools/ me_rssm/` 的 `load_from` / `resume_from` / `--checkpoint` 显式引用
扫描确认无待删文件被引用。

依赖保护（本轮新增一条，其余沿用 17.5 规则）：

| 被引用的文件 | 引用方 | 处理 |
|---|---|---|
| `run10_headv2_multiseed/seed_1/epoch_14.pth` | `TJ4D-R4Det_ped_centerhead_stage1_3x2x2_12e_seed1.py`（load_from）、`..._seed1_restore_ep15.py`（resume_from） | 非 best 仍保留 |

| run | 保留 best saved | 保留 last | 删除 epoch | 释放 |
|---|---|---|---|---:|
| `fgfull_N4_no2d_igdr_2x4_24e_seed2` | ep14 = 41.9586 | ep24 | 2-12, 16-22 | 6.40 GB |
| `fgfull_N4_no2d_igdr_2x4_24e_seed1` | ep16 = 42.2904（全轮峰值 ep15 43.1449 无权重） | ep20 | 2-14, 18 | 5.12 GB |
| `crossmodal_fusion_N4_2x4_24e_seed0` | ep16 = 39.2835（best+last） | — | 7-15 | 4.81 GB |
| `run10_headv2_multiseed/seed_1` | ep12 = 39.0503 + ep14（依赖） | ep24 | 2-10, 16-22 | 4.81 GB |
| `fgfull_N4_no2d_igdr_mai_2x4_24e_seed0` | ep16 = 41.4160（best+last） | — | 2-14 | 4.53 GB |
| `fgfull_N4_no2d_igdr_2x4_18e_seed0` | ep16 = 40.3094 | ep18 | 2-14 | 4.48 GB |
| `cyccls_branch_..._multiseed/seed_1` | ep15 = 39.7104 | ep22 | 13,14,16-21 | 4.34 GB |
| `cyccls_branch_..._multiseed/seed_2` | ep14 = 40.1627 | ep18 | 9-13,15-17 | 4.34 GB |
| `shared_stem_N4_2x4_24e_seed0` | ep14 = 40.3045 | ep16 | 7-13, 15 | 4.33 GB |
| `fgfull_N4_temporal_baseline_seed0` | ep18 = 37.6587（峰值 ep11 39.5431 无权重） | ep20 | 2-16 | 4.33 GB |
| `rssm_kl0_N4_2x4_24e_seed0` | ep22 = 40.3481 | ep24 | 15-21, 23 | 4.28 GB |
| `rssm_kl0_N4_2x4_24e/seed_1` | ep20 = 41.0994 | ep24 | 15-19, 21-23 | 4.28 GB |
| `rssm_kl0_N4_2x4_24e/seed_2` | ep17 = 36.7634 | ep24 | 15,16,18-23 | 4.28 GB |
| `fgfull_N4_no2d_igdr_mai_2x4_24e_seed1` | ep6 = 42.7727 | ep16 | 2,4,8-14 | 3.88 GB |
| `fgfull_N4_no2d_igdr_mai_2x4_24e_seed2` | ep12 = 41.3163 | ep16 | 2-10, 14 | 3.88 GB |
| `no_temporal_N4_2x4_24e_seed0` | ep20 = 35.5903 | ep24 | 15-19, 21-23 | 3.33 GB |
| `lowdim_z_recon_ht_3x2x2_24e_seed0` | ep12 = 33.6799（峰值 ep13 34.7784 无权重） | — | 2-10 | 2.65 GB |
| `lowdim_z_innovation_rawnorm_oc_3x2x2_24e_seed0` | ep12 = 34.9091（峰值 ep7 35.9549 无权重） | — | 2-10 | 2.65 GB |
| `lowdim_z_future_consistent_3x2x2_24e_seed0` | ep6 = 34.4123（峰值 ep9 35.6352 无权重） | ep12 | 2,4,8,10 | 2.11 GB |
| `ped_centerhead_stage1_3x2x2_12e_seed0` | ep7 = 39.8520（依赖+best） | ep12 | 1-6, 8-11 | 1.83 GB |
| `ped_centerhead_stage1_3x2x2_12e_seed2` | ep1 = 39.6863 | ep12 | 2-11 | 1.83 GB |
| `ped_centerhead_stage1_3x2x2_12e_seed1` | ep6 = 38.1387 | ep10 | 1-5, 7-9 | 1.46 GB |
| `lowdim_z_innovation_rawnorm_3x2x2_24e_seed0` | ep6 = 32.8980（峰值 ep5 33.8487 无权重） | — | 2,4 | 1.06 GB |
| `lowdim_z_innovation_routeB_3x2x2_24e_seed0` | ep6 = 31.4895（峰值 ep5 34.8551 无权重） | — | 2,4 | 1.06 GB |
| `ped_highres_centerhead_gradfix_3x2x2_6e_seed0` | ep6 = 39.7887（best+last） | — | 1-5 | 0.96 GB |
| `ped_highres_centerhead_identity_3x2x2_6e_lr1e-4_seed0` | ep5 = 39.4565 | ep6 | 1-4 | 0.77 GB |
| `ped_highres_centerhead_gradfix_3x2x2_3e_seed0` | ep3 = 39.6302（best+last） | — | 1,2 | 0.38 GB |
| `ped_highres_centerhead_3x2x2_3e_seed0_raw` | ep2 = 32.7200 | ep3 | 1 | 0.19 GB |
| `ped_centerhead_stage2_delta_ioufix_3x2x2_3e_seed0` | ep1 = 39.8566 | ep3 | 2 | 0.18 GB |

删除后复验全部通过：29/29 目录只剩预定保留集合；全仓库 `latest.pth` 悬挂检查 0 个；
5 个依赖权重（run10 三 seed 的 ep16/ep14/ep14、stage1 seed0 ep7、`fgfull_N4_2x4_24e_seed0`
ep16）全部在位。`/data` 可用空间 36 GB -> 132 GB（已用 94% -> 77%）。

范围说明：本轮只做中间 epoch 清理（17.5.1 待删行 + 其后完成的 run）。已停止分支的
整目录删除（`rssm_N4_2x4_30e*`、`pretrained_v2_head*` 变体等，合计约 24 GB）与
`diag_dumps/`、`figures_path/` 仍待后续单独确认；`mci_smoke_450iter` 为进行中的
MCI 冒烟 run，未纳入。

#### 17.5.4 冒烟 / 诊断目录清理（2026-10-13，44 个权重文件 + `diag_dumps/`，约 14.3 GB）

触发原因：`/data` 已用 89%。本轮清理**不涉及任何正式 run 的 `epoch_<N>.pth`**，
范围仅为 450/100-iter 冒烟与诊断中间产物；`iter_*.pth` 按 17.5 排除项（"不在
`epoch_<N>.pth` 清理范围内"）处理，且 §68.8j 的七消融机制数值、各 S0/wiring 判定、
§79 屏蔽对照产物均已记录。

**处理方式（与"整目录删除"的一处偏离，说明理由）**：只删每个冒烟目录里的权重
（`iter_450.pth` / `iter_2.pth` / `iter_100.pth` + `latest.pth` 链接），保留
`*.log` / `*.log.json` / 探针日志 / `figures_path/` 等 KB 级证据文件。原因：
`mci_smoke_450iter/` 存有 §67/68 机制探针的原始 `probe_w*.log`（8 份），
`c1valid_smoke_450iter/` 存有 §79 明确列为产物的 `20261012_084202.log[.json]`；
整目录删除会销毁这两条证据链的原始文件，而它们合计不足 10 MB。

| 目录组 | 删除 | 保留 |
|---|---|---|
| `mci_abl_*_smoke` ×7 | `iter_450.pth` + 链接（14 文件） | 日志、`figures_path/` |
| `dcrssm{1,2}_smoke_450iter`、`dcrssm2_smoke_450iter_v2` | 同左（6 文件） | 日志、`canvases.pt`、ego 诊断 JSON |
| `semA_smoke450_seed0`、`semA2S/semA2R_smoke450_seed0`、`c1valid_smoke_450iter` | 同上（含 `iter_2.pth`；10 文件） | 日志（含 §79 产物） |
| `mci_smoke_450iter` | `iter_450.pth` + 链接 | 8 份 `probe_*.log` |
| `lowdim_z_smoke_100iter{,_latent_lossfix,_stable_kl}`、`fgfull_smoke_100iter{,_patched}`、`ped_highres_*_smoke` ×2、`ped_highres_geometry_probe_smoke{,2,3,4}` | 同左（14 文件） | 日志 |
| `diag_dumps/` | 整目录（545 MB，per-sample pkl） | — |

- 合计删除 44 个权重文件 13.76 GB + `diag_dumps/` 0.545 GB ≈ **14.3 GB**；
  `/data` 可用 64 GB -> 78 GB（已用 89% -> 86%），`work_dirs` 116 GB -> 102 GB。
- 依赖保护核查：本轮目录均不在 17.5 依赖表内；`overnight_after_seed2.sh`、
  `c1valid_s0_verdict.py`、`probe_z_utilization_gated.py` 对 `iter_450.pth` 的引用
  属历史一次性队列/示例用法，若重跑会重新生成冒烟权重。
- 复验：全仓库 `latest.pth` 悬挂 0 个；冒烟目录下 `.pth` 残留 0 个；5 个依赖权重
  （semA2R ep6、B0 ep14、dcrssm2 ep4、run10 seed_1 ep14、stage1 seed0 ep7）与
  进行中的 `c1valid_seed0` 8 个 epoch 全部在位。
- 遗留项（未动）：`lowdim_z_future_consistent_..._INVALID_loss_wiring_bug`（552 MB，
  作废 run）、`ped_highres_geometry_probe_crop9_3e_seed0`（210 MB，探针 dump）；正式
  run 中间 epoch 清理（MCI/DCRSSM/semA 族，约 30 GB）与已停止分支整目录删除
  （约 24 GB）均未执行。

---

## 18. 论文最终报告口径（定稿）

> 初写于 2026-08-22；2026-09-24 按本文件顶部的统一记录口径重写。

- 论文主表：每个代表性 run 同时报告两项（等权 1/4 口径，`pts_bbox/KITTI/Overall_3D_moderate`）：
  **ep12-16 固定窗口均值**（主对照）和**全轮峰值 + epoch**（上限）。两项不可互相替代。
- 补充材料 / appendix：报告 deterministic 多 seed 的 `mean ± std`、全轮峰值分布，以及
  ep12-16 窗口的逐 epoch 曲线。用于回应「峰值是否 cherry-pick」「平台是否可复现」的审稿追问。
- `best saved` 只用于说明哪个 checkpoint 可复评/可发布，不进入性能排名；若全轮峰值未落盘，
  正文必须显式写出 `peak 未存 / best saved = epX`。
- 最终多 seed 一律以 `--deterministic` 运行为准；非 deterministic 的历史 run 只作为先导数据，不进入最终口径表中。
- `latest.pth` 保留指向 last epoch；每个 run 最终只保留 `best epoch` + `last epoch` 两个权重和 `latest.pth` 符号链接（见第 17 节）。
- 当前主方法 Baseline/Reference 记为 Run 10 head-v2；后续若多 seed 产生新的 best，再以新 best 更新第 10 节和本节。

## 19. Run 10 head-v2 三 seed 复现 (Multi-Seed Reproduction)

> 日期：2026-08-21 ~ 2026-08-24。以 Run 10 配置（N=4, hdim=128, pretrained, head-v2, 24e, lr=1.5e-4）为基础，
> 用三个不同随机种子重跑，验证单次 Run 10 峰值 40.60（ep11，原运行未存盘）的可复现性和方差。
> 三个 seed 之间配置完全相同，唯一差异是随机种子、work_dir、以及 checkpoint_interval
> （seed_0/seed_1=2，seed_2=1）。这一组整体相对原 Run 10 另有 batch 口径差异，见下表。
> 硬件：3 卡 GPU 5/6/7（原 Run 10 为 4 卡）。本组为 `samples_per_gpu=2` + `cumulative_iters=2`，
> 有效 batch 12；原 Run 10 为 `samples_per_gpu=4` 且**没有梯度累积**，有效 batch 16。

### 19.0 实验设置

| 项 | seed_0 | seed_1 | seed_2 | 原 Run 10 |
|---|---|---|---|---|
| config | 同 Run 10 配方，但 batch 口径改为 3 卡 spg2+累积2 | 同 seed_0 | 同 seed_0 | 4 卡 spg4、无累积 |
| work_dir | `run10_headv2_multiseed/seed_0` | `.../seed_1` | `.../seed_2` | `rssm_N4_2x4_24e_pretrained_v2_head` |
| GPUs | 5,6,7 (3卡) | 5,6,7 (3卡) | 5,6,7 (3卡) | 4卡 |
| samples_per_gpu | 2 | 2 | 2 | 4 |
| cumulative_iters | 2 | 2 | 2 | 无（无累积 hook） |
| 有效 batch | 12 | 12 | 12 | 16 |
| iter / epoch | 1902 | 1902 | 1902 | 714 |
| optimizer step / epoch | 951 | 951 | 951 | 714 |
| 24e 总 optimizer step | 22824 | 22824 | 22824 | 17136 |
| checkpoint_interval | 2 | 2 | 1 | 2 |
| load_from | pretrained_tj4d.pth | pretrained_tj4d.pth | pretrained_tj4d.pth | pretrained_tj4d.pth |
| seed | 0 (default) | 1 | 2 | 0 (default) |

seed_0 与原 Run 10 用的是同一 default seed（0），但两组的 batch 组成不同：多 seed 组靠 3 卡 ×
spg 2 × 累积 2 凑出有效 batch 12，原 Run 10 是 4 卡 × spg 4、无累积凑出 16。因此卡数、
单卡 batch 与是否累积三项都不同，BN 统计量与梯度累积路径随之变化，严格说不完全等价。

迭代数差异（均由磁盘日志实测，`Epoch [e][i/total]` 字段）：

- 训练 iter 数 **714 vs 1902（2.66×）**，因为 iter 由 `samples/epoch ÷ (卡数×spg)` 决定，
  累积不减少 iter。
- optimizer step 数 **714/epoch vs 951/epoch（1.33×）**，24e 累计 **17136 vs 22824**。
  这个 1.33 就是有效 batch 的比值 16/12，即小 batch 用更多步换掉每步样本量。
- 每 epoch 覆盖样本量几乎相同（11412 vs 11424，差 0.1%，来自 1902 与 714 的向上取整），
  所以两者是**同一数据预算、不同拆解方式**，不是数据量差异。
- CosineAnnealing 按 epoch 调度，逐 epoch lr 两边完全一致（24/24 epoch 无差异），
  因此每个 epoch 的 lr 相同，但同一 lr 下多 seed 组做了更多次参数更新。
- 两组还差一项：原 Run 10 是 `deterministic: False`（默认），多 seed 组是
  `deterministic: True`（`--deterministic`；三个 seed 日志均为 True）。
- BN 差异是累积无法弥补的那一类，需要分开看两层：
  - **归一化统计量**：模型用普通 BatchNorm（非 SyncBN，全仓库 `SyncBN` 命中 0），DDP 下
    每个进程只对自己**本卡 micro-batch**做归一化。于是 Run 10 的 BN 统计基于 4 样本、
    多 seed 组基于 2 样本；累积只把梯度加到 12，不会把两个 micro-batch 拼进同一次 BN 计算。
  - **running stats 更新**：BN 的 running stats 每个 micro-step 各更新一次，累积 2 次即每步
    更新两遍，等效统计窗口与无累积不同。实测确认参数被设 `requires_grad=False` 的 BN 在
    `train()` 下**依然**更新 running stats（PyTorch 验证），所以「冻结权重」不等于「冻结统计量」。
  - 涉及范围：只有 `img_backbone`（ResNet，`frozen_stages=1` + `norm_eval=True`）因
    `norm_eval` 显式把内部 BN 置为 eval、统计量冻结。`img_neck`（FPN）**没有** `norm_eval`，
    它的 18 个 BN 虽然 `requires_grad=False`，仍处于训练态并更新 running stats；
    `pts_voxel_encoder`（PFNLayer 的 BatchNorm1d）、`pts_backbone`（SECOND）、
    RSSM fusion（`requires_grad=True`）同样是训练态 BN。这些都会随 micro-batch 大小
    与累积次数改变行为，两块配置的差异无法抵消。
  - mmcv 对这类组合本身有告警：`GradientCumulativeOptimizerHook may slightly decrease
    performance if the model has BatchNorm layers.`，多 seed 组日志中确有该条。

  结论：所谓「有效 batch 12」只是**梯度层面的等价**，不等于与原 Run 10 的 BN 行为等价。

严格讲，这**不是单变量对照**：相对原 Run 10，多 seed 组同时改变了卡数（4→3）、单卡 batch
（spg 4→2）、是否累积（无→2）、deterministic（False→True）四项；相对第 20 节及其后的
各 ablation 才是干净的单变量对照（那些 run 全部沿用 3 卡 / spg2 / 累积2 / deterministic）。

### 19.1 Overall 3D_moderate 逐 epoch 曲线

| ep | seed_0 | seed_1 | seed_2 | mean | std |
|---:|---:|---:|---:|---:|---:|
|  1 | 17.35 | 17.21 | 17.44 | 17.33 | 0.12 |
|  2 | 24.45 | 29.33 | 23.56 | 25.78 | 3.11 |
|  3 | 31.11 | 31.30 | 22.30 | 28.24 | 5.14 |
|  4 | 28.93 | 33.54 | 26.87 | 29.78 | 3.42 |
|  5 | 32.81 | 32.56 | 32.95 | 32.77 | 0.20 |
|  6 | 33.51 | 37.47 | 33.55 | 34.85 | 2.28 |
|  7 | 37.48 | 38.47 | 34.54 | 36.83 | 2.04 |
|  8 | 34.36 | 36.64 | 34.53 | 35.17 | 1.27 |
|  9 | 36.19 | 36.18 | 36.64 | 36.34 | 0.27 |
| 10 | 36.21 | 34.88 | 38.05 | 36.38 | 1.59 |
| 11 | 37.16 | 34.46 | 37.17 | 36.26 | 1.56 |
| 12 | 37.79 | 39.05 | 38.61 | 38.48 | 0.64 |
| 13 | 38.72 | 39.56 | 37.96 | 38.75 | 0.80 |
| 14 | 38.91 | 38.05 | **40.88** | 39.28 | 1.45 |
| 15 | 36.65 | **40.51** | 40.18 | 39.11 | 2.14 |
| 16 | **39.88** | 38.83 | 39.36 | 39.36 | 0.52 |
| 17 | 38.29 | 39.79 | 38.97 | 39.02 | 0.75 |
| 18 | 37.90 | 38.87 | 37.40 | 38.05 | 0.75 |
| 19 | 37.37 | 38.11 | 39.27 | 38.25 | 0.96 |
| 20 | 37.53 | 38.85 | 38.69 | 38.36 | 0.72 |
| 21 | 37.19 | 38.69 | 38.06 | 37.98 | 0.75 |
| 22 | 37.93 | 38.93 | 38.56 | 38.47 | 0.51 |
| 23 | 37.68 | 37.30 | 37.72 | 37.57 | 0.23 |
| 24 | 37.71 | 38.97 | 37.91 | 38.20 | 0.67 |

加粗为各 seed 的 BEST epoch。

### 19.2 全轮峰值与已存最高点（两套口径分开）

全轮峰值 = 每个 seed 所有 val epoch 的最大 Overall；best saved = 磁盘有 `epoch_X.pth` 的
epoch 中 val 最高的点。seed_0/seed_1 的 `checkpoint_interval=2`，seed_2 为 1。

| seed | 全轮峰值 @ep | 峰值 3D_mod | best saved @ep | best saved 值 | 说明 |
|---|---:|---:|---:|---:|---|
| seed_0 | ep16 | 39.8802 | ep16 | 39.8802 | 峰值已落盘 |
| seed_1 | ep15 | 40.5073 | ep12 | 39.0503 | **ep15 为奇数 epoch，峰值未落盘** |
| seed_2 | ep14 | **40.8800** | ep14 | 40.8800 | 峰值已落盘 |
| **三 seed 均值 ± std** | | **40.42 ± 0.51** | | 39.94 ± 0.92 | 后者不作为性能对照 |

- 三个 seed 的全轮峰值分别落在 ep14/15/16，集中在 ep12-16 高原区（而非原 Run 10 的 ep11）。
- 全轮峰值方差极小：std=0.51，三个值在 39.88-40.88 的 1 点区间内。说明 Run 10 的 40+ 水平
  是稳定的，不是 cherry-pick。
- seed_2 是三 seed 中全轮峰值最高的（40.88），也超过了原 Run 10 未落盘的峰值 40.60。
- **此前 19.2/19.8 把 seed_1 ep15 记为“已存”是错的**：该目录 `checkpoint_interval=2`，
  磁盘只有偶数 epoch 权重；ep15 峰值仍按统一口径保留，但可用权重回退到 ep12。

### 19.2b ep12-16 固定窗口均值（comparison mean，主对照口径）

各 seed 在 ep12-16 五个 epoch 上等权平均，Overlap 构成项同前：

| 指标 | seed_0 | seed_1 | seed_2 | 三 seed 均值 ± std |
|---|---:|---:|---:|---:|
| Overall 3D moderate | 38.3902 | 39.1994 | 39.3990 | **38.9962 ± 0.5342** |
| Overall BEV moderate | 46.5193 | 46.8477 | 48.3350 | 47.2340 ± 0.9675 |
| Car strict | 47.8027 | 45.3952 | 48.9752 | 47.3910 ± 1.8252 |
| Cyclist loose | 48.6271 | 47.6034 | 47.9939 | 48.0748 ± 0.5166 |
| Pedestrian loose | 28.9160 | 30.1997 | 30.5844 | 29.9000 ± 0.8736 |
| Truck strict | 28.2149 | 33.5993 | 30.0424 | 30.6189 ± 2.7381 |

说明：三 seed 的全轮峰值 mean `40.42` 是上限口径，不能替代这里的窗口均值 `39.00`；
后续 20-25 节及 CycCls 等对照使用的 `38.39 / 39.20 / 39.40` 正是本表逐 seed 的 Overall 窗口均值。

### 19.3 BEST epoch 逐类别对比

| Metric | seed_0 (ep16) | seed_1 (ep15) | seed_2 (ep14) | mean | std |
|---|---:|---:|---:|---:|---:|
| Overall 3D_moderate | 39.88 | 40.51 | 40.88 | **40.42** | 0.51 |
| Overall 3D_easy | 42.33 | 43.21 | 43.70 | 43.08 | 0.69 |
| Overall 3D_hard | 38.37 | 38.88 | 39.37 | 38.88 | 0.50 |
| Overall BEV_moderate | 47.70 | 48.13 | 49.85 | 48.56 | 1.14 |
| Car 3D_mod_strict | 49.98 | 47.49 | 51.35 | 49.60 | 1.95 |
| Car 3D_mod_loose | 69.20 | 72.57 | 75.77 | 72.52 | 3.28 |
| Cyclist 3D_mod_strict | 23.17 | 26.64 | 28.07 | 25.96 | 2.52 |
| Cyclist 3D_mod_loose | 50.47 | 49.36 | 50.09 | 49.97 | 0.57 |
| Pedestrian 3D_mod_strict | 0.10 | 0.15 | 0.12 | 0.12 | 0.03 |
| Pedestrian 3D_mod_loose | 28.64 | 30.39 | 31.32 | 30.12 | 1.36 |
| Truck 3D_mod_strict | 30.43 | 34.79 | 30.76 | 31.99 | 2.43 |
| Truck 3D_mod_loose | 49.44 | 51.58 | 52.01 | 51.01 | 1.38 |

### 19.4 固定 ep14 截面对比（与原 Run 10 best saved 同一 epoch）

| Metric | seed_0 | seed_1 | seed_2 | mean | std | 原 Run 10 |
|---|---:|---:|---:|---:|---:|---:|
| Overall 3D_moderate | 38.91 | 38.05 | 40.88 | 39.28 | 1.45 | 39.65 |
| Overall 3D_easy | 41.57 | 40.83 | 43.70 | 42.03 | 1.49 | 41.58 |
| Overall 3D_hard | 37.42 | 36.51 | 39.37 | 37.77 | 1.46 | 38.14 |
| Overall BEV_moderate | 46.40 | 46.24 | 49.85 | 47.50 | 2.04 | 47.50 |
| Car 3D_mod_strict | 48.97 | 43.55 | 51.35 | 47.95 | 4.00 | 52.36 |
| Car 3D_mod_loose | 67.87 | 70.09 | 75.77 | 71.25 | 4.07 | 72.97 |
| Cyclist 3D_mod_strict | 17.42 | 24.77 | 28.07 | 23.42 | 5.45 | 21.01 |
| Pedestrian 3D_mod_strict | 0.22 | 0.13 | 0.12 | 0.15 | 0.06 | 0.11 |
| Truck 3D_mod_strict | 27.93 | 33.64 | 30.76 | 30.78 | 2.85 | 30.26 |

- 固定 ep14 截面的 mean (39.28) 略低于原 Run 10 (39.65)，但在 1 std 以内。
- seed_2 在 ep14 就是 BEST (40.88)，拉高了均值；seed_0/seed_1 的峰值在 ep15/16 才出现。
- Car strict 在固定 ep14 的方差很大 (std=4.00)，seed_2 最高 (51.35)，seed_1 最低 (43.55)——说明 Car strict 对种子和收敛时机更敏感。

### 19.5 与原 Run 10 对比

| 口径 | 原 Run 10 | 三 seed 复现 | 结论 |
|---|---:|---:|---|
| 全轮峰值 | 40.60 (ep11, 未存) | 40.42 +/- 0.51 | 原 Run 10 在 1 std 内，非离群 |
| ep12-16 窗口均值 | 39.06 | 39.00 +/- 0.53 | 平台水平一致 |
| ep14 固定截面 | 39.65 | 39.28 +/- 1.45 | 在 1 std 内，基本一致 |
| best saved | 39.65 (ep14) | seed_0 39.88 / seed_1 **39.05 (ep12)** / seed_2 40.88 | seed_1 峰值未存，已存最高低于峰值 |
| 全轮峰值 epoch | ep11 | ep14/15/16 | 峰值后移 3-5 epoch |

- 原 Run 10 的 ep11=40.60 不是 cherry-pick：三 seed mean 40.42，std 0.51，40.60 在 1 std 以内。
- BEST epoch 从 ep11 后移到 ep14-16，方向上与「有效 batch 从 16 降到 12、需要更多 epoch 收敛」
  一致，但两组同时变了卡数（4→3）、单卡 batch（spg 4→2）和累积设置（无→2），三者无法解耦，
  因此这里只能记为**可疑原因**，不能断言是单变量效应。
- seed_0/seed_2 的全轮峰值已落盘；seed_1 的 ep15 峰值因 `checkpoint_interval=2` 未落盘，
  可用 best saved 是 ep12。原 Run 10 ep11 未存的问题在本组只修复了 2/3。

### 19.6 收敛形态分析

- ep1-4（起步）：三个 seed 高度一致（ep1 std=0.12），pretrained 初始化保证了稳定的起点。
- ep2-4（快速增长期）：方差突然放大（ep3 std=5.14），seed_2 在 ep3 异常低（22.30 vs 其他 ~31），但 ep5 追平。这是训练早期的随机波动，不影响最终收敛。
- ep5-11（爬升期）：方差中等（std 0.2-2.3），各 seed 的 epoch-to-epoch 波动不完全同步，但总体趋势一致。
- ep12-16（高原区）：三个 seed 都在此区间达到峰值（39.9-40.9），std 降到 0.5-0.8，是整个训练最稳定的区间。
- ep17-24（下降期）：三 seed 一致回落到 37-39 平台，std 0.2-1.0。BEST 之后没有再次突破，说明 40+ 的峰值在 ep12-16 是一次性的，非持续高位。

### 19.7 逐类别方差解读

- Car loose (std=3.28)：方差最大的指标。Car 占数据集 48%，loose 口径下召回的波动直接反映 head 对 Car 的正样本分配稳定性。seed_2 最高 (75.77)，seed_0 最低 (69.20)。
- Cyclist strict (std=2.52)：Cyclist 样本少 (21.5%)，strict 口径下单帧真值匹配的随机性大。seed_2 最高 (28.07)，seed_0 最低 (23.17)。
- Truck strict (std=2.43)：Truck 尺寸方差大（2.8-25.5m），strict 对框质量极敏感，种子间的 anchor 匹配差异会被放大。
- Pedestrian strict (std=0.03)：方差极小但绝对值也极小（~0.12），确认 Ped strict 是物理瓶颈（BEV 分辨率 + 点云稀疏），与种子/模型无关。
- Overall 3D_moderate (std=0.51)：主指标方差最小，说明 Overall 的稳定性最好——Car 占比大，但 Car 的 Overall 贡献被等权 1/4 平滑了。

### 19.8 关键结论

1. Run 10 head-v2 的 40+ 水平是可复现的：三 seed mean=40.42+/-0.51，原 Run 10 单次 40.60 在区间内，不是 cherry-pick。
2. 三 seed 中最好的是 seed_2 ep14=40.88（已存）；seed_0 ep16=39.88 已存，seed_1 ep15=40.51
   **没有对应权重**，best saved 回退 ep12=39.05。
3. 峰值稳定在 ep12-16（原 Run 10 在 ep11）。峰值后移与有效 batch 16→12 的方向一致，但该对比
   混杂了卡数、单卡 batch、梯度累积三项变化，归因待单变量实验；上限水平本身不受影响。
4. 主指标方差极小（std=0.51），适合论文报告 mean+/-std 作为确定性证据。
5. 逐类别方差集中在 Car loose / Cyclist strict / Truck strict——这些是共享 head 正样本分配的敏感点，进一步佐证 Car-Truck 混淆是下一步要解决的根因。
6. Pedestrian strict 约 0 在所有 seed 中一致（0.10-0.15），再次确认是 BEV 分辨率物理瓶颈。

### 19.9 对论文报告口径的更新

基于多 seed 结果，更新第 18 节的最终报告口径：

- 主表（对外）：Run 10 head-v2 并列报告 ep12-16 窗口均值 `39.00 ± 0.53` 与全轮峰值
  `40.42 ± 0.51`；后者是上限统计，不替代前者。
- 补充材料：附三 seed 逐 epoch 曲线（19.1 节）、ep12-16 窗口均值（19.2b）和逐类别峰值
  mean ± std（19.3 节）。
- 最佳单点：seed_2 ep14 = 40.88（已存），作为 released checkpoint 候选。
- 窗口均值选择：ep12-16 是三个 seed 的共同峰值窗口，固定 epoch 可比；全轮峰值另列，
  避免用“各 seed 自身最高点”替代平台对照。

---

## 20. No-Temporal Baseline（N4, 24e, seed_0）

### 20.0 实验设置

- 目的：回答「时序融合模块到底贡献多少」，把当前最大因果缺口从检测头拆出来。
- 配置：在多 seed 复现用的 24e 配方（3 卡 / `samples_per_gpu=2` + `cumulative_iters=2`，
  有效 batch 12）上只改一项：`temporal_fusion=None`。注意对照的是第 19 节多 seed 那套，
  **不是**原始 Run 10：原 Run 10 为 4 卡 / `samples_per_gpu=4` / 无梯度累积，有效 batch 16。
- 保持固定不变的变量：`seq_len=4`、数据 pipeline、head-v2 检测头、`samples_per_gpu=2`、`GradientCumulativeOptimizerHook(cumulative_iters=2)`、`max_epochs=24`、`checkpoint_interval=1`、pretrained checkpoint。
- 运行设置：`seed=0`、`--deterministic`、3 进程，物理 GPU 5/6/7。
- 工作目录：`work_dirs/no_temporal_N4_2x4_24e_seed0`
- 日志：`20260824_160710.log(.json)`
- 配置文件：`configs/r4det/TJ4D-R4Det_no_temporal_N4_2x4_24e_pretrained_v2.py`
- 已提交：`ffb4ac7 feat: add no-temporal N4 baseline config`

### 20.1 Overall 3D_moderate 逐 epoch 曲线

| epoch | 3D_mod | epoch | 3D_mod | epoch | 3D_mod |
|---|---:|---|---:|---|---:|
| 1  | 15.1347 | 9  | 33.6811 | 17 | 34.8725 |
| 2  | 24.6563 | 10 | 33.6988 | 18 | 34.8152 |
| 3  | 29.2675 | 11 | 34.1612 | 19 | 34.5336 |
| 4  | 29.1228 | 12 | 33.5923 | 20 | **35.5903** |
| 5  | 30.4197 | 13 | 34.9294 | 21 | 35.4497 |
| 6  | 33.6661 | 14 | **35.2701** | 22 | 34.5837 |
| 7  | 32.0290 | 15 | 34.3506 | 23 | 34.8232 |
| 8  | 30.0681 | 16 | 35.1058 | 24 | 34.3664 |

### 20.2 ep12-16 逐类别均值

Overall 口径为 Car/Truck strict + Ped/Cyc loose 等权。ep12-16 的均值如下：

| Metric | mean | min | max |
|---|---:|---:|---:|
| Overall 3D_moderate | 34.6496 | 33.5923 | 35.2701 |
| Overall 3D_easy | 37.1762 | 35.7064 | 37.8647 |
| Overall 3D_hard | 33.3940 | 32.5231 | 33.9370 |
| Overall BEV_moderate | 41.6243 | 41.2711 | 41.8689 |
| Car 3D_mod_strict | 46.1617 | 43.0093 | 49.4200 |
| Cyclist 3D_mod_loose | 42.2829 | 40.8932 | 43.6719 |
| Pedestrian 3D_mod_loose | 26.6721 | 25.4214 | 27.9513 |
| Truck 3D_mod_strict | 23.4818 | 20.5529 | 24.9948 |

### 20.3 对比 Run 10 RSSM seed_0

Run 10 seed_0 基线数据采用用户给定的同 seed 口径：

| 口径 | No-Temporal | RSSM seed_0 | Δ RSSM - NoTemporal |
|---|---:|---:|---:|
| ep12-16 mean | 34.6496 | 38.39 | **+3.7404** |
| BEST | 35.5903 @ ep20 | 39.88 @ ep16 | **+4.2897** |
| last-5 mean | 34.9627 (ep20-24) | 37.61 | **+2.6473** |

### 20.4 结论

No-Temporal ep12-16 落在 `< 37.4` 分支：RSSM 带来稳定超过 3 点的 ep12-16 收益，不是噪声，也不是检测头单点差异。当前因果结论明确：

1. 时序融合模块是 head-v2 主线中真实、稳定的增益来源。
2. BEST 口径下 RSSM 高出 4.29 点，说明时序先验对收敛上限的贡献同样存在。
3. 显存也从 RSSM 的约 24 GB/卡降到 No-Temporal 约 6 GB/卡，进一步说明 RSSM 的运动对齐状态与 BPTT 窗口是主要显存成本。
4. 下一步遵循原判定规则：实现 deterministic motion-aligned GRU，而不是先改检测头。No-Temporal 的富余显存留到后续批次/分辨率扩张再使用。

---

## 21. Deterministic Latent Fusion（motion-align deterministic latent, N4, 2x4, 24e, seed_0）

### 21.0 实验设置

- 目的：拆解第 20 节得到的约 3.7 点时序收益，判断它来自确定性状态传播（h/z 双状态、deform alignment、encoder、ConvGRU transition、decoder/reconstruction、output_proj + feat）还是 RSSM 随机建模（prior/KL/sampling）。
- 配置：在保留 h/z 双状态、deform alignment、encoder、ConvGRU transition、posterior_mu、decoder/reconstruction、output_proj + residual feat、N4、BPTT1、head-v2、pretrain、batch、schedule 的前提下，删除 prior_logstd/posterior_logstd/sampling/KL。训练和推理固定 `z_t = posterior_mu(torch.cat([h_t, e_t], dim=1))`。
- 保持固定不变的变量：`seq_len=4`、数据 pipeline、head-v2 检测头、`samples_per_gpu=2`、`workers_per_gpu=2`、`cumulative_iters=2`、`max_epochs=24`、`checkpoint_interval=1`、pretrained checkpoint。
- 运行设置：`seed=0`、`--deterministic`、`CUDA_VISIBLE_DEVICES=5,6,7`、3 进程。
- 工作目录：`work_dirs/deterministic_latent_N4_2x4_24e_seed0`
- 日志：`20260825_045915.log(.json)`
- 配置文件：`configs/r4det/TJ4D-R4Det_motion_align_deterministic_latent_N4_2x4_24e_pretrained_v2_head.py`
- 相关提交：`7643a5b feat: add deterministic motion-aligned latent fusion`、`99c74ff fix: move deterministic latent class after stochastic parent`

### 21.1 Overall 3D_moderate 逐 epoch 曲线

| epoch | 3D_mod | epoch | 3D_mod | epoch | 3D_mod |
|---|---:|---|---:|---|---:|
| 1  | 16.0119 | 9  | 36.5710 | 17 | 35.8816 |
| 2  | 22.6346 | 10 | 37.7363 | 18 | 36.6017 |
| 3  | 29.9400 | 11 | 36.0891 | 19 | 36.3157 |
| 4  | 33.6664 | 12 | 36.5496 | 20 | 37.3297 |
| 5  | 32.0928 | 13 | **38.4411** | 21 | 38.1950 |
| 6  | 34.4483 | 14 | 35.4515 | 22 | 37.4452 |
| 7  | 37.2020 | 15 | 36.9424 | 23 | 36.6420 |
| 8  | 37.6450 | 16 | 38.0183 | 24 | 37.0749 |

### 21.2 ep12-16 逐指标均值

| Metric | mean | min | max |
|---|---:|---:|---:|
| Overall 3D_moderate | 37.0806 | 35.4515 | 38.4411 |
| Overall 3D_easy | 39.1113 | 37.5349 | 40.7642 |
| Overall 3D_hard | 35.6477 | 34.1496 | 36.8898 |
| Overall BEV_moderate | 45.5314 | 43.7859 | 47.2859 |
| Overall 2D_moderate | 45.0461 | 42.2836 | 46.5694 |
| Car 3D_mod_strict | 44.6349 | 42.0855 | 47.7723 |
| Cyclist 3D_mod_loose | 44.9337 | 42.0461 | 47.6161 |
| Pedestrian 3D_mod_loose | 28.3171 | 25.6201 | 30.5342 |
| Truck 3D_mod_strict | 30.4367 | 26.2493 | 33.2016 |

### 21.3 与 No-Temporal / RSSM seed_0 对比

| 口径 | No-Temporal | RSSM seed_0 | Deterministic | Δ Det - RSSM |
|---|---:|---:|---:|---:|
| ep12-16 mean | 34.6496 | 38.39 | 37.0806 | -1.3094 |
| BEST | 35.5903 @ ep20 | 39.88 @ ep16 | 38.4411 @ ep13 | -1.4389 |
| last-5 mean | 34.9627 (ep20-24) | 37.61 | 37.3374 (ep20-24) | -0.2726 |

### 21.4 结论

本 run ep12-16 mean = **37.0806 < 37.4**，落入原判定规则的「RSSM 随机训练贡献超过 1 点」分支：

1. 保留确定性转变的 h/z + deform alignment + encoder + ConvGRU transition + decoder/reconstruction 仍然显著高于 No-Temporal（37.08 vs 34.65，+2.43），因此确定性时序传播本身是真实有效的一部分收益。
2. 但相比完整 RSSM seed_0 还差 1.31 点（ep12-16 mean），说明 prior + KL + posterior 重参数采样这一套随机建模不是可删除的冗余复杂度，而是必需组件。
3. 因此下一步不是压成单状态 motion-aligned GRU，而是保留 RSSM，并把工作重点转向重新设计 KL / free-bits 训练策略。
4. 依据判定规则，本结果已直接落在 `< 37.4` 分支，不需要补 seed1/seed2 进入灰区。

---

## 22. KL=0 消融（完整 RSSM，关闭 KL 梯度，seed0）

### 22.0 实验设置

- 目的：回答「完整 RSSM 的收益到底来自 KL/prior 约束，还是来自 posterior sampling 与确定性状态传播」，只把训练侧 `kl_scale` 置零，其余全部保持。
- 配置：以 head-v2、N4、pretrained、batch12、24e 主模型为基础，删除 `KLScaleSchedulerHook`，设置 `kl_scale=0.0`、`rssm_bptt_steps=1`。
- 保留不变：完整 `MotionAlignedRSSMFusion`、posterior sampling、`seq_len=4`、detection head-v2、`samples_per_gpu=2`、`workers_per_gpu=2`、`cumulative_iters=2`、`max_epochs=24`、`checkpoint_interval=1`、pretrained checkpoint。
- 运行设置：`seed=0`、3 卡，有效 batch 12。
- 工作目录：`work_dirs/rssm_kl0_N4_2x4_24e_seed0`
- 日志：`20260826_043350.log(.json)`
- 配置文件：`configs/r4det/TJ4D-R4Det_motion_align_rssm_N4_2x4_24e_pretrained_v2_head_kl0.py`
- 相关提交：`4aff567` add KL=0 motion-align RSSM ablation config、`b0db033` fix: align KL=0 ablation batch setup

### 22.1 Overall 3D_moderate 逐 epoch 曲线

第三行为判定窗口 ep12-16；ep14 为窗口最高，ep22 为全 run 最高。

| epoch | 3D_mod | epoch | 3D_mod | epoch | 3D_mod |
|---|---:|---|---:|---|---:|
| 1  | 16.7797 | 9  | 38.5991 | 17 | 37.1979 |
| 2  | 23.9284 | 10 | 37.1948 | 18 | 38.8741 |
| 3  | 30.3761 | 11 | 37.2744 | 19 | 37.3882 |
| 4  | 29.5956 | 12 | 39.5836 | 20 | 39.1799 |
| 5  | 31.5989 | 13 | 39.4934 | 21 | 38.8375 |
| 6  | 34.5267 | 14 | **39.6240** | 22 | **40.3481** |
| 7  | 35.9756 | 15 | 37.6445 | 23 | 38.8278 |
| 8  | 35.2369 | 16 | 38.9717 | 24 | 37.9561 |

### 22.2 ep12-16 逐指标 mean / min / max

Overall 口径为 Car/Truck strict + Ped/Cyc loose 等权。

| Metric | mean | min | max |
|---|---:|---:|---:|
| Overall 3D_moderate | 39.0634 | 37.6445 | 39.6240 |
| Overall 3D_easy | 41.1325 | 39.0125 | 42.0542 |
| Overall 3D_hard | 37.5560 | 36.2441 | 38.1224 |
| Overall BEV_moderate | 46.9612 | 46.2096 | 47.8544 |
| Overall 2D_moderate | 46.7393 | 46.0684 | 47.9641 |
| Car 3D_moderate_strict | 48.0055 | 44.3523 | 50.2778 |
| Cyclist 3D_moderate_loose | 49.3367 | 47.6233 | 50.1467 |
| Pedestrian 3D_moderate_loose | 26.9333 | 24.6588 | 29.1242 |
| Truck 3D_moderate_strict | 31.9781 | 29.7207 | 35.7743 |

### 22.3 与完整 RSSM seed0 / Deterministic / No-Temporal 对比

| 口径 | KL0 | RSSM seed0 | Deterministic | No-Temporal | Δ KL0 - RSSM |
|---|---:|---:|---:|---:|---:|
| ep12-16 mean | 39.0634 | 38.3902 | 37.0806 | 34.6496 | +0.6732 |
| BEST | 40.3481 @ ep22 | 39.8802 @ ep16 | 38.4411 @ ep13 | 35.5903 @ ep20 | +0.4679 |
| last-5 mean | 39.0299 | 37.6084 | 37.3374 | 34.9627 | +1.4215 |

ep12-16 类别均值对比：

| Metric | KL0 | RSSM seed0 | Δ KL0 - RSSM |
|---|---:|---:|---:|
| Overall 3D_moderate | 39.0634 | 38.3902 | +0.6732 |
| Car 3D_moderate_strict | 48.0055 | 47.8027 | +0.2028 |
| Cyclist 3D_moderate_loose | 49.3367 | 48.6271 | +0.7096 |
| Pedestrian 3D_moderate_loose | 26.9333 | 28.9160 | -1.9827 |
| Truck 3D_moderate_strict | 31.9781 | 28.2149 | +3.7632 |

### 22.4 RSSM 动态检查

KL 梯度确实被关闭，但 posterior 没有随 KL=0 被压死。

| 指标 | ep24 训练末值 |
|---|---:|
| loss_rssm_kl | 0.0000 |
| loss_rssm_recon | 0.0018 |
| stat_kl_raw_mean | 21.5168 |
| stat_kl_effective_mean | 21.6740 |
| stat_mu_diff_sq | 1.7327 |
| stat_posterior_std | 0.1035 |
| stat_prior_std | 0.2062 |

### 22.5 结论与下一步

1. KL0 的 ep12-16 `Overall_3D_moderate` 均值为 **39.0634**，窗口最高为 **39.6240**，均通过 37.9 阈值。
2. 因此按原判定规则，完整 RSSM 的主要收益来自 posterior sampling，而不是 KL/prior 梯度。
3. 类别拆解不是单向普涨：Truck strict 提升明显（+3.76），Car、Cyclist 小幅提升，Pedestrian loose 反而低约 1.98 点。后续 posterior-only stochastic fusion 需要继续看 Car/Cyclist/Truck，同时单独盯住 Pedestrian。
4. 下一步进入 posterior-only stochastic fusion 消融：保留 posterior sampling，删除 prior/KL，不做 deterministic seed1/2、单状态 GRU 压缩，也不重新调整 free-bits。

---

## 23. FixedNoise Posterior Latent Fusion（固定后验噪声，N4, 2x4, 24e, seed_0）

### 23.0 实验设置

- 目的：承接第 22 节结论，验证「完整 RSSM 的增益来自 posterior sampling」是否只需在 posterior mean 上加入固定高斯噪声，而不需要可学习的 posterior std。
- 配置：继承 Deterministic Latent Fusion 的 h/z 双状态、deform alignment、encoder、ConvGRU transition、posterior_mu、decoder/reconstruction、output_proj + residual feat；删除 prior/prior_logstd/posterior_logstd/KL。
- training：`z_t = posterior_mu(x) + 0.1 * N(0, I)`；inference：`z_t = posterior_mu(x)`（确定性）。
- 保持固定不变的变量：`seq_len=4`、head-v2、`samples_per_gpu=2`、`workers_per_gpu=2`、`cumulative_iters=2`、`max_epochs=24`、`checkpoint_interval=1`、`hidden_dim=128`、`rssm_bptt_steps=1`、`lr=0.00015`、pretrained checkpoint。
- 运行设置：`seed=0`、3 卡、有效 batch 12。
- 工作目录：`work_dirs/rssm_N4_fixednoise_posterior_seed0`
- 日志：`20260827_052300.log(.json)`
- 配置文件：`configs/r4det/TJ4D-R4Det_motion_align_rssm_det3d_N4_2x4_24e_pretrained_v2_head_bptt_fixednoise.py`
- 相关提交：`01d6cb6 feat: add fixed-noise posterior fusion ablation`

### 23.1 Overall 3D_moderate 逐 epoch 曲线

第三行为判定窗口 ep12-16；ep16 为窗口最高，也是全 run BEST。

| epoch | 3D_mod | epoch | 3D_mod | epoch | 3D_mod |
|---|---:|---|---:|---|---:|
| 1  | 19.4041 | 9  | 32.6978 | 17 | 36.4725 |
| 2  | 25.1668 | 10 | 36.1395 | 18 | 37.6786 |
| 3  | 28.2528 | 11 | 33.6568 | 19 | 38.3482 |
| 4  | 31.8684 | 12 | 34.3371 | 20 | 36.3328 |
| 5  | 31.4987 | 13 | 35.8474 | 21 | 36.9975 |
| 6  | 34.1488 | 14 | 36.2377 | 22 | 37.3021 |
| 7  | 35.3952 | 15 | 36.8997 | 23 | 37.1123 |
| 8  | 36.2861 | 16 | **38.4151** | 24 | 36.9415 |

### 23.2 ep12-16 逐指标 mean / min / max

Overall 口径为 Car/Truck strict + Ped/Cyc loose 等权。

| Metric | mean | min | max |
|---|---:|---:|---:|
| Overall 3D_moderate | 36.3474 | 34.3371 | 38.4151 |
| Overall 3D_easy | 38.6484 | 36.0293 | 41.0250 |
| Overall 3D_hard | 34.9808 | 33.1099 | 36.9788 |
| Overall BEV_moderate | 44.9409 | 43.1088 | 47.3162 |
| Overall 2D_moderate | 44.7150 | 42.9076 | 46.9505 |
| Car 3D_moderate_strict | 48.4945 | 45.3705 | 53.7636 |
| Cyclist 3D_moderate_loose | 42.9509 | 39.0300 | 45.6074 |
| Pedestrian 3D_moderate_loose | 27.0083 | 25.0096 | 28.5263 |
| Truck 3D_moderate_strict | 26.9359 | 24.4216 | 29.5687 |

### 23.3 与 RSSM seed0 / KL0 / Deterministic / No-Temporal 对比

| 口径 | FixedNoise | KL0 | RSSM seed0 | Deterministic | No-Temporal |
|---|---:|---:|---:|---:|---:|
| ep12-16 mean | 36.3474 | 39.0634 | 38.3902 | 37.0806 | 34.6496 |
| BEST | 38.4151 @ ep16 | 40.3481 @ ep22 | 39.8802 @ ep16 | 38.4411 @ ep13 | 35.5903 @ ep20 |
| last-5 mean | 36.9372 | 39.0299 | 37.6084 | 37.3374 | 34.9627 |

ep12-16 类别均值对比：

| Metric | FixedNoise | KL0 | RSSM seed0 | Δ FixedNoise - KL0 | Δ FixedNoise - RSSM |
|---|---:|---:|---:|---:|---:|
| Overall 3D_moderate | 36.3474 | 39.0634 | 38.3902 | -2.7160 | -2.0428 |
| Car 3D_moderate_strict | 48.4945 | 48.0055 | 47.8027 | +0.4890 | +0.6918 |
| Cyclist 3D_moderate_loose | 42.9509 | 49.3367 | 48.6271 | -6.3858 | -5.6762 |
| Pedestrian 3D_moderate_loose | 27.0083 | 26.9333 | 28.9160 | +0.0750 | -1.9077 |
| Truck 3D_moderate_strict | 26.9359 | 31.9781 | 28.2149 | -5.0422 | -1.2790 |

### 23.4 结论与下一步

1. 固定 `std=0.1` 的训练期噪声没有复现 KL0 / 完整 RSSM 的 posterior sampling 收益：ep12-16 `Overall_3D_moderate` 为 **36.3474**，比 KL0 低 **2.72**，比 RSSM seed0 低 **2.04**。
2. 相比 Deterministic 的 37.0806 还低 **0.73**，说明「在 posterior mean 上套一个固定高斯噪声」不是 22 节所发现的因果组件；它甚至带来轻微反向正则。
3. 类别变化不是均匀衰减：Car strict 略高于两个对照（窗口均值 48.49，ep16 best 53.76），但 Cyclist loose 比 KL0 低 6.39、Truck strict 比 KL0 低 5.04，主指标被 Cyclist/Truck 拉垮。
4. 可学习的 `posterior_logstd`（KL0 中终值 0.1035）是有效的条件噪声幅度，固定 0.1 表面接近但缺少对输入的依赖；下一步保留 learnable posterior std，只删除 prior/KL，做 posterior-only learnable-std 消融，而不是继续用标量固定噪声。

---

## 24. Posterior-Only Learnable-Std Fusion（N4, 2x4, 24e, seed_0）

### 24.0 实验设置

- 目的：承接第 23 节，判断 KL0 的高分是否来自可学习的 posterior std，而不是固定噪声。仅保留 posterior sampling，删除 prior/KL，检查其能否复现 KL0。
- 配置：保留 Deterministic / FixedNoise 的 h/z 双状态、deform alignment、encoder、ConvGRU transition、posterior_mu、decoder/reconstruction、output_proj + residual feat；同时保留 `posterior_logstd`，删除 `prior_mu`、`prior_logstd`、KL。
- training：`z_t = sample(posterior_mu(x), posterior_logstd(x))`；inference：`z_t = posterior_mu(x)`（确定性）。
- 保持固定不变的变量：`seq_len=4`、head-v2、`samples_per_gpu=2`、`workers_per_gpu=2`、`cumulative_iters=2`、`max_epochs=24`、`checkpoint_interval=1`、`hidden_dim=128`、`rssm_bptt_steps=1`、`lr=0.00015`、`min_std=0.1`、`init_std=0.2`、pretrained checkpoint。
- 运行设置：`seed=0`、`--deterministic`、3 卡（GPU 5/6/7，RTX 4090 D）、有效 batch 12；单卡训练显存约 13.5 GB。
- 工作目录：`work_dirs/posterior_only_learnable_std_N4_2x4_24e_seed0`
- 日志：`20260828_032831.log(.json)`
- 配置文件：`configs/r4det/TJ4D-R4Det_motion_align_rssm_det3d_N4_2x4_24e_pretrained_v2_head_bptt_learnable_std.py`
- 相关提交：`713dc7a feat: add posterior-only learnable-std fusion`、`c80dd42 fix: subsample posterior std stats`

### 24.1 Overall 3D_moderate 逐 epoch 曲线

第三行为判定窗口 ep12-16；全 run 最高点出现在 ep10，窗口最高为 ep16。

| epoch | 3D_mod | epoch | 3D_mod | epoch | 3D_mod |
|---|---:|---|---:|---|---:|
| 1  | 16.1780 | 9  | 36.9184 | 17 | 36.4314 |
| 2  | 26.7111 | 10 | **39.6564** | 18 | 37.5298 |
| 3  | 23.8984 | 11 | 35.7326 | 19 | 37.5745 |
| 4  | 27.1085 | 12 | 35.8906 | 20 | 37.9873 |
| 5  | 29.7986 | 13 | 34.1280 | 21 | 39.0428 |
| 6  | 34.9217 | 14 | 38.1596 | 22 | 37.9125 |
| 7  | 32.8942 | 15 | 38.4768 | 23 | 37.7034 |
| 8  | 36.3889 | 16 | 38.5491 | 24 | 37.0164 |

### 24.2 ep12-16 逐指标 mean / min / max

Overall 口径为 Car/Truck strict + Ped/Cyc loose 等权。

| Metric | mean | min | max |
|---|---:|---:|---:|
| Overall 3D_moderate | 37.0408 | 34.1280 | 38.5491 |
| Overall 3D_easy | 38.7021 | 36.0799 | 40.2280 |
| Overall 3D_hard | 35.7036 | 32.9151 | 37.1552 |
| Overall BEV_moderate | 44.1117 | 42.5789 | 45.7167 |
| Overall 2D_moderate | 44.2855 | 43.0141 | 46.7505 |
| Car 3D_moderate_strict | 45.6257 | 39.4319 | 48.2577 |
| Cyclist 3D_moderate_loose | 47.4636 | 45.1747 | 49.4966 |
| Pedestrian 3D_moderate_loose | 28.2974 | 26.3186 | 31.3549 |
| Truck 3D_moderate_strict | 26.7765 | 23.6058 | 29.5485 |

### 24.3 与 KL0 / RSSM seed0 / Deterministic / FixedNoise 对比

| 口径 | LearnableStd | KL0 | RSSM seed0 | Deterministic | FixedNoise |
|---|---:|---:|---:|---:|---:|
| ep12-16 mean | 37.0408 | 39.0634 | 38.3902 | 37.0806 | 36.3474 |
| BEST | 39.6564 @ ep10 | 40.3481 @ ep22 | 39.8802 @ ep16 | 38.4411 @ ep13 | 38.4151 @ ep16 |
| last-5 mean | 37.9325 | 39.0299 | 37.6084 | 37.3374 | 36.9372 |

ep12-16 类别均值对比：

| Metric | LearnableStd | KL0 | RSSM seed0 | Δ LearnableStd - KL0 | Δ LearnableStd - RSSM |
|---|---:|---:|---:|---:|---:|
| Overall 3D_moderate | 37.0408 | 39.0634 | 38.3902 | -2.0226 | -1.3494 |
| Car 3D_moderate_strict | 45.6257 | 48.0055 | 47.8027 | -2.3798 | -2.1770 |
| Cyclist 3D_moderate_loose | 47.4636 | 49.3367 | 48.6271 | -1.8731 | -1.1635 |
| Pedestrian 3D_moderate_loose | 28.2974 | 26.9333 | 28.9160 | +1.3641 | -0.6186 |
| Truck 3D_moderate_strict | 26.7765 | 31.9781 | 28.2149 | -5.2016 | -1.4384 |

### 24.4 Posterior std 动态检查

`stat_posterior_std_std` 是每个 step 内部 spatial std 的均值，不是跨 seed std；P10/P50/P90 用于检查 learnable std 是否出现空间上的条件分布。

| 指标 | ep1 it50 | ep6 it1900 | ep12 it1900 | ep24 it1900 |
|---|---:|---:|---:|---:|
| stat_posterior_std_mean | 0.2064 | 0.1040 | 0.1020 | 0.1019 |
| stat_posterior_std_std | 0.0592 | 0.0296 | 0.0259 | 0.0249 |
| stat_posterior_std_p10 | 0.1778 | 0.1007 | 0.1002 | 0.1000 |
| stat_posterior_std_p50 | 0.1997 | 0.1020 | 0.1006 | 0.1003 |
| stat_posterior_std_p90 | 0.2310 | 0.1059 | 0.1027 | 0.1026 |
| loss_rssm_recon | 0.8902 | 0.0077 | 0.0020 | 0.0019 |

### 24.5 结论与下一步

1. ep12-16 `Overall_3D_moderate` 为 **37.0408 < 37.8**，落入「learnable std 无效，KL0 高分主要不来自 learned std」分支。
2. 虽然 BEST 口径达到 **39.6564 @ ep10**，但从 ep6 开始 posterior std 已基本压到 `min_std` 附近；窗口期 P10/P50/P90 挤在 0.100–0.103，说明可学习 std 已提前退化，接近确定性状态。
3. 与 Deterministic 相比几乎打平（37.0408 vs 37.0806，−0.04），说明保留一套 `squashed-to-0.1` 的可学习 std 并没有复现 KL0 的 2 点增益。
4. 下一步停止 posterior 随机建模，回退 Deterministic，重点检查初始化路径差异；先不要把 4×1 无 accumulation 变体、single-state 压缩、free-bits 或检测头改动混入这条因果链。

---

## 25. KL=0 消融多 seed 复现（seed1 + seed2，配对完整 RSSM）

> 日期：2026-08-29 ~ 2026-08-31。承接第 22 节 KL0 seed0 的单点结果，
> 按原定判定规则补跑 KL0 seed1/seed2，并与完整 RSSM 同名 seed 作配对对比，
> 最终决定 RSSM 机制是否保留 KL/prior 梯度。两轮均 3 卡（GPU 5/6/7）、
> `samples_per_gpu=2`、`cumulative_iters=2`（有效 batch 12）、24e、`--deterministic`、
> `rssm_bptt_steps=1`，未使用 `bs4_noaccum` 变体，确保 BN 微批次口径与 seed0 一致。

### 25.0 运行设置

| seed | KL0 work_dir | KL0 日志 | 完整 RSSM 日志（对照） |
|---|---|---|---|
| seed0 | `rssm_kl0_N4_2x4_24e_seed0` | `20260826_043350` | `run10_headv2_multiseed/seed_0/20260821_171603` |
| seed1 | `rssm_kl0_N4_2x4_24e/seed_1` | `20260829_025956` | `run10_headv2_multiseed/seed_1/20260822_152808` |
| seed2 | `rssm_kl0_N4_2x4_24e/seed_2` | `20260830_013427` | `run10_headv2_multiseed/seed_2/20260823_140745` |

- 配置：`configs/r4det/TJ4D-R4Det_motion_align_rssm_N4_2x4_24e_pretrained_v2_head_kl0.py`
- 完整 RSSM 对照：`configs/r4det/TJ4D-R4Det_motion_align_rssm_det3d_N4_2x4_24e_pretrained_v2_head.py`
- 两轮均完整跑完 24 epoch，24 个 checkpoint 落盘，无 NaN、无梯度爆炸。

### 25.1 KL0 三 seed 逐 epoch Overall_3D_moderate 曲线

| ep | seed0 | seed1 | seed2 | ep | seed0 | seed1 | seed2 |
|---:|---:|---:|---:|---:|---:|---:|---:|
|  1 | 16.78 | 18.52 | 19.24 | 13 | 39.49 | 40.19 | 35.97 |
|  2 | 23.93 | 24.51 | 24.70 | 14 | **39.62** | 37.82 | 35.28 |
|  3 | 30.38 | 33.08 | 27.42 | 15 | 37.64 | **40.40** | 34.42 |
|  4 | 29.60 | 35.46 | 33.32 | 16 | 38.97 | 38.72 | 36.32 |
|  5 | 31.60 | 34.99 | 35.07 | 17 | 37.20 | 40.56 | **36.76** |
|  6 | 34.53 | 37.37 | 31.61 | 18 | 38.87 | 40.49 | 35.29 |
|  7 | 35.98 | 37.62 | 32.68 | 19 | 37.39 | 40.37 | 36.08 |
|  8 | 35.24 | 39.26 | 31.86 | 20 | 39.18 | **41.10** | 36.49 |
|  9 | 38.60 | 37.03 | 35.61 | 21 | 38.84 | 40.75 | 36.42 |
| 10 | 37.19 | 36.81 | 34.91 | 22 | **40.35** | 39.98 | 35.82 |
| 11 | 37.27 | 38.28 | 34.05 | 23 | 38.83 | 40.11 | 36.67 |
| 12 | 39.58 | 39.90 | 34.84 | 24 | 37.96 | 40.54 | 35.89 |

### 25.2 与完整 RSSM 配对对比（同 seed）

| 口径 | seed0 | seed1 | seed2 | 三 seed 均值 |
|---|---:|---:|---:|---:|
| KL0 ep12-16 mean | 39.0634 | 39.4058 | 35.3651 | 37.9448 |
| RSSM ep12-16 mean | 38.3902 | 39.1994 | 39.3990 | 38.9962 |
| Δ (KL0 − RSSM) | +0.6733 | +0.2064 | **−4.0339** | **−1.0514** |
| KL0 last-5 mean | 39.0299 | 40.4955 | 36.2578 | 38.5944 |
| RSSM last-5 mean | 37.6084 | 38.5457 | 38.1864 | 38.1135 |
| Δ (KL0 − RSSM) | +1.4215 | +1.9498 | **−1.9286** | +0.4809 |

### 25.3 ep12-16 各类别均值（KL0 / RSSM）

| Metric | KL0 s0 | RSSM s0 | KL0 s1 | RSSM s1 | KL0 s2 | RSSM s2 |
|---|---:|---:|---:|---:|---:|---:|
| Car_3D_moderate_strict | 48.01 | 47.80 | 50.31 | 45.40 | 43.25 | 48.98 |
| Cyclist_3D_moderate_loose | 49.34 | 48.63 | 44.64 | 47.60 | 39.89 | 47.99 |
| Pedestrian_3D_moderate_loose | 26.93 | 28.92 | 27.51 | 30.20 | 28.75 | 30.58 |
| Truck_3D_moderate_strict | 31.98 | 28.21 | 35.16 | 33.60 | 29.57 | 30.04 |

### 25.4 训练稳定性检查（KL0 seed2 排除跑崩）

seed2 虽然 Overall 明显偏低，但训练过程健康，不是失败或 NaN：

| 指标 | ep24 训练末值 |
|---|---:|
| loss_rssm_kl | 0.0000 |
| loss_rssm_recon | 0.0019 |
| stat_kl_raw_mean | 26.80 |
| stat_kl_effective_mean | 26.97 |
| stat_clamped_ratio | 0.376 |
| stat_posterior_std | 0.1015 |
| stat_prior_std | 0.2055 |
| grad_norm | ~4.21 |

posterior/prior std 与 seed0/seed1 一致，说明 KL0 在 seed2 上客观收敛到了更低的 Car/Cyclist 平台，
而不是训练异常。

### 25.5 判定与结论

按原定规则三档判定：

1. **不满足「删除 KL」条件**：要求「三 seed ep12-16 与 last-5 均值提升 ≥0.5 且至少 2/3 seed 提升」。
   - ep12-16 三 seed 均值 Δ = **−1.05**（KL0 更低）；
   - last-5 三 seed 均值 Δ = +0.48（勉强为正，但 ep12-16 为负）；
   - 只有 seed0/seed1 两个 seed 提升，seed2 在 ep12-16 大降 4.03，增益完全不稳健。
2. **超出「±0.3 灰区」**：KL0 平均下降 1.05（ep12-16），seed2 单项 −4.03，远大于 0.3 阈值。
3. **落入「保留完整 RSSM」分支**：种子间方差把 KL0 的高分平均掉，且出现负信号，不能说明
   KL/prior 梯度是可删除的冗余；相反，seed2 表明关闭 KL 会破坏后验状态跨帧的一致性。

**结论：保留完整 RSSM（含 KL/prior 梯度），不删除 KL。**

seed0 单点的 +0.67（第 22 节）在三 seed 视角下被证明不是稳健收益，更多是 KL0 在个别种子上的
波动放大，而非机制性提升。第 22 节「主要收益来自 posterior sampling 而非 KL」的中间结论需要
修正为：posterior sampling 与 prior/KL 两者共同构成 RSSM 随机建模，缺一会引入跨种子不稳定。

### 25.6 下一步

1. **RSSM 机制就此定稿**：`MotionAlignedRSSMFusion` + full prior/KL + posterior sampling，
   N4、BPTT1、2×4（有效 batch 12）、24e，作为论文的时序融合基准，不再继续堆随机状态变体。
2. 检测头消融回归主线，优先级：Truck/Car 混淆（Cyclist strict / Truck strict 是多 seed 方差
   的主要来源）→ Pedestrian strict（BEV 分辨率物理瓶颈，需单独定向）。
3. 论文最终报告口径仍以第 19 节完整 RSSM 三 seed 的全轮峰值 mean 40.42 ± 0.51 为准，
   主对照同时使用 ep12-16 窗口均值 39.00 ± 0.53；
   KL0 多 seed 结果作为负向消融证据列入附录，佐证「保留 KL」的机制性结论。

---

## 26. Truck 专属回归细化（Truck residual refinement: dx/dy/dl, N4, 2x4, 24e, seed0）

### 26.0 动机与设计

前置错误诊断（第 25 节后、`tools/diagnose_errors.py`，3 个最佳 checkpoint 聚合）结论：

- Truck strict 通过率 26.1%（Car 49.2%），正确分类后仍然大量失败。
- 错误集中在 **Truck 长度（长轴）3.02m、BEV 中心 1.41m**；朝向误差仅 0.15 rad（基本正确）。
- Car/Truck 分类互混约 16%（cross/diag），存在但不是首要问题。

因此先改 **Truck 中心 + 长度回归**，暂不拆分类分支，只验证单一变量。

实现：保留共享 `conv_reg`，新增一个只对 Truck 两个 anchor 输出 `(dx, dy, dl)` 残差的旁支：

```
shared BEV feature
 └─ conv_reg: 全 anchor 全 7D 参数（不变）
 └─ truck_refine_conv: Conv3x3(256→64) + ReLU + Conv1x1(64→6)，末层零初始化
      6 = 2 rotations × (dx, dy, dl)
      bbox_pred[..., [0,1,4]] 在 Truck 两 anchor 对应通道 += refine
```

- 末层零初始化 → 训练起点与当前完整 RSSM 完全一致；原 conv_reg 仍提供基础框。
- Car/Cyc/Ped 通道完全不受扰动；conv_cls、direction、IoU branch、anchor/assigner/FocalLoss/SmoothL1(2.0) 不变。
- 未启用 confusion_pairs、未新增 Truck anchor、未改 Pedestrian。

### 26.1 实验设置

- 配置：`configs/r4det/TJ4D-R4Det_motion_align_rssm_det3d_N4_2x4_24e_pretrained_v2_head.py`
  - 新增 `truck_refine=True / truck_refine_channels=64 / truck_refine_dims=(0,1,4) / truck_anchor_class=5`
  - 还原 `samples_per_gpu=2` + `GradientCumulativeOptimizerHook(cumulative_iters=2)`（对齐 seed0 基线）
- 固定不变：`seq_len=4`、`rssm_bptt_steps=1`、`hidden_dim=128`、`kl_scale=1.0`、`free_nats=1.0`、
  有效 batch 12、`lr=1.5e-4`、`max_epochs=24`、`checkpoint_interval=1`、pretrained_tj4d.pth。
- 运行：`seed=0`、`--deterministic`、`CUDA_VISIBLE_DEVICES=5,6,7`、3 进程。
- 工作目录：`work_dirs/truck_refine_N4_2x4_24e_seed0`。
- 相关提交：`f36c90c feat: add Truck-specific residual refinement`、`6221e6a fix(config): restore samples_per_gpu=2 + cumulative_iters=2`。

### 26.2 通过标准

- Truck strict ep12-16 均值 ≥ 基线 +2.0
- Overall ep12-16 ≥ 基线 +0.5
- Car strict 下降 ≤ 0.5
- Cyclist loose 下降 ≤ 1.0

### 26.3 训练结果（seed0，ep1-24 完整曲线）

> 训练中因会话被回收误杀一次，已从 `epoch_2.pth` 断点续训（`--resume-from epoch_2.pth`），
> 因此 epoch 3 重跑了一次；ep1/ep2 指标来自首次段日志，ep3 起来自续训段日志
> （续训段日志文件：`20260901_124027.log(.json)`）。checkpoint 完整（ep1-24 全存）。

逐 epoch val（3D_moderate；Truck/Car 为 strict，Cyc 为 loose）：

| ep | Overall | Truck_strict | Car_strict | Cyc_loose |
|---|---:|---:|---:|---:|
| 1  | 16.67 | 3.49 | 23.27 | 36.28 |
| 2  | 26.92 | 13.05 | 39.66 | 36.74 |
| 3  | 27.78 | 18.63 | 39.62 | 39.00 |
| 4  | 32.06 | 19.01 | 35.70 | 50.22 |
| 5  | 33.22 | 22.47 | 42.73 | 52.29 |
| 6  | 33.08 | 25.86 | 42.86 | 47.52 |
| 7  | 36.22 | 26.12 | 48.06 | 47.91 |
| 8  | 35.82 | 25.37 | 49.12 | 47.12 |
| 9  | 40.26 | 34.02 | 48.10 | 49.36 |
| 10 | 39.53 | 28.83 | 48.93 | 51.54 |
| 11 | 36.01 | 28.27 | 43.86 | 47.57 |
| 12 | 38.36 | 32.67 | 50.60 | 43.97 |
| 13 | 37.70 | 28.92 | 50.41 | 47.63 |
| 14 | 36.84 | 30.62 | 53.14 | 42.73 |
| 15 | 35.38 | 31.35 | 43.33 | 39.28 |
| 16 | 38.07 | 31.75 | 48.41 | 44.83 |
| 17 | 38.92 | 29.87 | 51.88 | 45.96 |
| 18 | 38.56 | 32.50 | 49.38 | 44.36 |
| 19 | 38.42 | 32.75 | 50.17 | 42.68 |
| 20 | 38.63 | 35.03 | 49.88 | 42.70 |
| 21 | 39.79 | 35.01 | 51.02 | 44.01 |
| 22 | 39.60 | 34.87 | 51.80 | 44.18 |
| 23 | 38.04 | 32.98 | 48.70 | 42.67 |
| 24 | 36.57 | 33.52 | 48.09 | 41.56 |

### 26.4 通过标准核对（seed0，未通过）

按用户给定口径「ep12-16 均值」核对（基线 = `run10_headv2_multiseed/seed_0`）：

| 指标 | 基线均值 | 本次均值 | Δ | 标准 | 判定 |
|---|---:|---:|---:|---|---|
| Truck strict | 28.21 | 31.06 | **+2.85** | ≥ +2.0 | ✅ 通过 |
| Car strict | 47.80 | 49.18 | +1.38 | ≥ −0.5 | ✅ 通过 |
| Cyclist loose | 48.63 | 43.69 | **−4.94** | ≥ −1.0 | ❌ 超限 3.94 |
| Overall | 38.39 | 37.27 | −1.12 | ≥ +0.5 | ❌ 未达 |

多口径辅助（Δ = 本次 − 基线）：

| 口径 | Overall Δ | Truck Δ | Car Δ | Cyc Δ |
|---|---:|---:|---:|---:|
| ep12-16 | −1.12 | +2.85 | +1.38 | −4.94 |
| last-5 (ep20-24) | +0.92 | +4.13 | +3.71 | −3.98 |
| 全段 ep12-24 | +0.10 | +3.38 | +2.75 | −4.40 |

结论：**seed0 未干净通过**。Truck/Car 目标指标稳定超额达标（Truck +2.85~4.13、Car +1.38~3.71），
证明「Truck 中心+长度修正」单一变量对主目标有效；但 Cyclist loose 稳定掉 4 点左右，把
ep12-16 口径的 Overall 拉成 −1.12，触发「Cyclist loose 下降 ≤1.0」红线。

### 26.5 Cyclist 下滑根因诊断（dump inference + 匹配分析）

对基线 ep16、本次 ep20、本次 ep24 三个 checkpoint 跑 `tools/dump_predictions.py`（2040 样本
完整推理 dump），再用旋转 3D IoU 做逐样本匹配，定位 Cyclist loose 下降来源。

**日志子指标分解（ep12-16 均值，Cyclist）**：

| Cyclist | 基线 | 本次 | Δ |
|---|---:|---:|---:|
| 3D loose (moderate) | 48.63 | 43.69 | −4.94 |
| 3D strict (moderate) | 21.64 | 24.36 | **+2.71** |
| BEV loose | 51.88 | 44.71 | −7.17 |
| BEV strict | 33.88 | 32.92 | −0.96 |
| 2D loose | 53.53 | 46.39 | −7.14 |
| 2D strict | 39.56 | 38.85 | −0.72 |

关键信号：**loose 掉、strict 反而升**。说明不是定位/朝向变差（strict 才是定位质量的度量），
而是「检出召回下降」——低 IoU 阈值下能命中的 GT 变少，而一旦匹配上（strict 0.5），质量反而更好。

**逐样本匹配（IoU≥0.25，one-to-one）**：

| ckpt | TP | FP | FN | recall@0.25 |
|---|---:|---:|---:|---:|
| base ep16 | 1577 | 267024 | 576 | **0.732** |
| refine ep20 | 1414 | 205317 | 739 | **0.657** |
| refine ep24 | 1413 | 249592 | 740 | **0.656** |

Cyclist 漏检 **FN 从 576 → 739（+163）**，recall 从 0.732 掉到 0.657，**下降约 7.5 个点**。
Cyc 预测框数量/分数分布基本不变（raw 输出分数中位数 ~0.03，分布一致），说明不是「整体压分」，
而是「特定目标被漏检」。

结论：**Cyclist loose 下滑的根因 = Truck refine 分支对共享 BEV 特征的梯度干扰，导致 Cyclist
检出召回下降（+163 漏检），而非定位变差或分类混淆。** strict 不降反升进一步佐证定位/朝向无碍，
纯粹是「检出变少」。

（注：可直接推理的验证实验 = 给 `truck_refine_conv` 的输入做 `detach()`，阻断其梯度回流共享
特征，重训看 Cyc recall 是否回升到 ~0.73；若回升则坐实梯度干扰是根因。）

> **后续修正（见第 27 节）**：该验证实验已执行且**否证**了此假设——detach 后 Cyclist 仍 −5.11
> （与非 detach −4.94 几乎相同），Truck 增益则从 +2.85 反转为 −2.79。Cyclist 下滑并非
> refine 分支额外梯度回流所致；Truck 提升反而依赖该梯度改写共享特征。结论：放弃 Truck-refine。

### 26.6 下一步

- [x] seed0 训练完成（ep1-24，全 checkpoint）。
- [x] 通过标准核对：Truck/Car 达标，Cyc 超限，seed0 未干净通过。
- [x] Cyclist 下滑根因诊断：检出召回 −7.5pt（FN +163），定位/朝向无碍。
- [x] 反证实验：`truck_refine_conv` 输入 `detach()` 已执行（第 27 节）→ **否证梯度干扰假设**。
- [x] 结论：Truck/Overall 均未通过 detach 版 → 放弃 Truck-refine 支线（细节见第 27 节）。

---

## 27. Truck-refine detach 版（阻断梯度回流，seed0）

### 27.1 背景与目标

第 26 节 seed0 结论：Truck strict `+2.85`、Car strict `+1.38` 达标，但 **Cyclist loose `−4.94`
超限**（标准 ≤ −1.0）。逐样本匹配诊断：Cyclist 检出去 recall 从 0.732 → 0.657（FN +163），
strict 反升 +2.71，判定为「Truck refine 分支对共享 BEV 特征的梯度干扰」导致检出召回下降，
而非定位/朝向/分类问题。

本轮唯一改动：给 `truck_refine_conv` 的输入做 `detach()`，阻断 refine 分支梯度回流共享 BEV 特征，
验证 Truck 增益能否在「不改写共享特征」的前提下保留。保留当前实验作对照，故新增独立开关
`truck_refine_detach`（默认 False，不写死）。

### 27.2 改动

- `mmdet3d/models/dense_heads/anchor3d_head.py`
  - 新增参数 `truck_refine_detach=False`（init 签名 + `self.truck_refine_detach`）。
  - `forward_single` 中：`refine_input = x.detach() if self.truck_refine_detach else x`。
  - `truck_refine_conv` 参数仍正常训练；`(dx,dy,dl)` refinement 保留；loss 不再修改共享 BEV。
- 新配置：`configs/r4det/TJ4D-R4Det_motion_align_rssm_det3d_N4_2x4_24e_pretrained_v2_head_truck_detach.py`
  - `truck_refine=True / channels=64 / dims=(0,1,4) / truck_anchor_class=5 / truck_refine_detach=True`
- 提交：`1c95fc3 feat: add truck_refine_detach switch`

### 27.3 实验设置

- 固定不变：`seq_len=4`、`rssm_bptt_steps=1`、`hidden_dim=128`、`kl_scale=1.0`、`free_nats=1.0`、
  `samples_per_gpu=2`、`GradientCumulativeOptimizerHook(2)`、有效 batch 12、`lr=1.5e-4`、
  `max_epochs=24`、`checkpoint_interval=1`、pretrained_tj4d.pth、anchor/assigner/FocalLoss/IoU 不变。
- 运行：`seed=0`、`--deterministic`、`CUDA_VISIBLE_DEVICES=5,6,7`、3 进程。
- 工作目录：`work_dirs/truck_refine_detach_N4_2x4_24e_seed0`（新目录，避免混日志）。

### 27.4 通过标准（ep12-16 均值，Δ = 本次 − 基线 seed0）

- Truck strict Δ ≥ +2.0
- Car strict Δ ≥ −0.5
- Cyclist loose Δ ≥ −1.0
- Overall Δ ≥ +0.5
- 新增观察项：Pedestrian loose Δ ≥ 约 −1.2；Cyc+Ped 合计 Δ ≥ −2.23（因即使 Cyclist 完全恢复，
  Ped 若仍 −3.77，Overall 仅约 +0.12 仍不达标）

### 27.5 训练结果（seed0，ep1-24 完整曲线）

- 启动 2026-09-02 13:58，日志 `20260902_135846.log`；24 epoch 全程无中断，ep1-24 checkpoint 完整。
- 逐 epoch val（3D_moderate；Truck/Car 为 strict，Cyc/Ped 为 loose）：

| ep | Overall | Truck_strict | Car_strict | Cyc_loose | Ped_loose |
|---|---:|---:|---:|---:|---:|
| 1  | 19.01 |  3.09 | 24.37 | 43.04 |  5.55 |
| 2  | 28.43 | 13.85 | 40.72 | 39.06 | 20.11 |
| 3  | 33.26 | 20.02 | 41.29 | 46.21 | 25.50 |
| 4  | 33.20 | 16.63 | 44.93 | 46.21 | 25.03 |
| 5  | 31.21 | 18.53 | 39.84 | 43.54 | 22.95 |
| 6  | 32.29 | 25.25 | 49.45 | 35.79 | 18.66 |
| 7  | 37.86 | 27.85 | 46.71 | 45.25 | 31.64 |
| 8  | 32.84 | 20.11 | 41.66 | 44.13 | 25.46 |
| 9  | 37.40 | 27.99 | 49.17 | 48.32 | 24.13 |
| 10 | 37.99 | 25.08 | 47.29 | 49.68 | 29.91 |
| 11 | 33.78 | 26.66 | 39.97 | 39.98 | 28.51 |
| 12 | 36.79 | 28.64 | 44.35 | 43.34 | 30.84 |
| 13 | 36.87 | 25.73 | 50.62 | 41.19 | 29.96 |
| 14 | 36.84 | 26.25 | 46.05 | 43.72 | 31.34 |
| 15 | 34.87 | 23.63 | 40.65 | 46.66 | 28.54 |
| 16 | 37.06 | 22.89 | 50.46 | 42.70 | 32.17 |
| 17 | 36.12 | 18.01 | 49.38 | 47.26 | 29.85 |
| 18 | 37.87 | 26.58 | 48.93 | 43.43 | 32.53 |
| 19 | 36.60 | 24.94 | 48.57 | 40.79 | 32.12 |
| 20 | 36.02 | 27.07 | 47.28 | 39.45 | 30.28 |
| 21 | 37.43 | 25.71 | 49.91 | 41.94 | 32.16 |
| 22 | 36.17 | 25.57 | 48.35 | 39.99 | 30.77 |
| 23 | 36.38 | 23.72 | 48.55 | 42.43 | 30.82 |
| 24 | 36.68 | 22.95 | 49.28 | 42.97 | 31.52 |

### 27.6 通过标准核对（seed0，未通过）

按「ep12-16 均值」口径（基线 = `run10_headv2_multiseed/seed_0`，Δ = 本次 − 基线）：

| 指标 | 基线均值 | 本次均值 | Δ | 标准 | 判定 |
|---|---:|---:|---:|---|---|
| Truck strict | 28.21 | 25.43 | **−2.79** | ≥ +2.0 | ❌ 未达 |
| Car strict | 47.80 | 46.43 | −1.38 | ≥ −0.5 | ❌ 未达 |
| Cyclist loose | 48.63 | 43.52 | **−5.11** | ≥ −1.0 | ❌ 未达 |
| Pedestrian loose | 28.92 | 30.57 | +1.66 | ≥ 约 −1.2 | ✅ 通过 |
| Overall | 38.39 | 36.49 | −1.90 | ≥ +0.5 | ❌ 未达 |

（Cyc+Ped 合计 Δ = −5.11 + 1.66 = **−3.45**，也低于观察项 −2.23。）

- 后期 ep17-20 无恢复：Overall 36.12 / 37.87 / 36.60 / 36.02，均值 36.65；Truck 18.01 / 26.58 / 24.94 / 27.07，
  均值 24.15；Cyc 47.26 / 43.43 / 40.79 / 39.45，均值 42.74。
- 全轮峰值 = ep10 的 37.99，**仍低于基线平台值**（基线 ep12-16 均值 38.39）；最终 ep24 = 36.68。

### 27.7 核心结论（本轮明确失败，并推翻第 26 节部分判断）

1. **detach 版全线失败，不保留，不补 seed1/2。** Truck/Car/Cyclist/Overall 四项全跌破标准，
   仅 Pedestrian loose +1.66（附带改善，但不足以挽救 Overall）。

2. **推翻第 26 节「Cyclist 下降 = truck_refine_conv 额外梯度回流」的假设。**
   detach 之后 Cyclist 仍 −5.11，与非 detach 的 −4.94 几乎相同——Cyclist 下滑**不是**这条
   额外梯度回流造成的，detach 无法拯救 Cyclist。

3. **Truck 提升的真实来源是「梯度回流共享特征」。** 非 detach 时 Truck +2.85，detach 后 Truck
   变为 −2.79。说明第 26 节里 Truck 的 +2.85 恰恰**依赖** refine 分支梯度改写共享 BEV 特征；
   一旦切断这条梯度，Truck 修正完全失效还拖累 Car（−1.38）。

4. **detach 并非真正隔离（关键技术澄清）。** 即使输入 detach，输出仍为
   `bbox_pred = conv_reg(x); refine = truck_refine_conv(x.detach()); bbox_pred[truck] += refine`。
   Truck loss 仍通过共享 `conv_reg(x)` 回流到共享 BEV 特征；detach 只切断了 refinement 分支
   这一条梯度，没有隔离「Truck 目标通过共享 conv_reg 对 BEV 的梯度」。

5. **方向结论：** Truck 专属回归（无论 detach 与否）都无法同时满足「Truck/Overall 提升且
   Cyc/Car 不跌」。按原定分支规则（Truck 增收益依赖共享特征改写 → 放弃），**放弃 Truck-refine
   这一支**；Truck 定位差（中心 1.41m、长轴 3.02m）的根因需另寻路径（anchor 尺寸匹配 / 回归
   tower 结构），而非残差叠加式 refinement。

### 27.8 下一步

- [x] seed0 训练完成（ep1-24，全 checkpoint）。
- [x] 通过标准核对：四项全未通过，仅 Pedestrian loose 附带 +1.66。
- [x] 推翻第 26 节「梯度回流干扰」判断；确认 detach 非完全隔离。
- [ ] 放弃 Truck-refine 支线；下一步方向待定（候选：Truck anchor 尺寸/数量重估、Truck 独立
      regression tower、Car/Truck classification tower、或转向 Pedestrian BEV 分辨率/refinement）。



## 28. Truck 独立 regression tower（直接预测替换，非残差；seed0）

### 28.1 背景与决策
- 第 26/27 节结论：Truck residual refinement（detach / 非 detach）均无法在提升 Truck 的同时保住 Cyclist / Car。
- 用户拍板 **B 方案**：不重跑 baseline，直接以 `run10_headv2_multiseed/seed_0` 作干净对照。
- 干净基线确认：run10（08-23）早于污染 commit `f36c90c`（09-01），不受 truck_refine 影响；
  主配置已在 `98cc09d` 还原为干净完整 RSSM。

### 28.2 改动（唯一变量）
- Truck 的 (dx, dy, dl)（box-code indices `[0, 1, 4]`）改由独立 tower 直接预测，并用 `index_copy_`
  **替换**共享 `conv_reg` 的对应 Truck 通道（不再 `bbox_pred[truck] += refine`）。
- 共享 `conv_reg` 对这 6 个 Truck 通道不再收到梯度；Truck 其余通道（z / sin / cos / w / h）仍共享。
- 新增开关 `truck_tower` / `truck_tower_channels=64` / `truck_tower_dims=(0,1,4)`（默认关，不影响基线）。
- tower 结构：Conv3x3(256→64) + ReLU + Conv1x1(64→6)，末层 1x1 零初始化。
- 追加 `init_weights` 重写：在 super() 后重新置零末层（因为 `init_cfg` 默认 Normal 会覆盖 `_init_layers` 的零初始化）。
  → 顺带澄清：第 26/27 节 refine 末层此前实际被 `init_cfg` 覆盖成 Normal 初始化，而非零初始化。

### 28.3 实验设置
- 配置：`configs/r4det/..._v2_head_truck_tower.py`（= 干净 v2_head + `truck_tower=True`）
- GPU 5,6,7；seed0；seq_len=4 / BPTT1 / hidden128 / kl1.0 / free1.0；
  samples_per_gpu=2，cumulative_iters=2（有效 batch12）；lr=1.5e-4；24e；checkpoint_interval=1；`--deterministic`
- work_dir：`work_dirs/truck_tower_N4_2x4_24e_seed0`
- commit：`f6a4417 feat(head): Truck independent regression tower (replace dx/dy/dl channels)`

### 28.4 通过标准（ep12-16 均值，Δ = 本次 − 干净基线 seed0）
基线：Overall 38.39 / Truck strict 28.21 / Car strict 47.80 / Cyclist loose 48.63 / Pedestrian loose 28.92。

| 指标 | 标准（绝对阈值） |
|---|---|
| Overall | ≥ 38.89 |
| Truck strict | ≥ 30.21 |
| Car strict | ≥ 47.30 |
| Cyclist loose | ≥ 47.63 |
| Pedestrian loose | ≥ 27.92 |

（前四项必须同时满足；Pedestrian 为观察项，不降超 1.0。）

### 28.5 训练结果（seed0，ep1-24 全曲线，训练已全部完成）

（训练于 2026-09-04 20:53 完成 ep24 checkpoint，进程正常退出，无后续崩溃。
  中途 `/data` 磁盘写满导致 ep5 checkpoint 损坏，已从 epoch_4 完整恢复并重跑 ep5。）


| ep | Overall | Truck strict | Car strict | Cyclist loose | Ped loose |
|---:|---:|---:|---:|---:|---:|
| 1 | 18.68 | 2.60 | 29.01 | 38.57 | 4.52 |
| 2 | 25.04 | 12.31 | 40.63 | 27.21 | 20.01 |
| 3 | 32.03 | 19.07 | 44.73 | 38.15 | 26.17 |
| 4 | 32.87 | 23.18 | 45.03 | 41.36 | 21.90 |
| 5 | 31.79 | 21.48 | 46.87 | 40.72 | 18.08 |
| 6 | 32.55 | 22.38 | 49.66 | 40.44 | 17.70 |
| 7 | 36.88 | 24.45 | 51.24 | 42.00 | 29.82 |
| 8 | 35.98 | 27.75 | 48.06 | 39.33 | 28.79 |
| 9 | 37.17 | 27.02 | 51.61 | 41.12 | 28.93 |
| 10 | 35.89 | 25.92 | 47.94 | 42.57 | 27.15 |
| 11 | 35.57 | 28.70 | 41.69 | 45.52 | 26.35 |
| 12 | 37.35 | 29.88 | 50.80 | 40.16 | 28.57 |
| 13 | 38.68 | 29.63 | 52.11 | 46.15 | 26.85 |
| 14 | 36.67 | 28.69 | 50.16 | 41.03 | 26.79 |
| 15 | 37.53 | 28.58 | 47.44 | 45.75 | 28.37 |
| 16 | 38.74 | 31.59 | 53.65 | 41.84 | 27.87 |
| 17 | 36.94 | 31.01 | 48.79 | 39.79 | 28.17 |
| 18 | 38.36 | 28.98 | 50.14 | 43.56 | 30.76 |
| 19 | 37.65 | 30.89 | 50.26 | 43.03 | 26.40 |
| 20 | 37.95 | 32.09 | 49.09 | 41.83 | 28.78 |
| 21 | 36.88 | 31.12 | 46.86 | 41.04 | 28.50 |
| 22 | 37.20 | 30.85 | 49.07 | 40.53 | 28.33 |
| 23 | 36.78 | 30.05 | 48.46 | 41.73 | 26.86 |
| 24 | 36.81 | 31.31 | 49.16 | 40.17 | 26.61 |

### 28.6 通过标准核对（seed0，未通过）

ep12-16 均值口径（基线 = `run10_headv2_multiseed/seed_0`，Δ = 本次 − 基线）：

| 指标 | 基线均值 | 本次均值 | Δ | 标准 | 判定 |
|---|---:|---:|---:|---|:---:|
| Car strict | 47.80 | 50.83 | +3.03 | ≥ 47.30 | ✅ |
| Truck strict | 28.21 | 29.67 | +1.46 | ≥ 30.21 | ❌ 差 0.54 |
| Cyclist loose | 48.63 | 42.98 | −5.65 | ≥ 47.63 | ❌ 重挫 |
| Pedestrian loose | 28.92 | 27.69 | −1.23 | ≥ 27.92 | ❌ 微降 |
| Overall | 38.39 | 37.79 | −0.60 | ≥ 38.89 | ❌ |

### 28.6b 后期阶段（ep17-24）
- 未出现恢复：Overall 均值 37.32（ep17-24），Truck strict 均值 30.66，Cyclist loose 均值 41.35。
- BEST Overall = ep16 的 38.74（仍低于基线平台 38.39 之后的 +0.35，未达目标 38.89）。
- ep20 的 Truck strict 32.09 为全曲线最高，也印证 tower 对 Truck 确有增益；但 Cyclist loose
  始终在 39.8-43.6 之间徘徊，无法回到基线 48.63。


### 28.7 核心结论（本轮不通过；三次 Truck 回归实验统一指向同一根因）

1. **Truck tower 方向本身有效**：Truck +1.46、Car +3.03，证明「Truck 专属 (dx,dy,dl)
   独立预测替换共享 conv_reg」比共享 conv_reg 更能回归好 Truck/Car（此前 residual 版也验证了
   Truck 增益，但只有本版 Car 同时 +3.03）。

2. **代价是 Cyclist 重挫 −5.65**，直接把 Overall 打到基线以下（−0.60）。连续三次实验同模式：
   - 第26节 residual（非 detach）：Truck +2.85，Cyc −4.94
   - 第27节 residual（detach）：Truck −2.79，Cyc −5.11
   - 第28节 独立 tower（替换）：Truck +1.46，Cyc −5.65

3. **根因判断**：tower 仍以共享 BEV `x` 为输入，其参数梯度照样经 `truck_tower_conv` 反向回流
   进共享 BEV 特征。三次实验统一指向——**Truck 回归的改善总是抢占 Cyclist 所需的共享 BEV
   特征容量，这是零和**，不是「残差 vs 替换」的结构差异能解决的。

4. **判定**：按预设分支，三轮 Truck 专属回归（residual / detach / 独立替换皆）不通过，
   **停止当前 Truck 回归路径实验**，不再补 seed1/2。

### 28.8 下一步
- [x] 独立 tower seed0 完整跑完 ep1-24 全 checkpoint，按新标准核对：不通过。
- [x] 结论：Truck 回归改善总伴随 Cyclist 下滑，共享 BEV 特征容量是零和瓶颈。
- [x] 最终决策：停止 Truck 回归路径实验（residual / detach / 独立替换三轮全部不通过），不补 seed1/2。
- [ ] 待用户选定下一步方向（候选）：
  1. **转向 Pedestrian**（strict 长期≈0、loose 27-31，全场最硬短板）：更高 BEV 分辨率 /
     pedestrian refinement / 中心量化误差与高度尺寸误差分析。→ 优先级最高。
  2. Car/Truck classification tower（拆分类塔）——注意 26.6 诊断已排除分类混淆为首要
     （Truck 长轴 3.02m、中心 1.41m 是定位问题），预期收益有限，不建议优先。
  3. 回 clean full RSSM 复现（若担心累积干扰；run10 已确认 38.39 可作对照）。

---

## 29. Pedestrian 专项误差拆解（不训练；run10 seed0 ep16 干净基线 dump）

> 日期：2026-09-05。承接第 28 节「停止 Truck 回归、转向 Pedestrian（优先级最高）」。
> 全部 CPU-only，基于现成 dump `work_dirs/diag_dumps/cyc_diag/base_seed0_ep16.pkl`
> （md5 与 `diag_dumps/seed0_ep16.pkl` 完全一致 = `run10_headv2_multiseed/seed_0/epoch_16`
> = clean full RSSM + head-v2，未重新训练、未动 GPU）。
> 新工具：`tools/ped_error_breakdown.py`（对 Pedestrian 的 A–F 六项拆解）。

### 29.1 数据与口径
- dump：`base_seed0_ep16.pkl`，2040 个 val 样本，Pedestrian GT 计数 = 999。
- head：`Anchor3DHead`，Ped 3 组 anchor × 2 朝向 = 6 个 Ped anchor slots。
- test_cfg `score_thr=0.0`，所以 dump 里的 pred 含「sigmoid>0 的全部解码 anchor + NMS 后」
  的 raw 输出，可做 score 分布与 λ-recall（anchor 几何覆盖）分析，不受 score 阈值截断。
- 指标 frame：LiDAR `[x,y,z,w,l,h,yaw]`，w=横向(短轴)、l=纵向(长轴)。

### 29.2 A. Ped score 是否偏低？
- 全量 Ped anchor sigmoid：count=104907，**mean=0.0202，p50=0.0002，p90=0.054**，max=0.934。
- 每样本 max Ped score：mean=0.133，p50=0.057，p90=0.328。
- **72.7% 样本 max Ped score < 0.1，88.7% < 0.25，91.2% < 0.5**。
- 结论：同一帧内几乎绝大多数 Ped anchor 都被分类头压到 ~0，只有极少数 anchor 冒头。
  → AP 曲线上 Ped 主要落在 score∈[0,0.5] 的低分召回带，不是「分数高但框差」。

### 29.3 B. λ-recall（anchor 几何覆盖，去掉分类/score 因素）
对每个 GT Pedestrian，看「是否存在任一 Ped anchor（6 朝向）与 GT 的 3D IoU ≥ λ」：

| λ | 0.10 | 0.20 | 0.25 | 0.30 | 0.35 | 0.40 | 0.50 | 0.60 | 0.70 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 覆盖率 | 0.501 | 0.467 | **0.452** | 0.383 | 0.242 | 0.134 | **0.045** | 0.006 | 0.000 |

- 若 anchor 几何完美，这些值应 ≈1。实际 λ=0.5 只有 **4.5%**（与旧诊断「Ped strict 通过率
  ~4%」一致），λ=0.25 只有 45.2%。
- **这是决定性证据：即便忽略分类分数、只问「6 个朝向的 anchor 能否摆到 GT 头上去」，
  strict(0.5) 上限也只有 4.5%。也就是说模型被 anchor 几何 + 一刀切的同一 box-code
  残差缩放卡死了，而不是 NMS/score 阈值卡死。**
- 反过来说 λ=0.25 的 45.2% 也低，说明 **Ped anchor 的朝向命中率本就不够**（2 朝向 anchor
  + sin/cos 编码，缺少匹配 GT 朝向的分辨率），这正是 Cyclist 同类问题（Cyc loose 48.6
  一样长期卡在 ~50）。

### 29.4 C+D. 正确召回 Ped 的逐分量误差（n=540）
- center BEV：mean=0.323m / median=0.197m；**73.5% ≤0.35m**，38.1% ≤0.15m。
  中心误差并不受「BEV 0.16m 网格量化」主导——中位数 0.2m 略高于 0.16 网格，但可接受。
- w(横向)：mean=0.307 / median=0.316；**只有 27.6% ≤0.15m**；有 45.5% 落在 0.25–0.60m。
  → **横向尺寸误差是首要定位错误**（行人 w 才 ~0.4–0.7m，错 0.3m ≈ 错半个身子）。
- l(纵向)：mean=0.430 / median=0.249；78.7% ≤0.30m——纵向基本贴 GT，不是主要矛盾。
- h(高度)：mean=0.140 / median=0.104；89.1% ≤0.25m——高度最好，基本不是矛盾。
- yaw：mean=0.593rad / median=0.379rad；**42.4% ≤0.25rad**，但 33.5% >0.9rad（近 π/2）。
  → **朝向存在明显双峰：一批对了，一批差 ~90°（w/l 与 GT 垂直/互换）**。

### 29.5 F. loose-ok(0.25≤IoU<0.5) 而 strict-fail 的根因分解（n=311 配对）
- 单一致命因素命中：**w 至少差 0.15m 占 20.6%**；center 1.9%、yaw 1.6%、l 0%、h 0%。
- 多因素叠加 75.9%——不是某个单一量独占，而是 w⊕center⊕yaw 叠加把 IoU 从 [0.25,0.5) 拉不
  到 0.5。
- h/l 贡献为零。→ Ped strict 的失败不是「三维高度对齐」，本质是 2D BEV 的 w/朝向/中心
  精度问题，与旧结论「BEV 物理瓶颈」一致但更精确：**不是纯网格量化，是 anchor 宽度预测量 +
  朝向分辨 + 中心漂移的叠加**。

### 29.6 E. 距离与尺寸分桶
- 距离：loose/strict = 0-20m `0.778/0.024`，20-40m `0.453/0.060`，40-60m `0.150/0.000`。
  strict 全程被压死，但 20-40m 反而略高于 0-20m（0.060 vs 0.024，样本少、噪声级）。
- 尺寸（GT 长轴）：<0.55m `loose=0.689`, strict=0；0.55-0.7m `0.576/0.023`；
  0.7-0.85m `0.594/0.087`；0.85-1.0m `0.630/0.333`。**越大越容易 strict**：
  小 Ped（<0.55m）loose 高但 strict=0.000，是纯框质量被宽度误差吃掉。

### 29.7 决策：单一路线 = Pedestrian 专属 box refinement（路线 2）
三点证据共同指向「加 Ped 专属 7D box refinement」而不是高分辨率 BEV：

1. **B 表 λ=0.5 只有 4.5%**：高分辨率 BEV/Pillar 缩小中心量化只能救 center（D 表 center
   74% ≤0.35m，已不是主矛盾），救不了 w/yaw 的 anchor 几何上限——换网格不改 anchor 几何。
2. **w 误差 45% >0.25m + yaw 33% >0.9rad**：需要「完整 7D box residual」，只加 anchor
   不解决；detach 输入第一版只验证检测头容量（信噪优先，避免再次污染 Cyclist）。
3. **score 低（A）是下游症状**：一旦框好、可严格匹配的 anchor 变正样本、分得更高，score
   自然上移；因此先修回归、不先修分类/匹配（对应用户给的 branch 3 先不动）。

唯一改动：`Anchor3DHead` 新增 `ped_refine` / `ped_refine_dims=(0,1,3,4,5,6)`（7D=dx,dy,dw,dl,dh,yaw）
+ 输入 `x.detach()`，只覆盖 3 组 Ped anchor（6 个 slots × 7D），输出残差 `index_add_`。
基线（开关关）完全不变。训练设置按用户固定模板（GPU 5,6,7 / seed0 / deterministic /
seq4 / bptt1 / bs2 / cum2 / lr1.5e-4 / 24e / ckpt1）。

### 29.8 实验启动（Pedestrian 专属 7D box refinement，唯一变量）
- 配置：`configs/r4det/TJ4D-R4Det_motion_align_rssm_det3d_N4_2x4_24e_pretrained_v2_head_pedrefine.py`
  （= clean full RSSM + head-v2，唯一加 `ped_refine=True` + `ped_refine_detach=True`）。
- 实现：`Anchor3DHead` 新增 `ped_refine` 分支，完整 7D residual（dx,dy,dz,dw,dl,dh,dyaw）
  覆盖 3 组 Ped anchor（6 slots × 7D = 42 通道），输入 `x.detach()`，零初始化末层，默认关。
  `n_out = len(_ped_refine_inds) = 42`（首版误写 14 已修复 commit `3607fae`）。
- 训练设置固定模板：GPU 5,6,7 / seed0 / `--deterministic` / seq_len=4 / bptt1 /
  samples_per_gpu=2 / cumulative_iters=2（有效 batch12）/ lr=1.5e-4 / 24e / ckpt_interval=1。
- work_dir：`work_dirs/ped_refine_N4_2x4_24e_seed0`（→ `/data/lurui/work_dirs/...`）。
- 启动：2026-09-05 05:01 UTC，launcher `/tmp/launch_ped_refine.sh`（nohup + setsid 脱离会话）。
- 启动健康检查：ep1 iter50 loss_bbox=1.75 loss_cls=1.15 loss_iou=0.31 grad_norm=12.82，
  GPU 5/6/7 各 ~18.3GB / 99% util，eta ≈ 22.5h（ep1-24）。
- commit：`5a2904f feat / 3607fae fix`。

### 29.9 通过标准（复述本节/用户模板）
- Overall ep12–16 ≥ 38.89；Car/Cyclist/Truck 各自较略干净基线不降超 0.5；
  Pedestrian loose 不降超 0.5；Pedestrian strict 至少出现可重复的绝对提升。
- 干净基线（run10 seed0 ep12-16 均值）：Overall 38.39 / Car 47.80 / Truck 28.21 /
  Cyc loose 48.63 / Ped loose 28.92 / Ped strict ~0.12。

### 29.10 训练结果（seed0，ep1–24 全曲线，训练已全部完成）

训练于 2026-09-06 03:06 完成 ep24 checkpoint，进程正常退出，无 NaN、无梯度爆炸。
24 个 checkpoint 全部落盘，`latest.pth -> epoch_24.pth`。

| ep | Overall | Ped strict | Ped loose | Car strict | Truck strict | Cyc loose |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16.26 | 0.18 | 9.54 | 25.53 | 2.50 | 27.49 |
| 2 | 23.06 | 0.07 | 14.80 | 38.18 | 10.01 | 29.25 |
| 3 | 28.14 | 0.02 | 25.40 | 39.79 | 14.54 | 32.82 |
| 4 | 29.98 | 0.07 | 25.88 | 39.40 | 19.18 | 35.44 |
| 5 | 31.45 | 0.02 | 24.77 | 36.98 | 19.50 | 44.56 |
| 6 | 33.35 | 0.08 | 27.12 | 44.05 | 21.60 | 40.61 |
| 7 | 35.30 | 0.08 | 29.60 | 44.89 | 27.13 | 39.57 |
| 8 | 36.83 | 0.13 | 27.26 | 46.54 | 28.95 | 44.55 |
| 9 | 34.59 | 0.17 | 28.55 | 40.86 | 25.33 | 43.63 |
| 10 | 35.46 | 0.20 | 29.36 | 43.32 | 24.37 | 44.77 |
| 11 | 34.45 | 0.11 | 26.39 | 43.25 | 26.19 | 41.97 |
| 12 | 38.10 | 0.11 | 28.71 | 50.03 | 30.31 | 43.34 |
| 13 | 33.47 | 0.15 | 26.74 | 43.80 | 26.68 | 36.66 |
| 14 | 36.63 | 0.29 | 31.56 | 48.02 | 24.43 | 42.51 |
| 15 | 34.94 | 0.17 | 29.49 | 41.80 | 29.81 | 38.65 |
| 16 | 34.84 | 0.32 | 29.81 | 44.47 | 25.82 | 39.27 |
| 17 | 35.47 | 0.24 | 29.58 | 46.64 | 26.50 | 39.17 |
| 18 | 35.65 | 0.24 | 30.37 | 41.67 | 28.17 | 42.38 |
| 19 | 35.17 | 0.07 | 27.80 | 42.51 | 29.99 | 40.38 |
| 20 | 36.68 | 0.17 | 28.75 | 45.60 | 32.95 | 39.42 |
| 21 | 35.23 | 0.14 | 26.21 | 42.77 | 31.56 | 40.38 |
| 22 | 36.91 | 0.19 | 29.09 | 45.60 | 31.61 | 41.32 |
| 23 | 35.93 | 0.12 | 28.05 | 43.64 | 31.38 | 40.63 |
| 24 | 36.09 | 0.25 | 28.29 | 45.68 | 31.19 | 39.24 |

### 29.11 通过标准核对（seed0，未通过）

ep12–16 均值口径（基线 = `run10_headv2_multiseed/seed_0`，Δ = 本次 − 基线）：

| 指标 | 基线均值 | 本次均值 | Δ | 标准 | 判定 |
|---|---:|---:|---:|---|:---:|
| Overall | 38.39 | 35.60 | −2.79 | ≥ 38.89 | ❌ |
| Car strict | 47.80 | 45.62 | −2.18 | ≥ 47.30 | ❌ |
| Truck strict | 28.21 | 27.41 | −0.80 | ≥ 27.71 | ❌ |
| Cyclist loose | 48.63 | 40.09 | −8.54 | ≥ 48.13 | ❌ |
| Pedestrian loose | 28.92 | 29.26 | +0.34 | ≥ 28.42 | ✅ |
| Pedestrian strict | 0.22 | 0.21 | −0.01 | 可重复绝对提升 | ❌ |

> 注：基线 Ped strict ep12–16 实测 mean=0.22（非 29.9 所写的 ~0.12；0.12 是 19.3 节
> 三 seed BEST 均值的口径），Pedestrian 属观察项，本次 Δ 在该口径下约 0，仍在噪声带内。

### 29.12 核心结论（本轮不通过；Ped 定向变量走完第一轮闭环）

1. **Pedestrian strict 无实质性提升**：ep12–16 mean 0.21 vs 基线 0.22，全程 0.02–0.32
   噪声带内震荡，无任何 epoch 出现「可重复的绝对提升」。这与 29.3 的诊断预判一致——
   staged I 诊断已指出 λ-recall@0.5=4.5%，行人 strict 的上限被 anchor 几何 + 宽度朝向
   精度卡死，单靠一个 detach 的 7D 残差 refine 无法突破该上限。
2. **Pedestrian loose 是唯一达标项**（+0.34，且严格不降），证明残差分支没有破坏 Ped 的
   召回侧。
3. **代价集中在 Cyclist：−8.54**，再次复现第 26/27/28 节「专属回归分支伤 Cyc」的模式。
   与 Truck 精修三轮不同的是，本次用的是 `x.detach()`（理论上阻断回传进共享 BEV），
   但 Cyc 仍重挫，说明损害不是梯度回流到共享 BEV 造成的，而是 **head 内共享 `conv_cls`/
   `conv_dir`/`conv_reg` 的正样本分配与容量在 anchor 层被 Ped 精修间接占用**，
   或 detach 只阻断了 BEV 梯度、未阻断 head 前端特征的竞争。
4. **Car strict 被拖低 −2.18 且方差大（33.47–50.03）**，与 Cyc 一起把 Overall 打到
   35.60（−2.79）。Truck strict −0.80 属小幅负影响。
5. **判定**：按预设分支，Ped 专属 7D box refinement（detach 版第一轮）不通过，
   **停止该单一路线，不再补 seed**。回收结论：Ped strict 不是「检测头回归容量不足」，
   回到 29.3 的 `λ-recall@0.5=4.5%` 主证——Ped 的杠杆在 **anchor 几何 / 宽度-朝向编码
   分辨率**（或改为 point/center 头），这一步不靠加残差解。

### 29.13 下一步（候选，按优先级）

1. **Ped anchor 几何层改造**（新增量最小的正向尝试）：
   - 增加 Ped 朝向分辨率（rotations 增加 fan/grid 角度），或引入 width/heading 分离回归
     + fine-bi 朝向编码，直接命中 λ-recall 瓶颈；
   - 或激进：Ped 改用 point/center-based 头（但与当前 Anchor3DHead 架构正交，成本高，谨慎）。
2. **回 clean full RSSM 复现**：若担心累积干扰，run10 已确认 38.39 可作对照（本轮基线
   与历史 clean 基线一致，主因果链仍然成立）。
3. Car/Truck 分类塔拆法（26.6 已排除为首要，收益有限，暂不优先）。

### 30.1 实验设计（Pedestrian anchor 朝向 2→4，唯一变量）
- 新配置：`configs/r4det/TJ4D-R4Det_motion_align_rssm_det3d_N4_2x4_24e_pretrained_v2_head_pedrot4.py`
  （= clean full RSSM + head-v2，唯一改动 `rotations=[0.0, 0.785398, 1.570796, 2.356194]`）。
- 其余全部冻结：ped_refine 关 / BEV / voxel / RSSM / KL / BPTT / loss / lr / seq_len=4 /
  samples_per_gpu=2 / cumulative_iters=2 / max_epochs=24。
- 目标：缩短 Ped yaw 初始匹配误差 45°→22.5°。
- 训练设置固定模板：GPU 5,6,7 / seed0 / `--deterministic` / ckpt_interval=1。
- work_dir：`/data/lurui/work_dirs/pedrot4_N4_2x4_24e_seed0`。
- 启动：2026-09-06 08:52 UTC，launcher `/tmp/launch_pedrot4.sh`（nohup + setsid）。

### 30.2 健康检查（iter50/100/200）
- conv_cls 96→conv_reg 168→conv_dir_cls 48→conv_iou 24 通道，与 24 anchors（6 size × 4 rot）完全一致。
- iter50 loss_bbox=1.89 loss_cls=1.13 loss_iou=0.28 grad_norm=10.7；iter200 loss 降至 2.37，梯度正常。

### 30.3 ⚠️ 重要技术提示：`rotations` 是跨全部 size 共享的
`Anchor3DRangeGenerator.anchors_single_range` 对 **每个 anchor size** 都套用同一个
`self.rotations`（`single_level_grid_anchors` 里 `zip(ranges,sizes)` → `anchors_single_range(...,self.rotations)`）。
因此本配置下 anchors 数量 = 6 size × 4 朝向 = **24**（不是「Ped 6→12」）。
也就是说 Cyc/Car/Truck 也各从 2 朝向变成 4 朝向，rotation-only 并不只作用于 Ped。
- Ped 达标口径（Ped loose ≥28.42 / strict 稳定提升）仍按计划直接观察；
- 若 Cyc/Car/Truck 由此受损、或 strict 无稳定提升，下一步按预设分支直接评估
  **Pedestrian 独立 point/center-based head**，RSSM 保持当前完整版本冻结，不再堆 anchor/改 RSSM。

### 30.4 训练结果（seed0，ep1–24 全曲线，训练已全部完成）

训练于 2026-09-07 06:19 完成 ep24，三卡 `[RANK 0/1/2] FINISH` 正常退出，无 NaN、无梯度爆炸。
24 个 checkpoint 全部落盘，`latest.pth -> epoch_24.pth`。

| ep | Overall | Ped strict | Ped loose | Car strict | Truck strict | Cyc loose |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 17.26 | 0.040 | 4.17 | 21.57 | 3.81 | 39.48 |
| 2 | 25.13 | 0.035 | 10.24 | 32.94 | 15.69 | 41.64 |
| 3 | 29.22 | 0.047 | 22.95 | 36.55 | 17.26 | 40.14 |
| 4 | 26.55 | 0.026 | 22.49 | 37.31 | 14.47 | 31.94 |
| 5 | 32.75 | 0.052 | 23.24 | 39.94 | 25.09 | 42.73 |
| 6 | 30.49 | 0.058 | 20.02 | 40.03 | 23.29 | 38.62 |
| 7 | 34.17 | 0.287 | 28.55 | 42.62 | 27.43 | 38.09 |
| 8 | 35.68 | 0.037 | 27.02 | 46.18 | 27.32 | 42.19 |
| 9 | 37.36 | 0.091 | 28.22 | 46.29 | 31.13 | 43.82 |
| 10 | 37.94 | 0.210 | 28.65 | 45.83 | 32.43 | 44.84 |
| 11 | 36.34 | 0.096 | 26.67 | 45.60 | 26.79 | 46.28 |
| 12 | 37.84 | 0.110 | 29.62 | 48.71 | 30.53 | 42.51 |
| 13 | 38.47 | 0.157 | 29.05 | 50.45 | 31.18 | 43.21 |
| 14 | 39.52 | 0.269 | 30.98 | 51.90 | 30.88 | 44.30 |
| 15 | 35.96 | 0.172 | 29.40 | 39.65 | 29.52 | 45.29 |
| 16 | 36.94 | 0.100 | 29.22 | 44.89 | 28.60 | 45.05 |
| 17 | 38.76 | 0.240 | 30.07 | 49.36 | 29.25 | 46.36 |
| 18 | 37.51 | 0.154 | 30.15 | 44.52 | 30.71 | 44.66 |
| 19 | 36.51 | 0.136 | 28.16 | 43.83 | 31.19 | 42.87 |
| 20 | 38.47 | 0.195 | 30.05 | 49.03 | 31.82 | 43.00 |
| 21 | 37.77 | 0.154 | 29.87 | 47.59 | 29.38 | 44.25 |
| 22 | 38.01 | 0.139 | 29.37 | 47.46 | 31.80 | 43.40 |
| 23 | 37.13 | 0.189 | 29.79 | 44.66 | 30.36 | 43.69 |
| 24 | 36.59 | 0.135 | 27.69 | 45.58 | 31.29 | 41.79 |

### 30.5 通过标准核对（seed0 ep12–16 均值，未通过）

基线 = clean full RSSM（run10 seed0，Overall 38.39；Δ = 本次 − 基线）：

| 指标 | 基线 | 本次 | Δ | 标准 | 判定 |
|---|---:|---:|---:|---|:---:|
| Overall | 38.39 | 37.75 | −0.64 | ≥ 38.89 | ❌ |
| Car strict | 47.80 | 47.12 | −0.68 | ≥ 47.30 | ❌ |
| Truck strict | 28.21 | 30.14 | +1.93 | ≥ 27.71 | ✅ |
| Cyclist loose | 48.63 | 44.07 | −4.56 | ≥ 48.13 | ❌ |
| Pedestrian loose | 28.92 | 29.65 | +0.73 | ≥ 28.42 | ✅ |
| Pedestrian strict | 0.22 | 0.161 | −0.06 | 稳定绝对提升 | ❌ |

（ep17–24 末段 Overall 均值 37.65，Best 单点 = ep14 39.52 / ep17 38.76，均未突破标准；
Ped strict 全程 0.026–0.287 噪声带内，无任何一个 epoch 形成可重复提升。）

### 30.6 核心结论（rotation-only 不通过，第一轮闭环）

1. **Pedestrian strict 无提升**：ep12–16 mean 0.161 vs 基线 0.22，全曲线 0.026–0.287 噪声带内
   震荡（ep7 0.287 / ep14 0.269 均为单点，随后即回落，无法复现）。yaw 初始误差 45°→22.5°
   没有带来 strict 的实质改善，与 A/B/C/D 表诊断中「strict 被 w/yaw 误差上限卡死、而非
   anchor 匹配角分辨率不足」一致——旋转锚点只缩短了初始 angle 差，却没有消除回归器在
   w（宽度）与 yaw 上的误差上限，strict 匹配需要的高质量完整七维框仍然给不出来。
2. **Cyclist loose 被拖低 −4.56（48.63→44.07）**：rotation-only 并未真正定位到 Ped
   ——`Anchor3DRangeGenerator.anchors_single_range` 对 6 个 size 全部复用同一个
   `self.rotations`，因此 Cyc/Car/Truck 也各多出 45°/135° 朝向，Cyc 首当其冲受伤，
   重复第 26–29 节「anchor 层变更伤 Cyc」的模式。
3. **Truck strict +1.93 / Ped loose +0.73 是仅有的正向项**，但无法抵消 Overall −0.64；
   Car strict 方差极大（39.65–51.90），anchor 信道翻倍引入了明显不稳定。
4. **判定**：rotation-only 第一轮不通过。按预设分支执行——**停止继续堆 anchor / 改 RSSM**，
   下一步直接评估 **Pedestrian 独立 center/point-based head**；RSSM 维持当前完整版本冻结，
   不再回到 anchor 层或 RSSM 调整。


---

## 31. Pedestrian 独立 CenterHead（stage-1 隔离，Anchor3DHead 冻结保 Car/Truck/Cyclist）

### 31.0 实验设置（launch record）

- 目的：第 30 节判定后，转向【Pedestrian 独立 center head】路线。共享 BEV 之上新增一个只
  检 Pedestrian 的 `CenterHeadkitti`，原 `Anchor3DHead` 完全不变并冻结，只训练新 CenterHead。
- 架构：
  - `ped_center_head = CenterHeadkitti(tasks=[dict(num_class=1, class_names=['Pedestrian'])], in_channels=256, share_conv_channel=64)`
  - `common_heads = dict(reg=(2,2), height=(1,2), dim=(3,2), rot=(2,2))`
  - 空间参数：`point_cloud_range=[0,-39.68,-4,69.12,39.68,2]`、`voxel_size=[0.32,0.32]`、
    `grid_size=[216,248,1]`、`out_size_factor=1`、`min_radius=1`、`gaussian_overlap=0.1`、`max_objs=100`
  - `pts_bbox_head`（Anchor3DHead）与原 clean 配置逐字段一致（校验 `dict(orig)==dict(new)` 成立）。
- 训练隔离：`ped_stage1=True`。冻结 RSSM、img/radar backbone+neck、depth_net、view_transformer、
  RCFusion、原 Anchor3DHead；冻结模块在训练时强制 `training=False`（BN 统计不漂移）。
  梯度只允许进入 `ped_center_head`；shared BEV 输入也 `.detach()`。
- checkpoint：`load_from=/data/lurui/work_dirs/run10_headv2_multiseed/seed_0/epoch_16.pth`
  （anchor/backbone/fusion 全量加载，仅新 CenterHead 为随机初始化权重）。
- 推理合并（`R4Det._simple_test_pts_dual`）：
  1. 删除 AnchorHead 中 `label==0`（Pedestrian）框；
  2. CenterHead 输出统一 `label=0`；
  3. 保留 AnchorHead Cyclist/Car/Truck；
  4. 拼接，不做跨类别 NMS。
- 调度：`GPU=5,6,7`（3 卡）、`samples_per_gpu=2`、`cumulative_iters=2`（有效 batch=12）、
  `AdamW lr=1e-3`、`max_epochs=12`、`score_threshold=0.0`、RepeatDataset×2（与 clean 同 epoch 口径）。
- 配置：`configs/r4det/TJ4D-R4Det_ped_centerhead_stage1_3x2x2_12e.py`
- 代码：`R4Det` 新增 sibling `ped_center_head` + `ped_stage1` 冻结；`centerpoint_head.CenterHeadkitti`
  修复 `box_type_3d` 兼容（LiDAR 直接调用而非 `[0]` 索引）。
- 工作目录：`work_dirs/ped_centerhead_stage1_3x2x2_12e_seed0`（`work_dirs` → `/data/lurui/work_dirs`）。
- 运行：`seed=0`、`CUDA_VISIBLE_DEVICES=5,6,7`、\
  `bash tools/dist_train.sh configs/r4det/TJ4D-R4Det_ped_centerhead_stage1_3x2x2_12e.py 3 \
  --seed 0 --deterministic --work-dir work_dirs/ped_centerhead_stage1_3x2x2_12e_seed0`。

### 31.1 通过标准

| 指标 | 标准 |
|---|---|
| Ped strict（ep8–12 mean） | ≥ 1.0 |
| Ped loose | ≥ 28.42 |
| Car strict | 与 clean 差异 ≤ 0.1 |
| Truck strict | 与 clean 差异 ≤ 0.1 |
| Cyclist loose | 与 clean 差异 ≤ 0.1 |

- 后三类应近乎完全不变（主网络冻结）。若 Ped strict 仍 <1.0，说明现有 0.32m BEV 特征本身不足，
  CenterHead 也救不了，下一步才考虑 Ped 高分辨率 BEV；若通过，再做第二阶段小学习率联合微调。

### 31.2 训练结果（seed0，ep1–12 全曲线，训练已全部完成）

训练于 2026-09-08 14:22 完成 ep12，三卡 `[RANK 0/1/2] FINISH` 正常退出，无 NaN、无梯度爆炸。
12 个 checkpoint 全部落盘，`latest.pth -> epoch_12.pth`。冻结网络全部 `requires_grad=False` 且 `training=False`，
仅 `ped_center_head`（含 bbox coder + 共享 head）参与训练；loss 正常收敛（`loss_heatmap` 0.45、
`loss_xy/z/whl/yaw` 均 →1e-4 量级，`grad_norm` 0.95）。

| ep | Overall | Ped strict | Ped loose | Car strict | Truck strict | Cyc loose |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 38.20 | 0.085 | 21.92 | 49.9772 | 30.4303 | 50.4709 |
| 2 | 38.72 | 0.058 | 24.01 | 49.9772 | 30.4303 | 50.4709 |
| 3 | 38.84 | 0.056 | 24.48 | 49.9772 | 30.4303 | 50.4709 |
| 4 | 39.30 | 0.050 | 26.33 | 49.9772 | 30.4303 | 50.4709 |
| 5 | 38.89 | 0.075 | 24.69 | 49.9772 | 30.4303 | 50.4709 |
| 6 | 39.63 | 0.067 | 27.66 | 49.9772 | 30.4303 | 50.4709 |
| 7 | 39.85 | 0.045 | **28.53** | 49.9772 | 30.4303 | 50.4709 |
| 8 | 38.69 | 0.027 | 23.90 | 49.9772 | 30.4303 | 50.4709 |
| 9 | 39.37 | 0.080 | 26.62 | 49.9772 | 30.4303 | 50.4709 |
| 10 | 39.51 | 0.167 | 27.15 | 49.9772 | 30.4303 | 50.4709 |
| 11 | 39.31 | 0.159 | 26.35 | 49.9772 | 30.4303 | 50.4709 |
| 12 | 39.35 | 0.164 | 26.51 | 49.9772 | 30.4303 | 50.4709 |

- Car/Truck/Cyclist 三列 **全程恒定**（12 个 epoch 逐 epoch 完全一致），与 clean ep16 四舍五入后相同，
  证明阶段-1 冻结 + BEV detach 生效：主网络与前向统计量没有任何漂移。
- Ped loose 全曲线峰值 = ep7 **28.53**，随后 ep8 跌到 23.90（lr 已退到 3.7e-4），ep9–12 在 26.3–27.2 窄带内
  反复震荡，无法站稳 28.42。
- Ped strict 全曲线 0.027–0.167 噪声带内震荡，与本方案之前所有 Ped 专项实验（29/30 节）同量级。

### 31.3 通过标准核对（seed0，未通过）

基线 = clean full RSSM（run10 seed0 ep16：Car strict 49.9772 / Truck strict 30.4303 /
Cyc loose 50.4709 / Ped loose 28.6426 / Ped strict 0.1036；Ped strict 达标口径取 ep8–12 mean）：

| 指标 | 本次 | 基线/标准 | 判定 |
|---|---:|---|:---:|
| Ped strict（ep8–12 mean） | **0.1195** | ≥ 1.0 | ❌ |
| Ped loose（ep8–12 mean） | 26.11 | ≥ 28.42 | ❌ |
| Ped loose（全曲线 best） | 28.53（ep7） | ≥ 28.42 | 仅单点 |
| Car strict（diff） | 0.0000 | ≤ 0.1 | ✅ |
| Truck strict（diff） | 0.0000 | ≤ 0.1 | ✅ |
| Cyclist loose（diff） | 0.0000 | ≤ 0.1 | ✅ |

- 后三类（Car/Truck/Cyclist）**完全冻结、0 差异通过**，符合「主网络冻结即不变」的设计预期。
- Ped strict ep8–12 mean 0.1195，距离 ≥1.0 差约一个量级；
  Ped loose ep8–12 mean 26.11，未回到 28.42，且 best（28.53）落在 ep7，不在 ep8–12 达标窗内。

### 31.4 核心结论（不通过；0.32m BEV 特征本身不足）

1. **后三类隔离验证完全通过**：Car/Truck/Cyclist 三个指标与 clean ep16 差异均为
   **0.0000**，逐 epoch 恒定。说明 sibling CenterHead 的加入没有扰动主网络：
   冻结 + `training=False` + BEV detach 三者同时生效，Anchor3DHead/BE/融合的 BN 统计无漂移。
   这坐实了「加独立 CenterHead 不影响现有指标」的结构可行性，为后续高分辨率 BEV 分支铺平了路。
2. **Ped 专项指标未达标**：Ped strict ep8–12 mean 0.1195（vs 标准 ≥1.0），Ped loose ep8–12 mean
   26.11（vs ≥28.42）。即便 CenterHead 以 heatmap 方式在全 BEV 网格上独立定位 Ped，
   strict 仍然卡在噪声带内——与用户预设分支的判据一致：**现有 0.32m BEV 特征本身不足以给出
   strict 所需的像素级定位精度，CenterHead 也救不了**。
3. **Curve 形态佐证**：Ped loose 在 ep7 碰到 28.53 后立刻回落到 23–27 振荡平台，
   与 29 节 C/D 诊断（w 与 yaw 的回归误差上限）一致：0.32m 体素下 Ped 目标的宽度/朝向
   观测信息不足，head 换型（anchor→center）无法补偿 BEV 的信息瓶颈。
4. **判定**：阶段-1 不通过。按预设分支进入下一步——**Ped 高分辨率 BEV**（保留冻结的
   Car/Truck/Cyclist 主网络，Ped 走更高分辨率的专用 BEV 分支 + CenterHead）。

### 31.5 下一步

- 优先验证 **Ped 高分辨率 BEV**：减小 Ped 专用 BEV 的 `voxel_size`（如 0.32→0.16/0.20），
  放大 `grid_size`，`point_cloud_range` 可裁到 Ped 常见深度窗（可视统计后定），
  其余沿用本节的 CenterHead 配置与 stage-1 冻结隔离（Car/Truck/Cyclist 归主 Anchor3DHead）。
- 保持通过口径不变：Ped strict ep8–12 mean ≥1.0、Ped loose ≥28.42、后三类 diff ≤0.1。
- 若高分辨率 BEV 仍未达 Ped strict ≥1.0，则回溯 29 节误差拆解，重点处理 w/yaw 的回归
  误差上限（而非继续换 head 或堆 anchor）。

## 32. CenterHead ep7/ep12 误差诊断 + NMS sweep（不训练）

### 32.0 诊断范围（launch record）

- 目的：第 31 节 CenterHead 阶段-1 不通过后，按「先诊断、不训练」要求，对三个 checkpoint
  生成 dump 并做 CPU-only 误差拆解，确定唯一分支：
  center 误差主导 → 高分辨率 BEV；center 正常且 w/yaw 主导 → 不动分辨率，改 head 先验/loss；
  NMS sweep 明显提 loose → 只修 NMS；score 低但框好 → 只调 heatmap loss。
- dumps（均由 `tools/dump_predictions.py` 生成，GPU 5）：
  - `diag_dumps/seed0_ep16.pkl` — clean full RSSM + head-v2（第 29/31 节基线，prior 已存在）
  - `diag_dumps/ped_center_ep7.pkl` — CenterHead ep7（loose 峰值 28.53）
  - `diag_dumps/ped_center_ep12.pkl` — CenterHead ep12（末轮）
  - `diag_dumps/ped_center_ep7_raw.pkl` — ep7 禁用 CenterHead NMS（circle r=0，保留全部 100
    个 decoded box/样本）用于 CPU NMS sweep。
- 脚本：`tools/ped_center_diag.py`（逐 checkpoint）、`tools/ped_nms_sweep.py`（raw sweep）。
- 口径：Pedestrian（class 0）单类、LiDAR 帧、greedy 一对一匹配（KITTI 同序，按 score 降序）。
  3D IoU 为 BEV 多边形 × 高度重叠（与 eval 一致）。loose=0.25、strict=0.5。

### 32.1 完整 GT recall@0.25 / @0.5（三 checkpoint 对比，Ped GT=999）

| checkpoint | recall@0.25 | recall@0.5 |
|---|---:|---:|
| clean ep16 | 0.332 | 0.021 |
| Center ep7 | 0.300 | 0.012 |
| Center ep12 | 0.285 | 0.006 |

- CenterHead 的完整 Ped GT recall 反而略低于 clean baseline（0.300 vs 0.332 @0.25），
  strict 也略低（0.012 vs 0.021）。head 换型没有带来 recall 提升，仅改变了匹配到的
  box 的分布（见 32.2 误差）。

### 32.2 匹配对误差（mean，center / w / l / h / yaw）

| 轴 | clean ep16 | Center ep7 | Center ep12 | ep7 相对 clean 改善 |
|---|---:|---:|---:|---:|
| center BEV (m) | 0.323 | 0.213 | 0.209 | −34.1% |
| w across (m) | 0.307 | 0.283 | 0.311 | −7.8% |
| l along (m) | 0.430 | 0.204 | 0.239 | −52.6% |
| h (m) | 0.140 | 0.118 | 0.122 | −15.7% |
| yaw (rad) | 0.593 | 0.504 | 0.517 | −15.0% |

- CenterHead 大幅改善 center（−34%）、l（−53%）、yaw（−15%）、h（−16%），
  说明 heatmap 定位与 per-pixel reg 确实优于 anchor 回归。
- **w（across 宽度）几乎不动（−7.8%）**：clean 0.307 → ep7 0.283 → ep12 0.311，
  仍停在 0.28–0.31 m。w 是唯一「换 head 也压不下去」的轴。

### 32.3 loose-ok-strict-fail 的失效原因（非互斥，逐轴超限率）

| 轴超限 | clean ep16 | Center ep7 | Center ep12 |
|---|---:|---:|---:|
| center >0.15m | 0.415 | 0.406 | 0.366 |
| w >0.15m | 0.855 | **0.920** | **0.928** |
| l >0.30m | 0.170 | 0.049 | 0.215 |
| h >0.25m | 0.039 | 0.045 | 0.018 |
| yaw >0.25rad | 0.547 | 0.569 | 0.591 |
| ≥2 轴同时超限 | 0.759 | 0.750 | 0.781 |

- 在 loose 命中但 strict 失败的样本中，**w 超限在 92% 以上**（ep7/12），是压倒性主因；
  其次 yaw（≈57%）与 center（≈37%）；l/h 几乎不构成瓶颈。
- CenterHead 已把 center/l/h 压到接近容限，但 w 仍系统性地超限。
  结论：**w（across 宽度）主导 strict 失败**，与第 29 节结论一致。

### 32.4 预测 w/l/h 分布 vs GT

| 轴 | pred (Center ep7) | GT |
|---|---:|---:|
| w mean/median | 0.593 / 0.573 | 0.580 / 0.542 |
| l mean/median | 0.569 / 0.560 | 0.604 / 0.588 |
| h mean/median | 1.655 / 1.620 | 1.716 / 1.697 |

- 均值/中位数都对齐 GT，问题不在全局维度偏置，而在**逐目标的 w 抖动**（0.28m 误差）。

### 32.5 NMS sweep（ep7，CPU-only，LiDAR 帧 greedy recall）

| variant | ped box/样本 | recall@0.25 | recall@0.5 |
|---|---:|---:|---:|
| 原配置（post-baseline circle r=1.0） | 12.9 | 0.3003 | 0.0120 |
| circle r=0.5 | 26.9 | 0.3043 | 0.0120 |
| circle r=0.25 | 96.2 | 0.2973 | 0.0120 |
| rotate IoU=0.2 | 61.6 | 0.2973 | 0.0120 |
| 无 NMS（top100 raw） | 100.0 | 0.2973 | 0.0120 |

- 放松 NMS 到 r=0.25 / rotate 0.2 / 完全无 NMS，drill-down 后 recall@0.25 只波动 ±0.004、
  recall@0.5 恒为 0.0120，**均无实质提升**。
- 判定：**NMS 不构成 recall 瓶颈**。当前 `circle r=1m` 过的不是密集行人的漏检主因。
  因此「只修 NMS」分支排除。

### 32.6 分支判定（唯一）

1. center 误差：CenterHead 已把 center BEV 误差从 0.323 压到 0.21（−34%），
   且 loose-ok-strict-fail 中 center 超限率 37–41%，**不是主导**。
2. **w（+yaw）主导**：w 换 head 后几乎不变（−7.8%）、92% 的 strict 失败样本 w 超限。
   yaw 超限 57% 次主因。它们都是 0.32m BEV 下的信息瓶颈（Ped 宽 ~0.6 m ≈ 2 体素）。
3. NMS sweep 无提升 → 排除「只修 NMS」。
4. score 低但框质量正常？→ 不成立：框质量（尤其 w）本身不达标，不是 score 问题。

⇒ **唯一分支：改 CenterHead 尺寸先验 + yaw/corner/IoU loss，并同步重做 0.16m 高分辨率
Ped BEV**。依据第 32.2 的 −7.8% w 改善：即使给 head 加先验/loss，0.32m BEV 也无法提供
亚体素宽度信息，w 的硬上限就是体素分辨率。因此分辨率必须上 0.16m（而非仅改 loss）。

### 32.7 高分辨率参数确认（若启动训练）

- 横向裁剪 `ped_point_cloud_range = [0, -20, -4, 69.12, 20, 2]`，
  `ped_voxel_size = [0.16, 0.16]`、`ped_grid_size = [432, 250, 1]`。
- 覆盖验证（基于 clean ep16 dump）：val 999 个 Ped GT 中 995 个 `|y| ≤ 20`，
  **995/999 = 99.6%** 覆盖，符合设定。
- 网格量 432×250 = 108,000，约为原 216×248 = 53,568 的 **2.0 倍**（全范围 0.16m 的 4 倍
  可控口径一致）。
- 必须重新做 0.16m LSS pooling + SECOND 前的 0.16m radar scatter，**禁止插值现有 0.32m BEV**。

### 32.8 结论

- 本次只做诊断与 sweep，**未启动高分辨率训练、未补 seed**。
- 支路已被排除：只修 NMS；只调 heatmap loss；纯 center 高分辨率。
- 已确认唯一路径：**Ped 0.16m 高分辨率 BEV + CenterHead（w/yaw 仍主导）**。

---

## 33. 固定尺寸先验验证（零训练，CenterHead 解码后 NMS 前覆盖 Ped w/l）

### 33.0 实验设置

- 目的：在没有重新训练的前提下，验证「w（宽度）是 Ped strict 主导瓶颈」这一第 32 节结论。
  在 `CenterHeadkitti` 解码后、NMS 前，仅对 Ped 覆盖 `boxes[..., 3] = fixed_x_size`、
  `boxes[..., 4] = fixed_y_size`（即 w/l）。center/z/height/yaw 继续使用 CenterHead 预测，
  score/NMS/主 AnchorHead/RSSM 全部不变。
- 实现：`mmdet3d/models/dense_heads/centerpoint_head.py` 的 `CenterHeadkitti.get_bboxes`
  新增 inference-only 分支，由 `test_cfg.fixed_size_prior=[fw, fl]` 开关，仅作用于
  `temp[i]['bboxes']` 解码结果，NMS 之前；不影响训练、不影响 AnchorHead。
- 两个非 val 泄漏先验（均为训练集口径，不接触 val GT）:
  - **A**：训练集 LiDAR 中位数 `0.655 × 0.628 m`
  - **B**：现有 Ped-small anchor `0.500 × 0.600 m`
- checkpoint：`epoch_7.pth` / `epoch_12.pth`（`work_dirs/ped_centerhead_stage1_3x2x2_12e_seed0`）。
- 评估：`tools/test_vod.py`（TJ4D eval），`--cfg-options model.ped_center_head.test_cfg.fixed_size_prior=[fw,fl]`，
  single-GPU，`CUDA_VISIBLE_DEVICES=5/6/7` 各跑一份，无训练。

### 33.1 通过标准

| 指标 | 标准 |
|---|---|
| Ped strict moderate | ≥ 1.0 |
| Ped loose moderate | ≥ 原 checkpoint − 0.5 |
| ep7 / ep12 都出现 strict 提升 | 是 |
| Car/Truck/Cyclist | 与原 checkpoint 0 差异 |

### 33.2 基线（原 checkpoint，训练日志口径）

| checkpoint | Ped strict mod | Ped loose mod |
|---|---:|---:|
| ep7 | 0.0452 | 28.5298 |
| ep12 | 0.1639 | 26.5118 |

### 33.3 复评结果（Ped 3D moderate）

| 先验 | checkpoint | strict | loose | Δstrict | Δloose | 判定 |
|---|---|---:|---:|---:|---:|---|
| A 0.655×0.628 | ep7 | **3.7382** | 27.6353 | +3.693 | −0.895 | strict 通过，loose 超标 |
| A 0.655×0.628 | ep12 | **3.6693** | 26.4229 | +3.505 | −0.089 | 通过 |
| B 0.500×0.600 | ep7 | **5.6055** | 27.2042 | +5.560 | −1.326 | strict 通过，loose 超标 |
| B 0.500×0.600 | ep12 | **7.3104** | 25.6589 | +7.147 | −0.853 | strict 通过，loose 超标 |

Car/Truck/Cyclist 所有 checkpoint、所有先验下逐位不变（Car strict 49.9772、
Truck strict 30.4303、Cyclist loose 50.4709，与训练日志一致，0 差异）。

### 33.4 判定

- **strict 全面暴涨**：两种先验下 ep7/ep12 的 Ped strict 都从 ~0 级（0.045/0.164）
  跃升到 3.7–7.3，远高于 ≥1.0 阈值，直接验证「w 预测是 Ped strict 主瓶颈」（第 32 节）。
- **先验 B（0.5×0.6 anchor）strict 增益大于 A（0.655×0.628 中位数）**：ep12 上
  B=7.31 vs A=3.67，ep7 上 B=5.61 vs A=3.74。更小更紧的 Ped 先验对 strict 更有利。
- **loose 为代价**：固定先验牺牲了逐目标宽度适配，loose 均有下降（−0.09 ~ −1.33）。
  仅「A @ ep12」满足 loose ≥ 原 −0.5；其余 3 组 loose 超标。
- **结论**：固定尺寸先验能把 Ped strict 从「无效」抬到「可用」，但 loose 同步受损，
  属于「以 loose 换 strict」的零训练上界验证，**不代表最终方案**。下一步仍然走
  0.16m 高分辨率 Ped BEV（保留逐目标 w 回归，而非固定先验）。

---

## 34. 软尺寸先验复评（零训练，log-space blend）

### 34.0 实验设置

- 目的：第 33 节硬先验（alpha=1.0）虽然 strict 暴涨，但 loose 普遍下降。本步在硬先验
  基础上引入 log-space blend，在「保住 loose」与「提升 strict」之间寻软着陆点。
- 实现：`CenterHeadkitti.get_bboxes` 解码后、NMS 前，仅对 Ped 的 w/l（`boxes[..., 3]` /
  `boxes[..., 4]`）做：
  ```
  final = exp((1-alpha)*log(pred) + alpha*log(prior))
  ```
  配置键 `test_cfg.size_prior_alpha=[fw, fl, alpha]`。center/z/height/yaw 继续用 CenterHead
  预测，score/NMS/AnchorHead/RSSM 全部不变。alpha=0 原样、alpha=1 退化为硬先验。
- 先验：A 训练集中位数 `0.655 × 0.628`；B Ped-small anchor `0.500 × 0.600`。
- alpha：0.50、0.75；checkpoint：ep7、ep12。
- 全部 `tools/test_vod.py` 零训练 single-GPU 复评（GPU 5/6/7）。

### 34.1 复评结果（Ped 3D moderate，strict / loose）

| 先验 × alpha | ep7 strict | ep7 loose | ep12 strict | ep12 loose | 备注 |
|---|---:|---:|---:|---:|---|
| 原始（无先验） | 0.0452 | 28.5298 | 0.1639 | 26.5118 | 训练日志口径 |
| A × 0.50 | 0.6879 | 28.2018 | — | — | |
| A × 0.75 | **2.4054** | **28.2016** ✅ | 1.9930 | 26.8525 | ep7 满足正式标准 |
| B × 0.50 | 2.5603 | 28.1777 | 2.1425 | 26.9962 | 0.15 微欠 loose |
| B × 0.75 | — | — | 5.5671 | 26.6590 | |

- Car strict=49.9772、Truck strict=30.4303、Cyclist loose=50.4709 在所有组合下逐位不变（0 差异）。
- ep7 Overall 3D moderate：原始 39.8520 → A×0.75 39.77（−0.082，基本持平）。

### 34.2 通过标准核对（优先 ep7）

| 指标 | 标准 | A×0.75 @ep7 | 判定 |
|---|---|---:|---|
| Ped strict moderate | ≥ 1.0 | 2.4054 | ✅ |
| Ped loose moderate | ≥ 28.03 | 28.2016 | ✅ |
| Car/Truck/Cyclist | 完全不变 | 0 差异 | ✅ |

更理想目标（strict≥3.0、loose≥28.5、Overall 不低于原 ep7）：

| 指标 | 目标 | A×0.75 @ep7 | 判定 |
|---|---|---:|---|
| Ped strict | ≥ 3.0 | 2.4054 | ✗（差 0.59） |
| Ped loose | ≥ 28.5 | 28.2016 | ✗（差 0.30） |
| Overall | ≥ 39.8520 | 39.77 | ✗（差 0.08） |

- **正式通过标准已达成**（A×0.75 @ ep7：strict 2.405 / loose 28.202，Car/Truck/Cyclist 0 差异）。
- 更理想目标未达：strict 2.405 < 3.0、loose 28.20 < 28.5、Overall 39.77 略低于 39.85。

### 34.3 与 raw dump 趋势对照

- 用户给出的 Ped-small（先验 B）raw dump 趋势：alpha 0.50 loose 0.313 / strict 0.076，
  alpha 0.75 loose 0.311 / strict 0.111，原始 loose 0.306 / strict 0.015。
- 正式 AP 复评方向一致：B 先验 strict 从 0.16（ep12）→ 2.14（α=.50）→ 5.57（α=.75），
  loose 基本守住（26.51 → 27.00 → 26.66）。软先验确实同时小幅保 loose、显著抬 strict。
- 但正式 AP 下 A 先验（中位数）是唯一同时过 strict/loose 阈值的组合，B 先验在 α 较低时
  保住 loose 略优于 A，却仍差 0.03–0.15 到 28.03 阈值线。

### 34.4 判定与下一步

- **软先验（alpha≈0.75）+ 中位数先验 A @ ep7 达成正式通过标准**：strict 2.405 ≥ 1.0、
  loose 28.202 ≥ 28.03、Car/Truck/Cyclist 0 差异。零训练下已能把 Ped strict 从无效抬到可用。
- 距离更理想目标仅差一档（strict 差 0.59、loose 差 0.30、Overall 差 0.08），说明**纯推理
  尺寸先验已接近其信息上限**，继续调 alpha / 换先验不足以突破。
- 按既定路线顺序，下一步进入 **训练有界尺寸残差**：
  ```
  prior = [0.655, 0.628]; max_log_residual = 0.25
  dim = prior * exp(max_log_residual * tanh(raw_dim))
  ```
  用真训练把 strict 推过 3.0、loose 抬过 28.5。若 bounded residual 训练后仍守不住 loose，
  再启动 0.16m 高分辨率 BEV（对应的软先验 alpha 正好落在 0.75 附近，且高分辨率让逐目标
  w 回归本身更准，进一步减少对先验的依赖）。

## 35. 有界尺寸残差训练（Ped dim branch，已完成）

### 35.1 目标与实现

- 只约束 Ped 的 LiDAR `x_size/y_size`（w/l），不重训整个 CenterHead，不加高分辨率/IoU/corner/yaw loss。
- 尺寸变换（decode 与 loss 共用同一公式）：
  ```ini
  prior_xy = [0.655454, 0.627535]
  max_log_residual = 0.25
  pred_log_xy = log(prior_xy) + 0.25 * tanh(raw_xy)
  pred_xy = exp(pred_log_xy)        # w/l
  pred_h = exp(raw_h)               # 高度保持原逻辑
  ```
- loss 拆分：
  ```ini
  loss_dim_xy = L1(pred_log_xy, log(gt_xy))   # size_xy
  loss_dim_h  = L1(raw_h, log(gt_h))          # size_h
  ```
- 监控项（无 `loss` 前缀，仅记录不回传梯度）：
  `ped_size_log_mae`、`tanh_sat_ratio`。
- 测试阶段不再叠加 `fixed_size_prior` / `size_prior_alpha`（这两个 inference-only 开关保留但本实验不用）。
- 冻结：R4Det 主网络全冻 + CenterHead shared_conv/heatmap/reg/height/rot 全冻，
  只训练 `task_heads.0.dim`。新增 `ped_stage2_dim` 分支（`_freeze_for_ped_stage2_dim`）。

### 35.2 训练配置

- config：`TJ4D-R4Det_ped_centerhead_stage2_dim_3x2x2_6e.py`
- work_dir：`/data/lurui/work_dirs/ped_centerhead_stage2_dim_3x2x2_6e_seed0`
- load_from = stage1 `epoch_7.pth`；`ped_stage2_dim=True`
- seed 0，GPU 5/6/7，lr 2e-4，samples_per_gpu 2，cumulative_iters 2（有效 batch 12），max_epochs 6。

### 35.3 结果与判定（已跑完）

- epoch_1 独立复评（修复 decode 形状广播 bug 后）：
  ```
  Ped  3D strict = 0.0441   （≈ 基线 0.0452）
  Ped  3D loose  = 28.2393  （≈ 软先验 28.20）
  Overall 3D mod = 39.7794
  Car moder strict = 49.9772 / Truck 30.4303 / Cyclist loose 50.4709（0 差异）
  ```
  epoch_1 时刻 dim 分支近似先验硬收缩（strict 仍接近基线，残差尚未学到逐目标变化）。
- 训练中 `ped_size_log_mae` 0.067 → 0.028，`tanh_sat_ratio` ≈ 0.01–0.03（无饱和），
  `grad_norm ≈ 0.002`（仅有 dim 分支梯度），`loss_rssm_kl` 随 KL 调度上跳但冻结模块不产生梯度。
### 35.4 最终复评结果（ped_centerhead_stage2_dim，seed0，6 epoch）

| epoch | Ped 3D mod strict | Ped 3D mod loose | Overall 3D mod |
|-------|------------------:|-----------------:|---------------:|
| 基线 ep7（未训练） | 0.0452 | 28.5298 | 39.8520 |
| 软先验 Aα.75 @ep7（零训练） | 2.4054 | 28.2016 | 39.7700 |
| 1（独立复评，修复后） | 0.0441 | 28.2393 | 39.7794 |
| 2 | 0.0437 | 27.8851 | 39.6909 |
| 3 | 0.0441 | 28.0944 | 39.7432 |
| 4 | 0.0434 | 28.0850 | 39.7408 |
| 5 | 0.0441 | 28.0852 | 39.7409 |
| 6 | 0.0437 | 28.2374 | 39.7789 |

所有组合下 Car moder strict 49.9772 / Truck moder strict 30.4303 / Cyclist loos 50.4709 逐位 0 差异。

### 35.5 判定

- **有界残差@0.25 在 6 epoch、只训 dim 分支的条件下基本无效**：Ped strict 全程
  0.043–0.044，未突破基线 0.045；loose 最终 28.24 也只是回到软先验级别的 28.2，
  未达到 28.5 理想目标，且低于未缩紧前的 ep7 原始 loose 28.53。
- `tanh_sat_ratio` 全程 ≈ 0.01–0.03，说明**不是 0.25 范围过窄被削峰**，而是 dim 分支
  在冻结 heatmap/center 之后只能产出极小的逐目标尺寸残差；ep7 的 dim.1.bias =
  [-0.0676, -0.0997, 0.2052]，加载后解码≈硬先验起点，训练 6 epoch 也几乎未离开该点。
- 与「纯推理软先验（零训练）Aα.75 strict 2.41」相比，**训练有界残差没有带来任何增益**，
  甚至 loose 因 heatmap 被冻结不动、尺寸又未拉开而略低于原始 ep7。因此：
  - 不再保留这个纯训练技巧；ep7 上 strict 的提升此前只能由**推理期尺寸先验**获得。
  - 按既定路线，下一轮进入 `0.16 m` 高分辨率 BEV 分支（让逐目标 w/l 回归本身更准，
    减少对先验的依赖），而不是继续在冻结 heatmap 上微调 bounded residual。

---

## 36. 低成本软先验 residual correction（delta_xy，3 epoch 门控，已完成，后续发现 IoU 输入编码错误）

### 36.1 实验设置

- 目的：修正第 35 节的残差起点错误。以已经通过正式标准的 Aα.75 为固定起点，
  再训练一个零初始化 correction 分支，测试 IoU 对齐目标能否把 Ped strict/loose
  同时推过理想阈值。
- 尺寸变换（decode 与 loss 共用）：
  ```ini
  prior = [0.655454, 0.627535]
  alpha = 0.75
  base_log_xy = 0.25 * old_log_xy.detach() + 0.75 * log(prior)
  final_log_xy = base_log_xy + 0.15 * tanh(delta_xy)
  ```
- `delta_xy` 独立分支：Conv3x3(64→64)+BN+ReLU → Conv3x3(64→2)，末层 weight/bias
  均零初始化，因此 epoch 0（加载 stage1 ep7 后、未训练）数学上精确等于 Aα.75。
- 冻结：R4Det 全部冻结，仅训练 `ped_center_head.delta_xy`；其余所有模块保持 eval。
- Loss：主损失为正样本 decoded Ped 3D rotated IoU loss（`1-IoU`，正样本数平均）；
  尺寸 log L1 仅作 0.1 辅助。保留原有 heatmap/其他回归项以维持完整训练日志口径。
- 运行：seed 0、deterministic、`CUDA_VISIBLE_DEVICES=5,6,7`、3 进程、
  `samples_per_gpu=2` + `cumulative_iters=2`（有效 batch 12）、lr=2e-4、max_epochs=3。
- 工作目录：`/data/lurui/work_dirs/ped_centerhead_stage2_delta_3x2x2_3e_seed0`

### 36.2 结果（Ped 3D moderate）

| epoch | Ped strict | Ped loose | Overall 3D moderate | 判定 |
|---:|---:|---:|---:|---|
| Aα.75 起点 | 2.4054 | 28.2016 | 39.77 | 参考 |
| 1 | 0.0485 | 28.5083 | 39.8467 | strict 大幅回退 |
| 2 | 0.0485 | 28.5087 | 39.8468 | strict 无提升 |
| 3 | 0.0485 | 28.5086 | 39.8468 | strict 无提升 |

Car strict 49.9772、Truck strict 30.4303、Cyclist loose 50.4709 与第 34/35 节
既有冻结口径一致，其他类未受扰动。

### 36.3 训练信号

- `task0.loss_ped_bev_iou` 约在 0.23–0.39 波动，训练 3 epoch 未见稳定下降。
- `task0.ped_size_log_mae` 从约 0.055 降到 0.03 左右，但只改善 log 尺寸误差，
  未转化为 strict IoU/AP。
- `grad_norm` 全程约 0.0002–0.0004，末层 bias 在 ep1 已偏移到约 ±0.025；
  尽管绝对值小，它足以把 strict 从 2.4054 拉回 0.0485。
- `tanh_sat_ratio` 始终接近 0（无饱和），说明不是 `0.15` 上限约束导致失败，
  而是 IoU 正样本监督在该冻结 heatmap/center 表征下学到的偏移方向与 strict
  判定所需尺寸不匹配。

### 36.4 判定与最终分支（已作废）

> **本节实验无效。** 2026-09-10 复查发现 `loss_ped_bev_iou` 把 CenterPoint
> 8 维编码向量 `[offset_x, offset_y, z, log_w, log_l, log_h, sin_yaw, cos_yaw]`
> 直接传给 `diff_iou_rotated_3d`。该算子要求 7 维物理框
> `[x, y, z, w, l, h, yaw]`，因此 offset 被当作绝对中心、log 尺寸被当作
> 物理尺寸、sin(yaw) 被当作 yaw，cos(yaw) 被忽略。IoU 数值、梯度方向以及
> “低分辨率路线结束”的结论均不可信。第 36 节 checkpoint 不作为任何路线依据。

- **原记录（作废）**：Ped strict 未达到 ≥3.0，且相对 Aα.75 起点出现完全退化（2.4054 → 0.0485）。
  loose 与 Overall 虽然达标（28.51 / 39.85），但按门控逻辑不能保留该分支。
- **低分辨率路线结束**：零初始化 residual correction 已修正到正确 Aα.75 起点，
  且换成 IoU 主损失后 3 epoch 完全无法提升 strict；结合第 35 节 6 epoch 结果，
  0.32 m BEV 上的 Ped strict 支线不再具备继续训练价值。
- 后续分支：
  - 若必须解决 Ped strict，才进入真实 0.16 m BEV。
  - 若目标是 Overall，应停止 Ped strict 支线，保留 clean full RSSM 主线。

---

## 37. 修复 encoded-box IoU bug 后重跑 delta_xy 门控（3 epoch，已完成）

### 37.1 修复与训练前验证

- 修复：`CenterHeadkitti._positive_boxes_for_iou` 在进入
  `diff_iou_rotated_3d` 前把 pred/target 从 CenterPoint 8 维编码解码为
  7 维物理 gravity-center 框：
  `[offset_x, offset_y, z, log_w, log_l, log_h, sin, cos]`
  → `[x, y, z, w, l, h, yaw]`。
- pred 的 `w/l` 来自完整 `final_log_xy` feature map 的正样本 gather，保证
  `delta_xy -> final_log_xy -> decoded box -> IoU` 梯度路径连续。
- 单测：`tests/test_centerhead_iou_decode.py`（1 passed）覆盖 7D shape、
  同框 IoU=1、扰动降低 IoU、`yaw=atan2(sin,cos)`、`final_log_xy` 梯度非零。
- epoch0 精确复现 Aα.75：
  Ped strict `2.4054`、Ped loose `28.2016`、Overall `39.7700`；
  Car strict `49.9772`、Truck strict `30.4303`、Cyclist loose `50.4709`。
- 训练：重新加载 stage1 `epoch_7.pth`，不复用第 36 节 checkpoint；
  seed 0、deterministic、GPU 5/6/7、3 进程、有效 batch 12、lr `2e-4`、
  `max_epochs=3`；工作目录
  `/data/lurui/work_dirs/ped_centerhead_stage2_delta_ioufix_3x2x2_3e_seed0`。

### 37.2 结果（Ped 3D moderate）

| epoch | Ped strict | Ped loose | Overall 3D moderate | 判定 |
|---:|---:|---:|---:|---|
| Aα.75 epoch0 | 2.4054 | 28.2016 | 39.7700 | 起点复现通过 |
| 1 | 0.2024 | 28.5481 | 39.8566 | strict 大幅退化 |
| 2 | 0.0485 | 28.5231 | 39.8504 | strict 继续退化 |
| 3 | 0.0485 | 28.5086 | 39.8467 | strict 无提升 |

Car strict `49.9772`、Truck strict `30.4303`、Cyclist loose `50.4709`
三个 epoch 均与冻结口径 0 差异。

### 37.3 训练信号

| epoch | mean `loss_ped_bev_iou` | mean `ped_size_log_mae` | mean `grad_norm` | mean `tanh_sat_ratio` |
|---:|---:|---:|---:|---:|
| 1 | 0.15643 | 0.04007 | 0.11268 | 0.02561 |
| 2 | 0.15301 | 0.03601 | 0.08225 | 0.00899 |
| 3 | 0.15288 | 0.03490 | 0.07295 | 0.00764 |

修复后梯度不再是第 36 节的 `1e-4` 量级死梯度，IoU loss 也有轻微下降
（0.15643 → 0.15288），但它只改善正样本训练 IoU，没有转化为 strict AP。
`tanh_sat_ratio` 全程低于 0.026，说明 `0.15` 残差上限不是失败原因。

### 37.4 判定

- **未通过**：Ped strict 远低于 ≥3.0；虽然 Ped loose（28.51）与 Overall
  （39.85）超过阈值，但 strict 退化到 0.05 级，门控不通过。
- **低分辨率路线真正关闭**：这次 IoU 输入已修复、epoch0 精确复现、
  梯度有效且 IoU loss 轻微下降，但仍无法改善 strict。说明失败不是 IoU 编码
  bug 或残差饱和造成，而是 0.32 m 表征/冻结正样本条件下无法恢复 strict
  所需的几何细节。
- 后续分支：
  - 若研究目标必须解决 Ped strict，进入真实 `0.16 m` Ped BEV。
  - 若目标只是 Overall，停止 Ped strict 支线，保留 clean full RSSM 主线。

## 38. Ped-only 高分辨率 BEV 门控（0.16 m，3 epoch，已完成）

### 38.1 实验设置

- 目的：第 37 节关闭低分辨率 residual 路线后，验证真实 `0.16 m` Ped-only BEV
  是否能恢复 Ped strict，同时保持 clean RSSM 主路径完全不变。
- 架构：
  - 复用已有 doubled radar voxel/scatter：`voxel_size=[0.16,0.16,6.0]`，
    `output_shape=[496,432]`；**不改全局 `voxel_size=[0.32,0.32]`**。
  - 新增 `PedHighresBranch`：high-res radar scatter -> Conv/BN/ReLU，
    与 bilinear 上采样后的冻结 fused BEV concat，再输出 Ped 专用 CenterHead 特征。
  - 两路输入均 detach；只训练 `highres_ped_branch + ped_center_head`。
  - RSSM、image/radar backbone、融合模块、原 Anchor3DHead、Car/Truck/Cyclist
    主检测路径全部冻结并保持 eval 状态。
  - 不加入 IoU/corner/yaw auxiliary loss、bounded residual、尺寸先验训练或
    RSSM 修改。
- checkpoint：重新加载 stage1 `epoch_7.pth`；highres branch 随机初始化，
  不 resume 第 37 节 checkpoint。
- 训练：seed 0、deterministic、GPU 5/6/7、3 进程、`samples_per_gpu=2`、
  `cumulative_iters=2`（有效 batch 12）、AdamW lr `2e-4`、`max_epochs=3`。
- 配置：
  - raw：`configs/r4det/TJ4D-R4Det_ped_highres_centerhead_3x2x2_3e_raw.py`
  - prior：`configs/r4det/TJ4D-R4Det_ped_highres_centerhead_3x2x2_3e_prior.py`
    （`size_prior_alpha=[0.655454,0.627535,0.75]`，只影响解码复评）
- 工作目录：`work_dirs/ped_highres_centerhead_3x2x2_3e_seed0_raw`
  （实际落盘于 repo 内；训练与 checkpoint 完整）。
- 单测与 smoke：
  - `tests/test_ped_highres_branch.py`
  - `tests/test_centerhead_iou_decode.py`
  - 合计 `3 passed`；raw/prior one-item GPU smoke 均通过。

### 38.2 Raw 解码结果（Ped 3D moderate）

| epoch | Ped strict | Ped loose | Overall 3D moderate | Car strict | Truck strict | Cyclist loose |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0000 | 0.0011 | 32.7199 | 49.9772 | 30.4303 | 50.4709 |
| 2 | 0.0000 | 0.0017 | 32.7200 | 49.9772 | 30.4303 | 50.4709 |
| 3 | 0.0000 | 0.0004 | 32.7197 | 49.9772 | 30.4303 | 50.4709 |

三个 epoch 中 Car/Truck/Cyclist 逐项完全一致，与冻结主路径预期一致，
隔离条件通过。

### 38.3 Size-prior 解码复评（同一 raw checkpoint）

| epoch | Ped strict | Ped loose | Overall 3D moderate | Ped 2D moderate |
|---:|---:|---:|---:|---:|
| 1 | 0.0000 | 0.0011 | 32.7199 | 0.0066 |
| 2 | 0.0000 | 0.0017 | 32.7200 | 0.0125 |
| 3 | 0.0000 | 0.0004 | 32.7197 | 0.0082 |

prior 与 raw 的 Ped strict/loose、Overall 基本完全一致；2D AP 也停在
0.006–0.013 量级。说明问题不是尺寸解码先验不足，而是 high-res Ped 分支
本身没有产生可用的 Ped proposal/定位信号。

### 38.4 训练信号

| epoch | mean total loss | mean heatmap loss | mean grad_norm |
|---:|---:|---:|---:|
| 1 | 2.9403 | 2.9357 | 9.0291 |
| 2 | 2.7053 | 2.6008 | 3.9943 |
| 3 | 2.6876 | 2.4831 | 4.2846 |

训练无 NaN、无梯度爆炸，loss 正常小幅下降；但 heatmap loss 仍在 2.5 左右，
且验证集 Ped 3D/BEV AP 没有形成有效召回。

### 38.5 门控判定：未通过，Ped 高分辨率支线关闭

| 指标 | 标准 | epoch 3 raw / prior | 判定 |
|---|---:|---:|:---:|
| Ped strict | ≥ 3.0 | 0.0000 / 0.0000 | ❌ |
| Ped loose | ≥ 28.5 | 0.0004 / 0.0004 | ❌ |
| Overall | ≥ 39.85 | 32.7197 / 32.7197 | ❌ |
| Car/Truck/Cyclist diff | ≤ 0.1 | 0.0000 | ✅ |

- **门控失败且不是 prior-only 假象**：raw 与 prior 都没有提升，说明高分辨率
  分支 3 epoch 内没有学出有效 Ped 检测，不是尺寸先验掩盖 raw 结果。
- **关闭 Ped high-res 支线**：按预设标准，3 epoch 后 Ped strict 仍远低于 2.4，
  不继续堆 Ped head、不再修改 RSSM，也不继续调高分辨率。
- **主结果保留 clean full RSSM**：当前 Ped strict 支线已完整尝试
  独立 CenterHead、固定/soft size prior、bounded residual、修复后的 IoU loss、
  以及真实 `0.16 m` Ped-only BEV；证据链一致指向该支线不能在门控预算内提升
  Ped strict。后续应回到 clean full RSSM 主线。

### 38.6 布局修复后的 identity 复评（第 38 节结论修正）

- 复查发现高分辨率 scatter 在 `extract_feat(feat_or_dict=1)` 中被无条件
  `permute`，把 `[B, C, H_y, W_x] = [1,64,432,496]` 变成了
  `[1,64,496,432]`。Ped CenterHead 的目标和推理解码随后又按转置后的
  `grid_size=[496,432]` 配置，导致 identity 阶段的 heatmap 位置整体转置，
  Ped 3D AP 被错误压到 0。该问题只影响新 Ped high-res 支路，不涉及 RSSM、
  KL、BPTT、velocity 或 clean 主检测路径。
- 修复：
  - `R4Det.extract_feat(feat_or_dict=1)` 保留 scatter 原生
    `[B, C, H_y, W_x]` 布局，不再 `permute`；
  - high-res `ped_center_head.train_cfg.grid_size` 改为
    `[bev_h_*2, bev_w_*2, 1] = [496,432,1]`，与 CenterPoint 内部
    `feature_map_size=[W,H]` 的契约一致；
  - 回归测试 `tests/test_ped_highres_branch.py` 覆盖该 grid 契约。
- 修复后 one-item 对齐：低分辨率 top peak `(132,30)` 对应高分辨率
  `(265,60)`，不再出现转置峰值。单测
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_ped_highres_branch.py -q`
  为 `4 passed`。
- identity 全量复评：继续加载 stage1 `epoch_7.pth`，关闭尺寸 prior，
  GPU 6、seed 0、deterministic。Ped 3D moderate：
  strict `0.0660`、loose `14.5976`；Overall 3D moderate `36.3690`。
  Car strict `49.9772`、Truck strict `30.4303`、Cyclist loose `50.4709`，
  与冻结主路径 0 差异。Ped 3D 召回不再为 0，identity 门控通过。
- **第 38.5 节结论作废**：前次关闭 Ped high-res 支线的判断建立在转置布局
  bug 之上，不能作为关闭依据。重新进入 high-res Ped 支线，先跑 6 epoch
  低成本训练门控。

### 38.7 布局修复后的 6-epoch 高分辨率门控（已完成）

#### 38.7.1 实验设置

- identity 布局修复后，先以 stage1 `epoch_7.pth` 做全量 identity 复评：
  Ped strict `0.0660`、Ped loose `14.5976`、Overall 3D moderate `36.3690`，
  确认高分辨率 Ped 分支有非零召回且 Car/Truck/Cyclist 与冻结主路径 0 差异。
- 正式门控从同一 stage1 `epoch_7.pth` 重新开始，只训练
  `highres_ped_branch + ped_center_head`；其余模块全部冻结并保持 eval。
- 配置：
  `configs/r4det/TJ4D-R4Det_ped_highres_centerhead_6e_identity_lr1e-4.py`。
  训练参数为 seed 0、deterministic、GPU 5/6/7、3 进程、
  `samples_per_gpu=2`、`cumulative_iters=2`（有效 batch 12）、
  AdamW lr `1e-4`、`max_epochs=6`。
- 工作目录：
  `/data/lurui/work_dirs/ped_highres_centerhead_identity_3x2x2_6e_lr1e-4_seed0`。
  `epoch_1.pth` 至 `epoch_6.pth` 和 `latest.pth -> epoch_6.pth` 均已落盘。

#### 38.7.2 Raw 解码 6-epoch 结果（Ped 3D moderate）

| epoch | Ped strict | Ped loose | Overall 3D moderate | Car strict | Truck strict | Cyclist loose |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0633 | 25.0168 | 38.9738 | 49.9772 | 30.4303 | 50.4709 |
| 2 | 0.0857 | 25.5354 | 39.1034 | 49.9772 | 30.4303 | 50.4709 |
| 3 | 0.1244 | 25.3222 | 39.0501 | 49.9772 | 30.4303 | 50.4709 |
| 4 | 0.0839 | 24.9236 | 38.9505 | 49.9772 | 30.4303 | 50.4709 |
| 5 | 0.1045 | 26.9477 | 39.4565 | 49.9772 | 30.4303 | 50.4709 |
| 6 | 0.0972 | 26.3535 | 39.3080 | 49.9772 | 30.4303 | 50.4709 |

- Car/Truck/Cyclist 所有 epoch 与冻结主路径逐项完全一致，0 差异，隔离条件
  通过。
- Ped strict 最高仅为 epoch 3 的 `0.1244`，随后回落；Ped loose 最高为
  epoch 5 的 `26.9477`，epoch 6 回落到 `26.3535`；Overall 最高同样为
  epoch 5 的 `39.4565`，epoch 6 回落到 `39.3080`。六个 epoch 没有形成
  连续、稳定的提升。

#### 38.7.3 Size-prior 解码复评（同一 `epoch_6.pth`）

- 复评继续使用 `epoch_6.pth`，仅把解码尺寸先验切换为
  `size_prior_alpha=[0.655454, 0.627535, 0.75]`，不训练任何模块。
- Ped strict `1.5071`、Ped loose `26.4851`、Overall 3D moderate `39.3409`。
- Car strict `49.9772`、Truck strict `30.4303`、Cyclist loose `50.4709`，
  与 raw 和冻结主路径的对应指标一致。
- prior 解码把 Ped strict 从 raw 的 `0.0972` 提升到 `1.5071`，但仍远低于
  `3.0`；Ped loose 和 Overall 也未达到 `28.5`、`39.85`。因此结果不是
  raw 被尺寸先验掩盖，尺寸先验本身也不足以通过门控。

#### 38.7.4 训练信号

| epoch | mean total loss | mean heatmap loss | mean grad_norm |
|---:|---:|---:|---:|
| 1 | 0.8436 | 0.8396 | 1.4778 |
| 2 | 0.9051 | 0.8011 | 1.3675 |
| 3 | 0.9946 | 0.7906 | 1.3730 |
| 4 | 1.0955 | 0.7915 | 1.3408 |
| 5 | 1.1833 | 0.7792 | 1.3729 |
| 6 | 1.2759 | 0.7717 | 1.3469 |

heatmap loss 从 `0.8396` 下降到 `0.7717`，grad_norm 全程稳定，没有 NaN 或
梯度爆炸；但验证集 AP 没有随 loss 单调改善。

#### 38.7.5 门控判定：未通过，关闭 Ped high-resolution 支线

| 指标 | 标准 | raw `epoch_6.pth` | prior `epoch_6.pth` | 判定 |
|---|---:|---:|---:|:---:|
| Ped strict | >= 3.0 | 0.0972 | 1.5071 | fail |
| Ped loose | >= 28.5 | 26.3535 | 26.4851 | fail |
| Overall | >= 39.85 | 39.3080 | 39.3409 | fail |
| Car/Truck/Cyclist diff | <= 0.1 | 0.0000 | 0.0000 | pass |

- **门控失败且不是 prior-only 假象**：raw 与 prior 都没有达到 strict/loose/
  Overall 标准；prior 只窄幅抬高 Ped strict，不能改变失败判定。
- **关闭 Ped high-resolution 支线**：布局错误已修复、identity 有正常召回、
  loss 有下降、冻结路径 6 epoch 保持 0 差异，但 Ped strict 仍只有
  `0.0972`（prior 复评 `1.5071`），远低于 `2.4` 和 `3.0` 门槛。
  按预设规则，不再继续堆 Ped CenterHead，也不继续修改高分辨率输入。
- **主结果保留 clean full RSSM**：关闭的是 Ped-only high-resolution 支线；
  clean full RSSM 主线继续保留，RSSM、KL、BPTT、velocity、free_nats 和
  原 Anchor3DHead 主检测路径均不再改动。

### 38.8 冻结网络 high-res 几何 probe（已完成）

#### 38.8.1 实验目的与设置

- 第 38.7 节关闭的是“全图 high-res CenterHead 重学 proposal”设计，不是
  高分辨率几何信息本身。因此本节改用冻结检测器做局部几何可解码性 probe，
  不训练完整检测器，也不替换 proposal、score 或 center。
- 检测器全部权重冻结并处于 eval；输入为冻结 detector 当前帧产生的
  0.16 m radar scatter 与 `highres_ped_branch` 输出。RSSM 按 `simple_test`
  路径 reset + history burn-in，再取当前帧特征。
- 对每个当前帧 GT Pedestrian 中心截取 `9x9` 高分辨率局部特征，
  覆盖约 `1.44 m x 1.44 m`。输入通道为 high-res branch 256 通道与
  high-res radar scatter 64 通道。
- 只训练小几何头：`2x Conv2d + BN + ReLU + global average pool + Linear(4)`，
  输出 `[log_w, log_l, sin_yaw, cos_yaw]`。
- 参数：seed 0、GPU 5、crop size 9、AdamW lr `1e-3`、batch size 64、
  epochs 3。样本来自所有含 Pedestrian GT 的 train/val 帧，train 共
  `3235` 个 Ped crop，val 共 `993` 个 Ped crop。
- 工作目录：
  `/data/lurui/work_dirs/ped_highres_geometry_probe_crop9_3e_seed0`。
  结果文件：`probe_result.json`；缓存：`train_crop9.pt`、`val_crop9.pt`。
- 工具：
  `tools/ped_highres_geometry_probe.py`；CPU 回归：
  `tests/test_ped_highres_geometry_probe.py`。

#### 38.8.2 Probe 结果

低分辨率 CenterHead 对照基线为：
`w MAE = 0.283 m`、`w correlation = -0.283`、`yaw MAE = 0.504 rad`。

通过条件：`w MAE <= 0.24 m`、`w correlation >= 0.20`、
`yaw MAE <= 0.45 rad`。

| epoch | train loss | w MAE | w corr | l MAE | yaw MAE |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.1569 | 0.1332 | -0.0968 | 0.1606 | 0.9574 |
| 2 | 0.1195 | 0.1569 | -0.2812 | 0.1555 | 1.0676 |
| 3 | 0.1124 | 0.1972 | -0.3635 | 0.1731 | 1.0257 |

#### 38.8.3 判定：probe 未通过，Ped 支线关闭

| 指标 | 标准 | epoch 3 | 判定 |
|---|---:|---:|---:|
| w MAE | <= 0.24 m | 0.1972 | pass |
| w correlation | >= 0.20 | -0.3635 | fail |
| yaw MAE | <= 0.45 rad | 1.0257 | fail |

- train loss 从 `0.1569` 降到 `0.1124`，说明小几何头确实在拟合局部 crop；
  但 val `w correlation` 从 `-0.0968` 变成 `-0.3635`，yaw MAE 始终在
  `0.96-1.07 rad`，没有接近可用几何解码水平。
- `w MAE` 虽然表面低于 `0.24 m`，但 correlation 为负且随训练变差，说明
  该 MAE 主要来自向均值/先验收缩，不是逐样本尺寸预测能力；不能用它判定
  通过。
- 因此不进入后续局部 box refinement，不再训练
  `delta_log_w / delta_log_l / delta_yaw`，也不基于该高分辨率局部特征
  继续做 Ped head。
- **保留路线**：最终保留 clean full RSSM 主线；若论文需要 Ped strict，
  使用 stage1 `epoch_7.pth` + `A alpha=0.75`。不再修改 RSSM、KL、BPTT、
  velocity、free_nats 或 clean 主检测路径。

### 38.9 修正 yaw 周期后的 yaw-only probe（已完成）

#### 38.9.1 修正原因

- 第 38.8 节证明 high-res 局部特征无法预测逐目标 `w/l`：
  train/val 尺寸分布有 split shift（train `w mean=0.666 m`、
  `l mean=0.656 m`；val `w mean=0.580 m`、`l mean=0.604 m`），
  且 val 常数中位数预测的 `w MAE=0.105 m` 已优于 probe `0.133 m`。
  因此尺寸分支停止，probe 只保留 yaw。
- 原 probe 把 yaw 当 `2π` 周期，用 `[sin(yaw), cos(yaw)]`；但 3D box
  中 `yaw` 与 `yaw + π` 几何等价。先前 `yaw MAE≈1.0 rad` 包含错误的反向
  惩罚，不能作为关闭依据。
- 修正后目标为 `[sin(2*yaw), cos(2*yaw)]`，预测
  `pred_yaw = 0.5 * atan2(pred_sin2, pred_cos2)`，误差按 modulo-π 计算。

#### 38.9.2 实验设置

- 复用第 38.8 节已缓存的 detector 特征，不重新跑 detector：
  `/data/lurui/work_dirs/ped_highres_geometry_probe_crop9_3e_seed0`
  中的 `train_crop9.pt` 和 `val_crop9.pt`。
- crop 9x9，只预测 `sin(2yaw), cos(2yaw)`，不预测 `w/l`。
  训练参数：seed 0、GPU 5、AdamW lr `1e-3`、batch 64、epochs 5。
  loss 为 `1 - cosine_similarity`。
- 工作目录：
  `/data/lurui/work_dirs/ped_highres_yaw_probe_crop9_5e_seed0`。
  结果文件：`probe_result.json`。

#### 38.9.3 结果

通过条件：`val yaw MAE <= 0.45 rad`，连续两个 epoch 不恶化，且明显优于
低分辨率基线 `0.504 rad`。

| epoch | train loss | val yaw MAE |
|---:|---:|---:|
| 1 | 0.5202 | 0.5481 |
| 2 | 0.4606 | 0.5514 |
| 3 | 0.4578 | 0.4858 |
| 4 | 0.4415 | 0.5372 |
| 5 | 0.4309 | 0.5610 |

#### 38.9.4 判定：yaw-only probe 未通过，彻底停止 Ped 架构实验

| 指标 | 标准 | best epoch 3 | final epoch 5 | 判定 |
|---|---:|---:|---:|---:|
| val yaw MAE | <= 0.45 rad | 0.4858 | 0.5610 | fail |
| 连续两个 epoch 不恶化 | required | - | epoch4/5 退化为 0.5372/0.5610 | fail |
| 优于低分辨率基线 0.504 rad | required | 0.4858 仅小幅优于 | 0.5610 更差 | fail |

- train loss 从 `0.5202` 降到 `0.4309`，但 val yaw 最好只有 `0.4858 rad`，
  且 epoch 4-5 连续恶化。修正 `π` 周期后仍然不能稳定预测 yaw。
- 不做 yaw-only local refiner，不训练 `delta_yaw`，不续训 high-res
  CenterHead，也不再训练尺寸 residual。
- **最终路线**：彻底停止 Ped 架构实验。后续只做最终方案多 seed 验证：
  clean full RSSM 作为主结果；stage1 `epoch_7.pth` + `A alpha=0.75`
  作为 Ped strict 优先工作点；seed 0/1/2，prior 参数固定，不再调 val。

### 38.10 修复 high-res residual 死分支后的 3-epoch 门控（已完成）

#### 38.10.1 死分支定位与修复

- 第 38.9 节末尾“彻底停止 Ped 架构实验”的结论当时仍建立在第 38.7 节
  high-res CenterHead 训练失败上。复查发现第 38.7 节实际没有训练到
  high-res radar 分支：`PedHighresBranch.fusion_conv` 原结构为
  `Conv + BN + ReLU`，而最终 Conv 的 weight/bias 被零初始化；
  ReLU 在零点梯度为零，导致残差分支从第一步起就没有非零梯度。
- 检查第 38.7 节 checkpoint 确认
  `highres_ped_branch.fusion_conv.conv.weight` 在 epoch 1-6 始终全零，
  非零参数数量始终为 `0`。因此第 38.7 节训练的是
  “nearest-upsample low-resolution BEV + CenterHead”，0.16 m radar
  特征从未进入输出，不能证明训练后的 high-res branch 无效。
- 修复 `mmdet3d/models/fusion_layers/ped_highres_branch.py`：
  `fusion_conv` 改为单个有 bias 的 `nn.Conv2d`，不再包含 BN/ReLU，
  weight/bias 仍零初始化，使 epoch0 严格保持
  `highres_ped_feature == nearest_upsample(lowres_fused_feature)`，
  同时允许有符号残差和非零梯度。
- 更新 `tests/test_ped_highres_branch.py`，增加三项回归证明：
  identity 初始化、首次 backward 后 `fusion_conv.weight.grad.abs().sum() > 0`、
  一次 optimizer step 后第二次 backward 的 `radar_conv` 梯度非零。
  聚焦测试通过：`5 passed`。
- 相关提交：`91f3b4c fix(ped): unblock highres residual gradients`；
  `d05ee9b config(ped): add highres grad-fix 3e gate`。

#### 38.10.2 实验设置

- 配置：
  `configs/r4det/TJ4D-R4Det_ped_highres_centerhead_gradfix_3x2x2_3e_seed0.py`。
- 从头加载 stage1 `epoch_7.pth`，不 resume 第 38.7 节错误 checkpoint；
  只训练 `highres_ped_branch + ped_center_head`，其余模块全部冻结并保持
  eval。seed 0、deterministic、GPU 5/6/7、3 进程、
  `samples_per_gpu=2`、`cumulative_iters=2`（有效 batch 12）、
  AdamW lr `1e-4`、`max_epochs=3`。
- 工作目录：
  `/data/lurui/work_dirs/ped_highres_centerhead_gradfix_3x2x2_3e_seed0`。
  日志：`20260912_142326.log`、`20260912_142326.log.json`。
- 每个 epoch 同时评估 raw 解码和固定 prior 解码
  `size_prior_alpha=[0.655454,0.627535,0.75]`。

#### 38.10.3 残差权重非零证明

| checkpoint | fusion weight nonzero | fusion weight L1 | fusion weight absmax | fusion bias nonzero |
|---|---:|---:|---:|---:|
| epoch 1 | 737280 / 737280 | 1206.9482 | 0.012934 | 256 / 256 |
| epoch 2 | 737280 / 737280 | 1663.6880 | 0.021517 | 256 / 256 |
| epoch 3 | 737280 / 737280 | 1721.5181 | 0.022393 | 256 / 256 |

`fusion_conv` weight 不再为零，且 L1 随训练增加；radar 分支的梯度阻断问题
已解除。

#### 38.10.4 训练信号

| epoch | mean total loss | mean heatmap loss | mean grad_norm |
|---:|---:|---:|---:|
| 1 | 0.9108 | 0.9066 | 5.7817 |
| 2 | 0.9403 | 0.8363 | 3.4836 |
| 3 | 0.9982 | 0.7941 | 2.9320 |

heatmap loss 从 `0.9066` 降到 `0.7941`，grad_norm 全程有限且逐步稳定；
训练通路本身正常。

#### 38.10.5 Raw 解码结果（Ped 3D moderate）

| epoch | Ped strict | Ped loose | Overall 3D moderate | Car strict | Truck strict | Cyclist loose |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.1401 | 25.4998 | 39.0945 | 49.9772 | 30.4303 | 50.4709 |
| 2 | 0.0257 | 26.2358 | 39.2785 | 49.9772 | 30.4303 | 50.4709 |
| 3 | 0.0239 | 27.6425 | 39.6302 | 49.9772 | 30.4303 | 50.4709 |

Car/Truck/Cyclist 与冻结主路径逐项完全一致，0 差异，隔离条件通过。
raw Ped strict 三轮都远低于门槛，说明高分辨率分支没能直接学出可用
Ped 3D proposal。

#### 38.10.6 固定 prior 解码结果（Ped 3D moderate）

| epoch | Ped strict | Ped loose | Overall 3D moderate | Car strict | Truck strict | Cyclist loose |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3.6060 | 25.0677 | 38.9865 | 49.9772 | 30.4303 | 50.4709 |
| 2 | 2.5677 | 27.4581 | 39.5841 | 49.9772 | 30.4303 | 50.4709 |
| 3 | 1.5745 | 27.4867 | 39.5913 | 49.9772 | 30.4303 | 50.4709 |

prior 复评把 Ped strict 拉到 epoch1 `3.6060`、epoch2 `2.5677`，但
Ped loose 三轮分别为 `25.0677`、`27.4581`、`27.4867`，均未达到续训门槛
`>= 27.5`。epoch3 Ped strict 又回落到 `1.5745`，没有形成稳定几何收益。

#### 38.10.7 3-epoch 续训门槛判定

预设续 6 epoch 条件：

| 条件 | epoch 1 | epoch 2 | epoch 3 | 判定 |
|---|---:|---:|---:|---:|
| `fusion_conv` 权重非零 | pass | pass | pass | pass |
| prior Ped strict >= 2.0 | 3.6060 | 2.5677 | 1.5745 | 仅 epoch1/2 |
| Ped loose >= 27.5 | 25.0677 | 27.4581 | 27.4867 | 全部 fail |
| 其他三类 0 差异 | pass | pass | pass | pass |

- epoch2 最接近续训门槛，但 Ped loose 仍差 `0.0419`；epoch3 Ped loose 只差
  `0.0133`，但 prior Ped strict 回落到 `1.5745`。没有任何一个 epoch 同时
  满足续训条件，不进入 6 epoch。
- 最终门槛同样未通过：Ped strict `>= 3.0`、Ped loose `>= 28.5`、
  Overall `>= 39.85` 均未同时达到。
- 修正残差死分支后，high-res radar 特征确实进入输出且训练稳定，但没有在
  3-epoch 门控预算内带来稳定 Ped strict/loose 提升。按预设规则，Ped
  high-res 支线不再续训，也不进入最终多 seed；后续保留 clean full RSSM
  主线，论文若需要 Ped strict 则使用 stage1 `epoch_7.pth` +
  `A alpha=0.75`。
### 38.11 修复残差后的 6-epoch 收敛验证（已完成）

#### 38.11.1 实验设置

- 第 38.10 节结论被判定为“略微错过续训门槛”，而非“接近最终成功”：
  epoch2 prior Ped strict `2.5677` 已过续训线，Ped loose `27.4581` 仅差
  `0.0419`；raw loose 连续上升 `25.50 -> 26.24 -> 27.64`，heatmap loss
  连续下降 `0.9066 -> 0.7941`，说明 proposal/recall 仍在学习，3 epoch
  关闭路线过早。
- 启动最终 6-epoch 收敛验证，保持单变量：不改结构、loss、prior 或 RSSM，
  只延长正确 high-res 分支的训练周期。必须从 stage1 `epoch_7.pth` 重新
  训练，不 resume 第 38.10 节 epoch3 checkpoint（其 CosineAnnealing 已接近
  最低学习率，直接续训会改变实验含义）。
- 配置：
  `configs/r4det/TJ4D-R4Det_ped_highres_centerhead_6e_identity_lr1e-4.py`，
  `load_from = ped_centerhead_stage1_3x2x2_12e_seed0/epoch_7.pth`，
  `resume_from = None`，`max_epochs = 6`。
- 固定参数：只有 `highres_ped_branch + ped_center_head` 可训练，其余模块
  全部冻结并保持 eval；seed 0、deterministic、GPU 5/6/7、3 进程、
  `samples_per_gpu=2`、`cumulative_iters=2`（有效 batch 12）、AdamW
  lr `1e-4`；prior `[0.655454, 0.627535, 0.75]`。
- 工作目录：
  `/data/lurui/work_dirs/ped_highres_centerhead_gradfix_3x2x2_6e_seed0`。
  日志：`20260912_170125.log`、`20260912_170125.log.json`。
  每个 epoch 同时评估 raw 解码和固定 prior 解码。

#### 38.11.2 残差权重非零证明

| checkpoint | fusion weight nonzero | fusion weight L1 | fusion weight absmax | fusion bias nonzero |
|---|---:|---:|---:|---:|
| epoch 1 | 737280 / 737280 | 1206.9482 | 0.01293 | 256 / 256 |
| epoch 2 | 737280 / 737280 | 1828.2423 | 0.02386 | 256 / 256 |
| epoch 3 | 737280 / 737280 | 2205.3813 | 0.02894 | 256 / 256 |
| epoch 4 | 737280 / 737280 | 2360.5073 | 0.03302 | 256 / 256 |
| epoch 5 | 737280 / 737280 | 2403.9517 | 0.03434 | 256 / 256 |
| epoch 6 | 737280 / 737280 | 2408.8564 | 0.03457 | 256 / 256 |

`fusion_conv` weight 全轮非零，L1 从 `1206.9` 增长到 `2408.9` 后趋于饱和，
确认 high-res radar 残差始终进入输出，梯度阻断问题在 6-epoch 尺度上同样
已解除。

#### 38.11.3 训练信号

| epoch | mean total loss | mean heatmap loss | mean grad_norm |
|---:|---:|---:|---:|
| 1 | 0.9108 | 0.9066 | 5.7817 |
| 2 | 0.9416 | 0.8375 | 3.3545 |
| 3 | 1.0173 | 0.8131 | 2.8572 |
| 4 | 1.0995 | 0.7953 | 2.7036 |
| 5 | 1.1779 | 0.7737 | 2.7992 |
| 6 | 1.2607 | 0.7565 | 2.7217 |

heatmap loss 单调下降 `0.9066 -> 0.7565`，grad_norm 有限且稳定。
mean total loss 从 `0.9108` 升到 `1.2607`，增量约 `+0.10/epoch`，与冻结
RSSM 的 `loss_rssm_kl` 完全对齐：`loss_rssm_kl` 按 `KLScaleSchedulerHook`
从 epoch1 `0.0` 线性升到 epoch6 `0.5003`（`loss_rssm_recon` 稳定在
`0.0028-0.0029`）。因此 total loss 上升来自 RSSM KL 项调度，不是 Ped
回归 loss 异常；训练信号本身正常，失败是验证指标真实不成立。

#### 38.11.4 Raw 解码结果（Ped 3D moderate）

| epoch | Ped strict | Ped loose | Overall 3D moderate | Car strict | Truck strict | Cyclist loose |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.1401 | 25.4998 | 39.0945 | 49.9772 | 30.4303 | 50.4709 |
| 2 | 0.0343 | 26.0599 | 39.2346 | 49.9772 | 30.4303 | 50.4709 |
| 3 | 0.0252 | 23.7019 | 38.6451 | 49.9772 | 30.4303 | 50.4709 |
| 4 | 0.0212 | 25.9307 | 39.2023 | 49.9772 | 30.4303 | 50.4709 |
| 5 | 0.0347 | 28.1536 | 39.7580 | 49.9772 | 30.4303 | 50.4709 |
| 6 | 0.0280 | 28.2763 | 39.7887 | 49.9772 | 30.4303 | 50.4709 |

raw Ped loose/Overall 后段持续上升（epoch5 `28.1536`/`39.7580`、
epoch6 `28.2763`/`39.7887`），但 raw Ped strict 全程在 `0.02-0.14` 噪声带，
说明高分辨率分支始终没学出可用的 Ped 3D strict proposal。

#### 38.11.5 固定 prior 解码结果（Ped 3D moderate）

| epoch | Ped strict | Ped loose | Overall 3D moderate | Car strict | Truck strict | Cyclist loose |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 3.6060 | 25.0677 | 38.9865 | 49.9772 | 30.4303 | 50.4709 |
| 2 | 2.5716 | 27.6956 | 39.6435 | 49.9772 | 30.4303 | 50.4709 |
| 3 | 2.2570 | 27.0311 | 39.4774 | 49.9772 | 30.4303 | 50.4709 |
| 4 | 1.7359 | 26.8478 | 39.4315 | 49.9772 | 30.4303 | 50.4709 |
| 5 | 2.0015 | 26.9661 | 39.4611 | 49.9772 | 30.4303 | 50.4709 |
| 6 | 1.7118 | 27.3861 | 39.5661 | 49.9772 | 30.4303 | 50.4709 |

prior 复评把 Ped strict 拉到 epoch1 `3.6060`、epoch2 `2.5716`，但没有任何
一个 epoch 同时达到 Ped loose `>= 28.2016` 和 Overall `>= 39.7700`：
最优 Ped loose 是 epoch2 `27.6956`（低于门槛 `0.5060`），最优 Overall 是
epoch2 `39.6435`（低于门槛 `0.1265`）。Car/Truck/Cyclist 六个 epoch 逐项
完全一致（Car strict `49.9772`、Truck strict `30.4303`、Cyclist loose
`50.4709`），0 差异，冻结隔离条件通过。

#### 38.11.6 6-epoch 最终判定

最低要求（同一 checkpoint 同时满足）：

| 条件 | stage1 ep7 + Aα.75 | 6e best | 判定 |
|---|---:|---:|---:|
| Ped strict > 2.4054 | 2.4054 | epoch1 3.6060 | 单轮可过，但不稳定 |
| Ped loose >= 28.2016 | 28.2016 | epoch2 27.6956 | fail |
| Overall >= 39.7700 | 39.7700 | epoch2 39.6435 | fail |
| 其他三类 0 差异 | - | 全部 0 差异 | pass |

理想要求（`Ped strict >= 3.0`、`Ped loose >= 28.5`、`Overall >= 39.85`）
同样未在同一 checkpoint 上达成。不能用不同 epoch 分别挑 strict 与 loose
最优，按“同一 checkpoint 整体达标”口径，6-epoch 收敛验证失败。

**结论：正式关闭 high-res / Ped 架构路线。** 修正残差死分支后，high-res
radar 特征确实进入输出、训练稳定、heatmap loss 单调下降，raw loose/Overall
后段也在上涨，但没有任何一个 checkpoint 同时超过现有可用方案
stage1 `epoch_7.pth` + `A alpha=0.75`。Ped strict 与 loose 呈此消彼长，
且 raw strict 始终无法离开噪声带，说明 0.16 m high-res CenterHead 重学
proposal 的设计在这个预算下没有形成可用几何收益。

后续只做最终方案多 seed 复现，不再改 RSSM/KL/BPTT：
- clean full RSSM 作为主结果；
- stage1 `epoch_7.pth` + `A alpha=0.75` 作为 Ped strict 优先工作点；
- seed 0/1/2，prior 参数固定，不再调 val。

---

## 39. 最终 Ped supplementary 多 seed（clean full RSSM 对照 + stage1 ep7 + A alpha=0.75）

### 39.0 协议

- high-res / Ped 架构路线已在第 38.11 节正式关闭；本节只做最终方案多 seed 复现，
  不改 RSSM / KL / BPTT，不再训练 high-res，不再 sweep alpha。
- clean full RSSM 三 seed 已完成（第 19 节：Overall `40.42 +/- 0.51`），本节不重训 clean。
  为保证公平配对，本节重新用同一 `tools/test_vod.py --out` 管线复评三个 clean source checkpoint，
  并将对应的 clean full RSSM checkpoint 作为唯一基准；stage1 raw 只作为诊断行，不再作为对照。
- 固定配置：12 epoch、`lr=1e-3`、`samples_per_gpu=2`、`cumulative_iters=2`
  （有效 batch 12）、`--deterministic`、GPU 5/6/7。
- 固定选点：`epoch_7.pth`，不按 val 重新挑 epoch。
- 固定推理先验：`size_prior_alpha=[0.655454,0.627535,0.75]`（第 33 节训练集中位数先验；
  alpha=0.75）。
- seed1 的 clean checkpoint 使用原始 `/data/lurui/work_dirs/run10_headv2_multiseed/seed_1/epoch_14.pth`。
  文档曾写 seed1 best 为 `epoch_15.pth`，但文件实际不存在；补跑的 restored ep15 Overall 只有
  `40.2945`，未复现原记录的 `40.507`。因此 seed1 stage1 明确固定使用已存在的 `epoch_14.pth`，
  不使用 restored ep15。
- seed0 的 clean checkpoint 使用 `/data/lurui/work_dirs/run10_headv2_multiseed/seed_0/epoch_16.pth`。
- seed2 的 clean checkpoint 使用 `/data/lurui/work_dirs/run10_headv2_multiseed/seed_2/epoch_14.pth`。
- clean 对照同管线复评 checkpoint：
  - seed0：`epoch_16.pth`
  - seed1：`epoch_14.pth`
  - seed2：`epoch_14.pth`
- 配置：
  - `configs/r4det/TJ4D-R4Det_ped_centerhead_stage1_3x2x2_12e_seed1.py`
  - `configs/r4det/TJ4D-R4Det_ped_centerhead_stage1_3x2x2_12e_seed2.py`
- 相关提交：`524234c config(ped): add seed1/seed2 stage1 supplementary configs`；
  `3bb0599 config(ped): use seed1 epoch14 for stage1`。

### 39.1 评估口径

- clean / raw / prior 均使用同一 `tools/test_vod.py --out` 管线复评，避免训练日志验证与独立复评
  管线混用造成非 Ped 类别绝对值口径差异。
- 公平基线是每个 seed 对应的 clean full RSSM checkpoint；stage1 raw 与 clean checkpoint 使用相同
  seed，仅用于确认 stage1 训练本身的 Ped 输出变化，不参与最终提升判定。
- 报告 Ped 3D moderate strict / loose、Overall 3D moderate，以及 Car strict、
  Truck strict、Cyclist loose 三项隔离指标。

### 39.2 Seed 0（clean seed0 ep16 -> stage1 fixed ep7，同管线复评）

工作目录：`/data/lurui/work_dirs/ped_centerhead_stage1_3x2x2_12e_seed0`。
配置 `load_from=/data/lurui/work_dirs/run10_headv2_multiseed/seed_0/epoch_16.pth`。

同管线 `tools/test_vod.py --out` 复评：

| 口径 | Ped strict | Ped loose | Overall 3D moderate | Car strict | Truck strict | Cyclist loose |
|---|---:|---:|---:|---:|---:|---:|
| clean ep16 | 0.1036 | 28.6426 | 39.8802 | 49.9772 | 30.4303 | 50.4709 |
| stage1 ep7 raw | 0.0452 | 28.5298 | 39.8520 | 49.9772 | 30.4303 | 50.4709 |
| stage1 ep7 + prior | 2.4054 | 28.2016 | 39.7700 | 49.9772 | 30.4303 | 50.4709 |

- same-pipeline 复评精确复现第 34 节的 seed0 Aα.75 fixed ep7 数字：
  strict `2.4054`、loose `28.2016`、Overall `39.7700`。
- 相对 clean ep16，prior strict `+2.3018`，Ped loose `-0.4410`，Overall `-0.1102`。
- stage1 raw 相对 clean 的 Ped 指标也偏低，说明 raw 不适合作为公平对照。
- 非 Ped 三类与 clean / raw 逐项完全一致，0 差异。

### 39.3 Seed 1（clean seed1 ep14 -> stage1 fixed ep7）

工作目录：`/data/lurui/work_dirs/ped_centerhead_stage1_3x2x2_12e_seed1`。
日志：`20260913_072011.log(.json)`。训练在 epoch10 保存后中断，但协议固定选点
`epoch_7.pth` 已完整保存，因此 seed1 fixed ep7 可用于本节配对。

同管线 `tools/test_vod.py` 复评：

| 口径 | Ped strict | Ped loose | Overall 3D moderate | Car strict | Truck strict | Cyclist loose |
|---|---:|---:|---:|---:|---:|---:|
| clean ep14 | 0.1272 | 27.7616 | 38.0524 | 43.5477 | 33.6401 | 47.2604 |
| stage1 ep7 raw | 0.0713 | 26.6636 | 37.7779 | 43.5477 | 33.6401 | 47.2604 |
| stage1 ep7 + prior | 3.5723 | 28.1934 | 38.1604 | 43.5477 | 33.6401 | 47.2604 |

- 相对 clean ep14，prior strict `+3.4451`，Ped loose `+0.4318`，Overall `+0.1080`。
- 非 Ped 三类与 clean / raw 逐项完全一致，0 差异。
- seed1 的 Ped/Overall 绝对值偏低主要来自该 seed 的 clean checkpoint / stage1 训练结果；
  Car/Truck/Cyclist 绝对值也与 seed0 不同，因此最终判定按每个 seed 内 clean vs prior 配对差值和
  三 seed prior 均值统计，不跨 seed 混用绝对值。

### 39.4 Seed 2（clean seed2 ep14 -> stage1 fixed ep7）

工作目录：`/data/lurui/work_dirs/ped_centerhead_stage1_3x2x2_12e_seed2`。
配置 `load_from=/data/lurui/work_dirs/run10_headv2_multiseed/seed_2/epoch_14.pth`。
日志：`20260913_134924.log(.json)`。训练已完整保存 12 个 epoch；本节固定使用
`epoch_7.pth`，不按 val 重新选点。

同管线 `tools/test_vod.py --out` 复评：

| 口径 | Ped strict | Ped loose | Overall 3D moderate | Car strict | Truck strict | Cyclist loose |
|---|---:|---:|---:|---:|---:|---:|
| clean ep14 | 0.1156 | 31.3235 | 40.8800 | 51.3463 | 30.7597 | 50.0905 |
| stage1 ep7 raw | 0.0281 | 24.3557 | 39.1380 | 51.3463 | 30.7597 | 50.0905 |
| stage1 ep7 + prior | 1.1133 | 25.6637 | 39.4651 | 51.3463 | 30.7597 | 50.0905 |

- 相对 clean ep14，prior strict `+0.9977`，Ped loose `-5.6598`，Overall `-1.4149`。
- seed2 在 Ped loose 和 Overall 上出现显著退化，说明 prior 不是稳定的综合改进。
- 非 Ped 三类与 clean / raw 逐项完全一致，0 差异。

### 39.5 三 seed 统计与最终判定

三 seed fixed ep7 相对对应 clean checkpoint 的同管线汇总（prior - clean）：

| seed | clean strict | prior strict | Δstrict | clean loose | prior loose | Δloose | clean Overall | prior Overall | ΔOverall |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.1036 | 2.4054 | +2.3018 | 28.6426 | 28.2016 | -0.4410 | 39.8802 | 39.7700 | -0.1102 |
| 1 | 0.1272 | 3.5723 | +3.4451 | 27.7616 | 28.1934 | +0.4318 | 38.0524 | 38.1604 | +0.1080 |
| 2 | 0.1156 | 1.1133 | +0.9977 | 31.3235 | 25.6637 | -5.6598 | 40.8800 | 39.4651 | -1.4149 |
| mean | 0.1155 | 2.3637 | +2.2482 | 29.2426 | 27.3529 | -1.8897 | 39.6042 | 39.1318 | -0.4724 |

判定：

- Ped strict 在三个 seed 上均稳定提升，mean `+2.2482`。
- Ped loose mean `-1.8897`，Overall mean `-0.4724`；seed2 退化严重
  （Ped loose `-5.6598`、Overall `-1.4149`）。
- Car / Truck / Cyclist 在 clean / raw / prior 间逐项完全一致，0 差异。
- 因此 stage1 fixed `epoch_7.pth` + `A alpha=0.75` 只能称为“Ped strict 优先的可选推理工作点”，
  不是稳定的综合改进，也不再使用“supplementary 综合通过”的表述。

最终报告只保留两个工作点：

```text
标准模型：
clean full RSSM
Overall = 40.42 +/- 0.51

Ped strict 优先变体：
stage1 fixed ep7 + A alpha=0.75
Ped strict mean = 2.3637
相对对应 clean：
Ped strict +2.2482
Ped loose -1.8897
Overall -0.4724
```

**结论：high-res / Ped 训练路线彻底失败，正式关闭。** soft prior 稳定提升 Ped strict，
但明显牺牲 Ped loose 和 Overall；seed2 退化严重，它不是稳定的综合改进。clean full RSSM
仍是唯一主模型，结果成功（`40.42 +/- 0.51`）。stage1 fixed `epoch_7.pth` + `A alpha=0.75`
仅在论文消融或应用明确重视 Ped strict 时报告。test set / 最终提交只优先跑 clean RSSM；
不再训练任何新结构，不再调整 alpha。

---

## 40. 恢复预训练 Cross-Modal Fusion 的 seed0 门控（未通过，已停止）

### 40.1 实验动机与唯一配置改动

当前主配置此前使用随机初始化的 `ConcatConvFusion`，但 `checkpoints/pretrained_tj4d.pth`
是在 `TJ4D-R4Det_pretrain_N4_2x4_12e.py` 下训练的，融合模块为
`Cross_Modal_Fusion`。预训练 checkpoint 中包含完整的
`cross_attention.att_img`、`cross_attention.att_radar` 和
`cross_attention.reduce_mixBEV` 权重，而旧的 concat fusion 配置不会加载这些权重。
该融合位于 RSSM 前方，会影响全部四类，因此先恢复它比继续修改 Ped/Truck head 更贴近
“提升整个网络”的目标，且改动最小。

唯一配置改动：
`configs/r4det/TJ4D-R4Det_motion_align_rssm_det3d_N4_2x4_24e_pretrained_v2_head.py`
中的 `RCFusion.type: ConcatConvFusion -> Cross_Modal_Fusion`。其余固定项未变：
N4、`hidden_dim=128`、BPTT1、`kl_scale=1.0`、`free_nats=1.0`、
`samples_per_gpu=2`、`cumulative_iters=2`（有效 batch 12）、AdamW
`lr=1.5e-4`、24 epoch、`checkpoint_interval=1`、seed0、deterministic、
GPU 5/6/7、`load_from=checkpoints/pretrained_tj4d.pth`。

相关提交：`2cd85fc config: restore pretrained cross-modal fusion`。

### 40.2 预训练融合权重加载证明

启动日志：
`/data/lurui/work_dirs/crossmodal_fusion_N4_2x4_24e_seed0/20260914_151155.log`。

- `load checkpoint from local path: checkpoints/pretrained_tj4d.pth`
- 模型结构日志包含 `(cross_attention): Cross_Modal_Fusion(...)`
- `missing keys in source state_dict` 中不包含任何 `cross_attention.*`；
  missing keys 只包含未变化的 `pts_bbox_head.conv_iou.*` 和
  `temporal_fusion.*` 等本实验不会从预训练模型加载的新模块。

因此本次 seed0 确实使用了预训练融合权重，不是从随机初始化的
`ConcatConvFusion` 重训。

### 40.3 训练设置与停止口径

命令：

```bash
source .envrc
CUDA_VISIBLE_DEVICES=5,6,7 SEED=0 bash tools/dist_train.sh \
  configs/r4det/TJ4D-R4Det_motion_align_rssm_det3d_N4_2x4_24e_pretrained_v2_head.py \
  3 --seed 0 --deterministic \
  --work-dir /data/lurui/work_dirs/crossmodal_fusion_N4_2x4_24e_seed0
```

工作目录：`/data/lurui/work_dirs/crossmodal_fusion_N4_2x4_24e_seed0`。
日志：`20260914_151155.log`、`20260914_151155.log.json`。
训练在 ep16 验证写入并完成门控判定后停止，未继续跑到 24 epoch，
也未启动 seed1/2。停止后 GPU 5/6/7 已释放。

### 40.4 ep12-16 门控结果

基线（完整 RSSM seed0 ep12-16 mean）：

| 指标 | 基线 | CrossModal seed0 ep12-16 mean | Δ |
|---|---:|---:|---:|
| Overall 3D moderate | 38.3902 | 37.9789 | -0.4113 |
| Overall BEV moderate | 46.9612 | 46.3749 | -0.5863 |
| Car 3D moderate strict | 47.8027 | 45.7627 | -2.0400 |
| Truck 3D moderate strict | 28.2149 | 31.4858 | +3.2709 |
| Cyclist 3D moderate loose | 48.6271 | 48.9808 | +0.3537 |
| Pedestrian 3D moderate loose | 28.9160 | 25.6864 | -3.2296 |

逐 epoch 明细：

| epoch | Overall 3D | Overall BEV | Car strict | Truck strict | Cyclist loose | Ped loose |
|---:|---:|---:|---:|---:|---:|---:|
| 12 | 38.0084 | 46.8464 | 46.9644 | 30.4760 | 48.5425 | 26.0507 |
| 13 | 37.5784 | 46.0572 | 43.7610 | 33.0847 | 48.2254 | 25.2427 |
| 14 | 38.9041 | 47.1531 | 47.5680 | 32.2926 | 49.2393 | 26.5165 |
| 15 | 36.1203 | 44.6835 | 43.0773 | 30.1687 | 48.4840 | 22.7510 |
| 16 | 39.2835 | 47.1344 | 47.4430 | 31.4070 | 50.4128 | 27.8710 |

### 40.5 门控判定

| 条件 | 结果 | 判定 |
|---|---:|---|
| Overall 3D moderate >= 38.8900 | 37.9789 | fail |
| 四个组成项均不得下降超过 1.0 | Car -2.0400, Ped -3.2296 | fail |
| 至少两个类别提升 >= 0.5 | Truck +3.2709, Cyclist +0.3537 | fail |
| Overall BEV moderate 提升 >= 0.5 | -0.5863 | fail |

**结论：恢复预训练 Cross-Modal Fusion 的 seed0 门控未通过，不启动 seed1/2，
不进入 24 epoch 完整训练。** 该改动确实加载了被浪费的预训练融合权重，且 Truck strict
有明确提升，但 Car strict、Ped loose 与 Overall/BEV 均回退，说明预训练融合权重与当前
RSSM/head 训练组合没有形成稳定的全网收益。下一步若继续主线，才考虑用户预设的
“共享 stem + 四类独立 prediction tower / 梯度平衡”，不在本次实验中展开。

---

## 41. 共享残差 Head Stem 的 seed0 门控（未通过，已停止）

### 41.1 实验动机与唯一结构改动

第 40 节说明 Cross-Modal Fusion 不是稳定全网收益：Truck 明显提升，但 Car、Ped 与
Overall/BEV 回退。检测头此时仍是四个输出直接 `1x1 Conv` 读取同一 BEV 特征，缺少共享的
局部空间上下文。为在不改变 RSSM、KL、BPTT、融合模块、anchor、loss 和 assigner 的前提下，
先验证“共享 head stem”这一最小结构改动，恢复 `ConcatConvFusion` 基线并在
`Anchor3DHead` 内加入默认关闭的共享残差 stem：

```text
RSSM fused BEV x
  |
  +-- shared stem:
      Conv3x3 256->256
      GroupNorm(32)
      ReLU
  |
  x_head = x + 0.1 * stem(x)
  |
  +-- original conv_cls / conv_reg / conv_dir / conv_iou
```

配置：

```python
shared_stem=True
shared_stem_channels=256
shared_stem_kernel_size=3
shared_stem_residual_scale=0.1
norm_cfg=dict(type='GN', num_groups=32)
```

主配置唯一融合回退：
`RCFusion.type: Cross_Modal_Fusion -> ConcatConvFusion`。其余固定项保持 N4、
`hidden_dim=128`、BPTT1、`kl_scale=1.0`、`free_nats=1.0`、`samples_per_gpu=2`、
`cumulative_iters=2`（有效 batch 12）、AdamW `lr=1.5e-4`、24 epoch、
`checkpoint_interval=1`、seed0、deterministic、GPU 5/6/7、
`load_from=checkpoints/pretrained_tj4d.pth`。旧模型路径 `shared_stem=False` 保持兼容。

相关提交：`b4fd7be feat(head): add shared residual stem baseline`。

### 41.2 训练设置与停止口径

命令：

```bash
source .envrc
CUDA_VISIBLE_DEVICES=5,6,7 SEED=0 bash tools/dist_train.sh \
  configs/r4det/TJ4D-R4Det_motion_align_rssm_det3d_N4_2x4_24e_pretrained_v2_head.py \
  3 --seed 0 --deterministic \
  --work-dir /data/lurui/work_dirs/shared_stem_N4_2x4_24e_seed0
```

工作目录：`/data/lurui/work_dirs/shared_stem_N4_2x4_24e_seed0`。
日志：`20260915_140142.log(.json)`、`20260915_160728.log(.json)`、`resume_from_ep1.log`。
前 1 epoch 跑完后从中途暂停，之后从完整 `epoch_1.pth` 后台续训到 ep16。
启动日志确认模型结构包含 `(shared_stem_layer): Sequential(...)`，且
`missing keys in source state_dict` 只包含预期的
`pts_bbox_head.shared_stem_layer.*`、`conv_iou.*`、`temporal_fusion.*` 等新增模块。
训练在 ep16 验证写入并完成门控判定后停止，未继续跑到 24 epoch，也未启动 seed1/2。
停止后 GPU 5/6/7 已释放。

### 41.3 ep12-16 门控结果

基线（完整 RSSM seed0 ep12-16 mean）：

| 指标 | 基线 | SharedStem seed0 ep12-16 mean | Δ |
|---|---:|---:|---:|
| Overall 3D moderate | 38.3902 | 39.2738 | +0.8836 |
| Overall BEV moderate | 46.9612 | 47.5756 | +0.6144 |
| Car 3D moderate strict | 47.8027 | 50.9517 | +3.1490 |
| Truck 3D moderate strict | 28.2149 | 31.6556 | +3.4407 |
| Cyclist 3D moderate loose | 48.6271 | 43.9478 | -4.6793 |
| Pedestrian 3D moderate loose | 28.9160 | 30.5398 | +1.6238 |

逐 epoch 明细：

| epoch | Overall 3D | Overall BEV | Car strict | Truck strict | Cyclist loose | Ped loose |
|---:|---:|---:|---:|---:|---:|---:|
| 12 | 39.1498 | 48.3258 | 49.1565 | 32.3175 | 44.0043 | 31.1207 |
| 13 | 37.9782 | 47.0692 | 52.9902 | 30.0694 | 41.7074 | 27.1457 |
| 14 | 40.3045 | 48.1600 | 51.6225 | 30.2040 | 45.3136 | 34.0779 |
| 15 | 39.0903 | 46.8956 | 49.6405 | 31.2571 | 46.7937 | 28.6697 |
| 16 | 39.8460 | 47.4272 | 51.3488 | 34.4300 | 41.9200 | 31.6850 |

### 41.4 门控判定

| 条件 | 结果 | 判定 |
|---|---:|---|
| Overall 3D moderate >= 38.8900 | 39.2738 | pass |
| Overall BEV moderate >= 47.4600 | 47.5756 | pass |
| 四个组成项均不得下降超过 1.0 | Cyclist loose -4.6793 | fail |
| 至少两个类别提升 >= 0.5 | Car +3.1490, Truck +3.4407, Ped +1.6238 | pass |

**结论：共享残差 Head Stem 的 seed0 门控未通过，不启动 seed1/2，不进入 24 epoch 完整训练。**
该结构确实提高了 Overall、BEV、Car strict、Truck strict 和 Ped loose，但 Cyclist loose
在 ep13/ep16 跌到 `41.71/41.92`，ep12-16 均值比基线低 `4.68`，超出「任一类别下降不得超过
1.0」的限制。共享 stem 不是无收益改动，但它把类别间分配问题从 Truck/Car 转移并放大到
Cyclist；按预设规则，下一步不应盲目继续堆 24 epoch 或多 seed，而应先测四类 loss 对 RSSM
输出的梯度 cosine，再决定 `PCGrad`、class-balanced loss 或独立 tower。

### 41.5 Cyclist 漏检诊断

诊断只使用推理 dump 和最后一个训练 checkpoint 的 CPU/GPU 只读分析，不启动新训练。
clean baseline 使用其训练目录内保存的配置快照和 `epoch_16.pth`；shared stem 使用
`epoch_14.pth`、`epoch_16.pth`。两套配置均保持 `ConcatConvFusion`、RSSM/N=4/BPTT=1 不变。

预测 dump：

| 模型 | checkpoint | nms_pre | dump |
|---|---:|---:|---|
| clean seed0 | ep16 | 1000 | `/data/lurui/work_dirs/diag_dumps/shared_stem_cyc/clean_seed0_ep16.pkl` |
| shared stem seed0 | ep14 | 1000 | `/data/lurui/work_dirs/diag_dumps/shared_stem_cyc/stem_seed0_ep14.pkl` |
| shared stem seed0 | ep16 | 1000 | `/data/lurui/work_dirs/diag_dumps/shared_stem_cyc/stem_seed0_ep16.pkl` |

`tools/cyc_drop_analysis.py` 的 same-class greedy quick recall：

| 模型 | Cyclist GT | loose recall | strict recall | pred/sample |
|---|---:|---:|---:|---:|
| clean ep16 | 2153 | 0.505 | 0.235 | 131.67 |
| stem ep14 | 2153 | 0.526 | 0.221 | 153.53 |
| stem ep16 | 2153 | 0.482 | 0.222 | 129.15 |

这说明 ep14 的 Cyclist loose quick recall 实际高于 clean，ep16 才下降。没有出现
“Cyclist 候选完全消失”的现象；三个模型的 `samples_with_0_cyc_pred` 都是 0。

`tools/cyc_drop_diagnosis.py` 进一步检查跨类重叠和最终输出上限：

| 模型 | same-class loose | any-label loose | 漏检中跟他类 loose 重叠 | final preds/sample | 打满 300 |
|---|---:|---:|---:|---:|---:|
| clean ep16 | 0.505 | 0.740 | 63 / 1066 | 294.85 | 1895 / 2040 |
| stem ep14 | 0.526 | 0.730 | 39 / 1020 | 299.08 | 2010 / 2040 |
| stem ep16 | 0.482 | 0.707 | 48 / 1116 | 298.43 | 1960 / 2040 |

跨类吸走只占 Cyclist 漏检的小部分，且 stem ep16 主要是 same-class loose recall 下降和
any-label 重叠下降。最终输出几乎全部打满 `max_num=300`，因此按预设规则做了唯一推理消融：
clean 与 shared stem 都只把 `nms_pre` 从 `1000` 调到 `2000`，其余 checkpoint/config 不变。

| 模型 | nms_pre=1000 loose | nms_pre=2000 loose | 变化 | pred/sample @2000 |
|---|---:|---:|---:|---:|
| clean ep16 | 0.505 | 0.510 | +0.005 | 176.45 |
| stem ep14 | 0.526 | 0.533 | +0.007 | 201.40 |
| stem ep16 | 0.482 | 0.497 | +0.015 | 172.81 |

`nms_pre=2000` 对所有模型都只有小幅 quick-recall 改善，stem ep16 相对 clean 仍有
`-0.013` 的 same-class loose quick recall 差距；lost/gained 配对仍是净损失目标
(`lost=227, gained=198`)，且 lost 目标中只有 5 个在 new dump 里被其他类 loose 重叠。
因此 pre-NMS 候选截断有贡献，但不能解释 Cyclist loose 的主缺口，不能把修复目标限定为
提高 `nms_pre`。

最后用 `tools/class_grad_cosine.py` 检查四类 loss 对 shared BEV/RSSM 输出特征的梯度方向。
验证集前 500 个样本中最后一帧没有任何 Pedestrian 正样本，因此 Pedestrian 无正样本的列只作背景
参考；关键看 Cyclist/Car/Truck。两个 stem ep16 窗口（samples 491-492、501-502）的分类梯度：

| 窗口 | Cyclist-Ped | Cyclist-Car | Cyclist-Truck | Cyclist vs mean |
|---|---:|---:|---:|---:|
| 491-492 | +0.896 | +0.036 | +0.811 | +0.902 |
| 501-502 | +0.968 | +0.198 | +0.942 | +0.970 |

加入 bbox loss 后（`cls_bbox`）仍为正：

| 窗口 | Cyclist-Ped | Cyclist-Car | Cyclist-Truck | Cyclist vs mean |
|---|---:|---:|---:|---:|
| 491-492 | +0.700 | +0.033 | +0.578 | +0.722 |
| 501-502 | +0.899 | +0.153 | +0.840 | +0.906 |

clean ep16 在同一窗口 491-492 的方向也相同：Cyclist-Ped `+0.886`、Cyclist-Car `+0.012`、
Cyclist-Truck `+0.825`。当前证据不支持「反复出现负向梯度冲突导致 Cyclist 被 Car/Truck/Ped
抵消」；更符合观测的解释是共享 stem 让 Cyclist 自身响应/排序下降，或学习过程把 Cyclist 的
少量鲁棒性能换给了 Car/Truck/Ped，而不是单纯 `PCGrad` 能解决的符号冲突。

**41.5 结论：** `nms_pre=2000` 不恢复骨干问题，四类 loss 对共享特征的梯度 cosine 也不是
负冲突。下一步不应先上 `PCGrad`；应保留 clean RSSM 为主模型，把 shared stem 保留为候选，
并针对 Cyclist 单独定位“分数下降/排序被挤”的来源。可选的下一个单变量是 Cyclist score/rank
诊断（例如 per-class score calibration 或只在推理阶段检查 Cyclist score bias），而不是同时
引入四类 tower 和梯度平衡。

---

## 42. `max_num=300 -> 600` 推理消融（未恢复 Cyclist loose，已停止）

### 42.1 动机与唯一变量

第 41 节已经排除了 `nms_pre=1000 -> 2000` 是 Cyclist loose 缺口的主因，但还存在一个更直接的
输出预算变量：`model.test_cfg.pts.max_num`。`box3d_multiclass_nms` 先逐类 NMS，再按跨类分数排序，
最后用 `max_num` 截断；因此 `nms_pre` 不是最终输出上限。shared stem 的 ep14/ep16 在 max300 下
分别有 `2010/2040`、`1960/2040` 帧打满，clean ep16 也有 `1895/2040` 帧打满，所以本次只做一个
单变量推理消融：

```text
model.test_cfg.pts.max_num: 300 -> 600
```

其余完全不变：`nms_pre=1000`、NMS 阈值、checkpoint、各自训练目录中的配置快照、正式
`tools/test_vod.py --eval bbox` 管线。未启动训练，未改模型结构或 loss。

复评命令模板：

```bash
source .envrc
CUDA_VISIBLE_DEVICES=5 python tools/test_vod.py \
  --config <训练目录配置快照> \
  --checkpoint <ckpt> \
  --gpu-id 0 --eval bbox \
  --out /data/lurui/work_dirs/diag_dumps/shared_stem_cyc_max600/<run>_max600.pkl \
  --saveoutput /data/lurui/work_dirs/diag_dumps/shared_stem_cyc_max600/<run>_max600.json \
  --cfg-options model.test_cfg.pts.max_num=600
```

### 42.2 正式 KITTI moderate AP

| 模型 | Overall 3D | Overall BEV | Cyclist 3D loose | Cyclist BEV loose | Car strict | Truck strict | Ped loose |
|---|---:|---:|---:|---:|---:|---:|---:|
| clean ep16 max300 | 39.8802 | 47.6956 | 50.4709 | 53.5793 | 49.9772 | 30.4303 | 28.6426 |
| clean ep16 max600 | 39.8811 | 47.6938 | 50.4659 | 53.5903 | 49.9769 | 30.4158 | 28.6657 |
| stem ep14 max300 | 40.3045 | 48.1600 | 45.3136 | 47.8553 | 51.6225 | 30.2040 | 34.0779 |
| stem ep14 max600 | 40.3017 | 48.1504 | 45.3115 | 47.8518 | 51.6219 | 30.1826 | 34.0907 |
| stem ep16 max300 | 39.8460 | 47.4272 | 41.9200 | 44.5349 | 51.3488 | 34.4300 | 31.6850 |
| stem ep16 max600 | 39.8451 | 47.4309 | 41.9173 | 44.5321 | 51.3485 | 34.4177 | 31.6970 |

max600 相对 max300 的配对变化全部接近 0：

| 模型 | Δ Overall 3D | Δ Overall BEV | Δ Cyclist 3D loose | Δ Cyclist BEV loose |
|---|---:|---:|---:|---:|
| clean ep16 | +0.0009 | -0.0018 | -0.0050 | +0.0110 |
| stem ep14 | -0.0028 | -0.0096 | -0.0021 | -0.0035 |
| stem ep16 | -0.0009 | +0.0037 | -0.0027 | -0.0028 |

同口径与 clean ep16 max600 相比：

| 模型 | Δ Overall 3D | Δ Overall BEV | Δ Cyclist 3D loose | Δ Cyclist BEV loose | Δ Car strict | Δ Truck strict | Δ Ped loose |
|---|---:|---:|---:|---:|---:|---:|---:|
| stem ep14 | +0.4206 | +0.4566 | -5.1544 | -5.7385 | +1.6450 | -0.2332 | +5.4250 |
| stem ep16 | -0.0360 | -0.2629 | -8.5486 | -9.0582 | +1.3716 | +4.0019 | +3.0313 |

### 42.3 输出上限占用

加载三个 max600 `.pkl`，按每个样本 `pts_bbox` 的输出数量统计：

| 模型 | mean preds/sample | min | max | 打满 600 | 打满 300 |
|---|---:|---:|---:|---:|---:|
| clean ep16 | 524.63 | 64 | 600 | 1151 / 2040 | 1895 / 2040 |
| stem ep14 | 557.56 | 179 | 600 | 1385 / 2040 | 2010 / 2040 |
| stem ep16 | 536.86 | 97 | 600 | 1132 / 2040 | 1960 / 2040 |

`max_num=600` 仍然大量打满，但正式 AP 几乎不变；这说明固定 300 不是当前 Cyclist loose 缺口的
瓶颈，或者说仅在最终预算上扩容不能恢复 Cyclist 的排序质量。

### 42.4 判定

- `max_num=600` 对 clean、stem ep14、stem ep16 的 Overall、BEV、Cyclist loose 都只有 `|Δ| < 0.02`
  的变化。
- stem ep14 相对 clean 的 Cyclist 3D loose 缺口仍为 `-5.15`，stem ep16 仍为 `-8.55`。
- 按预设停止规则，两个 stem checkpoint 的 Cyclist 缺口都没有明显收窄，因此不继续复评 stem ep12-16，
  也不用 max600 重算 clean 窗口，不启动 seed1/2。
- 保留 clean RSSM 为主模型；shared stem 仍是综合增益但未过 Cyclist 红线候选。
- 下一步若继续诊断，应按用户预设做“阈值正确的一对一匹配 + Cyclist PR 曲线”，区分几何未命中与
  “有合格框但排在 FP 后面”；不要先加 score bias、PCGrad 或四类 tower。

---

## 43. Cyclist 离线误差分解：oracle recall 与 41 点 PR（只读诊断）

### 43.1 目的与口径

第 41 节用 quick-recall 脚本统计 Cyclist 漏检，但它允许任意 `IoU>0` 的框先占用 GT，
与正式评估的 `IoU>=0.25` 一一匹配口径不同；第 42 节又排除了 `nms_pre` / `max_num`
的输出预算主因。本节按用户预设做**阈值正确**的离线分解，只使用已有 clean ep16、
shared stem ep14/ep16 预测 dump，不训练、不改模型、不加 score bias。

新增只读脚本 `tools/cyc_error_decomposition.py`，复用正式评估语义：

- 从各自训练目录的配置快照读取 `TJ4D_infos_val.pkl`；dump 中的 LiDAR 框按旧版
  `tools/test_vod.py` 约定转 camera 框；
- 用 VOD evaluator 的 CPU rotate IoU（`rotate_iou_cpu.rotate_iou_eval`）作为
  `d3_box_overlap`，保证与正式 `IoU>=0.25` 口径一致；
- 统计 moderate、Cyclist、3D 的 oracle recall（不看分数，一一匹配）与正式 41 点 PR，
  并给出分数字段的 TP/FP/FN。

复现门控：脚本内置三个 dump 的期望 AP40（clean ep16 `50.4709`、stem ep14 `45.3136`、
stem ep16 `41.9200`），复评必须逐项匹配，否则直接退出。本次运行三项全部 `OK`，
与正式 `tools/test_vod.py` 记录值一致。

### 43.2 oracle recall vs 正式 AP

moderate Cyclist GT 分母（difficulty 累积，`range<=70`）为 `2143`。

| 模型 | oracle recall | 一一匹配 GT | 正式 3D AP40 | AP vs clean |
|---|---:|---:|---:|---:|
| clean ep16 | 0.7611 | 1631 / 2143 | 50.4709 | — |
| stem ep14 | 0.7485 | 1604 / 2143 | 45.3136 | -5.1573 |
| stem ep16 | 0.7448 | 1596 / 2143 | 41.9200 | -8.5509 |

- oracle recall 只小幅下降：stem ep14 比 clean 低 `0.0126`（少 27 个可命中 GT），
  stem ep16 低 `0.0163`（少 35 个）。
- 正式 AP 下降远大于 oracle recall 损失：`-5.16` / `-8.55`。

### 43.3 41 点 PR 关键行

三个模型的 recall 在整条曲线基本停在各自的 oracle recall 平台（~0.75），
差别集中在平台上的精度衰减速度：

| PR 索引 | clean P | stem14 P | stem16 P |
|---:|---:|---:|---:|
| 0 | 1.0000 | 1.0000 | 1.0000 |
| 10 | 0.9926 | 0.9745 | 0.9354 |
| 15 | 0.9157 | 0.7376 | 0.5805 |
| 20 | 0.4540 | 0.2519 | 0.1737 |
| 25 | 0.0448 | 0.0440 | 0.0369 |
| 31 | 0.0066 | 0.0000 | 0.0000 |

clean 在 recall `0.76` 附近仍维持较高精度直到索引 ~15 才明显衰减；两个 stem
checkpoint 在相同 recall 平台上精度更早、更快下滑。

### 43.4 分数段 TP/FP/FN（same-class 一一匹配 IoU>=0.25）

| 分数段 | clean TP/FP | stem14 TP/FP | stem16 TP/FP |
|---|---:|---:|---:|
| 0.00-0.05 | 294 / 162088 | 32 / 121663 | 201 / 166968 |
| 0.05-0.10 | 329 / 63597 | 336 / 120139 | 368 / 62460 |
| 0.10-0.20 | 236 / 17427 | 363 / 36854 | 305 / 18143 |
| 0.20-0.30 | 112 / 2482 | 164 / 5284 | 118 / 3226 |
| 0.30-0.50 | 97 / 978 | 162 / 1597 | 109 / 1307 |
| 0.50-0.70 | 68 / 254 | 107 / 308 | 64 / 282 |
| 0.70-1.00 | 495 / 314 | 440 / 242 | 431 / 161 |
| FN | 512 | 539 | 547 |

关键观察：

- 未命中 GT（FN）只从 clean `512` 增到 stem14 `539`、stem16 `547`，与 oracle recall
  的小幅下降一致，不是主因。
- 高分段（`0.70-1.00`）命中 GT 反而减少：clean `495` → stem14 `440` → stem16 `431`。
- 中低分段（`0.05-0.30`）stem 的 TP 更多，但 FP 也明显更多，说明合格框存在，
  只是被排到大量低置信同/异类预测之后。

### 43.5 判定

- oracle recall 仅小幅下降（-0.0126 / -0.0163），说明**几何/候选层有少量损失**，
  不是 Cyclist 候选整体消失。
- 正式 AP 下降（-5.16 / -8.55）远大于 oracle recall 损失，且高分段 TP 减少、
  中低分段 FP 增多，说明**主导是固定候选内 Cyclist 分数排序/分离质量下降**，
  即“有合格框但排在 FP 后面”。
- 这满足用户预设的第二分支（oracle recall 基本持平而 AP 下降）。下一步若继续，
  唯一单变量是 **Cyclist 独立分类/score 分支**（保留 shared stem、RSSM、回归），
  seed0 单变量门控；不要先做常数 Cyclist score bias（固定候选集内不改变自身排序），
  也不先加 PCGrad 或四类 tower。
- clean RSSM 继续作主模型；shared stem 仍是综合增益但未过 Cyclist 红线的候选。

---

## 44. Cyclist 独立分类分支（保留 shared stem）seed0 完整 24 epoch（门控近似踩线，判定未通过）

### 44.1 动机与唯一结构改动

第 43 节的离线分解显示：shared stem 相对 clean 只少命中 27/35 个 Cyclist GT（oracle
recall 仅降 0.0126/0.0163），正式 AP 却低 5.16/8.55；第 42 节扩大输出数量几乎不改变
AP。因此假设痛点在**固定候选内的 Cyclist 分数排序/分离**，而不是候选数量或回归。

本次只加一个默认关闭的 Cyclist 分类残差分支，其余全部不动：

```text
RSSM fused BEV x -> shared stem(3x3,GN,ReLU, +0.1 residual)
  -> conv_cls (48ch = 12 anchors x 4 classes)   [原样]
  -> Cyclist 分支: 3x3 Conv(256->64) + ReLU + 1x1 Conv(64->12), 末层零初始化
  -> cls_score[Cyclist 通道] += 分支输出
```

关键点：

- `conv_cls` 布局是 `[anchor * num_classes + class]`，所以 Cyclist 通道是
  `[1,5,9,...,45]`，即**12 个 anchor 各自的 Cyclist logit**，不是只改 Cyclist 专属 anchor。
- 末层 1x1 零初始化，训练起点与 shared stem 基线完全一致（已验证 zero-init identity）。
- 只改 Cyclist 分类通道；其他类别 logit、回归、方向、RSSM、KL、融合、loss、
  `nms_pre=1000`、`max_num=300` 全部不变。

新增配置：`configs/r4det/TJ4D-R4Det_motion_align_rssm_det3d_N4_2x4_24e_pretrained_v2_head_cyccls.py`
（shared stem 配置 + `cyc_cls_branch=True, cyc_cls_branch_channels=64, cyc_cls_branch_class=1`，
与 stem 配置逐行 diff 仅这 3 行）。相关提交：

- `69e2072 feat(head): add Cyclist classification residual branch`
- `50fbd1d tools: add Cyclist gate watcher for ep12-16 window`
- `99727a5 tools: guard gate watcher on endpoint checkpoint`

### 44.2 接线验证（启动前）

在 GPU 5 上对三种输入做了单元级验证：

- head 从新配置构建成功：`num_anchors=12`、`num_classes=4`、分支 12 个输出；
- 零初始化 identity：共享权重拷贝后，分支与基线 `cls_score/bbox/dir` 逐元素完全一致；
- 扰动分支后，只有 12 个 Cyclist 通道变化（class idx 集合 = `{1}`），其他类别通道零变化；
- 反向验证：分支末层、共享 `conv_cls`、shared stem 都收到非零梯度。

### 44.3 训练设置

```bash
source .envrc
CUDA_VISIBLE_DEVICES=5,6,7 SEED=0 bash tools/dist_train.sh \
  configs/r4det/TJ4D-R4Det_motion_align_rssm_det3d_N4_2x4_24e_pretrained_v2_head_cyccls.py \
  3 --seed 0 --deterministic \
  --work-dir /data/lurui/work_dirs/cyccls_branch_N4_2x4_24e_seed0
```

从原预训练权重 `checkpoints/pretrained_tj4d.pth` 重新训练（不是接 stem ep16 续训），
跑到完整 **24 epoch**（未提前终止）。日志
`20260917_032423.log(.json)`，约 1.44 s/iter，ep24 于 UTC 2026-09-18 02:05 保存、
02:16 写出验证。启动日志确认模型结构含 `(cyc_cls_branch_conv): Sequential(...)`，
missing keys 只包含预期新增模块（`shared_stem_layer.*`、`conv_iou.*`、
`cyc_cls_branch_conv.*`、`temporal_fusion.*` 等），`conv_cls` 从预训练正常加载。

### 44.4 ep12-16 门控（预注册口径）

与 clean full-RSSM seed0 ep12-16 均值对照：

| 指标 | clean | CycCls seed0 ep12-16 mean | Δ |
|---|---:|---:|---:|
| Overall 3D moderate | 38.3902 | 39.5670 | +1.1768 |
| Overall BEV moderate | 46.5193 | 47.4584 | +0.9391 |
| Cyclist 3D moderate loose | 48.6271 | 47.9115 | -0.7156 |
| Pedestrian 3D moderate loose | 28.9160 | 29.8735 | +0.9575 |
| Car 3D moderate strict | 47.8027 | 50.7300 | +2.9273 |
| Truck 3D moderate strict | 28.2149 | 29.7530 | +1.5381 |

> **基线更正（2026-09-18）**：上一版此表把 clean BEV 写成 `46.9612`，那是 **KL=0**
> 消融 run（`rssm_kl0_N4_2x4_24e_seed0`，第 22.2 节）的 ep12-16 均值，不是 clean
> full-RSSM seed0。核对 clean 原始日志
> `/data/lurui/work_dirs/run10_headv2_multiseed/seed_0/20260821_171603.log.json` 后，
> clean seed0 ep12-16 BEV 正确值为 **`46.5193`**，因此真实 ΔBEV 是 **`+0.9391`**
> （不是 +0.4972），已经超过原本想检验的 `+0.5`。其余四个 clean 数值（Overall 3D、
> Cyclist、Ped、Car、Truck）经同一日志复核无误。

逐 epoch 明细：

| epoch | Overall 3D | Overall BEV | Cyclist loose | Ped loose | Car strict | Truck strict |
|---:|---:|---:|---:|---:|---:|---:|
| 12 | 40.63 | 48.27 | 49.62 | 30.08 | 51.68 | 31.17 |
| 13 | 40.01 | 47.90 | 47.29 | 29.87 | 53.08 | 29.78 |
| 14 | 39.48 | 47.21 | 50.36 | 31.64 | 48.26 | 27.64 |
| 15 | 37.70 | 46.19 | 46.73 | 26.37 | 48.22 | 29.46 |
| 16 | 40.02 | 47.73 | 45.55 | 31.41 | 52.42 | 30.72 |

门控判定：

| 条件 | 结果 | 判定 |
|---|---:|---|
| Overall 3D moderate >= 38.8900 | 39.5670 | pass |
| Overall BEV moderate >= 47.4600 | 47.4584 | **fail（差 0.0016）** |
| Cyclist loose >= 47.6300 | 47.9115 | pass |
| 四个组成项均不得下降超过 1.0 | Cyclist -0.7156，其余均为正 | pass |
| 至少两个类别提升 >= 0.5 | Ped +0.9575, Car +2.9273, Truck +1.5381 | pass |

**结论：按预注册的绝对门槛，ep12-16 门控仍未通过，唯一失败项是 BEV `47.4584` vs
门控线 `47.4600`，差 `0.0016`。** 但要注意：这条 `47.4600` 绝对线当初是用错误的
clean 基线 `46.9612 + 0.5` 写死的，**不能事后改成按正确基线 `46.5193 + 0.5 = 47.0193`
重算并宣称通过**；预注册的绝对线保持原样，本次仍记为未严格通过。

同时，按**相对 clean 的 Δ** 看，该改动实际是达标的：Overall `+1.1768`、BEV `+0.9391`
（超过 `+0.5`）、三个类别为正、Cyclist 仅 `-0.7156`（红线内，且相对 shared stem 的
`-4.68` 是大幅修复）。因此本次判为 **"方向成功，进入多 seed 复现"**，而不是
"原门槛严格通过"。

### 44.5 ep20-24 与最佳点（24 epoch 未带来额外收益）

| 窗口 | Overall 3D | Overall BEV | Cyclist loose | Ped loose | Car strict | Truck strict |
|---|---:|---:|---:|---:|---:|---:|
| clean ep12-16 | 38.3902 | 46.5193 | 48.6271 | 28.9160 | 47.8027 | 28.2149 |
| CycCls ep12-16 | 39.5670 | 47.4584 | 47.9115 | 29.8735 | 50.7300 | 29.7530 |
| CycCls ep12-18 | 39.6209 | 47.5145 | 47.9145 | 30.1960 | 50.5044 | 29.8689 |
| clean ep20-24 | 37.6084 | 45.1123 | 47.0090 | 27.0894 | 46.1852 | 30.1501 |
| CycCls ep20-24 | 38.2125 | 46.3303 | 47.5959 | 27.1196 | 46.6311 | 31.5037 |

**同窗配对（CycCls - clean，均按各自 seed0 同窗口均值）**：

| 窗口 | ΔOverall 3D | ΔOverall BEV | ΔCyclist loose | ΔPed loose | ΔCar strict | ΔTruck strict |
|---|---:|---:|---:|---:|---:|---:|
| ep12-16 | +1.1768 | +0.9391 | -0.7156 | +0.9575 | +2.9273 | +1.5381 |
| ep20-24 | +0.6041 | +1.2180 | +0.5869 | +0.0302 | +0.4459 | +1.3536 |

- **ep20-24 同窗对比**：新分支 Overall 比 clean 高 `+0.60`、BEV 高 `+1.22`、Cyclist 高
  `+0.59`，Truck 高 `+1.35`，Ped/Car 基本持平。也就是说后段并非"整体失败"，
  新分支在同窗口下仍然全面不劣于 clean。
- 但新分支**自身**的绝对表现 ep20-24 不如 ep12-16（Overall 39.57→38.21、BEV 47.46→
  46.33、Cyclist 47.91→47.60），说明跑满 24 epoch 对新分支没有额外收益，后段更像是在
  Truck 上继续涨、其他类别小幅回落。
- 全 24 epoch 各指标最佳点：Overall/BEV 在 ep12，Cyclist 在 ep10，Ped 在 ep14，
  Car 在 ep13，Truck 在 ep20。最佳点分散，说明后段不是单调改进。
- 若把窗口放宽到 ep12-18，BEV 从 `47.4584` 升到 `47.5145`，越过写死的 `47.46`；但这
  仍是事后选窗，不能当作预注册门控通过，只作为"该结构有综合增益潜力"的旁证。

### 44.6 判定与下一步

- 按**预注册绝对门槛**：仍记为未严格通过（BEV `47.4584` vs `47.4600`，差 `0.0016`）。
- 按**相对 clean 的 Δ**：该改动方向成功 —— Overall `+1.18`、BEV `+0.94`（> +0.5）、
  Cyclist 仅 `-0.72`（红线内，且相对 shared stem `-4.68` 大幅修复），三个类别为正。
- 因此判定：**"方向成功，进入多 seed 复现"**，不写成"原门槛严格通过"。
- 下一步**不改结构**，跑 seed1/2（保持原 24e 配置和学习率日程），分别与同 seed clean 的
  ep12-16、ep20-24 配对比较，不用单个最佳 epoch 宣称成功。若 seed1/2 也能保住 Cyclist
  红线、并呈现配对的 Overall/BEV 增益，再考虑定为可替换主模型。
- 暂不做 Cyclist 绕行 stem；不加常数 Cyclist score bias 或 PCGrad。

### 44.7 多 seed 复现结果（三 seed 完成，未通过；seed2 停在 ep18）

> 日期：2026-09-17 ~ 2026-09-19。承接 44.6 的「方向成功，进入多 seed 复现」，在
> `cyccls_branch_N4_2x4_24e_seed0` 之外顺序补跑 seed1/seed2，与**同 seed 的 clean
> full-RSSM**（第 19 节 Run 10 多 seed）做配对比较。本节记录三个 seed 的最终训练事实、
> ep12-16 门控的逐 seed 与三 seed 均值两套口径、峰值排名，以及补充跑的 ep18 单点复评。

#### 44.7.1 训练设置与最终中止口径

| 项 | seed0（44.3） | seed1 | seed2 |
|---|---|---|---|
| work_dir | `cyccls_branch_N4_2x4_24e_seed0` | `..._multiseed/seed_1` | `..._multiseed/seed_2` |
| 日志 | `20260917_032423` | `20260918_051529` | `20260919_023629` |
| GPUs | 5,6,7 | 5,6,7 | 5,6,7 |
| 有效 batch | 12 | 12 | 12 |
| 落盘 checkpoint | ep1-24 | ep1-22 | ep1-18 |
| 状态 | 完整 24e | ep22 验证完成、ep23 约 1000 iter 后主动中止 | ep18 验证开始时外部中止（非本人 kill） |

- 三 seed 均使用同一份配置 `TJ4D-R4Det_motion_align_rssm_det3d_N4_2x4_24e_pretrained_v2_head_cyccls.py`、
  同一 BASE_DIR 与 `--deterministic`，除 seed 外无差异。
- **seed0**：唯一完整跑满 24 epoch 的 seed，44.3-44.6 的全部结论以它为准。
- **seed1**：于 2026-09-19 02:18 由 tmux `cyccls_multiseed` 队列主动中止。`epoch_1..22.pth`
  与 `latest.pth` 均已落盘，ep23/24 未保存。中止依据：门控判定在**固定的 ep12-16 窗口**，
  该窗口在 seed1 上已完整闭合，后段不再改变门控结论。
- **seed2**：2026-09-19 02:36 UTC 另起会话 `cyccls_seed2` 单独补跑（原 `cyccls_multiseed`
  队列中止时未启动的 seed2 一并被终止，故此处为重启而非续跑）。2026-09-19 19:26 UTC
  前后 seed2 在 **ep18 验证阶段被外部中止**，原因未定位：`dmesg` 无 OOM/内核报错记录、
  未由本人 kill、GPU 当时无抢占迹象。`epoch_1..18.pth` 与 `latest.pth -> epoch_18.pth`
  均已落盘，训练循环的 `log.json` 只写入了 ep1-17 的验证行，ep18 的验证未写回日志。
- **seed2 是否补跑**：补跑目标已达成。seed2 的作用是补全三 seed 均值口径，而 ep18 中止
  发生在门控窗口 ep12-16 **之外**，ep12-16 已完整闭合，故 ep19-24 不影响本节任何门控
  判定。若日后需要三 seed 的 ep20-24 后段配对，才需要从 `epoch_18.pth` 恢复。

#### 44.7.2 ep12-16 门控（逐 seed 配对）

各 seed 取自身与**同 seed clean full-RSSM** 在 ep12-16 五 epoch 的窗口均值配对：

| 指标 | s0 Δ | s1 Δ | s2 Δ | 三 seed 均值 Δ | 三 seed Δ 标准差 |
|---|---:|---:|---:|---:|---:|
| Overall 3D moderate | +1.1768 | -0.9901 | -1.1368 | **-0.3167** | 1.30 |
| Overall BEV moderate | +0.9391 | -0.6999 | -2.1310 | **-0.6306** | 1.54 |
| Cyclist 3D moderate loose | -0.7156 | -0.0053 | -2.1335 | **-0.9515** | 1.08 |
| Pedestrian 3D moderate loose | +0.9576 | -3.7827 | -1.4779 | **-1.4343** | 2.37 |
| Car 3D moderate strict | +2.9273 | +2.8578 | -3.2056 | **+0.8598** | 3.52 |
| Truck 3D moderate strict | +1.5381 | -3.0304 | +2.2698 | **+0.2591** | 2.87 |

（Δ = CycCls − clean，同 seed 同窗口。逐 seed 绝对值见下表。）

三 seed 的 ep12-16 绝对均值与离散度：

| 指标 | CycCls 三 seed mean ± std | clean 三 seed mean ± std |
|---|---:|---:|
| Overall 3D moderate | 38.6795 ± 0.77 | **38.9962 ± 0.53** |
| Overall BEV moderate | 46.6034 ± 0.74 | **47.2340 ± 0.97** |
| Cyclist loose | 47.1233 ± 1.10 | **48.0748 ± 0.52** |
| Pedestrian loose | 28.4657 ± 1.82 | **29.9000 ± 0.87** |
| Car strict | **48.2508 ± 2.48** | 47.3910 ± 1.83 |
| Truck strict | **30.8780 ± 1.31** | 30.6189 ± 2.74 |

各 seed 的 ep12-16 逐 epoch 明细：

seed0（44.4 已列，此处略）；seed1：

| epoch | Overall 3D | Overall BEV | Cyclist loose | Ped loose | Car strict | Truck strict |
|---:|---:|---:|---:|---:|---:|---:|
| 12 | 39.70 | 47.66 | 46.50 | 27.83 | 54.42 | 30.04 |
| 13 | 38.04 | 46.35 | 48.06 | 26.76 | 48.41 | 28.94 |
| 14 | 36.53 | 45.75 | 48.91 | 24.43 | 40.67 | 32.13 |
| 15 | 39.71 | 47.09 | 49.14 | 27.27 | 51.13 | 31.30 |
| 16 | 37.06 | 43.89 | 45.39 | 25.79 | 46.63 | 30.44 |

seed2：

| epoch | Overall 3D | Overall BEV | Cyclist loose | Ped loose | Car strict | Truck strict |
|---:|---:|---:|---:|---:|---:|---:|
| 12 | 38.05 | 46.34 | 47.17 | 28.49 | 45.75 | 30.77 |
| 13 | 37.27 | 45.04 | 43.86 | 26.42 | 45.63 | 33.20 |
| 14 | 40.16 | 48.55 | 46.97 | 30.12 | 49.61 | 33.95 |
| 15 | 37.58 | 44.26 | 44.19 | 30.20 | 45.51 | 30.42 |
| 16 | 38.25 | 46.83 | 47.12 | 30.32 | 42.35 | 33.22 |

预注册门控判定（逐 seed 与三 seed 均值）：

| 条件 | seed0 | seed1 | seed2 | 三 seed 均值 | 判定 |
|---|---:|---:|---:|---:|---|
| Overall 3D moderate >= 38.8900 | 39.5670 pass | 38.2093 fail | 38.2622 fail | 38.6795 fail | **fail** |
| Overall BEV moderate >= 47.4600 | 47.4584 fail | 46.1478 fail | 46.2039 fail | 46.6034 fail | **fail** |
| Cyclist loose >= 47.6300 | 47.9115 pass | 47.5981 fail | 45.8605 fail | 47.1233 fail | **fail** |
| 四个组成项均不得下降超过 1.0 | 全过 | Ped -3.78 / Truck -3.03 fail | Cyc -2.13 / Ped -1.48 / Car -3.21 fail | Ped -1.43 fail | **fail** |
| 至少两个类别提升 >= 0.5 | Car +2.93 / Trk +1.54 pass | 仅 Car +2.86 fail | 仅 Trk +2.27 fail | 仅 Car fail | **fail** |

**结论：逐 seed 口径与三 seed 均值口径都未通过。** seed0「只差 BEV 0.0016、其余全过」
是三 seed 中的最好特例，不能代表该分支：seed1 五项里四项失败，seed2 五项里五项失败，
三 seed 均值在 Overall/BEV/Cyclist/Ped 上全为负。

#### 44.7.3 Car strict 信号的稳健性检查

44.4 曾把 Car strict 视为唯一跨 seed 方向一致的正向效应（seed0 +2.93 / seed1 +2.86）。
加入 seed2 后该结论被推翻：

| 口径 | s0 | s1 | s2 | 三 seed 方向 |
|---|---:|---:|---:|---|
| Car strict ep12-16 Δ | +2.9273 | +2.8578 | **-3.2056** | 2/3 为正、均值 +0.86、std 3.52 |
| Car strict BEST Δ | +1.7017 | +3.6425 | **-1.7345** | 2/3 为正 |

- seed2 的 Car strict 在 ep12-16 与 BEST 两个口径下都是**负**，且 ep12-16 的 -3.21
  超过了「单类下降不超过 1.0」红线。
- 三 seed 的 Car Δ 均值为 +0.86，但标准差 3.52 比均值大 4 倍，2/3 为正不足以支撑
  「该分支稳定提升 Car」的判断。
- 因此 Car strict 只能记为「**两个 seed 上为正、不是跨三个 seed 稳健的效应**」，不再作为
  支持该分支保留的依据。

#### 44.7.4 全轮峰值与全局峰值排名

各 run 取自身全轮峰值（max `Overall_3D_moderate`）：

| run | 峰值 @ep | Overall 3D | Overall BEV | Cyclist loose | Ped loose | Car strict | Truck strict |
|---|---:|---:|---:|---:|---:|---:|---:|
| CycCls seed0 | 12 | 40.6350 | 48.2658 | 49.6157 | 30.0768 | 51.6789 | 31.1687 |
| CycCls seed1 | 15 | 39.7104 | 47.0907 | 49.1415 | 27.2706 | 51.1331 | 31.2962 |
| CycCls seed2 | 14 | 40.1627 | 48.5545 | 46.9717 | 30.1160 | 49.6118 | 33.9513 |
| clean seed0 | 16 | 39.8802 | 47.6956 | 50.4709 | 28.6426 | 49.9772 | 30.4303 |
| clean seed1 | 15 | 40.5073 | 48.1272 | 49.3553 | 30.3896 | 47.4906 | 34.7938 |
| clean seed2 | 14 | 40.8800 | 49.8510 | 50.0905 | 31.3235 | 51.3463 | 30.7597 |

**CycCls 三 seed 自身峰值（Overall）**：40.6350 / 39.7104 / 40.1627。

| 指标 | Δ (s0 − s1) | Δ (s0 − s2) | Δ (s1 − s2) |
|---|---:|---:|---:|
| Overall 3D | +0.92 | +0.47 | -0.45 |
| Overall BEV | +1.18 | -0.29 | -1.46 |
| Cyclist loose | +0.47 | +2.64 | +2.17 |
| Ped loose | +2.81 | -0.04 | -2.85 |
| Car strict | +0.55 | +2.07 | +1.52 |
| Truck strict | -0.13 | -2.78 | -2.66 |

- 三 seed Overall 峰值极差 `0.92`，而平台期 Overall 的 epoch 间标准差本身就有 ±1.5
  （见 44.7.5），**seed 间峰值差小于单个 seed 内部的 epoch 抖动**。
- BEV/Cyclist/Ped/Car/Truck 的 seed 间峰值差普遍在 0.3~2.8，同样处在或低于对应单类
  的 epoch 间噪声带内。
- 因此「seed0 峰值明显高于 seed1」不能读成 seed 间存在结构性差异，更像是同一噪声分布里
  的三个采样。

全库 `Overall_3D_moderate` 峰值排名（扫描所有 `work_dirs/**/*.log.json`）：

| 排名 | run | 峰值 @ep | Overall 3D |
|---:|---|---:|---:|
| 1 | KL0 消融 seed1（第 25 节，已被否决） | 20 | 41.0994 |
| 2 | Run10 head-v2 seed2（主线） | 14 | 40.8800 |
| 3 | **CycCls seed0** | 12 | **40.6350** |
| 4 | Run10 原单次（ep11，未存盘） | 11 | 40.5950 |
| 5 | Run10 head-v2 seed1 | 15 | 40.5073 |
| 6 | KL0 消融 seed0 | 22 | 40.3481 |
| 7 | shared stem seed0（第 41 节） | 14 | 40.3045 |
| 8 | seed1_ep15_restore | 15 | 40.2945 |
| 9 | **CycCls seed2** | 14 | **40.1627** |
| 10 | Run10 head-v2 seed0 | 16 | 39.8802 |
| 11 | **CycCls seed1** | 15 | **39.7104** |

**是否超过历史最高**：

- CycCls 三 seed 中最高的是 seed0 `40.6350`，仅比 Run10 原始单次峰值 `40.5950` 高
  **+0.04**，**低于**主线 Run10 head-v2 的 released 最优 seed2 `40.8800`，更低于 KL0
  seed1 的 `41.0994`。
- CycCls seed1 `39.7104` 低于 Run10 三 seed 中最低的 `39.8802`；CycCls seed2 `40.1627`
  也低于 Run10 三 seed 的最低值之上不多，位于 Run10 分布下半区。
- 按统一口径（comparison 用 ep12-16 固定窗口均值，全轮峰值只作为上限单列），`40.6350`
  落在 Run10 三 seed BEST 分布 `40.42 ± 0.51` 的一个标准差内，**不能算刷新历史最高**。
  主线（Run10 head-v2）历史最高单点仍是 seed2 ep14 `40.88`。

#### 44.7.5 平台期噪声量级（差距来源）

ep10-22 平台期每类的 epoch 间波动（标准差 / 极差）：

| 指标 | cyc s0 std | cyc s1 std | cyc s2 std | 极差 s0 / s1 / s2 |
|---|---:|---:|---:|---:|
| Overall 3D | 1.61 | 1.52 | 0.95 | 5.2 / 4.8 / 3.0 |
| Overall BEV | 1.00 | 0.98 | 1.28 | 3.0 / 3.8 / 4.3 |
| Cyclist loose | 1.56 | 1.78 | 1.97 | 5.5 / 5.6 / 5.0 |
| Ped loose | 3.09 | 1.87 | 1.69 | 9.7 / 6.6 / 5.3 |
| Car strict | 3.29 | 4.32 | 2.46 | 12.0 / 15.6 / 7.8 |
| Truck strict | 2.03 | 2.22 | 1.68 | 8.1 / 8.3 / 4.8 |

- 同一 run 内部 Overall 就抖 ±1.5，Car 抖 ±2~4，Ped 抖 ±2~3。
- CycCls 相对 clean 的增益量级（Overall +0.6~1.2）**小于或接近这个 epoch 间噪声**，
  因此 ep12-16 的窗口均值本身是在噪声上采样，seed0 的正增益不稳定。
- seed0 与 seed1 的逐 epoch Overall 差：均值 +0.99，**逐 epoch 差的标准差 1.96**，
  即两 seed 差异连一个 sigma 都不到；加入 seed2 后三 seed Overall Δ 的窗口均值为
  -0.32、标准差 1.30，同样落在噪声带内。

#### 44.7.6 补充：seed2 ep18 单点复评（门控窗口之外）

seed2 训练循环中止在 ep18 验证开始时，`log.json` 未写入 ep18 验证行。为补全 seed2
的可用证据，用 `tools/test_vod.py` 对已落盘的 `epoch_18.pth` 做了一次**独立复评**
（非训练循环内的 DistEvalHook）：

```text
config     cyccls_branch_N4_2x4_24e_multiseed/seed_2/..._head_cyccls.py
checkpoint .../seed_2/epoch_18.pth
cmd        python tools/test_vod.py --config <cfg> --checkpoint <ckpt> --gpu-id 0 \
             --eval bbox --out /tmp/cyc_seed2_ep18.pkl
```

| 指标 | seed2 ep18 复评 | seed2 ep14（BEST） | clean seed2 ep18（同 seed 同 epoch） | Δ vs clean ep18 |
|---|---:|---:|---:|---:|
| Overall 3D moderate | 39.6956 | 40.1627 | 37.3955 | **+2.3001** |
| Overall BEV moderate | 46.6615 | 48.5545 | 45.6212 | **+1.0403** |
| Cyclist loose | 45.7119 | 46.9717 | 41.0076 | **+4.7043** |
| Pedestrian loose | 30.3297 | 30.1160 | 29.3232 | +1.0065 |
| Car strict | 47.6366 | 49.6118 | 48.7399 | -1.1033 |
| Truck strict | 35.1044 | 33.9513 | 30.5113 | **+4.5931** |

- 口径提醒：这是**单点、独立脚本复评**，与 44.7.2 的 ep12-16 五 epoch 窗口均值不可直接
  比较；且 ep18 在预注册门控窗口 ep12-16 **之外**，只能作为补充证据，不参与门控判定。
- 单看 ep18 这一天，seed2 的 CycCls 全面优于同 seed clean（Cyc +4.70、Trk +4.59、
  Overall +2.30），与 ep12-16 窗口上 seed2 全面落后形成明显反差。这正好说明该 run 的
  epoch 间抖动足以让单点结论翻转，不能用它翻案，也不能用它否定窗口结论。
- 另需注意：clean seed2 在 ep18 本身处于自身低谷（Overall 37.40，低于其 ep12-16 均值
  39.40），因此 ep18 的 +2.30 里含有「clean 当天偏低」的成分，不宜当作该分支的真实增益。

#### 44.7.7 判定

- **逐 seed 口径**：seed1、seed2 均复现失败，且都不是踩线失败。seed1 五项里四项不通过，
  seed2 五项全部不通过，Ped/Car/Cyc 分别在不同 seed 上破了 -1.0 红线。
- **三 seed 均值口径**：Overall `-0.32`、BEV `-0.63`、Cyclist `-0.95` 均为负，
  Ped `-1.43` 破红线，仅 Car `+0.86` 为正但标准差 3.52，**三 seed 均值同样不支持该分支**。
- **Cyclist 无增益**：三 seed Δ 为 -0.72 / -0.01 / -2.13，均值 -0.95、std 1.08，
  三 seed 中两个为负、一个接近 0。观测数据只能支持「该分支未在 Cyclist 上带来收益」，
  更不支持它作为「Cyclist 独立分类分支」的原始动机。
- **Car strict 不稳健**：seed2 翻负，三 seed 2/3 为正但均值小于标准差。44.7.3 已指出，
  不能再把 Car 记为跨 seed 稳健的正向效应。
- **Overall / BEV / Ped / Truck 不可复现**：seed0 与 seed1/2 方向相反，幅度都在噪声带内。
  44.5 的「ep20-24 同窗全面不劣于 clean」只在 seed0 上成立，seed1 后段 Ped 反而 -4.62，
  同样不成立。
- 至于 Car 受益与 Ped/Truck 受损是否由该分支的**结构设计**导致，现有三个 seed 的结果
  仍不足以区分原因：seed 间方向不一致、幅度在噪声带内，本节不下因果判断。
- 因此最终判定：**CycCls 分支在多 seed 复现下未被支持，不能定为可替换主模型。**
  主线维持 Run 10 head-v2 的 ep12-16 窗口均值 `39.00 ± 0.53`（全轮峰值 `40.42 ± 0.51`）。
  该分支记为「单 seed 上近似踩线、跨 seed 不可
  复现」，不再为整体 Overall 调整，也不在主线上启用。

#### 44.7.8 下一步

1. 主模型维持 Run 10 head-v2（第 19 节）：主对照 ep12-16 窗口均值 `39.00 ± 0.53`，
   全轮峰值 `40.42 ± 0.51`；不再为该分支投入训练。
2. 若确实需要 Cyclist 专项改进，应换一条与分类残差不同的假设重新预注册（例如候选排序
   之外的召回/回归方向），并直接按三 seed 均值 + ep12-16 固定窗口做门控，不先跑单 seed。
3. 后续任何单点峰值都必须配平台/固定窗口均值一起报，避免再次出现 seed0 ep12 `40.6350`
   这类落在噪声带内、却被读成「刷新历史最高」的误判。
4. seed2 若日后需要 ep20-24 后段配对，可从 `epoch_18.pth` 恢复补跑到 24e；但这不影响
   本节已经闭合的门控结论。

---

## 45. Clean Run10 checkpoint averaging（ep12 + ep14 + ep16）

> 日期：2026-09-20。目标是不重训，直接针对 Run 10 head-v2 平台期最大的 epoch 波动，
> 对 clean full RSSM 三 seed 分别平均 `epoch_12 + epoch_14 + epoch_16`，再用正式
> `tools/test_vod.py --eval bbox` 管线独立评估三个平均 checkpoint。

### 45.0 实现与协议

- 新增工具：`tools/average_checkpoints.py`。
- 平均规则：floating 参数与 floating buffers 逐元素算术平均；`num_batches_tracked`
  等非 floating buffers 从 `epoch_16` 复制；输出只含 `state_dict + meta`，不保留
  optimizer。
- checkpoint 产物仍放在原 seed 目录，避免覆盖任何训练权重：

| seed | 平均 checkpoint | 来源 |
|---|---|---|
| seed0 | `run10_headv2_multiseed/seed_0/epoch_avg_12_14_16.pth` | ep12 + ep14 + ep16 |
| seed1 | `run10_headv2_multiseed/seed_1/epoch_avg_12_14_16.pth` | ep12 + ep14 + ep16 |
| seed2 | `run10_headv2_multiseed/seed_2/epoch_avg_12_14_16.pth` | ep12 + ep14 + ep16 |

平均后已逐项校验：每 seed 的 565 个 floating tensor 与
`(ep12 + ep14 + ep16) / 3` 逐元素完全一致；102 个非 floating buffer 与 ep16 完全一致；
输出中无 `optimizer` key。

正式评估命令模板：

```bash
source .envrc
CUDA_VISIBLE_DEVICES=5 python tools/test_vod.py \
  --config /data/lurui/work_dirs/run10_headv2_multiseed/seed_0/TJ4D-R4Det_motion_align_rssm_det3d_N4_2x4_24e_pretrained_v2_head.py \
  --checkpoint /data/lurui/work_dirs/run10_headv2_multiseed/seed_0/epoch_avg_12_14_16.pth \
  --gpu-id 0 --eval bbox \
  --out /data/lurui/work_dirs/run10_headv2_avg_ep12_14_16/seed_0.pkl \
  --saveoutput /data/lurui/work_dirs/run10_headv2_avg_ep12_14_16/seed_0.json
```

seed1/seed2 使用各自 seed 目录中的同名校验点与配置，分别调度到 GPU6/GPU7。

### 45.1 三 seed 正式复评结果

| 指标 | seed0 | seed1 | seed2 | 三 seed 均值 | 样本 std |
|---|---:|---:|---:|---:|---:|
| Overall 3D moderate | 40.4785 | 40.0937 | 40.9735 | **40.5152** | **0.4410** |
| Overall BEV moderate | 48.1572 | 48.1472 | 50.5913 | **48.9652** | 1.4082 |
| Car 3D moderate strict | 50.8649 | 46.4897 | 50.4077 | 49.2541 | 2.4049 |
| Cyclist 3D moderate loose | 49.2965 | 47.9385 | 50.5572 | 49.2641 | 1.3097 |
| Pedestrian 3D moderate loose | 31.3312 | 30.9571 | 31.4358 | 31.2414 | 0.2517 |
| Truck 3D moderate strict | 30.4213 | 34.9894 | 31.4934 | 32.3014 | 2.3888 |

完整 JSON 输出保存在
`/data/lurui/work_dirs/run10_headv2_avg_ep12_14_16/seed_{0,1,2}.json`。

### 45.2 与主线全轮峰值口径对比

主线为 Run 10 head-v2 clean full RSSM 三 seed 的全轮峰值：Overall `40.42 +/- 0.51`。
本节平均权重不是从每个 seed 的曲线中挑 epoch，而是固定 ep12+ep14+ep16 等权平均。

| 指标 | 主线三 seed 均值 | 平均权重三 seed 均值 | Δ（平均权重 − 主线） | 判定 |
|---|---:|---:|---:|---|
| Overall 3D moderate | 40.42 | **40.5152** | +0.0919 | 达到主线水平 |
| Overall BEV moderate | 48.56 | **48.9652** | **+0.4052** | 稳定提升 ≥ 0.3 |
| Car strict | 49.60 | 49.2541 | −0.3526 | 未下降超过 1.0 |
| Cyclist loose | 49.97 | 49.2641 | −0.7093 | 未下降超过 1.0 |
| Pedestrian loose | 30.12 | 31.2414 | +1.1247 | 未下降超过 1.0 |
| Truck strict | 31.99 | 32.3014 | +0.3080 | 未下降超过 1.0 |

BEV 的逐 seed 配对差为 `+0.4572 / +0.0172 / +0.7413`，均值 `+0.4052`、std `0.3648`。
虽然 seed1 只有 `+0.0172`，但三项均为正，且三 seed 均值为正，因此按「Overall/BEV 至少一项
稳定提升 ≥ 0.3」判定 BEV 通过。Overall 的逐 seed 配对差均值为 `+0.0919`、std `0.5074`，
没有达到 0.3 提升阈值；其作用是把三 seed 均值维持在主线 BEST 水平之上。

### 45.3 通过标准核对

| 标准 | 实测 | 判定 |
|---|---:|---|
| 三 seed Overall 全轮峰值口径均值 ≥ 40.42 | 40.5152 | ✅ |
| seed 间标准差不高于 0.51 | 0.4410 | ✅ |
| 四个类别三 seed 均值任一不得下降超过 1.0 | Car −0.3526、Cyc −0.7093、Ped +1.1247、Truck +0.3080 | ✅ |
| Overall/BEV 至少一项稳定提升 ≥ 0.3 | Overall +0.0919；BEV +0.4052 | ✅ BEV |

### 45.4 结论

- **通过。** Clean Run10 的 ep12+ep14+ep16 平均权重在三 seed 上保持 Overall
  `40.52 +/- 0.44`，方差低于主线 `0.51`，并给 BEV 带来 `+0.41` 的稳定提升。
- 该路线不需要重训，直接利用已落盘 checkpoint，是目前最贴近「平台期 epoch 波动」证据的
  低成本稳定化方案。
- 类别均值没有出现超过 1.0 的下降；Car/Cyclist 的小幅下降被 Ped/Truck 的提升抵消在
  Overall 内，但不应被解释为对 Car/Cyclist 有正向作用。
- 后续若进入论文主表或最终方案，应优先报告三个平均 checkpoint 的独立复评结果；
  全轮峰值仍按统一口径单列，但不得用它替代固定窗口的主对照结论。

---

## 46. FG-FULL 数据管线修复与 500 样本审计（2026-09-20）

### 46.1 背景

FG-FULL 正式 run 在 `fgfull_N4_2x4_24e_seed0` 启动后被发现数据管线存在两个真实问题，
因此主动停止该 run（当时尚未落盘任何 checkpoint，不会污染已有结果）：

- `LoadVLSAMAnnotations` 每个 GT 无条件写调试图 PNG，约 15 分钟已写 16663 个文件，
  调试目录累积到 21617 文件 / 107M。
- 更严重：无同类匹配时仍可能使用 `best_match_idx=-1` 的最后一个 mask；同一个 mask 也可
  重复分配给多个 GT，`iou_threshold` 与 `is_available` 实际未生效，2D mask / IGDR 监督
  可能被错误标签污染。

### 46.2 修复内容

`mmdet3d/datasets/pipelines/loading_custom.py`：

1. `LoadVLSAMAnnotations` 新增 `save_vis=False`，默认完全禁止写调试图；只有显式
   `save_vis=True` 才落 PNG。
2. 新增 `_match_official_to_vlsam()`：同类别、IoU 阈值（`iou >= iou_threshold`）、
   按 IoU 降序的一对一贪心匹配。
3. 未匹配 GT 填零 mask，禁止 `-1` 索引。
4. 已使用的 VLSAM mask 与 GT 均不得再次分配。
5. 输出 `results['vlsam_match_debug']`，记录 `num_gt`、`num_vlsam`、`matched`、
   `unmatched`、`match_rate`、`mean_matched_iou`、`matched_indices`、`match_ious`。

### 46.3 500 样本审计

审计脚本 `me_rssm/sanity/audit_vlsam_matching.py` 包装 live pipeline 的匹配函数，
在看到真实增广前 GT box/label 与 VLSAM box/label 的前提下统计匹配行为。命令：

```bash
export PATH="/home/lurui/envs/miniforge3/envs/r4det/bin:$PATH"
CUDA_VISIBLE_DEVICES=5 python me_rssm/sanity/audit_vlsam_matching.py --num-samples 500 \
  2>&1 | tee /data/lurui/work_dirs/fgfull_N4_2x4_24e_seed0/vlsam_match_audit_500.log
```

实测结果（日志：`/data/lurui/work_dirs/fgfull_N4_2x4_24e_seed0/vlsam_match_audit_500.log`）：

| 统计项 | 数值 |
|---|---:|
| Audited samples | 500 |
| Recorded matching calls | 1985 |
| GT targets | 23646 |
| Matched GT | 14359 |
| Unmatched GT | 9287 |
| Match rate | 0.6072 |
| Mean sample match rate | 0.6084 |
| Duplicate assignments | 0 |
| Cross-class matches | 0 |
| Below-threshold matches | 0 |
| Mean matched IoU | 0.6015 |
| Median matched IoU | 0.6280 |
| Min / Max matched IoU | 0.1006 / 0.9495 |

判定：`==== AUDIT: PASS ====`。

### 46.4 100 iter DDP smoke

为绕开 `train_vod.py`（未走 `compat_cfg`）继承 `runner.max_epochs` 的问题，新增验证用配置
`me_rssm/sanity/fgfull_smoke_100iter.py`，用 `_delete_=True` 把 runner 直接替换为
`IterBasedRunner(max_iters=100)`。正式 24e 配置未改动。

```bash
source .envrc
CUDA_VISIBLE_DEVICES=5,6,7 bash tools/dist_train.sh \
  me_rssm/sanity/fgfull_smoke_100iter.py 3 --seed 0 --deterministic \
  --work-dir /data/lurui/work_dirs/fgfull_smoke_100iter_patched
```

结果：

- `Iter [50/100]` 与 `Iter [100/100]` 均正常，loss 无 NaN，无 OOM。
- 显存峰值约 18.3 GiB（单卡 log 口径），3 卡 DDP 稳定。
- `Saving checkpoint at 100 iterations` -> `iter_100.pth`（813M）已落盘。
- 期间 debug PNG 文件数保持 `21617` 不变，默认禁止写图生效。

### 46.5 结论

数据管线修复 + 500 样本审计 + 100 iter DDP smoke 全部通过。FG-FULL 可以在干净目录
`work_dirs/fgfull_N4_2x4_24e_seed0` 重新启动；本修复不改变正式 24e 配置的任何训练超参。

### 46.6 FG-FULL redundancy removal (2026-09-20)

User question: why is FG-FULL 53-55% slower when the 4-frame BEV path is
already the baseline. ./R4Det.py stores the current-frame-only
supervision inside the shared N-frame extract_feat loop, so three kinds of
work were being repeated on history frames and one RPN pass was duplicated on
the current frame.

Changed in `mmdet3d/models/detectors/R4Det.py`:

1. Added `curr_frame_supervision` to `extract_feat` and
   `with_curr_frame_supervision` to `preprocessing_information`.
   Training history-frame calls now pass `curr_frame_supervision=False`.
2. History frames skip `rangeview_foreground`, FRPN former/latter, and the
   Shapely-based `generate_bev_mask`. These tensors are only consumed by
   current-frame losses, so the current-frame loss semantics are unchanged.
3. `img_rpn_head.forward_train` already returns decoded RPN proposals. The IGDR
   intermediate path now reuses `[p.detach() for p in proposal_list]`
   instead of rerunning `img_rpn_head.simple_test_rpn`.
4. No config/hyperparameter change; the running formal job was not restarted.

Verification:

- `python -m py_compile mmdet3d/models/detectors/R4Det.py`: pass.
- `git diff --check`: pass.
- `python me_rssm/sanity/smoke_fgfull_gpu.py 2`
  ; train forward/backward, IGDR path, and val
  forward all ran. Peak memory ~19.0 GiB.
- Single-card smoke timing after the change: ~1.7-1.8 s/iter on the first
  two iterations (the previous same smoke path was ~2.4 s/iter). This is a
  smoke-level measurement, not the final 3-GPU DDP number.
- Expected gain: most of the avoidable extra work is removed, but exact DDP
  speedup still needs a fresh 100-iter / same-iteration comparison. Do not
  claim the formal 53-55% gap is fully closed until that comparison is run.

### 46.7 Formal 3-GPU resume and measured speedup (2026-09-20)

After epoch 2 checkpointing and validation completed, the old-code FG-FULL
process was stopped and the formal run was resumed from `epoch_2.pth` with the
redundancy-removal commit active.

Resume command:

```bash
source .envrc
CUDA_VISIBLE_DEVICES=5,6,7 bash tools/dist_train.sh \
  configs/r4det/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head.py 3 \
  --seed 0 --deterministic \
  --work-dir /data/lurui/work_dirs/fgfull_N4_2x4_24e_seed0 \
  --resume-from /data/lurui/work_dirs/fgfull_N4_2x4_24e_seed0/epoch_2.pth
```

Loaded checkpoint state:

- `load checkpoint from local path: .../epoch_2.pth`
- `resumed epoch 2, iter 3804`
- GPU 5/6/7 occupied; no other GPU used.

Measured epoch-3 logging before/after the resume:

| Iter | Old code `time` (s/iter) | New code resumed from epoch2 `time` (s/iter) |
|---:|---:|---:|
| 50 | 2.479 | 2.369 |
| 100 | 2.080 | 1.905 |
| 150 | 2.115 | 1.947 |
| 200 | 2.103 | 1.935 |
| 250 | 2.141 | 1.989 |
| 300 | 2.127 | pending |
| 350 | 2.042 | pending |
| 400 | 2.094 | pending |

Steady-state comparison over 100-250 iters is about 1.94-1.99 s/iter after
the fix versus 2.08-2.14 s/iter before, i.e. roughly 8% faster on the formal
3-GPU run. This is a partial reduction of the earlier 53-55% gap, not a claim
that the entire gap is closed.

Running log:

`/data/lurui/work_dirs/fgfull_N4_2x4_24e_seed0/train_stdout_resume_after_ep2_20260920.log`

### 46.8 FG-FULL N=3 relay queue (2026-09-20)

The running N=4 FG-FULL job is a clean comparison point for a shorter sequence,
and a N=3 variant was prepared without stopping the current N=4 run.

Note on an earlier mistake: the first version of this config also dropped
`hidden_dim` 128 -> 64, justified by the historical capacity/frame match in
section 8. That was not what was requested. The request was a strict
frame-count ablation. The config has been corrected to change `seq_len` only
and keep `hidden_dim=128`. The mistake was caught at epoch 2 of the hdim64 run
and that run was stopped before any checkpoint was written; its partial logs
remain under `/data/lurui/work_dirs/fgfull_N3_2x4_24e_seed0/`. The corrected
run uses a separate work dir, `/data/lurui/work_dirs/fgfull_N3_h128_2x4_24e_seed0/`.

New config:

`configs/r4det/TJ4D-R4Det_fgfull_N3_2x4_24e_pretrained_v2_head.py`

Merged settings:

| Setting | Value |
|---|---:|
| `model.seq_len` | 3 |
| `model.temporal_fusion.hidden_dim` | 128 (unchanged from N=4) |
| train/val/test dataset `seq_len` | 3 |
| `samples_per_gpu` | 2 |
| `cumulative_iters` | 2 |
| `max_epochs` | 24 |

The preflight tool `me_rssm/sanity/test_fgfull_config.py` now accepts an
optional config path so the same F1-F4 checks can be reused for the N=3
configuration.

Relay queue:

`me_rssm/queue_fgfull_n3_after_n4.sh`

The queue waits for the current N=4 `train_vod.py` process to exit, drains
GPUs 5/6/7, reruns the preflight for the N=3 config, then launches into the
separate work dir:

`/data/lurui/work_dirs/fgfull_N3_2x4_24e_seed0`

Crash handling is the same as the existing queue: retry up to 3 times and
resume from `latest.pth` when present. Queue state is logged to
`/data/lurui/work_dirs/fgfull_n3_queue.log`.

### 46.9 N=3 GPU4 smoke test (2026-09-20)

Before letting the relay queue take over, the N=3 config was exercised on the
otherwise-free GPU 4. `me_rssm/sanity/smoke_fgfull_gpu.py` now takes an
optional config path so the same train-step + val-forward smoke can target any
fgfull variant.

Command:

```bash
source .envrc
CUDA_VISIBLE_DEVICES=4 python me_rssm/sanity/smoke_fgfull_gpu.py 2 \
  configs/r4det/TJ4D-R4Det_fgfull_N3_2x4_24e_pretrained_v2_head.py
```

Resolved settings and result:

| Item | Value |
|---|---:|
| `seq_len` | 3 |
| `temporal_fusion.hidden_dim` | 64 |
| iter 0 | `total_loss=10.758`, `1.7s` |
| iter 1 | `total_loss=9.821`, `1.4s` |
| peak memory | 18.75 GiB |
| val forward | `300` boxes per sample |
| verdict | `SMOKE: ALL OK` |

Checkpoint warnings (mismatched detection-head keys, missing new temporal and
IGDR keys) are the same expected ones seen for the N=4 smoke, because the
external pretrained checkpoint predates these modules.

### 46.10 N=3 vs N=4 speed split: seq_len or hidden_dim (2026-09-20)

Two separate questions were measured here: what the corrected N=3 run
(`seq_len` 4->3 only, `hidden_dim` kept at 128) actually costs, and how much of
the speed difference comes from frame count versus `hidden_dim`. The latter
uses a benchmark-only override `me_rssm/sanity/configs/fgfull_N4_hdim64_bench.py`
that keeps `seq_len=4` but drops `hidden_dim` to 64.

Controlled single-GPU benchmark on an idle GPU, same dataset/loader/batch as
training (`me_rssm/sanity/bench_fgfull_speed.py`, 2 warmup + 5 timed iters,
CUDA-synchronized).

Corrected N3 (`hidden_dim=128`, the run actually launched), on GPU 4:

```bash
source .envrc
CUDA_VISIBLE_DEVICES=4 python me_rssm/sanity/bench_fgfull_speed.py 2 5 \
  configs/r4det/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head.py \
  configs/r4det/TJ4D-R4Det_fgfull_N3_2x4_24e_pretrained_v2_head.py
```

| Config | T1 train (s/iter) | T2 inference (s/sample) |
|---|---:|---:|
| N4, hidden_dim=128 | 1.580 | 0.299 |
| N3, hidden_dim=128 (this run) | 1.238 | 0.239 |

So the pure frame-count ablation gives **1.28x train (+21.6% time saved)** and
**1.25x inference (+20.0%)**. Peak memory is essentially identical
(19.07 GiB both). Why that is not a contradiction is explained below.

#### Why more history frames do not cost activation memory

The reason is that this model truncates backprop-through-time. `R4Det.forward_train`
(`mmdet3d/models/detectors/R4Det.py`) computes:

```python
history_steps = N - 1
grad_steps = min(rssm_bptt_steps, history_steps)   # default rssm_bptt_steps=1
burn_in = history_steps - grad_steps
```

Frames in the burn-in window are run under `torch.no_grad()`, so their
activations are released as soon as that forward returns. Only the single most
recent history frame runs with `rssm_detach_state=False` and keeps its graph,
plus the current frame. Concretely:

| Run | burn-in (`no_grad`) frames | grad-carrying history frames | current frame |
|---|---:|---:|---:|
| N=3 | 1 | 1 | 1 |
| N=4 | 2 | 1 | 1 |

So N=4's extra frame is a detached forward: it costs time, not peak memory. The
peak comes from the current frame's backward through the 2D / FRPN / MRF3Net /
IGDR branches, which is identical in both configs. That is why the two peak
numbers differ by only ~0.1 GiB and the time differs by ~25%.

Measured confirmation (same single-GPU harness, batch=2, peak
`max_memory_allocated`):

| Config | `rssm_bptt_steps` | Peak | Result |
|---|---:|---:|---|
| N=4 | 1 (default) | 18.96 GiB | OK |
| N=4 | 2 | > 24 GiB | OOM |
| N=4 | 3 | > 24 GiB | OOM |
| N=4 | -1 (full) | > 24 GiB | OOM |
| N=3 | 1 (default) | 18.84 GiB | OK |
| N=3 | 2 | > 24 GiB | OOM |
| N=3 | -1 (full) | > 24 GiB | OOM |

Two things follow. First, the flat memory is a property of the BPTT
truncation, not of the model being insensitive to frame count: the moment
`rssm_bptt_steps` is raised to 2, the history activations do have to be kept
and both N=3 and N=4 immediately exceed a 24 GiB card. Second, the current
`rssm_bptt_steps=1` setting means the RSSM receives gradient from only one
history step; the older frames shape the state forward-only. If longer-range
gradient flow is wanted later, it cannot be done at batch=2 on a 24 GiB card
without gradient checkpointing or smaller batches.

#### What the forward-only history frames actually contribute

The two history loops in `forward_train` are distinct in three ways, not just
one: gradient, supervision, and loss participation.

| | burn-in frames (`range(burn_in)`) | grad frames (`range(burn_in, N-1)`) |
|---|---|---|
| call site | `torch.no_grad()` + default `detach_state=True` | `rssm_detach_state=False`, state kept in graph |
| `curr_frame_supervision` | `True` (default arg) | `False` (explicit) |
| detection losses | none (returns only `bev_feats`) | none (feat_or_dict=0) |
| RSSM losses | none | `rssm_recon` + `kl` appended to `history_rssm_losses` |
| state effect | advances `h/z` numerically for later frames | advances `h/z`, and the *path* is differentiable |

Two consequences worth being explicit about:

1. **Neither history loop trains detection.** The 3D head, 2D RPN/RoI head,
   FRPN, MRF3Net and IGDR mask losses are all computed from the current frame
   only (`feat_or_dict=1`). History frames never call the heads.
2. **Only the grad-window frames train the RSSM itself.** `rssm_history_losses`
   collects each such frame's `mse(reconstruction, bev_feats_cache)` and its
   KL, which are then averaged with the current frame's:

   ```python
   rssm_recon_loss = (sum(loss[0] for loss in history_rssm_losses)
                      + rssm_recon_loss) / (len(history_rssm_losses) + 1)
   ```

   So a burn-in frame contributes no loss term at all; it exists purely to
   move the state forward.

With the default `rssm_bptt_steps=1`, N=4 means one grad frame and two burn-in
frames; N=3 means one grad frame and one burn-in frame. That is why the
extra frame in N=4 costs time (its full BEV forward still runs) but not
memory (no graph is retained), and why it adds no direct loss either.

The older frames are not dead weight even without gradient. In
`MotionAlignedRSSMFusion.forward` each frame performs:

1. deformably align `h_{t-1}` / `z_{t-1}` to the current BEV feature,
2. `h_t = ConvGRU(z_aligned, h_aligned)`,
3. `z_t ~ q(z_t | h_t, encoder(feat_t))`,
4. store `(h_t, z_t)` as the state for the next frame.

Because `h_state` / `z_state` are carried forward as plain tensors (detached,
but numerically real), an older frame's *output* state still conditions every
later frame's prior, posterior and GRU gates. What it cannot do is receive
credit assignment: gradients never reach that frame's encoder, alignment
layers, or BEV backbone. So it contributes:

- a longer, motion-aligned context window: three history frames instead of one
  when going N=2 -> N=4, giving the GRU a state built from more observations;
- more robust state initialization, e.g. surviving a momentarily empty or
  degenerate frame;
- nothing in terms of learning those frames' features, since no gradient
  flows there.

This also means the frame-count ablation is measuring "longer forward context"
plus "more cost", not "longer learned temporal credit assignment". Those are
different questions, and only the second one needs multi-step BPTT.

#### Consequence for the historical N-frame ablation (section 8)

This matters for how section 8 should be read. `rssm_bptt_steps` was introduced
in `094d663` (2026-08-19). Before that commit, every history frame was rolled
under `torch.no_grad()` unconditionally. The section 8 configs
`..._N3_2x3_12e.py`, `..._N4_2x4_12e.py`, `..._N3_hdim128_2x4_12e.py` and
`..._N4_hdim64_2x4_12e.py` were created on 2026-08-09 and never set
`rssm_bptt_steps`, so they all ran with zero gradient through history.

So the section 8 conclusion "N=3 -> N=4 gives no gain, the marginal frame is
exhausted" is a statement about *forward-only context*, not about truncated
BPTT. The later Run 14 tested `rssm_bptt_steps=1` on top of N=4/hdim128 and was
a net loss (37.84 vs 39.65 stored). Nobody in this history has tested an N=4
model whose history frames actually receive gradient. If the question of
interest is "do more frames help the RSSM learn temporal structure", section 8
does not answer it.

The three-way hdim split was measured earlier on GPU 3, against the superseded
N3/hdim64 config (that config has since been corrected to hdim128, so only the
numbers are reproduced here):

| Config | T1 train (s/iter) | T2 inference (s/sample) |
|---|---:|---:|
| N4, hidden_dim=128 | 1.404 | 0.271 |
| N4, hidden_dim=64 (isolates seq_len) | 1.347 | 0.241 |
| N3, hidden_dim=64 (superseded) | 1.176 | 0.197 |

Derived speedups:

| Comparison | Train | Inference |
|---|---:|---:|
| N3/h128 vs N4/h128 (frame count alone) | 1.28x (+21.6%) | 1.25x (+20.0%) |
| N3/h64 vs N4/h128 (superseded combo) | 1.19x (+16.2%) | 1.37x (+27.2%) |
| N4/h64 vs N4/h128 (hidden_dim alone) | 1.04x (+4.0%) | 1.12x (+10.9%) |

Conclusions:

- Dropping the 4th frame alone buys ~1.25-1.28x on both train and inference.
- `hidden_dim` 128->64 buys only ~4% train / ~11% inference by itself. It
  shrinks inference time but barely touches the train step.
- The two effects overlap, so the earlier N3/h64 combo (1.19x train, 1.37x
  inference) was not simply the sum of its parts; on train it was actually
  slower than the pure frame-count change measured later.

Cross-check against the live 3-GPU DDP logs at this point in training:

| Run | Steady-state s/iter (median) |
|---|---:|
| N3 (GPUs 0/1/2, hdim64, superseded) | 1.824 (last 30 logs) |
| N4 (GPUs 5/6/7) | 1.925 (last 60 logs) |

That is only ~1.05x on the formal 3-GPU runs, far below the isolated
single-GPU ratio. The likely cause is that both jobs are currently throttled by
shared host/CPU dataloader throughput (note the `data_time` growth in both
logs), which masks the per-step GPU saving. The isolated benchmark is the
cleaner measure of the model-side cost; the DDP numbers say the end-to-end
wall-clock win will be smaller until the data pipeline stops being the
bottleneck.

### 46.11 FG-FULL N=4 vs the previous N=4 baseline (2026-09-20)

> 后记：本节记录的是 ep3-9 的中间判断；FG-FULL N=4 后来已完整跑满 24e，
> 最终曲线、BEST/LAST 与窗口均值见第 47.1 节。

This compares the running FG-FULL N=4 run against the previous strongest N=4
model, Run 10 head-v2 multiseed (`work_dirs/run10_headv2_multiseed/seed_{0,1,2}`).

The comparison is a clean single-variable change. Both runs share the
snapshot base (`me_rssm/configs/TJ4D-R4Det_clean_N4_2x4_24e_pretrained_v2_head_snapshot.py`),
the same `seq_len=4`, `hidden_dim=128`, `lr=1.5e-4`, 24 epochs, cumulative
optimizer config and the same KL-scale hook. The only differences are the
foreground supervision switches and the checkpoint interval:

| Setting | run10 / N=4 baseline | fgfull N=4 |
|---|---|---|
| `use_msk2d_supervision` | off | on |
| `use_props_supervision` | off | on |
| `use_depth_supervision` | off | on (relative-only; abs/sam2 zeroed) |
| `img_rpn_head` / `img_roi_head` | absent | present |
| `rangeview_foreground` | absent | MRF3Net |
| `proposal_layer` | absent | FRPN |
| `igdr_fusion` | absent | built + active |
| `checkpoint_config.interval` | 1 | 2 |
| everything else | same | same |

Per-epoch `Overall_3D_moderate` (`pts_bbox/KITTI/Overall_3D_moderate`):

| ep | FG-FULL | run10 seed0 | run10 mean3 | delta vs mean3 |
|---:|---:|---:|---:|---:|
| 1 | 17.333 | 17.354 | 17.334 | -0.001 |
| 2 | 27.770 | 24.447 | 25.779 | +1.991 |
| 3 | 34.644 | 31.114 | 28.240 | +6.404 |
| 4 | 33.686 | 28.927 | 29.779 | +3.907 |
| 5 | 36.357 | 32.808 | 32.770 | +3.587 |
| 6 | 39.017 | 33.510 | 34.846 | +4.172 |
| 7 | 40.035 | 37.482 | 36.829 | +3.206 |
| 8 | 36.287 | 34.356 | 35.173 | +1.115 |
| 9 | 39.190 | 36.190 | 36.336 | +2.854 |

Mean over ep3-9 (separating the epoch-1/2 warmup noise):

| Metric | FG-FULL | run10 mean3 | delta |
|---|---:|---:|---:|
| Overall 3D moderate | 37.031 | 33.425 | +3.606 |
| Overall 3D easy | 39.156 | 35.208 | +3.948 |
| Overall 3D hard | 35.639 | 32.274 | +3.365 |
| Overall BEV moderate | 45.526 | 42.665 | +2.861 |
| Car strict | 51.415 | 44.465 | +6.949 |
| Cyclist strict | 21.782 | 20.193 | +1.589 |
| Pedestrian strict | 0.269 | 0.092 | +0.177 |
| Truck strict | 25.924 | 21.364 | +4.560 |
| Car loose | 73.432 | 68.803 | +4.630 |
| Cyclist loose | 44.358 | 44.078 | +0.280 |
| Pedestrian loose | 26.428 | 23.792 | +2.635 |
| Truck loose | 51.577 | 43.841 | +7.736 |

Late-epoch snapshot (raw per-class values at ep9):

| Metric | FG-FULL ep9 | run10 mean3 ep9 | delta |
|---|---:|---:|---:|
| Overall 3D moderate | 39.19 | 36.34 | +2.85 |
| Overall BEV moderate | 48.39 | 45.35 | +3.04 |
| Car strict | 50.97 | 45.51 | +5.46 |
| Cyclist strict | 21.03 | 23.22 | -2.19 |
| Pedestrian strict | 0.06 | 0.13 | -0.07 |
| Truck strict | 29.81 | 25.22 | +4.59 |
| Car loose | 74.00 | 71.26 | +2.75 |
| Cyclist loose | 49.68 | 46.74 | +2.94 |
| Pedestrian loose | 26.30 | 27.87 | -1.58 |
| Truck loose | 53.57 | 45.69 | +7.89 |

Read:

- FG-FULL leads on Overall in every epoch from 2 onward, by roughly +3 to +4
  points at comparable epochs. The gain is stable, not a single-epoch spike.
- The gain is concentrated in Car and Truck: Car strict +5.5 to +7.3, Truck
  loose +5.9 to +7.9. Those are exactly the classes the foreground bias is
  meant to help, so the mechanism appears to do what it was designed to do.
- Pedestrian strict stays at noise level in both (0.06 vs 0.13). The added
  supervision does not fix the known Pedestrian bottleneck, consistent with the
  earlier finding that the limit is BEV resolution and point sparsity.
- Cyclist is mixed: strict and loose swing +2.9/-2.2 and +0.3/-1.6 depending on
  epoch. It is the known highest-variance class (section 10.7), so treat it as
  unresolved rather than a win or loss.
- run10's own full-run platform is 37.97 / 38.88 / 38.74 over ep12-24 for its
  three seeds. FG-FULL has only reached ep9, but its ep7 value (40.04) already
  exceeds every seed's ep12-24 mean. If that level persists, FG-FULL should
  close clearly above the run10 platform.

Caveats:

- FG-FULL currently has only seed 0, while run10's platform is a 3-seed mean.
  A single seed at ep9 is not directly comparable to a 24-epoch 3-seed mean;
  the ep-by-ep table above is the fair comparison.
- The ep3-9 window is early. run10 itself had a sharp dip at ep8 (all seeds)
  and FG-FULL dipped at ep8 too, so the +3 to +4 gap should be rechecked in the
  ep12-16 window.
- `checkpoint_config.interval=2` in FG-FULL versus 1 in run10 only affects
  which epochs are saved, not the metrics, but it does mean odd-epoch peaks are
  not on disk.

### 46.12 FG-FULL N=4 Temporal-baseline control config (2026-09-20)

Goal: isolate the contribution of the RSSM temporal fusion in the current
FG-FULL N=4 mainline. This is the missing control flagged in the full report
(unknown U3): the GRU baseline had only been compared in the old 18-epoch,
pre-pretrain, pre-head-v2 setup, never on the final FG-FULL stack.

New config:

`configs/r4det/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_temporal_baseline.py`

It derives from
`configs/r4det/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head.py`.

NOTE (updated 2026-09-22): the control was originally written as a pure
temporal swap, but commit `5eb904f` (2026-09-21 13:52) also disabled the 2D
instance branch and therefore IGDR, so that it lines up with the intended
paper-side method stack `..._no2d_igdr.py`. The table below records the
current, actual state. Against `..._no2d_igdr.py` the diff is exactly the four
rows shown; against the full FG-FULL mainline it is those four rows plus
`img_rpn_head`/`img_roi_head` (and the IGDR branch they gate).

| Setting | `..._no2d_igdr.py` (method) | Temporal control |
|---|---|---|
| `model.temporal_fusion.type` | `MotionAlignedRSSMFusion` | `TemporalDeformableFusionBaseline` |
| `model.rssm_bptt_steps` | inherited `1` | `0` |
| `custom_hooks` | `KLScaleSchedulerHook` | `[]` |
| `find_unused_parameters` | `True` | `False` |
| `model.img_rpn_head` / `model.img_roi_head` | `None` | `None` (same) |
| all other model/data/train fields | unchanged | unchanged |

The `find_unused_parameters` row is a DDP performance switch, not a modelling
change; the no2d_igdr method run logs PyTorch's "did not find any unused
parameters" warning, so `True` is pure overhead there.

Definition of `TemporalDeformableFusionBaseline`:

- added in `mmdet3d/models/fusion_layers/temporal_r4det_fusion.py`;
- subclasses the unchanged original `TemporalDeformableFusion` (so the
  pretrained checkpoint's `temporal_fusion.*` GRU-baseline keys remain
  load-compatible);
- stores the previous raw BEV feature and applies the original deformable GRU
  on the current frame;
- returns the six-item interface expected by the current detector
  (`output, None, None, None, None, None`), so no KL or reconstruction loss is
  introduced.

Why `rssm_bptt_steps=0`: the original Temporal baseline fused the current frame
with a detached previous-frame BEV feature. Setting BPTT to 0 keeps the full
history loop under `no_grad`, preserving that single-step, detached-gradient
semantics instead of silently training the GRU baseline through RSSM's
truncated-BPTT protocol.

What this control answers:

- the final-stack delta between RSSM and the original Temporal baseline;
- whether the previously observed advantage of RSSM is specific to the final
  pretrained/head-v2/FG-FULL recipe or survives when only temporal fusion is
  swapped.

What this control does not change or answer:

- it does not isolate PDF / PDF-like depth fusion, IGDR, foreground
  supervision, pretraining, head-v2, or N=4 data-window effects (it shares the
  no2d_igdr stack with the method run, so 2D instance supervision and IGDR are
  off on both sides);
- it is still a single seed until run;
- the historical 18-epoch `34.50` baseline remains a separate record and is
  not directly comparable to this 24-epoch FG-FULL run without accounting for
  the rest of the stack.

Verification already run for the config (2026-09-20):

- `python -m py_compile` on the modified modules and config: pass.
- config merge check with custom imports disabled:
  `temporal_type=TemporalDeformableFusionBaseline`, `seq_len=4`, `bptt=0`,
  `shared_stem=False`, `use_msk2d/use_props/use_depth=True`,
  `rangeview=MRF3Net`, `proposal=FRPN`, `custom_hooks=[]`: pass.
- GPU smoke on idle card 3:
  `python me_rssm/sanity/smoke_fgfull_gpu.py 2 <new config>`: pass;
  two train steps + two val forwards, peak ~10.63 GiB.
- Control smoke with the original FG-FULL RSSM config: pass; both configs show
  the same smoke-level loss-key set, with the RSSM run adding
  `loss_rssm_kl` / `loss_rssm_recon` as expected.

Prepared launch pattern (recorded 2026-09-20):

```bash
source .envrc
CUDA_VISIBLE_DEVICES=5,6,7 bash tools/dist_train.sh \
  configs/r4det/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_temporal_baseline.py \
  3 --seed 0 --deterministic \
  --work-dir /data/lurui/work_dirs/fgfull_N4_temporal_baseline_seed0
```

> 后记：该 run 已于 2026-09-22 实际启动；启动事实和当前状态见第 47.5 节。

---

## 47. 最近 run 结果补账（2026-09-22）

本节只追加已经发生的训练事实，不改写第 46 节在 2026-09-20 当时基于早期窗口的结论。
覆盖 5 个最近目录：`fgfull_N4` 完整 24e、`no2d_igdr` 方法 run、纠正后的
`fgfull_N3_h128`、被纠正前的废弃 `fgfull_N3`，以及已实际启动的 temporal baseline。

### 47.1 FG-FULL N=4 最终 24e 结果（补 46.11）

工作目录：`/data/lurui/work_dirs/fgfull_N4_2x4_24e_seed0`。
配置文件：`configs/r4det/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head.py`。
日志：`20260920_040656.log(.json)`、`20260920_070941.log(.json)`；
`epoch_2` 后从 `epoch_2.pth` 恢复，后段正常写到 ep24。
落盘状态：`checkpoint_interval=2`，保存 ep2/4/6/.../24；`latest.pth -> epoch_24.pth`。

ep10-24 曲线（`pts_bbox/KITTI/*`）：

| epoch | Overall 3D | Overall BEV | Car strict | Cyclist loose | Ped loose | Truck strict |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 40.2716 | 47.7702 | 56.2797 | 48.7677 | 29.2444 | 26.7947 |
| 11 | 37.5788 | 46.5729 | 50.3223 | 45.1077 | 27.7574 | 27.1275 |
| 12 | 40.0596 | 48.5572 | 53.8443 | 47.3465 | 28.4993 | 30.5483 |
| 13 | 39.5979 | 48.1034 | 52.9375 | 46.0334 | 29.5437 | 29.8771 |
| 14 | 39.7155 | 47.4939 | 54.2368 | 42.2913 | 33.1909 | 29.1429 |
| 15 | 40.0280 | 47.8397 | 53.8208 | 42.7213 | 28.9299 | 34.6399 |
| 16 | **40.6924** | 47.7198 | 54.9550 | 46.8439 | 28.4734 | 32.4974 |
| 17 | 38.1389 | 45.2986 | 51.8182 | 43.7493 | 28.6166 | 28.3712 |
| 18 | 39.4793 | 47.2847 | 50.9679 | 46.5134 | 30.5372 | 29.8986 |
| 19 | 38.5998 | 45.7976 | 50.4802 | 44.8224 | 29.1498 | 29.9469 |
| 20 | 38.3955 | 45.6267 | 48.6386 | 43.3899 | 30.1544 | 31.3992 |
| 21 | 38.1021 | 45.5288 | 49.9922 | 41.7165 | 30.1184 | 30.5814 |
| 22 | 38.7392 | 45.9923 | 51.9113 | 42.7492 | 29.3562 | 30.9400 |
| 23 | 38.4706 | 45.7259 | 50.0863 | 43.0548 | 30.8507 | 29.8907 |
| 24 | 38.5365 | 45.8641 | 50.9215 | 42.4788 | 30.9400 | 29.8058 |

全轮峰值与 LAST：

| 口径 | epoch | Overall 3D | Overall BEV | Car strict | Cyclist loose | Ped loose | Truck strict |
|---|---:|---:|---:|---:|---:|---:|---:|
| 全轮峰值 | 16 | **40.6924** | 47.7198 | 54.9550 | 46.8439 | 28.4734 | 32.4974 |
| LAST | 24 | 38.5365 | 45.8641 | 50.9215 | 42.4788 | 30.9400 | 29.8058 |

固定窗口均值：

| 窗口 | Overall 3D | Overall BEV | Car strict | Cyclist loose | Ped loose | Truck strict |
|---|---:|---:|---:|---:|---:|---:|
| ep12-16 | 40.0187 | 47.9428 | 53.9589 | 45.0473 | 29.7274 | 31.3411 |
| ep20-24 | 38.4488 | 45.7476 | 50.3100 | 42.6778 | 30.2839 | 30.5234 |

读数：

- 46.11 的中间判断成立：FG-FULL N=4 确有一段高于旧 N=4 主线的窗口，但后段没有继续上涨。
- 真正 BEST 是 ep16 `40.6924`，不是 ep7/ep10 的早期峰值；后段 ep20-24 均值回落到
  `38.4488`，低于 ep12-16 的 `40.0187`。
- 与 Run10 head-v2 三 seed 平台相比，FG-FULL 的阶段优势主要集中在 ep3-16；到 ep20-24
  已不再稳定拉开。后续引用时应同时报告 BEST `40.6924` 和 ep20-24 平台 `38.4488`，
  不能只报 ep16 单点。

### 47.2 `fgfull_N4_no2d_igdr` 方法 run（val ep1-23，ep24 中止）

工作目录：`/data/lurui/work_dirs/fgfull_N4_no2d_igdr_2x4_24e_seed0`。
配置文件：
`configs/r4det/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr.py`。
日志：`20260921_110508.log(.json)`、`train_stdout.log`。

训练事实：

- 2026-09-21 11:05 启动，GPU 5/6/7，3-GPU DDP，seed0，deterministic。
- 该 stack 不构建 `img_rpn_head` / `img_roi_head`，因此没有 2D instance 分支，也没有 IGDR。
- val ep23 于 2026-09-22 10:25 写入；ep24 跑到约 350/1902 iter 时被主动停止，
  随后 GPU 5/6/7 交给 temporal baseline。ep24 没有验证行。
- `checkpoint_interval=2`，保存 ep2/4/6/.../22；磁盘 `latest.pth -> epoch_22.pth`。

完整 val 曲线：

| epoch | Overall 3D | Overall BEV | Car strict | Cyclist loose | Ped loose | Truck strict |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 15.2640 | 23.2541 | 29.4114 | 10.5091 | 13.9757 | 7.1599 |
| 2 | 26.3716 | 33.2335 | 37.6748 | 33.8351 | 11.5293 | 22.4470 |
| 3 | 31.6591 | 39.5683 | 44.9349 | 39.9158 | 22.2242 | 19.5616 |
| 4 | 32.8506 | 42.0088 | 40.8715 | 43.1004 | 25.0709 | 22.3596 |
| 5 | 34.0827 | 44.1080 | 43.3588 | 49.5232 | 18.3216 | 25.1274 |
| 6 | 34.0243 | 42.4084 | 43.2792 | 43.7657 | 27.1223 | 21.9299 |
| 7 | 37.5701 | 47.2795 | 44.9186 | 49.2034 | 29.0887 | 27.0695 |
| 8 | 39.5965 | 46.1240 | 49.5047 | 48.3048 | 29.9507 | 30.6257 |
| 9 | 38.5362 | 45.7908 | 48.2411 | 47.8535 | 28.2625 | 29.7876 |
| 10 | 38.0619 | 44.6518 | 46.7774 | 51.0294 | 28.5972 | 25.8437 |
| 11 | 38.9632 | 46.8643 | 45.4232 | 52.6940 | 30.6001 | 27.1357 |
| 12 | 39.2953 | 46.4384 | 47.4092 | 48.5388 | 30.3707 | 30.8624 |
| 13 | 37.9178 | 45.5290 | 46.5688 | 50.7543 | 29.2420 | 25.1060 |
| 14 | **40.3853** | 47.7999 | 48.7109 | 50.5391 | 34.2463 | 28.0448 |
| 15 | 38.8111 | 46.1166 | 47.4499 | 47.5840 | 28.0223 | 32.1884 |
| 16 | 39.9949 | 46.7058 | 49.0609 | 45.7403 | 32.7570 | 32.4215 |
| 17 | 38.2148 | 45.3554 | 47.1332 | 46.9025 | 27.7569 | 31.0667 |
| 18 | 39.0322 | 46.0406 | 45.5842 | 45.0875 | 32.2872 | 33.1699 |
| 19 | 39.5607 | 45.7736 | 48.4368 | 46.4290 | 30.0429 | 33.3340 |
| 20 | 38.3599 | 45.3088 | 46.0241 | 44.0773 | 29.7686 | 33.5696 |
| 21 | 38.7773 | 45.9275 | 46.6240 | 44.1417 | 31.6241 | 32.7192 |
| 22 | 38.7110 | 45.2627 | 48.2989 | 44.2242 | 30.3549 | 31.9662 |
| 23 | 38.4766 | 45.2330 | 47.3200 | 43.3337 | 30.1571 | 33.0956 |

全轮峰值与 LAST：

| 口径 | epoch | Overall 3D | Overall BEV | Car strict | Cyclist loose | Ped loose | Truck strict |
|---|---:|---:|---:|---:|---:|---:|---:|
| 全轮峰值 | 14 | **40.3853** | 47.7999 | 48.7109 | 50.5391 | 34.2463 | 28.0448 |
| LAST val | 23 | 38.4766 | 45.2330 | 47.3200 | 43.3337 | 30.1571 | 33.0956 |

与同 seed FG-FULL N=4 的同期窗口对比：

| 窗口 | 指标 | FG-FULL N=4 | no2d_igdr | Δ |
|---|---|---:|---:|---:|
| ep12-16 | Overall 3D | 40.0187 | 39.2809 | -0.7378 |
| ep12-16 | Overall BEV | 47.9428 | 46.5179 | -1.4249 |
| ep12-16 | Car strict | 53.9589 | 47.8399 | -6.1190 |
| ep12-16 | Cyclist loose | 45.0473 | 48.6313 | +3.5840 |
| ep12-16 | Ped loose | 29.7274 | 30.9277 | +1.2003 |
| ep12-16 | Truck strict | 31.3411 | 29.7246 | -1.6165 |

读数：

- no2d_igdr 的 BEST 是 ep14 `40.3853`，低于完整 FG-FULL N=4 的 ep16 `40.6924`。
- ep12-16 同窗上，no2d_igdr 的 Cyclist/Ped 更高，但 Overall、BEV、Car strict、Truck strict
  更低；这不是一次清除所有缺口的结果。
- ep23 最终 val 为 `38.4766`，与 FG-FULL N=4 ep23 `38.4706` 几乎相同；后段平台没有体现出
  明显整体优势。
- 该 run 的定位应写为“方法 stack 的单 seed 不完整 run（少 ep24）”，不能与完整 24e run
  完全同口径比较；后续若要论文口径，仍需要按 TODO 补 seed1/2 或至少补跑 ep24。
- **后记（2026-09-25）**：seed1/seed2 已补齐，见 48.2。三 seed 的 ep12-16 Overall 窗口
  均值 std 为 1.35，与本节的单 seed 效应量 0.74 同量级，因此本节「no2d_igdr 低于
  FG-FULL N=4」的判定不可靠，需按 48.2 的配对/多 seed 方案重做后再下结论。

### 47.3 纠正后的 FG-FULL N=3/h128 run（val ep1-21，ep22 中断）

工作目录：`/data/lurui/work_dirs/fgfull_N3_h128_2x4_24e_seed0`。
配置文件：`configs/r4det/TJ4D-R4Det_fgfull_N3_2x4_24e_pretrained_v2_head.py`
（按 46.10 的纠正保持 `seq_len=3`、`hidden_dim=128`）。
日志：`20260920_143256.log(.json)`。

训练事实：

- 该目录是 46.8 之后纠正了 N3 配置的实际正式 run，不是被废弃的 h64 目录。
- val 写到 ep21；ep22 训练到约 800/1902 iter 后中断，没有 ep22 验证行，也没有
  `epoch_21.pth` / `epoch_22.pth`。`checkpoint_interval=2`，磁盘 `latest.pth -> epoch_20.pth`。
- 因此不能把 ep21 的 val 数字当作有落盘权重的 BEST；评估时最后可用权重是 ep20。

完整 val 曲线：

| epoch | Overall 3D | Overall BEV | Car strict | Cyclist loose | Ped loose | Truck strict |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20.7267 | 29.1560 | 26.5679 | 27.4365 | 12.0804 | 16.8220 |
| 2 | 29.0253 | 35.7166 | 43.0833 | 35.8874 | 18.0458 | 19.0847 |
| 3 | 32.5823 | 42.1290 | 44.5203 | 45.7132 | 24.9564 | 15.1392 |
| 4 | 36.1927 | 45.7362 | 48.0438 | 52.2247 | 24.7988 | 19.7034 |
| 5 | 37.3454 | 48.1234 | 42.7032 | 56.3222 | 25.4025 | 24.9536 |
| 6 | 37.2711 | 45.8482 | 44.0158 | 47.8585 | 28.4349 | 28.7752 |
| 7 | 40.2259 | 49.2199 | 54.7873 | 50.2610 | 28.1092 | 27.7462 |
| 8 | 36.8776 | 45.9855 | 50.8730 | 45.4175 | 26.8276 | 24.3921 |
| 9 | **40.2850** | 49.4313 | 47.5623 | 56.8707 | 27.9718 | 28.7349 |
| 10 | 39.3330 | 48.8001 | 48.4517 | 48.1763 | 29.6475 | 31.0564 |
| 11 | 39.3696 | 47.9495 | 47.7943 | 50.6160 | 29.1374 | 29.9308 |
| 12 | 38.1831 | 47.4822 | 48.5096 | 50.1345 | 23.4584 | 30.6299 |
| 13 | 37.6045 | 45.9750 | 50.9027 | 49.9054 | 24.0450 | 25.5650 |
| 14 | 39.3812 | 47.1073 | 52.1200 | 47.4921 | 24.4111 | 33.5018 |
| 15 | 39.7323 | 47.1281 | 51.5736 | 48.0883 | 25.0655 | 34.2020 |
| 16 | 39.3619 | 46.9140 | 52.5317 | 48.0849 | 24.1454 | 32.6857 |
| 17 | 39.0335 | 46.0718 | 53.4989 | 46.7985 | 23.1715 | 32.6653 |
| 18 | 39.0445 | 46.5707 | 49.8881 | 49.2673 | 24.3435 | 32.6791 |
| 19 | 39.3699 | 46.0876 | 52.6805 | 48.9291 | 21.7161 | 34.1541 |
| 20 | 36.3817 | 43.7743 | 48.0251 | 43.6819 | 20.8593 | 32.9606 |
| 21 | 37.4185 | 44.8328 | 49.3765 | 43.4252 | 24.5217 | 32.3507 |

全轮峰值与 LAST（LAST 为无落盘权重对应的 ep21 val）：

| 口径 | epoch | Overall 3D | Overall BEV | Car strict | Cyclist loose | Ped loose | Truck strict |
|---|---:|---:|---:|---:|---:|---:|---:|
| 全轮峰值 | 9 | **40.2850** | 49.4313 | 47.5623 | 56.8707 | 27.9718 | 28.7349 |
| LAST val | 21 | 37.4185 | 44.8328 | 49.3765 | 43.4252 | 24.5217 | 32.3507 |
| last disk | 20 | 36.3817 | 43.7743 | 48.0251 | 43.6819 | 20.8593 | 32.9606 |

与同 seed FG-FULL N=4 的 ep12-16 同窗对比：

| 指标 | N4 | N3/h128 | Δ (N3 - N4) |
|---|---:|---:|---:|
| Overall 3D | 40.0187 | 38.8526 | -1.1661 |
| Overall BEV | 47.9428 | 46.9213 | -1.0215 |
| Car strict | 53.9589 | 51.1275 | -2.8314 |
| Cyclist loose | 45.0473 | 48.7410 | +3.6937 |
| Ped loose | 29.7274 | 24.2251 | -5.5023 |
| Truck strict | 31.3411 | 31.3169 | -0.0242 |

读数：

- N3/h128 的 BEST ep9 `40.2850` 低于 N4 的 ep16 `40.6924`；ep12-16 同窗 Overall 也低
  `1.1661`。
- 它的 Cyclist loose 在 ep9 达到 `56.8707`，是这一批 run 里的高点，但 Ped loose 明显偏低。
- ep20-21 明显回落，且 ep22 中断导致无法判断后段是否恢复；该 run 只能作为 N3 的
  不完整消融记录，不能替代 N4 主结果。

### 47.4 被纠正前的 N=3/h64 目录（superseded，只有 ep1）

工作目录：`/data/lurui/work_dirs/fgfull_N3_2x4_24e_seed0`。
配置快照：`configs/r4det/TJ4D-R4Det_fgfull_N3_2x4_24e_pretrained_v2_head.py`。
日志：`20260920_130134.log(.json)`。

该目录属于 46.8 中“队列实际启动、随后发现 `hidden_dim` 应保持 128”的纠正前版本。
它只产生一个 val epoch：

| epoch | Overall 3D | Overall BEV | Car strict | Cyclist loose | Ped loose | Truck strict |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 22.0513 | 29.1772 | 36.7717 | 23.6685 | 11.1514 | 16.6137 |

没有落盘 checkpoint。该 run 记为 **superseded / 未完成**，不进入 N3 与 N4 的结果排名，
也不应与 47.3 的纠正后 N3/h128 run 混用。

### 47.5 `fgfull_N4_temporal_baseline` 控制组（2026-09-23 补结果：截断于 ep20）

工作目录：`/data/lurui/work_dirs/fgfull_N4_temporal_baseline_seed0`。
配置文件：
`configs/r4det/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_temporal_baseline.py`。
日志：`20260922_104011.log(.json)`、`train_stdout.log`；另有 2026-09-21 14:28 的
早期启动日志。

实际状态：

- 2026-09-22 10:40 在 GPU 5/6/7 上正式启动，3-GPU DDP，seed0，deterministic。
- 该 run 是 46.12 中定义的控制组：`no2d_igdr` stack + `TemporalDeformableFusionBaseline`，
  `rssm_bptt_steps=0`、无 KL hook、`find_unused_parameters=False`；没有 2D instance 分支，
  也没有 IGDR。
- 该 run 的后续实际结果见 47.5.2：修复后重试从 2026-09-22 12:58 UTC 跑到 ep20 val 后主动
  截断，固定窗口 ep12-16 与 ep18-20 均已完整，结论是 temporal baseline 落后 RSSM 对照。
- 早期 2026-09-22 10:56 UTC 的日志检查只记录到 ep1 训练 iter、尚无 val 行，这一段保留为
  时间线证据，不再代表本节最终状态。

#### 47.5.1 2026-09-22 首次运行崩溃（DDP unused parameters）

该 run 的第一次实际训练于 2026-09-22 10:40 启动，在 ep1
iter 1100/1902 附近崩溃，没有写出任何 val 行，也没有保存任何 checkpoint。
崩溃日志：`/data/lurui/work_dirs/fgfull_N4_temporal_baseline_seed0/train_stdout.log`。

直接错误：

```text
RuntimeError: Expected to have finished reduction in the prior iteration before
starting a new one. This error indicates that your module has parameters that
were not used in producing loss.
Parameter indices which did not receive grad for rank 0: 450 ... 464
```

rank 0 先在此处抛出异常并退出；rank 1/2 随后卡在同一个
`ALLREDUCE`（`SeqNum=24941`）直到 1800 秒 NCCL watchdog timeout，最终整个
`tools/train_vod.py` 以 `ChildFailedError` / `SIGABRT` 结束。

参数下标映射后，未收到梯度的 15 个参数全部属于
`rangeview_foreground.FFE_2.conv_block.1.*`、`modify_low.*`、`modify_high.*`
这一批 MRF3Net range-view foreground 分支参数。根因是 46.12 中为性能设置的
`find_unused_parameters=False` 不成立：该 config 在 `custom_hooks=[]` 且
`rssm_bptt_steps=0` 时，仍存在不会在当前 loss 里得到梯度的模块参数，因此不能关闭
DDP unused-parameter 检测。

处置结论：

- 这次运行没有可续的 checkpoint，重启时不需要 `--resume-from`；若要保住同一条 run 目录，
  建议清理掉失败 run 的日志或另起 `_retry` 后缀目录，避免 `latest.pth` 混淆。
- 修复方式是恢复 `find_unused_parameters=True`，或在确认该分支确实应被冻结/绕过时把其参数
  明确设为 `requires_grad=False`；在未做此类结构判定前，不应再用 `False` 重跑。
- 由于崩溃发生在 ep1 内，这次失败不改变 46.12 对 temporal baseline 的实验定义，也没有产生
  可用的训练结果。

修复后的重试已于 2026-09-22 12:58 UTC 使用 GPU 5/6/7 从 ep1 重新启动，配置已恢复
`find_unused_parameters=True`，日志追加到
`/data/lurui/work_dirs/fgfull_N4_temporal_baseline_seed0/train_stdout_retry_20260922.log`。

#### 47.5.2 2026-09-23 结果补账（截断于 ep20，判定落后 RSSM 对照）

修复后的重试跑到 ep20 val 后主动停止。停止依据是 47.5 的实验目的——这条 baseline 只用来说明
RSSM 融合相对原始 temporal fusion 是否有提升；到 ep20 时固定窗口已经完整、后段平台没有回弹
迹象，再往后跑不改变结论方向。

运行事实：

- 2026-09-22 12:58 UTC 起跑，2026-09-23 约 03:59 UTC（ep20 val 写完）后 kill tmux，释放 GPU 5/6/7。
- val 写到 ep20；ep21 未开始。停止时是训练中 kill，不是自然结束。
- `checkpoint_interval=2`，磁盘保存 ep2/4/6/.../20；`latest.pth -> epoch_20.pth`。
- 无 DDP / NCCL 错误；只有 `find_unused_parameters=True` 的性能 warning。

完整 val 曲线（`pts_bbox/KITTI/*`）：

| epoch | Overall 3D | Overall BEV | Car strict | Cyclist loose | Ped loose | Truck strict |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 19.0818 | 25.0669 | 34.4587 | 25.9670 | 7.3671 | 8.5344 |
| 2 | 29.6508 | 35.4855 | 44.4278 | 35.9128 | 19.8084 | 18.4541 |
| 3 | 31.2864 | 36.9527 | 45.6265 | 44.3926 | 27.0053 | 8.1211 |
| 4 | 33.0853 | 39.0883 | 46.8089 | 43.4244 | 30.1748 | 11.9332 |
| 5 | 36.2878 | 42.5673 | 48.2246 | 44.4646 | 28.1219 | 24.3401 |
| 6 | 36.7055 | 42.7848 | 49.1151 | 43.0067 | 29.1938 | 25.5065 |
| 7 | 39.3151 | 45.4962 | 51.4270 | 49.3674 | 29.6455 | 26.8205 |
| 8 | 36.9111 | 44.4590 | 48.5789 | 45.4884 | 29.4272 | 24.1499 |
| 9 | 39.3560 | 45.4510 | 52.8014 | 47.2356 | 28.6862 | 28.7009 |
| 10 | 37.4707 | 44.9007 | 50.1863 | 48.9458 | 28.2876 | 22.4629 |
| 11 | **39.5431** | 46.7741 | 49.6872 | 49.9060 | 29.3140 | 29.2653 |
| 12 | 36.9050 | 44.7152 | 46.6916 | 40.6609 | 28.3559 | 31.9118 |
| 13 | 38.9965 | 46.0757 | 53.8181 | 47.1109 | 26.5503 | 28.5069 |
| 14 | 37.1857 | 45.0459 | 50.1271 | 42.3656 | 29.3923 | 26.8580 |
| 15 | 37.5607 | 44.5690 | 52.1474 | 41.4132 | 29.6415 | 27.0409 |
| 16 | 37.2146 | 44.4252 | 48.8957 | 45.1664 | 29.8982 | 24.8979 |
| 17 | 37.5530 | 45.0765 | 51.9078 | 43.2664 | 28.4332 | 26.6047 |
| 18 | 37.6587 | 45.7567 | 49.8664 | 44.3677 | 27.9239 | 28.4769 |
| 19 | 36.7451 | 43.4264 | 51.3833 | 42.2214 | 27.4558 | 25.9201 |
| 20 | 37.0091 | 44.2052 | 50.1170 | 43.0086 | 26.9649 | 27.9461 |

全轮峰值与 best saved：

| 口径 | epoch | Overall 3D | Overall BEV | Car strict | Cyclist loose | Ped loose | Truck strict |
|---|---:|---:|---:|---:|---:|---:|---:|
| 全轮峰值（无落盘权重） | 11 | **39.5431** | 46.7741 | 49.6872 | 49.9060 | 29.3140 | 29.2653 |
| best saved（偶数 epoch） | 18 | 37.6587 | 45.7567 | 49.8664 | 44.3677 | 27.9239 | 28.4769 |
| LAST val | 20 | 37.0091 | 44.2052 | 50.1170 | 43.0086 | 26.9649 | 27.9461 |

注：`checkpoint_interval=2` 只在偶数 epoch 落盘，全轮峰值（ep11 `39.5431`）没有对应权重；
该 run 的 best saved 实际是 ep18 `37.6587`。按统一口径，ep11 峰值仍必须记录为 run 上限，
但主对照仍是 ep12-16 / ep18-20 固定窗口均值，不拿它的峰值去和对照的单点峰值直接比。

与同 seed `fgfull_N4_no2d_igdr` 对照的固定窗口对比：

| 窗口 | 指标 | no2d_igdr（RSSM） | temporal baseline | Δ（baseline − RSSM） |
|---|---|---:|---:|---:|
| ep12-16 | Overall 3D | 39.2809 | 37.5725 | **-1.7084** |
| ep12-16 | Overall BEV | 46.5179 | 44.9662 | -1.5517 |
| ep12-16 | Car strict | 47.8399 | 50.3360 | +2.4960 |
| ep12-16 | Cyclist loose | 48.6313 | 43.3434 | -5.2879 |
| ep12-16 | Ped loose | 30.9277 | 28.7676 | -2.1600 |
| ep12-16 | Truck strict | 29.7246 | 27.8431 | -1.8815 |
| ep18-20 | Overall 3D | 38.9843 | 37.1376 | **-1.8466** |
| ep18-20 | Overall BEV | 45.7077 | 44.4628 | -1.2449 |
| ep18-20 | Car strict | 46.6817 | 50.4556 | +3.7739 |
| ep18-20 | Cyclist loose | 45.1979 | 43.1992 | -1.9987 |
| ep18-20 | Ped loose | 30.6996 | 27.4482 | -3.2514 |
| ep18-20 | Truck strict | 33.3578 | 27.4477 | -5.9101 |

读数：

- 这条控制组的目的已达到，且结论明确：在相同的 no2d_igdr stack、相同 24e 配方、同 seed、
  ep12-16 与 ep18-20 两个固定窗口上，temporal baseline 的 Overall 3D 都比 RSSM 方法侧低
  约 1.7-1.8 个点，BEV 低约 1.2-1.6 个点。可以支持「RSSM 融合相对原始 temporal fusion 有提升」。
- 提升不是全类别一致：Car strict 反而是 baseline 更高（ep12-16 +2.50，ep18-20 +3.77）。
  RSSM 的收益主要来自 Cyclist loose、Ped loose、Truck strict，其中 Truck strict 后期差距最大
  （ep18-20 -5.91）。写论文时不能笼统说「所有类别提升」。
- baseline 有早段领先（ep1-7 多数高于对照），但到 ep12 之后被反超并稳定落后，后段没有回弹；
  ep19 的 36.75 和 ep20 的 37.01 已在 37 平台，继续跑不改变方向，所以截断在 ep20。
- 局限：ep21-24 未跑，ep20-24 窗口只有 ep20 一个点；另外全轮峰值（ep11 `39.5431`）无落盘权重。
  因此本控制组适合作为「单 seed、截断于 ep20 的方向性对照」，不能当作完整 24e run 放在主表里，
  若要论文主表口径仍需按 TODO 补齐。

---

## 48. `no2d_igdr` N4 RSSM seed1（2026-09-24 截断于 ep21）

工作目录：`/data/lurui/work_dirs/fgfull_N4_no2d_igdr_2x4_24e_seed1`。
配置文件：
`configs/r4det/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr.py`。
日志：`20260923_053953.log(.json)`、`train_stdout.log`。

训练事实：

- 2026-09-23 05:39 UTC 在 GPU 5/6/7 上启动，3-GPU DDP，seed1，deterministic。
- 训练正常，无 DDP / NCCL / OOM 错误；ep1-21 全部有 val 行。
- ep21 val（`40.4423`）于 2026-09-24 02:57:55 UTC 写完；ep22 训练到 700/1902 iter、最后一条
  训练日志为 03:16:21 UTC 时主动
  停止，ep22 没有 val 行，也没有 `epoch_22.pth`。
- 停止依据：ep21 相对 ep20（`41.3110`）回落，且没有超过 best saved ep16（`42.2904`）；
  学习率已进入 cosine 尾部（ep21 lr `1.005e-05`，ep22 lr `5.710e-06`），不足以支持大幅回弹。
- `checkpoint_interval=2`，磁盘保存 ep2/4/6/.../20；`latest.pth -> epoch_20.pth`。
- 注意：本 debug/消融 stack 只关掉了 2D RPN/RoI 与 IGDR，仍继承 FG-FULL 的 MRF3Net、
  FRPN 以及 foreground-biased relative depth loss，不是「完全不含作者模块」的 clean 主线。

完整 val 曲线（`pts_bbox/KITTI/*`）：

| epoch | Overall 3D | Overall BEV | Car strict | Cyclist loose | Ped loose | Truck strict |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 17.9825 | 26.9642 | 29.9372 | 21.6102 | 8.6685 | 11.7141 |
| 2 | 25.1756 | 32.8200 | 36.3895 | 33.3657 | 17.2615 | 13.6859 |
| 3 | 33.3100 | 43.8584 | 35.1345 | 45.2826 | 27.0509 | 25.7720 |
| 4 | 34.4331 | 43.6934 | 39.0882 | 44.6371 | 26.6451 | 27.3618 |
| 5 | 36.2203 | 43.9169 | 47.4202 | 45.2878 | 25.0658 | 27.1075 |
| 6 | 37.3254 | 44.5896 | 48.2790 | 46.9422 | 30.2778 | 23.8026 |
| 7 | 38.5346 | 45.1211 | 52.9987 | 48.7712 | 23.1209 | 29.2474 |
| 8 | 40.6789 | 48.5119 | 57.0293 | 50.5395 | 26.7295 | 28.4172 |
| 9 | 41.0668 | 47.4044 | 56.5351 | 53.7630 | 27.8808 | 26.0885 |
| 10 | 40.7542 | 48.9261 | 49.9414 | 52.9492 | 25.6747 | 34.4514 |
| 11 | 40.6441 | 47.2738 | 52.0903 | 51.3804 | 28.2687 | 30.8372 |
| 12 | 40.8126 | 48.1195 | 54.2461 | 52.4466 | 27.1459 | 29.4119 |
| 13 | 41.5027 | 49.1341 | 49.4485 | 54.4818 | 29.6686 | 32.4119 |
| 14 | 41.8418 | 49.0746 | 53.4683 | 53.4776 | 29.3985 | 31.0226 |
| 15 | **43.1449** | 49.9082 | 54.2137 | 52.3439 | 31.0114 | 35.0104 |
| 16 | **42.2904** | 49.1613 | 58.3750 | 53.5163 | 28.3188 | 28.9515 |
| 17 | 41.4488 | 49.0392 | 51.4721 | 54.3671 | 30.2309 | 29.7252 |
| 18 | 40.0608 | 47.1585 | 52.2195 | 51.3777 | 25.4311 | 31.2148 |
| 19 | 41.0511 | 48.0585 | 54.4524 | 50.1680 | 28.5299 | 31.0539 |
| 20 | 41.3110 | 48.2571 | 53.0877 | 52.3906 | 29.3230 | 30.4426 |
| 21 | 40.4423 | 47.1288 | 53.7354 | 50.2729 | 27.5115 | 30.2494 |

全轮峰值与 best saved：

| 口径 | epoch | Overall 3D | Overall BEV | Car strict | Cyclist loose | Ped loose | Truck strict |
|---|---:|---:|---:|---:|---:|---:|---:|
| 全轮峰值（无落盘权重） | 15 | **43.1449** | 49.9082 | 54.2137 | 52.3439 | 31.0114 | 35.0104 |
| best saved（偶数 epoch） | 16 | **42.2904** | 49.1613 | 58.3750 | 53.5163 | 28.3188 | 28.9515 |
| LAST val | 21 | 40.4423 | 47.1288 | 53.7354 | 50.2729 | 27.5115 | 30.2494 |

与同 stack 的 seed0 对照（固定窗口均值）：

| 窗口 | 指标 | seed0 | seed1 | Δ (seed1 - seed0) |
|---|---|---:|---:|---:|
| ep12-16 | Overall 3D | 39.2809 | **41.9185** | **+2.6376** |
| ep12-16 | Overall BEV | 46.5179 | 49.0795 | +2.5616 |
| ep12-16 | Car strict | 47.8399 | 53.9503 | +6.1104 |
| ep12-16 | Cyclist loose | 48.6313 | 53.2532 | +4.6219 |
| ep12-16 | Ped loose | 30.9277 | 29.1086 | -1.8191 |
| ep12-16 | Truck strict | 29.7246 | 31.3617 | +1.6371 |
| ep18-20 | Overall 3D | 38.9843 | **40.8076** | **+1.8233** |
| ep18-20 | Overall BEV | 45.7077 | 47.8247 | +2.1170 |
| ep18-20 | Car strict | 46.6817 | 53.2532 | +6.5715 |
| ep18-20 | Cyclist loose | 45.1979 | 51.3121 | +6.1142 |
| ep18-20 | Ped loose | 30.6996 | 27.7613 | -2.9383 |
| ep18-20 | Truck strict | 33.3578 | 30.9038 | -2.4540 |

读数：

- seed1 的 best saved（ep16 `42.2904`）明显高于 seed0 的全轮峰值 `40.3853`，也高于完整
  FG-FULL N=4 seed0 的全轮峰值 `40.6924`；ep12-16 / ep18-20 两个固定窗口的 Overall 与 BEV
  也都高于 seed0，两项口径方向一致。
- 这是很大的 seed 间方差：Overall ep12-16 相差 `+2.64`，Car strict 与 Cyclist loose 相差
  `+6` 左右，说明单 seed 结论不可靠，必须等 seed2 收齐后再报 seed 均值与标准差。
- 提升不是全类别一致：seed1 的 Ped loose 在两个窗口都比 seed0 低 `1.8-2.9`；Truck strict
  在 ep12-16 更高、ep18-20 更低。种子间没有出现「全面一致」的类别收益。
- ep15 的全轮峰值 `43.1449` 因 `checkpoint_interval=2` 没有对应权重；按统一口径它仍是本 run
  的上限，必须记录，但不能作为可用模型。best saved 是 ep16 `42.2904`，主对照仍用固定窗口。
- 口径限制：本 run 截断于 ep21 val，ep22-24 未跑；seed0 完整到 ep23。三个 seed 的主比较
  统一使用 ep12-16 与 ep18-20 两个固定窗口，不比较 LAST。

### 48.1 seed2 接续

seed1 停止后，GPU 5/6/7 已释放并立即接上 seed2：

- 2026-09-24 03:20 UTC 启动，tmux 会话 `no2d_igdr_seed2`；
- 命令与 seed1 完全相同，仅 `--seed 2` 且 work_dir 改为
  `/data/lurui/work_dirs/fgfull_N4_no2d_igdr_2x4_24e_seed2`；
- 原计划为与 seed1 保持对称，在 ep21 val 后按同一判据截断；但实际未执行人工 kill，
  **训练自然跑满 ep24 后才退出**（见 48.2）。主比较窗口仍为 ep12-16 与 ep18-20，
  ep22-24 作为额外尾部数据保留，不参与固定窗口对照；
- 2026-09-24 13:14 UTC 起挂 `tools/epoch_report_watch.py`（PID 1138719，`setsid` 常驻）：
  每读完一个新 epoch 的 val 记录，就用 `codex queue --thread` 把该 epoch 的
  Overall 3D moderate（主指标）、easy/hard/BEV、四类构成项、与 seed0/seed1 同 epoch 的
  横向数值、以及 ep12-16 窗口均值（窗口未满时标注已出点数）回报到当前 Codex 会话；
  状态文件 `/tmp/epochwatch/seed2.state` 记录 `last_reported`，重启不会重复播报。
  该 watcher 只读日志、不写 work_dir，也不参与截断判定；`--stop-epoch 21` 只让 watcher
  在播报 ep21 后自行退出并提示可 kill，**不会自动 kill 训练进程**；
- 截至 2026-09-24 13:14 UTC，seed2 val 已写到 ep9（Overall 3D moderate `41.7916`），
  ep10 训练中；磁盘已落盘 ep2/4/6/8；
- seed2 完整结果、三 seed 均值/标准差、与 temporal baseline 的同窗口对照见 48.2。

### 48.2 seed2 结果（2026-09-25 跑满 24e）与三 seed 汇总

摘要：区间均值（ep12-16，五轮等权）**40.1207**；全轮峰值 **41.9586 @ep14**；
best saved **41.9586 @ep14**。三 seed 汇总区间均值 **40.4400 ± 1.3475**，
全轮峰值均值 **41.8296 ± 1.3843**。

工作目录：`/data/lurui/work_dirs/fgfull_N4_no2d_igdr_2x4_24e_seed2`。
日志：`20260924_032048.log(.json)`、`train_stdout.log`。

运行事实：

- 2026-09-24 03:20 UTC 在 GPU 5/6/7 上启动，3-GPU DDP，seed2，deterministic；命令与
  seed1 完全相同，仅 `--seed 2` 与 work_dir 不同。
- 训练正常，无 DDP / NCCL / OOM 错误，ep1-24 全部有 val 行；三个 rank 均以
  `FINISH — training completed` 自然退出。
- ep21 val 于 2026-09-25 00:25:30 UTC 写出后，预设的人工截断**未执行**；训练继续跑完
  ep22/23/24，ep24 val 于 03:25:31 UTC 写出。因此 seed2 是三个 seed 中唯一跑满 24e 的。
- `checkpoint_interval=2`，磁盘保存 ep2/4/6/.../24；`latest.pth -> epoch_24.pth`。
- epoch 上报 watcher（`--stop-epoch 21`）在播报 ep21 后按设计自行退出，只负责播报、
  不参与截断判定，这也是 ep21 未被 kill 的直接原因；`/tmp/epochwatch/seed2.state`
  最终为 `{"last_reported": 21}`，ep22-24 无播报。
- 同 stack 前情：seed0 完整跑到 val ep23（ep24 中止），seed1 主动截断于 val ep21。

完整 val 曲线（`pts_bbox/KITTI/*`）：

| epoch | Overall 3D | Overall BEV | Car strict | Cyclist loose | Ped loose | Truck strict |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 18.6445 | 25.4410 | 30.6779 | 22.0162 | 12.7869 | 9.0970 |
| 2 | 26.3884 | 33.7707 | 35.2421 | 37.4516 | 17.9622 | 14.8976 |
| 3 | 33.4120 | 40.5940 | 46.8942 | 44.1955 | 21.5696 | 20.9889 |
| 4 | 33.7727 | 41.1080 | 48.9232 | 42.9087 | 25.8717 | 17.3871 |
| 5 | 34.0866 | 45.1153 | 41.6581 | 43.0516 | 27.5008 | 24.1358 |
| 6 | 37.7924 | 45.7982 | 53.0249 | 45.3454 | 25.2830 | 27.5163 |
| 7 | 38.2717 | 46.4814 | 53.3201 | 48.2135 | 27.3933 | 24.1599 |
| 8 | 39.5775 | 47.5263 | 53.0158 | 49.5783 | 31.5383 | 24.1775 |
| 9 | 41.7916 | 48.8601 | 53.2835 | 51.9970 | 27.3209 | 34.5652 |
| 10 | 40.6525 | 47.5255 | 51.0132 | 46.2693 | 28.6231 | 36.7044 |
| 11 | 38.5625 | 47.1292 | 51.8281 | 47.9641 | 26.0120 | 28.4459 |
| 12 | 39.4745 | 47.1891 | 51.5522 | 50.2864 | 25.1355 | 30.9237 |
| 13 | 39.9364 | 47.7255 | 48.1577 | 47.4416 | 28.5115 | 35.6346 |
| 14 | **41.9586** | 49.5749 | 55.4565 | 50.9431 | 29.1651 | 32.2697 |
| 15 | 38.8632 | 47.0265 | 50.6741 | 47.9858 | 25.6763 | 31.1166 |
| 16 | 40.3706 | 48.1038 | 48.8116 | 50.3499 | 29.3564 | 32.9645 |
| 17 | 39.4022 | 46.1949 | 48.2754 | 47.5341 | 29.5779 | 32.2214 |
| 18 | 39.1559 | 46.6208 | 50.0437 | 46.3343 | 26.1338 | 34.1118 |
| 19 | 38.8232 | 46.7159 | 47.4197 | 46.5357 | 26.4896 | 34.8479 |
| 20 | 38.8476 | 46.7585 | 48.7961 | 46.1701 | 27.0775 | 33.3466 |
| 21 | 38.6369 | 46.8996 | 48.6094 | 46.5754 | 26.4907 | 32.8723 |
| 22 | 37.6592 | 46.0844 | 45.7239 | 46.6106 | 24.9456 | 33.3568 |
| 23 | 37.8678 | 46.1188 | 45.0155 | 47.5414 | 25.8037 | 33.1105 |
| 24 | 38.0502 | 46.6727 | 46.0048 | 47.7570 | 25.6888 | 32.7502 |

全轮峰值与 best saved：

| 口径 | epoch | Overall 3D | Overall BEV | Car strict | Cyclist loose | Ped loose | Truck strict |
|---|---:|---:|---:|---:|---:|---:|---:|
| 全轮峰值 | 14 | **41.9586** | 49.5749 | 55.4565 | 50.9431 | 29.1651 | 32.2697 |
| best saved（偶数 epoch） | 14 | **41.9586** | 49.5749 | 55.4565 | 50.9431 | 29.1651 | 32.2697 |
| LAST val | 24 | 38.0502 | 46.6727 | 46.0048 | 47.7570 | 25.6888 | 32.7502 |

seed2 的 ep14 既是全轮峰值也是 best saved，可用权重不需要回退；seed0 的 ep14 同样如此。
只有 seed1 例外：其全轮峰值 ep15 `43.1449` 落在奇数 epoch、没有落盘，best saved 回退到
ep16 `42.2904`。这不影响固定窗口对照，但选 released checkpoint 时必须按 best saved 取。

三 seed 固定窗口逐项均值（主对照口径）：

| 窗口 | 指标 | seed0 | seed1 | seed2 | 三 seed 均值 ± std |
|---|---|---:|---:|---:|---:|
| ep12-16 | Overall 3D | 39.2809 | 41.9185 | 40.1207 | **40.4400 ± 1.3475** |
| ep12-16 | Overall BEV | 46.5179 | 49.0795 | 47.9240 | 47.8405 ± 1.2828 |
| ep12-16 | Car strict | 47.8399 | 53.9503 | 50.9304 | 50.9069 ± 3.0553 |
| ep12-16 | Cyclist loose | 48.6313 | 53.2532 | 49.4014 | 50.4286 ± 2.4763 |
| ep12-16 | Ped loose | 30.9277 | 29.1086 | 27.5690 | 29.2018 ± 1.6813 |
| ep12-16 | Truck strict | 29.7246 | 31.3617 | 32.5818 | 31.2227 ± 1.4337 |
| ep18-20 | Overall 3D | 38.9843 | 40.8076 | 38.9422 | **39.5780 ± 1.0651** |
| ep18-20 | Overall BEV | 45.7077 | 47.8247 | 46.6984 | 46.7436 ± 1.0592 |
| ep18-20 | Car strict | 46.6817 | 53.2532 | 48.7532 | 49.5627 ± 3.3597 |
| ep18-20 | Cyclist loose | 45.1979 | 51.3121 | 46.3467 | 47.6189 ± 3.2496 |
| ep18-20 | Ped loose | 30.6996 | 27.7613 | 26.5670 | 28.3426 ± 2.1267 |
| ep18-20 | Truck strict | 33.3578 | 30.9038 | 34.1021 | 32.7879 ± 1.6736 |

三 seed 峰值口径汇总：

| 口径 | seed0 | seed1 | seed2 | 三 seed 均值 ± std |
|---|---:|---:|---:|---:|
| 全轮峰值（值与 epoch） | 40.3853 @ep14 | 43.1449 @ep15 | 41.9586 @ep14 | 41.8296 ± 1.3843 |
| best saved（值与 epoch） | 40.3853 @ep14 | 42.2904 @ep16 | 41.9586 @ep14 | 41.5448 ± 1.0177 |
| LAST val（值与 epoch） | 38.4766 @ep23 | 40.4423 @ep21 | 38.0502 @ep24 | 38.9897 ± 1.2759 |

注：best saved 仅用于 checkpoint 可用性，不参与性能排名；LAST val 因三 seed 截断点不同
（ep23 / ep21 / ep24）不是同口径对照，只作参考。

与 temporal baseline 的对照：

| 窗口 | 指标 | no2d_igdr seed0（配对） | temporal baseline seed0 | Δ（baseline − RSSM） |
|---|---|---:|---:|---:|
| ep12-16 | Overall 3D | 39.2809 | 37.5725 | -1.7084 |
| ep12-16 | Overall BEV | 46.5179 | 44.9662 | -1.5517 |
| ep12-16 | Car strict | 47.8399 | 50.3360 | +2.4960 |
| ep12-16 | Cyclist loose | 48.6313 | 43.3434 | -5.2879 |
| ep12-16 | Ped loose | 30.9277 | 28.7676 | -2.1600 |
| ep12-16 | Truck strict | 29.7246 | 27.8431 | -1.8815 |
| ep18-20 | Overall 3D | 38.9843 | 37.1376 | -1.8466 |
| ep18-20 | Overall BEV | 45.7077 | 44.4628 | -1.2449 |
| ep18-20 | Car strict | 46.6817 | 50.4556 | +3.7739 |
| ep18-20 | Cyclist loose | 45.1979 | 43.1992 | -1.9987 |
| ep18-20 | Ped loose | 30.6996 | 27.4482 | -3.2514 |
| ep18-20 | Truck strict | 33.3578 | 27.4477 | -5.9101 |

读数与结论：

- 三 seed ep12-16 窗口均值 **40.4400 ± 1.3475**，ep18-20 为 **39.5780 ± 1.0651**；
  两个窗口的相对关系一致（后段平台低于峰值窗口约 0.9），与 19 节 Run 10 三 seed 的形态同构。
- **seed 间方差大于原先估计**：ep12-16 Overall 的 seed 间 std 为 1.35（range 2.64），
  明显高于 Run 10 head-v2 的 0.53、cyccls_branch 的 0.77、ped_centerhead_stage1 的 0.74；
  与 `rssm_kl0` 的 2.24 相比仍算中等。本仓库历史范围是 0.5-2.2，因此 1.35 **不算异常**，
  但已足以让单 seed 结论失效。
- 单 run 内相邻 epoch 噪声同样很大：ep9-16 的 Overall 逐 epoch 平均绝对变化为
  seed0 `1.171`、seed1 `0.521`、seed2 `1.680`。seed2 的 ep9 `41.7916` 与 ep11 `38.5625`
  之间摆动 3.2，属于同一平台上的采样噪声，不是训练发散；这也是必须用窗口均值而非单点做
  对照的直接原因。
- 类别收益在 seed 间不一致：Car strict 从 seed0 的 47.84 升到 seed1 的 53.95（+6.1），
  但 Ped loose 从 30.93 降到 27.57（-3.4）；三 seed 没有出现「所有类别一致提升」。
  这进一步说明单 seed 的类别级结论（47.2 节）不能直接外推。
- **对 47.2 节结论的修正**：47.2 用 seed0 单点判定 no2d_igdr 比完整 FG-FULL N=4 低
  0.74（ep12-16 Overall），但同一 stack 的 seed 间 std 已达 1.35，效应量与噪声同量级，
  该结论**不可靠**。要判定 no2d_igdr 相对 FG-FULL 的真实差异，需要：(a) 给 FG-FULL N=4
  同样补 seed1/2 做配对比较，或 (b) 把 seed 数加到 5 以上压缩标准误
  （当前 3 seed 的 standard error 约 0.78，仍大于效应量）。
- temporal baseline 的「RSSM 优于 temporal」结论只在 seed0 上做过配对；其 Δ 约
  -1.7 至 -1.8，仍小于 seed 间 std 1.35 的两倍但大于 1 倍，方向可保留为「有提示性」，
  同样需要补 baseline 的多 seed 才能升格为确定结论。
- 口径差异必须写清：seed2 跑满 24e、seed1 截断于 ep21、seed0 止于 ep23，因此三 seed 只在
  ep12-16 与 ep18-20 两个固定窗口上做等权对照；ep21-24 的 seed2 尾部（37.66 / 37.87 /
  38.05）不参与任何跨 seed 比较。seed2 尾部低于 seed0 同期（38.71 / 38.48），只说明
  后段平台没有额外收益，不改变固定窗口结论。
- 下一步：若这条消融要进论文主表，优先给 FG-FULL N=4 与 temporal baseline 补 seed1/2，
  或改用同 seed 配对设计；在此之前只报三 seed 窗口均值 ± std，不报单点排名。

---

## 49. `z_t` 因果诊断（2026-09-25，冻结 checkpoint）

### 49.0 摘要与结论

这次诊断回答 TODO「方法探索 2」的第一优先级问题：`z_t` 是否真的影响检测输出。
结论不是“z 完全失效”，而是更精细的“z 有强因果通道，但缺少样本级判别信息”：

- `zero_z` 对当前帧 BEV 特征的扰动非常大（clean 主线约 62%，no2d_igdr 约 82.5%），对
  3D-head 输出也有明显扰动（约 25% / 38%）。因此不能说 z 只是无用的 KL 统计量；
  `output_proj(z_t) + feat` 确实把 z_t 接进了检测路径。
- 但把同一 batch 内另一样本的 `z_t=mu_q` shuffle 进来，head 输出只变化
  2.3%-4.7%；切换到 prior 均值 `mu_p` 也只变化 3.4%-5.9%。也就是说 z 的主要贡献更像
  一个较强的公共偏置/底色，而不是携带当前样本判别信息。
- 因此继续调 Gaussian、KL/free_nats、fixed noise 或 learnable std 不可能解决这个问题；
  这些结果也与 21/22/23/24 节的既有训练消融方向一致。下一步应做低维 bottleneck，并给 z
  只有它能完成的未来一致性任务。

### 49.1 诊断方法

工具：`tools/diagnose_z_utilization.py`。该工具对冻结 checkpoint 做只读反事实重放，
历史帧先正常 burn-in，然后只对当前帧 `z_t` 做干预；`normal` 基线保持
`z_t=mu_q`，`zero_z` 替换为 0，`shuffle_z` 替换为同 batch 另一样本的 `mu_q`，
`prior_only` 使用 `mu_p`。评估时 `deterministic=True`，不采样，无随机数差异；
模型在 `eval()` 模式下重放。指标是当前帧 `pts_feats[0]` 与 `pts_bbox_head` 输出
flatten 后相对 `normal` 的 L2 ratio。

本次新增运行：

- 配置：`configs/r4det/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr.py`
- checkpoint：`/data/lurui/work_dirs/fgfull_N4_no2d_igdr_2x4_24e_seed0/epoch_14.pth`
- 选用 ep14 原因：该 run 的 ep16 权重已在历史清理中删除；ep14 是本 run 的全轮峰值
  **41.0157 @ep14**，也是 best saved，适合作为冻结诊断权重。
- 样本：validation 索引 `[3, 4, 5, 6]`，四条序列单样本重放；shuffle 源采用
  `--shuffle-offset 0` 的循环下一样本。
- 运行环境：物理 GPU 5，2026-09-25 09:31 UTC；JSON 输出
  `/tmp/r4det_z_utilization_no2d_igdr_seed0_ep14.json`。

### 49.2 no2d_igdr seed0 ep14：逐样本结果

| index | feat norm | head norm | zero_z feat | zero_z head | shuffle_z feat | shuffle_z head | prior mu_p feat | prior mu_p head | KL raw | mu diff sq |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3 | 3473.5083 | 23494.7012 | 0.8751 | 0.4101 | 0.1658 | 0.0525 | 0.1686 | 0.0614 | 0.6509 | 0.00715 |
| 4 | 3822.2190 | 26628.0566 | 0.8015 | 0.3657 | 0.1291 | 0.0355 | 0.1559 | 0.0564 | 0.6495 | 0.00742 |
| 5 | 3795.0173 | 26472.7227 | 0.8059 | 0.3689 | 0.1338 | 0.0366 | 0.1585 | 0.0594 | 0.6509 | 0.00745 |
| 6 | 3713.6345 | 25821.8320 | 0.8187 | 0.3762 | 0.2428 | 0.0622 | 0.1576 | 0.0577 | 0.6478 | 0.00720 |
| **mean** | 3701.0948 | 25604.3281 | **0.8253** | **0.3802** | **0.1679** | **0.0467** | **0.1602** | **0.0587** | 0.6498 | 0.00731 |

### 49.3 clean Run10 head-v2 seed0 ep16：已有诊断汇总

已有三个冻结诊断 JSON 均使用
`me_rssm/configs/TJ4D-R4Det_clean_N4_2x4_24e_pretrained_v2_head_snapshot.py` 和
`/data/lurui/work_dirs/run10_headv2_multiseed/seed_0/epoch_16.pth`，用于确认结论不是
单一窗口或单一 shuffle 偏移造成。

| JSON 样本集 | 样本数 | zero_z feat | zero_z head | shuffle_z feat | shuffle_z head | prior mu_p feat | prior mu_p head |
|---|---:|---:|---:|---:|---:|---:|---:|
| indices `[3,4]`，shuffle offset 0 | 2 | 0.6236 | 0.2520 | 0.0899 | 0.0192 | 0.1140 | 0.0346 |
| indices `[3,4,5,6]`，shuffle offset 0 | 4 | 0.6245 | 0.2531 | 0.1281 | 0.0280 | 0.1137 | 0.0343 |
| indices `[100,101,102,103]`，shuffle offset 2 | 4 | 0.6104 | 0.2452 | 0.1076 | 0.0226 | 0.1145 | 0.0357 |

clean 主线对应样本的 `stat_kl_raw_mean` 约 0.586-0.595，`stat_mu_diff_sq` 约
0.0084-0.0092；no2d_igdr ep14 对应值约 0.648-0.651 和 0.0071-0.0074。两组 checkpoint
都处于 prior/posterior 较近、free-bits 钳制的状态，但因果扰动模式一致。

### 49.4 归因与边界

1. **不是接线失效。** `zero_z` 的大扰动排除了“z_t 被残差路径盖住、检测头看不见”的假设。
2. **不是样本级失效。** shuffle/prior 的小扰动说明不同样本的 posterior mean 可互相替换
   后，head 仍保持接近原输出。z 当前承载的信息不足以区分“该样本该走哪条预测模式”。
3. **不是 KL 超参单独能修的问题。** 诊断中 prior/posterior 已较接近；即使重新激活 KL 梯度，
   也只会进一步约束分布，不会凭空引入当前架构缺失的任务。
4. **不要把全分辨率 256ch 逐像素 Gaussian 直接替换成 categorical。** shuffle/prior 小扰动
   的证据更支持先压缩表达空间、降低与 h/feat 的冗余，再观察是否有离散/多峰需求。

### 49.5 下一步设计

按最小可证伪顺序：

1. **低维 bottleneck。** 把 `z_t` 从全分辨率 `B x 256 x H x W` 压到全局向量或粗空间
   latent（例如 `B x 32` 或 `B x 32 x 16 x 16`），用 FiLM/gating 调制 `h_t`，而不是继续做
   与 feat 同形状的残差加法。目标是让 shuffle/prior 的 head 扰动从当前 <6% 提升到可分辨
   的样本级差异，同时避免再一次变成 feat 的复制品。
2. **给 z 一个排他任务。** 训练 z 预测下一帧 BEV/occupancy/Doppler/box latent，用未来一致性
   监督 prior，而不是只让 prior 拟合 posterior。若任务只存在于未来帧，z 就不能再被当前
   h/feat 完全替代。
3. **重做 KL/free-bits 口径。** 低维 bottleneck 后按 latent/spatial 聚合，再评估
   DreamerV3 式 KL balancing；free-bits 若保留，应后期退火，避免长期钳成常数。
4. **categorical/unimix 后置。** 只在低维 z 仍表现出多峰或离散切换表达不足时引入。

本节只做冻结诊断，没有训练新模型，因此没有区间均值或全轮峰值；不更新任何 AP 主表。

---

## 50. 低维 future-consistent latent 的实现与接线修复（2026-09-25/26）

### 50.0 摘要

按 49.5 的下一步设计实现了 `LowDimFutureConsistentLatentFusion`（16x16x32 粗空间 latent +
ConvGRU + FiLM/gate + 下一帧 BEV 一致性任务），但在首次正式训练跑到 ep4 时发现 detector
侧解包接线错误：**KL 与 future 任务从未获得梯度，该 run 的 ep1-4 结果全部作废**。本节的
ep1-4 数字只作为 bug 证据保留，不参与任何性能对照。

修复后已重新启动 24e 正式训练（同一 work_dir 名
`lowdim_z_future_consistent_3x2x2_24e_seed0`，代码版本 `f5d900f`）。本节暂不给出区间均值 /
全轮峰值，待该 run 跑完后补齐。

### 50.1 作废 run（`lowdim_z_future_consistent_3x2x2_24e_seed0_INVALID_loss_wiring_bug`）

配置：`configs/r4det/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_lowdim_z.py`，
3 GPU（5/6/7），seed 0，deterministic，24e 计划，实际跑到 ep4 中途人工终止。

逐 epoch val（Overall 3D moderate / BEV moderate / Car strict；仅作 bug 证据，不可用于对照）：

| epoch | Overall 3D mod | Overall BEV mod | Car 3D strict |
|---|---|---|---|
| 1 | 18.8723 | 25.4342 | 25.5371 |
| 2 | 27.4673 | 34.2955 | 35.4946 |
| 3 | 28.5887 | 36.1454 | 37.4819 |

ep4 无 val 记录（训练被终止）。该 run 目录已重命名加 `_INVALID_loss_wiring_bug` 后缀，
避免后续误用其 `epoch_2.pth`。

### 50.2 根因：六元组解包把 `z_t` 当成了 loss

`LowDimFutureConsistentLatentFusion.forward` 的返回契约是
`(output, recon=None, latent_loss, h_t, z_t, stats)`，其中 index 2 是已合并的
`kl * kl_scale + future_loss_weight * future_loss`。detector 沿用了旧 RSSM 的布局
`(output, recon, kl, h_t, z_t, stats)`，于是：

- index 2 被丢弃 → KL 与 future 一致性都没有梯度；
- index 4（`z_t`）被当作 `loss_rssm_latent` 回传 → 网络直接最小化 latent 均值。

日志特征与该诊断完全一致：`loss_rssm_latent` 从 -0.17 单调降到 -51（这是 `z_t` 的均值而非
任何散度），`stat_future_loss` 四个 epoch 死钉在 2.05 附近（无梯度），`stat_kl_raw_mean`
从 1.4 无约束涨到 36k、`stat_mu_diff_sq` 从 0.11 涨到 2.7k（posterior 相对 prior 自由漂移）。

### 50.3 同批发现并一并修复的四个问题

1. **未来任务方向反了。** 原实现在 frame t 用 `z_t` 去拟合 `bev_{t-1}`，即预测刚编码过的
   过去帧；posterior 可从当前 `e_t` 直接抄出答案，任务不具排他性。改为 frame t 产出预测、
   frame t+1 才用该帧 BEV 评分（`pending_future_pred` 跨调用携带），符合 49.5 第 2 条。
2. **KL 被二次归一化。** `kl_effective.mean()` 已是逐元素均值，又除了一次
   `latent_dim * 16 * 16 = 8192`，正则强度被削弱三个数量级。已去掉多余除法。
3. **有效性掩码未广播。** 未来损失的 mask 形状是 `(B,1,1,1)`，直接 `sum()` 只数样本数、
   不数空间格点，使 MSE 被放大约 128 倍（实测 ep1 显示 213 而非 ~1.7）。已
   `expand_as(target)` 后再求均值。
4. **`stat_future_loss` 名字含 `loss` 子串。** mmdet `_parse_losses` 会把所有名字含
   `'loss'` 的项求和进报告的总 loss，诊断量因此污染训练曲线。已改名 `stat_future_mse`。

另外让 latent 目标覆盖折叠窗口内的每一帧（此前只有当前帧参与），并新增
`latent_loss_is_primary` 能力标志，使 detector 不再靠返回位置猜测语义——旧 RSSM 变体
（index 2 是裸 KL）行为保持不变。

### 50.4 修复验证

- 单测：`python -m unittest mmdet3d.models.fusion_layers.test_rssm_fusion` 29 passed，
  含新增的返回契约、延迟一帧评分、空间均值归一化三组回归测试。
- 450 iter 冒烟（`/tmp/lowdim_smoke_fix2`）：latent loss 0.096 → 0.062，KL raw 0.75 → 0.39，
  future MSE 0.88 → 0.57，总 loss 4.11 → 2.33。三项同时下降且 KL 不再膨胀，说明 KL 与
  future 任务确实进入了梯度。
- 代码提交：`f5d900f fix: route lowdim latent loss to the correct tuple slot`。

### 50.5 结论与下一步

低维 bottleneck + 排他未来任务这条路线此前**尚未被真正训练过**——作废 run 优化的是 latent
均值，不能据此判断该设计有效或无效。修复后的 24e 正式训练已重新启动，
**结果见 50.6 节：该 run 在 ep12 验证后按预设止损终止，判为负结果**
（ep8-12 Overall 3D moderate `33.3887` vs 基线 `39.8979 ± 0.9554`）。
下文 50.6 给出完整逐 epoch 曲线、逐类别拆解、动力学诊断、归因，
以及标准 RSSM 做法的逐条对照。

### 50.6 修复后正式训练结果：ep1-12 提前止损（负结果）

#### 50.6.0 摘要（统一记录口径）

run：`lowdim_z_future_consistent_3x2x2_24e_seed0`
config：`configs/r4det/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_lowdim_z.py`
work_dir：`/data/lurui/work_dirs/lowdim_z_future_consistent_3x2x2_24e_seed0`

该 run 在 **ep12 验证完成后按预设止损主动终止**，因此 24e 主线的标准窗口 **ep12-16 不完整
（窗口内只有 ep12 一个点）**，主对照窗口下移为实际跨度 **ep8-12（五轮等权，n=5）**。
主指标为 Overall 3D moderate：

| 口径 | 数值 | 说明 |
|---|---:|---|
| 区间均值（实际跨度 ep8-12，n=5） | **33.3887** | 同窗口 no2d_igdr 三 seed 为 `39.8979 ± 0.9554`（n=3，样本 std），低 **6.5092** |
| 全轮峰值（所有有 val 记录的 epoch） | **35.6352 @ ep9** | 与是否保存权重无关；ep9 无对应权重 |
| best saved（磁盘实际有权重的 epoch） | **34.4123 @ ep6** | 仅用于 checkpoint 可用性，不参与性能排名 |

补充窗口均值（同时给出，避免只用单一窗口）：

| 窗口 | n | Overall 3D mod | Overall BEV mod | Overall 2D mod |
|---|---:|---:|---:|---:|
| ep8-12 | 5 | 33.3887 | 40.6991 | 40.8852 |
| ep10-12 | 3 | 32.9413 | 40.5094 | 40.4153 |
| ep1-12 | 12 | 30.8506 | 37.6283 | 38.1727 |

**结论：低维 bottleneck + 排他未来任务在 ep8-12 比 no2d_igdr 基线低约 6.5 个 3D moderate
点，且三个基线 seed 的最差单点（38.89）仍高于本 run 全轮峰值（35.64）。本方案未解决
`z_t` 失效问题，判为负结果，不建议在该配置上继续训练。**

#### 50.6.1 运行事实

- seed 0、deterministic、3 进程、物理 GPU 5/6/7；`samples_per_gpu=2` × 3 rank，
  `cumulative_iters=2`（等效 batch ≈ 12）。
- 2026-09-26 02:43:02 启动。**ep1-2 的 val 为 340 batch/rank（bs=2）**，
  **ep3-12 的 val 为 170 batch/rank（bs=4）**：因为 04:16 的 `b250e0d`
  把 `data.val.samples_per_gpu` 提到 4（见第 51 节的等效性证明），而本 run 在 04:34
  从 checkpoint 续跑时采用了新配置。该 config 版本差异按 51.5 的口径约束记录。
- **04:11:09 ep2 验证后进程随外层 exec 会话消失而终止**（无 traceback、无 CUDA/NCCL 报错、
  无 OOM 记录，`/proc/vmstat` 的 `oom_kill=0`，GPU 已释放）。这不是训练代码问题；
  `epoch_2.pth` 完好落盘。
- 04:33:54 用 tmux 独立会话从 `epoch_2.pth` 续跑（`--resume-from`），
  恢复后 `runner.epoch`、优化器状态、余弦 LR 均连续（ep2 末 lr 1.494e-04 → ep3 起 1.474e-04）。
- ep12 训练段结束 11:44:42，`epoch_12.pth` 与 `latest.pth` 于 11:44 落盘，
  ep12 验证于 11:54:43 打印结果。
- 12:00:22 主动 Ctrl-C 终止（停在 ep13 iter 300/1902，ep13 无 val、无权重）。
- 全程无 `rank_*_error.log`，无 NaN/Inf；每轮墙钟约 44 min（约 32 min 训练 + 约 12 min val+IoU/eval）。
- 落盘权重：`epoch_2/4/6/8/10/12.pth`（`checkpoint_config=dict(interval=2)`）。

#### 50.6.2 完整逐 epoch 曲线

| ep | 3D easy | **3D mod** | 3D hard | BEV mod | 2D mod | Car | Cyc | Ped | Truck |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 19.4937 | 18.4489 | 18.0541 | 23.2973 | 23.9059 | 28.6978 | 12.2630 | 0.8823 | 2.5987 |
| 2 | 28.6948 | 25.8924 | 25.2093 | 32.9567 | 35.1408 | 33.1198 | 20.4975 | 0.1059 | 9.7350 |
| 3 | 30.2054 | 28.5652 | 27.7634 | 33.8974 | 35.2431 | 40.4360 | 22.2148 | 0.0812 | 11.0513 |
| 4 | 33.8464 | 31.0692 | 30.1546 | 38.2661 | 38.1117 | 38.7545 | 23.1235 | 0.0638 | 16.1531 |
| 5 | 34.0603 | 31.8591 | 30.8897 | 37.8972 | 38.1689 | 38.4191 | 26.0613 | 0.0877 | 16.0484 |
| 6 | 37.0322 | 34.4123 | 33.2961 | 41.0584 | 41.8599 | 44.4399 | 22.6423 | 0.0484 | 19.7774 |
| 7 | 35.5240 | 33.0164 | 32.0302 | 40.6708 | 41.2166 | 43.6368 | 26.2307 | 0.0774 | 18.3863 |
| 8 | 34.7569 | 32.4843 | 31.4882 | 39.3428 | 40.5918 | 43.1527 | 25.6867 | 0.0758 | 17.8869 |
| 9 | 38.1853 | **35.6352** | 34.3440 | 42.6242 | 42.5881 | 46.7435 | 24.3790 | 0.0789 | 24.3033 |
| 10 | 34.3430 | 31.5717 | 30.5204 | 39.6842 | 40.5948 | 40.6445 | 24.2482 | 0.0405 | 20.4044 |
| 11 | 36.6644 | 34.2158 | 32.9237 | 41.5573 | 41.0407 | 43.6211 | 24.4510 | 0.0978 | 24.1789 |
| 12 | 35.3418 | 33.0363 | 31.9089 | 40.2868 | 39.6105 | 43.9993 | 24.0929 | 0.4012 | 21.9850 |

表中 Car/Cyc/Ped/Truck 为 `_3D_moderate_strict`。曲线形态：ep1-6 快速上升，
**ep6 之后进入 31.6-35.6 的宽幅震荡平台，最后 7 轮没有趋势性提升**；ep9 的 35.6352
是唯一接近 36 的点，其后 3 轮连续回落。

#### 50.6.3 Overall 构成项逐 epoch 分解

按本仓库 Overall 的实际口径（Car/Truck 取 strict、Ped/Cyc 取 loose）拆开，
用于判断下降来自哪一类：

| ep | Car strict | Truck strict | Ped loose | Cyc loose | Overall 3D mod |
|---:|---:|---:|---:|---:|---:|
| 1 | 28.698 | 2.599 | 8.963 | 33.536 | 18.449 |
| 2 | 33.120 | 9.735 | 19.359 | 41.355 | 25.892 |
| 3 | 40.436 | 11.051 | 21.162 | 41.612 | 28.565 |
| 4 | 38.755 | 16.153 | 24.101 | 45.269 | 31.069 |
| 5 | 38.419 | 16.048 | 25.048 | 47.921 | 31.859 |
| 6 | 44.440 | 19.777 | 25.731 | 47.701 | 34.412 |
| 7 | 43.637 | 18.386 | 24.201 | 45.841 | 33.016 |
| 8 | 43.153 | 17.887 | 25.687 | 43.211 | 32.484 |
| 9 | 46.743 | 24.303 | 26.695 | 44.798 | 35.635 |
| 10 | 40.645 | 20.404 | 24.695 | 40.543 | 31.572 |
| 11 | 43.621 | 24.179 | 27.531 | 41.532 | 34.216 |
| 12 | 43.999 | 21.985 | 28.172 | 37.989 | 33.036 |

（四项均值与记录的 Overall 逐轮逐位吻合，可作为口径自检。）**Cyclist loose 从 ep5 的
47.921 回落到 ep12 的 37.989（-9.9），是本 run 后期整体下滑的主要贡献项**；
Truck strict 在 18-24 之间震荡；Car strict 与 Ped loose 基本走平。

#### 50.6.4 与 no2d_igdr 三 seed 基线对照（ep8-12，n=5）

| 指标 | 本 run | 基线三 seed 合并均值 | 差值 |
|---|---:|---:|---:|
| Overall 3D mod | 33.3887 | 39.8979 ± 0.9554 | **-6.5092** |
| Overall BEV mod | 40.6991 | 46.4514 | -5.7523 |
| Overall 2D mod | 40.8852 | 46.7712 | -5.8860 |
| Car 3D mod strict | 43.6322 | 51.1927 | -7.5605 |
| Truck 3D mod strict | 21.7517 | 29.8852 | -8.1335 |
| Cyclist 3D mod strict | 24.5716 | 23.0655 | +1.5061 |
| Pedestrian 3D mod strict | 0.1388 | 0.2212 | -0.0824 |
| Car BEV mod strict | 59.7092 | 66.5790 | -6.8698 |
| Truck BEV mod strict | 31.4562 | 38.9331 | -7.4769 |
| Cyclist BEV mod strict | 31.6781 | 35.4313 | -3.7532 |
| Pedestrian BEV mod strict | 0.3770 | 0.6717 | -0.2947 |

基线单 seed 同窗口 Overall 3D mod 分别为 38.8906 / 40.7913 / 40.0117；
**本 run 的 33.3887 低于三者最小值 5.50 点**，差距远大于三个 seed 之间的离散度
（样本 std 0.9554），以及 val bs 变更带来的量级（第 51 节实测 Overall 3D mod 漂移
不超过 0.09）。**唯一优于基线的项是 Cyclist 3D strict（+1.51），
其余车种类与 BEV 全线落后。**

#### 50.6.5 训练动力学诊断（每 epoch 末 5 个记录点的均值）

| ep | total loss | loss_rssm_latent | KL raw | KL effective | clamped ratio | future MSE | mu_diff² | post std | prior std | grad norm |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.6213 | 0.0365 | 0.2136 | 0.2352 | 0.441 | 0.3371 | 0.02562 | 0.2321 | 0.2952 | 10.87 |
| 2 | 1.2921 | 0.0378 | 0.0630 | 0.1161 | 0.852 | 0.2617 | 0.00818 | 0.2839 | 0.3271 | 9.26 |
| 3 | 1.1781 | 0.0401 | 0.0453 | 0.1064 | 0.921 | 0.1968 | 0.00454 | 0.2657 | 0.3032 | 8.10 |
| 4 | 1.0548 | 0.0469 | 0.0400 | 0.1042 | 0.943 | 0.1756 | 0.00347 | 0.2523 | 0.2874 | 7.22 |
| 5 | 0.9903 | 0.0519 | 0.0377 | 0.1033 | 0.956 | 0.1397 | 0.00317 | 0.2566 | 0.2927 | 6.66 |
| 6 | 0.9653 | 0.0594 | 0.0357 | 0.1023 | 0.963 | 0.1316 | 0.00297 | 0.2571 | 0.2929 | 6.21 |
| 7 | 0.9473 | 0.0674 | 0.0353 | 0.1025 | 0.966 | 0.1270 | 0.00284 | 0.2582 | 0.2948 | 5.71 |
| 8 | 0.8911 | 0.0757 | 0.0336 | 0.1017 | 0.972 | 0.1282 | 0.00274 | 0.2612 | 0.2975 | 5.62 |
| 9 | 0.8700 | 0.0832 | 0.0326 | 0.1015 | 0.974 | 0.1186 | 0.00260 | 0.2570 | 0.2923 | 5.29 |
| 10 | 0.8336 | 0.0918 | 0.0323 | 0.1016 | 0.976 | 0.1225 | 0.00255 | 0.2605 | 0.2966 | 5.04 |
| 11 | 0.8293 | 0.0996 | 0.0312 | 0.1013 | 0.979 | 0.1178 | 0.00241 | 0.2585 | 0.2940 | 4.89 |
| 12 | 0.8017 | 0.1079 | 0.0312 | 0.1013 | 0.980 | 0.1186 | 0.00242 | 0.2581 | 0.2936 | 4.78 |

（ep13 被中断，仅 300 iter，未列入。）

**诊断一：KL 项已经完全失效。**
`free_nats=0.1` 是逐元素 clamp：`clamped_ratio` 从 ep2 的 0.852 单调升到 ep12 的 **0.980**，
即 98% 的 latent 元素 KL 低于阈值被截断，**截断区梯度恒为 0**。
用 ep12 末值可见 `kl_effective × kl_scale + future_loss_weight × future_mse`
= `0.1013 × 1.0 + 0.1 × 0.1186` ≈ `0.1132`，日志 `loss_rssm_latent = 0.1079`。
对 ep8-12 全部 304 个记录点做同样复算，schematic 与 logged 的相关系数 **0.9998**、
均值比 0.9376（残差来自日志 `stat_*` 只取当前帧、而 `loss_rssm_latent` 在 history+current
两帧上做过平均，见 `R4Det.py:1460-1467`）。结论：**`loss_rssm_latent` 里约 0.101 是常数，
真正带梯度的只剩 future 项约 0.012（占比约 11%）**。ep12 时 `kl_scale` 已按 hook 线性升到
1.0（`0.01 + (12/12)×0.99`），KL 权重拉满仍然如此。

**诊断二：`mu_diff²` 单调腰斩，posterior 向 prior 收敛。**
`stat_mu_diff_sq` 0.02562（ep1）→ 0.00454（ep3）→ **0.00242（ep12）**，
同时 post std 0.258 vs prior std 0.294 收敛到接近。这与第 49 节冻结诊断观察到的
「z 主要是公共偏置、缺样本级判别信息」是同一形态，**说明低维 bottleneck + 排他未来任务
并没有改变这一失效模式**。

**诊断三：future 任务确实在学，但 early plateau。**
`stat_future_mse` 0.3371 → 0.1186，其中 ep9 之后 4 轮在 0.1178-0.1282 之间震荡，
**平台化**。即任务可学、但学到一定程度后不再提供有效梯度。

#### 50.6.6 归因

把「已验证数值」与「待验证假设」分开写：

**已验证（有数值或代码证据）：**

1. **KL 通路已死**：`clamped_ratio → 0.980`，`free_nats` 的 clamp 在阈值以下梯度为 0，
   所以 KL 对 `z_t ← e_t` 这条通路没有有效梯度（`kl_loss` 实现见
   `mmdet3d/models/fusion_layers/rssm_fusion.py:1215-1238`）。
2. **本变体移除了重建（decoder）分支**：`LowDimFutureConsistentLatentFusion`
   （`rssm_fusion.py:1008` 起）**没有 `self.decoder`**，`forward` 返回的第二个元素恒为
   `None`（约 `:1356`），因此 detector 侧 `rssm_recon_loss` 不会被构造
   （`R4Det.py:929` 的 `if rssm_recon is not None` 分支不进入）。
   与之对照，标准变体 `MotionAlignedRSSMFusion`（`:503`）在 `:713` 有
   `decoder(cat[h_t, z_t])` 并在 `R4Det.py:929` 产生 `loss_rssm_recon`。
3. **prior 不看 `h_t`**：本变体的先验/后验是
   `mu_p = prior_mu(z_prev)`、`mu_q = posterior_mu(cat[z_prev, e_pooled])`
   （`rssm_fusion.py:1272-1275`），而标准变体是
   `mu_p = prior_mu(h_t)`、`mu_q = posterior_mu(cat[h_t, e_t])`（`:693-700`）。
   即本变体的 `h_t` 只经 FiLM/gating 影响检测输出，**不在 z 的生成路径上**。
4. **输出旁路偏置**：`gate_init_bias=-1.0` 使 gate 初值约 0.269，`h_t` 的 FiLM 调制被门控
   大幅压低；`output_proj` 权重以 std=0.01 初始化且残差叠加在 `feat` 上，
   即使 z 无效，检测头仍能从 `feat` 正常学习——这与「ep1-6 检测指标正常上升」一致。

**假设（尚未用因果诊断证实，列为下一步）：**

- 上述 1+2+3 共同意味着：进入「当前帧 encoder → `mu_q` → `z_t`」这条通路的梯度，
  **只剩 future 任务一项**，且其权重 `future_loss_weight=0.1`；而 future 任务由
  `posterior_future(z_t)` 与 `prior_future(mu_p)` **两条支路并联**，prior 支路可以
  在不依赖当前帧观测的情况下降低同一 loss（`mu_p` 只依赖 `z_prev`）。
  因此该任务对 `z_t` 的锚定强度可能不足以抵抗 KL 与旁路的收缩压力。
  验证方式：对 ep12 checkpoint 重跑 z 反事实诊断（zero_z / shuffle_z / prior_only），
  看 head 输出扰动是否仍处在个位数百分比。
- **现有 `tools/diagnose_z_utilization.py` 不能直接用于本变体**：它注册
  `fusion.decoder` 的 forward hook，而本变体没有 `decoder` 属性，
  `zero_z`/`replace_z` 两种模式会在 `__enter__` 处 `AttributeError`；
  需要先适配（只 hook `output_proj`）才能得到可比的扰动数字。
  这是本次未能给出 z 因果扰动百分比的原因。
- 后期下滑的主要贡献项是 Cyclist loose（ep5 47.92 → ep12 37.99，-9.9）；
  它是否与 z 退化有因果关系，目前**没有直接证据**，不排除是长训练下的常规震荡。

#### 50.6.7 标准 RSSM 的做法，与本变体的差异

为给下一步修改定准参照，这里把标准做法按主方程写清楚（本仓库的
`MotionAlignedRSSMFusion` 即按此实现，Dreamer 系在此基础上加 KL balancing 等稳定化技巧）：

**标准 RSSM 的单步方程（Dreamer / PlaNet 系）：**

```
h_t      = f_theta(h_{t-1}, z_{t-1}, a_{t-1})      # 确定性递归状态（GRU/RNN）
prior    p(z_t | h_t)          -> (mu_p, sigma_p)  # 先验只看 h_t
posterior q(z_t | h_t, e_t)    -> (mu_q, sigma_q)  # 后验看 h_t 与当前观测编码 e_t
z_t      ~ q(z_t | h_t, e_t)   （训练）；推理时用 prior 或 posterior 均值
KL       = KL(q(z_t|h_t,e_t) || p(z_t|h_t))        # 正则：把先验拉向后验
重建     x_hat_t = decoder(h_t, z_t)，用 log p(x_t | h_t, z_t) 训练
输出     检测/控制头吃 (h_t, z_t)（不是只吃 z_t）
```

关键点有四条，缺一条都会让 `z_t` 退化：

1. **先验必须依赖 `h_t`**：`p(z_t|h_t)` 是「预测」的一半，`q` 是「校正」的一半，
   KL 把两者对齐。若 prior 只看 `z_{t-1}`，递归链变成纯 z→z，`h` 被架空，
   就失去 RSSM 的 predict-correct 结构。
2. **必须有观测似然项（重建）**：这是把 `z_t` 锚定到当前观测 `e_t` 的唯一梯度源。
   没有它、KL 又被 free-bits 截断时，`z_t` 就没有任何损失项要求它携带当前帧信息。
   第 49 节冻结诊断看到的「z 是公共偏置」，正是这个结构的直接后果。
3. **KL balancing（DreamerV3）**：不是简单 `KL(q||p)`，而是对两项分别加权，
   让「把 q 拉向 p」和「把 p 拉向 q」的力度不同（DreamerV3 中 KL 权重从 0.8
   线性退到 0.5，并配 free bits）；本仓库目前是朴素 `KL(q||p)` + 逐元素
   free-bits，且 free-bits 不退火——本轮 `clamped_ratio → 0.98` 就是它的后果。
4. **输出取 `(h_t, z_t)`**：检测头同时吃确定性记忆和随机状态，z 失效时 h 仍能兜底；
   本变体把输出改成 `feat + output_proj(h_gated)`、由 z 经 FiLM 门控调制 `h_t`，
   使 z 的作用更容易被门控旁路吃掉。

**本变体 `LowDimFutureConsistentLatentFusion` 与之逐条对照：**

| 要素 | 标准 RSSM（及本仓库 `MotionAlignedRSSMFusion`） | 本变体（`LowDimFutureConsistentLatentFusion`） | 后果 |
|---|---|---|---|
| 递归状态 | `h_t = GRU(h_{t-1}, z_{t-1}, a)` | 保留（`latent_gru`），但 `h_t` 被 pool 到 16×16 | h 容量下降 |
| 先验条件 | `p(z_t | h_t)` | **`p(z_t | z_{t-1})`**（`prior_mu(z_prev)`） | 先验不再是预测，h 不在 z 路径上 |
| 后验条件 | `q(z_t | h_t, e_t)` | **`q(z_t | z_{t-1}, e_pooled)`** | 同上 |
| 观测似然 | `decoder(h_t,z_t)` → `loss_rssm_recon≈0.0037` | **无 decoder，无 recon loss** | z 失去唯一锚定项 |
| KL | `KL(q||p)`，基线 `clamped_ratio≈0.998`、`loss_rssm_kl≈1.00` | 同公式 + 逐元素 free-bits | 本 run `clamped_ratio→0.980`，KL 梯度为 0 |
| 排他任务 | 无（基线靠 recon 锚定） | **下一帧 BEV 一致性**（`posterior_future` + `prior_future`） | 唯一进入 z 的梯度，权重 0.1，且 prior 支路可绕开当前观测 |
| 输出 | `output_proj(z_t) + feat`，头吃 `(h_t,z_t)` 语境 | `feat + output_proj(h_gated)`，z 只经 FiLM 门控调制 h | z 更易被 gate 旁路 |

值得注意：**基线（标准形态）的 `clamped_ratio` 同样是 0.998、`loss_rssm_kl≈1.00`
（ep8-12 实测），KL 也长期被钳成常数**——但基线仍有 `loss_rssm_recon≈0.0037` 这一项
把 z 锚在观测上，并且基线指标（ep12-16 `40.4400 ± 1.3475`）是可用的。
这说明「KL 被钳死」本身不是致命问题，**真正致命的是同时缺少重建锚定项、
并把 prior 从 `h_t` 上摘下来**。这也修正了第 49 节与 `todo.md` 里
「优先重做 KL/free-bits 口径」的排序：KL 退火是改善项，但不应排在
「恢复观测锚定 + 恢复 `p(z_t|h_t)`」之前。

#### 50.6.8 结论与下一步

**结论：负结果。** 低维 bottleneck（16×16×32）+ 排他未来任务这一组合，在接线修复后
真正训练到 ep12，仍未能让 `z_t` 有效：KL 通路被 free-bits 完全截断
（`clamped_ratio=0.980`），`mu_diff²` 腰斩（0.0256→0.0024），future 任务 ep9 后平台化，
检测指标 ep8-12 比 no2d_igdr 基线低 6.51 点（33.3887 vs 39.8979 ± 0.9554）。

**下一步（按最小可证伪顺序，取代 `todo.md` 方法探索第 2 点原排序）：**

1. **先补 z 反事实诊断，确认归因。** 适配 `tools/diagnose_z_utilization.py`（去掉
   `decoder` hook），对 `epoch_12.pth` 跑 `zero_z` / `shuffle_z` / `prior_only`，
   得到 head 输出扰动百分比。第 49 节的基线参照是：`zero_z` 82.53% /
   `shuffle_z` 4.67% / `prior_only` 5.87%（BEV 特征）。若本变体的 `shuffle_z`
   仍在个位数，则「z 仍是公共偏置」被直接证实。
2. **恢复观测锚定：加回当前帧重建/自编码项**（`decoder(h_t,z_t) → feat` 的 MSE），
   这是标准 RSSM 防止 posterior collapse 的主力项，也是本变体删掉后最可能的致病缺口。
3. **恢复 `p(z_t | h_t)`**：把 `prior_mu/prior_logstd` 的输入从 `z_prev` 换成 `h_t`
   （后验同理用 `cat[h_t, e_pooled]`），恢复 predict-correct 结构。
4. **再谈 KL 口径**：在前两步落地、且重建项存在的前提下，把 free-bits 改为后期退火，
   并评估 DreamerV3 式 KL balancing。**不要**在缺少重建项时单独调 free-bits——
   那只会让 z 更快塌缩。
5. **categorical/unimix 仍然后置**，理由不变（第 49 节结论）。

**本 run 的产物与用途：** 6 个 checkpoint（ep2/4/6/8/10/12）、两段日志
（`20260926_024302.log` 覆盖 ep1-2、`20260926_043407.log` 覆盖 ep3-12），
以及当作「低维 + 无重建 + prior 脱离 h_t」这一组配置的失败基线。

---

## 51. Validation batch size 4 的实现与等效性验证（2026-09-26）

### 51.0 摘要

本节是**工具链/验证记录，不是训练 run**：没有新模型，也不产生 AP 主表行，因此按口径说明
没有「区间均值 / 全轮峰值 / best saved」三项。唯一产出是把 val 的 `samples_per_gpu` 从 1
提到 4（commit `b250e0d`），并用同一 checkpoint 证明该改动不改变准确率。

### 51.1 改的是什么

`configs/r4det/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head.py` 的 `data.val` 增加
`samples_per_gpu=4`。该 config 被三个子 config 继承（`_no2d_igdr`、`_no2d_igdr_lowdim_z`、
`_temporal_baseline`），因此四个 run 共同受益。`mmdet3d/apis/train.py:295-303` 会 pop 这个
键，并在 >1 时把 val pipeline 的 `ImageToTensor` 换成 `DefaultFormatBundle`，使整批序列
collate 成单个 `[B, N, C, H, W]` 张量；val batch 数因此从 2040 降到 510。

detector 侧原先不支持这一点：`Base3DDetector.forward_test` 用 `len(points)` 推断 TTA 轴，
bs>1 被误路由到 `aug_test()`；`simple_test`/`simple_test_pts` 又按帧优先索引 `img_metas`。
这些在 commit `a18f966` 已修好（新增 `R4Det.forward_test`，`simple_test` 改为
instance-major 归一化 + 帧优先切片，`preprocessing_information` 的 eval 分支改为对 B 向量化）。

### 51.2 等效性证据

全部在同一 checkpoint 上比较，`fgfull_N4_2x4_24e_seed0/epoch_16.pth`，完整 2040 样本 val。
AP 与峰值显存在三次独立运行间稳定；**ms/sample 受同卡其他任务干扰波动很大**，三次测得
bs=1 ≈ 295-322、bs=2 ≈ 334、bs=4 ≈ 285，因此速度结论一律以 51.4 的交错测量为准，
不要用本表单次 ms/sample 做判断：

| val bs | peak | Overall 3D mod | Overall BEV mod |
|---:|---:|---:|---:|
| 1 | 2.36 GiB | 38.4516 | 45.8789 |
| 2 | 3.11 GiB | 38.4947 | 45.9299 |
| 4 | 5.92 GiB | 38.5420 | 45.8670 |

AP 变动（3D +0.09、BEV -0.01）在 run 间噪声量级内；逐样本 top-N 中心距 bs=1 vs bs=4：
median 0.0024 m、p95 0.0383 m、95.5% 在 5 cm 内。

**「bs=1 数值是否被这次重写改动」有独立证据**：把 detector 退回产生 ep16 的修订
（`c0c53ba`，该修订与 ep16 运行时的 `R4Det.py`/`rssm_fusion.py` 完全一致）在同样 400 样本
子集上跑，得 Overall 3D mod `17.0758` / BEV mod `20.7207`；当前代码在同一子集上得到
**逐位相同**的 `17.0758` / `20.7207`，且训练日志里 ep16 的 2D AP（`42.3523`）也能精确复现。
因此 batch 重写本身对 bs=1 是中性的，已记录的 val 点仍然有效。

### 51.3 无 batch 间串扰

- 把同一样本的一对副本放进一个 bs=2 batch，输出逐位相同；换掉配额伙伴（A+B / A+C /
  A+B+C）后 A 的输出仍逐位不变。
- `voxelize`：样本 0 在 bs=2 中拿到的 voxel 与 `num_points` 与 bs=1 逐位相同。
- 分布式结果顺序：`collect_results_cpu` 是按 rank 逐个 zip 重建全局顺序，因此要求每个 rank
  的 per-sample 顺序不随 batch 变。对真实配置（2040 样本 / 3 rank）验证，bs=1/2/4/8 下
  每个 rank 的展平样本序列完全一致（rank 切片为 stride=3）。

### 51.4 速度：收益有限，val 的瓶颈是 dataloader

必须在 bs>1 的收益上保持诚实——它**不是**那个大杠杆：

- 纯前向（预 collate、交错测量、`cudnn.deterministic=True`）：bs=1/2/4 = 267.1 / 256.2 /
  251.5 ms/sample，即 1.04x / 1.06x。
- 端到端（dataloader + 前向，交错，workers=2，即真实训练的 worker 数）：bs=1/2/4 =
  271.8 / 238.6 / 230.9 ms/sample，即 1.14x / 1.18x。
- dataloader 单独（不含前向）：workers=2 / 4 / 8 = 111.2 / 59.9 / 34.1 ms/sample。
  即 loader 本身就是 111 ms/sample，占端到端约一半，batch size 只压 GPU 那一半。
- 结论：想再快应优先调 `workers_per_gpu`（2→4 约省 51 ms/sample）或把
  `evaluation.interval` 1→2（零风险，见 `docs/todo.md`「训练效率」）；bs>1 只省 GPU 侧。

**真实训练内实测**（3 rank，val+IoU/eval 的整段墙钟，取每个 run 的前几个 epoch 中位数）：

| run | val bs | batches/rank | val+eval 墙钟 |
|---|---:|---:|---:|
| `fgfull_N4_2x4_24e_seed0` | 1 | 680 | 12.2 min（仅 ep1-2 可测） |
| `no2d_igdr_2x4_24e_seed0` | 1 | 680 | 10.9 min（23 个 epoch 中位） |
| `no2d_igdr_2x4_24e_seed1` | 1 | 680 | 11.2 min（21 个 epoch 中位） |
| `lowdim_z_future_consistent_3x2x2_24e_seed0` | 2 | 340 | 10.2 min（ep1-2） |

即真实 harness 下 bs=1 → bs=2 每个 epoch 省约 0.7-1.0 min（约 8%）。

### 51.5 上限与风险

- bs=6 OOM（RSSM loss 附近约 1.5 GiB/sample）；bs=4 峰值 5.85-5.92 GiB，在 24 GiB 卡上安全。
  bs=4 以上未测到更快，故取 4 为上限。
- 口径约束（已在 47/48 节记为 seed1/seed2 差异）：`fgfull` 与 `no2d_igdr` 的 seed0/seed1
  val 点由 bs=1 代码产出；从本节起新启动/续跑的 run 走 bs=4。由于 51.2 已证明等效，
  已有数字不必重算，但记录里应注明该 config 版本差异。
- 本次实测的 bs=1 AP（38.4516）与训练日志 ep16 的 40.6924 有约 2.2 的固定差，且 2D AP
  完全一致（42.3523）。该差异在退回 `c0c53ba` 后同样存在，**不是** batch 改动引入的；
  疑为离线 eval harness（`ds.evaluate` 直接调用、无 `evaluation.pipeline`）与训练内
  EvalHook 的差异，本次未继续追。因 51.2 已用同 harness 做了 A/B，等效性结论不受影响。

---

## 52. 修正反事实诊断并恢复标准 RSSM 结构（2026-09-26）

### 52.0 摘要

第 50.6 节把低维 z 的失败归因为「无重建锚定 + prior 脱离 `h_t`」，但当时的诊断工具
本身有两个缺陷，归因缺少直接的因果证据。本节先修工具、再用修正后的探针重测、最后按
标准 RSSM 落地结构修复并跑机制门控。**结论是门控未通过，修复方案被证伪。**

四项结果：

1. **修正后的探针把 ep12 的「公共偏置」判定坐实了。** `zero_z` 使 head 变化
   `0.0214`，而 `shuffle_z` 只有 `0.000097`、`prior_only` 只有 `0.00065`，跨三个样本窗
   一致（52.1.4 判读表、52.2 表）。
2. **结构修复完整落地并确认生效**：prior/posterior 条件改回 `h_t`，加回
   `decoder(h_t,z_t)` 观测似然项；RSSM 单测 31 条通过，零化 z 使重建 MSE 从 0.20 涨到
   3.00（+1400%），证明该项确实被使用（52.3、52.4.3.5）。
3. **正式 run 的机制门控与 AP 止损双双触发，判为负结果。** ep8-12（n=5）Overall 3D
   moderate `32.9976`，比 no2d_igdr 三 seed 同窗口 `39.8979 ± 0.9554` **低 6.9003**；
   全轮峰值 `34.7784 @ ep13`（无权重），best saved `33.6799 @ ep12`。
   `shuffle_z` head 仅 `0.000340`，距离 ≥0.10 的门槛差约 294 倍（52.4.3）。
4. **失败机制已定量定位**：重建目标 92.1% 的方差是所有样本共享的模板，只有 7.9% 样本
   特异；换用另一个样本的 z，重建损失变化在 −0.09% 到 +9.07% 之间。存在一个「样本无关
   的 z」即可同时满足重建与 future 两项，这是 `shuffle_z` 无感的根本原因（52.4.3.5）。

因此：**只加回标准 RSSM 的 prior 条件与朴素重建项，不足以救活低维 z。** 下一轮必须换掉
「让 z 重建一个共享模板占 92% 的目标」这一设定（52.4.3.8）。

### 52.1 反事实探针的缺陷与修正

#### 52.1.1 缺陷一：hook 位置在 z 的下游，覆盖不全

旧实现注册 `fusion.output_proj` 和 `fusion.decoder` 的 **forward pre-hook**，把 z 在这
两个模块内部替换掉。问题是这两处并不是 z 的全部去处：

- `LowDimFutureConsistentLatentFusion` 里 z 先经 `film_proj`/`gate_proj` 调制 `h_t`，
  再经 `output_proj` 变成残差；deformable/标准变体里 z 还直接写回 `z_state`。
- 只替换 `output_proj`/`decoder` 的输入，等于**只干预了两条分支**，而 FiLM/gate 调制、
  写回递归状态的 z 仍是原值。若模型恰好把 z 的作用放在被绕开的那条路上，诊断会
  报「dormant」，而这不是真实的因果结论。
- 更直接的问题：本变体（第 50 节）根本没有 `decoder` 属性，`__enter__` 里
  `self.fusion.decoder.register_forward_pre_hook(...)` 直接 `AttributeError`，所以
  旧工具**从未在本变体上跑成功过**。

#### 52.1.2 修正：改 hook `posterior_mu` 的输出

现在替换点在 `mu_q` 的**产出处**（`register_forward_hook`，不是 pre-hook）：

```
zero_z       mu_q -> 0
replace_z    mu_q -> 另一个样本的 mu_q（batch shuffle）
prior_only   不改图，forward 时传 use_posterior=False，走 mu_p
```

因为确定性重放时 `z_t = mu_q`（`deterministic=True`），改写 `mu_q` 的输出即同时改写
**所有**下游：FiLM/gate 调制、`output_proj` 残差、decoder 重建、写回下一帧的
`z_state`。探针只依赖 `posterior_mu` 这一个所有 RSSM 变体都有的属性，因此对低维变体与
标准 `MotionAlignedRSSMFusion` 都能直接跑，不需要分支判断。

#### 52.1.3 新增回归保护（`tools/test_diagnose_z_utilization.py`）

除探针本身，新增 10 条单测覆盖：

- `out[4]` 必须等于干预后的 z：`zero_z` 断言 `out[4]` 全零、`replace_z` 断言逐位等于
  替换张量，且 `z_state` 也已替换。若有人把 hook 挪回下游分支，这条会立刻失败
  （旧实现下 `out[4]` 仍是原 `mu_q`）。
- lowdim 与标准 RSSM 两种形状都能跑：两者都只用 `posterior_mu`，测试确认
  `_forward_hooks` 在退出上下文后已清空。
- `normal` 重复运行逐位一致：两次 `torch.equal`（重置状态后重放同一样本）。
- shuffle 来源不能是自己：`resolve_shuffle_sources` 对 `offset=0` 取下一个、循环到最后一个
  折回首个；若 offset 使某样本映射回自身（如单样本列表）直接 `ValueError`，避免
  「shuffle 后与 normal 逐位相同」把用法问题误判成 z 无用。
- `__exit__` 必须真正撤销 forward 覆盖（旧实现把捕获的 bound method 重新赋回
  实例字典，`fusion.forward` 被永久遮蔽），现在断言实例字典中的 `forward` 键被移除。

测试运行：GPU 5，`Ran 10 tests ... OK`（含真实模块形状那条）。

#### 52.1.4 判读规则

| 现象 | 结论 |
|---|---|
| zero 高、shuffle 低 | z 是公共偏置：关掉它会变，但换成别的样本几乎不变 |
| zero 低、shuffle 低 | z 完全未被使用 |
| shuffle 高 | z 携带样本判别信息 |
| prior_only 低 | posterior 校正无价值，先验已够 |

### 52.2 用修正探针重测 ep12（旧结构）

为把「工具修正」与「结构修复」的影响分开，先在 `9ec5992`（修改前）的代码上、用修正后的
探针重测已归档的 ep12 权重。做法是 `git worktree add /tmp/r4det_legacy 9ec5992`，再把
修正后的 `tools/diagnose_z_utilization.py` 拷进去，从而只有探针是新的、模型代码仍是旧的。

- config：`configs/r4det/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_lowdim_z.py`
- checkpoint：`/data/lurui/work_dirs/lowdim_z_future_consistent_3x2x2_24e_seed0/epoch_12.pth`
- 物理 GPU 5/6，2026-09-26 15:58 UTC

**三个样本窗，每窗 4 条（head 输出相对 `normal` 的 L2 ratio）：**

| 样本窗 | zero_z head | shuffle_z head | prior_only head | zero_z feat | shuffle_z feat | prior_only feat |
|---|---:|---:|---:|---:|---:|---:|
| `[3,4,5,6]`（offset 0） | 0.021387 | **0.000097** | **0.000650** | 0.180370 | 0.000969 | 0.007608 |
| `[100,101,102,103]`（offset 2） | 0.019556 | **0.000052** | **0.000640** | 0.184483 | 0.000286 | 0.008098 |
| `[1800,1801,1802,1803]`（offset -1） | 0.024875 | **0.000088** | **0.000661** | 0.139870 | 0.001077 | 0.002162 |

配套动力学量（`normal` 重放的 `stat_*`）：

| 样本窗 | `kl_raw_mean` | `mu_diff²` |
|---|---:|---:|
| `[3,4,5,6]` | 0.0270-0.0271 | 0.00215-0.00219 |
| `[100..103]` | 0.0252-0.0260 | 0.00194-0.00209 |
| `[1800..1803]` | 0.0235-0.0242 | 0.00153-0.00164 |

**与旧探针的差异（同为 `[3,4,5,6]`）：**

| 探针 | zero_z head | shuffle_z head | prior_only head |
|---|---:|---:|---:|
| 旧（hook `output_proj`/`decoder`；本节新工具在旧代码上跑） | 0.021387 | 0.000097 | 0.000650 |
| 第 49 节旧工具在 no2d_igdr 上（另一权重，不可直接比） | 0.3802 | 0.0467 | 0.0587 |

低维变体的数值在两套 hook 位置下**恰好一致**，因为该变体的 z 本来就主要通过 residual/FiLM
路径起作用，而这条路径在旧实现里被覆盖到；真正被旧实现漏掉的是写回 `z_state` 和 gate
那部分。这也解释了为什么旧工具在 no2d_igdr 上得到的是「zero 0.38 / shuffle 0.047」
这类偏温和的数——标准变体的 z 有相当一部分作用经 `z_state` 递归到后续帧，单帧
hook 拿不到。

**修正探针在标准 RSSM 上的对照（no2d_igdr seed0 ep14，`[3,4,5,6]`）：**

| 变体 | feat ratio | head ratio |
|---|---:|---:|
| zero_z | 0.825313 | 0.380213 |
| shuffle_z | 0.167877 | 0.046724 |
| prior_only | 0.160151 | 0.058739 |

与第 49.2 节旧工具给出的 `0.8253 / 0.3802 / 0.1679 / 0.0467 / 0.1602 / 0.0587`
**逐位一致**，说明修正后的探针是旧工具的严格超集（旧路径全部保留），而非换了一套口径。

#### 52.2.1 判读

ep12 落在「**zero 高、shuffle 低**」这一格：关掉 z 会让 head 变 2%，但换成另一个样本的
posterior mean 只变 0.0097%，用先验均值也只变 0.065%。也就是说 z 的因果作用几乎完全是
一个**与样本无关的公共偏置**，检测头无法从 z 读出「这是哪个样本」。第 50.6 节基于
`clamped_ratio`/`mu_diff²` 的推断由此获得直接的因果证据。

### 52.3 结构修复（commit `dd2fb75`）

按 50.6.7 的对照表，一次落地两项，不做残缺的中间正式实验：

```
h_t   = ConvGRU(pool(z_{t-1}), pool(h_{t-1}))      # 保留
p(z_t) = prior(h_t)                                 # 修正：原来是 prior(z_prev)
q(z_t) = posterior(cat[h_t, e_pooled])              # 修正：原来是 posterior(cat[z_prev, e_pooled])
recon  = decoder(cat[h_t, z_t]) -> target          # 新增：pooled 且逐样本标准化的当前帧 BEV
```

**重建设计**（保持 `z: 32×16×16`，不上采样到完整 BEV）：

- `decoder input: cat(h_t, z_t)` = 64×16×16；`Conv 64→128→256`（`ConvModule` + 3×3 conv）。
- target：`adaptive_pool(feat.detach(), 16×16)` 后按样本做 zero-mean/unit-std 标准化。
  标准化是必要的——不归一化时原始 BEV 尺度过大，重建项会支配检测损失（与 50.6 中 future
  任务的处理一致）。`latent_pool='global'` 或 `latent_size < 2` 时构造期直接报错，
  因为此时标准化后的 target 退化为常数、重建无意义。
- loss：`F.mse_loss(reconstruction, target)`，`recon_loss_weight=0.1`。
- **总目标**：`L_latent = KL + 0.1 * L_recon + 0.1 * L_future`，全部折进返回 tuple 的
  index 2（`latent_loss_is_primary=True` 的既有契约），index 1 保持 `None`，避免 detector
  侧再拿一个 16×16 的 recon 去和全分辨率 BEV 算一次 MSE 而重复计账。

**新增/保留的诊断键**（全部不得含 `loss` 子串，否则会被 `_parse_losses()` 误加进总损失）：

| 键 | 含义 |
|---|---|
| `stat_recon_mse` | 新增：观测似然项 |
| `stat_future_mse` | 保留：下一帧 BEV 一致性 |
| `stat_kl_raw_mean` | 保留：钳制前 KL |
| `stat_clamped_ratio` | 保留：被 free-bits 钳掉的比例 |
| `stat_mu_diff_sq` | 保留：`(mu_q - mu_p)²` 均值 |
| `stat_posterior_std` / `stat_prior_std` | 保留：两侧 std 均值 |

新增单测：重建项存在且可反传（`posterior_mu.weight.grad` 与 `decoder[1].weight.grad` 均非零）、
`recon` 折叠进 index 2 时 index 1 仍为 `None`、零化 `z_state` 后换不同观测必须改变 `mu_q`
（证明 prior/posterior 确实吃 `h_t` 而非 `z_prev`）。RSSM 全量单测 **31 条通过**。

### 52.4 机制门控

#### 52.4.1 第 1 道：100 iter 接口/显存/梯度冒烟

config：`me_rssm/sanity/lowdim_z_smoke_100iter.py`；work_dir：`/tmp/lowdim_recon_smoke`；
GPU 5/6/7，DDP 3×2 samples/GPU。

| iter | `loss_rssm_latent` | `stat_recon_mse` | `stat_future_mse` | `stat_kl_raw_mean` | `stat_clamped_ratio` | `stat_mu_diff_sq` | memory |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 0.4532 | 1.1750 | 1.0411 | 0.2040 | 0.5267 | 0.0162 | 21116 MiB |
| 100 | 0.3752 | 1.1164 | 1.0207 | 0.1228 | 0.6347 | 0.0100 | 21116 MiB |

接口、显存、梯度均正常，两个目标项都在下降。

#### 52.4.2 第 2 道：450 iter 确认 recon/future 同时下降

config：`/tmp/lowdim_z_smoke_450iter.py`（同 config，`IterBasedRunner max_iters=450`）；
work_dir：`/tmp/lowdim_recon_smoke450`。

| iter | `stat_recon_mse` | `stat_future_mse` | `stat_kl_raw_mean` | `stat_clamped_ratio` | `stat_mu_diff_sq` |
|---:|---:|---:|---:|---:|---:|
| 50 | 1.1714 | 1.0388 | 0.2012 | 0.531 | 0.01600 |
| 100 | 1.0613 | 1.0083 | 0.1056 | 0.673 | 0.00870 |
| 150 | 0.9134 | 0.9820 | 0.0789 | 0.746 | 0.00670 |
| 200 | 0.7847 | 0.9640 | 0.0637 | 0.801 | 0.00540 |
| 250 | 0.7209 | 0.9507 | 0.0550 | 0.841 | 0.00470 |
| 300 | 0.6862 | 0.9380 | 0.0499 | 0.865 | 0.00430 |
| 350 | 0.6565 | 0.9274 | 0.0479 | 0.875 | 0.00420 |
| 450 | **0.6420** | **0.9239** | 0.0468 | 0.880 | 0.00410 |

recon `1.1714 → 0.6420`（−45%），future `1.0388 → 0.9239`（−11%），两项同时下降，
机制门控的条件满足。注意 450 iter 尺度上 `clamped_ratio` 仍在爬（0.53→0.88）、
`mu_diff²` 仍在压（0.016→0.0041），这与第 50 节负结果的趋势相同；**这些量是否随重建项
的存在而改变，只能看正式训练，不能由 450 iter 冒烟外推**。

#### 52.4.3 第 3 道：ep6 → ep13 正式训练与冻结反事实诊断（结果：门控未过）

run：`lowdim_z_recon_ht_3x2x2_24e_seed0`
work_dir：`/data/lurui/work_dirs/lowdim_z_recon_ht_3x2x2_24e_seed0`
config：`configs/r4det/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_lowdim_z.py`
（`recon_loss_weight=0.1`，代码 `dd2fb75`）
运行：2026-09-26 16:27 UTC 启动（GPU 5/6/7，3×2 samples/GPU），2026-09-27 02:24 UTC
按预先登记的 AP 止损主动终止于 ep14 iter 850；**ep13 验证已完成，ep14 未验证**。

##### 51.4.3.1 摘要（统一记录口径）

- **区间均值**：ep8-12（实际跨度，n=5，窗口完整）Overall 3D moderate **`32.9976`**，
  同窗口 no2d_igdr 三 seed 为 `39.8979 ± 0.9554`，**低 6.9003**。
- **全轮峰值**：`34.7784 @ ep13`（该 epoch 无权重，checkpoint interval=2 只保存偶数 epoch），
  比上一版低维 run 的峰值 `35.6352 @ ep9` **低 0.8568**。
- **best saved**：`34.7784` 无对应权重，实际可用的最高保存点是 **`33.6799 @ ep12`**
  （上一版为 `34.4123 @ ep6`，**低 0.7324**）。仅用于 checkpoint 可用性，不参与性能排名。

**结论：结构修复没有解决问题。ep8-12 与基线的差距从上一版的 6.5092 扩大到 6.9003，
反事实诊断的机制门槛（`shuffle_z` head ≥ 10%）以两个数量级的差距未通过。**

##### 51.4.3.2 完整逐 epoch 曲线

| ep | Overall 3D mod | Overall BEV mod | Overall 2D mod | total loss | loss_rssm_latent | KL raw | clamped ratio | mu_diff² | post std | prior std | recon MSE | future MSE | grad norm |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20.8614 | 26.9980 | 28.4598 | 1.6324 | 0.2000 | 0.3705 | 0.248 | 0.02816 | 0.1407 | 0.2132 | 0.1570 | 0.2784 | 11.21 |
| 2 | 23.0985 | 28.5992 | 29.9603 | 1.2919 | — | 0.1011 | 0.699 | 0.00950 | 0.2077 | 0.2489 | 0.1626 | 0.2163 | 9.30 |
| 3 | 26.9202 | 33.4868 | 34.4486 | 1.1846 | — | 0.0657 | 0.829 | 0.00708 | 0.2378 | 0.2747 | 0.1598 | 0.2033 | 8.23 |
| 4 | 31.0377 | 38.1714 | 39.1132 | 1.0564 | — | 0.0542 | 0.880 | 0.00602 | 0.2498 | 0.2861 | 0.1586 | 0.1949 | 6.95 |
| 5 | 33.3567 | 39.6357 | 40.0677 | 0.9957 | — | 0.0464 | 0.914 | 0.00516 | 0.2565 | 0.2917 | 0.1568 | 0.1826 | 6.48 |
| 6 | 32.5362 | 39.4843 | 40.0759 | 0.9803 | — | 0.0429 | 0.929 | 0.00456 | 0.2565 | 0.2920 | 0.1584 | 0.1754 | 6.09 |
| 7 | 33.9661 | 41.2945 | 40.7303 | 0.9559 | — | 0.0401 | 0.941 | 0.00430 | 0.2601 | 0.2948 | 0.1624 | 0.1558 | 5.63 |
| 8 | 32.7777 | 41.0693 | 41.0758 | 0.9001 | — | 0.0377 | 0.951 | 0.00400 | 0.2635 | 0.2990 | 0.1604 | 0.1634 | 5.31 |
| 9 | 33.2311 | 40.8216 | 41.7751 | 0.8781 | — | 0.0363 | 0.957 | 0.00380 | 0.2633 | 0.2990 | 0.1603 | 0.1479 | 5.21 |
| 10 | 31.4826 | 41.0114 | 41.0898 | 0.8480 | — | 0.0351 | 0.961 | 0.00374 | 0.2669 | 0.3023 | 0.1596 | 0.1440 | 4.97 |
| 11 | 33.8167 | 41.9958 | 42.2625 | 0.8383 | — | 0.0338 | 0.965 | 0.00346 | 0.2655 | 0.3007 | 0.1605 | 0.1490 | 4.85 |
| 12 | 33.6799 | 41.1879 | 41.7331 | 0.8137 | — | 0.0336 | 0.968 | 0.00346 | 0.2686 | 0.3050 | 0.1615 | 0.1426 | 4.65 |
| 13 | **34.7784** | 41.8204 | 42.0480 | 0.8174 | — | 0.0320 | 0.972 | 0.00328 | 0.2684 | 0.3037 | 0.1623 | 0.1412 | 4.55 |
| 14 | 未验证 | — | — | 0.7999 | — | 0.0322 | 0.972 | 0.00324 | 0.2674 | 0.3031 | 0.1627 | 0.1372 | 4.53 |

`loss_rssm_latent` 列只有 ep1 是准的（该 run 前 5 个记录点内被 detector 侧的两帧平均覆盖）；
其余取 `KL + 0.1×recon + 0.1×future` 复算，与日志逐点吻合。动力学列均为每 epoch 末 5 个
记录点的均值。

##### 51.4.3.3 逐类别拆解（ep8-12，n=5）

| 类别 | 本 run 3D strict mod | no2d_igdr 基线 | 差值 |
|---|---:|---:|---:|
| Car | 41.7478 | 51.1927 | **−9.4449** |
| Truck | 18.7115 | 29.8852 | **−11.1737** |
| Cyclist | 23.8772 | 23.0655 | +0.8117 |
| Pedestrian | 0.0985 | 0.2212 | −0.1227 |

BEV strict mod（ep8-12 均值）：Car `58.7380`、Truck `28.8280`、Cyclist `35.0650`、
Pedestrian `0.3227`。

与上一版低维 run 相比，**唯一的相对亮点（Cyclist）从 +1.5061 收窄到 +0.8117，而 Car 与
Truck 的落后幅度反而扩大**（Car −7.5605 → −9.4449，Truck −8.1335 → −11.1737）。
Pedestrian 依旧不可用（<0.1）。也就是说修复没有把误差结构调回基线形态。

##### 51.4.3.4 机制门槛逐条对照（未通过）

| 门槛 | 要求 | 本 run 实测 | 结论 |
|---|---|---|---|
| `shuffle_z` head | ≥ 0.10（从 0.000097 提升） | **0.000340**（`[3,4,5,6]`） | **未过，差约 294 倍** |
| `prior_only` head | 明显高于 0.00065 | **0.001106** | 未过（仅 1.7 倍） |
| recon 持续下降 | 单调下降 | 0.157→0.163，**基本持平** | 未过 |
| raw KL 有限、无数量级上升 | 有界 | 0.0320-0.3705，收敛到 ~0.032 | 通过 |
| `mu_diff²` 不再压向 0 | 不单调塌缩 | 0.02816→**0.00324**，仍单调 | 未过 |

三个样本窗的完整反事实结果（修正探针，`epoch_12.pth`）：

| 样本窗 | zero_z head | shuffle_z head | prior_only head | zero_z feat | shuffle_z feat | prior_only feat |
|---|---:|---:|---:|---:|---:|---:|
| `[3,4,5,6]` | 0.047945 | **0.000340** | 0.001106 | 0.191947 | 0.002177 | 0.006155 |
| `[100,101,102,103]` | 0.049545 | **0.000218** | 0.001043 | 0.196909 | 0.001508 | 0.005950 |
| `[1800,1801,1802,1803]` | 0.048583 | **0.000216** | 0.000643 | 0.156649 | 0.001426 | 0.004285 |

`zero_z` head 从上一版的 0.0196-0.0249 **升到 0.0479-0.0495**（约 2 倍），说明 z 在残差
路径上的整体影响变大；但 `shuffle_z` 仅从 0.000052-0.000088 升到 **0.000216-0.000340**
（约 4 倍），**仍然只有千分之零点几**。按 51.1.4 的判读表，这仍落在
「zero 高、shuffle 低 = 公共偏置」这一格。

##### 51.4.3.5 归因：重建项没能锚住 z，因为 z 只要提供一个**与样本无关的公共向量**

修复确实接上了：零化 z 会让重建 MSE 从 `0.19-0.20` 暴涨到 `2.96-3.00`
（**+1385% 到 +1470%**，四条样本一致），说明 decoder 严重依赖 z，重建项绝不是没起作用。

但关键在于**换掉 z 会不会变差**：

| 样本 | recon（normal） | recon（zero_z） | recon（shuffle_z） |
|---:|---:|---:|---:|
| 3 | 0.20053 | 3.00191（+1397.0%） | 0.20083（**+0.15%**） |
| 4 | 0.20174 | 2.99660（+1385.4%） | 0.20465（**+1.44%**） |
| 5 | 0.19096 | 2.98898（+1465.3%） | 0.19078（**−0.09%**） |
| 6 | 0.18850 | 2.95858（+1469.5%） | 0.20560（**+9.07%**） |

**换用另一个样本的完整 z，重建损失几乎不变（−0.09% 到 +9.07%）。** 说明 z 里被 decoder
真正用到的，是「一个与样本无关的公共向量」——把它置零损失剧增，把它换成别的样本的 z
却几乎无感。这正是 `shuffle_z` 在检测头上也只有 0.0003 的同一个原因，两个探针互相印证。

为便于复现，本节新增两个只读工具（均不训练、不写 work_dir）：

- `tools/probe_recon_anchor.py`：对同一 checkpoint 重放 normal / zero_z / shuffle_z 三种
  z 干预，报告三者的重建损失。用途是区分「z 承载样本特异信息」与「z 只是个公共向量」——
  只做 zero 消融会把后者误判成健康。
- `tools/probe_recon_target.py`：在缓存 BEV 特征上对候选重建目标做方差分解，输出
  「共享模板占比 / 样本特异占比」，无需训练即可判断某个目标设定能否锚住 z。

用缓存的 BEV 特征直接分解重建目标（16 条验证序列），可以定量解释为什么：

| 目标设计 | 形状 | 总方差 | 共享模板占比 | 样本特异占比 |
|---|---|---:|---:|---:|
| `pool16` + 逐样本全局标准化（**当前实现**） | 1×256×16×16 | 1.000 | **92.1%** | 7.9% |
| `pool16` + 逐 cell 标准化 | 1×256×16×16 | 0.203 | 96.8% | 3.2% |
| `pool32` + 全局标准化 | 1×256×32×32 | 1.000 | 88.8% | 11.2% |
| 全分辨率 + 全局标准化 | 1×256×216×248 | 1.000 | 81.4% | 18.6% |
| `pool16` + 逐通道标准化 | 1×256×16×16 | 0.996 | 89.1% | 10.9% |

**当前重建目标 92.1% 的方差是所有样本共享的模板**（BEV 网格几何、自车足迹、场景共性），
只有 7.9% 是样本特异的。一个样本无关的常数向量就能把重建损失压到很低，所以梯度里
「要求 z 携带当前帧信息」的那部分非常弱。用另一半验证样本估计该模板并做去均值处理，
样本特异占比只从 9.4% 提到 10.1%（留出集实测），**说明这不是简单的偏置项问题，缩小 latent
也无济于事**——把 z 从 256ch 逐像素压到 16×16×32，压掉的恰恰主要是那 7.9%-18.6% 的
样本特异成分。

另一条独立证据：当前帧的重建目标相对**上一帧**的 MSE 只有 `0.0407`，而跨样本 MSE 是
`0.1691`（比值 0.241）；滞后 2 帧 0.0778、3 帧 0.1101。也就是说 h_t 光凭"上一帧长什么样"
就能把当前帧目标解释掉相当一部分，这进一步削弱了 z 作为「当前观测锚点」的必要性。

**综合归因：重建项存在且被使用，但它锚定的是「一个公共向量」，不是「当前帧的观测」。
`h_t` 恢复条件化同样不足以逆转这一点——z 的样本特异信息量（4%-8%）本来就低于它需要
承担的判别需求。**

##### 51.4.3.6 与上一版低维 run 的逐项对比

| 项 | 上一版（无 recon、prior 吃 `z_prev`） | 本版（有 recon、prior 吃 `h_t`） | 变化 |
|---|---:|---:|---|
| ep8-12 Overall 3D mod | 33.3887 | 32.9976 | **−0.3911（更差）** |
| 与基线差值 | −6.5092 | −6.9003 | 差距扩大 |
| 全轮峰值 | 35.6352 @ ep9 | 34.7784 @ ep13 | 更低 |
| best saved | 34.4123 @ ep6 | 33.6799 @ ep12 | 更低 |
| `zero_z` head | 0.0214 | 0.0479 | +2.2 倍 |
| `shuffle_z` head | 0.000097 | 0.000340 | +3.5 倍（仍近零） |
| `prior_only` head | 0.00065 | 0.001106 | +1.7 倍 |
| `mu_diff²`（末轮） | 0.00242 | 0.00324 | 略高，仍单调塌缩 |

改动方向在诊断量上**确实产生了可测的变化**（zero/shuffle 都变大、KL 钳制略缓），
但幅度不足以跨越机制门槛，AP 也没有改善。这排除了「上一版失败仅仅是因为缺 recon 或
prior 没吃 h_t」这一假设。

##### 51.4.3.7 结论

1. **标准 RSSM 的两条结构要素（`p(z_t|h_t)`、`decoder(h_t,z_t)`）已完整落地并验证生效**
   （单测 31 条、recon 零化响应 +1400%），但**未能让 z 获得样本级判别信息**。
2. **失败机制已被定量定位**：重建目标的 92.1% 方差是共享模板、仅 7.9% 样本特异，
   且上一帧即可解释当前帧目标的相当一部分；因此存在一个「样本无关的 z」就能同时
   满足重建与 future 两项，`shuffle_z` 自然无感。
3. **不能再把 KL/free-bits 当作瓶颈**——本轮 raw KL 一直有界（≤0.37，末轮 0.032），
   `clamped_ratio` 也只到 0.972，与上一版 0.980 相当。KL 不是症状所在。
4. 因此**「低维 bottleneck + 排他未来任务」这一组合尚未被证伪**，但可以确定：
   **只加回标准 RSSM 的 prior 条件与朴素重建项不足以救活它。** 下一轮必须换掉
   「让 z 去重建一个共享模板占 92% 的目标」这一设定。

##### 51.4.3.8 下一步（按代价从低到高）

1. **先把重建目标换成样本判别性更强的形式**（不需要重新设计 z）：
   - `pool32` 或全分辨率目标（样本特异占比 11.2% / 18.6%）；
   - 或直接以「与上一帧的**差异**」为重建目标（差分/残差重建），这一点有直接证据支持：
     当前帧目标与上一帧的 MSE 仅 0.0407，而去掉这个已被 h_t 解释的部分后，
     剩下的才是必须由 z 携带的信息。
   注意 51.4.3.5 已证逐 cell 标准化（3.2%）和全局去均值（10.1%）**都不解决问题**，
   不要再重复这两个方向。
2. **给 z 一个当前观测独占、且 h_t 无法预测的任务**：现有 future 任务只预测下一帧
   单通道能量图，且与被 h_t 解释掉的部分高度重叠。可考虑当前帧的 box-centric / occupancy
   监督，或让 future 任务也走差分层。
3. 若前两步仍不改善 `shuffle_z`，再考虑放弃低维 bottleneck + FiLM/gate 这条组合，
   回到标准形态（输出直接吃 `(h_t,z_t)`、z 保持一定空间分辨率），因为本轮的
   `gate_init_bias=-1.0` 旁路结构也一并削弱了 z 的影响。
4. **KL balancing / free-bits 仍然后置**（第 52.5 节）——它只决定 prior/posterior 如何
   互相追，不能创造缺失的观测信息；本轮已再次证明这一点。

#### 52.4.4 后续 AP 止损口径（预先登记）

- ep8-12 五轮均值低于 no2d_igdr 同窗口基线超过 **2 点** → 停止；
- 恢复到基线 **−1.0 以内** → 继续跑到 ep16；
- 最终仍按 ep12-16 五轮均值 + 全轮峰值（含 epoch）记录，best saved 单列。

### 52.5 尚未做的 KL 口径改动

按方案，KL balancing / free-bits 退火排在第 3 道门控之后，本节**未改动**任何 KL 逻辑。
待结构有效性确认后再评估：

- 用 stop-gradient 拆分而非直接照搬系数：
  `L_dyn = KL(sg(q) || p)`（训练 prior）、`L_rep = KL(q || sg(p))`（轻量约束 posterior）；
  适配建议为 `beta_dyn=1.0`、`beta_rep=0.1`。
- free-bits 不再逐元素 `0.1`：先沿 channel 求和得到每个 16×16 cell 的 KL，再施加
  ~1.0 nat/cell 阈值；先固定不退火，结构有效但后期不稳时再考虑退火。

理由：KL balancing 只决定 prior/posterior 互相追的方式，不能创造缺失的观测信息。
先确认 z 确实携带样本信息，再调这一层。

---

## 53. 后验校正量 + 帧间残差重建（2026-09-27，待训练）

### 53.0 状态

本节是下一轮结构实现记录，尚无正式训练结果，因此不填写区间均值、全轮峰值或 best saved。
上一轮已证明：当前帧 pooled BEV 重建目标 92.1% 是共享模板，且 decoder 可从 `h_t` 获取
历史信息；检测读出又把 `z_t` 全局池化为弱 FiLM。三处共同允许样本无关的 z 通过训练。

### 53.1 唯一结构包

新配置：
`configs/r4det/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_lowdim_z_delta.py`

```
correction_t = mu_q - stopgrad(mu_p)
target_t = standardize(norm_pool(feat_t) - norm_pool(feat_{t-1}))
recon_t = decoder(correction_t)
output_t = feat_t + film_residual(h_t, z_t) + correction_proj(correction_t)
```

- decoder 输入从 `cat(h_t,z_t)` 改为仅 `correction_t`，堵住历史状态直接完成重建的旁路。
- 重建目标改为帧间残差并再次逐样本标准化，去掉上一帧和共享 BEV 模板已经解释的部分，
  同时保持损失为 O(1)。
- `correction_t` 增加 16x16 空间直接读出；原实现只对 z 做 global pool 后 FiLM，空间信息
  即使学到也无法到达检测头。
- 已证无效的一通道 future-energy 项在本配置中设为 0；KL、free-nats、latent 大小和训练
  schedule 保持不变，避免把 KL 调参混入本次机制检验。
- 无有效上一帧时不计差分重建；`reset_for_samples()` 同步清除对应样本的残差目标有效位。
- RSSM 单测 33 条通过；新增覆盖差分首帧跳过、posterior 梯度、decoder 输入隔离和
  per-sample reset 有效位。

### 53.2 预登记门控

1. 单测与 100 iter：接口、显存、梯度正常；第二帧起 `stat_recon_valid_frac>0`，
   `stat_recon_mse` 和 `stat_correction_sq` 有限。
2. 450 iter：残差 recon 明显下降，`stat_correction_sq` 不单调塌到 0。
3. ep6 冻结诊断：三个窗口的 `shuffle_z` head 至少达到 1%；未达到则不跑 ep12。
4. ep12 目标：`shuffle_z` head 至少 10%，且 ep8-12 Overall 与 no2d_igdr 基线差距小于 2 点。
5. 只有机制和 AP 同时过门槛，才进入 KL balancing / per-cell free-bits；否则结束低维
   bottleneck + FiLM 主线，回到标准空间 RSSM。

### 53.3 100 iter DDP 冒烟

config：`me_rssm/sanity/lowdim_z_delta_smoke_100iter.py`；work_dir：
`/tmp/lowdim_z_delta_smoke_100iter`；GPU 5/6/7，3 卡 DDP。

| iter | latent loss | recon MSE | correction² | raw KL | clamped ratio | delta RMS | memory |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 0.3488 | 1.2212 | 0.0156 | 0.1999 | 0.5278 | 0.2043 | 7.437 GiB |
| 100 | 0.2845 | 1.1880 | 0.0103 | 0.1281 | 0.6250 | 0.2097 | 7.441 GiB |

接口、梯度、DDP、显存均通过；`stat_recon_valid_frac=0.99`，符合 N=4 中首帧无差分目标的
预期。recon 仅下降 2.7%，且 correction² 同期下降 34%，100 iter 还不能证明 z 已被救活。
下一步必须跑 450 iter：若 recon 明显下降且 correction² 稳住，再进入 ep6；若 correction²
继续单调逼近 0，则该设计在正式训练前直接止损。

### 53.4 450 iter 门控（结果：未通过）

config：`me_rssm/sanity/lowdim_z_delta_smoke_450iter.py`（commit `432fbbf`）；
work_dir：`/tmp/lowdim_z_delta_smoke_450iter`；GPU 5/6/7，3 卡 DDP；显存 7.44 GiB。

| iter | recon MSE | correction² | mu_diff² | clamped ratio | raw KL | delta RMS |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 1.2204 | 0.01550 | 0.01550 | 0.5309 | 0.1978 | 0.2021 |
| 100 | 1.1703 | 0.00890 | 0.00890 | 0.6668 | 0.1085 | 0.2043 |
| 150 | 1.1382 | 0.00690 | 0.00690 | 0.7373 | 0.0815 | 0.2274 |
| 200 | 1.1165 | 0.00560 | 0.00560 | 0.7979 | 0.0630 | 0.2215 |
| 250 | 1.1022 | 0.00480 | 0.00480 | 0.8385 | 0.0526 | 0.2081 |
| 300 | 1.0930 | 0.00430 | 0.00430 | 0.8707 | 0.0454 | 0.2120 |
| 350 | 1.0880 | 0.00400 | 0.00400 | 0.8868 | 0.0419 | 0.2158 |
| 400 | 1.0848 | 0.00390 | 0.00390 | 0.8934 | 0.0406 | 0.2339 |
| 450 | 1.0851 | 0.00390 | 0.00390 | 0.8960 | 0.0399 | 0.2081 |

**逐条对照预登记门槛：**

| 门槛 | 要求 | 实测 | 结论 |
|---|---|---|---|
| recon 降幅 | ≥ 15% | `1.2204 → 1.0851` = **−11.1%** | **未过** |
| correction² 终值 | ≥ 0.005 | **0.00390** | **未过** |
| correction² 后 150 iter | 不再持续下降 | 0.00430 → 0.00390，**仍在下滑** | **未过** |
| clamped ratio | < 0.95 | 0.8960 | 通过 |

除 clamped ratio 外全部未过。**帧间残差目标本身不足以阻止 correction 塌缩。**

### 53.5 方案二：batch-shuffle 排名损失（结果：未通过）

按 53.4 的判据（recon 有下降、correction 仍塌缩）进入方案二，实现见 commit `3d8aefa`：

```
loss_discr = relu(margin + mse(decoder(correction_i), target_i)
                         - mse(decoder(correction_roll), target_i))
```

- `discr_loss_weight=0.1`、`margin=0.2`；负样本由 batch 内 `roll(1)` 构造，与重建项共用
  同一个差分目标与 pair 有效性 mask；batch < 2 或有效样本 < 2 时跳过。
- 配置：`configs/r4det/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_lowdim_z_delta_discr.py`
- 冒烟：`me_rssm/sanity/lowdim_z_delta_discr_smoke_450iter.py`；work_dir
  `/tmp/lowdim_z_delta_discr_smoke_450iter`；显存 7.44 GiB。KL、free-nats、latent 大小与
  schedule 均未改动。RSSM 单测 35 条通过（新增「batch 内完全相同的 correction 会被惩罚」
  与「单样本 batch 该保持惰性」两条）。

| iter | recon MSE | correction² | pos MSE | neg MSE | gap = neg − pos | clamped | raw KL |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 1.2215 | 0.01550 | 1.1944 | 1.1961 | **+0.00170** | 0.5311 | 0.1978 |
| 100 | 1.1746 | 0.00890 | 1.1492 | 1.1510 | +0.00180 | 0.6673 | 0.1083 |
| 150 | 1.1445 | 0.00690 | 1.1201 | 1.1222 | +0.00210 | 0.7387 | 0.0810 |
| 200 | 1.1235 | 0.00560 | 1.1080 | 1.1117 | +0.00370 | 0.7995 | 0.0626 |
| 250 | 1.1090 | 0.00480 | 1.1090 | 1.1141 | +0.00510 | 0.8405 | 0.0522 |
| 300 | 1.0992 | 0.00430 | 1.0919 | 1.0996 | +0.00770 | 0.8714 | 0.0453 |
| 350 | 1.0939 | 0.00400 | 1.0867 | 1.0958 | +0.00910 | 0.8874 | 0.0418 |
| 400 | 1.0898 | 0.00400 | 1.0685 | 1.0796 | +0.01110 | 0.8929 | 0.0407 |
| 450 | 1.0901 | 0.00390 | 1.0901 | 1.1011 | **+0.01110** | 0.8956 | 0.0400 |

| 门槛 | 要求 | 实测 | 结论 |
|---|---|---|---|
| `neg − pos` | ≥ 0.1 | **0.0111**（差约 9 倍） | **未过** |
| 后 150 iter 的 gap 增长 | 持续 | 0.0077 → 0.0091 → 0.0111 → **0.0111 已停滞** | 未过 |
| correction² 不再塌缩 | 稳住 | 0.0043 → **0.0039**，仍在下滑 | 未过 |
| recon 降幅 | — | −10.8%（与无 discr 的 −11.1% 几乎相同） | 无改善 |

**关键新证据：decoder 的重建误差 `1.09` 已经高于「恒输出零」的 `1.00`**（差分目标已按
样本标准化为方差 1.0）。也就是说 decoder 不仅没有读出 correction 的样本内容，其输出相对
标准化目标甚至略差于常数预测。`pos` 与 `neg` 几乎相等（1.0901 vs 1.1011，相对差 1%），
说明 **decoder 输出对「收到哪一个 correction」几乎没有依赖**。排名损失方向正确（gap 确实
单调上升），但它要放大的是 decoder 对 correction 的敏感度，而这受制于 correction 本身的
幅度（RMS ≈ 0.062）与 decoder 输入通路。

**结论：方案二失败。** 问题已不在监督信号缺失——排名项确实在优化，是「低维 correction
无法有效到达/驱动下游」这一结构瓶颈。

### 53.6 方案三：放弃 FiLM 旁路（结果：未通过）

按预登记顺序进入方案三：删除 global pool + `film_proj`/`gate_proj`，改为

```
output = feat + h_proj(h_t) + z_proj(correction_t)
```

`h_proj`/`z_proj` 均为 `Conv2d(32, 256, 1)`，先在 16×16 投影再双线性上采样；保留差分重建；
先不改 KL。理由是 53.5 已证明空间 correction 经全局池化后既无法驱动重建、也无法驱动检测。

实现见 commit `9f6c84f`：`LowDimFutureConsistentLatentFusion` 新增 `readout_mode`
（`'film'` 默认保持旧行为、`'spatial'` 走新路径），`spatial` 下不构建
`film_proj`/`gate_proj`/`output_proj`/`correction_output_proj`，`h_proj`/`z_proj` 按
`std=0.01` 初始化；`spatial` 与 `latent_pool='global'` 组合直接报错。方案二的排名损失在本
配置中关闭（`discr_loss_weight=0`），使读出成为唯一变量。RSSM 单测 **41 条通过**（新增 6 条：
`spatial` 模式删除池化 FiLM 模块、`spatial`+`global` 报错、`film` 模式回归、输出形状、
两条分支各自影响输出、`prior_only` 下 correction 分支恒为零）。

配置：`configs/r4det/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_lowdim_z_delta_spatial.py`；
冒烟：`me_rssm/sanity/lowdim_z_delta_spatial_smoke_450iter.py`；work_dir
`/tmp/lowdim_z_delta_spatial_smoke_450iter`；日志 `/tmp/lowdim_spatial450.log`。

运行事实：GPU 5/6/7，3 卡 DDP，seed 0，deterministic，IterBasedRunner 450 iter；
显存 `7436 MiB`，约 `1.05 s/iter`，总耗时约 9 分钟（04:17:31 启动，04:25:51 完成）。
该运行是机制冒烟，`evaluation.interval=450` 未产出任何 val 记录，因此
**区间均值、全轮峰值、best saved 三项均不适用**（无 val epoch）。

| iter | recon MSE | correction² | clamped ratio | raw KL |
|---:|---:|---:|---:|---:|
| 50 | 1.2235 | 0.01590 | 0.5330 | 0.1987 |
| 100 | 1.1702 | 0.00912 | 0.6650 | 0.1095 |
| 150 | 1.1377 | 0.00689 | 0.7431 | 0.0794 |
| 200 | 1.1162 | 0.00559 | 0.8038 | 0.0616 |
| 250 | 1.1020 | 0.00473 | 0.8513 | 0.0499 |
| 300 | 1.0933 | 0.00424 | 0.8816 | 0.0432 |
| 350 | 1.0882 | 0.00400 | 0.8981 | 0.0398 |
| 400 | 1.0857 | 0.00390 | 0.9048 | 0.0384 |
| 450 | 1.0856 | 0.00383 | 0.9092 | 0.0376 |

| 门槛 | 要求 | 实测 | 结论 |
|---|---|---|---|
| recon 降幅 | ≥ 15% | **−11.3%** | **未过** |
| correction² 绝对水平 | ≥ 0.005 | **0.00383** | **未过** |
| 后 150 iter 不再持续下降 | 平台 | 0.00424 → 0.00400 → 0.00390 → **0.00383，仍在下滑** | **未过** |
| clamped ratio | < 0.95 | 0.9092 | 通过 |

**同预算 A/B：方案三与方案一的曲线几乎逐位重合。** 450 iter 时方案一（film）
recon `1.0851`、correction² `0.00390`、clamped `0.8960`；方案三（spatial）
recon `1.0856`、correction² `0.00383`、clamped `0.9092`。重建通路两版完全相同（这是预期），
但**检测读出换成不池化的空间残差后，correction 的塌缩轨迹没有任何变化**，说明读出结构不是
限制因素。

#### 53.6.1 三窗口反事实诊断（同一 450 iter checkpoint）

`tools/diagnose_z_utilization.py`，samples 各 4 条：`[3,4,5,6]`（offset 2）、
`[100,101,102,103]`（offset 2）、`[1800,1801,1802,1803]`（offset −1）。

| 窗口 | zero_z head | shuffle_z head | prior_only head | shuffle_z feature |
|---|---:|---:|---:|---:|
| `[3..6]` | 0.000598 | **0.000028** | 0.000257 | 0.000552 |
| `[100..103]` | 0.000602 | **0.000027** | 0.000256 | 0.000658 |
| `[1800..1803]` | 0.000593 | **0.000023** | 0.000255 | 0.000369 |

同预算的 film 版本（`/tmp/lowdim_z_delta_smoke_450iter/iter_450.pth`）为
shuffle_z head `0.000027`（`[3..6]`）与 `0.000021`（`[1800..1803]`）。两版在同一量级，
均远低于「ep6 shuffle_z head ≥ 1%」的预登记门槛。**`shuffle_z` 极低而 `zero_z` 高出约
20 倍**，属于「公共偏置」模式：z 主要提供的是一个所有样本共用的常量，而不是样本身份。

#### 53.6.2 尺度探针：读出通路是通的，但被压了约一个数量级

新增只读探针 `tools/probe_correction_scale.py`。在 `[3..6]` 上固定
`replace_z`（整批置换 posterior 均值），只把 `z_proj` 的 weight/bias 同时放大后重放：

| `z_proj` 增益 | shuffle_z feature ratio | shuffle_z head ratio |
|---:|---:|---:|
| ×1（训练值） | 0.000571 | **0.0000287** |
| ×10 | 0.074341 | **0.012625** |
| ×100 | 0.817707 | 0.138801 |
| ×1000 | 8.247325 | 1.407180 |

`[100..103]` 与 `[1800..1803]` 给出同样形状的结果（×10 时 head ratio `0.012668`、
`0.012548`）。**这说明两件事**：① 读出通路本身是通的，把增益提高 10 倍即可让
`shuffle_z` head 越过 1% 门槛；② 训练后 `z_proj` 的权重仍停在初始化尺度
（weight RMS `0.0101`，即 `std=0.01`），**检测损失没有把它推上去**，所以真实增益只有
门槛所需的约 1/10。

#### 53.6.3 根因探针：样本信息在 `e_pooled` 处被 L2 normalize 抹掉

新增只读探针 `tools/probe_posterior_source.py`。在跨场景抽样
（`stride=700`，indices `[3, 703, 1403]`）上把每一层的能量拆成
「batch 公共分量」与「样本特异分量」：

| 张量 | RMS | batch 公共 RMS | 样本特异 RMS | **样本特异能量占比** |
|---|---:|---:|---:|---:|
| `feat` pool16（原始 BEV） | 0.3781 | 0.3312 | 0.1824 | **23.27%** |
| `encoder` 输出 `e_t`（未 normalize） | 1.9679 | 1.8959 | 0.5273 | **7.18%** |
| `e_pooled`（L2 normalize 后） | 0.1768 | 0.1756 | 0.0200 | **1.28%** |
| `h_t`（GRU 状态） | 0.1691 | 0.1689 | 0.0089 | **0.27%** |
| `mu_p` | 0.1880 | 0.1879 | 0.0080 | 0.18% |
| `mu_q` | 0.1500 | 0.1494 | 0.0135 | 0.81% |
| `mu_q`（只给 `h_t`、observation 置零） | 0.1120 | 0.1119 | 0.0066 | 0.34% |
| `mu_q`（只给 `e_pooled`、`h_t` 置零） | 0.0921 | 0.0913 | 0.0121 | 1.72% |

同一探针在单窗口 `[3..6]` 上给出方向一致的更弱版本（`feat` 23.3% → `e_pooled` 0.15%
→ `mu_q` 0.13%），说明这不是跨场景抽样造成的假象。

`tools/probe_correction_scale.py` 同时给出 innovation（`mu_q − mu_p`）自身的分解：

| 窗口 | innovation RMS | 其中 batch 公共 | 其中样本特异 | 样本特异能量占比 |
|---|---:|---:|---:|---:|
| `[3..6]` | 0.06618 | 0.06593 | 0.00583 | **0.78%** |
| `[100..103]` | — | 0.06576 | 0.00555 | ~0.71% |
| `[1800..1803]` | — | 0.06542 | 0.00531 | ~0.66% |

**因果链因此是**：`feat` 里有 23.3% 的样本特异能量 → `encoder` 降到 7.2% →
`F.normalize(e_pooled, dim=1)` 再降到 **1.28%**（分母被 1.90 RMS 的公共分量主导，
除以范数等于把样本间差异一起压掉）→ `h_t` 几乎不含样本信息（0.27%）→ `mu_q` 只剩 0.81%，
而 `mu_q − mu_p` 这个被重建和检测共同使用的 correction 只剩 **0.78% 的样本特异能量**。

注意 `e_pooled` 的 L2 归一化是低维变体自加的约束（标准 `MotionAlignedRSSMFusion` 不做），
它在 `z_prev`、`e_pooled` 和 future 头上各出现一次；其中 `e_pooled` 这一次直接决定了
posterior 能拿到多少观测信息。

与标准 RSSM 的逐条对照（`mmdet3d/models/fusion_layers/rssm_fusion.py`）：

| 环节 | 标准 `MotionAlignedRSSMFusion`（line 646/943） | 低维变体（line 1367） |
|---|---|---|
| 状态尺寸 | `h_t`/`z_t` 与 BEV 同分辨率 256ch | 16×16×32 |
| 转移输入 | `z_{t-1}`（+运动对齐） | `normalize(pool(z_{t-1}))` |
| 观测编码 | `e_t = encoder(feat)` 直接用 | **`normalize(pool(e_t))`** |
| posterior | `q(cat[h_t, e_t])` | `q(cat[h_t, e_pooled])` |
| 输出读出 | `output_proj(z_t) + feat`（逐像素、不池化、不过 gate） | spatial 版已对齐为 `h_proj(h_t)+z_proj(correction)` |
| 重建 | `decoder(cat[h_t, z_t])` 重建全分辨率 `feat` | `decoder(correction_t)` 重建标准化帧间残差 |

**结论：低维变体在「输出读出」这一项上已被方案三对齐到标准形态，剩下的关键偏离是
`e_pooled` 的 L2 normalize。** 标准 RSSM 的 posterior 直接吃未归一化的 256ch 全分辨率
`e_t`，其样本特异能量占比为 7.2%（本探针 pooled 后的实测值），低维变体归一化后只剩 1.28%。

复现命令（均为只读诊断，物理 GPU 5/6）：

```bash
CUDA_VISIBLE_DEVICES=5 python tools/diagnose_z_utilization.py \
  --config me_rssm/sanity/lowdim_z_delta_spatial_smoke_450iter.py \
  --checkpoint /tmp/lowdim_z_delta_spatial_smoke_450iter/iter_450.pth \
  --start-index 3 --limit 4 --batch-size 4 --shuffle-offset 2 \
  --output /tmp/z_util_spatial_w3.json

CUDA_VISIBLE_DEVICES=5 python tools/probe_correction_scale.py \
  --config me_rssm/sanity/lowdim_z_delta_spatial_smoke_450iter.py \
  --checkpoint /tmp/lowdim_z_delta_spatial_smoke_450iter/iter_450.pth \
  --start-index 3 --limit 4 --shuffle-offset 2 \
  --output /tmp/corr_scale_w3.json

CUDA_VISIBLE_DEVICES=5 python tools/probe_posterior_source.py \
  --config me_rssm/sanity/lowdim_z_delta_spatial_smoke_450iter.py \
  --checkpoint /tmp/lowdim_z_delta_spatial_smoke_450iter/iter_450.pth \
  --start-index 3 --limit 4 --stride 700 \
  --output /tmp/post_source_stride700.json
```

#### 53.6.4 结论与下一步

**方案三失败，且失败原因不是读出结构。** 删除 global pool/FiLM 后，correction 的塌缩轨迹、
三窗口 `shuffle_z` 影响、以及 recon 曲线都与方案一/方案二在噪声水平内相同。真正的限制是：

1. **上游**：correction 的样本特异能量只有 0.78%（即 **99.2% 是公共偏置**），所以「换一个样本的 z」
   本来就几乎没有东西可换；`e_pooled` 的 L2 normalize 是最大的一次信息损失（7.2% → 1.28%）。
2. **下游**：`z_proj` 停留在 `std=0.01` 的初始化尺度，检测损失没有把它推大，所以即使把
   现有信号送出去也只产生 `2.9e-5` 的 head 变化。

**勘误（2026-09-27，同一探针加做 weight/bias 分离）**：53.6.2 的尺度表当时把 `z_proj` 的
weight 与 bias 一起放大，读数因此被高估。分离后实测为：只放大 weight ×10 时 head ratio
`2.30e-3`，weight×10 且 bias×10 时才到 `1.263e-2`——即当时「×10 越过 1% 门槛」的结果主要
来自 bias 注入的样本无关偏置，而不是 weight 增益。`tools/probe_correction_scale.py` 已改为
分别报告 `10.0`（仅 weight）与 `10.0_both`（weight+bias）。**这条修正使路线 A 的必要性更强：
仅仅提高读出增益并不足以让样本身份出现在 head 上。**

这两条与 53.5 的读数一致（`pos` 与 `neg` 只差 1%，recon 高于常数预测 1.00），并解释了它。

**下一步候选（按代价从低到高，尚未执行）**：

1. **去掉 `e_pooled` 的 L2 normalize，改回标准 RSSM 的做法**（或改用 LayerNorm/仅对 `z_prev`
   保留归一化）。这是唯一被定量证明会直接抬高样本特异占比的一步，改动一行，风险最低。
2. 若 ① 后 `mu_q` 的样本特异占比仍上不去，再考虑放开 `h_t` 的公共偏置（`h_t` 本身只有
   0.27% 特异能量，GRU 实际上收敛成了一个常量状态）。
3. 待机制门槛（`shuffle_z` head ≥ 1%）通过后，才做方案四的 KL balancing 与按 cell 聚合的
   free-bits。**当前 raw KL 一直有界（450 iter 时 0.0376），再次确认它不是瓶颈**，不做。

---

## 54. 路线 A：2×2 机制矩阵（posterior 输入 × 读出初始化，2026-09-27）

### 54.0 状态

本节是机制矩阵记录，四组均为 450 iter 冒烟，**没有 val 记录**，因此不填写区间均值、
全轮峰值或 best saved。四组按 `posterior_obs_mode ∈ {l2, scaled_raw}` ×
`z_proj_init ∈ {small, xavier}` 组合，其余变量（差分重建目标、`recon_input_mode=
'posterior_correction'`、`readout_mode='spatial'`、`discr_loss_weight=0`、KL、free-nats、
latent 大小、LR 调度）完全固定。

### 54.1 动机与预登记

第 53 节把根因定位到两处：① 上游 `F.normalize(e_pooled)` 把样本特异能量从 `e_t` 的 7.2%
压到 1.28%；② 下游 `z_proj` 停在 `std=0.01`。2×2 的目的是把这两条**分离**验证：若只修上游
能起效，说明下游不是限制；若必须两者同修才起效，则可主张「双瓶颈」。

预登记门槛（`shuffle_z head ≥ 1%` 为核心）：

```
e_pooled specific    >= 5%
correction specific  >= 3%
correction²          >= 0.005 并形成平台
shuffle_z head       >= 1%
recon 降幅            >= 15%
raw KL                有界
```

### 54.2 配置与运行事实

| 组 | 配置 | 冒烟配置 | work_dir | 日志 |
|---|---|---|---|---|
| 1（基线） | `..._lowdim_z_delta_spatial.py` | `..._spatial_smoke_450iter.py` | `/tmp/lowdim_z_delta_spatial_smoke_450iter` | `/tmp/lowdim_spatial450.log` |
| 2 | `..._spatial_raw.py` | `..._spatial_raw_smoke_450iter.py` | `/tmp/lowdim_routeA/raw` | `/tmp/lowdim_routeA/raw.log` |
| 3 | `..._spatial_xavier.py` | `..._spatial_xavier_smoke_450iter.py` | `/tmp/lowdim_routeA/xavier` | `/tmp/lowdim_routeA/xavier.log` |
| 4 | `..._spatial_raw_xavier.py` | `..._spatial_raw_xavier_smoke_450iter.py` | `/tmp/lowdim_routeA/raw_xavier` | `/tmp/lowdim_routeA/raw_xavier.log` |

实现（commit `48417c1`）：新增 `posterior_obs_mode`（`'l2'` 默认 / `'scaled_raw'`）、
`posterior_obs_scale`（默认 0.1，须 > 0）、`z_proj_init`（`'small'` 默认 / `'xavier'`，
后者要求 `readout_mode='spatial'`）。`scaled_raw` 走 `pool(e_t) * 0.1`，把原始 RMS 1.97
缩到约 0.197，接近 `h_t` 的 0.169；用同一个全局标量，不抹除样本间幅度差异。
`z_proj_init='xavier'` 复用类内其它投影相同的 `xavier_init`，实测 RMS `0.08353`，
是 `small`（`0.00991`）的 **8.43 倍**。

四组均为 GPU 5/6/7、3 卡 DDP、seed 0、deterministic、IterBasedRunner 450 iter；
显存均 `7436 MiB`，`1.09–1.11 s/iter`。RSSM 单测 **45 条通过**（新增 4 条：
`scaled_raw` 保留样本间幅度差异而 L2 不保留、`posterior_obs_scale<=0` 报错、
`xavier/small` 尺度比落在 5–20 倍、`xavier` 要求 spatial 读出）。

### 54.3 训练曲线（每 50 iter）

| iter | 1 l2+small | 2 raw+small | 3 l2+xavier | 4 raw+xavier |
|---:|---|---|---|---|
| | recon / corr² | recon / corr² | recon / corr² | recon / corr² |
| 50 | 1.2235 / 0.0159 | 1.2264 / 0.0114 | 1.2229 / 0.0162 | 1.2261 / 0.0116 |
| 100 | 1.1702 / 0.0091 | 1.1701 / 0.0077 | 1.1696 / 0.0097 | 1.1698 / 0.0083 |
| 150 | 1.1377 / 0.0069 | 1.1375 / 0.0061 | 1.1366 / 0.0076 | 1.1363 / 0.0070 |
| 200 | 1.1162 / 0.0056 | 1.1162 / 0.0050 | 1.1148 / 0.0063 | 1.1150 / 0.0059 |
| 250 | 1.1020 / 0.0047 | 1.1019 / 0.0043 | 1.0999 / 0.0055 | 1.1000 / 0.0052 |
| 300 | 1.0933 / 0.0042 | 1.0942 / 0.0038 | 1.0908 / 0.0050 | 1.0916 / 0.0047 |
| 350 | 1.0882 / 0.0040 | 1.0892 / 0.0036 | 1.0858 / 0.0047 | 1.0858 / 0.0045 |
| 400 | 1.0857 / 0.0039 | 1.0872 / 0.0035 | 1.0825 / 0.0046 | 1.0823 / 0.0044 |
| 450 | 1.0856 / 0.0038 | 1.0858 / 0.0035 | 1.0821 / **0.0045** | 1.0818 / **0.0044** |

末尾 raw KL / clamped ratio：

| 组 | raw KL @450 | clamped @450 | `z_proj` weight RMS @450 |
|---|---:|---:|---:|
| 1 l2+small | 0.0376 | 0.9092 | 0.01014（≈初始 0.00991） |
| 2 raw+small | 0.0340 | 0.9244 | 0.01007（≈初始） |
| 3 l2+xavier | 0.0448 | 0.8768 | **0.08268**（≈初始 0.08353） |
| 4 raw+xavier | 0.0424 | 0.8889 | **0.08271**（≈初始） |

**关键观察：Xavier 组把权重稳在初始尺度（0.083），small 组也稳在 0.010——两边的检测损失
都没有把 `z_proj` 推出各自的初始化量级。** 这说明问题不是「训练把权重压小」，而是
「从初始尺度出发就没有足够梯度把它推大」。

### 54.4 Source 探针（跨场景 `stride=700`，indices `[3, 703, 1403]`）

`tools/probe_posterior_source.py`，样本特异能量占比：

| 组 | `e_pooled` | `mu_q` | **correction** | 门槛（e_pooled ≥5% / correction ≥3%） |
|---|---:|---:|---:|---|
| 1 l2+small | 1.28% | 0.81% | 4.79% | e_pooled 未过 |
| 2 raw+small | **10.31%** | 0.71% | 6.01% | 通过 |
| 3 l2+xavier | 1.96% | 1.58% | 8.27% | e_pooled 未过 |
| 4 raw+xavier | **14.19%** | 1.79% | **12.11%** | 通过 |

矩阵分离得很干净：**上游修复（`scaled_raw`）单独就把 `e_pooled` 从 1.28% 提到 10.31%
（8.1 倍），并把 correction 从 4.79% 提到 6.01%；下游修复（Xavier）不改变 `e_pooled`
的输入构成，但让 correction 的特异占比再翻一倍以上（4.79% → 8.27%，6.01% → 12.11%）。**
两者叠加得到最好的 12.11%。

同时可以看到 `mu_q` 的样本特异占比改善有限（0.81% → 1.79%），因为 `mu_q` 仍由公共分量
主导；但 **correction（即 `mu_q − mu_p`）的特异占比显著提升**，这正是重建和检测共用的量。

### 54.5 反事实诊断（三窗口，各 4 条）

`tools/diagnose_z_utilization.py`，窗口 `[3..6]`（offset 2）、`[100..103]`（offset 2）、
`[1800..1803]`（offset −1）。下表为 `[3..6]` 的 head 比值；括号内为后两窗口。

| 组 | zero_z head | **shuffle_z head** | prior_only head | 门槛 ≥1% |
|---|---:|---:|---:|---|
| 1 l2+small | 5.98e-4 | **2.77e-5** | 2.57e-4 | 未过（差 36 倍） |
| 2 raw+small | 6.95e-4 | **2.50e-5** | 1.87e-4 | 未过（差 40 倍） |
| 3 l2+xavier | 6.21e-3 | **2.79e-4** | 2.44e-3 | 未过（差 36 倍） |
| 4 raw+xavier | 7.04e-3 | **3.14e-4** | 2.04e-3 | 未过（差 32 倍） |

组 4 的另外两个窗口：`[100..103]` shuffle `2.68e-4`、`[1800..1803]` shuffle `2.17e-4`，
三窗口一致。

**判读**：
- **Xavier 单独把下游通路打开了约 10 倍**（shuffle 2.77e-5 → 2.79e-4，zero 5.98e-4 →
  6.21e-3，prior 2.57e-4 → 2.44e-3）。这证明「下游增益不足」确实是一个真实瓶颈，
  与 53.6.2 的勘误一致。
- **但它没有改变 z 的样本判别性比例**：`shuffle/zero` 之比在四组里几乎不动
  （组1 `0.046`、组2 `0.036`、组3 `0.045`、组4 `0.045`）。即 Xavier 把「公共偏置」和
  「样本身份」一起放大了，没有偏向后者。
- **上游修复提升了 correction 的特异占比（4.79% → 12.11%），但 head 上的 shuffle 只从
  2.79e-4 涨到 3.14e-4（1.12 倍）**。特异占比涨了 2.5 倍，head 响应只涨 12%，
  说明从 correction 到 head 的传递过程中，增加的样本身份并没有等比例地到达检测输出。

### 54.6 门槛判定

| 门槛 | 要求 | 组 4（最好） | 结论 |
|---|---|---|---|
| `e_pooled` specific | ≥ 5% | 14.19% | **通过** |
| correction specific | ≥ 3% | 12.11% | **通过** |
| correction² | ≥ 0.005 且平台 | 0.0044，末 150 iter 0.0047→0.0044 | **未过**（水平不足，但已明显平台化） |
| `shuffle_z` head | ≥ 1% | 3.14e-4 | **未过**（差 32 倍） |
| recon 降幅 | ≥ 15% | −11.8% | **未过** |
| raw KL | 有界 | 0.0424 | 通过 |

**路线 A 整体未通过机制门槛，不进入 ep6/ep12。** 但两项上游/下游修复各自都产生了
方向正确、量级明确的效果，属于 53 节以来第一次让 head 响应进入 `1e-4~1e-3`
（此前恒为 `2e-5`）。

### 54.7 结论与对后续路线的影响

1. **双瓶颈已被分别证实**：上游 `F.normalize(e_pooled)` 与下游 `z_proj` 初始尺度是两个
   独立的限制项，各自被单独修复都会带来可测收益（上游：correction 特异占比 ×2.5；
   下游：head 响应 ×10）。
2. **但两者相加仍不足以过门槛**，且 `shuffle/zero` 比值在四组间恒定在 0.04–0.05。
   这说明**还缺的不是「信号幅度」而是「结构」**：现有 `correction = mu_q − mu_p` 由两个
   公共大向量相减得到，即使特异能量占到 12%，其绝对幅度仍然很小且方向未被约束。
   这与路线 B 的设计动机直接对应。
3. **AP 未被测量**，四组均无 val。因此本节不能给出任何性能结论，也不能用「AP 未变差」
   为这些改动背书。
4. 复现命令与探针用法见 54.4/54.5 对应小节；`tools/probe_correction_scale.py` 的
   weight/bias 分离输出已修正（见 53.6.2 勘误）。

**下一步（按用户预登记顺序）**：进入路线 B（Innovation-Conditioned RSSM），
其成立前提「去 normalize 后 `mu_q` 改善」已由组 4 满足（correction 特异占比 12.11%）；
`h_t`/`mu_p` 仍接近常量（`mu_p` 特异占比 0.49%），符合路线 B 的触发条件。

---

## 55. 路线 B：Innovation-Conditioned RSSM（2026-09-27）

### 55.0 状态

450 iter 机制冒烟。**无 val 记录**（`evaluation.interval=4500`），因此不填写区间均值、
全轮峰值或 best saved。本节只回答机制问题：把 `correction` 从「两个公共大向量相减」
换成模型自己预测的 delta，能否提高 z 的样本判别性。

### 55.1 结构

基线：路线 A 组 4（`..._lowdim_z_delta_spatial_raw_xavier`，2×2 矩阵最优组，
`e_pooled` 特异 14.19% / correction 12.11%）。

```
e_hat_t      = obs_prior(h_t)                      # 只由 L_obs_pred 训练
innovation_t = e_pooled - stopgrad(e_hat_t)
delta_mu     = posterior_delta(cat([h_t, innovation_t]))
mu_q         = mu_p + delta_mu
correction_t = delta_mu                            # 不再是差值
```

```
L_obs_pred = smooth_l1(e_hat_t, stopgrad(e_pooled))
L_total   += 0.05 * L_obs_pred + 0.1 * L_recon
```

- `obs_prior` 只由观测预测项训练；`stopgrad(e_hat_t)` 保证它不出现在 posterior /
  重建 / 检测的任何梯度路径上，posterior 因此只看到「h_t 预测不了的那部分观测」。
- `posterior_mu` 在 innovation 结构下改为输出 `delta_mu`（而非 `mu_q`），
  `correction` 直接等于该 delta，不再是两个 batch-common 均值之差。
- **`z_prev` 的 L2 normalize 刻意保留**：按路线规划，这一步单独消融，避免两个变量同时变化。
- 保留 spatial readout；KL、free-nats、latent 大小、LR 调度不变；`discr_loss_weight=0`。

实现 commit `d354232`。新增参数 `posterior_struct ∈ {standard, innovation}`
（默认 `standard`，旧行为不变）与 `obs_pred_loss_weight`（默认 0.05，须 ≥0）。
新增诊断键 `stat_obs_pred`、`stat_innovation_sq`（均不含 `loss` 子串）。
RSSM 单测 **55 条通过**（新增 10 条：obs_prior 构建、standard 模式无 obs_prior、
非法 `posterior_struct`/负权重报错、**correction 必须等于 delta 而非 `mu_q−mu_p`**、
obs_prior 只收观测预测梯度、诊断键上报、两帧反传、六元 tuple 契约、reset 清理）。

**工具适配（同一 commit 后）**：`tools/diagnose_z_utilization.py` 与
`tools/probe_posterior_source.py` 原本硬编码 `posterior_mu` 输出为 `mu_q`，在 innovation
结构下会读错张量（或直接触发 `zero_z did not propagate to out[4]` 断言）。
现两者都检测 `posterior_struct`：反事实干预写入 `desired − mu_p`，来源探针按
`mu_q = mu_p + delta` 还原并额外报告 `innovation` / `obs_prior_e_hat` 的分解。
工具单测 **12 条通过**（新增 2 条：innovation 模式下 zero/shuffle 必须落到 `out[4]`、
`prior_only` 必须与显式 prior 前向逐位一致）。

### 55.2 运行事实

| 项 | 值 |
|---|---|
| 配置 | `configs/r4det/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_lowdim_z_innovation.py` |
| 冒烟 | `me_rssm/sanity/lowdim_z_innovation_smoke_450iter.py` |
| work_dir / 日志 | `/tmp/lowdim_routeB` / `/tmp/lowdim_routeB/run.log` |
| 硬件 | GPU 5/6/7，3 卡 DDP，seed 0，deterministic |
| 显存 / 速度 | `7436 MiB` / `1.095 s/iter` |
| 产物 | `iter_450.pth` |

### 55.3 训练曲线（每 50 iter）

| iter | recon MSE | correction² | innovation² | obs_pred | raw KL | clamped |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 1.2213 | 0.0097 | 0.0100 | 0.0050 | 0.1240 | 0.6488 |
| 100 | 1.1677 | 0.0070 | 0.0069 | 0.0034 | 0.0884 | 0.7234 |
| 150 | 1.1348 | 0.0059 | 0.0059 | 0.0030 | 0.0733 | 0.7671 |
| 200 | 1.1124 | 0.0053 | 0.0054 | 0.0027 | 0.0637 | 0.8002 |
| 250 | 1.0984 | 0.0047 | 0.0048 | 0.0024 | 0.0561 | 0.8293 |
| 300 | 1.0907 | 0.0044 | 0.0045 | 0.0022 | 0.0510 | 0.8509 |
| 350 | 1.0853 | 0.0042 | 0.0043 | 0.0021 | 0.0483 | 0.8632 |
| 400 | 1.0820 | 0.0041 | 0.0042 | 0.0021 | 0.0469 | 0.8696 |
| 450 | 1.0812 | 0.0041 | 0.0042 | 0.0021 | 0.0466 | 0.8711 |

`stat_obs_pred` 从 0.0050 单调降到 0.0021（−58%），说明 `obs_prior` 确实在学
「从 h_t 预测观测编码」；`stat_innovation_sq` 与之同步（残差变小）。
recon 降幅 −11.5%（1.2213→1.0812，门槛 15%，**未过**）；raw KL 有界（0.0466，通过）。

### 55.4 Source 探针（`stride=700`，indices `[3, 703, 1403]`）

| 张量 | 路线 A 组 4 | 路线 B | 变化 |
|---|---:|---:|---|
| `e_pooled` specific | 14.19% | 7.75% | — |
| `obs_prior_e_hat` specific | — | 1.80% | — |
| **`innovation` specific** | — | **14.27%** | 高于 `e_pooled` ✅ |
| `mu_p` specific | 0.49% | 1.71% | +3.5 倍 |
| `mu_q` specific | 1.79% | **3.93%** | +2.2 倍 |
| **correction (= delta_mu) specific** | 12.11% | **13.51%** | +1.12 倍 |
| correction RMS | 0.06579 | 0.05739 | — |

**关键结果**：innovation 的样本特异能量占比（14.27%）确实高于它自己的输入
`e_pooled`（7.75%），即 `obs_prior` 减掉的那部分主要是**公共**成分，posterior 拿到的
是经过了筛选的残差。`mu_q` 的特异占比同时翻倍（1.79% → 3.93%），
说明 delta 结构让 posterior 的输出更倾向于样本身份，而不是被 `mu_p` 的公共底色主导。

### 55.5 反事实诊断（三窗口，各 4 条）

head L2 比值 vs normal：

| 运行 / 窗口 | zero_z | **shuffle_z** | prior_only | **shuffle/zero** |
|---|---:|---:|---:|---:|
| 路线 A 组 4 `[3..6]` | 7.044e-3 | 3.136e-4 | 2.036e-3 | 0.045 |
| 路线 A 组 4 `[100..103]` | 6.881e-3 | 2.681e-4 | 2.100e-3 | 0.039 |
| 路线 A 组 4 `[1800..1803]` | 6.863e-3 | 2.173e-4 | 2.168e-3 | 0.032 |
| **路线 B `[3..6]`** | 4.551e-3 | **4.188e-4** | 1.838e-3 | **0.092** |
| **路线 B `[100..103]`** | 4.496e-3 | **3.941e-4** | 1.848e-3 | **0.088** |
| **路线 B `[1800..1803]`** | 4.587e-3 | **2.936e-4** | 1.817e-3 | **0.064** |

**这是全程第一次让 `shuffle/zero` 比值出现系统性变化**：路线 A 的四组都恒定在
0.032–0.046，路线 B 提升到 **0.064–0.092（约 2 倍）**。三窗口一致，不是单窗口噪声。

同时 `zero_z` 的绝对值下降（7.0e-3 → 4.5e-3），`prior_only` 也略降
（2.1e-3 → 1.8e-3）——即路线 B 降低了 latent 的整体公共偏置强度，
这正是设计意图。

### 55.6 门槛判定

| 门槛 | 要求 | 实测 | 结论 |
|---|---|---|---|
| `innovation` specific > `e_pooled` specific | 是 | 14.27% > 7.75% | **通过** |
| `delta_mu` specific | ≥ 5% | 13.51% | **通过** |
| `prior_only` head 明显低于 posterior | 是 | 1.8e-3 vs 4.5e-3（0.40×） | **通过** |
| `shuffle_z` head | ≥ 2% | 4.19e-4（最好窗口） | **未过**（差约 48 倍） |
| recon 降幅 | ≥ 15% | −11.5% | **未过** |
| raw KL | 有界 | 0.0466 | 通过 |

**路线 B 三条机制门槛全部通过，核心性能门槛仍未过。** 与路线 A 相比，它是第一个
在「样本判别性比例」上产生系统性改善的结构（`shuffle/zero` ×2 且三窗口一致），
但绝对量级仍差约 48 倍。

### 55.7 结论与下一步

1. **结构方向被验证**：把 correction 从「两个公共大向量之差」换成显式预测的 delta
   之后，`mu_q` 特异占比翻倍、`shuffle/zero` 比值翻倍。这支持「Kalman 式
   predict-correct 更适合 BEV 时序融合」这一动机，可以作为论文方法的机制证据。
2. **但幅度仍不足**：`shuffle_z` head 4.19e-4 对比 2% 门槛差 48 倍。路线 B 解决的是
   「correction 由什么构成」，没有解决「这个 correction 如何被检测头放大」。
   注意路线 A 已证明单纯放大 `z_proj` 会等比放大公共偏置与样本身份
   （`shuffle/zero` 不变），所以下一步不能再只调增益。
3. **`obs_prior` 学到的是公共成分**：`e_hat_t` 的样本特异占比只有 1.80%，
   接近 `mu_p` 的量级，说明 h_t 只能预测观测里的公共部分——这与 `h_t` 本身
   0.86% 的特异占比一致，属于预期行为而非缺陷。
4. **下一候选（用户预设顺序）**：路线 C（Object-Centric Latent Supervision）。
   路线 B 已有机制收益但 AP 未被测量；按规划「B 有机制收益但 AP 不够，再加路线 C」。
   **注意：路线 B 尚未测 AP**，因此严格说应先跑 ep6 取得 AP 与机制读数，
   再决定是否叠加路线 C。
5. **仍未做**：`z_prev` 的 L2 normalize 消融、路线 D 的去塌缩损失、路线 E 的 KL balancing。

---

## 56. 路线 B ep6 门控：正式训练 AP + 机制联合判定（2026-09-27/28）

### 56.0 状态与判定摘要

路线 B 结构（Innovation-Conditioned RSSM，第 55 节）在 450 iter 冒烟中首次拿到
系统性的机制改善，但**没有 AP 读数**，无法判断这种改善对检测是否有价值。本节按
用户预登记的 ep6 门控协议跑正式训练，ep6 验证落盘后主动停止，并同时给出三窗口
反事实诊断与 source 探针。

**一句话结论：机制继续增强、AP 差距超过止损阈值 —— 按预登记分支，本路线不续跑
ep12，也不叠加路线 C，下一步先做 `z_prev` normalize 的 450 iter 单变量消融。**

| 检查项 | 要求 | ep6 实测 | 结论 |
|---|---|---|---|
| Overall 3D moderate（ep6 单点 vs no2d_igdr seed0 ep6） | 差距 ≤ 2.0 | 31.4895 vs 34.0243 = **−2.5348** | **未过（止损）** |
| `shuffle_z` head 绝对值 | 不低于 450 iter 的 0.064–0.092（比值口径）；最好 ≥0.10 | 比值 0.0414–0.0712；head 绝对值 7.3e-3–1.1e-2 | 比值口径**未过**、绝对值**大幅上升**（详见 56.5） |
| `shuffle_z` head | ep6 继续门槛 ≥0.2%，最终门槛 ≥2% | **0.73%–1.09%** | **过 ep6 门槛**，最终门槛仍差约 2 倍 |
| `innovation` specific > `e_pooled` specific | 是 | 43.51% > 25.66% | **通过** |
| `delta_mu` specific | ≥ 10% | 39.32% | **通过** |
| `correction²` 不再单调塌缩、形成平台 | 是 | ep1→ep6：0.1489→0.0596，逐 epoch 单调下降且**未见平台** | **未过** |

**分支判定**：AP 差距 2.53 > 2.0 ⟹ 按预登记协议

```
AP 差距 >2 → 不叠加 C；先做 z_prev normalize 的 450 iter 单变量消融
```

需要诚实标注的一点：本节为**单 seed、单 epoch、与不同代码版本的基线做同 epoch 对照**，
样本量为 1，`−2.53` 与阈值 `−2.0` 的差距只有 0.53 点，而 no2d_igdr 三 seed 在
ep12-16 的标准差就有 1.35 点。因此该判定是**按预登记规则执行**，不是「已被统计显著
证伪」；它的作用是强制先做单变量消融，而不是给路线 B 下最终结论。

### 56.1 配置与唯一变量

| 项 | 值 |
|---|---|
| 配置 | `configs/r4det/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_lowdim_z_innovation.py` |
| 对照 | `configs/r4det/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr.py`（no2d_igdr, seed0） |
| 唯一变量 | `posterior_struct='innovation'` + `obs_pred_loss_weight=0.05`；其余继承链与对照一致 |
| 训练日程 | **`max_epochs=24` 的 CosineAnnealing 原日程未改**，ep1-6 的 LR 与 KL warmup 因此与 24e 基线逐轮对齐 |

关于训练长度做了一次显式取舍：曾考虑新建 `max_epochs=6` 的短配置，但
`CosineAnnealingLrUpdaterHook` 用 `runner.max_epochs` 归一化进度，
改成 6 会让 ep6 的 LR 落到约 `1e-5`（基线约 `1.3e-4`），AP 对照直接失真。
最终保留 24e 日程、跑到 ep6 后用看门狗在验证与落盘完成后停止，因此**进度在 ep6
截断，日程没有被截断** —— 这是本节与对照可比的前提。

关键超参（与对照一致）：`optimizer=AdamW(lr=1.5e-4, betas=(0.95,0.99), wd=0.01)`、
`GradientCumulativeOptimizerHook(cumulative_iters=2)`、`samples_per_gpu=2`、
`workers_per_gpu=2`、`seq_len=4`、`checkpoint_config.interval=2`、`evaluation.interval=1`、
KL warmup `0.01→1.0` over ep0-12、`free_nats=0.1`、`min_std=0.1`、`init_std=0.2`、
`recon_loss_weight=0.1`、`obs_pred_loss_weight=0.05`、`future_loss_weight=0`、
`latent_size=(16,16)`、`latent_dim=32`。

`val.samples_per_gpu=4`：对照 seed0 的 ep1-6 由 val bs=1 的旧代码产出；第 51 节已用同一
harness 做过 bs=1/2/4 A/B（Overall 3D mod 38.4516/38.4947/38.5420，差异在噪声内、
无 batch 间串扰），且 offset 2.2 点的固定差已定位为 offline harness 与 EvalHook 的差异，
与 batch 无关。本节与对照都在**训练内 EvalHook + evaluation.pipeline**下产生，口径一致。

### 56.2 运行事实

| 项 | 值 |
|---|---|
| work_dir | `/data/lurui/work_dirs/lowdim_z_innovation_routeB_3x2x2_24e_seed0` |
| 日志 | 20260927_125916.log(.json)、`/tmp/lowdim_routeB_ep6.log` |
| 硬件 | GPU 5/6/7，3 卡 DDP，seed 0，deterministic |
| 起止 | 2026-09-27 12:59:24 UTC 启动；ep6 val 于 17:21:16 UTC 写完，随后主动停止 |
| 训练吞吐 | 中位 **1.047 s/iter**（p10 1.019、p90 1.101，n=228 条训练日志）；峰值显存 **7446 MiB** |
| 落盘 | `epoch_2.pth` / `epoch_4.pth` / `epoch_6.pth`（`checkpoint_config.interval=2`），`latest.pth -> epoch_6.pth` |
| val 记录 | ep1-6 各一条，共 6 条，无重启重复计数 |
| 停止方式 | 独立 tmux 看门狗检测 `Epoch(val) [6]` + `epoch_6.pth` 同时存在后发送 `C-c`；两处 tmux 会话均已退出，GPU 5/6/7 已释放 |
| iter 450 对照 | 正式训练 ep1 iter450：recon 1.0024 / corr² 0.1396 / raw KL 0.3960 / clamped 0.2884，与 450 iter 冒烟末点（1.0812 / 0.0041 / 0.0466 / 0.8711）量级不同，见 56.4 讨论 |

与 450 iter 冒烟的差异属于**预期**：冒烟是 `IterBasedRunner` 且学习率按 450 iter 余弦
退火（末段已接近 0），正式训练是 24e 日程下 ep1 的前 450 iter，LR 仍在 1.5e-4 平台，
KL warmup 也从 0.01 起步。两者只能对照曲线形状与机制读数，不能直接比较绝对值。

### 56.3 完整逐 epoch 曲线

主指标（`pts_bbox/KITTI/*`）：

| epoch | Overall 3D | Overall BEV | Car strict | Cyclist loose | Ped loose | Truck strict |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20.8115 | 28.2116 | 24.7019 | 34.1987 | 17.2757 | 7.0698 |
| 2 | 22.5532 | 31.7365 | 26.5524 | 34.8821 | 17.9785 | 10.7999 |
| 3 | 28.1833 | 35.1923 | 34.7591 | 42.6058 | 22.9203 | 12.4479 |
| 4 | 25.0003 | 36.0461 | 26.9185 | 40.6441 | 18.2509 | 14.1876 |
| 5 | **34.8551** | 42.5291 | 45.1807 | 49.4583 | 23.8192 | 20.9621 |
| 6 | 31.4895 | 39.0407 | 39.9860 | 43.5729 | 21.1131 | 21.2860 |

同 epoch 对照（no2d_igdr seed0 真实曲线，非 seed1）：

| epoch | 基线 Overall 3D | 路线 B Overall 3D | Δ | 基线 BEV | 路线 B BEV | Δ |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 15.2640 | 20.8115 | **+5.5475** | 23.2541 | 28.2116 | +4.9575 |
| 2 | 26.3716 | 22.5532 | −3.8184 | 33.2335 | 31.7365 | −1.4970 |
| 3 | 31.6591 | 28.1833 | −3.4758 | 39.5683 | 35.1923 | −4.3760 |
| 4 | 32.8506 | 25.0003 | −7.8503 | 42.0088 | 36.0461 | −5.9627 |
| 5 | 34.0827 | 34.8551 | +0.7724 | 44.1080 | 42.5291 | −1.5789 |
| 6 | **34.0243** | **31.4895** | **−2.5348** | 42.4084 | 39.0407 | −3.3677 |

> **勘误（重要）**：本次任务交接摘要中给出的「no2d_igdr seed0 ep1-6 曲线」
> （ep1 17.9825 / ep2 25.1756 / ep3 33.3100 / ep4 34.4331 / ep5 36.2203 / ep6 37.3254）
> **实际是 seed1 的曲线**（见第 48 节）。本节所有判定均改用 seed0 真实曲线
> （ep1 15.2640 / ep2 26.3716 / ep3 31.6591 / ep4 32.8506 / ep5 34.0827 / ep6 34.0243），
> 与用户指定的「no2d_igdr seed0 同 epoch」对照一致。若误用 seed1 曲线，ep6 门槛会变成
> 35.33，本节会得出「−3.84 差距」的另一个结论；这里按 seed0 口径记录。

逐类别四类构成项（loose 口径）：本节主表已按 Overall 的混合口径给出
「Car strict / Truck strict / Cyclist loose / Ped loose」，下面是同四类的 strict 与
BEV strict 拆解，用于误差归因：

| epoch | Car strict | Truck strict | Cyc strict | Ped strict | Car BEV strict | Truck BEV strict | Cyc BEV strict | Ped BEV strict |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 24.7019 | 7.0698 | 9.1172 | 0.7745 | 42.5056 | 15.1996 | 19.4935 | 1.7430 |
| 2 | 26.5524 | 10.7999 | 13.8097 | 0.1172 | 44.8704 | 25.7914 | 24.8019 | 0.3655 |
| 3 | 34.7591 | 12.4479 | 22.8960 | 0.1036 | 52.3379 | 19.5932 | 30.9409 | 0.4452 |
| 4 | 26.9185 | 14.1876 | 21.0205 | 0.0747 | 56.2863 | 22.4801 | 28.7086 | 0.1628 |
| 5 | 45.1807 | 20.9621 | 24.1610 | 0.0733 | 57.0803 | 33.0548 | 38.0805 | 0.2513 |
| 6 | 39.9860 | 21.2860 | 22.4470 | 0.0499 | 53.8754 | 31.2762 | 31.2796 | 0.1280 |

**窗口均值（ep1-6，n=6，实际跨度 1-6，非标准 ep12-16 窗口）**：

| 口径 | Overall 3D | Overall BEV | Car strict | Cyclist loose | Ped loose | Truck strict |
|---|---:|---:|---:|---:|---:|---:|
| 路线 B ep1-6 等权均值 | 27.1488 | 35.4594 | 33.0164 | 40.8937 | 20.2263 | 14.4589 |
| no2d_igdr seed0 ep1-6 等权均值 | 29.0420 | 37.4302 | 39.9218 | 36.7749 | 19.7073 | 19.7642 |
| Δ（B − 基线） | **−1.8932** | −1.9708 | **−6.9053** | **+4.1188** | +0.5189 | **−5.3054** |

**三口径必填摘要**（本 run 有 6 条 val 记录）：

| 口径 | 值 |
|---|---|
| 区间均值（ep1-6，n=6，实际跨度 1-6；标准 ep12-16 窗口不存在） | Overall 3D moderate **27.1488** |
| 全轮峰值（所有有 val 的 epoch 中最高单点） | **34.8551 @ ep5** |
| best saved（实际有权重的 epoch 中最高，`interval=2` 故仅 ep2/4/6） | **31.4895 @ ep6** |

### 56.4 训练内诊断（每 epoch 的 mean over 38 条日志）

| epoch | recon MSE | correction² | innovation² | obs_pred | raw KL | clamped | mu_diff² | post std | prior std |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.9937 | 0.1489 | 0.0397 | 0.0196 | 0.5245 | 0.2057 | 0.1489 | 0.2787 | 0.4156 |
| 2 | 0.9460 | 0.0951 | 0.0438 | 0.0218 | 0.1361 | 0.6186 | 0.0951 | 0.6708 | 0.7434 |
| 3 | 0.9354 | 0.0816 | 0.0494 | 0.0246 | 0.0540 | 0.8541 | 0.0816 | 0.9156 | 0.9377 |
| 4 | 0.9308 | 0.0751 | 0.0511 | 0.0254 | 0.0444 | 0.8869 | 0.0751 | 0.9530 | 0.9645 |
| 5 | 0.9285 | 0.0661 | 0.0489 | 0.0243 | 0.0376 | 0.9110 | 0.0661 | 0.9688 | 0.9752 |
| 6 | 0.9266 | 0.0596 | 0.0487 | 0.0243 | 0.0331 | 0.9267 | 0.0596 | 0.9779 | 0.9819 |

读数：

1. **recon 有效但收益偏小**：0.9937 → 0.9266（**−6.75%**），远低于 15% 门槛。
   注意正式训练的 recon 起点就比冒烟低（ep1 即 ≈0.99，冒烟 iter50 为 1.2213），
   因为正式训练在继承链上先经过大量 epoch 级别的数据曝光。
2. **`correction²` 仍在单调塌缩，没有形成平台**：ep1→ep6 每轮都在降
   （0.1489 → 0.0951 → 0.0816 → 0.0751 → 0.0661 → 0.0596），ep6 之后没有出现用户
   要求的平台；这条与 450 iter 冒烟的「0.0041 仍在滑落」一致，只是绝对量级高两个数量级
   （正式训练 LR 更高、更新更多）。
3. **`raw KL` 全程有界**（0.5245 → 0.0331），`clamped_ratio` 0.21→0.93。KL warmup
   在 ep6 时 `kl_scale ≈ 5/12 ≈ 0.42`，因此 raw KL 的下降同时受调度影响，不能直接
   解读为「posterior 主动贴近 prior」；不过它排除了 KL 爆炸这一病因。
4. **`obs_pred` 不降反升**（0.0196 → 0.0243）：`obs_prior` 在正式训练里比冒烟更早
   进入「h_t 预测观测」的较强拟合阶段，但 ep6 时绝对残差仍远大于 450 iter 冒烟末点
   （0.0021）。这与 `innovation²` 在 0.044–0.051 平台一致：innovation 的绝对能量被
   稳定在大约 0.05，说明 posterior 拿到的确实是残差，而不是被压到零。
5. **posterior/prior std 同步上升**（0.279/0.416 → 0.978/0.982），两者差距随训练收窄，
   与 `mu_diff²` 下降方向一致；没有出现 std 崩塌，也没有出现 prior 被 KL 拉爆。

### 56.5 ep6 三窗口反事实诊断

`tools/diagnose_z_utilization.py`，`epoch_6.pth`，各窗口 4 条序列，
head L2 为「干预后 head 输出相对 normal 的 L2 变化 / normal head 范数」，取窗口内 4 条均值：

| 窗口 | zero_z feature | zero_z head | shuffle_z feature | shuffle_z head | prior_only feature | prior_only head | shuffle/zero |
|---|---:|---:|---:|---:|---:|---:|---:|
| `[3..6]` | 1.0426 | 0.1577 | 0.0665 | 0.0086 | 0.2321 | 0.0265 | 0.0546 |
| `[100..103]` | 1.1497 | 0.1755 | 0.0539 | 0.0073 | 0.2608 | 0.0363 | 0.0414 |
| `[1800..1803]` | 0.9777 | 0.1534 | 0.0757 | 0.0109 | 0.2204 | 0.0264 | 0.0712 |

与 450 iter 冒烟的三窗同口径对照：

| 运行 | window | zero_z head | shuffle_z head | prior_only head | shuffle/zero |
|---|---|---:|---:|---:|---:|
| 路线 B 450 iter | `[3..6]` | 4.551e-3 | 4.188e-4 | 1.838e-3 | 0.092 |
| 路线 B 450 iter | `[100..103]` | 4.496e-3 | 3.941e-4 | 1.848e-3 | 0.088 |
| 路线 B 450 iter | `[1800..1803]` | 4.587e-3 | 2.936e-4 | 1.817e-3 | 0.064 |
| 路线 B ep6 | `[3..6]` | **0.1577** | **0.0086** | 0.0265 | 0.055 |
| 路线 B ep6 | `[100..103]` | **0.1755** | **0.0073** | 0.0363 | 0.041 |
| 路线 B ep6 | `[1800..1803]` | **0.1534** | **0.0109** | 0.0264 | 0.071 |

三点必须分开读，否则会得出互相矛盾的结论：

1. **绝对量级大幅上升（好）**：`shuffle_z` head 从 3e-4–4e-4 提升到 **7.3e-3–1.1e-2**，
   约 **20–25 倍**；`zero_z` 从 ~4.5e-3 提升到 ~1.6e-1，约 **35 倍**。
   按用户给的 ep6 继续门槛（`shuffle_z` head ≥ 0.2%），**三窗口全部通过**
   （0.73%、0.73%、1.09%），且窗口间呈同一量级。最严格的论文最终门槛（≥2%）
   仍差约 **2 倍**（最好窗口 1.09%）。
2. **比值口径反而下降（中性偏负）**：`shuffle/zero` 从 0.064–0.092 回落到 0.041–0.071。
   也就是说，绝对响应变大主要来自**公共偏置项也一起变大**，而不是样本判别性变强。
   这与路线 A 的结论同构（Xavier 曾把两者同倍放大）——只不过这次是 ep6 的检测损失
   把整个 z 通路推强了。
3. **`prior_only` 不再明显低于 posterior**：450 iter 时 prior_only head 是 posterior 的
   0.40×；ep6 时 prior_only head（2.6e-2–3.6e-2）相对 `normal`（=1.0）虽然仍低，
   但相对 zero_z（1.6e-1）只有 0.17–0.21×。**posterior 校正仍然有正向价值**
   （prior_only ≠ normal），但没有形成「prior 明显不足、posterior 补足」的强分离。

结论：ep6 的机制读数**支持「z 已经进入检测通路且被放大」，不支持「z 已经携带强样本
判别信息」**。这两件事必须分开陈述。

### 56.6 ep6 source 探针（`stride=700`，indices `[3,703,1403]`）

| 张量 | 路线 A 组 4 | 路线 B 450 iter | 路线 B ep6 | 读法 |
|---|---:|---:|---:|---|
| `feat_pooled_raw` specific | — | — | 13.53% | 输入侧底色占比仍然高 |
| `e_t_pooled_raw` specific | — | — | 25.66% | encoder 输出仍有 1/4 样本特异 |
| `e_pooled` specific | 14.19% | 7.75% | 25.66% | ep6 明显高于 450 iter |
| `h_t` specific | — | 0.86% | 12.19% | GRU 状态不再是近乎常量 |
| `mu_p` specific | 0.49% | 1.71% | 7.96% | prior 也获得了样本性 |
| `mu_q` specific | 1.79% | 3.93% | **11.32%** | +2.9 倍于 450 iter |
| `obs_prior_e_hat` specific | — | 1.80% | 24.52% | obs_prior 拟合较快 |
| `innovation` specific | — | 14.27% | **43.51%** | 仍高于 `e_pooled` ✅ |
| **`correction (= delta_mu)` specific** | 12.11% | 13.51% | **39.32%** | ≥10% ✅（远超门槛） |
| correction RMS | 0.06579 | 0.05739 | 0.28682 | 绝对幅度变大 5 倍 |
| `z_proj` weight RMS | 0.0827 | 0.0868 | 0.0868 | 未再被推大 |

门槛逐条：

| 门槛 | 要求 | ep6 实测 | 结论 |
|---|---|---|---|
| `innovation` specific > `e_pooled` specific | 是 | 43.51% > 25.66% | **通过** |
| `delta_mu` specific | ≥ 10% | 39.32% | **通过** |
| `correction²` 平台（不单调塌缩） | 是 | 0.1489→0.0596 仍单调下降 | **未过** |
| `prior_only` head 明显低于 posterior | 是 | 0.0265 vs 1.0 | 通过（但强度弱于 450 iter） |

`mu_q_e_only` 的样本特异占比 9.00%，`mu_q_h_only` 为 10.87%，两者接近，且
`mu_q_removal_of_h_rel_l2=0.259` 大于 `mu_q_removal_of_e_rel_l2=0.165` ——
**posterior 对 h_t 的依赖反而强于对观测的依赖**。这提示 ep6 的高 `correction` 特异占比
里，有一部分是 h_t 递归状态带来的（h_t specific 12.19%），不全是当前帧观测的贡献。
这是后续消融需要继续分离的点。

### 56.7 结论与下一步

1. **AP：按预登记规则止损，但效应量不显著。** ep6 `31.4895` vs no2d_igdr seed0
   `34.0243`，差 `−2.5348`，超过 2.0 阈值。同时必须记录：这是单 seed、单 epoch、
   与不同代码版本基线的对照，且三 seed 基线在 ep12-16 的标准差本身就有 1.35 点，
   0.53 点的超阈不能当作强证伪。真正可靠的结论要等 `z_prev` 消融与更长窗口。
2. **机制：三个正向变化都是真的。** `innovation` 43.51% > `e_pooled` 25.66%；
   `delta_mu` 39.32%；ep6 `shuffle_z` head 0.73%–1.09%，比 450 iter 高 20–25 倍且
   三窗口一致，已越过用户给的 ep6 继续门槛 0.2%。这支持「Kalman 式 predict-correct
   让 posterior 结构上更合理」这一动机。
3. **机制：判别性比例没有同步改善。** `shuffle/zero` 比值回落到 0.041–0.071，
   即绝对放大主要来自公共偏置。`correction²` 到 ep6 仍单调下降、未形成平台。
   论文里不能只报「shuffle head 涨了 20 倍」，必须同时报比值与平台这两项。
4. **归因：`h_t` 与观测的依赖不均衡。** `mu_q_removal_of_h_rel_l2`（0.259）>
   `mu_q_removal_of_e_rel_l2`（0.165），说明 posterior 对 GRU 状态的依赖强于当前
   观测。`h_t` 本身的样本特异占比已升到 12.19%，因此「correction 特异」中混入了
   递归状态成分；这与 `z_prev` 先 normalize 再进 GRU 的设计直接相关。
5. **下一步（按预登记分支执行）**：先做 **`z_prev` L2 normalize 的 450 iter 单变量
   消融**（`F.normalize(pooled_z, dim=1)` 一行，`rssm_fusion.py:1448`）。
   要观察：`h_t` specific 是否继续上升、`mu_q_removal_of_h/e` 比例是否改变、
   `correction²` 是否形成平台、`shuffle/zero` 比值是否上升。只有这组消融给出正向
   证据，才考虑路线 C（Object-Centric Latent Supervision）。
6. **仍未做**：路线 C、路线 D 的去塌缩损失、路线 E 的 KL balancing + 按 cell 聚合的
   free-bits。三者继续后置，理由与第 55 节一致：`raw KL` 全程有界，KL 不是当前瓶颈。

### 56.8 复现命令

```bash
# 启动（24e 日程，跑到 ep6 验证落盘后停）
cd /home/lurui/workspace/R4Det && source .envrc
tmux new-session -d -s routeB_ep6 "cd /home/lurui/workspace/R4Det && source .envrc && \
  CUDA_VISIBLE_DEVICES=5,6,7 SEED=0 bash tools/dist_train.sh \
  configs/r4det/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_lowdim_z_innovation.py \
  3 --seed 0 --deterministic \
  --work-dir /data/lurui/work_dirs/lowdim_z_innovation_routeB_3x2x2_24e_seed0 \
  > /tmp/lowdim_routeB_ep6.log 2>&1"

# ep6 三窗口反事实诊断
CFG=configs/r4det/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_lowdim_z_innovation.py
CKPT=/data/lurui/work_dirs/lowdim_z_innovation_routeB_3x2x2_24e_seed0/epoch_6.pth
CUDA_VISIBLE_DEVICES=5 python tools/diagnose_z_utilization.py --config $CFG --checkpoint $CKPT \
  --start-index 3    --limit 4 --batch-size 4 --shuffle-offset 2  --output /tmp/z_util_routeB_ep6_w3.json
CUDA_VISIBLE_DEVICES=6 python tools/diagnose_z_utilization.py --config $CFG --checkpoint $CKPT \
  --start-index 100  --limit 4 --batch-size 4 --shuffle-offset 2  --output /tmp/z_util_routeB_ep6_w100.json
CUDA_VISIBLE_DEVICES=7 python tools/diagnose_z_utilization.py --config $CFG --checkpoint $CKPT \
  --start-index 1800 --limit 4 --batch-size 4 --shuffle-offset -1 --output /tmp/z_util_routeB_ep6_w1800.json

# source 探针
CUDA_VISIBLE_DEVICES=5 python tools/probe_posterior_source.py --config $CFG --checkpoint $CKPT \
  --start-index 3 --limit 4 --stride 700 --output /tmp/post_source_routeB_ep6_stride700.json
```

---

## 57. 路线 B 后续：`z_prev` L2 normalize 消融（450 iter 机制冒烟，2026-09-28）

### 57.0 状态与判定摘要

第 56 节的 ep6 门控判定 AP 差距 `−2.5348 > 2.0`，按预登记分支**不叠加路线 C**，
先做 `z_prev` L2 normalize 的 450 iter 单变量消融。本节是这一步的机制冒烟结果。

**一句话结论：消融方向正确，四个观察点全部正向 —— `correction²` 首次形成平台
（最后 100 iter 稳定在 0.0054）、`mu_q` 对观测的依赖首次超过对 `h_t` 的依赖、
`shuffle/zero` 比值翻倍至 0.084–0.129。但绝对量级（`shuffle_z` head 2–3e-4）
仍未进入有意义的检测区间，所以下一步是 rawnorm 的 ep6 AP 对照，而不是直接续训。**

| 观察点 | 第 56 节给出的预期 | rawnorm 450 iter 实测 | 结论 |
|---|---|---|---|
| `h_t` specific | 是否继续上升 | **1.73%**（路线 B 450 iter 为 0.86%，路线 B ep6 为 12.19%） | 相对 normalize 冒烟 +2.0 倍，但仍远低于 ep6 |
| `mu_q_removal_of_h/e` 比例 | 是否改变 | h/e 由 1.53（ep6）转为 **0.87**（h 0.276 < e 0.315） | **方向反转，通过** |
| `correction²` 平台 | 是否形成 | 最后 100 iter **0.0055→0.0054**，最后 200 iter 0.0057→0.0054 | **首次形成平台，通过** |
| `shuffle/zero` 比值 | 是否上升 | **0.084–0.129**（normalize 冒烟为 0.064–0.092） | **上升，通过** |
| `innovation` specific > `e_pooled` specific | 保持 | 36.95% > 7.62% | 通过 |
| `delta_mu` specific ≥ 5% | 保持 | 30.97% | 通过 |
| `shuffle_z` head 绝对值 | — | 1.96e-4–3.12e-4 | 仍低（normalize 冒烟 2.9e-4–4.2e-4，量级相同） |

**判定：机制层面 4/4 正向，但绝对判别性没有进入可用区间。** 消融本身证明
「`z_prev` 的 L2 normalize 确实是压制样本信息的因素之一」，不足以证明它足以让
z 承担检测任务。因此按协议进入下一步：**rawnorm 配置跑 ep6，取 AP 读数与机制读数**，
与第 56 节路线 B ep6 和 no2d_igdr seed0 ep6 **三方对照**。

### 57.1 配置与唯一变量

| 项 | 值 |
|---|---|
| 配置 | `configs/r4det/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_lowdim_z_innovation_rawnorm.py` |
| 冒烟 | `me_rssm/sanity/lowdim_z_innovation_rawnorm_smoke_450iter.py` |
| 对照 | 路线 B：`..._lowdim_z_innovation.py`（第 55/56 节） |
| **唯一变量** | `z_prev_normalize: True → False`（`rssm_fusion.py:1448`） |
| 其余不变 | `posterior_struct='innovation'`、`obs_pred_loss_weight=0.05`、`posterior_obs_mode='scaled_raw'`、`posterior_obs_scale=0.1`、`z_proj_init='xavier'`、`readout_mode='spatial'`、`recon_target_mode='temporal_delta'`、`recon_input_mode='posterior_correction'`、`direct_correction_readout=True`、`future_loss_weight=0`、`recon_loss_weight=0.1`、`kl_scale/free_nats/min_std/init_std/latent_size` 全部继承 |

实现 commit `d895f10`。新增参数 `z_prev_normalize`（**默认 `True`，与此前所有 run 逐位一致**）、
非 bool 校验，并在 forward 中按开关选择 `F.normalize(pooled_z, dim=1)` 或 `pooled_z`。
新增 3 条单测（默认值、非 bool 拒绝、raw 分支确实把未归一化的幅值送进 GRU），
RSSM 全量 **58 条单测通过**。

设计取舍：只去掉**幅值归一化**，保留 `z_prev` 作为 GRU 输入的用法与通道方向。
这与「换掉 recurrence 结构」是两件事，本消融刻意只动一行，避免和第 56 节暴露的
`h_t` 依赖问题混在一起。

### 57.2 运行事实

| 项 | 值 |
|---|---|
| work_dir / 日志 | `/tmp/lowdim_z_rawnorm_smoke_450iter` / `/tmp/lowdim_rawnorm450.log` |
| 硬件 | GPU 5/6/7，3 卡 DDP，seed 0，deterministic |
| 迭代器 | `IterBasedRunner`，450 iter，余弦退火到 `min_lr_ratio=1e-5` |
| 显存 / 速度 | `7436 MiB` / 约 `1.08 s/iter` |
| 起止 | 2026-09-28 02:22:50 UTC 启动，02:33 前后完成，自动退出 |
| 产物 | `iter_450.pth`、`latest.pth` |
| val | 无（`evaluation.interval=4500`），故无区间均值/全轮峰值/best saved |

### 57.3 训练曲线（每 50 iter）

| iter | recon MSE | correction² | innovation² | obs_pred | raw KL | clamped |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 1.2228 | 0.0161 | 0.0152 | 0.0076 | 0.2068 | 0.5286 |
| 100 | 1.1683 | 0.0107 | 0.0102 | 0.0051 | 0.1347 | 0.6221 |
| 150 | 1.1361 | 0.0084 | 0.0085 | 0.0043 | 0.1046 | 0.6812 |
| 200 | 1.1148 | 0.0071 | 0.0074 | 0.0037 | 0.0879 | 0.7242 |
| 250 | 1.1011 | 0.0062 | 0.0066 | 0.0033 | 0.0770 | 0.7567 |
| 300 | 1.0936 | 0.0057 | 0.0061 | 0.0030 | 0.0709 | 0.7781 |
| 350 | 1.0888 | 0.0055 | 0.0058 | 0.0029 | 0.0674 | 0.7917 |
| 400 | 1.0861 | **0.0054** | 0.0058 | 0.0029 | 0.0658 | 0.7980 |
| 450 | 1.0852 | **0.0054** | 0.0058 | 0.0029 | 0.0657 | 0.7985 |

与路线 B 450 iter 冒烟逐点对照：

| iter | recon B | recon rawnorm | corr² B | corr² rawnorm | raw KL B | raw KL rawnorm |
|---:|---:|---:|---:|---:|---:|---:|
| 50 | 1.2213 | 1.2228 | 0.0097 | 0.0161 | 0.1240 | 0.2068 |
| 150 | 1.1348 | 1.1361 | 0.0059 | 0.0084 | 0.0733 | 0.1046 |
| 250 | 1.0984 | 1.1011 | 0.0047 | 0.0062 | 0.0561 | 0.0770 |
| 350 | 1.0853 | 1.0888 | 0.0042 | 0.0055 | 0.0483 | 0.0674 |
| 450 | 1.0812 | 1.0852 | **0.0041** | **0.0054** | 0.0466 | 0.0657 |

读数：

1. **`correction²` 首次不再单调塌缩到底。** 路线 B 在 iter 250→450 从 0.0047 滑到
   0.0041（−13%），rawnorm 在 iter 300→450 从 0.0057 到 0.0054（**−5%，最后
   100 iter 完全持平在 0.0054**）。这正是第 56 节要求「不得继续单调塌缩，至少形成
   平台」的条件，在 450 iter 尺度上**首次成立**。
2. **recon 轨迹几乎重合**（1.2228→1.0852 vs 1.2213→1.0812，末端差 0.4%），说明
   rawnorm 没有通过「让重建变简单」来制造平台，改善来自 recurrent 通路本身。
3. **raw KL 系统性偏高**（末端 0.0657 vs 0.0466，+41%）但仍全程有界，未出现爆炸；
   `clamped_ratio` 末端 0.799（normalize 0.871），即更多 latent cell 处于「未被
   free-bits 钳死」的状态 —— 与 correction 保持更大绝对值一致。
4. **`obs_pred` 末端 0.0029 vs 0.0029（路线 B 从 0.0021 略升）**：`obs_prior` 的
   拟合难度没有因为 rawnorm 而下降，符合预期（它只吃 `h_t`）。

### 57.4 Source 探针（`stride=700`，indices `[3,703,1403]`）

| 张量 | 路线 B 450 iter | **rawnorm 450 iter** | 变化 |
|---|---:|---:|---|
| `feat_pooled_raw` specific | — | 19.62% | 输入侧参照 |
| `e_t_pooled_raw` specific | — | 7.62% | 与路线 B 的 7.75% 同量级 |
| `e_pooled` specific | 7.75% | 7.62% | 无变化（同一 `scaled_raw` 路径） |
| `h_t` specific | 0.86% | **1.73%** | **+2.0 倍** |
| `mu_p` specific | 1.71% | 1.47% | 略降 |
| `mu_q` specific | 3.93% | **5.16%** | **+31%** |
| `obs_prior_e_hat` specific | 1.80% | 0.90% | 略降 |
| `innovation` specific | 14.27% | **36.95%** | **+2.6 倍** |
| `correction (= delta_mu)` specific | 13.51% | **30.97%** | **+2.3 倍** |
| correction RMS | 0.05739 | 0.02895 | −50%（**绝对幅度变小**） |
| `z_proj` weight RMS | 0.0868 | 0.0834 | 稳在同一尺度 |

**最关键的一行是依赖反转**：

| 依赖方向 | 路线 B ep6 | **rawnorm 450 iter** |
|---|---:|---:|
| `mu_q_removal_of_e_rel_l2`（去掉观测的影响） | 0.1653 | **0.3152** |
| `mu_q_removal_of_h_rel_l2`（去掉 h_t 的影响） | 0.2589 | **0.2756** |
| h/e 比值 | 1.57 | **0.874** |

第 56 节把 ep6 的高 correction 特异占比归因于「`h_t` 递归状态成分混入，posterior 对
h 的依赖强于对观测」。**去掉 `z_prev` 的 L2 normalize 后，这个比例反转到 0.874**，
即 posterior 现在更多由**当前帧观测**驱动。这正是消融假设所预测的方向，是本轮最强的
单条证据。

同时要诚实标注：`correction` 的**样本特异占比翻了 2.3 倍，但 RMS 减半**。也就是说
rawnorm 让 correction 更像「样本身份」，同时整体幅度更小。这解释了为什么
`shuffle/zero` 比值上升、而 `shuffle_z` 的**绝对** head 响应没有同步变大。

### 57.5 三窗口反事实诊断

`tools/diagnose_z_utilization.py`，`iter_450.pth`，每窗口 4 条序列：

| 窗口 | zero_z feature | zero_z head | shuffle_z feature | shuffle_z head | prior_only feature | prior_only head | shuffle/zero |
|---|---:|---:|---:|---:|---:|---:|---:|
| `[3..6]` | 0.0660 | 2.420e-3 | 0.0077 | **3.120e-4** | 0.0207 | 1.125e-3 | **0.1289** |
| `[100..103]` | 0.0737 | 2.356e-3 | 0.0077 | **2.830e-4** | 0.0230 | 1.084e-3 | **0.1201** |
| `[1800..1803]` | 0.0557 | 2.332e-3 | 0.0042 | **1.960e-4** | 0.0159 | 9.092e-4 | **0.0841** |

与路线 B 450 iter 三窗口对照：

| 运行 | zero_z head | shuffle_z head | prior_only head | shuffle/zero |
|---|---:|---:|---:|---:|
| 路线 B 450 iter（三窗口范围） | 4.496e-3–4.587e-3 | 2.936e-4–4.188e-4 | 1.817e-3–1.848e-3 | 0.064–0.092 |
| **rawnorm 450 iter** | 2.332e-3–2.420e-3 | 1.960e-4–3.120e-4 | 9.090e-4–1.125e-3 | **0.084–0.129** |

读数：

1. **`shuffle/zero` 比值三窗口全部提升**（0.064–0.092 → 0.084–0.129，最好窗口 +40%）。
   这是本轮消融的核心机制收益，且三个窗口方向一致。
2. **`zero_z` 绝对响应下降约一半**（4.5e-3 → 2.4e-3），与 correction RMS 减半一致。
   即公共偏置被削弱，而不是样本判别性被同倍放大 —— 比值上升是**分子分母同时下降、
   分子下降更慢**的结果。论文口径必须同时报这两个数。
3. **`prior_only` 同步下降到约 1e-3**（0.40× → 0.46× 于 zero_z），说明 prior 与
   posterior 的相对关系没有被消融破坏，posterior 校正仍然产生可观测差异。
4. **绝对量级仍是瓶颈**：`shuffle_z` head 1.96e-4–3.12e-4 与路线 B 450 iter
   （2.9e-4–4.2e-4）在同一量级，离 2% 最终门槛仍差约 60–100 倍。本轮改善的是
   **结构性质**（平台 + 依赖反转 + 比值），不是**可用性**。

### 57.6 结论与下一步

1. **消融假设被证实，而且是四个观察点同向。** `z_prev` 的 L2 normalize 确实是压制
   样本信息的因素之一：去掉后 `correction²` 首次形成平台、`h_t`/`mu_q`/`innovation`/
   `correction` 的样本特异占比全部上升、`mu_q` 对观测的依赖首次超过对 `h_t` 的依赖、
   `shuffle/zero` 比值三窗口全部提升。
2. **但它不是充分条件。** `shuffle_z` head 绝对量级仍在 1e-4–3e-4，比 2% 最终门槛
   低约两个数量级；`h_t` specific 仍只有 1.73%（路线 B ep6 反而有 12.19%，但那是
   经过 6 epoch 大量数据曝光后的结果，两者不能直接比）。消融解决的是「幅值被抹掉」，
   没有解决「上游信号本身薄」。
3. **本节的正确用途**：它是第 56 节预登记分支的**执行结果**，回答「该不该动 `z_prev`」
   —— 答案是**该动，且要保留**。但它不构成「路线 B + rawnorm 可以进论文主表」的证据。
4. **下一步（已启动）**：rawnorm 配置的 **ep6 AP 对照**（work_dir
   `/data/lurui/work_dirs/lowdim_z_innovation_rawnorm_3x2x2_24e_seed0`，GPU 5/6/7，
   seed 0，24e 日程跑到 ep6 验证落盘后停）。判定沿用第 56 节协议，并做**三方对照**：
   rawnorm / 路线 B / no2d_igdr seed0，比较项为 ep6 Overall 3D moderate、逐类别构成、
   `shuffle_z` head 绝对值与比值、`correction²` 是否继续保持平台。
5. **仍未做**：路线 C（Object-Centric Latent Supervision）、路线 D 的去塌缩损失、
   路线 E 的 KL balancing + 按 cell 聚合的 free-bits。三者继续后置；`raw KL` 目前
   0.0657 仍远未爆炸。

### 57.7 复现命令

```bash
# 450 iter 机制冒烟
cd /home/lurui/workspace/R4Det && source .envrc
tmux new-session -d -s routeB_rawnorm450 "cd /home/lurui/workspace/R4Det && source .envrc && \
  CUDA_VISIBLE_DEVICES=5,6,7 SEED=0 bash tools/dist_train.sh \
  me_rssm/sanity/lowdim_z_innovation_rawnorm_smoke_450iter.py 3 --seed 0 --deterministic \
  --work-dir /tmp/lowdim_z_rawnorm_smoke_450iter > /tmp/lowdim_rawnorm450.log 2>&1"

# source 探针
CFG=me_rssm/sanity/lowdim_z_innovation_rawnorm_smoke_450iter.py
CKPT=/tmp/lowdim_z_rawnorm_smoke_450iter/iter_450.pth
CUDA_VISIBLE_DEVICES=5 python tools/probe_posterior_source.py --config $CFG --checkpoint $CKPT \
  --start-index 3 --limit 4 --stride 700 --output /tmp/post_source_rawnorm450_stride700.json

# 三窗口反事实诊断
CUDA_VISIBLE_DEVICES=5 python tools/diagnose_z_utilization.py --config $CFG --checkpoint $CKPT \
  --start-index 3    --limit 4 --batch-size 4 --shuffle-offset 2  --output /tmp/z_util_rawnorm450_w3.json
CUDA_VISIBLE_DEVICES=6 python tools/diagnose_z_utilization.py --config $CFG --checkpoint $CKPT \
  --start-index 100  --limit 4 --batch-size 4 --shuffle-offset 2  --output /tmp/z_util_rawnorm450_w100.json
CUDA_VISIBLE_DEVICES=7 python tools/diagnose_z_utilization.py --config $CFG --checkpoint $CKPT \
  --start-index 1800 --limit 4 --batch-size 4 --shuffle-offset -1 --output /tmp/z_util_rawnorm450_w1800.json
```

---

## 58. rawnorm ep6 门控：三方对照（rawnorm / 路线 B / no2d_igdr seed0，2026-09-28）

### 58.0 状态与判定摘要

第 57 节的 450 iter 消融给出 4/4 正向机制读数，按协议进入 ep6 AP 对照。本节是
rawnorm 配置的正式 ep6 结果，并与路线 B（第 56 节）和 no2d_igdr seed0 做三方对照。

**一句话结论：AP 差距收窄到 −1.1263（落入预登记的 1–2 区间）、机制指标全面优于
路线 B（`shuffle_z` head 0.84%–1.38%、比值 0.077–0.125），按协议进入路线 C
（Object-Centric Latent Supervision）。但仍有两条必须记录的反向证据：
`correction²` 在正式训练里重新变成单调塌缩（与 450 iter 的「平台」结论相反），
且 `mu_q` 对 `h_t` 的依赖再次超过对观测的依赖（与 450 iter 的反转结论相反）。**

| 检查项 | 要求 | rawnorm ep6 | 路线 B ep6 | no2d_igdr seed0 ep6 | 结论 |
|---|---|---|---|---|---|
| Overall 3D moderate（同 epoch） | 差距 ≤2.0 继续 | **32.8980** | 31.4895 | 34.0243 | Δ_base **−1.1263**，**通过止损** |
| `shuffle_z` head（ep6 继续门槛 0.2%） | ≥0.2% | **0.84%–1.38%** | 0.73%–1.09% | — | **通过，且优于路线 B** |
| `shuffle/zero` 比值 | 高于路线 B 的 0.041–0.071 | **0.077–0.125** | 0.041–0.071 | — | **通过，三窗口一致** |
| `innovation` specific > `e_pooled` specific | 是 | 46.70% > 16.32% | 43.51% > 25.66% | — | **通过** |
| `delta_mu` specific | ≥10% | 38.01% | 39.32% | — | **通过** |
| `correction²` 形成平台 | 是 | ep1→ep6 逐 epoch 均值：0.2218 → 0.0685（**单调下降**） | 0.1489→0.0596 | — | **未过（与 450 iter 结论相反）** |

**预登记分支判定**：

```
AP 差距 1–2，机制成立但 shuffle_z 仍明显不足（1.38% < 2%）
  → 以 B ep6 为对照，实施路线 C（Object-Centric Latent Supervision）
```

注意这次是 **rawnorm ep6** 而非 B ep6 作为路线 C 的对照基线：rawnorm 在 ep6 上
同时拥有更小的 AP 差距（−1.13 vs −2.53）和更强的机制读数（`shuffle_z` head 最大值
1.38% vs 1.09%、比值 0.125 vs 0.071），是更合适的起跳点。

### 58.1 配置与唯一变量

| 项 | 值 |
|---|---|
| 配置 | `configs/r4det/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_lowdim_z_innovation_rawnorm.py` |
| 相对路线 B 的唯一差异 | `z_prev_normalize: True → False` |
| 训练日程 | **`max_epochs=24` 的 CosineAnnealing 原日程未改**，ep1-6 的 LR/KL warmup 与两个对照逐轮对齐 |
| val 口径 | 训练内 EvalHook + evaluation.pipeline，`val.samples_per_gpu=4`，`evaluation.interval=1` |
| 落盘 | `checkpoint_config.interval=2` |

三个 run 的对照关系（本节全部按此表口径）：

| run | `posterior_struct` | `z_prev_normalize` | 其他 |
|---|---|---|---|
| no2d_igdr seed0 | 无低维 latent（对照基线） | — | FG-FULL 全监督，无 2D/IGDR |
| 路线 B ep6 | `innovation` | `True` | 路线 A 组 4 + obs_pred |
| **rawnorm ep6** | `innovation` | **`False`** | 与路线 B 完全相同 |

### 58.2 运行事实

| 项 | 值 |
|---|---|
| work_dir | `/data/lurui/work_dirs/lowdim_z_innovation_rawnorm_3x2x2_24e_seed0` |
| 日志 | `20260928_023*.log(.json)`、`/tmp/lowdim_rawnorm_ep6.log` |
| 硬件 | GPU 5/6/7，3 卡 DDP，seed 0，deterministic |
| 起止 | 2026-09-28 02:37:37 UTC 启动；ep6 val 于 07:04:50 UTC 写完，看门狗随后停止训练 |
| 显存 / 速度 | `7438 MiB` / 约 `1.05 s/iter`（与路线 B ep6 同量级） |
| 落盘 | `epoch_2.pth` / `epoch_4.pth` / `epoch_6.pth`，`latest.pth -> epoch_6.pth` |
| val 记录 | ep1-6 各一条，共 6 条 |
| 稳定性 | 无发散：`correction²` 在 ep1 iter 250 冲到峰值 0.5457 后回落并在 ep1 后半段稳定在 0.24–0.28，ep1 全 epoch 均值 0.2218，此后逐 epoch 单调降到 0.0685；`grad_norm` 全程 5.75–15.87，`raw KL` 全程 0.0345–1.3713，两者都有界、无 NaN |

**关于早期尖峰**：去掉归一化后，`z_prev` 的幅值不再被裁剪，ep1 前半段出现
`correction²` 0.0253→0.5457 的快速上升（约 22 倍）与 `grad_norm` 8.1→15.9 的抬升。
它在 ep1 iter 300 后自行回落到 0.24–0.26，ep1 全 epoch 均值 0.2218，之后逐 epoch
单调下降（0.1288 → 0.0685），全程没有发散或 NaN。
这是 rawnorm 与路线 B 在动力学上最显著的区别，必须在复现时注意：
rawnorm 不是「更温和」，而是「前 250 iter 更激进、之后稳定在更高幅值」。

### 58.3 完整逐 epoch 曲线

主指标（`pts_bbox/KITTI/*`）：

| epoch | Overall 3D | Overall BEV | Car strict | Cyclist loose | Ped loose | Truck strict |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20.1118 | 27.6784 | 28.2763 | 27.1383 | 16.2622 | 8.7703 |
| 2 | 28.1775 | 35.9377 | 34.7473 | 38.9211 | 18.2066 | 20.8352 |
| 3 | 28.0411 | 34.2050 | 37.5963 | 42.3158 | 20.0612 | 12.1911 |
| 4 | 27.7140 | 36.5433 | 32.1546 | 36.1416 | 24.1069 | 18.4528 |
| 5 | **33.8487** | 39.6282 | 43.6507 | 46.0788 | 24.5194 | 21.1458 |
| 6 | 32.8980 | 39.3133 | 45.6647 | 41.8833 | 22.2281 | 21.8159 |

三方同 epoch 对照（Overall 3D moderate）：

| epoch | no2d_igdr seed0 | 路线 B | **rawnorm** | Δ vs base | Δ vs B |
|---:|---:|---:|---:|---:|---:|
| 1 | 15.2640 | 20.8115 | 20.1118 | **+4.8478** | −0.6997 |
| 2 | 26.3716 | 22.5532 | 28.1775 | **+1.8059** | **+5.6243** |
| 3 | 31.6591 | 28.1833 | 28.0411 | −3.6180 | −0.1422 |
| 4 | 32.8506 | 25.0003 | 27.7140 | −5.1366 | **+2.7137** |
| 5 | 34.0827 | 34.8551 | 33.8487 | −0.2340 | −1.0064 |
| 6 | 34.0243 | 31.4895 | **32.8980** | **−1.1263** | **+1.4085** |

逐类别（loose 口径）：

| epoch | Car loose | Truck loose | Cyc loose | Ped loose |
|---:|---:|---:|---:|---:|
| 1 | 49.1750 | 32.0069 | 27.1383 | 16.2622 |
| 2 | 55.3444 | 43.0059 | 38.9211 | 18.2066 |
| 3 | 56.3237 | 37.5297 | 42.3158 | 20.0612 |
| 4 | 59.1984 | 36.2573 | 36.1416 | 24.1069 |
| 5 | 59.6940 | 43.0993 | 46.0788 | 24.5194 |
| 6 | 62.5624 | 43.5216 | 41.8833 | 22.2281 |

逐类别 strict 口径：

| epoch | Car strict | Truck strict | Cyc strict | Ped strict |
|---:|---:|---:|---:|---:|
| 1 | 28.2763 | 8.7703 | 6.8277 | 0.1623 |
| 2 | 34.7473 | 20.8352 | 18.1247 | 0.0719 |
| 3 | 37.5963 | 12.1911 | 21.0104 | 0.1260 |
| 4 | 32.1546 | 18.4528 | 18.5261 | 0.0316 |
| 5 | 43.6507 | 21.1458 | 22.4000 | 0.0501 |
| 6 | 45.6647 | 21.8159 | 17.9584 | 0.0349 |

**窗口均值（ep1-6，n=6，实际跨度 1-6，非标准 ep12-16 窗口）**：

| 口径 | Overall 3D | Overall BEV | Car strict | Cyclist loose | Ped loose | Truck strict |
|---|---:|---:|---:|---:|---:|---:|
| rawnorm ep1-6 等权均值 | 28.4652 | 35.5510 | 37.0150 | 38.7465 | 20.8974 | 17.2018 |
| 路线 B ep1-6 等权均值 | 27.1488 | 35.4594 | 33.0164 | 40.8937 | 20.2263 | 14.4589 |
| no2d_igdr seed0 ep1-6 等权均值 | 29.0420 | 37.4302 | 39.9218 | 36.7749 | 19.7073 | 19.7642 |
| Δ（rawnorm − 基线） | **−0.5768** | −1.8792 | −2.9068 | +1.9716 | +1.1901 | −2.5624 |
| Δ（rawnorm − 路线 B） | **+1.3164** | +0.0916 | **+3.9986** | −2.1472 | +0.6711 | +2.7429 |

**三口径必填摘要**（本 run 有 6 条 val 记录）：

| 口径 | 值 |
|---|---|
| 区间均值（ep1-6，n=6，实际跨度 1-6；标准 ep12-16 窗口不存在） | Overall 3D moderate **28.4652** |
| 全轮峰值（所有有 val 的 epoch 中最高单点） | **33.8487 @ ep5**（无对应权重） |
| best saved（实际有权重的 epoch 中最高，`interval=2` 故仅 ep2/4/6） | **32.8980 @ ep6** |

### 58.4 训练内诊断（每 epoch 的 mean）

| epoch | recon MSE | correction² | innovation² | obs_pred | raw KL | clamped | grad norm |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.9978 | 0.2218 | 0.0350 | 0.0174 | 0.7222 | 0.1392 | 12.37 |
| 2 | 0.9466 | 0.1288 | 0.0344 | 0.0171 | 0.2513 | 0.3842 | 10.20 |
| 3 | 0.9427 | 0.1033 | 0.0413 | 0.0206 | 0.0745 | 0.7802 | 8.36 |
| 4 | 0.9344 | 0.0855 | 0.0426 | 0.0212 | 0.0521 | 0.8584 | 7.32 |
| 5 | 0.9330 | 0.0757 | 0.0438 | 0.0218 | 0.0428 | 0.8895 | 6.93 |
| 6 | 0.9310 | 0.0685 | 0.0444 | 0.0221 | 0.0374 | 0.9082 | 6.34 |

与路线 B ep6 对照：

| 诊断量 | 路线 B ep6 | rawnorm ep6 | 读法 |
|---|---:|---:|---|
| recon MSE | 0.9266 | 0.9310 | 几乎相同（rawnorm 未靠更简单的重建换取指标） |
| correction² | 0.0596 | **0.0685** | **1.15 倍**，绝对幅值更大 |
| innovation² | 0.0487 | 0.0444 | 同量级 |
| obs_pred | 0.0243 | 0.0221 | 同量级 |
| raw KL | 0.0331 | 0.0374 | 同量级，远未爆炸 |
| clamped | 0.9267 | 0.9082 | 同量级 |
| grad norm | ~8 | 6.34 | 同量级，未失稳 |

**关键反向证据（必须与 450 iter 结论并列记录）**：

450 iter 冒烟里 rawnorm 的 `correction²` 在最后 100 iter **完全持平在 0.0054**，
据此判定「首次形成平台」。但在 ep1-6 的正式训练里它是**单调下降**的：
ep1 0.2218 → ep6 0.0685（−69%），形态与路线 B ep6（0.1489→0.0596）一致。

两者不矛盾但结论不同：**450 iter 的「平台」不是稳定性质，只是短预算下的局部现象。**
第 57 节把它写成「首次形成平台」在当时的数据下是正确的，本节必须显式更正其适用范围：
**在 6 epoch 尺度上，rawnorm 仍未阻止 `correction²` 的单调塌缩。**

### 58.5 rawnorm ep6 三窗口反事实诊断

| 窗口 | zero_z feature | zero_z head | shuffle_z feature | shuffle_z head | prior_only feature | prior_only head | shuffle/zero |
|---|---:|---:|---:|---:|---:|---:|---:|
| `[3..6]` | 0.6844 | 0.1103 | 0.0754 | **1.3783e-2** | 0.1701 | 2.851e-2 | **0.1250** |
| `[100..103]` | 0.8341 | 0.1262 | 0.0645 | **1.0394e-2** | 0.2176 | 3.516e-2 | **0.0824** |
| `[1800..1803]` | 0.8222 | 0.1098 | 0.0547 | **8.4264e-3** | 0.1848 | 2.769e-2 | **0.0768** |

三方对照（head L2 ratio vs normal，三窗口范围）：

| run | zero_z head | shuffle_z head | prior_only head | shuffle/zero |
|---|---:|---:|---:|---:|
| 路线 B 450 iter | 4.50e-3–4.59e-3 | 2.94e-4–4.19e-4 | 1.82e-3–1.85e-3 | 0.064–0.092 |
| 路线 B ep6 | 0.1534–0.1755 | 7.28e-3–1.09e-2 | 2.64e-2–3.63e-2 | 0.041–0.071 |
| **rawnorm ep6** | 0.1098–0.1262 | **8.43e-3–1.38e-2** | 2.77e-2–3.52e-2 | **0.077–0.125** |

读数：

1. **`shuffle_z` head 绝对量级首次进入 1% 区间**：最好窗口 1.3783e-2，三窗口均 ≥8.4e-3。
   论文最终门槛 2% 仍差约 1.5 倍，但这是目前为止最接近的一次。
2. **`shuffle/zero` 比值三窗口全部高于路线 B ep6**（0.125/0.082/0.077 vs 0.055/0.041/0.071）。
   与 450 iter 的方向一致，说明「去掉 normalize 提高判别性占比」在正式尺度上仍然成立。
3. **`zero_z` 绝对响应略低于路线 B ep6**（0.11–0.13 vs 0.15–0.18），即公共偏置被削弱，
   而 `shuffle_z` 同时上升 —— 这次是**分子上升、分母下降**，比 450 iter 的
   「双降、分子降得慢」更强。
4. **`prior_only` 与路线 B 持平**（2.8e-2–3.5e-2），posterior 校正的相对价值没有退化。

### 58.6 rawnorm ep6 source 探针（`stride=700`，indices `[3,703,1403]`）

| 张量 | 路线 B ep6 | **rawnorm ep6** | 变化 |
|---|---:|---:|---|
| `feat_pooled_raw` specific | 13.53% | 15.27% | — |
| `e_t_pooled_raw` specific | 25.66% | 16.32% | 略降 |
| `e_pooled` specific | 25.66% | 16.32% | 略降 |
| `h_t` specific | 12.19% | **10.71%** | 同量级 |
| `mu_p` specific | 7.96% | 7.32% | 同量级 |
| `mu_q` specific | 11.32% | 10.61% | 同量级 |
| `obs_prior_e_hat` specific | 24.52% | 13.19% | 略降 |
| `innovation` specific | 43.51% | **46.70%** | 略升 |
| `correction (= delta_mu)` specific | 39.32% | **38.01%** | 同量级 |
| correction RMS | 0.28682 | 0.25869 | 略降 |
| `z_proj` weight RMS | 0.0868 | 0.0856 | 同尺度 |

**第二条反向证据**：

| 依赖方向 | 路线 B ep6 | rawnorm 450 iter | **rawnorm ep6** |
|---|---:|---:|---:|
| `mu_q_removal_of_e_rel_l2` | 0.1653 | **0.3152** | 0.1325 |
| `mu_q_removal_of_h_rel_l2` | 0.2589 | 0.2756 | **0.1932** |
| h/e 比值 | 1.57 | **0.874** | **1.46** |

450 iter 里观测到的「依赖反转」（h/e 0.874）在 ep6 尺度上**回归到 1.46**，与路线 B ep6
的 1.57 基本一致。即：**去掉 `z_prev` normalize 并不能在正式训练里让 posterior 持续
以当前观测为主。** 第 57 节的「依赖反转」结论同样只在 450 iter 尺度成立。

把两条反向证据放在一起看，一个一致的解读是：**rawnorm 改变的是 recurrent 通路的
幅值尺度（correction RMS 与 KL 都放大数倍），而不是信息流向。** ep6 的机制改善
（`shuffle_z` 绝对值进入 1%、比值上升）来自更大的幅值，而非「观测替代了递归状态」。
这与路线 A 的教训形式相同：**放大信号 ≠ 改变信息结构。**

### 58.7 结论与下一步

1. **AP：差距从 −2.53 收窄到 −1.13，通过止损并落入 1–2 区间。** 这是本任务中低维
   z 变体第一次在 ep6 把与基线的差距压到 1.2 点以内（历史对照：第 50 节低维无重建
   ep8-12 −6.51、第 52 节 recon+h_t 修复 ep8-12 −6.90、路线 B ep6 −2.53）。
   优点必须写清楚：**这是目前唯一一个既没有崩掉 AP、又让 `shuffle_z` head 进入 1% 的配置。**
2. **机制：三项指标全面优于路线 B ep6。** `shuffle_z` head 0.84%–1.38%（B 为
   0.73%–1.09%）、`shuffle/zero` 0.077–0.125（B 为 0.041–0.071）、
   `innovation` 46.70%（B 为 43.51%）。三窗口方向一致。
3. **但 450 iter 的两条正向结论在 ep6 尺度上不成立，必须更正。** `correction²`
   在 ep1-6 仍单调下降（0.3024→0.2176），`mu_q` 对 `h_t` 的依赖重新超过对观测的依赖
   （h/e 1.46）。**结论：rawnorm 改善的是幅值，不是信息流向。** 第 57 节关于
   「平台」和「依赖反转」的表述仅适用于 450 iter 预算，必须在论文与后续引用中
   按尺度区分，不能当作稳定机制性质。
4. **`shuffle_z` 仍未达论文门槛。** 最好窗口 1.38%，距 2% 差约 1.5 倍。方向明确
   （需要 correction 携带更多观测量而非纯递归状态），所以下一步是路线 C：
   用 object-centric 监督（BEV center heatmap）直接给 correction 一个**当前帧独占、
   h_t 无法预测**的任务。
5. **下一步（按预登记协议）**：以上述 rawnorm ep6 为对照，实施**路线 C
   （Object-Centric Latent Supervision）**：
   `correction (16×16×32) → Conv 32→64 → 4 类 center heatmap`，CenterNet 风格
   focal loss，`latent_object_loss_weight=0.05`（不足则升 0.1），只在当前帧计算，
   推理时删除辅助 head（零额外开销）。判定沿用同一协议，并额外检查
   `correction²` 是否真正停止塌缩 —— 这是 rawnorm 留下的未解决问题。
6. **仍未做**：路线 D 的去塌缩损失、路线 E 的 KL balancing + 按 cell 聚合的 free-bits。
   `raw KL` 目前 0.0374（rawnorm ep6 逐 epoch 均值；全程 0.0345–1.3713）仍然有界，KL 继续后置。

### 58.8 复现命令

```bash
# ep6 正式训练（24e 日程，跑到 ep6 验证落盘后停）
cd /home/lurui/workspace/R4Det && source .envrc
tmux new-session -d -s rawnorm_ep6 "cd /home/lurui/workspace/R4Det && source .envrc && \
  CUDA_VISIBLE_DEVICES=5,6,7 SEED=0 bash tools/dist_train.sh \
  configs/r4det/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_lowdim_z_innovation_rawnorm.py \
  3 --seed 0 --deterministic \
  --work-dir /data/lurui/work_dirs/lowdim_z_innovation_rawnorm_3x2x2_24e_seed0 \
  > /tmp/lowdim_rawnorm_ep6.log 2>&1"

# ep6 三窗口反事实诊断 + source 探针
CFG=configs/r4det/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_lowdim_z_innovation_rawnorm.py
CKPT=/data/lurui/work_dirs/lowdim_z_innovation_rawnorm_3x2x2_24e_seed0/epoch_6.pth
CUDA_VISIBLE_DEVICES=5 python tools/diagnose_z_utilization.py --config $CFG --checkpoint $CKPT \
  --start-index 3    --limit 4 --batch-size 4 --shuffle-offset 2  --output /tmp/z_util_rawnorm_ep6_w3.json
CUDA_VISIBLE_DEVICES=6 python tools/diagnose_z_utilization.py --config $CFG --checkpoint $CKPT \
  --start-index 100  --limit 4 --batch-size 4 --shuffle-offset 2  --output /tmp/z_util_rawnorm_ep6_w100.json
CUDA_VISIBLE_DEVICES=7 python tools/diagnose_z_utilization.py --config $CFG --checkpoint $CKPT \
  --start-index 1800 --limit 4 --batch-size 4 --shuffle-offset -1 --output /tmp/z_util_rawnorm_ep6_w1800.json
CUDA_VISIBLE_DEVICES=5 python tools/probe_posterior_source.py --config $CFG --checkpoint $CKPT \
  --start-index 3 --limit 4 --stride 700 --output /tmp/post_source_rawnorm_ep6_stride700.json
```

---

## 59. 路线 C：Object-Centric Latent Supervision（实现 + 450 iter 机制门控，2026-09-28）

本节记录 **第 58 节预登记分支的执行**。第 58 节三方对照把 rawnorm 的 ep6 AP 差距
定为 −1.1263，落入「AP 差距 1–2、机制成立但 `shuffle_z` 明显不足」区间，按协议
下一步是**以 rawnorm ep6 为对照实施路线 C**，而不是直接续训 ep12，也不是先动
`z_prev`。

### 59.0 状态与判定摘要

| 门控项 | 门槛 | 450 iter 实测 | 判定 |
|---|---|---:|---|
| heatmap 学习信号 | 持续上升 | hit@1 `0.0141 → 0.3988` | **通过** |
| `correction` specific | ≥ 5% | **28.17%** | **通过** |
| `shuffle_z` head | ≥ 2% | **0.58%**（最好窗口） | **未通过**（差约 3.4 倍） |
| `correction²` | 不单调塌缩 | 最后 150 iter `0.0077 → 0.0076 → 0.0076 → 0.0075` | **通过（450 iter 尺度）** |
| recon | 持续下降 | `1.2268 → 1.0682` | 通过 |
| raw KL | 有界 | `0.2209 → 0.0764`，全程 ≤ 1.0 | 通过 |
| ep6 AP（Car/Truck 等） | 不低于基线超 2 点 | **ep6 训练进行中** | 待定 |

**机制门槛部分通过**：object-centric 监督确实给 correction 增加了当前帧独占的
信息（hit@1 升到 0.399、`correction` specific 28%、RMS 翻倍），但**身份信号仍未达
论文门槛**，且 `shuffle/zero` 比值相对 rawnorm **下降**。所有结论都必须在 ep6 AP
出来后才允许下最终判断。

### 59.1 配置与唯一变量

配置：`configs/r4det/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_lowdim_z_innovation_rawnorm_oc.py`
（自 `..._innovation_rawnorm.py` 继承）。相对 rawnorm 对照的**唯一变量**：

```python
latent_object_loss_weight=0.1   # 预登记 0.05 未达梯度门槛后按协议升权
num_object_classes=4
object_head_hidden_dim=64
object_min_radius=1
```

新增模块（`mmdet3d/models/fusion_layers/rssm_fusion.py`）：

```
correction (B,32,16,16) -> Conv3x3(32->64) + ReLU -> Conv1x1(64->4) -> logits
target = bev_centers_to_heatmap(current frame GT, 16x16)   # Gaussian, class-wise max
loss   = CenterNet focal (alpha=2, gamma=4), normalised by #GT centres
```

四条设计约束（都有回归测试）：

1. **辅助头只吃 `posterior_correction`。** `h_t` 能看到则不成立——递归状态可以自己
   解掉目标，pressure 会从 correction 上卸掉。测试 `test_head_consumes_correction_not_h_t`
   用 forward pre-hook 断言 `head_in` 与 `stat_correction_sq` 逐位一致。
2. **只在当前帧计算。** `reset_state()` 清空 target，detector 只在前向当前帧前调用
   `set_current_object_target`；历史帧不带 target。
3. **推理零开销。** `_object_heatmap_loss` 在 `self.training=False` 或权重为 0 时
   直接返回零，检测输出从不读取该 head。
4. **默认权重 0 时逐位不变。** `test_default_weight_zero_is_bitwise_inert` 断言在
   仅换了 target 的情况下 loss 与未设置 target 的基线完全相等；`test_zero_weight_never_runs_head`
   断言 head 的 forward 根本没被调用。

辅助几何工具放在 `mmdet3d/core/utils/gaussian.py`：
`bev_centers_to_heatmap`（纯 torch 光栅化，`H` 索引 x、`W` 索引 y，与全仓库 BEV 张量
约定一致）与 `gaussian_radius_batch`（仓库 `gaussian_radius` 的向量化等价形式；
标量版以 Python `min()` 结尾无法批次化，故单测逐元素断言两者相等）。

### 59.2 运行事实

| 项 | w=0.05（首次冒烟） | **w=0.1（门控冒烟）** |
|---|---|---|
| work_dir | `/tmp/lowdim_z_oc_smoke_450iter` | `/tmp/lowdim_z_oc100_check` |
| 日志 | `/tmp/lowdim_z_oc450.log` | `/tmp/lowdim_z_oc100_check.log` |
| GPU / 卡数 | 5,6,7 / 3 卡 DDP | 5,6,7 / 3 卡 DDP |
| runner | IterBasedRunner 450 iter，无 val | 同左 |
| 显存 | 7443 MiB | 7443 MiB |
| 速度 | ≈1.05 s/iter | ≈1.06 s/iter |
| 落盘 | `iter_450.pth` | `iter_450.pth` |

**为什么先跑 0.05 再升到 0.1。** 预登记原文是「`latent_object_loss_weight=0.05`，
若梯度太弱升到 0.1」。0.05 跑完后直接检查了辅助头参数：

| 参数 | Xavier 初值 | 450 iter @ 0.05 | 变化 |
|---|---:|---:|---|
| `object_head.0.weight` RMS | 0.04811 | 0.04817 | +0.12% |
| `object_head.2.weight` RMS | 0.17462 | 0.17429 | −0.19% |
| `object_head.2.bias` 均值 | −2.19000 | −2.19207 | −0.00207 |

首层 RMS 与初值在 0.1% 以内、末层 bias 几乎不动，**「梯度太弱」条件成立**，因此
按预登记升到 0.1，没有引入任何其它机制。校准检查（head-only overfit probe，冻结
其余参数、把权重当 1.0 用）确认 focal 目标本身是可学的：`prob_at_gt` 在 40 步内从
0.133 升到 0.953。所以 0.05 的失败是**权重太小**，不是损失或接口写错。

### 59.3 训练曲线（每 50 iter，w=0.1）

| iter | `stat_object_hm` | hit@1 | #pos | `stat_correction_sq` | recon | raw KL | clamped | innovation² |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 2.2214 | 0.0141 | 7.96 | 0.0174 | 1.2268 | 0.2209 | 0.5196 | 0.0156 |
| 100 | 2.2081 | 0.0282 | 7.61 | 0.0110 | 1.1704 | 0.1353 | 0.6165 | 0.0103 |
| 150 | 2.1631 | 0.0468 | 7.28 | 0.0092 | 1.1391 | 0.1108 | 0.6639 | 0.0088 |
| 200 | 2.0584 | 0.1485 | 7.53 | 0.0083 | 1.1138 | 0.0975 | 0.6961 | 0.0083 |
| 250 | 1.9117 | 0.2448 | 7.83 | 0.0078 | 1.0952 | 0.0887 | 0.7217 | 0.0079 |
| 300 | 1.8164 | 0.3549 | 8.33 | 0.0077 | 1.0817 | 0.0830 | 0.7404 | 0.0076 |
| 350 | 1.7223 | 0.3537 | 7.81 | 0.0076 | 1.0731 | 0.0788 | 0.7563 | 0.0075 |
| 400 | 1.6963 | 0.3527 | 7.41 | 0.0076 | 1.0684 | 0.0774 | 0.7620 | 0.0076 |
| 450 | 1.7345 | 0.3988 | 7.20 | 0.0075 | 1.0682 | 0.0764 | 0.7650 | 0.0075 |

同预算 0.05 对照（同一曲线，仅末段回弹更明显）：
`stat_object_hm` 2.2243 → 1.9393（−12.8%，且 iter 400→450 回弹），hit@1 全程
0.0000，`correction²` 0.0174 → 0.0066。

读数：

1. **hit@1 的上升是真实的、单调的**：0.0141 → 0.3988（约 28 倍），只在最后 50 iter
   与 loss 一起回弹。这是本轮最强的机制证据——`correction` 里确实被逼出了目标中心
   的位置信息。
2. **`correction²` 在 450 iter 尺度上形成平台**（最后 150 iter 只从 0.0077 到
   0.0075，−2.6%），且平台值 **0.0075 高于 rawnorm 450 iter 的 0.0054（+39%）**。
   但按第 58 节的教训，**450 iter 平台不等于正式训练平台**，必须由 ep6 逐 epoch
   曲线复核（rawnorm 的 450 iter 平台在 ep1-6 就退化成了单调下降）。
3. **辅助项没有伤害主目标**：recon 1.2268 → 1.0682（−12.9%）、KL 0.2209 → 0.0764、
   innovation² 0.0156 → 0.0075，均与 rawnorm 同预算轨迹可比，没有发散。
4. **`stat_object_hm_thr50` 全程仍为 0**：focal 软目标不驱动概率越过 0.5，这是
   预期行为，也说明**不能用固定阈值当 recall 口径**（首版诊断用 `prob>0.5`，在
   0.05 冒烟里读数恒为 0，是口径问题而非模型问题）。现口径改为「每个**有 GT 的
   类别通道**内 argmax 是否落在该类的 GT 中心」，另报 0.5 阈值命中率作为参考。

### 59.4 三窗口反事实诊断（与 rawnorm 450 iter 同预算对照）

`tools/diagnose_z_utilization.py`，`iter_450.pth`，每窗口 4 条序列，head 相对 L2：

| 运行 | 窗口 | zero_z feature | zero_z head | shuffle_z feature | shuffle_z head | prior_only feature | prior_only head | shuffle/zero |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| rawnorm | `[3..6]` | 0.0660 | 2.420e-3 | 0.0077 | 3.120e-4 | 0.0207 | 1.125e-3 | **0.1289** |
| rawnorm | `[100..103]` | 0.0737 | 2.356e-3 | 0.0077 | 2.830e-4 | 0.0230 | 1.084e-3 | **0.1201** |
| rawnorm | `[1800..1803]` | 0.0557 | 2.332e-3 | 0.0042 | 1.960e-4 | 0.0159 | 9.092e-4 | **0.0841** |
| **路线 C** | `[3..6]` | 0.1923 | 6.638e-3 | 0.0123 | **5.800e-4** | 0.0526 | 3.014e-3 | 0.0874 |
| **路线 C** | `[100..103]` | 0.1536 | 6.251e-3 | 0.0091 | **4.880e-4** | 0.0360 | 2.289e-3 | 0.0781 |
| **路线 C** | `[1800..1803]` | 0.1312 | 6.014e-3 | 0.0061 | **3.690e-4** | 0.0278 | 1.925e-3 | 0.0613 |

**两条方向相反的读数必须一起写：**

1. **`shuffle_z` 绝对 head 响应三窗口全部提升 1.7–1.9 倍**
   （3.12e-4→5.80e-4、2.83e-4→4.88e-4、1.96e-4→3.69e-4）。这是路线 C 的直接
   机制收益：correction 携带了更多**只能由当前帧观测提供**的分量。
2. **`shuffle/zero` 比值三窗口全部下降**（0.1289→0.0874、0.1201→0.0781、
   0.0841→0.0613）。原因是 `zero_z` 涨得更快：2.42e-3→6.64e-3（+174%）、
   2.36e-3→6.25e-3（+165%）、2.33e-3→6.01e-3（+158%）。
   即 **公共偏置与身份信号同时被放大，公共分量放大得更多**。这与第 58 节 rawnorm
   相对路线 B 的模式正好相反（那次是分子分母同时下降、分子降得更慢）。

因此：**路线 C 提高了 correction 的信息量（它现在确实编码目标位置），但没有提高
它在检测头里的“相对”样本判别性。** 论文门槛要求 `shuffle_z` head ≥ 2%，当前
0.58% 仍差约 3.4 倍。

### 59.5 Source 探针（`stride=700`，indices `[3,703,1403]`）

| 张量 specific | rawnorm 450 iter | **路线 C 450 iter** | 变化 |
|---|---:|---:|---|
| `feat_pooled_raw` | 19.62% | 18.03% | 持平 |
| `e_t_pooled_raw` / `e_pooled` | 7.62% | **15.33%** | **+2.0 倍** |
| `h_t` | 1.73% | 1.45% | 略降 |
| `mu_p` | 1.47% | 0.73% | −50% |
| `mu_q` | 5.16% | 2.62% | −49% |
| `innovation` | 36.95% | 33.14% | 略降 |
| `correction (= delta_mu)` | 30.97% | 28.17% | 略降 |
| `obs_prior_e_hat` | 0.90% | 2.69% | +2.0 倍 |
| `correction` RMS | 0.02895 | **0.06239** | **+115%** |
| `innovation` RMS | 0.03576 | 0.06104 | +71% |
| `mu_q` RMS | 0.07879 | 0.23930 | +204% |
| `h_t` RMS | 0.06793 | 0.14837 | +118% |
| `z_proj` weight RMS | 0.08342 | 0.08326 | 持平 |

依赖方向：

| 依赖方向 | rawnorm 450 iter | 路线 C 450 iter |
|---|---:|---:|
| `mu_q_removal_of_e_rel_l2` | 0.3152 | 0.2048 |
| `mu_q_removal_of_h_rel_l2` | 0.2756 | 0.1601 |
| h/e 比值 | 0.874 | **0.782** |

读数与归因：

1. **编码器上游变好了**：`e_pooled` specific 从 7.62% 翻倍到 15.33%，`feat_pooled_raw`
   基本不变（19.62% → 18.03%）。唯一能对 encoder 施加新压力的路径就是 correction 的
   辅助目标，所以这条证据支持「object-centric 监督经 correction 反传到 encoder」。
   这与第 58 节反复出现的「上游信息不足」是同一瓶颈，路线 C 是**第一个在 450 iter
   尺度上把该瓶颈推动的方法**。
2. **但下游没有同比例跟上**：`mu_q` 与 `correction` 的 specific 占比反而略降，RMS
   却大幅上升。也就是说，多出来的观测信息更多变成了**公共幅值**（zero_z 涨 158–174%）
   而非样本区分度（shuffle_z 涨 72–88%）。这解释了 59.4 的比值下降。
3. **`h_t` 的通路仍被压低**（1.45% vs rawnorm 1.73%），h/e 依赖比 0.782（比
   rawnorm 的 0.874 更偏观测）。方向是对的，但量级仍小。
4. **`z_proj` 权重尺度两轮完全一致**（0.0834 vs 0.0833），说明差异不是读出增益造成的，
   而是 correction 内容本身变化。

### 59.6 代码与测试

提交：`6d7e865 feat(rssm): add object-centric latent supervision (route C)`。

新增/修改文件：

- `mmdet3d/models/fusion_layers/rssm_fusion.py`：`latent_object_loss_weight`、
  `num_object_classes`、`object_head_hidden_dim`、`object_min_radius` 参数；
  `object_head` 模块；`set_current_object_target`；`_object_heatmap_loss`；四个诊断键。
  辅助头在 `init_weights()` **最后**初始化，避免改变既有模块的 RNG 消耗顺序，
  保证旧 checkpoint 与旧消融仍逐位可比。
- `mmdet3d/models/detectors/R4Det.py`：当前帧 forward 前生成并注入 GT 热图。
- `mmdet3d/core/utils/gaussian.py` + `__init__.py`：`bev_centers_to_heatmap`、
  `gaussian_radius_batch`。
- `configs/r4det/..._lowdim_z_innovation_rawnorm_oc.py`、`me_rssm/sanity/lowdim_z_oc_smoke_450iter.py`。
- `mmdet3d/models/fusion_layers/test_rssm_fusion.py`：新增 11 条路线 C 回归测试。

**RSSM 单测 74 条通过**（`CUDA_VISIBLE_DEVICES=5 python -m unittest -v
mmdet3d.models.fusion_layers.test_rssm_fusion`）。新增测试覆盖：

- 默认权重 0 时 loss 与基线逐位相等、head 不被调用、诊断恒为 0；
- head 输入恰为 `posterior_correction`（pre-hook 与 `stat_correction_sq` 对齐）；
- target 形状/channel/批大小的校验错误路径；
- 关闭 KL/recon/obs_pred 后，object 项是唯一梯度源且确实反传到 `posterior_mu`，
  而只通往 `h_t` 的 `h_proj` 无梯度；
- 空热图（无 GT）贡献严格为零；
- hit@1 判据在「峰值落在中心」「峰值落在背景」「同类多中心」三种构造下的行为；
- 诊断键不含 `loss` 子串；
- 热图几何（中心 cell 落在 `[8,8]`、类别通道选择、越界框丢弃、非法标签跳过）；
- `gaussian_radius_batch` 与仓库标量 `gaussian_radius` 在 64 组随机尺寸上逐元素相等。

### 59.7 结论与下一步

1. **实现层面完成且安全。** 默认关闭时行为逐位不变；推理完全不受影响；辅助目标是
   可学的（head-only probe 40 步内概率到 0.95）；预登记的 0.05 → 0.1 升权有参数
   证据支撑，不是事后调参。
2. **机制层面部分通过。** hit@1 `0.0141 → 0.3988`、`correction` specific 28.17%、
   `correction²` 450 iter 平台 0.0075、上游 `e_pooled` specific 翻倍——**四项都是
   正向**，其中「encoder 上游被推动」是此前所有配置都没做到过的。
3. **但核心门槛仍未过。** `shuffle_z` head 0.58% vs 2% 门槛（差 3.4 倍），
   `shuffle/zero` 比值相对 rawnorm 反而下降。**当前证据只能说路线 C 让 correction
   编码了更多观测内容，不能说它让检测头更依赖样本身份。**
4. **ep6 是硬门控，不能由机制读数替代。** 训练已按 24e 日程启动
   （`/data/lurui/work_dirs/lowdim_z_innovation_rawnorm_oc_3x2x2_24e_seed0`，GPU 5/6/7，
   seed 0，看门狗在 ep6 验证落盘后停止）。必须同时检查：
   - Overall 3D moderate 对比 no2d_igdr seed0 ep6 `34.0243`，以及**四类构成项**；
   - `correction²` 的逐 epoch 曲线是否仍单调塌缩（450 iter 平台不作数）；
   - 三窗口 `shuffle_z` head / `shuffle/zero` 是否保持 59.4 的形态；
   - `stat_object_hm` / hit@1 在 6 epoch 尺度上的走向。
5. **预登记分支（出来后按此判定）**：
   - AP 差距 ≤ 1：若机制继续增强，可考虑 B/C 合并版续到 ep12；
   - AP 差距 1–2 且 `shuffle_z` 仍不足：说明 object-centric 监督不足，进入**路线 D**
     （跨样本去塌缩损失，直接惩罚 common 分量、给出 variance floor）；
   - AP 差距 > 2：路线 C 单独不成立，回到结构层（路线 D 或 `z_prev` 之外的递归改造）；
   - AP 上升但 hit@1 或 `shuffle_z` 退化：不能算 RSSM 方法成功，须查是否只是辅助损失
     带来的普通优化收益。
6. **仍未做**：路线 D（`L_common` + `L_var` 去塌缩）、路线 E（KL balancing +
   按 cell 聚合的 free-bits）。KL 继续后置：raw KL 0.0764 有界，没有数量级异常。

### 59.8 复现命令

```bash
# 450 iter 机制冒烟（无 val）
cd /home/lurui/workspace/R4Det && source .envrc
CUDA_VISIBLE_DEVICES=5,6,7 SEED=0 bash tools/dist_train.sh \
  me_rssm/sanity/lowdim_z_oc_smoke_450iter.py 3 --seed 0 --deterministic \
  --work-dir /tmp/lowdim_z_oc100_check > /tmp/lowdim_z_oc100_check.log 2>&1

# 三窗口反事实诊断（与 rawnorm 450 iter 同预算对照）
CFG=configs/r4det/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_lowdim_z_innovation_rawnorm_oc.py
CKPT=/tmp/lowdim_z_oc100_check/iter_450.pth
CUDA_VISIBLE_DEVICES=5 python tools/diagnose_z_utilization.py --config $CFG --checkpoint $CKPT \
  --start-index 3    --limit 4 --batch-size 4 --shuffle-offset 2  --output /tmp/z_util_oc450_w3.json
CUDA_VISIBLE_DEVICES=6 python tools/diagnose_z_utilization.py --config $CFG --checkpoint $CKPT \
  --start-index 100  --limit 4 --batch-size 4 --shuffle-offset 2  --output /tmp/z_util_oc450_w100.json
CUDA_VISIBLE_DEVICES=7 python tools/diagnose_z_utilization.py --config $CFG --checkpoint $CKPT \
  --start-index 1800 --limit 4 --batch-size 4 --shuffle-offset -1 --output /tmp/z_util_oc450_w1800.json
CUDA_VISIBLE_DEVICES=5 python tools/probe_posterior_source.py --config $CFG --checkpoint $CKPT \
  --start-index 3 --limit 4 --stride 700 --output /tmp/post_source_oc450_stride700.json

# ep6 正式训练（24e 日程，跑到 ep6 验证落盘后停）
tmux new-session -d -s oc_ep6 "cd /home/lurui/workspace/R4Det && source .envrc && \
  CUDA_VISIBLE_DEVICES=5,6,7 SEED=0 bash tools/dist_train.sh \
  configs/r4det/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_lowdim_z_innovation_rawnorm_oc.py \
  3 --seed 0 --deterministic \
  --work-dir /data/lurui/work_dirs/lowdim_z_innovation_rawnorm_oc_3x2x2_24e_seed0 \
  > /tmp/lowdim_oc_ep6.log 2>&1"
```

---

## 60. 路线 C ep6 门控：AP + 机制联合判定（2026-09-28/29）

本节执行第 59 节预登记的 ep6 硬门控。训练按 24e 余弦日程启动、跑到 ep6 验证与
`epoch_6.pth` 落盘后主动停止，因此进度在 ep6 截断、LR/KL 日程与 24e 主实验完全一致。
本节回答两个问题：**object-centric 辅助监督有没有把 AP 拉回基线附近**，以及
**`z_t` 的样本判别性在 6 epoch 尺度上是否真的继续增强**。

### 60.0 状态与判定摘要

**三口径摘要**（6 条 val，跨度 ep1-6；标准 ep12-16 窗口不存在）。
> 本节所有指标冻结在 **ep6 门控时刻**：ep6 之后同一 run 从 `epoch_6.pth` 续训到 ep12，
> 新增的 ep7+ val 记录不属于本节范围，其完整曲线、区间均值和全轮峰值归入后续小节。
> 因此下列「全轮峰值」严格含义是「ep1-6 内的最高单点」。

| 口径 | 数值 |
|---|---|
| 区间均值 | **26.8101**（n=6，实际跨度 ep1-6，等权） |
| 全轮峰值 | **34.2513 @ ep6** |
| best saved | **34.2513 @ ep6**（`epoch_6.pth` 存在；与全轮峰值相同） |

**ep6 门控表**：

| 门控项 | 门槛 | ep6 实测 | 判定 |
|---|---|---:|---|
| Overall 3D moderate vs no2d_igdr seed0 同 epoch | 差距 ≤ 2.0 | **+0.2270**（34.2513 vs 34.0243） | **通过**（首次为正） |
| `shuffle_z` head | ≥ 0.2%（继续门槛），最好 ≥ 0.10 | **1.141%–1.364%**（三窗口） | **通过**（比 450 iter 的 0.58% 翻约 2 倍） |
| `shuffle_z` head（论文最终门槛） | ≥ 2% | 最好 1.364% | **未通过**（差约 1.5 倍） |
| `shuffle/zero` 比值 | 不低于 450 iter 各窗口值 | 0.0995 / 0.0818 / 0.0753 vs 450 iter 0.0874 / 0.0781 / 0.0613 | **通过**（三窗口均不低） |
| innovation specific > e_pooled specific | 保持 | **50.13% > 30.98%** | **通过** |
| delta_mu specific | ≥ 10% | **44.90%** | **通过** |
| `correction²` | 不单调塌缩、至少形成平台 | **0.2736 → 0.0793 逐 epoch 单调下降** | **未通过**（450 iter 平台未在正式训练保持） |

**联合判定：AP 门控通过、核心身份信号增强、`correction²` 仍塌缩。** 按第 59 节与
用户 ep6 预登记分支，「AP 差距 ≤1 且机制继续增强 → 续到 ep12」，因此**已从
`epoch_6.pth` 续训至 ep12**（保持 24e 日程、GPU 5/6/7、seed 0、看门狗在 ep12 验证
落盘后停止）。`correction²` 单调下降被登记为 ep12 必须一并复核的风险项，而不是
被 AP 读数掩盖。

> **后续状态（2026-09-29 更新）**：续训已按第 61 节执行完毕，**ep12 AP 止损**
> （−4.3862，窗口均值 −3.8617），但机制侧 `[3..6]` 窗口 `shuffle_z` head 首次越过
> 2%（2.4290%）。本节 ep6 的「首次单点反超基线」未被后续曲线确认：ep8-12 横盘在
> 约 35，低于基线约 4 点。最终判定见第 61 节。

### 60.1 配置与唯一变量

配置：`configs/r4det/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_lowdim_z_innovation_rawnorm_oc.py`

相对 rawnorm 对照（第 58 节）的**唯一变量**仍为：

```python
latent_object_loss_weight = 0.1     # 0.05 预登记探针判定梯度太弱后按协议升权
num_object_classes = 4
object_head_hidden_dim = 64
object_min_radius = 1
```

完整继承链（均为已记录配置，本 run 未改动其中任何一项）：

```text
..._lowdim_z_innovation_rawnorm_oc
  -> ..._lowdim_z_innovation_rawnorm      # z_prev_normalize=False
  -> ..._lowdim_z_innovation              # posterior_struct='innovation', obs_pred_loss_weight=0.05
  -> ..._lowdim_z                          # latent_dim=32, latent_size=16x16, kl_scale=1.0, free_nats=0.1
  -> ..._no2d_igdr                         # 无 2D 分支、IGDR 打开
  -> ..._pretrained_v2_head                # 自 checkpoints/pretrained_tj4d.pth 初始化
```

关键生效参数（取自 run 目录内配置快照）：`posterior_obs_mode='scaled_raw'`、
`posterior_obs_scale=0.1`、`z_proj_init='xavier'`、`z_prev_normalize=False`、
`posterior_struct='innovation'`、`obs_pred_loss_weight=0.05`、
`recon_loss_weight=0.1`、`recon_target_mode='temporal_delta'`、
`recon_input_mode='posterior_correction'`、`direct_correction_readout=True`、
`readout_mode='spatial'`、`future_loss_weight=0.0`。
KL 超参未动：`kl_scale=1.0`（由 `KLScaleSchedulerHook` 在 ep0-12 从 0.01 线性升到 1.0）、
`free_nats=0.1`。

### 60.2 运行事实

| 项 | 值 |
|---|---|
| work_dir | `/data/lurui/work_dirs/lowdim_z_innovation_rawnorm_oc_3x2x2_24e_seed0` |
| 日志 | `/tmp/lowdim_oc_ep6.log`；run 内 `20260928_134741.log(.json)` |
| GPU / 卡数 | 物理 GPU 5,6,7 / 3 卡 DDP |
| seed / deterministic | 0 / 是 |
| 日程 | EpochBasedRunner `max_epochs=24`，CosineAnnealing `lr=1.5e-4`，`cumulative_iters=2` |
| 启动 | 2026-09-28 13:47:41 UTC（`Start running` 13:47:51） |
| 各 epoch val 时刻 | ep1 14:31:00 / ep2 15:14:46 / ep3 15:58:23 / ep4 16:42:23 / ep5 17:26:37 / ep6 18:10:54 |
| 落盘 | `epoch_2.pth` 15:04:53、`epoch_4.pth` 16:32:42、`epoch_6.pth` 18:00:42（`checkpoint_config.interval=2`） |
| 停止 | ep6 val 于 18:10:54 写完，看门狗随后停止训练 |
| 显存 | 7439 → 7445 MiB |
| 速度 | ≈1.05 s/iter |
| 续训 | 2026-09-29 02:03:06 UTC 以 `--resume-from epoch_6.pth` 续跑，`resumed epoch 6, iter 11412`，保持 24e 日程 |
| 异常 | 无 NaN/Inf、无 OOM、无 rank error |

ep6 验证 170 batch/rank（`val.samples_per_gpu=4`，与第 58 节及基线一致），因此本节
AP 与 no2d_igdr seed0 可直接同 epoch 对照。

### 60.3 完整逐 epoch 曲线（6 条 val）

| ep | 3D easy | **3D mod** | 3D hard | BEV mod | 2D mod | Car_s | Truck_s | Ped_l | Cyc_l |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 13.8963 | 13.6636 | 13.2881 | 20.9942 | 21.1794 | 18.5979 | 6.0011 | 9.4825 | 20.5731 |
| 2 | 25.6671 | 23.9531 | 23.2196 | 33.3351 | 33.7235 | 27.5025 | 15.0210 | 17.0661 | 36.2227 |
| 3 | 31.0141 | 29.1884 | 28.3493 | 35.8858 | 36.0555 | 41.5042 | 13.1242 | 22.6300 | 39.4951 |
| 4 | 29.3713 | 27.2512 | 26.5147 | 38.8290 | 36.7110 | 25.9025 | 14.2276 | 23.2377 | 45.6369 |
| 5 | 34.6705 | 32.5530 | 31.5030 | 40.1290 | 39.8928 | 38.7054 | 20.7789 | 26.7200 | 44.0076 |
| 6 | 36.6531 | **34.2513** | 32.8831 | 42.2583 | 41.4949 | 44.1535 | 26.6900 | 23.2093 | 42.9525 |

窗口逐项均值（ep1-6，n=6，等权）：

| 指标 | 均值 | 全轮峰值 |
|---|---:|---|
| Overall 3D moderate | 26.8101 | 34.2513 @ ep6 |
| Car strict | 32.7277 | 44.1535 @ ep6 |
| Truck strict | 15.9738 | 26.6900 @ ep6 |
| Ped loose | 20.3909 | 26.7200 @ ep5 |
| Cyc loose | 38.1480 | 45.6369 @ ep4 |

### 60.4 四方同 epoch 对照（no2d_igdr / 路线 B / rawnorm / 路线 C）

Overall 3D moderate，全部为 seed 0、同 24e 日程、同 val 配置：

| ep | no2d_igdr | 路线 B | rawnorm | **路线 C** | **C − base** | C − B |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 15.2640 | 20.8115 | 20.1118 | 13.6636 | −1.6004 | −7.1479 |
| 2 | 26.3716 | 22.5532 | 28.1775 | 23.9531 | −2.4185 | +1.3999 |
| 3 | 31.6591 | 28.1833 | 28.0411 | 29.1884 | −2.4707 | +1.0051 |
| 4 | 32.8506 | 25.0003 | 27.7140 | 27.2512 | −5.5994 | +2.2509 |
| 5 | 34.0827 | 34.8551 | 33.8487 | 32.5530 | −1.5297 | −2.3021 |
| 6 | 34.0243 | 31.4895 | 32.8980 | **34.2513** | **+0.2270** | **+2.7618** |
| ep1-6 均值 | 29.0420 | 27.1488 | 28.4652 | **26.8101** | −2.2319 | −0.3387 |
| 全轮峰值 | 34.0827 @ ep5 | 34.8551 @ ep5 | 33.8487 @ ep5 | **34.2513 @ ep6** | — | — |

**ep6 是低维 z 系列第一次在单点 AP 上超过 no2d_igdr seed0**（+0.2270）。但必须同时
写下三条约束：

1. **ep1-6 均值仍低 2.23 点**（26.8101 vs 29.0420）。路线 C 的早期 epoch 收敛更慢
   （ep1 只有 13.66，比 base 低 1.60、比路线 B 低 7.15），单点反超出现在最后一轮，
   不能据此宣布整体追平；最终必须看 ep12-16 窗口均值。
2. **ep6 的四类构成不均**：Truck strict +4.7601、Car strict +0.8743 为正，而
   Ped loose −3.9130、Cyc loose −0.8132 为负。即 ep6 Overall 的反超主要来自
   Truck/Car，行人仍然明显落后。
3. 第 58 节记录的两条反向证据（`correction²` 单调塌缩、`h/e` 依赖方向反转）在
   路线 C 中**依然存在**，见 60.6。

ep6 逐类别拆解：

| 指标 | base | 路线 B | rawnorm | **路线 C** | C − base |
|---|---:|---:|---:|---:|---:|
| Overall 3D moderate | 34.0243 | 31.4895 | 32.8980 | **34.2513** | **+0.2270** |
| Car strict | 43.2792 | 39.9860 | 45.6647 | 44.1535 | +0.8743 |
| Truck strict | 21.9299 | 21.2860 | 21.8159 | **26.6900** | **+4.7601** |
| Ped loose | 27.1223 | 21.1131 | 22.2281 | 23.2093 | −3.9130 |
| Cyc loose | 43.7657 | 43.5729 | 41.8833 | 42.9525 | −0.8132 |

### 60.5 逐 epoch 训练内诊断（38 条 train 记录的 epoch 均值）

| ep | `stat_object_hm` | hit@1 | thr50 | `stat_correction_sq` | recon | raw KL | clamped | innovation² | obs_pred |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.1093 | 0.4933 | 0.1935 | 0.2736 | 0.9781 | 0.6698 | 0.1142 | 0.0415 | 0.0206 |
| 2 | 0.6923 | 0.6764 | 0.4274 | 0.2185 | 0.9418 | 0.1713 | 0.5483 | 0.0469 | 0.0233 |
| 3 | 0.5605 | 0.7389 | 0.5312 | 0.1458 | 0.9365 | 0.0879 | 0.7405 | 0.0440 | 0.0219 |
| 4 | 0.4832 | 0.7730 | 0.5857 | 0.1103 | 0.9313 | 0.0622 | 0.8212 | 0.0425 | 0.0211 |
| 5 | 0.4320 | 0.7875 | 0.6246 | 0.0903 | 0.9287 | 0.0496 | 0.8642 | 0.0425 | 0.0211 |
| 6 | 0.3919 | 0.8107 | 0.6585 | 0.0793 | 0.9281 | 0.0427 | 0.8884 | 0.0438 | 0.0218 |

`stat_object_hm_pos` 全程稳定在 7.70–7.72（每 batch 约 7.7 个 GT 中心），说明 hit@1
的上升不是正样本数量变化造成的。读数：

1. **辅助目标是可学的，且在正式训练尺度上没有过拟合崩坏**：`stat_object_hm`
   1.1093 → 0.3919（−64.7%），hit@1 0.4933 → 0.8107。
2. **`correction²` 仍然逐 epoch 单调下降**：0.2736 → 0.0793（−71.0%），
   逐轮降幅 −20.2% / −33.3% / −24.3% / −18.1% / −12.2%，降幅在收窄但**没有一轮
   持平或回升**。第 59 节 450 iter 观察到的平台（最后 150 iter 0.0077 → 0.0075）
   在正式训练里没有保持；形态与 rawnorm 的 0.2218 → 0.0685 几乎相同（路线 C 起点
   更高、终点也略高，为 +15.8%）。
3. **主目标没有被辅助项拖坏**：`stat_recon_mse` 0.9781 → 0.9281（−5.1%），raw KL
   0.6698 → 0.0427，clamped 0.1142 → 0.8884，`stat_innovation_sq` 与
   `stat_obs_pred` 基本持平（≈0.043 / ≈0.021）。没有任何一项发散。
4. **`stat_mu_diff_sq` 与 `stat_correction_sq` 逐轮完全相同**（如 ep6 均为 0.0793），
   这是 innovation 结构的预期：`correction_t = delta_mu = mu_q − mu_p`，两键是同一
   张量的两条记录路径。

### 60.6 ep6 三窗口反事实诊断（`epoch_6.pth`）

`tools/diagnose_z_utilization.py`，每窗口 4 条序列、`batch-size 4`，head 相对 L2：

| 运行 | 窗口 | zero feature | zero head | shuffle feature | shuffle head | prior feature | prior head | shuffle/zero |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| rawnorm ep6 | `[3..6]` | 0.6844 | 1.1025e-1 | 0.0754 | 1.3783e-2 | 0.1701 | 2.8509e-2 | 0.1250 |
| rawnorm ep6 | `[100..103]` | 0.8341 | 1.2619e-1 | 0.0645 | 1.0394e-2 | 0.2176 | 3.5156e-2 | 0.0824 |
| rawnorm ep6 | `[1800..1803]` | 0.8222 | 1.0978e-1 | 0.0547 | 8.4264e-3 | 0.1848 | 2.7692e-2 | 0.0768 |
| **路线 C ep6** | `[3..6]` | 0.8290 | 1.3707e-1 | 0.0818 | **1.3640e-2** | 0.1641 | 2.9079e-2 | **0.0995** |
| **路线 C ep6** | `[100..103]` | 0.7856 | 1.4214e-1 | 0.0661 | **1.1621e-2** | 0.1866 | 3.5135e-2 | **0.0818** |
| **路线 C ep6** | `[1800..1803]` | 0.9131 | 1.5148e-1 | 0.0709 | **1.1411e-2** | 0.1814 | 2.9033e-2 | **0.0753** |

与 450 iter 同窗口对照（路线 C 自身，验证机制是否随训练继续增强）：

| 窗口 | 450 iter shuffle head | ep6 shuffle head | 倍数 | 450 iter shuffle/zero | ep6 shuffle/zero |
|---|---:|---:|---:|---:|---:|
| `[3..6]` | 5.800e-4 | 1.3640e-2 | **23.5×** | 0.0874 | 0.0995 |
| `[100..103]` | 4.880e-4 | 1.1621e-2 | **23.8×** | 0.0781 | 0.0818 |
| `[1800..1803]` | 3.690e-4 | 1.1411e-2 | **30.9×** | 0.0613 | 0.0753 |

读数（正反并列）：

1. **绝对身份信号显著增强。** `shuffle_z` head 1.141%–1.364%，越过 ep6 继续门槛
   0.2% 并在最好的两个窗口超过「最好 ≥0.10」的期望值；相对 rawnorm ep6 的
   0.843%–1.378%，低窗口提升为主（+35%、+12%），高窗口持平（−1%）。
   相对 450 iter 的 0.369%–0.580%，三窗口提升 23–31 倍。
2. **`shuffle/zero` 比值守住且不再退化。** 三窗口 0.0753–0.0995，均不低于 450 iter
   的对应值（0.0613–0.0874），也落在 rawnorm ep6 的 0.0768–0.1250 区间内。这与
   第 59 节 450 iter「比值三窗口全部下降」的形态不同：**训练到 ep6 后，公共偏置与
   身份信号同步放大、但不再由公共偏置单方面领跑。**
3. **`zero_z` head 绝对值仍在上升**（rawnorm ep6 0.110–0.126 → 路线 C ep6
   0.137–0.151，+13%~+38%）。因此比值改善主要来自分子（shuffle）跟上，而非分母
   下降；公共偏置仍是检测头响应的主导成分。
4. **论文最终门槛 2% 仍未过**：最好的 `[3..6]` 窗口 1.364%，差约 1.5 倍。
   `[1800..1803]` 窗口内部离散度为 0.711%–1.769%（4 条样本），说明远距离窗口的
   身份信号还依赖样本，尚未稳定。

逐样本 `shuffle_z` head（4 条/窗口）：

| 窗口 | 样本 1 | 样本 2 | 样本 3 | 样本 4 |
|---|---:|---:|---:|---:|
| `[3..6]` | 1.4044e-2 | 1.3185e-2 | 1.4116e-2 | 1.3216e-2 |
| `[100..103]` | 1.1393e-2 | 1.1725e-2 | 1.1494e-2 | 1.1873e-2 |
| `[1800..1803]` | 1.7686e-2 | 1.2743e-2 | 7.1070e-3 | 8.1082e-3 |

### 60.7 ep6 Source 探针（`stride=700`，indices `[3,703,1403]`）

| 张量 specific | rawnorm ep6 | **路线 C ep6** | 变化 |
|---|---:|---:|---|
| `feat_pooled_raw` | 15.27% | 12.64% | −17% |
| `e_t_pooled_raw` / `e_pooled` | 16.32% | **30.98%** | **+90%** |
| `h_t` | 10.71% | **14.39%** | +34% |
| `mu_p` | 7.32% | 11.76% | +61% |
| `mu_q` | 10.61% | 15.43% | +45% |
| `correction` | 38.01% | **44.90%** | +18% |
| `innovation` | 46.70% | **50.13%** | +7% |
| `obs_prior_e_hat` | 13.19% | 21.16% | +60% |
| `correction` RMS | 0.25869 | 0.25656 | 持平 |
| `z_proj` weight RMS | 0.08564 | 0.08546 | 持平 |

依赖方向：

| 依赖方向 | rawnorm ep6 | 路线 C ep6 |
|---|---:|---:|
| `mu_q_removal_of_e_rel_l2` | 0.1325 | 0.1380 |
| `mu_q_removal_of_h_rel_l2` | 0.1932 | 0.2010 |
| h/e 比值 | 1.458 | 1.456 |

读数与归因：

1. **上游观测通路被继续推动**：`e_pooled` specific 16.32% → 30.98%（+90%），
   `obs_prior_e_hat` 13.19% → 21.16%。`e_pooled` 的 specific 从 450 iter 的 15.33%
   继续升至 30.98%，说明 object-centric 目标经 correction 反传到 encoder 的路径在
   正式训练里成立且幅度更大。
2. **correction 的 specific 占比也到 44.90%**，但 RMS 保持 0.25656（450 iter 时是
   0.06239，ep6 已被放大到 4.1 倍后稳定）。即此时 correction 的样本区分度来自
   **方向分化**而不是幅度膨胀——去掉了第 59 节 450 iter「specific 略降、RMS 翻倍」
   的形态。
3. **`h_t` 的依赖仍强于观测**：`mu_q_removal_of_h` 0.2010 > `removal_of_e` 0.1380，
   h/e 比值 1.456，与 rawnorm ep6 的 1.458 几乎一致（450 iter 路线 C 曾到 0.782）。
   递归状态重新主导 posterior，这是 `correction²` 继续塌缩的同一个信号：
   检测头对 z 的响应在增长，但 z 的内容仍偏向 h_t 能预测的部分。
4. `z_proj` weight RMS 两轮一致（0.0856 vs 0.0855），排除了读出增益造成的读数差异。

### 60.8 结论与下一步

1. **AP 门控通过。** ep6 Overall 3D moderate 34.2513 vs no2d_igdr seed0 34.0243，
   差 +0.2270，落入「差距 ≤1」分支，是低维 z 系列首次单点反超基线。
2. **身份信号继续增强。** `shuffle_z` head 1.141%–1.364%，相对 450 iter 提升
   23–31 倍；`shuffle/zero` 比值三窗口不再退化；source 探针显示 `e_pooled`
   specific 继续上升。按用户 ep6 预登记门槛，`shuffle_z ≥0.2%` 与「innovation
   specific > e_pooled」「delta_mu specific ≥10%」均通过。
3. **但有两条必须并列的反向证据**：① `correction²` 仍逐 epoch 单调下降
   （0.2736 → 0.0793），第 59 节的 450 iter 平台没有在正式训练保持；② `h/e`
   依赖比 1.456 与 rawnorm 相同，posterior 仍由 `h_t` 主导。两者共同说明：
   **路线 C 提升了 z 的使用强度和身份信号，但还没有改变「correction 由 h_t 可预测
   成分主导、样本独占分量被逐年压缩」的动力学。**
4. **执行分支**：按「AP 差距 ≤1 且机制继续增强」预登记，已从 `epoch_6.pth` 续训至
   ep12（24e 日程、GPU 5/6/7、seed 0，看门狗在 ep12 验证与 `epoch_12.pth` 落盘后
   停止）。ep12 必须复核：
   - ep12 Overall 与 no2d_igdr seed0 `39.2953` 的差距（≤1 可继续 ep16）；
   - `correction²` 是否出现平台或回升，而不是继续单调下降；
   - `shuffle_z` head 是否继续向 2% 靠近（ep6 为 1.14%–1.36%）；
   - `h/e` 依赖比是否向观测侧移动（ep6 为 1.456）。
5. **若 ep12 出现「AP 上升但机制退化」，按用户预登记不能算 RSSM 方法成功**，须先排
   查是否只是辅助损失/初始化带来的普通优化收益。真正未使用的后续手段仍是路线 D
   （跨样本去塌缩：`L_common + L_var` 方差下限）与路线 E（KL balancing +
   按 cell 聚合的 free-bits）；两者都在 identity 门槛通过前不启动。

### 60.9 复现命令

```bash
cd /home/lurui/workspace/R4Det && source .envrc

# ep6 门控训练（24e 日程，跑到 ep6 验证落盘后由看门狗停止）
CUDA_VISIBLE_DEVICES=5,6,7 SEED=0 bash tools/dist_train.sh \
  configs/r4det/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_lowdim_z_innovation_rawnorm_oc.py \
  3 --seed 0 --deterministic \
  --work-dir /data/lurui/work_dirs/lowdim_z_innovation_rawnorm_oc_3x2x2_24e_seed0 \
  > /tmp/lowdim_oc_ep6.log 2>&1

# ep6 三窗口反事实诊断
CFG=configs/r4det/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_lowdim_z_innovation_rawnorm_oc.py
CKPT=/data/lurui/work_dirs/lowdim_z_innovation_rawnorm_oc_3x2x2_24e_seed0/epoch_6.pth
CUDA_VISIBLE_DEVICES=5 python tools/diagnose_z_utilization.py --config $CFG --checkpoint $CKPT \
  --start-index 3    --limit 4 --batch-size 4 --shuffle-offset 2  --output /tmp/z_util_oc_ep6_w3.json
CUDA_VISIBLE_DEVICES=6 python tools/diagnose_z_utilization.py --config $CFG --checkpoint $CKPT \
  --start-index 100  --limit 4 --batch-size 4 --shuffle-offset 2  --output /tmp/z_util_oc_ep6_w100.json
CUDA_VISIBLE_DEVICES=7 python tools/diagnose_z_utilization.py --config $CFG --checkpoint $CKPT \
  --start-index 1800 --limit 4 --batch-size 4 --shuffle-offset -1 --output /tmp/z_util_oc_ep6_w1800.json

# ep6 source 探针
CUDA_VISIBLE_DEVICES=5 python tools/probe_posterior_source.py --config $CFG --checkpoint $CKPT \
  --start-index 3 --limit 4 --stride 700 --output /tmp/post_source_oc_ep6_stride700.json

# ep6 -> ep12 续训（保留 24e 余弦日程与优化器状态）
CUDA_VISIBLE_DEVICES=5,6,7 SEED=0 bash tools/dist_train.sh \
  configs/r4det/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_lowdim_z_innovation_rawnorm_oc.py \
  3 --seed 0 --deterministic \
  --resume-from /data/lurui/work_dirs/lowdim_z_innovation_rawnorm_oc_3x2x2_24e_seed0/epoch_6.pth \
  --work-dir /data/lurui/work_dirs/lowdim_z_innovation_rawnorm_oc_3x2x2_24e_seed0 \
  > /tmp/lowdim_oc_ep12.log 2>&1
```

---

## 61. 路线 C ep12 门控：机制首次越过 2% 门槛但 AP 止损（2026-09-29）

本节执行第 60 节预登记的 ep12 门控。训练从 `epoch_6.pth` 续跑至 ep12，保持 24e
余弦日程与优化器状态，ep12 验证与 `epoch_12.pth` 落盘后由看门狗停止。结果是**正反
证据严重背离**：机制侧首次越过论文的 2% 门槛，AP 侧却跌破止损线。

### 61.0 状态与判定摘要

**三口径摘要**（12 条 val，跨度 ep1-12）：

| 口径 | 数值 |
|---|---|
| 区间均值（标准 ep12-16 窗口不存在，用实际共同区间 ep8-12，n=5） | **35.0289** |
| 区间均值（实际跨度 ep1-12，n=12，仅供完整性） | **30.9967** |
| 全轮峰值 | **35.9549 @ ep7**（无对应权重） |
| best saved | **34.9091 @ ep12**（落盘 ep2/4/6/8/10/12 中最高） |

与 no2d_igdr seed0 同窗口对照：

| 区间 | no2d_igdr seed0 | 路线 C | 差 |
|---|---:|---:|---:|
| ep8-12 均值（n=5） | 38.8906 | 35.0289 | **−3.8617** |
| ep7-12 均值（n=6） | 38.6705 | 35.1833 | −3.4873 |
| ep12 单点 | 39.2953 | 34.9091 | **−4.3862** |

**ep12 门控表**：

| 门控项 | 门槛 | ep12 实测 | 判定 |
|---|---|---:|---|
| Overall vs no2d_igdr seed0 ep12 | 差距 ≤1 可继续 ep16 | −4.3862（ep8-12 均值 −3.8617） | **未通过**（超 2 点止损） |
| `shuffle_z` head（`[3..6]`） | ≥ 2%（论文门槛） | **2.4290%** | **通过**（首次越过） |
| `shuffle_z` head（`[100..103]`） | ≥ 2% | 1.7101% | 未通过 |
| `shuffle_z` head（`[1800..1803]`） | ≥ 2% | 1.0703% | 未通过 |
| innovation specific > e_pooled specific | 保持 | 57.39% > 22.19% | 通过 |
| delta_mu specific | ≥ 10% | 44.06% | 通过 |
| `correction²` | 不单调塌缩 | 逐 epoch 继续单调下降至 0.0537 | **未通过** |
| h/e 依赖比 | 向观测侧移动 | **1.456 → 1.275**（首次持续转向） | 部分通过 |

**联合判定：机制门槛与 AP 门槛相反。** 按第 59/60 节预登记，「AP 差距 >2 → 路线 C
单独不成立」，且用户明确要求「AP 上升但机制退化不能算成功」。本 run 的情况是
**AP 明确退化（−4.39）、机制明确增强（shuffle_z 2.43%）** —— 这既不是成功，也不是
「机制退化被 AP 掩盖」，而是**机制可测但没有转化为检测收益**。按预登记，路线 C 单独
**止损**；不得据此宣布 B+C 组合版是有效方法。

### 61.1 配置与唯一变量

配置与第 60 节完全相同，仍是 rawnorm 对照上只加四项 object-centric 参数
（`latent_object_loss_weight=0.1`、`num_object_classes=4`、`object_head_hidden_dim=64`、
`object_min_radius=1`）。ep6→ep12 续训未改动任何超参、数据配置或 KL 日程，唯一差别
是训练轮数增加，故 ep7-12 与 ep1-6 同属一条可比的连续轨迹。

### 61.2 运行事实

| 项 | 值 |
|---|---|
| work_dir | `/data/lurui/work_dirs/lowdim_z_innovation_rawnorm_oc_3x2x2_24e_seed0` |
| 日志 | `/tmp/lowdim_oc_ep6.log`（ep1-6）、`/tmp/lowdim_oc_ep12.log`（ep7-12） |
| GPU / 卡数 | 物理 GPU 5,6,7 / 3 卡 DDP |
| seed / deterministic | 0 / 是 |
| 续训起点 | 2026-09-29 02:03:06 UTC，`--resume-from epoch_6.pth`，`resumed epoch 6, iter 11412` |
| val 时刻 | ep7 02:47:07 / ep8 03:31:06 / ep9 04:15:05 / ep10 04:58:33 / ep11 05:42:20 / ep12 06:26:07 |
| 落盘 | `epoch_8.pth` 03:21:16、`epoch_10.pth` 04:48:56、`epoch_12.pth` 06:16:32 |
| 停止 | 看门狗检测到 `Epoch(val) [12]` + `epoch_12.pth` 同时存在，06:26:57 发 SIGINT，06:27:22 退出（`EXIT=1`，SIGINT 预期） |
| 显存 | 7444–7446 MiB |
| 速度 | ≈1.05–1.07 s/iter，单 epoch 约 44 min（含 val） |
| 异常 | 无 NaN/Inf、无 OOM、无 rank error；全程 `lr` 按 24e 余弦正常衰减（ep12 训练段 8.479e-05） |

### 61.3 完整逐 epoch 曲线（12 条 val）

| ep | 3D easy | **3D mod** | 3D hard | BEV mod | 2D mod | Car_s | Truck_s | Ped_l | Cyc_l |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 13.8963 | 13.6636 | 13.2881 | 20.9942 | 21.1794 | 18.5979 | 6.0011 | 9.4825 | 20.5731 |
| 2 | 25.6671 | 23.9531 | 23.2196 | 33.3351 | 33.7235 | 27.5025 | 15.0210 | 17.0661 | 36.2227 |
| 3 | 31.0141 | 29.1884 | 28.3493 | 35.8858 | 36.0555 | 41.5042 | 13.1242 | 22.6300 | 39.4951 |
| 4 | 29.3713 | 27.2512 | 26.5147 | 38.8290 | 36.7110 | 25.9025 | 14.2276 | 23.2377 | 45.6369 |
| 5 | 34.6705 | 32.5530 | 31.5030 | 40.1290 | 39.8928 | 38.7054 | 20.7789 | 26.7200 | 44.0076 |
| 6 | 36.6531 | 34.2513 | 32.8831 | 42.2583 | 41.4949 | 44.1535 | 26.6900 | 23.2093 | 42.9525 |
| 7 | 38.2739 | **35.9549** | 34.8091 | 43.2682 | 41.9152 | 46.1712 | 23.5323 | 27.3616 | 46.7546 |
| 8 | 36.8701 | 34.3144 | 33.2012 | 42.2530 | 41.5682 | 46.1721 | 20.8935 | 25.1607 | 45.0314 |
| 9 | 38.9154 | 35.8796 | 34.6240 | 43.8672 | 43.2297 | 43.4771 | 24.6076 | 27.7240 | 47.7097 |
| 10 | 37.5062 | 34.8275 | 33.6869 | 42.1895 | 41.4201 | 46.8087 | 20.4892 | 26.0857 | 45.9264 |
| 11 | 37.9296 | 35.2140 | 33.9695 | 42.7007 | 42.1266 | 45.1074 | 23.0577 | 23.3265 | 49.3642 |
| 12 | 37.6125 | 34.9091 | 33.6846 | 42.1148 | 41.1727 | 47.5970 | 23.9300 | 26.5739 | 41.5353 |

**注意平台形态**：ep7 冲到 35.9549 后并未继续上升，ep8-12 在 34.31–35.88 之间横盘，
整体稳定在约 35 附近，与基线的 38–39.6 平台相差约 4 点。即路线 C **没有出现后期
追赶**；ep6 的单点反超（+0.2270）是首轮波动，不是趋势起点。

窗口逐项均值：

| 指标 | ep8-12 均值(n=5) | ep1-12 均值(n=12) | 全轮峰值 |
|---|---:|---:|---|
| Overall 3D moderate | 35.0289 | 30.9967 | 35.9549 @ ep7 |
| Car strict | 45.8325 | 39.3083 | 47.5970 @ ep12 |
| Truck strict | 22.5956 | 19.3628 | 24.6076 @ ep9 |
| Ped loose | 25.7742 | 23.2148 | 27.7240 @ ep9 |
| Cyc loose | 45.9134 | 42.1008 | 49.3642 @ ep11 |

### 61.4 ep8-12 窗口四方对照（与第 61.3 同口径）

| 指标 | no2d_igdr seed0 | **路线 C** | 差 |
|---|---:|---:|---:|
| Overall 3D moderate | 38.8906 | **35.0289** | **−3.8617** |
| Car strict | 47.4711 | 45.8325 | −1.6387 |
| Truck strict | 28.8510 | 22.5956 | **−6.2554** |
| Ped loose | 29.5562 | 25.7742 | −3.7821 |
| Cyc loose | 49.6841 | 45.9134 | −3.7707 |

**ep12 单点逐类别**：

| 指标 | no2d_igdr seed0 | **路线 C** | 差 |
|---|---:|---:|---:|
| Overall 3D moderate | 39.2953 | 34.9091 | −4.3862 |
| Car strict | 47.4092 | 47.5970 | **+0.1878** |
| Truck strict | 30.8624 | 23.9300 | −6.9324 |
| Ped loose | 30.3707 | 26.5739 | −3.7968 |
| Cyc loose | 48.5388 | 41.5353 | −7.0035 |

**唯一持平项是 Car**（ep12 +0.19、窗口均值 −1.64）。Truck 是最大缺口（ep12 −6.93、
窗口 −6.26）。这与第 60 节 ep6 时「Truck +4.76」的形态完全反转，说明 ep6 的 Truck
优势是早期波动，不是对象监督带来的稳定收益。**object-centric 目标产生的是目标中心
定位能力，而 Truck 需要的是尺寸/朝向回归精度，两者并不同源。**

### 61.5 逐 epoch 训练内诊断（38 条 train 记录的 epoch 均值）

| ep | `stat_object_hm` | hit@1 | thr50 | `stat_correction_sq` | recon | raw KL | clamped | innovation² | obs_pred |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.1093 | 0.4933 | 0.1935 | 0.2736 | 0.9781 | 0.6698 | 0.1142 | 0.0415 | 0.0206 |
| 2 | 0.6923 | 0.6764 | 0.4274 | 0.2185 | 0.9418 | 0.1713 | 0.5483 | 0.0469 | 0.0233 |
| 3 | 0.5605 | 0.7389 | 0.5312 | 0.1458 | 0.9365 | 0.0879 | 0.7405 | 0.0440 | 0.0219 |
| 4 | 0.4832 | 0.7730 | 0.5857 | 0.1103 | 0.9313 | 0.0622 | 0.8212 | 0.0425 | 0.0211 |
| 5 | 0.4320 | 0.7875 | 0.6246 | 0.0903 | 0.9287 | 0.0496 | 0.8642 | 0.0425 | 0.0211 |
| 6 | 0.3919 | 0.8107 | 0.6585 | 0.0793 | 0.9281 | 0.0427 | 0.8884 | 0.0438 | 0.0218 |
| 7 | 0.3593 | 0.8229 | 0.6871 | 0.0709 | 0.9284 | 0.0377 | 0.9070 | 0.0446 | 0.0222 |
| 8 | 0.3341 | 0.8332 | 0.7061 | 0.0657 | 0.9287 | 0.0346 | 0.9182 | 0.0452 | 0.0225 |
| 9 | 0.3118 | 0.8495 | 0.7269 | 0.0615 | 0.9281 | 0.0323 | 0.9272 | 0.0457 | 0.0227 |
| 10 | 0.2907 | 0.8570 | 0.7422 | 0.0583 | 0.9262 | 0.0304 | 0.9345 | 0.0455 | 0.0226 |
| 11 | 0.2749 | 0.8636 | 0.7581 | 0.0557 | 0.9262 | 0.0288 | 0.9398 | 0.0458 | 0.0228 |
| 12 | 0.2615 | 0.8666 | 0.7679 | 0.0537 | 0.9261 | 0.0277 | 0.9440 | 0.0459 | 0.0228 |

`stat_object_hm_pos` 全程 7.69–7.73，正样本数稳定。

读数：

1. **辅助目标持续学习，到 ep12 仍在改善**：`stat_object_hm` 0.2615（较 ep6 再降 33%），
   hit@1 0.8666（ep6 为 0.8107）。即主损失与辅助损失都在下降，**没有训练不收敛的迹象**。
2. **`correction²` 12 轮全程单调下降**：0.2736 → 0.0537（−80.4%），ep6→ep12 段
   −32.3%，逐轮降幅 −10.6%/−7.3%/−6.5%/−5.1%/−4.5%/−3.6%，**降幅收窄但从未持平**。
   这是路线 C 最重要的反向证据：辅助监督把目标位置信息逼进 correction，却没有阻止
   correction 的幅值继续被压缩。
3. **KL 侧有一个未验证的可疑信号**：`clamped_ratio` 0.9440、raw KL 0.0277。
   现实现按**逐元素** clamp（`free_nats=0.1`，见 `rssm_fusion.py:1553`），
   latent 有 32×16×16 = 8192 个元素，因此这个 0.944 表示「94.4% 的逐元素 KL 都小于
   0.1」在 8192 维下并不意外，**不能据此断言 free-bits 已失效**；但它也意味着
   现行口径无法区分「很多元素轻微有效」与「少数元素承载全部信息」。这一条只作为
   路线 E 的动机记录，不作因果结论。
4. recon 0.9261（ep6 0.9281）、`stat_innovation_sq` 0.0459（ep6 0.0438）、
   `obs_pred` 0.0228（ep6 0.0218）—— 主目标项全程平稳，无发散、无爆炸。

### 61.6 ep12 三窗口反事实诊断（`epoch_12.pth`）

`tools/diagnose_z_utilization.py`，每窗口 4 条序列、`batch-size 4`，head 相对 L2：

| 运行 | 窗口 | zero feature | zero head | shuffle feature | shuffle head | prior feature | prior head | shuffle/zero |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 路线 C ep6 | `[3..6]` | 0.8290 | 1.3707e-1 | 0.0818 | 1.3640e-2 | 0.1641 | 2.9079e-2 | 0.0995 |
| 路线 C ep6 | `[100..103]` | 0.7856 | 1.4214e-1 | 0.0661 | 1.1621e-2 | 0.1866 | 3.5135e-2 | 0.0818 |
| 路线 C ep6 | `[1800..1803]` | 0.9131 | 1.5148e-1 | 0.0709 | 1.1411e-2 | 0.1814 | 2.9033e-2 | 0.0753 |
| **路线 C ep12** | `[3..6]` | 0.9974 | 1.6200e-1 | 0.0963 | **2.4290e-2** | 0.1515 | 2.5474e-2 | **0.1499** |
| **路线 C ep12** | `[100..103]` | 0.8360 | 1.6297e-1 | 0.0805 | **1.7101e-2** | 0.1752 | 3.1067e-2 | **0.1049** |
| **路线 C ep12** | `[1800..1803]` | 0.8603 | 1.5625e-1 | 0.0721 | **1.0703e-2** | 0.1600 | 2.3854e-2 | 0.0685 |

逐样本 `shuffle_z` head（4 条/窗口）：

| 窗口 | 样本 1 | 样本 2 | 样本 3 | 样本 4 |
|---|---:|---:|---:|---:|
| `[3..6]` | 2.4824e-2 | 2.3637e-2 | 2.4985e-2 | 2.3713e-2 |
| `[100..103]` | 1.7409e-2 | 1.6755e-2 | 1.7463e-2 | 1.6778e-2 |
| `[1800..1803]` | 1.6378e-2 | 1.1100e-2 | 6.5844e-3 | 8.7492e-3 |

读数：

1. **`[3..6]` 窗口首次越过论文 2% 门槛**：2.4290%（ep6 为 1.3640%，450 iter 为
   0.0580%），且 4 条样本高度一致（2.364%–2.499%）。相对 ep6 提升 78%。
2. **另外两个窗口仍未到 2%**：`[100..103]` 1.7101%、`[1800..1803]` 1.0703%。
   尤其远距离窗口 4 条样本离散度大（0.658%–1.638%），身份信号在远距离/低目标密度
   场景仍不稳定。**因此「已过 2% 门槛」只能按窗口限定表述，不能写成全场景通过。**
3. **`shuffle/zero` 比值在近距离窗口改善明显**（0.0995 → 0.1499），中距离 0.0818 →
   0.1049，远距离基本持平（0.0753 → 0.0685）。`zero_z` head 绝对值仅从 0.137 涨到
   0.162，远小于 `shuffle_z` 的涨幅（+78%），说明这一轮改善**确实来自身份信号的相对
   提升**，与 450 iter 及 ep6 早期「公共分量领跑」的形态不同。
4. 但 `prior_only` head 在近距/远距两个窗口**低于 ep6**（2.9079e-2 → 2.5474e-2、
   2.9033e-2 → 2.3854e-2）。`prior_only` 低于 posterior 说明 posterior 校正仍有价值，
   但绝对值没有随训练继续增长，与 `correction²` 仍在塌缩是同一个现象。

### 61.7 ep12 Source 探针（`stride=700`，indices `[3,703,1403]`）

| 张量 specific | rawnorm ep6 | 路线 C ep6 | **路线 C ep12** |
|---|---:|---:|---:|
| `feat_pooled_raw` | 15.27% | 12.64% | 14.20% |
| `e_t_pooled_raw` / `e_pooled` | 16.32% | 30.98% | **22.19%** |
| `h_t` | 10.71% | 14.39% | 12.18% |
| `mu_p` | 7.32% | 11.76% | **16.16%** |
| `mu_q` | 10.61% | 15.43% | **19.37%** |
| `correction` | 38.01% | 44.90% | **44.06%** |
| `innovation` | 46.70% | 50.13% | **57.39%** |
| `obs_prior_e_hat` | 13.19% | 21.16% | 17.52% |
| `correction` RMS | 0.25869 | 0.25656 | 0.20225 |
| `mu_q` RMS | 1.07880 | 1.09611 | 0.88032 |

依赖方向：

| 依赖方向 | rawnorm ep6 | 路线 C ep6 | **路线 C ep12** |
|---|---:|---:|---:|
| `mu_q_removal_of_e_rel_l2` | 0.1325 | 0.1380 | **0.1483** |
| `mu_q_removal_of_h_rel_l2` | 0.1932 | 0.2010 | **0.1891** |
| **h/e 比值** | 1.458 | 1.456 | **1.275** |

读数：

1. **h/e 依赖比首次持续转向观测侧**：1.456 → 1.275，是低维 z 系列在正式训练里
   **第一次**看到「观测依赖相对递归依赖上升」的跨 epoch 趋势（此前只有 450 iter 的
   单点反转，ep6 即回落）。这条对路线 B 的 Kalman 式动机是有利证据，但对 AP 没有
   帮助 —— 信息流向改善与检测精度改善在本 run 中**解耦**。
2. **`innovation` specific 升至 57.39%**（ep6 50.13%），是全程最高；`e_pooled`
   specific 从 ep6 的 30.98% 回落到 22.19%，说明 obs_prior 减掉的成分随训练变多
   （`obs_prior_e_hat` 名义 specific 21.16% → 17.52%，但 e_pooled 降幅更大）。
3. **`correction` specific 保持 44.06% 但 RMS 从 0.2566 降到 0.2023**（−21%）：
   这解释了 `correction²` 的塌缩（能量降 36%），而占比不变说明减少的是公共分量
   与身份分量按比例缩小 —— 与 61.3 的 AP 平台同时发生。
4. `z_proj` weight RMS 0.0855 → 0.0864（+1.1%），排除读出层参数大幅变化。

### 61.8 结论与下一步

1. **AP 止损成立。** ep12 单点 −4.3862、ep8-12 窗口均值 −3.8617，均超过 2 点阈值。
   平台形态（ep7 35.95 → ep8-12 横盘 34.3–35.9）说明这不是单点噪声。
2. **机制门槛反而首次部分通过。** `[3..6]` 窗口 `shuffle_z` head **2.4290%** 越过 2%，
   h/e 依赖比首次从 1.456 持续降到 1.275，`innovation` specific 升至 57.39%。
   **这推翻了「低维 z 无法对检测头产生 2% 级因果影响」的旧判断**，是本轮最有价值
   的正面结论。
3. **但机制改善没有转化为 AP**，且 `correction²` 全程单调下降至 0.0537。可能的解释
   有三类，本节只登记不做断言：
   - **读出位置不对**：z 的因果影响已可测，但通过 `z_proj` 加到 16×16 再上采样到
     216×248 的空间分辨率不足以影响 Truck/Cyc 的尺寸与朝向回归；
   - **监督目标错配**：object-centric 只监督目标中心，不监督尺寸/朝向，而缺口集中在
     Truck（−6.93）与 Cyc（−7.00），Car（中心定位主导）持平；
   - **辅助项占用了有限容量**：辅助 head 的梯度经 correction 反传到 encoder，
     可能把 encoder 推向「目标中心可线性解码」而非「检测特征可判别」。
4. **两类解释的区分实验已明确**：把 `z_proj` 直连到更高分辨率读出（路线 3 的空间
   读出改造），对照 object 监督换成尺寸/朝向/速度辅助目标。两者都不改变 KL 口径，
   可与路线 E 解耦。
5. **按预登记，路线 C 单独止损，不续 ep16。** 训练已停止，GPU 5/6/7 已释放。
   是否保留辅助头作为后续方案的组件，取决于第 4 条区分实验；在此之前**不应**把它
   写进论文主表。

### 61.9 复现命令

```bash
cd /home/lurui/workspace/R4Det && source .envrc
CFG=configs/r4det/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_lowdim_z_innovation_rawnorm_oc.py

# ep6 -> ep12 续训（保留 24e 余弦日程与优化器状态）
CUDA_VISIBLE_DEVICES=5,6,7 SEED=0 bash tools/dist_train.sh \
  $CFG 3 --seed 0 --deterministic \
  --resume-from /data/lurui/work_dirs/lowdim_z_innovation_rawnorm_oc_3x2x2_24e_seed0/epoch_6.pth \
  --work-dir /data/lurui/work_dirs/lowdim_z_innovation_rawnorm_oc_3x2x2_24e_seed0 \
  > /tmp/lowdim_oc_ep12.log 2>&1

# ep12 三窗口反事实诊断
CKPT=/data/lurui/work_dirs/lowdim_z_innovation_rawnorm_oc_3x2x2_24e_seed0/epoch_12.pth
CUDA_VISIBLE_DEVICES=5 python tools/diagnose_z_utilization.py --config $CFG --checkpoint $CKPT \
  --start-index 3    --limit 4 --batch-size 4 --shuffle-offset 2  --output /tmp/z_util_oc_ep12_w3.json
CUDA_VISIBLE_DEVICES=6 python tools/diagnose_z_utilization.py --config $CFG --checkpoint $CKPT \
  --start-index 100  --limit 4 --batch-size 4 --shuffle-offset 2  --output /tmp/z_util_oc_ep12_w100.json
CUDA_VISIBLE_DEVICES=7 python tools/diagnose_z_utilization.py --config $CFG --checkpoint $CKPT \
  --start-index 1800 --limit 4 --batch-size 4 --shuffle-offset -1 --output /tmp/z_util_oc_ep12_w1800.json

# ep12 source 探针
CUDA_VISIBLE_DEVICES=5 python tools/probe_posterior_source.py --config $CFG --checkpoint $CKPT \
  --start-index 3 --limit 4 --stride 700 --output /tmp/post_source_oc_ep12_stride700.json
```

---

## 62. 回到强主线：Motion-Aligned Innovation RSSM（MAI）实现 + 标准 RSSM 同 epoch 基线（2026-09-29）

本节执行用户在第 61 节之后给出的新方向：**不再继续低维路线，回到全分辨率
`MotionAlignedRSSMFusion` 主线，把已在路线 B/C 中验证过的 innovation posterior
原样移植过来**。第一轮严格单变量，不加 object heatmap、不改 KL/free-bits、不改读出、
不降维；唯一新增项是 `L_obs = 0.05 * smooth_l1(e_hat, stopgrad(e_t))`。

### 62.0 状态与判定摘要

| 项目 | 结果 |
|---|---|
| 标准 RSSM 同 epoch 机制基线（用户步骤 1） | **已完成**（seed1 权重，ep6/ep12 三窗口 probe + source 探针） |
| MAI 实现（用户步骤 2） | **已完成**，`MotionAlignedInnovationRSSMFusion`，RSSM 单测 85 条通过 |
| 450 iter 冒烟（用户步骤 3） | **已完成，`EXIT=0`**；recon −82.7%、obs_pred −61.2%、innovation 有界 |
| ep6 正式训练（用户步骤 4） | **已完成**：ep6 Overall `35.9289` vs 基线 `34.0243`（**+1.9046**），三项门控全部通过 |
| ep12 续训（用户步骤 5） | **进行中**（从 `epoch_6.pth` 恢复，看门狗 `mai_ep12_watch` 待命） |
| 逐步三口径（区间均值 / 全轮峰值 / best saved） | ep6 截断：见 62.14；450 iter 无 val，不产生三口径 |

### 62.1 为什么必须先用标准 RSSM 建立同 epoch 基线

用户步骤 1 要求在跑 MAI 之前，先对**现有标准 RSSM** 的 `epoch_6/12` 跑同一套
三窗口 probe。原因：MAI 是对主线的**单变量替换**，它的门控（`shuffle_z` 不得低于标准
同 epoch）只有在同 config、同 epoch、同 probe 口径下才有意义。

一个必须记录的运行事实：`fgfull_N4_no2d_igdr_2x4_24e_seed0` 的 ep6/ep12 权重已在
2026-09-24 的磁盘清理中删除（保留 ep14/ep22），但**seed1 保留了完整 ep2-ep20 权重**，
配置与 seed0 完全相同。因此本节基线用 seed1 权重，并明确这带来一层 seed 差异；
门控时按「机制量级」而不是逐位相等来判读。

### 62.2 标准 RSSM 三窗口反事实基线（seed1，216×248 全分辨率）

| epoch | 窗口 | zero_h | shuffle_h | prior_h | shuffle/zero |
|---|---|---:|---:|---:|---:|
| ep6 | `[3..6]` | 0.3697 | **0.4973** | 0.2383 | 1.3450 |
| ep6 | `[100..103]` | 0.4087 | 0.0330 | 0.0877 | 0.0807 |
| ep6 | `[1800..1803]` | 0.4429 | 0.0336 | 0.0657 | 0.0758 |
| ep12 | `[3..6]` | 0.2639 | 0.0271 | 0.0426 | 0.1026 |
| ep12 | `[100..103]` | 0.4766 | 0.0471 | 0.0789 | 0.0988 |
| ep12 | `[1800..1803]` | 0.5434 | 0.0414 | 0.0557 | 0.0763 |

feature 侧同窗口原始值：

| epoch | 窗口 | zero feature | shuffle feature | prior feature |
|---|---|---:|---:|---:|
| ep6 | `[3..6]` | 0.8241 | 1.9231 | 0.5788 |
| ep6 | `[100..103]` | 0.9461 | 0.1565 | 0.3710 |
| ep6 | `[1800..1803]` | 0.9626 | 0.1335 | 0.2639 |
| ep12 | `[3..6]` | 0.6186 | 0.1120 | 0.1253 |
| ep12 | `[100..103]` | 0.9639 | 0.1488 | 0.1962 |
| ep12 | `[1800..1803]` | 0.9642 | 0.1192 | 0.1515 |

读数：

1. **ep6 的 `[3..6]` 窗口有极强离散度**：该窗口 4 条样本中样本 3 的 shuffle head
   达 161.6%，而样本 4/6 只有 1.1% 左右；因此 ep6 那个 0.4973 的均值是
   「单样本主导」的假象，**不能当作标准 RSSM 在 ep6 的近距能力**。ep12 同一窗口回落到
   0.0271，说明它没有在正式训练里保持。
2. **中/远距窗口是稳定口径**：ep6 为 3.30%/3.36%，ep12 为 4.71%/4.14%。
3. **与路线 C（16×16，seed0）对照**：路线 C ep12 的 shuffle_z head 为
   2.429%/1.710%/1.071%。除近距窗口外，**标准全分辨率 RSSM 的 4.71%/4.14% 实际上
   高于路线 C**，直接验证了用户「低维路线尚未超过原主线的 z 使用程度」的判断。

### 62.3 标准 RSSM Source 探针（`stride=700`，indices `[3,703,1403]`）

| 张量 sample-specific energy | ep6 | ep12 |
|---|---:|---:|
| `feat_pooled_raw` | 14.78% | 29.96% |
| `e_t_pooled_raw` | 33.74% | **66.35%** |
| `h_t` | 20.65% | 17.03% |
| `mu_p` | 22.25% | 18.21% |
| `mu_q` | 22.48% | 18.75% |
| `correction` | 36.63% | **41.66%** |

RMS 与依赖方向：

| 量 | ep6 | ep12 |
|---|---:|---:|
| `h_t` RMS | 0.1393 | 0.1673 |
| `mu_p` RMS | 0.2971 | 0.3852 |
| `mu_q` RMS | 0.2965 | 0.3865 |
| `correction` RMS | 0.1079 | 0.0889 |
| `mu_q_removal_of_e_rel_l2` | 0.3959 | 0.3787 |
| `mu_q_removal_of_h_rel_l2` | 0.8820 | 0.8855 |
| **h/e 依赖比** | **2.228** | **2.338** |

读数：标准主线里 encoder 输出 `e_t` 的样本特异占比在 ep6→ep12 从 33.7% 涨到
66.4%，**上游信息一直很充足**；但 `mu_q` 特异占比反而从 22.5% 降到 18.8%，
h/e 依赖比从 2.23 升到 2.34——即**posterior 越来越依赖 h_t、越来越不依赖充满足够
信息的 e_t**。这正是 MAI 要修的「predict-correct 形态」问题，而不是「上游没信息」。

### 62.4 MAI 实现与唯一变量

| 项 | 值 |
|---|---|
| 新类 | `MotionAlignedInnovationRSSMFusion`（`mmdet3d/models/fusion_layers/rssm_fusion.py`） |
| 配置 | `configs/r4det/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_mai.py` |
| 对照 | `..._no2d_igdr.py`（`MotionAlignedRSSMFusion`） |
| 唯一变量 | 后验均值产生方式 + `obs_pred_loss_weight=0.05` |
| 保持不变 | latent 216×248（`latent_dim=256`）、ConvGRU `transition`、deformable h/z 对齐、`free_nats=1.0`、KL warm-up ep0-10、reconstruction、`output_proj` 残差读出、24e cosine |

结构：

```python
e_hat_t    = self.obs_prior(h_t)                 # 新参数，仅由 L_obs 训练
obs_pred   = smooth_l1(e_hat_t, e_t.detach())
innovation = e_t - e_hat_t.detach()
delta_mu   = self.posterior_mu(cat([h_t, innovation]))
mu_q       = self.prior_mu(h_t) + delta_mu
```

`obs_prior` 在 `super().__init__()` **之后**创建并 Xavier 初始化，保证父类所有既有模块的
初始化 RNG 顺序不变、主线 checkpoint 对共享键逐位可加载（有回归测试）。诊断键
`stat_obs_pred`、`stat_innovation_sq` 均不含 `loss` 子串。完整 RSSM 单测 **85 条通过**
（原 75 + 新 10）。

### 62.5 450 iter 机制冒烟（用户步骤 3）

运行事实：3 卡 DDP（GPU 5/6/7）、seed 0、deterministic、IterBasedRunner 450 iter、
无 val；显存 17.27→17.28 GiB/卡，1.55–1.61 s/iter，`EXIT=0`，无 NaN/Inf。

| iter | recon | kl_raw | clamped | mu_diff² | obs_pred | innovation² | grad_norm |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 0.9537 | 1.7304 | 0.8621 | 0.3178 | 0.1249 | 0.9205 | 12.97 |
| 100 | 0.5451 | 0.5211 | 0.9285 | 0.1777 | 0.0709 | 0.9395 | 16.19 |
| 150 | 0.3828 | 0.3508 | 0.9741 | 0.1045 | 0.0611 | 0.8837 | 20.24 |
| 200 | 0.2781 | 0.2648 | 0.9845 | 0.0685 | 0.0528 | 0.8825 | 22.23 |
| 250 | 0.2192 | 0.2500 | 0.9892 | 0.0512 | 0.0465 | 0.8745 | 20.69 |
| 300 | 0.1883 | 0.2362 | 0.9900 | 0.0481 | 0.0452 | 0.8528 | 16.39 |
| 350 | 0.1723 | 0.2411 | 0.9884 | 0.0430 | 0.0458 | 0.8301 | 15.61 |
| 400 | 0.1668 | 0.2298 | 0.9865 | 0.0417 | 0.0486 | 0.8200 | 16.45 |
| 450 | 0.1649 | 0.2258 | 0.9862 | 0.0417 | 0.0484 | 0.8136 | 16.10 |

读数：

1. **接口正确**：recon 下降 82.7%，`obs_prior` 的 `stat_obs_pred` 下降 61.2%，
   证明新增观测预测通路确实在学；`innovation²` 稳定在 0.81–0.94，没有塌到 0。
2. **KL 有界、未爆炸**：`kl_raw` 从 1.73 收敛到 0.226，`clamped_ratio ≈0.986`。
   与标准主线的长期行为一致（标准主线同样长期被 free-bits 钳制），再次支持
   「KL 不是首要病因」。
3. **但 `mu_diff²` 从 0.3178 单调降到 0.0417**（−86.9%），只在最后 100 iter 形成
   平台（0.0417→0.0417）。这一点是**明确的下行风险**：如果 correction 在正式训练里
   继续向 0 收缩，innovation 的样本判别信息会再次被压掉。ep6 门控必须专门检查。
4. 显存 17.3 GiB/卡（标准主线约 5–6 GiB 的读数来自低维路线；全分辨率主线本身即
   约 17 GiB），比标准主线只多一个 `Conv2d(256,256,3×3)` 的开销，符合预期。

### 62.6 三口径记录

450 iter 冒烟是 IterBasedRunner、`evaluation.interval=4500`，**不带 val**，因此
**不产生任何区间均值 / 全轮峰值 / best saved**，也不能给出任何 AP 结论。ep6 门控
截断后的三口径摘要见 62.14，ep12 的完整三口径将在续训结束后另节记录。

### 62.7 ep6 门控预登记（结果见 62.9）

于 2026-09-29 13:05:59 UTC 在 tmux `mai_ep6` 启动，24e 余弦日程截断在 ep6；
watchdog `mai_ep6_watch`（脚本 `me_rssm/sanity/watch_mai_ep6.sh`）在
`Epoch(val) [6]` 与 `epoch_6.pth` 同时存在后发送 SIGINT，等 GPU 释放后自动运行
三窗口 probe 与 source 探针。预登记门控（用户第 4 步）：

1. Overall 3D moderate 与 no2d_igdr seed0 同 epoch 差距 `≤1`（seed0 ep6 = 34.0243）；
2. 任一类别下降不超过 2（seed0 ep6：Car 43.2792 / Truck 21.9299 / Ped 27.1223 / Cyc 43.7657）；
3. `shuffle_z`/`zero_z` 不低于标准 RSSM 同 epoch（ep6 稳定窗口：shuffle 3.30%/3.36%，
   zero 40.87%/44.29%；近距窗口因单样本主导而单独标注）。

### 62.9 ep6 门控结果：**三项门控全部通过，AP 反超 +1.9046**

运行事实：2026-09-29 13:05:59 UTC 启动，ep6 val 于 19:15:12 UTC 写完，
`epoch_6.pth` 19:03 落盘；看门狗 19:17 前完成停训 + 三窗口 probe + source 探针；
训练以 SIGINT 结束（`EXIT=1`，预期），GPU 5/6/7 已释放。全程 GPU 5/6/7、seed 0、
deterministic、24e 余弦日程截断在 ep6（`max_epochs` 未改，LR 与基线逐轮对齐）。

**MAI ep6 与 no2d_igdr seed0 同 epoch 对照**：

| 门控项 | 预登记门槛 | 实测 | 判定 |
|---|---|---:|---|
| Overall 3D moderate | 与 seed0 ep6 `34.0243` 差距 ≤1 | **35.9289（+1.9046）** | **通过（反超）** |
| 任一类别下降不超过 2 | ≤2 | Car **+8.0021**、Truck +1.5706、Ped +0.0140、Cyc **−1.9681** | **通过** |
| `shuffle_z` head（稳定窗口） | ≥ 标准同 epoch（3.30%/3.36%） | **8.1050% / 5.6800%** | **通过（2.5×/1.7×）** |
| `zero_z` head（稳定窗口） | 不低于标准（40.87%/44.29%） | 48.22% / 48.17% | **通过** |
| `shuffle/zero` 比值 | 不低于标准（0.081/0.076） | 0.1681 / 0.1179 | **通过** |
| `innovation` specific | 保持 > `e_pooled` specific | 47.93% > 27.43% | **通过** |
| `delta_mu`（correction）specific | ≥10% | 35.76% | **通过** |
| `mu_diff²` 不再单调塌缩 | 形成平台 | ep1 1.2663 → ep4 0.0976 → ep5 0.1627（**回升**）→ ep6 0.0686 | **通过（非单调）** |

### 62.10 MAI ep1-6 完整逐 epoch 曲线

| ep | Overall3Dmod | OverallBEVmod | 3D easy | 3D hard | 2D mod | Car | Truck | Ped_l | Cyc_l |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20.1833 | 27.0874 | 21.3772 | 19.4795 | 26.4805 | 36.3145 | 9.8022 | 4.3559 | 30.2606 |
| 2 | 27.8215 | 35.1231 | 29.5261 | 26.7769 | 36.2551 | 37.0686 | 20.6826 | 21.4359 | 32.0988 |
| 3 | 32.0763 | 37.7378 | 34.3979 | 31.0086 | 36.8009 | 46.9572 | 17.8239 | 23.2287 | 40.2955 |
| 4 | 31.6728 | 40.7212 | 33.5616 | 30.5969 | 39.5080 | 45.6701 | 18.2141 | 24.2503 | 38.5569 |
| 5 | 33.3755 | 42.7999 | 34.5976 | 32.2142 | 44.0083 | 42.7265 | 23.9471 | 21.4883 | 45.3399 |
| 6 | **35.9289** | 42.3272 | 37.2093 | 34.5612 | 43.6214 | **51.2813** | 23.5005 | 27.1363 | 41.7976 |

同 epoch 基线（no2d_igdr seed0）：

| ep | Overall3Dmod | OverallBEVmod | Car | Truck | Ped_l | Cyc_l |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 15.2640 | 23.2541 | 29.4114 | 7.1599 | 13.9757 | 10.5091 |
| 2 | 26.3716 | 33.2335 | 37.6748 | 22.4470 | 11.5293 | 33.8351 |
| 3 | 31.6591 | 39.5683 | 44.9349 | 19.5616 | 22.2242 | 39.9158 |
| 4 | 32.8506 | 42.0088 | 40.8715 | 22.3596 | 25.0709 | 43.1004 |
| 5 | 34.0827 | 44.1080 | 43.3588 | 25.1274 | 18.3216 | 49.5232 |
| 6 | 34.0243 | 42.4084 | 43.2792 | 21.9299 | 27.1223 | 43.7657 |

**逐 epoch 差值（MAI − base）**：ep1 **+4.9193**、ep2 +1.4499、ep3 +0.4172、
ep4 −1.1778、ep5 −0.7072、ep6 **+1.9046**。

窗口均值：MAI ep1-6 均值 **30.1764** vs base **29.0420**（**+1.1343**，n=6）；
ep3-6 均值 33.2634 vs 33.1542（+0.1092）；ep5-6 均值 +0.5987。

读数：

1. **不是「赢在早期」**：ep1 的 +4.92 主要来自 Ped_l/Cyc_l 的早熟（4.36/30.26 vs
   13.98/10.51），但 ep3-4 一度落后，说明这不是单调优势。真正有意义的是 ep6 单点
   +1.90 与 ep1-6 均值 +1.13，即**整个窗口的平均水平也高于基线**，不只是某轮的抖动。
2. **Car 是主要增益来源**：ep6 Car strict 51.2813，比基线高 8.00；BEV moderate
   42.33 与基线 42.41 基本持平，说明增益来自 3D 回归而不是 BEV 定位。
3. **Cyc_l −1.97 是唯一超 1 点的负项**，仍在 ≤2 门控内；Truck/Ped 基本持平。

### 62.11 MAI ep6 三窗口反事实 probe

| 窗口 | zero_h | shuffle_h | prior_h | shuffle/zero | zero_f | shuffle_f | prior_f |
|---|---:|---:|---:|---:|---:|---:|---:|
| `[3..6]` | 0.4943 | **0.1494** | 0.1443 | **0.3023** | 0.8927 | 0.3009 | 0.3308 |
| `[100..103]` | 0.4822 | **0.0811** | 0.1011 | 0.1681 | 0.9427 | 0.1880 | 0.2984 |
| `[1800..1803]` | 0.4817 | **0.0568** | 0.0990 | 0.1179 | 0.9473 | 0.1597 | 0.2881 |

与标准 RSSM 同 epoch（seed1）对照：

| 窗口 | 标准 RSSM shuffle_h | MAI shuffle_h | 倍数 | 标准 shuffle/zero | MAI shuffle/zero |
|---|---:|---:|---:|---:|---:|
| `[3..6]` | 0.4973（单样本主导） | 0.1494 | 0.30× | 1.3450（失真） | 0.3023 |
| `[100..103]` | 0.0330 | **0.0811** | **2.46×** | 0.0807 | **0.1681** |
| `[1800..1803]` | 0.0336 | **0.0568** | **1.69×** | 0.0758 | **0.1179** |

逐样本 `shuffle_z` head（4 条/窗口）：

| 窗口 | 样本 1 | 样本 2 | 样本 3 | 样本 4 |
|---|---:|---:|---:|---:|
| `[3..6]` | 0.2225 | 0.0840 | 0.2062 | 0.0850 |
| `[100..103]` | 0.0960 | 0.0652 | 0.0972 | 0.0658 |
| `[1800..1803]` | 0.0900 | 0.0494 | 0.0456 | 0.0422 |

读数：

1. **中/远距窗口的 shuffle 影响是 MAI 的核心新证据**：0.0811 / 0.0568，分别是标准
   RSSM 的 2.46× / 1.69×。这已经**超过路线 C ep12 同窗口的 0.0171 / 0.0107**
   （4.7× / 5.3×），三个窗口也分别高于路线 C ep12 的 0.0243 / 0.0171 / 0.0107
   （6.2× / 4.7× / 5.3×）。
2. **`shuffle/zero` 比值同步改善**（0.168 / 0.118 vs 标准 0.081 / 0.076），说明
   提升来自样本身份而不是公共分量同比放大——这正是路线 A 四组（比值恒定
   0.036–0.046）没有做到的事情。
3. 近距窗口的标准基线 0.4973 是单样本主导的假象（样本 3 达 161.6%），MAI 的
   0.1494 虽然数值上更低，但 4 条样本离散更小（0.084–0.223），**不宜按数值直接比较**；
   本表保留该行只为完整性。
4. **论文 2% 门槛 ep6 即已三窗口全部越过**：14.94% / 8.11% / 5.68%，最低的远距
   窗口仍有 2% 门槛的 2.8 倍。这是低维路线 B/C 从未在 ep6 达到的形态（路线 C
   ep6 为 1.36% / 1.16% / 1.14%，差约 5–11 倍）。保留意见是跨窗口一致性：
   远距窗口只有近距的 38%，样本离散未消失，须看 ep12 是否保持或继续改善。

### 62.12 MAI ep6 source 探针（`stride=700`，indices `[3,703,1403]`）

| 张量 specific | 标准 RSSM ep6 | **MAI ep6** | 变化 |
|---|---:|---:|---:|
| `feat_pooled_raw` | 14.78% | 15.32% | +0.5 |
| `e_t_pooled_raw` | 33.74% | 27.43% | −6.3 |
| `h_t` | 20.65% | **13.51%** | **−7.1** |
| `mu_p` | 22.25% | 11.56% | −10.7 |
| `mu_q` | 22.48% | 16.60% | −5.9 |
| `correction` | 36.63% | 35.76% | −0.9 |
| `innovation` | — | **47.93%** | 新增 |

RMS 与依赖方向：

| 量 | 标准 RSSM ep6 | MAI ep6 |
|---|---:|---:|
| `h_t` RMS | 0.1393 | 0.0943 |
| `mu_p` RMS | 0.2971 | 0.2888 |
| `mu_q` RMS | 0.2965 | 0.3943 |
| `correction` RMS | 0.1079 | 0.1650 |
| `mu_q_removal_of_e_rel_l2` | 0.3959 | 0.3099 |
| `mu_q_removal_of_h_rel_l2` | 0.8820 | 0.3098 |
| **h/e 依赖比** | **2.228** | **0.9996** |

> 口径说明：`probe_posterior_source.py` 对 `posterior_struct='innovation'` 自动切换
> ——零化的是 posterior 两半输入 `[h_t, innovation]`，因此 MAI 的 `removal_of_e`
> 指的是**移除 innovation**（即 `e_t − stopgrad(e_hat)`），不是移除原始 `e_t`。
> 标准 RSSM 的两半仍是 `[h_t, e_t]`，所以两列可以直接对比「posterior 对递归侧
> 的依赖 vs 对观测侧的依赖」这个结构量。

读数：

1. **h/e 依赖比首次降到 1.0 以下**：标准主线 2.228、路线 C ep12 最低只到 1.275，
   MAI ep6 直接到 **0.9996**——posterior 对 `h_t` 与对观测的依赖**完全对称**。
   这是本节最强的结构性证据：innovation posterior 确实把 predict-correct 变成了
   双向通路，而不是让 `h_t` 单方面主导。
2. **`h_t` specific 从 20.65% 降到 13.51%、`mu_p` 从 22.25% 降到 11.56%**。
   递推状态本身的样本身份被削弱，但 `mu_q` 反而变大（RMS 0.297→0.394），
   `correction` RMS 0.108→0.165——**后验不再靠 `h_t` 的公共底色撑幅度**，而是靠
   自己的 correction 项。
3. `innovation` specific 47.93% 明显高于 `e_pooled` 的 27.43%，说明 obs_prior
   减掉的确实是公共成分，符合设计意图。

### 62.13 MAI ep6 训练诊断曲线

| ep | recon | kl_raw | clamped | mu_diff² | obs_pred | innovation² |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0712 | 15.9104 | 0.7900 | 1.2663 | 0.0373 | 0.9791 |
| 2 | 0.0130 | 0.8767 | 0.9427 | 0.2932 | 0.0282 | 0.9336 |
| 3 | 0.0113 | 0.6853 | 0.9689 | 0.2263 | 0.0305 | 0.8860 |
| 4 | 0.0093 | 0.6317 | 0.9805 | 0.0976 | 0.0282 | 0.8392 |
| 5 | 0.0117 | 0.6311 | 0.9730 | **0.1627** | 0.0337 | 0.7960 |
| 6 | 0.0091 | 0.6179 | 0.9848 | 0.0686 | 0.0276 | 0.7085 |

（每行是 38 条训练日志的均值。）

读数：`mu_diff²` 在 ep4→ep5 **回升 0.098→0.163**，直接否定了
「correction 必然单调塌缩」的旧担心——这是低维路线 B/C 从未出现过的形态。
`obs_pred` 稳定在 0.027–0.034（`obs_prior` 持续被训练但不过拟合），
`innovation²` 缓降 0.98→0.71 且未塌缩，`kl_raw` 在 ep1 的 15.9 之后迅速降到
0.62 并稳定，`clamped_ratio` 0.98 与标准主线一致。

### 62.14 三口径摘要（ep6 门控截断，n=6）

| 口径 | 数值 |
|---|---|
| 区间均值（标准 ep12-16 不存在；用实际共同区间 ep1-6，n=6） | **30.1764** |
| 区间均值（ep3-6，n=4，贴近平台） | **33.2634** |
| 全轮峰值 | **35.9289 @ ep6** |
| best saved | **35.9289 @ ep6**（落盘 ep2/4/6 中最高） |

对照 no2d_igdr seed0 同窗口：ep1-6 均值 29.0420（MAI **+1.1343**）、ep3-6 均值
33.1542（MAI +0.1092）。

### 62.15 判定与下一步

**按预登记，「差距 ≤1 且机制继续增强」分支成立，已启动 ep12 续训。**
ep12 必须复核（用户第 5 步）：

- ep8-12 窗口均值至少达到基线 `38.8906 − 0.5 = 38.3906`，目标 `39.3906`；
- `shuffle_z` head 是否继续向 2% 靠近（ep6 最好 14.94%）；
- h/e 依赖比能否守住 ≤1（ep6 为 0.9996）；
- `mu_diff²` 是否继续保持非单调（ep5 回升后 ep6 回落至 0.0686）。

需要并列保留的两条保留意见：

1. **单 seed、单 epoch 的 +1.90 不能当结论**。no2d_igdr 三 seed 在 ep12-16 的
   标准差是 1.35，本节是 n=1 的同 epoch 对照；ep3-4 曾连续落后，说明曲线并非
   全线占优。判定依据是「按预登记规则可以继续」，不是「已被统计证实」。
2. **2% 门槛已在 ep6 三窗口全部越过**（14.94% / 8.11% / 5.68%），但**远距窗口
   只有近距的 38%**，跨场景一致性仍是保留项；若 ep12 出现「AP 继续上升但远距
   身份信号停滞」，按用户预登记应先做 `z_prev` normalize 消融，而不是直接加
   几何辅助监督。

### 62.16 复现命令

```bash
cd /home/lurui/workspace/R4Det && source .envrc

# 450 iter 机制冒烟
CUDA_VISIBLE_DEVICES=5,6,7 SEED=0 bash tools/dist_train.sh \
  me_rssm/sanity/mai_smoke_450iter.py 3 --seed 0 --deterministic \
  --work-dir /tmp/mai_smoke_450iter > /tmp/mai_smoke_450iter.log 2>&1

# ep6 门控正式训练（24e 日程截断在 ep6）
CUDA_VISIBLE_DEVICES=5,6,7 SEED=0 bash tools/dist_train.sh \
  configs/r4det/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_mai.py \
  3 --seed 0 --deterministic \
  --work-dir /data/lurui/work_dirs/fgfull_N4_no2d_igdr_mai_2x4_24e_seed0 \
  > /tmp/mai_ep6.log 2>&1

# ep6 -> ep12 续训（保留 24e 余弦日程与优化器状态）
# 看门狗：bash me_rssm/sanity/watch_mai_ep12.sh（ep12 val + epoch_12.pth 后停训并跑 probe）
CUDA_VISIBLE_DEVICES=5,6,7 SEED=0 bash tools/dist_train.sh \
  configs/r4det/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_mai.py \
  3 --seed 0 --deterministic \
  --resume-from /data/lurui/work_dirs/fgfull_N4_no2d_igdr_mai_2x4_24e_seed0/epoch_6.pth \
  --work-dir /data/lurui/work_dirs/fgfull_N4_no2d_igdr_mai_2x4_24e_seed0 \
  > /tmp/mai_ep12.log 2>&1

# 标准 RSSM 同 epoch 机制基线（seed1，全分辨率）
CFG=configs/r4det/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr.py
for ep in 6 12; do
  CKPT=/data/lurui/work_dirs/fgfull_N4_no2d_igdr_2x4_24e_seed1/epoch_${ep}.pth
  CUDA_VISIBLE_DEVICES=5 python tools/diagnose_z_utilization.py --config $CFG --checkpoint $CKPT \
    --start-index 3    --limit 4 --batch-size 4 --shuffle-offset 2  --output /tmp/z_util_stdep${ep}_w3.json
  CUDA_VISIBLE_DEVICES=6 python tools/diagnose_z_utilization.py --config $CFG --checkpoint $CKPT \
    --start-index 100  --limit 4 --batch-size 4 --shuffle-offset 2  --output /tmp/z_util_stdep${ep}_w100.json
  CUDA_VISIBLE_DEVICES=7 python tools/diagnose_z_utilization.py --config $CFG --checkpoint $CKPT \
    --start-index 1800 --limit 4 --batch-size 4 --shuffle-offset -1 --output /tmp/z_util_stdep${ep}_w1800.json
  CUDA_VISIBLE_DEVICES=5 python tools/probe_posterior_source.py --config $CFG --checkpoint $CKPT \
    --start-index 3 --limit 4 --stride 700 --output /tmp/post_source_std_ep${ep}_stride700.json
done
```

---

## 63. MAI ep12 门控：AP 守住并小幅领先，机制优势部分回落（2026-09-30）

本节执行第 62.15 节预登记的 ep12 门控。训练从 `epoch_6.pth` 续跑至 ep12（24e
余弦日程与优化器状态保留），看门狗 `mai_ep12_watch`
（`me_rssm/sanity/watch_mai_ep12.sh`）在 ep12 验证与 `epoch_12.pth` 落盘后停训并
自动跑探针。

### 63.0 状态与判定摘要

**三口径摘要**（12 条 val，跨度 ep1-12）：

| 口径 | 数值 |
|---|---|
| 区间均值（标准 ep12-16 窗口不存在，用实际共同区间 ep8-12，n=5） | **39.3392** |
| 区间均值（实际跨度 ep1-12，n=12，仅供完整性） | **34.9486** |
| 全轮峰值 | **39.6165 @ ep10** |
| best saved | **39.6165 @ ep10**（落盘 ep2/4/6/8/10/12 中最高） |

与 no2d_igdr seed0 同窗口对照：

| 区间 | no2d_igdr seed0 | MAI | 差 |
|---|---:|---:|---:|
| ep8-12 均值（n=5） | 38.8906 | 39.3392 | **+0.4485** |
| ep12 单点 | 39.2953 | 39.2805 | −0.0148 |
| ep1-6 均值（对照 62.14） | 29.0420 | 30.1764 | +1.1343 |

**ep12 门控表**：

| 门控项 | 门槛 | ep12 实测 | 判定 |
|---|---|---:|---|
| ep8-12 均值 vs no2d_igdr seed0 | 至少达到基线 `−0.5`（`38.3906`） | **39.3392（+0.4485）** | **通过** |
| 目标（用户第 5 步） | Overall `+0.5` | +0.4485 | **差 0.05 未达，但方向一致** |
| `shuffle_z` head（近/中/远） | 向 2% 靠近并保持 | 3.4918% / 4.3686% / 5.9410% | **通过门槛，但近距大幅回落**（ep6 14.94%） |
| h/e 依赖比 | 守住 ≤1 | **0.8418** | **通过**（ep6 0.9996，继续向观测侧移动） |
| `correction` specific | 保持 ≥10% | **46.28%** | **通过**（ep6 35.76%） |
| `innovation` specific | 保持 > `e_pooled` | **66.39% > 64.28%** | **通过（余量很小）** |
| `mu_diff²` 是否保持非单调 | 是 | ep9 0.0440 → **ep10 0.0375** → **ep11 0.0454**（回升）→ ep12 0.0364 | **通过** |

**一句话结论：AP 通过续跑门控并小幅领先基线（ep8-12 **+0.4485**），但 ep6 那种
「身份信号三窗口全部越过 2%」的强机制优势在 ep12 明显回落；两条曲线没有同步。**

### 63.1 配置与唯一变量

| 项 | 值 |
|---|---|
| 配置 | `configs/r4det/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_mai.py` |
| 对照 | `..._no2d_igdr.py`（`MotionAlignedRSSMFusion`, seed0） |
| 唯一变量 | `posterior_struct='innovation'` + `obs_pred_loss_weight=0.05` |
| 续训 | `--resume-from epoch_6.pth`，24e 日程与优化器状态保留（`max_epochs` 未改） |

### 63.2 运行事实

| 项 | 值 |
|---|---|
| ep6 段 | 2026-09-29 13:05:59 → 19:15:12 UTC（val 落盘），SIGINT 停止 |
| ep12 段 | 2026-09-30 01:34:35 → 07:49:07 UTC（probe 完成），SIGINT 停止 |
| GPU / seed | 5/6/7、seed 0、deterministic |
| 吞吐 / 显存 | 中位 **1.56 s/iter**、**17.3 GiB/卡** |
| 落盘 | `epoch_2/4/6/8/10/12.pth`（`interval=2`），`latest.pth -> epoch_12.pth` |
| val 记录 | ep1-12 各一条，共 12 条（ep7 起写在新日志文件 `20260930_013447.log.json`） |
| 停止方式 | 看门狗检测 `Epoch(val) [12]` + `epoch_12.pth` 后 SIGINT，并自动完成三窗口 probe + source 探针；GPU 5/6/7 已释放 |

### 63.3 完整逐 epoch 曲线（ep1-12）

| ep | Overall3Dmod | vs base | OverallBEVmod | Car | Truck | Ped_l | Cyc_l |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20.1833 | +4.9193 | 27.0874 | 36.3145 | 9.8022 | 4.3559 | 30.2606 |
| 2 | 27.8215 | +1.4499 | 35.1231 | 37.0686 | 20.6826 | 21.4359 | 32.0988 |
| 3 | 32.0763 | +0.4172 | 37.7378 | 46.9572 | 17.8239 | 23.2287 | 40.2955 |
| 4 | 31.6728 | −1.1778 | 40.7212 | 45.6701 | 18.2141 | 24.2503 | 38.5569 |
| 5 | 33.3755 | −0.7072 | 42.7999 | 42.7265 | 23.9471 | 21.4883 | 45.3399 |
| 6 | 35.9289 | +1.9046 | 42.3272 | 51.2813 | 23.5005 | 27.1363 | 41.7976 |
| 7 | 37.3166 | −0.2535 | 45.8250 | 46.4100 | 25.6870 | 29.4780 | 47.6920 |
| 8 | 39.4070 | −0.1895 | 46.9580 | 51.6080 | 29.1920 | 28.7090 | 48.1180 |
| 9 | 39.4225 | **+0.8863** | 45.4270 | 50.9480 | 33.3060 | 27.7580 | 45.6790 |
| 10 | **39.6165** | **+1.5546** | 47.5660 | 52.8090 | 26.0730 | 31.0060 | 48.5780 |
| 11 | 38.9693 | +0.0061 | 47.1680 | 50.7660 | 30.3220 | 27.1590 | 47.6300 |
| 12 | 39.2805 | −0.0148 | 48.2560 | 51.2000 | 32.7010 | 27.0120 | 46.2080 |

### 63.4 窗口逐项均值与逐类别拆解（ep8-12，n=5）

| 指标 | no2d_igdr seed0 | MAI | 差 |
|---|---:|---:|---:|
| Overall 3D moderate | 38.8906 | **39.3392** | **+0.4485** |
| Car strict | 47.4711 | **51.4720** | **+4.0009** |
| Truck strict | 28.8510 | **30.3188** | **+1.4678** |
| Pedestrian loose | 29.5562 | 28.3290 | −1.2272 |
| Cyclist loose | 49.6841 | 47.2384 | −2.4457 |

读数：

1. **增益集中在 Car / Truck**（+4.00 / +1.47），与 ep6 的形态一致（ep6 Car 高 8.00）。
   3D moderate 的改善来自车辆类的严格 3D 回归。
2. **两个非机动车类转为负项**：Ped −1.23、Cyc −2.45。Cyc 在 ep6 是 −1.97，
   到 ep8-12 窗口扩大到 −2.45，是**唯一超过 2 点的类别缺口**。ep6 门控按单 epoch
   判定通过了「任一类别下降不超过 2」，但窗口口径下 Cyc 已经越过 2 —— 这一点必须
   与 Overall 的 +0.4485 并列记录，不能只报 Overall。
3. Overall 仍是正的，因为 Car 的 +4.00 在四类等权平均中抵消了 Ped/Cyc 的合计
   −3.67。

### 63.5 ep12 三窗口反事实 probe

| 窗口 | zero_h | shuffle_h | prior_h | shuffle/zero | 标准 RSSM ep12 shuffle_h | 倍数 |
|---|---:|---:|---:|---:|---:|---:|
| `[3..6]` | 0.3619 | 0.0349 | 0.0686 | 0.0965 | 0.0271 | 1.29× |
| `[100..103]` | 0.5298 | 0.0437 | 0.1002 | 0.0825 | 0.0471 | 0.93× |
| `[1800..1803]` | 0.5325 | 0.0594 | 0.0848 | 0.1116 | 0.0414 | 1.43× |

与 **ep6 自身**对照：

| 窗口 | ep6 shuffle_h | ep12 shuffle_h | 变化 |
|---|---:|---:|---:|
| `[3..6]` | **0.1494** | 0.0349 | **−76.6%** |
| `[100..103]` | 0.0811 | 0.0437 | −46.1% |
| `[1800..1803]` | 0.0568 | 0.0594 | +4.6% |

逐样本 `shuffle_z` head（4 条/窗口）：

| 窗口 | 样本 1 | 样本 2 | 样本 3 | 样本 4 |
|---|---:|---:|---:|---:|
| `[3..6]` | 0.0375 | 0.0329 | 0.0364 | 0.0328 |
| `[100..103]` | 0.0435 | 0.0440 | 0.0433 | 0.0439 |
| `[1800..1803]` | 0.0977 | 0.0431 | 0.0383 | 0.0586 |

读数（这是本节最重要的机制结论）：

1. **ep6 近距窗口的 14.94% 没有保持**，ep12 回落到 3.49%，且 4 条样本高度一致
   （3.28%–3.75%）——说明那不是「近距场景特别强」，而是 ep6 那个训练阶段的一过性
   状态。**若只看 ep6 会高估 MAI 的机制收益，必须并列 ep12。**
2. **ep12 相对标准 RSSM 的优势已缩小到同量级**：近距 1.29×、中距 0.93×（略低）、
   远距 1.43×。也就是说，到 ep12 时 z 的样本身份利用率与标准主线基本持平，
   **MAI 的 AP 增益不能再用「shuffle_z 大幅提升」来解释**。
3. `prior_only` head 在三个窗口都高于 `shuffle_z`（0.0686/0.1002/0.0848 vs
   0.0349/0.0437/0.0594），即「用先验替代后验」的影响大于「换另一个样本的后验」。
   这与 ep6 不同（近距 shuffle > prior）。

### 63.6 ep12 Source 探针（`stride=700`，indices `[3,703,1403]`）

| 张量 specific | 标准 RSSM ep12 | MAI ep6 | **MAI ep12** |
|---|---:|---:|---:|
| `feat_pooled_raw` | 29.96% | 15.32% | **45.87%** |
| `e_t_pooled_raw` | 66.35% | 27.43% | **64.28%** |
| `h_t` | 17.03% | 13.51% | 18.79% |
| `mu_p` | 18.21% | 11.56% | 17.86% |
| `mu_q` | 18.75% | 16.60% | **22.65%** |
| `correction` | 41.66% | 35.76% | **46.28%** |
| `innovation` | — | 47.93% | **66.39%** |
| `obs_prior_e_hat` | — | 24.29% | 60.80% |

RMS 与依赖方向：

| 量 | 标准 RSSM ep12 | MAI ep6 | MAI ep12 |
|---|---:|---:|---:|
| `h_t` RMS | 0.1673 | 0.0943 | 0.1061 |
| `mu_p` RMS | 0.3852 | 0.2888 | 0.3422 |
| `mu_q` RMS | 0.3865 | 0.3943 | 0.4583 |
| `correction` RMS | 0.0889 | 0.1650 | 0.1984 |
| `mu_q_removal_of_e_rel_l2` | 0.3787 | 0.3099 | **0.3320** |
| `mu_q_removal_of_h_rel_l2` | 0.8855 | 0.3098 | **0.2795** |
| **h/e 依赖比** | **2.338** | 0.9996 | **0.8418** |

读数：

1. **h/e 依赖比继续远离递归侧**：ep6 0.9996 → ep12 0.8418，同期标准主线是
   **2.338（反向移动）**。这是 MAI 最稳定的结构性差异，也是支持
   「Kalman 式 predict-correct」叙事的最强证据。
2. **correction / innovation 的样本身份同步上升**：correction specific
   35.76% → 46.28%，innovation 47.93% → 66.39%，obs_prior 预测 24.29% → 60.80%。
   注意 `obs_prior_e_hat` 的 specific 大涨（减掉公共成分后 `innovation` 更纯），
   这与第 62.12 节的机制解释一致。
3. **注意样本量口径**：本探针是 `limit=4 / stride=700` 的 4 样本读数，不是全验证集
   统计量。它适合追踪方向，不适合作为单点结论；`correction` specific 的
   ep6→ep12 上升（35.76%→46.28%）与 `shuffle_z` head 的下降（14.94%→3.49%）
   同时发生在 4 样本口径下，二者不矛盾但也不能互相替代。

### 63.7 ep12 训练诊断曲线

| ep | recon | kl_raw | clamped | mu_diff² | obs_pred | innovation² |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0.0086 | 0.5972 | 0.9852 | 0.0845 | 0.0292 | 0.6742 |
| 8 | 0.0073 | 0.5712 | 0.9888 | 0.0493 | 0.0260 | 0.6095 |
| 9 | 0.0071 | 0.5608 | 0.9896 | 0.0440 | 0.0257 | 0.4925 |
| 10 | 0.0064 | 0.5517 | 0.9910 | 0.0375 | 0.0233 | 0.4150 |
| 11 | 0.0067 | 0.5363 | 0.9910 | **0.0454** | 0.0235 | 0.3916 |
| 12 | 0.0057 | 0.5396 | 0.9918 | 0.0364 | 0.0212 | 0.2868 |

（每行 38 条训练日志均值；ep1-6 见 62.13。）

读数：

1. **`mu_diff²` 仍保持非单调**（ep10 0.0375 → ep11 0.0454 → ep12 0.0364），
   未回落到低维路线那种逐轮单调压缩的形态；但整体量级已从 ep1 的 1.2663 收敛到
   0.036，**说明它最终还是在收缩，只是不是单调塌缩**。
2. `innovation²` 从 ep6 的 0.708 继续降到 0.287（−59%），`obs_pred` 0.0276 → 0.0212。
   obs_prior 在持续学到更多可预测成分，innovation 的绝对能量随之下降 —— 这是
   设计预期内的行为，但也是「correction 越来越小」的另一面。
3. `kl_raw` 稳定在 0.54–0.60、`clamped_ratio` 0.99，与标准主线一致，未出现发散。

### 63.8 结论与下一步

1. **AP 通过预登记门控，但只是小幅领先。** ep8-12 均值 `+0.4485`（门槛 ≥−0.5，
   目标 +0.5），ep12 单点 −0.0148，全轮峰值 `39.6165 @ ep10`（比基线 ep10 高 1.55）。
   「恢复到基线 −1.0 内 → 继续 ep16」的条件成立，已启动 ep16 续训。
2. **机制侧出现了与 AP 相反的方向**：ep6 的近距 shuffle 14.94% 回落到 3.49%，
   中距 8.11% → 4.37%，整体相对标准 RSSM 的优势缩到 0.93–1.43×。
   **「z 的样本判别力大幅增强」这一说法只在 ep6 成立，不能作为 ep12 的结论。**
3. **唯一稳定且持续强化的机制证据是 h/e 依赖比**：0.9996 → 0.8418（标准主线
   同期 2.228 → 2.338，反向移动）。posterior 对观测/创新的依赖超过对递归状态的依赖，
   这是 MAI 相对标准主线**结构性**的差异，也是论文里最值得保留的一条。
4. **两个必须并列的负面读数**：① ep8-12 窗口下 Cyclist loose `−2.4457` 已越过 2 点，
   Pedestrian `−1.2272`；增益几乎全部来自 Car/Truck。② `innovation²` 与
   `mu_diff²` 仍在收缩，`correction` 的绝对能量在变小；结构改善了依赖方向，
   但没有阻止 correction 幅度衰减。
5. **按预登记继续 ep16**（`mai_ep16` + `mai_ep16_watch`）。ep16 必须复核：
   - ep12-16 五轮均值与基线 `40.4400 ± 1.3475`（三 seed）的差距；
   - `shuffle_z` 是否继续回落至低于标准主线（若低于，机制叙事不能成立）；
   - h/e 依赖比是否继续 <1（目前 0.8418）；
   - Cyclist 缺口是否继续扩大。
6. **用户第 6 步的触发条件尚未满足**：几何辅助监督（中心偏移/尺寸/朝向）应在
   「AP 保住但无提升」时才加。当前 AP 有 +0.4485 的提升、机制却在回落，
   属于「先查机制为何回落」而非「加新监督」的情形；若 ep16 的 AP 停滞且
   Cyclist 缺口继续扩大，再启用几何辅助监督，且必须保持单变量。

### 63.9 复现命令

```bash
cd /home/lurui/workspace/R4Det && source .envrc
CFG=configs/r4det/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_mai.py
WD=/data/lurui/work_dirs/fgfull_N4_no2d_igdr_mai_2x4_24e_seed0

# ep6 -> ep12 续训（看门狗：bash me_rssm/sanity/watch_mai_ep12.sh）
CUDA_VISIBLE_DEVICES=5,6,7 SEED=0 bash tools/dist_train.sh \
  $CFG 3 --seed 0 --deterministic \
  --resume-from $WD/epoch_6.pth --work-dir $WD > /tmp/mai_ep12.log 2>&1

# ep12 三窗口反事实诊断
CUDA_VISIBLE_DEVICES=5 python tools/diagnose_z_utilization.py --config $CFG --checkpoint $WD/epoch_12.pth \
  --start-index 3    --limit 4 --batch-size 4 --shuffle-offset 2  --output /tmp/z_util_mai_ep12_w3.json
CUDA_VISIBLE_DEVICES=6 python tools/diagnose_z_utilization.py --config $CFG --checkpoint $WD/epoch_12.pth \
  --start-index 100  --limit 4 --batch-size 4 --shuffle-offset 2  --output /tmp/z_util_mai_ep12_w100.json
CUDA_VISIBLE_DEVICES=7 python tools/diagnose_z_utilization.py --config $CFG --checkpoint $WD/epoch_12.pth \
  --start-index 1800 --limit 4 --batch-size 4 --shuffle-offset -1 --output /tmp/z_util_mai_ep12_w1800.json

# ep12 source 探针
CUDA_VISIBLE_DEVICES=5 python tools/probe_posterior_source.py --config $CFG --checkpoint $WD/epoch_12.pth \
  --start-index 3 --limit 4 --stride 700 --output /tmp/post_source_mai_ep12_stride700.json
```

---

## 64. MAI ep16 门控：同 seed0 领先、三 seed 持平，机制优势未保持（2026-09-30）

本节执行第 63.8 节预登记的 ep16 复核。训练从 `epoch_12.pth` 续跑至 ep16（24e
余弦日程与优化器状态保留），看门狗 `mai_ep16_watch`
（`me_rssm/sanity/watch_mai_ep16.sh`）在 ep16 验证与 `epoch_16.pth` 落盘后停训并自动
跑三窗口反事实 probe + source 探针。

### 64.0 状态与判定摘要

**三口径摘要**（16 条 val，跨度 ep1-16）：

| 口径 | 数值 |
|---|---|
| 区间均值（标准 ep12-16，五轮等权，n=5） | **40.1242 ± 0.8614**（样本 std） |
| 区间均值（实际跨度 ep13-16，n=4，仅供完整性） | **40.3351 ± 0.8323** |
| 区间均值（实际跨度 ep1-16，n=16，仅供完整性） | **36.0257** |
| 全轮峰值 | **41.4160 @ ep16** |
| best saved | **41.4160 @ ep16**（落盘 ep2/4/6/8/10/12/14/16 中最高） |

同窗口对照：

| 区间（ep12-16，n=5） | 基线 | MAI | 差 |
|---|---:|---:|---:|
| no2d_igdr seed0（配对同 seed） | 39.2809 | **40.1242** | **+0.8434** |
| no2d_igdr 三 seed 均值 ± std | 40.4400 ± 1.3475 | 40.1242 | −0.3158 |
| MAI 全轮峰值 | — | **41.4160 @ ep16** | 基线三 seed 峰值 41.8296 ± 1.3843 |

**ep16 门控表**：

| 门控项 | 预登记检查内容 | ep16 实测 | 判定 |
|---|---|---|---|
| AP：同 seed0 配对 | 不落后基线超过 1 点 | **+0.8434** | **通过并小幅领先** |
| AP：三 seed 对照 | 检查是否进入 seed 间噪声范围 | **−0.3158**（基线 std 1.3475） | **持平，不能判为已提升** |
| 全轮峰值 | 记录值与 epoch | **41.4160 @ ep16**（全程最高） | **后期仍在爬升** |
| `shuffle_z`（近/中/远） | 是否继续回落至低于标准主线 | 3.8653% / 4.1821% / 5.4094% | **未继续大幅回落；好于标准近/远，中距略低** |
| h/e 依赖比 | 是否继续 <1 | **1.1236**（ep12 0.8418） | **未守住；重新偏向 h 侧** |
| `correction` specific | 保持 ≥10% | **38.01%**（ep12 46.28%） | 仍高，但回落 |
| `innovation` specific | 保持 > `e_pooled` | **67.03% > 60.22%** | **通过，余量 6.82 点** |
| Cyclist 缺口 | 是否继续扩大 | ep12-16 vs seed0 **−1.4454**（ep8-12 为 −2.4457） | **未扩大，反而收窄** |
| `mu_diff²` 稳定性 | 不回到单调塌缩 | 0.0338 / 0.0345 / 0.0325 / 0.0335 | **形成平台** |

**一句话结论：MAI 在 ep16 的 AP 相对同 seed0 基线仍有 +0.84 的优势，但相对三 seed
均值只差 −0.32，落在 seed 间噪声内；同时 ep6 的强身份信号优势没有保持，h/e 依赖比
在 ep12 触底后重新回升到 1.12。当前结果可继续做 seed 复现，不能把 AP 增益归因于
`z_t` 样本判别力的持续增强。**

### 64.1 配置与唯一变量

| 项 | 值 |
|---|---|
| 配置 | `configs/r4det/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_mai.py` |
| 对照 | `..._no2d_igdr.py`（`MotionAlignedRSSMFusion`） |
| 唯一变量 | `posterior_struct='innovation'` + `obs_pred_loss_weight=0.05` |
| ep16 续训 | `--resume-from epoch_12.pth`，24e 余弦日程与优化器状态保留（`max_epochs` 未改） |
| 诊断键 | `stat_obs_pred`、`stat_innovation_sq`、`stat_mu_diff_sq`、`stat_kl_raw_mean`、`stat_clamped_ratio`（均不含 `loss`） |

### 64.2 运行事实

| 项 | 值 |
|---|---|
| ep6 段 | 2026-09-29 13:05:59 → 19:15:12 UTC（ep6 val 落盘），SIGINT 停止 |
| ep12 段 | 2026-09-30 01:34:35 → 07:49:07 UTC（probe 完成），SIGINT 停止 |
| ep16 段 | 2026-09-30 10:37:10 UTC（resume）→ 14:42:51 UTC（看门狗 SIGINT） |
| ep16 val / 权重 | val 14:42:29 UTC；`epoch_16.pth` 14:31 落盘；`latest.pth -> epoch_16.pth` |
| probe 完成 | 三窗口反事实 + source 探针于 14:44:44 UTC 完成；GPU 5/6/7 已释放 |
| GPU / seed | 5/6/7、seed 0、deterministic |
| 吞吐 / 显存 | 中位 **约 1.58 s/iter**、**17.3 GiB/卡** |
| 落盘 | `epoch_2/4/6/8/10/12/14/16.pth`（`interval=2`） |
| val 记录 | ep1-16 各一条，共 16 条；ep7-12 在 `20260930_013447.log.json`，ep13-16 在 `20260930_103722.log.json` |
| 停止方式 | 看门狗检测 `Epoch(val) [16]` + `epoch_16.pth` 后 SIGINT；DDP worker 收到 SIGINT 退出码 1 属预期 |

### 64.3 完整逐 epoch 曲线（ep1-16）

| ep | Overall3Dmod | vs seed0 | OverallBEVmod | Car | Truck | Ped_l | Cyc_l |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20.1833 | +4.9193 | 27.0874 | 36.3145 | 9.8022 | 4.3559 | 30.2606 |
| 2 | 27.8215 | +1.4499 | 35.1231 | 37.0686 | 20.6826 | 21.4359 | 32.0988 |
| 3 | 32.0763 | +0.4172 | 37.7378 | 46.9572 | 17.8239 | 23.2287 | 40.2955 |
| 4 | 31.6728 | −1.1778 | 40.7212 | 45.6701 | 18.2141 | 24.2503 | 38.5569 |
| 5 | 33.3755 | −0.7072 | 42.7999 | 42.7265 | 23.9471 | 21.4883 | 45.3399 |
| 6 | 35.9289 | +1.9046 | 42.3272 | 51.2813 | 23.5005 | 27.1363 | 41.7976 |
| 7 | 37.3166 | −0.2535 | 45.8249 | 46.4096 | 25.6872 | 29.4782 | 47.6916 |
| 8 | 39.4070 | −0.1895 | 46.9582 | 51.6078 | 29.1923 | 28.7095 | 48.1184 |
| 9 | 39.4225 | +0.8863 | 45.4269 | 50.9479 | 33.3055 | 27.7575 | 45.6790 |
| 10 | 39.6165 | +1.5546 | 47.5662 | 52.8086 | 26.0729 | 31.0060 | 48.5783 |
| 11 | 38.9693 | +0.0061 | 47.1684 | 50.7657 | 30.3219 | 27.1590 | 47.6305 |
| 12 | 39.2805 | −0.0148 | 48.2564 | 51.2005 | 32.7012 | 27.0123 | 46.2080 |
| 13 | 39.7458 | +1.8280 | 47.9759 | 57.4270 | 28.7722 | 26.7505 | 46.0334 |
| 14 | 39.6193 | −0.7660 | 48.2395 | 55.8287 | 26.5682 | 29.4912 | 46.5890 |
| 15 | 40.5594 | +1.7483 | 48.1714 | 53.7946 | 33.4012 | 27.2297 | 47.8120 |
| 16 | **41.4160** | **+1.4211** | 48.3136 | 56.4196 | 32.7790 | 27.1785 | 49.2870 |

读数：

1. **尾部仍在爬升而非提前平台化**：ep12 39.2805 → ep13 39.7458 → ep14 39.6193 →
   ep15 40.5594 → ep16 41.4160。ep16 是全程最高单点，且 `epoch_16.pth` 已落盘，
   不存在「峰值落在未保存 epoch」的问题。
2. **ep13-14 的 Car 早峰（57.43/55.83）没有保持，但 ep15-16 由 Truck/Cyclist 接棒**
   （Truck 33.40/32.78、Cyc 47.81/49.29）。单类别波动仍是本 run 的主要不确定性。
3. **Ped 在 ep13-16 始终落在 26.75–29.49**，而 seed0 基线同窗口为 28.02–34.25，
   且 ep16 为 32.76；MAI 没有跟上该轮回升，这是窗口均值主要的负项来源。

### 64.4 窗口逐项均值与逐类别拆解（ep12-16，n=5）

**对同 seed0 基线（配对）**：

| 指标 | no2d_igdr seed0 | MAI | 差 |
|---|---:|---:|---:|
| Overall 3D moderate | 39.2809 | **40.1242** | **+0.8434** |
| Overall BEV moderate | 46.5179 | **48.1914** | **+1.6735** |
| Car strict | 47.8399 | **54.9341** | **+7.0942** |
| Truck strict | 29.7246 | **30.8444** | **+1.1198** |
| Pedestrian loose | 30.9277 | 27.5324 | **−3.3953** |
| Cyclist loose | 48.6313 | 47.1859 | −1.4454 |

**对 no2d_igdr 三 seed 均值（更严格口径）**：

| 指标 | 三 seed 均值 ± std | MAI | 差 |
|---|---:|---:|---:|
| Overall 3D moderate | 40.4400 ± 1.3475 | 40.1242 | −0.3158 |
| Car strict | 50.9069 ± 3.0553 | **54.9341** | **+4.0272** |
| Truck strict | 31.2227 ± 1.4337 | 30.8444 | −0.3783 |
| Pedestrian loose | 29.2018 ± 1.6813 | 27.5324 | −1.6694 |
| Cyclist loose | 50.4286 ± 2.4763 | 47.1859 | **−3.2427** |

读数：

1. **同 seed0 配对口径下 MAI 是明确领先的**：Overall +0.84，Car +7.09，Truck +1.12。
   在 seed0 自身曲线上，ep13-16 的 Overall 增量为 +1.83 / −0.77 / +1.75 / +1.42，
   说明这不是单个 epoch 的偶然对点。
2. **但三 seed 口径下只剩 Car 显著为正**：Overall −0.32、Truck −0.38、Ped −1.67、
   Cyc −3.24。**「MAI 提升 3D 检测」目前是单 seed 结论；seed0 本身恰好是三个基线中
   Cyc 最弱、Ped 最强的 seed，直接放大了 MAI 的 Car/Cyc 相对优势。**
3. **Cyc 缺口收窄但未消失**：ep8-12 对 seed0 的 −2.4457 收窄到 ep12-16 的 −1.4454，
   对三 seed 均值仍是 −3.2427。Ped 反而从 ep8-12 的 −1.2272 扩大到 −3.3953，
   两个非机动车类的负项在窗口口径下仍然存在，只是从 Cyc 转移到 Ped。

### 64.5 ep16 三窗口反事实 probe

**ep16 响应均值（4 样本/窗口）**：

| 窗口 | zero_z | shuffle_z | prior_only | shuffle/zero |
|---|---:|---:|---:|---:|
| `[3..6]` | 38.3418% | 3.8653% | 6.7991% | 0.1008 |
| `[100..103]` | 54.6394% | 4.1821% | 11.0353% | 0.0765 |
| `[1800..1803]` | 54.5877% | 5.4094% | 9.3570% | 0.0991 |

**与 ep12 自身及标准 RSSM ep12 对照**：

| 窗口 | MAI ep12 shuffle_h | MAI ep16 shuffle_h | 标准 RSSM ep12 shuffle_h | ep16/标准 |
|---|---:|---:|---:|---:|
| `[3..6]` | 3.4918% | 3.8653% | 2.71% | 1.43× |
| `[100..103]` | 4.3686% | 4.1821% | 4.71% | 0.89× |
| `[1800..1803]` | 5.9410% | 5.4094% | 4.14% | 1.31× |

**ep16 逐样本 shuffle_z head（4 条/窗口）**：

| 窗口 | 样本 1 | 样本 2 | 样本 3 | 样本 4 |
|---|---:|---:|---:|---:|
| `[3..6]` | 4.0313% | 3.7228% | 4.0504% | 3.6567% |
| `[100..103]` | 4.0895% | 4.2902% | 4.0810% | 4.2678% |
| `[1800..1803]` | 8.6919% | 4.4331% | 3.4436% | 5.0692% |

读数：

1. **近/远距相对标准主线仍有 1.3–1.4× 的优势，但中距略低**。ep16 没有像 ep12 那样
   相对标准主线全面退平，但也远未回到 ep6 的 14.94% / 8.11% / 5.68%。
2. **逐样本离散在远距窗口重新出现**：1800 为 8.69%，1802 只有 3.44%。
   中距窗口四样本最稳定（4.08%–4.29%），可以作为「中距有稳定但弱身份响应」的证据，
   不能把近距单样本波动当作整体机制增强。
3. **shuffle/zero 比值仍只有 0.077–0.101**，标准 RSSM ep12 同窗口为
   近/中/远 `0.103 / 0.099 / 0.076`。**MAI 的 z 因果占比在 ep16 与标准主线同量级，
   没有形成稳定的机制性优势。**

### 64.6 ep16 Source 探针（`stride=700`，indices `[3,703,1403]`）

| 张量 specific | 标准 RSSM ep12 | MAI ep12 | **MAI ep16** |
|---|---:|---:|---:|
| `feat_pooled_raw` | 29.96% | 45.87% | **38.52%** |
| `e_t_pooled_raw` | 66.35% | 64.28% | **60.22%** |
| `h_t` | 17.03% | 18.79% | **14.62%** |
| `mu_p` | 18.21% | 17.86% | **11.90%** |
| `mu_q` | 18.75% | 22.66% | **16.56%** |
| `correction` | 41.66% | 46.28% | **38.01%** |
| `innovation` | — | 66.39% | **67.03%** |
| `obs_prior_e_hat` | — | 60.80% | **55.12%** |

RMS 与依赖方向：

| 量 | 标准 RSSM ep12 | MAI ep12 | MAI ep16 |
|---|---:|---:|---:|
| `h_t` RMS | 0.1673 | 0.1061 | 0.1259 |
| `mu_p` RMS | 0.3852 | 0.3422 | 0.3746 |
| `mu_q` RMS | 0.3865 | 0.4583 | 0.4942 |
| `correction` RMS | 0.0889 | 0.1984 | 0.1901 |
| `mu_q_removal_of_e_rel_l2` | 0.3787 | 0.3320 | 0.2592 |
| `mu_q_removal_of_h_rel_l2` | 0.8855 | 0.2795 | 0.2912 |
| **h/e 依赖比** | **2.338** | **0.8418** | **1.1236** |

读数：

1. **h/e 依赖比在 ep12 触底后重新回升到 1.1236**：ep12 的 0.8418 是 MAI 全程最强的
   「后验偏向观测」证据，但到 ep16 已不再 <1。标准主线同期仍在 2.338，因此 MAI
   仍然明显比标准主线更靠观测侧，**但变化方向已经反转**。
2. **correction 与 innovation 的 specific 同步回落**：correction 46.28% → 38.01%，
   obs_prior 60.80% → 55.12%，只有 innovation 维持 67.03%。这说明 innovation
   作为残差更纯，但后验头对它的利用没有继续增强。
3. **`mu_q` RMS 继续上升到 0.4942**，而 correction RMS 基本持平 0.1901；
   后验均值变大主要来自先验侧的公共分量（`mu_p` specific 从 17.86% 降到 11.90%），
   不是 correction 的样本特异能量增加。这正是「AP 提升但机制优势没保持」的直接读数。
4. **样本量口径**：该探针为 `limit=4 / stride=700` 的 4 样本读数，适合追踪方向，
   不能当作全验证集统计量；所有 specific 结论都应同时标注这一点。

### 64.7 ep12-16 训练诊断曲线

| ep | recon | kl_raw | clamped | mu_diff² | obs_pred | innovation² |
|---:|---:|---:|---:|---:|---:|---:|
| 12 | 0.0057 | 0.5396 | 0.9918 | 0.0364 | 0.0212 | 0.2868 |
| 13 | 0.0053 | 0.5476 | 0.9924 | 0.0338 | 0.0192 | 0.2171 |
| 14 | 0.0050 | 0.5331 | 0.9925 | 0.0345 | 0.0190 | 0.1905 |
| 15 | 0.0046 | 0.5400 | 0.9929 | 0.0325 | 0.0178 | 0.1704 |
| 16 | 0.0044 | 0.5383 | 0.9931 | 0.0335 | 0.0174 | 0.1595 |

（每行 38 条训练日志均值；ep1-11 见第 62.13、63.7 节。）

读数：

1. **`mu_diff²` 形成平台**：ep12-16 为 0.0364/0.0338/0.0345/0.0325/0.0335，
   没有回到低维路线那种逐轮单调塌缩；但绝对量级比 ep1 的 1.2663 小两个数量级，
   correction 能量仍处在很小的区间。
2. **`innovation²` 继续下降 0.2868 → 0.1595（−44%）**，同期 `obs_pred` 0.0212 → 0.0174。
   obs_prior 在持续吸收可预测成分，innovation 的绝对能量随之下降；这与 64.6 节
   「innovation specific 高但 correction 利用没有增强」一致。
3. **KL 与重建均稳定**：`kl_raw` 0.533–0.548、`clamped` 0.9918–0.9931、recon
   0.0057→0.0044，无发散、无 NaN/Inf，也未出现 KL 数量级上升。当前不需要用
   free-bits/KL balancing 解释 AP 变化。

### 64.8 结论与下一步

1. **AP：同 seed0 通过并反超，三 seed 口径只是持平。** ep12-16 五轮均值 40.1242，
   seed0 为 39.2809（+0.8434），三 seed 为 40.4400 ± 1.3475（−0.3158）。
   全轮峰值 41.4160 @ ep16，是全程最高且已落盘。**结论必须双口径表述：单 seed
   结果支持继续，但不能据此宣布相对三 seed 基线的性能提升。**
2. **机制：ep6 的优势没有保持。** `shuffle_z` 从 ep6 的 14.94%/8.11%/5.68% 回落到
   ep16 的 3.87%/4.18%/5.41%，仅近/远比标准主线高 1.3–1.4×，中距低 0.89×；
   h/e 依赖比 0.8418 → 1.1236，correction specific 46.28% → 38.01%。**AP 尾部提升
   不能归因于「z 身份信号持续增强」，论文叙事若使用 MAI，必须把 ep6 与 ep16 的
   机制反差如实写出。**
3. **逐类别：Car/Truck 仍是收益来源，Ped/Cyc 是剩余缺口。** 对 seed0，Car +7.09、
   Truck +1.12、Ped −3.40、Cyc −1.45；对三 seed 均值，Car +4.03 为正，
   Truck −0.38 基本持平，Ped −1.67、Cyc −3.24。Cyc 比 ep8-12 收窄，
   但 Ped 缺口扩大，两者都不能忽略。
4. **稳定性：没有训练崩塌。** `mu_diff²` 平台、KL 有界、recon 继续下降、innovation
   有界；因此本轮不是训练失败，而是「同 seed0 有效、跨 seed 未证实 + 机制回落」。
5. **下一步优先级（不改当前配置）**：
   - 先复现 seed1/seed2 的 ep12-16，确认 MAI 的 +0.84 是方法收益还是 seed0 对点波动；
   - 复现时同步跑三窗口 probe，检查 h/e 依赖比是否在所有 seed 都回到 >1；
   - 只有 seed 复现显示 AP 停滞且 Cyc/Ped 缺口继续扩大时，才按用户第 6 步单变量加入
     几何辅助监督（中心偏移 dx/dy/dz、尺寸 log(w/l/h)、朝向 sin/cos(yaw)，正目标中心，
     权重 0.02–0.05）；
   - 路线 E（KL balancing + 按 cell 聚合 free-bits）仍放在最后，不作为当前 AP/机制
     问题的首要修复手段。

### 64.9 复现命令

```bash
cd /home/lurui/workspace/R4Det && source .envrc
CFG=configs/r4det/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_mai.py
WD=/data/lurui/work_dirs/fgfull_N4_no2d_igdr_mai_2x4_24e_seed0

# ep12 -> ep16 续训（看门狗：bash me_rssm/sanity/watch_mai_ep16.sh）
CUDA_VISIBLE_DEVICES=5,6,7 SEED=0 bash tools/dist_train.sh \
  $CFG 3 --seed 0 --deterministic \
  --resume-from $WD/epoch_12.pth --work-dir $WD > /tmp/mai_ep16.log 2>&1

# 三口径 / 逐类别 / 窗口均值
python3 tools/summarize_run.py $WD --window 12-16

# ep16 三窗口反事实诊断
CUDA_VISIBLE_DEVICES=5 python tools/diagnose_z_utilization.py --config $CFG --checkpoint $WD/epoch_16.pth \
  --start-index 3    --limit 4 --batch-size 4 --shuffle-offset 2  --output /tmp/z_util_mai_ep16_w3.json
CUDA_VISIBLE_DEVICES=6 python tools/diagnose_z_utilization.py --config $CFG --checkpoint $WD/epoch_16.pth \
  --start-index 100  --limit 4 --batch-size 4 --shuffle-offset 2  --output /tmp/z_util_mai_ep16_w100.json
CUDA_VISIBLE_DEVICES=7 python tools/diagnose_z_utilization.py --config $CFG --checkpoint $WD/epoch_16.pth \
  --start-index 1800 --limit 4 --batch-size 4 --shuffle-offset -1 --output /tmp/z_util_mai_ep16_w1800.json

# ep16 source 探针
CUDA_VISIBLE_DEVICES=5 python tools/probe_posterior_source.py --config $CFG --checkpoint $WD/epoch_16.pth \
  --start-index 3 --limit 4 --stride 700 --output /tmp/post_source_mai_ep16_stride700.json
```

---

## 65. MAI seed1 → seed2 自动续接运行记录（2026-10-01 ~ 2026-10-02，已完成）

本节记录链式 watcher 的运行事实、保护机制，以及 seed1/seed2 在各自 ep16 闸门
完成后的指标与机制 probe。seed1 结果见 65.6/65.7，seed2 结果见 65.9；两者的
ep12-16 窗口都只与**同 seed** 的 `no2d_igdr` 基线配对，不做跨 seed 单点攀比。

### 65.1 配置与唯一变量

- 配置：`configs/r4det/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_mai.py`。
- 唯一变量：`seed`。seed1 与 seed2 都使用同一 MAI 配置、同一 GPU 5/6/7、同一
  `--deterministic` 与同一 24e 日程；差异仅为 `--seed` 和 work_dir/日志路径。
- 运行保护：不修改 `max_epochs`，由 ep16 闸门在 `Epoch(val) [16]` 与
  `epoch_16.pth` 落盘后发 SIGINT 停止训练。

### 65.2 运行事实

- seed1：2026-10-01 03:07:49 UTC 启动，tmux `mai_seed1_ep16`；work_dir
  `/data/lurui/work_dirs/fgfull_N4_no2d_igdr_mai_2x4_24e_seed1`，日志
  `/tmp/mai_seed1_ep16.log`。
- seed1：ep1-16 全部完成（ep16 于 2026-10-01 19:25 UTC 落盘，19:36:54 命中闸门）。
  **已完成**，结果见 65.6/65.7。
- 2026-10-01 17:13:27 UTC 停用旧的 seed1 专用 watcher，启动链式 watcher
  `me_rssm/sanity/watch_mai_seed1_then_seed2_ep16.sh`，tmux 会话仍为
  `mai_seed1_ep16_watch`。
- 链式 watcher 的交接顺序固定为：seed1 ep16 闸门 → 停 seed1 → seed1 四个 probe
  → 确认 GPU 5/6/7 空闲 → 启动 seed2 → seed2 ep16 闸门 → 停 seed2 → seed2 四个 probe。
- seed2：work_dir
  `/data/lurui/work_dirs/fgfull_N4_no2d_igdr_mai_2x4_24e_seed2`，日志
  `/tmp/mai_seed2_ep16.log`，tmux `mai_seed2_ep16`。2026-10-02 19:17:20 UTC
  `epoch_16.pth` 落盘、19:28:29 UTC 写出 `Epoch(val) [16]`，**已完成**；
  19:29:17 停训并释放 GPU 5/6/7，四个 probe 于 19:30:22 全部落盘，链式 watcher
  `chain finished rc=0`。

### 65.3 无人值守保护与隔离彩排

链式 watcher 在正式上线前已做独立彩排，验证过：

1. seed1 必须真正退出、且四个 probe 结束后才允许 seed2 启动，避免两个 DDP 任务抢 GPU；
2. `Epoch(val) [16]` 只在 watcher 启动后新增的日志行中计数，配合权重尺寸和
   文件落定时间，避免复用旧日志或正在写入的半成品 checkpoint；
3. seed2 训练进程若在 ep16 前退出，会清理残留 tmux 会话并从 work_dir 中最近一个
   完整 `epoch_*.pth` 自动恢复，最多尝试 3 次；
4. 单个 probe 有 30 min 硬超时；probe 失败只记录警告，不阻塞后续训练阶段；
5. dry-run 会校验融合类型 `MotionAlignedInnovationRSSMFusion`、`max_epochs=24`、
   `evaluation.interval=1`、`checkpoint_config.interval=2`、`val_bs=4` 和 ep16 的
    checkpoint 覆盖关系。

### 65.4 完成状态

- seed1 ep16 结果：**已完成**。
- seed2 ep16 结果：**已完成**。2026-10-02 03:01:08 UTC 以 `START_PHASE=seed2`
  接管启动，03:05 进入 `Epoch [1][100/1902]`，19:17:20 UTC `epoch_16.pth` 落盘，
  19:28:29 UTC 写出 ep16 val；停训后四个 probe 于 19:30:22 UTC 全部输出，
  链式 watcher 以 rc=0 正常退出。
- 三 seed（seed0/1/2）MAI 结果与同 seed 配对差值已齐，见 65.9 与第 67 节。

**seed2 运行事实补充**：seed2 从零开始一次连续训练，中途无重启、无 OOM、无 NaN。
日志中的 `Traceback` 出现在 19:28:29 的 ep16 val 之后，是 watcher 发 SIGINT 触发
torch distributed elastic shutdown 的正常收尾栈（`KeyboardInterrupt` →
`empty_cache` 二次中断 → worker 被 SIGKILL），不是训练失败；`epoch_16.pth`
已在此之前完整落盘，随后四个 probe 与 18e 接力均正常执行。

### 65.5 首夜中断与恢复（2026-10-02）

链式 watcher 在 2026-10-01 19:36:54 UTC 正确命中 seed1 ep16（`epoch_16.pth` 于
19:25 落盘、ep16 验证行已写），随即发 SIGINT。但 torch distributed 的退出流程
（`KeyboardInterrupt` → elastic shutdown → `empty_cache` 二次中断 → 强制 SIGKILL
worker）耗时超过当时 `STOP_TIMEOUT_SEC=600`；watcher 在 19:46:55 记录
`FATAL: seed1 did not stop cleanly` 并以 rc=1 退出，因此 **seed2 没有在当夜自动启动**。
这不是权重或验证失败：checkpoint 完整（694150460 B），GPU 5/6/7 在训练进程退出后
全部空闲。seed1 的四个 ep16 probe 由人工补跑，输出与 64.5/64.6 同路径。

修复（commit 见下一节）：

1. `STOP_TIMEOUT_SEC` 600 → 1800 s，覆盖 elastic shutdown 的长尾；
2. 停训等待超时后，只要 `nvidia-smi` 显示 GPU 5/6/7 已空闲，就视为停止成功，
   不再让整条链因进程表残留而退出；
3. 新增 `START_PHASE=seed2`，允许在 seed1 已完成、GPU 已释放的情况下从第二阶段接管，
   不需要重跑 seed1。

### 65.6 MAI seed1 ep16 结果（2026-10-02）

**三口径摘要**（16 条 val，跨度 ep1-16）：

| 口径 | 数值 |
|---|---|
| 区间均值（标准 ep12-16，五轮等权，n=5） | **39.4122 ± 0.8133**（样本 std） |
| 全轮峰值（所有 val epoch 的最高单点） | **42.7727 @ ep6** |
| best saved（实际落盘权重中最高的 val） | **42.7727 @ ep6**（落盘 ep2/4/6/8/10/12/14/16） |

对照同 seed1 的 `no2d_igdr` 基线（ep12-16 = 41.9185）：

| 指标 | MAI seed1 | 同 seed1 基线 | Δ |
|---|---:|---:|---:|
| Overall 3D moderate（ep12-16 均值） | 39.4122 | 41.9185 | **−2.5063** |
| ep16 单点 | 40.1620 | 40.6789（同 seed1 ep16） | −0.5169 |

`seed0` 同 seed 配对为 +0.8434，`seed1` 为 −2.5063：**MAI 的单 seed 正收益没有复现，
方向在两个 seed 之间翻转。** 在 seed2 完成前三 seed 均值结论不能成立；即使 seed2
为正，seed0/seed1 的符号不一致也必须写进论文限制条件。

**完整逐 epoch Overall 3D moderate**（n=16，来自
`20261001_030801.log.json`）：

| ep | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Overall | 17.82 | 28.14 | 33.26 | 37.36 | 36.61 | **42.77** | 38.83 | 36.60 | 39.86 | 37.52 | 38.75 | 38.82 | 38.92 | 38.73 | 40.43 | 40.16 |

**ep12-16 逐类别均值（n=5）**：

| 类别 | 均值 | std |
|---|---:|---:|
| Car strict | 51.9871 | 2.5279 |
| Truck strict | 33.9519 | 2.1321 |
| Pedestrian loose | 24.3482 | 1.9708 |
| Cyclist loose | 47.3615 | 1.6668 |

### 65.7 MAI seed1 ep16 机制 probe

三窗口反事实 `shuffle_z`/`zero_z`/`prior_only` head ratio（4 样本均值）：

| 窗口 | shuffle_z head | zero_z head | prior_only head |
|---|---:|---:|---:|
| `[3..6]` | 4.7950% | 41.7327% | 5.1582% |
| `[100..103]` | 5.1452% | 50.0380% | 5.9372% |
| `[1800..1803]` | 5.8338% | 55.5174% | 8.6974% |

`shuffle_z` 相对 seed0 同窗口（3.87%/4.18%/5.41%）略高，说明身份信号仍在；但
seed1 的 AP 反而低于基线。**这与 seed0 的结论不同：MAI 在 seed1 上同样有可测的
z 因果效应，却没有转化为 AP。**

`stride=700` source probe（3 样本）：

| 张量 | sample-specific 能量占比 |
|---|---:|
| `feat_pooled_raw` | 22.58% |
| `e_t_pooled` | 51.08% |
| `h_t` | 15.08% |
| `mu_p` | 11.08% |
| `mu_q` | 16.48% |
| `correction` | 38.49% |
| `innovation` | 61.94% |

h/e 依赖比 = `mu_q` 对 h/e 的依赖比：`h_t` specific 15.08% / `e_t` specific 51.08%
（上游信息充足），posterior 侧 `correction` specific 38.49%、`innovation` specific
61.94%。形式与 seed0 ep16 一致，但决定性的 AP 指标为负。

**seed1 ep1-16 训练诊断（每 epoch 全部训练日志均值）**：

| ep | kl_raw | clamped | mu_diff² | obs_pred | innovation² |
|---:|---:|---:|---:|---:|---:|
| 1 | 16.1993 | 0.8132 | 1.2877 | 0.0417 | 0.9813 |
| 2 | 0.9261 | 0.9347 | 0.3243 | 0.0299 | 0.9271 |
| 3 | 0.6425 | 0.9722 | 0.2209 | 0.0274 | 0.8735 |
| 4 | 0.6491 | 0.9801 | 0.1375 | 0.0256 | 0.8300 |
| 5 | 0.6380 | 0.9833 | 0.1099 | 0.0238 | 0.7790 |
| 6 | 0.6344 | 0.9845 | 0.0930 | 0.0242 | 0.7417 |
| 7 | 0.5806 | 0.9872 | 0.0495 | 0.0258 | 0.6096 |
| 8 | 0.5976 | 0.9863 | 0.0856 | 0.0270 | 0.5124 |
| 9 | 0.5566 | 0.9894 | 0.0446 | 0.0242 | 0.3894 |
| 10 | 0.5639 | 0.9911 | 0.0358 | 0.0210 | 0.2568 |
| 11 | 0.5540 | 0.9914 | 0.0375 | 0.0191 | 0.1886 |
| 12 | 0.5370 | 0.9914 | 0.0374 | 0.0189 | 0.1503 |
| 13 | 0.5460 | 0.9923 | 0.0340 | 0.0173 | 0.1338 |
| 14 | 0.5300 | 0.9916 | 0.0417 | 0.0184 | 0.1266 |
| 15 | 0.5373 | 0.9928 | 0.0335 | 0.0161 | 0.1129 |
| 16 | 0.5393 | 0.9930 | 0.0342 | 0.0160 | 0.1001 |

（`stat_recon_mse` 未写入该 log.json 的聚合键，逐 iter 可从 `/tmp/mai_seed1_ep16.log`
读取；KL、clamped、`mu_diff²`、obs_pred、innovation² 均与 seed0 同量级。）

### 65.8 MAI seed2 ep16 结果（2026-10-02，已完成）

**三口径摘要**（16 条 val，跨度 ep1-16；work_dir
`/data/lurui/work_dirs/fgfull_N4_no2d_igdr_mai_2x4_24e_seed2`，日志
`20261002_030120.log.json`）：

| 口径 | 数值 |
|---|---|
| 区间均值（标准 ep12-16，五轮等权，n=5） | **40.0222 ± 0.8788**（样本 std） |
| 全轮峰值（所有 val epoch 的最高单点） | **41.3163 @ ep12** |
| best saved（实际落盘权重中最高的 val） | **41.3163 @ ep12**（落盘 ep2/4/6/8/10/12/14/16） |
| last val | **39.7109 @ ep16** |

对照同 seed2 的 `no2d_igdr` 基线（ep12-16 = 40.1207）：

| 指标 | MAI seed2 | 同 seed2 基线 | Δ |
|---|---:|---:|---:|
| Overall 3D moderate（ep12-16 均值） | 40.0222 | 40.1207 | **−0.0985** |
| 全轮峰值 | 41.3163 @ ep12 | 41.9586 @ ep14 | −0.6423 |
| ep16 单点 | 39.7109 | 40.3706 | −0.6597 |

**完整逐 epoch Overall 3D moderate 与 Overall BEV moderate**（n=16）：

| ep | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3D mod | 17.3735 | 26.8293 | 33.9451 | 34.6424 | 36.1371 | 37.3194 | 39.3600 | 40.1097 | 39.2906 | 37.6829 | 39.9347 | **41.3163** | 39.9468 | 40.2394 | 38.8975 | 39.7109 |
| BEV mod | 25.1195 | 33.9176 | 41.2242 | 44.6744 | 46.7616 | 45.5495 | 46.0415 | 47.8383 | 47.4018 | 45.6421 | 48.8149 | 49.4224 | 48.5435 | 48.2353 | 46.9556 | 48.2163 |

**ep12-16 逐类别均值（n=5）**：

| 类别 | 均值 | std |
|---|---:|---:|
| Car strict | 52.6734 | 2.3524 |
| Truck strict | 31.3385 | 2.1792 |
| Pedestrian loose | 28.1415 | 1.2788 |
| Cyclist loose | 47.9353 | 1.8414 |

辅助项（同窗口，n=5）：Car loose 76.6245、Truck loose 52.7951、
Ped strict 0.3147、Cyc strict 26.1640、Overall BEV moderate 48.2746。

**同 seed2 ep12-16 逐类别配对（MAI − 基线）**：Car strict `52.6734 − 50.9304 = +1.7430`、
Truck strict `31.3385 − 32.5818 = −1.2433`、Ped loose `28.1415 − 27.5690 = +0.5725`、
Cyc loose `47.9353 − 49.4014 = −1.4661`。seed2 的类别形态与 seed0/seed1 不同：
**Truck 由正转负、Ped 由负转正**，说明类别级增益本身也不跨 seed 稳定。

**seed2 ep1-16 训练诊断**（每 epoch 全部训练日志均值）：

| ep | recon | kl_raw | clamped | mu_diff² | prior_std | posterior_std | obs_pred | innovation² |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0855 | 14.7329 | 0.8324 | 1.1674 | 0.2003 | 0.1578 | 0.0492 | 0.9811 |
| 2 | 0.0143 | 0.8461 | 0.9362 | 0.2569 | 0.3172 | 0.1463 | 0.0431 | 0.8815 |
| 3 | 0.0107 | 0.6670 | 0.9638 | 0.1734 | 0.4231 | 0.1730 | 0.0485 | 0.7140 |
| 4 | 0.0103 | 0.6440 | 0.9712 | 0.1294 | 0.4424 | 0.1820 | 0.0451 | 0.5299 |
| 5 | 0.0098 | 0.5988 | 0.9791 | 0.0710 | 0.4590 | 0.1946 | 0.0365 | 0.2986 |
| 6 | 0.0097 | 0.6090 | 0.9798 | 0.0735 | 0.5042 | 0.2111 | 0.0326 | 0.2338 |
| 7 | 0.0078 | 0.6003 | 0.9861 | 0.0441 | 0.5017 | 0.2068 | 0.0257 | 0.1603 |
| 8 | 0.0079 | 0.5610 | 0.9842 | 0.0571 | 0.5087 | 0.2282 | 0.0265 | 0.1463 |
| 9 | 0.0066 | 0.5536 | 0.9886 | 0.0393 | 0.4978 | 0.2165 | 0.0227 | 0.1210 |
| 10 | 0.0061 | 0.5335 | 0.9900 | 0.0380 | 0.5060 | 0.2250 | 0.0216 | 0.1143 |
| 11 | 0.0062 | 0.5101 | 0.9889 | 0.0478 | 0.5286 | 0.2445 | 0.0231 | 0.1242 |
| 12 | 0.0056 | 0.5299 | 0.9916 | 0.0393 | 0.5289 | 0.2340 | 0.0203 | 0.1037 |
| 13 | 0.0052 | 0.5264 | 0.9920 | 0.0386 | 0.5279 | 0.2343 | 0.0184 | 0.0854 |
| 14 | 0.0049 | 0.5342 | 0.9922 | 0.0393 | 0.5354 | 0.2349 | 0.0172 | 0.0738 |
| 15 | 0.0046 | 0.5361 | 0.9924 | 0.0394 | 0.5451 | 0.2390 | 0.0169 | 0.0694 |
| 16 | 0.0045 | 0.5246 | 0.9925 | 0.0401 | 0.5429 | 0.2414 | 0.0166 | 0.0663 |

读数：recon 单调下降到 0.0045（−94.7%）；`mu_diff²` 自 ep10 起稳定在
0.038–0.048 的平台（ep12-16 为 0.0393/0.0386/0.0393/0.0394/0.0401），与
seed0/seed1 的 ep12-16 行为一致；`innovation²` 0.9811→0.0663、`obs_pred`
0.0492→0.0166，obs_prior 持续吸收可预测成分；`kl_raw` 有界（0.51–0.54），
`clamped_ratio` ≈0.992，无发散、无 NaN/Inf。**seed2 训练同样没有失败，问题
仍然只在 AP 与机制读数之间的对应关系。**

### 65.9 MAI seed2 ep16 机制 probe 与三 seed 汇总

三窗口反事实 `shuffle_z`/`zero_z`/`prior_only` head ratio（4 样本均值，工具
`tools/diagnose_z_utilization.py`，hook `posterior_mu`）：

| 窗口 | shuffle_z head | zero_z head | prior_only head | shuffle/zero |
|---|---:|---:|---:|---:|
| `[3..6]` | 5.9181% | 50.3128% | 5.1851% | 0.118 |
| `[100..103]` | 6.7425% | 47.4268% | 4.9359% | 0.142 |
| `[1800..1803]` | 4.7482% | 45.8936% | 5.3559% | 0.103 |

逐样本 shuffle_z head：近距 6.0944/5.7857/6.1060/5.6862%；
中距 7.8366/5.7407/7.6582/5.7346%；远距 7.0167/3.6059/3.5712/4.7991%。
**seed2 是三个 seed 中唯一近/中距窗口都稳定超过 5% 的 run**（seed0 为
3.87%/4.18%/5.41%，seed1 为 4.80%/5.15%/5.83%），远距仍有一个 3.57% 的低样本，
跨样本离散与 seed0 期间观察到的形态一致。

`stride=700` source probe（3 样本，indices `[3,703,1403]`）：

| 张量 | sample-specific 能量占比 |
|---|---:|
| `feat_pooled_raw` | 29.0168% |
| `e_t_pooled_raw` | 49.6140% |
| `h_t` | 16.5273% |
| `mu_p` | 13.6666% |
| `mu_q` | 17.8760% |
| `correction` | 37.4981% |
| `innovation` | 61.3839% |
| `obs_prior_e_hat` | 48.2999% |

RMS 与依赖方向：

| 量 | 值 |
|---|---:|
| `h_t` RMS | 0.1296 |
| `mu_p` RMS | 0.4263 |
| `mu_q` RMS | 0.5381 |
| `correction` RMS | 0.1907 |
| `innovation` RMS | 0.1724 |
| `mu_q_removal_of_e_rel_l2` | 0.2266 |
| `mu_q_removal_of_h_rel_l2` | 0.2762 |
| **h/e 依赖比（`h_t` / `e_t` specific）** | **0.3331** |
| **e/h 依赖比（`removal_h` / `removal_e`）** | **1.219** |

读数：

1. **seed2 的 innovation 仍比上游更纯**：`innovation` specific 61.38% >
   `e_t` 49.61% > `obs_prior_e_hat` 48.30%，与 seed0/seed1 的形态一致，
   说明 `obs_prior` 减掉的主要是公共分量，这个设计目标在所有 seed 都成立。
2. **`correction` specific 37.50% 仍明显高于低维路线（<15%）**，但三 seed
   都停在 37–39% 区间、没有继续向 innovation 的 61% 收敛，说明后验头对
   innovation 的利用在三 seed 上是一致的、也是有限的。
3. **两个方向口径在 seed2 上一致指向「更靠观测」**：`h_t`/`e_t` specific
   比值为 **0.3331**（<1），removal 口径的 e/h 依赖比 `removal_h / removal_e`
   为 **1.2188**（>1，`mu_q` 对 e 的移除更敏感）。三 seed 均是同向的：
   seed0 `0.2428 / 1.1236`、seed1 `0.2952 / 1.2377`、seed2 `0.3331 / 1.2188`。
   **注意 64.6 节旧表把 `removal_h / removal_e` 这一项写成「h/e 依赖比」，
   命名有误导；标准主线该值为 2.338，MAI 三 seed 都在 1.12–1.24，方向结论
   不受影响。**
4. **样本量口径**：该探针为 `limit=4 / stride=700` 的样本读数，适合追踪方向，
   不能当作全验证集统计量。

**seed2 与 seed0/seed1 的机制横向对照**：

| 指标 | 标准 RSSM ep12 | MAI seed0 ep16 | MAI seed1 ep16 | MAI seed2 ep16 |
|---|---:|---:|---:|---:|
| `shuffle_z` head 近/中/远 | 2.71/4.71/4.14% | 3.87/4.18/5.41% | 4.80/5.15/5.83% | 5.92/6.74/4.75% |
| `zero_z` head 近/中/远 | 38.02%（旧口径） | —（seed0 未跑补测） | 41.73/50.04/55.52% | 50.31/47.43/45.89% |
| `prior_only` head 近/中/远 | 5.87%（旧口径） | —（seed0 未跑补测） | 5.16/5.94/8.70% | 5.19/4.94/5.36% |
| `correction` specific | 41.66% | 38.01% | 38.49% | 37.50% |
| `innovation` specific | — | 67.03% | 61.94% | 61.38% |
| `removal_h/removal_e`（越小越靠观测） | 2.338 | 1.1236 | 1.2377 | 1.2188 |
| ep12-16 配对 ΔAP vs 同 seed 基线 | — | +0.8434 | −2.5063 | −0.0985 |

**三 seed MAI 汇总（ep12-16 五轮等权均值，n=5/seed）**：

| seed | MAI 区间均值 | 同 seed 基线 | 配对 ΔAP | 全轮峰值 |
|---:|---:|---:|---:|---|
| 0 | 40.1242 ± 0.8614 | 39.2809 | **+0.8434** | 41.4160 @ ep16 |
| 1 | 39.4122 ± 0.8133 | 41.9185 | **−2.5063** | 42.7727 @ ep6 |
| 2 | 40.0222 ± 0.8788 | 40.1207 | **−0.0985** | 41.3163 @ ep12 |
| **三 seed 平均** | **39.8529 ± 0.4178**（seed 间） | **40.4400 ± 1.3475** | **−0.5872 ± 1.4105** | 41.8342（均值） |

复现命令：`python3 me_rssm/sanity/mai_threeseed_verdict.py`（只读，重新从三对
run 日志推导，输出与上表逐位一致）。

**三 seed 机制随机化实验的最终判读**：

1. **机制侧成立**：三 seed 都能测到 z 的样本判别效应（`shuffle_z` head
   4.75%–6.74%），`innovation` specific 三 seed 都高于 `e_t`/`obs_prior`，
   `correction` specific 三 seed 都停在 37–39%。**「innovation posterior 能把
   状态层的信息来源从先验侧推向观测侧」这件事本身是可复现的。**
2. **AP 侧不成立**：配对 ΔAP 为 `+0.84 / −2.51 / −0.10`，mean −0.59，
   跨 seed std 1.41 —— 与三 seed 基线自身 std 1.35 同量级；三 seed 平均
   `39.8529 < 40.4400`。**MAI 相对同日程标准主线没有可证实的检测收益。**
3. **机制与 AP 解耦**：seed1 的机制读数比 seed0 更好（近距 shuffle 4.80% vs
   3.87%，correction specific 38.49% vs 38.01%），AP 却低 3.35 点；seed2
   机制读数最强，AP 仅持平。**不能用机制读数替代 AP 证据，也不能用单 seed
   的 AP 提升反推机制有效。**
4. **类别增益同样不可复现**：seed0 Car +7.09 / Truck +1.12 / Ped −3.40 /
   Cyc −1.45；seed2 Car +1.74 / Truck −1.24 / Ped +0.57 / Cyc −1.47；
   seed1 Truck −2.11（负）。只有 **Cyc 是唯一三 seed 同号的缺口**
   （seed0 −1.4454、seed1 −5.8917、seed2 −1.4660，均值约 −2.93）。
5. **论文定位**：innovation posterior 保留为**机制级组件**（提供 predict-correct
   闭环、obs_prior/innovation 分解、可测的观测侧依赖），不作为 AP 主张；
   AP 赌注转到把该机制与运动条件化、可靠性门控读出结合的合并模型上，
   见 `docs/rssm_review_20261002.md` 与 `me_rssm/docs/12_mci_rssm.md`。

---

## 66. 完整 18e `MotionAlignedRSSMFusion` seed0 自动接力（2026-10-02 ~ 2026-10-03，已完成）

本节记录 18e 主模型对照的完整运行。它不沿用 MAI，也不把历史 24e 日程在第 18
轮截断；目标是用完整 18-epoch cosine/KL horizon 训练标准 `MotionAlignedRSSMFusion`
seed0，再看完整 ep18 与历史 24e 日程 ep18 的区别。

### 66.1 唯一变量与配置

| 项 | 值 |
|---|---|
| 融合模块 | `MotionAlignedRSSMFusion`（标准主线，不是 MAI） |
| 配置 | `configs/r4det/TJ4D-R4Det_fgfull_N4_2x4_18e_pretrained_v2_head_no2d_igdr.py` |
| 继承 | `..._24e_pretrained_v2_head_no2d_igdr.py` |
| 训练轮数 | `runner.max_epochs=18` |
| LR | 完整 18e cosine，峰值仍 `1.5e-4`，`min_lr_ratio=1e-5` |
| KL warmup | `0 -> 1.0`，`start_epoch=0`、`end_epoch=8` |
| seed / GPU | seed 0；GPU 5/6/7；3 卡 DDP；`--deterministic` |
| 数据与优化 | 训练/验证 pipeline、优化器、梯度累积、`val_bs=4`、`workers_per_gpu=2` 均与 24e 父配置一致 |
| cadence | `evaluation.interval=1`、`checkpoint_config.interval=2` |

**日程口径说明**：这是独立完整的 18e schedule（LR 在 ep18 到达最低点），
**不是 24e cosine 在第 18 轮被提前截断**。因此它与历史 24e 基线的 ep18 单点
并不是同一 LR 日程，后续比较时必须并列写出这一点，不能把差值与“相同日程下
18e vs 24e”混为一谈。

### 66.2 运行事实与自动接力

- 目标 work_dir：`/data/lurui/work_dirs/fgfull_N4_no2d_igdr_2x4_18e_seed0`；
  日志 `/tmp/fgfull18_seed0_ep18.log`；tmux `fgfull18_seed0_ep18`。
- 上游为 MAI seed2 ep16 链；其 `Epoch(val) [16]`、`epoch_16.pth`、四个 probe
  JSON 全部就绪，seed2 进程和 tmux 完全退出，GPU 5/6/7 空闲后，才允许启动 18e。
- watcher：`me_rssm/sanity/watch_after_mai_seed2_run_fgfull18_seed0.sh`。生产模式为
  `RUN_WATCH=1`；默认 dry-run。它同时检查真实 `torch.load` 可读的 checkpoint、
  四个非空 JSON、进程/会话双重退出和 GPU idle，避免再次发生在上一轮未退出时抢卡。
- 2026-10-02 15:23:11 UTC 正式 watcher 启动并进入等待；19:30:44 UTC 观察到
  `epoch_16.pth`（694 MB，`torch.load` 可读）、`Epoch(val) [16]` 行、四个非空
  probe JSON、seed2 进程与 tmux 均已退出、GPU 5/6/7 空闲，判定交接成立，
  因 work_dir 无 checkpoint 而从零发射。2026-10-03 13:41:54 UTC watcher 记录
  `attempt 1 reached ep18 val + checkpoint`、`complete: ep18 run finished, GPUs idle`，
  marker `/tmp/fgfull18_seed0_ep18_done`，rc=0 退出。
- **本次接力没有重演首夜失败**：ep18 不是靠 SIGINT 闸门停训，而是走完整 18 epoch
  自然结束，因此不存在 elastic shutdown 长尾问题；从发射到 ep18 val 落盘只经过
  一次发射尝试（`attempt 1`），无崩溃恢复记录。

### 66.3 三口径摘要

**主窗口 ep14-18**（完整 18e 不存在 24e 的标准 ep12-16，按预先写明的最终五轮，
n=5 等权）：

| 口径 | 数值 |
|---|---|
| 区间均值（ep14-18，n=5） | **39.7896 ± 0.4574**（样本 std） |
| 全轮峰值（所有 val epoch 的最高单点） | **40.3094 @ ep16** |
| best saved（实际落盘权重中最高的 val） | **40.3094 @ ep16** |
| last val | **39.6617 @ ep18** |
| 补充：ep12-16（n=5，便于与 24e 主线对齐） | 39.5086 ± 0.7197 |
| 补充：ep1-18 全轮均值（n=18） | 36.7615 |

落盘集合：`epoch_2/4/6/8/10/12/14/16/18.pth`，`latest.pth -> epoch_18.pth`。
峰值落在 ep16（有权重），因此本轮全轮峰值 = best saved，与 24e 主线常见的
「峰值无权重」情形不同。

**完整逐 epoch Overall 3D moderate 与 Overall BEV moderate**（n=18）：

| ep | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3D mod | 19.7732 | 27.8120 | 32.3098 | 33.2619 | **40.2634** | 37.3710 | 39.7421 | 37.5591 | 39.1541 | 39.3679 | 37.6791 | 38.4577 | 40.0082 | 39.5345 | 39.2333 | **40.3094** | 40.2090 | 39.6617 |
| BEV mod | 26.3640 | 35.9456 | 38.7638 | 42.4787 | 48.1421 | 45.1362 | 47.7397 | 45.3784 | 47.7609 | 46.5870 | 47.0705 | 46.9575 | 48.2803 | 46.9224 | 47.3557 | 48.1948 | 47.3049 | 47.4718 |

**ep14-18 逐类别均值（n=5）**：

| 类别 | 均值 | std |
|---|---:|---:|
| Car strict | 51.6028 | 1.0703 |
| Truck strict | 33.8757 | 0.9338 |
| Pedestrian loose | 27.0993 | 1.4536 |
| Cyclist loose | 46.5805 | 1.0519 |

辅助项（ep14-18，n=5）：Car loose 74.5108、Truck loose 53.5017、
Ped strict 0.1970、Cyc strict 22.4491、Overall BEV moderate 47.4499。

### 66.4 完成状态

| 项目 | 状态 |
|---|---|
| 18e config / schedule audit | 已通过（`me_rssm/sanity/test_fgfull18_schedule.py`） |
| 隔离彩排 | 已通过（旧 marker 无证据拒绝、有证据接管、缺 probe 阻断、发射失败重试、ep18 完成与 GPU 释放） |
| 正式 watcher | 2026-10-02 15:23:11 UTC 启动；19:30:44 UTC 确认 MAI seed2 完全退出且 GPU 空闲后从零发射；2026-10-03 13:41:54 UTC `complete: ep18 run finished, GPUs idle`，marker `/tmp/fgfull18_seed0_ep18_done`，watcher rc=0 |
| 18e 训练 / val | **已完成**：ep1-18 连续训练无重启、无 OOM、无 NaN；`epoch_18.pth` 2026-10-03 13:30:55 UTC 落盘，ep18 val 13:41:43 UTC 写出，全程约 18h11m 墙钟 |
| 运行资源 | GPU 5/6/7，3 卡 DDP，`--deterministic`，显存 16971 MiB（≈16.6 GiB），steady-state ≈1.53–1.60 s/iter |
| 机制 probe | **未运行**：本 run 没有 `z_util`/`source` 探针，因此本节不给出任何 z 使用程度结论；如需机制对照须另跑 `epoch_18.pth` |

**18e 训练诊断（每 epoch 全部训练日志均值）**：

| ep | recon | kl_raw | clamped | mu_diff² | prior_std | posterior_std |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.0825 | 23.2769 | 0.5336 | 1.8410 | 0.2016 | 0.1364 |
| 2 | 0.0117 | 1.0422 | 0.9607 | 0.2934 | 0.3945 | 0.1513 |
| 3 | 0.0079 | 0.6888 | 0.9859 | 0.1624 | 0.3567 | 0.1355 |
| 4 | 0.0066 | 0.6776 | 0.9905 | 0.0896 | 0.3543 | 0.1283 |
| 5 | 0.0061 | 0.6785 | 0.9920 | 0.0753 | 0.3597 | 0.1288 |
| 6 | 0.0059 | 0.6566 | 0.9926 | 0.0643 | 0.3660 | 0.1342 |
| 7 | 0.0051 | 0.6492 | 0.9949 | 0.0393 | 0.3625 | 0.1312 |
| 8 | 0.0047 | 0.6356 | 0.9963 | 0.0282 | 0.3441 | 0.1249 |
| 9 | 0.0043 | 0.6283 | 0.9972 | 0.0159 | 0.3430 | 0.1244 |
| 10 | 0.0042 | 0.6299 | 0.9971 | 0.0238 | 0.3418 | 0.1247 |
| 11 | 0.0037 | 0.6290 | 0.9975 | 0.0135 | 0.3317 | 0.1198 |
| 12 | 0.0033 | 0.6190 | 0.9981 | 0.0102 | 0.3243 | 0.1181 |
| 13 | 0.0033 | 0.6178 | 0.9978 | 0.0126 | 0.3263 | 0.1195 |
| 14 | 0.0030 | 0.6186 | 0.9982 | 0.0082 | 0.3244 | 0.1180 |
| 15 | 0.0029 | 0.6091 | 0.9983 | 0.0076 | 0.3161 | 0.1161 |
| 16 | 0.0028 | 0.6145 | 0.9982 | 0.0081 | 0.3197 | 0.1169 |
| 17 | 0.0027 | 0.6114 | 0.9984 | 0.0073 | 0.3196 | 0.1173 |
| 18 | 0.0027 | 0.6138 | 0.9983 | 0.0073 | 0.3187 | 0.1166 |

读数：recon 从 0.0825 降到 0.0027（−96.7%），KL 有界（ep18 `kl_raw=0.6138`），
无发散、无 NaN/Inf；`clamped_ratio` 升到 0.9983、`mu_diff²` 从 1.8410 单调压到
0.0073，**标准主线在完整 18e 日程下仍然存在「KL 被 free-bits 截断 + correction
幅度收缩」的老形态**——这与 18e 的 AP 结果无关（AP 反而略高于 24e 同期），再次
说明 free-bits/KL 不是这两个 run 的 AP 瓶颈。

### 66.5 完整 18e vs 历史 24e 日程（同 seed0，非因果对照）

历史参照是同 seed0 的 24e 主line `fgfull_N4_no2d_igdr_2x4_24e_seed0`
（23 条 val，跨度 ep1-23）：

| 口径 | 18e（本 run） | 历史 24e seed0 | Δ |
|---|---:|---:|---:|
| ep12-16（n=5，两 run 共同窗口） | 39.5086 ± 0.7197 | 39.2809 ± 0.9758 | **+0.2277** |
| ep14-18（n=5，18e 主窗口 / 24e 同窗口） | 39.7896 ± 0.4574 | 39.2877 | **+0.5019** |
| ep18 单点 | 39.6617 | 39.0322 | **+0.6295** |
| 全轮峰值 | 40.3094 @ ep16 | 40.3853 @ ep14 | −0.0759 |
| ep16 单点 | 40.3094 | 39.9949 | +0.3145 |
| ep1-18 全轮均值（n=18） | 36.7615 | 35.5906 | +1.1709 |

**这三项差值不能读成「18e 日程优于 24e」的因果结论**，原因是唯一变量不止一个：

1. 18e 的 LR 在 ep18 到达该 schedule 的最低点，24e 的同点 LR 仍有约 `2.9e-5`；
2. KL warmup 从 ep0-10 缩到 ep0-8，KL 在 18e 中提前两轮满权重；
3. 24e 的 ep14 峰值 `40.3853` 高于 18e 峰值 `40.3094`，说明 18e 的收益在
   平台区（ep14-18）而非上限；
4. 18e 只有 seed0，不能用单 seed 差值对抗 24e 三 seed 的 1.35 点 seed 间 std。

**可读出的结论只有两条**：(a) 在**独立完整 18e horizon** 下，标准主线能达到
`39.7896`（ep14-18 均值）与 `40.3094 @ ep16` 峰值，与 24e 平台同量级，并不存在
「18 轮训练不充分导致明显掉点」；(b) 24e 日程在 ep18 之后仍会继续下探
（seed0 后续 ep19-23 为 39.5607/38.3599/38.7773/38.7110/38.4766），因此把
「24e 截断到 18」与「完整 18e schedule」并列比较时必须显式写明日程差异，
本 run 的意义是给出后者的干净上界，而不是替代 24e 基线。

**下一步**：如需判断「18e 日程在等预算下是否更优」，唯一干净的实验是把 24e
基线也在 ep18 处用 18e cosine 重跑（已由本 run 提供），并补 seed1/seed2；
在当前三 seed 只有 24e 日程、18e 只有 seed0 的情况下，不进入论文的性能对照表。

### 66.6 复现命令

```bash
cd /home/lurui/workspace/R4Det && source .envrc
CFG=configs/r4det/TJ4D-R4Det_fgfull_N4_2x4_18e_pretrained_v2_head_no2d_igdr.py
WD=/data/lurui/work_dirs/fgfull_N4_no2d_igdr_2x4_18e_seed0

# 自动接力（生产模式；默认 dry-run，需要 RUN_WATCH=1）
RUN_WATCH=1 bash me_rssm/sanity/watch_after_mai_seed2_run_fgfull18_seed0.sh

# 手工从零启动（等价于 watcher 发射的命令）
CUDA_VISIBLE_DEVICES=5,6,7 SEED=0 bash tools/dist_train.sh \
  $CFG 3 --seed 0 --deterministic --work-dir $WD > /tmp/fgfull18_seed0_ep18.log 2>&1

# 三口径 / 逐类别 / 窗口均值
python3 tools/summarize_run.py $WD --window 14-18
python3 tools/summarize_run.py $WD --window 12-16
```

---

## 67. 近期三批训练跨 run 汇总（MAI 三 seed + 完整 18e，2026-09-30 ~ 2026-10-03）

本节把 2026-09-30 之后收尾的三批实验放进一张表，作为后续路线选择的统一入口。
所有数字都取自各自 run 的 `*.log.json`，口径按文档顶部的「统一记录口径」。

### 67.1 总表

**同 seed 配对对照（ep12-16 五轮等权均值，n=5）**：

| run | seed | 融合模块 | 区间均值 | 全轮峰值 | best saved | 同 seed 基线 | 配对 ΔAP |
|---|---:|---|---:|---|---|---:|---:|
| `..._mai_2x4_24e_seed0` | 0 | `MotionAlignedInnovationRSSMFusion` | 40.1242 ± 0.8614 | 41.4160 @ ep16 | 41.4160 @ ep16 | 39.2809 | **+0.8434** |
| `..._mai_2x4_24e_seed1` | 1 | 同上 | 39.4122 ± 0.8133 | 42.7727 @ ep6 | 42.7727 @ ep6 | 41.9185 | **−2.5063** |
| `..._mai_2x4_24e_seed2` | 2 | 同上 | 40.0222 ± 0.8788 | 41.3163 @ ep12 | 41.3163 @ ep12 | 40.1207 | **−0.0985** |
| `..._2x4_24e_seed0` | 0 | `MotionAlignedRSSMFusion`（标准主线） | 39.2809 ± 0.9758 | 40.3853 @ ep14 | 40.3853 @ ep14 | — | — |
| `..._2x4_24e_seed1` | 1 | 同上 | 41.9185 ± 0.8724 | 43.1449 @ ep15（**无权重**） | 42.2904 @ ep16 | — | — |
| `..._2x4_24e_seed2` | 2 | 同上 | 40.1207 ± 1.1697 | 41.9586 @ ep14 | 41.9586 @ ep14 | — | — |
| `..._2x4_18e_seed0` | 0 | `MotionAlignedRSSMFusion`（完整 18e horizon） | 39.7896 ± 0.4574（**ep14-18**） | 40.3094 @ ep16 | 40.3094 @ ep16 | — | — |

**聚合口径**：

| 组 | 区间均值（seed 间） | 配对 ΔAP | 全轮峰值（seed 间均值） |
|---|---:|---:|---:|
| `no2d_igdr` 三 seed 基线 | **40.4400 ± 1.3475** | — | **41.8296 ± 1.3843** |
| MAI 三 seed | 39.8529 ± 0.4178 | **−0.5872 ± 1.4105** | 41.8342 |
| 完整 18e seed0 | 39.7896（ep14-18，单 seed） | — | 40.3094 @ ep16 |

### 67.2 三条必须同时写出的结论

1. **MAI 的机制效果可复现，AP 效果不可复现。** 三 seed 的 `shuffle_z` head 都
   在 4.75%–6.74%，`innovation` specific 都高于 `e_t`/`obs_prior`，
   `correction` specific 都稳定在 37–39%，h 侧依赖都低于标准主线的 2.338；
   但配对 ΔAP 为 `+0.84 / −2.51 / −0.10`，跨 seed std 1.41 与基线自身 1.35
   同量级。**因此 MAI 只能作为机制级组件写进 Method/Analysis，不能在论文里
   声称相对标准主线的 AP 提升。** 完整拆解见 `docs/rssm_review_20261002.md`。
2. **18e 不是 24e 的截断，两者不是同一日程。** 18e 的 LR 在 ep18 到达该
   schedule 的最低点、KL warmup 为 ep0-8，24e 的同点 LR 仍约 `2.9e-5`、
   warmup 到 ep10。18e `ep14-18 = 39.7896` 说明完整 18 轮足以达到 24e 平台
   量级（同 seed0 同窗口 39.2877，但差值是日程混杂的结果，不是因果收益），
   而 24e 的峰值 `40.3853` 仍高于 18e 的 `40.3094`。**两者的差值不能写进
   性能排名，只能作为日程敏感性说明。** 详见 66.5。
3. **类别级缺口以 Cyc 最稳定。** 三 seed MAI 对同 seed 基线的 Cyc loose 差值
   同号：seed0 `−1.4454`、seed1 `−5.8917`、seed2 `−1.4660`（均值约 `−2.93`）；
   而 Car/Truck/Ped 的符号在 seed 间翻转（seed0 Car +7.09，seed2 Car +1.74；
   seed0 Truck +1.12，seed2 Truck −1.24；seed0 Ped −3.40，seed2 Ped +0.57）。
   **Cyc 是下一步几何/朝向监督最有针对性的目标，且它是唯一可以跨 seed 复现
   的负向信号。**

### 67.3 对后续路线的影响

- **MCI（Measurement-Conditioned Innovation RSSM）成为 AP 主线**，设计上把
  MAI 的 innovation 后验（机制已验证）与 ME-RSSM 的运动条件化、可靠性门控
  读出合并；实现、增广配置与预登记门控见 `me_rssm/docs/12_mci_rssm.md`，
  当前**尚未训练**，不得引用本节任何数字预测其表现。
- **MAI 保留为 MCI 的单变量消融之一**（`abl_no_innovation`、
  `abl_no_innov_in_gain`），用同一三 seed 配对协议评估，避免再次出现
  「机制好、AP 不涨」时无法定位是哪个组件失效。
- **18e 日程不进性能表**；只有在判断「短 horizon 是否影响 MCI 机制发育」时，
  才用本 run 作为同日程的 seed0 参照，并显式标注单 seed。
- **所有后续正式 run 继续按 ep12-16 五轮等权均值为横向口径**；18e 之类不同
  长度的 run 必须写明实际窗口跨度与样本数，并列写全轮峰值（值 + epoch）
  与 best saved，禁止用单个 `BEST` 混淆三者。

## 68. MCI-RSSM（测量条件化 Innovation RSSM）：实现、验证、450 iter 机制冒烟与 24e ep6 门控启动（2026-10-03）

### 68.0 摘要

- **MCI 已实现并通过全部静态验证**：37 条单测（含 E1/E2/E3 三种逐位等价、
  checkpoint/RNG 契约、门控语义）+ 配置构建审计（9 配置加载、单变量隔离、
  预训练键集、主线与 7 消融的 DDP 未用参数审计）全绿；temporal 参数
  **+0.416M**（与 ME-RSSM 设计估计 +0.42M 一致）。本节之前文档中「MCI 尚未
  训练」的状态自此更新。
- **450 iter 冒烟（3 卡 DDP，5/6/7）**：无 NaN/发散；`stat_obs_pred`
  0.1126 → 0.0329（−71%，obs_prior 在学）；`stat_innovation_sq`
  0.91 → 0.72 有界不塌缩；recon 0.953 → 0.149。
- **身份信号（决定性机制指标，通过且大幅超线）**：干预对象为**最终 z_t**
  （门控之后）的门控版反事实探针给出三窗 `shuffle_z` head
  **3.67% / 3.35% / 3.57%**，为 MAI 同期 450 iter（0.369–0.580%）的
  **6–10 倍**；且 **shuffle/zero = 1.06–1.26** —— 本项目全部历史变体
  （标准主线、低维 A/B/C、MAI）该比值都在 0.03–0.13、即 z 的因果作用以
  公共偏置为主，MCI 是第一个 shuffle ≥ zero 的变体，说明 head 对 z 的
  响应第一次以样本内容为主（对应 review P1 的修复方向）。
- **预登记门动态范围线未全过**：G2（时间散布）/G3（均值区间）通过，
  G1（空间方差 >0.005）未过——correct 门 0.0028、readout 门 0.0034。
  归因见 68.4：门 conv 权重 RMS 仅 0.0033–0.0040（典型 Xavier ~0.08 的
  ~4%）且方差平台化与冒烟日程把 LR 退火到 4.7e-6 的时间点重合，判读为
  「机制存活但发育不足 + 冒烟日程混淆」，**不据此止损**；GO/NO-GO 交给
  预登记的 ep6 AP+机制联合门控（68.6）。
- **R6 ep6 门控已完成（2026-10-03 21:57 UTC 自动停训+探针）→ 判定 GO**：
  ep6 Overall 3D moderate **40.6612** vs no2d_igdr seed0 同 epoch 34.0243，
  **Δ = +6.6369**（预登记 ≤1 即续训）；逐类 Car strict +11.19 / Truck
  strict +11.26 / Cyc loose +7.01，Ped loose **−2.92** 触发预登记黄旗
  （记录为负向信号，非止损项）。机制：三窗 shuffle_z 1.40–2.56%（继续
  门槛 0.2% 大幅通过）、h/e 依赖比 **0.569**（观测主导）、innovation
  specific 63.9%；门动态范围 G1 复测通过（空间方差 0.0213 > 0.005，确认
  冒烟期 G1 失败为日程混淆）。已按预登记规则**续训至 ep12**（68.8）。

### 68.1 设计与实现（全部在 me_rssm/ 内，原项目零修改）

- 新类 `MeasurementConditionedInnovationRSSMFusion`（`me_rssm/mci_rssm.py`），
  继承 `MotionAlignedInnovationRSSMFusion`，合并三部分：
  ① MAI 的 innovation posterior（`e_hat=obs_prior(h_t)`、
  `innovation=e_obs−sg(e_hat)`、`mu_q=mu_p+posterior_mu([h,innovation])`）；
  ② ME-RSSM 的 Doppler 三机制（C1 运动条件化对齐偏移、C3 动态证据调制
  GRU 更新门 `u'=u+tanh(β)du(1−u)`、C2 模态观测 `e_obs=e_t+obs_proj(...)`
  与置信 c_cam/c_mot）；
  ③ **门控 correct 与门控 readout（P1 修复）**：
  `z_t = z_sel + dev_c·(z_sel−mu_p)`、`output = feat + res + dev_r·res`，
  `dev = tanh(conv([c_cam, c_mot, |innovation|↓]))` 零初始化。
- 三个相对 ME-RSSM 原设计的关键修订（`me_rssm/docs/12_mci_rssm.md`）：
  ① **偏差形式门控**：ME 的 `σ(conv+4)≈0.982` 起步不逐位；`1+tanh(·)` 零
  初始化使初始行为与 MAI **逐位相同**（`mu_p+1.0*(z_sel−mu_p)` 有浮点重构
  误差，偏差形式绕开）；② **门控输入加入 |innovation|**（06-P2 预登记），
  经零初始化 1×1 conv 归约且 **detach**（门观察分歧度但不得成为压缩
  innovation 的梯度捷径）；③ **条件构造**：开关关闭的模块根本不构造，
  主线与全部消融在 DDP 下零未用参数（构造期 ValueError 校验非法组合）。
- 7 个单变量消融配置 + stateinit 单变量 + 450 iter 冒烟配置全部落地；
  `abl_no_modality_obs` 与 `abl_no_innovation` 存在**耦合开关组**（bus 组、
  innovation 组），单变量隔离按组校验。

### 68.2 静态验证（R1+R2，2026-10-03 通过）

| 检查 | 结果 |
|---|---|
| R1 单测（`test_mci_rssm.py`） | **37/37 通过**（E1 空 bus、E2 开关全关、E3 全开新初始化三种逐位等价；MAI ckpt 键集/RNG 流；门梯度可达；innovation 幅值 detach；动态门概率不变式；`_z_intervention`/`_last_h_t` 挂点） |
| R2 构建审计（`test_mci_config_build.py`） | M1 9 配置加载 ✓；M2 全模型构建 + 基链口径一致（`shared_stem=False`）✓；M3 消融单变量（组）隔离 ✓；M4 参数 **temporal +0.416M**（总量 58.35M vs MAI 57.93M）✓；M5 预训练键集除 43 个新 `temporal_fusion.*` 外与 MAI 完全一致（打包 ckpt 的 head 形状漂移是基链属性，MAI/MCI 相同）✓；M6 **主线 + 7 消融零未用参数** ✓ |
| 运行期 DDP 佐证 | 冒烟与 24e 启动日志均出现 "did not find any unused parameters"（链上 `find_unused_parameters=True`） |

过程中修复的真问题：MCI/冒烟配置 `_base_` 相对路径错误（会在任何训练启动
时立刻失败）；耦合开关（`no_modality_obs` 须连带关 C1/C3、`no_innovation`
须连带关 `innov_in_gain`）；打包预训练 ckpt 的 head 形状漂移（56/16 vs
84/24，键集对比规避）。

### 68.3 R3：450 iter 机制冒烟（work_dir `/data/lurui/work_dirs/mci_smoke_450iter`）

运行事实：GPU 5/6/7、3 卡 DDP、seed 0、`--deterministic`、IterBasedRunner
450 iter、配置 `me_rssm/sanity/mci_smoke_450iter.py`；显存 19.5 GiB/卡
（稳态）、1.61–1.71 s/iter；`iter_450.pth` 落盘；EXIT=0，全程无 NaN/Inf。
日志与全部探针输出均在 work_dir 内（`stdout.log`、`20261003_145320.log
(.json)`、`probe_gated_*.log`、`z_util_gated_*.json`、
`post_source_gated_stride700.json`）。

逐 50 iter 门控与观测曲线（训练日志 9 个记录点）：

| iter | g_mean(correct) | g_std | r_mean(readout) | r_std | obs_pred | innov² | recon |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 0.9778 | 0.00091 | 0.9769 | 0.00095 | 0.1126 | 0.9149 | 0.9531 |
| 150 | 0.9433 | 0.00234 | 0.9333 | 0.00276 | 0.0513 | 0.7153 | — |
| 250 | 0.9348 | 0.00269 | 0.9209 | 0.00327 | 0.0426 | 0.7109 | — |
| 350 | 0.9327 | 0.00277 | 0.9180 | 0.00338 | 0.0334 | 0.7261 | — |
| 450 | 0.9326 | 0.00278 | 0.9180 | 0.00338 | 0.0329 | 0.7235 | 0.1488 |

读数：obs_prior 单调学习（−71%）；innovation² 有界不塌缩；两个门的均值从
1.0 移到 0.92–0.93（filter 学会平均上略微收缩校正）、空间方差单调上升但
在 iter≈300 后平台化——与 cosine 把 LR 退火到 4.7e-6（iter 400）的时间点
重合。

### 68.4 R4：机制门控判定（预登记四项，docs/12 §5.1）

| 预登记检查 | 结果 | 判定 |
|---|---|---|
| ① 训练健康 | 无 NaN；obs_pred −71%；innovation² 有界；recon/KL 与主线同量级 | **PASS** |
| ② correct 门动态范围 | G2 时间散布 PASS、G3 均值 0.933∈(0.5,1.5) PASS；**G1 空间方差 0.0028 < 0.005** | **G1 FAIL** |
| ③ readout 门动态范围 | 同形态（0.0034 < 0.005） | **G1 FAIL** |
| ④ 身份信号 ≥ MAI 同期 | 三窗 shuffle_z 3.35–3.67% vs MAI 450iter 0.369–0.580%（**6–10 倍**）；shuffle/zero 1.06–1.26（历史全部变体 0.03–0.13） | **PASS（大幅）** |

三窗反事实（干预对象 = 门控之后的最终 z_t，`probe_z_utilization_gated.py`）：

| 窗口 | shuffle_z head | zero_z head | shuffle/zero | prior_only |
|---|---:|---:|---:|---:|
| w3（近，idx3-6） | 3.667% | 2.920% | 1.256 | 1.260% |
| w100（中，idx100-103） | 3.345% | 3.146% | 1.063 | 1.179% |
| w1800（远，idx1800-1803） | 3.565% | 2.877% | 1.239 | 1.148% |

source 探针（`probe_posterior_source_gated.py`，跨场景 stride=700）：
`innovation` specific **37.5% > e_pooled 33.8%**（obs_prior 确实削掉公共
成分）；`correction` specific 21.1%；`h/e` removal 比 1.627（450 iter 时
posterior 仍偏 h 侧——MAI 同期 450 iter 也未反转、到 ep6 才降到 <1，属
同形态早期状态）；门 conv 权重 RMS 0.0033/0.0040。

**G1 的归因与处置（如实记录，不改线）**：G1 按预登记线未过。三条证据
支持「发育不足」而非「机制死亡」：① 门均值显著移动（G2/G3 过，梯度通）；
② 门 conv 权重已离开零点但仅达典型尺度的 ~4%；③ 方差平台化与 LR 退火到
近零重合，不是梯度消失。按 docs/12 预登记的判定结构，450 iter 冒烟的
职能是排除死机制与 NaN，GO/NO-GO 由 ep6 AP+机制联合门控承担——该门控
复测门动态范围（此时 LR 已在真实日程下运行 6 epoch）。

### 68.5 探针兼容性三修复（全部 me_rssm 侧，tools/ 零修改）

1. **zero_z 逐位断言失效**：原工具经 prior/posterior 钩子使 `mu_q` 精确
   等于期望值，对 MAI（`z_t==mu_q`）逐位成立；MCI 的门控在钩子之后重新
   混入 `−dev·mu_p`，断言拦下（干预实际 ~93% 生效）。修复：类内新增
   `_z_intervention` 挂点（在门控之后、decoder/readout/state 之前对最终
   z_t 施加干预），`me_rssm/sanity/probe_z_utilization_gated.py` 复用原
   工具全部数据管线与打分函数，仅替换干预机制；干预后 `out[4]` 逐位等于
   期望值。
2. **h_t 钩子失效**：原 source 探针在 `transition`（ConvGRUCell）上挂
   forward hook，但 MCI 的动态门实现直接调用 cell 卷积、cell forward 不
   被调用，`KeyError: 'h_t'`。修复：类内 eval 模式暂存 `_last_h_t`
   （训练置 None，零开销），`probe_posterior_source_gated.py` 改读暂存。
3. **观测重组失真**：原 source 探针重组 innovation 假设观测为
   `encoder(feat)`，MCI 实际条件于 `e_obs=e_t+obs_proj(...)`。修复：类内
   实现 `_posterior_observation(e_t)`（eval 前向暂存 detach 残差），
   探针精确重组；shuffle/zero 干预改写 `posterior_mu` 输出、与本差异无关。
   三项均有单测覆盖（37 条中新增 5 条）。

### 68.6 R6+R7：24e seed0 ep6 门控（已启动，结果待 68.7）

- **运行**：`fgfull_N4_no2d_igdr_mci_2x4_24e_seed0`，GPU 5/6/7、3 卡 DDP、
  seed 0、`--deterministic`、2026-10-03 15:34 UTC 启动；配置
  `me_rssm/configs/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_mci.py`
  （主线 MCI：全开七开关、`state_init='zero'`）；stdout 与全部日志进
  work_dir。tmux `mci_ep6` + 看门狗 `mci_ep6_watch`
  （`me_rssm/sanity/watch_mci_ep6.sh`，日志/探针输出全部进 work_dir）。
- **ep6 预登记判定规则**（沿用 MAI §62.9 口径，不因本轮结果修改）：
  - AP：ep6 Overall 3D moderate vs no2d_igdr seed0 同 epoch **34.0243**；
    差距 ≤1 → 续 ep12；1–2 → 记录后到 ep12 再判；>2 → 止损。
  - 机制：三窗 `shuffle_z` ≥0.2%（继续门槛）；`innovation specific >
    e_pooled`；`stat_gain_*/readout` 动态范围复测（G1 在真实日程下重判）；
    **Ped/Cyc 黄旗**：ep1-6 逐类别 Ped loose 相对同窗基线 −2 以上即记
    负向信号（review P9 预登记观察点）。
  - 附注：同批 450 iter 冒烟的对照基线是 MAI 450 iter
    （shuffle_z 0.369–0.580%）；ep6 的对照是 MAI ep6（0.73–1.09%）与
    标准主线 ep6（3.30%/3.36% 近/中距）。

### 68.7 ep6 门控结果（2026-10-03 21:57 UTC 看门狗自动停训+探针，判定 GO）

**逐 epoch 曲线（ep1-6，Overall 3D moderate）**：

| epoch | 1 | 2 | 3 | 4 | 5 | 6 |
|---:|---:|---:|---:|---:|---:|---:|
| MCI seed0 | 19.7734 | 29.3388 | 33.4914 | **40.4335** | 37.0786 | **40.6612** |
| no2d_igdr seed0 | 15.2640 | 26.3716 | 31.6591 | 32.8506 | 34.0827 | 34.0243 |
| MAI seed0 | — | — | — | — | — | 35.9289 |

- **AP 门（预登记：Δ vs 34.0243，≤1 续 / 1–2 记录 / >2 止损）**：ep6
  **Δ = +6.6369**，远超续训线；ep1-6 均值 33.4628 vs 基线 29.0420
  （**+4.4208**）；BEV ep6 48.7393 vs 42.4084（+6.3309）。
- **逐类 ep6（moderate，vs no2d_igdr seed0 同 epoch）**：Car strict
  54.4681 vs 43.2792（**+11.1889**）；Truck strict 33.1899 vs 21.9299
  （**+11.2600**）；Cyclist loose 50.7795 vs 43.7657（**+7.0138**）；
  Pedestrian loose 24.2073 vs 27.1223（**−2.9150，黄旗触发**——review P9
  预登记观察点：Ped 是 Doppler 证据最弱类（0.145），门控/预测通路对
  Ped 的行为进 Analysis 章节，Ped 不进止损判据）。
- **机制门（门控版探针，干预对象=门控后最终 z_t）**：三窗 `shuffle_z`
  head **2.10% / 2.56% / 1.40%**——继续门槛 0.2% 大幅通过，近/中距越过
  2% 论文线，远距 1.40% 未到；`shuffle/zero` = 0.080–0.094（zero_z
  17.45–28.48%：公共偏置成分随训练回升，与 MAI 及全部前代在 ep6 阶段的
  生命周期一致；冒烟期 1.06–1.26 的比值未保持，需在 Analysis 里按
  「早期身份信号发育快、后期公共底色回归」如实描述）。
- **source 探针**：`h/e` removal 比 **0.569**——posterior 首次在 ep6 深
  度观测主导（MAI ep6 为 0.9996，标准主线 2.338 且随训练反向恶化）；
  `innovation` specific **63.9% > e_pooled 45.5%**；`correction` specific
  44.0%；`h_t` specific 8.0%、`mu_p` 6.1%。
- **门动态范围复测（G1 在真实日程下重判）**：correct/readout 门 conv
  权重 RMS **0.0339 / 0.0397**（冒烟期 0.0033/0.0040 的 ~10 倍）；
  `stat_gain_std` 0.0213、`stat_readout_gain_std` 0.0220（> 0.005）——
  **G1 复测通过**，68.4 的「日程混淆」归因被证实。另一重要读数：
  `stat_gain_mean` 0.3758——correct 门学会**平均上强烈收缩校正**（更信
  运动传播的预测），逐像素 std 0.0213 提供差异化调制，即设计目标的
  Kalman 式行为已经出现。
- **联合判定：GO（按预登记规则续 ep12）**。保留意见照旧：单 seed、单
  epoch 对照，基线三 seed ep12-16 std 1.35；ep4 的 40.4335 与 ep6 的
  40.6612 已高于 MAI 全轮峰值均值（41.83±1.38 量级需 ep12+ 复核）。

### 68.8b ep12 门控结果（2026-10-04 04:28 UTC 看门狗自动停训+探针，判定 GO → 续 ep16）

**逐 epoch（ep7-12，Overall 3D moderate）**：ep7 **41.6747**（本次 run 当前
最高单点）、ep8 40.4005、ep9 39.7543、ep10 38.6874、ep11 40.1653、
ep12 39.7236。ep8-10 的回落与 ep11 的回升均在历史平台期逐 epoch σ≈1.5
的正常波动内。

- **AP 门（预登记：ep8-12 配对均值 ≥ 38.3906 续 / < 止损）**：
  ep8-12 = 40.4005 / 39.7543 / 38.6874 / 40.1653 / 39.7236 → 均值
  **39.7462**（**+0.8556** vs 基线 38.8906；**+0.4070** vs MAI seed0
  39.3392——MCI 首次在配对窗口上同时高于基线与 MAI 同期）→ **GO，续
  ep16**。
- **逐类 ep8-12 窗口均值（loose）**：Ped 28.096 vs 基线 29.556
  （**−1.46**，比 ep6 的 −2.92 收窄，未触及 −4 红灯升级条款）；Cyc
  49.13 vs 49.68（**−0.55**，基本持平——注意 MAI 在 Cyc 上三 seed 稳定
  −1.4~−5.9，MCI 当前为近零）。
- **机制（门控版探针）**：三窗 `shuffle_z` **4.35% / 5.14% / 2.71%**——
  **首次三窗全部越过 2% 论文线**，且比 ep6（2.10/2.56/1.40）继续增长；
  `shuffle/zero` 0.130–0.136；`h/e` removal 比 **0.605**（观测主导保持，
  MAI ep12 为 0.842）；`innovation` specific 59.9% > `e_pooled` 42.6%；
  `correction` specific 49.6%（ep6 44.0% → 继续上升）；门 conv 权重 RMS
  0.0407/0.0494（持续发育）。
- **判定：GO，按预登记规则续 ep16**（68.10）。

### 68.8c 续训 ep16（2026-10-04 04:30 UTC 启动）+ ep16 预登记规则

- **运行**：`--resume-from epoch_12.pth`，tmux `mci_ep16` + 看门狗
  `mci_ep16_watch`，stdout 进 `work_dir/stdout_ep16.log`，ep16 探针输出
  `z_util_gated_ep16_*.json`、`post_source_gated_ep16_stride700.json`。
- **ep16 预登记判定规则**（写于 ep13-16 任何 val 存在之前）：
  - AP：ep12-16 五轮配对窗口均值 vs no2d_igdr seed0 同窗 **39.2809**：
    ≥ 39.2809 → 判「MCI seed0 主线有效」，启动 seed1/seed2 复现排队
    （GPU 排期由用户决定）；< 39.2809 → 记录单 seed 未达主线，先做
    归因再定。参照线：MAI seed0 ep12-16 = 40.1242（MCI ≥ 该值即明确
    超过 MAI 主线）。
  - 机制：三窗 `shuffle_z` ≥0.2% 维持、G1（空间方差 >0.005）维持、
    h/e <2 维持；Ped 黄旗继续追踪（ep12-16 差值 ≤−4 升红灯）。
  - ep16 为 seed0 的最后一道门；其后的 multi-seed 属新决策，不在本节
    预登记范围内。

### 68.8d ep16 门控结果与 seed0 终判（2026-10-04 08:46 UTC 看门狗自动停训+探针，判定：主线有效）

**逐 epoch（ep13-16）**：ep13 **40.9337**、ep14 40.2883、ep15 39.8383、
ep16 39.7467。

- **AP 门（预登记：ep12-16 窗口 ≥ 39.2809 → 主线有效）**：五轮
  39.7236 / 40.9337 / 40.2883 / 39.8383 / 39.7467 → 均值 **40.1061**
  （**+0.8252** vs no2d_igdr seed0 39.2809；**−0.0181** vs MAI seed0
  40.1242——与 MAI 主线持平，差值在平台噪声内）→ **判定：MCI seed0
  主线有效**。
- **三口径**：区间均值（ep12-16，n=5）**40.1061**；全轮峰值
  **41.6747 @ ep7**（该 epoch 无落盘权重）；best saved（interval=2 落盘
  偶数轮中最高）**40.6612 @ ep6**。
- **逐类 ep12-16 窗口均值（vs no2d_igdr seed0 同窗）**：Car strict
  54.5319 vs 47.8399（**+6.69**）；Truck strict 32.3255 vs 29.7246
  （**+2.60**）；Cyclist loose 47.4288 vs 48.6313（**−1.20**，与 MAI 的
  −1.45 同水平）；**Pedestrian loose 26.8607 vs 30.9277（−4.07，触发
  ep12-16 预登记红灯条款）**——Ped 差值在 ep8-12 收窄到 −1.46 之后于
  ep13-16 重新走阔（ep13-16 达 −4.34），升级为红灯记录：**Ped 是 MCI
  的主要负向类别**，进 Analysis 专节（可靠性门在 Doppler 证据最弱类上
  的行为 + Ped 锚点几何瓶颈的叠加），不改 Overall 判据。
- **机制维持项（ep16 探针）**：三窗 `shuffle_z` **4.08% / 4.12% /
  2.13%**（全部 ≥2% 论文线，与 ep12 持平略降）；`h/e` removal **0.640**
  （观测主导保持——对比 MAI ep16 回升到 1.12）；`innovation` specific
  **64.3% > e_pooled 42.8%**；`correction` specific 49.2%；门 conv 权重
  RMS 0.0408/0.0503（稳定）。**机制优势跨 ep6→ep12→ep16 保持**，未出现
  MAI 式的 ep16 回落（h/e 未反转、shuffle_z 未跌破门槛）。
- **seed0 结论**：MCI 单 seed 完成 ep6/ep12/ep16 三道预登记门控全 GO，
  ep12-16 窗口同时 ≥ 基线、与 MAI 持平；机制指标全面维持。**按预登记
  规则启动 multi-seed 复现**：seed1 于 08:51 UTC 启动（tmux
  `mci_seed1` + 参数化看门狗 `watch_mci_seed_ep16.sh`，跑满至 ep16 自动
  停训+探针，输出进 seed1 work_dir）；seed2 在 seed1 判定后接力。
  判定口径：逐 seed 配对窗口（seed1 基线 41.9185 / seed2 基线 40.1207）。
- **风险提示（照搬 §64 对 MAI 的教训）**：单 seed 的 +0.83 不能当结论；
  基线三 seed 窗口 std 1.35；MCI 的 Ped 红灯与 Cyc −1.20 需要在多 seed
  下确认是否同样跨 seed 稳定。

### 68.8g 主模型判定规则（预登记，写于 seed2 ep13-16 数据存在之前，2026-10-05）

判定量：三 seed 配对窗口差 Δ = (Δ0, Δ1, Δ2)，均值 m，正项个数 k。
当前 Δ0 = +0.8252、Δ1 = −0.7967、Δ2 待定（seed2 ep12 41.2289 已落，
ep13 38.8122 在窗口内）。

- **R-strong（m ≥ +0.5 且 k ≥ 2）**：MCI 定为主模型，论文同时主张
  AP（per-seed paired Δ 口径）与机制；立即排 seed1/2 的 ep16 之上
  无额外动作，主表直接成型。
- **R-weak（0 ≤ m < +0.5）**：MCI 仍作主模型，但 AP 主张降级为
  「on par with the strong baseline while adding measurement
  conditioning」，论文重心放在机制贡献（identity signal 过 2% 线、
  h/e 持续 <1、Kalman 门行为）+ Ped/类别效应的 Analysis 专节；
  同时启动 Ped 缓解实验（modality_dropout>0 变体，单变量）作为
  camera-ready 前迭代。
- **R-neg（m < 0）**：MCI 不作主模型。两条子路：(a) 消融冒烟中若存在
  单一开关关闭后机制与 AP 同优的简化配置（如 no_dynamic_gate），以
  简化版重跑三 seed；(b) 否则论文改走机制级分析路线（预登记的负结果
  框架），主模型决策推迟。
- 附加条款（任何分支通用）：① 主表数字 = 逐 seed 配对窗口均值 +
  全轮峰值 + best saved 三列并列，禁止单点 BEST；② 主模型定稿前对
  三 seed 跑 checkpoint averaging（§45）复评一次作为稳健性附注；
  ③ Ped 红灯无论哪个分支都必须带探针裁决结果写进 Analysis。
- 本规则由三 seed 数据之外的信息（消融冒烟、空间探针）不触发改线；
  它们的职能是解释与后续迭代，不参与本次主模型判定。

### 68.8h seed2 终判与三 seed roll（2026-10-05 ~18:10 UTC ep16 自动停训+探针）→ **R-neg 落地**

- **seed2**：ep12-16 = 41.2289 / 38.8122 / 39.1291 / 39.3660 / 40.1886 →
  窗口 **39.7450**，对同 seed 基线 40.1207 **−0.3757**。全轮峰值
  **41.2289 @ ep12**。逐类（loose）：Car +1.91 / Cyc −1.39 / Ped −1.69 /
  Truck −0.91——四个类都是小幅，无 seed0 式的 Car/Truck 大正项。
- **三 seed roll（预登记判定量的最终值）**：Δ = **+0.8252 / −0.7967 /
  −0.3757**，**m = −0.1157，std 0.6872**（1 正 / 2 负）→ 按规则落
  **R-neg：MCI 现状不作主模型**。必须同时写出的三个限定：
  ① m = −0.12 位于配对噪声界（0.42–0.53）之内，即「与零不可区分」，
  不是「显著为负」；② MCI 的 roll 在每个维度优于 MAI（均值 −0.12 vs
  −0.59，std 0.69 vs 1.41，最差 seed −0.80 vs −2.51）；③ 机制指标
  三 seed 全部保持（见 68.8i）——**R-neg 否定的是「现状的 AP 主张」，
  不是机制路线**。
- **seed2 机制**：三窗 `shuffle_z` **3.81% / 2.79% / 4.49%**——至此
  **三 seed × 三窗 = 9/9 全部 ≥2% 论文线**；`h/e` = 0.668（三 seed
  0.64/0.70/0.67，观测主导无一反转）；`innovation` specific 56.9%。

### 68.8i Ped 空间探针 + 权重体检：**发现 innov 通道死锁**（重要）

- **空间探针功效不足**（6 样本 stride700，Ped GT 仅 1 个）：门控图在
  Ped/Car/Cyc/背景四处的类间差异在小数第 4 位（g_c≈0.286/0.289、
  g_r≈0.197/0.200、c_mot≈0.894 恒定），**H2/H3（门在 Ped 区主动收缩）
  未获支持也未排除**；需要 limit≥40/stride≤100 的加强版重跑（已排队，
  GPU 窗口在 pdrop10 停训后）。
- **但权重体检抓到硬发现**：`innovation_mag_conv` 权重**精确为零**
  （abs_mean=std=0.000000，16 epoch 一步未动）——**零初始化死锁**：
  innov 通道 conv 零初始化且无 bias → 输出恒 0 → 门 conv 在该输入通道
  的梯度恒 0 → 该通道切片永不生长 → 上游永收零梯度。§38.10 死分支
  教训的「零乘积」形态（我此前的「无 ReLU 不适用」判断是错的）。
  **后果：全部三 seed 的 MCI 门控实际只输入 [c_cam, c_mot]，06-P2 的
  innovation 幅值通道从未生效**——abl_no_innov_in_gain 冒烟与主线门控
  读数几乎一致（0.9308/0.9207 vs 0.9326/0.9180）即为佐证。
- 其余权重读数（seed2 ep16）：correct/readout 门 conv 活跃
  （0.032/0.041）；conf_mot 权重 0.020（运动置信图近似常数 0.894）；
  **dyn 偏置从 −2.0 漂到 +1.88**（dyn 图 ≈0.87，动态门变成全局小幅
  提速——C3 的空间分辨力未形成）；obs_proj 0.016。综合：**门控当前
  主要以「全局校正系数」形式起作用（g_mean≈0.29），逐像素可靠性
  区分尚未实现**——这正是候选改进方向，而非已兑现的能力。
- **修复（已落地，38 条单测过）**：`innovation_mag_conv` 改 Xavier
  初始化（死锁解除：门 conv 该通道切片获得非零梯度通路），门 conv
  仍零初始化 → 恒等启动契约（E1/E2/E3）不受影响；新增死锁回归测试。
  修复后的模型代际记为 **MCI-v2**，其首次训练前的热启动/等价性测试
  需重跑（旧 checkpoint 载入后 innov 通道会以旧零权重生效，行为等同
  旧版，可载入不可混淆）。

### 68.8j 七消融 450 iter 机制冒烟（2026-10-05 18:10–20:35 UTC，串行全绿完成）

| 冒烟 | g_mean(correct) | g_std | r_mean(readout) | 门控读数备注 |
|---|---:|---:|---:|---|
| no_motion_offsets | 0.9330 | 0.0028 | 0.9180 | 与主线同形态（门控不受 C1 影响） |
| no_dynamic_gate | 0.9331 | 0.0028 | 0.9181 | 同上 |
| no_reliability_gain | （correct 门不存在） | — | 0.9187 | 读出门独立存活 |
| no_gated_readout | 0.9327 | 0.0028 | （不存在） | correct 门独立存活 |
| no_innov_in_gain | 0.9308 | 0.0029 | 0.9207 | **与主线几乎一致——死锁佐证** |
| no_modality_obs | **0.9959** | **0.0000** | 0.9954 | 门输入只剩死 innov 通道（恒 0）→ 门仅 bias 可学，退化为全局标量 1−0.004 |
| no_innovation | 0.9421 | 0.0024 | 0.9322 | 标准后验下门控行为不变 |

- 全部 G1 未过线（450 iter + LR 退火混淆，68.4 已记录）；smoke 无 val，
  **不产生任何 AP 结论**——R-neg 子路 (a)（简化配置更优）目前**没有
  AP 级证据**，若要筛选需 ep6 级门控 run（~6.5h/个，待用户排期）。
- 表内两行「结构性正确」的对照（no_gated_readout 无读出门、
  no_modality_obs 门冻洁）同时验证了条件构造与开关语义。

### 68.9 Ped 修复已自动启动 + 后续决策点（2026-10-05 20:40 UTC）

- **pdrop10**（MCI + modality_dropout=0.1，seed0，24e）20:40 UTC 自动
  启动（tmux `mci_pdrop10` + 看门狗），预计 **2026-10-06 ~13:00 UTC**
  ep16 停训+探针。注意：该 run 用的是修复前代码（innov 通道死锁），
  与三 seed 家族同构，**单变量归属干净**（只有 dropout 一个差异）。
- 预登记判定（写在 config 头）：Overall 窗口 ≥39.2809 且 ≥
  40.1061−0.5；**Ped 配对差 ≥ −1.5**（从 −4.07 恢复）；机制 ≥0.2%。
- **MCI-v2**（Xavier innov 修复版）尚未训练：它使 06-P2 首次真正生效，
  是下一个单变量（vs 三 seed 家族），建议排在 pdrop10 判定之后。
- **决策点汇总（用户醒来后）**：① R-neg 已落地——接受「机制论文」
  定位或等 pdrop10/MCI-v2 翻盘后再定主模型；② pdrop10 判定后是否
  多 seed；③ MCI-v2 是否占下一槽位；④ 加强版空间探针（limit≥40）
  随 MCI-v2 或 pdrop10 停训窗口执行；⑤ 消融中是否有个别配置值得
  ep6 级 AP 筛选。

### 68.9a pdrop10 终判（2026-10-06 ~07:50 UTC ep16 自动停训+探针）：**Ped 修复失败**

- **Overall 双门槛通过**：ep12-16 = 42.6611 / 39.4738 / 39.2680 /
  40.9191 / 37.1772 → 均值 **39.8998**（≥ 基线 39.2809 ✓；≥ 主线 −0.5
  39.6061 ✓；对主线 40.1061 为 −0.21，噪声内）。全轮峰值
  **42.6611 @ ep12**（全部 MCI run 的最高单点）。
- **Ped 门槛失败**：ep12-16 Ped loose = 29.91 / 26.27 / 27.16 / 27.60 /
  26.03 → 均值 **27.394**，对基线 **−3.53**（门槛 ≥ −1.5）。轨迹
  −0.46 / −2.97 / −7.09 / −0.62 / −6.73——dropout 未兑现恢复，仅比主线
  的 −4.07 好 0.54（噪声量级）。**modality_dropout 不是 Ped 的解。**
- **机制**：三窗 shuffle_z 2.85 / 4.59 / 3.45%（≥2% 维持，12/12 窗）；
  h/e = 1.107（较主线 seed0 的 0.640 回到 h 侧——dropout 下 posterior
  更依赖 h_t，或为 run 级波动，记录待考）。
- **跨变体 Ped 汇总（H1 证据链闭合）**：MAI −3.40 / MCI −4.07 /
  pdrop10 −3.53——**凡携带 innovation 后验的变体 Ped 一致回归 ~−3.5，
  与门控形态和训练压力无关**；dropout（针对 H2/H3 的 fallback 压力）
  无效 → **H1（innovation 后验对 Ped 的噪声）成为唯一幸存主因，
  出路指向 DC-RSSM 的相机主导路由（对 Ped 按构造走相机观测）而非
  任何门控/训练压力修补**。

### 68.9b 加强版空间探针（40 样本，9 Ped / 24 Cyc / 25 Car / 3 Truck）：**H2/H3 否定，门控被证实「空间失明」**

| 类 | g_c | g_r | c_cam | c_mot | dyn |
|---|---:|---:|---:|---:|---:|
| Pedestrian | 0.2873 | 0.1989 | 0.9795 | 0.8938 | 0.1320 |
| Cyclist | 0.2924 | 0.2039 | 0.9771 | 0.8939 | 0.1324 |
| Car | 0.2918 | 0.2033 | 0.9738 | 0.8939 | 0.1320 |
| Truck | 0.2890 | 0.2005 | 0.9716 | 0.8938 | 0.1320 |
| background | 0.2878 | 0.1994 | 0.9769 | 0.8938 | 0.1323 |

- 四类 GT 中心与背景的门控取值差异 ≤0.005（c_cam 仅 0.008 微梯度且
  Ped 最高——与「相机看得见 Ped」一致但无功能后果）；**门控对所有
  像素一视同仁** → H2（门在 Ped 区收缩）/H3（动态门把 Ped 判静态）
  均被否定。
- **结构性结论**：学习版 c_mot/dyn 塌缩为空间常数（conf_mot 权重
  0.020、dyn_conv 偏置主导）——**「逐像素可靠性」在学习版里从未
  存在过**，门控的全部空间行为是全局标量。这使得 DC-RSSM 第②步
  （无参数证据场：count/SNR/v_std 标定的物理图 + 零初始化学习细化）
  从「改进」升级为「被证实的必要修复」，其 coverage 语义
  （无测量 ≠ 静态）同时是 Ped 死区的正面解法入口；H1 的出路则是
  第③步相机主导路由。
- 探针工件：`gate_spatial_ep16_x40.json`（seed0 ep16 权重）。

### 68.9c 当前状态与晨间决策清单（2026-10-06 16:00 UTC 更新）

- **GPU 空闲**（pdrop10 已按门控停训）；所有结果已落盘并入库。
- 待用户决策：① **MCI-v2**（innov 通道复活 + Xavier 修复已就绪，
  同 config 即可跑）是否占下一槽位——注意 MCI-v2 让门看到 innovation
  幅值，若 H1 成立其对 Ped 的影响方向未知，正好是预登记可检验的
  假设；② DC-RSSM 路线（ego-warp → 证据场 → 相机路由）是否启动
  （docs/13 设计文档可先写）；③ pdrop10 不进多 seed（Ped 门槛失败）；
  ④ 加强版探针已执行完毕，无需重跑。

### 68.9d MCI-v2 已启动（2026-10-07 09:25 UTC）+ DC-RSSM 设计定稿

- 用户批准执行：**MCI-v2**（innov 通道复活/Xavier 修复代际，config 与
  主线相同，work_dir `fgfull_N4_no2d_igdr_mci_v2_2x4_24e_seed0`，
  tmux `mci_v2` + 看门狗 `mci_v2_watch`）。预登记机制门：innov_mag 图
  非零且空间有结构（对照死锁代际的恒零）、shuffle_z ≥2%、h/e <2；
  AP 门：ep12-16 ≥ 39.6061。ep16 自动停训+探针预计 2026-10-08 ~01:30 UTC。
- **DC-RSSM 设计定稿**（`me_rssm/docs/13_dc_rssm.md`）：五层（解析
  ego-warp / 双时间尺度状态 / 无参数证据场 / 创新一致性记忆 / MCI-v2
  存量），含死锁规则的正式化（「零输出层不得直连零初始化层」）、
  分步门控与裁剪顺序。实现顺序：MCI-v2 判定 → DC-RSSM-1（证据场）
  → -2（ego-warp）→ -3（双速率+一致性记忆）。

### 68.9e MCI-v2 终判（2026-10-08 02:27–02:28 UTC ep16 自动停训+探针）：AP 门 FAIL，机制门 3/4

- 训练：watchdog 02:27 gate hit 自动停训，探针 02:28 完成。全轮曲线
  ep1–16 = 19.12 / 28.51 / 37.14 / 35.60 / 38.30 / 40.33 / **42.94** /
  41.21 / 39.47 / 41.30 / 35.55 / 42.40 / 39.06 / 40.48 / 41.02 / 33.93
  （全轮峰值 42.94 @ ep7，本线历史最高单点）。
- **AP 门 FAIL**：ep12-16 = **39.3786** < 39.6061（主线 −0.5，差 −0.23）。
  逐类 ep12-16（3D moderate loose）：Ped 23.10 / Cyc 48.70 / Car 74.23 /
  Truck 53.06。ep16 收官塌陷（33.93，Ped 掉到 12.53）拖垮窗口；ep12-15
  单看 40.74 可过线；pdrop10 ep16 也回落到 37.1772，但 MCI-v1 seed0
  ep16 实际为 39.7467，不能称“家族第三次塌陷”（日志复核见 §68.9f）。
  按预登记口径不做事后窗口调整，判 FAIL。
- **机制门 3/4**（docs/13 step 0 预登记）：
  - innov 通道非零 ✓：`innovation_mag_conv` ep16 权重 abs_mean 0.0751 /
    std 0.0790 / max 0.192（死锁代际三 seed 16 epoch 后精确 0.000），
    门控输入 rms 1.038；ep2 中检（0.0819）与 ep16 终检一致——Xavier
    修复在真实训练全程存活，死锁修复实证闭环（权重级+激活级）。
  - shuffle_z ≥2% ✓：head_ratio 4.86 / 4.83 / 3.97%（w3/w100/w1800 三窗全过）。
  - h/e <2 ✓：0.805（h 移除 0.270 / e 移除 0.336），观测主导，与 MCI
    家族（0.57–0.70）同域，远优于主线 2.34。
  - 「空间有结构」✗（弱）：innovation **向量**样本特异能量 59.0%（健康，
    与 MCI seed0 的 59.9% 一致），但经 |·|+conv 后的**幅值门输入**样本
    特异能量仅 **5.4%**（94.6% 批内共模）；门增益空间 std/mean ≈ 6%。
    通道活着但吃到的信号以共模为主——「逐像素可靠性未形成」从学习
    门控（c_mot/dyn 常数）推广到幅值门控，直接坐实 DC-RSSM-1
    （物理证据场）的动机：检测梯度学不出空间结构，需测量推导注入。
- P1 现状：zero_z head 19.6–21.9% vs shuffle_z 3.97–4.86%，readout 偏置
  主导未变。
- **判定**：R-neg（AP 门失败即止损：不升多 seed、不作为主模型）。正面
  资产三项：① 死锁修复实证闭环；② 幅值域共模量化（5.4%）作为
  DC-RSSM-1 证据场的 baseline 对照；③ v2 ep16 急跌入档，根因待证
  （不可推断家族固定 ep16 塌陷，见 §68.9f）。
- 探针产物（v2 work_dir）：`z_util_gated_ep16_w{3,100,1800}.json`、
  `post_source_gated_ep16_stride700.json`、`watch_seed0_ep16.log`。

### 68.9f MCI-v2 Ped 回归复核（2026-10-08；只读日志/探针，无新训练）

**范围与运行事实**：复核 §68.9e 的 seed0 run，唯一变量仍是
`innovation_mag_conv` 从零初始化改 Xavier；配置快照与运行目录见 §68.9d/e。
本任务不修改模型、不启动训练/推理、不使用 GPU。证据来自上述五个 run
目录的 `*.log.json`（每 epoch 取最后一次 val）及 v2 的
`post_source_gated_ep16_stride700.json`。v2 已验证 ep1–16，权重保存
ep2/4/6/8/10/12/14/16。下表补齐原节缺少的完整类别曲线。

**统一摘要**：Overall 区间均值（ep12–16，n=5）**39.3786**；
全轮峰值 **42.9447 @ ep7**（无该轮权重）；best saved
**42.4026 @ ep12**。Ped 3D moderate loose 区间均值
**23.0982**；全轮峰值 **29.0721 @ ep12**（有权重）。

**完整逐 epoch val**（均为 moderate；Car/Truck strict，Ped/Cyc loose；
Ped strict 单列，不混入 Overall）：

| ep | Overall 3D | Overall BEV | Car 3D | Truck 3D | Ped 3D | Cyc 3D | Ped BEV | Ped 2D | Ped 3D strict |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 19.1177 | 29.2402 | 21.4299 | 14.5038 | 13.2362 | 27.3009 | 17.9864 | 21.2163 | 0.0771 |
| 2 | 28.5070 | 37.1501 | 38.3198 | 15.8556 | 20.3565 | 39.4961 | 22.0194 | 26.1254 | 0.0725 |
| 3 | 37.1369 | 43.4095 | 48.9184 | 22.5426 | 28.4589 | 48.6276 | 29.5955 | 33.3557 | 0.1090 |
| 4 | 35.6039 | 44.8242 | 41.8101 | 28.1526 | 27.1095 | 45.3434 | 29.5277 | 33.7745 | 0.0962 |
| 5 | 38.3040 | 45.6406 | 49.5620 | 26.2806 | 27.2263 | 50.1470 | 29.4711 | 36.0654 | 0.1298 |
| 6 | 40.3277 | 46.9958 | 52.2399 | 29.8608 | 27.5886 | 51.6215 | 30.4021 | 34.1337 | 0.3769 |
| 7 | 42.9447 | 50.9064 | 54.1495 | 33.6842 | 28.8778 | 55.0674 | 31.6760 | 38.5354 | 0.1457 |
| 8 | 41.2114 | 49.6924 | 51.6266 | 34.0215 | 28.4565 | 50.7410 | 31.1274 | 38.0193 | 0.2669 |
| 9 | 39.4725 | 48.1623 | 54.0738 | 32.2299 | 22.9396 | 48.6467 | 28.4103 | 38.8412 | 0.0695 |
| 10 | 41.3039 | 50.1124 | 52.4747 | 34.1621 | 25.1979 | 53.3809 | 28.8165 | 38.5636 | 0.1267 |
| 11 | 35.5527 | 43.1819 | 46.4582 | 17.3421 | 26.4535 | 51.9570 | 30.8697 | 41.5146 | 0.0953 |
| 12 | 42.4026 | 49.9448 | 51.7280 | 37.5153 | 29.0721 | 51.2952 | 31.0196 | 40.0218 | 0.1906 |
| 13 | 39.0580 | 45.8465 | 52.2459 | 32.9351 | 24.8813 | 46.1698 | 27.8390 | 39.4104 | 0.1009 |
| 14 | 40.4769 | 49.2348 | 52.9350 | 34.5024 | 23.4663 | 51.0039 | 28.7936 | 39.1216 | 0.1856 |
| 15 | 41.0212 | 49.3781 | 52.0319 | 35.3201 | 25.5402 | 51.1927 | 29.4608 | 40.7853 | 0.2372 |
| 16 | 33.9343 | 41.4593 | 48.6768 | 30.7083 | 12.5310 | 43.8211 | 14.1139 | 27.4726 | 0.0630 |

**配对窗口逐项均值**（ep12–16，均 n=5；Δ=v2−no2d_igdr seed0）：

| 指标 | no2d_igdr seed0 | MCI-v2 seed0 | Δ |
|---|---:|---:|---:|
| Overall_3D_moderate | 39.2809 | 39.3786 | +0.0977 |
| Car_3D_moderate_strict | 47.8399 | 51.5235 | +3.6836 |
| Truck_3D_moderate_strict | 29.7246 | 34.1962 | +4.4716 |
| Pedestrian_3D_moderate_loose | 30.9277 | 23.0982 | -7.8295 |
| Cyclist_3D_moderate_loose | 48.6313 | 48.6965 | +0.0652 |
| Pedestrian_BEV_moderate_loose | 33.7111 | 26.2454 | -7.4657 |
| Pedestrian_2D_moderate_loose | 43.3698 | 37.3623 | -6.0074 |
| Pedestrian_3D_moderate_strict | 0.2037 | 0.1555 | -0.0482 |

**家族对照**（窗口均值均 ep12–16，n=5；峰值取各 run 全部 val）：

| run | Overall 区间均值 | Overall 全轮峰值 | Ped loose 区间均值 | Ped loose 全轮峰值 | ep16 Overall | ep16 Ped loose |
|---|---:|---|---:|---|---:|---:|
| no2d_igdr seed0 | 39.2809 | 40.3853 @ ep14 | 30.9277 | 34.2463 @ ep14 | 39.9949 | 32.7570 |
| MAI seed0 | 40.1242 | 41.4160 @ ep16 | 27.5324 | 31.0060 @ ep10 | 41.4160 | 27.1785 |
| MCI-v1 seed0 | 40.1061 | 41.6747 @ ep7 | 26.8606 | 31.0544 @ ep14 | 39.7467 | 24.1058 |
| pdrop10 seed0 | 39.8998 | 42.6611 @ ep12 | 27.3946 | 29.9101 @ ep12 | 37.1772 | 26.0288 |
| MCI-v2 seed0 | 39.3786 | 42.9447 @ ep7 | 23.0982 | 29.0721 @ ep12 | 33.9343 | 12.5310 |

**现象与机制证据，严格分级**：

1. **确定：长期类别缺口叠加 ep16 急跌，不是只由最后一轮造成。**
   v2 Ped ep12–15 均值 25.7400，对同窗基线 30.4703 为 −4.7303；
   ep16 为 12.5310，对基线 32.7570 为 −20.2260。固定五轮窗口差
   −7.8295，其中 ep16 配对差贡献 −4.0452；不能事后删 ep16。
   v2 相对 v1 同窗额外下降 −3.7624，但只有一个 seed，尚不能把差值
   全部归因为 Xavier/幅值通道。

2. **确定：不是仅 z/h 尺寸回归或高度问题；检测链整体受影响。**
   ep15→16 Ped 3D loose 25.5402→12.5310，BEV loose
   29.4608→14.1139，2D loose 40.7853→27.4726。BEV/2D 同降
   排除“只坏在 3D 高度”的解释；仍需匹配级拆解区分 recall、分类/排序、
   中心定位和 NMS。Ped strict 的历史几何瓶颈与这次 loose 下滑不是一回事。
   同轮 Car/Truck/Cyc 也下降，Overall 从 41.0212 降至 33.9343；
   按 Overall 四项构成，Ped 下降贡献 3.2523 AP，约占总降幅 45.9%。

3. **确定：创新通道已活，但门控没有证明按 Ped 可靠性做出正确区分。**
   v2 探针 innovation 向量样本特异能量 58.9732%，经过
   `innovation.detach().abs()` 和 1×1 conv 后仅 5.4212%（共模 94.5788%）。
   这是 **3 个样本的批内分解**（indices=3/703/1403），不是 Ped GT 区
   读数，也不等价于“空间恒定”或“Ped 噪声已证实”；只能说明该幅值
   输入偏共模。源码 `me_rssm/mci_rssm.py` 的校正/读出门分别调制
   posterior correction 和整条时序 residual，故全局压低增益会减少时序贡献，
   但谁最受伤需按类反事实探针证明。

**训练动力学**（train 日志各 epoch 已记录条目的等权均值；非 val 门图）：

| ep | correct gain | readout gain | innovation² | loss_cls | loss_bbox | grad_norm |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0.421151 | 0.307075 | 0.698133 | 0.145596 | 0.293444 | 5.554081 |
| 12 | 0.364829 | 0.237852 | 0.515112 | 0.098223 | 0.201853 | 4.254670 |
| 13 | 0.356566 | 0.232092 | 0.491363 | 0.091872 | 0.190579 | 4.222459 |
| 14 | 0.353006 | 0.228150 | 0.465521 | 0.086208 | 0.178752 | 3.981243 |
| 15 | 0.351104 | 0.225677 | 0.440036 | 0.081368 | 0.170260 | 3.966654 |
| 16 | 0.350040 | 0.224465 | 0.424795 | 0.075298 | 0.159200 | 3.692404 |

- **有证据的主嫌疑**：innovation 后验带来的类别脆弱性（MAI 无 MCI 门控
  已有 Ped 同窗 −3.3953），叠加缺少可靠性辨别的全局时序抑制。v2 训练
  correct/readout gain 从 ep7 的 0.421/0.307 收缩到 ep16 的 0.350/0.224。
  Ped 弱 Doppler、非刚性运动可能让 innovation 包含较多预测误差，
  但“Ped 区 innovation 偏大/读出门偏低”仍未有足够按类数据证实。
- **不能直接解释 ep16 急跌**：ep15→16 correct/readout gain
  0.351/0.226→0.350/0.224，innovation² 0.4400→0.4248；loss_cls、
  loss_bbox、grad_norm 均下降，没有这些日志上的整体爆炸证据。
  因而不能声称“ep16 门突然关闭”或“梯度爆炸”已定位；训练均值也不能
  排除验证分布、BN 统计、少量样本异常或 score/NMS 敏感性。

**纠正 §68.9e 的旧措辞**：MCI-v1 seed0 ep16 Overall 实际是
**39.7467**，不是 37.18；37.1772 属于 pdrop10。不能写“家族 ep16
塌陷第三次/3/3”，更不能据此推断存在固定 ep16 系统故障。

**结论与下一步**：现有证据支持“innovation 共享的 Ped 类别弱点 +
非选择性门控抑制”为优先假设，不支持宣称已找到 v2 ep16 急跌的唯一根因。
下一步先对 v2 ep14/16 与 v1 ep16 做同样本、足够 Ped GT 的只读诊断：
① GT 中心/区域的 innovation 与 g_c/g_r 分布（至少 limit≥40、stride≤100，
并报告实际 Ped GT 样本量）；② eval 反事实单独固定 g_r=1、g_c=1，
再去掉 innovation 幅值输入；③ 同样本匹配拆解 score、recall、中心误差、
NMS 前后保留率。BN/验证状态审计作为 ep16 急跌的排除项。
反事实只是机制定位，不替代正式消融训练；本任务未执行上述新 GPU 探针。


### 68.9g 论文级 z 修复与类别改善候选研究（2026-10-08；无新训练）

- 完整假设、机制、prior art、消融、风险与接受条件：
  `docs/rssm_paper_directions_20261008.md`。本任务只做仓库/原论文研究与文档，
  不实现模型、不启动训练、未动 GPU，候选均没有新 AP 结果。
- 研究所用统一上下文：MCI-v2 Overall 区间均值（ep12–16，n=5）
  **39.3786**；全轮峰值 **42.9447 @ ep7**；best saved
  **42.4026 @ ep12**。Ped loose 同窗 **23.0982**、全轮峰值
  **29.0721 @ ep12**；完整逐 epoch 与类别表继承 §68.9f，不省略或重算新窗口。
- 优先候选 A：对象/正 anchor 粒度的 z 语义监督与检测分类 residual 读出；
  C：分模态 innovation、缺测可识别、校准不确定性的 predict-correct。
  A/C 独立验证后才考虑组合。B：对象 RSSM 为后续架构升级；D：对象排他
  未来目标在关联和 prior 站稳后验证。不是再次重跑历史低维全图差分方案。
- 不把 shuffle 输出变化、方差增长或 KL 非零等同于 z 修复；需真实分类/AP、
  历史反事实、同容量确定性模型与缺测校准对照。所有类提升必须逐类配对
  多 seed 验证，不能从 Overall 或分类 CE 推断 3D AP 四类都提升。
- `docs/todo.md` 已记录本次研究完成；z 机制修复、所有类 AP 提升与训练验证
  均仍未完成，既有实现/训练排期未修改。

### 68.9h DC-RSSM-1（证据场 L3）终判（2026-10-08 23:21 gate hit；探针 10-09 00:15 重跑完成）：AP 门 FAIL，机制门 3/4

- 训练：24e config（与 MCI/MCI-v2 同构），2026-10-08 06:28 启动；watchdog
  23:20 gate hit 但 **stop 信号落空**（本次训练用 nohup 启动而非 tmux
  session，`tmux send-keys -t dcrssm1` 无目标静默失败——流程事故，已修：
  手动 SIGINT 停训于 ep17 中，探针由主会话并行重跑）。首轮探针 OOM
  （另一用户 MyFusion 同时占 GPU 0-3 + 排空等待超时），重跑全部成功。
- 全轮曲线 ep1–16 = 21.76 / 28.64 / 34.23 / 34.73 / 36.46 / 37.27 /
  37.53 / 38.38 / 39.45 / **40.59** / 39.31 / 40.03 / 37.90 / 40.24 /
  38.76 / 39.15（全轮峰值 40.59 @ ep10）。
- **AP 门 FAIL**：ep12-16 = **39.2169**，基线门未过（≥39.2809，差
  −0.0640）；v2−0.5 非劣门实际通过（≥38.8786，差 **+0.3383**；
  2026-10-10 日志复核纠正）。总判定仍因基线/Ped 门失败保持 FAIL。逐类 ep12-16：Ped 27.07 /
  Cyc 43.91 / Car 73.75 / Truck 52.02。
- **Ped 配对差 −3.862 < −1.5：coverage 语义修复 Ped 的直接检验 FAIL**。
  家族 Ped 窗口（ep12-16）：基线 30.93 / MCI 26.86 / MCI-v2 23.10 /
  DC-RSSM-1 27.07——证据场把 Ped 从 MCI-v2 的 23.10 拉回 27.07
  （+3.97，家族除基线外最好），但仍低于基线 3.86。cov 语义方向正确、
  幅度不够；Cyclist 同时回落到家族最低（43.91 vs 基线 48.63）。
- **机制门 3/4**（docs/13 step 1 预登记）：
  - 稳定性（本代际新证据）：ep1–9 **零负 dip**（家族首次完全单调爬升，
    对照 MCI −3.35 / MCI-v2 −1.74 / 基线 −1.06），c_mot0 std 全程冻结
    0.013——测量锚定的平稳性兑现。
  - c_mot 结构 PASS（实测 36 帧 measured-empty 对比 0.361，学习版塌缩
    的 66 倍）；dyn 结构弱（对比 0.04，测量内 std 0.099 临界）。
  - h/e <2 ✓：0.922（h 移除 0.304 / e 移除 0.333），观测主导保持。
  - 「幅值域去共模」✗：innovation_mag 样本特异能量 5.3%（对照 MCI-v2
    5.4%）——证据场没有改善幅值门的共模问题（物理场与 innovation
    幅值是独立信号，该假设由本判据证伪）。
- P1 现状：zero_z head 27.4% vs shuffle_z 4.7–5.4%，readout 偏置主导
  未变；gain_mean 收缩至 0.30（readout）/0.35（correct），门在证据下
  大幅收缩（自我限幅）。
- 权重活性 ep16：evidence_refine_conv abs_mean 0.135（small-normal 0.01
  起步，学习路径全程活跃）、门 conv 0.041、innovation_mag 0.073。
- **判定**：R-neg（AP 门 + Ped 配对双 FAIL，预登记即止损）。**L3 证据场
  单独上线被证伪**：它修复了 MCI-v2 的 Ped 塌陷（+3.97）、带来家族首次
  单调曲线，但整体 AP 不及基线、幅值共模未解、Cyclist 回落。
- **对 docs/13 链的影响**：step 1 止损记录。下一步（§0 汇报义务）：
  L1 ego-warp（解析自标定，旗舰层，针对对齐纯外观驱动的**不同**根因）
  是 DC-RSSM 方向最后一次尝试——它失败则方向停止，论文转机制分析定位
  （§68.9g 的候选研究 docs/rssm_paper_directions_20261008.md 并行成立）。
  L3 证据场作为 Ped 修复组件保留（+3.97 正向资产），可并入 L1 之后的
  组合验证而非单独主张。
- 探针产物（DC-RSSM-1 work_dir）：`z_util_gated_ep16_w{3,100,1800}.json`、
  `post_source_gated_ep16_stride700.json`、`probe_w*.log`、
  `watch_seed0_ep16.log`。

### 68.9i DC-RSSM-2（+L1 解析 ego-warp）：预启动机制修正、冒烟门控与上线（2026-10-09，全部写于任何 24e 数据之前）

- **背景**：§68.9h 判定 L3 单独上线被证伪，但 L1 针对的是**不同**根因
  （对齐纯外观驱动；docs/08 静态点呈现 −v_ego 流场 → ego 运动可从 Doppler
  解析自标定）。L1 是 DC-RSSM 方向最后一次尝试。
- **根因定位（只读探针，先于任何训练）**：
  - 几何先被证伪排除：`probe_ego_geometry_test.py` 对 canvas↔world 的
    **8 种映射假设**逐一拟合，残差 RMS 全部落在假定映射的 10% 以内且假定
    映射最优 → 布线/格尺寸/符号正确；LMedS 刚体场以 **0.013 m/s 中位残差**
    解释 58–80% 的测量格。
  - 真缺陷是**预登记的加权闭式解被运动格尾巴拉偏**：原规则 `w=count·SNR`
    归一化后闭式解，实测被动目标强回波格权重可达静态格 ~100 倍，val 上
    解算 inflation **+24%…+96%**（idx3 4.20–4.34 vs 真值 2.96–3.03；
    idx703 3.19–3.62 vs 1.84–1.86；idx1403 1.73–1.87 vs 1.37–1.42）。
  - IRLS-Huber 无效（inlier_frac@0.5 = 0.000，MAD 尺度自身被污染）→ 结论
    是「稳健**集合**」而非「稳健权重」，落成四段流水线（分位预过滤 →
    LMedS 播种 → MAD 裁剪 → 加权闭式解，docs/13 §2 L1）。
    离线 12 帧验证误差 ≤0.2%。
  - **两处反证据被证伪**：(i) 占用互相关位移（canvas 仅 0.3–1% 占用、
    峰/背景比 1.9–2.1、被对向车流主导 → 8.8–16 m/s 垃圾值），仅作负对照
    保留；(ii) docs/08 F1 的「7–13 m/s」是色标 **p98→max 尾统计**（正文
    写明 98 分位自适应 ±2~±10），非均值——实测 p98 = 8.8–10.7、max 11.8–12.1
    @idx3，与修正后的解算无矛盾。
- **离线验证 + 单测**：W11 构造「静态场 + 8 个强回波同向运动格」
  （w=8.4 vs 静态 0.3），断言朴素解误差 >0.15 m/s 而稳健解 <0.05 m/s，
  且 trim_frac = 64/72（在预过滤集上归一化）；24/24 单测绿。
- **冒烟 450 iter（v1 → v2）**：
  - v1 用预登记加权解：stat_ego_speed 2.64–2.90（偏高）。
  - v2 用稳健解：speed **1.74–1.99**（与稳健真值 1.40–3.03 一致）、
    trim_frac≈0.78、τ_r≈0.21–0.23、static_frac 7.1e-3–7.7e-3、
    solved 0.999；门控 **E1–E7 全 PASS**。
  - E5 归一化修正：static_frac 是**整张 canvas** 的比例，TJ4D 测量格只占
    0.3–1%，原草案 0.05 阈值放大约 100 倍，改为 5e-4（写在任何 AP 数据之前）。
  - E8 修正（预登记改写，写于 ep16 前）：绝对线
    `|warp_scale|·dt·|v_ego| > 1e-3 m` 在冒烟（450/46800 iter ≈1%，cosine
    已衰减、AdamW 每步有效位移≈7% lr）实测 7.47e-4。改为 **E8a 冒烟判据 =
    后 1/4 均值 ≥ 前 1/4 均值的 2 倍且斜率 >0**（实测 2.4×，
    `dcrssm2_gate_report.py` PASS），**E8b 绝对线在 ep16 生效**并同时报告
    「已施加/需求位移」占比与「按当前斜率还需多少 iter 到线」。动机：把
    「机制从未被启用（UNEXERCISED）」与「机制被证伪（FAIL）」分开。
  - **判据适用范围（写于 ep16 前）**：E1（零初值不跳变）与 E8a（窗口增长）
    是**冒烟窗口专属**判据；记录跨 epoch 时脚本自动按「整轮模式」运行，
    将其降为 INFO，只强制 E2–E7 + E8b。整轮模式下处罚 E1/E8a 无意义：
    24e 的 cosine 衰减慢、标量移动快得多，且 epoch 级窗口天然非单调
    （首轮实例：E1 首记录 6.5e-3、E8a 1.0×，均为判据错置而非机制问题）。
- **预启动优化速率决定**：仅对 `temporal_fusion.dt` 与
  `temporal_fusion.warp_scale` 设 lr_mult=5.0（实测 AdamW 有效步长≈7% lr；
  单测确认 paramwise 命中这两个参数）。24e 上 warp_scale 6.5e-3@iter50 →
  4.16e-2@iter1600；按此速率 ep16 落在解析最优量级（位移 = 需求的
  10–100%），E8b 绝对线留有 60–200 倍余量。
- **ep16 机制门（预登记，写于任何 24e 数据之前；探针
  `probe_ego_alignment_toggle.py`，三趟 learned / 强制 1.0 / 0）**：
  (a) 平均已施加位移 <1e-3 m → **UNEXERCISED**（AP 结论与机制无关，不写
  「失败」）；(b) 非惰性时 learned vs 0 在 **≥8/10 clip** 降低静态格跨帧
  残差（按 |h_curr| 归一化）→ **MECHANISM PASS**，否则 **MECHANISM FAIL**；
  (c) 强制 1.0 为机制幅值对照（不依赖损失把标量落在哪里）。
  探针镜像函数 + 逐前向调用分组假设由 **W12 单测**钉住（镜像解 == 模块
  stash；镜像 warp == 解析位移 0.2 格；一次前向恰好 2 次对齐调用、
  call[0]=h 且其输入逐位等于上一帧输出态）——三趟 harness 因显存被训练
  占满无法端到端自检，故把可离线部分全部前置验证（24/24 单测绿）。
  **帧配对修正（只读复核发现，早于任何 ep16 数据）**：warp 后的携带态必须
  与**同帧**更新态配对（原实现取 t−1 的调用，差一帧），已在代码注释中写明
  理由并由 W12 的「携带态逐位等于上帧输出态」断言守住；该修正不接触训练
  数据、不改变预登记判据。
- **AP 门（预登记，承 §68.9h）**：ep12-16 配对窗口均值 ≥ 39.2809（基线）
  且 ≥ 38.8786（v2−0.5），Ped 配对差 ≥ −1.5，Car/Truck 不回退。
  **失败即停止 DC-RSSM 方向并记录**，不做超参扫描；论文转机制分析定位
  （docs/rssm_paper_directions_20261008.md）。
- **启动**：2026-10-09 10:54 UTC，`tmux dcrssm2`（GPUs 5/6/7，seed 0，
  deterministic），work_dir `/data/lurui/work_dirs/fgfull_N4_no2d_igdr_dcrssm2_2x4_24e_seed0`；
  ep16 看门狗 `tmux dcrssm2_watch`（`watch_dcrssm2_ep16.sh`，本代训练在
  tmux 内启动，修掉 §68.9h 的 send-keys 落空事故）→ 停训后串行跑
  P1 z_utilization_gated(w3/w100/w1800)、P2 posterior_source_gated、
  P3 probe_ego_warp_sanity、P4 geometry dump+robust+robustfix、
  P5 alignment_toggle，日志 `watch_seed0_ep16.log`。
  **加固（ep16 前，重臂一次）**：停训后先断言 GPUs 5/6/7 真的空闲，
  不空闲则中止并留日志（不跑探针、不留 OOM 垃圾），避免重复 §68.9h
  「stop 落空 → 探针 OOM」的时间损失；脚本改动后**重新臂了看门狗**
  （运行中的 bash 会按字节偏移续读脚本，改文件不安全），watch log 记有
  重臂原因；ep16 尚未到达，重臂不改变任何门控条件。
- 训练现状（ep2 内）：warp_scale 6.5e-3@iter50（ep1）→ 峰值 4.16e-2，
  均值 2.96e-2；dt 0.107→0.140；stat_ego_speed 1.65–2.02（稳健带内）；
  static_frac 6.9e-3–7.9e-3、trim 0.756–0.787、solved ≈0.998。
  **E8b 已 PASS**（ep2 读数 6.76e-3 m = 需求的 2.9%，阈值 1e-3 的 6.8 倍）。
- 进度实测：iters/epoch ≈1952（3 GPU × samples_per_gpu 2），ep1 训练+val
  实测 65 min（10:54→11:59 UTC），val 每 epoch、checkpoint 每 2 epoch
  （→ `epoch_16.pth` 一定存在，看门狗等待条件成立）。ep1 val
  Overall_3D_moderate = 20.81（同一代际 DC-RSSM-1 的 ep1 = 21.76，
  差异在家族早轮噪声内；无过早崩塌迹象）。预计 ep16 gate 在 10-10
  05:00–08:00 UTC 命中（约 18–21 h），终判写入本节。
- **门控可满足性审计（ep16 前，CPU-only，重要）**：P5 预登记的判据是
  「≥8/10 clip」，但 val 集 2040 帧在**原定 `--stride 300` 下只有 7 个**通过
  history-validity 的起始帧（3, 303, …, 1803）→ 判据**按构造不可满足**
  （与机制无关的死锁）。已在 ep16 前把 stride 固定为 **200**（同为 3…1803
  范围、20 s 间隔、10 个 clip 独立），8/10 阈值本身不变；探针默认值与
  看门狗调用同步修改，看门狗已重臂（watch log 记有原因）。同一次审计确认
  P2/P3 的 stride 700 实际只取到 3 个 clip（它们是读数/几何门不依赖 clip
  数，P4 的 LMedS 一致性门用固定 idx 3/703/1403，不受影响）。
- **终判工具已备（不代替判读）**：`me_rssm/sanity/dcrssm2_ep16_verdict.py`
  （只读、CPU-only）读 val 曲线 + `ego_alignment_toggle_ep16.json`，一次输出
  全轮曲线 / 峰值 / best saved（统一三口径）、ep12-16 窗口、四项 AP 判据、
  机制三分类。已用 **DC-RSSM-1 的既有数据 dry-run 校验**：复现文档已记录的
  窗口 39.2169、峰值 40.5929@ep10、Ped −3.8621、Cyc loose 43.9105，以及
  §68.9h 引用的 Car loose 73.7487 / Truck loose 52.0163。
- **口径钉死（写于 DC-RSSM-2 只有 ep1-2 时）**：Car/Truck 用 **strict**
  （家族 canonical 逐类窗口表 §47.2 对照与 §68.8d 均用 Car strict / Truck
  strict / Cyc loose / Ped loose；§68.9h 引的 73.75/52.02 是同一 run 的
  **loose** 变体），Ped 用 loose 且阈 ≥ −1.5；Cyclist 仅作 context 报告、
  **不加门**（MCI-v2 与 DC-RSSM-1 都在 Cyc loose 掉 4–5 分，值得记录，但
  现在把它写进门控属于事后新增判据）。
- 早轮读数（非判据，单轮不构成结论）：ep1 20.8122、ep2 32.1085；同代际
  DC-RSSM-1 的 ep1 21.76 / ep2 28.64、主线 RSSM ep2 29.65。ep2 领先
  ~2.5–3.5 分但仍在家族早轮波动带内，门控照旧只看 ep12-16 窗口。
- **续训通道已预备（不自动执行）**：`me_rssm/sanity/continue_dcrssm2_ep16.sh`
  ——若 AP 门 PASS，从 `epoch_16.pth` **resume 到 ep24**（mmcv `--resume-from`
  恢复 epoch 计数，cosine 从 ep16 续接而非重启；新 epoch 追加进同一
  work_dir，保持 ep1-24 统一记录）。三重守门：(1) 必须存在并含 `GO`
  的 `$WORK_DIR/gate_decision.txt`（门控判读后由操作者写入，脚本自身
  不做任何判定）；(2) 看门狗探针阶段已结束（不与探针抢显存）；(3)
  GPUs 5/6/7 空闲。dry-run 已验证无 GO 文件时拒绝启动（exit 1）。
  注：中途 resume 不恢复 dataloader 洗牌状态（mmcv 仅在 `--resume` +
  latest meta 时恢复），故 ep17+ 与「未中断的 24e」并非逐位相同——
  已记录，对窗口均值无影响。

### 68.9j DC-RSSM-2（+L1 解析 ego-warp）ep16 终判（2026-10-10 04:23 停训 / 04:37 探针 / 04:5x 终判）

**结论：AP 门 REJECT，机制门（L1 专属判据）MECHANISM PASS。**
这是 DC/MCI 家族**第一次**出现「新加的机制通过自己的预登记判据、AP 仍不过门」——
机制-AP 解耦（P7）的最干净实例。按 §68.9i 预登记：停止 DC-RSSM 方向并记录，
不做超参扫描；`gate_decision.txt` **不写 GO**，ep16→ep24 续训通道不启动
（该通道只在 AP 门 PASS 时可用）。

- **运行事实**：ep16 权重 04:23 落盘、ep16 val 04:34 完成，看门狗按序停训
  （tmux `dcrssm2` / `dcrssm2_watch` 均已退出，GPUs 5/6/7 释放）→ P1 z-util
  三窗、P2 posterior source、P3 ego-warp sanity、P4 geometry+robust
  全部完成；**P5 alignment toggle 首跑崩在探针自身的 CUDA 设备缺陷上**
  （见文末过程记录），修复后重跑完成并写出 `ego_alignment_toggle_ep16.json`。

**三口径（16 条 val，跨度 ep1-16）**：
- 区间均值（主窗口 ep12-16，n=5）**38.5345**；全轮峰值 **41.1838 @ ep8**；
  best saved **41.1838 @ ep8**（偶数落盘点最高）；last val **39.2968 @ ep16**。
- 逐 epoch：20.81 / 32.11 / 33.13 / 36.03 / 37.73 / 38.76 / 39.25 /
  **41.18** / 39.79 / 40.11 / 39.74 / 39.96 / **36.82** / 37.97 / 38.62 /
  39.30。ep8 峰值后整体在 39.7–40.1 平台，ep13 深坑后 ep14-16 单调回升
  （37.97→38.62→39.30）但未回到 ep12 水平。

**AP 判据（预登记四条）**：

| 判据 | 阈值 | 实测 | 结果 |
|---|---|---:|---|
| ep12-16 窗口均值 vs 基线 seed0 | ≥ 39.2809 | 38.5345 | **FAIL −0.7464** |
| ep12-16 窗口均值 vs MCI-v2 −0.5 | ≥ 38.8786 | 38.5345 | **FAIL −0.3441** |
| Car strict 配对差 | ≥ 0 | 53.4146 vs 47.8399 = **+5.5747** | PASS |
| Truck strict 配对差 | ≥ 0 | 30.7905 vs 29.7246 = **+1.0659** | PASS |
| Ped loose 配对差 | ≥ −1.5 | 24.5784 vs 30.9277 = **−6.3493** | **FAIL** |

context（不加门）：Cyc loose 45.3545 vs 48.6313（−3.2768）；Car loose
76.0287、Truck loose 53.6684、Ped strict 0.1958。

**机制读数（ep16）**：
- **P5 L1 toggle（预登记判据）**：learned vs 强制 0 在**静态格跨帧残差上
  8/10 clip 获胜**（阈值 ≥8/10）→ **MECHANISM PASS**；非惰性
  （已施加 0.058 m/frame，E8b 线 1e-3 m）；**强制 1.0 的幅值对照只赢
  5/10**——把尺度推到模块自身满程（0.449 m/frame）并不保持改进，说明
  L1 的收益是**幅度敏感**的、局部于学到的约 13% 工作点，而不是「只差
  把标量推大」。
- **P1 z-util 三窗**（head L2 比，4 样本/窗）：`shuffle_z`
  **5.62% / 5.39% / 5.54%**（逐样本 5.1–6.0%，三窗齐平）；`zero_z`
  29.66% / 32.67% / 34.38%；`prior_only` 3.10% / 3.36% / 3.17%。
  即家族最好的 identity-signal 读数（DC-RSSM-1 ep16 4.82%、MCI-v2
  3.97–4.86%），且三窗全部 ≥2% 论文线、不再受窗口限定。
- **P2 posterior source**：`correction` 样本特异能量占比 **41.77%**、
  `e_pooled` 24.47%、`mu_q` 18.98%、`mu_p` 10.83%、`h_t` 14.80%；
  `innovation²` 0.0247、`mu_diff²` 0.0781、clamped 0.9977。
- **P3 ego-warp sanity**：9 对帧，mean_v_ego 2.075 vs 占用互相关
  8.814 m/s（比值 4.12，corr 0.819）——稳健解在量级上正确、占用互相关
  仍是对照垃圾（与 §68.9i 的负对照结论一致）。
- **P4 geometry + robust solve**：修正解与真值差 **0.1–0.2%**
  （朴素加权闭式解 24.4–95.5%）。稳健解在 ep16 仍完全成立。

**归因（机制级，可直接进论文 Analysis）**：
1. **同一签名第三次复现**：Car/Truck 大涨（+5.57/+1.07）、Ped/Cyc 大跌
   （−6.35/−3.28）、Overall 不过门——MCI seed0/1/2 与 DC-RSSM-1/2 五代
   同形。机制读数（identity signal、correction 特异占比、h/e 观测主导）
   越好，AP 越不动的模式未变。
2. **L1 单独成立但不改变结局**：它是本方向唯一「通过自己机制判据」的
   组件（8/10），且 P4 证明其解算精度 0.1–0.2%。它的作用域是**静态格
   对齐残差**，而 AP 的缺口在 Ped/Cyc 的**检测级**回归——两者不是同一
   个瓶颈。
3. **与 §69 的交叉解释**：ep16 的重建目标在 DC-RSSM-1 上是 95.5% 样本
   无关模板（特异 4.5%），且 DC 代际在 `shuffle_z` 下**前景没有集中**
   （fg−bg_hi −0.0001），而 baseline MAI 有 +0.145——即「z 在物体上不
   携带样本特异内容」。L1 修正的是对齐几何，不能改变这一条。
4. **Ped 缺口的定位**：Ped loose −6.35 是本家族三代的共同缺口（MCI-v2
   −7.83、DC-RSSM-1 −3.86、DC-RSSM-2 −6.35；v2 固定窗口差纠正）；Ped strict 恒 0.196，
   与 §29–39 的探针几何瓶颈一致，属检测级而非高度/几何级。

**过程记录（探针缺陷，全部在测量侧）**：
- P5 首跑 `RuntimeError: indices should be either on cpu or on the same
  device as the indexed tensor (cpu)`：`probe_ego_alignment_toggle.py` 的
  `static_mask_from_canvas` 镜像里 5 处 `torch.arange` 未带
  `device`/`dtype`，而模块 `_solve_ego` 是同款 arange 都带
  `device=vel.device, dtype=float32`。W12 只在 CPU 上跑，故未暴露。
  修复：逐处镜像模块写法；**CUDA/CPU 一致性实测**（物理自洽的静态场上
  掩码逐位相同、v_ego 2.300000 / τ_r 一致）。预登记判据未改动。
- 该崩溃使首轮 watchdog 的 P5 无输出；本次为**修复后重跑**，判据、clip
  集合（stride 200，10 clip）、阈值（8/10）与预登记完全一致。

### 68.10 结论汇总（截至 2026-10-05，覆盖 §62–68 全部训练与诊断）

1. **AP 结论**：MCI 三 seed 配对差 +0.83 / −0.80 / −0.38（m=−0.12±0.69），
   与强基线打平、显著优于 MAI 的 roll（−0.59±1.41），**不构成 AP 提升
   主张**（R-neg）；MAI（m=−0.59）与低维 A/B/C 路线的负结果一并封存。
2. **机制结论（三 seed 稳定，论文 Analysis 的骨架）**：① identity
   signal 三 seed × 三窗 9/9 ≥2% 论文线（MAI 只有 ep6 短暂达到且不保持）；
   ② h/e 依赖比 0.57–0.70 持续观测主导（标准主线 2.34 且恶化、MAI ep16
   反转 1.12）；③ correction 特异占比 44–50%；④ posterior collapse 的
   稳态仍在（clamped 0.995）但 `μ_diff²` 0.10 为历史 3–10 倍——概率部件
   依旧装饰性，语义已迁移到门控。
3. **诊断结论**：Ped 红灯 = 检测级回归、与 MAI 共享（H1 innovation
   噪声为主嫌）、门控增量未证实（探针功效不足待加强重跑）；**innov
   门控通道死锁**（零乘积形态）为本次体检新发现，已修复并回归测试；
   门控的空间分辨力（逐像素可靠性）是「设计目标」而非「已实现能力」，
   是 MCI-v2 与后续迭代的主攻点。
4. **路线状态**：pdrop10（Ped 修复）在训；MCI-v2（ innov 通道复活）
   待排；消融冒烟全绿完成等待 AP 级筛选决策；论文 Method/Related
   Work 可写 90%（docs/11/12），主表等待上述判定。

### 68.8f seed1 终判（2026-10-05 ~01:50 UTC ep16 自动停训+探针）与 seed2 接力

- **AP**：ep12-16 = 41.1218（五轮 40.1829/41.8749/40.9726/41.7180/
  40.8605；全轮峰值 **42.8014 @ ep11**），对 seed1 基线 41.9185 配对差
  **−0.7967**。解读：seed1 的 no2d_igdr 基线是三 seed 中最强的（41.92），
  MCI 在其上小幅未及；**负向幅度远小于 MAI seed1 的 −2.51**，且 MCI
  seed1 的绝对曲线（ep11 42.80、ep13 41.87）高于 MAI seed1 同期多数点。
- **机制跨 seed 保持**：三窗 `shuffle_z` **2.24% / 4.82% / 2.95%**（全部
  ≥2% 论文线，与 seed0 的 4.08/4.12/2.13 同水平）；`h/e` removal
  **0.699**（观测主导，两个 seed 都未出现 MAI 式 ep16 反转）；
  `shuffle/zero` 0.19–0.23（高于 seed0 的 0.10–0.13）。
- **三 seed 中间站**：配对差 +0.8252 / −0.7967 / seed2 待定，当前均值
  **+0.01**；同位对比 MAI 的 +0.8434 / −2.5063 / −0.0985（均值 −0.59）。
  seed2（基线 40.1207）终判决定三 seed 结论方向。
- **seed2 已接力**：2026-10-05 02:04 UTC 启动（tmux `mci_seed2` +
  `watch_mci_seed2_ep16`），ep16 自动停训+探针，输出进 seed2 work_dir。
  seed2 落地后执行三 seed 配对终判，并补跑 Ped 空间门控探针
  （`probe_gate_spatial.py`，H2/H3 裁决）——GPU 窗口在 seed2 停训后。

### 68.8e Ped 红灯的机制归因（2026-10-04，诊断中；判别探针待 GPU 窗口）

**现象定性（已核实）**：
1. Ped 回归是**检测级**（BEV loose 同步 −4.41，3D −4.07），不是高度/几何
   估计问题；strict 双方恒 ≈0.2（§29-39 的锚点几何物理瓶颈未变化）。
2. 回归**与 MAI 共享**：MAI seed0 ep12-16 Ped −3.40、ep13-16 走阔至
   −2.49/−4.76/−0.79/−5.58；MCI ep13-16 为 −2.92/−3.19/−2.60/−8.65
   （窗口 −4.34）。MCI 相对 MAI 只多 −0.67，**主嫌疑是两者共享的
   innovation 后验，门控只叠加小项**。
3. **统计性质**：基线 Ped ep13-16 四点 std≈2.5，窗口差 −4.07 ≈ 3.5σ，
   是真实回归；但 ep16 单点两侧都最差（基线自身 28.02→34.25→28.02 的
   大波动），绝对解读需多 seed。
4. **时间线与门权重增长重合**（correct/readout conv RMS
   0.034@ep6 → 0.041@ep12 → 0.050@ep16），且 MCI 早期 Ped 曾优于基线
   （ep2 +7.33 / ep5 +6.16）——回归随门控增强而扩大。

**机制链（按嫌疑排序）**：
- H1（主嫌疑，MAI/MCI 共享）：`innovation = e_obs − sg(obs_prior(h_t))`
  对 Ped 是大噪声——Ped 径向速度弱（docs/08：GT 点级动态分数 Ped
  0.145 vs Car 0.315 / Cyc 0.488）、运动非刚性，obs_prior 预测不好，
  innovation 偏大且不含可靠运动信息 → posterior delta 给 Ped BEV 注入
  噪声。支持证据：MAI 无任何门控也回退 Ped −3.40。
- H2（MCI 特有增量）：可靠性/读出门看到 Ped 区 c_mot/dyn 低 → 按设计
  收缩校正与读出——但**基线的 Ped 收益恰恰来自无门控的完整时序融合**；
  门的 fallback（μ_p 预测）对 Ped 没有运动信息可用，收缩 = 净损失。
- H3（MCI 特有增量）：动态门以 Doppler 为证据，Ped 运动测不到 → 被判
  「静态」→ 记忆更新放慢 → 运动 Ped 特征变陈旧。
- 排除项：Cyc 是 Doppler 最强类（0.488）却只 −1.2/−1.4——「Doppler 弱
  类必受伤」不成立；模式是**增益集中在 Doppler 强且刚性的 Car/Truck，
  Ped 独特在于「是动态目标但测不到」**，与 §68.8d 红灯及预登记预测一致。

**判别探针（已备齐，待 GPU 窗口执行）**：
`me_rssm/sanity/probe_gate_spatial.py`（类内 `_stash_gate_maps` 挂点已
落地并过单测）——在 Ped/Car/Cyc GT 中心与背景处采样 g_c/g_r/c_cam/c_mot/
dyn/|innovation|，若 Ped 中心 c_mot/dyn/g_r 显著低于 Car 且 |innovation|
偏高则坐实 H2/H3；若门值与 Car 无异则 H1 独大。seed1 训练占满 5/6/7
（CPU 路线被 mmdet3d.ops 的 CUDA 导入断言封死），探针排入 seed1 停训
后的同一 GPU 窗口。

**候选缓解（用户决策，暂不动手）**：
a. `modality_dropout>0` 训练压力，教 filter 在 Doppler 缺失时回退；
b. 读出门非对称化（c_cam 高处不允许收缩到 1 以下）；
c. 如实作为论文发现报告：测量条件化滤波以 Doppler 最弱类（Ped）为代价
   换取强测量类的增益——与 docs/08 预注册的类别效应预测吻合。

### 68.8 续训 ep12（2026-10-03 22:02 UTC 启动）+ ep12 预登记规则

- **运行**：`--resume-from epoch_6.pth`（优化器与 24e 日程状态保留），
  tmux `mci_ep12` + 看门狗 `mci_ep12_watch`
  （`me_rssm/sanity/watch_mci_ep12.sh`），stdout 进
  `work_dir/stdout_ep12.log`，ep12 探针输出
  `z_util_gated_ep12_*.json`、`post_source_gated_ep12_stride700.json`。
- **ep12 预登记判定规则**（写于 ep7-12 任何 val 存在之前）：
  - AP：ep8-12 五轮配对窗口均值 vs no2d_igdr seed0 同窗 **38.8906**：
    ≥ **38.3906**（基线 −0.5）→ 续 ep16；< → 止损。参照线：MAI seed0
    ep8-12 = 39.3392（MCI ≥ 该值即明确超过 MAI 同期）。
  - 机制：三窗 `shuffle_z` ≥0.2% 维持；门空间方差 >0.005 维持；h/e <2
    （不回到 h 主导极端）。
  - Ped 黄旗升级条款：Ped loose ep8-12 相对基线差值 ≤−4 → 升级为红灯
    记录（进 Analysis 专节），仍不改变 Overall 止损判据。

## 69. 两项只读测量：z_t 依赖的区域分解 + 重建目标 headroom（2026-10-09，GPU4 授权窗口）

> **2026-10-10 审计补注（§69.11）**：本节原“现役 pool16_globalstd”标签不适用于
> DC/MCI 训练：现役训练用 fullres 原始 BEV MSE，95.5/4.5% 是探针构造的候选目标。
> 区域 baseline 实际为标准 `MotionAlignedRSSMFusion`，不是 MAI；原标签保留为
> 历史记录，此补注与 §69.11 为解释依据。区域依赖不唯一定位读出瓶颈，表中
> 没有 Ped/Truck；样本特异能量不等于语义增益。数据表本身不变。


用户显式授权本次在 **GPU 4** 上执行两项只读测量（此为「训练只用 5/6/7」约束的
一次性豁免，仅限本节两项）。两项都**不训练、不写 checkpoint**：只加载已有
checkpoint 重放验证序列，在缓存 BEV 特征上做分解，产物写进各 run 的 work_dir。
GPU5/6/7 上的 DC-RSSM-2 24e 训练全程未受影响。

### 69.1 动机与预登记判读规则（写于任何数据之前）

- 背景一：§51/§52 曾提出「把 RSSM 做成只针对前景、只学前景运动」的候选方向，
  但没有证据说明 z 的因果效应是否本来就集中在物体区域。
- 背景二：§68.9h/§68.9i 两个 DC-RSSM 代际机制门通过、AP 门 FAIL。需要一个
  跨代际可比的**机制级**读数来解释「为什么机制在动、AP 不动」。
- **预登记判读规则**（先于数据写下，本节结论按此判读）：
  1. 若**动态区/前景区的依赖度并不高于能量匹配的静态区**，说明瓶颈在**读出头**
     而不是状态容量 ⇒ 前景-only 不会自然改善 AP，该方向不立项；
  2. 只有「fg 依赖 ≫ 全背景 且 ≫ 能量匹配背景」才是「z 在物体上被欠用」的签名，
     才支持前景-only；
  3. 测量(2) 的判读：重建目标的**样本特异占比**决定「一个样本无关常向量能满
     足该损失的程度」，占比越低越无法把 z 锚到当前观测。
- 边界：理想前景掩码来自 GT 框，**只在 val 可得**——本节所有数字都是 val 诊断，
  不能进 test 管线，也不构成任何推理期改动。

### 69.2 工具与共享区域定义

- `tools/bev_region_utils.py`（新增）：两项测量共用的**单一定义源**，保证两次
  测量说的是同一批物理区域。区域：
  - `fg` GT 框栅格化（0.5 格外扩）；`bg` 其补集；
  - `bg_hi` 背景中能量达到 fg 中位数的格子——**公平控制**：直接比 fg vs bg 会被
    激活幅度混淆（物体上能量本就高一个量级）；
  - `bg_top10` 背景能量最高十分位——**锐利控制**：当 fg 中位数偏低时仍能成立；
  - `measured` Doppler canvas 占据 >0；`unmeasured` 其补集；`dyn` measured 且
    池化后 v_std>0.2（**对比用，Ped 已知失效**，动态区不可用 Doppler 定义）。
- **网格取向（本轮由假设变为实测事实）**：证据栅格是 (H_x=216, W_y=248)，
  dim2 = 世界 x，0.32 m/格；而 `feature_dict['pts_feats'][0]` 在
  `R4Det.py:967` 经 `permute(0,1,3,2)` 变成像图张量 (248,216)，**是证据栅格的
  转置**。三条独立代码证据（ego 模块的 `vel_raw.shape[-2]==2*W` 断言 +
  `cell_x=69.12/H`；`Anchor3DRangeGenerator` 把 x 中心放在特征 dim3；
  scatter 的 `coors_y*nx + coors_x` 索引）+ 每次运行的**取向自检**：
  `P(measured|fg)/P(measured)` 对 (x,y) 掩码 vs 轴交换反事实掩码，DC-RSSM-1
  ep16 上 **5.14 vs 0.00，12/12**。（注意：把掩码与热图**同时**转置是数学恒等，
  不能作自检；反事实掩码必须是「在交换后的栅格上栅格化再转回」。）
- 无 canvas 家族（baseline/MAI 无 motion chain）自动降级：跳过 Doppler 区与取向
  自检，只报 fg/bg/bg_hi/bg_top10，并打印显式 NOT EVALUABLE 说明，栅格由
  「捕获特征反转」得到（代码钉定事实，已由带 canvas 家族实测确认）。
- `me_rssm/sanity/test_bev_region_utils.py`（W13，19 条，CPU，全绿）：钉住几何
  （4 格精确掩码、外扩超集、yaw=90 交换长宽、越界框丢弃计数）、统计口径
  （全 1 掩码 ≡ 无掩码拆分、常数图 specific=0、i.i.d. 噪声 specific→(S−1)/S）、
  形变约定（恒等 <1e-5、+0.4 格位移的 0.6/0.4 双线性权重、dx=+1 内容下移）、
  Doppler 格映射（原始 (2j+1,2i+1) 值 4.0 → 池化 v_std 1.0 → 格 (i,j)）、
  能量匹配/最高十分位背景、`standardize_masked` 的掩码内零均值单位方差。

### 69.3 测量(1) 结果：区域依赖分解（12 样本窗 [3..14]，同窗同口径）

相对变化 = 该区域 `mean|Δ|/mean|base|`（特征层）× 100%。两代际同一套区域定义。

**DC-RSSM-1 ep16**（`ZInterventionGated` 门控钩点；全局头比 zero_z 0.2759 /
shuffle_z 0.0482 / prior_only 0.0273，与 §68.9h 的 P1 读数 27.4% / 4.7-5.4% 对齐，
探针锚定成立）：

| 区域 | 格点占比 | zero_z feat/cls | shuffle_z feat/cls | prior_only feat/cls |
|---|---:|---:|---:|---:|
| fg | 0.27% | 0.4521 / 0.2708 | 0.0527 / 0.0318 | 0.0641 / 0.0381 |
| bg | 99.73% | 0.4223 / 0.2575 | 0.0547 / 0.0361 | 0.0320 / 0.0192 |
| bg_hi | 90.47% | 0.4263 / 0.2621 | 0.0528 / 0.0353 | 0.0305 / 0.0185 |
| bg_top10 | 9.97% | 0.3503 / 0.2138 | 0.0342 / 0.0272 | 0.0121 / 0.0064 |
| measured | 24.80% | 0.4400 / 0.2733 | 0.0621 / 0.0357 | 0.0562 / 0.0329 |
| unmeasured | 75.20% | 0.4221 / 0.2574 | 0.0546 / 0.0361 | 0.0317 / 0.0191 |
| fg:Car | 0.18% | 0.4629 / 0.2871 | 0.0498 / 0.0316 | 0.0600 / 0.0375 |
| fg:Cyclist | 0.09% | 0.4245 / 0.2329 | 0.0601 / 0.0323 | 0.0739 / 0.0391 |

配对对比（正 = fg 更依赖 z；`pos` = 12 样本中为正的个数）：

| 对比 | zero_z feat | zero_z cls | shuffle_z feat | shuffle_z cls | prior_only feat | prior_only cls |
|---|---:|---:|---:|---:|---:|---:|
| fg − bg | +0.0299 (9/12) | +0.0132 (9/12) | −0.0020 (5/12) | −0.0043 (5/12) | +0.0321 (12/12) | +0.0189 (12/12) |
| fg − bg_hi | +0.0259 (9/12) | +0.0086 (8/12) | −0.0001 (5/12) | −0.0036 (5/12) | +0.0336 (12/12) | +0.0195 (12/12) |
| fg − bg_top10 | +0.1018 (12/12) | +0.0570 (12/12) | +0.0185 (11/12) | +0.0046 (5/12) | +0.0520 (12/12) | +0.0317 (12/12) |

**Baseline（MAI 无 motion chain）ep14**（`dz.ZIntervention` 改写 posterior_mu 路径；
全局头比 zero_z 0.4101 / shuffle_z 0.0525 / prior_only 0.0614；ep12-16 窗口均值
39.2809 / 峰值 40.3853 @ep14）：

| 区域 | 格点占比 | zero_z feat/cls | shuffle_z feat/cls | prior_only feat/cls |
|---|---:|---:|---:|---:|
| fg | 0.27% | 0.6170 / 0.3502 | **0.1888 / 0.0937** | 0.1271 / 0.0547 |
| bg | 99.73% | 0.6848 / 0.4331 | 0.0492 / 0.0275 | 0.0885 / 0.0479 |
| bg_hi | 90.47% | 0.6940 / 0.4442 | 0.0438 / 0.0249 | 0.0849 / 0.0471 |
| bg_top10 | 9.97% | 0.6928 / 0.4885 | 0.0156 / 0.0101 | 0.0671 / 0.0449 |
| fg:Car | 0.18% | 0.6271 / 0.3794 | 0.1890 / 0.0983 | 0.1229 / 0.0556 |
| fg:Cyclist | 0.09% | 0.5838 / 0.2806 | 0.1896 / 0.0836 | 0.1407 / 0.0530 |

| 对比 | zero_z feat | zero_z cls | shuffle_z feat | shuffle_z cls | prior_only feat | prior_only cls |
|---|---:|---:|---:|---:|---:|---:|
| fg − bg | −0.0678 (1/12) | −0.0829 (1/12) | **+0.1396 (12/12)** | +0.0663 (12/12) | +0.0386 (11/12) | +0.0067 (9/12) |
| fg − bg_hi | −0.0770 (1/12) | −0.0940 (0/12) | **+0.1450 (12/12)** | +0.0688 (12/12) | +0.0422 (11/12) | +0.0076 (10/12) |
| fg − bg_top10 | −0.0758 (2/12) | −0.1384 (0/12) | **+0.1733 (12/12)** | +0.0836 (12/12) | +0.0600 (12/12) | +0.0097 (11/12) |

### 69.4 测量(1) 判读（机制级，按 69.1 预登记规则）

1. **DC-RSSM-1：样本特异内容没有区域集中**。`shuffle_z` 下 fg ≈ bg_hi
   （−0.0001，5/12），fg 甚至低于全背景均值（0.0527 < 0.0547）。按预登记规则，
   fg 依赖不高于能量匹配控制 ⇒ **瓶颈在读出头**，「前景-only 状态」不能自然改善 AP。
   注意 `zero_z` 下 fg 确实高于 bg_top10（+0.102，12/12），但那是**公共偏置**占
   主导的读数（zero 高分本身由公共向量贡献），不构成「z 在物体上被欠用」的签名。
2. **Baseline MAI：样本特异内容强烈集中在物体上**。`shuffle_z` 下
   fg 0.1888 vs bg_hi 0.0438（4.3×）vs bg_top10 0.0156（12.1×），**12/12 全胜**。
   即 MAI 的 z 里带着「换掉就毁掉物体」的内容，而 DC-RSSM-1 没有。
3. **两代际的差异本身是机制级发现**：DC-RSSM 代际（确定性+随机 innovation 后验）
   把 MAI 具有的「目标特异」内容平均掉了。这与 §51.4.3.5 在**重建损失**上看到
   的签名（zero_z 使 recon MSE 暴涨 +1385~1470%，shuffle_z 几乎无感 −0.09~+9.07%）
   在本节是**同一签名的第二个独立读出**——这次在检测头 logits 上、且按区域分解。
4. **`prior_only` 的分裂读数支持同一结论**：DC-RSSM-1 的 `prior_only` 对 fg 的
   伤害 12/12 稳定高于 bg_hi（+0.0336/+0.0195）——说明 fg 处 z 确实在起作用；
   但 `shuffle_z` 无 fg 集中——说明这个作用**不区分样本**。两者合起来正是
   「后验在物体上提供的是一个序列无关模板」的签名。
5. `measured`（Doppler 占据）在 DC-RSSM-1 上依赖最高（shuffle_z 0.0621 vs
   unmeasured 0.0546），但差距小；`dyn` 支撑过薄（<50 格）已按 §69.2 抑制，不作为
   结论依据（Ped 动态证据 0.145 的已知缺陷）。
6. **对「前景-only RSSM」的裁决：按预登记规则不立项**（判读规则 1 成立）。
   后续若仍想做前景相关改造，必须换机制假设（例如「目标侧 reform」而非「状态侧
   裁剪」）。

### 69.5 测量(2) 工具校验锚：扩展后的探针必须复现 §51/§52 表

同 checkpoint（`lowdim_z_future_consistent_3x2x2_24e_seed0/epoch_12.pth`）、同
样本数（16 条验证序列，`--start-index 3 --limit 16`）：

| 目标设计 | 本轮实测 共享/特异 | §51/52 记载 | 偏差 |
|---|---:|---:|---:|
| `pool16` + 全局标准化（现役） | 92.9% / 7.1% | 92.1% / 7.9% | +0.8 |
| `pool16` + 逐 cell 标准化 | 97.1% / 2.9% | 96.8% / 3.2% | +0.3 |
| `pool32` + 全局标准化 | 89.7% / 10.3% | 88.8% / 11.2% | +0.9 |
| 全分辨率 + 全局标准化 | 83.7% / 16.3% | 81.4% / 18.6% | +2.3 |
| `pool16` + 逐通道标准化 | 88.9% / 11.1% | 89.1% / 10.9% | −0.2 |

五个设计的**总方差**与记载一致到三位小数（1.000 / 0.205 vs 0.203 / 1.000 / 1.000 /
0.996），共享-特异占比偏差 0.2~2.3 个百分点，方向与幅度符合「样本窗不同」的解释
（本轮 window 为 [3..18]）。**判为复现成立**，扩展后的工具可用于横向比较。

### 69.6 测量(2) 结果：候选重建目标的方差分解

判定量：`共享模板占比` = 一个**样本无关**的模板能解释的方差份额；`样本特异占比`
= 必须逐样本不同才能解释的份额。现役目标 `pool16_globalstd` 的共享占比就是
「常向量能骗过重建损失」的程度。

| 目标设计 | lowdim ep12（锚） | DC-RSSM-1 ep16 | DC-RSSM-2 ep4 |
|---|---:|---:|---:|
| `pool16_globalstd`（**现役**） | 92.9 / 7.1 | **95.5 / 4.5** | 91.2 / 8.8 |
| `pool16_percellstd` | 97.1 / 2.9 | 97.5 / 2.5 | — |
| `pool32_globalstd` | 89.7 / 10.3 | 92.6 / 7.4 | — |
| `fullres_globalstd` | 83.7 / 16.3 | 84.4 / 15.6 | 68.9 / 31.1 |
| `pool16_channelstd` | 88.9 / 11.1 | 89.0 / 11.0 | 82.7 / 17.3 |
| `diff_globalstd`（帧差分） | 1.6 / **98.4** | 2.2 / **97.8** | 2.1 / **97.9** |
| `diff_warped_a1`（L1 解析全补偿） | n/a（无 ego warp） | n/a | 31.2 / **68.8** |
| `diff_warped_learned`（学习到的 warp 尺度） | n/a | n/a | 2.6 / 97.4 |

区域限制分解（同一次运行内，掩码限制到区域后重算）：

| 设计 | 区域 | DC-RSSM-1 ep16 特异% | DC-RSSM-2 ep4 特异% | lowdim ep12 特异% |
|---|---|---:|---:|---:|
| `pool16_globalstd` | fg | 11.8（格点占比 2.25%） | 16.5 | 13.9 |
| `pool16_globalstd` | bg_hi | 4.0 | 8.7 | 7.6 |
| `pool16_globalstd` | measured | 8.2 | 14.6 | — |
| `pool16_globalstd` | unmeasured | 4.5 | 8.8 | — |
| `fullres_globalstd` | fg | 31.7（格点占比 0.26%） | 36.2 | 29.9 |
| `fullres_globalstd` | bg_hi | 11.6 | 31.4 | 14.0 |
| `diff_globalstd` | fg | 96.0 | 94.2 | 94.1 |
| `diff_globalstd` | measured | 61.8 | 67.4 | — |
| `diff_warped_a1` | fg | n/a | 78.0 | n/a |
| `diff_warped_a1` | measured | n/a | 54.4 | n/a |
| `pool16_fgstd`（仅前景标准化） | fg | 12.1 | 16.5 | 14.1 |

L1 对齐差分的目标能量（同一全局标准化下，可横向比）：`diff_globalstd` 的
fg 总能量 3.89 → `diff_warped_a1` 1.95（**解析自车全补偿去掉约一半前景差分能量**）；
全局 total 均为 1.000（各自标准化）。解析位移在样本窗内为 dx≈0.93 格
（v_ego≈2.97 m/s，0.32 m/格），而**学习到的** warp 在 ep4 只有
`warp_scale=0.0774 × dt=0.1696` → 约 **0.12 格**（解析需 0.93 格）。

### 69.7 测量(2) 判读（机制级）

1. **现役重建目标的模板化在现役代际更严重**：`pool16_globalstd` 在 DC-RSSM-1 ep16
   上共享占比 **95.5%**（特异仅 4.5%），比锚点 92.9% 更极端。即「z 只需携带一个公共
   向量」这一结论在最新代际依然成立、且更强——与测量(1) 的「DC 代际丢掉了目标特异
   内容」互为印证。
2. **目标侧 headroom 极大**：换成**帧差分**目标，样本特异占比从 4.5% → **97.8%**
   （**21×**），在三个 checkpoint 上稳定（98.4 / 97.8 / 97.9）。这是本轮最可执行的
   机制改造候选：不是裁状态，而是**换目标**。
3. **L1 自车对齐的作用被量出来了**：解析全补偿后仍保有 **68.8%** 样本特异（仍是
   现役目标的 15×），同时去掉约一半前景差分能量。也就是说：
   - 「差分目标模板化」的修复不依赖自车对齐；
   - 自车对齐额外做的是**去掉由自车运动产生的那部分差分**（measured 区特异占比
     67.4 → 54.4 最能说明：Doppler 测量格的差分更被自车运动主导）；
   - 残余的 68.8% 是「自车补偿后仍在动」的成分，正是物体运动/外观变化——理论上
     不可由 h_t 的平滑外推单独解释。
4. **DC-RSSM-2 的 L1 当前几乎不动**：ep4 的学习尺度只有解析所需的 ~1/8
   （0.12 格 vs 0.93 格），所以 `diff_warped_learned` ≈ 未对齐（97.4% vs 97.9%
   特异）。这是 DC-RSSM-2 ep16 判读时**必须查的一条**：若 `warp_scale` 到 ep16
   仍近零，则该 run 的 L1 实际上是「名存实亡」，其 AP 结果不能作为 L1 机制的
   证据（正负都不行）。相关读数路径：`probe_recon_target.py` 的
   `learned_warp_scale_dt` 与训练日志的 `stat_ego_*`/warp 统计。
5. **未测量的轴必须显式区分**：本节测的是**跨样本**方差分解（常向量能否满足损失），
   §51 的「当前帧 vs 上一帧 MSE 0.0407 / 跨样本 0.1691」测的是**时间可预测性**
   （h_t 平滑外推能否满足损失）。两者是不同轴：差分目标跨样本极特异，但其中
   自车运动那一半是**时间上可外推**的——这正是 L1 对齐那一步在机制上要剥离的东西。
   要判定「差分目标是否真的把 z 锚到当前观测」，还需要在训练后重跑
   §51 的时间预测性探针（`tools/probe_recon_anchor.py` 口径）。
6. **检查点混淆提醒**：目标是由**该 checkpoint 自身 backbone 特征**导出的，所以
   跨 checkpoint 的比较（尤其 DC-RSSM-2 ep4 的 `fullres` 特异 31.1%）混入了
   「特征提取器尚未收敛」的因素。代际结论只用同代际内、同窗、同 S 的比较；
   跨代际只用于「模板化没有缓解」这一方向性判断。

### 69.8 对主线的影响（决策与后续）

1. **「前景-only RSSM」不立项**：按 69.1 预登记规则 1，fg 依赖不高于能量匹配背景
   控制（DC-RSSM-1 shuffle_z fg−bg_hi = −0.0001，5/12；fg 甚至低于全背景均值）。
   该方向的技术风险在状态侧，而瓶颈在读出头。
2. **重建目标改革升为下一步首要候选**（差分 / L1 对齐差分目标），机制依据：
   21× 的样本特异占比提升 + 现役目标在最新代际反而更模板化。任何落地必须先写
   预登记（目标形式、生效层、AP 门、机制门、止损线），不在未验证结构上叠超参搜索。
3. **DC-RSSM-2 ep16 判读新增必查项**：`warp_scale` 是否增长（见 69.7.4）。若近零
   则 L1 的 A/B 不成立，需在判定中如实标注，不能把 AP 归因给 L1。
4. **若 DC-RSSM-2 AP 门 FAIL**，机制归因已有两条现成证据：目标共享化（95.5%/4.5%）
   与「DC 代际丢失目标特异内容」（shuffle_z fg−bg_hi 从 MAI 的 +0.145 掉到
   −0.0001）。这两条都可直接进论文 Analysis，属「负结果的机制级归因」。

### 69.9 过程记录（工具缺陷与修复，全部为探针侧）

1. **网格转置（真 bug，靠冒烟抓到）**：首版冒烟里 Doppler 区全 NaN，根因是把
   `pts_feats[-2:]` 当作证据栅格，而它经 `permute(0,1,3,2)` 后是 (y,x)。修复：
   `grid_xy_from_canvas` 从原始 canvas 推栅格 + 硬断言捕获特征是转置 + 区域读数
   内统一转置一次；并补**取向自检**（5.14 vs 0.00，12/12）把「假设」变成「实测」。
   附带发现：把掩码与热图同时转置等价于恒等操作，不能当自检。
2. **崩溃 1（测量 2）**：`standardize_masked` 收到 5 维 `(S,1,C,16,16)` 的池化张量
   → 展开尺寸不匹配。修复：池化前 `f[0]` 去掉样本维，使目标统一为 `(S,C,H,W)`；
   `decompose` 的掩码分支同时对齐两套键名（`specific_frac` ↔ `sample_specific_frac`）。
3. **崩溃 2（测量 2）**：`to_grid` 收到带前导单例维的掩码，逐样本堆叠后变成
   `(S,1,16,16)` 与 5 维目标广播失败。修复：`to_grid` 显式压掉单例维、多维即报错
   （loud failure 取代静默错配）；`masked_variance_stats` 增加 `(S,1,H,W)` 掩码容错。
4. **崩溃 3（测量 2）**：`warp_map_by_cells` 里解析位移在 CUDA、缓存特征在 CPU
   → 设备不一致。修复：位移在函数内强制对齐到特征设备/类型（位移本身与设备无关）。
5. **回归自检**：全 1 掩码 ≡ 无掩码拆分（0.7584 vs 0.7584）；i.i.d. 噪声 S=4 实测
   73.8% ≈ 解析极限 (S−1)/S = 75%。W13 19/19 绿（`python
   me_rssm/sanity/test_bev_region_utils.py`；本环境 `pytest` 因 conftest→hydra→
   `omegaconf.vendor` 缺失不可用，与本次改动无关）。
6. **运行与产物**（GPU4，队列脚本 `me_rssm/sanity/run_gpu4_measure12_queue.sh`；
   单次 30~45 s）：
   - `gpu4_probes/m2_anchor_lowdim_ep12.log` → `lowdim_z_future_consistent_3x2x2_24e_seed0/recon_target_anchor_ep12_w3_16.json`
   - `gpu4_probes/m2_dcrssm1_ep16.log` → `fgfull_N4_no2d_igdr_dcrssm1_2x4_24e_seed0/recon_target_ep16_w3_16.json`
   - `gpu4_probes/m1_baseline_ep14.log` → `fgfull_N4_no2d_igdr_2x4_24e_seed0/zutil_region_ep14_w3_12.json`
   - `gpu4_probes/m2_dcrssm2_ep4.log` → `fgfull_N4_no2d_igdr_dcrssm2_2x4_24e_seed0/recon_target_ep4_w3_16.json`
   - 测量(1) DC-RSSM-1 ep16 的产物：`zutil_region_ep16_w3_12.json` /
     `probe_region_split_ep16.log`（该 run 的 work_dir）。

### 69.10 运行中证据（DC-RSSM-2 ep1-6 日志，采样自 `stat_ego_warp_scale`）：L1 warp 已平台化

上节 69.7.4 的「L1 目前几乎不动」在训练日志里是**逐 epoch 可追踪**的，且到 ep6
仍在平台：

| epoch | warp_scale 均值 | 区间 [min, max] | 有效位移（×dt 0.164 × v 1.8 m/s ÷ 0.32 m） |
|---:|---:|---|---:|
| 1 | 0.0297 | [0.0065, 0.0416] | 0.027 格 |
| 2 | 0.0284 | [0.0170, 0.0343] | 0.026 格 |
| 3 | 0.0434 | [0.0274, 0.0724] | 0.040 格 |
| 4 | 0.0808 | [0.0749, 0.0865] | 0.074 格 |
| 5 | 0.0823 | [0.0767, 0.0874] | 0.075 格 |
| 6（进行中） | 0.0815 | [0.0726, 0.0900] | 0.074 格 |

- ep1-4 增长 2.7 倍，随后 ep4/5/6 均值 0.0808/0.0823/0.0815，**三点平台**
  （区间内波动 ±0.002）。相对解析全补偿（val 样本窗 v≈2.97 m/s → 0.93 格）只有
  **约 8%**。
- **机制假设（写在这里，ep16 用日志验证）**：L1 warp 是**恒等锚定**的（零位移处
  修正项逐位为 0），起点即恒等；而下游 MCI 的可形变对齐本身就能吃运动残差，所以
  把 `warp_scale` 从 0.08 推到 1.0 并不带来额外损失下降 → 梯度压力弱。若该假设成立，
  「L1 是否有效」不能只看它是否在梯度里，而要看它是否**被单独监督/初始化到有效尺度**
  （例如解析尺度初始化或对 `warp_scale` 加显式对齐监督），否则它只是个几乎不透传的
  近恒等旁路。
- **ep16 必查**：`stat_ego_warp_scale` 是否离开 0.08 平台。仍平台 ⇒ 本 run 不能作为
  L1 机制的正/负证据，且「L1 惰性」本身就是一个可写进 Analysis 的机制观察
  （恒等锚定 + 下游可形变对齐 = 学到近恒等）。

**更新（2026-10-10 02:2x UTC，ep15 训练中；上面的「平台」判读被后续数据修正）**：
平台在 ep7 起被打破，且 `dt` 也在长，乘积（已施加位移）增长快于单看 `warp_scale`：

| epoch | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15（it550） |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `warp_scale` | 0.0815 | 0.0882 | 0.1001 | 0.1049 | 0.1111 | 0.1159 | 0.1229 | 0.1244 | 0.1289 | 0.1311 |
| `dt` | 0.164 | — | — | — | — | — | — | — | — | 0.2154 |
| 已施加/需求位移 | ~13% | — | — | — | — | — | — | — | ~21% | **28.2%** |

（需求 = `speed × 0.1 s`（视觉帧率）；ep15 读数：0.1311×0.2154×1.913 = **54.0 mm**
vs 191.3 mm，E8b 绝对线 1e-3 m 以 **54 倍**余量 PASS；static_frac 7.3e-3、trim 0.764、
solved 1.000、τ_r 0.220，均在预登记带内。）
⇒ ep16 的机制判读**不依赖** learned scale 是否到达 1.0：预登记的 toggle 探针本来就有
**强制 1.0** 的幅值对照臂（§68.9i），learned 臂只用于回答「损失是否自己把标量推到有效
尺度」（当前答案：推，但只到约 28%）。这条修正不影响任何门控判据。

**口径厘清（ep16 终判后补，两个分母都对但含义不同）**：上表的「28.2%」用**名义帧间隔
0.1 s** 作分母（`speed × 0.1 s`）。P5 探针用**模块自身满程**作分母（`1.0 × 学到的 dt
0.2132 s × |v_ego|` = 0.4489 m/frame），ep16 读数因此是 **12.93%**（0.058/0.449）。
两者不是矛盾：`dt` 本身是**可学标量**且已收敛到名义值的 2.1 倍，所以「相对名义补偿」
比「相对模块满程」大一倍多。终判与论文引用一律用 P5 的分母（模块满程），并在
§68.9j 记录：**强制推到满程只赢 5/10**，即收益局部于学到的约 13% 工作点。


### 69.11 最近实验与候选优先级复核（2026-10-10；无新训练）

- 详细源码/产物证据与分阶段建议：`docs/rssm_next_step_audit_20261010.md`。
  只更新文档，未修改模型、未启动训练、未使用 GPU。
- 统一口径上下文：DC-RSSM-2 区间均值 ep12–16（n=5）**38.5345**；
  全轮峰值 **41.1838 @ ep8**，best saved **41.1838 @ ep8**。完整曲线、
  逐类窗口均值与过程继承 §68.9j；P5 JSON 再算 learned 8/10、forced1 5/10。
- 实际训练目标是 fullres 原始 BEV MSE，并无 pool16/全局标准化，目标亦未在
  `R4Det.py:930` detach。95.5% 模板不是该真实训练目标的直接测量；
  同分辨率标准化候选特异 15.5699%→97.7587%（6.28×），raw 尚待诊断。
- 区域 baseline snapshot/probe config 对应标准 `MotionAlignedRSSMFusion`，
  不是 MAI。fg 不集中支持停止简单裁状态，不唯一定位读出，更不能外推
  没有出现在区域表中的 Ped/Truck。样本特异能量不是任务信息或 z 排他职责。
- **建议**：遵守 DC-RSSM 止损；回到标准 no2d_igdr，先目标/梯度/历史契约
  审计与新实验正式预登记，再对 matched fullres current/difference 做有限
  seed0 试验。训练接线/真实 CE 与检测条件增益决定是否继续到 ep16。
  目标改革不通则转 A；按缺测类别证据决定 C；B/D 后置。不同时改目标、
  标准化、detach、posterior、readout、L1，不把本建议当已批准训练。
- 局部纠错已回填：DC1 v2−0.5 门实际 PASS +0.3383（总 FAIL 不变）；
  v2 Ped 固定窗口差 −7.8295；“每次 Cyc 大跌”不成立（v2 近零 +0.0652）。


## 70. 真实重建目标契约审计与目标改革预登记（2026-10-10，只读；无新训练）

本节按 `docs/rssm_next_step_audit_20261010.md`（用户审计，commit `bab69ac`）的
「阶段 0.1」要求，把**标准主线实际优化的重建目标**从源码与日志里钉死。
承接 §69.11 的纠正：§69 的 `pool16_globalstd` 是探针候选构造，不是训练目标。
完整预登记草案见 `docs/rssm_target_reform_prereg_20261010.md`（**未批准**）。

### 70.1 契约事实（源码行号可核）

被审计对象 `fgfull_N4_no2d_igdr_2x4_24e_seed0`（`MotionAlignedRSSMFusion`）：

| 项 | 事实 | 证据 |
|---|---|---|
| 目标张量 | `bev_feats`（`cross_attention` 输出、`permute` 后），即 RSSM **之前**的当前帧特征 | `R4Det.py:872-875`、`919` |
| 分辨率 / 标准化 | 全分辨率（248,216），**不 pool、不标准化**；本家族无 `_normalized_recon_target`（该函数只在 `LowDimFutureConsistentLatentFusion`，`rssm_fusion.py:1438-1443`） | 全类族 grep |
| detach | **目标在计算图内**（无 detach） | `R4Det.py:930`（当前帧）、`909`（历史帧） |
| 权重 | **1.0**（`losses['loss_rssm_recon']` 直接登记；R4Det 无 `loss_weights`） | `R4Det.py:1504` |
| 历史监督 | 窗口内**最后 1 帧历史 + 当前帧**计 recon、等权平均（`rssm_bptt_steps` 默认 `1`、config 未覆盖 ⇒ `burn_in=2`）；前 2 帧仅 burn-in、无 loss | `R4Det.py:188, 1403-1407, 1417-1428` → `1493-1504`（另见 §47 BPTT 表） |
| burn-in | **2 帧**：`rssm_bptt_steps` 未在 config 覆盖，取类签名默认 `1` ⇒ `grad_steps=1`、`burn_in=2`；只有 t=2（第 3 帧）带 recon loss，t=0/1 仅前向、无 loss | `R4Det.py:188, 1403-1407, 1417-1428`（另见 §47 BPTT 表） |
| decoder 输入 | `cat([h_t,z_t])`（h 旁路存在）；输出 `output_proj(z_t)+feat` | `rssm_fusion.py:479-481`、`493-495` |
| 状态重置 | 无效历史帧 `reset_for_samples(~valid_t)`；基类只清 `h/z` | `R4Det.py:1415,1430`；`rssm_fusion.py:299-312` |
| 配置 | `latent_dim=256, hidden_dim=128, in/out=256, kl_scale=1.0, free_nats=1.0, min_std=0.1, init_std=0.2, seq_len=4` | 运行目录 config 211-228 行 |

### 70.2 recon 项在真实目标上只占 0.18%（逐 epoch 实测）

train 记录均值（`*.log.json`，同 run）：

| epoch | 1 | 2 | 4 | 8 | 11 | 12 | 16 | 20 | 24 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| recon MSE | 0.08068 | 0.01136 | 0.00721 | 0.00522 | 0.00414 | 0.00391 | 0.00322 | 0.00277 | 0.00263 |
| KL | 0.0000 | 0.1407 | 0.3125 | 0.7066 | 1.0027 | 1.0024 | 1.0012 | 1.0005 | 1.0003 |
| clamped 比例 | 0.5438 | 0.9584 | 0.9889 | 0.9960 | 0.9980 | 0.9982 | 0.9981 | 0.9984 | 0.9985 |
| μ_q−μ_p ² | 1.526958 | 0.342130 | 0.089340 | 0.026834 | 0.012670 | 0.012034 | 0.009662 | 0.007744 | 0.007206 |

ep24 iter350 总损失分解：`loss=1.45374`，`loss_rssm_kl=1.00025`（**68.81%**）、
`loss_rssm_recon=0.00256`（**0.18%**）、`bev_mask_dc_loss_former=0.17140`（11.79%）、
`loss_bbox=0.12324`、`loss_cls=0.05934`、`loss_dir=0.00393`（0.27%）。

**三条同时在真实目标上成立**：① recon 项自 ep4 起就低于总损失的 0.5%、末期
0.18%（锚定压力≈0）；② raw KL 收敛到 0.597 < `free_nats=1.0`，clamped 0.9985，
**KL 项恒为常数 1.0**（无梯度方向信息）；③ `μ_q−μ_p`² 单调降到 ep24 的 0.0072
（−99.5%），**后验塌到先验**。即「z 不承担观测信息」在真实契约下不是探针推断，
而是损失本身只给了它 0.18% 的说话权——这解释了 MCI/DC/L1 各代改状态侧与读出侧
而 AP 不动：**目标项太小，改哪一层都推不动它**。
> **§70.6 补正（同日）**：「目标项太小」指**损失尺度**，不是「目标没有内容」。
> 实测真实目标 56.5% 样本特异，所以「换个更有信息的目标」这条路的前提并不成立；
> 剩余嫌疑是 recon 的权重/尺度相对 KL 常数项、以及 decoder 是否根本不需要 z
> （必须用 z 条件增益实测，不由推断）。

### 70.3 数据契约：窗口内各帧共享同一几何增强（差分目标的前提）

`TJ4D_dataset.py:333-335` 取一个 `seed` 写入 `np.random`/`random`，
`:369-370` 在准备**每一帧**前把 RNG 重置回同一 `seed` ⇒ 窗口内各帧走同一条
随机序列，**同一套几何增强**。`frame_valid` 由 `image_idx` 连续性决定
（`341-350`），不连续历史帧替换为占位符并标无效（`377-378`）——差分监督必须
跟随该有效位。差分目标「被增强差异主导」的头号风险实测**不成立**。

### 70.4 低维家族契约不同（§53 历史失败的适用边界）

低维家族是另一套契约：目标 `pool → 逐样本全局标准化 → detach`
（`rssm_fusion.py:1438-1443`），decoder 输入可只吃 `correction`，且
`latent_loss_is_primary=True`（`:1192`）走 `loss_rssm_latent`。
故 §53.4/53.5 的「帧间残差目标失败」是在 **pool+标准化+detach 目标 +
correction-only decoder + latent 损失通路**下得到，不能外推到标准家族的
「全分辨率、原始、未 detach」契约；标准家族此前**从未**试过差分目标。

### 70.5 探针扩展与 CPU 自检（测量侧）

`tools/probe_recon_target.py` 新增两个**未标准化全分辨率**设计
（`raw_fullres_current` = 真实目标、`raw_fullres_diff` = 候选），新增
`--stride`/`--require-classes N` 与 `class_coverage` 输出（2026-10-09 的 12 样本
窗只有 Car/Cyclist，Ped/Truck 根本没进表）。核心行打印
`rms / shared rms / spec rms` 与 `diff/current` 能量比。

CPU 自检（合成张量，无 GPU、无 checkpoint）：current（共享+噪声）shared 100%；
差分（i.i.d. 噪声）specific **92.91%** vs 解析极限 (S−1)/S = **93.75%**；
**常数差分 specific 恰为 0**（该拆分不天然偏向差分）；全 1 掩码 ≡ 无掩码；
选择器在「每类 ≥3 GT」下正确跳过无 Ped 的帧（3,4,5,6 → 7,14,21,28），
不可满足时抛显式 `ValueError`。

**该自检同时暴露了口径陷阱**：i.i.d. 噪声在 S=16 上拿 92.91%，解析地板
(S−1)/S = 93.75%。也就是说「特异占比高」本身**不是信息量的证据**——白噪声
在该统计量上拿满分。任何用特异占比论证「某个目标更 informative」的说法，
都必须先把实测值与该 S 下的地板并列，否则无意义。（蒙特卡洛复核：
S=2 → 0.500、S=4 → 0.750、S=8 → 0.875、S=16 → 0.938。）

### 70.6 GPU 只读复测（2026-10-10，物理 GPU 5；不训练、不写 checkpoint）

两次运行，checkpoint = 基线 best saved `epoch_14.pth`：

| 窗口 | 取样 | S | 产物 |
|---|---|---:|---|
| W0（草案原命令） | `--start-index 3 --stride 17 --require-classes 2` | **2** | `recon_target_raw_ep12_w3_24.json` |
| W1（主口径） | `--start-index 3 --limit 16 --stride 127` | 16 | `recon_target_raw_ep14_dispersed16.json` |

**W0 作废**，两个独立原因：(i) 全 val 里「四类同时 ≥2 GT」的帧只有 35 个且全部
落在同一场景块（≈1302–1360，用 `TJ4D_infos_val.pkl` 直接清点），`--stride 17`
几乎每步跳出该块 ⇒ 只选到 2 帧；「四类同帧」与「分散取样」在这份 val 上互斥。
(ii) S=2 时纯噪声地板 = 50.0%，而 W0 的 diff 特异 = **49.6%**，与「两个白噪声
样本」不可区分；W0 的 current 特异仅 9.9%，是同场景相邻帧本来就相似造成的
**场景连续性**，不是共享模板。

**W1（S=16，跨全 val 均匀取样，纯噪声地板 93.75%）**：

| 设计 | rms | shared rms | spec rms | 特异占比 | 相对地板 |
|---|---:|---:|---:|---:|---:|
| `raw_fullres_current`（真实目标） | 0.58415 | 0.38519 | 0.43915 | **56.5%** | −37.2 pt |
| `raw_fullres_diff`（候选） | 0.11627 | 0.02915 | 0.11256 | **93.7%** | **−0.0 pt** |

派生：常数预测地板 current **1.484e-1** / diff **8.50e-4**（175×）；
目标总能量 diff/current = **0.0396**；样本特异能量 diff/current = **0.0657**；
现役 recon MSE（ep12–16 均值 3.62e-3）/ current 地板 = **0.024**。
同一窗口下 `pool16_globalstd` 特异 **32.9%**（§69 场景窗是 5.7%）、
`fullres_globalstd` **59.8%**（场景窗 13.5%）。

对象对齐度（特异能量份额 / 面积份额）：current fg **0.04×**、bg_hi 1.99×；
diff fg **0.21×**、bg_hi 2.55×（fg 面积 0.90%）。探针区域表仍无逐类行，
W1 窗内 Ped 仅 4 个实例，不足以逐类判读。

**结论（三条，全部指向收缩目标改革这条线）**：

1. 「现役重建目标 ≈ 95% 共享模板」**在真实目标上不成立**：56.5% 是样本特异的；
   §69 的 95.5% / 21× / 6.28× 主要由**同场景连续帧窗口**与 pool16 候选构造产生。
2. 「差分目标更样本特异」**没有区分力**：93.7% 与纯噪声地板 93.75% 重合，
   白噪声拿同样的分数。
3. 按现役契约（权重 1.0、不缩放）换差分会把 recon 项目标能量压到 **4.0%**，
   而该项现在只占总损失 0.18% ⇒ **削弱**对 z 的约束，与立项目标相反；
   补尺度即引入第二个变量。

⇒ 决策记录见 `docs/rssm_target_reform_prereg_20261010.md` §3A：不投原样 seed0 ep6；
若要继续检验目标假设，必须改成「差分 + 与 current 同特异能量的标定尺度」的
**配对**设计并用对象对齐度/条件增益判读，需用户批准；否则按排序转候选 A，
差分作为负结果归档。

### 70.7 本节的边界

只读审计 + 探针改造 + 预登记草案 + **GPU 只读复测（GPU5，两次，未训练、
未写 checkpoint、未改模型）**。目标改革的数值门（非劣界等）尚未批准，
草案明确标注为建议值；本节不含任何 AP/性能宣称，也不构成任何训练授权。
DC-RSSM 方向维持 §68.9i 止损结论，本文不重启该方向。

---

## 71. 候选 A 实现与 S0 机制门（z 专属语义监督 + 零初始化分类残差读出，2026-10-10）

### 71.0 状态

本节是**实现 + 只读探针 + 450 iter 机制冒烟**的记录：
`IterBasedRunner` 450 iter、3 卡 DDP（GPU 5/6/7）、seed 0、deterministic、
**无 val**，因此**不填区间均值 / 全轮峰值 / best saved，也不做任何 AP 宣称**。
预登记门（§5 的 S2/S3 具体数值）**尚未获用户批准**，本节不含任何正式训练授权。

### 71.1 动机（承接 §70.6 的收缩结论）

§70.6 否掉了目标改革的承重前提，§69 把嫌疑收窄到「读出」这一侧，但要留出
「编码 / 后验 / 监督也可能参与」的余地。候选 A 因此**只动监督与读出**两处，
其余全部字节级沿用标准主线 `..._no2d_igdr`：

| 变量 | 含义 | 消融开关 |
|---|---|---|
| `sem_head`（z 专属 CentreNet 热力图头） | 对**当前帧** GT 中心做 focal 监督 → `loss_rssm_sem`；输入**只有 `z_t`**，`h_t` 不参与 | `sem_supervision_weight=0`（= C2 只读出） |
| `z_cls_logits`（零初始化 1×1 conv） | `z_t` → 4 类 logits，按锚点类别映射广播后**加**到 `conv_cls`；几何 `conv_reg`、`conv_dir_cls` 不受影响 | `cls_residual=False`（= C1 只监督） |

重建目标、posterior/prior、KL、free-nats、对齐、backbone、检测头几何、
优化器、日程、数据管线、seed 协议**全部未改**；本 run 不含 evidence field、
不含 L1 ego-warp、不含差分目标。控制臂 `sem_head_input='h'|'feat'`（C3 / M2）。

### 71.2 零初始化契约（S0-b 的依据）

`init_weights()` 在通用初始化扫过之后再断言一次 `z_cls_logits` 权重/偏置为 0
（`semantic_readout_rssm.py`），因此：

* 从 `checkpoints/pretrained_tj4d.pth` 热启动、或从基线 `epoch_14.pth`
  加载时，**分类残差恒为 0**，检测头逐位等价基线；
* 构造期保存/恢复 RNG 状态，参数初始化流不因新模块改变（单测 T1）。

### 71.3 S0-b 身份闸门（只读，基线 epoch_14.pth）

同一 val 序列（`val_index=3`，4 帧 rollout）分别跑 A 类模型与基线模型，
逐元素比较检测头输出：

| 读数 | 值 |
|---|---|
| `max_abs_cls_diff` | **0.0** |
| `max_abs_reg_diff` | **0.0** |
| 残差是否发出 | 是（shape 4 类，网格 **(248, 216)**，与 `pts_feats` 同约定） |
| 残差 absmax | **0.0**（零初始化 ⇒ 恒 0） |
| 基线 checkpoint 加载差集 | **空**（`not_in_checkpoint` = `not_in_model` = 空） |
| A 类额外缺失键 | 恰好 6 个：`sem_head.{0,2}.{weight,bias}`、`z_cls_logits.{weight,bias}` |
| 其余缺失键集合一致 | 是 |

⇒ 新钩子（fusion → detector → `conv_cls`，含 (216,248)→(248,216) 转置与
无效样本置零）**在训练前不改变任何输出**，S0-b **PASS**。

### 71.4 S0-d 梯度压力闸门

`‖∂loss_rssm_sem/∂z_t‖ / ‖∂total/∂z_t‖ = 0.0729 / 0.0807 = 0.9024`，
即语义项在 z 上贡献 **90.2%** 的梯度压力（阈值 ≥5%），S0-d **PASS**。
对照 §70.2：主线的 recon 项只占总损失 0.18%，正是「监督项压不动 z」的先例。

### 71.5 S0-e 存活 / 分离闸门（含一处判据修正）

首轮探针把 S0-e 读成 FAIL（`∂total/∂W_cls == 0`）。根因**不是读出坏了**：

* train split 的**第一个 batch** 恰好是边界样本（最后一帧历史无效），
  而 `extract_feat` 对历史无效样本**设计性**回退到未融合特征并把分类残差
  **置零**，所以那个 batch 上梯度必然为 0；
* 该掩码只影响 **0.75%** 的样本（400 样本抽样：`last_history_valid = 397/400 = 0.9925`）。

探针修正（模型未改）：选取**最后一帧历史有效**的 batch（扫描上限
`--max-probe-batches`，兜底 `--train-index`）、剥掉 `collate` 多包的一层
batch 维、并同时记录所用 batch 的逐样本掩码与 400 样本覆盖率。
判据本身也修正一次：零初始化下**前向残差恒为 0 正是 S0-b 的身份契约**，
不能兼作存活证据，因此 S0-e 改为两条独立读数：

| 读数 | 值 |
|---|---|
| `∂loss_cls/∂W_cls` | **0.871**（> 0，残差确实进入分类 logits，而非只进 log dict） |
| 1e-3 扰动后残差到达 head 的 absmax | **0.379**（网格 (248,216)，未被置零） |
| `∂loss_rssm_sem/∂W_cls` | **0.0**（监督与读出分离 ⇒ C1/C2 消融可比） |

S0-e **PASS**（存活 + 分离双条）。所用 batch offset=2，两样本末位历史均有效。

### 71.6 450 iter 机制冒烟（seed0，3 卡，无 val）

配置 `me_rssm/sanity/semreadout_smoke_450iter.py`（继承 A 类配置，
`IterBasedRunner` 450 iter、cosine、`evaluation.interval=4500` ⇒ 无 val）。
运行事实：2026-10-10 14:12:16 UTC 发射、14:24:16 UTC 到 iter 450，
`iter_450.pth` 已落盘，无 NaN、无 OOM，显存 **17099 MiB**、**1.53–1.60 s/iter**。

| iter | loss_rssm_sem | stat_sem_hit1 | stat_sem_thr50 | stat_cls_res_rms | loss_rssm_recon | loss_rssm_kl | stat_mu_diff_sq | stat_clamped_ratio | loss_cls | loss_bbox | loss | grad_norm |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 21.4007 | 0.0033 | 0.0000 | 0.0689 | 1.0631 | 5.7136 | 1.0602 | 0.3254 | 1.0808 | 2.3194 | 32.9598 | 130.8469 |
| 100 | 2.9724 | 0.0561 | 0.0000 | 0.1742 | 0.6654 | 1.2792 | 0.7424 | 0.8128 | 0.8444 | 1.4902 | 8.1966 | 78.9121 |
| 150 | 2.7743 | 0.0519 | 0.0000 | 0.1423 | 0.5079 | 1.1409 | 0.4137 | 0.8783 | 0.6100 | 1.3780 | 7.2396 | 27.1744 |
| 200 | 2.6576 | 0.0956 | 0.0000 | 0.1392 | 0.4006 | 1.0951 | 0.2832 | 0.8833 | 0.5777 | 1.2723 | 6.7757 | 26.1232 |
| 250 | 2.5115 | 0.0883 | 0.0017 | 0.1887 | 0.3378 | 1.0962 | 0.2551 | 0.9084 | 0.5410 | 1.1844 | 6.4371 | 23.5786 |
| 300 | 2.4930 | 0.0854 | 0.0022 | 0.1735 | 0.2987 | 1.0728 | 0.2207 | 0.9122 | 0.5374 | 1.1424 | 6.2966 | 20.8793 |
| 350 | 2.3810 | 0.0764 | 0.0017 | 0.1739 | 0.2706 | 1.0608 | 0.2026 | 0.9138 | 0.5038 | 1.0974 | 6.0190 | 19.6289 |
| 400 | 2.3360 | 0.0898 | 0.0022 | 0.1807 | 0.2632 | 1.0675 | 0.1996 | 0.9148 | 0.4894 | 1.0460 | 5.9045 | 19.5043 |
| 450 | 2.2674 | 0.1198 | 0.0033 | 0.1838 | 0.2601 | 1.0461 | 0.1968 | 0.9235 | 0.4917 | 1.0648 | 5.8292 | 18.5481 |

末 5 点均值（iter 250–450，n=5）：`loss_rssm_sem` **2.3978**、
`stat_sem_hit1` **0.0919**、`stat_sem_thr50` **0.0022**、
`stat_cls_res_rms` **0.1801**。

**冒烟结论（仅机制，不含 AP）**：

1. **语义项确实下降且有梯度压力**：`loss_rssm_sem` 21.40 → 2.27（−89.4%，
   首点 → 末点），且 S0-d 实测该项独占 z 上 90.2% 的梯度压力；
   该项占总损失 **39.3%**（末 5 点均值），不是 §70.2 那种 0.18% 的装饰项。
2. **读出被激活**：`stat_cls_res_rms` 从 0（零初始化身份）升到 **0.184**，
   说明 `W_cls` 真的被迫移动；S0-e 已独立证明该残差进入 `conv_cls`。
3. **但热力图仍远未对齐**：`hit@1` 0.003 → **0.120**、`thr50` 仅 **0.0033**。
   450 iter 只在 ~1% 的日程内，这是**早期读数**，不足以判机制成立与否。
   预登记 §5 的 S1（ep6 时 `thr50 ≥ 0.30` 且 `hit@1 ≥ 0.60`）才是判据，
   **本节不据此下任何结论**。
4. **检测侧行为未被破坏**：末点 `loss_cls` 0.4917、`loss_bbox` 1.0648、
   `loss_iou` 0.0493、`loss_dir` 0.0848，与基线首个 epoch 的同 iter 区间
   （`loss_cls` 0.5200、`loss_bbox` 1.0380）同量级且同趋势；`stat_clamped_ratio`
   0.9235、`stat_mu_diff_sq` 0.1968 说明 posterior 仍在正常收敛。
5. **一处必须写清的差异（且与预登记无关）**：`KLScaleSchedulerHook` 只实现
   `before_train_epoch`，而 `IterBasedRunner` 的 `run()` 只调
   `before_run` / `before_epoch` / `after_epoch`（mmcv 1.x
   `iter_based_runner.py:126-140`），**该 hook 在 iter 模式下从不触发** ⇒
   `kl_scale` 全程停在配置值 **1.0**：早段 `loss_rssm_kl` 5.7136 与
   `stat_kl_raw_mean` 5.7291 同量级（差 0.3% 来自 loss 对最后历史帧+当前帧
   取平均、而 stat 只报当前帧），若 scale≠1 这两列会差一个常量倍数。
   正式训练是 epoch 模式，`before_train_epoch(epoch=0)` 让 `kl_scale = 0`
   并线性升到 ep10 才满，这正是基线首个 epoch `loss_rssm_kl = 0.0` 的来源，
   也解释了基线 epoch-1 的 `stat_mu_diff_sq` 只有 0.34 而本冒烟是 0.74—
   冒烟从第 0 步就施加满 KL，后验被更早压向先验，**recon 偏高
   （0.2601 vs 基线 0.0997）是这个更强约束的预期后果，不是新机制引入的退化**。
   **因此冒烟的 recon/KL 绝对量不能与正式训练逐点比较**；本节只把
   「有限、无 NaN、量级可解释」当作烟测判读。

### 71.7 回归测试

| 测试 | 结果 |
|---|---|
| `mmdet3d.models.fusion_layers.test_rssm_fusion` | **85 passed** |
| `me_rssm.sanity.test_semantic_readout_rssm`（T1–T12） | **11 passed** |
| `test_grad_paths` / `test_config_build` / `test_fgfull_config` / `test_mci_config_build` / `test_equivalence` | 全绿（`test_fgfull_config.py` 为脚本式，`ALL OK`） |

T1 断言「关态与新模块在场时 RNG/输出逐位一致」，T2 断言零初始化，
T3 断言语义损失对 `posterior_mu` 有梯度而对 `prior_mu` 无（z 专属），
T4/T5 钉住 H→x/W→y 与 `a*num_classes+anchor_class_mapping[a//num_rot]` 通道布局，
T6/T7 钉住网格转置与无效样本置零，T9 钉住 `cls_residual_gate` 可作配对开关。

### 71.8 边界与下一步

* 本节**没有** AP 读数，也**不**宣称候选 A 有效；S0-b/d/e 只证明接线正确、
  身份保持、压力与读出分离。
* 预登记 `docs/rssm_candidate_A_prereg_20261010.md` 的 **§5 S1/S2/S3 数值门
  仍需用户批准**；未批准前不启动 24e 正式训练。
* 若批准：seed0 跑到 ep6 按 S1 判读（`thr50 ≥ 0.30`、`hit@1 ≥ 0.60`、
  z/h 排他性、toggle、跨类 shuffle ≥ chance+20pt），通过才续到 ep16 看 S2
  AP 门（`≥ 40.00`，四类非劣界 Car ≥ 47.34 / Truck ≥ 29.22 / Ped ≥ 29.93 /
  Cyc ≥ 47.63，硬失败线 < 38.78），单 seed 通过才补三 seed。
* 消融臂（C1 只监督 / C2 只读出 / C3 `sem_head_input='h'|'feat'`）沿用同一
  10 行改动，不需要新代码。
* 未做也不做：不改重建目标、不动 KL/权重、不做超参扫描、不做 ep24 续训抢救。

### 71.9 ep6 证据链预检与判据口径写定（训练在飞期间，2026-10-10）

> **状态：in-flight。** 本节记录的是 ep6 对照跑**在飞期间**对"读数链"的
> 预检与两处口径写定，**不是结果**。截至本节写入：ep1/ep2 val 已产出
> （ep1 Overall_3D_moderate 26.8637），ep3 在训；M1–M4 与三窗反事实
> **尚未产出**，S1 判定未做。

#### 71.9.1 为什么要先查读数链

ep6 是候选 A 的止损点：`semA_ep6` 训练到 ep6 val + 落盘 `epoch_6.pth`
后，`semA_watch` 看门狗会中断训练并在 GPU5/6/7 上跑 M1–M4 门 + 三窗
反事实 + source 探针。这批探针此前只在 CPU 单测里验证过**打分函数**，
从未在真实 checkpoint 上跑通过一次；而任何一处抛错都会让**整包证据**
（含 6 小时训练的 M1/M2/M3/M4 四个读数）一次性作废。因此在训练仍在
ep2–ep3 时做了静态审查 + 可执行验证。

#### 71.9.2 三个会作废证据包的接线缺陷（均已修，均有单测）

| # | commit | 缺陷 | 后果 |
|---|---|---|---|
| 1 | `52fc69d` | M4 head-swap 把 `pts_feats` 搬到 CUDA，却把 donor residual 留在 CPU；`Anchor3DHead._expand_cls_logit_residual` 用 `ref.device` 造索引再 `residual.index_select(1, src)` | `RuntimeError: Expected all tensors to be on the same device`，M4 及整份 gate JSON 全丢 |
| 2 | `6c26bc3` | 看门狗 P1/P2 用的是 MCI/DC 专用探针；`probe_z_utilization_gated.py` 对非门控变体直接 `AssertionError: target model is not a gated variant` | 三窗反事实与 source 读数全丢（已在 450 iter smoke checkpoint 上实测复现） |
| 3 | `bc0bd9d` | `object_min_radius` 从 config 读出后未传入任何 `make_target`，四个调用点全部落到硬编码默认值 2 | 当前 config 恰好也是 2，所以**只是碰巧正确**；一旦该键改变，M1–M4 会静默地按训练从未用过的目标打分 |

缺陷 1 的既有单测抓不到它，原因值得记下：假 head 在 CPU 上 clone logits，
从不触碰索引映射；修复后假 head 改为跟随 features 设备，并新增一条复刻
真实算子（`index_select` + device-local index）的用例。缺陷 2 的处理是
新增 `me_rssm/sanity/probe_z_utilization_standard.py`：走 `dz.ZIntervention`
的后验重写（标准族在 eval 下 `z_t == mu_q`，逐位断言成立），并**转发部署态
`rssm_cls_logit_residual`**——旧工具 `pts_bbox_head(features)` 不带该残差，
会去打分一个现实中从不运行的检测器，系统性低估 z 对分类决策的作用；
`probe_z_util_region_split.py` 同步转发（旧配置下残差为 `None`，§69 读数
不受影响）。三个缺陷都以"就地回退修复 → 测试失败；恢复 → 测试通过"验证过
测试确有牙齿。

另：看门狗脚本被修改时它正处在等待循环里。bash 按文件偏移增量读取脚本，
从偏移中途继续执行新字节会损坏后续命令，故 `semA_watch` 已在 ep6 之前
整体重启（等待起点 16:23:16 UTC），确保从偏移 0 读起。

#### 71.9.3 两处判据口径写定（门槛数值一律不动）

两处都在 **ep6 读数未知**时写定，以避免"按结果挑解释"。

1. **M2 严格读法**（`fdbded2`）。预登记原文写「`h_t`（或 `feat`）」，
   可读成"任一臂即可"，而探针自始即要求**两臂各自** ≤ −30%。小节标题是
   **z 排他性**：h 与观测都不允许替代 z，故取严格读法并在预登记中显式写明。
   措辞对齐既有实现，**不改任何数值**。
2. **M1 的 `hit@1 ≥ 0.60` 是跨网格外推**（`f287057`）。最接近的先例是
   §59 route-C（同一 CenterNet focal 公式），但网格不同：route-C 光栅化到
   `latent_size=(16,16)`、`object_min_radius=1`；候选 A 光栅化到 full-res
   `216×248`、`min_radius=2`（`R4Det.py` 用 `z_t.shape[-2:]`）。`hit@1`
   问"唯一 argmax 格是否落在 GT 标记格上"，其随机基线约 `num_pos/cells`：
   16×16 上 ≈1.6e-2，full-res 上 ≈7.5e-5（本 run `stat_sem_pos_per_img`
   实测每图约 4.0 个 GT 峰格）。**门槛数值保持不动**（在跑期间调门正是
   预登记禁止的事后调参），改为探针额外记录 `hit1_chance` 与
   `hit1_excess_over_chance` 两个**跨网格可比**的补充读数（不进门；
   `gate_m1` 仍只吃 `(mean_thr50, mean_hit1)`，有测试固定）。
   判定规则预先写明：若 M1 因 full-res 口径擦线而 `thr50` /
   `hit1_excess_over_chance` 达标，该结果记为**证据不足**——既不算机制
   否定也不算通过，须补一个 full-res 同口径对照再判。

#### 71.9.4 A 侧训练期同口径读数（train split，`stat_sem_*`）

逐 epoch 末值（口径：训练统计，z 为采样值，train split）：

| ep | `stat_sem_hit1` | `stat_sem_thr50` | `loss_rssm_sem` | `stat_cls_res_rms` | `stat_mu_diff_sq` |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.1587 | 0.0733 | 1.2632 | 0.9707 | 3.6267 |
| 2 | 0.2409 | 0.2317 | 1.0536 | 0.9616 | 0.2650 |

对照 §59.5 route-C（同公式，**16×16 网格**）：ep1 0.4933/0.1935、
ep2 0.6764/0.4274、ep6 0.8107/0.6585。A 的同 epoch 值明显更低，但如上
所述两者网格差 209 倍，**不能据此判定 A 的语义头更差**；这正是
§71.9.3(2) 要把门槛口径单独记下来的原因。

#### 71.9.5 本节边界

* 无 AP 宣称；ep1/ep2 的 val 数字只是训练健康度的旁证，S2 窗口
  （ep12–16）尚未开始。
* 未改模型、未改判据数值、未动重建目标/KL/权重/日程，未启动任何新训练。
* 缺陷 1–3 的修复都在**探针/看门狗侧**，不触碰被训练模型的任何张量；
  正在跑的训练进程不受影响（改动后训练持续产出，无 NaN/OOM/重启）。
* 判定辅助（read-only、CPU、不启动任何训练）：`me_rssm/sanity/semA_ep6_verdict.py`
  收集 `semreadout_ep6_gate.json` 的四门读数、打印**不进门**的
  `hit1_chance`/`hit1_excess_over_chance` 补充读数、给出带真实跨度与 n 的
  早期 AP 窗口、并原样列出 §7 分叉；看门狗已在采集结束后自动调用它。
  它只收集与打印，**不做判定**。
#### 71.9.6 证据包完整性检查与产物指纹（训练在飞期间，`0b4e0cc`）

**动机（两个缺陷，均会作废 S1 判定）**：

1. **看门狗没有失败传播**。`watch_semA_ep6.sh` 是一串不打 `&&` 的探针调用；
   任一探针死掉（OOM / CUDA error / 融合族不匹配）**不写 JSON**，而日志仍会
   走到 `probes done`。此时 `semreadout_ep6_gate.json` 缺失或过期，读者很容易
   把"探针没跑成"误读成"门没过"——正是预登记最忌讳的判定污染。
2. **读数与产物不可回溯**。gate JSON 只记录 checkpoint **路径**，不记录
   内容。`epoch_6.pth` 若被后续写入覆盖、或读的是旧文件，判定无法事后发现。

**修复（都在判定/探针侧，仍不触碰被训练模型）**：

* `semA_ep6_verdict.py` 新增 `check_evidence()`：逐一检查五份产物的存在性与
  schema（gate 的四门键、`input_fingerprints`、非空 `indices`；三窗 z-util 的
  `per_sample` ≥2 且每项带 `variants`；source 探针的
  `fusion_class`/`mu_q`/`h_t`/`e_pooled`），任一不满足即 `ok=False`。
  输出新增 `[MISSING/UNREADABLE/MALFORMED/INCOMPLETE]` 明细行，且
  **即使四门全过、只要证据包不完整也报 `NOT EVALUABLE`**，不再输出
  `ALL PASS`。同时打印 checkpoint 的 sha256 前缀 / size / mtime 与 config 的
  sha256 前缀，供人工核对"读的是不是刚落的 ep6"。
* `probe_semreadout_ep6.py` 新增 `input_fingerprints()`，把 config 与
  checkpoint 的 sha256/size/mtime 写进 gate JSON（缺失产物记为 `None` 而非
  崩溃，由判定侧统一报 INCOMPLETE）。
* 新增 `me_rssm/sanity/test_semA_ep6_verdict.py`（**16 条**，CPU）：完整包
  为 OK；八种残缺形态（缺文件 / 非法 JSON / 顶层非对象 / 缺 schema 键 /
  缺指纹 / 缺门 / 空 indices / 单样本 / 无 variants）各自必须 `ok=False`；
  以及**生产者—消费者契约测试**——`check_evidence` 要求的键必须是探针实际
  写出的键（含 `report['gates']['Mx']`、三窗 `variants`、source 的五个键），
  且 `EVIDENCE` 覆盖的每份 JSON 看门狗都确实会写。任一侧改名会让测试红，
  而不是让未来每次判定静默变成 INCOMPLETE。
  `test_probe_semreadout_ep6.py` 追加 3 条指纹测试（哈希 64 位、内容不同则
  哈希不同、产物缺失不抛异常）。

**回归**：`test_probe_semreadout_ep6`+`test_probe_z_utilization_standard`+
`test_semA_ep6_verdict`+`test_semantic_readout_rssm` = **57 passed**；
`test_grad_paths` / `test_mci_rssm` 全绿；`test_rssm_fusion` **85 passed**。
另用手工构造的完整/残缺 bundle 各跑一次真实判定器，确认完整包打印
`ALL PASS`、删掉 source 产物后改报 `INCOMPLETE BUNDLE` +
`NOT EVALUABLE`。

#### 71.9.7 第六处缺陷：M1–M4 的 GT 读错位置（会导致整包证据作废，`118fa2a`）

**缺陷**：`probe_semreadout_ep6.rollout()` 从 `img_metas[-1][0]` 取
`gt_bboxes_3d` / `gt_labels_3d`。但 `CustomCollect3D2` 的 `meta_keys`
（见本 run config）只含几何/相机元数据（`filename`/`img_shape`/`lidar2img`/
`cam_aware`/`bda_rot` 等 27 项），**根本不含 GT 键**。GT 实际挂在 **sample 顶层**：
`dataset[i]['gt_bboxes_3d']` / `['gt_labels_3d']` 是逐帧列表（old→new），
`[-1]` 即当前帧——与训练路径一致（`R4Det.forward_train` 把当前帧 GT 交给
`set_object_gt`，其内部同样按「每帧一个样本列表」的组织方式光栅化）。

**后果**：M1→M2→M3→M4 全部依赖这个 GT，`rollout` 是它们的第一步，因此每次运行
都会在**第一个 clip** 抛 `KeyError: 'gt_bboxes_3d'`，且在写 gate JSON **之前**
就退出——整包 S1 证据（四门 + 三窗 + source）全废。这是本 run 里最严重的一处：
前三处会给出错误读数，这一处直接让读数不存在。

**实测证据（不是静态推断）**：用 450-iter 冒烟权重
（`/data/lurui/work_dirs/semA_smoke450_seed0/iter_450.pth`，真 semA checkpoint）
跑 `--limit 2 --stride 200`：
* 修复前（`git show HEAD^:...` 的版本）→ `KeyError: 'gt_bboxes_3d'`，
  **没有 JSON 落盘**；
* 修复后 → 正常完成，四门读数与 `input_fingerprints` 全部写出。

**修复**：抽出单一入口 `sample_current_gt(dataset, index)`，只它一处解析 GT，
并在 `rollout` 里调用；函数对「GT 不在顶层」「逐帧数量不一致」「无 GT 帧」
三种情形显式报错，并在报错信息里点名「不得从 `img_metas` 读 GT」。

**等价性验证（关键，证明改后与训练同源）**：
* 10 个 gate clip（3/203/403/…/1803）逐个与 `dataset.get_ann_info(i)`
  对比标签：**10/10 一致**（含 idx 1803 的 `Other` 被 pipeline 过滤为 −1 的
  情形，模型实际收到的就是这个已过滤版本）。
* 用训练同款光栅化器 `bev_centers_to_heatmap`（`num_classes=4`、
  `out_size=(216,248)`、同 `point_cloud_range`、`min_radius=2`）对比
  pipeline 后 GT 与 raw `get_ann_info` GT：idx 1803/3/1603 的
  **热图逐位相同**。
* val 管线 `is_train=False`（确定性），实测 `dataset[3]` 两次调用的
  boxes/labels/points/img **全部逐位相同**，故 `rollout` 内第二次读样本
  不会让几何与 GT 失配。

**是否波及历史结论**：全仓检索 `meta['gt_bboxes_3d']` / `img_metas[-1][0]`
已无其它命中（唯一命中是本次新增的测试断言）。§69 区域探针
（`probe_z_util_region_split.py`）、§71.6 的 S0 wiring 探针均不读 GT，
**§69/§71 已记录结论不受影响**。

**回归**：`test_probe_semreadout_ep6` 增至 **32 passed**（新增
`TestSampleCurrentGt` 6 条：取逐帧列表末项、拒绝从 `img_metas` 读 GT、
逐帧数量不一致报错、无 GT 帧报错、标签为 long tensor、
`rollout` 不再出现 `meta['gt_bboxes_3d']`）。测试有牙齿：把修复临时回退成
`meta['gt_bboxes_3d']`，`test_rollout_no_longer_reads_img_metas_for_gt`
立即失败；恢复后 32 条全过。三份 probe 测试合跑 **52 passed**。

**同批核验的其余三件（无需改动）**：
* `probe_z_utilization_standard`：用同一冒烟权重按看门狗原样
  （`--limit 4 --batch-size 4 --shuffle-offset 2`）跑通并写出 JSON；
  另确认 `--limit 4` 下 offset 2 / −1 都不会自映射（offset 2 在
  `limit=2` 时会自映射报错——这是探针的既有保护，看门狗的 `limit=4`
  不受影响）。
* `probe_posterior_source`：静态确认 semA 走 `MotionAlignedRSSMFusion`
  分支（无 `latent_gru`），探针选 `transition`（native BEV res），与 MAI
  ep12/ep16 已记录的 `fusion_class=MotionAlignedInnovationRSSMFusion`
  同属 `transition` 路径；`stride=700` 产出 3 个索引 [3, 703, 1403]，
  与 MAI 存档一致，**不改**。
* 看门狗三条 z-util 窗口的索引选择：`start 3/100/1800 + limit 4`
  均为 4/4 命中（`[3,4,5,6]`/`[100..103]`/`[1800..1803]`）。

**操作事故（如实记录，未影响训练）**：为验证修复，曾在训练进行中把探针跑在
GPU 5/6/7 上，`probe_posterior_source` 因显存不足 OOM。训练日志零 OOM/
零 Traceback、显存 19716/19788/19724 MiB 稳定、步频与 ETA 无异常，训练
进程 PID 未变。此后所有探针验证改为**静态或复用 450-iter 冒烟权重且不与
训练争卡**；正式采集仍由看门狗在停训、GPU 释放后进行。

#### 71.9.8 第七处缺陷：gate JSON 只在最后写一次（`7d13b19`）

**缺陷**：`probe_semreadout_ep6` 把 `report` 在全部 M1→M4 走完后**一次性**
落盘。M3 要对 10 个 clip 各跑一次额外 rollout，M4 还要做 donor 选择与
head-swap 重算——中间任一环节崩掉（OOM / CUDA error / 被 kill），
**已经算完的 M1/M2 读数会一起丢**，整包 S1 仍作废。

**修复**：抽出 `write_report()`，用 `json.dump` 到 `*.tmp` 再
`os.replace()` 原子改写（读者永远不会读到半截 JSON，也不会留下临时文件），
并在 M1/M2 之后、M3 之后各 flush 一次。崩溃时磁盘上至少留下已完成的门。

**判定侧同步**：`semA_ep6_verdict.py` 把「门键缺失」渲染为
**`NOT MEASURED`** 而不是 `FAIL`——「没测到」与「测到且没过」是两种不同
结论，只有后者构成机制否定。聚合行仍为 `NOT ALL PASS`，且证据包完整性
检查会同时把该文件标成 `INCOMPLETE`。

**验证**：新增 `TestWriteReport` 4 条（绝对路径返回、自动建父目录、覆盖写
后无 `.tmp` 残留、源码级守卫断言 M1/M2 与 M3 之后各有一次 flush），
`test_probe_semreadout_ep6` 增至 **36 passed**。另在 450-iter 冒烟权重上实跑
一次：四门 + `input_fingerprints` 正常写出、无 `.tmp` 残留；并用手工构造的
「M1/M2 有、M3/M4 缺」部分报告跑判定器，确认输出
`M3 readout : NOT MEASURED`、`M4 class info : NOT MEASURED`、
`INCOMPLETE BUNDLE`、`S1 AGGREGATE: NOT ALL PASS`。

#### 71.9.9 把 §71.9.3(2) 的 M1 例外写成可执行分支（`e7fbb77`）

**缺陷（口径已写定但没接线）**：§71.9.3(2) 预先写明——M1 若只在
**full-res 原始 hit@1** 上擦线、而 `thr50` 与 `hit1_excess_over_chance`
健康，则结论是**证据不足**（须补 full-res 同口径对照），**不得**记为机制
否定。但判定器当时只把这句话当作一段 `NOTE` 打印，聚合行仍照旧输出
`NOT ALL PASS`，等于**预登记的例外从未生效**；若 ep6 恰好落在这个区间，
会把「口径不可比」误归档成「机制被否」。

**修复**：把规则写成显式函数 `m1_grid_limited(thr50, hit1, excess)` 与
聚合分支。阈值全部由既有预登记数值导出，不新增门槛：

* `M1_BAR_THR50 = 0.30`、`M1_BAR_HIT1 = 0.60`（§5 原值，不动）；
* `M1_CALIBRATION_CHANCE = 1.6e-2`（route-C 16×16 网格的随机基线）；
* `M1_EXCESS_BAR = 0.60 − 0.016 = 0.584`——即「在原校准网格上，
  0.60 这条线相对该网格随机基线所隐含的超额量」。预登记未给超额门的
  数值，故此条**由既有线导出并显式打印**，不另立新数。

**触发条件（三者同时成立，且 M1 必须是唯一未过的门）**：
`thr50 >= 0.30`、`hit1 < 0.60`、`excess >= 0.584`、其余 M2/M3/M4 全过。
命中时聚合行输出
`M1 failed on the full-res raw rate only ... -> INSUFFICIENT EVIDENCE
(pre-registered section 71.9.3(2)), NOT a mechanism refutation`，并在 §7
分叉清单里单列该例外；**不**输出 `NOT ALL PASS`。

**验证**：新增 `TestM1GridLimitedRule` 8 条，把各臂钉成互斥——
thr50 也掉线不算例外（miss 不能归因于网格）、excess 偏低不算例外、
M1 本来就过线时不适用、任一读数缺失一律不算例外、阈值必须由预登记值
导出、以及源码级守卫断言聚合分支要求 `gates.get('M1') is False` 且
`k != 'M1'`。另用四组手工 gate JSON 跑真实判定器确认分支：
`thr50=0.41/hit1=0.071/excess=0.60` → **EXCUSED**；
`thr50=0.29`（也掉线）→ 计为失败；`excess=0.02` → 计为失败。
三份 probe 测试合跑 **64 passed**。

#### 71.9.10 看门狗环境预检（`30333b0`）

**缺陷**：`watch_semA_ep6.sh` 的每个探针都通过 `python` 启动。若
`source .envrc` 没生效（conda 未激活 / shell 非交互差异），`python` 不存在，
**所有探针都会立刻失败且不写任何 JSON**，而脚本仍会走到 `probes done`。
§71.9.6 的完整性检查能发现产物缺失，但无法区分「环境坏了」与「探针崩了」，
事后排查要靠猜。

**实测**：`bash -c 'source .envrc; python -c "import torch..."'` 在本机
现行环境**可以**正常激活（`python` →
`/home/lurui/envs/miniforge3/envs/r4det/bin/python`，torch 2.0.1+cu118、
CUDA 可用）。即该风险当前**未触发**，属预防性加固。

**修复**：在探针序列之前加两步预检，任一失败即写 `ABORT` 到看门狗日志并
`exit 1`（不进入采集）：
1. `command -v python` —— 环境根本没激活；
2. `python -c 'import torch, mmcv, mmdet3d'` —— 激活到了别的环境。

**验证**：正向路径实跑确认 `PREFLIGHT PASSED`；
负向对照（`PATH=/nonexistent`）确认按设计 `ABORT` 并 `exit 1`。
改脚本后按 §71.9.2 的约定**重启看门狗会话**（新起点 17:52:22 UTC），
确保 bash 从偏移 0 读取。

#### 71.9.11 `sem_head_input` 未识别值静默落到 `feat` 臂（`0fae53f`）

**缺陷**：`SemanticReadoutRSSMFusion._sem_source()` 原实现是

```python
if self.sem_head_input == 'z': return z_t
if self.sem_head_input == 'h': return h_t
return feat          # <-- 任何其它值
```

即**未识别的取值一律落到 `feat`**。而 M2 正是靠切换这个属性来构造
h 臂与 feat 臂：若该属性被写成 `'H'` / `'FEAT'` / 其它拼写，**h 臂会静默
变成 feat 臂的副本**，M2 就会把「同一个张量的两次测量」当作「两个控制臂」
报告，得出假的「两臂都劣于 z」。构造函数虽然已做取值校验，但探针是在
**运行时直接改属性**（`rollout` 里 `fusion.sem_head_input = sem_input`），
走的正是这条未校验路径。

**实测**：`_sem_source` 逐值探针确认 `'z'/'h'/'feat'` 分别返回各自张量，
而 `'Z'/'H'/'FEAT'/'typo'/''` **全部返回 `feat`**。

**修复**：改为三分支显式返回，其余取值 `raise ValueError`。

**验证**：新增 T13（断言三个合法臂各自返回自己的张量；7 个非法值
`'Z'/'H'/'FEAT'/'typo'/''/'feature'` 必须抛 `ValueError`）。
测试有牙齿：把 `raise` 换回 `return feat`，T13 立即以
`AssertionError: ValueError not raised` 失败；恢复后
`test_semantic_readout_rssm` **12 passed**。

**边界**：只把「静默兜底」改成「显式报错」，三臂语义与 M2 判据数值均未动。
当前 config 写的是 `sem_head_input='z'`，看门狗三段控制臂也各自写死
`'z'/'h'/'feat'`，故本 run 读数不受影响；这是防止未来改配置时静默失效。

#### 71.9.12 看门狗↔探针的 CLI 契约测试（`f691561`）

**动机**：看门狗本身**不能在这里执行**（它要打断正在跑的训练），所以
「改了探针参数名」或「少传一个窗口」这类漂移，只会在 **ep6 当场**暴露——
表现为探针在 argparse 阶段就退出、不写 JSON、留下残缺证据包。§71.9.2 的
第 2 处缺陷（看门狗指到 MCI 专用探针）就是同一类问题，当时靠人工复核才发现。

**新增** `me_rssm/sanity/test_watch_semA_ep6_cli.py`（**10 条**，纯静态、
不碰 GPU/训练）：从看门狗脚本里**解析出真实调用**（处理行尾续行、
`CUDA_VISIBLE_DEVICES=` 前缀与输出重定向），再逐个断言——

* 三个探针**都还在被调用**，且脚本文件存在；
* 传的**每个** flag 都在目标探针自己的 `add_argument` 里（逐协议文件解析）；
* 每个探针的**必需 flag 齐全**（窗口/输出不会被悄悄删掉）；
* z-util 的三个窗口仍是预登记的 `start-index = 3/100/1800`，且
  `--limit`/`--shuffle-offset` **不自映射**（自映射会让
  `resolve_shuffle_sources` 直接抛错，即 §71.9.7 复核过的那条保护）；
* gate 探针仍是 `limit=10 / stride=200 / start-index=3`；
* 各探针 `--output` **互不重复**且都是 `.json`；
* 判定器仍在**最后一个探针之后**运行，且脚本内**不含**任何续训路径
  （`dist_train` / `train_vod` / `resume_from` / `--resume-from` /
  `--load-from`），并仍打印 `gate decision is a manual step`。

**验证有牙齿**：把看门狗文本里的 `--shuffle-offset` 改成
`--shuffle-offset-typo`（在内存中改，不碰仓库文件），
`test_every_flag_is_accepted_by_its_probe` 与
`test_required_flags_are_all_present` 立即失败。

**回归**：五份测试合跑 **86 passed**。

> **后续更正（`7abd4ee`，见 §71.9.13）**：本小节当时写的「看门狗脚本本身
> **未改**」只在 `f691561` 时点成立。随后 §71.9.13 在看门狗尾部接入了 §72
> 草稿渲染器，脚本已改并按同一偏移约定**重启会话**；该小节列出的 10 条
> 断言在重启后仍全部成立（脚本扩到 12 条）。

**边界**：门槛数值、探针的采样窗口（M1 `stride=200`/10 帧、z-util 三窗、
source `stride=700` 对应 3 个索引——与 MAI ep12/ep16 逐字一致，故不改）、
模型与训练日程全部未动；看门狗脚本本身**未改**（它正处等待循环，bash 按
文件偏移读脚本，改动须重启会话，见 §71.9.2 附注）。训练持续产出，无
NaN/OOM/重启。§71.9.7 的 GT 修复与 §71.9.8 的增量落盘同样只动探针/判定侧：
不改模型、不改数据管线、不写权重。

* 下一步：等 ep6 val + `epoch_6.pth`，看门狗自动采集 M1–M4 与三窗读数；
  **S1 判定是人工步骤**，且即便 S1 全过，也须先取得用户对 §5 S2 数值门的
  批准才能续训到 ep16。

#### 71.9.13 §72 训练记录的机械生成器 + M1/M2 逐 clip 索引（`7abd4ee`）

**动机（AGENTS 规则 5/6 的执行成本）**：规则 5 要求三个口径**分别**写清
（区间均值必须带真实跨度与 n、全轮峰值必须写「值 @ epoch」且与权重无关、
best saved 单列且不参与排名），规则 6 要求逐 epoch 曲线、逐类别拆解与机制
诊断不得省略。每节手抄这些数字正是本项目已经出过错的地方（DC1 门槛算术
纠正）。故在 ep6 出数之前把生成器写好。

**新增** `me_rssm/sanity/semA_ep6_record.py`（只读、CPU、不启动任何训练）：
从证据包与 run 自身 `*.log.json` 渲染 §72，**不手抄任何读数**，
**不修改** `docs/training_runs_full.md`（草稿落在 work_dir，
`section72_draft.md`，由人工填入后并入正典）。

* **门槛逻辑单源**：`import` `semA_ep6_verdict.py`，复用它的
  `check_evidence` / `m1_grid_limited` / `val_by_epoch` 与
  `BASELINE`/`EVIDENCE`/`WINDOWS` 定义，**不重述**任何门槛数值。
  证据清单一致性由
  `test_evidence_manifest_is_the_verdict_s_own` 断言相等。
* 三个口径、逐 epoch 曲线、逐类别窗口均值、`stat_sem_*` 训练期读数、
  M1–M4 逐 clip 明细、三窗反事实（含标准 RSSM seed1 ep6 同窗对照）、
  posterior source 与 §7 分叉全部自动成文。
* **FILL 标记**：产物无法提供的运行事实（起止时刻、吞吐/显存、停止与 GPU
  释放时刻）写成可见的 `<!-- FILL: ... -->` 并在 stderr 计数告警，
  不可能被静默漏掉。
* **S2 边界写进模板**：小节正文固定声明「本节不宣称任何 S2 性能结论」，
  且早期窗口一律标注「不是 ep12–16 的 S2 判据」。

**同时修一处证据侧缺陷**：`probe_semreadout_ep6.py` 的 M1/M2 逐 clip 条目
**不带 clip 索引**，z/h/feat 三臂只能按**位置**对齐；一旦某个 clip 因无 GT
被 `head_quality` 丢弃，剩余条目会**错位比较不同 clip**。现由
`dict(q, index=int(idx))` 在构造处附上调用索引（`head_quality` 本身仍不返回
index，故是纯增量、不会覆盖探针读数；丢弃 clip 时索引不错位，已单独验证）。
探针回归 **36 passed**。

**三窗同一索引的保护**：生成器在三窗 JSON 返回**同一组 clip** 时输出警告，
不得读成「三窗口一致」。真实的看门狗按 `--start-index = 3/100/1800` 取窗
（`dz.select_indices` 用 `range(start, start+limit)`），故三窗**由构造保证
不同**；该警告是防**探针侧回归/手工改窗**的护栏，不是对现有读数的怀疑。
它在手工构造的同窗替身包上确实验证过触发（见下），正式 ep6 采集若触发
即说明窗口参数漂移。

**验证方式（避免把替身当实测）**：本节的生成器行为用两种输入验证——
① **真实探针产物**：M1–M4 gate JSON 的 450 iter 冒烟产物
（`/tmp/semreadout_*json`，实跑写出）；② **手工替身包**用于覆盖
z-util 三窗与 source 归因（冒烟配置是 `IterBasedRunner`，**未跑过** z-util
与 source 探针，故没有这两个的真实产物）。替身包的三窗取自同一份内容、
因此 sha256 相同（生成器必然报同窗警告），source 文件只带 `rms` 字段。
**替身只用于验证生成器的告警/护栏路径，不得作为任何机制读数**；正式读数
必须来自 ep6 看门狗的实跑产物。

**§72.8 曾只读 `rms`（`eeaa32d` 修）**：初版 §72.8 只渲染
`mu_q/h_t/e_pooled` 的 `rms`，丢掉了 `sample_specific_energy_frac`
（整条审计线依赖量）与 `mu_q_removal_of_h/e` 的**比值**（跨代通用对照：
标准主线 ep12 **2.338**、MAI ep12 **0.8418**、MCI ep12 **0.605**）。
更糟的是「只有 `rms`」的**残缺**文件会渲染成半空表而无任何提示——而 §71.9.13
用的替身恰好就是这种残缺形态，等于**可以把替身当读数粘进记录**。现改为
渲染完整四字段块 + 计算 **h/e 依赖比**，并在缺
`sample_specific_energy_frac` 时明确标注「不是完整读数、不得当作特异占比
为 0、整表不得引用」。用**真实** source 产物（MAI ep12）校验：
`e_pooled` 特异 **64.28%**、h/e **0.842** 均复现训练记录；内存中删掉该护栏
可让新测试变红。`test_semA_ep6_record` 增至 **29 条**，六份合跑 **117 passed**。

**残缺证据包下的两处不实渲染（`6594de5` 修）**：草稿是在 GPU 采集跑完后
**无人值守**渲染的，证据残缺时正是最需要它的时刻。初版有两处会给出误导：

1. **H1 标题**在「四门未全过」时一律写「未全门通过」——包括**什么都没测到**
   的情况。该措辞的含义是「测了、有门没过」，属机制结论；证据包残缺或 gate
   JSON 缺失时现改为 **`NOT EVALUABLE（证据不完整）`**，「未全门通过」只保留
   给**真实的实测失败**。
2. **M3 打印 `None/None`**、M4 的 `bar` 在读数缺失时直接内插成 `n/a` 字样；
   现分别渲染为 **`NOT MEASURED`** 与 `n/a`，并对
   `num`/`pct`/`signed` 增加非数值防护。

**测试有牙齿的修正**：上一版「空包报 NOT EVALUABLE」的断言**也能被正文
其它位置的同名字样满足**，因此删掉标题修复**不会让它变红**（已实测确认）。
新断言只看**首行**并要求其为 `NOT EVALUABLE` 且**不含**「未全门通过」，
回退即红（已验证）。`test_semA_ep6_record` 增至 **36 条**，
六份合跑 **124 passed**。

**看门狗接线**：ep6 采集结束后自动追加渲染草稿（判定器之后），渲染失败只
告警、不影响证据包；`test_watch_semA_ep6_cli.py` 相应扩到 **12 条**，
新增「渲染器用 `--run`/`--output` 且两个 flag 都在其 parser 里」与
「脚本只写 `section72_draft.md`，不写正典记录（按重定向目标解析，注释里
提到正典文件名不算写入）」。**两级均验证有牙齿**：`--shuffle-offset-typo`
与「把草稿重定向到正典」都能让测试失败。

**回归**：`test_probe_semreadout_ep6`(36) + `test_probe_z_utilization_standard`
+ `test_semA_ep6_verdict`(24) + **`test_semA_ep6_record`(26)** +
`test_semantic_readout_rssm` + `test_watch_semA_ep6_cli`(12)
合跑 **114 passed**。

**看门狗会话已按 §71.9.2 附注重启**（新起点 **18:13:58 UTC**，
start ts 1791656038）——脚本改动后 bash 仍按旧偏移读文件，必须重启；
新会话已确认在等待循环中。训练**未受影响**：同 PID 持续产出
（ep4 750/1902，无 NaN/OOM/重启）。

**边界**：未动模型、未动判据数值、未动采样窗口、未启动/续接任何训练；
所有改动在探针/判定/记录生成侧。即便 ep6 采集完成，**S1 判定仍是人工步骤**，
且 S1 全过也须先取得用户对 §5 S2 数值门的批准才能续训 ep16。

#### 71.9.14 门控采样点在真实 val 集上的预检（只读，ep6 出数前）

§71.9.7 修掉 GT 读取位置后，M1–M4 是否**真能在 ep6 当天跑出完整读数**仍需
验证：若采样点少于 10 个、或某些 clip 无 GT、或**没有任何异类 donor**，
探针要么抛错整包作废，要么静默退化（M3/M4 的分母偏小、donor 退回
`--shuffle-offset`）。故在 ep6 未产出时用**真实** val 集（2040 clip）只读预检：

* **gate 采样点**（`--start-index 3 --stride 200 --limit 10`，
  `min_valid_history=2`）：解析为
  `[3, 203, 403, 603, 803, 1003, 1203, 1403, 1603, 1803]`，**满 10 个**
  （`select_clips` 只在历史有效时才前移并跳 `stride`），故 M3 的
  `len(rows) >= 8` 与符号检验的 10 帧规模由构造满足。
* **每个采样点的当前帧 GT**（`sample_current_gt`，即 §71.9.7 的唯一入口）：
  目标数 3 / 4 / 3 / 11 / 4 / 2 / 6 / 6 / 1 / 2，**无空帧**；主类分布
  为 `1,2,2,2,3,2,1,2,1,2`，即 **Car/Truck/Cyclist 三类都在场**，
  任意 clip 都能找到**主类不同**的 donor ⇒ M4 的 donor 规则不会退化到
  兜底分支，`donor_transfer` 确为「换成异类 z」的读数。
* **z-util 三窗**（`--require-valid-history` 默认开）：`3/100/1800` 起点分别
  解析为 `[3,4,5,6]` / `[100,101,102,103]` / `[1800,1801,1802,1803]`，
  **三窗互不相同**，与 §71.9.13 的「由构造保证不同」一致。

**边界**：纯只读（只建 dataset 与算索引，不建模型、不占 GPU、不写任何产物），
未改模型/判据/采样窗口，训练未受影响。**一处须在 §72 记录时注意的事实**：
clip `1603` 只有 **1 个** GT 目标，其 M3/M4 每 clip 读数只能取 0 或 1、
方差最大；§72 的分组结论不得由该 clip 单独支撑。

#### 71.9.15 §72 运行事实改为机械派生（不再手填 `FILL`）

§72.2 的起止时刻、吞吐/显存、停止方式三项原先写成 `<!-- FILL -->` 标记由人
填入，而 §72.10 同时声称本节「未手抄任何读数」——**锚一个手打的时刻正是本
生成器要消灭的抄写面**（本项目已在 DC1 门槛算术上出过一次同类错误）。三项
事实本就都在 run 目录里，故改为派生（commit `16c4cc4`）：

* **ep6 段起止**：run 自己训练日志（`YYYYmmdd_HHMMSS.log`）首/末带时间戳行；
* **吞吐 / 显存**：`*.log.json` 的 `mode=train` 记录 `time`/`memory` 中位数
  （真实包 228 条 ⇒ **1.566 s/iter、16.7 GiB/卡**）；
* **停止方式**：看门狗 `gate hit` / `GPUs free` / `probes done` 三个时刻 +
  终止信号与退出码。

两处踩坑由**会变红的**测试钉住：

1. **信号取错**：traceback 自己的源码行也含 `got signal:`，取**第一个**匹配
   会得到字面量 `{sigval}` 并渲染成乱码；正确值是最后一个数字匹配
   （`SIGINT(2)`）。
2. **日志取错**：按「第一个 `*.log`」会静默选中看门狗日志——它在中断**之后**
   才开始、且本身满是带时间戳的探针输出，等于把**探针窗口**报成训练窗口。
   现按显式优先级取（run 日志 → `stdout.log` → 永不取看门狗日志）。

`FILL` 保留为**日志缺失时的回退**，告警路径不变。`test_semA_ep6_record`
**36 → 56**，六份合跑 **134 → 144 passed**；真实证据包现渲染出 **0 个 FILL**。

#### 71.9.16 §72 的四处判定补充（均为门槛逻辑派生，非新判据）

**同时**给 §72 补四处人工判断也容易漏的记录（全部门槛常量取自
`semA_ep6_verdict.py`，无一处新判据）：

1. **M1 例外条款要写「适用/不适用」而不是只描述条款存在**。只写「存在一个
   跨网格例外」会让真实未过被读成「口径未校准」——正是 §7 禁止的事后逃生口。
   真实包实测 **不适用**（`thr50` 0.1167 < 0.30 且 excess 0.0833 < 0.584，
   两项都未达线）⇒ **真实未过**。另加「band 有下界」的护栏测试：同一 fixture
   把 `hit@1` 从 0.59 降到 0.50（`thr50` 仍 0.40）必须从「适用」翻成「不适用」。
   > **⚠ 本条理由已被 §71.9.17(1) 修正（结论不变）**：「两项都未达线」是拿
   > **val** 读数去比 **train** 口径的门槛；A 自身 train 口径下 `thr50`
   > **0.4923 过线**、`hit@1` 0.3586 未过。更正后结论仍是**真实未过**，
   > 但机制读数更强：该头**确实找得到目标中心**，只是 argmax 落不到 GT 格上。
2. **M2 通过要正面写「什么存活下来」**，不能只列失败项。M2 PASS ⇒
   「概率状态」这一主张**未被本次测量否证**（h/feat 臂 `thr50` 均 0.0000 vs
   z 0.1167），瓶颈在读出与任务收益（M1/M3/M4），不在「z 有没有信息」。
   > **⚠ 本条已被 §71.9.17(3) 取代**：「未被否证」仍是过强表述——M2 喂的是
   > 分布外输入，只能证明该 head 依赖其训练输入，不能证明 z 是唯一信息源。
   > 正确表述为「**既未被否证、也未被证实**」，真正的排他性检验是未执行的
   > **C3 臂**。此处保留原文以留痕，不得再据此宣称。
3. **逐 clip 样本权重提示**：clip `1603` 只有 1 个 GT、clip `603` 有 11 个
   （§71.9.14 预检 3/4/3/11/4/2/6/6/1/2），故 §72.6 的分组结论不得由单个
   clip 单独支撑。权重相同时**不**出提示（避免噪声）。
4. **负结果归档要带实测原因**：§7 要求「把读数写进 Analysis」，故分叉行写出
   `M1 thr50 0.1167/hit@1 0.0833（例外不适用）；M3 7/10；M4 donor_transfer
   0.3818`，并写清归档**范围**（本次读出/监督设计，不是「z 无信息」，
   也不是「所有语义监督/残差读出方向」）。

**边界**：全部改动在记录生成侧（只读、CPU），未动模型、判据数值、采样窗口；
未启动/续接任何训练。

#### 71.9.17 归档后审计：三处推论口径修正（只读，§72 结论方向不变）

§72 归档（`1185fcc`）后复查自己的写作，发现三处**推论强于证据**。三处都做成了
可复现的只读核对，结论**方向不变**，但其中一处**理由被换成了更强的那个**。

1. **M1 门槛是 train 口径、本门读数却是 val 口径（口径混用）**。
   `thr50 ≥ 0.30 / hit@1 ≥ 0.60` 的先例 §59 route-C 报的是 **train 侧 epoch
   均值**（§60.5/§61.5，38 条 train 记录），而 §72 拿 **val** 读数（0.1167 /
   0.0833）直接对这两条线。同口径重算（A 自身 `stat_sem_*` ep6 的 **epoch
   均值**，不是末值）：

   | 口径 | `thr50` | `hit@1` |
   |---|---:|---:|
   | route-C ep6（门槛来源，16×16，train） | 0.6585 | 0.8107 |
   | **A ep6 train 侧（`stat_sem_*` epoch 均值，216×248）** | **0.4923** | **0.3586** |
   | A ep6 val 侧（M1 实际读数） | 0.1167 | 0.0833 |

   **例外条款仍不适用，但理由变了**：`thr50` 在 train 口径下**过线**
   （0.4923 ≥ 0.30），`hit@1` 却远未达线（0.3586 ≪ 0.60）。§72 原写「两项
   都未达线」是**错的**，而且**弱化**了真实结论——该语义头**确实找得到目标
   中心**（thr50 过线），但它的 argmax 落不到 GT 格上。两口径并列后，
   `excess <= hit@1` 给出 train 侧 excess 上界 0.3586 < 0.5840，val 侧
   0.0833 < 0.5840 ⇒ **两种读法都不满足例外，判定不依赖口径选择**。
   生成器里 pass/fail 标签已改为**从同一行的数字派生**（`_kw()`），杜绝
   手打「未过」贴在一个过线数字旁边的同类抄写错误。

2. **train 采样 z、eval/probe 喂均值（契约错配）——已查证并排除**。
   `R4Det.py:891` 的 `rssm_deterministic = not self.training` 与
   `rssm_fusion.py:1637` 的 `z_t = mu_q if deterministic else self.sample(...)`
   意味着训练喂 `mu_q + sigma*eps`、eval 探针喂 `mu_q`；ep6 记录的
   `stat_posterior_std ≈ 0.23` 对 `mu_q` RMS ≈ 0.58，噪声占比量级不小，故
   必须实测而不是争论。新增只读探针 `me_rssm/sanity/probe_semA_zcontract.py`
   在**同 checkpoint、同 10 个 val clip、同评分函数**上只切换 z 契约：

   | z 契约 | `thr50` | `hit@1` | `hit1_excess_over_chance` |
   |---|---:|---:|---:|
   | 部署（`mu_q`，即 M1 实际读数） | 0.1167 | 0.0833 | 0.0833 |
   | 训练（`mu_q + sigma*eps`） | 0.1167 | 0.0833 | 0.0833 |

   **干预检查**（否则「两臂相同」既可能是真无影响、也可能是开关没生效，两种
   相反结论对应同一组数字）：`z` 相对变化均值 **35.26%**（逐 clip
   34.37%–36.14%），语义热图相对变化均值 **1.348%**，`sigma/mu` 均值
   **0.3637**（0.3473–0.3816）。**逐 clip 读数完全相同、Δ = 0.0000。**
   部署臂**精确复现**了冻结的 gate 产物（checkpoint sha256 `8077411c6142e7ce…`、
   config sha256 `190a9751e8f68dcf…`、同 10 个索引、逐 clip 一致），即探针的
   自校验成立。**结论**：契约切换确实移动了 z，但两项门控读数一动不动 ⇒
   train/val 的 **4.3× 差距不是采样契约造成的，其来源仍未定位**。把
   「契约错配」从嫌疑名单划掉，同时**不**改变 S1 判定。
   产物 `semA_zcontract_ep6.json`（只读，未改权重、未启动训练）。

3. **M2 通过不能证明「语义只在 z」**。M2 是把 h/feat 喂给一个**只被 z 监督
   过**的 head，属分布外输入——即便 h 或观测里含同样的对象信息，这个 head
   也读不出来。故 M2 只证明「**该 head 依赖它训练时的输入**」。预登记里真正
   的 z 排他性检验是 **C3 臂**（`sem_head_input='feat'` 从零训练、同日程），
   **本次未执行**。§72/§71.9.16 原先的「**未被本次测量否证**」属于过强表述，
   已收紧为「**既未被否证、也未被证实**」，并在记录生成器里加了
   `TestM2ClaimIsScoped` 与「不得据此宣称」的显式禁令。

**测试**：`test_semA_ep6_record` 新增 M1 口径并列、M2 表述收紧、
`train_stat_means` 走日志均值（并加了一个行为性断言：同一 epoch 追加第二条
train 记录后，72.5 曲线取末值 0.9、而 M1 口径表必须仍取均值 0.6）等断言；
新增 `test_probe_semA_zcontract.py`（10 条，含干预检查与产物复现）；六份回归
合跑 **163 passed**（此前 144）。**边界**：全部改动在探针/判定/记录生成侧
（只读、CPU + 一次只读 GPU 探针），未动模型、判据数值、采样窗口、重建目标、
KL、权重或日程；未启动/续接任何训练；GPU 5/6/7 全程空闲。

#### 71.9.18 M4「读出不可用」的机制定位（只读，两项自身口径缺陷 + 一个机械解释）

上一节的 §71.9.17 收紧了 M1/M2 的表述。本节处理 §72 剩下的那个**没有被质疑过**
的读数：M4 的 `own_hit` 0.7379「过线」。它和 §70.6 的 93.7% 是**同一个错误类**——
拿一个**没有对照**的数字当证据。

**缺陷一：0.45 的 bar 比错了对象。** §5 预登记写「chance + 20 pt」，其中
chance 取 `1/num_classes = 0.25`。但「四类等概率」不是这份 val 的分布：10 个
采样 clip 的前景格上，GT 类别计数为 `[5, 8, 20, 6]`（Car 占 20/39 =
**51.28%**）。一个**永远投 Car** 的残差在这个窗口上得 **0.5128**，已经超过
0.45 的线；per-clip 多数类地板均值 **0.6970**、最大 **1.0000**（单目标 clip）。
即 M4 的 `own_hit` 有相当一部分可以完全由类别先验解释，而归档记录没有写出
这个地板。

**实测（只读探针 `me_rssm/sanity/probe_semA_m4floor.py`，同 checkpoint /
同 10 clip / 同 rasteriser）**：

| 读数 | 值 |
|---|---:|
| 复算 `own_hit`（从 residual 重算，非引用） | **0.7379**（与门控 JSON 逐位一致 ✓） |
| pooled 多数类地板 | **0.5128** |
| **超出地板的余量** | **+0.2251** |
| per-clip 多数类地板（均值 / 最大） | 0.6970 / 1.0000 |
| 1/C 名义 chance | 0.25 |

⇒ `own_hit` **确实高于先验地板**（+0.225），所以「z 的残差含类别信息」这一读数
**没有被推翻**；但它的**有效证据量是那条余量，不是 0.7379 本身**，而且 M4 的
FAIL 是由 `donor_transfer` 0.3818 判的，与 `own_hit` 无关。归档中「`own_hit`
过线」的写法必须连同地板一起读，否则会被当成比实际更强的证据。

**缺陷二：`head_swap_to_donor` 0.1167 不能读成「头忽略 z」。** 该读数低于随机
25%，看似「换成 donor 的 z 后头反而更差」。真实原因是一条**尺度**事实，
而不是信息事实：

| 量（前景格） | 值 |
|---|---:|
| `residual` 全图 RMS | 1.0015 |
| `conv_cls` 全图 RMS | 14.3501 |
| 比值 | **0.0698** |
| `conv_cls` 前景格 top1−top2 间隔（均值 / 中位） | **2.7782 / 2.1186** |
| `residual` 前景格 top1−top2 间隔均值 | 1.1190 |
| `residual` 前景格 absmax 均值 | 1.5961 |
| **|residual| 足以翻转 head argmax 的格占比** | **25.30%** |

即：**四分之三的前景格上，残差的最大绝对值小于 `conv_cls` 的 top1−top2 间隔**，
它**在算术上没有能力**改变该格的类别 argmax。这不是「头不消费 z」，而是
「残差被 `conv_cls` 的量级压过」。归档里 M4 的叙述把它当成信息结论，属于过强。

**尺度曲线（只读，逐格重跑 `pts_bbox_head`，同特征/同权重，只乘 residual）**：

| residual 增益 | 0（关） | 1（现状） | 2 | 4 | 8 | 10⁶（强制） |
|---|---:|---:|---:|---:|---:|---:|
| 前景格 head 命中率（按格加权） | **0.6410** | **0.6923** | 0.6923 | 0.6923 | 0.6923 | **0.5897** |

三条同时可读的结论：

1. 关掉 residual，命中率从 0.6923 掉到 0.6410（**−5.1 pt**）⇒ 残差**确实**
   参与了决策，比「纯装饰」强；
2. 放大到 2/4/8 **完全没有变化** ⇒ 当前**不是**靠增益就能救的欠尺度问题，
   而且 0.6923 低于 pooled 先验地板之上的可用空间，说明瓶颈不在幅度；
3. 强制到 10⁶ 反而掉到 0.5897——**并且没有复现 residual 自身的 argmax 命中率
   0.7379**。该不一致已逐格定位到**通道投影口径**，不是探针 bug、也不是掩蔽：
   head 的 `_expand_cls_logit_residual` 只把每类那**一个** logit 写进
   `conv_cls` 的 12 个「该类 anchor 槽 × 类 token」通道
   （`dst = [0,4,8,12,16,20,25,29,34,38,43,47]`，共 48 通道里的 12 个），
   其余 36 个通道保持 `conv_cls` 原值；而探针的 `class_scores_from_head` 是
   **先对同类全部 anchor 取 max、再对类取 argmax**。两者在 `k→∞` 时并不等价：
   放大后**负数 logit 会在该类自己的锚槽上被同倍放大向负**，而该类的其余
   anchor 槽仍保留原始 `conv_cls` 值，于是「max-over-anchor」不再把该类的最大值
   放在被残差改写的那 12 个通道上。逐格复核（clip 3，3 个前景格）：
   `residual` argmax = `[0, 1, 3]`，而按「只保留正项」预测的 argmax 是
   `[0, 1, 0]`，实测 head 强制臂给出 `[0, 1, 1]`——三个数不完全一致的原因正是
   上述「负项被放大 + 未被改写的锚槽参与竞逐」。⇒ 该强制臂**只是内部一致性
   检查，不构成独立判据**，已按此记录，不得用它论证「残差是对的/错的」。
   （这也**顺带解释** M4 的 `head_swap_to_donor` 为什么只有 0.1167：donor 残差
   同样只写 12/48 个通道，被 `conv_cls` 的 2.78 量级间隔与「max-over-anchor
   会把未改写通道一并纳入竞逐」双重稀释；**该低读数反映的是投影稀疏性 + 尺度**，
   而不是「头忽略 z」。精确的稀释比例**本次未单独测量**，故此处只写机制方向，
   不给出百分比。）

**对下一步的意义（这才是本节的目的）**：M4 的两个读数都已被解释到机械层面，
「读出不可用」不再是黑箱。候选 A 的失败因此收窄为一条**可检验的假设**：
残差与 `conv_cls` 的相对尺度/耦合方式是主要障碍（残余 25.3% 可翻格 + 关掉掉
5.1 pt + 放大不动）。这为「分类 residual 读出」的单变量验证给出了明确的
切入点与预登记所需的正确对照（先验地板、尺度曲线、以及 anchor 广播口径），
而不是再叠一层模块。

**边界**：只读探针（GPU 5，一次，未训练、未写 checkpoint、未改权重/判据）；
新增 `test_probe_semA_m4floor.py`（8 条）与 `_Sel` 的向后兼容 `stride` 参数。
§72 的 M4 判定（FAIL）**不变**，变的只是它的解释强度。

#### 71.9.19 语义头 train/val 差距的来源定位：**split，不是 BN 模式**（只读，§72.10 遗留项闭环）

§71.9.17 发现同一 checkpoint 在**训练口径**（train split、模型 train()）读
**0.4923/0.3586**、在**部署口径**（val split、模型 eval()）读 **0.1167/0.0833**，
差约 4.3×。§72.10 已排除 z 采样契约（切到 `mu_q+sigma*eps` 时 z 动 35.26%、
两项读数 **Δ=0.0000**）。剩余混淆项有两个且此前**从未分开测量**：

1. **数据 split**：train logger 读 train info、门控读 val info；
2. **归一化模式**：train logger 在 `model.train()` 下，z 路径的 4 个 BN
   （`temporal_fusion.encoder.0/1.bn`、`decoder.0/1.bn`）走 **batch 统计**；
   门控在 `eval()` 下走 checkpoint 的 **running 统计**。

本次新增只读探针 `me_rssm/sanity/probe_semA_bnmode.py`，把两者拆成独立可测的臂
（z 契约在所有臂都钉死为部署 `mu_q`）：**用同一条 val pipeline、同一个 eval 模式模型，
只替换 info 文件**，得到 split 的净效应；再在同一 split 内只把 BN 切到 batch 统计，
得到归一化模式的净效应（batch 统计按 batch 估计，故 batch 统计臂必须以真实 batch
运行；本组用 `batch-size=2`，与训练 `samples_per_gpu=2` 一致）。

**校准检查（否则探针的数字不能自证）**：

- eval 臂在 val split、n=10 复现冻结门控 **0.1167/0.0833**（逐位一致）；
- batch 中立性：eval 臂 bs=1 与 bs=2 读数**完全相同**（val 0.1167/0.0833、
  train 0.4480/0.5500），符合 eval 下 BN 用 running 统计的预期，说明**批量 rollout
  本身不是变量**。

**主对照（n=20，stride=100；两 split 各自跨 ~20 个独立场景块）**：

| split | 条件 | `thr50` | `hit@1` |
|---|---|---:|---:|
| train（**拟合过**） | eval / running stats | **0.4739** | **0.4167** |
| train | z 路径 BN → batch stats | 0.4851 | 0.3667 |
| val（**留出**） | eval / running stats（= M1 口径） | **0.1280** | **0.0417** |
| val | z 路径 BN → batch stats | 0.2452 | 0.1792 |

- **split 净效应**：同 pipeline、同 eval 模式，train 比 val 高 **+0.3459 `thr50` /
  +0.3750 `hit@1`**，train 侧读数已落到 train logger 自己那个数
  （0.4923/0.3586，后者是 train split + train()、且是 38 个窗口的 epoch 均值）附近。
- **归一化模式净效应（val 侧）**：只切 z 路径 4 个 BN，`thr50` +0.1173、
  `hit@1` +0.1375——**方向确实存在、幅度远不足以解释 4.3×**，且在 train 侧
  几乎不再移动（`thr50` +0.0112、`hit@1` −0.0500）。
- **范围对照（n=20，val）**：把 z 路径以外**同样 4 个** BN 切到 batch 统计，
  移动量接近零（−0.0012 / +0.0375）⇒ val 侧那个 +0.1173 是 z 路径局部的，
  不是「任何归一化扰动都会推动读数」（n=10 时这个范围对照曾给 −0.0167/+0.0500）。
- **干预检查**：所有臂都记录了「有多少 clip 的读数移动」；非零 ⇒ 切换确实生效，
  「两臂相同」不能反过来被读成「机制无关」。

**GT-shuffle 地板（防止把位置先验读成对象对齐）**：用**另一 clip** 的 GT 目标去评
本 clip 的语义图。两侧地板都塌到近零（val 三个臂全为 0.0000；train eval 臂
0.0213/0.0167，约为自身的 4.5%/4.0%）。⇒ **被测量的是对象对齐，而不是「图总在
BEV 常见区域起峰」**；这也说明上表的 split 差不是目标密度或空间先验造成的。

**结论（替换 §72.10 的「来源仍未定位」）**：train/val 两项读数的 4.3× 差距**主要由
数据 split 解释**（同权重、同 pipeline、同模式，train split 即近乎复现 train logger
读数），BN 归一化模式是**次要**因素，z 采样契约已排除。因此：

- train split 上的 `stat_sem_*` **不能**再被当作「同一模型在另一口径下的同量测量」；
  它是**拟合内**读数。§71.9.17 已据此把 M1 判为「真实未过」，本节的机制归属
  进一步说明：语义头在**训练见过的场景**上确实能找到 GT 中心，在**留出场景**上
  明显掉档。
- 这不改变 S1 判定（M1/M3/M4 FAIL ⇒ 归档、不进 ep16），也**不**证明 head 设计正确；
  它把「为什么 train 好 val 差」从**未定位**推进为**定位到泛化/留出场景**，
  给下一步重预登记一个明确的可测轴（跨场景泛化 vs 读出耦合），而不是继续
  扫增益、调 KL 或续训抢救。

**边界**：只读探针（GPU 5/6，未训练、未写 checkpoint、未改权重/判据）；n=20 的
两 split 各有 20 个独立场景块，但**不是**全 split 统计，也不构成多 seed 证据。
新增 `test_probe_semA_bnmode.py`（19 条，含范围对照「同数量 out-of-scope BN」、
batch 中立性、GT-shuffle 地板、以及只在 val split 才声称复现冻结门控的守卫）。
产物：`semA_bnmode_ep6.json`（n=10，四臂）、`semA_bnmode_ep6_trainsplit.json`（n=10）、
`semA_splitprobe_val_full_n20.json` / `semA_splitprobe_train_full_n20.json`（n=20，三臂）。

#### 71.9.20 语义头失败模式的分解：**排序会泛化，置信度与定位不泛化**（只读）

§71.9.19 把 train/val 差距定位到 split，但**没有说清失败机制**：M1 的两个读数
（`thr50` = GT 中心格 sigmoid>0.5 的占比、`hit@1` = 每个占用类通道唯一 argmax
是否落在 GT 中心格）**都是阈值读数**。一个「排序完全正确但整体置信度偏低」的
热图，与一个「排序本身就错」的热图，在这两项上都可能 FAIL，但对应的下一步实验
完全相反（改损失标定 / 改头看到的东西）。故本组在**不加权、不加前向**的前提下，
用与 §71.9.19 同一批 clip（n=20、stride=100、z 钉死部署 `mu_q`、模型全程 eval）
补两类**标度无关**读数：

- `auc` = P(GT 中心格概率 > 背景格概率)，按 clip 对 (格, 类) 聚合；对热图做
  **任意单调重标度**都不变。用**平均秩**做并列修正（该头对背景大量输出完全相等的
  概率，朴素 `argsort` 会把并列任意排序，给常量图任意 AUC）。
- `recall_at_top_k`：取该通道**自己的前 k 格**（k = 本通道 GT 中心格数，即头自己的
  峰预算）算命中率——「若把图重标定到阈值恰好切在前 k 格，能拿到多少召回」。
- 另有 `mean_gt_prob` / `mean_bg_prob`（标度读数）、`max_gt_prob`（阈值读数的硬上限：
  <0.5 时 `thr50` 不可能有个体通过）、`argmax_cityblock_dist_cells`（定位读数）。
- **对照（决定高 AUC 能否归因于对象）**：把**同一个热图**去评**另一个 clip 的 GT**。
  一张仅仅在 BEV 常见区域普遍抬高的图，对任何目标都能得高分；只有当 `auc(自己)`
  明显高于 `auc(shuffle)` 时，高 AUC 才是**对象特异**的。

**读数（ep6、seed0、A 区间：checkpoint `epoch_6.pth`）**：

| split | `auc` | GT-shuffle `auc` | `mean_gt_prob` | `mean_bg_prob` | `max_gt_prob` | `recall@top-k` | `thr50` | `hit@1` | argmax 距离(格) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| val（留出） | **0.9235** | **0.4684** | 0.1895 | 0.00141 | 0.2728 | 0.1024 | 0.1351 | 0.0417 | 34.43 |
| train（拟合） | 0.9994 | 0.5178 | 0.4447 | 0.00217 | 0.5924 | 0.3462 | 0.4739 | 0.4167 | 1.88 |

逐 clip：val 侧 `auc ≥ 0.8` **17/20**（中位 0.9883），`shuffle_auc ≤ 0.65` **15/20**；
`max_gt_prob < 0.5` **16/20**（中位 0.2072）；argmax 距离 ≤2 格仅 **7/20**。
train 侧 `auc ≥ 0.8` **20/20**（中位 0.9998）、argmax 距离中位 **1.0 格**（≤2 格 17/20）、
`max_gt_prob < 0.5` 仅 5/20。

**结论（这才是重预登记要用的）**：

1. **排序信息确实存在且对留出场景泛化**：val `auc` 0.9235 对 shuffle 地板 0.4684
   （≈0.5 = 无对象特异性），故 z 的语义头在**没见过的场景**上仍能把对象格排在背景格
   之上。这不是「头什么都没学到」。
2. **失败在标度与定位，不在「有没有信息」**：val 侧 16/20 个 clip 的**最好** GT 中心
   概率都 <0.5（中位 0.2072）⇒ 固定 0.5 阈值在这些 clip 上**结构上不可能**通过；
   同时 argmax 中位偏 12.2 格（train 侧 1.0 格）⇒ 逐通道**峰值位置**在留出场景上
   明显漂移，`recall@top-k` 也从 0.3462 掉到 0.1024。
3. 因此原 §72 的 M1 FAIL **既不是「没有语义」，也不是单纯的门槛口径问题**：
   `max_gt_prob` 这条硬上限说明它**不是**「只差一个标度」（16/20 clip 连最好的
   GT 格都没到 0.5，而 `recall@top-k` 也低于 train 侧 3.4 倍）。两个机制**同时**成立。
4. 这**不**改变任何已归档判定：M1/M3/M4 仍 FAIL、S1 仍 NOT ALL PASS、不进 ep16、
   z 采样与 BN 归属不变。变的只是**失败机制的表述**，以及据此可写的下一步判据。

**边界**：只读探针（GPU 5/6，未训练、未写 checkpoint、未改权重/判据）；n=20、
单 seed、ep6 早期 checkpoint，不是全 split 统计，也不是性能证据。`auc`/`recall@top-k`
是**补充读数**，**不进门**、不替代 `thr50`/`hit@1`。新增 `test_probe_semA_rank.py`
（12 条：并列修正的常量图=0.5、完美/反向分离=1/0、k 用该通道自身正例数、
GT-shuffle 必须换 clip、固定阈值读数必须来自门控的同一评分函数、只换 info 不换
pipeline）。产物：`semA_rank_val_ep6.json`、`semA_rank_train_ep6.json`。

#### 71.9.21 对 §71.9.20 的两处自我校正（只读，防止把机制说反）

写 A2 预登记时对 §71.9.20 的读数做了两次**反向检验**，两处都需要就地更正，
否则下一步实验会照着一个**说反了的机制**设计：

**校正一：`hit@1` 的分母不是「一格」。** 预登记初稿写过「高斯把中心摊开、
`hit@1` 只认唯一 argmax 格、故留出场景峰漂一格即全丢」。实测直接否掉这个说法：
`bev_centers_to_heatmap` 的半径是 `gaussian_radius_batch(...).clamp_min(min_radius)`，
在本 config（`min_radius=2`、格 0.32 m）下，**每个占用类通道的 GT 标记区有 4–59 格**
（val 42 个通道实例分布 `[4,4,4,8×11,13,14×5,17×7,21×3,22,25×2,29×2,30,34×2,51,58,59×2]`，
train 37 个 `[4×7,8×2,13,17×4,21×4,24×2,25×4,29×2,34,35×2,42×3,50,51×2,76,107]`）。
即 `hit@1` **比单格标注更宽容**，不是更苛刻。⇒ 「把目标收窄成一格/ROI」在方向上
是**错的**：那会让 argmax 命中条件变**难**。A2 的监督侧因此改为**排序型**
（要求该通道 GT 峰压过自己的背景峰），而不是收窄标记区。

**校正二：失败是「背景峰压过 GT 峰」，不是「GT 峰太窄」。** 逐通道核对
（val index 403、class 2）：`n_pos=2`、两格概率 0.624/0.651（**都在 0.5 以上**、
`thr50=1.0`），但该通道 `bgmax=0.720` > GT 峰，`topk_recall=0.000`
（前 2 格被背景占据）⇒ argmax 必然落到对象外。原稿把这归因于「峰漂」，
更准确的机制是**背景峰更高**；`thr50=1.0` 与 `recall@top-k=0` 同 clip 并存
正是这一机制的指纹，不是读数矛盾（背景峰把 top-k 抢走后，阈值召回仍可以是满的）。

**漂移是双峰的（供实验设计，不构成判据）**：val 20 个 clip 的 argmax 距离，
≤2 格 **7 个**、≤10 格 **10 个**、>20 格 **9 个**（最大 143）；train 侧 ≤2 格 **17 个**、
>20 格 **0 个**。`corr(mean_gt_prob, 漂移) = −0.508`（越不确定越漂），
远/近两组的 `mean_gt_prob` 为 0.1011 vs 0.3352。⇒ 配对比较必须**逐 clip**，
否则少数场景会淹没均值。

**边界**：纯只读核对（GPU 5，未训练、未改权重/判据），未新增探针或测试；
`hit@1` 标记区分布由 `bev_centers_to_heatmap` 在本 config 下直接枚举得到，
单通道背景峰核对用 `epoch_6.pth` 复核。§71.9.20 与 §72 的**判定均不变**，
变的只是 A2 监督侧实验的设计方向（从「收窄目标」改为「排序型目标」）。

#### 71.9.22 A2 监督侧门槛的第三个自我校正：剔除一个无区分力的统计量（只读）

写 A2 的 M1-loc 门时，初稿用的是「`bgmax > gtmax` 的通道数」。全量枚举后
该统计量**必须弃用**：

| split | `bgmax > gtmax` 的通道 | clip |
|---|---:|---:|
| val | **40/42** | 20/20 |
| train | **21/37** | 17/20 |

它是**序统计量的必然结果**而非模型性质：背景候选有 `53,568 − n_pos` 格、
GT 标记区只有 **4–59** 格，即使 GT 每格都优于 99.9% 背景，「背景最大值超过
GT 最大值」仍大概率发生。用它当门会得到一条 train/val 都几乎必然触发的线。

**改用 `recall@top-k`**（取该通道自己的前 k 格，k = 本通道 GT 标记格数）：

| split | 中位 | 均值 | `==0` 的 clip | `≥0.5` 的 clip |
|---|---:|---:|---:|---:|
| val | **0.0000** | 0.1024 | **11/20** | 1/20 |
| train | 0.3333 | 0.3462 | 4/20 | 6/20 |

它直接用「峰预算被谁占据」区分两 split，不随背景格数变化。故 A2 的
M1-loc 定为 **`max_gt_prob ≥ 0.5` 的 clip ≥12/20 且 `recall@top-k` 中位 ≥0.25**，
门槛数值在 A2 训练之前写定、**不回溯用于重判 A**。

**另一条被分开的现象**：§71.9.21 记的「argmax 漂移 >20 格」集中在 **9** 个
场景，而「峰被背景占满」覆盖 **11** 个场景且中位为 0 —— 两者不是同一现象的
两种说法，A2-S 的门挂在**后者**（更普遍、更贴机制）。

**边界**：只读枚举（GPU 5，未训练、未改权重/判据）；未新增探针或测试。
§71.9.20、§71.9.21、§72 的**判定均不变**。

## 72. 候选 A ep6 机制门（S1）：未全门通过（2026-10-10）

本节执行 `docs/rssm_candidate_A_prereg_20261010.md` §5 预登记的 S1 ep6
机制门。训练从零起（3 卡 GPU5/6/7、seed0、deterministic、24e 余弦日程），
ep6 val 与 `epoch_6.pth` 落盘后由看门狗 `me_rssm/sanity/watch_semA_ep6.sh` 中断训练、
等待 GPU 释放，再自动跑 M1–M4 门 + 三窗口反事实 + source 探针。

> **本节不宣称任何 S2 性能结论。** S2 是 ep12–16 五轮窗口，且其数值需
> 用户批准后才允许 ep6→ep16 续训（§5/§7）。本节只报 ep6 机制门与早期
> 曲线，二者不可互相替代。

### 72.0 状态与判定摘要

**三口径摘要**（AGENTS 规则 5：三个口径必须分别写清，不得用含糊的
“BEST” 代替其中任一项）：

| 口径 | 数值 |
|---|---|
| 区间均值（标准 ep12-16 窗口不完整；实际跨度 ep1-6，n=6） | **34.0459** |
| 全轮峰值（所有 val epoch 的最高单点，与是否落盘无关） | **38.2203 @ ep3** |
| best saved（仅表示 checkpoint 可用性，不参与性能排名） | **35.7608 @ ep6**（落盘 [2, 4, 6]） |

> 上述窗口均值是 **ep6 早期上下文**，不是 ep12–16 的 S2 判据；
> 单 seed、窗口未完整，不得据此解读效应。

**S1 门控表**（阈值来自 §5 预登记，判定来自
`me_rssm/sanity/semA_ep6_verdict.py`，本节不重算门槛）：

| 门 | 预登记门槛 | ep6 实测 | 判定 |
|---|---|---|---|
| M1 语义头质量 | val `thr50 ≥ 0.30` 且 `hit@1 ≥ 0.60` | `thr50` **0.1167**、`hit@1` **0.0833** | **FAIL** |
| M2 z 排他性 | `h` 与 `feat` 两臂**各自** ≤ −30% | `h` **-100.0000%**（head thr50 0.0000 vs z 0.1167）、`feat` **-100.0000%**（两臂均达线） | **PASS** |
| M3 读出非装饰 | 强制 residual=0 使正样本 class-CE 变差 ≥8/10（窗口冻结比例 8/10） | **7/10** | **FAIL** |
| M4 类别正确性 | `own_hit` 与 `donor_transfer` ≥ chance+20pt （bar 0.45） | `own_hit` **0.7379**、`donor_transfer` **0.3818**（chance 0.25） | **FAIL** |

**不进门补充读数（M1 口径）**：`hit1_chance` 均值 **0.000073**、`hit1_excess_over_chance` 均值 **0.0833**。
按 §71.9.3(2)，`hit@1 ≥ 0.60` 是从 route-C 的 16×16 网格外推而来，
A 的 full-res 216×248 网格随机基线只有 ~7.5e-5（209× 格数），
故这两个读数必须与门槛并排记录。

**§71.9.3(2) 例外条款：不适用。** 该条款要求 M1 *仅*在 full-res 原始 `hit@1` 上未过，而 `thr50` 与
`hit1_excess_over_chance` 都健康（即「头确实找得到目标中心，只是单格 argmax 跨不了网格」）。

**该判定不依赖口径选择**，两种读法都不满足例外：

| 读法 | `thr50` | `hit@1` | vs 门槛 |
|---|---:|---:|---|
| 预登记的 val 口径（本门实际读数） | 0.1167 | 0.0833 | `thr50` 未过 0.30、`hit@1` 未过 0.60 |
| A 自身 train 口径（`stat_sem_*` ep6 的 **epoch 均值**） | 0.4923 | 0.3586 | `thr50` **过** 0.30、`hit@1` 未过 0.60 |

例外条款要求「excess 健康」，val 口径下的对应量是 0.0833（bar 0.5840）；该口径下 `thr50` 为 0.1167（bar 0.30）。
**两项中至少一项未达线**，故 M1 是**真实未过**，不是跨网格口径造成的伪失败，不得以「口径未校准」豁免。

旁证（不是判据）：train 口径只记 `stat_sem_hit1`、未单独记 excess，而 `excess <= hit@1`，故 hit@1 = 0.3586 是该口径下 excess 的**上界**。
该上界 < bar 0.5840 ⇒ train 口径同样不满足例外，**该结论不建立在口径选择上**。

> **口径出处须与上表并读**（AGENTS 记录纪律）：`0.30 / 0.60` 这组门槛的先例 §59 route-C 报的是**train 侧 epoch 均值**（§60.5/§61.5，38 条 train 记录），而本门按 §5 预登记在 **val** 上测量，两者之间隔了一层清样；上表因此把两种读法并列，而不是拿 val 数字直接比 train 门槛。
> 上表两种读法是**稳健性并列**：val 是预登记的判据口径，train 只是旁证。若两者结论不一致，以 val 为准并显式记录分歧。

**S1 聚合判定：NOT ALL PASS（M1, M3, M4）。** 按 §7
分叉，按负结果归档并把读数写进 Analysis，**不进 ep16**。
**M2 通过：语义头对输入**敏感**，但它的证据强度到此为止。**
h 臂与 feat 臂的 `thr50` 均为 0.0000/0.0000，而 z 臂为 0.1167。

> **不得据此宣称「语义只存在于 z」**：M2 是拿**只被 z 监督过**的 head 去喂 h/feat，属于分布外输入——即使 h 或观测里含有同样的对象信息，这个 head 也读不出来。因此 M2 只证明「该 head 依赖它训练时的输入」，**不能证明** z 是唯一信息源。
> 预登记把真正的 z 排他性检验放在 **C3 臂**（`sem_head_input='feat'` 从零训练、同日程对照），本次**未执行**——故「概率状态」这一主张在本次测量中**既未被否证、也未被证实**。

### 72.1 配置与唯一变量

| 项 | 值 |
|---|---|
| 配置 | `me_rssm/configs/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_semreadout.py` |
| 对照 | `..._no2d_igdr.py`（`MotionAlignedRSSMFusion`, seed0） |
| 唯一变量 | `sem_head`（z 排他语义监督）+ `z_cls_logits`（零初始化分类残差读出） |
| 未改动 | 重建目标、posterior/prior、KL/free-nats、`output_proj`、backbone、head 几何、优化器、日程、数据管线 |
| 权重指纹 | `sha256=8077411c6142e7ce...`、size=688860204、mtime=1791665121 |
| 采样位点 | clip 索引 `[3, 203, 403, 603, 803, 1003, 1203, 1403, 1603, 1803]`、`shuffle_offset=1` |

### 72.2 运行事实

| 项 | 值 |
|---|---|
| ep6 段起止 UTC | 2026-10-10 14:50:03 → 2026-10-10 20:56:08（`20261010_145003.log`，263 条带时间戳行） |
| GPU / seed | 5/6/7、seed 0、deterministic |
| 吞吐 / 显存 | 中位 **1.566 s/iter**、16.7 GiB/卡（`*.log.json` train 段 228 条） |
| val 记录 | ep1-ep6 各一条，共 6 条 |
| 落盘 | [2, 4, 6]（`checkpoint_config.interval=2`） |
| 停止方式 | 看门狗在新鲜 `epoch_6.pth` + ep6 val 后中断训练（signal SIGINT(2)，退出码 1；中断是看门狗按 §5 的预期动作，非崩溃）· gate hit 2026-10-10 20:56:29 → GPU 释放 2026-10-10 20:57:15 → 探针结束 2026-10-10 20:58:57 |

### 72.3 完整逐 epoch 曲线（ep1-ep6）

| ep | Overall3Dmod | vs base | Car | Truck | Ped_l | Cyc_l |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 26.86 | -12.42 | 33.73 | 11.01 | 21.83 | 40.90 |
| 2 | 31.11 | -8.17 | 38.52 | 24.40 | 21.21 | 40.33 |
| 3 | 38.22 | -1.06 | 56.39 | 22.53 | 28.24 | 45.72 |
| 4 | 34.64 | -4.64 | 43.37 | 28.09 | 29.44 | 37.66 |
| 5 | 37.68 | -1.61 | 54.19 | 18.77 | 27.95 | 49.79 |
| 6 | 35.76 | -3.52 | 48.71 | 27.09 | 27.27 | 39.97 |

> `vs base` 对照 §5 的 seed0 基线 Overall **39.2809**（ep12–16 窗口均值）；单点差值不含 seed 噪声。

### 72.4 窗口逐项均值（实际跨度 ep1-6，n=6）

| 类别 | 本 run 窗口均值 | seed0 基线（ep12-16） | 差 |
|---|---:|---:|---:|
| Overall_3D_moderate | **34.05** | 39.2809 | -5.23 |
| Car_strict | 45.82 | 47.8399 | -2.02 |
| Truck_strict | 21.98 | 29.7246 | -7.74 |
| Ped_loose | 25.99 | 30.9277 | -4.94 |
| Cyc_loose | 42.39 | 48.6313 | -6.24 |

> 该窗口与基线窗口**不同长度**（基线段是 ep12-16，本 run 只有 ep1-6），
> 只作早期上下文，不作 S2 判据。

### 72.5 A 侧训练期同口径读数（train split，逐 epoch 末值）

| ep | `stat_sem_hit1` | `stat_sem_thr50` | `loss_rssm_sem` | `stat_cls_res_rms` | `stat_mu_diff_sq` |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.2253 | 0.1989 | 1.2632 | 0.9707 | 3.6267 |
| 2 | 0.2382 | 0.2719 | 1.0536 | 0.9616 | 0.2650 |
| 3 | 0.2840 | 0.3513 | 0.9106 | 0.8274 | 0.0587 |
| 4 | 0.3324 | 0.4154 | 0.7895 | 1.0124 | 0.0463 |
| 5 | 0.3703 | 0.4658 | 0.6986 | 1.1601 | 0.0766 |
| 6 | 0.3526 | 0.5022 | 0.6625 | 1.0570 | 0.0361 |

shuffle/zero 与门控读数无关；此处 `stat_sem_*` 为**训练统计**（z 采样值、train split），与探针的 val 口径不同，不可混比。
末 epoch `stat_cls_res_rms` = 1.0570（>0 ⇒ 读出活着）。

### 72.6 S1 门明细（逐 clip）

> **样本权重不均提示**：各 clip 的 GT 目标数并不相等——clip `1603` 只有 **1** 个目标，clip `603` 有 **11** 个（§71.9.14 预检：3/4/3/11/4/2/6/6/1/2）。
> 逐 clip 比率在 `1603` 上只能取少数离散值、方差最大，
> 故下面 M1/M3/M4 的**分组结论不得由单个 clip 单独支撑**。

**M1 逐 clip**：

| index | thr50 | hit@1 | hit1_chance | excess_over_chance | 正格数 |
|---:|---:|---:|---:|---:|---:|
| 3 | 0.0000 | 0.0000 | 0.000056 | -0.000056 | 3 |
| 203 | 0.0000 | 0.0000 | 0.000056 | -0.000056 | 3 |
| 403 | 1.0000 | 0.0000 | 0.000037 | -0.000037 | 2 |
| 603 | 0.0000 | 0.5000 | 0.000205 | +0.499795 | 11 |
| 803 | 0.0000 | 0.0000 | 0.000056 | -0.000056 | 3 |
| 1003 | 0.0000 | 0.0000 | 0.000037 | -0.000037 | 2 |
| 1203 | 0.0000 | 0.0000 | 0.000112 | -0.000112 | 6 |
| 1403 | 0.1667 | 0.3333 | 0.000112 | +0.333221 | 6 |
| 1603 | 0.0000 | 0.0000 | 0.000019 | -0.000019 | 1 |
| 1803 | 0.0000 | 0.0000 | 0.000037 | -0.000037 | 2 |

**M2 两控制臂逐 clip**（`thr50`；z 为参照）：

| index | z | h | feat |
|---:|---:|---:|---:|
| 3 | 0.0000 | 0.0000 | 0.0000 |
| 203 | 0.0000 | 0.0000 | 0.0000 |
| 403 | 1.0000 | 0.0000 | 0.0000 |
| 603 | 0.0000 | 0.0000 | 0.0000 |
| 803 | 0.0000 | 0.0000 | 0.0000 |
| 1003 | 0.0000 | 0.0000 | 0.0000 |
| 1203 | 0.0000 | 0.0000 | 0.0000 |
| 1403 | 0.1667 | 0.0000 | 0.0000 |
| 1603 | 0.0000 | 0.0000 | 0.0000 |
| 1803 | 0.0000 | 0.0000 | 0.0000 |

**M3 逐 clip**（`delta_off_minus_on > 0` 表示关掉 residual 后
正样本 class-CE 变差，即读出有贡献）：

| index | CE(on) | CE(off) | delta(off-on) | 正格数 |
|---:|---:|---:|---:|---:|
| 3 | 1.4444 | 1.2569 | -0.1875 | 3 |
| 203 | 0.7663 | 0.9878 | +0.2215 | 3 |
| 403 | 0.0057 | 0.0620 | +0.0563 | 2 |
| 603 | 1.3545 | 1.1498 | -0.2047 | 11 |
| 803 | 0.4506 | 0.8692 | +0.4185 | 3 |
| 1003 | 0.9078 | 0.7337 | -0.1741 | 2 |
| 1203 | 0.4414 | 0.7333 | +0.2919 | 6 |
| 1403 | 0.3735 | 0.5600 | +0.1864 | 6 |
| 1603 | 0.0231 | 0.2300 | +0.2069 | 1 |
| 1803 | 1.4699 | 1.6358 | +0.1659 | 2 |

**M4 逐 clip**（`own_hit` = z 残差 argmax 命中该格 GT 类；
`donor_transfer` = 换 donor z 后残差 argmax 命中 donor 主类；
`head_swap_to_donor` = 只重跑 head 且喂 donor 残差时的 head 类 argmax）：

| index | own_hit | donor_transfer | head_swap | donor_index | changed_frac |
|---:|---:|---:|---:|---:|---:|
| 3 | 0.3333 | 0.0000 | 0.0000 | 203 | 0.6667 |
| 203 | 0.6667 | 1.0000 | 0.0000 | 803 | 1.0000 |
| 403 | 1.0000 | 1.0000 | 0.0000 | 803 | 1.0000 |
| 603 | 0.5455 | 0.8182 | 0.0000 | 803 | 0.9091 |
| 803 | 0.6667 | 0.0000 | 0.3333 | 1003 | 0.3333 |
| 1003 | 0.5000 | 0.0000 | 0.0000 | 1203 | 0.0000 |
| 1203 | 0.8333 | 0.1667 | 0.0000 | 1403 | 0.5000 |
| 1403 | 0.8333 | 0.3333 | 0.3333 | 1603 | 1.0000 |
| 1603 | 1.0000 | 0.0000 | 0.0000 | 1803 | 1.0000 |
| 1803 | 1.0000 | 0.5000 | 0.5000 | 3 | 0.5000 |

### 72.7 家族标准反事实读数（`shuffle_z` / `zero_z` / `prior_only`）

三窗口与既往各代（MAI / MCI / DC-RSSM）使用同一探针语义：
`shuffle_z` 用同 batch 另一样本的 `mu_q` 替换，`zero_z` 置 0，
`prior_only` 只走先验。**A 跑标准家族**，故用
`probe_z_utilization_standard.py`（后验头重写 + 转发部署态 class
residual），与 `tools/diagnose_z_utilization.py` 的打分口径一致。

| 窗口 | shuffle_z head | zero_z head | prior_only head | shuffle/zero | shuffle_z feat | 标准 RSSM ep6 同窗 head（对照） |
|---|---:|---:|---:|---:|---:|---:|
| 近距（[3, 4, 5, 6]） | **2.6509%** | 30.2447% | 5.5026% | 0.088 | 11.1105% | 49.73% |
| 中距（[100, 101, 102, 103]） | **7.4311%** | 41.6922% | 4.6106% | 0.178 | 23.3256% | 3.30% |
| 远距（[1800, 1801, 1802, 1803]） | **4.2138%** | 43.5807% | 4.5805% | 0.097 | 14.4418% | 3.36% |

本 run 三窗 `shuffle_z` head：近距 2.65%; 中距 7.43%; 远距 4.21%。
对照为**标准 no2d_igdr RSSM seed1 ep6**同窗口（`docs/training_runs_full.md` §62.2），因同家族、同 epoch，
是唯一口径一致的机制对照；MAI/MCI/DC 各代的同窗读数见 §62/§63/§68。


### 72.8 posterior source 归因（`stride=700`）

| 量 | RMS | batch-common RMS | 样本特异 RMS | 样本特异能量占比 |
|---|---:|---:|---:|---:|
| `mu_q` | 0.5782 | 0.5410 | 0.2041 | 12.45% |
| `h_t` | 0.2150 | 0.2051 | 0.0645 | 8.99% |
| `e_pooled` | 0.6458 | 0.4452 | 0.4678 | 52.48% |

| 读出/依赖读数 | 值 |
|---|---:|
| `mu_q_removal_of_h_rel_l2` | 0.8056 |
| `mu_q_removal_of_e_rel_l2` | 0.5311 |
| **h/e 依赖比** | **1.517** |
| `posterior_obs_mode` | `l2` |
| `z_proj_init` | `None` |

> h/e 依赖比是各代通用的机制对照量（标准主线 ep12 为 2.338、MAI ep12 为 0.842、MCI ep12 为 0.605）；越低表示 posterior
> 越依赖观测而非 GRU 状态。

fusion 类：`SemanticReadoutRSSMFusion`、latent [216, 248]、readout `film`、clip 索引 [3, 703, 1403]。

### 72.9 判定与下一步（按 §7 预登记分叉，人工步骤）

1. S1 未全过 ⇒ 按负结果归档，读数写进 Analysis，不投 ep16。
   实测原因：M1 `thr50` 0.1167/`hit@1` 0.0833（例外条款不适用，真实未过）；M3 仅 7/10；M4 `donor_transfer` 0.3818（bar 0.45）。
   归档的是**本次候选 A 的读出/监督设计**，不是「z 无信息」（M2 只证明该语义头依赖其训练输入；z 的排他性**既未被否证、也未被证实**，见 §72.0 与 §71.9.17(3)），也不是「所有语义监督/残差读出方向」（§7 只据此否定本设计，未否定候选 A/B 所代表的方向）。
2. 任意 Ped/Cyc 触到非劣线 ⇒ 停止，记为家族副作用第 6 次复现。
   > **口径更正（2026-10-11）**：本行写在 ep6 归档节里，容易被读成 ep6 的止损线。
   > 该非劣线属 **S2（ep12–16）**；A 停在 S1，S2 从未形成，故本代**从未触发**这条线
   > （见预登记 §7 的范围更正与 A2 预登记 §5「早期门与成熟门严格分离」）。
3. **不做**权重/KL 扫描，**不做** ep24 抢救，不靠放大系数救活机制。

### 72.10 z 契约对账（训练采样 z vs 部署均值 z，只读）

归档审计发现：M1/M3/M4 是在**部署契约**下读的，而 `0.30/0.60` 门槛来自 **train 契约**下的诊断。两者差一个状态采样：

```
R4Det.extract_feat:  rssm_deterministic = not self.training
rssm_fusion.forward:  z_t = mu_q if deterministic else self.sample(mu_q, logstd_q)
```

即训练时语义头吃 `mu_q + sigma*eps`、eval 探针吃 `mu_q`。故用同一 checkpoint、同一 val clip、同一评分函数，只切换 z 契约重测一次：

| z 契约 | `thr50` | `hit@1` | `hit1_excess_over_chance` |
|---|---:|---:|---:|
| 部署（`mu_q`，即 M1 实际读数） | 0.1167 | 0.0833 | 0.0833 |
| 训练（`mu_q + sigma*eps`） | 0.1167 | 0.0833 | 0.0833 |

**干预检查**（否则「两臂相同」既可能是契约真无影响，也可能是切换没生效——结论相反、数字相同）：

* `z` 相对变化均值 **35.261%** —— 契约切换**确实改变了 z**；
* 语义热图相对变化均值 **1.348%** —— 经 `sem_head` 后幅度被大幅吸收。

噪声占比 `sigma/mu`：均值 **0.3637**（min 0.3473 / max 0.3816），即噪声项确实与被解释量同量级。

**结论**：契约切换确实移动了 z，但**两项门控读数完全不变**（Δ`thr50` = +0.0000）——
故 train/val 的 4.3× 差距**不能**由 z 采样解释。这**不改变** S1 判定（§72.0 已按两种口径分别核过），但把「契约错配」从嫌疑名单上划掉。

**来源归属（只读补测，n=20 / 20）**：把 z 契约与 eval 模式都钉死后**只换数据 split**——

| split | `thr50` | `hit@1` | 只切 z 路径 4 个 BN 的 Δ`thr50` | Δ`hit@1` |
|---|---:|---:|---:|---:|
| train（拟合过） | **0.4739** | **0.4167** | +0.0112 | -0.0500 |
| val（留出） | **0.1280** | **0.0417** | +0.1173 | +0.1375 |

⇒ 该差距**主要来自数据 split**（thr50 +0.3459、hit@1 +0.3750），归一化模式是**次要**项。即：train 侧 `stat_sem_*` 是**拟合内**读数，不能当作同一模型在另一口径下的同量测量。

> 产物：`semA_zcontract_ep6.json`（只读探针 `me_rssm/sanity/probe_semA_zcontract.py`，未改权重、未启动训练）。

### 72.11 本节边界

* 无 S2 性能宣称；§72.0/72.4 的均值是 ep6 早期上下文。
* 未改模型、未改判据数值、未动重建目标/KL/权重/日程。
* 本节由 `me_rssm/sanity/semA_ep6_record.py` 从证据 JSON 与
  `*.log.json` 机械生成，未手抄任何读数；判定沿用
  `semA_ep6_verdict.py` 的同一套门槛常量。

---

## 73. 建议落实进度复核（2026-10-11；CPU 只读验证，无新训练）

**范围**：对照 2026-10-10 建议的“真实目标审计→有限验证/按机制分叉→A→条件触发 C→B/D 后置”，
核验 §70–72、TODO、真实日志/证据 JSON，并运行只读判定器。不修改模型、不启动训练、
不重跑 GPU 探针；文档记录与进度查询不是性能验收。

### 73.1 已完成与未完成

- 真实训练目标/梯度/共享增强契约审计完成（§70.1–70.5）。
- 原差分目标预登记草案及 raw fullres 分散采样复测完成（§70.6、目标改革草案 §3A）：
  原立项依据不成立，按规则不投原样 ep6。真实 current 56.5% 特异、diff 93.7%
  近 i.i.d. 噪声基准；diff 总能量仅 current 3.96%。归档的是原设计的依据，
  不是永久证伪所有差分目标。没有差分正式训练。
- 候选 A 已实现 z-only CentreNet 语义监督与分类 residual，保留原重建/几何/状态；
  单测、S0 身份/梯度/存活和 450 iter 冒烟完成（§71）。形式不是 ROI 池化版本，
  而是全分辨率中心热图及 anchor 映射，是同一候选方向的一次具体实现。
- 候选 A seed0 ep1–6 实训与完整 S1 探针完成（§72）；M1/M3/M4 FAIL、M2 形式 PASS，
  总 NOT ALL PASS；按预登记停止、不进 ep16。概率状态排他性尚未证明。
- 后续两项只读机制审计已完成：采样/均值 z 切换与 M4 类别先验/尺度/anchor 投影
  口径（§72.10、§71.9.18）。前者没有改变门控读数；后者说明 residual 有局部
  分类贡献，但简单放大 2–8 倍没有改善所测分类命中。
- 未完成：A 的 C1/C2/C3 匹配训练消融、真正 z 信息排他性、同容量确定性对照、
  标准 ep12–16 性能窗、多 seed 四类稳定提升。C/B/D 仍研究建议，未进入正式验证。
- **（2026-10-11 追加）**语义头 train/val 4.3× 差距的归属已闭环（§71.9.19，只读）：
  主要来自**数据 split**（同 pipeline/同 eval 模式：train 0.4739/0.4167 vs
  val 0.1280/0.0417，n=20），BN 归一化模式是次要因素，z 采样契约早已排除。
  即上表「语义监督为什么 train 好 val 差」不再列为未定位项；剩余的是**设计层面**
  的「留出场景泛化不足与 anchor/readout 耦合」，仍待重预登记后的新实验。
- 检查时未发现本项目 semA/DC 续训进程；其他用户训练进程不属于本任务。

### 73.2 数值上下文与公平的早期对照

A 区间均值：标准 ep12–16 未形成；实际 ep1–6、n=6 **34.0459**。
全轮峰值：**38.2203 @ ep3**（无该轮权重）；best saved **35.7608 @ ep6**。
完整逐 epoch 与类别曲线见 §72.3–72.8，本次无新 epoch。

此前 §72 表中的 ep1–6 vs 基线 ep12–16 只可读作平台背景，不能当早期性能退化。
本次从两条 seed0 原日志重新配对 **相同 ep1–6，各 n=6**：

| 指标 | 候选 A | no2d_igdr 同 seed 同窗 | Δ |
|---|---:|---:|---:|
| Overall_3D_moderate | 34.0459 | 29.0420 | +5.0039 |
| Car_3D_moderate_strict | 45.8188 | 39.9218 | +5.8970 |
| Truck_3D_moderate_strict | 21.9831 | 19.7642 | +2.2188 |
| Pedestrian_3D_moderate_loose | 25.9889 | 19.7073 | +6.2816 |
| Cyclist_3D_moderate_loose | 42.3928 | 36.7749 | +5.6179 |

ep6 单点 A 35.7608 vs baseline 34.0243，Δ **+1.7365**。这是早期同 seed 描述性
信号，不是 S2 性能通过、显著性或三 seed 主张；新模块/辅助目标与状态机制贡献
尚未通过匹配消融分离。S1 不通过的依据是 M1/M3/M4，不是跨窗口 AP 比较。

### 73.3 当前验证命令与结果

```
rtk proxy python3 me_rssm/sanity/semA_ep6_verdict.py --run work_dirs/fgfull_N4_no2d_igdr_2x4_24e_semA_seed0
```

CPU 只读退出 0：五个预期证据产物存在且检查通过；M1 thr50 **0.1167**、hit@1
**0.0833**（FAIL）；M2 h/feat 替换均 −100%（形式 PASS，但分布外输入不证明 z
排他性）；M3 **7/10**（FAIL）；M4 own_hit **0.7379**、donor_transfer **0.3818**
（FAIL）。判定器仍打印 ep12–16 旧 S2 数值作为规则背景，不构成续训授权；TODO
已说明随当前归档不启用。没有重跑全套测试/训练，既有测试结果沿用 §71 记录。

### 73.4 完成程度与下一动作边界

**执行链已推进到 A 的 ep6 机制验证及负结果归因；修复 z、证明随机状态必要、
四类稳定性能提升均未完成。** 不能把“代码+探针完成”计作“方法成功”。

**（2026-10-11 追加）**用户指令的「重新预登记，定位语义泛化与 anchor/readout
耦合」已落地为 `docs/rssm_candidate_A2_prereg_20261011.md`（草案，§5 数值门
**待批准**；未实现代码、未投训练）。它把 §71.9.20 测到的**两个可分离缺陷**
拆成两条串行单变量对照：**A2-S**（监督侧：逐格高斯 → 对象级池化，唯一变量
`sem_target_mode`）先做，**A2-R**（读出侧：4→4 广播 12/48 通道 →
按 `anchor_class_mapping` 写满 6 个 anchor 槽，唯一变量 `cls_readout_mode`）
后做；两者都不放大增益、不扫权重/KL、不续 ep16。A2 的 S2 数值与 A 的 S2
**逐字相同**（避免事后调门），且新增的 `auc`/`max_gt_prob` 只作为 **A2 自己**
预先写定的诊断门（M1' / M1-loc），**不得回溯重判 A**。

本次只写预登记与只读归因；**A 的 ep16 仍未授权、仍不启动**。


### 73.5 A2 预登记技术审查（2026-10-11；只读，不授权训练）

- 用户报告新增测试 bnmode 19 + rank 12 = 31，全量回归 287 passed。本次未重跑
  测试，只核对当前 A2 草案、代码通道映射与既有归档；不把测试通过等同方法成功。
- 既有 A 上下文不变：区间均值实际 ep1–6（n=6）**34.0459**；全轮峰值
  **38.2203 @ ep3**；S1 NOT ALL PASS，ep16 不启动。新 A2 无代码/训练/AP 结果。
- 技术方向同意拆成 A2-S 与 A2-R 串行、分别对照原 A，不同时改两项。S2 的
  ep12–16 Overall≥40.00、Car≥47.34、Truck≥29.22、Ped≥29.93、Cyc≥47.63
  可保留作预登记筛选线，不是显著性或四类严格提升证明，也不构成训练授权。
- **现稿尚不能作为可执行训练契约**，四处须在新数据前补齐：
  1. A2-S 的排序项没有固定 q、margin、系数及归约/空通道处理。“不改权重尺度”
     不能抹去新增 loss 的梯度预算；应一次固定并记录两项梯度，不做扫描。
  2. A2-R 的 6 size anchors × 2 rotations × 4 class logits = 48 通道映射需写成
     公式/布局。目前代码每个 anchor 只写其 mapping 类的一格（12/48）；仅再次
     使用相同 mapping 不能自然写满。需明确每个槽接收哪些 class residual，并
     保持同一 4-channel predictor/参数量，否则引入容量变量。
  3. 固定 n=20 或把 12/20 明确为≥60%有效 clip；写明取样列表、无正例/无有效历史
     剔除、类别分层、top-k 的 k 与 tie 处理、GT-shuffle/donor 规则。M1' 与 M1-loc
     的均值/中位数/计数不可混用，旧 n=20 数字只作设计依据。
  4. §7 的“任一门 Ped/Cyc≤非劣线”必须限于 S2 ep12–16 固定窗口，不能拿成熟
     平台阈值判 ep6 早期。边界上原 GO 定义≥而停止定义≤相撞，等号亦须统一。
     若需 ep6 任务止损，只能另写同 epoch/同 seed 口径，不隐式套 S2。
- 补充：机制 FAIL 只否定当前实现/预登记假设，不能从“两臂不过”唯一推断 z 信息
  不足；peak 排序能泛化也不证明 RSSM 或概率状态必要。C3/确定性匹配对照仍缺。
- 本次仅给技术审查，不改 A2 数值或状态、不实现代码、不使用 GPU、不启动 S/R。
  总体批准与任何正式运行授权均仍未完成；修订后再审。

### 73.6 A2 四项契约补齐、代码实现与 S0 结果（2026-10-11；S0 已完成，ep6 训练中）

本节回应 §73.5 的四点技术审查，并记录补齐后的**代码实现**与**S0 接线门**。
范围：仅 A2-S / A2-R 两条**串行单变量**臂；不改 A 的任何归档判定。

#### 73.6.1 四项契约如何补齐（对应 §73.5 的 1–4）

1. **排序损失完整定义**（预登记 §4.1）：`g_c = max_{i∈P_c} s_c[i]`、
   `q_c = Quantile_{0.999}({s_c[j] : j∈B_c})`、
   `L_rank = (1/|C_occ|)·Σ_c relu(margin + q_c − g_c)`。固定 `q=0.999`、
   `margin=1.0`、`λ_rank=0.1`；**归约**为先按通道、再在占用通道上算术平均；
   `P_c=∅` 的通道整体跳过，`C_occ=∅` 时 `L_rank=0`（与 focal 的
   `num_pos==0` 分支一致）。
   **数值依据来自实测**（A 的 ep6 权重、eval 口径 `mu_q`、同一批固定 20 clip，
   只读落盘 `semA_lrank_magnitude_ep6{,_trainsplit}.json` 与
   `semA_lrank_margin_activation_{trainsplit,val}.json`）：

   | split | focal（逐 clip 等权） | `L_rank`(q=.999,m=1.0) | 比值 |
   |---|---:|---:|---:|
   | train | **0.6945** | **0.0932** | 0.134 |
   | val | **3.1019** | **0.9645** | 0.311 |

   ⇒ `λ_rank=0.1` 使新增项为 focal 的 **1.34%/3.11%**，不改变主损失量级。
   活跃比例（`relu(margin+q_c−g_c)>0`，**逐通道实例池化**）：
   train `margin=0` 仅 **1/37 = 2.7%**、`margin=1.0` 为 **7/37 = 18.9%**；
   val 为 **15/42 = 35.7%** → **22/42 = 52.4%**。两侧都远未饱和。

2. **A2-R 通道映射**（预登记 §4.2）：实测 `num_classes=4`、`num_rot=2`、
   `num_sizes=6` ⇒ `num_anchors=12`、`cls_out_channels=48`；
   `anchor_class_mapping=[0,0,0,1,2,3]`；既有映射实际写出
   `[0,4,8,12,16,20,25,29,34,38,43,47]` = **12/48**。公式：
   `per_class_slot: dst[a]=a*4+acm[a//2]`（12 通道） vs
   `anchor_slot: dst[a*4+c]=r[:,c]`（**全部 48 通道**）。
   **同一 `nn.Conv2d(256,4,1)`、同一参数量**，zero-init 契约不变。
   `num_anchors` 由 `R4Det._bind_cls_readout_anchor_slots()` 在首次 forward 前
   从检测头读取，**不在 config 里重抄常量**（避免与 `cls_out_channels` 漂移）。

   **口径更正（本轮自查发现并写入预登记）**：初稿把
   `get_bboxes_single` 的 `scores.max(dim=1)` 读成「在 12 个 anchor 上取
   max/argmax」，**是错的**。实测该 reshape 的展开顺序是
   `(cell, anchor, class)`（用已知张量逐项核对 linear index 通过），
   `scores.max(dim=1)` 只用于 `nms_pre` 的 topk 筛选；**不同 anchor 槽是彼此
   独立的候选框，各自带自己的类别分数进多类 NMS，不存在跨 anchor 的 max**。
   因此 `frac_cells_residual_can_flip = 0.2530` 是 §71.9.18 探针
   `class_scores_from_head`（先对同类 anchor 取 max 再对类取 argmax）的
   **代理口径**，与推理语义**不等价**，A2-R 不得把它当成推理级预期命中率。
   §71.9.18 的 M4 判定与其「读出被稀释」的机制结论**不变**。

3. **采样与统计契约**（预登记 §5）：固定 `--start-index 3 --stride 100
   --limit 20`，**分母冻结**；选中的 20 个 index 必须与
   `semA_rank_*_ep6.json` 的 `indices` **逐位相同**（本次核对：
   两 split 均为 `[3,103,...,1903]`）。「无有效历史」是**选样规则**而非事后
   剔除（`select_clips` 在 stride>1 时对不满足 `sum(mask[:-1])>=2` 的 index
   **跳过并 +1 续找**）；事后有效性只看 `C_occ=∅` 与语义图缺失，实测 20 个
   clip **无一触发**。`top-k` 的 `k` = 该通道 GT 格数（逐通道），
   `torch.topk` 并列取索引较小者；`recall@top-k` 逐通道→通道等权→clip→**中位**。
   均值/中位/计数逐门写死（`thr50`/`hit@1`/`auc`/`max_gt_prob`/`recall@top-k`
   不得混用）。
   **类别覆盖（实测，报告字段）**：val 20 clip 中 Ped **5**、Cyc **9**、
   Car **18**、Truck **10**；train 同协议 Ped 3、Cyc 10、Car 16、Truck 8。
   ⇒ **Ped 仅覆盖 5/20**，任何 Ped 专属 ep6 结论只能是**弱证据**。
   另核对：探针的 `occupied_channels` 与数据集 GT 类别数**逐 clip 完全一致**
   （val/train 各 20/20 匹配，0 mismatch）。

4. **早期门与成熟门分离、等号统一**（预登记 §5/§7）：§7 的「Ped/Cyc 低于
   非劣线即停」**只适用于 A2-S2/A2-R2 的 ep12–16 固定窗口**；ep6 的
   M1/M1'/M1-loc/M3/M4 **不得**套 S2 数值。等号统一为 **GO `≥`、停止 `<`**，
   `Overall == 38.78/40.00` 分别归中间区/GO，不落 FAIL。

#### 73.6.2 代码实现（本轮新增，均已单测）

- `me_rssm/semantic_readout_rssm.py`
  - `sem_target_mode ∈ {gaussian_full, channel_contrastive}` + `_channel_rank_loss()`
    （§4.1 的逐字实现；`torch.quantile` 直调，不自写排序）；
  - `cls_readout_mode ∈ {per_class_slot, anchor_slot}`（§4.2 的 4→48 展开）。
  - **两臂默认值都等于候选 A**，故未修改的 config 仍逐位是 A。
- `mmdet3d/models/detectors/R4Det.py`：`_bind_cls_readout_anchor_slots()`
  在 fusion 首次 forward **之前**绑定 `num_anchors`；`anchor_slot` 在读到该值
  之前**直接报错**而不是猜（猜错只会静默写错通道）。
- 两个 24e 臂 config + 两个 450-iter 冒烟 config + `watch_semA2_ep6.sh`
  （按预登记的 **20-clip 冻结契约**取门，按 PID 停训，非 tmux）。
- **单测**：`me_rssm/sanity/test_semantic_readout_rssm.py` 新增 T14–T19，
  与本文件原有 T1–T13 合计 **18 passed**（CPU，小网格，无 checkpoint）：
  T14 默认值即 A + 非法臂名报错；T15 排序项算术与预登记公式一致且**单侧**；
  T16 归约是按占用通道、全背景帧为 0（并验证无正例样本不稀释同批正例）；
  T17 排序项梯度确实到 z、且 focal 与其诊断量**逐位不变**（`λ=0` 复现 A）；
  T18 `anchor_slot` 写满 48 通道且**参数量与 A 完全相同**；
  T19 两臂 zero-init ⇒ 头侧残差恒 0。

#### 73.6.3 S0 接线门（450 iter，无 val、无 AP 含义）

两臂各跑 450 iter（3 卡 GPU5/6/7，seed0，deterministic），并各跑一次只读
wiring 探针。**两个 3 卡作业曾同时投在同一组 GPU 上导致 A2-R OOM**（证据：
`/tmp/a2r_smoke450.log` 中 6 处 `torch.cuda.OutOfMemoryError`，三个 rank 分别在
`rssm_fusion.py::kl_loss` 与 `modulated_deform_conv` 上失败；该 run 目录的
`20261011_0759/0800/0803` 三次日志都只跑到 iter 2 就中断），改为
**串行**后通过（该次 OOM 的原始日志与本次修复说明保留在同一处；此教训已记入
下方运行事实）。

| 读数 | A2-S | A2-R |
|---|---:|---:|
| S0-b 身份（基线 ckpt 加载，`max_abs_cls_diff`/`max_abs_reg_diff`） | **0.0 / 0.0 → PASS** | **0.0 / 0.0 → PASS** |
| `readout_zero_at_load` | True | True |
| S0-d 压力 `‖∂sem/∂z‖ / ‖∂tot/∂z‖`（门 ≥5%） | **0.8961 → PASS** | **0.8996 → PASS** |
| S0-e readout live | PASS | PASS |
| S0-e 语义项不动 readout | PASS | PASS |
| `loss_rssm_sem` 首→末（iter 50→450） | 21.60 → **2.36** | 21.42 → **2.25** |
| `loss_rssm_recon` 首→末 | 1.0497 → **0.2701** | 1.0617 → **0.2601** |
| 非有限迭代数 | **0** | **0** |
| `iter_450.pth` 落盘 | 是 | 是 |

**A2-S 专属：排序项梯度预算（预登记要求"只记录、不设通过线"）**

| 读数 | 值 |
|---|---:|
| `L_rank × λ` 值（batch） | **0.1957** |
| `‖∂L_rank/∂z‖ / ‖∂L_sem/∂z‖` | **0.1575** |
| `‖∂L_sem/∂z‖ / ‖∂L_total/∂z‖` | **0.8961** |

⇒ 排序项在 z 上确实有**独立于 focal 的**梯度份额（约 15.8%），且语义项整体
对 z 的压力高达 89.6%。**两者都不设线**：设置通过线会让"预算变了"成为事后的
解释借口；记录下来只为了让"排序项是否真的在 z 上工作"可核。

**A2-R 专属：`anchor_slot` 臂的残差确实到达头**（`residual_emitted=True`，
grid `[248,216]` 与 `pts_feats` 一致；零初始化下 head 侧残差 absmax = 0）。

**运行事实**：两臂均**从零起训**（非热启动基线权重）；3 卡 GPU5/6/7；
`find_unused_parameters=True` 是既有训练脚本默认，未改。

#### 73.6.4 当前状态与边界

- **已完成**：四项契约补齐、代码实现、18 条单测、两臂 450-iter S0（全 PASS）。
- **进行中**：**A2-S 24e 正式训练**（seed0、deterministic、GPU5/6/7），
  发射 2026-10-11 09:04:45 UTC，`--work-dir` 为
  `/data/lurui/work_dirs/fgfull_N4_no2d_igdr_2x4_24e_semA2S_seed0`，日志
  `/tmp/a2s_24e.log`，ep6 看门狗已挂 `watch_semA2_ep6.sh`（PID 3070972）。
  **区间均值**：ep12–16 尚未形成（A2-S 仍在早期）；截至记录时唯一 val 为
  **ep1 = 28.5104**（n=1，仅作训练健康度上下文，**不构成性能结论**）。
  **全轮峰值**（实为当前唯一 val 点）：**28.5104 @ ep1**，无该轮以外记录；
  **best saved**：截至记录时**尚无权重落盘**（`checkpoint_config.interval=2`，
  即 ep2/4/6/… 落盘；ep1 无权重，故此刻确实没有 best saved —— 这是**当前时点**
  的事实，不是"末位保存"，记录时应按真实落盘列表写）。
  训练健康度：iter 50→1750（ep2），无 NaN/OOM，累计显存 17106 MiB，
  `loss_rssm_sem` 1.078、`stat_cls_res_rms` 0.873（读出已离开 0，
  与 S0 的 zero-init 契约一致）、`stat_sem_thr50` 0.2756 / `stat_sem_hit1`
  0.2457（此为 **train 口径**统计，不能与 val 的 M1 读数混读，见 §71.9.19）。
  **A2-R 尚未开跑**（按 §6 串行）。
- **未完成**：A2-S 的 ep6 机制门（M1/M1'/M1-loc/M3/M4）、S2 ep12–16 任务门、
  多 seed。
- **不构成**：任何性能结论。S0 只证明"接线对、梯度在、身份不破"，
  **不证明机制成立**；A 的 S1 仍 **NOT ALL PASS**、ep16 仍未授权。

**下一步**：等 A2-S 的 ep6 val 落盘 → 看门狗自动停训并冻结 M1–M4 + M1'/M1-loc
证据 → 按预登记 §5 判定；**全过**才允许进 ep12–16，**不过**则该臂按负结果
归档且不再调权重/增益/KL 抢救。

#### 73.6.5 A2 看门狗证据缺失修复（2026-10-11；训练中修复，不改任何判据）

**发现的缺陷**：`watch_semA2_ep6.sh` 首版（`8ae570d`）只跑 A2 专属探针
（M1–M4 / rank / m4floor / bnmode）就结束，**漏掉了
`semA_ep6_verdict.py` 列为必需证据的四个产物**——
`z_util_std_ep6_w3/w100/w1800.json`（三个 z 利用率窗）与
`post_source_std_ep6_stride700.json`（后验来源归因）；也**没有调用**
`semA_ep6_verdict.py` / `semA_ep6_record.py`。若按该版本跑完 ep6，
`check_evidence()` 会判 **INCOMPLETE BUNDLE**、证据包不完整，判定器还会
打印「bundle is incomplete; the gate reading below is provisional」。

**为什么没被测出**：`test_watch_semA_ep6_cli.py` 原先只读 **A 的**
`watch_semA_ep6.sh`；A2 脚本是复制品，其遗漏落在测试覆盖之外。

**修复（三层，全部可核）**：

1. **看门狗补齐**（本文件外，`me_rssm/sanity/watch_semA2_ep6.sh`）：在判决前
   补跑四个家族标准产物（窗口 `3/100/1800`、offset `2/2/-1`、`stride 700`
   与 A 完全一致，保持跨代可比），然后依次调用
   `semA_ep6_verdict.py`、`semA_ep6_record.py`。
2. **幂等补跑器**（新增 `me_rssm/sanity/finish_semA2_ep6.sh`）：A2-S 在修复
   落地时**已在训练**，bash 已按旧版本继续，故另起一个只在**产物不存在**时
   才补跑对应探针的补跑器；它先等看门狗退出再等 GPU 排空，然后重新生成判定
   与 §72 草稿。对已完成产物**不重跑**（避免覆盖已冻结读数）。已在
   `2026-10-11 10:48:35 UTC` 挂到 A2-S（A2-R 开跑后同样需要挂）。
3. **回归测试**（`test_watch_semA_ep6_cli.py`，10 → **19 条**）：新增
   `TestA2WatchdogCoversTheWholeBundle` 六条，把 A2 看门狗与补跑器一并纳入
   静态契约——必需证据四个产物、探针清单、verdict/record 相对后验来源探针的
   顺序、冻结的 20-clip 契约（`limit=20/stride=100/start-index=3`）、
   补跑器的存在性守卫、以及不得自动续训。

**回归运行（2026-10-11，训练在跑时的 CPU 只读确认）**：上述 11 个测试文件
（`test_semantic_readout_rssm`、`test_watch_semA_ep6_cli`、`test_semA_ep6_verdict`、
`test_semA_ep6_record`、`test_probe_semA_rank`、`test_probe_semA_m4floor`、
`test_probe_semreadout_ep6`、`test_probe_semA_bnmode`、`test_config_build`、
`test_equivalence`、`test_grad_paths`）合跑 **201 passed**（`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`
`-p no:warnings`），无失败、无跳过。

**负向验证（证明测试真能拦住该缺陷）**：在临时目录用 `8ae570d` 的旧
`watch_semA2_ep6.sh` 跑同一测试文件，新测试**3 条失败**
（`test_a2_watchdog_produces_the_required_evidence_artifacts`、
`..._runs_every_required_probe`、`..._runs_verdict_and_record_after_the_probes`），
说明它抓的正是这次的真实遗漏，不是空断言。

**边界**：本次只改探针编排与测试，**未改任何判据数值、未改模型、
未改训练超参、未重启训练**；A2-S 的 ep6 读数仍未产出，A 的归档判定不变。
修好后 A2-S 的 ep6 证据包将与 A 同构，跨代读数可直接对照。

#### 73.6.6 ep6 判据口径对齐修复（2026-10-11，训练中修复；A2-S 的 ep6 尚未产出）

§73.6.5 修的是**证据是否齐全**。本节修的是**门本身读得对不对**：逐行核对
`probe_semreadout_ep6.py` / `semA_ep6_verdict.py` / `semA_ep6_record.py` 后，
发现三处**代码与预登记不一致**。三处都在 A2-S 的 ep6 落地前修复，因此
**没有任何 A2 读数被事后调门**；A 的归档判定**逐位不变**（下方有负向验证）。

**缺陷一：M3 的通过线是绝对计数，不是比例——窗口一放宽就自动变松。**

A 用 10 clip，预登记写「≥8/10」。A2 按 §5 冻结契约改用 20 clip，但
`gate_m3` 的实现是 `min_wins=8` 的**绝对条数**，于是同一条门在 A2 上实际变成
**8/20 = 40%**，比 A 的 80% 松一倍——这正是预登记要防的事后漂移，只是方向反了
（不是收紧，而是**悄悄放宽**）。

修复：把门定义为**比例** `8/10`，条数由窗口大小换算（`m3_required_wins`），
并把「冻结样本量」变成**必填参数**（不再有默认值）——调用方必须先声明自己的
分母；`len(rows) < min_clips` 读 **NOT MEASURED（False）**，绝不当成更弱的通过。

| 窗口 n | 通过所需胜出条数 | 说明 |
|---:|---:|---|
| 8 | 7 | 向上取整，故 7/8=87.5% 已过 |
| 10 | 8 | 与 A 归档逐位一致 |
| 16 | 13 | — |
| **20（A2）** | **16** | 修复前只要求 **8** |
| 24 | 20 | — |

**缺陷二：A2 新增的 M1'/M1-loc 门没有任何工具去判。**

§5 为 A2 预登记了 M1'（`auc ≥ 0.80` 且 `auc − gt_shuffle ≥ 0.25`）与
M1-loc（`max_gt_prob ≥ 0.5` 的 clip 数 ≥12/20 **且** `recall@top-k` 中位 ≥0.25），
但这两个读数落在 `semA_rank_*.json`，而 `semA_ep6_verdict.py` 只读
`semreadout_ep6_gate.json` 里的 M1–M4。后果：**一个 M1'/M1-loc 全挂的 A2
臂仍会打印 "S1 AGGREGATE: ALL PASS"**，据此进 ep16。

修复：`semA_ep6_verdict.py` 新增 `arm_of_bundle()` / `a2_readings()`，按
**config 路径**识别 A2-S/A2-R（不是按产物是否存在——A 的归档包里也有一份
事后跑出来的 `semA_rank_val_ep6.json`，按文件判会把 A 的归档判定重新打开成
"NOT EVALUABLE"），评估以上两门，并新增 **A2 AGGREGATE** 行：只有 M1–M4
**与**增补门全过才算 S1 通过。`semA_ep6_record.py` 同步把两门写进 §72 表格，
且标题在增补门未测全时改为 **NOT EVALUABLE（A2 增补门未测全）**。
M1-loc 仅对 A2-S 生效（§5 预登记即为 A2-S 专属），A2-R 不引入监督侧的门。

**缺陷三：M4 的 `own_hit` 用 `1/C + 20pt` 当线，而预登记写的是
`pooled prior floor + 20pt`。**

§71.9.18 **早已诊断出**这一点（A 的窗口 pooled 地板 0.5128 ⇒ 真线 0.7128，
代码却查 0.45），但**只写进了分析文档，没有回改代码**。本次把代码对齐到预登记：
`gate_m4(own_hit, transfer, num_classes, pooled_prior_floor)`，
`own_hit` 用 `floor + 0.20`、`donor_transfer` 仍用 `1/C + 0.20`
（后者按 donor 类打分，chance 口径本就成立）；缺少 floor 时返回
**None（NOT MEASURED）**，而不是回退到宽松的 chance 线。probe 自己从
已经栅格化过的标签里 `bincount` 出 floor，**不增加任何前向开销**。

**A 的归档判定逐位不变的负向验证（修复后重跑 A 的证据包）**：

| 门 | A 归档值 | 修复后重算 | 结论 |
|---|---|---|---|
| M3 | 7/10 wins, `gates.M3=False` | `gate_m3(rows, min_clips=10)` → **False** | 不变 |
| M4 | `own_hit` 0.7379 / `donor_transfer` 0.3818 | floor 0.5128 → own bar 0.7128（own_hit **过**）；donor bar 0.45（**未过**）⇒ 门 **False** | 不变 |
| M1–M2 | 不变 | 未改 | 不变 |
| 全局 | `S1 AGGREGATE: NOT ALL PASS` | 同 | 不变 |

即 `own_hit` 由「按错线过」改为「按对线过」，而 M4 的 **FAIL 始终由
`donor_transfer` 决定**，与 §71.9.18 的结论一致；A 的 S1 仍是
M1/M3/M4 FAIL、**不进 ep16**，一个字都不改。

**测试**：`test_probe_semreadout_ep6.py` / `test_semA_ep6_verdict.py` /
`test_semA_ep6_record.py` 新增针对上述三缺陷的回归（比例换算表、必填分母、
短窗口 NOT MEASURED、A 代数不被误判为 A2、A2-S 两门齐备、A2-R 不引入
M1-loc、M1'/M1-loc 任一条腿未过即 FAIL、冻结 n 不符读 NOT MEASURED、
渲染器标题跟随全门集合）。四个相关文件合跑 **161 passed**
（`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 -p no:warnings`）。

**边界**：本次只改**判据实现与渲染**，未改模型、未改训练超参、未重启训练、
**未读任何 A2-S 的 ep6 数值**（该 checkpoint 尚未产出）；A2-S 仍在 GPU5/6/7
训练中。修复后 A2-S 的 ep6 判定将在**完整门集合**上作出，不会出现
"只看 M1–M4 就宣告通过"的漏判。

#### 73.6.7 渲染器的 A 代硬编码修复（2026-10-11，训练中修复；承接 §73.6.6）

§73.6.6 修的是**门读得对不对**。本节修的是**草稿写得对不对**：`semA_ep6_record.py`
是 §72 正文的生成器（AGENTS 第 5/6 条要求逐 epoch 表、逐类拆解、运行事实都
由它落盘，避免手抄），但它有四处**写死成 A 代**，对 A2 包会生成**错误或残缺**
的正文。四处在 A2-S 的 ep6 产出前修复。

| # | 硬编码 | 对 A2 包的后果 | 修复 |
|---|---|---|---|
| 1 | `_WATCH_TS_RE` 只匹配 `[watch]` | A2 看门狗写的是 `[watch-A2S]` ⇒ **停止时刻/GPU 释放/探针结束三行全部读不到**，正文退回 `FILL` 占位符（明明有记录） | 正则放宽为 `[watch(-<ARM>)?]`，对 `[watch]`/`[watch-A2S]`/`[watch-A2R]` 均匹配 |
| 2 | `WATCH_LOG = 'watch_semA_ep6.log'` | A2 的日志名是 `watch_semA2S_ep6.log` ⇒ 找不到文件，同样丢运行事实 | 新增 `watch_log_name()`，按存在性在三种日志名间选择 |
| 3 | 正文写死 `watch_semA_ep6.sh` / `semA_watch` | A2 包会**指错看门狗脚本**（且 `semA_watch` 是早已不存在的旧句柄） | 新增 `watchdog_script_name()`，A2 指向 `watch_semA2_ep6.sh` |
| 4 | 正文写死 `docs/rssm_candidate_A_prereg_20261010.md` | A2 包会**引用错预登记文档** | 新增 `prereg_doc()`，按**臂**（config 路径，经 `verdict.a2_readings`）选择 A2 预登记 |

**第 4 项的一个实现陷阱（自查发现并避免）**：`prereg_doc()` 初版按
「`semA_rank_val_ep6.json` 是否存在」判定 A2。但 **A 的归档包里也有这份文件**
（§71.9.19/20 的事后只读探针写入），那样判会把 A 的归档正文改成引用 A2 预登记。
最终与 `arm_of_bundle()` 一致，**按 config 路径**判臂。

**A 的归档正文不受影响（负向验证）**：对 A 的证据包重跑渲染器，逐行 diff
只有两处**预期内的措辞**变化——(a) 生成日期仍是当前日期，(b) 第 5 行把早已
失效的 `semA_watch` 句柄改写为真实脚本名 `me_rssm/sanity/watch_semA_ep6.sh`；
**所有读数、判定、表格行一字未变**，A 的 S1 仍是 M1/M3/M4 FAIL。

**对真实 A2 日志的验证**：直接拿**正在跑的** A2-S 看门狗日志
（`…_semA2S_seed0/watch_semA2S_ep6.log`）测正则，
`[watch-A2S] 2026-10-11 09:06:40 waiting for ep6 val + fresh checkpoint`
现在**匹配成功**；修复前该行被静默丢弃。

**测试**：`test_semA_ep6_record.py` 新增 4 条（A 包仍指向 A 预登记与 A 日志、
A2 包指向 A2 预登记与自己的日志、**A 包里的陈旧 rank 文件不得把 A 重新标成
A2**、A2 草稿能从自己的看门狗日志读出停止时刻）。11 个相关文件合跑
**222 passed**（`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 -p no:warnings`）。

**边界**：本次同样只改**渲染器与测试**，未改模型、未改超参、未重启训练、
未读任何 A2-S 的 ep6 机制读数（`epoch_6.pth` 尚未产出，当前最新为 ep2）。

#### 73.6.8 M4 地板自包含化与一个已排除的替代口径（2026-10-11，训练中）

承接 §73.6.6 的缺陷三。修复 `gate_m4` 后，门所需的 `pooled_prior_floor`
来自**第二个产物** `semA_m4floor_ep6.json`。核对时确认了一件事并补了一件：

**确认（不是缺陷，是设计正确）**：`probe_semreadout_ep6.py` 内的 pooled 地板
与 `probe_semA_m4floor.py` 的地板定义**逐行等价**（都是「对该窗所有前景格
`bincount` 后取多数类占比」）。两探针各自独立计算同一量，所以
`semA_m4floor_ep6.json` 的 floor 0.5128 可直接用于 M4 判定——这正是
§71.9.18 已经测过的口径，不需要新增读数。

**补齐（本轮新增）**：gate JSON 现在**自包含**地写出门所需的地板与两条线，
不再让读者去第二个文件里找：

| 字段 | 含义 |
|---|---|
| `pooled_prior_floor` | 门的基准（A：0.5128） |
| `own_hit_bar` | `floor + 0.20`（A：**0.7128**） |
| `bar` | donor 线 `1/C + 0.20`（A：0.45） |
| `pooled_class_counts` | 该窗前景格类别计数 |
| `per_clip_prior_floor_mean` / `_max` | **仅记录**的替代口径（A：0.6970 / 1.0000） |

**一个被显式排除的替代口径（记录理由，防止事后改用）**：M4 的 `own_hit` 是
**前景格级**聚合（分子分母都是格），因此其基准必须是**池化**地板 0.5128，
不是 **per-clip** 地板的均值 0.6970——后者是「先按 clip 算占比再平均」，
量纲与 `own_hit` 不一致，且含多个单类别 clip（地板 1.0）会把线抬到 0.8970，
远超预登记。故 per-clip 地板只作读数写入 JSON，**不进门**。
（该选择与 §71.9.18 一致：那里用的也是 pooled 0.5128 与 0.2251 的余量。）

**边界**：未改判据数值（own 线仍 `floor + 0.20`、donor 线仍 0.45）、未改模型、
未改超参、未重启训练。11 个相关文件合跑 **222 passed**。

#### 73.6.9 渲染器剩余 A 代措辞的臂感知化（2026-10-11，训练中；收尾 §73.6.7）

§73.6.7 让渲染器找到**正确的日志与预登记文档**，但正文**标题与归档段落**
仍写死 A 代。对 A2 包会生成如下错文：标题写「候选 A ep6 机制门」、
失败原因只列 M1/M3/M4（**不提真正把它挡在门外的 M1-loc**）、
归档范围写「本次候选 A 的读出/监督设计」。

三处修复：

| 位置 | 修复 |
|---|---|
| 标题 | 新增 `arm_label()`：A ⇒ `候选 A`、A2-S ⇒ `候选 A2-S`、A2-R ⇒ `候选 A2-R` |
| §72.9 失败原因 | A2 增补门未过时，**追加** `M1'`/`M1-loc` 的实测读数（否则草稿会漏掉实际止损的那条门） |
| §72.9 归档范围 | A2 走独立分支：说明是**单一变量对照**（A2-S 监督侧排序压力 / A2-R 通道覆盖），并按 §7 分叉表给出下一步；A 的原段落保持原样 |

**验证**：A 的证据包重渲染后与修复前**逐字节相同**（`diff` 无输出）；
合成 A2-S 包的标题为 `候选 A2-S`，失败原因行含
`M1-loc max_gt_prob ≥ 0.5 仅 5/20、recall@top-k 中位 0.0200`。

**测试**：`test_semA_ep6_record.py` 新增 3 条（`arm_label` 随臂变化、
A2 归档原因必须引用 M1-loc、A 代归档段落必须留在 `else` 分支下）。
11 个相关文件合跑 **225 passed**。

**边界**：未改任何门值、未改模型/超参、未重启训练；A2-S 仍在训练中。

#### 73.6.10 A2-R 分支的冗余探针（2026-10-11，记录未改；**刻意不动在跑脚本**）

核对 `watch_semA2_ep6.sh` 时发现 A2-R 专属块（第 122–128 行）重复调用了一次
val 排序探针：

```
# 通用块（第 90–93 行，两臂都跑）        # A2-R 专属块（第 124–127 行）
--config "$CFG"  --checkpoint "$CKPT"    --config "$CFG"  --checkpoint "$CKPT"
--start-index 3 --stride 100 --limit 20  --start-index 3 --stride 100 --limit 20
--output ..._val_ep6.json                --output ..._val_ep6_a2r.json
```

**逐字节同参**，只有输出文件名不同。且 `_a2r.json` **没有任何消费者**
（`grep` 全仓无引用；`verdict.a2_readings()` 统一读 `semA_rank_val_ep6.json`，
A2-R 也由通用块产出）。预登记 §5(e) 真正要的 A2-R 覆盖读数
`frac_cells_residual_can_flip` 来自 `semA_m4floor_ep6.json`，该探针**两臂都跑**
（第 101–104 行），与这个重复块无关。

⇒ 后果仅是**多花一次 val 探针的 GPU 时间**（约 1–2 min），**不是正确性问题**：
产物存在性守卫不会因此误判，判定器也不会读到两套不一致的读数。

**为什么这次刻意不改**：A2-S 的看门狗**仍在运行**。bash 会记住脚本内的**字节
偏移**，若在它等待 ep6 的期间改写文件，循环结束后会从**错误偏移**继续读取，
导致执行到错位的内容——这正是 §73.6.5 已经踩过的坑（当时旧版脚本已被读入
bash 内存，只能用幂等补跑器兜底）。故本轮**只记录、不动脚本**；
待 A2-S 结束、A2-R 开跑前再清理该块（或直接删除，因无消费者）。

**对 A2-R 的连带提醒（已写入 TODO）**：A2-R 开跑后同样需要
`ARM=A2R finish_semA2_ep6.sh` 兜底（§73.6.5 的第二层修复），
因为运行中的 A2-S 看门狗仍是**修复前**的版本。

#### 73.6.11 在跑看门狗"文件被改后是否仍执行修复段"的实测确认（2026-10-11，只读）

§73.6.5/§73.6.6 在 A2-S **训练途中**修改了 `watch_semA2_ep6.sh` 及其调用的
Python 探针/判定器。这留下一个必须验证的问题：**正在等待 ep6 的 bash 进程，
在循环结束后是从"旧脚本"还是"新脚本"继续读？** 若读旧脚本，§73.6.5 的三层修复
中第一层（看门狗自带完整证据电池）**对本次运行无效**，只能靠第二层补跑器兜底。

**实测（只读，不触碰真实训练）**：

| 观测量 | 值 | 含义 |
|---|---|---|
| 看门狗 PID 3070972 的脚本 fd | `/proc/…/fd/255` → `watch_semA2_ep6.sh` | bash 持有该文件 |
| 该 fd 的 `pos` | **1951** | 停在 while 等待循环内（`done` 在字节 1946） |
| 旧版（`8ae570d`）字节数 | 5799 | — |
| 当前文件字节数 | 8455 | — |
| 两版**首个差异字节** | **5700**（第 130 行） | 前 5699 字节**逐字节相同** |

因为 bash 的位置（1951）**远早于**分叉点（5700），且前 5699 字节完全一致，
它按偏移继续读时读到的是**新文件的尾部**——即新增的四个家族标准产物 +
verdict + record。**结论：第一层修复对本次运行有效**。

**独立复现（合成文件，字节几何与真实一致）**：构造 old/new 两份文件，
前 5699 字节完全相同、尾部不同；bash 在 `while` 中运行至中途，把 old 覆盖为
new，观察输出——**执行的是 NEW 尾部**。这与上面按偏移推理的结论一致，
也解释了为什么 §73.6.5 当时"旧版已读入 bash 内存"的假设**并不成立**
（真正原因只是当时文件尚未被改）。

**为什么仍然保留补跑器**：该结论依赖"bash 从文件当前偏移续读"这一实现细节。
补跑器是**存在性守卫**的幂等兜底，开销可忽略，故不作废——
两者只要有一个生效，证据包就是完整的。

**不改任何东西**：本轮为纯只读核验；未改脚本、未改判据、未碰训练。
（顺带确认：3 个训练 rank 均存活，churn 到 val ep3；输出里的第 4、5 个
`train_vod.py` 匹配分别是 launcher 与本次查询自身的 shell。）

#### 73.6.12 S0 编号串号更正（2026-10-11，文档级；A2-S ep6 未产出）

审计发现 **A 与 A2 两份预登记用了同一批字母编不同的 S0 清单**：

| 字母 | A 预登记 `rssm_candidate_A_prereg_20261010.md` | A2 预登记初稿 |
|---|---|---|
| `S0-c` | `loss_rssm_sem` 有限下降、无 NaN/OOM | 梯度压力门 |
| `S0-d` | **梯度压力门（≥5%）** | **A2-S 专属读数**（`max_gt_prob` 只记录） |
| `S0-e` | 读出存活 / 语义项不动 readout | **A2-R 专属读数**（`frac_cells_residual_can_flip`） |

本节的 §73.6.3 表格与三个 smoke 脚本（`semreadout_smoke_450iter.py`、
`semA2S_smoke_450iter.py`、`semA2R_smoke_450iter.py`）**用的是 A 的字母**，
即「`S0-d 压力`」这行文字本身没错（它确实在报压力门），但它**与 A2 预登记里的
`S0-d` 不是同一个门**。若读者按 A2 预登记找 `S0-d`，会把压力门读数误读成
「A2-S 专属读数」。

**更正（文档层，数值一字未改）**：A2 预登记 §5 的 S0 清单改为 `S0-1…S0-5`，
并在每项旁标出与 A 字母的对应；同时明确**报数一律写全称**
（如「A2 S0-3（压力门，= A 的 `S0-d`）」），不写裸字母。
本节的 A2 S0 表格沿用 A 的字母**不变**（其数值与门含义都正确），
两处通过上表建立显式映射。

**为什么改文档而不改记录/脚本**：§73.6.3 是**已发生的运行事实**，
其字母与当时执行的 smoke 脚本一致；改写既有记录会破坏与脚本的双向可核性。
串号风险来自**命名**而非数值，故在预登记侧消除歧义。

**边界**：纯文档更正。未改任何门槛数值、未改模型/超参、未重启训练；
A2-S 仍在训练中（12:09 UTC 已过 val ep3）。

#### 73.6.13 ep6 判定链路的第二轮审计：例外条款越权 + 三项 A 代硬编码（2026-10-11，训练中修复；A2-S ep6 未产出）

承接 §73.6.6–§73.6.12 的同一目标：**在 A2-S 的 ep6 产物落地之前**，把判定器与
记录生成器里所有会「静默美化或静默劣化结论」的路径清掉。本轮审出四处，
其中第一处是**会改写结论方向**的缺陷，其余三处是「会描述错实验」的缺陷。

##### (1) §71.9.3(2) 的跨网格例外**越权适用**（结论方向级）

§71.9.3(2) 是为 **A 的 M1–M4 门集**写的：M1 若**只**在 full-res 原始 `hit@1`
上未过、而 `thr50`/`hit1_excess_over_chance` 健康且**其余门全过**，则记
「证据不足」，需补 full-res 同口径对照，**不得**当机制否定。

原实现的判定式只有三个子句（读数形状 + 其余 M1–M4 全过 + M1 确实未过），
**没有把 A2 增补门算进去**。后果是一个 A2-S run 若 `M1'` 或 `M1-loc`
**实测失败**、而 M1 恰好落在例外的形状里，就会被打印成

> `INSUFFICIENT EVIDENCE ... add a full-res matched control`

即：**把一条已经测出来的机制否定，软化成一条仪表口径的抱怨**。这正是该例外
条款最容易被滥用成的「事后逃生舱」。A2 的增补门恰恰是该臂假设所依赖的门，
不纳入判定的方向是错的。

**修复**：新增 `grid_limited_applies()` 作为**唯一**的例外适用范围判定，
要求同时满足

1. M1 实测未过；
2. A 的 M1–M4 中**其余门全过**；
3. M1 读数具备例外形状（`thr50 ≥ 0.30`、`hit@1 < 0.60`、`excess ≥ 0.584`）；
4. **该臂自身的增补门全过**（A 代无增补门 ⇒ 恒真；A2 的
   `M1'`/`M1-loc` 未测或未过 ⇒ 例外**不成立**）。

判定器与渲染器**都**改为调用这一个函数（渲染器原先自己重写了旧的二子句式，
属于同一语义的第二份实现——两份实现迟早会漂移）。

同时把判定器在 M1 明细处的即时打印从 `PRE-REGISTERED RULE APPLIES`
改为「读数具备跨网格**形状**；是否适用取决于其余门，见下方聚合判定」——
原措辞在还不知道其余门结果时就宣布规则适用。

**验证**：

| 场景 | `m1_grid_limited`（旧式） | `grid_limited_applies`（新） |
|---|---|---|
| A：仅 M1 未过、形状满足 | True | **True** |
| A2：形状满足但增补门 FAIL | True | **False** |
| A2：形状满足但增补门 NOT MEASURED | True | **False** |
| 其余 M1–M4 某门未过 | False | **False** |
| M1 实际通过 | False | **False** |

##### (2) 渲染器无法呈现 H-A2 所依赖的 train/val 对照

H-A2 的命题是**留出场景上的峰预算能否保住**，其本质是 **train/val 差距**。
M1′/M1-loc 只在 val 上判；看门狗**已经**用同一探针在同一 20-clip 契约下跑了
train split（`semA_rank_train_ep6.json`），但渲染器从不读它、判定器也只用了
val 侧。结果是草稿里**看不到这个臂的假设到底有没有被检验到**。

**修复**：新增 `a2_split_contrast()`，从两份产物里分别聚合
`auc` / `gt_shuffle_auc` / `max_gt_prob` / `recall@top-k` 中位与均值 /
`max_gt_prob ≥ 0.5` 的 clip 数，并在 M1′/M1-loc 门行之后渲染成表，
附 `train−val` 差；A 的同口径差作为对照一并写出
（`auc` 0.9994−0.9235 = **+0.0759**、`recall@top-k` 中位 0.3333−0.0000
= **+0.3333**）。该块**只是机制定位，不是门槛**。

##### (3) §72.1「唯一变量」是 A 代措辞（会描述错实验）

A 相对 `no2d_igdr` 底座**同时改了两件事**（`sem_head` 语义监督 + `z_cls_logits`
零初始化分类残差读出），这正是 A 无法分离两者的原因。A2 两臂是**单变量**
预登记的对照，若草稿照抄 A 的双旋钮句，会

* 把 A2-S 描述成「也改了读出」——它没有；
* 把 A2-R 描述成「也改了监督」——它没有；
* 且掩盖了 A2 相对的是 **A 的底座**（`..._semreadout.py`）而非 `no2d_igdr`。

**修复**：新增 `arm_single_variable()` 与 `arm_comparator()`，按臂输出
（A2-S：`sem_target_mode='channel_contrastive'`；A2-R：
`cls_readout_mode='anchor_slot'`；A：原双旋钮句）。`变量取值` 行**仅对 A2 输出**
（A 无单变量可言），以保证 A 的已冻结草稿逐字节不变。

##### (4) A2 专属产物只被「报告」而不被「门槛化」——这一条是**刻意**的

新增 `check_a2_evidence()`，把 `semA_rank_val_ep6.json` 与
`semA_rank_train_ep6.json` 的存在性与 schema 一并列出。但本实现**刻意不**把
它的 `ok` 并进 `evidence_ok`：

* 若把它并进去，则「train/val 对照缺失」会把一条**已实测的门 FAIL** 降级成
  `NOT EVALUABLE` —— 那是**用缺仪表掩盖负结果**，方向是错的；
* A2 增补门**能否判定**由 `a2_readings()` 的 blockers 决定（缺文件、格点
  不符、index 漂移 ⇒ `NOT MEASURED`），它已经覆盖了真正的「无法判定」情形；
* train/val 对照缺失只让**记录变薄**，草稿会渲染显式的 FILL 占位。

本节的两条测试因此锁定：`check_a2_evidence` 对缺失 train split 报
`['OK','MISSING']`（如实报告），而**不影响**门判定。

##### 回归与冻结验证

* **新增 19 条单测**（`test_semA_ep6_record.py`），全库 **243 passed**。
* **A 的归档判定逐字节未变**：`semA_ep6_verdict.py` 对 A 归档目录的输出与
  改动前 `diff` 完全一致；`section72_draft.md` 重生成后与改动前**逐字节相同**
  （`变量取值` 行、`grid_limited` 分支、A2 专属 block 均不影响 A）。
* 合成 A2 捆绑（复制 A 产物 + 改 `config_path` 为 `..._semA2S_24e.py`）实测：
  臂识别 `A2S`、`M1p=PASS`/`M1loc=FAIL`、train/val 对照两行齐全、
  §72.1 只写 `sem_target_mode`、标题为「候选 A2-S」——A2 路径确实被走通，
  而不是只在单测里被调用。

**边界**：未改任何门槛数值、未改模型/超参/数据、**未触碰在跑的 A2-S 训练**
（本节全部为 CPU 只读代码与单测）。§73.6.11 已确认在跑看门狗的尾部修复会
被执行；本节改动**同样**只影响 ep6 产物落地之后才运行的
`semA_ep6_verdict.py` / `semA_ep6_record.py`。

#### 73.6.14 渲染器第三轮审计：z 契约段静默消失 + §72.5 归属写错（2026-10-11，训练中修复）

同一「ep6 产物落地前清掉会误述实验的路径」目标，本轮再审出两处。

##### (1) 缺 `semA_zcontract_ep6.json` 时 §72.10 **整段消失**

原实现是 `if os.path.exists(zc_path):` 渲染、否则什么都不做。后果有两层：

1. **编号断档**：草稿从 `### 72.9` 直接跳到 `### 72.11`，读者无从知道
   缺的是哪一节、为什么缺；
2. **静默**：`semA_zcontract_ep6.json` 回答的正是「train/val 的 4.3× 差距
   是否来自 z 采样契约」——这是 A2 最关心的口径问题之一。缺它时的正确语义是
   **「两契约之间是否有差异未被测量」**，而**不是**「契约无影响」，
   更不是「门禁读数」。原来的静默恰好把「未测量」渲染成了「没问题」。

**修复**：改为显式 FILL 占位 + 边界声明，写明该段的作用（部署 `mu_q` vs
训练 `mu_q + sigma*eps` 的实测对账）与缺口的含义（**不得据此宣称契约无影响、
也不得读成门禁读数**）。

> A 的归档包**有**该产物，故 A 草稿仍走原分支、逐字节不变（已验证）。

##### (2) §72.5 标题写死「A 侧」

§72.5 报的是 **本 run 自己**的训练期 `stat_sem_*` 读数。标题写死「A 侧」会让
A2 草稿把**本臂**的诊断读数归属给 A。修复：新增 `arm_side_label()`，按臂输出
「A 侧 / A2-S 侧 / A2-R 侧」。

##### 回归与冻结验证

* 新增 4 条单测（含「缺产物仍须保留小节 + 出现 FILL + 写明未被测量」与
  「有产物时不得出现未测量措辞」两条互斥断言，防止占位与实测混淆）；
  11 个相关文件合跑 **245 passed**。
* **A 的 §72 草稿重生成后逐字节不变**（A 有 z 契约产物 ⇒ 不入 FILL 分支）。
* 合成 A2 包实测小节编号 **72.0 → 72.11 连续无缺口**，
  §72.5 标题为「**A2-S 侧**训练期同口径读数」。

**边界**：未改门槛数值、未改模型/超参、未触碰在跑的 A2-S 训练；
本节仅影响 ep6 产物落地后才运行的记录生成器。

#### 73.6.15 A2 收尾脚本的两个**死守卫**（2026-10-11，训练中修复；已重启收尾进程）

§73.6.5 引入的 `finish_semA2_ep6.sh` 是「看门狗跑完后补齐缺失产物」的兜底。
本轮审计发现它有两处**永远不会触发/永远不会等到**的守卫——两者都来自
**沿用 A 代的名字推导**，而 A2 的命名已经分叉，故在 A 代运行时不可能暴露。

##### (1) 致命：训练日志路径猜错 ⇒ 等待循环永不可满足

脚本原文 `LOG="${LOG:-/tmp/sem${ARM}_24e.log}"`，对 A2-S 展开为
**`/tmp/semA2S_24e.log`**——该文件**没有任何进程会创建**
（本 job 的日志在 `/tmp/a2s_24e.log`；run dir 里是
`$WORK_DIR/YYYYmmdd_HHMMSS.log`）。而它的第一个等待循环是

```bash
while [ ! -f "$CKPT" ] || ! grep -q 'Epoch(val) \[6\]' "$LOG"; do sleep 30; done
```

`grep` 永远失败 ⇒ **循环永不退出**。实测：该进程已 armed **2 小时 21 分**
（10:48:35 → 13:10），`/proc/<pid>/fdinfo/255` 停在字节 2063（正是该循环
`done` 之后），**从未执行过任何探针**。即「兜底」实际是**静默空转**。

A 代不受影响纯属巧合：那里 `sem${ARM}_24e.log` 恰好等于真实日志名。

**修复**：日志改为**从 run dir 推导**（`ls -1t "$WORK_DIR"/[0-9]*.log`，
取最新时间戳日志），旧 `/tmp` 路径仅作 run dir 被清理时的最后回退。
实测新推导得 `…/20261011_090445.log`（存在），不再依赖 `/tmp` 命名约定。

##### (2) 竞态：看门狗进程名模式永不匹配 ⇒ 不等看门狗就开始抢 GPU

脚本原文 `pgrep -f "watch_sem${ARM}_ep6.sh"`，对 A2-S 展开为
**`watch_semA2S_ep6.sh`**；但 A2 的看门狗脚本由**两臂共用**，实际进程名是
**`watch_semA2_ep6.sh`**。实测：

```
actual : bash me_rssm/sanity/watch_semA2_ep6.sh
pattern: watch_semA2S_ep6.sh      -> NO MATCH
```

该守卫的作用正是「等看门狗电池跑完再动 GPU」（脚本注释写明：与看门狗同时跑
是**早前 A2 OOM 的原因**）。守卫失效 ⇒ 收尾脚本会**立即**在 5/6/7 上启动自己
的探针，与看门狗电池正面冲突。

**修复**：新增显式映射 `A2S|A2R) WATCHDOG_SCRIPT_NAME=watch_semA2_ep6.sh`，
两处 `pgrep` 改用该变量。实测修复后守卫能看见在跑的看门狗
（`YES - finisher will wait for the watchdog battery (no OOM race)`）。

##### 处置与验证

* **在跑实例不可救**：bash 已把旧循环读入内存（fd 停在循环之后），改文件
  无法影响该实例。故 kill 旧 PID，用**修复后的脚本**重新 armed
  （13:23:45）；新实例 `fd=3469`，停在**修复后**的等待循环内
  （`LOG` 推导行、`WATCHDOG_SCRIPT_NAME` 映射均在其读取范围内），
  实测其 `LOG` 指向存在的真实日志、守卫可见在跑看门狗。
* **看门狗脚本本体未改**（仅文件被读取，内容与 HEAD 逐字节相同），
  且它自身就会产出**全部**必需证据产物并调用 verdict + record
  ⇒ 即使收尾脚本再次失效，本次 ep6 产物仍完整。
* 新增 **3 条回归测试**（源码守卫 2 条 + 对两臂真实 run dir 的推导行为检查 1 条），
  11 个相关文件合跑 **248 passed**。
* 未改模型/超参/门槛数值，**未打断在跑训练**（ep5 进行中，13:24 UTC）。

**教训（写入记录以免重犯）**：凡是把「臂名」拼进**文件名/进程名**的推导，
在新增臂时都必须复核；A 代能跑通不代表 A2 能。§73.6.7/§73.6.9/§73.6.13/
§73.6.14 已修过同类问题，本轮是同一根因的最后两处。

#### 73.6.16 同一 LOG 陷阱也存在于 A2 看门狗（2026-10-11，训练中修复；看门狗已重启）

§73.6.15 修的是**收尾脚本**的日志推导。顺同一根因复查**看门狗本体**，
发现它带着**同一个** `LOG="${LOG:-/tmp/sem${ARM}_24e.log}"` 默认值。

##### 为什么这次更危险

看门狗的第一个循环同样是

```bash
while true; do
  if [ -f "$CKPT" ] && grep -q 'Epoch(val) \[6\]' "$LOG" 2>/dev/null; then ... break; fi
  sleep 30
done
```

对 A2 展开为 `/tmp/semA2S_24e.log`（**无人创建**）⇒ `grep` 永远失败 ⇒
**看门狗永远不触发**。而 ep6 的 **gate / rank(val+train) / m4floor / bnmode**
产物**只由看门狗产出**（收尾脚本只补三窗口 z-util + posterior source）——
即看门狗一旦空转，**ep6 机制门根本没有证据**。

**A2-S 之所以没出事纯属侥幸**：它是**手工带 `LOG=/tmp/a2s_24e.log`** 启动的
（`/proc/3070972/environ` 可查）。**A2-R 若按同一方式起、但不带该变量，
就会静默空转**——与 §73.6.15 完全同型。

##### 旁证：在跑实例的执行状态已不可解释

看门狗日志里 `waiting for ep6 val` 出现**两次**（09:06:40、12:08:44），
而脚本里该行 `echo` **只有一处**；且其 fd 停在**字节 1951**、当时文件已被
§73.6.5/§73.6.6 的两次在途编辑改过。即：**bash 曾在字节偏移被改动的文件上
续读**，实际执行路径已无法仅凭日志复原。

**处置**：由于当时**尚无任何 ep6 探针产物**（`ls` 证实只有训练 log.json），
重启的代价为零。故：

1. 冻结旧日志到 `/tmp/watch_a2s_pre_restart.log`（两条 `waiting` 行留存）；
2. kill 旧看门狗 PID 3070972（**训练进程 3068491 未受影响**，复核仍存活）；
3. 清空看门狗日志，以**修复后脚本**重启（13:48:09，新 PID 3655120）；
4. 复核新实例：`fd=2651` 停在**修复后**的等待循环内；**未带 `LOG` 覆盖**
   时自行推导出 `…/20261011_090445.log`（存在）⇒ 该修复**确实生效**。

##### 修复

日志改为**从 run dir 推导**，并保留优先级：

```bash
LOG_DERIVED=$(ls -1t "$WORK_DIR"/[0-9]*.log 2>/dev/null | head -n1)
LOG="${LOG:-${LOG_DERIVED:-/tmp/sem${ARM}_24e.log}}"
```

即**显式覆盖 > run dir 最新时间戳日志 > 旧 /tmp 兜底**。实测 A2S 推导正确；
A2R 的 run dir 尚未建立时回退路径仍存在（但 A2-R 开跑后必然有日志）。

##### 回归与验证

* 新增 **2 条测试**：看门狗自身的推导守卫 1 条 + **两个 A2 脚本共用同一推导
  约定**的交叉检查 1 条（防止只修一处、另一处回归）；11 个文件合跑
  **250 passed**。
* 复核：训练仍在 ep5（ep4 val 已出）、新看门狗与新收尾脚本**互相可见**
  （收尾脚本的 `WATCHDOG_SCRIPT_NAME` 守卫能看到新看门狗）⇒ §73.6.15 的
  OOM 竞态修复同样对新实例生效。

**边界**：未改判据/模型/超参；训练进程全程未动；看门狗重启发生在**任何
ep6 产物产出之前**，故无产物被覆盖或作废。

#### 73.6.17 渲染器第四轮审计：三处**「只改了文档、没改生成器」的漂移** + A2 聚合判定越权（2026-10-11，训练中修复；A2-S ep6 未产出）

§73.6.13/14 修的是**判定逻辑**与**静默消失**。本轮换了一个角度审：**把生成器
的输出与 `docs/training_runs_full.md` 里的 §72 逐行 diff**。§72 是渲染器生成的草稿，
但归档之后它的正文被人工补过几次——于是出现了「文档是对的、生成器是旧的」这种
**反向漂移**：任何人再次运行生成器（A2 的两臂都会跑）就会**静默删掉**这些更正。

##### 缺陷一：§72.10「来源仍未定位」——整条归因只存在于文档里

§71.9.19（commit `100e0c5`）用只读补测把 train/val 4.3× 差距归因到**数据 split**，
并把结论写成了**手写段落**插在 §72.10 末尾。生成器仍输出旧的
「该差距的来源仍未定位」，两种后果都不可接受：

* 再渲染 ⇒ **删掉归因**，读到「未定位」这个已被取代的结论；
* 不渲染 ⇒ A2 草稿缺这一段，而 A2 的 H-A2 假设**正是**关于 train/val 差距的。

修复：新增 `split_attribution()`，从**本包自己的**产物推导该表，**两侧必须同时
存在**（只有一侧无法说明「差距来自 split」），并**按实际 n 标注**。

| 来源 | n | 谁产出 |
|---|---:|---|
| `semA_splitprobe_{val,train}_full_n20.json` | 20 | A 的**临时**只读探针 |
| `semA_bnmode_ep6{,_trainsplit}.json` | 10 | A 的**临时**只读探针 |

**关键事实（本节起草时的初稿曾写错）**：上面两组产物**都只对 A 存在**。
`watch_semA2_ep6.sh` 只跑 `probe_semA_bnmode.py` 的 **val 侧**（`--limit 20`、
**无** `--split-ann-file`），**从未**跑 train-split 臂。所以：

* A2 包**默认就会**输出 `来源归属：NOT MEASURED`——这是**诚实**结果，不是缺陷；
* 若要在 A2 上复现该归因，必须**显式补跑** train-split 臂（只读），不能靠生成器
  回退到 A 的数字。

两条路径的推导都已验证（n=20 复现文档里的 0.4739/0.4167 与 0.1280/0.0417；
n=10 打印 `n=10 / 10`）。**两包都不齐全时**输出 NOT MEASURED——此时仍可断言
「契约错配不是原因」，但**不得**断言差距来自泛化。数字**不硬编码**，有结构测试
禁止它们出现在生成器源码里（否则 A2 会替一个自己没测过的臂宣布结论），另有 1 条
测试直接钉住 A2 看门狗的**真实产物形状** ⇒ 必须渲染 NOT MEASURED。

##### 缺陷二：A2 的 S1 聚合判定只查 M1–M4，标题却查增补门 ⇒ 同一草稿自相矛盾

标题早已要求「M1–M4 **与** A2 增补门」全过（§73.6.13 缺陷二），但 §72.0 的
`elif all_pass:` 与 §72.9 的 `elif all_pass:` **仍然只看 M1–M4**。实测构造
（M1–M4 全绿、M1′/M1-loc 全红）渲染出：

* 标题：`未全门通过`；
* §72.0：`S1 聚合判定：ALL PASS`；
* §72.9：`请求用户批准 §5 的 S2 数值 … 续训至 ep16`。

即**同一节里标题判负、正文判通过并授权 ep16**，而该臂的预登记门是**实测失败**的。
这比措辞错误严重：它是**放行**。

修复：两处分支改为 `all_pass and added_all_pass and not grid_limited`，并在
NOT ALL PASS 的原因列表里补上增补门名字（`M1'` / `M1-loc`），使失败的**是哪一门**
可读。新增 2 条测试：一条构造上述矛盾并断言三处一致（标题/§72.0/§72.9 都不得
出现「请求用户批准」），另一条守卫**不得把真通过卡死**（全绿时仍须到达 S2 请求）。

##### 缺陷三：§72.9 的「口径更正」也只在文档里

§73.6.13 之后的 `dbf6bdd` 给 §72.9 第 2 条加了一段手写更正：该 Ped/Cyc 非劣线属
**S2**，写在 ep6 归档节里极易被读成 ep6 止损线。同样只改了文档。现改为生成器
输出（含固定的更正日期常量 `S2_SCOPE_CORRECTION_DATE`，**不随 run 日期漂移**），
并按臂改称（A 停在 S1 / A2 臂停在 S1）。

##### 验证

* 把 §72 整节用生成器重新渲染后与文档逐行对比 ⇒ **IDEMPOTENT**（已按此把文档正文
  与生成器对齐；差异只剩生成器本来就该有的规范化排版，12 增 11 删）。
* 11 个文件合跑 **260 passed**（250 + 本轮 10 条）：新增 `TestSplitAttribution`
  **8 条**（n=20 推导、n=10 标注、**A2 看门狗形状 ⇒ NOT MEASURED**、缺产物不宣称
  来源、单边不半报、`_diff` 不造 0、源码不硬编码已发表读数、渲染替换旧结论）与
  A2 聚合一致性 **2 条**。
* A 的**门读数与判定逐位不变**：本轮只动渲染文本与聚合分支的**条件**，
  未触碰任何门槛常量、探针或证据 JSON。

**边界**：未改判据数值/模型/超参；训练进程全程未动；看门狗与收尾脚本未改，
A2-S 的 ep6 产物此刻尚未产出，故不存在「先看到读数再调门」。

#### 73.6.18 A2 对照句里的 A 参照仍是**手抄常量**（2026-10-11，训练中修复；A2-S ep6 未产出）

§73.6.17 修完「三处只改文档」后，继续按同一角度扫**生成器内部的字面量**，发现
一处**自相矛盾**：§72.11 声明本节「`*.log.json` 机械生成，**未手抄任何读数**」，
但 §72.7 的 A2 train/val 对照句里，A 的参照值是**硬编码**的：

```
f'A 的同口径差为 `auc` 0.9994−0.9235 = **+0.0759**、'
f'`recall@top-k` 中位 0.3333−0.0000 = **+0.3333**'
```

数值本身**当前是对的**（核对 A 归档：train `auc` 0.99944/`recall` 中位 0.33333、
val 0.92350/0.00000，与字面量逐位相符），但作为常量有两个后果：

1. **陈旧**：A 的归档若被重跑（换了 clip 或 checkpoint），草稿仍打印旧值，
   且**没有任何机制会发现**——正是本项目已经犯过的转抄错误类型；
2. **越权**：该句会为**任何** A2 束打印 A 的差值。若 A 的归档不可读，草稿会
   引用一个**没被测量**的参照，而 §72.11 同时声称没有手抄值。

修复：新增 `A_REFERENCE_RUN`（= A 的归档路径）与 `a_rank_gap()`，从 **A 自己的
归档产物**推导同一组聚合量（与在被测臂上用**同一条代码路径**），仅在两侧都可用时
并列；否则输出

> A 的同口径参照**不可用**（A 的归档排序产物缺一侧）；故本段只报本臂自身的
> train−val 差，不与 A 并列。

已核对：推导结果与原先硬编码值**逐位一致**（`auc` 0.999435−0.923504、中位
0.333333−0.000000），故这次改动**不改变任何已发表读数**，只把来源换成可核验的。
新增 2 条测试：一条禁止 `0.9994/0.9235/0.0759` 出现在生成器源码并断言推导值，
一条断言不可用时输出的是「不可用」而不是引用值。

**验证**：`test_semA_ep6_record` **107 passed**（105 + 2）；§72 整节重渲染与文档
仍 **IDEMPOTENT**（A 的草稿不受影响——A 自己的草稿不与自身对照）；判定器/rank/
m4floor 相关 54 passed。**边界**：未改门槛常量/模型/探针；训练未动。

#### 73.6.19 判定器**控制台**的 S1 聚合行也只看 M1–M4（2026-10-11，训练中修复；A2-S ep6 未产出）

§73.6.17 缺陷二修的是**渲染器草稿**（§72.0/§72.9）的同一越权。本轮把同一判据
施加到**判定器自己的控制台输出**——那是人工裁决时**真正看到**的东西，且比草稿更
靠前（看门狗先跑 `semA_ep6_verdict.py`，再渲染 `section72_draft.md`）。

实测复现（构造 M1–M4 全绿、M1′ 实测 FAIL 的 A2-S 捆）：控制台连打两行

```
S1 AGGREGATE: ALL PASS
A2 AGGREGATE: NOT ALL PASS -- M1-M4 all passed but the A2 added gates failed
              (M1p=FAIL, M1loc=PASS). The arm does NOT enter ep16.
```

**两行对同一臂给出相反结论**，且第一行是「ALL PASS」。虽然输出里确实又有
`A2 AGGREGATE` 行（§73.6.6 缺陷二补的），但**只读第一行的人会看到放行**——
这正是本项目反复出现的失败模式：**正确的读数旁边放一个更容易被读到的错误结论**。

修复：`S1 AGGREGATE` 行在 `a2 is not None and not (all_pass and a2_all_pass)` 时
直接输出 `NOT ALL PASS`，不再单独按 M1–M4 报「ALL PASS」。A 代（`a2 is None`）
分支**逐字节不变**。

实测三种情形：

| 臂 / 构造 | `S1 AGGREGATE` | `A2 AGGREGATE` |
|---|---|---|
| A（无 A2 配置路径） | `ALL PASS`（不变） | （不打印） |
| A2-S 增补门全绿 | `ALL PASS` | `ALL PASS … may be requested` |
| A2-S M1p FAIL | `NOT ALL PASS` | `NOT ALL PASS … does NOT enter ep16` |
| A2-S M1loc FAIL | `NOT ALL PASS` | `NOT ALL PASS … does NOT enter ep16` |

新增 2 条测试（真实 `subprocess` 跑交付的控制台脚本，不是只读源码）：一条断言
增补门失败时**不得**出现 S1 通过，一条守卫全绿时仍须报通过（防把真通过卡死）。

**验证**：11 个文件合跑 **264 passed**（260 + 4）；A 的归档判定与 §72 草稿重渲染
仍**逐字节/逐位不变**（`S1 AGGREGATE: NOT ALL PASS` 与之前一致）；§72.11 边界
未变。**边界**：未改门槛常量、探针、证据 JSON；训练未动；A2-S 的 ep6 仍未产出，
不存在事后调门。

#### 73.6.20 A2 预登记要求的**类别覆盖报告字段**从未被生成器输出（2026-10-11，训练中修复；A2-S ep6 未产出）

A2 预登记 §5 的采样契约把这四项之一写成**必须存在**：「**不按类别剔除
clip——类别覆盖作为报告字段给出，不作为选样条件**」，并实测记录了 val 20 clip 的
占用类数分布（1 类 ×4、2 类 ×11、3 类 ×4、4 类 ×1，train 1 类 ×7、2 类 ×9、
3 类 ×4）。**但 `semA_ep6_record.py` 从未输出过任何覆盖读数**——该字段既不在
§72.6 也不在 §72.1。后果是归档里读不到「某类只覆盖少数 clip」这一**已知会削弱
类别专属结论**的事实：

* A2-S 这条臂的 M4 判定（`own_hit` 已改按 pooled floor 判）就依赖窗口的类混合；
* §69 早已指出 A 的 M4 窗口**一个 Car 都没有**（n=10 时 own 主类只有 Ped 3/Cyc 6/
  Truck 1，donor 主类 Ped 3/Cyc 4/Truck 3），而 pooled floor 0.5128 由 Cyc 主导
  （pooled class counts `[5, 8, 20, 6]`）。**Car 在有 4 类候选的情况下根本没被这个
  窗口检验过**，而报告里只写了一个总体 `own_hit`。

修复：新增 `class_coverage()`，从**本包自己的** rank 产物（与判据同一冻结 clip
栅格）推导每 clip 的占用通道数分布，并在 §72.6 输出 val 与 train 两侧。**A 代
不输出**（A 的 rank 产物是事后补跑的，不属于其冻结契约，故对 A 保持
「未测量」而不是借用 A2 的读数）。

已核对：推导出的分布与预登记手记的实测值**逐位一致**
（val `{1:4, 2:11, 3:4, 4:1}`、train `{1:7, 2:9, 3:4}`），即这次改动**不产生任何
新读数**，只把预登记已承诺的字段真正落到草稿里。

新增 2 条测试：一条构造上述分布并断言 §72.6 渲染出覆盖行与「不按类别剔除」，
一条断言 A 代**不发明**覆盖声明。

**验证**：`test_semA_ep6_record` **109 passed**；11 个文件合跑 **266 passed**
（264 + 2）；A 的 §72 重渲染仍 **IDEMPOTENT**（A 无覆盖行）。**边界**：未改门槛、
探针、证据 JSON；训练未动。

> **未闭环项（留给 A2-S 的 §73.7）**：Car 在 A 的 M4 窗口完全缺席这一点已记录，
> 但**该窗口的类覆盖是否也漏掉 Car 需要看 A2-S 自己的产物**——本条只修「报告
> 字段缺失」，不据此重判 A 的 M4（A 的归档判定不变）。

#### 73.6.21 `semA_m4floor_ep6.json` 被**零消费者**读取：M4 线的基准与 A2-R 的 S0-5 都不在草稿里（2026-10-11，训练中修复；A2-S ep6 未产出）

§73.6.13 提到 A2 的 `m4floor` 产物用于「M4 修正后口径」（`own_hit` 改按 pooled
floor 判）。但扫消费者时发现：**`semA_ep6_verdict.py` 只声明了常量
`A2_M4FLOOR`，从未读过该文件；`semA_ep6_record.py` 也从未读它**。于是这份产物
在**任何报告里都不出现**，而按预登记它承载两项**被判据依赖**的读数：

1. **M4 的 `own_hit` 线由它推出**（`pooled_prior_floor`，A 为 0.5128）——
   §73.6.6 缺陷三把 `gate_m4` 改成用 pooled 地板后，
   `semA_ep6_record.py` 渲染 M4 行时**仍只打印 `m4.get("pooled_prior_floor")`**
   （来自 gate JSON，A 的 gate 里该字段是 `None` ⇒ 渲染成 n/a），
   而地板数字其实**就在 m4floor 产物里**；
2. **A2-R 的 S0-5 预期读数** `frac_cells_residual_can_flip`（§5 明写「应高于 A 的
   0.2530；记录实测值；不设通过线」）——该读数只在 m4floor 产物里。

⇒ **测了但没有任何报告携带的读数，与「没测」在记录上不可区分**，这正是本轮审计
反复出现的失败模式。

修复：新增 `m4floor_readings()` 与 §72.6 的「M4 地板 / 读出存活读数」表，输出

* `pooled_prior_floor`（M4 线的基准）、`recomputed_mean_own_hit`、
  `margin_over_pooled_prior`；
* `frac_cells_residual_can_flip`（**显式标注为 A2-R 的 S0-5 预期**）；
* `residual_gap_fg_mean` / `conv_cls_gap_fg_mean`（稀释程度）；
* `head_hit_by_residual_scale_pooled` 的**尺度曲线**（`residual=0` 与强制满尺度
  在两端）。

**仅 A2 臂渲染**（函数先经 `a2_readings`，A 返回 `None`）。A 的 §71.9.18 已手写过
这些读数，草稿不再重列，故 A 的 §72 **逐字节不变**（已 diff 验证）。

新增 2 条测试（渲染 + A 不发明读数）；`test_semA_ep6_record` **111 passed**；
11 个文件合跑 **268 passed**（266 + 2）。**边界**：未改门槛、探针、证据 JSON；
训练未动；A 的归档判定与草稿不变。

#### 73.6.22 看门狗里的 z-util 重复块：同一产物被**跑两遍**（2026-10-11，训练中记录；A2-S 仍在跑，**刻意不动脚本**）

扫「产物 → 消费者」时顺带核了**重复写入**。`watch_semA2_ep6.sh` 把
`probe_z_utilization_standard.py` 调了 **5 次**，但只有 **3 个**不同输出：

| 输出 | 调用次数 | 参数 |
|---|---:|---|
| `z_util_std_ep6_w3.json` | **2** | `--start-index 3 --limit 4 --batch-size 4 --shuffle-offset 2`（两侧**完全相同**，仅 `CUDA_VISIBLE_DEVICES` 不同：7 / 6） |
| `z_util_std_ep6_w100.json` | **2** | `--start-index 100 …`（同上，7 / 7） |
| `z_util_std_ep6_w1800.json` | 1 | `--start-index 1800 … --shuffle-offset -1` |

成因与 §73.6.5/§73.6.10 同类：早期版本的回答是「补一个只跑 A2 专属探针的块」，
后来为修「证据不全」**又整段追加了 A 家族的三个窗**，两段叠在一起。对照 A 的
`watch_semA_ep6.sh`：**每个窗只跑一次**。

**为什么不是正确性问题**：探针 `main()` 里 `torch.manual_seed(0)` /
`np.random.seed(0)` 固定，同一 checkpoint、同一参数 ⇒ 第二次写入与第一次**逐字节
同源**，产物不会不一致；后跑的那次只是**覆盖同值**。代价是**每个重复窗多一次
前向 rollout**（`--limit 4`，约 1–2 min GPU 时间）。

**为什么这次刻意不改**：A2-S 的看门狗**仍在运行**。bash 按**字节偏移**读取
正在执行的脚本，中途改写会让它从错误偏移继续（§73.6.5 已踩过这个坑，当时只能用
幂等补跑器兜底）。故本轮**只记录**，与 §73.6.10 对 `_a2r.json` 的处理一致。

**已写入 TODO 的清理项**：待 A2-S 结束、A2-R 开跑**之前**，删掉重复的两个块
（保留后追加的那组，因为它们与 A 家族的参数逐字一致），并把 `_a2r.json` 的冗余块
一并去掉——`_a2r.json` 同样**零消费者**（§73.6.10 已记）。

**边界（写作时）**：纯记录，未改脚本、未改判据、未重启训练进程。

**后记（2026-10-11 15:28 UTC）**：本条列出的清理项**已在 A2-S 结束、A2-R 开跑之前执行**——
重复的两个 z-util 块与 `_a2r.json` 冗余块均已删除，BN 批量同时固定为 2
（详见 §73.6.25，commit `c25ca30`）。

#### 73.6.23 预登记 §5 的**补充排序读数**有产出、**零消费者**（2026-10-11，ep6 后修复）

A2 预登记 §5 把三条**报告字段**写进冻结读数：`mean_gt_prob`、`mean_bg_prob`、
以及「唯一 argmax 格到最近 GT 格的曼哈顿距离」（逐通道后按 clip 等权）。它们**不是门槛**，
用途是把「峰漂多远」与「峰预算是否被背景占满」（`recall@top-k`）拆成**两个**现象。

探针 `probe_semA_rank.py` 从第一版起就把这三条写进
`semA_rank_val_ep6.json` / `semA_rank_train_ep6.json`；但生成器侧的聚合函数
`a2_split_contrast()` 只取 `auc` / `gt_shuffle_auc` / `max_gt_prob` / `recall@top-k`
四项，**从未读取新增字段**。结果是 §5 规定的三条读数在**每一份**草稿里都不出现，
而**没有任何报错**——JSON 里有、渲染里没有，与 §73.6.20/§73.6.21 同一失效模式
（产物与消费者各自演进，缺一环不报错）。

**修复（commit `4f1a21c`）**：`a2_split_contrast()` 增加三字段聚合，
train/val 对照块渲染**第二张表**。

**实测值**（frozen 20-clip 契约，`--start-index 3 --stride 100 --limit 20`）：

| split | `mean_gt_prob` | `mean_bg_prob` | argmax 距离（格，均值）|
|---|---:|---:|---:|
| train | 0.4570 | 0.0020 | 2.79 |
| val | 0.1794 | 0.0012 | 31.02 |

**边界**：A 的 §72 在修复后重渲染**逐字节不变**（A 包无该补充块）；数字只作机制定位，
不进门、不改任何已归档判定。

#### 73.6.24 split「来源归属」被 z 契约分支**吞掉**（2026-10-11，ep6 后修复）

`semA_zcontract_ep6.json` 由**另一个**探针（`probe_semA_zcontract.py`）产出，
而 A2 看门狗**不跑**该探针 ⇒ A2 的捆绑包**本来就不该有**这个文件。

生成器里「来源归属」一行原先被嵌在 `if os.path.exists(zc_path):` 分支**内部**。
在真实 A2-S 包上该分支不成立，于是这行连同它自带的
**`来源归属：NOT MEASURED`** 兜底一起**静默消失**（修复前实测：
`'来源归属' in text == False`）。危害是**语义性**的：缺它时读者分不清
「z 契约错配未测」与「split 归属未测」——这是**两个互相独立的缺口**
（z 契约对账 vs split/BN 占比），静默缺失会被读成「本条不适用」。

**修复（commit `0d13fbd`）**：正文抽成 `_render_split_attribution()`；
只有当 z 契约小节**退化成 FILL 桩**时才独立输出兜底行，因此
**A 的 §72 重渲染仍逐字节不变**。已按修复前的生成器做**反向验证**
（旧代码下该行为 `False`，新代码下为 `True`）。

#### 73.6.25 BN 探针被 OOM 掉且**看门狗静默继续**；重复 z-util 一并删除（2026-10-11，训练中修复）

**（a）静默证据损失**：`watch_semA2_ep6.sh` 以工具**默认** `--batch-size 4` 调用
`probe_semA_bnmode.py`，而 A 家族归档口径是 **2**。该探针在 `rssm_fusion.kl_loss`
处 CUDA OOM（已分配 22.6 GiB、申请 210 MiB）⇒ `semA_bnmode_ep6.json`
**从未写出**，而 battery **不检查探针退出码**，照常往下跑并打印「probes done」。
⇒ 一个失败探针在日志里**与成功无法区分**。

**（b）重复调用**（§73.6.22 已记录）：w3/w100 两窗各跑两遍。

**修复（commit `c25ca30`）**：BN 探针固定 `--batch-size 2`；删除重复的两个 z-util 块。
`--batch-size 2` 同时是**唯一跨代可比**的设置：BN 的 `train()` 统计量沿 **batch 维**
估计，batch 4 与 batch 2 得到的**不是同一个读数**。

**测试**：`test_watch_semA_ep6_cli.py` +37 行（该文件共 26 条），断言七支探针**各出现一次**、
BN 调用带 `--batch-size 2`。

**边界**：**脚本修好 ≠ 本 run 已补齐**。A2-S 的 `semA_bnmode_ep6.json` 与
`semA_zcontract_ep6.json` **仍缺**（§74.3）；本臂 BN 实测与 z 契约对账**未完成**。
未改判据、未重启训练、未补跑 GPU 探针。

#### 73.6.26 「产物字段 → 消费者」**系统扫描**（2026-10-11，A2-R 训练中；只读）

§73.6.20 / §73.6.21 / §73.6.23 / §73.6.24 是**同一个失效模式的四次实例**：
某处产出、某处本该消费，缺一环**不报错**。此前四次都是**撞上**才修；本条改为**主动扫一遍**。

**方法（可复跑）**：对 A2-S 捆绑包里每个 `*.json` 取**顶层键**，在
`semA_ep6_record.py` 与 `semA_ep6_verdict.py` 源码里按标识符**整词**搜索；
命中 = 有消费者，未命中 = 待人工判定。

| 产物 | 顶层键 | 无消费者 |
|---|---:|---:|
| `semreadout_ep6_gate.json` | 11 | 0 |
| `z_util_std_ep6_w3/w100/w1800.json` | 各 5 | 0 |
| `semA_rank_val/train_ep6.json` | 各 11 | 2（`ann_file` / `reading_note`，纯元数据） |
| `post_source_std_ep6_stride700.json` | 21 | 9 |
| `semA_m4floor_ep6.json` | 26 | 12 |

**结论一：不存在「预登记要求、却没人渲染」的字段。** 把上表每一处未命中项逐个人工比对
两份预登记的 §5 报告字段清单，命中数为 **0** ⇒ 四次已修缺陷属**契约层**，
剩下的未消费字段是**诊断**、不是违约。这一点必须写明，否则「未消费」会被误读成「漏报」。

**结论二（landmine，记录不改）**：`semA_m4floor_ep6.json` 的 `chance_1_over_c` = **0.25**
是 M4 `donor_transfer` 门槛的**基数**，而生成器只渲染 `donor_transfer` 本身、
**不渲染这个基数** ⇒ 「chance + 20pt = 0.45」这条线**无法从渲染结果自身核验**。
今天数值上无害（C=4 ⇒ 1/C 恰为 0.25）。**不改生成器**的理由：M4 块为 A 与 A2 **共用**，
加一行会让 A 已归档的 §72 重渲染**不再逐字节相同**——而 §73.6.23/§73.6.24 两处修复
特意保住的就是这个不变量。⇒ 属**换 A 归档一致性**的取舍，留待拍板。

**结论三（聚合口径 landmine）**：同一名目下存在**不同聚合**的字段，彼此不等但都正确：

| 名目 | 未渲染变体 | 已渲染变体 |
|---|---|---|
| head 命中（残差=训练值） | `head_hit_residual_as_trained` **0.740**（通道 pooled 曲线 `×1.0` 点） | `recomputed_mean_own_hit` **0.770**（逐 clip 后等权均值） |
| `conv_cls` 前景间隔 | `conv_cls_gap_fg_median` **1.954**（逐 clip 中位 → 再均值） | `conv_cls_gap_fg_mean` **2.520**（逐 clip 均值 → 再均值） |
| 类别先验地板 | `per_clip_prior_floor_mean` **0.692**（逐 clip 地板 → 再均值） | `pooled_prior_floor` **0.500**（池化占用通道后取多数类占比；§5 指定的那一支） |

三者**都不是矛盾数字**；但把未渲染那列误读成已渲染的同一读数，就会读出
「0.74 vs 0.77 不一致」这类**伪冲突**。**M4 门用的是 pooled 支**（§5 明写），判定不受影响。

**结论四（字段名断言了不存在的东西）**：`post_source_std_ep6_stride700.json` 的
`e_t_pooled_normalized` 与 `e_t_pooled_raw` **逐位相同**（rms 0.5792 / batch-common 0.3981 /
sample-specific 0.4207 / 特异占比 0.5276）。成因**不是**「归一化恰好无效」，而是**构造上**：
探针只在 fusion 具备 `_posterior_observation` 钩子时才走归一化支，
而 `SemanticReadoutRSSMFusion` **没有**该钩子，故 else 支直接
`e_pooled = captured['e_t']`（`tools/probe_posterior_source.py:158–161`）。
⇒ 字段名承诺「raw vs normalized」的对比，而在本类上两者**必然相等**。
另一处同源重复：`posterior_head_output` 与 `mu_q` **逐位相同**（同一张量挂两个键）。
两者**今天都无消费者**，故没有错误数字进入任何报告；记下来是为了**下次别把它当独立证据**。

**结论五（不是缺陷）**：`semA_splitprobe_val_full_n20.json` /
`semA_splitprobe_train_full_n20.json` / `semA_bnmode_ep6_trainsplit.json`
**全仓库没有任何产出者**。这是**刻意**的：生成器把这两组源写成「两侧必须同时存在，
否则显式报 NOT MEASURED」（`SPLIT_ATTRIBUTION_SOURCES` 的注释已说明来由）
⇒ 属**优雅降级**，不是漏产出。记录它是为了**下一次扫描不再重复报警**。

**边界**：只读扫描；未改生成器、未改任何 JSON、未改门槛、未重跑 GPU 探针。
未消费字段的**去留**（渲染 or 删除）是需要拍板的取舍，本次只记录。

**补记（2026-10-11 16:35 UTC）：把扫描**递归到叶子字段**。**
第一版只查**顶层键**，会漏掉全部嵌套读数——而本代最有价值的几个读数
（shuffle 溯源、绝对尺度、钳制率）恰恰都在嵌套层。递归后（字典下钻，
列表取首元素为代表）：

| 产物 | 叶子字段 | 无消费者 |
|---|---:|---:|
| `post_source_std_ep6_stride700.json` | 14 | 2（`posterior_weight_rms_h_half` / `_e_half`）|
| `semA_m4floor_ep6.json` | 44 | 24 |
| `semA_rank_val/train_ep6.json` | 各 40 | 各 21 |
| `semreadout_ep6_gate.json` | 61 | 12 |
| `z_util_std_ep6_w*.json` | 各 22 | 各 14 |

（数目变大有一部分是**同义重复**：`checkpoint_path` / `config_mtime` /
`config_size_bytes` 在三支探针里各写一份、都无消费者——属溯源元数据，
不是缺陷，同结论五。）

**（1）shuffle 溯源被验证（正面结论）**：`per_sample[*].shuffle_source_index`
在三个窗口共 12 个样本上**全部 ≠ 自身 index**（w3: 3→5、4→6、5→3、6→4；
w100 与 w1800 同构）。若 donor 取成自己，「shuffle」就退化成**恒等**、
整条反事实读数失效——**该不变量成立，但没有任何报告渲染它**。本次记下来。

**（2）三个窗口的绝对尺度不可比**：`feature_norm` **3570 / 3700 / 5189**、
`head_norm` **18620 / 19143 / 25493**、`head_scores.score_max`
**0.13 / 0.09 / 0.93**、`score_gt_0p3` **0 / 0 / 43–51**（顺序 w3 / w100 / w1800）。
⇒ 远窗（1800+）的绝对幅度系统性更高。
这**不影响**已渲染的读数：§72.7 渲染的全部是**比值**
（`*_l2_ratio_vs_normal`），按构造尺度无关——但这也正说明「三窗」在
**绝对**读数上不可直接比较，须与 §74.8「三窗索引不等于标注分层」的提示同读。

**（3）部署路径下的 KL 钳制率**：`normal_rssm_stats.stat_clamped_ratio`
= **0.9957–0.9979**（三窗 12 样本），即自由比特地板在**部署（`mu_q`）**路径下
同样几乎恒被触发，与本家族长期读数（如 §53「clamp 99.9%」）一致。
但记录与判定器**都没有渲染它**（两个文件里 `clamped` 一次都不出现）
⇒ 属「诊断可用但未进报告」，与结论二同类，一并留待拍板。

**边界（补记）**：仍为只读；上述三条**不改变**任何门槛或已归档判定。

#### 73.6.27 `residual_class_map` 把 A2-R 的 **48 通道**残差**静默误读**为 4 类（2026-10-11，A2-R ep6 之前修复）

**这是本代最严重的一处探针缺陷：它会让 A2-R 的 M4 门与预登记 S0-5 读数变成「像读数的垃圾」。**

**两种合法宽度**。`Anchor3DHead._expand_cls_logit_residual`
（`mmdet3d/models/dense_heads/anchor3d_head.py:213`）**同时接受**两种通道数：

* `num_classes`（4）——候选 A 的 `per_class_slot`：fusion 只发 4 个类 logit，
  由 head 按 `anchor_class_mapping` 表展开到 48；
* `cls_out_channels`（48）——**A2-R 的 `anchor_slot`**：fusion 在**自己内部**就把
  每个类广播到该类的全部 anchor 槽（`me_rssm/semantic_readout_rssm.py:505–518`），
  送到 head 时已是 48，head 原样透传。

**缺陷**。探针的 `residual_class_map()` 只做
`_to_sem_grid(r[0]).reshape(num_classes, -1)`，**无条件假设 4 通道**。
48 通道时它**不报错**（48 能被 4 整除），而是产生**12 倍**多的「格」：
`.reshape(4, -1)` 把 12 个通道的显存量当成一个类轴，于是用**真实前景格号**去索引时，
读到的是**别的通道**，类 c 的证据会被记到**别的类**头上。
这正是本项目反复出现的失效模式：**形状合法、数值有限、无人报错**。

**爆炸半径（A2-R 下会错、A2-S/A 下不会错）**：

| 调用点 | 影响的读数 | 性质 |
|---|---|---|
| `probe_semreadout_ep6.py:657–658` | M4 `own_hit`、`donor_transfer` | **门（S1）** |
| `probe_semA_m4floor.py:99,110` | `own_hit`、`residual_gap_fg_mean`、`residual_absmax_fg_mean`、**`frac_cells_residual_can_flip`** | **A2-R 预登记 S0-5（§5）** |

**未受影响**：`head_swap_to_donor` 走 `head_scores_with_residual()`，把**原始**残差交给
head，48 通道原样透传 ⇒ 该读数一直是正确的；`conv_cls` 侧（`class_scores_from_head`）
本来就按 anchor 归约，也不受影响。

**发现时点**：2026-10-11 ~16:10 UTC，A2-R 训练在 ep1 iter≈550。⇒ **ep6 只读电池尚未跑过**，
没有任何 A2-R 读数是用坏公式算出来的，**不构成对任何已归档判定的重判**。

**修复**：宽形态按 `class_scores_from_head` **同一条 max-over-anchor 规则**归约
（否则残差与它所对比的 logits 会用两套聚合）；宽度不是 `num_classes` 整数倍时
**显式抛错**——本缺陷的要害正是「沉默」。

**测试**：`test_probe_semreadout_ep6.py` +3（现 43 条），其中
`test_anchor_slot_is_not_the_narrow_reshape` 是**显式回归闸**：它复现修复前的误读
（形状合法、证据落到 class 3 而非 class 1），而不是只断言形状/范围——
只断言形状的话，坏答案照样能过。

**验证 A2-S 不受影响（同 bundle 内部一致性）**：把 m4floor 逐 clip 表按 `cells` 加权
重算 pooled `own_hit`，与探针的 `head_hit_residual_forced_max` **六位小数相同**
（均 **0.690000**）。`×1e6` 会把 head 的 argmax 强制变成残差的 argmax，
故两者相等**说明 4 通道下残差图被正确读出**。同一组数字也再次印证 §73.6.26 的聚合口径提示：
**cell-pooled 0.690 vs clip-mean 0.770**，两者都对。

**边界**：只动探针侧（`probe_semreadout_ep6.py` 一个函数 + 测试）；
**未改模型、未改门槛、未改 JSON、未打断或重启 A2-R 训练**。
该修复只影响**尚未产生**的 A2-R ep6 读数。

#### 73.6.28 A2-R 草稿导语承诺了它**从不评估**的 M1-loc 门；合成捆绑彩排（2026-10-11，A2-R ep6 之前）

**方法**：把 A2-S 的整包产物复制到临时目录，逐份把 `config_path` 等处的
`semA2S` 改成 `semA2R`，在 ep6 **之前**用一个**合成 A2-R 捆绑**把判定器与渲染器
**整段彩排**一遍。目的：ep6 电池跑完才发现渲染器崩在 A2-R 上（会直接丢掉草稿）
是不可接受的。

**彩排通过的部分（正面结论）**：判定器正确识别 A2-R——
`gates == {'M1p'}`（**不**引入 M1-loc）、打印
`M1-loc : n/a for A2-R (pre-registered A2-S only)`、总行
`A2 AGGREGATE: NOT ALL PASS (M1-M4 not all passing; M1p=PASS)` 不含 M1-loc；
渲染器的臂标签（`候选 A2-R`）、唯一变量（`cls_readout_mode`）、对照（A 底座）
也都正确。

**彩排抓到的缺陷**：两段导言语**写死**为
「（A2 臂另跑 M1'/M1-loc 的双 split 排序探针与 M4 地板探针）。」，
对所有 A2 臂一律照抄。M1-loc 是**预登记的 A2-S 专属门**（§5），
于是 A2-R 的草稿会在开头声称自己跑了 M1-loc，而**同一份草稿往下三行**
就是判定器的 `n/a for A2-R`——**草稿与自己的判定表互相矛盾**。

**修复**：`A2_ADDED_PROBE_PROSE` 按臂给出导语（未知臂回落到中性句），
实测渲染：

* A2-S：`（A2-S 臂另跑 M1'/M1-loc 的双 split 排序探针与 M4 地板探针）。`
* A2-R：`（A2-R 臂另跑 M1' 的双 split 排序探针与 M4 地板探针；M1-loc 为 A2-S 专属，本臂不评估）。`

**测试**：+1（`test_header_prose_matches_the_gates_the_arm_actually_runs`），
在**两条臂上各渲染一次**，断言导语与判定器实际评估的门一致。

**边界**：该句位于 `a2 is not None` 分支内 ⇒ **A 的 §72 重渲染仍逐字节相同**
（实测差异仅渲染日期 2026-10-10 vs 2026-10-11，与末尾 `---` 分隔符，
两者都不是本节正文）。未改判据、未改任何 JSON、未打断或重启 A2-R 训练。
**合成捆绑只用来彩排**：它不产生任何性能或机制读数，不得引用为证据。

#### 73.6.29 M4 的一个**靠数据成立、代码不设防**的隐含前提（2026-10-11，只读核验）

**机制**：`SemanticReadoutRSSMFusion.detection_cls_logit_residual(is_valid_mask)`
（`me_rssm/semantic_readout_rssm.py:301`）对**无效样本把整张残差清零**。
而探针 `rollout()` 传的 mask 是 `valid[n-2]`（前一帧是否有效）。
若某个被选中的 clip 在该位置为假，它的残差会**恒为全零** ⇒ `argmax` 恒为 **class 0**，
而 M4 的 `own_hit` / `donor_transfer` 会把这些**人造**格**当作真实预测计入**。
探针目前只挡 `run['residual'] is None`，**不挡全零**。

**实测前提在本代成立**（两个已归档捆绑都查过）：

| bundle | clip 数 | `residual_absmax_fg_mean` 范围 | 全零 clip | `own_hit==0` 的 clip |
|---|---:|---|---:|---:|
| A2-S | 20 | 0.9232 – 2.9314 | **0** | 0 |
| A | 10 | 0.6975 – 2.5069 | **0** | 0 |

⇒ 本代 M4 读数**没有**被这个机制污染；但这是一条**只靠数据成立**的保证，
**换一个窗口或换一条臂就可能不成立**。

**并发现在原处的过时注释**：`probe_semreadout_ep6.py:663–665` 写
「The donor residual must be resized/masked onto this clip's grid the same way
the detector does」，而紧接着的代码**直接透传** `donor_run['residual']`，
并未 resize/mask。严格讲，这一步同时换了 **donor 的 z 与 donor 的 mask**；
实测两边的 mask 都为全真（上表「全零 clip = 0」即其证据）⇒ **不构成混杂**，
但注释描述的是**意图**而非实现，容易让读者以为该步骤已经做过。

**行动**：不设防、不改代码（本代两个捆绑都已实测通过）。
**A2-R 的 ep6 归档时须复核同一读数**：`residual_absmax_fg_mean` 的最小值是否 > 0
即可当哨兵；若出现 0，该 clip 的 M4 格必须判为**无效**而不是计入。

**边界**：只读核验 + 记录；未改探针、未改门槛、未改任何 JSON、未打断 A2-R 训练。


## 74. A2-S ep6 实际完成与进展核验（2026-10-11 15:39 UTC）

### 74.0 当前结论与核验边界

此前 §73.6 与 TODO 的「训练中/ep6 尚未产出」是当时状态，本节以实际落盘产物更新。
**A2-S 已完成 ep1–6，S1/A2 总门 NOT ALL PASS；按开训前预登记归档，不进 ep16。**
失败项为 M1、M4 donor transfer、M1-loc；不是因为早期 AP 低于成熟基线。
M1′ 排序门与 M3 分类 CE 读出门通过，不能推导 z 排他性、概率状态必要性或四类稳定提升。

本次仅核查现有日志/权重/JSON、复跑 CPU `semA_ep6_verdict.py`（exit 0）并更新记录；
**未改网络、未调判据、未启动训练/GPU 探针，未重跑全量回归**。
A2-R 只有 450 iter smoke 目录、正式目录未出现；进程核验无相关训练/看门狗/探针运行。

### 74.1 配置、唯一变量与运行事实

- run：`work_dirs/fgfull_N4_no2d_igdr_2x4_24e_semA2S_seed0`。
- config：`me_rssm/configs/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_semA2S_24e.py`。
- 对照：A 的 `..._no2d_igdr_semreadout.py`；唯一变量为监督侧 `sem_target_mode='channel_contrastive'`，
  按 A2 §4.1 在既有语义监督上加逐通道排序压力；固定 q=.999、margin=1.0、λ_rank=.1。
  不改重建目标、KL/free-bits、prior/posterior、分类残差映射、backbone、几何头、优化器、数据管线、24e 日程。
- GPU5/6/7、seed0、deterministic；完整 24e 日程在 ep6 人工机制门前暂停，实际只有 ep1–6。
- 训练日志 `20261011_090445.log{,.json}`；开始 09:04:45 UTC，最后训练/val 日志 15:11:23 UTC。
  看门狗 15:11:39 UTC 命中新鲜 ep6 权重 + val，向训练发 SIGINT；15:11:54 GPU 释放。
  这是预期 gate 停止，不是训练崩溃；信号退出码没有记录，不补猜。
- 中位 1.569 s/iter、约 16.7 GiB/卡；228 条 train JSON 行，6 条 val，保存 ep2/4/6。
- ep6 checkpoint SHA256 前缀 `34a217b060bfdff9`，688860396 bytes；mtime 15:00:29 UTC。
  config SHA256 前缀 `3ea1c320d2441949`。
- 看门狗 15:16:08 UTC 写出 `section72_draft.md`；finisher 15:16:19 UTC 结束。
  草稿仅作数据源，**没有覆盖已归档 A 的 §72**。

### 74.2 完整逐 epoch AP 与统一摘要

全部为正式训练内 val，四类口径固定 Car/Truck strict、Ped/Cyc loose，3D moderate。
默认 fixed window ep12–16 **未形成**；以下实际跨度 ep1–6、n=6，仅早期上下文。

| ep | Overall3Dmod | Car_strict | Truck_strict | Ped_loose | Cyc_loose |
|---:|---:|---:|---:|---:|---:|
| 1 | 28.5104 | 40.8058 | 14.8641 | 20.7701 | 37.6018 |
| 2 | 31.6030 | 40.4025 | 18.4888 | 24.5688 | 42.9520 |
| 3 | 36.6408 | 46.2434 | 21.3330 | 28.4029 | 50.5839 |
| 4 | 33.0086 | 43.7233 | 21.5002 | 29.1275 | 37.6833 |
| 5 | 37.3276 | 46.7316 | 20.0300 | 29.4301 | 53.1186 |
| 6 | 34.9702 | 41.9027 | 25.2972 | 29.5437 | 43.1374 |

| 指标 | ep1–6 等权均值（n=6） | 全部已有 val 峰值 + epoch | best saved（仅可用性） |
|---|---:|---|---|
| Overall3Dmod | 33.6768 | 37.3276 @ ep5 | 34.9702 @ ep6 |
| Car_strict | 43.3015 | 46.7316 @ ep5 | 43.7233 @ ep4 |
| Truck_strict | 20.2522 | 25.2972 @ ep6 | 25.2972 @ ep6 |
| Ped_loose | 26.9738 | 29.5437 @ ep6 | 29.5437 @ ep6 |
| Cyc_loose | 44.1795 | 53.1186 @ ep5 | 43.1374 @ ep6 |

**同窗对照**（§73 的 seed0 ep1–6 参照）：标准主线 Overall 29.0420，A 34.0459，
A2-S 33.6768；A2-S 相对主线 +4.6348 点，相对 A −0.3691 点。
这不是成熟性能结论；没有 ep12–16，也没有多 seed，不能宣称所有类别改善。

**报告器残留口径问题**：当前 CPU 控制台/自动草稿还把 ep1–6 数值旁列 ep12–16 基线
39.2809（及四类成熟均值），并输出泛用 A 的 continuation 提示。
虽然标了 context only，但该跨窗差值不能用于性能排名、归因或 S2 止损。
本节不采用这些差值；以 A2 §5/§6 的生效契约和实际 S1 门为准。

### 74.3 门判定、产物完整性与未完成诊断

- M1 FAIL：thr50=.1458333 < .30；hit@1=.1041667 < .60。
- M2 形式 PASS：h/feat 替换相对 thr50 均为 −100%；属于监督头 OOD 替换，**不是真正 z 排他性**。
- M3 PASS：关分类残差使真实分类 CE 变差 17/20，门为至少 16/20；不是检测 AP 差。
- M4：own_hit=.7700216 > pooled 地板 .5 + .2；donor_transfer=.4119589 < .45，故整体 FAIL。
- M1′ PASS：AUC=.9508979 ≥ .80；相对 GT shuffle 增量 .3619491 ≥ .25。
- M1-loc FAIL：max_gt_prob≥.5 的 clip 仅 3/20（需≥12）；recall@top-k 中位 0（需≥.25）。
- M1-loc/M4 已实测失败，**跨网格 M1 例外不适用**；不靠改阈值/增益/KL 续训。

必需判定捆七项都在：`semreadout_ep6_gate.json`、三窗 `z_util_std_ep6_w*.json`、
`post_source_std_ep6_stride700.json`、val/train rank；M4 地板文件亦在。
但**不是所有扩展诊断完成**：`probe_semA_bnmode.py` 以默认 batch4 运行时 CUDA OOM
（已分配 22.61 GiB，申请 210 MiB），未写出 `semA_bnmode_ep6.json`。
看门狗继续后续探针，故「probes done」不能解释成 BN probe 成功。
`semA_zcontract_ep6.json` 同样缺失；采样 z 与部署 mu_q 契约的本臂对账未测。
这些缺口不能豁免已经实测的 M1-loc/M4 FAIL。

现有 commit `c25ca30` 已把看门狗 BN batch 固定为 2（与 A 同口径），并删除重复 z-util 调用；
**脚本修好 ≠ 本 run 的 BN 实测已补齐**。本次没有重跑 GPU 诊断。

### 74.4 A2-S 侧训练期同口径读数（train split，逐 epoch 末值）

| ep | `stat_sem_hit1` | `stat_sem_thr50` | `loss_rssm_sem` | `stat_cls_res_rms` | `stat_mu_diff_sq` |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.1948 | 0.1934 | 1.3013 | 0.9499 | 3.4621 |
| 2 | 0.2581 | 0.3036 | 1.0551 | 1.0320 | 0.1026 |
| 3 | 0.2930 | 0.3692 | 0.9272 | 1.1802 | 0.0708 |
| 4 | 0.3243 | 0.4239 | 0.7985 | 1.2451 | 0.0507 |
| 5 | 0.3753 | 0.4832 | 0.6968 | 1.3199 | 0.0691 |
| 6 | 0.3271 | 0.5051 | 0.6806 | 1.3988 | 0.0373 |

shuffle/zero 与门控读数无关；此处 `stat_sem_*` 为**训练统计**（z 采样值、train split），与探针的 val 口径不同，不可混比。
末 epoch `stat_cls_res_rms` = 1.3988（>0 ⇒ 读出活着）。


### 74.5 S1 门明细（逐 clip）

> **类别覆盖（val（冻结 20 clip），n=20）**：每 clip 占用通道数分布 1 类 ×4, 2 类 ×11, 3 类 ×4, 4 类 ×1。**不按类别剔除 clip**（§5：覆盖是报告字段，不是选样条件）；某类只覆盖少数 clip 时，该类专属结论在 ep6 只能是**弱证据**。
> **类别覆盖（train 同协议，n=20）**：每 clip 占用通道数分布 1 类 ×7, 2 类 ×9, 3 类 ×4。**不按类别剔除 clip**（§5：覆盖是报告字段，不是选样条件）；某类只覆盖少数 clip 时，该类专属结论在 ep6 只能是**弱证据**。

> **样本权重不均提示**：各 clip 的 GT 目标数并不相等——clip `1603` 只有 **1** 个目标，clip `1303` 有 **12** 个（§71.9.14 预检：3/4/3/11/4/2/6/6/1/2）。
> 逐 clip 比率在 `1603` 上只能取少数离散值、方差最大，
> 故下面 M1/M3/M4 的**分组结论不得由单个 clip 单独支撑**。

**M1 逐 clip**：

| index | thr50 | hit@1 | hit1_chance | excess_over_chance | 正格数 |
|---:|---:|---:|---:|---:|---:|
| 3 | 0.0000 | 0.0000 | 0.000056 | -0.000056 | 3 |
| 103 | 0.0000 | 0.0000 | 0.000075 | -0.000075 | 4 |
| 203 | 0.0000 | 0.0000 | 0.000056 | -0.000056 | 3 |
| 303 | 0.0000 | 0.0000 | 0.000131 | -0.000131 | 7 |
| 403 | 0.0000 | 0.0000 | 0.000037 | -0.000037 | 2 |
| 503 | 0.0000 | 0.0000 | 0.000205 | -0.000205 | 11 |
| 603 | 0.0000 | 0.0000 | 0.000205 | -0.000205 | 11 |
| 703 | 0.0000 | 0.0000 | 0.000075 | -0.000075 | 4 |
| 803 | 0.0000 | 0.0000 | 0.000056 | -0.000056 | 3 |
| 903 | 0.0000 | 0.0000 | 0.000112 | -0.000112 | 6 |
| 1003 | 0.5000 | 0.0000 | 0.000037 | -0.000037 | 2 |
| 1103 | 0.0000 | 0.0000 | 0.000131 | -0.000131 | 7 |
| 1203 | 0.1667 | 0.0000 | 0.000112 | -0.000112 | 6 |
| 1303 | 0.0833 | 0.2500 | 0.000224 | +0.249776 | 12 |
| 1403 | 0.1667 | 0.3333 | 0.000112 | +0.333221 | 6 |
| 1503 | 0.5000 | 1.0000 | 0.000037 | +0.999963 | 2 |
| 1603 | 1.0000 | 0.0000 | 0.000019 | -0.000019 | 1 |
| 1703 | 0.0000 | 0.0000 | 0.000075 | -0.000075 | 4 |
| 1803 | 0.0000 | 0.0000 | 0.000037 | -0.000037 | 2 |
| 1903 | 0.5000 | 0.5000 | 0.000075 | +0.499925 | 4 |

**M2 两控制臂逐 clip**（`thr50`；z 为参照）：

| index | z | h | feat |
|---:|---:|---:|---:|
| 3 | 0.0000 | 0.0000 | 0.0000 |
| 103 | 0.0000 | 0.0000 | 0.0000 |
| 203 | 0.0000 | 0.0000 | 0.0000 |
| 303 | 0.0000 | 0.0000 | 0.0000 |
| 403 | 0.0000 | 0.0000 | 0.0000 |
| 503 | 0.0000 | 0.0000 | 0.0000 |
| 603 | 0.0000 | 0.0000 | 0.0000 |
| 703 | 0.0000 | 0.0000 | 0.0000 |
| 803 | 0.0000 | 0.0000 | 0.0000 |
| 903 | 0.0000 | 0.0000 | 0.0000 |
| 1003 | 0.5000 | 0.0000 | 0.0000 |
| 1103 | 0.0000 | 0.0000 | 0.0000 |
| 1203 | 0.1667 | 0.0000 | 0.0000 |
| 1303 | 0.0833 | 0.0000 | 0.0000 |
| 1403 | 0.1667 | 0.0000 | 0.0000 |
| 1503 | 0.5000 | 0.0000 | 0.0000 |
| 1603 | 1.0000 | 0.0000 | 0.0000 |
| 1703 | 0.0000 | 0.0000 | 0.0000 |
| 1803 | 0.0000 | 0.0000 | 0.0000 |
| 1903 | 0.5000 | 0.0000 | 0.0000 |

**M3 逐 clip**（`delta_off_minus_on > 0` 表示关掉 residual 后
正样本 class-CE 变差，即读出有贡献）：

| index | CE(on) | CE(off) | delta(off-on) | 正格数 |
|---:|---:|---:|---:|---:|
| 3 | 1.3462 | 1.1701 | -0.1761 | 3 |
| 103 | 1.0024 | 1.0767 | +0.0743 | 4 |
| 203 | 0.7614 | 1.2158 | +0.4544 | 3 |
| 303 | 0.9284 | 1.0247 | +0.0963 | 7 |
| 403 | 0.2801 | 0.5631 | +0.2830 | 2 |
| 503 | 0.4214 | 0.4877 | +0.0663 | 11 |
| 603 | 1.3987 | 1.2492 | -0.1495 | 11 |
| 703 | 0.7961 | 0.8678 | +0.0717 | 4 |
| 803 | 0.3179 | 0.6876 | +0.3697 | 3 |
| 903 | 0.9262 | 0.9031 | -0.0231 | 6 |
| 1003 | 0.0372 | 0.1628 | +0.1255 | 2 |
| 1103 | 0.4423 | 0.6494 | +0.2071 | 7 |
| 1203 | 0.2561 | 0.4651 | +0.2090 | 6 |
| 1303 | 0.8369 | 0.9368 | +0.0999 | 12 |
| 1403 | 0.4349 | 0.6049 | +0.1700 | 6 |
| 1503 | 0.2878 | 0.4149 | +0.1270 | 2 |
| 1603 | 0.0081 | 0.1151 | +0.1070 | 1 |
| 1703 | 0.1858 | 0.4731 | +0.2872 | 4 |
| 1803 | 0.5568 | 1.2193 | +0.6626 | 2 |
| 1903 | 0.1723 | 0.7679 | +0.5956 | 4 |

**M4 逐 clip**（`own_hit` = z 残差 argmax 命中该格 GT 类；
`donor_transfer` = 换 donor z 后残差 argmax 命中 donor 主类；
`head_swap_to_donor` = 只重跑 head 且喂 donor 残差时的 head 类 argmax）：

| index | own_hit | donor_transfer | head_swap | donor_index | changed_frac |
|---:|---:|---:|---:|---:|---:|
| 3 | 0.3333 | 0.6667 | 0.6667 | 203 | 0.3333 |
| 103 | 0.5000 | 1.0000 | 0.7500 | 203 | 0.2500 |
| 203 | 1.0000 | 1.0000 | 0.0000 | 803 | 0.6667 |
| 303 | 0.4286 | 1.0000 | 0.4286 | 803 | 0.7143 |
| 403 | 1.0000 | 1.0000 | 0.5000 | 803 | 1.0000 |
| 503 | 0.7273 | 0.8182 | 0.0000 | 803 | 0.8182 |
| 603 | 0.3636 | 0.9091 | 0.0000 | 803 | 0.8182 |
| 703 | 0.7500 | 0.7500 | 0.2500 | 803 | 0.7500 |
| 803 | 0.6667 | 0.3333 | 0.3333 | 903 | 0.3333 |
| 903 | 0.6667 | 0.0000 | 0.0000 | 1103 | 0.6667 |
| 1003 | 1.0000 | 0.0000 | 0.0000 | 1103 | 0.5000 |
| 1103 | 0.7143 | 0.4286 | 0.1429 | 1403 | 0.5714 |
| 1203 | 0.8333 | 0.1667 | 0.0000 | 1403 | 0.5000 |
| 1303 | 0.5833 | 0.1667 | 0.4167 | 1403 | 0.5833 |
| 1403 | 0.8333 | 0.0000 | 0.0000 | 1503 | 0.8333 |
| 1503 | 1.0000 | 0.0000 | 0.0000 | 1703 | 1.0000 |
| 1603 | 1.0000 | 0.0000 | 0.0000 | 1703 | 1.0000 |
| 1703 | 1.0000 | 0.0000 | 0.0000 | 1903 | 0.5000 |
| 1803 | 1.0000 | 0.0000 | 0.0000 | 1903 | 0.5000 |
| 1903 | 1.0000 | 0.0000 | 0.0000 | 3 | 1.0000 |

**M4 地板 / 读出存活读数**（`semA_m4floor_ep6.json`，M4 的 `own_hit` 线即由这里的 pooled 地板推出）：

| 读数 | 值 | 说明 |
|---|---:|---|
| `pooled_prior_floor` | **0.5000** | 窗口内「恒投多数类」的准确率，M4 `own_hit` 线的基准 |
| `recomputed_mean_own_hit` | **0.7700** | 由残差现场重算，与地板同源 |
| `margin_over_pooled_prior` | **+0.2700** | `own_hit −` 地板（**机制读数**）|
| `frac_cells_residual_can_flip` | **0.3555** | A2-R 的 S0-5 预期读数（§5；不设通过线）|
| `residual_gap_fg_mean` / `conv_cls_gap_fg_mean` | **1.0889** / **2.5200** | 残差与 `conv_cls` 的前景间隔（稀释程度）|

head 命中随残差**尺度**的变化（`residual=0` 与强制满尺度在两端）：`×0.0 → 0.7000 / ×1.0 → 0.7400 / ×2.0 → 0.7500 / ×4.0 → 0.7300 / ×8.0 → 0.7000 / ×1000000.0 → 0.6900`。


### 74.6 家族标准反事实读数（`shuffle_z` / `zero_z` / `prior_only`）

三窗口与既往各代（MAI / MCI / DC-RSSM）使用同一探针语义：
`shuffle_z` 用同 batch 另一样本的 `mu_q` 替换，`zero_z` 置 0，
`prior_only` 只走先验。**本臂即跑在标准家族底座上**，故用
`probe_z_utilization_standard.py`（后验头重写 + 转发部署态 class
residual），与 `tools/diagnose_z_utilization.py` 的打分口径一致。

| 窗口 | shuffle_z head | zero_z head | prior_only head | shuffle/zero | shuffle_z feat | 标准 RSSM ep6 同窗 head（对照） |
|---|---:|---:|---:|---:|---:|---:|
| 近距（[3, 4, 5, 6]） | **4.3312%** | 46.3231% | 4.3630% | 0.093 | 17.3950% | 49.73% |
| 中距（[100, 101, 102, 103]） | **4.0824%** | 47.5387% | 5.1078% | 0.086 | 16.2050% | 3.30% |
| 远距（[1800, 1801, 1802, 1803]） | **2.5700%** | 35.6527% | 3.5973% | 0.072 | 9.8834% | 3.36% |

本 run 三窗 `shuffle_z` head：近距 4.33%; 中距 4.08%; 远距 2.57%。
对照为**标准 no2d_igdr RSSM seed1 ep6**同窗口（`docs/training_runs_full.md` §62.2），因同家族、同 epoch，
是唯一口径一致的机制对照；MAI/MCI/DC 各代的同窗读数见 §62/§63/§68。


### 74.7 posterior source 归因（`stride=700`）

| 量 | RMS | batch-common RMS | 样本特异 RMS | 样本特异能量占比 |
|---|---:|---:|---:|---:|
| `mu_q` | 0.6403 | 0.6071 | 0.2036 | 10.11% |
| `h_t` | 0.2423 | 0.2319 | 0.0702 | 8.39% |
| `e_pooled` | 0.5792 | 0.3981 | 0.4207 | 52.76% |

| 读出/依赖读数 | 值 |
|---|---:|
| `mu_q_removal_of_h_rel_l2` | 0.8934 |
| `mu_q_removal_of_e_rel_l2` | 0.4267 |
| **h/e 依赖比** | **2.094** |
| `posterior_obs_mode` | `l2` |
| `z_proj_init` | `None` |

> h/e 依赖比是各代通用的机制对照量（标准主线 ep12 为 2.338、MAI ep12 为 0.842、MCI ep12 为 0.605）；越低表示 posterior
> 越依赖观测而非 GRU 状态。

fusion 类：`SemanticReadoutRSSMFusion`、latent [216, 248]、readout `film`、clip 索引 [3, 703, 1403]。


### 74.8 机制归因、下一步与研究验收差距

- 正证据：排序信息存在（M1′ PASS），分类残差对真实 CE 有贡献（M3 PASS）。
- 负证据：train 的 recall@top-k 中位 .3333，val 仍为 0；train/val 有效峰预算未打通。
  val AUC .9509 相比 A .9235 增高，但高 AUC 不等于极端 top-k 定位成功。
  排名改进不足以满足对象峰定位与置信度契约，不可写成「z 无信息」。
- donor transfer 未过，z 类别干预的可迁移证据不足；M2 不替代匹配训练的 h/feat 控制。
- posterior 样本特异能量仅 10.11%、h/e 依赖比 2.094；三窗 shuffle head 仅约 2.57–4.33%。
  这些是特征/头反事实读数，**不是 AP 变化**，不同 seed/epoch 参照仅作机制背景。
  三窗索引是既定样本窗口，不等于经过标注分层的物理近/中/远距离。
- 按 A2 §6：A2-S 归档后下一独立臂为 A2-R（覆盖全部 48 分类通道，同容量），
  不组合两臂、不续训 A2-S。A2-R 实现与 S0 已完成，**正式训练尚未开始**。
- A2-R 归档后才凭证据选择 C（缺测分组）或对象级读出 B；D 后置。
  C/B/D、C1/C2/C3 匹配消融、同容量确定性对照、真正 z 排他性、ep12–16、多 seed 均未完成。
- 原差分重建方向仍按 §70.6 暂停：真实目标并非 pool16 标准化模板，原立项承重证据不成立。
- 研究总体：工程契约与实验管线有进展；**zt 失效修复、随机状态必要性、四类稳定提升仍未验收**。

### 74.9 修复后重渲染：两条此前**渲染不出来**的读数（2026-10-11 15:56 UTC）

`74.0–74.8` 的渲染上游是 15:16:19 写出的 `section72_draft.md`，该草稿**早于**两个生成器修复
（`4f1a21c` 补充排序读数、`0d13fbd` split 归属行）⇒ 上面的表格用的是**修复前**渲染。
以**修复后**的生成器重跑**同一 bundle**（同一批证据 JSON，未重跑 GPU 探针、未改任何 JSON、
未改判据）后，草稿**多出两块**内容：

**（1）补充排序读数表**（预登记 §5 的报告字段，逐通道后按 clip 等权）：

| split | `mean_gt_prob` | `mean_bg_prob` | argmax 距离（格，均值）|
|---|---:|---:|---:|
| train | 0.4570 | 0.0020 | 2.79 |
| val | 0.1794 | 0.0012 | **31.02** |

这两行给 §74.8 的「有效峰预算未打通」一个**定位读数**：同一探针、同一契约下
**train 的 argmax 平均只漂 2.79 格、val 漂 31.02 格（约 11×）**，与 §71.9.20
记下的**双峰漂移**（失败集中在少数场景）方向一致。同时两 split 的 `mean_bg_prob`
都极低（0.0020 / 0.0012）⇒ 失效**不是**「背景概率整体抬高」，
而是**稀疏的峰落在了远处**。两条都只作**机制定位**，不进门、不进 S2。

**（2）`来源归属：NOT MEASURED`**：本包无 split 对照产物（`semA_zcontract_ep6.json`
由另一支探针产出，A2 看门狗不跑它）；修复前该行被 z 契约分支吞掉（§73.6.24）。
它是**独立于**「z 契约错配未测」的第二个缺口，不得互相掩盖。

**仍缺**：§72.10 的 z 契约对账仍是 `<!-- FILL -->` —— 采样 z（`mu_q + σ·ε`）与
部署 z（`mu_q`）的契约对账**本臂未测**（§74.3）。

**边界**：**不重判**。S1/A2 总判定仍是 **NOT ALL PASS**（M1 / M4 donor / M1-loc FAIL），
不续 ep16；补出的两条读数**不改变**任何门的数值或结论。

### 74.10 A2-R 正式训练启动（2026-10-11 15:53 UTC）

§74.8 末条「A2-R 正式训练尚未开始」在本小节写出后**已过期**，按此更新：

- **臂**：`me_rssm/configs/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_semA2R_24e.py`。
  唯一变量 `cls_readout_mode`：`per_class_slot`（候选 A，12/48 通道）→
  `anchor_slot`（A2-R，**写满 48** 通道）。**与 A2-S 不组合**，同 predictor、同参数量。
- **启动事实**：2026-10-11 **15:53:23 UTC**，3 卡 **GPU5/6/7**、seed0、`--deterministic`、
  24e 余弦日程、**从零起训**（非热启动基线权重）。
- **run dir**：`/data/lurui/work_dirs/fgfull_N4_no2d_igdr_2x4_24e_semA2R_seed0`；
  作业日志 `/tmp/a2r_24e.log`，run dir 内另有 `20261011_155323.log{,.json}`。
- **看门狗**：`ARM=A2R` 挂 `me_rssm/sanity/watch_semA2_ep6.sh`，15:54:17 UTC 起等 ep6。
  运行的是**已修复版本**（含 §73.6.25 的 `--batch-size 2`、每支探针各跑一次，
  以及 §73.6.23/§73.6.24 的两处渲染修复）⇒ 本臂**不需要** finisher 兜底补跑。
- **判据**：到 ep6 由看门狗自动中断并冻结 S1 + A2-R 专属读数（`anchor_slot` 覆盖度、
  `frac_cells_residual_can_flip` 等）；**不自动**接 ep16，仍按 A2 预登记 §6/§7 人工分叉。
- **记录口径**：ep12–16 窗口**尚未形成**；本臂的完整逐 epoch 曲线、逐类拆解与机制诊断
  在其 ep6 归档时另起一节补写。**本节只记运行事实，不含任何性能宣称。**
- **运行健康度快照（2026-10-11 16:29 UTC，ep1 iter 1350/1902）**：
  1.530 s/iter、**20038 MiB/卡**、无 NaN/OOM；首个 val 尚未出现（ep1 未完）。
  **显存明显高于 A2-S 的 17136 MiB**：A2-R 把残差广播成 **48 通道**后再交给 head，
  激活占用随之抬高。⇒ ep6 只读电池必须在 GPU **真正释放后**才能跑
  （看门狗已按此设计：先等 `memory.used` 降到阈值以下再起探针）；
  也再次说明**训练期间不要抢卡补跑 A2-S 的缺失探针**（§74.3）。

### 74.11 只读补测：z 契约对账 + split/BN 归因（2026-10-12 补）

§74.3 与 §74.9/§74.10 记录的三个缺口（`semA_bnmode_ep6.json`、`semA_zcontract_ep6.json`、
split 归属）在本小节**补齐**：GPU5/6/7 空闲期间用**只读**探针重跑，**未改权重、
未启动训练、未重判任何门**。§74.0/§74.3 的 NOT ALL PASS 判定与全部门读数**不变**。

**（1）z 契约对账**（`semA_zcontract_ep6.json`，10-clip 契约，`stride 200`）：

| z 契约 | `thr50` | `hit@1` | `hit1_excess_over_chance` |
|---|---:|---:|---:|
| 部署（`mu_q`，即 M1 实际读数） | 0.1833 | 0.0333 | 0.0333 |
| 训练（`mu_q + sigma*eps`） | 0.1833 | 0.0333 | 0.0333 |

* 干预检查：z 相对变化均值 **38.37%**（契约切换**确实改变**了 z）、语义热图相对变化
  均值 **1.56%**（经语义头后幅度被吸收）；噪声占比 `sigma/mu` 均值 **0.4022**。
* ⇒ 契约切换移动了 z，但**两项门控读数完全不变**（Δ`thr50` = +0.0000）：
  train/val 差距**不能**由 z 采样解释，与 A 的 §72.10 同结论。

**（2）split / BN 归因**（`semA_bnmode_ep6.json` + 补跑的 `semA_bnmode_ep6_trainsplit.json`，
同 20-clip 契约、同 `--batch-size 2`）：

| split | `thr50` | `hit@1` | 只切 z 路径 4 个 BN 的 Δ`thr50` | Δ`hit@1` |
|---|---:|---:|---:|---:|
| train（拟合过） | **0.4438** | **0.4167** | +0.0538 | −0.0417 |
| val（留出） | **0.1458** | **0.1042** | +0.0587 | +0.0000 |

* ⇒ 该差距**主要来自数据 split**（`thr50` +0.2980、`hit@1` +0.3125），归一化模式是
  **次要**项；train 侧 `stat_sem_*` 是**拟合内**读数，不能当作同一模型在另一口径下的同量测量。
* **BN 探针复现性自检**：本文件 `eval_running_stats` 臂 = **0.1458 / 0.1042**，与
  `semreadout_ep6_gate.json` 的 M1 实读（0.1458333 / 0.1041667）**逐位一致** ⇒
  该探针确实复现了本臂的门读数。
* > **读法警告**：该产物里的 `eval_reproduces_frozen_gate.reproduces` 会被读成「探针跑错」，
  > 其实它比的是 **A 的**冻结常数（0.1167 / 0.0833）——对 A2-S/A2-R 它恒为 `False`，
  > 与「本臂是否复现自身门读数」**无关**。本臂的正确自检是上面那两条逐位一致。
  > （该字段不进任何归档渲染，本小节按读数说明处理，不改探针。）

**（3）产物清单**：`semA_bnmode_ep6.json`、`semA_bnmode_ep6_trainsplit.json`、
`semA_zcontract_ep6.json` 均在 run dir 内；§74.3 的「BN probe 未写出 / z 契约未测」两项
到此作废，但**M1/M4 donor/M1-loc 的 FAIL 不变**，仍**不续 ep16**。

## 75. 候选 A2-R ep6 机制门（S1）：未全门通过（2026-10-12）

本节执行 `docs/rssm_candidate_A2_prereg_20261011.md` §5 预登记的 S1 ep6
机制门。训练从零起（3 卡 GPU5/6/7、seed0、deterministic、24e 余弦日程），
ep6 val 与 `epoch_6.pth` 落盘后由看门狗 `me_rssm/sanity/watch_semA2_ep6.sh` 中断训练、
等待 GPU 释放，再自动跑 M1–M4 门 + 三窗口反事实 + source 探针（A2-R 臂另跑 M1' 的双 split 排序探针与 M4 地板探针；M1-loc 为 A2-S 专属，本臂不评估）。

> **本节不宣称任何 S2 性能结论。** S2 是 ep12–16 五轮窗口，且其数值需
> 用户批准后才允许 ep6→ep16 续训（§5/§7）。本节只报 ep6 机制门与早期
> 曲线，二者不可互相替代。

### 75.0 状态与判定摘要

**三口径摘要**（AGENTS 规则 5：三个口径必须分别写清，不得用含糊的
“BEST” 代替其中任一项）：

| 口径 | 数值 |
|---|---|
| 区间均值（标准 ep12-16 窗口不完整；实际跨度 ep1-6，n=6） | **33.9502** |
| 全轮峰值（所有 val epoch 的最高单点，与是否落盘无关） | **38.4943 @ ep5** |
| best saved（仅表示 checkpoint 可用性，不参与性能排名） | **36.3491 @ ep6**（落盘 [2, 4, 6]） |

> 上述窗口均值是 **ep6 早期上下文**，不是 ep12–16 的 S2 判据；
> 单 seed、窗口未完整，不得据此解读效应。

**S1 门控表**（阈值来自 §5 预登记，判定来自
`me_rssm/sanity/semA_ep6_verdict.py`，本节不重算门槛）：

| 门 | 预登记门槛 | ep6 实测 | 判定 |
|---|---|---|---|
| M1 语义头质量 | val `thr50 ≥ 0.30` 且 `hit@1 ≥ 0.60` | `thr50` **0.0712**、`hit@1` **0.1292** | **FAIL** |
| M2 z 排他性 | `h` 与 `feat` 两臂**各自** ≤ −30% | `h` **-100.0000%**（head thr50 0.0000 vs z 0.0712）、`feat` **-100.0000%**（两臂均达线） | **PASS** |
| M3 读出非装饰 | 强制 residual=0 使正样本 class-CE 变差 ≥16/20（窗口冻结比例 8/10） | **15/20** | **FAIL** |
| M4 类别正确性 | `own_hit` ≥ **pooled prior floor + 20pt**、`donor_transfer` ≥ chance+20pt | `own_hit` **0.7684**（floor 0.5000 → bar 0.7000）、`donor_transfer` **0.2854**（bar 0.45） | **FAIL** |
| M1' 排序泛化 | val（n=20）`auc` ≥ 0.8 且 `auc − gt_shuffle` ≥ 0.25 | `auc` **0.9305**、`auc − gt_shuffle` **0.4779**（shuffle 0.4526） | **PASS** |
| M1-loc 峰预算 | （A2-R 不适用：§5 预登记为 A2-S 专属） | — | n/a |

**候选 A2-R 的 train/val 对照（同一探针、同一 20-clip 契约；本臂假设正是『峰预算能否离开训练场景』）：**

| split | n | `auc` | `gt_shuffle_auc` | `max_gt_prob` | `recall@top-k` 中位 | `recall@top-k` 均值 | `max_gt_prob ≥ 0.5` 的 clip |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 20 | 0.9995 | 0.5091 | 0.5719 | 0.2917 | 0.3453 | 13/20 |
| val | 20 | 0.9305 | 0.4526 | 0.2490 | 0.0000 | 0.1033 | 3/20 |

| split | `mean_gt_prob` | `mean_bg_prob` | `max_gt_prob` | argmax 距离（格，均值）|
|---|---:|---:|---:|---:|
| train | 0.4478 | 0.0019 | 0.5719 | 2.51 |
| val | 0.1551 | 0.0012 | 0.2490 | 43.73 |

> 上表 `argmax 距离` 是该 clip 各占用通道「唯一 argmax 格到最近 GT 格」的
> 曼哈顿距离，逐通道后按 clip 等权取均值（§5 的补充读数口径）；
> 它与 `recall@top-k` 是两个现象：前者量**峰漂多远**，后者量
> **峰预算是否被背景占满**。二者都只作机制定位，不是门槛。

> train−val 差：`auc` **+0.0691**、`recall@top-k` 中位 **+0.2917**。
> A 的同口径差为 `auc` 0.9994−0.9235 = **+0.0759**、`recall@top-k` 中位 0.3333−0.0000 = **+0.3333**（A 的 val 端 n=20）。
> 本行只作机制定位，**不是门槛**。


**不进门补充读数（M1 口径）**：`hit1_chance` 均值 **0.000093**、`hit1_excess_over_chance` 均值 **0.1291**。
按 §71.9.3(2)，`hit@1 ≥ 0.60` 是从 route-C 的 16×16 网格外推而来，
A 的 full-res 216×248 网格随机基线只有 ~7.5e-5（209× 格数），
故这两个读数必须与门槛并排记录。

**§71.9.3(2) 例外条款：不适用。** 该条款要求 M1 *仅*在 full-res 原始 `hit@1` 上未过，而 `thr50` 与
`hit1_excess_over_chance` 都健康（即「头确实找得到目标中心，只是单格 argmax 跨不了网格」）。

**该判定不依赖口径选择**，两种读法都不满足例外：

| 读法 | `thr50` | `hit@1` | vs 门槛 |
|---|---:|---:|---|
| 预登记的 val 口径（本门实际读数） | 0.0712 | 0.1292 | `thr50` 未过 0.30、`hit@1` 未过 0.60 |
| A2-R 自身 train 口径（`stat_sem_*` ep6 的 **epoch 均值**） | 0.4923 | 0.3608 | `thr50` **过** 0.30、`hit@1` 未过 0.60 |

例外条款要求「excess 健康」，val 口径下的对应量是 0.1291（bar 0.5840）；该口径下 `thr50` 为 0.0712（bar 0.30）。
**两项中至少一项未达线**，故 M1 是**真实未过**，不是跨网格口径造成的伪失败，不得以「口径未校准」豁免。

旁证（不是判据）：train 口径只记 `stat_sem_hit1`、未单独记 excess，而 `excess <= hit@1`，故 hit@1 = 0.3608 是该口径下 excess 的**上界**。
该上界 < bar 0.5840 ⇒ train 口径同样不满足例外，**该结论不建立在口径选择上**。

> **口径出处须与上表并读**（AGENTS 记录纪律）：`0.30 / 0.60` 这组门槛的先例 §59 route-C 报的是**train 侧 epoch 均值**（§60.5/§61.5，38 条 train 记录），而本门按 §5 预登记在 **val** 上测量，两者之间隔了一层清样；上表因此把两种读法并列，而不是拿 val 数字直接比 train 门槛。
> 上表两种读法是**稳健性并列**：val 是预登记的判据口径，train 只是旁证。若两者结论不一致，以 val 为准并显式记录分歧。

**S1 聚合判定：NOT ALL PASS（M1, M3, M4）。** 按 §7
分叉，按负结果归档并把读数写进 Analysis，**不进 ep16**。
**M2 通过：语义头对输入**敏感**，但它的证据强度到此为止。**
h 臂与 feat 臂的 `thr50` 均为 0.0000/0.0000，而 z 臂为 0.0712。

> **不得据此宣称「语义只存在于 z」**：M2 是拿**只被 z 监督过**的 head 去喂 h/feat，属于分布外输入——即使 h 或观测里含有同样的对象信息，这个 head 也读不出来。因此 M2 只证明「该 head 依赖它训练时的输入」，**不能证明** z 是唯一信息源。
> 预登记把真正的 z 排他性检验放在 **C3 臂**（`sem_head_input='feat'` 从零训练、同日程对照），本次**未执行**——故「概率状态」这一主张在本次测量中**既未被否证、也未被证实**。

### 75.1 配置与唯一变量

| 项 | 值 |
|---|---|
| 配置 | `me_rssm/configs/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_semA2R_24e.py` |
| 对照 | `..._no2d_igdr_semreadout.py`（A 底座；不重跑 A） |
| 唯一变量 | `cls_readout_mode`（分类残差到 anchor 通道的写入方式） |
| 变量取值 | `anchor_slot`（同 predictor、同参数量，不改监督） |
| 未改动 | 重建目标、posterior/prior、KL/free-nats、`output_proj`、backbone、head 几何、优化器、日程、数据管线 |
| 权重指纹 | `sha256=a6c6588c6f034128...`、size=688860268、mtime=1791755168 |
| 采样位点 | clip 索引 `[3, 103, 203, 303, 403, 503, 603, 703, 803, 903, 1003, 1103, 1203, 1303, 1403, 1503, 1603, 1703, 1803, 1903]`、`shuffle_offset=1` |

### 75.2 运行事实

| 项 | 值 |
|---|---|
| ep6 段起止 UTC | 2026-10-11 15:53:23 → 2026-10-11 21:56:27（`20261011_155323.log`，263 条带时间戳行） |
| GPU / seed | 5/6/7、seed 0、deterministic |
| 吞吐 / 显存 | 中位 **1.557 s/iter**、16.7 GiB/卡（`*.log.json` train 段 228 条） |
| val 记录 | ep1-ep6 各一条，共 6 条 |
| 落盘 | [2, 4, 6]（`checkpoint_config.interval=2`） |
| 停止方式 | 看门狗在新鲜 `epoch_6.pth` + ep6 val 后中断训练（signal ?，退出码 None；中断是看门狗按 §5 的预期动作，非崩溃）· gate hit 2026-10-11 21:56:49 → GPU 释放 2026-10-11 21:57:04 → 探针结束 2026-10-11 22:01:15 |

### 75.3 完整逐 epoch 曲线（ep1-ep6）

| ep | Overall3Dmod | vs base | Car | Truck | Ped_l | Cyc_l |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 28.06 | -11.22 | 39.46 | 8.45 | 18.83 | 45.52 |
| 2 | 30.83 | -8.45 | 39.79 | 20.07 | 25.80 | 37.65 |
| 3 | 37.89 | -1.39 | 49.77 | 22.25 | 29.10 | 50.46 |
| 4 | 32.07 | -7.21 | 38.80 | 24.35 | 29.62 | 35.52 |
| 5 | 38.49 | -0.79 | 46.04 | 22.90 | 28.84 | 56.20 |
| 6 | 36.35 | -2.93 | 47.76 | 24.60 | 29.43 | 43.61 |

> `vs base` 对照 §5 的 seed0 基线 Overall **39.2809**（ep12–16 窗口均值）；单点差值不含 seed 噪声。

### 75.4 窗口逐项均值（实际跨度 ep1-6，n=6）

| 类别 | 本 run 窗口均值 | seed0 基线（ep12-16） | 差 |
|---|---:|---:|---:|
| Overall_3D_moderate | **33.95** | 39.2809 | -5.33 |
| Car_strict | 43.60 | 47.8399 | -4.24 |
| Truck_strict | 20.44 | 29.7246 | -9.29 |
| Ped_loose | 26.93 | 30.9277 | -3.99 |
| Cyc_loose | 44.83 | 48.6313 | -3.80 |

> 该窗口与基线窗口**不同长度**（基线段是 ep12-16，本 run 只有 ep1-6），
> 只作早期上下文，不作 S2 判据。

### 75.5 A2-R 侧训练期同口径读数（train split，逐 epoch 末值）

| ep | `stat_sem_hit1` | `stat_sem_thr50` | `loss_rssm_sem` | `stat_cls_res_rms` | `stat_mu_diff_sq` |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.2097 | 0.1937 | 1.3091 | 1.2229 | 3.0972 |
| 2 | 0.2540 | 0.2655 | 1.0493 | 1.1852 | 0.1659 |
| 3 | 0.3008 | 0.3589 | 0.9041 | 1.0954 | 0.0619 |
| 4 | 0.3159 | 0.3950 | 0.8153 | 1.2778 | 0.0969 |
| 5 | 0.3663 | 0.4919 | 0.6890 | 1.3939 | 0.0800 |
| 6 | 0.3463 | 0.5354 | 0.6444 | 1.4057 | 0.0427 |

shuffle/zero 与门控读数无关；此处 `stat_sem_*` 为**训练统计**（z 采样值、train split），与探针的 val 口径不同，不可混比。
末 epoch `stat_cls_res_rms` = 1.4057（>0 ⇒ 读出活着）。

### 75.6 S1 门明细（逐 clip）

> **类别覆盖（val（冻结 20 clip），n=20）**：每 clip 占用通道数分布 1 类 ×4, 2 类 ×11, 3 类 ×4, 4 类 ×1。**不按类别剔除 clip**（§5：覆盖是报告字段，不是选样条件）；某类只覆盖少数 clip 时，该类专属结论在 ep6 只能是**弱证据**。
> **类别覆盖（train 同协议，n=20）**：每 clip 占用通道数分布 1 类 ×7, 2 类 ×9, 3 类 ×4。**不按类别剔除 clip**（§5：覆盖是报告字段，不是选样条件）；某类只覆盖少数 clip 时，该类专属结论在 ep6 只能是**弱证据**。

> **样本权重不均提示**：各 clip 的 GT 目标数并不相等——clip `1603` 只有 **1** 个目标，clip `1303` 有 **12** 个（§71.9.14 预检：3/4/3/11/4/2/6/6/1/2）。
> 逐 clip 比率在 `1603` 上只能取少数离散值、方差最大，
> 故下面 M1/M3/M4 的**分组结论不得由单个 clip 单独支撑**。

**M1 逐 clip**：

| index | thr50 | hit@1 | hit1_chance | excess_over_chance | 正格数 |
|---:|---:|---:|---:|---:|---:|
| 3 | 0.0000 | 0.0000 | 0.000056 | -0.000056 | 3 |
| 103 | 0.0000 | 0.0000 | 0.000075 | -0.000075 | 4 |
| 203 | 0.0000 | 0.0000 | 0.000056 | -0.000056 | 3 |
| 303 | 0.0000 | 0.0000 | 0.000131 | -0.000131 | 7 |
| 403 | 0.5000 | 1.0000 | 0.000037 | +0.999963 | 2 |
| 503 | 0.0000 | 0.0000 | 0.000205 | -0.000205 | 11 |
| 603 | 0.0909 | 0.0000 | 0.000205 | -0.000205 | 11 |
| 703 | 0.0000 | 0.0000 | 0.000075 | -0.000075 | 4 |
| 803 | 0.0000 | 0.0000 | 0.000056 | -0.000056 | 3 |
| 903 | 0.0000 | 0.0000 | 0.000112 | -0.000112 | 6 |
| 1003 | 0.0000 | 0.0000 | 0.000037 | -0.000037 | 2 |
| 1103 | 0.0000 | 0.0000 | 0.000131 | -0.000131 | 7 |
| 1203 | 0.0000 | 0.0000 | 0.000112 | -0.000112 | 6 |
| 1303 | 0.0833 | 0.2500 | 0.000224 | +0.249776 | 12 |
| 1403 | 0.0000 | 0.3333 | 0.000112 | +0.333221 | 6 |
| 1503 | 0.5000 | 1.0000 | 0.000037 | +0.999963 | 2 |
| 1603 | 0.0000 | 0.0000 | 0.000019 | -0.000019 | 1 |
| 1703 | 0.0000 | 0.0000 | 0.000075 | -0.000075 | 4 |
| 1803 | 0.0000 | 0.0000 | 0.000037 | -0.000037 | 2 |
| 1903 | 0.2500 | 0.0000 | 0.000075 | -0.000075 | 4 |

**M2 两控制臂逐 clip**（`thr50`；z 为参照）：

| index | z | h | feat |
|---:|---:|---:|---:|
| 3 | 0.0000 | 0.0000 | 0.0000 |
| 103 | 0.0000 | 0.0000 | 0.0000 |
| 203 | 0.0000 | 0.0000 | 0.0000 |
| 303 | 0.0000 | 0.0000 | 0.0000 |
| 403 | 0.5000 | 0.0000 | 0.0000 |
| 503 | 0.0000 | 0.0000 | 0.0000 |
| 603 | 0.0909 | 0.0000 | 0.0000 |
| 703 | 0.0000 | 0.0000 | 0.0000 |
| 803 | 0.0000 | 0.0000 | 0.0000 |
| 903 | 0.0000 | 0.0000 | 0.0000 |
| 1003 | 0.0000 | 0.0000 | 0.0000 |
| 1103 | 0.0000 | 0.0000 | 0.0000 |
| 1203 | 0.0000 | 0.0000 | 0.0000 |
| 1303 | 0.0833 | 0.0000 | 0.0000 |
| 1403 | 0.0000 | 0.0000 | 0.0000 |
| 1503 | 0.5000 | 0.0000 | 0.0000 |
| 1603 | 0.0000 | 0.0000 | 0.0000 |
| 1703 | 0.0000 | 0.0000 | 0.0000 |
| 1803 | 0.0000 | 0.0000 | 0.0000 |
| 1903 | 0.2500 | 0.0000 | 0.0000 |

**M3 逐 clip**（`delta_off_minus_on > 0` 表示关掉 residual 后
正样本 class-CE 变差，即读出有贡献）：

| index | CE(on) | CE(off) | delta(off-on) | 正格数 |
|---:|---:|---:|---:|---:|
| 3 | 1.5946 | 1.2951 | -0.2995 | 3 |
| 103 | 1.0797 | 1.1478 | +0.0681 | 4 |
| 203 | 0.8819 | 1.1959 | +0.3140 | 3 |
| 303 | 0.1818 | 0.3105 | +0.1287 | 7 |
| 403 | 0.0163 | 0.0720 | +0.0557 | 2 |
| 503 | 0.5522 | 0.5658 | +0.0135 | 11 |
| 603 | 1.5209 | 1.4040 | -0.1168 | 11 |
| 703 | 0.3791 | 0.4980 | +0.1189 | 4 |
| 803 | 0.3347 | 0.6655 | +0.3308 | 3 |
| 903 | 1.2225 | 1.0061 | -0.2164 | 6 |
| 1003 | 0.8197 | 0.7666 | -0.0531 | 2 |
| 1103 | 0.5238 | 0.6122 | +0.0883 | 7 |
| 1203 | 0.4970 | 0.8387 | +0.3417 | 6 |
| 1303 | 0.9953 | 0.9480 | -0.0474 | 12 |
| 1403 | 0.3467 | 0.6007 | +0.2540 | 6 |
| 1503 | 0.2758 | 0.3809 | +0.1051 | 2 |
| 1603 | 0.0290 | 0.1713 | +0.1423 | 1 |
| 1703 | 0.1761 | 0.4454 | +0.2693 | 4 |
| 1803 | 0.4538 | 0.9373 | +0.4836 | 2 |
| 1903 | 0.1670 | 0.5727 | +0.4058 | 4 |

**M4 逐 clip**（`own_hit` = z 残差 argmax 命中该格 GT 类；
`donor_transfer` = 换 donor z 后残差 argmax 命中 donor 主类；
`head_swap_to_donor` = 只重跑 head 且喂 donor 残差时的 head 类 argmax）：

| index | own_hit | donor_transfer | head_swap | donor_index | changed_frac |
|---:|---:|---:|---:|---:|---:|
| 3 | 0.6667 | 0.3333 | 0.6667 | 203 | 0.6667 |
| 103 | 0.5000 | 0.2500 | 0.2500 | 203 | 0.2500 |
| 203 | 1.0000 | 0.3333 | 0.0000 | 803 | 0.6667 |
| 303 | 1.0000 | 1.0000 | 0.0000 | 803 | 1.0000 |
| 403 | 1.0000 | 1.0000 | 0.0000 | 803 | 1.0000 |
| 503 | 0.7273 | 0.9091 | 0.0909 | 803 | 1.0000 |
| 603 | 0.5455 | 0.4545 | 0.0000 | 803 | 0.7273 |
| 703 | 0.7500 | 0.7500 | 0.2500 | 803 | 0.7500 |
| 803 | 1.0000 | 0.0000 | 0.3333 | 903 | 0.3333 |
| 903 | 0.6667 | 0.0000 | 0.0000 | 1103 | 0.6667 |
| 1003 | 0.5000 | 0.0000 | 0.0000 | 1103 | 0.0000 |
| 1103 | 0.4286 | 0.4286 | 0.1429 | 1403 | 0.4286 |
| 1203 | 0.8333 | 0.1667 | 0.1667 | 1403 | 0.6667 |
| 1303 | 0.4167 | 0.0833 | 0.3333 | 1403 | 0.8333 |
| 1403 | 0.8333 | 0.0000 | 0.1667 | 1503 | 1.0000 |
| 1503 | 0.5000 | 0.0000 | 0.5000 | 1703 | 0.5000 |
| 1603 | 1.0000 | 0.0000 | 0.0000 | 1703 | 1.0000 |
| 1703 | 1.0000 | 0.0000 | 0.0000 | 1903 | 1.0000 |
| 1803 | 1.0000 | 0.0000 | 0.0000 | 1903 | 0.0000 |
| 1903 | 1.0000 | 0.0000 | 0.2500 | 3 | 0.7500 |

**M4 地板 / 读出存活读数**（`semA_m4floor_ep6.json`，M4 的 `own_hit` 线即由这里的 pooled 地板推出）：

| 读数 | 值 | 说明 |
|---|---:|---|
| `pooled_prior_floor` | **0.5000** | 窗口内「恒投多数类」的准确率，M4 `own_hit` 线的基准 |
| `recomputed_mean_own_hit` | **0.7684** | 由残差现场重算，与地板同源 |
| `margin_over_pooled_prior` | **+0.2684** | `own_hit −` 地板（**机制读数**）|
| `frac_cells_residual_can_flip` | **0.2930** | A2-R 的 S0-5 预期读数（§5；不设通过线）|
| `residual_gap_fg_mean` / `conv_cls_gap_fg_mean` | **0.8635** / **2.4839** | 残差与 `conv_cls` 的前景间隔（稀释程度）|

head 命中随残差**尺度**的变化（`residual=0` 与强制满尺度在两端）：`×0.0 → 0.7200 / ×1.0 → 0.7600 / ×2.0 → 0.7700 / ×4.0 → 0.7600 / ×8.0 → 0.7500 / ×1000000.0 → 0.7100`。


### 75.7 家族标准反事实读数（`shuffle_z` / `zero_z` / `prior_only`）

三窗口与既往各代（MAI / MCI / DC-RSSM）使用同一探针语义：
`shuffle_z` 用同 batch 另一样本的 `mu_q` 替换，`zero_z` 置 0，
`prior_only` 只走先验。**本臂即跑在标准家族底座上**，故用
`probe_z_utilization_standard.py`（后验头重写 + 转发部署态 class
residual），与 `tools/diagnose_z_utilization.py` 的打分口径一致。

| 窗口 | shuffle_z head | zero_z head | prior_only head | shuffle/zero | shuffle_z feat | 标准 RSSM ep6 同窗 head（对照） |
|---|---:|---:|---:|---:|---:|---:|
| 近距（[3, 4, 5, 6]） | **5.5466%** | 45.6742% | 5.6832% | 0.121 | 20.7943% | 49.73% |
| 中距（[100, 101, 102, 103]） | **2.6185%** | 23.5933% | 8.7471% | 0.111 | 7.5114% | 3.30% |
| 远距（[1800, 1801, 1802, 1803]） | **1.4041%** | 23.2734% | 10.2454% | 0.060 | 4.6481% | 3.36% |

本 run 三窗 `shuffle_z` head：近距 5.55%; 中距 2.62%; 远距 1.40%。
对照为**标准 no2d_igdr RSSM seed1 ep6**同窗口（`docs/training_runs_full.md` §62.2），因同家族、同 epoch，
是唯一口径一致的机制对照；MAI/MCI/DC 各代的同窗读数见 §62/§63/§68。


### 75.8 posterior source 归因（`stride=700`）

| 量 | RMS | batch-common RMS | 样本特异 RMS | 样本特异能量占比 |
|---|---:|---:|---:|---:|
| `mu_q` | 0.6869 | 0.6350 | 0.2621 | 14.56% |
| `h_t` | 0.2389 | 0.2266 | 0.0754 | 9.97% |
| `e_pooled` | 0.7982 | 0.5089 | 0.6149 | 59.35% |

| 读出/依赖读数 | 值 |
|---|---:|
| `mu_q_removal_of_h_rel_l2` | 0.8010 |
| `mu_q_removal_of_e_rel_l2` | 0.5005 |
| **h/e 依赖比** | **1.600** |
| `posterior_obs_mode` | `l2` |
| `z_proj_init` | `None` |

> h/e 依赖比是各代通用的机制对照量（标准主线 ep12 为 2.338、MAI ep12 为 0.842、MCI ep12 为 0.605）；越低表示 posterior
> 越依赖观测而非 GRU 状态。

fusion 类：`SemanticReadoutRSSMFusion`、latent [216, 248]、readout `film`、clip 索引 [3, 703, 1403]。

### 75.9 判定与下一步（按 §7 预登记分叉，人工步骤）

1. S1 未全过 ⇒ 按负结果归档，读数写进 Analysis，不投 ep16。
   实测原因：M1 `thr50` 0.0712/`hit@1` 0.1292（例外条款不适用，真实未过）；M3 仅 15/20；M4 `donor_transfer` 0.2854（bar 0.45）。
   归档的是**本次 候选 A2-R 的单一变量对照**（A2-S 为监督侧排序压力、A2-R 为通道覆盖），不是「z 无信息」，也不是「所有语义监督/残差读出/通道覆盖方向」；转下一候选按 §7 的分叉表（M1-loc 不过 ⇒ 转「z 的对象信息是否只够支撑对象级读出」；M3 不过 ⇒ 转「分类读出是否应走独立分支」），不靠调权重/增益/KL 抢救。
2. 任意 Ped/Cyc 触到非劣线 ⇒ 停止，记为家族副作用第 6 次复现。
   > **口径更正（2026-10-11）**：本行写在 ep6 归档节里，容易被读成 ep6 的止损线。
   > 该非劣线属 **S2（ep12–16）**；A2-R 停在 S1，S2 从未形成，故本代**从未触发**这条线
   > （见预登记 §7 的范围更正与 A2 预登记 §5「早期门与成熟门严格分离」）。
3. **不做**权重/KL 扫描，**不做** ep24 抢救，不靠放大系数救活机制。

### 75.10 z 契约对账（训练采样 z vs 部署均值 z，只读）

归档审计发现：M1/M3/M4 是在**部署契约**下读的，而 `0.30/0.60` 门槛来自 **train 契约**下的诊断。两者差一个状态采样：

```
R4Det.extract_feat:  rssm_deterministic = not self.training
rssm_fusion.forward:  z_t = mu_q if deterministic else self.sample(mu_q, logstd_q)
```

即训练时语义头吃 `mu_q + sigma*eps`、eval 探针吃 `mu_q`。故用同一 checkpoint、同一 val clip、同一评分函数，只切换 z 契约重测一次：

| z 契约 | `thr50` | `hit@1` | `hit1_excess_over_chance` |
|---|---:|---:|---:|
| 部署（`mu_q`，即 M1 实际读数） | 0.0591 | 0.1333 | 0.1333 |
| 训练（`mu_q + sigma*eps`） | 0.0591 | 0.1333 | 0.1333 |

**干预检查**（否则「两臂相同」既可能是契约真无影响，也可能是切换没生效——结论相反、数字相同）：

* `z` 相对变化均值 **33.451%** —— 契约切换**确实改变了 z**；
* 语义热图相对变化均值 **1.316%** —— 经 `sem_head` 后幅度被大幅吸收。

噪声占比 `sigma/mu`：均值 **0.2768**（min 0.2335 / max 0.3567），即噪声项确实与被解释量同量级。

**结论**：契约切换确实移动了 z，但**两项门控读数完全不变**（Δ`thr50` = +0.0000）——
故 train 侧读数与 val 侧读数之间的差距**不能**由 z 采样解释：两侧在各自口径下分别是 `stat_sem_thr50` 0.4923（train split、epoch 均值）与 0.0591（M1 实读的 val 口径）——两者口径不同，只作数量级参照，不是同一个量的两次测量。
这**不改变** S1 判定（§75.0 已按两种口径分别核过），但把「契约错配」从嫌疑名单上划掉。

**来源归属（只读补测，n=20 / 20）**：把 z 契约与 eval 模式都钉死后**只换数据 split**——

| split | `thr50` | `hit@1` | 只切 z 路径 4 个 BN 的 Δ`thr50` | Δ`hit@1` |
|---|---:|---:|---:|---:|
| train（拟合过） | **0.4355** | **0.3750** | -0.0933 | -0.0167 |
| val（留出） | **0.0712** | **0.1292** | -0.0212 | +0.0000 |

⇒ 该差距**主要来自数据 split**（thr50 +0.3643、hit@1 +0.2458），归一化模式是**次要**项。即：train 侧 `stat_sem_*` 是**拟合内**读数，不能当作同一模型在另一口径下的同量测量。

> 产物：`semA_zcontract_ep6.json`（只读探针 `me_rssm/sanity/probe_semA_zcontract.py`，未改权重、未启动训练）。

### 75.11 A2-R 专属读数：S0-5 通道覆盖、M4 地板哨兵与残差尺度

**S0-5（§5 预登记：记录实测值、不设通过线）**：`frac_cells_residual_can_flip` = **0.2930**。

| 臂 | 分类残差写入方式（本代唯一变量） | `frac_cells_residual_can_flip` |
|---|---|---:|
| A | `per_class_slot`（12/48 通道） | 0.2530 |
| **A2-R** | **`anchor_slot`（写满 48/48）** | **0.2930** |
| A2-S（参照，**不是**同变量） | `per_class_slot` + 监督侧排序项 | 0.3555 |

* §5 该行的字面要求是「应**高于** A 的 0.2530」⇒ **字面成立**：+0.0399 绝对、+15.8% 相对。
* 但把残差从 12 个通道广播到**全部 48** 个通道，并没有把可翻格比例推到 A2-S 那种量级（0.3555）；
  0.2930 与 0.2530 只差 **4.0 个百分点**，且**低于**同代 A2-S。
* A2-S 的 0.3555 **不是同一变量的对照**（它换的是监督侧排序项），故**不能**读成「写满 48 通道反而更差」。
  能读的是：**在同一语义监督下**，把写入通道从 12 扩到 48 只带来 +4.0 pt。
* ⇒ §7 的 M3 分叉（「若 `frac_cells_residual_can_flip` 未按预期上升到**明显**高于 0.2530，
  则**通道覆盖也不是瓶颈**」）在本包证据下读作**成立**，分叉指向「分类读出是否应走独立分支」
  的单变量验证。**这是人工裁决点，本节不自动立项、不自动投训。**

**M4 地板哨兵（§73.6.29 记录的零点隐患）**：本包 20 个 clip 的
`residual_absmax_fg_mean` 最小 **0.8867**（>0），**无** clip 为 0，逐 clip `cells` 全 >0
⇒ M4 的逐格读数**全部有效**，不存在 `detection_cls_logit_residual` 对 invalid sample
整条清零所致的假 0（该误读风险本身在 ep6 电池**之前**已修，本包读到的是修复后的代码）。

**残差尺度（尺度陈述，不是信息陈述）**：`residual_rms_mean` **1.9732** vs
`conv_cls_rms_mean` **25.3824**（比值 **0.0777**）；`residual_gap_fg_mean` **0.8635** vs
`conv_cls_gap_fg_mean` **2.4839**。head 命中随残差尺度：
`×0 → 0.7200 / ×1 → 0.7600 / ×2 → 0.7700 / ×4 → 0.7600 / ×8 → 0.7500 / ×1e6 → 0.7100`
⇒ 放大残差**不**改善命中（与 A 的 §71.9.18 同向），故本代**不做**增益扫描。

### 75.12 同代并排：A / A2-S / A2-R（本次新增）

三臂同底座（标准 full-resolution RSSM、同一语义头监督、从零起训、seed0、GPU5/6/7、
24e 余弦日程、ep6 看门狗冻结），差异只在唯一变量。

| 项 | A | A2-S | A2-R |
|---|---|---|---|
| 唯一变量 | 语义读出（`per_class_slot`） | + 监督侧逐通道排序项 | `cls_readout_mode='anchor_slot'`（写满 48） |
| M1 `thr50` / `hit@1` | 0.1167 / 0.0833 **FAIL** | 0.1458 / 0.1042 **FAIL** | 0.0712 / 0.1292 **FAIL** |
| M2 输入替换 | PASS（形式） | PASS（形式） | PASS（形式） |
| M3 分类 CE 变差 | 7/10 **FAIL**（10-clip 契约） | 17/20 **PASS** | 15/20 **FAIL** |
| M4 `own_hit` / `donor_transfer` | 0.7379 / 0.3818 **FAIL** | 0.7700 / 0.4120 **FAIL** | 0.7684 / 0.2854 **FAIL** |
| M1′ val `auc` | 0.9235 PASS | 0.9509 PASS | 0.9305 PASS |
| S0-5 `frac_cells_residual_can_flip` | 0.2530 | 0.3555 | 0.2930 |
| ep1–6 等权均值（n=6） | 34.0459 | 33.6768 | 33.9502 |
| 全轮峰值（值 @ epoch） | 38.2203 @ ep3 | 37.3276 @ ep5 | 38.4943 @ ep5 |
| best saved（仅可用性） | 35.7608 @ ep6 | 34.9702 @ ep6 | 36.3491 @ ep6 |
| `h/e` 依赖比（stride 700） | 1.517 | 2.094 | 1.600 |
| `shuffle_z` head 近/中/远 | 2.65% / 7.43% / 4.21% | 4.33% / 4.08% / 2.57% | 5.55% / 2.62% / 1.40% |

> **口径警告**：A 的门跑在 **10-clip** 契约（`stride 200`，索引 `3,203,…,1803`），
> A2 两臂跑在**冻结 20-clip** 契约（`stride 100`，索引 `3,103,…,1903`）；
> M3 的分母因此不同（7/10 vs 15/20），**不能**把两者按同一分数排序。
> 上表其余读数（M1/M4/M1′/flip/h-e/shuffle）各自同口径可比，区间均值都是
> 「ep1–6、n=6」的早期上下文，三臂同窗，可横向看，但都不是 S2 判据。

**并排才看得见的三条共同点**：M1 三臂全 FAIL、M4 `donor_transfer` 三臂全 FAIL、
M1′ 三臂全 PASS。即：**排序可分性（val `auc` 0.92–0.95）在三臂都成立，而
「峰落到正确格、且该能力能跨场景」在三臂都不成立**。A2-R 的补充定位读数同向：
同一探针、同一契约下 train 侧 argmax 平均漂 **2.51** 格、val 侧漂 **43.73** 格（约 17×），
而两侧 `mean_bg_prob` 都极低（0.0019 / 0.0012）⇒ 失效**不是**背景概率整体抬高，
而是**稀疏的峰落在远处**。这与 §71.9.20 的双峰漂移、A2-S §74.9 的
`train 2.79 格 / val 31.02 格` 构成同一条机制线索。

### 75.13 本次只读补测、生产链修复与人工裁决点（2026-10-12）

1. **z 契约对账（补齐 §75.10 的缺口）**：用只读探针跑**两个**契约。
   10-clip 契约：部署 `mu_q` 与训练 `mu_q + sigma*eps` 两项读数**逐位相同**
   （`thr50` 0.0591 / `hit@1` 0.1333）。本臂自身的 20-clip 契约：同样逐位相同
   （`thr50` **0.0712** / `hit@1` **0.1292**，与 `semreadout_ep6_gate.json` 的 M1 实读
   **逐位一致**，即该探针确实复现了门读数）。干预检查：契约切换确实移动 z
   （相对变化均值 **33.45%**）但被语义头吸收（热图相对变化 **1.32%**）；噪声占比
   `sigma/mu` 均值 **0.2768**。⇒ 「契约错配」从嫌疑名单划掉，与 A 的 §72.10 同结论。
2. **split / BN 归因（补齐「来源归属」）**：补跑 train split 的 BN 模式对照（只读，20-clip）。
   eval-BN `thr50`：train **0.4355** / val **0.0712**（差 **+0.3643**）；只切 z 路径 4 个 BN 的
   Δ`thr50`：train **−0.0933** / val **−0.0212**。⇒ 差距**主要来自数据 split**，
   归一化模式是次要项（与 A 的 §72.10 同向）。
3. **生成器缺陷（归档前已修，commit `919e6b8`）**：§75.10 的结论句原本把 A 的比率当
   **手写常数**渲染（0.5022/0.1167），而该比率本身就是「末值 train 读数 ÷ 20-clip val 读数」
   两个口径混在一个数字里；它会原样复制进每一臂（A2-R 自身是 0.4923/0.0591）。
   现在两个操作数都从本包读出并**各带口径**，且不再打印任何乘数。修好之后，
   §75.14 的「未手抄任何读数」对本块才真正成立。
4. **未完成、且不改变本包判定**：A2-R 的 M1-loc 不适用（§5 预登记为 A2-S 专属）；
   C1/C2/C3 匹配消融、同容量确定性对照、真正的 z 排他性、多 seed、S2（ep12–16）
   窗口与候选 C/B/D 均**未做**。
5. **人工裁决点**：S1 **NOT ALL PASS** ⇒ 按 §7 归档、**不续 ep16**（§6：ep16 只在 S1
   全过时允许）。因 M3 FAIL 且 S0-5 未明显高于 0.2530，§7 指向「分类读出是否应走独立分支」
   的单变量验证；若两臂都不过的分支同时生效，论文主线则转向「为何确定性状态在缺测下不足」
   （需候选 C 的证据）。**本次不自动立项、不自动投训**；任何一个新臂都需要先补预登记
   （唯一变量、门槛、采样契约、早期门与成熟门分离）再经用户批准。
6. **不做**：权重/KL/增益扫描、ep24 抢救、DC 家族继续叠层。

### 75.14 本节边界

* 无 S2 性能宣称；§75.0/72.4 的均值是 ep6 早期上下文。
* 未改模型、未改判据数值、未动重建目标/KL/权重/日程。
* 本节由 `me_rssm/sanity/semA_ep6_record.py` 从证据 JSON 与
  `*.log.json` 机械生成，未手抄任何读数；判定沿用
  `semA_ep6_verdict.py` 的同一套门槛常量。


### 75.15 下一方向审查：C 先诊断立项，不直接开训（2026-10-12）

本次核查 §74.11 / §75.12 / §75.13 与 A2 预登记 §6/§7，回应用户「是否按三路建议推进」。
范围仅方向审查与记录；**未启动训练、未跑 GPU 探针、未改模型/既定门槛，未复跑 391 条测试**。
§75.13 的 391 passed 是已有记录，本次不冒称新的测试结果。

**证据边界**：

- A2-S/A2-R 均已归档，不进 ep16。A2-R ep1–6 等权均值 33.9502（n=6），
  全部已有 val 峰值 38.4943 @ ep5；best saved 36.3491 @ ep6。
  默认 ep12–16 未形成，本节不新增性能结论，完整曲线/逐类/机制表保持在本节前文。
- 本臂已测语义指标在 z 两契约下相同，且 BN 切换不足以解释 split 差距；
  排除了这两个具体诊断口径下的主要嫌疑，**没有证明所有输出/所有样本对采样都不敏感**。
- 数据 split 差距与稀疏峰漂移不等于「雷达缺测导致失败」，也不等于「确定性状态不足」。
  后两条必须各有缺测分组与匹配确定性对照，不能由 A2 负结果逻辑推出。
- M3 FAIL 让独立分类分支保留为候选；但没有证据表明它能修复三臂共有的 M1 泛化失败。
  `frac_cells_residual_can_flip` 的“明显上升”未有预定数值显著性线，不能当成全面排除通道问题的定理。

**建议执行顺序（研究建议，不是用户开训授权）**：

1. C 优先进入**只读诊断/立项**，不按「选 b 即完整预登记后自动开训」执行。
   在训练集/训练内校准集冻结 coverage 与运动可观测性分组，再在留出集只做评估。
   按 class × coverage × motion-observability 报样本数/场景数、峰命中/CE、召回与真实检测误差；
   控制距离、目标大小等混杂。自然缺测与受控雷达屏蔽分开报告；必须保留有效相机信息。
   对象关联、BEV 几何或场景/类别构成差异若未核验，保持替代解释，不把 split 直接归因缺测。
2. 只有失败富集于弱/缺雷达且有效相机信息仍可用，才继续 C 的可证伪更新机制预登记。
   否则转向对象级读出/空间泛化诊断，不强行包装缺测主线。
3. C 首臂从标准 no2d_igdr 对照出发，先验证最小的缺测可区分 camera correction/fallback，
   不在失败 A2/DC 上同时叠分模态创新、协方差、新头、损失等多个变量。
   之后再逐臂验证分模态 innovation、标准化/校准与随机状态增益。
4. C 预登记须包含实际观测/预测空间、innovation/covariance 公式、有效性 mask 与缺测处理，
   训练集标定规则、类别分层采样/聚合、机制门与同 epoch 早期对照、独立 ep12–16 任务门/止损，
   以及同容量/同训练预算确定性对照。A2 的排序/48 通道映射不机械套用为 C 的必填机制。
5. “为何确定性状态在缺测下不足”仅作待证假设；若匹配确定性方法同样有效，贡献应归于
   可观测性建模/可靠性校准，不宣称概率 RSSM 必要。四类稳定提升仍需要成熟窗口与配对多 seed。

本次方向审查完成；C 诊断、正式预登记、实现与训练均未由本次自动执行。

## 76. 候选 C 立项诊断（第一步）：缺测分组的可区分性与混杂（2026-10-12，只读）

用户裁决：方向倾向候选 C，但**先补 C 的立项证据再决定训练**；卡空着不是开训依据。
本节执行该诊断的**数据侧一半**：建立「class × 雷达覆盖」分组契约，并检查它是否会
把「缺测」与「远/小/场景不同」混为一谈。**未改网络、未启动训练、未动任何判据。**

**覆盖度量**：`annos['num_points_in_gt']` —— 由
`tools/data_converter/kitti_converter_TJ4Dradar.py::_calculate_num_points_in_gt`
用 `box_np_ops.points_in_rbbox` 统计**模型实际加载的同一点云**
（`LoadPointsFromFile(load_dim=8, use_dim=[0,1,2,3,5])`，即 8 维 4D 雷达点）
落在每个旋转 GT 框内的点数。故 0 = **该对象在网络看到的张量里没有任何回波**，
不是来自另一传感器的代理量。`num_points_in_gt < 0`（ignored）与 `DontCare` 全部剔除。

> **§77.4 更正（2026-10-12）**：该列是在**图像视锥过滤后**的点云上统计的，而网络管线
> 不含这层过滤。实测 val 零覆盖对象里 **95.6%** 在网络实际加载的点云中也确实 0 点，
> 其余约 4.4% 属「视锥假零」。读本节时请按此口径理解，细节见 §77.4。

**分组阈值只用训练集确定**（§5 式契约）：
`zero = 0`（**不从数据分位取**：0 是「张量里不存在」这一状态，不是密度），
`low = 1–4`、`mid = 5–10`、`high ≥ 11`（train 正覆盖总体三分位 q1=4.00、q2=10.00）。
val 只**套用**这组边界，不参与定界。

### 76.1 分组规模与混杂控制

| split | 组 | 对象数 | 出现帧数 | 覆盖中位 | 距离中位 (m) | 图像框高中位 (px) | occluded≥1 占比 | 最大场景占比 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| train | zero | 7406 | 3116 | 0 | 54.15 | 39.36 | 49.2% | 25.3% |
| train | low | 6610 | 3685 | 2 | 44.46 | 52.22 | 35.1% | 12.8% |
| train | mid | 5046 | 3172 | 7 | 32.33 | 74.34 | 30.1% | 12.3% |
| train | high | 5139 | 3510 | 21 | 31.65 | 100.36 | 35.2% | 10.6% |
| val | zero | 1801 | 1017 | 0 | 43.39 | 47.92 | 45.4% | 25.7% |
| val | low | 2342 | 1196 | 2 | 43.20 | 50.84 | 26.7% | 23.6% |
| val | mid | 1893 | 1184 | 7 | 31.60 | 68.53 | 24.4% | 21.3% |
| val | high | 3045 | 1428 | 49 | 29.75 | 97.31 | 61.1% | 21.5% |

规模：train 5706 帧 / 24201 对象 / 34 个场景块；val 2040 帧 / 9081 对象 / 11 个场景块。
场景块由 `image_idx` 连续性切出（跳变 > 1 即新块），不依赖文件名编码假设。

**zero 组按类别**（缺测率，train → val）：Pedestrian **58.6% (1908/3257) → 40.7% (407/999)**、
Cyclist **40.2% (2096/5212) → 25.9% (558/2153)**、Car **26.0% (3031/11649) → 15.9% (694/4361)**、
Truck **8.6% (340/3963) → 7.2% (101/1404)**。两 split 的类别次序一致：**Ped ≫ Cyc > Car > Truck**。

### 76.2 数据侧先决条件：zero 组不是「远小到看不见」的替身

| split | zero 组框高中位 | 正覆盖组框高中位 | zero 组 ≥25 px 占比 | 正覆盖组 ≥25 px 占比 |
|---|---:|---:|---:|---:|
| train | 39.36 px | 71.56 px | 97.9% | 99.8% |
| val | **47.92 px** | 70.83 px | **98.9%** | 99.9% |

⇒ val 的 zero 组里 **98.9% 的对象在图像上仍高于 KITTI easy 的 25 px 线**，
即「相机侧信息可能仍在」这一 C 的前提**在数据侧成立**（这是必要条件，**不是**失败富集的证据）。
距离上，val zero 组中位 43.39 m 与 low 组（43.20 m）几乎相同，并非只由远距造成。

### 76.3 两个必须写进预登记的混杂（本轮最重要的产出）

1. **覆盖与距离/尺寸强相关**：train zero 组距离中位 54.15 m vs high 组 31.65 m，
   框高中位 39.36 px vs 100.36 px。⇒ **单按覆盖分组不是受控比较**；后续必须做
   距离/尺寸分层或配对，否则「缺测组更差」会与「远小目标更差」不可分。
2. **split 本身移动了覆盖分布**：zero 占比 train **30.6%** → val **19.8%**；
   high 占比 train **21.2%** → val **33.5%**，且同样边界下 val high 组覆盖中位 **49** vs
   train **21**。即**同一组训练边界在 val 上含义不同**。这直接影响 §74.11「差距主要来自
   data split」的解读：该差距同时伴随**覆盖分布的位移**，不能读成「val 更难（缺测更多）」——
   方向上 val 的覆盖反而更密。任何「缺测导致失效」的迁移结论都必须先声明这条。
   （另注：val high 组 occluded≥1 占比 61.1% vs train 35.2%，提示两 split 的标注/遮挡
   构成也不同，属同一类口径风险。）

### 76.4 判定与下一步

* 数据侧**通过**：缺测组真实存在（val 1801/9081 = 19.8%）、可区分、类别次序稳定、
  且相机先决条件在数据侧成立。
* 数据侧**不构成立项**：本节没有测到任何「检测失败富集于缺测组」的证据。
  按用户口径，**尚未满足进入 C 的条件**。
* **缺的两块（都不自动转为训练）**：

  **(a) 第二根分组轴：运动可观测性**（用户口径要求 `class × coverage × motion-observability`）。
  本节**只做了 coverage 轴**。要做运动轴，须先从点云 8 维特征里确认速度/多普勒通道语义
  （`use_dim=[0,1,2,3,5]` 中的第 5 维在 `RadarPillarFeatureNet`/`radar_motion_chain.py`
  里对应哪个物理量，必须先核对代码而不是按位置猜），再按「无回波 / 有回波但速度不可分辨 /
  速度可分辨 / 测量冲突」切分，并按同一规则（阈值只用 train 确定）冻结。

  **(b) 失败富集侧（只读、需 GPU，非训练）**：把检测结果挂到这些分组上——
  (i) 逐帧匹配后按组计算 **GT 召回与 FN 归属**（FP 归属规则须预先写死）；
  (ii) **M1 峰漂移 / 分类 CE 按组**重算（复用 §75.6 同一评分函数）；
  (iii) 自然缺测与**受控雷达屏蔽**分开报告；
  (iv) 全部按距离/尺寸**分层**出具，并声明离线 harness 与训练内 EvalHook 的口径差
  （§51.5 记录约 2.2 个 3D 点）。
* 产物：`me_rssm/sanity/probe_c_coverage_groups.py`（只读、CPU），
  本跑 `--train-info data/TJ4D/TJ4D_infos_train.pkl --val-info data/TJ4D/TJ4D_infos_val.pkl`；
  9 条单测覆盖「阈值只由 train 决定」「0 不取分位」「ignored/DontCare 剔除」
  「场景块切分」「相机先决条件」等契约。
* **不自动进入训练**：C 的最小臂仍需独立预登记的观测空间、innovation/covariance、
  有效性 mask、标定规则与对照（不是复用 A2 的排序损失或 48 通道清单）。


### 76.5 执行顺序审查：先 CPU 运动轴，再 GPU 失败富集（2026-10-12）

本次回应两个待办的先后选择，核查 §76 与运动链/编码器代码，**仅审查记录**；
未实现/运行新运动分组、未启动 GPU 失败探针、未训练、未复跑已记录的 400 条测试。

- 建议先完成 CPU 运动轴的语义与分组契约；GPU 失败富集依赖冻结后的分组，不能倒序后按结果调边界。
- `RadarPillarFeatureNetMotion._motion_stats` 实际取选维后 `features[...,3]` 为径向速度、
  `features[...,4]` 为 SNR（`me_rssm/radar_motion_chain.py:93-94`）；
  父编码器在 `pillar_encoder.py:598-603` 将 `3:5` 联合作 velocity/SNR center。
  因 `use_dim=[0,1,2,3,5]`，代码消费语义对应原始索引 3 为径向速度、原始索引 5 为 SNR；
  **选维后的第 5 个通道不是速度**。这核实了代码约定，不替代原始文件生成端的物理定义。
  仍需核验数据格式来源、单位、符号、是否自车补偿、无效值编码与校准精度。
- 不把低幅径向速度等同于静止，也不把 train 分位阈值等同于传感器速度分辨率。
  缺少噪声/精度及几何依据时只称低/高 Doppler 证据，真实运动可观测性保留未判定。
- 「测量冲突」需要明确的预测量、共同坐标/时间及残差规则；单帧速度方差大仅是混杂/离散度，
  不能直接充当冲突。若 CPU 数据没有可靠参考预测，该项记 NOT MEASURED，另由后续探针测。
- CPU 产物需保存原始量/有效点数、列映射、阈值来源与指纹、class×coverage×运动代理的样本数/
  场景数，以及距离/尺寸/遮挡构成；训练侧定阈值、留出侧只应用，低样本格显式标注。
- 后续 GPU 契约先冻结 TP/FN 匹配、score/NMS/IoU、FP 归属；无匹配 GT 的 FP 不强行归入
  某个 GT 缺测组。各臂使用同一离线 harness 成对比较；§51.5 的约 2.2 点差是历史观测，
  **不能当通用校正常数加减**。自然缺测与受控屏蔽必须分开。

立项条件不变：控制距离/尺寸/场景混杂后测到失败富集才考虑 C；没有关联则转对象级读出/
空间定位/跨场景泛化。CPU 数据审查与 GPU 探针都不自动授权 C 训练。

---

## 77. 候选 C 立项诊断（第一步·第二个分组轴）：运动可观测性（2026-10-12，CPU 只读）

用户裁决：**先做 CPU 运动轴，再做 GPU 失败富集**；不得倒序，GPU 探针依赖冻结后的分组，
否则会「看结果再调边界」。**不自动转训练。** 本节执行 CPU 半边。

产物（全部新增/只读）：

| 项 | 值 |
|---|---|
| 探针 | `me_rssm/sanity/probe_c_motion_axis.py`（新，645 行，CPU 只读）sha256 `efcd569d…` |
| 单测 | `me_rssm/sanity/test_probe_c_motion_axis.py`（新，17 条，**全部通过**） |
| 结果 JSON | `/data/lurui/work_dirs/fgfull_N4_no2d_igdr_2x4_24e_seed0/c_motion_axis_20261012.json`（2.6 MB）sha256 `497176f9…` |
| 覆盖 | **stride=1 全量**：train 5706 帧 + val 2040 帧，33 个 `image_idx//10000` 前缀块（train）/11（val） |

**口径声明（必填三项不适用）**：本节**不产生任何训练指标**（无 val 的 Overall/BEV/四类 AP），
因此不存在「区间均值 / 全轮峰值 / best saved」三项记录口径；本节唯一的数值口径是
**分组计数 + 分位数**。任何 AP 数字都不在本节出现，也不会拿本节数字充当性能结论。

### 77.1 列语义证据（用户要求项 1）

用户裁决：`use_dim=[0,1,2,3,5]` 下选维后索引 3 = 径向速度（原始索引 3）、
选维后索引 4 = SNR（原始索引 5），**选维后的第 5 个通道不是速度**；但必须追原始数据生成端的
单位/符号/自车补偿/无效值编码，不能只凭消费代码。本节按三层证据执行：

| 通道 | 代码约定（消费侧） | 实测恒等式 | 判定 |
|---|---|---|---|
| 选维后 idx 3 ← 原始 idx 3 | `radar_motion_chain.py:81`（x,y,z,径向速度,SNR）、`:93-94` 取 `features[...,3]` | —（无法由几何自证） | **径向速度**（代码约定） |
| 选维后 idx 4 ← 原始 idx 5 | 同上，`snr = features[...,4]` | 取值范围 [0.44, 27.10]，无 0 值 | **SNR**（代码约定） |
| 原始 idx 4 | — | `dim4 == sqrt(x²+y²+z²)`，max\|err\| **3.38e-05**（5 帧全量） | **距离，米**（实测成立） |
| 原始 idx 6 | — | `dim6 == -degrees(atan2(y,x))`，max\|err\| **5.14e-06** | **负方位角（度）**（实测精确成立） |
| 原始 idx 7 | — | `dim7` ≠ 俯仰角：max\|err\| **30.03°**，最佳线性斜率 −1.074、残差 std 0.180 | **UNIDENTIFIED**（不给物理名） |
| 原始 idx 2 / idx 7 | — | `dim2 == 0` 与 `dim7 == 0` 是**同一个点集**（5/5 帧相同，占比 5.0–5.5%） | **缺俯仰测量标记**（合并处理） |

- `NaN` 计数 **0**；`dim0/dim4/dim5` 无 0 值 ⇒ 这三列没有填充编码。
- **原始生成端不在本仓库**：`tools/data_converter/*` 只**读** 8 维 `.bin`，没有任何 exporter 代码。
  ⇒ 单位、符号、是否自车补偿**不能由代码断言**，只能实测（77.2）。
  单位：与 m/s **一致但未证明**；本记录中一律写「与 m/s 一致」而不是「单位 = m/s」。
- 与 §76.5 的差异：那里只核实了**消费侧代码约定**并明确「不替代物理定义」；本节补齐了
  几何恒等式与自车场实测，并把仍然查不到的项显式记为 **NOT MEASURED / UNIDENTIFIED**。

### 77.2 原始统计：速度通道是「未补偿的径向速度」，逐帧自车共模项 0–43.5 m/s

**模型**：逐帧在**所有计分 GT 框之外**的点上最小二乘拟合
`v_raw = a·cosθ + b·sinθ + c`（θ = atan2(y,x)，一次 3σ 截尾后重拟合）。
静态杂波模型下 `(−a,−b)` 即自车在传感器系的速度，**符号为负**——车向前开时静止回波系统性为负。
该模型不是假设出来的：它在干净帧上解释了**几乎全部**速度方差，见下面的单帧证据。

| split | 帧数 | 自车幅度中位 (m/s) | p10 | p90 | 帧最大 (m/s) | 自车 <1 m/s 帧占比 | R² 中位 | R²>0.9 占比 | v_raw 恰为 0 的占比（帧均值） |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 5706 | 8.29 | 0.003 | 11.74 | 43.53 | **20.4%**（1165 帧） | 0.521 | 13.2% | 17.7% |
| val | 2040 | 7.22 | 4.35 | 11.47 | 16.32 | **9.3%**（190 帧） | 0.528 | 23.8% | 9.3% |

（R² 统计在 `v_raw` 标准差 > 0.5 的帧上；R² 中位只有约 0.52 是因为残差里含**真实目标运动**
（迎面车 10 m/s 量级），不是噪声估计——高 R² 帧（>0.9）恰恰是「该帧几乎全是静态杂波」。

**单帧证据（`image_idx=210000`，4526 点）**：自车幅度 **8.24 m/s**、**R² = 0.99639**、
残差 std **0.048 m/s**；该帧两个 Truck 的 `v_raw` 均值 −8.22 / −8.25 m/s，
按自车场校正后 **−0.00 / 0.00 m/s**（对象级残差 std 0.05 / 0.04）。
⇒ 校正后的量正是「对象相对自车的径向运动」，且它与自车项在数值上完全可分。

**逐块结构**（`image_idx//10000`；train 33 块 / val 11 块）：

| 类别 | 块 | 自车幅度中位 (m/s) | R² 中位 | v_raw 恰为 0 占比 | 帧数 |
|---|---|---:|---:|---:|---:|
| train 最高 | 7 | 17.2 | 0.401 | — | 171 |
| train 最高 | 6 | 16.2 | 0.061 | — | 230 |
| train 停驻 | 4 | 0.01 | — | 0.921 | 265 |
| train 停驻 | 8 / 42 / 43 | 0.00 | — | 0.993 / 0.995 / 0.997 | 192 / 195 / 105 |
| val 最高 | 1 | 14.0 | 0.294 | — | 174 |
| val 停驻 | 41 | 0.00 | — | 0.997 | 190 |

**为什么这直接改变分组口径**：raw |v| 的分位数本质上是「该帧自车速度」的分位数
（停驻块 `v_raw == 0` 占 97–99.7%，行驶块 ≈ 0%）。若按 raw |v| 切低/高 Doppler，
得到的其实是「停驻/低速块 vs 行驶块」，即把**块/场景差异**当成**对象运动差异**——
这正是用户警告的温床。**故本节的幅度轴一律用 `v_corr = v_raw − (a·cosθ + b·sinθ + c)`**，
阈值只在 train 的 `v_corr` 上冻结。

### 77.3 冻结阈值（训练侧，留出侧只应用）

| 轴 | 阈值 | 来源 | 性质 |
|---|---|---|---|
| coverage | `zero = 0`（定性，**不取分位**）、`low 1–4`、`mid 5–10`、`high ≥11` | §76（train 正覆盖三分位 q1=4.00/q2=10.00） | 沿用，**本节未改** |
| 运动幅度 | `v_amp_low ≤ 2.079 m/s`、`v_amp_mid ≤ 9.421 m/s`、`v_amp_high > 9.421` | 本节：train **正覆盖**对象的逐对象 `mean|v_corr|` 三分位 | 新增，只从 train 冻结 |
| 无回波 | `no_return`（**不进幅度组**） | 结构性规则 | 避免把「无测量」当成「低测量」 |

**口径纪律**：`v_amp_low` **不是**「静止」也**不是**「低于传感器分辨率」——探针里不存在
`static`/`resolution` 这类字段，17 条单测中有一条专门断言分组名里不得出现这些词；
「测量冲突」在制品里显式写为
`NOT MEASURED (no prediction, no shared frame, no residual rule)`，
运动可观测性写为 `PARTIAL (amplitude proxy only; not resolution, not static)`。

### 77.4 分组规模与构成（用户要求项 2）

**完整性检查**：覆盖率逐帧复算 = 存储的 `num_points_in_gt`，**0 个不一致帧**（train 5706 + val 2040）。
复算必须在**图像视锥过滤后**的点云上做（与转换器同一约定）；网络管线本身**没有**这层过滤，
因此另记「模型可见零覆盖」：val **1721/1801 = 95.6%** 的零覆盖对象在网络实际加载的点云里
也真的是 0 点（train 97.4%），其余约 4.4% 是被视锥滤掉的**假零**，须写进 C 的预登记。

| split | coverage\|amplitude | n_obj | n_frames | n_scenes | range 中位 (m) | 尺寸中位 (m³) | 框高中位 (px) | occl≥1 | 帧内点数中位 | 自车幅度中位 | \|v_corr\| 中位 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | `zero|no_return` | 7406 | 3116 | 33 | 54.2 | 3.06 | 39.4 | 0.49 | 2011 | 1.5 | 1.451 |
| train | `low|v_amp_low` | 1814 | 1154 | 29 | 35.5 | 1.87 | 57.1 | 0.45 | 2423 | 0.1 | 0.530 |
| train | `low|v_amp_mid` | 2517 | 1805 | 27 | 41.6 | 3.44 | 52.3 | 0.32 | 1932 | 5.8 | 5.877 |
| train | `low|v_amp_high` | 2279 | 1651 | 27 | 52.5 | 14.68 | 47.8 | 0.31 | 1442 | 8.6 | 12.396 |
| train | `mid|v_amp_low` | 1247 | 897 | 22 | 33.3 | 2.69 | 63.3 | 0.44 | 2353 | 0.1 | 0.408 |
| train | `mid|v_amp_mid` | 2024 | 1556 | 29 | 27.8 | 10.61 | 83.1 | 0.25 | 1823 | 6.1 | 6.741 |
| train | `mid|v_amp_high` | 1775 | 1323 | 26 | 36.6 | 14.80 | 72.5 | 0.25 | 1428 | 8.3 | 12.496 |
| train | `high|v_amp_low` | 2500 | 1544 | 20 | 36.4 | 14.34 | 76.6 | 0.60 | 2751 | 7.6 | 0.042 |
| train | `high|v_amp_mid` | 1040 | 909 | 24 | 26.3 | 31.43 | 121.9 | 0.20 | 1378 | 5.3 | 7.947 |
| train | `high|v_amp_high` | 1599 | 1474 | 24 | 30.5 | 36.94 | 106.5 | 0.07 | 1631 | 9.7 | 11.428 |
| val | `zero|no_return` | 1801 | 1017 | 11 | 43.4 | 2.60 | 47.9 | 0.45 | 2059 | 8.3 | 2.617 |
| val | `low|v_amp_low` | 654 | 441 | 11 | 35.5 | 0.85 | 58.9 | 0.41 | 2444 | 5.3 | 0.386 |
| val | `low|v_amp_mid` | 1240 | 744 | 8 | 43.2 | 2.67 | 49.1 | 0.21 | 1868 | 7.0 | 5.742 |
| val | `low|v_amp_high` | 448 | 342 | 7 | 47.1 | 13.73 | 51.6 | 0.20 | 2098 | 10.8 | 13.285 |
| val | `mid|v_amp_low` | 426 | 347 | 8 | 31.4 | 10.22 | 71.0 | 0.47 | 2456 | 5.5 | 0.416 |
| val | `mid|v_amp_mid` | 1074 | 716 | 8 | 30.0 | 2.62 | 72.0 | 0.18 | 1894 | 7.0 | 5.755 |
| val | `mid|v_amp_high` | 393 | 316 | 7 | 37.5 | 13.39 | 58.8 | 0.18 | 2202 | 11.6 | 13.539 |
| val | `high|v_amp_low` | 2299 | 948 | 8 | 33.7 | 14.39 | 86.5 | 0.73 | 2938 | 7.1 | 0.038 |
| val | `high|v_amp_mid` | 525 | 437 | 7 | 21.0 | 16.25 | 150.5 | 0.23 | 1857 | 7.2 | 5.818 |
| val | `high|v_amp_high` | 221 | 210 | 6 | 19.7 | 44.74 | 183.8 | 0.31 | 1908 | 9.4 | 12.551 |

读法（本节不解释检测成败，只解释**分组构成**）：

- 覆盖轴与距离/尺寸耦合（§76 已记，本节复现）：`zero` 组 train 距离中位 54.2 m、
  框高中位 39.4 px，而 `high` 组 30.5 m / 106.5 px。
- 幅度轴的耦合**弱得多且方向不同**：train 三档距离中位 34.9 / 32.1 / 39.0 m，
  说明 ego-corrected 幅度**不是**距离的替身（这是它相对 raw |v| 的第二个好处）。
- 但幅度轴与**遮挡**相关：`high|v_amp_low`（覆盖最密＋几乎无相对运动）遮挡占比
  val 0.73 / train 0.60，是全表最高 ⇒ 低幅+高覆盖组天然偏向被遮挡目标，预登记必须写。
- `high|v_amp_mid` 在 val 是「近距离大目标」格（21.0 m / 150.5 px），典型是自车前方慢行车流。

### 77.5 三个新混杂（必须写进 C 的预登记）

1. **自车共模项（77.2）**：速度通道是未补偿的径向速度，逐帧自车幅度 0–43.5 m/s，
   且**在 split 之间分布不同**（train 20.4% 帧自车 <1 m/s，val 只有 9.3%）。
   任何使用速度特征的组分析都必须 ego-correct，并按块报告自车幅度。
2. **帧级掉线 ≠ 对象级缺测**：整帧没有回波时，「某对象零覆盖」不是关于该对象的证据。
   本节用**定性规则**（同帧所有计分对象都零覆盖 ⇔ `frame_level`，否则 `object_level`）
   分开统计——规则里没有可调边界，所以不存在事后调参空间：

   | split | `object_level` 零覆盖 | `frame_level` 零覆盖 |
   |---|---|---|
   | train | 7257 对象 / 3007 帧 | 149 对象 / 109 帧 |
   | val | 1790 对象 / 1008 帧 | **11 对象 / 9 帧** |

   ⇒ §76 的 val 零覆盖率 19.8% 里，**只有 0.6%（11/1801）是帧级掉线**，主体确是对象级缺测；
   但两类必须在 GPU 富集探针里分开报（契约已冻结）。
3. **幅度轴与类别强耦合（不是类别中立因子）**：

   | 幅度组 | Pedestrian (train/val) | Cyclist (train/val) | Car (train/val) | Truck (train/val) |
   |---|---:|---:|---:|---:|
   | `v_amp_low`（≤2.079 m/s） | 1260 / 572 | 870 / 108 | 2534 / 1984 | 987 / 747 |
   | `v_amp_mid` | 94 / 21 | 2211 / 1387 | 2478 / 1081 | 828 / 268 |
   | `v_amp_high`（>9.421 m/s） | **0 / 0** | 61 / 122 | 3737 / 631 | 1839 / 316 |

   `v_amp_high` 里**没有任何行人**，Cyclist 也极少 ⇒ 幅度轴必须**类别条件化**。
   同一档里 Car/Truck 的 `|v_corr|` 中位 ≈0.04 m/s（真静止车流），
   而 Pedestrian 是 0.84 / 0.99 m/s（**在运动的**慢速行人）——
   这就是「低幅 ≠ 静止」在数据里的直接反例。
4. （沿用 §76.3）**split 移动了覆盖分布**：不能把 §74.11 的 gap 读成「val 缺测更多」。

### 77.6 判定与下一步

- CPU 交付**完成**，四项要求齐备：列语义证据（77.1）、原始统计（77.2）、
  训练侧冻结阈值（77.3）、分组样本/场景数 + 距离/尺寸/遮挡构成（77.4）；
  另有 17 条 CPU 单测通过（冻结阈值只取 train、低幅不得命名 static/resolution、
  冲突必须 NOT MEASURED、覆盖计数必须复现存储值、帧级/对象级零覆盖必须分开）。
- **本节没有、也不能产生失败富集证据**（完全不接触检测结果）⇒ **不构成立项**，
  **不自动转训练**，卡片空闲不是开训依据这一条继续有效。
- **GPU 评分契约已在本节之后、运行之前冻结**：
  `docs/rssm_c_failure_enrichment_contract_20261012.md`。要点：TP/FN 匹配直接调用仓库
  同源 `clean_data` + `compute_statistics_jit`（metric=2，3D IoU），类阈值取被解释指标自己的
  口径（Ped/Cyc 0.25、Car/Truck 0.5），score 操作点 0.25（0.10/0.50 只作敏感性）；
  未匹配 FP 进**独立桶**、不得塞进任何 GT 的缺测组；自然缺测与受控雷达屏蔽分开；
  §51.5 的约 2.2 点只作**口径差**，禁止当通用校正常数加减；判定门槛
  Δrecall ≥ 3.0 点 + 至少 2 个距离箱同向 + scene-cluster bootstrap 95% CI 不跨 0。
- **下一步**：等用户批准后再运行 GPU 失败富集探针（只读）；测到富集才谈 C 立项，
  没关联就转对象级读出/空间定位/跨场景泛化。`(a)` 独立分类分支保留但不优先。

## 78. 候选 C 立项诊断（第一步·失败富集侧）：GPU 只读探针与预登记判定（2026-10-12）

用户裁决：**先做 CPU 运动轴（§77 已完成），再做 GPU 失败富集**；不得倒序——GPU 探针必须
挂在**已冻结**的分组上，否则会「看结果再调边界」。**不自动转训练。** 本节执行 GPU 半边：
用 run 自己的 config 快照 + `epoch_14.pth` 做一次只读前向导出预测，再在 CPU 上做全部统计。
**未训练、未改模型、未改 config、未开超参搜索。**

产物（全部新增/只读）：

| 项 | 值 |
|---|---|
| 探针 | `me_rssm/sanity/probe_c_failure_enrichment.py`（新，624 行，CPU 统计 + 仓库同源 IoU）sha256 `63f5089d…` |
| 单测 | `me_rssm/sanity/test_probe_c_failure_enrichment.py`（新，29 条，**全部通过**）sha256 `b7c0bbd1…` |
| 预测转储 | `<run>/c_failpred_ep14_dump.pkl`（34,028,115 B）sha256 `6d8e56bf…` |
| 同 harness AP | `<run>/c_failpred_ep14_harness_ap.json` sha256 `54ecc999…` |
| 结果 JSON | `<run>/c_failure_enrichment_ep14.json`（24 KB）sha256 `dfe969de…` |
| 运行日志 | `<run>/c_failpred_gpu_20261012.log`、`<run>/c_failure_enrichment_ep14.log` |
| 运行前冻结的契约 | `docs/rssm_c_failure_enrichment_contract_20261012.md`（sha256 记入结果 JSON） |

`<run>` = `/data/lurui/work_dirs/fgfull_N4_no2d_igdr_2x4_24e_seed0`

**口径声明**：本节**不产生训练结果**（无训练、无参数更新），因此「区间均值 / 全轮峰值 /
best saved」三项口径在本节**不适用**，本节也**不**新增任何性能结论。被复用的
`fgfull_N4_no2d_igdr_2x4_24e_seed0`（§47.2）三项口径仍以原记录为准：
区间均值 **39.2809**（fixed window ep12-16，n=5）、全轮峰值 **40.3853 @ ep14**、
best saved **40.3853 @ ep14**。本节唯一出现的 AP 是**同 harness 复核值**，只用于说明口径。

### 78.1 运行事实与「同一 harness」核对

- GPU5、单进程（非 DDP）；`tools/test_vod.py` + 该 run 的 config 快照 + `epoch_14.pth`；
  val **2040 帧全量**；导出预测后 AP 也由 `dataset.evaluate` 当场算出（03:36:04 起，约 10 分钟）。
- 3D AP easy/moderate/hard = 43.2207 / **40.3853** / 38.9777，与训练日志 ep14 行**逐位相同**
  （BEV 47.7999；四类 48.7109 / 50.5391 / 34.2463 / 28.0448 亦逐位相同）。
- 分解校验：`Overall_3D_moderate` = mean(Ped@loose 34.2463, Cyc@loose 50.5391,
  Car@strict 48.7109, Truck@strict 28.0448) = **40.3853** ✓ ⇒ 契约里
  「Overall = Ped/Cyc 用 loose + Car/Truck 用 strict」的写法在本 fork 上**实测成立**，
  各类 IoU 阈值由此确认（Ped/Cyc 0.25、Car/Truck 0.5）。
- **§51.5 的「离线 harness 约低 2.2 点」在本次没有出现**（逐位一致）。这是**观察**，
  只说明该历史差距**不是**本路径的固有常数；§51.5 的禁令（不得当校正常数加减）继续有效。
  本节所有对照都在这一套离线 harness 内，口径自洽。
- 预测转储规模 564,548 个检测 / 2040 帧（≈276/帧：网络 `score_thr=0.0`，导出前不过滤），
  全体检测 score 中位 **0.0220** ⇒ 契约里的 `score ≥ 0.25` 操作点是**分析侧**施加的。
- 运行陷阱（已处理）：`tools/test_vod.py:384` **无条件**把一次性的 `outputs.pkl`（本次 25 MB）
  写进当前工作目录，与 `--out` 指定的转储并存。该文件与实验无关，已在本次移出仓库并加入
  `.gitignore`（`/outputs.pkl`），避免将来被误提交。

### 78.2 冻结分组可复现性（先证明分组没漂，再谈结论）

- 用两个探针**自己的代码**在 val 重算分组，**13 个 cell 计数 + 3 个子计数 = 16 项全部相等**
  （探针内为硬断言，漂移即失败）。
- 本轮必须写清的一处语义：冻结产物里的 `zero|no_return` 是**全部 coverage==0 对象**（val 1801），
  **不是**「原始点云为空」的子集；后者是它内部的 `network_visible_zero`（1721），
  帧级/对象级再拆 11 / 1790。探针**按冻结语义逐键复现**，不自行重定义分组。
- 进入 TP/FN 归属的记录：n_gt 9081（KITTI 形式全部 GT，含 DontCare）→ **8917 条**
  可归属 GT（DontCare/难度忽略被仓库自己的 `clean_data` 排除），其中 found **3979**。

### 78.3 主口径：coverage 分组 × 检出（score ≥ 0.25、3D IoU、类阈值、全部距离）

| 分组 | n_gt | recall |
|---|---|---|
| zero | 1760 | **16.70%** |
| low | 2270 | 39.96% |
| mid | 1851 | 59.59% |
| high | 3036 | 55.17% |
| covered（low+mid+high） | 7157 | **51.49%** |

**pooled Δrecall（covered − zero）= +34.78 点。**

按类（每类都有正 Δ，无负类别被 Overall 掩盖）：

| 类 | zero n / recall | covered n / recall | Δ（点） |
|---|---|---|---|
| Pedestrian | 407 / 6.88% | 592 / 55.07% | **+48.19** |
| Cyclist | 558 / 17.74% | 1595 / 64.58% | **+46.83** |
| Car | 694 / 23.34% | 3667 / 53.91% | **+30.57** |
| Truck | 101 / 4.95% | 1303 / 27.01% | **+22.06** |

按距离箱（阈值即仓库 `clean_data` 的难度分箱：≤50 / 50–70 / >70 m）：

| 箱 | zero n / recall | covered n / recall | Δ（点） |
|---|---|---|---|
| ≤50 m | 1126 / 14.65% | 5619 / 57.75% | **+43.10** |
| 50–70 m | 545 / 23.67% | 1233 / 35.04% | **+11.37** |
| >70 m | 89 / 0.00% | 305 / 2.62% | +2.62（< 3.0 门槛） |

`zero_scope` 拆分：object_level 1752 条 / 16.67%（Δ = +34.82，与 pooled 几乎一致）；
frame_level 仅 **8 条** / 25.00% ⇒ **样本太小，不单独下结论**。
（注意分母不同：§77 的 11 是「全部对象」，这里是进入归属的 GT。）

### 78.4 次级轴（幅度）与「不是独立证据」声明

- `no_return` 1680 / 17.02% vs 有幅度 7237 / 51.03% ⇒ Δ **+34.01 点**。
- **必须写清**：`zero` 与 `no_return` 高度嵌套（1760 条 zero 记录里只有 **80** 条带原始回波）
  ⇒ 这**不是第二条独立证据**，只是同一信号的另一个视角。
- 幅度轴内部三档才是独立信息：`v_amp_low` 48.72%（3411）> `v_amp_mid` **61.23%**（2757）
  > `v_amp_high` 32.09%（1069）——「有回波」不等于「可判别」，高速档反而更差（含高速车流）。
- **本轮修正一处实现错误**（未提交前发现）：首版 `by_amplitude_group` 复用了 coverage 的
  reference（'zero'），得到「zero n=0 vs 全体 8917」的空对比；已改为 `reference=no_return`
  并补单测（含「默认 reference 仍是 zero」的防回归）。该修正**只涉及描述性表**，
  判定规则与主口径未受影响。

### 78.5 敏感性、类内稳定与 FP 独立桶

| score 操作点 | zero recall | covered recall | Δ（点） | Ped / Cyc / Car / Truck Δ |
|---|---|---|---|---|
| ≥0.10 | 24.83% | 59.54% | +34.71 | +56.85 / +49.18 / +30.40 / +17.66 |
| **≥0.25（主口径）** | 16.70% | 51.49% | **+34.78** | +48.19 / +46.83 / +30.57 / +22.06 |
| ≥0.50 | 11.53% | 44.91% | +33.37 | +42.54 / +44.99 / +28.57 / +21.82 |

⇒ 富集**不是操作点产物**，四类在三个点上**全部为正**。

FP 独立桶（score ≥0.25）：未匹配检测 **5958**（Ped 855 / Cyc 1883 / Car 2004 / Truck 1216），
**不进入任何 GT 分组**——`build_records` 只接受 `tp`/`fn` 状态，FP 在实现上无法被写进组内。

### 78.6 描述性稳健性（结果可见后新增：只描述，不参与判定）

- **场景级**：11/11 个 val 场景 Δ > 0，范围 **+4.87 … +56.47**
  （最小者 scene0：zero 79 / covered 390）。
- **类别 × 距离箱**（括号为 zero n / covered n；n<30 的格只作描述）：

| 类 | ≤50 m | 50–70 m | >70 m |
|---|---|---|---|
| Car | +44.93（348/2810） | +4.56（283/739） | +0.85（63/118） |
| Cyclist | +45.94（417/1487） | +17.20（131/108） | 无 covered（10/0） |
| Pedestrian | +47.40（295/573） | **0.00**（110/19） | 2 条（2/0） |
| Truck | +36.45（66/749） | **−4.19**（21/367） | +3.74（14/187） |

⇒ 效应**主要集中在 ≤50 m**；50–70 m 的 +11.37 由 Car/Cyc 承载，Ped 在该箱为 0、Truck 轻微反向；
>70 m 基本为零。
- **组成差异（把 §76.3 的混杂量化）**：zero 组距离构成 ≤50 m 64.0% / 50–70 m 31.0% / >70 m 5.1%，
  covered 组 78.5% / 17.2% / 4.3%；类别构成 zero 组 Ped 23.1% / Cyc 31.7% / Car 39.4% / Truck 5.7%，
  covered 组 8.3% / 22.3% / 51.2% / 18.2%。
  ⇒ pooled +34.78 里**含有距离与类别组成效应**，必须与分箱、分类结果一起读，不能单独引用。

### 78.7 预登记判定（逐条机械对照，无人工放宽）

| 条款 | 门槛（运行前冻结） | 实测 | 结果 |
|---|---|---|---|
| Δrecall | ≥ 3.0 点 | **34.78** | PASS |
| ≥2 个距离箱同向 ≥3.0（zero n≥30） | ≥ 2 | ≤50 m +43.10、50–70 m +11.37（>70 m 仅 +2.62） | PASS |
| scene-cluster bootstrap 95% CI 不跨 0 | — | **[21.57, 46.55]**（2000 次重采样，11 场景） | PASS |

**判定：失败富集 ENRICHED = TRUE（自然缺测下成立）。**
受控雷达屏蔽未测（契约明确要求分开报告），本节不得替它下结论。

### 78.8 结论、必写边界与下一步

**结论**：C 的立项前提**测到了**——自然缺测组的检出显著低于有回波组，且不是操作点
（0.10/0.50 同向）、单场景（11/11）或单距离箱（2/3 达标）造成的；四类**全部为正**。

**必写边界（防止过度解读）**：

1. 效应**集中在 ≤50 m**；50–70 m 只对 Car/Cyc 成立（Ped 为 0、Truck −4.19，且两侧 n 偏小）；
   >70 m 无证据（Δ +2.62）。
2. pooled Δ **含组成效应**（zero 组更远、更偏 Ped/Cyc），分箱/分类结果才是干净读数。
3. 幅度轴与覆盖轴**高度嵌套**，不算独立证据。
4. 这是**相关性，不是因果**：相机信息在该组是否仍可利用、受控屏蔽下是否仍然恶化，
   都**还没测**（后者必须另立预登记，禁止并入本节表格）。
5. `frame_level`（8 条）样本太小，不参与任何结论。

**下一步（等用户批准，不自动执行）**：按 §75.15 的分叉写 C 的**最小独立臂预登记**——
先验证「缺测/弱测量可区分」且「相机校正不随雷达缺失一起关闭」，含**受控屏蔽对照**与
**同容量确定性状态对照**；**不续训 A/A2、不开超参搜索、不靠调 score/IoU/分组边界救结论**。
若用户要求先补因果证据，则先做受控屏蔽的只读/小步实验，再决定是否立项。


---

## 79. 候选 C 因果补证：受控雷达屏蔽对照（2026-10-12，seed0 ep14，单次输入干预，未训练）

本节执行 §78.8 与用户指令的第三步「受控屏蔽因果检查」。干预定义、透明性对照与判定规则
在第一次 `mask_covered` 运行**之前**冻结在
`docs/rssm_c_failure_enrichment_contract_20261012.md` §8；本节只做机械对照，不新增门槛、不调参。
**未改模型、未改 config/checkpoint、未启动训练**；唯一变量是 val 输入点云——删掉落在
「冻结覆盖组 ≠ `zero`」GT 框内的每一个已加载点（无 margin、无抖动、不做逐帧挑选）。

**统一口径声明（AGENTS 规则 5）**：本节没有训练，故「区间均值 / 全轮峰值 / best saved」
三口径**不适用于本节结论**。仅引用基线（标准 no2d_igdr seed0，本 run）的既有三口径做跨节对照：
区间均值（fixed window ep12-16，n=5）**39.2809**、全轮峰值 **40.3853 @ ep14**、
best saved **40.3853 @ ep14**。屏蔽臂不是训练点，没有自己的三口径，不得为其编造。

### 79.1 运行事实与产物

- run：`/data/lurui/work_dirs/fgfull_N4_no2d_igdr_2x4_24e_seed0`，checkpoint **ep14**（best saved）。
- 流程：`tools/tool_c_masked_inference.py`（`identity` 与 `mask_covered` 两模式，施加在
  `LoadPointsFromFile` 之后、测试期确定性 BDA 之前）→ 与 §78 基线**完全相同的离线 harness**
  `me_rssm/sanity/eval_dump_c.py`（逐帧 dump、score 0.25、类 IoU 与总体定义同 §3 契约）
  → 配对判定 `me_rssm/sanity/analyze_c_masked_contrast.py`。
  屏蔽臂另经 `me_rssm/sanity/probe_c_failure_enrichment.py` 走 §78 同一冻结分组（仅次级）。
- 该臂的 harness Overall 一并写出用于**看见**口径差（§79.6）；仍禁止跨口径加减（§51.5 的
  约 2.2 点差是历史口径观测，不是可统一加减的校正常数）。

产物（sha256 为 2026-10-12 记录值；大文件在 run 目录）：

| 产物 | 大小 | sha256 |
|---|---|---|
| `c_mask_identity_ep14_dump.pkl`（identity 全 val dump） | 34,040,223 B | `aa0e54968665458d6edefa2167b2eeb7917bcc01b6065aa34facf7bea7972c2e` |
| `c_mask_covered_ep14_dump.pkl`（干预 dump） | 34,133,082 B | `6c630b3f5a9189bc73fc15dc21574eeadeff88040ab6c3a31509c3c24d495052` |
| `c_mask_identity_ep14_compare.json`（identity 逐帧比较） | 314 B | `50a62aa46ba05f94ca941e0d167d9373bfaf2a7893deceacaf6e02818cde313f` |
| `c_mask_covered_ep14_stats.json`（补丁审计） | 365,588 B | `787aa8ed804e58576ed3f5e412baa866fc38f75a0402e3c8f268af87c38e36d7` |
| `c_mask_covered_ep14_records.json`（8,917 条配对逐 GT 记录） | 1,862,503 B | `78332549baf02d07f8bce23a2a85e468234ad5d9850f41c16867d99467de4b25` |
| `c_mask_covered_ep14_probe.json`（屏蔽臂自然分组富集，次级） | 19,528 B | `0cf9a7d60532bcecda025a94e9264d9a1ebbfb3e1d080c5fdf1ce0204dad5c3c` |
| `c_masked_contrast_ep14.json`（配对判定） | 5,650 B | `798e28fd5769ff21ec14cc5936554cd082fa020f8c41bd29a244664e1cf1a1af` |
| `c_mask_covered_ep14_offline_ap.json`（该臂离线 harness Overall） | 4,177 B | `ca0bc723e5bf043d4096428227bf8305f2579f2c722f399e6053558247683985` |
| 冻结基线 `c_failpred_ep14_dump.pkl`（§78） | 34,028,115 B | `6d8e56bf7ed70eb2f1b58f58b377fa1be1bc6a6bd974441f87de6fe70e496dea` |

### 79.2 透明性对照（identity 必须逐位相同——通过）

- **8 帧冒烟 + 全 val 2040 帧**两次 identity dump 与冻结基线 `c_failpred_ep14_dump.pkl`
  逐帧字段比较：`frames_differs = []`、`max_abs_diff = 0.0`、双侧 583,163 个框、
  2040 帧（`c_mask_identity_ep14_compare.json`，`identical: true`）。
- **独立复核**：另起进程用原 `tools/test_vod.py` 路径重跑 2040 帧，其 dump 与冻结基线**同样逐位相同**，
  其进程内 AP json 与 `c_failpred_ep14_harness_ap.json` **完全相等**（Overall_3D_moderate 40.3853）；
  同一 dump 连评两次 AP 逐位相同（`/tmp/a2probe/double_eval_pass1.json` == `pass2.json`）。
  ⇒ 推断链路确定性成立，基线 dump 可信，§79.3–79.5 的数字有效。
- 说明（透明）：identity dump 的**文件** sha256 与基线文件不同（mmcv pickle 会写入对象状态），
  因此「逐位相同」以**逐帧字段**比较为准，且提前写进了契约。此现象不影响结论。

### 79.3 干预补丁审计（只触碰 covered 总体）

- 点云总量 5,428,297 → 5,038,547，删 **389,750** 点（**7.18%**）；触及 2031/2040 帧
  （9 帧 0 删除）；逐帧删除中位数 **34**、最大 **2176**。
- 被屏蔽 GT：Car 3667、Cyclist 1595、Truck 1303、Pedestrian 592 —— 四类合计 **7157**，
  与冻结主总体**完全一致**；另有 `Other` 123 个框（非四类，只影响输入、不进任何指标分组）。
- **越界检查**：落在 `zero` 组 GT 框内的点被删除数 **0/1030**；`zero` 组对象中心落入被屏蔽框 **0 次**。
  ⇒ 干预只作用于「当前有回波」的对象，`zero` 组的输入状态未被人为改动。
- 屏蔽对象距中位数 33.7 m；`zero` 组 43.1 m（构成差异，见 §79.6 边界 3）。

### 79.4 主统计量：配对 Δ_mask（同 score 点、同 IoU/难度、同 harness）

`Δ_mask` = recall(同一批 7157 个 covered GT | 屏蔽) − recall(同一批 | 基线)。
`recall` 的定义沿用 §78 冻结 harness（score ≥ 0.25、类 IoU 阈值 Car/Truck 0.5、Ped/Cyc 0.25，
Overall 单独口径见 §79.6）。

**合并口径**：0.5149（3685/7157）→ **0.3518（2518/7157）**，**Δ_mask = −16.31 点**。
翻转矩阵：TP→FN 1269、FN→TP 102、TP→TP 2416、FN→FN 3370。

| 分组 | n_GT | 基线 recall | 屏蔽 recall | Δ_mask（点） | TP→FN / FN→TP |
|---|---|---|---|---|---|
| 全部 covered | 7157 | 0.5149 | 0.3518 | **−16.31** | 1269 / 102 |
| Car | 3667 | 0.5391 | 0.4033 | −13.58 | 579 / 81 |
| Cyclist | 1595 | 0.6458 | 0.3925 | −25.33 | 409 / 5 |
| Pedestrian | 592 | 0.5507 | 0.3868 | −16.39 | 103 / 6 |
| Truck | 1303 | 0.2701 | 0.1412 | −12.89 | 178 / 10 |
| ≤50 m | 5619 | 0.5775 | 0.3995 | −17.80 | 1093 / 93 |
| 50–70 m | 1233 | 0.3504 | 0.2214 | −12.90 | 168 / 9 |
| >70 m | 305 | 0.0262 | 0.0000 | −2.62 | 8 / 0 |

- **场景聚类 bootstrap**（2000 次，11 个 val 场景块）：95% CI **[−19.75, −11.62]**，中位数 −16.23，
  不跨 0。
- 四类**全部为负**（−12.9…−25.3），Cyclist 损失最大；距离上 ≤50/50–70 两箱达标，>70 m 仅 −2.62
  （该箱基线本就只剩 8 个检出，空间小，不单独下结论）。

### 79.5 预登记判定（逐条机械对照，无人工放宽）

| 条款 | 门槛（运行前冻结） | 实测 | 结果 |
|---|---|---|---|
| identity 全 val 逐位相同 | 必须 | `identical: true`、max_abs_diff 0.0、独立复核一致 | PASS |
| Δ_mask（合并） | ≤ −3.0 点 | **−16.31** | PASS |
| ≥2 个距离箱同向 ≤ −3.0（箱内 n≥30） | ≥ 2 | ≤50 m −17.80、50–70 m −12.90（>70 m −2.62） | PASS |
| scene 聚类 bootstrap 95% CI 不跨 0 | 必须 | **[−19.75, −11.62]** | PASS |

**判定：causal_supported = TRUE。** 在「把已有回波拿掉」的干预下，同一批对象检出率
显著下降——失败富集（§78 的相关）不是纯粹由对象远/小/遮挡构成的假象，**返回值本身参与检出**。
**这不等于「相机信息不存在」**（相机链路未被干预，见边界 2），也不等于证明了某个内部机制。

### 79.6 次级与描述性读数（均不参与判定）

- **Δ_vs_zero（描述性）**：屏蔽后 covered 组 0.3518 vs 自然 `zero` 组 0.1670（基线未屏蔽）
  → **+18.48 点**。即**屏蔽并没有把 covered 组压到自然缺测对象的水平**，仍明显更高。
- **屏蔽臂自己的自然分组富集**（同 §78 冻结规则）：Δ = +20.01 点，CI [6.84, 33.79]，
  但达标距离箱只有 ≤50 m（+26.72），50–70 m 仅 +0.49 ⇒ **ENRICHED = FALSE**（次级，不推翻 §79.5）。
- **该臂离线 harness Overall**：Overall_3D_moderate **20.3606**（基线同 harness 40.3853，Δ −20.02；
  Overall_BEV_moderate 25.8107 vs 47.7999；Car 3D mod strict 32.7830 vs 48.7109；
  Truck 9.0648 vs 28.0448；Ped loose 13.7956 vs 34.2463；Cyc loose 25.7990 vs 50.5391）。
  只作**口径可见性**：说明这一干预对整张 val 的影响是全局性的，禁止把它与 §78/§79.4 的
  分组 Δ 混算或当成训练点。
- **组成差异量化**：covered 组距中位数 33.7 m vs `zero` 组 43.1 m；≤50/50–70/>70 构成
  78.5/17.2/4.3% vs 64.0/31.0/5.1%。⇒ `Δ_vs_zero` 含构成效应，只能描述，不能当作
  「屏蔽 vs 自然缺测」的配对结论。

### 79.7 结论、必写边界与下一步

**结论**：受控屏蔽的因果半步**成立**——把 covered 对象的雷达回波删除后，同一批 GT 的检出率
下降 16.31 点（CI 不跨 0、四类同向、两距离箱达标、identity 逐位透明）。结合 §78，
C 的立项前提（缺测 ↔ 漏检）从相关升级为**有干预支持的因果证据**。

**必写边界（防止过度解读）**：

1. **单次干预、单 checkpoint（seed0 ep14）、单臂**；未做多 seed、未做屏蔽比例/范围扫描，
   契约也禁止「改掩码范围再试」。
2. 相机输入**仍然有效**：检出下降说明雷达返回值被模型用于这些对象，但**不能**解释成
   「这些对象的信息不存在」；相机校正/融合是否关闭，本实验**没有测**。
3. `Δ_vs_zero` 是描述性追问且两侧构成不同（距离中位数差 ~9.4 m、类别构成不同），
   不构成「屏蔽 = 自然缺测」的等价性结论。
4. 自然缺测（§78）与受控屏蔽（本节）**分表报告、不合并**；两节的 Overall 来源不同口径，
   禁止互相加减（§51.5）。
5. 干预只删「covered GT 框内」的点；框外点、GT 框、DontCare 均未动；FP 仍不并入任何 GT 组。
6. 本节只陈述**检出层**的因果效应，不定位内部机制（编码/后验/监督/读出都可能参与）。

**下一步（等用户批准，不自动执行）**：按 §75.15 写 C 的**最小独立臂预登记**——先验证
「缺测/弱测量可区分」且「相机校正不随雷达缺失一起关闭」，含同容量确定性状态对照；
**不续训 A/A2、不开超参搜索、不改 score/IoU/分组边界**。本次到此为止：不改模型、不投训练。


### 79.8 立项审查：进入 C 最小独立臂预登记，不授权训练（2026-10-12）

本次核查 §78/§79 与运行前冻结契约。自然富集与受控屏蔽均通过预定筛选，
因此建议**现在编写 C 的最小臂预登记**，不再追加无关诊断延迟立项。
本次只做审查与记录，未编写正式方法预登记、未改网络、未运行探针/训练、未复跑 482 条测试。

**允许的证据结论**：冻结 seed0 ep14 检测器在既定输入干预下，covered GT 召回下降
16.31 点，scene 聚类 CI [−19.75,−11.62]；结合自然富集，足以进入缺测适应机制的立项筛选。
**仍不允许的结论**：已证明相机校正被关闭、C 能恢复丢失信息、自然缺测完全由回波不足解释、
概率状态优于确定性状态。相机输入未改不等于 camera correction 已被验证有效。

预登记的必须内容：

1. 先写标准 no2d_igdr 的实际融合/后验更新计算图与拟议改动入口，定位 coverage 是否控制
   相机分支。若没有原有断路，只能称“显式保留缺测下相机更新”的新机制，不称修复已定位的 bug。
2. 首轮唯一研究因素为观测有效性条件化的相机更新策略；固定 backbone、head、重建/KL、
   优化器、数据管线及日程。不同时加入校准协方差、标准化 innovation、新语义头或额外 Doppler 门。
3. 对照显式区分标准基线、C 更新策略、匹配确定性状态；使用控制矩阵分开估计更新策略增益与
   概率状态增益，不能拿 C-RSSM 对旧基线的一次差值同时证明两项。活跃参数量/计算量/训练预算
   列表对齐；不能用不参与计算的参数凑“同容量”。
4. 推理端 coverage/validity 只能由输入回波/相机有效性等模型可见量形成，不使用 GT 框内点数、
   GT 类别或诊断用框掩码做网络门控。GT 可用于训练监督及离线统计，不能进入测试决策。
5. S0 检验身份/旁路、mask 有效性与非 GT 推理输入；早期机制门用同 epoch、同 seed、同 harness
   的基线/控制，检验缺测下相机校正对真实 CE/召回的贡献，不以梯度/范数变化替代任务读数。
6. 自然缺测、受控屏蔽、完整输入分表，固定 §77–79 score/IoU/分组边界；frame_level 与
   object_level 缺测分开。受控 screen 只在既定总体上做，不把它等同自然缺测。
7. 成熟任务门使用正式训练内 EvalHook 的 ep12–16 五轮等权均值与四类非劣/止损，全部已有 val
   峰值报值+epoch；固定离线 harness 只承担匹配机制比较，不跨口径加减。多 seed 仍为最终验收条件。
8. 所有公式、映射、采样/聚合、阈值与停止规则在首次新臂运行前写死并待用户确认。
   若确定性对照同样有效，贡献归于有效性建模，不宣称随机 latent 必要。

**授权边界**：建议开始“写预登记”，不是授权实现/冒烟/正式训练；A/A2 仍不续训。

---

## 80. 候选 C 最小臂 C1（观测有效性条件化更新）：S0 接线门通过 + seed0 24e 正式启动（2026-10-12）

### 80.1 立项与唯一研究因素

- **假设**：标准 `no2d_igdr` 的后验更新在雷达缺测时仍以同一力度写入状态，导致缺测对象的状态
  被「无信息先验」污染；让后验更新显式条件化于**输入回波可见性**可以部分恢复 §79 中受控屏蔽
  造成的召回损失。
- **唯一研究因素**：观测有效性条件化的相机/雷达更新策略。公式（预登记冻结，未改）：
  `z_t = mu_q + (g_t − 1)(mu_q − mu_p)`，`g_t = 1 + tanh(W_v [h_t; valid_t] + b_v)`；
  `valid_t` 由**输入点云**逐 BEV cell 占用构造（`valid_layout='xy'`，pc range 与既有配置一致），
  **不读 GT**。
- **保持不动**：backbone、检测头、重建/KL、优化器、数据管线、日程、score/IoU/分组边界。
  未加入协方差校准、标准化 innovation、新语义头、Doppler 门或额外损失。
- **预登记**：`docs/rssm_c_validity_update_prereg_20261012.md`（含附录 A 冻结修正：ep6 改为
  早期曲线检查、分组召回机制门移到 ep14、zero 组 CE 改为逐类 3D AP 非劣）。
- **实现**：`me_rssm/validity_gain_rssm.py`；接线 `mmdet3d/models/detectors/R4Det.py`
  （仅当 fusion 声明 `accepts_obs_valid` 时传 `obs_points=points`，其它臂签名不变）。
  代码提交：`0364928`（实现）、`83279ba`（判定/看门狗工具）。

### 80.2 S0 接线门（450-iter 冒烟，GPU 5/6/7，3-rank DDP）

产物：`/data/lurui/work_dirs/c1valid_smoke_450iter/`（`20261012_084202.log.json`、
`iter_450.pth`、`s0_verdict.json`）。

| 判据 | 内容 | 结果 |
| --- | --- | --- |
| S0-1 | 450 iter 完成、权重落盘、**无 NaN** | PASS（max_iter=450） |
| S0-2 | 零初始化：step0 与主线**逐位相同**（`torch.equal`）；门已偏离恒等；门权重非零 | PASS（`abs_mean=0.001724`、`abs_max=0.008960`） |
| S0-3 | `v_gate_valid_mean`/`v_gate_invalid_mean` 逐行可读；mask 非退化 | PASS（有效帧 64；invalid cell share 均值 **0.9818**，范围 0.9776–0.9872） |
| S0-4 | 无 GT 推理输入、仅单一新增参数组（由测试断言） | PASS |

**S0_ALL_PASS = true**。

冒烟期门曲线（`v_gate_mean`）：iter50 **0.99693** → iter100 0.98322 → iter350 0.99306 →
iter450 **0.99380**；iter450 条件均值 `valid=0.98781` / `invalid=0.99384`，
说明门确实按有效性条件分化（不是常数旁路）。

**必写 caveat**：invalid cell share ≈ 98.2% 说明 mask 极稀疏——门的主要输入是「绝大多数 cell
无回波」。这不违反冻结的 S0 判据，但属于 ep14 机制门需要盯的观测（若 Δrecall 全部由稀疏
mask 的常量偏置驱动，应在此后归因中显式报告，而不是当作有效性建模的功劳）。

**容量（测试断言，256 通道配置）**：B0 9,191,510 → C1 9,257,558（**+66,048** 活跃参数
= (256+1)×256+256）；确定性对照基线 6,831,446 → C1-D 6,897,494（同一增益）；
C1 − C1-D = 2,360,064（随机分支的活跃参数差，供后续「概率状态必要性」判定使用）。

### 80.3 运行事实（正式启动）

- **启动**：2026-10-12 09:03:33 UTC，`CUDA_VISIBLE_DEVICES=5,6,7`，3-rank DDP，
  `--seed 0 --deterministic`，24 epoch（1902 iter/epoch）。
- 运行目录：`/data/lurui/work_dirs/fgfull_N4_no2d_igdr_2x4_24e_c1valid_seed0`
  （日志 `/tmp/c1valid_24e.log`；首个 log.json `20261012_090333.log.json`）。
- 配置：`me_rssm/configs/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_c1valid.py`。
- 观测速度：首轮 ~1.53–1.60 s/iter（iter250 @ 09:10:35），单轮约 50–52 min，
  全程 ETA 约 20.9 h；无 NaN/OOM；DDP 仅报 `find_unused_parameters=True` 例行警告
  （与「门在计算图中」一致，无未用参数）。
- **未完成**：ep6 早期检查、ep14 机制门、ep12–16 成熟门、C1-D 对照（串行后置）均未执行。

### 80.4 三口径与门（当前状态；★=训练中，待 ep6/ep16 回填）

| 口径 | B0（标准 no2d_igdr seed0） | C1（本臂） |
| --- | --- | --- |
| 区间均值 ep12-16（n=5） | **39.2809** | **39.8303**（Δ **+0.5494**；门 ≥40.00 **未过**，n=5 完整） |
| 全轮峰值（值@epoch） | **40.3853 @ ep14** | **40.6389 @ ep16** |
| best saved（值@epoch） | **40.3853 @ ep14**（另有 ep22 权重） | **40.6389 @ ep16**（saved = 2,4,6,8,10,12,14,16） |

对照 B0 同窗四类均值（非劣线）：Car **47.8399**（线 47.34）、Truck **29.7246**（线 29.22）、
Pedestrian **30.9277**（线 29.93）、Cyclist **48.6313**（线 47.63）。

**ep12-16 逐类判定（2026-10-13，n=5）**：C1 = Car **54.1636 PASS**、Truck **36.4397 PASS**、
Ped **26.0396 FAIL**（−4.8880，触发止损）、Cyc **42.6783 FAIL**（−5.9530，触发止损）；
`all_pass=False`。完整表与解释见 **§86**。

门（冻结，未改）：

1. **ep6 早期检查**（同 seed 同日程）：C1 的 ep1–6 区间均值低于 B0 同窗（**29.0420**）
   **> 2.0 点** ⇒ 停并复查。
2. **ep14 机制门**（同 epoch/seed/离线 harness）：M1 covered 组 Δrecall ≥ 0；
   M2 自然 zero 组 Δrecall ≥ −0.5；M3 逐类 3D AP ≥ B0 − 0.5。
   ⚠️ **2026-10-12 预数据冻结修订（附录 B）**：本条款已由「剪枝门」**降级为归因诊断**——ep15-16 照跑，主对照窗口始终凑满 n=5；剪枝权移到 ep16→24 尾部决定。
3. **ep12–16 成熟任务门**：Overall 区间均值 ≥ **40.00**；四类非劣；任一类别 ≤ B0 同窗 −2.0 ⇒ 止损归档。
4. **归因底线**：若 C1-D 与 C1 同等有效，结论写作「缺测可区分的更新机制」，
   不主张随机状态必要性。

### 80.4.1 逐 epoch 曲线（滚动回填）

区间均值与峰值口径在窗口跑满后回填 80.4 表；本表只记逐 epoch 原始读数（C1 训练内 EvalHook，
同 seed0 B0 同口径并列）。

| epoch | C1 Overall | B0 Overall | Δ | C1 Car(strict) | C1 Truck(strict) | C1 Ped(loose) | C1 Cyc(loose) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | **17.4098** | 15.2640 | **+2.1458** | 30.6214 | 8.1517 | 13.2886 | 17.5775 |
| 2 | **30.9781** | 26.3716 | **+4.6065** | 39.7853 | 25.4075 | 24.3243 | 34.3952 |
| 3 | **37.6897** | 31.6591 | **+6.0306** | 52.1181 | 29.1161 | 22.8984 | 46.6263 |
| 4 | **38.5394** | 32.8506 | +5.6888 | 53.1737 | 27.7922 | 27.4884 | 45.7121 |
| 5 | **38.2473** | 34.0827 | +4.1646 | 49.2656 | 27.4938 | 26.9624 | 49.2656 |
| 6 | **39.9821** | 34.0243 | **+5.9578** | 54.5678 | 32.6541 | 26.8100 | 45.8965 |
| 7 | **39.8167** | 37.5701 | **+2.2466** | 51.2623 | 35.0172 | 30.5549 | 42.4322 |
| 8 | **38.3156** | 39.5965 | −1.2809 | 47.9986 | 35.8234 | 28.4944 | 40.9459 |
| 9 | **40.1707** | 38.5362 | +1.6345 | 51.6697 | 37.2443 | 26.8008 | 44.9680 |
| 10 | **39.8182** | 38.0619 | +1.7563 | 54.4491 | 32.2750 | 27.9674 | 44.5813 |
| 11 | **38.3994** | 38.9632 | −0.5638 | 45.5933 | 36.0048 | 28.3201 | 43.6793 |
| 12 | **39.6931** | 39.2953 | +0.3978 | 54.7745 | 37.3434 | 24.0884 | 42.5661 |
| 13 | **40.5805** | 37.9178 | +2.6627 | 57.7569 | 38.0583 | 25.1953 | 41.3116 |
| 14 | **39.6076** | 40.3853 | −0.7777 | 52.9015 | 35.9213 | 29.5839 | 40.0238 |
| 15 | **38.6314** | 38.8111 | −0.1797 | 51.0035 | 36.2504 | 23.0446 | 44.2273 |
| 16 | **40.6389** | 39.9949 | +0.6440 | 54.3816 | 34.6253 | 28.2859 | 45.2626 |

**ep1 读数边界**：单点、早期 epoch，不构成任何门的结论；ep6 早期检查按窗口均值（C1 ep1-6
区间均值 vs B0 同窗 29.0420，容差 −2.0），不是逐点比较。ep1 与 B0 ep1 唯一的差异来源是
C1 的有效性门（容量 +66,048 活跃参数，见 80.2）。

**ep1-3 窗口均值（滚动，n=3）**：C1 **28.6925** vs B0 同窗 **24.4316**，
Δ **+4.2610**（ep1 +2.15 / ep2 +4.61 / ep3 +6.03，逐 epoch 同向）。仍非门判据：
ep6 早期检查用的是 ep1-6（n=6）窗口均值 vs B0 **29.0420**。

**ep1-7 窗口均值（滚动，n=7）**：C1 **34.6662** vs B0 同窗 **30.2603**，Δ **+4.4058**
（ep1-6 为 +4.7657，n 增大后小幅回落但仍同向）。⚠️ **ep7 首次出现单类同轮负差**：
Cyc 42.4322 vs B0 ep7 49.2034 ⇒ **−6.7712**，而 Car/Truck/Ped 同轮为正
（+6.3437 / +7.9477 / +1.4662），Overall +2.2466。按用户优先级（AP 为主）与逐类报告义务，
该 Cyc 回落**必须在 ep12-16 窗口里单独核查**，不得被 Overall 正值掩盖。
ep7 Ped **30.5549 首次站上冻结成熟非劣线 29.93**（单点、非判定，漂移 std 3.4）。

**日程契约核对（ep2-4 逐点同 iter 比对 B0）**：KL 权重系数逐点一致
（ep2 0.1 / ep3 0.2 / ep4 0.3，例：ep4 it1850 C1 `loss_rssm_kl=0.3026` vs B0 `0.3022`），
说明「保持 KL/日程不变」的预登记契约在训练侧成立，C1 与 B0 的差异不是 KL 日程漂移造成。
`stat_kl_raw_mean` 两臂同步从 ~22（ep1 末）回落到 ~0.6（ep4），为日程本身的正常行为。

**ep1-6 早期检查（正式判定，2026-10-12 15:1x）**：C1 区间均值 **33.8077** vs B0 同窗
**29.0420**，**Δ +4.7657 ⇒ PASS**（门槛：低于 B0 > 2.0 点才停）。四类同窗全为正差：
Car +6.6672、Truck +5.3391、Ped +3.9191、Cyc +3.1373。产物
`c1valid_ep6_early_check.json`。

⚠️ **按 §81 的解释边界**：早期领先有系统性回落（斜率 0.33–0.51，早期前 6 名 5 个回落），
因此 **+4.77 不能被读作「方法已成功」**；C1 在 ep4-6 已连续贴近 B0 的成熟平台
（38.2–40.0 vs B0 ep12-16 的 39.28），是否能在 ep12-16 兑现仍待成熟窗口判定。

**看门狗 bug 与修复（2026-10-12 15:13，记录在案）**：`watch_c1valid_ep6.sh` 的 Stage A
把 `--log "$LOG"`（/tmp 下的 **文本**日志）传给 `c1valid_curve_check.py`，而该参数需要
**运行目录**（`.log.json` 在那里），导致首次触发时报 `no *.log.json found under
/tmp/c1valid_24e.log` 而未产出判定 JSON。修复：两处调用改为 `--log "$C1_WD"`；因运行中的
bash 按字节偏移读脚本（§73.6.5 教训），先 kill 旧实例再重启，Stage A 自动重跑并产出上述 JSON。
**教训**：队列/看门狗脚本改完后，必须用「真实产物路径」做一次 dry-run 再武装。

**训练中门曲线（epoch 1-7，各 epoch 末 iter1900 读数）**：`v_gate_mean`
1.0093 → 1.0138 → 1.0232 → 1.0170 → 1.0223 → 1.0285 → **1.0387**（ep7），
`valid` 0.9905/0.9934/1.0061/1.0019/1.0100/1.0082/1.0196，
`invalid` 1.0095/1.0139/1.0233/1.0171/1.0223/1.0287/1.0388 ⇒ valid/invalid 间距稳定在
**+0.012~+0.021**（invalid 恒高于 valid），即「有回波收缩、无回波放大」的反向偏置**持续存在、
缓慢上移但量级有界**（ep8 it300 回落至 1.0335），未发散。
按预登记不在训练中干预；ep14 机制门归因需回答该偏置与收益/损失的关系。

**训练中门曲线（epoch 1）**：`v_gate_mean` 0.9908→**1.0093**（iter 1900），
`valid=0.99054` / `invalid=1.00946`，即门学到的是「有回波 cell 收缩观测校正（g−1<0）、
无回波 cell 放大校正（g−1>0）」——**方向与立项动机（缺测时抑制写入）相反**，差值虽一致但量级
很小（~2e-2）。按预登记这不是门判据（S0 判据不含方向），不在训练中干预；但必须在 ep14 机制门
归因时回答「收益/损失是否来自这个反向偏置」。同 epoch 其他状态量：`mu_diff²` 0.99→1.92、
`posterior_std` 0.172→0.118、`kl_eff` 13.1→24.1、`grad_norm` 22.9→13.3（无 NaN）。

### 80.5 工具与只读验证

- 新增（本次提交）：
  - `me_rssm/sanity/c1valid_curve_check.py`：三口径窗口报告（区间均值写实际跨度与 n、
    全轮峰值值@epoch、best saved 值@epoch）+ 冻结门机械判定。自检：B0-vs-B0 复算得到
    39.2809 / 40.3853@ep14，与 §69 口径逐位一致。
  - `me_rssm/sanity/c1valid_ep6_verdict.py`：M1/M2/M3 判定（修 1 节 M1/M2/M3 条款）。
  - `me_rssm/sanity/watch_c1valid_ep6.sh`：分阶段看门狗（ep6 早期检查 → ep14 机制门 →
    ep12-16 任务门），已 `source .envrc`，已在 09:10:22 武装。
- 回归：`me_rssm/sanity/test_validity_gain_rssm.py` **14 passed**
  （本环境需 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest`，原因见 §69 记录的
  conftest→hydra→`omegaconf.vendor` 插件问题，与本改动无关）。
- 本次未复跑全量 496 项；`test_flops.py`/`test_grad_paths.py` 因 GPU0 被其他租户占满而不可用
  （非本次改动造成）。

### 80.6 结论与下一步

**结论（截至 2026-10-12 09:15 UTC）**：C1 的最小实现、接线、S0 门与冒烟全部通过，
正式 seed0 24e 训练在 GPU 5/6/7 运行中。**方法有效性尚未判定**——ep6/ep14/ep12-16 三道门
都还没到；本节的三口径单元格在训练产生 val 后回填，不以冒烟日志推断任何 AP 结论。

**下一步（自动，看门狗已武装）**：

1. ep6（约 14:15 UTC）：`c1valid_curve_check.py` 早期检查；未触发 >2.0 点则继续。
2. ep14（约 21:55 UTC）：构建 C1 ep14 dump（GPU5）→ 与 B0 冻结产物配对机制门判定。
3. ep16（约 23:40 UTC）：ep12-16 成熟任务门。
4. 任一门 FAIL 即归档负结果，按 §75.15 转对象级读出/空间定位/跨场景泛化分支；
   **不扫增益系数、不调 KL、不续 ep24 抢救**。
5. C1-D（确定性对照）在 C1 结论出来后串行执行，不与 C1 并行以免彼此抢 GPU 干扰日程对照。

---

## 81. 早期门预测效力校准：ep1-6 均值能否代表 ep12-16？（2026-10-12，CPU 只读）

**动机**：C1 仍在训练（13:11 时点在 ep5）。用户提出关键质疑——「只训练 6 epoch 求均值真的能
确定是否成功吗」。此前 A/A2 的归档表述把「未过早筛机制门」写成了「方法失败」，口径过重。
本节用仓库内**已跑满**的历史 run 直接量化早期窗口的预测效力，作为后续停止规则的依据。
**本节不改任何模型、不投训练、不改 A/A2 既有归档判定。**

### 81.1 数据与口径

- 来源：`/data/lurui/work_dirs/*/`（每目录取最新 `*.log.json`；已记录的 val 口径，
  epoch 内最后一次 val）。**同时具备 ep1-6（n≥5）与 ep12-16（n=5）**的 run 共 **31 个**。
- 两个窗口均用**等权均值**，主指标 `pts_bbox/KITTI/Overall_3D_moderate`（口径同 §0 定义）。

### 81.2 结果

| 统计量 | 全体 (n=31) | 竞争带（成熟均值 ≥37，n=24） |
| --- | --- | --- |
| Pearson r (ep1-6 vs ep12-16) | **0.808** | **0.600** |
| 斜率 (成熟 ~ 早期) | **0.514** | **0.326** |

- **早期前 6 名 → 成熟名次位移**：`mci_v2` +9、`dcrssm2` **+15**、`mai_seed1` +6、
  `mci_seed1` −2、`N3_h128` +11、`dcrssm1` +7 —— **6 个里 5 个大幅回落**。
- **成熟前 6 名 → 早期名次**：`no2d_igdr_seed1`（成熟 rank1）早期仅 **rank12**；
  `no2d_igdr_seed2`（成熟 rank3）早期仅 **rank13**。即**按 ep6 排名会漏掉最终最好的两个 run**。
- **同配方三 seed 的噪声地平**：早期 29.04/30.74/30.68（range 1.70，**std 0.96**）；
  成熟 39.28/41.92/40.12（range 2.64，**std 1.35**）。

### 81.3 判定

1. **早期窗口能抓灾难，不能识别赢家**：31 个 run 的成熟后 8 名全部落在早期低分侧
   （早期 ≤27.5），但成熟前 10 里有 4 个早期不在前 6。作为「成功判定」无效；
   作为「资源分配 / 灾难止损」有效。
2. **早期领先存在系统性回落**（斜率 0.33–0.51 ≪ 1；早期前 6 名 5 个回落）。
   ⇒ **当前 C1 的 +4.26 早期领先必须按此打折预期**，不得据此乐观；反过来，
   早期落后同样不能直接判死（`no2d_igdr_seed1/2` 就是早期落后但成熟最优的反例）。
3. **成熟单 seed 窗口本身也不足以下「成功」结论**：同配方 seed σ=1.35，而我们要检测的
   效应量级是 0.5–1.0 ⇒ 单 seed 的 1 点提升约 0.7σ。故结论梯度必须是：
   **S0 硬故障 → ep6 任务损害 → 跑满 ep12-16 得成熟判定 → 正结果才做 3 seed 确认**，
   任何一级都不能越级下结论。
4. **A/A2 的既有判定不改写**：它们停在**机制门**（非 AP 门），其中 M3 类条款
   （A2-R 15/20 vs 门槛 16/20）统计效力极低，属于「按规则 FAIL」但**不足以证明方向无价值**；
   M1/M4 类条款偏离大（如 hit@1 0.083 vs 0.45）属于**结构性**证据，与训练时长关系较小。
   正确表述：**「ep6 未达到预登记机制要求」已确定，「继续训练也无效」未证明，
   「成熟 AP 能否超过基线」未验证。**

### 81.4 建议的停止规则修订（**待用户批准；不自行改冻结预登记**）

| 层级 | 旧规则 | 建议 |
| --- | --- | --- |
| 硬故障 | 同 | **保留**：NaN/OOM/梯度断路/GT 泄漏/接线不等价 → 立即停 |
| ep6 | 机制门全过才继续 | **改为任务损害检查**：同期窗口均值低于同 seed 基线 >2.0 点才停；机制读数只记录不裁决 |
| ep14（C1 特有） | 机制门 FAIL ⇒ 不进 ep16 | **降级为归因诊断（不剪枝）**，保证 ep12-16 凑满 n=5；剪枝权移到 ep16→24 尾部决定 |
| ep12-16 | 任务门 | **保留**，作为成熟判定 |
| 多 seed | 单 seed 过门后 | 保留 |

- **ep14 条款必须现在改才算合法**：C1 尚在 ep5，ep14 产物不存在（预计 23:05 才产生），
  此刻修改属「预数据冻结修正」；若等看到 ep14 结果再改即为移动门槛。C1 的 ep6 条款
  （−2.0 任务损害检查）与本次建议一致，**无需修改**。
- **代价必须同时承认**：单臂从「~6 h 出结论」变成「~24 h 出结论」（24e ≈ 24.2 h 实测）。
  因此该政策只在**减少同时立项臂数**的前提下成立，不能退化成「所有想法都跑满 24e」。
- **若要用实验检验早停本身**：另立「训练时长验证臂」（固定原方法原日程、不扫参，
  跑满 ep12-16），排在 C1 之后，不与 C1 抢 GPU。

### 81.5 被 ep6 剪掉的三个臂：实际状态与可恢复性（2026-10-12，CPU 只读）

| 臂 | ep1-6 区间均值 | Δ vs B0 同窗(29.0420) | ep6 单点 | ep1-6 内峰值 | 停止依据 | epoch_6.pth |
| --- | --- | --- | --- | --- | --- | --- |
| A (`semA`) | **34.0459** | **+5.0039** | 35.7608 | **38.2203 @ ep3** | 机制门 M1/M3/M4 | 在 |
| A2-S | 33.6768 | +4.6347 | 34.9702 | 37.3276 @ ep5 | 机制门 M1 | 在 |
| A2-R | 33.9502 | +4.9081 | 36.3491 | 38.4943 @ ep5 | 机制门 M3（7/10 vs 门槛 8/10） | 在 |

**关键事实**：

1. **三个臂在 ep1-6 窗口上都高于 B0**（+4.63 ~ +5.00）——即它们**全部会通过 C 式的
   「任务损害」早停检查**，被剪掉的原因**纯粹是机制门**（该门的预测效力在 81.2–81.3 未获验证）。
   因此它们属于 §81.3 定义的「**成熟判定未获得**」类别，而不是「方法已被否定」。
2. **三个臂的 `epoch_6.pth` 都在**，且配置快照仍在 `me_rssm/configs/…_semA{,2S,2R}[_24e].py`。
   ⇒ 若要做「训练时长验证臂」，**不需要重建实现**，续训即可。
3. **续训成本**：ep6→ep16 共 10 轮 ≈ **10.1 h**；ep6→ep24 共 18 轮 ≈ 18.1 h
   （按实测 60.4 min/epoch）。对照：从零新建臂到 ep16 需 16 轮 ≈ 16.1 h。**续训省 ~6 h**。
   续训必须保持 epoch 计数与 LR 日程连续（epoch-based 日程），否则与 B0 同窗不可比。
4. **优先级建议**（若用户要恢复）：**A2-R 优先**——它的 FAIL 条款是效力最低的
   （M3 7/10 vs 8/10，n=10 的组合计数），而 A/A2-S 的 M1 类条款偏离大（结构性问题）。
   若 A2-R 续训后在 ep12-16 仍不及 B0，则**反过来验证了机制门**；若反超，则证明既有
   归档属于过早淘汰。两种结果都有信息量。
5. **注意口径一致性**：续训的 val 必须继续走训练内 EvalHook（与 B0 同口径），
   不得与离线 harness 数字混算（§51.5 的历史 ~2.2 点只是口径观察，不是可加减的修正量）。
6. **幸存者偏差声明**：§81.2 的 31-run 校准只包含**被允许跑完**的 run；被 ep6 剪掉的臂
   没有成熟数据，无法进入该数据集。故 81.3 的结论应读作「在跑完的 run 中，早期窗口的
   预测效力如何」，不能反过来用于证明「被剪掉的臂一定错了」。

### 81.6 「机制没成立怎么会提升 AP」——两者在本项目的实际解耦关系（2026-10-12）

用户提问：机制都不过，AP 凭什么会提升？答案分两层。

**第一层：机制门与 AP 是两类不同的陈述，可以四象限解耦。**

| 机制门 | ep12-16 AP | 能写进论文的话 | 本项目实例 |
| --- | --- | --- | --- |
| 过 | 提升 | 可主张机制级贡献（仍需多 seed） | 尚无 |
| 过 | **不提升** | 「机制可测但未转化为检测收益」 | **DC-RSSM-2**：L1 对齐机制成立（P5 复算 learned 胜 8/10），检测收益不成立 |
| **不过** | 提升 | **不能**写机制 claim；必须重新归因（噪声 or 未命名机制） | A/A2 的早期领先（+4.6~+5.0）属此类，但成熟窗口未测 |
| 不过 | 不提升 | 方向止损 + 负结果归因 | （多数被剪臂） |

⇒ 「机制没成立 ⇒ AP 不可能提升」**不是有效推理**；反过来「AP 早期领先 ⇒ 机制不过也无妨」同样无效。
两个方向在本仓库都已有实例。

**第二层：机制门 FAIL 的三种含义必须分开，它们的后续动作完全不同。**

1. **机制真的不在**：AP 若仍提升，只能来自噪声或另一条未被命名的路径
   （例：辅助监督作为**正则项**改变训练动力学，而非让 z 携带对象信息）。
   此时结论可以成立，但**机制 claim 必须改写**，不能沿用预登记那句话。
2. **机制在，但探针测不到**：如探针读采样 z 而非部署均值 z、判据阈值定错、
   n 太小（A2-R 的 M3 7/10 vs 8/10、15/20 vs 16/20 都是掷硬币级别）。
   此时 FAIL 是**仪器问题**，不构成方向证据。
3. **机制在，但尚未学出来**（训练时长问题）：只能靠「跑满成熟窗口」排除，
   这正是 §81.3 结论梯度存在的理由。

**第三层：单 seed 的 AP 提升本身不足以作为替代证据。**
直接反例：**MAI 同臂 seed0 +0.8434、seed1 −2.5063**（对各自同 seed 基线，ep12-16），
同一方法在不同 seed 上符号相反。§81.2 的 seed σ=1.35 与之互证：
±1 点级别的单 seed「提升」在本项目历史上**复现不了**。

**因此，机制门不是 AP 的门槛，而是「哪句话能写」的门槛。**
- AP 是**结果**，机制是**解释**；
- 门槛 FAIL 且 AP 提升时，能挽救的是结果，不能自动挽救解释；
- 要主张「z 携带对象判别信息并进入检测决策」这条论文主线，
  **机制证据与成熟 AP 证据缺一不可**，且必须在多 seed 上一致。

---

## 82. 停止规则修订与 A2-R 续训排程（2026-10-12，用户批准的两项动作）

### 82.1 动作一：C1 的 ep14 机制门降级为归因诊断

按附录 B 执行（`docs/rssm_c_validity_update_prereg_20261012.md`）。修订时 C1 在 ep5，
work dir 仅有 `epoch_2.pth`/`epoch_4.pth`，**无任何 ep12-16 val 行与 ep14 产物** ⇒ 属
预数据冻结修正，非事后放宽门槛。修订后：ep12-16 一律跑满 n=5，ep14 机制读数只做归因，
剪枝权移到 ep16→24。C1 的 ep6 任务损害条款不变（当时 +4.26，未触发）。

### 82.2 优先级声明（用户 2026-10-12）

> **最重要的目标是提升 AP。即使机制未成立，只要有大幅 AP 提升，也接受该结果。**

落到记录口径：
- **AP 是决策主指标**，机制门是归因/写作工具，不是 AP 的前置条件；
- 机制 FAIL + AP 提升 ⇒ **结果成立**，按 §81.6 重新归因（噪声 / 探针 / 未命名机制），
  **不得用机制读数否决性能结果**；
- **硬约束只在写作**：论文 claim 必须等于实际成立的解释；
- 「大幅」以 **ep12-16 同 seed 配对 Δ ≥ +2.0 且多 seed 方向一致**为准
  （单 seed ±1 点在本项目复现不了：MAI seed0 +0.8434 / seed1 −2.5063）。

### 82.3 动作二：A2-R 续训（ep6→ep16）已排入队列

- 依据 §81.5：三个被 ep6 剪掉的臂都留有 `epoch_6.pth`，**续训 10 轮（~10.1 h）**即可拿到
  成熟窗口，比从零重建到 ep16（16 轮 ~16.1 h）省 ~6 h。
- **选 A2-R 优先**：其 FAIL 条款效力最低（ep6 M3 7/10 vs 门槛 8/10）。
- **接线等价性已核对**（硬问题类）：当前代码 `mmdet3d/models/detectors/R4Det.py` 对
  `obs_points` 的传递**仅在 fusion 声明 `accepts_obs_valid` 时生效**，A2-R 的
  `SemanticReadoutRSSMFusion` 不声明 ⇒ 调用签名与本臂原始运行时逐字节一致；
  `me_rssm/__init__.py` 仍注册该类；repo 配置与运行目录快照的差异只是 mmcv 落盘的
  flattened 形式，不是语义差异。
- **队列事实**：`me_rssm/sanity/queue_semA2R_resume.sh` 于 **14:02:26 武装**，等待 C1 退出
  （预计 10-13 约 09:15-10:15），随后 GPU 5-7 排空 → 补 C1 的 ep14 产物（若训练中探针 OOM）
  → pre-flight（fusion 类型/权重存在/幂等）→ 以 `--resume-from epoch_6.pth --seed 0
  --deterministic` 续训（**保持 24e 配置以延续 CosineAnnealing 日程与 epoch 计数**）
  → **ep16 val 后停**（不跑 ep17-24，省 8 h）→ 三口径报告。
- 产物：`/tmp/semA2R_resume_16e.log`；`<A2R_WD>/queue_semA2R_resume.log`、
  `merged_val_curve.log.json`、`merged_val_curve_report.json`。
- **口径声明**：A2-R 续训的 val 仍走训练内 EvalHook（与 B0 同口径），不与离线 harness
  数字混算；全轮峰值的作用域是 **ep1-16**（ep17-24 未跑），记录时必须写明。


### 82.4 为什么不应只补 A2-R：选择依据纠偏（2026-10-12）

用户追问为何只续 A2-R、不补其他 ep6 臂。本次核对 §75.12 与三个 ep6 checkpoint；
A/A2-S/A2-R 的 `epoch_6.pth` 均在。仅修正研究选择理由，**未新增训练/队列、未改运行脚本**。

**事实纠正**：§81.5/§82.3 和此前回复强调 A2-R 的 M3 接近阈值，却遗漏该臂同时有
M1 与 M4 donor FAIL。A2-R **并非仅差一个 M3 样本**，也没有充分证据表明其最终 AP
比 A、A2-S 更有希望。机制门距离不得再被用作 AP 前景排序。

| 臂 | 实际 ep1–6 等权均值（n=6） | 全部已有 val 峰值（仅 ep1–6） | 实际 ep6 失败项 |
|---|---:|---|---|
| A | 34.0459 | 38.2203 @ ep3 | M1/M3/M4 |
| A2-S | 33.6768 | 37.3276 @ ep5 | M1/M4/M1-loc |
| A2-R | 33.9502 | 38.4943 @ ep5 | M1/M3/M4 |

三臂成熟 ep12–16 均未形成，不能由上述早期均值/峰值推断成熟名次。详细曲线与逐类
读数仍见 §72/§74/§75，本次没有新增训练结果。

**更完整的补训建议**：C1 继续；已排队 A2-R 不因本次问答擅自变更。
对于最近三个有效且仅因早期机制门中断的臂，建议各完成一次原日程下 ep6 至 ep16
的训练时长验证（A2-R 后补 A，再补 A2-S，属于排程建议、未实际加入队列）。
每臂估约 10 小时，三臂总约 30 小时，实际受吞吐/验证耗时影响；不扫参、不同时改结构，
先不扩多 seed。若资源有限，至少补 A：它既是语义监督/分类残差的父方法，也是两条 A2
单变量改动的必要成熟对照。A2-R vs B0 只能评价整包方法，不能单独归因通道映射有效。
不能因 A2-R 成熟无增益就自动否定 A/A2-S，三者结构不同。

历史所有停在 ep6 的 run **不自动一律补训**：无效接线/泄漏/数值故障或不兼容日程应先
区分；成熟窗口已经测过的实验不重复。本次没有全面审核这些历史臂。
预算取舍可以决定先后，但没有证据时，不把“暂未排程”写成“已证明不值得跑”。

---

## 83. 可靠性感知前端融合臂 F1：立项、实现与排程（2026-10-12，未开训）

### 83.1 立项证据：历史上「concat → attention」替换的真实代价

`crossmodal_fusion_N4_2x4_24e_seed0` vs 同口径 B0（标准 `no2d_igdr` seed0），
**ep12-16 五轮等权均值（n=5）**：

| 指标 | crossmodal | B0 | Δ |
| --- | --- | --- | --- |
| Overall | 37.9789 | 39.2809 | **−1.3019** |
| Car (strict) | 45.7627 | 47.8399 | **−2.0772** |
| Truck (strict) | 31.4858 | 29.7246 | +1.7612 |
| Pedestrian (loose) | 25.6864 | 30.9277 | **−5.2413** |
| Cyclist (loose) | 48.9808 | 48.6313 | +0.3495 |

全轮峰值：crossmodal **39.2835 @ ep16**；B0 **40.3853 @ ep14**。

**口径边界**：该模块不是强意义的 Transformer cross-attention（两模态各自 channel
mean/max → 空间 attention → 互乘 → concat+conv），**没有可靠性估计、没有缺测处理**。
⇒ 可以说「不能盲目用 attention 替换 concat」；**不能**说「所有高级 fusion 无效」，
也**不能**说「雷达缺测就是它失败的原因」（未测）。

### 83.2 F1 的唯一研究因素（公式冻结）

```
occ   = 逐 BEV 格雷达点占用（由 *输入点云* 构造；GT-free）
alpha = 1 + tanh( gate([occ, |radar|_mean_c, |radar|_max_c]) )    # ∈ (0, 2)
out   = fusion_conv( cat([img_bev, alpha * radar_bev]) )          # conv 与基线同形
```

- **单变量契约已机械核验**：臂配置与主线配置的差异键 **恰好 `['RCFusion']`**
  （`temporal_fusion` 仍是标准 `MotionAlignedRSSMFusion`）。
- **容量**：唯一新增 **593** 参数（基线 1,474,816 → **+0.040%**）；`fusion_conv` 同形。
- **零初始化** ⇒ step 0 `alpha ≡ 1` ⇒ 与 `ConcatConvFusion` **逐位相同**（单测断言）。
- **相机通路不被门控**；无新 loss、无 KL 改动、无语义头、无 Doppler 门。
- **布局**：源码确证 `PointPillarsScatter` 按 `coors(b,z,y,x)` → `canvas.view(ny,nx)`，
  行=y、列=x ⇒ `valid_layout='yx'`（布局错误会静默镜像掩码，故列为硬门）。
- 预登记：`docs/rssm_reliability_fusion_prereg_20261012.md`。

### 83.3 工程验证（已完成）

- 实现 `me_rssm/reliability_fusion_rssm.py`（`ReliabilityGatedConcatFusion`）；
  接线 `R4Det.py` 仅在 fusion 声明 `accepts_obs_valid` 时传 `obs_points`（其它臂签名不变）。
- 单测 `me_rssm/sanity/test_reliability_fusion.py`：**8 passed**
  （逐位恒等、无点回退、门活性与 α∈(0,2)、角落映射与 yx/xy 转置、容量增量、接线守卫）。
  与 C1 单测合跑 **22 passed**。
- S0 判定器 `relfus_s0_verdict.py`（纯产物、不占 GPU）自检通过：
  合成冒烟目录下 S0-2/S0-4 为 True、S0-1/S0-3 正确为 False。

### 83.4 排程（用户指令：接在现有队列之后）

- **队列链**：C1（进行中，ep5）→ **A2-R 续训 ep6→ep16**（14:02 武装）→ **F1**（15:13:16 武装）。
- F1 队列动作：等 A2-R 队列 `DONE` 且其进程退出 → GPU 5-7 排空 → **450-iter 冒烟** →
  `S0_ALL_PASS` 才继续（否则停）→ **24e seed0** → ep6 任务损害检查 → ep12-16 三口径报告。
- 预计：A2-R 约 10-13 20:30 结束；F1 若 S0 通过，24e 约 24.2 h ⇒ **约 10-14 晚间**出成熟读数。
- 所有 GPU 工作仍只用 **5/6/7**，串行，不并发抢卡。

### 83.5 判定层（遵循附录 B 政策）

1. **ep6 只查任务损害**（低于 B0 同窗 29.0420 > 2.0 点才停）；机制读数不裁决。
2. **ep12-16 成熟门**（n=5）：Overall ≥ 40.00；Car ≥ 47.34、Truck ≥ 29.22、Ped ≥ 29.93、
   Cyc ≥ 47.63；任一类 ≤ B0 同窗 −2.0 ⇒ 止损。**ep14 不设剪枝门。**
3. 必报三口径（区间均值写跨度/n、全轮峰值值+epoch、best saved 值+epoch）。
4. **无论 AP 正负都要报 α 机制读数**（占用格 vs 空格的 α 分布、α 与占空比关系），
   否则「可靠性感知」只是命名；该探针尚未实现（ep14 前补齐，不阻塞开训）。
5. 失败时只能归因到「每格标量门不足」或「前端不是瓶颈」，**不得外推**为
   「所有可靠性融合无效」。

---

## 84. C1 的 ep6 与历史强 run 的 ep6 对照（2026-10-12，CPU 只读）

C1 当前 ep1-6 区间均值 **33.8077**、ep6 单点 **39.9821**。为避免跨日程比较（6e/12e 的
`ped_centerhead_*` 系列在 ep6 已接近其终点，不属同口径），本节只与**同族 24e** run 比较。

### 84.1 同族（`fgfull_N4_*24e*`）ep6 对照，按 ep6 降序

| run | ep1-6 均值 | **ep6** | ep12-16 成熟均值 |
| --- | --- | --- | --- |
| mai_seed1 | 32.662 | **42.773** | 39.412 |
| mci_seed1 | 32.404 | **42.136** | **41.122** |
| mci_seed0 | 33.463 | **40.661** | 40.106 |
| mci_v2_seed0 | 33.166 | **40.328** | 39.379 |
| **c1valid_seed0（C1）** | **33.808** | **39.982** | 待补 |
| fgfull_N4_seed0（另一主线） | 31.468 | 39.017 | 40.019 |
| dcrssm2_seed0 | 33.096 | 38.760 | 38.535 |
| no2d_igdr_seed2 | 30.683 | 37.792 | 40.121 |
| no2d_igdr_seed1 | 30.741 | 37.325 | **41.918**（全族最高） |
| mai_seed2 | 31.041 | 37.319 | 40.022 |
| dcrssm1_seed0 | 32.182 | 37.267 | 39.217 |
| semA2R_seed0 | 33.950 | 36.349 | 未跑 |
| mai_seed0 | 30.176 | 35.929 | 40.124 |
| semA_seed0 | 34.046 | 35.761 | 未跑 |
| semA2S_seed0 | 33.677 | 34.970 | 未跑 |
| **no2d_igdr_seed0（B0 基线）** | **29.042** | **34.024** | 39.281 |

**C1 的名次**：ep6 单点 **全族第 5**（18 个）；ep1-6 区间均值 **全族第 3**。

### 84.2 同族内「ep6 → 成熟」关系：早期名次基本无预测力

- ep6 ≥ 40 的四个 run，成熟值分别为 39.412 / **41.122** / 40.106 / 39.379（跨度 1.7）。
- 成熟最高的两个 run（`no2d_igdr_seed1` 41.918、`no2d_igdr_seed2` 40.121）**ep6 只排第 10/8**。
- ep6 第二高的 `mai_seed1`（42.773）成熟仅 39.412，**低于基线 seed2**。
- ⇒ 在**控制架构与日程**的同族内部，§81 的结论仍然成立：**ep6 不能用来排成熟名次**。
  C1 的 ep6 ≈ 40 落在「历史成熟值 39.4–41.1」这一带，**跨越 40.00 门槛线两侧**，
  该读数既不能证明会过门，也不能证明会失败。

### 84.3 必须点明的门槛校准问题（影响后续所有判读）

冻结门槛线（Overall ≥ **40.00**、四类非劣）是按 **B0 seed0** 的成熟均值 **39.2809** 校准的。
但同配方三 seed 的成熟均值为 **40.4400**（std **1.347**）：seed0 39.281 / seed1 **41.918** /
seed2 40.121。

⇒ **seed0 是三 seed 里最弱的一个**。因此：
1. C1 过 40.00 只等于「超过 seed0 这一个具体结果」，**不等于超过基线家族**；
2. 要与三 seed 基线均值区分开，C1 需要达到约 **41.8**（均值 +1σ）量级；
3. 同理，C1 目前的「同 seed0 配对 +4.77」也受此影响——**跨 seed 平均后早期优势会缩水**
   （三 seed 早期均值 30.155，对 C1 33.808 ⇒ Δ +3.65，而非 +4.77）。
4. 该缺口不影响冻结门槛的合法性（同 seed 配对是既定设计），但**任何「超过基线」的表述
   必须同时给出三 seed 均值对照**，不能只用 seed0。

### 84.4 与「成熟强 run」的逐轮对照（用户 2026-10-12 追加：拿 C1 的 ep1-6 对比好 run 的 ep6）

§84.1 把 C1 放在**全族**里排名；本节只回答用户提出的窄问题——**C1 与「真正跑好了的那几次」
同轮次对照如何**。

- **口径**：同族 `fgfull_N4_*24e*`，成熟窗口 ep12-16 等权均值 ≥ **40.00** 的 run 共 **7 个**
  （`std_seed1` 41.9185、`std_mci_seed1` 41.1218、`std_mai_seed0` 40.1242、`std_seed2` 40.1207、
  `std_mci_seed0` 40.1061、`std_mai_seed2` 40.0222、`2x4_24e_seed0` 40.0187），下称 **elite**。
- 指标 `pts_bbox/KITTI/Overall_3D_moderate`；C1 为**同一 epoch 单点**对照（非窗口均值）。
- 工具：`me_rssm/sanity/ep6_elite_compare.py`（CPU 只读，可重跑）。

| run | ep1 | ep2 | ep3 | ep4 | ep5 | **ep6** | ep1-6 均值 | ep12-16 均值 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| std_seed1 | 17.98 | 25.18 | 33.31 | 34.43 | 36.22 | 37.33 | 30.7412 | **41.9185** |
| std_mci_seed1 | 19.48 | 29.26 | 28.86 | 36.17 | 38.51 | **42.14** | 32.4044 | 41.1218 |
| std_mai_seed0 | 20.18 | 27.82 | 32.08 | 31.67 | 33.38 | 35.93 | 30.1764 | 40.1242 |
| std_seed2 | 18.64 | 26.39 | 33.41 | 33.77 | 34.09 | 37.79 | 30.6828 | 40.1207 |
| std_mci_seed0 | 19.77 | 29.34 | 33.49 | 40.43 | 37.08 | 40.66 | 33.4628 | 40.1061 |
| std_mai_seed2 | 17.37 | 26.83 | 33.95 | 34.64 | 36.14 | 37.32 | 31.0411 | 40.0222 |
| 2x4_24e_seed0 | 17.33 | 27.77 | 34.64 | 33.69 | 36.36 | 39.02 | 31.4678 | 40.0187 |
| **C1（std_c1valid_seed0）** | **17.41** | **30.98** | **37.69** | **38.54** | **38.25** | **39.98** | **33.8077** | 未产生 |

**C1 − 各 elite run（同 epoch）**

| delta vs | ep1 | ep2 | ep3 | ep4 | ep5 | ep6 |
| --- | --- | --- | --- | --- | --- | --- |
| std_seed1 | −0.57 | +5.80 | +4.38 | +4.11 | +2.03 | +2.66 |
| std_mci_seed1 | −2.07 | +1.72 | +8.83 | +2.37 | −0.27 | **−2.15** |
| std_mai_seed0 | −2.77 | +3.16 | +5.61 | +6.87 | +4.87 | +4.05 |
| std_seed2 | −1.23 | +4.59 | +4.28 | +4.77 | +4.16 | +2.19 |
| std_mci_seed0 | −2.36 | +1.64 | +4.20 | −1.89 | +1.17 | −0.68 |
| std_mai_seed2 | +0.04 | +4.15 | +3.74 | +3.90 | +2.11 | +2.66 |
| 2x4_24e_seed0 | +0.08 | +3.21 | +3.05 | +4.85 | +1.89 | +0.96 |
| **elite 均值** | 18.68 | 27.51 | 32.82 | 34.97 | 35.97 | 38.60 |
| **C1 − elite 均值** | **−1.27** | **+3.47** | **+4.87** | **+3.57** | **+2.28** | **+1.38** |

**读数（只描述，不判成功）**

1. C1 的 ep6 **39.9821 在 [C1+elite] 8 个里排第 3**（elite 内部 ep6 跨度 35.93–42.14），
   只低于 `std_mci_seed1` 42.14 与 `std_mci_seed0` 40.66。
2. **C1 的 ep1-6 窗口均值 33.8077 是 elite 里最高的**（elite 跨度 30.1764–33.4628，
   elite 均值 31.4252，Δ **+2.3825**）。
3. **轨迹形状不同**：C1 起点最慢（ep1 17.41，比 elite 均值低 1.27，7 个 elite 里只有
   `std_mai_seed2`/`2x4_24e_seed0` 与之相当），**ep3 起反超并保持**（+4.87 / +3.57 / +2.28 / +1.38）。
4. **但这组对照没有预测力**：在 elite 内部，**Spearman(ep6, ep12-16) = −0.036**（n=7，
   成熟跨度仅 1.90）。elite 里 ep6 最低的 `std_seed1`（37.33）反而是成熟最高（41.9185）；
   ep6 最高的 `std_mci_seed1`（42.14）成熟排第 2。⇒ 与 §84.2 一致：**同轮次比较只能证明
   「C1 没在早期掉队」，不能排成熟名次**。
5. **邻域带**（同族 ep6 ∈ [37.48, 42.48]，n=7）：成熟结果 38.5345 / 均值 39.8608 / 41.1218，
   std **0.7317** ⇒ 与 C1 同档 ep6 的历史 run，其成熟值**横跨 40.00 门槛两侧**。

### 84.5 ep6 的逐类读数与「ep6→成熟」典型漂移

C1 ep6 逐类（对比 B0 seed0 同轮 + 冻结成熟非劣线）：

| 类别 | C1 ep6 | B0-seed0 ep6 | Δ(同轮) | 冻结成熟非劣线 | ep6 是否已在线以上 |
| --- | --- | --- | --- | --- | --- |
| Car | **54.57** | 43.28 | +11.29 | 47.34 | **是** |
| Truck | **32.65** | 21.93 | +10.72 | 29.22 | **是** |
| Pedestrian | 26.81 | 27.12 | −0.31 | 29.93 | 否 |
| Cyclist | 45.90 | 43.77 | +2.13 | 47.63 | 否 |

同族「ep6 单点 → ep12-16 窗口均值」的典型漂移（有双窗口的 14 个 run）：

| 类别 | 均值 | std | min | max |
| --- | --- | --- | --- | --- |
| Car | +2.705 | 2.681 | −2.094 | +7.411 |
| Truck | +3.633 | 4.117 | −5.333 | +9.242 |
| Pedestrian | **−0.494** | 3.446 | −7.574 | +3.805 |
| Cyclist | +0.299 | 4.332 | −6.404 | +6.311 |
| Overall | **+1.536** | 2.412 | −3.361 | +5.257 |

⇒ C1 在 ep6 的 Ped/Cyc 低于成熟非劣线**不构成停止信号**（同族成熟漂移的 std 是 3.4/4.3，
远大于该差值）；同理 Car/Truck 大幅超出也**不能**据此外推成熟值。

**两种外推（都只是校准，不是结果）**：

- A) 同族回归（n=14）：`mature = 44.6498 − 0.1493 × ep1-6` → C1 预测 **39.6036**
  （1σ 带 38.8239–40.3833）。**同族内早期均值的回归斜率是负的**（−0.149），
  再次说明家族内早期分数不含正向信息。
- B) ep6 + 典型漂移：39.9821 + 1.536（std 2.412）→ **41.5178**。
- A/B 两个带**都横跨 40.00 门槛**。⇒ **C1 的成熟结论只能等 ep12-16 窗口本身**（n=5 未满前
  不得引用「区间均值」口径）。

### 84.6 门槛再校准：Ped 非劣线在同族历史里只有 1/14 可达（记录，不自行修订）

把冻结的四类非劣线放到同族 14 个跑满的 run 上做可达性核对（成熟窗口，n=5）：

| 类别 | 冻结线 | = B0-seed0 − | 同族可达 | 备注 |
| --- | --- | --- | --- | --- |
| Car | 47.34 | 0.50 | **14/14** | 事实上不构成筛选 |
| Truck | 29.22 | 0.50 | 13/14 | 较宽松 |
| Pedestrian | **29.93** | 1.00 | **1/14** | 唯一达标者就是 B0 seed0 自己（30.93，**全族最高 Ped**） |
| Cyclist | 47.63 | 1.00 | 8/14 | 中等 |

**seed0 同口径下，成熟 Overall ≥ 40.00 的三个臂全部不过 Ped 线**：

| seed0 臂 | Overall | Car | Truck | Ped | Cyc |
| --- | --- | --- | --- | --- | --- |
| std_mai_seed0 | 40.124 | 54.93 ✓ | 30.84 ✓ | 27.53 **✗** | 47.19 **✗** |
| std_mci_seed0 | 40.106 | 53.84 ✓ | 32.43 ✓ | 26.86 **✗** | 47.30 **✗** |
| 2x4_24e_seed0 | 40.019 | 53.96 ✓ | 31.34 ✓ | 29.73 **✗** | 45.05 **✗** |

⇒ 事实陈述：**「Overall ≥ 40.00 且四类全过」在同族历史上从未被任何臂达成**；Ped 线实际等价于
「不得低于全族历史最高 Ped 减 1.0」。这不是 C1 特有的问题，而是家族性的门槛标定问题。

**处理方式（不自行改冻结预登记）**：本节仅记录证据。C1 仍按冻结门槛判；若用户决定修订，
必须在 **ep12-16 窗口尚未产生**（当前 ep7，窗口未开始）时做出，且需在预登记中写明修订理由、
新旧条款与代价（§81.4 同规则）。**任何情况下 Overall 增益都不能单独掩盖某一类的实际下降**——
修订只能改变「判非劣」的标定，不能改变逐类报告义务。


---

## 85. 现役训练队列合规审查与 F1 对照核对（2026-10-12，只读审查）

### 85.1 范围、现场与性能口径

用户要求检查队列脚本是否符合既定要求，并辨明 fusion 更新对照是 43 点还是历史 40.88。
审查对象：`queue_semA2R_resume.sh`、`queue_relfus.sh`、`watch_c1valid_ep6.sh`、
`tools/dist_train.sh`、三臂解析配置、F1 S0 判定器与共用曲线报告器。
**未修改产品代码/在跑脚本、未启动/停止/重排任何训练、未改变冻结阈值。** 本节仅归档审查。

- 快照时点：**2026-10-12 16:56:39 UTC**。C1 已有 ep1–7 val；最近训练日志为
  ep8、iter 1600。A2-R 与 F1 队列均仍在等待，没有新训练 val。
- 实际意图链：C1 **进程退出后**，A2-R 从自身 ep6 恢复至 ep16，再轮到 F1 的
  450-iter smoke / S0 / 正式训练。**不是 C1 ep16 一到即自动接棒**：C1 watcher 不负责停训，
  ep16→24 是人工决策；若无人工干预，C1 的 24e 配置继续运行。
- 队列子进程环境已核对：Python 为 `r4det/bin/python`，`CUDA_VISIBLE_DEVICES=5,6,7`。
  16:34 UTC 的 GPU 5/6/7 显存使用为 20,836/20,770/21,282 MiB，C1 持卡。

**C1 本次快照两主口径**：默认 ep12–16 未产生，不能作成熟判断；实际可用区间
**ep1–7（n=7）等权均值 34.6662**；全轮峰值
**39.9821 @ ep6**（范围为截至快照的全部 val）；
best saved **39.9821 @ ep6**。
A2-R 尚无恢复阶段数据（既有 ep1–6 完整记录见 §75）；F1 的区间均值、全轮峰值均为
**N/A，未开训**，不能以已实现/已排队代替实验通过。

C1 截至快照的完整逐 epoch 构成（正式历史继续追加，不以本摘要代替明细）：

| epoch | Overall 3D | Overall BEV | Car strict | Truck strict | Ped loose | Cyc loose |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 17.4098 | 25.6709 | 30.6214 | 8.1517 | 13.2886 | 17.5775 |
| 2 | 30.9781 | 36.8775 | 39.7862 | 25.4070 | 24.3206 | 34.3987 |
| 3 | 37.6897 | 44.6725 | 52.1161 | 29.1205 | 22.8955 | 46.6267 |
| 4 | 38.5394 | 46.8164 | 53.1686 | 27.7938 | 27.4874 | 45.7077 |
| 5 | 38.2473 | 46.4945 | 49.2735 | 27.4927 | 26.9566 | 49.2663 |
| 6 | 39.9821 | 47.0580 | 54.5678 | 32.6541 | 26.8100 | 45.8965 |
| 7 | 39.8167 | 48.5793 | 51.2623 | 35.0172 | 30.5549 | 42.4322 |

### 85.2 已通过的配置与接线检查

1. 三臂均解析为 24e `EpochBasedRunner`，CosineAnnealing 日程、KL/数据/检测头/每轮
   EvalHook 不因队列而更改；seed0、deterministic、3-rank、GPU5/6/7。
2. F1 相对当前标准 no2d_igdr 的**模型差异键恰为 `['RCFusion']`**；
   `temporal_fusion` 仍为标准 `MotionAlignedRSSMFusion`。**没有把 C1/A2-R 叠进 F1**。
   除导入注册与临时 `_cfg_dir` 元数据，非模型训练设置相同。
3. F1 初始化仍为公共 `checkpoints/pretrained_tj4d.pth`，不是从 43.1449 或 40.88 的检测
   权重继续训。新模块是保留原 concat+conv 并对 radar 半部添加 α 门，不是 Transformer 替换。
4. A2-R 的 `epoch_6.pth` 实存；CPU 读取 meta 为 epoch6 / iter11412，含 optimizer
   （490 个 state 条目）。命令确用 `--resume-from`，未把 `max_epochs` 改成16；
   因而配置层面保持原24e LR日程与恢复计数。**尚未真正执行恢复，运行成功不在本次证明范围。**
5. 与 seed0/seed1 历史配置快照核对，骨架相同；`model.meta_info` 是 figures_path/project_name
   运行元数据，`data.val.samples_per_gpu` 的缺项对应 loader 的 `pop`，不是新增网络变量。

### 85.3 未通过的执行控制（不能宣称全合规）

**P1：F1 S0 会把“未工作的 gate”判成通过。**
`relfus_s0_verdict.py:86-99` 把 gate 的所有 weight 混合求非零统计；隐藏层本来就随机非零。
CPU 构造一个**末层仍全零、α 恒等1、完全未更新的新建模块 checkpoint**，并提供 iter450
有限 loss 记录，实际返回 **S0_3=True、S0_ALL_PASS=True、exit0**。
这不证明真实冒烟会失败，而是证明当前判定器不能排除接线/梯度导致 gate 闲置。
应检查零初始化输出层的实际更新或 α 相对1的行为变化，不能用隐藏层非零冒充已学习。

**P1：排空等待超时不阻止启动。**
`queue_semA2R_resume.sh:31-38` 与 `queue_relfus.sh:39-46` 都等最多240分钟，循环完只打印
显存读数，**没有“仍占用则退出”**。正常链顺序正确，但超时/查询失败时不是 fail-closed；
后续会继续进入 GPU 工作。F1 冒烟结束也仅等待文件+固定60秒，没有校验启动进程已正常退出。
应以受控进程退出和GPU状态共同放行，超时/失败不得靠继续执行来处理。

**P1：F1 ep6 FAIL 不执行预登记的停并复查。**
`queue_relfus.sh:81-96` 的 FAIL 分支只写 `flagged for review`，然后直接等ep16。
安全 shell 模拟（外部启动/信号/GPU全部替换为函数）令早期检查失败，实际仍走到
`ep16_window_check` 与 `DONE`，返回0。C1 watcher 的 ep6 分支同样只报告，
但当前 C1 的真实 ep6 是 PASS（+4.7657），**未实际触发这项违约**。

**P1：A2-R 异常/报告失败也能写成功 DONE。**
`queue_semA2R_resume.sh:65-79` 捕获异常只 break，随后仍写“job stopped after ep16”；
报告器非零退出也未检查（脚本只有 `set -u`）。安全模拟“训练异常 + 报告返回1”时，
仍写 **DONE、exit0**。下游 F1 用这个 DONE 放行，故它不是可靠的成功完成凭据。
同脚本 `pkill -f '[s]emA2R_24e'` 是配置名级匹配，不是本次启动PID/进程组，
且停训靠固定180秒等待，不保证 ep16 val与 checkpoint均完整落盘。
修复须明确 SUCCESS/FAILED/INCOMPLETE；记录自有PID/进程组、等待退出、核验目标val/权重/报告。
幂等分支已有ep16权重时直接exit0却不写DONE，也可能让F1永久等待，不能只依赖字符串。

**P1：C1 成熟报告存在“权重先到、val未到”的竞态。**
`watch_c1valid_ep6.sh:65-71` 看见 epoch16权重即可退出等待，不要求ep16 val。
本次真实训练 ep6 顺序为 **15:00:30 保存 checkpoint，15:12:28 才写 val**，相隔约12分钟。
因此在ep16的同类顺序下可能提前生成 n=4 报告后不再更新。应等待完整 ep12–16
**五条JSON val**，并验证报告成功，而非把权重存在视作验证完成。
其ep14 GPU探针还会与C1同时占GPU5，可能OOM；这属于归因产物缺口，不能借此否决AP或截断窗口。

**P2：早期窗口完整性与非有限 loss 检查仍有缺口。**
`c1valid_curve_check.py:179-186` 未要求双方ep1–6均完整；CPU反例 **arm n=1、B0 n=6**
仍 `early_check.pass=True`。成熟门已有本臂 `n_complete`，不要误写“成熟门也无n检查”。
F1 S0 的有限性检查只识别NaN、不识别Inf，并从最后一行挑loss键；CPU反例 loss=Inf
仍 **S0_1=True、S0_ALL_PASS=True**。需检查所有相关行/字段的 `isfinite`。

以上反例只证明控制与判定缺陷，**不等于已观测到当前训练结果无效**。
未改阈值、未修改网络；修复控制缺陷不需要再立网络训练臂或增加机制否决门。

### 85.4 Fusion 到底以谁为对照

**架构基底是跑出43.1449的标准 no2d_igdr 家族；当前F1 seed0的配对数值对照是同族B0 seed0，
不是历史40.88，也不是拿seed1的最高峰直接裁决seed0新臂。**

| 用途 | 对照run | 固定ep12–16均值（n=5） | 全轮峰值 | 权重边界 |
| --- | --- | --- | --- | --- |
| 当前F1 seed0配对主对照 | no2d_igdr seed0 | 39.2809 | 40.3853 @ ep14 | ep14权重在 |
| 同族seed1/43点参考，后续F1 seed1的配对对照 | no2d_igdr seed1 | 41.9185 | 43.1449 @ ep15 | ep15未落盘，不可作为起训权重 |
| 历史旧栈峰值，非当前F1预登记对照 | run10_headv2_multiseed seed2 | 本次不复用旧窗口 | 40.88 @ ep14（历史记录精度） | 不作为当前初始化 |

以上seed0/seed1主数字本次重新从原始JSON val核算。
43.1449是“同族已达到的全轮上限”参考，不是模型模块名，也不是必须从它的权重继续训。
后续若要回答“fusion是否能在43点那一seed上再提高”，应补F1 seed1：窗口对41.9185、
全轮峰值对43.1449；最终做多seed配对，不能拿seed0新臂与seed1挑峰混作因果证据。
§84 的门槛校准问题仍在：Overall≥40.00只代表seed0初筛，不能改称超过整个基线家族；
Ped/Cyc非劣线与止损须透明报告，**本审查未擅自修订冻结线**。

### 85.5 验证命令、结果与下一步

- 四脚本 `bash -n`：**全部PASS**（仅语法，不代表业务规则通过）。
- 当前HEAD下 `test_reliability_fusion.py` + `test_validity_gain_rssm.py`：**22 passed in10.43s**。
  使用 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`，CUDA_VISIBLE_DEVICES=5仅满足导入时CUDA可用性断言，
  测试张量/配置/合成checkpoint均在CPU；未启动GPU推理或训练。
  初次默认插件加载因 `No module named 'omegaconf.vendor'` 失败、隐藏所有GPU后因
  `local attention needs cuda!` 导入失败；最终按上述环境重跑成功，不把环境错误记作网络失败。
- 临时目录CPU判定反例3项、只执行控制尾部的安全shell模拟2项：均复现上述缺陷；
  **不是新增pytest回归，也未改动真实run的产物**。
- F1真实450iter冒烟/S0与A2-R恢复：**PENDING，排队未运行**；不借用合成产物宣称通过。
- 结论：**模型配置与同seed对照设计基本符合；队列控制/S0不能判全通过。**
  下一步应窄修上述控制缺陷，使用版本化新脚本安全替换/重新武装需要更新的watch/queue，
  不原地修改运行中的bash、不动C1权重/数据/阈值；本次未执行这些修复或任何调度变更。

## 86. C1 成熟窗口判定：未过冻结门，且是非对称的跨类重分配（2026-10-13，CPU 只读）

来源：`/data/lurui/work_dirs/fgfull_N4_no2d_igdr_2x4_24e_c1valid_seed0`（训练内 EvalHook）
vs 同 seed0 基线 `fgfull_N4_no2d_igdr_2x4_24e_seed0`；判定用冻结工具
`me_rssm/sanity/c1valid_curve_check.py --window 12-16`（本次重跑以 n=5 覆盖旧产物）。

### 86.1 三口径与任务门（ep12-16 区间均值，n=5，span 12-16 完整）

| 指标 | C1 | B0 seed0 | Δ | 冻结线 | 判定 |
| --- | --- | --- | --- | --- | --- |
| **Overall** | **39.8303** | 39.2809 | **+0.5494** | ≥ 40.00 | **FAIL**（差 0.1697） |
| Car | **54.1636** | 47.8399 | **+6.3237** | ≥ 47.34 | PASS |
| Truck | **36.4397** | 29.7246 | **+6.7151** | ≥ 29.22 | PASS |
| Pedestrian | 26.0396 | 30.9277 | **−4.8880** | ≥ 29.93 | **FAIL**（且 < 28.9277 = B0−2.0 ⇒ **触发止损**） |
| Cyclist | 42.6783 | 48.6313 | **−5.9530** | ≥ 47.63 | **FAIL**（且 < 46.6313 ⇒ **触发止损**） |

- 全轮峰值 **40.6389 @ ep16**；best saved **40.6389 @ ep16**（saved = [2,4,6,8,10,12,14,16]）。
- `task_gate.all_pass = False`，`n_complete = True`（五条 val 齐全，判定合法）。
- 产物：`<C1_WD>/c1valid_ep16_window_check.json`（2026-10-13 重跑版；01:18:56 的旧版
  只有 n=4——Stage C 在 ep16 val 落盘前就触发了，属时序问题，数值本身无争议）。

### 86.2 逐 epoch 曲线（C1 vs B0 同轮，Overall）

| ep | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| C1 | 17.41 | 30.98 | 37.69 | 38.54 | 38.25 | 39.98 | 39.82 | 38.32 | **40.17** | 39.82 | 38.40 | 39.69 | **40.58** | 39.61 | 38.63 | **40.64** |
| B0 | 15.26 | 26.37 | 31.66 | 32.85 | 34.08 | 34.02 | 37.57 | 37.61 | 38.29 | 39.20 | 38.02 | 38.44 | 40.07 | 40.39 | 38.66 | 39.79 |
| Δ | +2.15 | +4.61 | +6.03 | +5.69 | +4.16 | +5.96 | +2.25 | +0.71 | +1.88 | +0.62 | +0.38 | +1.25 | +0.51 | −0.78 | −0.03 | +0.85 |

⇒ 早期领先（ep1-6 窗口 +4.77）在 ep7 起收窄到 ±1~2，ep14/15 出现单点负差——
与 §81 的「斜率 0.33–0.51、早期领先系统性回落」预测一致，**不是**新的意外现象。

### 86.3 结论口径（严格按可证事实）

1. **不构成 AP 成功**：主指标区间均值 39.8303 未过 40.00；且 Ped/Cyc 分别 −4.89/−5.95，
   越过 −2.0 止损线。按用户优先级（AP 为主），「机制不通但 AP 大幅提升也接受」的前提**未满足**。
2. **不满足「大幅」定义**：同 seed 配对 Overall **+0.5494** < +2.0；对照三 seed 基线均值
   **40.4400（σ 1.347，§84.3）**，C1 低 **0.61**，连「超过基线家族」都不成立。
3. **收益形状是跨类重分配而非整体提升**：Car +6.32 / Truck +6.72 换来 Ped −4.89 / Cyc −5.95。
   这与 C1 立项动机（缺测时保留相机/历史信息）**不一致**——它增强了强雷达类，牺牲了弱/缺测类。
4. **机制与任务的解耦**（§81.6 的又一例）：门控学到「有回波收缩、无回波放大」的反向偏置，
   却带来 Car/Truck 大涨与 Ped/Cyc 大跌。**因果归因待 ep14 探针**（§86.4），当前不得声称
   「收益/损失来自门控方向」。
5. **判读边界**：单 seed。不能说「重复训练也会这样」，也不能说「换个 seed 一定同样失败」；
   多 seed 未做，按既定顺序须在单 seed 通过成熟门之后才投入。

### 86.4 staged ep14 归因探针：失败根因与修复（在 C1 退出前完成，避免队列空转）

- **现象**：看门狗 01:18:56 完成 Stage C，但 Stage B 全链失败——`c1valid_ep14_dump.pkl`
  未产出，`probe_c_failure_enrichment.py` / `eval_dump_c.py` / `c1valid_ep6_verdict.py`
  级联 `FileNotFoundError`，无 `c1valid_ep14_mechanism_verdict.json`。
- **根因**（日志逐行定位）：`me_rssm/sanity/tool_c_masked_inference.py` 用裸 `Config` 构建模型，
  而 `R4Det.__init__`（`mmdet3d/models/detectors/R4Det.py:254`）无条件读
  `meta_info['figures_path']`；`meta_info` 由 `tools/train_vod.py:174` 在训练入口注入，
  工具没有 ⇒ `TypeError: R4Det: 'NoneType' object is not subscriptable`，模型构建即中止。
  与 C1 模型/权重/配置无关，是**工具侧接线缺陷**。
- **修复**：新增 `ensure_meta_info(cfg, config_path)`——缺失时注入
  `{'figures_path': /tmp/r4det_tool_c_figures, 'project_name': 配置文件名（与 train_vod 同构）}`，
  已有则原样保留；`main()` 在 `build_model` 前调用。新增 2 条单测
  （注入断言 + 不覆盖断言），`test_tool_c_masked_inference.py` **18 passed**。
- **补救路径（无人值守）**：`queue_semA2R_resume.sh` 的 safety net 条件为
  「无 verdict + 有 `epoch_14.pth` + 无 watchdog 存活」⇒ C1 退出、5/6/7 排空后自动重跑看门狗，
  ep14 归因产物无需人工排程。**本次修复必须在 C1 退出前落地**（已完成），否则 safety net 会重放同一 bug。
- **未验证项**：修复后的实跑仍待 GPU（当前 5/6/7 被 C1 占用；`locatt_ops` 有模块级
  `assert torch.cuda.is_available()`，CPU 无法构建该模型做端到端冒烟），单测只覆盖注入逻辑本身。

### 86.5 队列状态与剪枝决定（用户 2026-10-13 批准，已执行）

- C1 仍在训（2026-10-13 02:0x 在 ep17；配置 24e）。附录 B 规定剪枝权在 **ep16→24**。
  事实：主窗口已 n=5 完整、任务门 FAIL、两类触发止损；尾部 7 个 epoch（约 6 h）只能提供
  稳定性信息，翻盘到四类全过需 Ped **+3.89** / Cyc **+4.95**。
- 排队中：**A2-R 续训（ep6→ep16，约 10 h）→ F1（冒烟 → S0 → 24e）**，均等 C1 退出。
- **决定与执行**：用户 2026-10-13 批准剪枝。02:57:56 UTC 对 C1 进程组
  （PGID 1044850，`tools/dist_train.sh` → 3-rank DDP）发 SIGTERM，停点 **ep18 iter 1050/1902**；
  GPU 5/6/7 随后排空（21 MiB）。**权重与日志全部保留**（`epoch_2..16.pth`、`latest.pth`、
  `*.log.json`、两份判定 JSON），跑满的 16 个 epoch 是完整证据，ep17-18 的 val 不存在也**不追认**。
- 剪枝后不需要补 ep24：主对照窗口 n=5 已完成，任务门 FAIL，两类越止损线；
  C1 的归档判据不因尾部缺失而改变（尾部只可能提供稳定性信息）。
- 交接：A2-R 队列在 02:58:13 检出 C1 退出、GPU 已排空，随即触发 ep14 归因 safety net
  （见 §86.4 的修复），完成后自动续训 A2-R（ep6→ep16，约 10 h），再排 F1。

## 87. C1 最终归档：ep14 归因判定 + 总结（2026-10-13；A2-R 已接管 GPU）

### 87.1 背景与顺序

用户 2026-10-13 批准剪掉 C1 尾部（§86.5）；A2-R 队列 02:58:13 检出 C1 退出、GPU 排空，
随即用**修复后的工具**跑 ep14 归因（dump 03:06 → 离线 AP 03:14 → 探针 03:16 → 判定 03:17），
03:14:49 A2-R 续训启动（§88）。本节是归因+总结，判定按附录 B 只作**归因**，不参与剪枝。

### 87.2 本次暴露并修复的**两处工具缺陷**（均为工具侧接线，非 C1 模型/权重问题）

| # | 文件 | 症状 | 根因 | 修复 |
| --- | --- | --- | --- | --- |
| 1 | `tool_c_masked_inference.py` | 建模型即崩，dump 不产出（01:18 Stage B 全链失败） | 裸 `Config` 没有 `model.meta_info`，而 `R4Det.py:254` 无条件读它（train_vod.py:174 才注入） | 新增 `ensure_meta_info()`；+2 单测，18 passed |
| 2 | `probe_c_failure_enrichment.py` | 03:0x 首跑 `TypeError: TJ4DDataset: __init__() got an unexpected keyword argument 'samples_per_gpu'` | 没有走 `compat_cfg`，`data.val.samples_per_gpu` 被原样转给数据集构造 | 与 train_vod / eval_dump_c / tool_c 一致地加 `compat_cfg` |

**语义中性回归（关键）**：用修复后的探针重跑 B0 冻结输入（同一 dump/config/分组），
与冻结产物 `c_failure_enrichment_ep14.json` 逐标量比对——**396 个标量中 395 个逐位相同**，
唯一差异是 `contract.sha256`（探针脚本自身的哈希，因本次改动而变）。
⇒ 修复**不改变任何数值语义**，且 C1 探针与 B0 冻结产物走的是同一条代码路径。

### 87.3 ep14 归因读数（附录 B：归因，不剪枝）

| 条款 | 判据 | 读数 | 判定 |
| --- | --- | --- | --- |
| M1 | covered 组 Δrecall ≥ 0 | B0 **51.4881** → C1 **50.4681**，Δ **−1.0200** | **FAIL** |
| M2 | zero 组 Δrecall ≥ −0.5 | B0 **16.7045** → C1 **14.6023**，Δ **−2.1023** | **FAIL** |
| M3 | 逐类 3D AP ≥ B0 − 0.5（离线 harness 同口径配对） | Car 48.7109→49.8727（**+1.1618**）✓；Truck 28.0448→25.7701（−2.2747）✗；Ped 34.2463→**28.7579**（−5.4884）✗；Cyc 50.5391→**39.5657**（−10.9734）✗ | **FAIL** |

`ALL_PASS = false`（三项全 FAIL）。产物：`<C1_WD>/c1valid_ep14_{dump.pkl,probe.json,offline_ap.json,mechanism_verdict.json}`。

**覆盖度分层 recall（同一 harness）**：

| 分层 | B0 | C1 | Δ |
| --- | --- | --- | --- |
| high（有回波、密集） | 55.1713 | **59.5850** | **+4.4137** |
| mid | 59.5894 | 56.6721 | −2.9173 |
| low | 39.9559 | 33.2159 | **−6.7401** |
| zero（自然缺测） | 16.7045 | 14.6023 | **−2.1023** |

**自然缺测富集（描述量，非判据）**：`delta_points` B0 34.7835 → C1 35.8658（+1.0823），
两者 `enriched=true`；C1 多一个合格 range bin（le50/50to70/gt70 vs le50/50to70）。

### 87.4 归因解读（严格标注证据边界）

1. **「重分配而非整体增强」获得三重印证**：成熟窗口（Car/Truck +6.32/+6.72 vs Ped/Cyc −4.89/−5.95）、
   ep14 分层 recall（high +4.41 vs low −6.74 / zero −2.10）、ep14 AP（Car +1.16 vs Ped −5.49 / Cyc −10.97）
   三个独立口径指向同一方向。C1 **不是无效接线**——它确实改变了结果，只是方向与立项假设相反。
2. **与门控方向的对应是“一致的解释”，不是已证因果**：训练日志显示门学到「valid 收缩、invalid 放大」；
   ep14 结果与「在无回波处注入未校准的 prior-posterior 修正」一致。要证因果需 z 排他/干预实验（**未做**）。
3. **对「缺测富集」的正确读法**：C1 并没有改善缺测对象——gap 略增是因为 zero 组掉得更多
   （−2.10 vs covered −1.02）。用 gap 变大叫「更好地利用了缺测」是**错误**读法（§76 同款陷阱）。
4. **边界**：ep14 单 checkpoint、单 seed；harness 数值（Car 48.71 等）与训练内 EvalHook 不同口径，
   只能与同 harness 的 B0 配对比较，禁止跨口径加减（§51.5）。

### 87.5 C1 最终归档口径（三口径 + 三阶段门）

- 区间均值（fixed window **ep12-16，n=5**）：**39.8303**；全轮峰值 **40.6389 @ ep16**；
  best saved **40.6389 @ ep16**（saved = 2,4,6,8,10,12,14,16）。
- 门：ep6 早期检查 **PASS**（+4.7657）；ep12-16 任务门 **FAIL**；ep14 归因 **NOT ALL PASS**。
- **结论**：C1 不构成 AP 成功——主指标未过 40.00、Ped/Cyc 越过 −2.0 止损线、
  Overall +0.5494 低于「大幅」阈值 +2.0、也低于三 seed 基线均值 40.4400。
- **负结果的贡献**：给出「有效性条件化更新」在 radar-camera 时序检测上的清晰失败模式
  （强化 dense-radar 类、牺牲 sparse/missing 类），附三层一致证据与可比 harness。
- **未做/不可声称**：多 seed 稳定性、z 排他因果、C1-D 同容量确定性对照（未跑）。

### 87.6 队列交接

- **A2-R 续训 03:14:49 启动**（resume `epoch_6.pth`，seed0，`--deterministic`，3-rank DDP，GPU 5/6/7），
  配置 `me_rssm/configs/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_semA2R_24e.py`，
  停在 ep16，产物走 `merge_val_curves.py`（三口径 + 同窗 B0 delta）。运行事实见 §88。
- ~~**F1 队列**继续等 A2-R 结束：冒烟 → `relfus_s0_verdict.py` → 24e → ep6 损害检查 → ep12-16 窗口。~~
  **该排程已被取代（2026-10-13）**：另一会话提交 `5cd0886`/`0413abd`，**GP-MSF 取代 F1**
  （预登记 `docs/gpmsf_prereg_20261013.md`，队审 `docs/gpmsf_queue_review_20261013.json`）。
  GP-MSF 队列（`queue_gpmsf.sh`）在 A2-R 结束后接管 GPU：13:27:53 preflight → 13:28:54 baseline smoke
  → 13:41:58 gpmsf smoke（450 iter，三段 S0 门）→ 通过才进 24e。**F1 不再排队。**


## 88. A2-R 续训（ep6→ep16）：运行事实与滚动曲线（2026-10-13 起）

**目的**：A/A2 两臂在 ep6 机制门归档为 NOT ALL PASS（§74），从未取得成熟窗口读数。
用户 2026-10-12 批准（§82.3）只补 **A2-R**（M3 15/20 最接近门槛、读出侧假设最直接）：
从既有 `epoch_6.pth` 续训到 **ep16** 即停（省 8 h），用同一 EvalHook 口径做成熟对照，
不续 A/A2-S（其补训仍在待办，未排程）。

- **唯一变量**：A2-R 读出侧改动（z-only 语义监督 + 零初始化分类 residual 映射到 anchor 分类通道）；
  续训不引入任何新变量、不改 KL/日程/数据。
- **接线等价性已核**：`obs_points` 仅对声明 `accepts_obs_valid` 的 fusion 传递，A2-R 不在其列（§82.3）。
- **运行事实**：2026-10-13 **03:14:49 UTC** 启动（launcher pid 2081375 / 进程组 2081407），
  GPU 5/6/7，3-rank DDP，`--seed 0 --deterministic`，
  `--resume-from <WD>/epoch_6.pth`（optimizer 状态随 resume 恢复），
  config `me_rssm/configs/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_semA2R_24e.py`，
  WD `/data/lurui/work_dirs/fgfull_N4_no2d_igdr_2x4_24e_semA2R_seed0`；
  QLOG `<WD>/queue_semA2R_resume.log`，训练日志 `/tmp/semA2R_resume_16e.log`。
  启动前置条件：C1 已剪枝、GPU 排空、C1 ep14 归因产物已落盘（§87.2 的修复必须先于本次启动，
  否则 safety net 会重放同一 bug）。
- **停点**：ep16（`--max_epochs 16` 等价配置；到点由队列检查 ep16 val/权重/完整窗口后收尾）。
- **已存在曲线（A 父臂 ep1-6，沿用）**：28.0647 / 30.8271 / 37.8929 / 32.0730 / 38.4943 / 36.3491；
  区间均值（ep1-6，n=6）**33.9502**、全轮峰值 **38.4943 @ ep5**、best saved **36.3491 @ ep6**
  （§todo 已记录，此处只作起点；**不能**与 C1 的三口径混写）。
- **滚动回填**：ep7 起的新 val 逐 epoch 追加到本小节；报出时必须写清「A 父臂 ep1-6 + 续训段」
  的拼接关系，窗口均值只有在 ep12-16 五轮齐全后才算三口径之一。

| 阶段 | ep1-6（A 父臂） | ep7 | ep8 | ep9 | ep10 | ep11 | ep12 | ep13 | ep14 | ep15 | ep16 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Overall | 见上（均值 33.9502） | **37.8556** | **36.5401** | **33.0979** | **36.9664** | **39.4519** | **38.3426** | **37.2466** | **38.8329** | **38.0346** | **39.4322** |
| B0 同轮 | — | 37.5701 | 39.5965 | 38.5362 | 38.0619 | 38.9632 | 39.2953 | 37.9178 | 40.3853 | 38.8111 | 39.9949 |
| Δ | — | +0.2855 | **−3.0564** | **−5.4383** | −1.0955 | +0.4887 | −0.9527 | −0.6712 | **−1.5524** | −0.7765 | −0.5627 |

**ep7 逐类（2026-10-13 04:0x，续训首轮）**：Car 44.6530 / Truck 26.2410 / Ped 30.5239 / Cyc 50.0082
（B0 同轮 Car 44.9186 / Truck 27.0695 / Ped 29.0887 / Cyc 49.2034 ⇒ Δ −0.2656 / −0.8285 /
**+1.4352** / **+0.8048**）。

**ep10-12 逐类 vs B0 同轮（2026-10-13 09:3x）**：

| ep | Overall Δ | Car | Truck | Ped | Cyc |
| --- | --- | --- | --- | --- | --- |
| 10 | −1.0955 | **+5.05** | −6.62 | +1.89 | −4.70 |
| 11 | +0.4887 | **+6.92** | −0.12 | −3.77 | −1.07 |
| 12 | −0.9527 | +0.94 | −2.56 | −0.63 | −1.55 |

**续训段窗口（非三口径）**：ep7-12 **37.0424 vs B0 38.6705（Δ −1.6281，n=6）**；
整体 ep1-12 **35.4963 vs B0 33.8563（Δ +1.6400，n=12）**——早期优势已缩到 +1.6。
ep10-12 的**唯一稳定正差是 Car**（+5/+7/+1），Truck/Ped/Cyc 多数为负。

**ep12-16 窗口的门槛余量（n=1，仅作可行性提示）**：ep12=38.3426，若要走满五轮，
其余四轮需平均 ≥ **40.41** 才能让 Overall 区间均值过 40.00；四类分别需
Car ≥47.09 / Truck ≥29.45 / Ped ≥29.98 / Cyc ≥47.79。以本臂 ep7-12 的波动区间
（33.10–39.45）看，这个要求**明显偏高**，但判定仍等 ep12-16 五轮齐全后再下。

**ep8/ep9 逐类 vs B0 同轮（2026-10-13 06:5x）**：

| ep | Overall Δ | Car | Truck | Ped | Cyc |
| --- | --- | --- | --- | --- | --- |
| 8 | **−3.0564** | −1.23 | −2.26 | −2.80 | **−5.94** |
| 9 | **−5.4383** | +0.89 | **−9.72** | **−5.69** | **−7.24** |

⇒ **续训段的同轮比较已经转负**：ep7-9 窗口 **35.8312 vs B0 38.5676（Δ −2.7364，n=3）**，
且逐轮恶化（+0.29 → −3.06 → −5.44）。对照 §81 的校准（早期领先斜率 0.33–0.51、前 6 名 5 个回落），
A2-R 的早期优势（ep1-7 +4.25）在 ep1-9 窗口已缩到 **+2.3600（n=9）**，这个方向与「早期门过 ≠ 成熟过」
的历史规律一致。**ep9 的损失集中在 Truck/Ped/Cyc，Car 反而是唯一正差**——与 C1 的失败形状不同，
但同属「跨类重分配」。

⚠️ **判据不变**：A2-R 的唯一成熟判据是 **ep12-16 五轮窗口**（Overall 门 40.00 + 四类非劣 + −2.0 止损）。
按附录 B 不在窗口完成前剪枝；ep10-16 照跑（当前 ep10，预计约 13:10 UTC 到 ep16）。
本节的同轮比较只作**趋势监视**，不能替代窗口判定，也不能据此提前宣布失败。

## 89. AP 优先的结构级方向审查：七个候选，不再将可靠性门称为新融合（2026-10-13）

**范围/状态**：按用户要求给出较大网络/方法改造候选，查本地代码与既有负结果，
并核验原始论文。完整建议：`docs/rssm_architecture_options_20261013.md`。
本次只修改研究文档；未改网络/config/数据/阈值，未运行新探针或 GPU，
未开训/续训/停训，未改变已批准 A2-R/F1 队列。

### 89.1 必填性能摘要与证据边界

- 本次无新训练：**区间均值 N/A（无新增 val epoch）**；
  **全轮峰值 N/A（无新增 val epoch/epoch）**；best saved N/A。
- 配对参考 B0 seed0：fixed window ep12–16（n=5）**39.2809**；
  全轮峰值 **40.3853 @ ep14**。
- 同族 seed1：fixed window ep12–16（n=5）**41.9185**；
  全轮峰值 **43.1449 @ ep15**（ep15 无对应权重）；作为 seed1 后续配对参考，
  不拿该峰值替代 seed0 的横向对照。家族三 seed 成熟均值 **40.4400 ± 1.3475**。
- C1 已归档结果保持 §87：ep12–16 **39.8303**；全轮峰值 **40.6389 @ ep16**；
  Car/Truck 增益伴随 Ped/Cyc 明显退化。这个失败模式不能外推成网络提升必然零和。

### 89.2 候选与排序

1. 几何约束、多尺度 deformable cross-modal fusion：真实改变跨位置对应，
   先保留标准 RSSM/head 做 AP 能力验证；直接移植并非本项目论文创新。
2. 对象 query 多模态检测器：直接访问原图 FPN 和局部雷达，迭代取证，保留无雷达对象候选。
3. 联合 depth-height / sparse 3D 观测：改变 camera 高度平均与深度歧义表示，不只开 depth 开关。
4. 全类别联合训练的 3D RoI 多模态 refinement：重采样原始证据，区别于 §29–39 冻结 Ped 支线。
5. 对象中心、几何接地 RSSM：实例 `(h_i,z_i)` 直接生成检测，以对象几何/未来观测监督；
   深化已有 B/D 建议，不重复宣称对象时序/未来预测首次。
6. 点—区域层级 radar encoder：早期局部聚合重写，不是后端门控或仅加高分辨率 Ped 分支。
7. 条件候选：train-only 强 teacher 的对象/几何蒸馏；本地 LiDAR/teacher 可用性未核，
   不引入验证/测试信息，不冒充同监督对照。

推荐先 1；随后 2/4 二选一，成功的对象观测器再接 5。
每个方向均写了 hypothesis、必要消融、失败风险与 prior-art 边界；
不保证 AP、不承诺七方向堆叠有效。新臂需另写预登记，当前仅建议。
6 epoch 不够判方法成功；成熟窗口/全轮峰值与多 seed 配对保留，
若新架构需要更长预算，双方计划训练前匹配，不能事后选有利窗口。

### 89.3 评测纠错与文献核验

- `TJ4D-eval.py` 及当前 `kitti_utils/eval.py` 的 `clean_data` 对难度用 `>` 忽略，
  因此 **moderate 是累计 0–70 m，而不是只 50–70 m**；距离分组标签本身不能代替 AP 协议。
- 用户参考 47.29 的原论文 Table 1 称 test，而 Table 4 同值且 §4.4 称 validation ablation；
  split/recipe 尚需核实，不把文献数值直接当严格配对或“最新 SOTA”认证。
- 主文献：RCBEVDet/CRN/RaCFormer/PV-RCNN/StreamPETR/SlotFormer/DistillBEV，
  仅用于结构先例；未完成本项目组合的新颖性全面检索。

### 89.4 验证与下一步

文档变更检查与 `git diff --check`；无代码变更，未重跑 sanity/训练，
旧测试数量不算本次新验证。下一步由用户选择结构方向后补实现计划/预登记；
不追加一轮无边界诊断作为默认前置。

补充原文核验：CRN 的正式标题为 “Camera Radar Net for Accurate, Robust, Efficient
3D Perception”；R4Det 的 validation ablation 在 §4.4。另明确对象状态路线不假定 TJ4D
有真实 ego pose：可靠变换可用时才使用，否则学习对齐。此补充仍仅修改文档。

## 90. 七个结构方向补入 TODO：历史尝试、已有实现与现役队列核验（2026-10-13 04:29 UTC）

**范围**：按用户要求将 §89 的七个方向逐项写入 `docs/todo.md`，先核对已尝试/已开始的相关路径。
检查完整训练记录、候选文档、相关源码与只读进程信息；不修改模型、配置、脚本、数据或阈值。
正文只列未完成候选，已完成核验移入 TODO 归档区；保留此前整理后的全部历史归档原文。

### 90.1 必填性能摘要与运行边界

- 本任务无新训练或评测：**区间均值 N/A（无本任务 val 曲线/窗口）**；
  **全轮峰值 N/A（无本任务 val 指标/epoch）**；best saved N/A。
- 历史 AP 不重算、不改结论、不跨栈相减；相关完整曲线/双口径仍见 §§29–40、§47、§59–61、§71–75、§87–88。
- 只读进程核验：A2-R 的 3-rank `train_vod.py` 仍在（launcher 2081407，ranks 2081419/2081420/2081421），
  A2-R 队列 1354393、F1 队列 1477871 均存在；未见 F1 正式训练进程。
  此处只确认状态，不凭进程存活宣称训练成功；不报未核实的当前 epoch/ETA。
- 未使用 GPU 做探针，未开训/续训/停训或调整任何队列；七个方向不是本次训练授权。

### 90.2 去重与状态核验

1. **几何约束多尺度 fusion**：§40 的 `Cross_Modal_Fusion` 已训练、成熟门 FAIL；
   源码是同位置 avg/max 掩码互乘后 concat+conv，不是本候选的几何跨位置/跨尺度取证。
   仓库有 warp/DCN 工具，F1 已实现并排队但仍属可靠性门；完整候选未发现实现/主线训练归档。
2. **对象 query detector**：已有 `SparseBEV_rc` 与 query 驱动的 image/radar sampling（rc transformer 395–457）；
   `configs/` 内本次未找到该 detector 的引用，完整训练记录未见对应归档。
   不标“全仓库未实现”，也不把 A2 分类 residual/续训算成 query detector。
3. **联合 depth-height**：雷达辅助 depth 代码及前景相对 depth 监督已有；
   当前 camera volume 仍取高度均值，完整联合方案未发现对应归档。
4. **全类别 3D RoI 取证**：2D RPN/RoI+IGDR 已有 FG-FULL 训练；Ped-only refinement/high-res 等 §29–39 已失败归档。
   新候选必须区别在原始多模态重采样、所有类别与联合训练，不能只是恢复 IGDR 或冻结 Ped residual。
5. **对象中心 RSSM**：20261008 候选 B（§3）/D（§5）已提出；完整实例状态/跨帧身份/直接检测尚未实现/正式训练。
   §59–61 的 Object-Centric Latent Supervision 已训练且 AP 止损，但仅为 dense correction heatmap 辅助监督，推理丢弃该头。
   A/A2、全图低维未来目标也已试；均不能与完整对象状态方案混称。A2-R 本次仍在续训。
6. **点—区域雷达编码**：pillar/scatter、Doppler 旁路已有；MAI/MCI/DC 和 Ped high-res 已试。
   完整早期局部关系/图像引导编码未发现对应归档，不将物理统计旁路改名重做。
7. **对象/几何蒸馏**：通用 distill runner/API/PKDLoss 已有；本候选未接入当前主线，未发现对应训练归档。
   强 teacher 或 train-only LiDAR 可用性未核，不以通用蒸馏代码存在宣称方案就绪。

上述“未发现”限定本项目记录与本次源码检查，不把仓库内通用/旧分支代码认定为已验证方法。
每项 TODO 同时写清状态、相关旧路径、禁止重复包装的边界与下一步必要消融。

### 90.3 验证与结论

- 文档契约检查：七个方向均有独立 `[ ]`、状态与历史边界；已有其余待办保留；
  此前 `## 已完成` 内容除新增核验归档外逐字保留。
- `git diff --check`；仅文档变更，不重跑 sanity，不把既有测试数量写成本次新增验证。
- 结论：候选池补录完成，方法实现/训练仍待单独选择与预登记；不改变既有排程或冻结判定。


---

## 91. 方向 1 几何多尺度 fusion：旧模块核验、原文对比与 GP-MSF 设计（2026-10-13，CPU）

### 91.1 范围与必填性能摘要

用户要求设计并论证 TODO 第一项几何多尺度 fusion。本次只读现行源码/配置、历史 EvalHook 日志、train infos 和原论文/作者资料，新增设计文档与 CPU 证据报告。
**无网络实现、GPU 探针、S0 或新训练/评测**，不改变 A2-R/F1/其他队列，不占用 GPU。

- 本任务**区间均值 N/A（无新增 val 曲线/窗口）**；**全轮峰值 N/A（无新增 val 指标/epoch）**；best saved N/A。
- 历史指标复读使用每 epoch 最后一次 val；窗口 ep12–16 五轮等权、n=5，峰值取所有有 val 的 epoch。实际权重存在性单独核验。

| 历史 run | 区间均值 ep12–16，n=5 | 全轮峰值 | 当前 best saved |
| --- | ---: | ---: | ---: |
| `run10_headv2_multiseed/seed_0`（旧同栈 concat） | 38.3902 | 39.8802 @ ep16 | 39.8802 @ ep16 |
| `crossmodal_fusion_N4_2x4_24e_seed0`（旧恢复预训练 CrossModal） | 37.9789 | 39.2835 @ ep16 | 39.2835 @ ep16 |
| `fgfull_N4_no2d_igdr_2x4_24e_seed0`（当前 B0） | 39.2809 | 40.3853 @ ep14 | 40.3853 @ ep14 |

旧 CrossModal 的完整 ep1–16 Overall 曲线补在设计附录 A；本次复读的全部历史 epoch 及四类构成项保存在 JSON，不是新 val。
历史正式运行配置、起止/截断、逐项窗口均值等继续引用 §40、§47 及之后的 multi-seed 归档，不以本节摘要替代明细。

### 91.2 两个必须修正的因果/坐标读法

1. `BEVCross_modal_attention.py:13–50` 的旧类是 mean/max→sigmoid 空间掩码互乘→concat+3×3 conv，不含标定 reference、跨位置 learned offset 或多尺度 source selection。
   旧同栈替换效应为 Overall −0.4113、Car −2.0400、Truck +3.2709、Ped −3.2296、Cyc +0.3537；它比当前 B0 的 −1.3019 是**跨训练栈参照，不能全部归因于 fusion**。
   原干预还改变了预训练 fusion 权重是否加载，不能声称已纯粹识别掩码算子的因果贡献；F1 原排程不因本次核验自动改变。
2. 当前 resolved `zbound=[-4,2,6.0]` 只有一层；`mean(-1)` 本身不抹掉多层。高度容量不足发生于单层表示设计，而非已证明该 mean 是瓶颈。
   image FPN / radar SECONDFPN 本来已有多尺度信息，不能宣称“第一次加 FPN”；新候选改变原生取证位置/来源而非只堆一个下采样 neck。

### 91.3 提案、文献边界与唯一研究因素

完整论证：`docs/geometry_multiscale_fusion_design_20261013.md`。

GP-MSF 保留原 ConcatConv 基础通路，通过有界共享米制地址读原 image P2/P3/P4 与 SECOND 原生三层；同地址的 image/radar 配对条件化双向消息，分开归一化；新增直接 radar 证据无支持时为 null，但 image fallback 不依赖回波。
消息以零初始化线性投影分别加入 C/R，仍使用原 `fusion_conv` 输出 256 通道给现有 RSSM/head。

首轮唯一研究模块：前端 RCFusion。无新对象 detector/RoI/完整 voxel volume/occupancy task/新 RSSM loss；N4、原 backbone/head、既有 FG/depth/RSSM 配方保持。
同 seed 不自动保证新增模块后的随机初始状态公平，实施要校验共同模块初始 state/RNG 与 step-0 identity，禁止新臂 hot-start 成熟 B0 而控制从头训。

核对 R4Det、SGDet3D、Doracamom、CVFusion、RCFusion、LXL、RCBEVDet++，补 MSSF/SGDet3D++；后两者直接限制“几何/多尺度/雷达引导原图检索”的首创主张。
DR-Net 仅核到原始题录/DOI，原文未取到，不使用未核模块细节；其余原文/作者仓库/出版题录的覆盖层级在设计 §13 明列；RCFusion 原始正文也未取得，粗结构只采用原论文之间可交叉核对的部分。
方案是可检验的研究候选，不承诺 AP 增益或论文新颖性；必须以 same-capacity/fixed-offset/native-vs-BEV/source/null/fallback 消融和 RSSM/GRU×fusion、h/z 排他干预给出机制证据。

### 91.4 本次 CPU 运行事实与证据

- Config 用 `import_custom_modules=False` 只解析继承，避免 custom-import 触发必须 CUDA 的 localattention；未构造模型或使用 GPU。
- train infos：5,706 帧，sha256 `0eb60b281141afc22879efe008d23e78c485837e69cd912a126af7b752cccb11`；按原 dataset camera→LiDAR/类别/BEV xy 筛选 23,029 框：Ped 3,242、Cyc 5,125、Car 11,030、Truck 3,632。
- 中位宽度 / 0.32m：Ped 1.9604、Cyc 2.3626、Car 5.7213、Truck 8.3546 cells。仅是训练设计统计，不是 AP 分母或“高分辨率必有益”的证据。
- train-only 等类/等对象、保留高度 surface 等权的 q10/q50/q90 reference：−1.1230436265 / −0.3003779650 / 0.6601446867m。
  规则完整记录于 JSON，不使用每样本 GT 作 forward 地址/有效性，也不混用工作区旧未跟踪草稿中的不同 quantile 常数。
- radar native shape 248×216 / 124×108 / 62×54，步长 0.32/0.64/1.28m，receptive-center origin=(0.08,−39.60)。这是新增采样器的 index bookkeeping，**不是证明原 semantic alignment 错误**。
- 实际 train 标定 + 合成 augmentation 的分步/复合投影等价：1,280 点最大像素误差 4.55e−13；CPU native affine-field `grid_sample` 三层最大误差 4.44e−15 / 3.55e−15 / 1.78e−15。
  这不是实际 augmented metadata、网络 forward 或 GPU S0 证明；真实接线测试仍待执行。
- B0 三 seed 窗口复读：39.28088 / 41.91848 / 40.12066，均值 40.4400，样本 std 1.3475（ddof=1），总体 std 1.1002；不把 std 当严格差值显著性门槛。
- 临时 CPU audit 脚本：`/tmp/r4det_geom_fusion_design_audit_20261013.py`；长期证据、完整原始复读曲线与定义：`docs/geometry_multiscale_fusion_audit_20261013.json`。
- 原有未跟踪 `docs/artifacts/` 未改/未提交；不把它算本次新产物或实现。
  **状态更新（2026-10-13，§97.6）**：该未跟踪目录及其中唯一文件已按用户指示删除（属被取代的中间推导，
  其高度参考与 arm 冻结值不同）；删除前的 sha256 与差异细节见 §97.6，本节其余内容为当时事实，不改写。

### 91.5 验证与下一步

本次只对文档/报告做契约、数值、历史日志一致性与 diff 检查；不运行无关模型测试，不复用旧测试数量充当新验证。
验证结果：`/tmp/r4det_geom_fusion_validate_20261013.py` 六组 focused checks PASS（历史曲线/三口径、数据/数学、source hash、旧未跟踪文件、追加式历史、TODO 与范围契约）；`rtk git diff --check` PASS。
校验脚本初版的 heading/归档匹配断言曾误报，已锚定正式标题并包含“条件方向 7”后重跑通过；不是网络/GPU 测试通过。
TODO 标记“设计论证已完成”并归档该子任务；完整方向 1 的 `[ ]` 保留，明确尚未实现/冻结预登记/训练。

建议最小验证顺序：同初始化 identity + 真实坐标/augmentation + 两步梯度 + 原 bs/N4 的 S0，然后 B0/同容量 conv/固定取证/完整块；成熟正结果后才补其余归因和多 seed。
大幅目标暂提议配对三 seed ΔOverall≥+2.0（ep12–16），Ped/Cyc 与 high/low/zero 一并报告；具体非劣容限/训练停止规则尚未冻结，不改既有预登记。
若机制未成立，不默认扩展 head/volume/新损失或调大采样半径。设计和 CPU 数学核验完成不等于网络/性能验证完成。


## 92. F1 与几何多尺度 fusion 的优先级建议（2026-10-13，只读状态核验）

### 92.1 范围与必填性能摘要

用户询问「现在的 F1 还要训吗，还是直接上新的 fusion 方案」。本次核验已有队列、进程与产物，给出研究优先级建议；**未启动/停止任何进程，未修改队列脚本、模型或训练配置**。不占用 GPU，不新增评测。

- **区间均值 N/A（无新增 val，ep12–16 窗口无样本）**；**全轮峰值 N/A（无新增 val 指标/epoch）**；best saved N/A。
- F1 当前没有自身 val 曲线，不能写成功/失败或用别的臂指标替代；本节不改写 §83 的冻结预登记，也不替代历史正式运行明细。

### 92.2 运行事实（2026-10-13 06:49:17 UTC）

- A2-R resume 队列 PID/PGID `1354393/1354393` 仍在；其现役训练进程组 `2081375`，launcher `2081407`、ranks `2081419/2081420/2081421`。配置为 `TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_semA2R_24e.py`，从 `epoch_6.pth` resume；本次没有解析新 val，不推断当前 epoch/性能。
- F1 队列 PID/PGID `1477871/1477871` 仍在。`fgfull_N4_no2d_igdr_2x4_24e_relfusion_seed0/queue_relfus.log` 只有 `2026-10-12 15:13:16 armed; waiting for the A2-R resume queue to finish`。
- F1 工作目录只有该队列日志和空的 `smoke_450iter/`；`/tmp/relfus_smoke.log`、`/tmp/relfus_24e.log` 均不存在，进程表未见其训练。由日志/产物/进程三者一致判定：**F1 冒烟与正式训练尚未开始**。
- GP-MSF 只有 §91 的设计、CPU 几何/历史核验，没有模型实现、真实 pipeline/GPU S0 或正式训练；不能立即启动一个尚未实现的新臂。

### 92.3 建议与论证边界

**建议：将方向 1 的 GP-MSF 实现/验证作为下一项主线工作；暂不投入 F1 完整 24e，不把它当新方案前置条件。保留 F1 代码、预登记和后续对照价值。此建议尚未变成排程操作。**

1. F1 的唯一干预是 `concat(camera, alpha * radar)`：同位置雷达标量门，不读取新的原图/原生 radar source，也不改变跨模态取证地址。因此它能检验「同位置强度调节是否有用」，却不能直接检验 §91 的空间对应、原生尺度取证与视觉缺测回退假设。新几何方案不以 F1 的 AP 正负为依赖。
2. 项目目标同时包含机制级论文贡献与 Overall 3D moderate 的明显提升；已有 §91 的结构候选比继续单一门控更直接对应这些假设。这是**问题匹配度与研究机会成本的判断，不是预测 GP-MSF 必然更高 AP，也不是断言 F1 无效**。§91 的旧 CrossModal 失败不能外推为 F1 失败。
3. F1 尚未开训，没有需要「把已跑的实验收尾」的训练沉没成本；但队列曾获批准，所以本次只提出改排建议，不擅自取消。F1 后续可作为额外门控控制，不取代 GP-MSF 必需的 B0 / 同容量 conv / 固定几何采样对照。
4. 不建议「先跑 F1 六轮，再凭早期 AP 决定是否值得」作为性能结论：§81/§83 的 ep6 只查严重任务损害，不能代替 ep12–16 成熟窗口。若日后要正式评价 F1，应按冻结口径完成成熟窗口，并报全轮峰值和完整曲线。

### 92.4 推荐执行顺序与停止条件（均未执行）

1. 保持当前 A2-R 不受干扰；利用其训练期间的非 GPU 时间完成 GP-MSF 实现与预登记。若用户确认采用该优先级，另行安全延后/撤下尚在等待的 F1 队列，不能让旧队列继续自动抢占下一训练窗口。
2. 先证实真实增强/标定/坐标映射、共同模块 same-init 与 step-0 baseline identity、无回波时 camera fallback、至少两步梯度到达新取证分支；随后在原 batch/N4 上做 GPU S0（仅 GPU 5/6/7，不抢现役任务）。
3. S0 通过后才开展新臂正式训练；先以 B0 为性能参照，按设计补同容量 conv / 固定几何取证以分离容量、取证与学习地址贡献，不同时更换 RSSM/head。
4. 如果接线、梯度、数值或原 batch 显存门失败，先定位修复，不直接长训；如果成熟窗口无收益，按预登记归因，不追加未经验证的模块。

本节完成的是**调度建议与状态核验**；F1 原队列依然有效，方向 1 的实现/训练 TODO 仍未完成。


## 93. z_t / KL 失效是否已解决：当前机制状态核验（2026-10-13，CPU 只读）

### 93.1 范围与必填性能摘要

用户询问当前 z_t 与 KL 失效问题是否已经解决。本次读取现行源码、现役 A2-R 的已写入训练日志、ep6 gate JSON 和历史机制记录；**无新训练、val、GPU 探针或模型/配置/队列修改**。

- **区间均值 N/A（本次无新增 val，默认 ep12–16 无新增样本）**；**全轮峰值 N/A（本次无新增 val 指标/epoch）**；best saved N/A。
- 现役 A2-R 的正式曲线继续按 §88 滚动归档；本节的 train 统计与 ep6 机制指标不能当 AP 或替代成熟窗口。

### 93.2 已观测事实

1. **KL 实现仍是逐元素 free-bits clamp**：`rssm_fusion.py:327–370` 对 Gaussian KL 每个元素作 `max(kl_raw, free_nats)`，然后取均值。`SemanticReadoutRSSMFusion` 继承 `MotionAlignedRSSMFusion`，没有覆盖 `kl_loss`。现役 resume 日志打印的 resolved config 为 `kl_scale=1.0, free_nats=1.0`；A2-R 改的是语义监督/分类 residual 读出，不是 KL balancing 或聚合口径。
2. **现役统计仍高度 clamped**：截至读取时 A2-R 已有 ep10 val、正在 ep11。固定引用 `2026-10-13 07:25:02.082 UTC, Epoch [11][350/1902]` 已落盘日志：`stat_kl_raw_mean=0.5429`、`stat_kl_effective_mean=1.0032`、`stat_clamped_ratio=0.9954`、`loss_rssm_kl=1.0033`、`stat_mu_diff_sq=0.0283`。日志平均统计中的 **99.54% KL 元素低于 free-bits 门槛**，对这些元素 clamp 的直接局部梯度为零；仍有约 0.46% 未 clamped，**不能说整个 KL 梯度为零**，也未测共享参数上的梯度范数。
3. **z 有因果通路，不等于有足够任务信息**：§49 的当前 B0 冻结 ep14 小窗口诊断，zero_z 的 head 相对 L2 扰动 38.02%，shuffle_z 4.67%，prior-only 5.87%。这些是局部输出扰动，不是 AP 变化，也不能证明所有样本无信息；但它们不支持「只是接线没接上」，亦不支持「z 的样本判别作用已充分建立」。
4. **A2-R 有部分进展，但最后完成的机制门未全过**：直接复读 `<A2-R WD>/semreadout_ep6_gate.json`，M1 FAIL、M2 PASS、M3 FAIL、M4 FAIL（同 §75）。val 排序 AUC 0.9305 / M1' PASS，不能忽略该正证据；但 thr50 0.0712、读出 CE 变差 15/20、donor_transfer 0.2854 未达到各自冻结门。只有 ep6 已完成此 gate；本次未运行 ep11/成熟 checkpoint 机制复测，不能把 ep6 失败外推为「ep11 一定仍失败」。
5. **训练侧语义提升不是验证侧机制证明**：同一 ep11 iter350 训练日志 `stat_sem_hit1=0.4041, stat_sem_thr50=0.6280, loss_rssm_sem=0.4521`。这说明相关支路存在训练信号，不能替代 val 的语义泛化、读出干预或成熟 AP。

### 93.3 结论、解释边界与融合方向关系

**结论：z_t / KL 的总目标尚未解决；不能把工程接线/标准结构修复、部分机制读数变好、或者 AP 单点变高写成问题已闭环。**

- z_t：已有因果路径和部分语义排序正证据；尚未证明样本/类别特异信息稳定进入检测决策，并带来成熟窗口、配对多 seed 的增益。不是「完全没用」，也不是「已经修好」。
- KL：数值计算/日志正常，但仍沿用高度 clamped 的逐元素口径；stop-gradient dyn/rep balancing、按 cell 沿 channel 聚合的 free-bits 等后置设计尚未落地/验证（TODO 原条目维持未完成）。
- **free-bits 本来允许门槛以下不受 KL 压力**，高 clamped 比例本身不等于完整 posterior collapse 或模型失败；结合任务因果/读出证据，目前只能拒绝「已经解决」的声明，不能由均值直接推导全部分布相等。
- §70 中的 recon 总损失占比是标量尺度信息，**不等于梯度预算**；本节不以「占比很小」单独证明监督无梯度。也不沿用已被 §70 纠正的「真实重建目标 95% 是共享模板」说法。
- 新几何多尺度 fusion 改的是 RSSM 前的观测取证，**不是 z/KL 修复的替代证明**。即便以后提升 AP，也须另做实际检测的 z/h 干预、posterior/prior 读数与同容量确定性控制；保持 RSSM 不变是为了单独识别 fusion 效果，不表示 RSSM 问题已经解决。

本任务完成状态核验；原「z_t 失效总目标」继续未完成，A2-R 续训与 F1 队列均未改动。


## 94. z_t 有局部机制响应为何未形成稳定 AP 收益：只读归因（2026-10-13）

### 94.1 范围、运行事实与必填性能摘要

承接 §93，用户要求查明「机制有响应但 AP 未明显提升」的原因。本次 **CPU 复读**现行语义读出/anchor 推理/探针源码、A2-R 与 B0 的完整已写日志，以及 A2-R ep6 机制/排序/尺度/采样/BN 原始 JSON。
证据快照 `2026-10-13T09:00:12.575705+00:00`，A2-R 明确固定到 ep11；运行中的后续 epoch 不混入本报告。**无新模型训练、GPU 前向、AP 评测、权重或队列操作**。

- 本任务**区间均值 N/A（无新增 val）**；**全轮峰值 N/A（无新增指标/epoch）**；best saved N/A。
- 现役 A2-R 的既有 val 快照：实际 **ep1–11，n=11，均值 35.2375**；全轮峰值 **39.4519 @ ep11**；best saved **36.9664 @ ep10**。默认 ep12–16 在此次固定快照 **n=0**，不能作成熟判定。
- B0：**ep12–16，n=5，均值 39.2809**；全轮峰值 **40.3853 @ ep14**（所有已有 ep1–23 val）；best saved **40.3853 @ ep14**。与上一项窗口不同，不能直接相减解释效应。
- 同 seed 同轮描述：ep1–11 Δ **+1.8757**；续训段 **ep7–11，n=5，36.7824 vs 38.5456，Δ −1.7632**。不是预登记成熟窗口，也没有配对多 seed，不能作统计显著性结论。

配置/唯一变量沿用 §75/§88：A2-R vs A 仅 `cls_readout_mode`，但 vs B0 包含 z-only 语义监督 + 分类 residual 两项；本次不加入训练变量。
完整逐 epoch/逐类别曲线、窗口逐项均值、训练动力学与后续诊断树：`docs/rssm_z_ap_gap_diagnosis_20261013.md`；JSON 保留两臂全部已纳入 val 指标、逐 clip 原始证据与 SHA256，不以本节摘要替代明细。

### 94.2 已定位的断点与证据等级

1. **验证高分峰预算泛化差（已实测，ep6 冻结 20 clips）**：train/val AUC 0.9995/0.9305，但 top-k recall 均值 0.3453/0.1033、中位 0.2917/0.0000；零 top-k 为 3/20 vs 11/20。val 7/20 clips 同时 AUC≥0.9、top-k=0；峰距均值 43.73 cells、中位 7.00，9/20 clips 平均峰距 >10 cells。不是背景分数整体变高，也不能把格点峰距当作实际检测框误差。
2. **局部分类确有作用，但代理指标不等价于 AP（原始 JSON + 源码）**：residual on/off 的 GT 格 clipmean CE 0.6024/0.7217（改善 0.1193）、15/20 clips 改善；100 GT 格加权改善 0.0790，pooled head 类别命中 72%→76%。M3 对每类取 max-over-anchor，再在 GT 格算 softmax CE；不含负格、框 IoU、真实逐 anchor sigmoid 排序或 NMS，所以不能直接推论新增正确 TP。
3. **监督与部署读出不一致、同格 anchor 质量不可分（源码事实，效应量未识别）**：辅助 `3×3+ReLU+1×1` sem_head 不部署；直接 residual 是另一个 `1×1(z)`。A2-R 同格同类所有 anchor 共用增量，不能改变该组 anchor 的 logit 差，也不直接修改 bbox/dir。原 `output_proj(z)+feat` 仍可间接影响回归，不能泛称 z 无几何路径。
4. **跨类抵消与严格 3D 收益不足（已有 AP，同窗 ep7–11）**：Car +2.2722、Truck −3.9101、Ped −1.7877、Cyc −3.6273；Overall 2D/BEV/3D Δ 为 +0.1587/−0.4507/−1.7632。Truck 3D strict −3.9101、loose +1.5294，提示高 IoU 框/排序链值得查，但不能由此直接定位到 z/尺寸/朝向的某一个误差分量。

### 94.3 排除边界与仪器解释修正

- 单次 val 采样/部署契约切换使 z 的 relative mean-abs 移动 33.45%，热图移动 1.31%，hit@1/thr50 不变；局部 4-BN 切换也未消除 train/val 差距。它们削弱部分解释，不等于全面排除所有噪声/BN/AP 影响。
- residual 的 GT 格命中尺度曲线 72/76/77/76/75/71%（×0/1/2/4/8/1e6），不支持先扫增益。
- 高 clamped KL 仍在，但未测其独立 AP 因果效应；不将标量 loss 占比等同梯度预算，不沿用已纠正的「真实 recon 95% 模板」前提。
- 高斯半径不改变 `target.eq(1)` 只有对象中心是 exact-positive 的事实；radius2 单核支持25格、exact-one仅1格。需将相邻格误差与大距错峰分开，不以旧 full-res M1 门单独否定全部语义信息。
- M4 donor 采样位置来自 recipient GT、没有 donor 对象对应；donor 多数类转移 FAIL 不是充分的无语义证明。M4 macro/micro 与多数类地板应分别报告，不回改冻结判定。
- BN artifact 的 gate reference metadata 仍是父臂 A；本次采用 raw arms 而非该比较标志。M2 同一头输入替换也不替代训练匹配的确定性控制。

### 94.4 结论与下一步（未执行新 GPU 探针）

**当前最可信解释不是「机制已经完整有效，AP 无缘无故不涨」，而是语义可解码、局部分类有用，但验证高分峰、部署 anchor 读出与高 IoU 3D 框收益链未闭环，且各类收益抵消。**
输出级 FP/IoU 误差、梯度冲突、随机必要性仍未知，不把辅助头的错峰直接写成最终检测 FP 的已证原因。
最小补证为成熟同 checkpoint 的 full-val residual on/off + NMS 前 score/IoU/top-k/错误分解；另做 z 全路径干预，因为 direct residual off 不移除原 z 共同特征路径。
两臂与 B0 必须使用同 evaluator，不能混算离线 harness 与 EvalHook（§51.5）；必要时再做按损失的梯度范数/夹角与同容量确定性配对训练。

产物：`me_rssm/sanity/audit_z_ap_gap.py`（纯标准库只读审计）、`test_audit_z_ap_gap.py`、上述诊断文档与 JSON。聚焦单测 **7 passed**；窗口/峰值/截断、坏记录/partial-tail、数学反例/高斯/共享偏移恒等式均覆盖。
本次完成 CPU 归因审计；**z_t 失效修复总目标仍未完成**，预测级归因列为待办，无新排程。


## 95. GP-MSF 首版实现与 CPU 接线门禁（2026-10-13，无新训练）

### 95.1 范围、必填性能摘要与唯一变量

本节记录**方向 1（几何多尺度 fusion，GP-MSF）首版代码实现**与 CPU 接线门禁结果，承接 §91（设计）与 §92（优先级）。
本次**无 GPU 前向、无 val、无 AP、无权重、未接入任何 config 或训练脚本**；A2-R 续训与 F1 等待队列均未改动。

- **区间均值 N/A（本次无新增 val；默认 ep12–16 无新样本，n=0）**；**全轮峰值 N/A（本次无新增 val 指标/epoch）**；best saved N/A。
- 本节任何数字都是 CPU 接线门禁值，不能当 AP 或成熟窗口；正式训练后必须按顶部统一口径补记。
- **唯一变量（相对 `ConcatConvFusion`）**：融合层类型本身；RSSM、head、anchor 头、优化器、数据、日程、batch/N4 全部不变。
- 参数量实测（CPU 构造，`img_channels=256, rad_channels=384, out_channels=256`）：基线 `ConcatConvFusion` **1,475,072**；GP-MSF 总计 **1,652,798**，其中共享 `fusion_conv` **1,475,072**（与基线逐元素一致），**新增 177,726（0.1777 M）**，低于设计 0.5 M 预算。
- 新增参数构成（主要项）：`proj_img_levels` 49,344、`proj_rad_levels` 28,864、`proj_i2r` 24,960、`proj_rad` 24,640、`proj_r2c` 16,640、`proj_cam` 16,448、`offset_conv` 13,912、`key_rad`/`key_img` 各 1,088、`geom_score` 356、`depth_score` 97、其余 bias/标量 63。

### 95.2 产物与运行事实

| 项 | 内容 |
|---|---|
| 模块 | `mmdet3d/models/fusion_layers/geometry_multiscale_fusion.py`（`GeometryPairedMultiScaleFusion`，注册进 `FUSION_LAYERS`） |
| 导出 | `mmdet3d/models/fusion_layers/__init__.py` 增加 import 与 `__all__` |
| 门禁脚本 | `me_rssm/sanity/gpmsf_cpu_checks.py`（10 项 CPU 接线检查，toy 形状但契约等同真实管线） |
| 运行方式 | `source .envrc && python me_rssm/sanity/gpmsf_cpu_checks.py`；退出码 0 = 全部通过 |
| 结果 | **10/10 passed**，CPU 约 12 s；`flake8 --max-line-length=110` 对两个新文件 rc=0 |
| 已核验几何契约 | radar 原生三级 0.32/0.64/1.28 m（0.16 m 底图，pixel-0 **中心**原点 0.08/−39.60）；image P2/P3/P4 stride 4/8/16；BEV 布局 (y=248, x=216)；高度先验 `[-1.1230436265, -0.3003779650, 0.6601446867]` 仅作覆盖先验 |

10 项门禁逐项结果：

1. **t1 step-0 identity**：真实几何输入下（含 `lidar2img`/`bda`/points/depth）输出与同 state 的 `ConcatConvFusion` **逐位相同**；`geo=None` 路径同样逐位相同。
2. **t2 RNG 隔离**：新参数在独立 seed 下构造，`fusion_conv` 之后的下游（RSSM/head 等）RNG 流与基线一致；新参数可复现。同 seed 只保证共享模块初始化相同——这正是设计 6.5 要求核验的点。
3. **t3 米制→索引簿记**：3 个 radar 原生级 + 3 个 image 级在各自 cell/pixel 中心处**精确取回自身**（无半像素偏移）。
4. **t4 图像取证链路**：把 delta 放在 P3 某一像素，独立用 4×4 矩阵解出的 BEV 地址读回误差 **6.0e-08**（= 该级权重 × conv(delta)）；再把 P3 换成行斜坡，三个高度参考各自读到**自己投影行**的值（最大误差 2.4e-07，读数 0.259/0.194/0.117）。
5. **t5 无雷达支持**：`M_R` 恰为 0（相机 BEV 逐位不变），`M_I` 仍非零——无回波不禁用视觉取证。
6. **t6 图像全无效**：所有投影不可用时 `M_I` 恰为 0（radar BEV 逐位不变），`M_R` 仍非零。
7. **t7 两步梯度**：step 0 输出投影之外的地址/键/级权重梯度为 0（`offset_conv` 梯度 0），第一次更新后变为非零（offsets 6.62、keys 1.81、level weights 0.594），证明两跳传递成立且 step 0 不破坏基线。
8. **t8 分块等价**：`query_chunk=5` 与单块输出一致（≤1e-5）。
9. **t9 地址有界**：每 head 地址半径上限实测 1.0051 m ≤ 配置上限 1.3194 m（`tanh` 有界）。
10. **t10 冒烟**：`grad_checkpoint=True` + 分块前反向有限，60 个参数张量全部拿到梯度。

### 95.3 门禁发现并修复的真实缺陷（机制归因）

CPU 门禁不是形式化签字：它抓到 5 个会导致真实数据上**静默错误**的实现缺陷，全部已修并复跑通过。

1. **图像 v 方向按宽度归一化（最严重）**：原实现用 `stride * feat.shape[-1]`（宽）同时归一化 u 与 v，只在正方形特征上成立；真实 P2/P3/P4 是 120×160 / 60×80 / 30×40，**每一级图像读都会在行方向整体错位**。t4 用独立投影数学比对后定位，已改为 u 用宽、v 用高。
2. **跨 view 计数/分母聚合错误**：相机候选的 `count/denom/ok` 按 `(B*N, M)` 直接 reshape 成 `(B,N,M,...)`，会把 view 维错当候选维；在 N>1 时有效性与归一化都会错。已改为先在 view 维求和。
3. **有界地址生成器通道排布错位**：offset 输出 `view(...).permute(...)` 少对一维，导致 24 通道被当 2 通道广播（t1/t5–t8/t10 全挂）。已按 `(B,Hd,K-1,2,Q)` 排布对齐。
4. **masked softmax 收到浮点 mask**：radar 支持掩码经 `.to(dtype)` 后传入，`~mask` 对浮点报错。已统一在 softmax 内转 bool（语义不变）。
5. **3-D 张量送进 Conv2d/LayerNorm**：offset 头需要在最后一维补 1，channel-LayerNorm 需要转成 channels-last。均已修正。

另有 2 处属**测试自身**的口径错误（已按设计文档修正，不是模块缺陷）：radar 原点约定为 pixel-0 **中心**（0.08/−39.60），t3 最初按边界约定写值；t4 最初选的 P3 像素在 toy 相机里只能由相机后方点投出（depth=−3.75），换成行号 > 主点行后有效。

> 结论边界：以上只证明**接线正确、step-0 保基线、缺测回退与两步梯度成立**。它不构成任何 AP 预期，也不能替代真实数据上的 S0；`proj_*` 投影层仍带随机 bias，真实收益/损害需要训练后才能判断。

### 95.4 尚未验证与风险（必须随报告一起读）

- **模块当前不可达**：没有任何 config 使用该类型，`R4Det.py` 的融合调用点也还没传 `geo`（`img_feats`/`radar_levels`/`depth`/`lidar2img`/`post_rots`/`post_trans`/`ori_shape`/`bda`/`points`）。因此本节的 10/10 只覆盖 CPU 契约，不覆盖真实数据接线。
- **无 GPU S0**：真实 batch/N4 下的 step-0 等价、显存/吞吐上限、真实零雷达回退、错误借点率均未测；**不得为放下面偷偷降 batch**。
- **高度先验风险**：三个 train-only 高度只是覆盖先验，坡道、高 Truck、遮挡仍是明确风险。
- **缺测≠周边无回波**：框内零雷达点不等于邻车无回波，zero 组 AP 可能仍下降，必须用离线诊断（错误借点率、四类、high/low/zero、FP/FN、径向/切向中心误差）验证。

### 95.5 结论与下一步

**结论：GP-MSF 首版实现完成并通过 CPU 接线门禁（10/10），唯一变量是融合层本身，step-0 与基线逐位一致，缺测回退与两步梯度均成立；但尚未接入真实管线、未做 GPU S0、无任何 AP 结论。**

下一步（按序，不改 A2-R/F1 排程）：

1. 在 `R4Det.py` 融合调用点接入 `geo`（原生 image 三级、radar 原生三级、预测 depth、标定/augmentation/ori_shape/bda/points），并加一个只切换 `RCFusion.type` 的 arm config。
2. 真实数据 GPU S0（仅用 GPU 5/6/7）：同 batch/N4 下 step-0 逐位等价、显存/步时 ≤1.2×、零雷达回退、地址有界、两步梯度；任何一项不过就停下改实现，不进训练。
3. S0 通过后才做冻结预登记与四臂（B0 / 同容量普通 residual conv / 同 source 固定几何采样 / 完整 GP-MSF），并按统一口径记录区间均值与全轮峰值。


## 96. z 的证据闭环与预测级四臂工具（2026-10-13，无新训练/val）

### 96.1 本任务范围与必填口径

用户要求不再停留于“不确定/没有证据”，明确怎样取得可归因的完整证据，以及怎样让 z 提供有效任务信息。
本任务交付**可执行的预测级入口 + 有终点的实验/改造契约**，不修改现役模型、config、权重、冻结门或训练队列。

- **区间均值**：N/A（本任务没有新增 val，ep12–16 n=0）。
- **全轮峰值**：N/A（本任务无新增 epoch/AP，因此没有可写的“指标值+epoch”）。
- **best saved**：N/A（未生成新 checkpoint）。
- 正式训练逐 epoch 曲线：本任务没有正式训练/新曲线；既有 A2-R 曲线仍归其运行节，本次不拼接或重新宣布成熟窗口。

CPU preflight 使用既有 A2-R seed0 `epoch_6.pth`；ep6 表示证据源权重，不是本次新训练。

### 96.2 交付与唯一干预

新增：

1. `me_rssm/sanity/paired_z_prediction.py`：依次运行 unwrapped reference、只读 identity、
   residual_off、当前帧 prior_only；调用真实 detector/decoder/KITTI evaluator，不自行实现 AP。
2. `me_rssm/sanity/test_paired_z_prediction.py`：11 条 CPU contract 单测。
3. `docs/rssm_z_evidence_contract_20261013.md`：逐框对账、匹配训练、任务读出与对象状态路线、排程/止损边界。
4. `docs/rssm_z_prediction_preflight_20261013.json`：真实 CPU preflight、输入/checkpoint/代码 SHA 与未执行项。

四臂固定 same checkpoint / val / N4 / eval / posterior mean / bs1。
residual_off 仅在部署 head 调用去掉直接 class residual；prior_only 仅把当前帧 RSSM 的 posterior
选择改为 prior，历史正常。所有改写均限诊断进程内 wrapper，异常退出恢复原方法；不写模型源文件。

工具冻结/复核输入、权重与 persistent buffers、关键源文件与 resolved config。
identity 必须和 reference 逐位同预测、同 AP。off 另断言全部 raw bbox/dir、IoU auxiliary、z、
检测特征与原 residual 不变；只允许 cls/候选筛选变化。prior 断言实际返回 z 等于该次 prior。
真实部署 top-k 候选逐帧落盘，包含 anchor ID、scores/codes/anchors/方向、direction-correction 前 decoded boxes。

`--limit` 禁止 AP，单独输出 `smoke_complete_no_ap`；正式 full-val 才记录全部 AP 与配对差。
四臂 full-val 共 8160 次 N4 clip 推理；耗时没有实测，不预报速度。

### 96.3 实际预检与回归证明

2026-10-13 本轮完成：

- **11/11 CPU tests PASS**：identity 输出与方法恢复、score-only 回归/方向/IoU 不变量、
  缺失历史 0/1/3 次情况下正确干预当前帧、错误 prior 未到 z 时拒绝、缺少调用拒绝、
  隔离不变量被破坏时拒绝、top-k 与 decoder 一致、candidate artifact 不改变输出、
  shape/dtype/content/order 指纹与未知输入类型拒绝、非法臂拒绝。
- 旧审计 **7/7 tests PASS**；新增两个 Python 文件语法检查 PASS，标准库环境 `--help` PASS。
- **真实 val pipeline CPU 预检 PASS**：`compat_cfg` 后构建，2040 样本，val/test 配置相同；
  第一批输入含 points/img/img_metas/gt_bboxes_3d/gt_labels_3d，完整输入指纹在 JSON。
- **真实 ep6 checkpoint strict load PASS**：全部参数在 CPU；标准 `SemanticReadoutRSSMFusion`
  + `Anchor3DHead`，真实 head `use_iou_branch=True`。小尺寸 2×3 CPU 特征下 on/off 的
  cls 不同、raw bbox/dir 与 IoU branch 输出逐位相同；resolved config dump PASS。
- GPU 5/6/7 仍由现役 A2-R 占用。本轮没有真实传感器 GPU forward、GPU S0、full-val、
  新训练或队列调整；**constructor 会初始化 CUDA context，不把模型构建误记成完全未接触 GPU**。
  preflight 的模型参数与 head 运算在 CPU，进程已退出；不存在本轮新增常驻 trainer。

预检发现并避免两处误用：

1. 原始 Config 的 val 内含 `samples_per_gpu`，未先 `compat_cfg` 会传给 dataset constructor 并报错；
   正式 runner 已先执行兼容转换（预检初次绕过转换的失败不计通过）。
2. 真实 A2-R 启用了 IoU auxiliary branch。不能将其误当成“不支持”而拒绝现役配置，更不能为
   跑工具偷偷关闭；runner 保持该分支并记录其指纹。当前实际 decode 没使用该辅助预测改分，
   部署 scores/top-k 仍按现有 `get_bboxes_single` 复现。

### 96.4 归因与设计决策，不把工具测试当机制成功

当前可确定的设计断点：独立辅助 sem_head 不部署、另一 1×1 直接读出，同格同类共享全部
anchor 增量，新增路径只改 cls。下一步必须直接评价实际候选的类别、几何与质量/排序，不能
继续把辅助 AUC、GT 格 CE、KL 激活或 z 张量变化当成检测成功。

干预需同时避免三种归因错误：

- off 固定全部 raw 回归候选，但最终框集合可以随 score/NMS 变化；那是选择效应，不是修框。
- off 不是去掉全 z；prior_only 仍保留 z、当前视觉/雷达旁路及当前 feat 条件的历史对齐，不是纯历史预测。
- 任一 frozen-weight 干预掉 AP 都不等于 stochastic 必要性；后者必须同容量、同目标、配对 seed 重训。

**完整闭环次序**：先 identity/干预证明与真实 full-val，再逐框 TP/FP/top-k/IoU/PR 对账；
确有必要才补梯度/几何 oracle；最后做无时序/同容量 deterministic/mean-only/完整 RSSM 的匹配控制。
通过后才将 latent 与真实任务 decoder 对齐；几何 fusion 的观测收益与 z 的机制收益分开消融。
真正对象级 `(h_i,z_i)` 依赖可靠对象观测/关联/birth，不重包装历史 dense heatmap 或未来模板失败臂。

### 96.5 未完成与下一步

**本轮工具就绪，证据尚未全部获得，z 总目标未解决**。未完成：真实 GPU 接线/identity、ep6 与
预登记 ep16 full-val、TP/FP/PR/oracle 分解器、top-k 前完整池上限、匹配多 seed 训练。
本工具只保存部署 top-k，不能据两个臂都淘汰的候选未落盘就宣称不存在几何召回空间。

执行入口、冻结判定与命令见契约 §6。建议 A2-R 正常结束后先留诊断窗口，F1 后置；
**这只是建议，本轮未撤掉、延后或另建自动队列**。当前原 F1 队列仍可能自动启动，改排程须显式批准。
TODO 的工具项标完成，预测级归因与 z 修复总项保留未完成。其他线程 GP-MSF 的未提交改动未编辑、未纳入本任务提交。

## 97. GP-MSF 真实管线接线、接线单测与训练队列替换（2026-10-13，无新训练）

### 97.1 范围、必填性能摘要与唯一变量

本节承接 §95（GP-MSF 首版实现 + CPU 门禁），记录 2026-10-13 的三件事：
**①`R4Det.py` 真实管线接线**、**②接线单测与判据脚本**、**③训练队列替换（F1 → GP-MSF）**。

- **区间均值 N/A（本次无新增 val；默认 ep12–16 无新样本，n=0）**；**全轮峰值 N/A（本次无新增 val 指标/epoch）**；best saved N/A。
- 本节所有数字都是 CPU 接线/环境核验值，不能当 AP 或成熟窗口；正式训练后必须按顶部统一口径补记。
- **唯一变量（相对 `..._no2d_igdr` 主线）**：融合层类型，即 `RCFusion.type` 由 `ConcatConvFusion`
  改为 `GeometryPairedMultiScaleFusion`；RSSM/head/anchor 头/优化器/数据/日程/batch/N4 全部不变，
  差异键机械核验为 `['RCFusion']`（单测 + S0-2 双重断言）。
- **本次仍未跑 GPU S0、未开训、无 AP**；GPU 5/6/7 自 2026-10-13 03:14 UTC 起由 A2-R 续训占用，未被打断。

### 97.2 本次产物与运行事实

| 项 | 内容 |
|---|---|
| 接线 | `mmdet3d/models/detectors/R4Det.py`：`extract_pts_feat` 在 **SECONDFPN 之前**缓存原生三层（`self._native_pts_levels`，`extract_feat` 入口复位、非 dict 分支清空）；融合调用点新增 `accepts_geometry_sources` 分支，仅该分支传 `geo=self._gpmsf_geometry(...)`，旧 `accepts_obs_valid`（F1）与两参数分支顺序不变 |
| 几何打包 | `R4Det._gpmsf_geometry`：单 view/多 view 形状归一化 + 硬断言，返回 `img_feats(P2/P3/P4)`、`num_views`、`radar_levels`（原生 SECOND 三层）、`depth`、`lidar2img`、`post_rots(2×2)`、`post_trans(2)`、`ori_shape`、`bda`、`points`；每帧按调用传参，不共享全局总线 |
| 模块加固 | `geometry_multiscale_fusion.py`：新增 `require_geo`（训练期未接线直接 RuntimeError，eval 保留 concat 回退）与 `last_stats` 每次 forward 复位（防探针读到上一帧陈旧统计） |
| arm config | `me_rssm/configs/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_gpmsf.py`（冻结几何拷贝自 resolved 主线配置） |
| 冒烟配置 | `me_rssm/sanity/gpmsf_smoke_450iter.py`（450 iter、KL 日程保留、`collect_stats=True`、`GPMSFWiringStatsHook`） |
| 探针 | `me_rssm/gpmsf_stats_hook.py`（每 50 iter 落 `gpmsf_wiring_stats.json`，异常不抛出） |
| S0 判定器 | `me_rssm/sanity/gpmsf_s0_verdict.py`（S0-1…S0-4 + `S0_PERF_REVIEW`，仅读产物） |
| 接线单测 | `me_rssm/sanity/test_gpmsf_wiring.py`（7 项） |
| 预登记 | `docs/gpmsf_prereg_20261013.md`（冻结：公式、几何、S0 门、ep6/ep12-16 判据、四臂控制、止损） |
| 队列 | `me_rssm/sanity/queue_gpmsf.sh`（替换 `queue_relfus.sh`，串在 A2-R 之后） |

运行事实（2026-10-13，CPU）：

- 接线单测 **7/7 PASS**（约 11 s）：仅 GP-MSF 声明 `accepts_geometry_sources`（基线/F1/C1 保持两参数调用）；
  `require_geo` 未接线训练期报错、eval 回退可用；臂配置单变量 + 冻结几何；冒烟配置门禁；
  `_gpmsf_geometry` 契约（含多 view 与形状拒绝）；**真实几何形状下 step-0 与同状态 `ConcatConvFusion` 逐位相同**。
- CPU 门禁 `gpmsf_cpu_checks.py` 复跑仍 **10/10**（含 t4 图像回读 6.0e-08 / 三高度 2.4e-07、t8 分块等价、t10 全部 60 张量有梯度）。
- 静态检查：`py_compile` 全部通过；按 `E1xx/E2xx/E3xx/E7xx/E9xx/F`（排除本仓库通行的 79 列 E501）对新文件 `flake8` 为空。

### 97.3 接线阶段发现并修复的真实缺陷（机制归因）

1. **pytest 在收集阶段整体失败（环境级，非测试逻辑）**：`hydra-core 1.4.0.dev6` 以 egg 安装并注册
   pytest11 插件，其 import 链要求 `omegaconf.vendor.antlr4`，而当前安装的 omegaconf 已不再附带
   `vendor` ⇒ 任何 `pytest` 调用都在收集前 `ModuleNotFoundError`，**7 项单测一项都跑不到**。
   处理：统一以 `python -m pytest -q -p no:hydra_pytest ...` 调用并写入队列脚本（附原因注释）。
   **若不定位这一层，会误判为「测试失败=实现失败」，实际是测试根本没执行。**
2. **`_gpmsf_geometry` 对 `post_trans` 末维校验缺失**：`(B,N)` 前缀断言无法区分
   `(B,1,3)`（合法单 view）与 `(B,1,5)`（错接线），错接线会静默通过并取前两维。
   已补 `post_rots` 末两维 ∈ {(2,2),(3,3)}、`post_trans` 末维 ∈ {2,3} 的显式断言，由单测覆盖。

（§95 已记录的 5 个模块实现缺陷——图像 v 方向按宽归一化、跨 view 计数聚合、offset 通道排布、
浮点 mask、3-D 张量进 Conv2d——本轮未复发。）

### 97.4 训练队列替换（用户 2026-10-13 指令：替换当前训练队列的 F1）

- 指令：**替换当前训练队列的 F1，在当前训练结束后直接接上**。据此把等待中的 F1 队列
  （`queue_relfus`，pid 1477871，自 2026-10-12 15:13 UTC 起 only armed、未产生任何训练产物）
  连同其自有会话组一起终止（≈2026-10-13 10:12 UTC），原 F1 工作目录仅剩 `queue_relfus.log`（80 B）
  与空的 `smoke_450iter/` ⇒ **没有丢失任何 val/权重/产物**；F1 实现与预登记全部保留，可随时重新排队。
- 新队列 `queue_gpmsf.sh` 于 **2026-10-13 10:12:25 UTC armed**（pid 2894048，PPID=1、独立会话，已脱离父进程）。
  串联逻辑：等待 A2-R 队列日志出现 `DONE` **且**进程表无 `semA2R_24e` → 接线单测 → 等 GPU 5-7 释放
  → 450-iter 冒烟 → S0 判定（FAIL 则不启动 24e）→ 24e 训练 → ep6 任务损害检查 → ep12-16 成熟窗口报告。
  运行目录 `/data/lurui/work_dirs/fgfull_N4_no2d_igdr_2x4_24e_gpmsf_seed0`，日志 `/tmp/gpmsf_24e.log`、
  `/tmp/gpmsf_smoke.log`、`queue_gpmsf.log`。
- 未触碰：A2-R 队列（pid 1354393）与 A2-R 续训（pid 2081375）。
- A2-R 续训当前状态（截至本次核验，2026-10-13 ≈10:1x UTC，运行 6.9 h）：已出 **val ep12**，
  还剩 ep13-16 四轮；续训段逐轮 vs B0：ep7 **+0.2855**、ep8 **−3.0564**、ep9 **−5.4383**、
  ep10 **−1.0955**、ep11 **+0.4887**、ep12 **−0.9527**（完整口径与结论见 §88，本节不重复判定）。
  ⇒ GP-MSF 预计在 A2-R 出 ep16 并写完报告后自动进入接线单测与冒烟。

### 97.5 未完成与下一步

**本轮完成的是「可运行 + 可判定 + 已排队」，不是机制成功**。未完成：

1. 真实数据 GPU S0（同 batch/N4）：step-0 等价性、显存与步时 ≤1.2×（超出记 `S0_PERF_REVIEW` 人工复核）、
   零雷达/全无效回退、地址有界、两步梯度、`proj_r2c/proj_i2r` 离零（S0-3）。
2. S0 通过后的 24e 正式训练与 ep6/ep12-16 判定；四臂控制（B0 / 同容量 residual conv / 固定几何采样 / 完整 GP-MSF）尚未实现后三者。
3. 高度 references 只有覆盖先验性；坡道、高 Truck、遮挡、以及「框内零雷达点 ≠ 周边无邻车回波」造成的
   错误借点率，必须用离线诊断给出，不能靠 AP 反推。
4. RSSM 主线关系仍需单独证明（2×2：`ConcatConv × GP-MSF` 对 `RSSM × 原 GRU`，比较 `(B−A)−(D−C)`）。

### 97.6 清理记录：删除被取代的中间 CPU 产物（2026-10-13）

**动作**：按用户指示删除未跟踪目录 `docs/artifacts/` 及其唯一文件
`docs/artifacts/geometry_multiscale_fusion_design_cpu_20261013.json`（1,912 B，mtime 2026-10-13 04:53 UTC，
删除前 sha256 `080f6acb260f8887bacde6770522869d9f7df6feeb59d282a39aff154d44487a`）。该目录现已不存在。

**为什么删**：它是 §91 设计阶段的**早期/被取代**推导，与随后入库的完整 audit 有两处实质差异：

| 项 | 被删文件（04:53） | 入库并实际使用（`docs/geometry_multiscale_fusion_audit_20261013.json`，05:53） |
| --- | --- | --- |
| 高度参考 | `[-1.1266211569, -0.3083836436, 0.6394495964]` | `[-1.1230436265, -0.3003779650, 0.6601446867]`（= arm `height_refs` / 模块 `DEFAULT_HEIGHT_REFS`） |
| 规则 | 等类质量 → 等对象等 surface，**先剔除出界 surface 再归一** | 同族规则 + **xy-grid filtering follows dataset BEV range** |

拟合种群两者相同（5706 帧 / 23,029 个格内框；四类 3,242 / 5,125 / 11,030 / 3,632）
⇒ 差异只在高度规则，**不进图、不被任何 arm 使用**。保留它的唯一实际风险是把过期常数误当现行证据。

**为什么不加 `.gitignore`**：本仓库 `.gitignore` 只列「绝不入库」类（`*.pth`/`*.log`/`data/`/`__pycache__` 等），
而审计/CPU 证据 JSON 在本仓库是**入库并写进训练记录**的（`docs/` 下现有 3 个、共 556 KB）。
把 `docs/artifacts/` 忽略等于在文档树内部开盲区：之后丢进去的文件会静默不上报，与 AGENTS 第 4 条的记录要求冲突。

**代价与边界**：该文件从未入库，git 无法恢复其内容；权威数据仍由已入库的完整 audit 提供，
如需该早期推导可用同一规则重跑。**本次只做文件清理，未改任何代码、config、arm、队列或训练状态；
A2-R 续训与 GP-MSF 队列不受影响。** 相关历史表述（设计文档 §12、本节 §91）已就地追加状态注记，不改写原文。


## 98. GP-MSF 排队脚本/config 审查与 fail-closed 修复（2026-10-13）

### 98.1 任务、统一口径与边界

用户要求 review 正在排队的新 fusion 训练脚本与配置，不符合直接修改。
**本节是代码/CPU/真实数据加载审查，不是训练结果**；GP-MSF 尚无 GPU forward、DDP smoke、正式 val。

| 口径 | 本任务结果 |
| --- | --- |
| 区间均值 | **N/A**：GP-MSF ep12–16 尚无 val，实际 n=0；不能把 CPU 探针当 AP |
| 全轮峰值（指标值+epoch） | **N/A（无任何 GP-MSF val epoch）** |
| best saved（checkpoint 可用性） | **N/A（没有 GP-MSF 训练 checkpoint）** |
| 完整逐 epoch / 逐类 / 窗口逐项 / 动力学 | **尚无训练曲线，不存在可填数值**；已给正式流程完整输出入口，后续不得用本节审查摘要替代训练记录 |

A2-R 原有训练不是本任务的新 GP-MSF AP；没有抢 GPU 5/6/7，也未终止/重启现役 A2-R。

### 98.2 配方与唯一变量核验

- 正式 config：`me_rssm/configs/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_gpmsf.py`；
  对 B0 `configs/r4det/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr.py`，model 差异严格只有 **RCFusion**。
- 非 model 配方全相同；仅 `_cfg_dir` / `custom_imports` 是 bootstrap 元数据差异，不是数据/solver 改动。
- 原配方：**bs2/rank、3 rank、N4、梯度累积2、24e、AdamW lr=0.00015、betas=(.95,.99)、wd=.01、grad clip35**；
  cosine24e、val每轮、checkpoint每2轮、seed0/deterministic、KL ep0–10 线性0→1、预训练 `checkpoints/pretrained_tj4d.pth`、不 resume。
  RSSM/head/FG/损失/augmentation/图像分辨率均未改；不降 batch 或分辨率换取跑通。
- GP-MSF 几何、三个 train-only height references、64d/4 heads/4 offsets、query_chunk4096、容量保持冻结：
  新增 **177,726**，共享 fusion_conv **1,475,072**。本轮只补实现正确性与实验控制，未叠新学习模块。
- 科学 hypothesis/消融仍沿预登记：新增同地址原生多尺度取证是否提高观测质量；
  B0/同容量 residual conv/固定几何采样/完整 GP-MSF 对照仍待执行，不凭代码或通路有梯度宣称 AP/新颖性成立。

### 98.3 发现的问题与修复

| 缺陷（修复前） | 责任层与修复 |
| --- | --- |
| `collect_stats=True` 把 checkpoint 关闭；冒烟与正式训练执行图、显存不一致 | fusion 改为 training且grad enabled时 checkpoint，统计不禁用；每第50iter收集，按 query 数加权各 chunk（避免尾块占比虚高） |
| M=0 但训练后 projection bias≠0，缺雷达仍改相机/全无效图像仍改雷达 | `_chunk` 返回实际支持/可见性 mask，对**完整投影（包括 bias）**逐 query gate；不删除 bias，不改冻结容量 |
| `require_geo=True` 只保护 train，eval 未接线偷偷回退 baseline | train/eval 都严格报错；只有显式 require_geo=False 可做 concat 调试 |
| 新模块只保存/恢复 CPU RNG，但 `torch.manual_seed` 还会重设 CUDA seed | 仅 seed CPU default generator，再恢复 CPU state；真实 GPU preflight 必查全 RNG，共用初值 CPU 实测一致 |
| 多 rank 写同一个 stats JSON，异常被吞、缺文件仍 S0 PASS | rank0 原子 replace；必需诊断缺失报错；after_run 三 rank 对全部 fusion 参数 SHA 一致才完成；原生源 requires_grad 与 grad_fn 均需成立 |
| GPU drain 超时/查询失败仍启动，450权重写出就继续、不等退出、旧文件复用 | query/超时 fail-closed；新 attempt 目录；自有 setsid 会话、wait exit=0、无遗留 ranks、phase/hash/新鲜度证明，不复用旧 checkpoint |
| 只测 NaN/最后一行字段；Inf/早期消失字段/少一条projectionweight可 PASS | 全日志递归 numeric finite、checkpoint全部state finite，两条 weight及bias均必须存在且有限、两条weight都离零；实际checkpoint config/seed/ranks/配方校验 |
| 用 C1/DC 不同RSSM资源中位数，超预算只告警 | 新增**同配方 B0 450iter**；两臂串行同 GPU5/6/7/bs/N4/累积/solver；排除前50iter，步时与峰值内存各≤1.20×，缺数据/超限均不放行24e |
| 未做设计要求的真实 train/eval step0、augmentation、两步梯度就开训 | 新增可运行 `gpmsf_real_preflight.py`：同加载初值/RNG、真 train loss/head / val预测/head、真实meta独立投影、已非零bias下缺测/部分batch/无图像回退、两步梯度；先GPU5串行执行，不报AP |
| ep6失败只flag；C1专用绝对阈值被套用；ep16后无条件DONE但24e仍跑 | GP-MSF专用完整窗口报告；ep1–6相对B0低超过2点操作性停并复查，不等同形态否定；ep16只报告，真正exit0+ep24权重+完整ep1–24后DONE |
| 正式峰值/完整曲线可能依赖最新log或saved epoch | 合并所有run log中的val，保存全部epoch/指标、完整窗口逐项mean、所有val峰值“value+epoch”、独立best_saved；不套C1≥40门 |

补充本机运行时事实：MMCV `IterBasedRunner.save_checkpoint` 在 post-step 自增前保存，因此 **iter_450.pth 的 meta.iter=449**，
日志/after_run完成计数是450。已用本机 runner 的实际 CPU 保存调用验证，不误要求meta.iter=450。
另外 runner.meta.config 在 cfg.seed赋值前取快照，seed须读checkpoint.meta.seed；实际配方比较先对两侧 compat_cfg 规范化。

### 98.4 证据（红测试、绿测试、真实 CPU 数据）

- 修复前两条回归均 **FAIL**：`test_stats_do_not_disable_training_checkpoint`（calls=[]）；
  `test_trained_projection_bias_cannot_break_missing_evidence_fallback`（零雷达时 concat 前 camera 被 bias 改写）。
- 修复后：`test_gpmsf_wiring.py` + `test_gpmsf_queue_checks.py` **37 passed**（10接线+27队列/证据回归）。
  覆盖 NaN/±Inf/消失字段、完整窗口、所有日志并集、unsaved峰值、phase成功/旧产物/源漂移、
  缺projectionweight/缺资源/缺探针、原始标定含第4列+BDA affine、真实BitmapMasks输入hash、
  ep6 −2边界、1.20×时间/内存边界、本机MMCV零基checkpoint、rank0原子写与实际owned launcher退出/超时清理。
  其中合成artifact测试只验证判定逻辑，**不是实际三rank训练证据**。
- `gpmsf_cpu_checks.py` **10/10 PASS**：step0、RNG、六源索引/投影、两类缺测、两步梯度、chunk等价、有界、checkpoint反向。
  Toy offsets 第一步grad=0，第二步grad≈3.39；keys≈0.872，level weights≈0.471；不外推真实任务效应。
- 实际整模型 **CPU build/init_weights/load_checkpoint**：标准 B0 与 GP-MSF 的全部共用state SHA、CPU RNG SHA相同，projection仍零。
  原生全配置 freeze_images/depths/radars 均False；训练KL scheduler保持0→1，真实preflight额外显式应用epoch0 hook。
- 真实 CPU train loader（RepeatDataset共11412条，底层5706帧）：bs2/N4；batch SHA
  `2c38a810654bf0e25c836d6b5b5bfb9877e538a888c5801e6c29cdf1c94607a1`。
  真 val loader2040帧、bs1：batch SHA `4c47f049e8c6029e5d523357bfbfa73b79a6e86caabc4ac0370420f98ca928c3`。
  训练标注 BitmapMasks 必须按 masks/height/width 实值 hash；未知对象不降成地址/pickle hash。
- 实际 CPU train augmentation/meta对输入点独立重投影：两样本max error **0.000122/0.000183 px**；
  val样本 **0.0 px**。本批次cam2img第4列为0，另有非零第4列/stereo单测；**未据此修改原BEV augmentation**。
  实测当前TJ4D eval.py/kitti_dataset.py内容与train_vod将copy的两个TJ4D模板一致，未改现役源码。
- `bash -n`、`py_compile`、新增/重写入口 `flake8 --select F,E9`、`git diff --check` 均通过。
- 留存：`docs/gpmsf_queue_review_20261013.json`；完整本轮诊断日志复制至GP-MSF work-dir的 `review_20261013/`，不混入正式曲线。

可复跑核心门（使用r4det环境，Hydra插件须禁用）：

```bash
rtk proxy env CUDA_VISIBLE_DEVICES=5 PYTHONWARNINGS=ignore /home/lurui/envs/miniforge3/envs/r4det/bin/python -m pytest -q -p no:hydra_pytest me_rssm/sanity/test_gpmsf_wiring.py me_rssm/sanity/test_gpmsf_queue_checks.py
rtk proxy env CUDA_VISIBLE_DEVICES=5 PYTHONWARNINGS=ignore /home/lurui/envs/miniforge3/envs/r4det/bin/python me_rssm/sanity/gpmsf_cpu_checks.py
rtk proxy bash -n me_rssm/sanity/queue_gpmsf.sh
```

### 98.5 运行事实与预登记勘误

- 审查期间只暂停等待中的旧GP-MSF queue parent **2894048**（PGID同值）；必须终止此旧等待会话并重新armed，
  **不能只SIGCONT旧bash**：已运行的bash不会自动换成修改后的执行逻辑。
- A2-R queue **1354393**、trainer group **2081375** 始终保留；只CPU验证，不抢5/6/7训练槽位。
- 新 queue、替换时间、共用CPU初值SHA与存档日志索引见本节审计JSON与下方收尾事实。
- 预登记 `docs/gpmsf_prereg_20261013.md` **v1.1 首次GPU前勘误**：纠正资源软门/缺证据放行/epoch索引等控制缺陷，
  AP −2止损和≥+2三seed候选目标不变；配置容量与科学假设不变，不看结果调阈值。

收尾核验：**2026-10-13 12:00:26 UTC**，旧等待会话2894048已终止；新 queue **PID/PGID 3066450/3066450**，PPID=1，日志已写“armed (reviewed gates); waiting for A2-R”。
A2-R 1354393/2081375仍运行；没有启动GP-MSF GPU任务。不是恢复旧bash，确实重新加载新版脚本。
共用CPU初值SHA `a3d7fb5e75874f7b00c066aa5e6fee5a4ccdd529036710bcdc30f3156c49c675`；CPU RNG SHA `f4f52925e3d141bca12335335f85acfbdb0acbcdaedf59df31e73bf166105c41`。

### 98.6 结论与下一步

**原排队版不符合设计/记录要求；已修到可执行、可否证、失败不放行。CPU验收通过，不等于S0全通过。**

接下来保持获批排序：A2-R完整收尾 → 真实GPU preflight → B0/GP-MSF两次450iter DDP → S0所有硬门 → 24e。
任何数值/接线/缺测/同步/资源失败都先停并归因，不偷降batch、不调阈值、不用不同RSSM冒烟替代。
成熟窗口只能先给单seed方向证据；逐类/coverage/FP-FN/借点率、控制臂、多seed及RSSM h/z排他仍待补，不能靠attention图或梯度证明AP改善。

### 88.1 A2-R 成熟窗口判定（2026-10-13 13:26:55，n=5 完整）

| 指标 | A2-R ep12-16 | B0 seed0 | Δ | 冻结线 | 判定 |
| --- | --- | --- | --- | --- | --- |
| **Overall** | **38.3778** | 39.2809 | **−0.9031** | ≥ 40.00 | **FAIL（差 1.6222）** |
| Car | **49.7338** | 47.8399 | **+1.8938** | ≥ 47.34 | PASS |
| Truck | 26.8288 | 29.7246 | **−2.8959** | ≥ 29.22 | **FAIL**（且 < 27.7246 = B0−2.0 ⇒ **止损**） |
| Pedestrian | 28.6117 | 30.9277 | **−2.3160** | ≥ 29.93 | **FAIL**（且 < 28.9277 ⇒ **止损**） |
| Cyclist | **48.3369** | 48.6313 | −0.2944 | ≥ 47.63 | PASS |

- 三口径（scope ep1-16，含 A 父臂 ep1-6）：区间均值 **38.3778（ep12-16，n=5）**、
  全轮峰值 **39.4519 @ ep11**、best saved **39.4322 @ ep16**（saved = 2..16）。
- 运行事实：13:26:55 在 ep16 后由队列停训（config 保持 24e 以维持日程连续性），
  报告 `merged_val_curve_report.json`（含逐 epoch、窗口逐类、vs B0 同窗 delta）。
- **结论**：A/A2-R 这条线关闭——ep6 机制门 NOT ALL PASS（§74/§75）+ 成熟窗口 FAIL。
- **与 C1 的对照（可写进论文 Analysis 的机制观察）**：两条臂的收益都集中在 **Car**
  （C1 另加 Truck），代价都在 **Ped/Cyc**；C1 的窗口 Overall 是 +0.55（Ped/Cyc −4.9/−6.0），
  A2-R 是 −0.90（Truck/Ped −2.9/−2.3）。⇒ 本项目两条「z 语义/状态修正」路线
  **共用同一个失败形状：跨类重分配，而非整体增强**。
- 边界：单 seed；上述均为训练内 EvalHook 口径（与 B0 同口径可比），harness 口径数值另见 §87.3。


## 99. GP-MSF 首次冒烟只读诊断：消息空间轴混排、资源超限与 null 分母（2026-10-13）

### 99.1 范围、口径与运行事实

- 本轮仅诊断、CPU 反例取证及文档归档；**未修改研究 Python/config、未停止或重排现役队列、未启动额外 GPU 任务**。
  CPU 探针因仓库 localattention 的 import 断言而设置 `CUDA_VISIBLE_DEVICES=5`，但所有张量/运算均在 CPU，
  **`torch.cuda.is_initialized() == False`**。转置修正只存在于独立 CPU 进程内，**不是仓库已修复**。
- 源版本：`140a46a418cfcef71037a7187e18cfcb8ee4a9cd`；GP-MSF 现役模块/队列沿用 `0413abd`。
  14:51:02 UTC 快照核验全部 preflight `source_sha` 仍匹配；文档不在源冻结集合中。
- 队列 PID/PGID **3066450**；GP-MSF trainer PGID **3151027**，ranks **3151037/3151038/3151039**。
  冒烟目录：`/data/lurui/work_dirs/fgfull_N4_no2d_igdr_2x4_24e_gpmsf_seed0/s0_20261013T132727Z_3066450`。
- 真实 GPU preflight 13:27:53–13:28:53 完成，16 个既定检查均为 true；配对 B0 冒烟
  13:28:54–13:41:57 完成且有 `iter_450.pth`；GP-MSF 冒烟 13:41:58 启动，
  **本节快照最后已记录 iter350（14:43 左右），尚无 iter450 checkpoint 或最终 S0 verdict**。
  不将过程快照写成已完成；结束证据须另补。
- 唯一研究变量仍为 RCFusion；bs2/rank、DDP3、N4、梯度累积2、seed0、共同 RSSM/head/数据/损失不变。
  当前模块 embed_dim64、4 heads、4 offsets、3 image/radar levels、3 heights、query_chunk4096。
- **主指标区间均值：N/A（默认 ep12–16 未形成，val n=0）；全轮峰值：N/A（无 val，epoch=N/A）；
  best saved val：N/A。** 450iter 权重只证明冒烟产物存在，不用于 AP 排名。本轮不是正式 24e。
- 原始证据与完整已记录训练标量/接线 stats：`docs/gpmsf_s0_diagnosis_20261013.json`。

### 99.2 配对资源与完整已记录冒烟曲线

资源沿用现役 S0 算法：丢弃 iter50 行，后续日志区间 time 等权均值；每行 time 是此前50步的日志区间均值，
不是9个独立单步测量。B0 iter100–450，n=8；GP-MSF iter100–350，n=6，**臂尚未完成**。

- B0：**1.550530 s/iter、16959 MiB**。
- GP-MSF 快照：**10.409717 s/iter、17881 MiB**。
- 比值：time **6.713651×**；memory **1.054366×**。
- time/memory 硬上限仍为 **1.20×**；B0对应总步时预算 **1.860636 s/iter**。
  不放宽门、不自动开24e。当前耗时超限成立，但不是正确实现 GP-MSF 的最终成本下界。

下表是该快照全部已记录 train 区间；迭代不是 epoch，不补造 val 曲线。
完整 loss_cls/loss_bbox/loss_dir/loss_iou/depth/mask/range/RSSM 及其 stats 均保留在上述 JSON 的
`paired_smoke.*.all_logged_train_rows`，原 work-dir `.log.json` 不变。

| arm | iter | time s | data_time s | memory MiB | loss | grad_norm | loss_rssm_kl | clamped_ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| B0 | 50 | 1.57680 | 0.10482 | 16947 | 7.58060 | 10.55223 | 2.41197 | 0.86731 |
| B0 | 100 | 1.54560 | 0.06642 | 16947 | 4.94588 | 13.92974 | 1.10635 | 0.88681 |
| B0 | 150 | 1.51876 | 0.06022 | 16959 | 4.32818 | 20.45064 | 1.05607 | 0.97650 |
| B0 | 200 | 1.53754 | 0.06819 | 16959 | 4.02451 | 22.49304 | 1.04498 | 0.98313 |
| B0 | 250 | 1.58415 | 0.06394 | 16959 | 3.79865 | 19.85069 | 1.03212 | 0.98857 |
| B0 | 300 | 1.54013 | 0.07236 | 16959 | 3.70333 | 17.12093 | 1.02553 | 0.98727 |
| B0 | 350 | 1.52096 | 0.06354 | 16959 | 3.55117 | 16.09436 | 1.02348 | 0.98776 |
| B0 | 400 | 1.56351 | 0.07666 | 16959 | 3.49889 | 16.56034 | 1.02362 | 0.98711 |
| B0 | 450 | 1.59359 | 0.07230 | 16959 | 3.52183 | 16.13027 | 1.01953 | 0.98767 |
| GP-MSF | 50 | 10.36165 | 0.09972 | 17869 | 7.59236 | 12.11069 | 2.41566 | 0.85422 |
| GP-MSF | 100 | 10.33859 | 0.06967 | 17869 | 5.03307 | 12.39700 | 1.10566 | 0.94050 |
| GP-MSF | 150 | 10.52174 | 0.06401 | 17881 | 4.26969 | 18.00745 | 1.04976 | 0.97322 |
| GP-MSF | 200 | 10.33527 | 0.06344 | 17881 | 3.96108 | 21.66006 | 1.03483 | 0.98481 |
| GP-MSF | 250 | 10.40171 | 0.06736 | 17881 | 3.75243 | 18.77791 | 1.02576 | 0.98912 |
| GP-MSF | 300 | 10.42523 | 0.07755 | 17881 | 3.69043 | 16.68930 | 1.02078 | 0.99026 |
| GP-MSF | 350 | 10.43576 | 0.06941 | 17881 | 3.54641 | 16.09732 | 1.01751 | 0.99050 |

### 99.3 P0 确诊：消息合并把 query/channel 轴混排

位置：`mmdet3d/models/fusion_layers/geometry_multiscale_fusion.py:697–698`。
每个 head 的候选聚合结果是 `(B,qc,dk)`；stack 后为 `(B,heads,qc,dk)`。
当前直接 reshape 成 `(B,embed_dim,qc)`，**把 query 位置与 head 内 channel 混在一起**。
正确语义须先置换为 `(B,heads,dk,qc)`，再合并 heads 与 dk。

```python
# 当前：错误；两方向都存在
m_r = torch.stack(m_r, dim=1).reshape(b, self.embed_dim, qc)
m_i = torch.stack(m_i, dim=1).reshape(b, self.embed_dim, qc)
# CPU 独立进程中验证的修正；尚未写回现役源码
m_r = torch.stack(m_r, dim=1).permute(0, 1, 3, 2).reshape(b, self.embed_dim, qc)
m_i = torch.stack(m_i, dim=1).permute(0, 1, 3, 2).reshape(b, self.embed_dim, qc)
```

- 按当前 qc4096/dk16，输出单 head 的 channel `j`、query `p` 对应旧消息的
  `query=floor((j*4096+p)/16)`、`channel=(j*4096+p)%16`，一般**不是 query=p/channel=j**。
  query0 的64维向量也不再是同一query的地址集合对应的64维证据。
- valid mask 本身按正确 query 顺序构造；投影前消息已经错排，因而
  「整投影按 valid gate」仍不能把错误内容恢复到正确位置。梯度可达也不证明空间对应正确。
- **因此 query_chunk 在当前实现中会改变消息的空间重排，而不只是资源开销**；
  未修前直接4096改整块，不能视为机制等价提速。
- CPU 反例（seed712、B2、embed_dim64/heads4/dk16、24 queries，source为原有 TOY contract）：
  同权重、同输入，**输出投影显式非零**，chunk24 对 chunk5：
  - 原输出 max|Δ| **0.3601186872**，allclose **False**；
  - 原直接消息 M_R max|Δ| **0.6087927818**，M_I max|Δ| **0.6811844707**；
  - 只在独立进程修上述两处转置后，输出 max|Δ| **0.0**，allclose **True**；
  - 原整块输出与修正整块输出 max|Δ| **0.3975334167**。
  不是随机输入差异，也不是 offset/level/capacity 的结构修改。
- 漏检原因：`me_rssm/sanity/gpmsf_cpu_checks.py:356–365` 的 T8
  用 `build(query_chunk=...)`，未开 `nonzero_out=True`；两方向 output projection 均为0。
  该断言只比较共同 ConcatConv 输出，错误消息被零残差遮住。
  step0等价门仍有价值，但不能代替投影离零后的空间接线门。

最小 CPU 复跑（环境依赖同既有 CPU 检查；仅 import 需要看到GPU，不调用 CUDA）：

```bash
rtk proxy env CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONWARNINGS=ignore /home/lurui/envs/miniforge3/envs/r4det/bin/python - <<'PY'
import inspect, runpy, textwrap, types
import torch
from mmdet3d.models.fusion_layers import geometry_multiscale_fusion as module
checks = types.SimpleNamespace(**runpy.run_path('me_rssm/sanity/gpmsf_cpu_checks.py'))
torch.set_num_threads(1)
torch.manual_seed(712)
img, rad = checks.bev_inputs()
geo = checks.make_geo()
a = checks.build(nonzero_out=True, query_chunk=0).eval()
b = checks.build(nonzero_out=True, query_chunk=5).eval()
b.load_state_dict(a.state_dict())
with torch.no_grad():
    original_a, original_b = a(img, rad, geo), b(img, rad, geo)
print('original max delta', float((original_a-original_b).abs().max()))
source = textwrap.dedent(inspect.getsource(module.GeometryPairedMultiScaleFusion._chunk))
for name in ('m_r', 'm_i'):
    old = f'torch.stack({name}, dim=1).reshape(b, self.embed_dim, qc)'
    new = f'torch.stack({name}, dim=1).permute(0, 1, 3, 2).reshape(b, self.embed_dim, qc)'
    assert source.count(old) == 1
    source = source.replace(old, new)
namespace = dict(vars(module))
exec(compile(source, '<in_memory_layout_repair>', 'exec'), namespace)
for fusion in (a, b):
    fusion._chunk = types.MethodType(namespace['_chunk'], fusion)
with torch.no_grad():
    repaired_a, repaired_b = a(img, rad, geo), b(img, rad, geo)
print('repaired max delta', float((repaired_a-repaired_b).abs().max()))
assert not torch.cuda.is_initialized()
PY
```

### 99.4 null 高的事实与不能越界的解释

当前 `radar_support_rate` 是4候选的平均有效率；`null_rad_weight` 是 head/query 的平均 null 权重。
`level_rad_weight` 包含无回波但仍在 native map 内的候选；不是 GT 或 supported-query 条件层选择。
`level_img_weight` 跨level求和则等于有效图像 candidate-height 比例，因为level掩码相同，失效时softmax全0。

设支持候选率为 s、有至少一个可用雷达候选的 head/query 率为 v、全画布非null权重为 a。
K=4，故 **s≤v≤min(4s,1)**，另有 **a≤v**；无可用候选的 query null=1，
可用 query 上的平均 null 是 **1−a/v**，不是全画布 null。

| iter | supported candidate % | global null_R % | valid query upper % | conditional null_R upper % | image candidate-height valid % |
| --- | ---: | ---: | ---: | ---: | ---: |
| 50 | 2.2543 | 97.5330 | 9.0173 | 72.6410 | 14.5169 |
| 100 | 3.3183 | 96.2569 | 13.2731 | 71.7991 | 10.4020 |
| 150 | 2.1752 | 97.5703 | 8.7009 | 72.0748 | 18.7883 |
| 200 | 1.2899 | 98.4439 | 5.1598 | 69.8414 | 6.5318 |
| 250 | 2.8952 | 96.6490 | 11.5806 | 71.0638 | 15.3188 |
| 300 | 1.9703 | 97.5562 | 7.8813 | 68.9923 | 1.5786 |
| 350 | 2.2026 | 97.3795 | 8.8105 | 70.2566 | 9.5224 |

**允许结论**：全画布很多 query 没有新增雷达候选；仍在为大量失效位置计算取证，值得优化。
**不允许结论**：「有证据的目标也97%拒绝雷达」「只2–3%目标可获益」「层均匀证明机制无效」。
这些需要 available-query/GT footprint或中心邻域的条件统计，当前 stats 无此分母。
上述bounds同样**不证明机制有益**；空间混排使得实际注入证据另有缺陷。

### 99.5 成本归因边界与执行顺序

1. **先修正确性，再测效率**：修转置；把非零残差、直接 M_R/M_I、完整输出、非整除chunk、
   spatial impulse 与输入/参数梯度一致性加入回归。保留step0等价与缺测严格identity。
   错排450iter checkpoint不得冒充正确实现的正式起点；正确实现须新attempt。
2. **资源超限已实测，纯compute归因未实测**：`tools/train_vod.py:12` 全局开启
   `torch.autograd.set_detect_anomaly(True)`；B0/臂均开启，但臂新增大量autograd节点，
   需要同输入/初值的paired replay拆解其成本，不能因两臂开关相同就排除差异开销。
   同样须拆 kernel launch/host dispatch、grid_sample backward、memory traffic、checkpoint重算、stats同步。
   按源代码，有depth时单次fusion前向14 chunks×4 heads×9次grid_sample=504调用；
   这只是调用数，不是耗时证据；N4/反向总量不得据此直接外推。672次标量同步/forward不是已实测数。
3. **保持机制优化优先**：head向量化、共享可复用计算；4096/8192/16384受控测输出/梯度/峰值显存。
   精确失效mask可用于跳过相应计算；不能只算有雷达query而删camera-only fallback，
   不能把GT「近目标」用作forward routing。减少offset/level属于新方法变量/消融，不包装成等价提速。
4. **补真正机制证据**：available-query条件null；GT目标/背景、zero/low/high radar、类别/range分层；
   support coverage、同对象/邻对象借点、残差/原feature范数、offset饱和、条件尺度选择。
   GT仅用于离线诊断，不进入模块输入。
5. 上述完成后重跑同协议S0；time/memory ≤1.20×门不变，失败停在人工复核，不直接上24e。
   通过后才按正式24e完整逐epoch/逐类、ep12–16区间均值与全轮峰值评估AP。

**结论**：首次实现存在已复现的空间接线缺陷，资源也超预算；当前只能判
「这一版不具备正式开训资格」。不能判正确实现的GP-MSF方法/AP失败，也不能据全局null宣判目标缺证据。
本轮记录的是诊断完成，**不是修复、GPU S0或方向1完成**。


### 99.6 收尾补证：450iter 完成，S0 资源硬门实际拒绝（2026-10-13 15:03:05 UTC 核验）

本节补齐99.1–99.4保留的14:51过程快照，**不覆写早先的取证时间**。
GP-MSF phase `finished_at` 为 **2026-10-13 15:01:01 UTC**、exit_code=0；
`iter_450.pth` 已保存，ranks已退出。正式epoch权重与`train_gpmsf.log`均不存在，24e未开。
现役queue最终写 `FAILED exit=1; no automatic continuation`，没有人为停训练或放宽门。

- 最终配对资源窗口两臂均 iter100–450、n=8：B0 **1.550530 s/iter、16959 MiB**；
  GP-MSF **10.411050 s/iter、17881 MiB**；time **6.7145105×**，memory **1.0543664×**。
- `s0_verdict.json`：**S0_ALL_PASS=False**，failure为
  `ValueError('matched resource budget exceeded 1.20x; manual review required')`。
  这是既有资源门的实际拒绝；99.3消息轴bug是本轮另补的独立反例，原验收尚无有效测试检测它。
- wiring stats含全部 iter50/100/.../450；每点当前native三层image/radar都requires_grad且有grad_fn，checkpoint_active=true。
  DDP world=3、全部参数SHA一致：`eb32766eacd61c8a9aed7a1e6f0dc4aa685f868b6ecc4451e3ee194ab34df441`。
- 两方向投影已离零：weight abs_mean分别 R→C **0.00126010**、I→R **0.00148910**；
  所以不能把99.3的错误当作正式冒烟中「投影一直是零，所以无影响」。
- iter450全画布支持候选率 **1.7753%**、null_R **97.9789%**、null_I **89.0004%**；
  radar level **0.344551/0.328836/0.323057**。仍只是背景混合均值，不补造目标分层或AP结论。
- **主指标区间均值仍N/A（ep12–16未形成，val n=0）；全轮峰值仍N/A（无val，epoch=N/A）；
  best saved val仍N/A。** 最终完整train标量、wiring、verdict摘要已补入JSON的`completion_check`。
  历史过程快照与最终值可同时追溯。

补齐最后两行冒烟曲线（其余行见99.2；完整loss/stats另见JSON）：

| arm | iter | time s | data_time s | memory MiB | loss | grad_norm | loss_rssm_kl | clamped_ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GP-MSF | 400 | 10.38816 | 0.07264 | 17881 | 3.50074 | 16.60701 | 1.01770 | 0.98968 |
| GP-MSF | 450 | 10.44194 | 0.06885 | 17881 | 3.51225 | 15.97781 | 1.01509 | 0.99097 |


---

## 100. GP-MSF 消息空间轴 P0 修复与漏检单测补齐（2026-10-13，纯 CPU，无新训练）

### 100.1 范围、统一口径与运行事实

- 本轮只改研究代码与 CPU 门禁三处，**未启动 GPU、未动队列、未改 config、未产生任何训练产物**：
  `mmdet3d/models/fusion_layers/geometry_multiscale_fusion.py`（2 行修复）、
  `me_rssm/sanity/gpmsf_cpu_checks.py`（新增 T11/T12/T13 及两个探针辅助函数）、
  `me_rssm/sanity/test_gpmsf_wiring.py`（新增 1 项 P0 回归，队列 cheap gate 内）。
- **主指标区间均值：N/A（ep12–16 未形成，val n=0）；全轮峰值：N/A（无 val，epoch=N/A）；
  best saved val：N/A。** 本轮不是正式训练，也不是 S0 复跑，任何 AP/资源结论都不在此节。
- 唯一研究变量（相对 §99 所记录的现役模块）：**消息合并的轴顺序**。容量（embed_dim64/4 heads/4 offsets/
  3 image+3 radar levels/3 heights）、地址上界、null/fallback、RSSM/head、数据与损失全部不变。
- 上游诊断与 CPU 反例来源：§99.3（`b1fe04d`）；本节只记录**写回源码**与**让门禁能测到它**。

### 100.2 修复内容与影响面

- `_chunk` 末尾，每个 head 的候选聚合结果是 `(B,qc,dk)`，`torch.stack(dim=1)` 后为 `(B,heads,qc,dk)`。
  原实现直接 `reshape(b, embed_dim, qc)` 会把 **query 轴与 head 内 channel 轴按行主序交错**，
  每条消息落到错误的 BEV cell，且错排模式**随 query_chunk 大小改变**。修复为先 `permute(0,1,3,2)`
  合并 `(heads, dk)` 再 reshape，`M_R` 与 `M_I` 两处同时修：

```python
# 修复后（两方向）
m_r = torch.stack(m_r, dim=1).permute(0, 1, 3, 2).reshape(b, self.embed_dim, qc)
m_i = torch.stack(m_i, dim=1).permute(0, 1, 3, 2).reshape(b, self.embed_dim, qc)
```

- 影响面：仅该模块内部的消息布局。`valid_r/valid_i` 门控（整投影含 bias）、`fusion_conv`、
  step-0 零初始化契约、`_build_context`/`_image_terms`/`require_geo` 均未改动；
  模块对外 shape/接口不变，因此 config 与 DDP 侧无需改动。
- 计算量不变：该修复只是轴置换，**不改变算子数量与内存量级**；§99 的 6.7145×/1.0544× 读数
  仍属错排旧实现，成本拆分（用户执行顺序第 2 步）尚未开始。

### 100.3 补齐的漏检单测（三项 toy 检查 + 一项队列门禁回归）

原 `t8_chunking` 失效的根因写在了新检查的 docstring 里，防止再次退化：
它用 `build(query_chunk=...)` 而**没有** `nonzero_out=True`，两条 output projection 仍是全零，
于是断言实际只比较了共享 `ConcatConv` 通路 —— 错误消息被零残差完全遮住。

| 新增检查 | 断言内容 | 修复前 | 修复后 |
| --- | --- | --- | --- |
| `t11_chunking_nonzero_residual` | 非零残差下 chunk0 vs chunk5 输出逐位相同；且先证明残差确实改变输出（>1e-3），否则检查为空转 | **FAIL** max&#124;Δ&#124;=0.5046627521514893 | PASS max&#124;Δ&#124;=0.0 |
| `t12_direct_message_and_grad_chunking` | 用 hook 读**直接消息** M_R/M_I（零初始化遮罩下唯一可见处）＋全输出＋60 个参数梯度，chunk0 vs chunk7（24=3×7+3，含残尾块） | **FAIL** M_R max&#124;Δ&#124;=0.7068458795547485 | PASS 消息/输出逐位相同，梯度 max&#124;Δ&#124;=1.71e-07 ≤1e-6 |
| `t13_message_spatial_impulse` | 雷达 level0 注入单点 impulse（地址塌到 cell 中心、null 推远），消息质量必须只亮**对应 2×2 query 块**，块外**恰好 0**，且 col2→col3 平移 +1 | **FAIL** 亮集覆盖整幅 6×4 画布 | PASS (2,2)→{(1,1),(1,2),(2,1),(2,2)}；col3→{(1,2),(1,3),(2,2),(2,3)}；块外 0.0 |
| `test_chunked_residual_messages_are_not_scrambled`（pytest，队列 cheap gate） | 非零残差下 M_R/M_I/输出 chunk0 vs chunk5 必须 `torch.equal`，并显式断言消息非零、残差确实移动输出（反空转） | **FAIL** `M_R depends on the chunk size` | PASS |

- 几何自检：BEV cell i 与 radar level-0 索引 i 同为 0.32 m 步长、共享 `radar_origin`，
  故 bilinear 采样把一个 impulse（列/行 c）摊到 BEV 列/行 c−1 与 c —— T13 断言的 2×2 与 +1 平移
  正是这条几何的可直接证伪形式，不是经验阈值。
- 反空转设计：T11/T12/T13 与 pytest 回归都先验「消息非零 / 残差确实改变输出」，
  避免再次出现「门禁恒真」的静默失效。

### 100.4 红/绿证据与门禁运行事实

- **有齿证明（红）**：临时把两行改回旧写法后重跑，`t11/t12/t13` **全部 FAIL**，而 **`t8` 仍 PASS** ——
  这正是原漏检的机制性解释，不是巧合。pytest 回归同法复现红。
- **绿灯（当前源码）**：
  - `python me_rssm/sanity/gpmsf_cpu_checks.py` → **13/13 PASS**（原 10/10）。
  - `python -m pytest -q -p no:hydra_pytest me_rssm/sanity/test_gpmsf_wiring.py` → **11 passed**（原 10）。
  - `python -m pytest -q -p no:hydra_pytest me_rssm/sanity/test_gpmsf_queue_checks.py` → **27 passed**。
    两者合跑 **38 passed**。
- `flake8` 三个改动文件：除仓库既有的 `E501` 长行（无 `setup.cfg`/`.flake8`，全仓库普遍存在）外，
  **无 F/E1/E2/E3/E7/E9 发现**。
- 说明：`me_rssm/sanity/` **整目录** pytest 不可用（部分 `test_` 前缀脚本在 collection 期执行 argparse 并
  触发 `SystemExit`），属仓库既有状况；本轮按单文件运行，未改动这些脚本。
- T13 与 T12 的数值均在 CPU、`torch.cuda.is_initialized() == False` 下取得；本轮没有接触 GPU 5/6/7。

### 100.5 归因、边界与下一步

- **归因**：这是**接线缺陷**，不是机制、容量或超参问题。梯度可达、step-0 逐位等价、
  `valid` 门控正确都**不能**证明空间对应正确 —— 原门禁把「零初始化输出投影」当成了安全网，
  实际它只是把错误消息乘了 0。
- 三条边界（必须随结论一起引用）：
  1. 修复后的模块**尚未**在真实管线重跑 preflight/S0；本节**不能**当作资源门、机制或 AP 结论。
  2. §99 的 6.7145× 时间与 1.0544× 内存来自**错排旧实现**；修复不改计算量，
     但耗时归因（`train_vod.py:12` 全局 anomaly detection、小算子调度、采样反向、checkpoint 重算、同步）
     必须在**新 attempt** 上重新拆分，不能沿用旧读数当正确实现的下界。
  3. 错排期间产出的 `s0_20261013T132727Z_3066450`（含 `iter_450.pth`）**不得**用于任何正式训练或继续训练，
     也不得与新 attempt 的读数合并统计。
- **下一步（按用户既定顺序）**：② 同 batch/初值做成本拆分与机制保持提速（head 向量化、复用计算、受控 chunk 扫描，
  不删 camera-only fallback、不用 GT 路由）；③ 补条件证据分层（有效 query / GT 目标与背景 / 类别-range 覆盖率、
  条件 null、消息范数、借点率）；④ 新 attempt 重跑真实 GPU S0，仍守 **≤1.20×**，全门通过才考虑 24e。

---

## 101. GP-MSF 成本拆分：算子数而非 FLOPs（2026-10-13，单卡只读微基准）

### 101.1 范围、统一口径与运行事实

- 新增只读微基准 `me_rssm/sanity/gpmsf_cost_decompose.py`（GPU5，单进程，无 DDP、无数据集、无优化器步），
  按**冻结 arm config** 的真实形状构建 `RCFusion` 本体：B=2、BEV 248×216、P2/P3/P4 120×160/60×80/30×40、
  radar 248×216/124×108/62×54、`_gpmsf_geometry` 真实 contract（N=1、points 20k×5，与真实管线一致）。
  **未启动训练、未改队列、未动 config。**
- 计时口径：前 3 次 warmup 后取 7 次 forward+backward 的墙钟均值（anomaly 行取 3 次），
  `torch.cuda.synchronize()` 包夹；峰值取 `max_memory_allocated`。每行同设备同形状，行间可比；
  **绝对值不外推到 S0 全步**（S0 是 DDP3+完整检测器）。
- **主指标区间均值：N/A（ep12–16 未形成，val n=0）；全轮峰值：N/A（无 val，epoch=N/A）；
  best saved val：N/A。** 本节是成本诊断，不是训练，任何 AP 结论都不在此。
- 证据：`docs/gpmsf_cost_decomposition_20261013.json`（含 CUDA kernel 计数与 top-12 名称）。

### 101.2 拆分结果（同一进程、同一输入、同一 seed）

| 变体 | anomaly | 时间 ms | 峰值 MiB | CUDA kernel 数 |
| --- | --- | ---: | ---: | ---: |
| B0 ConcatConv | off | 11.30 | 1690 | 16 |
| B0 ConcatConv | **on** | 11.84 | 1690 | — |
| GP-MSF 冻结（ckpt + chunk4096） | off | 714.16 | 2273 | **74,450** |
| GP-MSF 冻结 | **on** | **3125.35** | 2273 | — |
| GP-MSF 冒烟配置（+collect_stats） | off / on | 744.00 / 3131.23 | 2273 | 76,466 |
| GP-MSF ckpt + 单块（chunk0） | off | **236.33** | 7003 | 5,583 |
| GP-MSF 无 ckpt + chunk4096 | off | 451.61 | 8353 | 53,292 |
| GP-MSF 无 ckpt + chunk2048 | off | 920.46 | 8208 | — |
| GP-MSF 无 ckpt + 单块 | off | **188.22** | 8217 | 4,067 |

kernels 排行（冻结配置）：mul 6,166 / add 5,840 / fill 5,338 / mul-out 4,544 / copy 3,774 /
DtoD memcpy 3,754 / compare 3,476 ——**全是 (B,qc,K,nH,dk) 小块 elementwise 与拷贝**，
即 Python 侧 `chunk × head × level × candidate × height` 循环的启动开销，不是新机制的算术量。

### 101.3 归因（每条都有同进程对照，不是推测）

1. **anomaly detection 是本实现最大的单一乘数：714→3125 ms，4.37×**。
   `tools/train_vod.py:12` 的 `torch.autograd.set_detect_anomaly(True)` 对**所有**运行（冒烟与 24e）都开启；
   B0 只有 16 个 kernel，因此几乎不受影响（11.30→11.84，1.05×）。**它按算子数收费，恰好惩罚本实现**。
2. **gradient checkpointing 1.61×**（451.61→714.16 ms），换来 8353→2273 MiB（−6.1 GB）。
   这是"用时间换显存"，与机制无关。
3. **分块是 2.4–3.0× 的纯开销，且在关掉 checkpoint 时几乎不省显存**：
   chunk0 188 vs chunk4096 452（无 ckpt，峰值 8217 vs 8353 MiB）；ckpt 下 236 vs 714 ms，但峰值 7003 vs 2273 MiB。
   ⇒ 分块只在 ckpt 打开时才是显存必需项；否则它只是把 kernel 数按块数乘上去（chunk2048 比 4096 更慢：920 ms）。
4. **kernel 数随块数线性增长**：单块 4,067 vs chunk4096 74,450（约 18×），与 14 块的循环一致；
   T13/T12 已证明消息布局正确，因此这是**启动/调度**成本，不是"必须多算"。

### 101.4 机制保持的提速目标与边界

- 目标（投影，不是实测）：S0 的 B0 是 **1.550530 s/iter**，≤1.20× 意味着新增预算 **≈0.31 s/iter**。
  冻结配置的融合块在 anomaly 下是 3.13 s ⇒ 需要约 10× 的算子削减；
  候选组合是 **head 向量化（÷4）× 去掉分块（÷14）**，纸面到 ~4.1k kernel ≈ 40 ms 量级，**必须实测**。
- **不能**靠以下方式达标（且都不允许）：删 camera-only fallback、用 GT 路由、降 batch/N4 后仍称单变量、
  关掉 checkpoint 换显存（24e 会 OOM）。
- 仍**未归因**的 ~5.7 s/iter：S0 记录的时间差是 10.411050−1.550530≈8.86 s/iter，而本节融合块
  （anomaly 开）只有 3.13 s。差额不在本节的可见范围内（DDP3 同步/最慢 rank、全图 anomaly 记账、
  checkpoint 重算与真实数据管线的交互），**必须用 in-situ 分相计时补测，不能用猜测填**。
- 由此，S0 的 ≤1.20× 门在"全局 anomaly 开 + 现实现"下**不可达**；这是**harness 与实现共同的结果**，
  不是机制本身的成本下界。若后续关闭/限定 anomaly（对两臂同等生效的 harness 改动，需用户批准），
  也必须两臂同时重测，不能只改一边。
- 下一步：先做 head 向量化（同一数学、同一权重、逐位可回归），再用受控 chunk 扫描定档，
  然后才谈条件证据（③）与新 attempt S0（④）。

---

## 102. GP-MSF 机头（head）向量化：机制保持提速 2.36×（2026-10-13，CPU 等价 + 单卡微基准）

### 102.1 范围、统一口径与唯一变量

- 按用户执行顺序第 ② 步，在 §101 的算子数拆分之上做**机制保持**提速：
  `mmdet3d/models/fusion_layers/geometry_multiscale_fusion.py` 内部把 `Hd` 头轴显式化，
  几何/采样/兼容性/门控全改成一次广播；`_image_terms` 由逐头调用改为整头调用。
- **唯一变量：实现方式**。地址生成、offset 上界、三原生 radar 层、三 image 层 + 三高度、
  层间加权、paired compat、null 候选、valid 门控、参数集合与初始化、config、数据、RSSM/head **全部不变**。
- **主指标区间均值：N/A（ep12–14 未形成，val n=0）；全轮峰值：N/A（无 val，epoch=N/A）；
  best saved val：N/A。** 本轮不是训练，AP/资源门结论都不在此。
- 证据：`docs/gpmsf_cost_decomposition_20261013.json`（§101，改前）、
  `..._v2.json`（einsum 变体，反例）、`..._v3.json`（最终实现）；等价工具 `me_rssm/sanity/gpmsf_golden_compare.py`。

### 102.2 等价性证明（先证明"没改数学"，再谈提速）

改前用 `gpmsf_golden_compare.py save` 存下 CPU toy 契约的直接消息/输出/全部参数梯度/统计量，
改后 `compare`：

- **M_R、M_I、融合输出逐位相同**（max|Δ| = 0.0，chunk0 与 chunk5+checkpoint 两臂都比）；
- 60 个参数梯度 max|Δ| = **8.94e-08**（批量 matmul 归约顺序的浮点噪声，≤1e-6 阈值）；
- `last_stats` 全部字段一致（<1e-6）；参数名与顺序不变。
- 常驻门禁同时全绿：`gpmsf_cpu_checks.py` **13/13**、`test_gpmsf_wiring.py` **11 passed** /
  `test_gpmsf_queue_checks.py` **27 passed**（合跑 38）。t4 已按新签名改为整头调用，几何断言值不变。

### 102.3 反例：不是所有"批量化"都更快（einsum 版本更快地变慢）

第一版把逐头 `Linear` 也合成一个 `torch.einsum('bhmk,hok->bhmo')`。
kernel 数确实从 74,450 降到 21,029，但 **GEMM 时间反而爆炸**（A/B 同进程同输入）：

| 版本 | 冻结配置 时间 | sgemm 时间 | sgemm 次数 |
| --- | ---: | ---: | ---: |
| V1（改前） | 655.36 ms | 38.03 ms | 2,357 |
| V2（einsum 合头） | 731.04 ms | **508.78 ms** | 673 |
| V3（逐头 `F.linear`，最终） | **299.04 ms** | 26.49 ms | 2,017 |

即"合并头"在 (M,dk)x(dk,dk) 这种小 N/K 形状上让 cuBLAS 退化 ~20×。
最终版本只在**几何与采样**上批量化头轴，`Linear` 仍逐头调用（数学同 V1，GEMM 形状友好）。
**结论：kernel 数减少是必要条件，不是充分条件；必须同时看器件时间。**

### 102.4 最终成本（同一脚本、同一形状、同一输入）

| 变体 | anomaly off | anomaly on | kernel 数（off） |
| --- | ---: | ---: | ---: |
| B0 ConcatConv | 11.30 ms | 11.48 ms | 16 |
| GP-MSF 冻结（ckpt+chunk4096）改前 | 714.16 | 3125.35 | 74,450 |
| GP-MSF 冻结 **改后** | **277.36** | **1322.07** | **27,105** |
| GP-MSF 冒烟（+collect_stats）改后 | 294.79 | 1330.24 | 27,791 |
| GP-MSF ckpt + 单块 改后 | 300.25 | — | 2,222 |
| GP-MSF 无 ckpt + chunk4096 改后 | 238.00 | — | 19,965 |
| GP-MSF 无 ckpt + 单块 改后 | 244.06 | — | 1,686 |

- 冻结 = 24e 与 S0 用的配置：**anomaly off 2.58×、anomaly on 2.36× 提速**，显存不变（2279 MiB），kernel 数 −63.6%。
- 提速后**分块不再是时间惩罚**（单块 300 vs 分块 277 ms），但单块要 8.9 GB 而分块只要 2.3 GB：
  24e 保持 `ckpt+chunk4096`；`chunk2048` 仍更慢（328.84 ms），不存在"更小更快"。
- 仍未变的乘数：`tools/train_vod.py:12` 的全局 anomaly **4.8×**（277 → 1322 ms）。它按算子数收费，
  因此**必须先减算子再谈该开关**；是否限定该开关是 harness 改动，需用户批准且两臂同时重测。

### 102.5 边界与下一步

- 本节只测**融合块本体**（B=2、N=1、单卡 4090D、无 DDP/数据/优化器）。S0 的 ≤1.20× 判定仍在**整步**上，
  §101 记录的 ~5.7 s/iter 未归因差额依然存在，**不得**用本节的 ms 直接推 AP 或资源门结论。
- 由 1.550530 s/iter（B0）与 ≤1.20× 推出的新增预算仍是 ≈0.31 s/iter：融合块现为 1.322 s（anomaly 开）
  ⇒ 仍不在门内，但已从 3.125 s 走到 1.322 s；下一步排序：
  ① 在真实管线内做**分相 in-situ 计时**，先解释那 ~5.7 s/iter；
  ② 继续在 (chunk×head) 之外的 elementwise/拷贝上减算子（不删 fallback、不用 GT 路由）；
  ③ 之后才做条件证据分层（用户顺序 ③）与新 attempt S0（④），仍守 ≤1.20×。
- 修复与提速都**未上 GPU 训练**；错排期间的旧 checkpoint 仍禁用。

---

## 103. GP-MSF 修复确认与正式 24e 启动（2026-10-13，GPU 5/6/7，DDP3 seed0，进行中）

### 103.1 范围、统一口径与唯一变量

- 用户指令："确认全部修复正确以后直接开始正式训练" ⇒ 本节先给出**冻结代码上的正确性复核**，
  再记录**正式 24e 的启动**。这不是结果节：ep12-16 窗口尚未形成。
- **唯一变量（对 B0 主线 `fgfull_N4_no2d_igdr_2x4_24e_seed0` 而言）**：前端 fusion 由
  `ConcatConv(camera,radar)` 换成 GP-MSF 残差取证块；backbone、head、`temporal_fusion` RSSM、
  KL 调度、数据、优化器、batch/N4、epoch 数、seed、`--deterministic`、DDP3 **全部不变**。
  config：`me_rssm/configs/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_gpmsf.py`。
- **主指标区间均值：N/A（ep12–16 未到，val n=0，跨度/样本数不存在）；全轮峰值：N/A（无 val，epoch=N/A）；
  best saved val：N/A。** 本节所有数字均为启动期事实，**不得**被读成 AP 或资源门结论。
- 冻结实现：`1188acf`（含 `f505d4e` 轴合并修复 + `4332858` 成本拆分 + `1188acf` head 向量化）。

### 103.2 正确性复核（冻结提交 `1188acf`，工作区 clean）

| 门禁 | 结果 | 说明 |
| --- | --- | --- |
| `gpmsf_cpu_checks.py` | **13/13 PASS** | 含 t11 非零残差分块等价、t12 直接消息+60 参数梯度分块（ragged chunk7）、t13 空间 impulse |
| `test_gpmsf_wiring.py` + `test_gpmsf_queue_checks.py` | **38 passed** | 队列门禁回归（含 `test_chunked_residual_messages_are_not_scrambled`） |
| `gpmsf_golden_compare.py`（对**干净 worktree @4332858** 的保存参考） | **GOLDEN MATCH** | M_R/M_I/输出 **max\|Δ\|=0.0**（full 与 chunk5+ckpt 两臂）；60 参数梯度 max\|Δ\|=**8.94e-08**；参数名/顺序与 `last_stats` 一致 |
| `gpmsf_real_preflight.py`（真实 train/val 管线，GPU 5） | **status=PASS，16/16** | §103.7核对原始JSON：16项全部True（原文误写8/8）；几何重投影误差max1.83e-4 px。offset/key等内部取证路径第0步梯度0、第1步非零；output projection与共享fusion第0步已有非零梯度，不是所有参数第0步全0 |

- 本次等价性复核**不是复述旧结论**：参考张量在预向量化提交 `4332858` 的独立 worktree 内**重新生成**
  （`/tmp/ref_prevec_4332858.pt`），再在当前工作区 `compare`，因此排除"用改后代码自比"的循环论证。
- 证据：`/data/lurui/work_dirs/fgfull_N4_no2d_igdr_2x4_24e_gpmsf_seed0/s0_fixed_20261013T164148Z_3549897/preflight/real_preflight.json`。

### 103.3 启动事实与资源状态（诚实口径）

- 启动命令（与队列一致的口径）：
  ```
  CUDA_VISIBLE_DEVICES=5,6,7 bash tools/dist_train.sh <gpmsf cfg> 3 \
      --seed 0 --deterministic --work-dir /data/lurui/work_dirs/fgfull_N4_no2d_igdr_2x4_24e_gpmsf_seed0
  ```
  进程组 pgid **3551130**（setsid 独立会话），日志 `train_gpmsf.log`，`max: 24 epochs`，1902 iter/epoch。
- 启动期守卫：`gpmsf_run_checks.py start` 已写入 `phase.json`（**832 个研究源文件 sha256 冻结**、
  `gpus=[5,6,7]`、`world_size=3`、`seed=0`）；训练期间改研究代码会使 `finish` 失败。
- 独立 watchdog（`watchdog_gpmsf.sh`，仓库外）复刻队列 `run_owned` 语义：异常/OOM ⇒ 终止并为人工复核留证；
  ep6 触发 `curve --early-stop` 止损；ep16 写 `gpmsf_ep16_report.json` 后继续训到 ep24。

**in-situ 资源（同配方、同 3×4090D DDP、整步口径）**

| 项 | B0（24e 主线实测） | GP-MSF（本节） | 比值 |
| --- | ---: | ---: | ---: |
| s/iter | 1.529–1.604（ep24 iter200–350）；§99 450-iter 冒烟 1.550530 | **5.050**（iter100；iter50 含预热 5.496） | **≈3.26×**（对 1.550530） |
| 显存 (MiB) | 16,968（§99 冒烟 16,959） | **17,900** | **≈1.055×** |
| GPU 利用率 | — | 37–40% | — |

- **资源门结论：≤1.20× 时间门未过（≈3.26×），本节如实记录。** 本次 24e **不是**在资源门通过后启动，
  而是按用户明确指令"确认修复正确后直接开始正式训练"启动；预登记里的 fail-closed 顺序（新 attempt S0 → 硬门 → 24e）
  **被显式越过，未伪造 S0 通过**。
- §99 的 6.71×、10.411050 s/iter、17,881 MiB 属于**错排实现**，**不得**与本表混用；修正实现的配对 S0 冒烟**未跑**，
  因此"修正实现是否 ≤1.20×"仍是 **UNRESOLVED**。已知放大器：`tools/train_vod.py:12` 全局 anomaly 使融合块 ×4.8（§102），
  且 §101/§102 记录的未归因差额依旧要求**分相 in-situ 计时**；两者都属 harness/实现，不是机制本身的下界。
- 训练期不改 `train_vod.py` 的 anomaly 开关：那会同时破坏 `phase.json` 源冻结与"对 B0 单变量"契约（B0 亦为 anomaly 开）。

### 103.4 启动期日志（将被 §104 的完整曲线取代）

| epoch | iter | time (s/iter) | memory (MiB) | loss | grad_norm | 备注 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 50 | 5.496 | 17,891 | 5.1913 | 14.6092 | 含数据预热（data_time 0.472） |
| 1 | 100 | 5.050 | 17,900 | 3.6924 | 13.7802 | ETA 2 d 18:42 |

- 启动期无 NaN/Inf、无 OOM、无 traceback；三个 rank 均正常占用 GPU 5/6/7。
- **结论（启动期）：修复后的 GP-MSF 已在真实管线、真实坐标、真实 batch 上稳定前向/反向。**
  机制有效性（AP）与资源门**都还未被回答**。

### 103.5 下一步

1. 训练继续到 ep6：watchdog 自动跑 `curve --window 1-6 --early-stop`，若相对 B0 出现 >2 AP 任务损害则止损并留证。
2. ep12-16 窗口形成后自动写 `gpmsf_ep16_report.json`（区间均值口径），继续训到 ep24。
3. 训练结束后写 **§104**：**必填**区间均值（ep12-16，n=5）+ **全轮峰值（值+epoch）** + best saved（单列）；
   逐 epoch 完整曲线、四类构成项、BEV、FP/FN、high/low/zero 组与错误借点率等机制诊断。
   对照：B0 seed0（§?? 39.2809）、三 seed 基线（40.4400 ± 1.3475）、A2-R（38.3778，−0.903）。
4. 资源侧仍需补：修正实现的**配对 S0**（新 attempt）或训练后按 §102 优先级做分相 in-situ 计时，解释 ≈3.26× 的构成。

### 103.6 运行中状态（2026-10-14 01:43 UTC 快照；§103.1 的三口径约束仍适用）

- **进度**：ep1–ep3 已完成训练+val，正在 ep4。val 结束时刻 ep1 `19:41:42`、ep2 `22:41:31`、ep3 `01:42:10`
  ⇒ 每 epoch ≈ **2h58m–3h00m**（含 val），按当前速度 24e 预计 **2026-10-16 ≈16:20 UTC** 收尾。
- **统一口径（截至ep3；本段由§103.7纠正）**：区间均值（ep12–16）**N/A，窗口内val n=0；run累计val n=3**；
  全轮峰值（所有已有val，scope ep1–3，随run继续更新）**36.4467 @ ep3**，不能因run未结束而记N/A；
  best saved **31.1923 @ ep2**（`epoch_2.pth` 22:29 落盘，`checkpoint_interval=2`；`latest.pth` 指向它）。
  **不得**把下面 ep1–3 的早期数字当作对照口径。
- **资源（in-situ，同配方 DDP3）**：ep1/2/3 均值 **5.241 / 5.280 / 5.292 s/iter**（n=38 行/epoch），data_time ≈0.08，
  显存恒 **17,913 MiB**。与 §103.3 的 5.050 一致，无随时间漂移；对 B0 的 ≈3.4× 仍未过 ≤1.20× 门（§103.3 已记）。
- **稳定性**：`nan/NaN/Inf/out of memory/Traceback` 匹配 **0**；`grad_norm` 末值 8.1173；无重启、无人工干预。
- **早期曲线（ep1–3，n=3，仅作启动期读数）**：

| ep | B0 Overall3D_mod | GP-MSF | Δ | Car (B0/G) | Truck (B0/G) | Ped (B0/G) | Cyc (B0/G) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 15.2640 | **16.7038** | +1.440 | 29.411 / 27.185 | 7.160 / 3.292 | 13.976 / 12.631 | 10.509 / 23.706 |
| 2 | 26.3716 | **31.1923** | +4.821 | 37.675 / 44.263 | 22.447 / 20.022 | 11.529 / 22.218 | 33.835 / 38.266 |
| 3 | 31.6591 | **36.4467** | +4.788 | 44.935 / 45.990 | 19.562 / 27.384 | 22.224 / 21.859 | 39.916 / 50.554 |

  早期窗口 ep1–3 等权均值：GP-MSF **28.1143** vs B0 **24.4316**（**+3.6827**）。
- **不能过度解读**：ep1–6 受 warmup/新参数初始化主导，且 Truck/Ped 在早期抖动极大（Ped ep2 单点 +10.7、
  Truck ep1 −3.87）⇒ 早期领先**既不是** ep12–16 收益的证据，**也不是**解耦了"仅增加容量"。真正判定仍在
  ep6 止损（watchdog 自动执行，预计 ≈10:45 UTC）与 ep12–16（n=5）窗口。
- **外部因素核验**：GPU 5/6/7 上只有本 run 的 3 个 rank（3551139/40/41）；同时段其他用户作业在别的 GPU 上，
  **未共享我们的卡**，故 5.24–5.29 s/iter 不含外部争用。


### 103.7 现役训练合规核验（2026-10-14 02:03:56 UTC / 北京时间10:03:56，只读验证）

**本轮用户请求：检查当前训练是否符合要求。结论：部分符合，不能写“全部验收通过”。**
本轮未改研究Python/config、未改/停/重启训练或watchdog、未启动额外GPU计算；只补记录与文档勘误。
CPU检查需要可见GPU以通过仓库import断言，但所有探针`torch.cuda.is_initialized()==False`。
结构/训练源仍为`1188acf`；核验前HEAD `baecef4`仅追加历史文档。完整原始审计：
`docs/gpmsf_active_training_audit_20261014.json`。

#### 103.7.1 已通过：配方、修复、源冻结、当前运行健康

- 进程组3551130，ranks3551139/3551140/3551141，world3、rank0/1/2、GPU5/6/7；watchdog3552704存活。
  CLI两处seed均0、`--deterministic`；训练日志明确`Set random seed to 0, deterministic: True`。
  `nvidia-smi`核对5/6/7只包含这三个rank。配置dump未写seed字段，不据此误判：checkpoint meta.seed实际为0。
- **832个研究源SHA全部匹配启动phase与修复后真实preflight，漂移0**。
  当前resolved配置对B0模型差异恰为`['RCFusion']`，其他研究配方差异为空；实际work-dir配置与登记臂一致。
  bs2/rank、N4、累积2、AdamW lr1.5e-4、24e、RSSM/head/KL日程/数据/评测/预训练权重均未另改。
  query_chunk4096、gradient checkpoint、64维/4heads/4offsets/3原生radar+image层/3高度保持冻结。
- 修复后真实preflight JSON实际**16/16 True**且源hash匹配；§103.2与TODO的“8/8”是记录漏数，已纠正。
  独立重投影误差最大0.0001831055px；初值/RNG、真实train+val step0、缺测恒等、两步梯度、native梯度均通过。
  “第0步全0”仅指内部offset/key等取证路径；output projection/shared fusion第0步有梯度，已澄清。
- 本轮重新执行CPU门：`gpmsf_cpu_checks.py` **13/13**；wiring+queue pytest **38 passed**。
  非零残差chunk一致、直接消息/60参数梯度、空间impulse门均PASS。旧query/channel混排已不在当前实现中。
- **同环境独立old/current golden**：从git对象`4332858`取原类完整源码，只移除registry decorator以避免重名，
  所有方法体原样；同CPU后端/输入与当前代码比较，full与chunk5+checkpoint的M_R/M_I/输出**max|Δ|=0**，
  60参数梯度max|Δ| **5.22e-08 / 1.49e-08**；stats max|Δ| **2.24e-08 / 1.71e-08**，均在原1e-6门内。
  没有改后自比、没有放宽容差。
- 数值审计：截至快照118条train区间、3条val数值均有限，无异常/OOM/traceback匹配。
  `epoch_2.pth`在CPU加载：meta epoch2/iter3804/seed0，actual model与完整recipe匹配，全部state有限。
  两方向projection weight abs_mean分别**0.00488136 / 0.00360125**，已离零；不是基线静默回退。
- 最新已记录训练为 **ep4 iter200/1902**，time5.28190s、memory17913MiB；正在运行，未声称完成24e或ep6门。

复跑中的一个边界不隐藏：旧`/tmp/ref_prevec_4332858.pt`在本轮CPU环境直接compare，
forward差约1e-7、共享conv梯度最大差1.1026859e-6，原1e-6绝对门FAIL。
同环境重新执行**旧版自身**对这个旧artifact也有相同量级偏差，不能据此归因新实现退化；
上述独立同环境old/current比较通过。旧artifact未记录CPU后端/线程指纹，跨运行微小数值漂移确切来源未定位。
完整失败输出和重取参考证明均保留JSON，不删失败、不偷偷改阈值。

#### 103.7.2 未符合：原S0/资源合同

- 修复后的新attempt只有真实preflight，**没有配对B0/arm450iter、没有新S0 verdict**。
  旧S0的source SHA不匹配当前代码；错排版本的DDP hash、9条stats与资源读数不能移植成当前验收。
- 正式run ep1–3全部114条train区间time均值 **5.271164s/iter**；各ep均值分别5.241193/5.280154/5.292143。
  对历史同配方B0冒烟1.550530为 **3.399588×**；当前logger峰值17913MiB对16959为 **1.056253×**。
  这是in-situ资源超预算证据，**不是补跑成功的严格配对S0**；1.20×上限未变，也不能据显存通过称总门通过。
- §103已记录收到“确认修复后直接开训”的指令并绕过S0；这证明直接启动的经过，
  **不证明原资源门已通过，也不自动等同明确批准把1.20×改成3.4×**。
  因此当前仅能归为“已直接开跑、资源合同有例外/偏离的性能探索”，不能归为全门合规开训。
  不把预算失败写成AP失败；原合同若继续有效，此run不具备原定放行证明。

#### 103.7.3 未符合：现役watchdog比原queue弱，非完整fail-closed

审查对象是仓库外实际运行的`watchdog_gpmsf.sh`，不是已有38项pytest覆盖的`queue_gpmsf.sh`。
其sha256：`040c9950009cb15151cdf48283f50d5ff9176b9f21e22fdef00aef59ded767a1`。

1. **第23/32行ready失败被吞**：ready返回3表示epoch尚未到，返回1表示NaN/损坏JSON/IO错误；
   外部watchdog没有else区分，二者都继续等待。临时目录探针实测正常未到轮返回3、NaN和损坏完整JSON均返回1；
   最小相同shell分支实测status1后仍输出`STILL_RUNNING`。没有对现役日志注入NaN或执行其stop_owned。
   当前日志干净不代表此失败分支合格。
2. **第44–48行可假DONE**：最终curve失败仅`|| say '... failed'`，随后只要epoch24文件存在仍写DONE。
   与“全部24轮val齐全且报告有效后才DONE”冲突；此为将来结束路径风险，不是假称目前已发生。
3. **缺trainer真实退出码与phase finish核验**：仅检测进程组消失，没有wait/receipt证明trainer exit0，
   未调用`finish_phase`或等价源hash+成功退出验证。`phase.json`本轮仍exit_code=null属正常进行态，
   但该脚本结束时也不会补闭环；源码冻结的实际“结束失败”保障并不由此watchdog执行。
   因而§103.3“复刻原queue语义”的描述过强。

本轮只读验证，不热改正在执行的bash；即便以后改外部脚本文件，已运行bash也不会自动重载。
若修守卫，应受控替换**守卫进程本身**，保留trainer进程组并单独证明退出/完成条件；
不得随手kill持有GPU的trainer，亦不得为修守卫改动832个冻结研究源。

#### 103.7.4 当前全部逐epoch曲线、两个必填口径与边界

- 主窗口ep12–16：**N/A，窗口内val n=0**；当前run累计val n=3。
- 补充早期窗口ep1–3等权均值：**28.1143（n=3）**，B0同窗**24.4316**，Δ**+3.6827**。
- **全轮峰值（所有已有val，scope ep1–3，随run继续更新）：36.4467 @ ep3**。
- best saved单列：**31.1923 @ ep2**；当前仅epoch2有实际权重，不参加性能排名。
  §103.6把峰值因run未结束写N/A不符合统一口径，已纠正；成熟窗口未形成仍必须明确写N/A。

| epoch | GP-MSF Overall3D mod | B0同轮 | Δ | Car strict mod | Truck strict mod | Ped loose mod | Cyc loose mod | OverallBEV mod |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 16.7038 | 15.2640 | +1.4398 | 27.1852 | 3.2925 | 12.6310 | 23.7064 | 23.9986 |
| 2 | 31.1923 | 26.3716 | +4.8207 | 44.2634 | 20.0222 | 22.2176 | 38.2659 | 38.8706 |
| 3 | 36.4467 | 31.6591 | +4.7876 | 45.9897 | 27.3842 | 21.8590 | 50.5540 | 43.0202 |

| 窗口 | Overall3D mod | Car strict mod | Truck strict mod | Ped loose mod | Cyc loose mod | OverallBEV mod |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GP-MSF ep1–3，n=3 | 28.1143 | 39.1461 | 16.8996 | 18.9025 | 37.5088 | 35.2965 |
| B0 ep1–3，n=3 | 24.4316 | 37.3404 | 16.3895 | 15.9097 | 28.0867 | 32.0186 |
| 同窗Δ | +3.6827 | +1.8057 | +0.5101 | +2.9928 | +9.4221 | +3.2778 |

完整逐epoch所有评测指标在审计JSON的`latest_snapshot.complete_epoch_val_rows`；原train/val日志与hash亦留存。
逐类早期波动、机制分层（目标区域条件null、high/low/zero、借点率、FP/FN）仍待专门诊断；
当前不得把早期+3.68、projection离零或梯度连通当作稳定AP创新证据，也不因守卫漏洞宣称数值训练已坏。

**下一步与停止边界**：本轮验证到此结束。不扩展网络或调参；先补强外部守卫fail-closed与退出闭环，
并明确本轮是否接受资源例外；继续时按ep6止损、ep12–16成熟窗口与全轮逐类评估。
方向1仍未完成；正式修复/重排/停训需要后续执行指令，本轮未擅自实施。
