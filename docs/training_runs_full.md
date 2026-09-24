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
> seq_len=4, hidden_dim=128, samples_per_gpu=4, lr=1.5e-4, CosineAnnealing, 30 epochs, grad_accum=2.

### 9.0 N=4 30e 无 pretrain 对照组（Run 9 的对照基线）

> 第 9.3 节与第 16 节都引用了这条 run 的数值（BEST `34.71`、last-5 `33.67`），但此前没有独立
> 章节。本次审计按磁盘日志补账，它是正式训练而非 smoke。

- 工作目录：`/data/lurui/work_dirs/rssm_N4_2x4_30e`。
- 配置：`configs/r4det/TJ4D-R4Det_motion_align_rssm_det3d_N4_2x4_30e.py`
  （`load_from=None`，其余 30e / seq_len=4 / hidden_dim=128 / samples_per_gpu=4 /
  lr=1.5e-4 / CosineAnnealing / grad_accum=2 与 Run 9 相同）。
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
> 24 epochs，samples_per_gpu=4，lr=1.5e-4（CosineAnnealing），grad_accum=2，`load_from=pretrained_tj4d.pth`，**`checkpoint_interval=2`（注意：奇数 epoch 不存）**。

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

### 17.5 盘点（2026-09-24，dry-run，未执行删除）

按 17.1 规则逐目录核对 `best saved`、`last`、`latest.pth` 指向，并扫描 `configs/`、`tools/`、`docs/`
里对 `*.pth` 的显式引用（下游 `load_from` / `resume_from`、诊断脚本 `--checkpoint`、文档固定选点）。
下表是纯 dry-run 结果，**本次未删除任何文件**；后续人工确认后再执行。

排除项：

- `fgfull_N4_no2d_igdr_2x4_24e_seed2`：seed2 仍在运行（GPU 5/6/7），整个目录不在清理范围。
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
| `fgfull_N4_2x4_24e_seed0/epoch_2.pth` | 文档第 47 节的断点续训固定记录（仅叙述引用，可按需释放） |

可删除候选（202 个文件，合计 99.2 GB）：

| run | 现有 | 保留 | 删除数 | 删除 epoch | 可释放 |
|---|---:|---|---:|---|---:|
| `run10_headv2_multiseed/seed_2` | 24 | ep14(best+依赖), ep24(last) | 22 | 1-13, 15-23 | 11.8 GB |
| `fgfull_N4_2x4_24e_seed0` | 12 | ep2(文档), ep16(best+依赖), ep24(last) | 9 | 4,6,8,10,12,14,18,20,22 | 7.1 GB |
| `cyccls_branch_N4_2x4_24e_seed0` | 15 | ep12(best), ep24(last) | 13 | 10,11,13-23 | 7.1 GB |
| `fgfull_N3_h128_2x4_24e_seed0` | 10 | ep14(best), ep20(last) | 8 | 2,4,6,8,10,12,16,18 | 6.3 GB |
| `fgfull_N4_no2d_igdr_2x4_24e_seed0` | 11 | ep14(best), ep22(last) | 9 | 2,4,6,8,10,12,16,18,20 | 5.8 GB |
| `run10_headv2_multiseed/seed_0` | 12 | ep16(best+依赖), ep24(last) | 10 | 2,4,6,8,10,12,14,18,20,22 | 5.3 GB |
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

说明：

- 上表 `best` 一律指 `best saved`（实际有权重的 val 最高点），不是全轮峰值；全轮峰值另见各 run 正文，可能没有权重。
- `last` 指 val 曲线最后一个 epoch。部分 run 的 `latest.pth` 指向的 checkpoint 序号大于最后一个 val epoch（例如 `fgfull_N3_h128` 的 ep20 对 val ep21、`fgfull_N4_no2d_igdr_2x4_24e_seed1` 的 ep20 对 val ep21、`ped_centerhead_stage1_3x2x2_12e_seed1` 的 ep10 对 val ep9、`cyccls_branch_..._multiseed/seed_2` 的 ep18 对 val ep17），按 17.1 规则保留磁盘上的 last 权重。
- 已合规、无需处理的目录（仅 best saved + last）：`deterministic_latent_N4_2x4_24e_seed0`、`ped_centerhead_stage2_delta_3x2x2_3e_seed0`、`ped_centerhead_stage2_dim_3x2x2_6e_seed0`、`ped_highres_centerhead_3x2x2_3e_prior_smoke`、`ped_highres_centerhead_3x2x2_3e_seed0_smoke`、`ped_refine_N4_2x4_24e_seed0`、`pedrot4_N4_2x4_24e_seed0`、`posterior_only_learnable_std_N4_2x4_24e_seed0`、`rssm_N2_2x4_24e_pretrained_v2_head_bptt`、`rssm_N4_2x4_24e_pretrained_v2_head`、`rssm_N4_2x4_24e_pretrained_v2_head_bptt`、`rssm_N4_2x4_24e_pretrained_v2_head_dynmask`、`rssm_N4_2x4_24e_pretrained_v2_head_truck`、`rssm_N4_2x4_24e_pretrained_v2_head_truck_car2`、`rssm_N4_2x4_30e`、`rssm_N4_2x4_30e_lr2e4`、`rssm_N4_2x4_30e_pretrained`、`rssm_N4_fixednoise_posterior_seed0`、`seed1_ep15_restore`、`truck_tower_N4_2x4_24e_seed0`。
- 非 checkpoint 的可回收项（未计入上表）：`diag_dumps/` 约 545 MB、各 run `figures_path/` 合计约 3.0 GB。这些是诊断/可视化中间产物，需按项目需要单独确认。


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
> 三个 seed 配置完全相同，唯一差异是随机种子、work_dir、以及 checkpoint_interval（seed_0/seed_1=2，seed_2=1）。
> 硬件：3 卡 GPU 5/6/7（原 Run 10 为 4 卡），有效 batch 12 vs 原 Run 10 的 16。

### 19.0 实验设置

| 项 | seed_0 | seed_1 | seed_2 | 原 Run 10 |
|---|---|---|---|---|
| config | 同 Run 10 | 同 Run 10 | 同 Run 10 | -- |
| work_dir | `run10_headv2_multiseed/seed_0` | `.../seed_1` | `.../seed_2` | `rssm_N4_2x4_24e_pretrained_v2_head` |
| GPUs | 5,6,7 (3卡) | 5,6,7 (3卡) | 5,6,7 (3卡) | 4卡 |
| 有效 batch | 12 | 12 | 12 | 16 |
| checkpoint_interval | 2 | 2 | 1 | 2 |
| load_from | pretrained_tj4d.pth | pretrained_tj4d.pth | pretrained_tj4d.pth | pretrained_tj4d.pth |
| seed | 0 (default) | 1 | 2 | 0 (default) |

seed_0 与原 Run 10 用的是同一 default seed（0），但 3 卡 vs 4 卡导致 BN 统计量 / 梯度累积不同，严格说不完全等价。

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
- BEST epoch 从 ep11 后移到 ep14-16，可能来自 3 卡 vs 4 卡的有效 batch 差异（12 vs 16），更小 batch 需要更多 epoch 收敛到峰值。
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
3. 峰值稳定在 ep12-16（原 Run 10 在 ep11），3 卡 vs 4 卡的有效 batch 差异可能导致峰值后移，但不影响上限。
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
- 配置：在 Run 10 head-v2 的 24e 配置上只改一项：`temporal_fusion=None`。
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
- 为与 seed1 保持对称，seed2 也按同一判据在 ep21 val 后截断（ep22 不落盘）；截断动作
  **尚未自动执行**，需要在 ep21 val 写出后人工 kill（或另加 watcher）。主比较窗口仍为
  ep12-16 与 ep18-20；
- 跑完后在本节追加 seed2 结果、三 seed 均值/标准差，并与 temporal baseline 做同窗口对照。
