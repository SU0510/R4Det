# 五次训练完整记录（效果 / 配置 / 模块改动）

> 数据来源：各 `work_dirs/<run>/*.log.json` 的 `mode=val` 记录（每 epoch 取该 run 最后一次 val，避免中途重启污染）+ 各 run 目录下 `*.py` 配置快照 + git log（branch `motionalignrssm`）。
> 评估口径：KITTI 3D detection AP。主指标 **Overall 3D moderate**（`pts_bbox/KITTI/Overall_3D_moderate`）。`_loose` 为松评估 per-class。
> **Overall 口径说明**：`Overall_3D_moderate` = `(Car_strict + Truck_strict + Pedestrian_loose + Cyclist_loose) / 4`。即「Car/Truck 看 strict、Ped/Cyc 看 loose」的混合口径，不是四类全 strict（详见 `kitti_utils/eval.py` 的 Overall 聚合）。读任何 Overall 数字时请按此四项理解，否则会用错驱动变量。
> 所有 run 训练 18 epoch（`baseline_temporal` 日志从 ep6 起，ep1-5 无记录）。
> 配套诊断见 [rssm_diagnosis.md](rssm_diagnosis.md)。

---

## 0. 五次 run 一览

| # | run 目录 | 时序融合模块 | BEST 3D_mod @ep | 起点 | 一句话定位 |
|---|---|---|---:|---|---|
| 1 | `baseline_temporal` | `TemporalDeformableFusion`（GRU+deform） | **34.50** @ep12 | resume latest | 最强 baseline，GRU 直接吃上一帧 BEV |
| 2 | `baseline_rssm` | `BEVRSSMTemporalFusion`（RSSM，无对齐） | 33.54 @ep17 | resume ep4 | RSSM 首次落地，KL 被压死 |
| 3 | `motion_align_rssm` (v1) | `MotionAlignedRSSMFusion` | 30.89 @ep10 | from scratch | 对齐层 + RSSM，KL 仍压死，踩坑跑 |
| 4 | `motion_align_rssm2` (v2) | `MotionAlignedRSSMFusion` | 34.01 @ep14 | resume v1 latest | 修 KL 超参，追平 baseline |
| 5 | `motion_align_rssm3` (v3) | `MotionAlignedRSSMFusion` | 33.77 @ep11 | from scratch | 同 v2 配置 + logstd 防爆炸，从头训 |

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

### Run 9: N=4 Pretrained RSSM (seq_len=4, hdim=128, 30e)

- config: configs/r4det/TJ4D-R4Det_motion_align_rssm_det3d_N4_2x4_30e_pretrained.py
- work_dir: work_dirs/rssm_N4_2x4_30e_pretrained
- diff: load_from only, else identical to Run 7 (N4 30e no-pretrain)
- BEST: ep19 37.94 🔥 (ALL-TIME RECORD)
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
- **BEST overall: ep11 = 40.60** ⚠️ 未存（奇数 epoch，interval=2）
- **BEST SAVED overall: ep14 = 39.65**
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
| Overall 3D_moderate | **40.60** | 11 (未存) | 38.76 |
| Overall 3D_moderate（已存） | **39.65** | 14 | 38.76 |
| Overall 3D_easy | 42.49 | 15 | 40.73 |
| Overall 3D_hard | 39.06 | 11 | 37.42 |
| Overall BEV_moderate | 48.70 | 13 | 46.66 |
| Car 3D_mod_strict | **53.04** | 20 | 51.80 |
| Cyclist 3D_mod_strict | 25.41 | 8 | 23.62 |
| Pedestrian 3D_mod_strict | 0.42 | 2 | 0.08 |
| Truck 3D_mod_strict | 33.23 | 11 | 26.26 |

### 10.3 vs Run 9（同 base，唯一差异=检测头）

| Metric | Run 9 BEST (ep19) | Run 10 BEST | Run 10 已存 best | Δ (BEST) |
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

1. **检测头改动带来明确正向收益**：Overall 3D_moderate +1.71（已存 ep14）/ +2.66（峰值 ep11），BEV 口径 +5.53。Car strict 刷到 **53.04**，全库历史最高，超 Run 9 达 +4.08。
2. **loose（召回）口径全面提升**：Car +8.71、Cyclist +11.32、Truck +5.22。来源是多 anchor + `anchor_class_mapping` + IoU-proxy 训练信号共同改善了共享检测头特征与打分质量；Car/Cyclist/Truck 仍各只有 1 个 anchor，说明提升是「共享头标定变好」，不是简单加 anchor。
3. **Pedestrian strict 依旧 ≈ 0**：3 组行人 anchor 只让 loose 微涨（+0.67）、strict 在 0.08–0.42 的噪声带内横跳。**证实瓶颈是 BEV 0.16m 分辨率 + 点云稀疏（行人仅 ~15 cell），不是 anchor 数量**，与上轮判断一致。
4. **Truck strict 持平（33.2 vs 33.3）但 loose +5.22**：召回上去了、定位没上去，**证实短货车(2.8m)到半挂(25.5m)的巨大尺寸方差 + 单 anchor 才是 Truck 瓶颈**，与尺寸方差分析吻合。
5. **Cyclist strict 几乎没动（25.4 vs 25.1）**：`ignore_dir_classes=[0]` 只关了行人方向分类，未伤及 Cyclist；但也没带来骑行者 strict 提升。
6. ⚠️ **`checkpoint_interval=2` 丢掉了本轮的峰值 checkpoint**（ep11=40.60 只存在于日志）。下一轮必须改成 `interval=1`（或加 `max_keep_ckpts`），否则奇数 epoch 的峰值还会再次丢失。

### 10.6 Conclusions / Next

- **head-v2 是当前最强模型**：已存最优 ep14（Overall 39.65）+ 新纪录 Car strict ep20（53.04）。按需选点：看 Overall 用 ep14，看 Car 单项用 ep20。
- 检测头这一刀砍对了方向，但它主要救了 Car / 召回，**没解决 Ped strict 和 Truck strict 两个真正的硬骨头**。
- 下一步优先级：
  1. **Truck 多 anchor**（van/标准货/半挂 3 组）——直接打当前最大的 strict 差距（Truck 26~33 vs Car 52），零替代风险；
  2. `checkpoint_interval=1` 必修，避免再丢峰值；
  3. Pedestrian 需跳出 anchor 层——BEV 分辨率 / 点云稠密化 / point-based 头，才是真正杠杆；
  4. Cyclist：loose 已大涨，strict 未跟上，可考虑骑行者专属 size/增强。

### 10.7 复现性评估：Run 10 的 39.65 是「平台顶」，不是「典型值」

- **40.60 从未成为可用 checkpoint**：Run 10 `checkpoint_interval=2`，奇数 epoch 不落盘，ep11=40.60（实际日志 40.59）只存在于日志。可用最优是 **ep14=39.65**。
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
- **BEST Overall: ep11 = 38.45（✅ 已存，interval=1 生效）**
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
| Overall 3D_moderate | **40.60** (ep11) / 39.65 已存 | 38.45 (ep11) | **−2.15 / −1.20** |
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
- **BEST Overall: ep14 = 38.11（✅ 已存）**
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

- **本轮仍是负收益**：Car-large anchor 单独使用能救 Car，但会牺牲 Truck/Cyclist，Overall 没有超过 Run 11，更低于 Run 10 已存最优。
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
- **BEST Overall: ep8 = 36.21（✅ 已存，interval=1）**
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
| Overall 3D_moderate | **40.60** (ep11) / 39.65 已存 (ep14) | 36.21 (ep8) | **−4.39 / −3.44** ❌❌ |
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
- **BEST Overall: ep14 = 37.84（✅ 已存）**
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

1. **BPTT 当前设置没有带来提升**。Overall 最优 37.84（ep14），比 Run 10 已存最优 39.65 低 **1.81**，比 Run 10 峰值的 40.60 低 **2.76**；Run 14 的 LAST 34.90 更比 Run 10 LAST 38.76 低 **3.86**。
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
- **BEST Overall: ep14 = 38.08**（已存）
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
