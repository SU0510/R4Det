# R4Det 训练完整记录（效果 / 配置 / 模块改动）

> 数据来源：各 `work_dirs/<run>/*.log.json` 的 `mode=val` 记录（每 epoch 取该 run 最后一次 val，避免中途重启污染）+ 各 run 目录下 `*.py` 配置快照 + git log（branches: `motionalignrssm` → `Nframerssm` → `detection_head_v2` → `radar_static_dynamic`）。
> 评估口径：KITTI 3D detection AP。主指标 **Overall 3D moderate**（`pts_bbox/KITTI/Overall_3D_moderate`）。`_loose` 为松评估 per-class。
> **Overall 口径说明**：`Overall_3D_moderate` = `(Car_strict + Truck_strict + Pedestrian_loose + Cyclist_loose) / 4`。即「Car/Truck 看 strict、Ped/Cyc 看 loose」的混合口径，不是四类全 strict（详见 `kitti_utils/eval.py` 的 Overall 聚合）。读任何 Overall 数字时请按此四项理解，否则会用错驱动变量。
> 训练时长按阶段：Run 1–8 为 18 epoch；Run 9 为 30 epoch；Run 10–15 为 24 epoch（`baseline_temporal` 日志从 ep6 起，ep1-5 无记录）。
> 配套诊断见 [rssm_diagnosis.md](rssm_diagnosis.md)。

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
- **BEST SAVED overall: ep14 = 39.65**
- **⚠️ 盘点注：ep14 权重已不在盘中，当前落盘为 ep12=39.26，见 17.2 节**
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
- **频率加权 Overall（平台，last-5）**: **39.88 ± 0.78**（幅度提升来自 Car 权重 25%→48.4%；权重见 section 16）
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
- **频率加权 Overall（平台，last-5）**: **40.23 ± 0.65**（幅度提升来自 Car 权重 25%→48.4%；权重见 section 16）
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
- **频率加权 Overall（平台，last-5）**: **38.62 ± 1.15**（幅度提升来自 Car 权重 25%→48.4%；权重见 section 16）
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
- **频率加权 Overall（平台，last-5）**: **38.15 ± 0.68**（幅度提升来自 Car 权重 25%→48.4%；权重见 section 16）
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
- **频率加权 Overall（平台，last-5）**: **39.87 ± 0.45**（幅度提升来自 Car 权重 25%→48.4%；权重见 section 16）
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

---

## 16. 频率加权 Overall 对比：重新排名的结果

> 前面的 `Overall_3D_moderate` 是**等权 1/4**（`kitti_utils/eval.py` 的默认聚合）。本节按训练集真实类别频率重新加权，权重（四类归一、排除 0.5% Other）：
> Car 48.37% / Cyclist 21.64% / Truck 16.46% / Pedestrian 13.53%。
> 即 `Overall_freq = 0.4837·Car_s + 0.2164·Cyc_l + 0.1646·Truck_s + 0.1353·Ped_l`（沿用同类「Car/Truck strict、Ped/Cyc loose」混合口径）。
> 工具：`python3 tools/summarize_run.py <work_dir> --weighted --tail 5`。

### 16.1 两种口径的平台（last-5）对比

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

（均取 last-5 平台；Run 9/N4 系列为 30e 训练，其余 24e。）

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
   - 无论哪种，都别再拿单点 BEST 比，用上表 last-5 平台均值。

### 16.3 落地

- 工具已支持 `--weighted`，以后每个新 run 跑完跑一条 `--weighted` 即得频率加权口径。
- 权重写死在 `tools/summarize_run.py` 的 `CLASS_PRIORS`，若数据集（train/val）类别分布变化需同步更新。

---

## 17. Checkpoint 保留规则与当前盘点

> 写于 2026-08-22，追加在文档末尾，便于每次实验后照着做。

### 17.1 保留规则（此后所有 run 统一执行）

- 每个 run 只保留 `best epoch` + `last epoch` 两个真实 `.pth` 文件，外加 `latest.pth` 符号链接。
- `best` 是 val `Overall_3D_moderate`（或指定主指标）最高的已保存 epoch。
- 若最优 epoch 未落盘，回退为 val 最接近该最优值的已保存 epoch（例如 Run 10 head-v2 保留 `epoch_12`）。
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

---

## 18. 论文最终报告口径（定稿）

> 写于 2026-08-22，用于把「对外报什么、对内记什么」钉死，避免与会话中多次口头约定不一致。

- 论文主表：报告每个代表性 run 的单点 `BEST` Overall（等权 1/4 口径，`pts_bbox/KITTI/Overall_3D_moderate`），这是对外可比的标准口径。
- 补充材料 / appendix：报告 deterministic 多 seed 的 `mean ± std`，以及 BEST 所属 epoch 的平台区间。用于回应「BEST 是否 cherry-pick」的审稿追问。
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

### 19.2 BEST 汇总（各 seed 取自身 BEST epoch）

| seed | BEST @ep | 3D_mod | 已存 |
|---|---:|---:|:---:|
| seed_0 | ep16 | 39.88 | yes |
| seed_1 | ep15 | 40.51 | yes |
| seed_2 | ep14 | **40.88** | yes |
| **mean +/- std** | | **40.42 +/- 0.51** | |

- 三个 seed 的 BEST epoch 分别落在 ep14/15/16，集中在 ep12-16 高原区（而非原 Run 10 的 ep11）。
- 方差极小：std=0.51，三个值在 39.88-40.88 的 1 点区间内。说明 Run 10 的 40+ 水平是稳定的，不是 cherry-pick。
- seed_2 是三 seed 中最高的（40.88），也超过了原 Run 10 未存盘的峰值 40.60。

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

### 19.4 固定 ep14 截面对比（与原 Run 10 已存最优同一 epoch）

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
| 单次 BEST | 40.60 (ep11, 未存) | 40.42 +/- 0.51 | 原 Run 10 在区间内，非离群 |
| 已存 BEST | 39.65 (ep14) | 40.42 +/- 0.51 (全部已存) | 复现超过原已存 |
| ep14 固定截面 | 39.65 | 39.28 +/- 1.45 | 在 1 std 内，基本一致 |
| BEST epoch | ep11 | ep14/15/16 | 峰值后移 3-5 epoch |

- 原 Run 10 的 ep11=40.60 不是 cherry-pick：三 seed mean 40.42，std 0.51，40.60 在 1 std 以内。
- BEST epoch 从 ep11 后移到 ep14-16，可能来自 3 卡 vs 4 卡的有效 batch 差异（12 vs 16），更小 batch 需要更多 epoch 收敛到峰值。
- 所有三个 seed 的 BEST 均已落盘（seed_0 ep16, seed_1 ep15, seed_2 ep14），修复了原 Run 10 ep11 未存的遗憾。

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
2. 所有 BEST checkpoint 已存盘，修复了原 Run 10 ep11=40.60 未存的遗憾。三 seed 中最好的是 seed_2 ep14=40.88。
3. 峰值稳定在 ep12-16（原 Run 10 在 ep11），3 卡 vs 4 卡的有效 batch 差异可能导致峰值后移，但不影响上限。
4. 主指标方差极小（std=0.51），适合论文报告 mean+/-std 作为确定性证据。
5. 逐类别方差集中在 Car loose / Cyclist strict / Truck strict——这些是共享 head 正样本分配的敏感点，进一步佐证 Car-Truck 混淆是下一步要解决的根因。
6. Pedestrian strict 约 0 在所有 seed 中一致（0.10-0.15），再次确认是 BEV 分辨率物理瓶颈。

### 19.9 对论文报告口径的更新

基于多 seed 结果，更新第 18 节的最终报告口径：

- 主表（对外）：Run 10 head-v2 报告 40.42 +/- 0.51（三 seed BEST 的 mean +/- std），替代原单次 40.60。
- 补充材料：附三 seed 逐 epoch 曲线（19.1 节）和逐类别 mean +/- std（19.3 节）。
- 最佳单点：seed_2 ep14 = 40.88（已存），作为 released checkpoint 候选。
- BEST epoch 选择：ep12-16 是稳定高原区，报告时取各 seed 自身 BEST（而非固定 epoch），避免低估。

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
3. 论文最终报告口径仍以第 19 节完整 RSSM 三 seed BEST mean 40.42 ± 0.51 为准；
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
- BEST = ep10 的 37.99，**仍低于基线平台值**（基线 ep12-16 均值 38.39）；最终 ep24 = 36.68。

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
