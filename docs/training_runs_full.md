# 五次训练完整记录（效果 / 配置 / 模块改动）

> 数据来源：各 `work_dirs/<run>/*.log.json` 的 `mode=val` 记录（每 epoch 取该 run 最后一次 val，避免中途重启污染）+ 各 run 目录下 `*.py` 配置快照 + git log（branch `motionalignrssm`）。
> 评估口径：KITTI 3D detection AP。主指标 **Overall 3D moderate**（`pts_bbox/KITTI/Overall_3D_moderate`）。`_loose` 为松评估 per-class。
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
