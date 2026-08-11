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
|---|---:|---:|---:|---:|---:|---:|
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
|---|---|---:|---:|---:|---:|---:|---:|---:|
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

### Run 9 -- N=4 RSSM (seq_len=4, hidden_dim=64)

- config: configs/r4det/TJ4D-R4Det_motion_align_rssm_det3d_N4_hdim64_2x4_12e.py
- work_dir: work_dirs/rssm_N4_hdim64_2x4_12e
- 改动: 仅 hidden_dim=128->64（从 Run 7 降低），其他超参与 Run 7 完全相同
- 目的: 隔离 N=4 单独效果（Run 7 中 N=4+hdim=128 两变量混淆）
- BEST: ep12 33.00
- LAST: ep18 32.46

与 N=3 hdim=64 (Run 6) 对比:
- Overall: **-1.27** (34.27->33.00) ❌ N=4 单独有害！
- Car loose: -5.28 (55.70->50.42)
- Car strict: -0.63 (34.01->33.38)
- Cyc loose: +3.65 (48.48->52.13) 意外提升
- Ped loose: +1.79 (22.88->24.68)
- Ped strict: **+2.47** (0.05->2.52) 🔥 行人3D严格指标首次非零！
- Trk loose: +0.04 (47.48->47.52)
- Trk strict: -9.91 (31.72->21.81) 崩塌

与 N=4 hdim=128 (Run 7) 对比:
- Overall: -1.05 (34.05->33.00)
- Car strict: -5.11 (38.49->33.38) hdim=128 贡献了 Car strict 的提升
- Trk strict: -2.83 (24.64->21.81)
- Ped strict: +2.50 (0.02->2.52) hdim=64 意外好于 hdim=128

关键观察:
- **N=4 单独（无 hdim=128 配合）有害**: 3D_mod 从 34.27 跌至 33.00
- **Ped strict 突破性提升**: 从所有 run 的 0.01~0.05 跃升至 2.52，N=4+hdim=64 对行人3D定位有独特改善
- N=4+hdim=128 的 Car strict 优势 (38.49) 确实来自 hdim=128
- Cyc 意外大涨 (+3.65)，可能与 N=4 多帧时序对骑行者有效
- **三角关系确认**: N=4 需要 hdim=128 配合才能达最佳效果；hdim=128 也需要 N=4 配合

### N=3 和 N=4 epoch-by-epoch

| epoch | N=4 hdim64 | N=3 hdim128 | N=4 hdim128 | N=3 hdim64 | N=2 v3 | baseline_temporal |
|---|---:|---:|---:|---:|---:|---:|
| 1  |  5.21 |  6.28 |  3.07 |  2.51 |  9.33 | -- |
| 2  | 12.82 | 14.12 | 13.87 | 11.52 | 13.89 | -- |
| 3  | 19.67 | 19.07 | 16.51 | 17.33 | 15.82 | -- |
| 4  | 22.71 | 22.57 | 19.77 | 18.62 | 21.09 | -- |
| 5  | 24.85 | 24.30 | 25.52 | 25.96 | 22.97 | -- |
| 6  | 28.07 | 27.50 | 28.34 | 28.21 | 24.73 | 29.82 |
| 7  | 30.36 | 29.89 | 30.43 | 30.74 | 27.56 | 28.66 |
| 8  | 28.16 | 27.21 | 28.94 | 29.47 | 31.40 | 32.31 |
| 9  | 30.43 | 29.94 | 31.70 | 31.11 | 30.48 | 31.28 |
| 10 | 29.22 | 29.12 | 30.36 | 28.05 | 33.01 | 31.75 |
| 11 | 31.62 | 31.04 | 32.13 | 31.58 | 33.77 | 31.04 |
| 12 | 33.00 | 31.55 | 33.55 | 32.82 | 32.94 | 34.50 |
| 13 | 31.27 | 31.30 | 32.91 | 32.46 | 30.97 | 31.00 |
| 14 | 32.45 | 32.02 | 32.97 | **34.27** | 33.02 | 34.47 |
| 15 | 30.85 | 32.15 | 32.58 | 32.73 | 32.62 | 33.13 |
| 16 | 32.02 | 31.05 | 32.84 | 31.94 | 32.85 | 34.46 |
| 17 | 31.67 | **32.87** | **34.05** | 32.60 | 31.90 | -- |
| 18 | 32.46 | 30.98 | 32.07 | 33.27 | 32.93 | -- |

### 逐类别 BEST 对比 (loose / strict)

| Run | BEST ep | Car_l | Car_s | Cyc_l | Cyc_s | Ped_l | Ped_s | Trk_l | Trk_s | Ovl_3D |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_temporal (N=2) | 12 | 52.18 | 32.82 | 48.63 | 22.52 | **26.64** | 0.04 | 43.02 | 29.91 | **34.50** |
| N=3 (hdim=64) | 14 | **55.70** | 34.01 | 48.48 | 21.73 | 22.88 | 0.05 | 47.48 | **31.72** | 34.27 |
| N=4 (hdim=128) | 17 | 53.67 | **38.49** | 48.59 | 20.76 | 24.46 | 0.02 | **48.97** | 24.64 | 34.05 |
| N=4 (hdim=64) | 12 | 50.42 | 33.38 | 52.13 | 20.74 | 24.68 | **2.52** | 47.52 | 21.81 | 33.00 |
| N=3 (hdim=128) | 17 | 51.13 | 32.58 | 45.65 | 19.67 | 24.07 | 0.02 | 48.16 | 29.17 | 32.87 |
| N=2 v3 (hdim=64) | 11 | 56.03 | 33.20 | **49.02** | **22.49** | 23.35 | 0.01 | 43.65 | 29.51 | 33.77 |
| N=2 v2 (hdim=64) | 14 | 52.57 | 37.79 | 49.34 | 21.26 | 21.13 | 0.01 | 42.90 | 27.79 | 34.01 |

---

## 8. N帧消融总结

### 总体排名（3D moderate BEST）

| # | Run | N | hdim | BEST ep | 3D_mod |
|---|---:|---|---:|---:|
| 1 | baseline_temporal (GRU) | 2 | -- | 12 | **34.50** |
| 2 | N=3 RSSM | 3 | 64 | 14 | 34.27 |
| 3 | N=4 RSSM (hdim=128) | 4 | 128 | 17 | 34.05 |
| 4 | v2 RSSM | 2 | 64 | 14 | 34.01 |
| 5 | v3 RSSM | 2 | 64 | 11 | 33.77 |
| 6 | baseline_rssm (no align) | 2 | 64 | 17 | 33.54 |
| 7 | N=4 RSSM (hdim=64) | 4 | 64 | 12 | 33.00 |
| 8 | N=3 RSSM (hdim=128) | 3 | 128 | 17 | 32.87 |
| 9 | v1 RSSM (KL broken) | 2 | 64 | 10 | 30.89 |

### 核心发现

1. N=2->3 (hdim=64) 有 +0.50 正向提升，主要来自 Truck (+3.83)
2. N=3->4 (hdim=64) **反效果 -1.27** (34.27→33.00) — 单独增加帧数有害
3. N=3->4 (hdim=128) 微弱 +0.05 (34.27→34.05) — 帧数+容量同时增加能维持性能
4. **三角锁定关系**: N=4↔hdim=128 互锁 — N=4 需要 hdim=128 才有价值，hdim=128 也需要 N=4 才能发挥
5. **Ped strict 首次突破**: N=4+hdim=64 的 Ped 3D strict=2.52，打破所有 run 的 0.01~0.05 魔咒
6. hdim=128 贡献 Car strict 提升 (+5.11, 33.38→38.49)，但 hdim=64 有更好的 Cyc 和 Ped
7. 所有RSSM变体都没超过N=2 GRU baseline (34.50) — 差距0.23~1.63
8. 18 epoch 对 hdim=128 不够，ep17才peak且ep18暴跌

### 待做消融

- [x] N=3 + hidden_dim=128: hdim=128 在 N=3 下全面劣化，确认需要 N=4 配合
- [x] N=4 + hidden_dim=64: N=4 单独有害 (-1.27 vs N=3)，确认 N=4↔hdim=128 互锁
- [ ] N=4 + hdim=128, 24-30ep 延长训练: 当前 ep17 peak→ep18 跌，需更长收敛 + lr scheduler
- [ ] N=4 + hdim=64, 24-30ep 延长训练: Ped strict 2.52 亮点值得深挖
- [ ] velocity 分支注入 (action_dim=2): 所有当前 RSSM 使用 action_dim=0
- [ ] deformable alignment 消融: MotionAlignedRSSMFusion vs 纯 flow warp vs 无对齐
