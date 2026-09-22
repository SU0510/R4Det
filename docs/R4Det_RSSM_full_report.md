# R4Det + RSSM 项目全景研究报告

> **文档定位**：面向汇报的单份完整研究记录。覆盖项目网络结构、代码实现、全部 40+ 次训练实验的目的/改动/结果/失败原因，以及本次审计对每条结论的独立验证情况。
> **写作日期**：2026-09-19。基于仓库 commit `a76d194`（分支 `goal-test`）时的代码、`/data/lurui/work_dirs/` 全部日志与 checkpoint、git 全部 196 个提交。
> **证据等级标注**：
> - 【已验证】= 本报告作者独立从代码/日志/checkpoint/git 复核过；
> - 【记录口径】= 来自 `docs/training_runs_full.md`（下称《训练记录》）等已有文档，本次未能或未需复核（多数为已被删除的早期 run）；
> - 【未知/待验证】= 现有材料无法确认，禁止当作结论使用。

---

## 目录

1. [项目概览](#1-项目概览)
2. [Baseline 网络](#2-baseline-网络)
3. [当前 RSSM 网络](#3-当前-rssm-网络)
4. [完整数据 / Tensor 流](#4-完整数据--tensor-流)
5. [RSSM 数学机制与代码对应](#5-rssm-数学机制与代码对应)
6. [全部实验时间线](#6-全部实验时间线)
7. [实验矩阵总表](#7-实验矩阵总表)
8. [每次训练详解：结果与问题](#8-每次训练详解结果与问题)
9. [失败原因分类与证据链](#9-失败原因分类与证据链)
10. [Baseline / 原时序模块 / RSSM 差异总表](#10-baseline--原时序模块--rssm-差异总表)
11. [当前可信结论](#11-当前可信结论)
12. [尚未解决的问题与未知项](#12-尚未解决的问题与未知项)
13. [下一步实验建议](#13-下一步实验建议)
- [附录 A：一致性核查记录](#附录-a一致性核查记录)
- [附录 B：主线配置关键快照](#附录-b主线配置关键快照)
- [附录 C：复现入口与工具](#附录-c复现入口与工具)
- [附录 D：证据文件索引](#附录-d证据文件索引)

---

## 1. 项目概览

### 1.1 任务与代码基础

- **任务**：4D 毫米波雷达 + 单目相机的多模态 BEV 3D 目标检测（四类：Pedestrian / Cyclist / Car / Truck），时序融合。
- **代码基座**：mmdet3d 框架上的 **R4Det**（TJ4DRadSet 论文官方代码，2025-05 落库，`main` 分支至今停在 `9418d11`，无 RSSM 内容）。本项目全部 RSSM/时序/检测头工作在其分支链上完成：
  `trainbaseline` → `trainrssm` → `motionalignrssm` → `Nframerssm` → `detection_head_v2` → `radar_static_dynamic` → **`goal-test`（当前，HEAD `a76d194`）**。【已验证：`git branch -a` 与各分支 tip】
- **工作时段**：2026-07-26（RSSM 首个提交 `256f0c8`）至 2026-09-19，共 196 个提交。

### 1.2 数据集【已验证】

| 项 | 值 | 证据 |
|---|---|---|
| 数据集 | TJ4DRadSet（TJ4D），雷达+相机 | `docs/guidance/dataset.md`，`/data/TJ4D/` |
| train / val | **5706 / 2040** 帧（`TJ4D_infos_train.pkl` / `TJ4D_infos_val.pkl`） | pkl 实际计数 |
| 训练集类别分布 | Car 11649 (48.4%)、Cyclist 5212 (21.6%)、Truck 3963 (16.5%)、Pedestrian 3257 (13.5%)，共 24081 框 | pkl 实际计数 |
| val 集 GT | Car 4361、Cyc 2153、Truck 1404、**Ped 999** | pkl 实际计数（与《训练记录》§29.1/§41.5 完全一致） |
| 数据限制 | **无 ego 位姿 / 无 timestamp / 无速度标注**（pose 目录为空）→ action/velocity 分支接不上真实信号 | `work_dirs/rssm_diagnosis.md`（07-31 诊断） |
| 雷达点格式 | `load_dim=8, use_dim=[0,1,2,3,5]`：x, y, z, 径向速度 v(维3), SNR(维4)，共 5 通道 | 主配置 `LoadPointsFromFile` |

### 1.3 评估口径【已验证】

- KITTI 式 3D/BEV/2D AP，**AP40（40 点）**；难度 easy/moderate/hard，逐 GT 预计算 difficulty（commit `7374435`：三档全不达标者记 -1 忽略而非 hard）。
- 每类两套 IoU 阈值：**strict**（Car 0.7 / Ped & Cyc 0.5）与 **loose**（统一 0.25）。
- **主指标 `Overall_3D_moderate` 的真实构成**（`mmdet3d/core/evaluation/kitti_utils/eval.py:925-937`，拼接后取均值）：

  ```
  Overall = mean( Pedestrian_loose, Cyclist_loose, Car_strict, Truck_strict )
  ```
  即「Car/Truck 看 strict、Ped/Cyc 看 loose」的混合口径。**读任何 Overall 数字都必须按此理解**；谁强谁弱会随口径改变（见 §8.4 的频率加权重排名）。
- 频率加权口径（《训练记录》§16，工具 `tools/summarize_run.py --weighted`）：`0.4837·Car_s + 0.2164·Cyc_l + 0.1646·Trk_s + 0.1353·Ped_l`，权重与实测类别分布吻合【已验证权重≈实际分布】。

### 1.4 一句话总结整个项目

> 把 RSSM（循环状态空间模型）接进 R4Det 的 BEV 时序融合位置，经过 5 个阶段（模块落地 → 帧数/容量消融 → 预训练+检测头 → 机制拆解 → 类别专项），最终**主线模型 = 预训练 backbone + N=4 Motion-Aligned RSSM + head-v2**，三 seed 复现 **Overall 3D moderate = 40.42 ± 0.51**【已验证】，显著优于无时序基线 35.59 与 GRU 时序基线（18e 口径 34.50）；Pedestrian strict ≈ 0 被证实为 BEV 分辨率/点云稀疏的物理瓶颈，所有试图修复它的支线（refinement / CenterHead / 高分辨率 BEV / 尺寸先验 / probe）均未通过门控。

---

## 2. Baseline 网络

### 2.1 原始 R4Det（单帧，main 分支 / 预训练用）

模型 `mmdet3d/models/detectors/R4Det.py`（`R4Det(MVXFasterRCNN)`）。主线训练配置只保留 3D 检测监督（`use_depth_supervision / use_props_supervision / use_msk2d_supervision = False`），2D RPN/ROI、IGDR、backward_projection、voxelpainting 全部关闭（`use_sa_radarnet=False` 等）。【已验证：主配置】

单帧前向（`extract_feat`，R4Det.py:761-955）：

```
radar 点 (N,5) ─ 中分辨率体素化(0.16m) ─ RadarPillarFeatureNet(5→12维装饰→64ch)
   ─ PointPillarsScatter → [496,432] 0.16m BEV
   ─ SECOND(3 段, strides 2, out 64/128/256) ─ SECONDFPN → (B,384,248,216) 雷达 BEV
camera (480×640) ─ ResNet50(+FPN 256ch) ─ GeometryDepth_Net(上下文+离散深度 72 bins)
   ─ ViewTransformerLSS(downsample=8)：context×depth 外积后按 (x,y,z) 网格 splat 池化
     → (B,256,216,248,1)（z 仅 1 格，zbound 步长 6.0）→ z 维求均值 → (B,256,216,248)
     → permute → (B,256,248,216) 相机 BEV（与雷达 BEV 同画布）
两路 BEV ─ ConcatConvFusion(cat 256+384 → 3×3 conv → 256) → 216×248 BEV（断言 bev_h_=216, bev_w_=248）
   ─ Anchor3DHead（单层，12 anchor × 4 类）
```

- 体素化用了 **voxel_size/2 = 0.16m** 的 pillar + Scatter 到双倍网格 `[496,432]`，再由 FIRST 层 SECOND stride 2 回到 0.32m 等效分辨率——这是 §32.7「必须重做 0.16m scatter」这一设计的来源。【已验证：主配置 `pts_voxel_layer.voxel_size=[0.16,0.16,6.0]`、`pts_middle_encoder.output_shape=[bev_w_*2, bev_h_*2]`】
- 相机深度：`GeometryDepth_Net`（use_radar_depth=False 时纯图像深度），`dbound=[1,73,1]`；主线关闭深度监督（`loss_depth_prob=0.0`、`use_depth_supervision=False`），但 depth_net 仍前向产生 context/depth 供 LSS 使用。【已验证】

### 2.2 时序 Baseline：`TemporalDeformableFusion`（Run 1，GRU）

`mmdet3d/models/fusion_layers/temporal_r4det_fusion.py`（120 行，全文已读）【已验证】：

```
输入: feat_curr(B,256,H,W), feat_prev(上一帧真实 BEV 特征)
offset_mask_generator: cat(curr,prev) → offset+mask (零初始化=identity 起步)
h̃ = ModulatedDeformConv2d(prev, offset, mask)     ← 把上一帧 BEV warp 对齐到当前帧
z,r = σ(conv_z / conv_r(cat(curr, h̃)))             ← GRU 式门控 (1×1 conv)
h_t = (1−z)·feat_curr + z·conv_h(cat(curr, r·h̃))
output = output_layer(h_t)                          ← 3×3 ConvModule，Xavier 初始化，主动改造特征
```

- 特点：历史来源是**上一帧的真实 BEV 特征**（无内部状态、无随机性、无 KL）；输出层 Xavier 初始化，从一开始就改造特征。
- 18e 成绩 BEST **34.50 @ep12**【记录口径：work_dir 已删除，日志无法复核】。

### 2.3 No-Temporal Baseline（§20）

`temporal_fusion=None`、其余与主线（head-v2、pretrained、N4 数据、24e）完全一致：BEST 35.59 @ep20，ep12-16 均值 34.65【已验证：日志复算 35.5903 / 34.6496，逐位一致】。注意它仍用 seq_len=4 的数据管线，只是不用时序融合。

---

## 3. 当前 RSSM 网络

主线 = `MotionAlignedRSSMFusion`（rssm_fusion.py:501-725）+ Anchor3DHead v2。以下全部【已验证：代码逐行读过】。

### 3.1 模块构成（rssm_fusion.py）

| 组件 | 实现 | 作用 |
|---|---|---|
| `encoder` | ConvModule 256→hidden→latent（3×3×2） | 观测编码 e_t = encoder(feat_t) |
| `transition` | `ConvGRUCell`（reset/update/candidate 三个 3×3 conv） | 确定性状态传播 h_t = GRU(z_{t-1} 对齐[, action], h_{t-1} 对齐) |
| `align_h_*` / `align_z_*` | offset/mask conv（**零初始化=identity 起步**）+ `ModulatedDeformConv2d` | 把 h_{t-1}/z_{t-1} warp 到当前帧（motion-align） |
| `prior_mu / prior_logstd` | 3×3 conv on h_t | p(z_t\|h_t) |
| `posterior_mu / posterior_logstd` | 3×3 conv on [h_t, e_t] | q(z_t\|h_t, e_t) |
| `decoder` | ConvModule latent+h → hidden → in_channels | 重建 feat（产生 recon loss） |
| `output_proj` | 3×3 conv latent→256 | 输出 = output_proj(z_t) **+ feat**（残差） |
| 状态 | `h_state/z_state` 成员变量，`reset_state()`/`reset_for_samples(mask)` | 跨 forward 调用维护；`reset_for_samples` 已改为非 in-place（BPTT autograd 兼容，`d4406b2`） |

变体类（同一文件内，均【已验证】）：

| 类 | 相对完整 RSSM 的差异 | 用于 |
|---|---|---|
| `BEVRSSMTemporalFusion` | 无 motion-align（Run 2） | 早期 baseline |
| `MotionAlignedRSSMFusion` | 主线 | Run 3 以后全部 |
| `DeterministicMotionAlignedLatentFusion` | 删 prior/logstd/KL/采样，z=μ_q | §21 |
| `FixedNoisePosteriorLatentFusion` | 上述 + 训练期 z=μ_q+0.1·N(0,I) | §23 |
| `PosteriorOnlyLearnableStdLatentFusion` | 保留 posterior_logstd，删 prior/KL | §24 |

### 3.2 检测头 v2（Anchor3DHead 改造，anchor3d_head.py）

- anchor：6 组 size（Ped×3 [0.5,0.6,1.60]/[0.6,0.8,1.69]/[0.75,0.9,1.80]、Cyc [0.78,1.77,1.60]、Car [1.84,4.56,1.70]、Truck [2.66,10.76,3.47]）× 2 朝向 = **12 anchor**；`anchor_class_mapping=[0,0,0,1,2,3]`。
- `ignore_dir_classes=[0]`：行人关闭方向分类（dir loss 权重置零）。
- `use_iou_branch=True`：`conv_iou` 每 anchor 1 通道；训练目标为**中心度代理** `exp(−BEV中心距/框对角线)` 的 L1（anchor3d_head.py:592-612），**推理时不参与打分**（`get_bboxes` 不读它）。
- 可选开关（默认关，全部已实现）：`confusion_pairs`（Car↔Truck 互斥 logit 压制 BCE）、`truck_refine`/`truck_refine_detach`、`truck_tower`、`ped_refine(7D, detach)`、`shared_stem`、`cyc_cls_branch`。

### 3.3 当前仓库状态的重要警告（本次审计发现）

1. **主配置 ≠ clean 基线**：`TJ4D-R4Det_motion_align_rssm_det3d_N4_2x4_24e_pretrained_v2_head.py` 当前含 `shared_stem=True`（commit `b4fd7be` 09-15 加入，§41 门控失败后**从未回退**）。想复跑 clean full-RSSM 必须先把 `shared_stem=True` 改回 False（或从 `work_dirs/run10_headv2_multiseed/seed_*/` 的配置快照复制）。【已验证：读当前配置 + git】
2. ~~**`BEVRSSMTemporalFusion.forward` 缺 return**~~（**2026-09-22 已修复**）：commit `99c74ff`（08-25，移动 Deterministic 类时）误删了基类 forward 末尾的 `return output, reconstruction, kl*kl_scale, h_t, z_t, stats`，使 `baseline_rssm` 配置返回 None、不可直接复跑。该行已恢复（`mmdet3d/models/fusion_layers/rssm_fusion.py`），修复后该方法与 `99c74ff^` 版本 AST 一致、`baseline_rssm` 可复跑。历史 Run 2（07-27）训练时该行本就存在，故其 33.54 不受影响，无需重估。【已验证：git show 对比 + git log -L + 单测 + 实跑 forward】
3. **`baseline_rssm` 配置文件被后来覆盖**：现文件是 kl_scale=1.0/free_nats=1.0/warmup 0→1.0(ep0-5)（v2/v3 风格），与《训练记录》§4 所记 Run 2 实际训练值（kl_scale=0.1 / free_nats=0.0 / warmup 0→0.1 ep0-3）不符 → Run 2 的真实超参现在只能信记录。【已验证：读现配置 vs 记录】
4. 预训练 checkpoint `checkpoints/pretrained_tj4d.pth`（1.09GB，1421 个张量）内含：img_backbone/img_neck/depth_net/pts_voxel_encoder/pts_backbone/pts_neck + **Cross_Modal_Fusion 的 `cross_attention.*` 权重**（主线 ConcatConvFusion 不加载）+ 一个 **`temporal_fusion.*` 是 GRU 基线模块的权重**（与 RSSM 结构不匹配 → 每次训练 RSSM 均为随机初始化）。【已验证：checkpoint 键值枚举】

---

## 4. 完整数据 / Tensor 流

以主线（N=4, 24e, batch 3卡×2×累积2=12）为准。B=batch（单卡 2），单帧 BEV 网格 216×248（0.32m）。

### 4.1 数据集侧（TJ4D_dataset.py:322-380）【已验证】

- `__getitem__` 返回 `seq_len` 帧序列（**old→new**，最后一帧=当前帧）。有效性按 `image_idx` 连续性判定：一旦断档，该帧及更早帧全部 invalid，用空占位帧替代并标记 `is_prev_frame_valid=False`。
- 训练时每帧独立做增广（同一 seed 重置）；历史帧 GT 也加载（供 depth/proposal 等潜在监督；主线只用品当前帧 GT 算检测 loss）。

### 4.2 模型侧逐层 shape（R4Det.forward_train，R4Det.py:1254-1528）

```
对每个 batch 样本 i, 帧序列 t=0..3 (N=4):

RSSM 状态:   h_state ∈ (B,256,216,248),  z_state ∈ (B,256,216,248)   # latent_dim=256=in_channels
             序列开始 reset_state() → h_0=z_0=0（不是从观测初始化）

帧 t (历史):
  radar (N_p,5) → pillar(0.16m,≤10点) → PFN 64ch → Scatter (B,64,432,496)
    → SECOND+FPN → radar BEV (B,384,248,216)
  img (B,3,480,640) → ResNet50+FPN (B,256,60,80) → depth_net → context+depth(72bins)
    → LSS → (B,256,8,216,248) 沿深度求和 → camera BEV (B,256,248,216)①
  ConcatConvFusion: cat(256,384)→conv→ (B,256,248,216) → permute → (B,256,216,248)
  RSSM(burn-in 段: no_grad, detach_state=True):
     h_a = DeformAlign(h_{t-1}; feat),  z_a = DeformAlign(z_{t-1}; feat)
     h_t = ConvGRU(z_a, h_a)            # (B,256,216,248)
     e_t = encoder(feat)
     μ_q,σ_q ← posterior([h_t,e_t]); μ_p,σ_p ← prior(h_t)
     z_t = μ_q（burn-in 段 deterministic=True）
     状态存储: h_state=h_t.detach(), z_state=z_t.detach()
  ↓ 仅最后 rssm_bptt_steps=1 个历史帧: detach_state=False（计算图保留到本步）
  ↓ 若某样本最后历史帧 invalid: 当前帧输出回退为未融合 feat（torch.where 掩码, R4Det.py:898-901）

帧 t=3 (当前, feat_or_dict=1):
  同上前向（有梯度）;  z_t = sample(μ_q, σ_q)（训练, deterministic=False）或 μ_q（推理）
  KL = free-bits-clamped( q‖p ) × kl_scale          # 标量
  recon = decoder([h_t,z_t])                         # (B,256,216,248)
  output = output_proj(z_t) + feat → pts_feats[0]    # (B,256,216,248)
  → Anchor3DHead:
     shared_stem(若开): x = x + 0.1·stem(x)
     conv_cls 1×1 → (B,48,216,248)   # 12 anchor × 4 类
     conv_reg 1×1 → (B,84,216,248)   # 12 anchor × 7 维 box code
     conv_dir 1×1 → (B,24,216,248);  conv_iou 1×1 → (B,12,216,248)
```

① LSS 输出 [b,c,x,y,z]=(B,256,216,248,1)，`mean(-1)` 池化 z 维（ViewTransformerLSS.py `voxel_pooling` 返回 [b,c,x,y,z]），再 permute(0,1,3,2)（R4Det.py:800-801）；72 个深度 bin 已在 lift 阶段（depth×feat 外积）被消耗。

### 4.3 Loss 汇总（训练）

| loss | 公式/权重 | 证据 |
|---|---|---|
| `loss_cls` | FocalLoss(use_sigmoid, α=0.25, γ=2.0), w=1.0；MaxIoUAssigner：Ped/Cyc pos≥0.35/neg≤0.2，Car/Truck pos≥0.5/neg≤0.35 | 主配置 |
| `loss_bbox` | SmoothL1(β=1/9), w=2.0, sin-difference 处理 yaw；**注意**：顶层 `code_weights=[2,2,1,…]` 并未接入 train_cfg（train_cfg.pts 无 `code_weight` 键）→ 对 anchor head 不生效（死配置） | anchor3d_head.py:560-563 + 主配置 |
| `loss_dir` | CE, w=0.2；Pedestrian 权重置零（ignore_dir_classes=[0]） | anchor3d_head.py:576-586 |
| `loss_iou` | L1( iou_pred, exp(−center_dist/diag) )，训练 only | anchor3d_head.py:592-612 |
| `loss_rssm_kl` | free-bits clamp(KL(q‖p), min=free_nats) 均值 × kl_scale（hook 线性 warmup 0→1.0 over ep0-10） | rssm_fusion.py:327-370 + hook |
| `loss_rssm_recon` | MSE(decoder([h,z]), feat)，BPTT 窗口+当前帧取平均 | R4Det.py:1376-1387 |
| `stat_*` | KL/std/mu_diff² 等 6 项监控，只记日志不进 loss | rssm_fusion.py:359-368 |

### 4.4 推理（simple_test，R4Det.py:1070-1251）

`reset_state()` → 历史 3 帧 `no_grad` 前向仅更新状态 → 当前帧 `z_t=μ_q`（确定性后验均值）→ 解码 NMS（`nms_pre=1000, nms_thr=0.5, score_thr=0.0, max_num=300`）。**prior 在推理路径上完全不参与**。若有 ped_center_head（Ped 支线），按「删 Anchor 头 Ped 框 + CenterHead 框全部置 label=0 + 拼接（无跨类 NMS）」合并（`_simple_test_pts_dual`，R4Det.py:1586-1626）。

---

## 5. RSSM 数学机制与代码对应

### 5.1 状态空间公式 ↔ 代码

| 数学对象 | 公式 | 代码位置（rssm_fusion.py） |
|---|---|---|
| 确定性状态 h_t | `h_t = ConvGRU( align(z_{t-1})[, action], align(h_{t-1}) )`；r/u/h̃ 为标准 GRU 门（式见 ConvGRUCell docstring） | `MotionAlignedRSSMFusion.forward` L672-688；`ConvGRUCell.forward` L44-61 |
| 随机状态 z_t（后验） | `q(z_t\|h_t,e_t)=N(μ_q, σ_q²)`，e_t=encoder(feat_t)；训练 `z_t=μ_q+εσ_q`，推理 `z_t=μ_q` | L694-705；`sample()` L320-325 |
| 先验 p(z_t\|h_t) | `N(μ_p, σ_p²)`，由 h_t 过 3×3 conv | L690-692 |
| 转移观测/action | action=ego velocity，`action_dim=2` 时代**恒为零**（数据集无位姿）；08-05 起 `action_dim=0` 整体移出计算图 | L657-663；`b64dd1f` |
| 观测编码 e_t | 2 层 ConvModule 256→128(hdim)→256(latent) | L694-695, encoder 定义 L137-154 |
| 解码器/重建 | `x̂_t = decoder([h_t, z_t])` → `loss_rssm_recon = MSE(x̂_t, feat)` | L710-711 + R4Det.py:1376-1387 |
| 输出 | `output = output_proj(z_t) + feat`（残差融合回 BEV） | L713-715 |
| KL | 逐元素高斯 KL，`min_logstd+softplus(·)` 软下界 + `clamp(max=0)`（σ≤1），free-bits `clamp(kl, min=free_nats)`，×kl_scale | `kl_loss()` L327-370 |
| σ 约束与初始化 | logstd bias 解析解使初始 σ≈0.2（min_std=0.1, init_std=0.2） | `_logstd_bias_value()` L246-260 |
| KL 调度 | `KLScaleSchedulerHook.before_train_epoch` 直接改 `temporal_fusion.kl_scale`（按 runner.epoch，resume 安全） | `mmdet3d/core/hook/kl_scale_scheduler.py` |
| 截断 BPTT | 序列长 N=4；前 `N-1-bptt` 帧 no_grad burn-in；最后 1 帧 `detach_state=False` 保留图；当前帧全梯度；窗口内 KL/recon 取平均 | R4Det.py:1313-1346, 1376-1384 |

### 5.2 训练 vs 推理的行为差异【已验证】

| | 训练 | 推理 |
|---|---|---|
| z_t 来源 | `sample(μ_q, σ_q)`（重参数化） | `μ_q`（确定性） |
| 历史帧 | burn-in（无梯度）+ 1 帧带梯度 | 全部 no_grad |
| prior | 参与 KL（受 warmup 缩放） | 不参与 |
| recon loss | 有 | 无 |
| 状态复位 | 每个训练 step `reset_state()`；invalid 样本 `reset_for_samples` | 每个测试序列 `reset_state()` |

### 5.3 实测动力学（posterior collapse）【已验证：run 日志 stat_* 字段】

- 所有完整 RSSM run（含 BPTT）收敛后 `stat_clamped_ratio→99~100%`、`stat_mu_diff²→0.008~0.04`、`post_std≈prior_std`：**prior 与 posterior 几乎重合、KL 梯度归零**（《训练记录》§9.5、§14.4、§15.5；`rssm_diagnosis.md`）。推理只用 μ_q，因此该现象对当前 AP 中性甚至有利（posterior 自由编码观测）——这是 07-31 诊断文档的核心结论，后续 KL0/Deterministic 系列消融（§21-25）进一步证实。
- learnable-std 变体（§24）中 posterior std 从 ep6 起压到 min_std≈0.10 平台（P10/P50/P90 全挤在 0.100–0.103）——随机性实际消失。

---

## 6. 全部实验时间线

（日期=commit 日期【已验证：`git log --date=iso`】；Run 编号沿用《训练记录》）

### 阶段 0：基座（2025-05 ~ 2026-07-25）
原 R4Det 论文代码落库、数据准备（main/trainbaseline 分支）。

### 阶段 1：时序融合模块对比（07-26 ~ 08-05，分支 trainrssm→motionalignrssm）
| 日期 | commit | 事件 |
|---|---|---|
| 07-26 | `256f0c8` | RSSM v1 落地（BEVRSSMTemporalFusion + KL hook + R4Det 接 6 元组）→ **Run 2** |
| 07-27 | `2168bed` | rssm v1 update（output_proj 零初始化确立） |
| 07-31 | `57da70f` | MotionAlignedRSSMFusion（deform 对齐）→ **Run 3** |
| 07-31 | `c10307f` | KL 超参修正（kl_scale 1.0 / free_nats 1.0 / warmup 5）→ **Run 4** |
| 08-01 | `f39dedd`/`bc06162`/`74283ff` | logstd 上界 σ≤1、IoU CUDA、tqdm → **Run 5** |
| ~08-05 | — | `rssm_diagnosis.md` 诊断：KL 死是稳态、velocity 恒零、零初始化是起步慢主因 |
| 08-05 | `b64dd1f` | **output_proj 零初始化→Xavier** + action_dim=0 移除 velocity 分支 |

### 阶段 2：N 帧与 hidden_dim 消融（08-06 ~ 08-15，分支 Nframerssm）
08-06 seq_len=3 支持（`80742aa`）→ Run 6 (N3/hdim64, BEST 34.27)；08-09~08-11 N4/hdim128 → Run 7 (34.05)、Run 8 (N3/hdim128, 32.87)、N4/hdim64 (33.00)；08-12 下载 TJ4D 官方预训练权重（detectron2 stub 系列 fix）；08-13 **head-v2**（`d24e1e4`：Ped 3 anchor + ignore_dir + IoU branch；KL warmup 12→10、30e→24e）；08-13 Run 9（N4+pretrain 30e, BEST ep19 37.94）；08-15 **Run 10**（head-v2 24e, 已存 BEST ep14 39.65 / 峰 ep11 40.60 未存盘）。

### 阶段 3：anchor / mask / BPTT 变体（08-15 ~ 08-20）
Run 11 Truck×3 anchor（38.45，Car −5）；Run 12 Car-large anchor（38.11，Truck −5.98）；Run 13 Doppler 动静 mask（36.21，全场回退）；Run 14 N4 BPTT（37.84）；Run 15 N2 BPTT（38.08）。08-17 自主实验协议文档（`f017cda`）。08-21~08-24 **Run 10 三 seed 复现**（deterministic、3 卡）：40.42±0.51，seed2 ep14=40.88 为 released 候选（§19）。

### 阶段 4：RSSM 机制消融（08-24 ~ 08-31）
§20 No-Temporal（35.59）→ §21 Deterministic latent（38.44）→ §22 KL=0 seed0（40.35）→ §23 FixedNoise（38.42）→ §24 Posterior-only learnable-std（39.66@ep10，窗口 37.04）→ §25 KL0 三 seed 复现 → **结论：保留完整 RSSM（KL/prior + posterior sampling 共同必需）**。

### 阶段 5：Truck 专项（09-01 ~ 09-05）
§26 Truck residual refine（Truck +2.85 但 Cyc −4.94）→ §27 detach 版（否证梯度干扰假设；Truck −2.79）→ §28 独立 tower（Truck +1.46，Cyc −5.65）→ **三轮全部不通过，Truck 回归路线停止**；共享 BEV 特征容量被判定为 Car/Truck/Cyclist 零和瓶颈。

### 阶段 6：Pedestrian 专项（09-05 ~ 09-14）
§29 误差拆解（λ-recall@0.5=4.5% 决定性证据；w/yaw 误差主导）→ 7D ped refine（不通过）；§30 rotation 2→4（不通过，rotations 跨类共享）；§31 独立 CenterHead stage-1（隔离 0 差异通过、Ped strict 仍 0.12）；§32 诊断+NMS sweep（w 主导、NMS 非瓶颈）；§33/34 零训练固定/软尺寸先验（**strict 0.05→2.41 @Aα0.75**）；§35 有界残差训练（无效）；§36 delta_xy（**IoU 输入编码 bug，作废**）；§37 修复后重跑（仍不通过，低分辨率路线关闭）；§38 高分辨率 0.16m BEV 支线（含**转置布局 bug 38.6、残差死分支 38.10** 两次工程事故修复后仍不通过；几何/yaw probe 失败）→ §39 最终多 seed：stage1 ep7+Aα0.75 = 「Ped strict 优先工作点」（strict +2.25 均值，loose −1.89，Overall −0.47，seed2 大退化）→ **Ped 训练路线正式关闭**。

### 阶段 7：结构门控与 Cyclist（09-14 ~ 09-19）
§40 恢复预训练 Cross-Modal Fusion（Truck +3.27 但 Car −2.04/Ped −3.23，不通过）→ §41 共享残差 stem（Overall +0.88/Car +3.15 但 Cyc −4.68，不通过；Cyclist 漏检诊断 + 梯度 cosine 排除负冲突）→ §42 max_num 300→600 推理消融（无变化）→ §43 Cyclist oracle recall vs PR（几何损失小、排序质量损失大）→ §44 Cyclist 分类残差分支（保留 shared stem）：seed0「方向成功」（Overall +1.18 / Cyc 仅 −0.72），**seed1 复现失败**（Ped −3.78 / Truck −3.03 破红线）；唯一跨 seed 稳健信号 = Car strict +2.9。seed2 于 09-19 02:36 启动，**进行中**。

---

## 7. 实验矩阵总表

状态分类：✅完成且达标 ｜ ⚠️完成但未达标/效果不佳 ｜ ❌明确失败 ｜ 🛠️代码/配置问题 ｜ ⏸️未完成 ｜ ❓无法确认。

| Run | 名称 | 唯一变量 | BEST Overall 3D_mod | 状态 | 本报告核验 |
|---|---|---|---:|---|---|
| 1 | baseline_temporal (GRU) | TemporalDeformableFusion | 34.50 @ep12 (18e) | ✅（当时最强 baseline） | 【记录口径】日志已删 |
| 2 | baseline_rssm | 无对齐 RSSM | 33.54 @ep17 (18e) | ⚠️ KL 压死 | 【记录口径】；现配置与当时超参不符（§3.3-3） |
| 3 | motion_align v1 | +deform 对齐（kl0.1/free0） | 30.89 @ep10 | ❌ 踩坑跑 | 【记录口径】 |
| 4 | v2 | KL 超参放开+续训 v1 | 34.01 @ep14 | ⚠️ 含续训贡献 | 【记录口径】 |
| 5 | v3 | logstd 上界, 从头训 | 33.77 @ep11 | ⚠️ 持平略降 | 【记录口径】+ 统计已验证（diagnosis.md） |
| 6 | N=3 hdim64 | seq_len 2→3 | 34.27 @ep14 (18e) | ⚠️ +0.50 | 【记录口径】 |
| 7 | N=4 hdim128 | N 与 hdim 同时改 | 34.05 @ep17 | ⚠️ Car strict 历史高 38.49 | 【记录口径】 |
| 8 | N=3 hdim128 | hdim 64→128 (N3) | 32.87 @ep17 | ❌ 全面劣化 | 【记录口径】 |
| — | N=4 hdim64 | hdim 128→64 (N4) | 33.00 @ep12 | ⚠️ | 【记录口径】 |
| — | N4 30e 无预训练 | 30e | 34.71 @ep22 | ✅ 对照 | 【记录口径】 |
| — | N4 30e lr2e-4 | lr 1.5→2e-4 | 30.08 | ❌ lr 过大 | 【记录口径】 |
| 9 | N4 30e + pretrained | load_from 官方权重 | **37.94 @ep19** | ✅ 最大单点增益 +3.23 | 【已验证】日志 ep19=37.9420 |
| 10 | head-v2 24e（原单次） | 检测头 v2 | 40.60 @ep11（未存）/39.65 已存 | ✅ | 【已验证】40.5950/日志；快照 spg=4×4卡=16、interval=2 |
| 11 | + Truck×3 anchor | anchors | 38.45 @ep11 | ❌ Car −5.01 | 【记录口径】ckpt 已清 |
| 12 | + Car-large anchor | anchors | 38.11 @ep14 | ❌ Truck −5.98 | 【记录口径】 |
| 13 | Doppler 动静 mask | +score 通道+velocity 门控 | 36.21 @ep8 | ❌ 全线回退 | 【已验证】BEST/日志；⚠️ 实为 3卡×spg4=batch12（记录未注明） |
| 14 | N4 BPTT | rssm_bptt_steps=1 | 37.84 @ep14 | ❌ 负收益 | 【记录口径】 |
| 15 | N2 BPTT | N4→N2/hdim128→64 | 38.08 @ep14 | ❌ 仍低于 Run10 | 【记录口径】 |
| §19 | Run10 三 seed 复现 | seed 1/2 + 3 卡 | **40.42±0.51**（s0 39.88/s1 40.51/s2 40.88） | ✅ 论文口径 | 【已验证】三 seed BEST 逐位一致 |
| §20 | No-Temporal | temporal_fusion=None | 35.59 @ep20 | ✅（作对照） | 【已验证】 |
| §21 | Deterministic latent | 删随机性 | 38.44 @ep13 | ⚠️ −1.31 vs RSSM | 【已验证】 |
| §22 | KL=0 seed0 | 关 KL 梯度 | 40.35 @ep22 | ⚠️ 单点更好 | 【已验证】 |
| §23 | FixedNoise | 固定 0.1 噪声 | 38.42 @ep16 | ❌ | 【已验证】 |
| §24 | Posterior learnable-std | 删 prior 保留 std | 39.66 @ep10 | ❌ 窗口 37.04 | 【已验证】 |
| §25 | KL0 seed1/2 | 多 seed | s1 41.10 / s2 36.76 | ❌ 不稳健→保留 KL | 【已验证】s1 ep20=41.0994 为全库日志最高 Overall |
| §26 | Truck residual refine | head 分支 | 40.26 @ep9（窗口未过） | ❌ Cyc −4.94 | 【已验证】窗口均值 37.27 |
| §27 | 同上 detach | detach | 37.99 @ep10 | ❌ 全项未达 | 【记录口径】ckpt 已清（best/last 保留 2 个） |
| §28 | Truck 独立 tower | 替换式 tower | 38.74 @ep16 | ❌ Cyc −5.65 | 【已验证】 |
| §29 | Ped 7D refine (detach) | head 分支 | 38.10 @ep12（窗口 35.60） | ❌ Ped strict 无提升 | 【已验证】 |
| §30 | Ped rotation 2→4 | rotations（**跨类共享**） | 39.52 @ep14（窗口 37.75） | ❌ | 【已验证】 |
| §31 | Ped CenterHead stage1 | 冻结全网+新头 | 39.85 @ep7（Ped strict 0.12） | ❌（隔离机制通过） | 【已验证】ckpt 12 个齐全；后三类逐位恒定 |
| §33/34 | 零训练尺寸先验 | 推理期覆盖/blend | Aα0.75@ep7: strict 2.41/loose 28.20 | ✅（零训练上界） | 【记录口径】（复评记录完整） |
| §35 | 有界尺寸残差 6e | 只训 dim 分支 | strict 0.044 | ❌ 无效 | 【记录口径】 |
| §36 | delta_xy 3e | IoU 主损失 | strict 回退 0.05 | 🛠️ **IoU 编码 bug → 整节作废** | 【记录口径】 |
| §37 | 修复 IoU 后重跑 | 同上 | strict 0.05 | ❌ 低分辨率路线关闭 | 【记录口径】+单测 test_centerhead_iou_decode |
| §38.1-5 | 高分辨率 0.16m BEV 3e | Ped 高分辨支线 | Ped AP≈0 | 🛠️ **scatter 转置布局 bug**（38.6 修复后结论作废） | 【记录口径】+单测 |
| §38.7 | 布局修复后 6e | 同上 | strict 0.097/prior 1.51 | ❌ | 【记录口径】 |
| §38.8/9 | 冻结几何/yaw probe | 只训小 probe 头 | w corr −0.36 / yaw 0.49-0.56 rad | ❌ | 【记录口径】+工具与 json |
| §38.10 | 残差死分支修复 3e | fusion_conv 去 BN/ReLU | prior strict 3.61@ep1 不稳 | 🛠️→❌ | 【记录口径】（权重非零证明表） |
| §38.11 | 6e 收敛验证 | 同上 | 无 checkpoint 同时达标 | ❌ 正式关闭 | 【记录口径】 |
| §39 | Ped 多 seed 补充 | stage1 ep7+Aα0.75 | strict +2.25±, loose −1.89, Overall −0.47 | ⚠️ 仅 strict 优先工作点 | 【记录口径】（seed1 ep15 恢复失败插曲已记录） |
| §40 | 预训练 Cross-Modal Fusion | RCFusion 类型 | 窗口 37.98（ep16 停） | ❌ | 【已验证】窗口均值/ckpt 16 个 |
| §41 | 共享残差 stem | head stem | 窗口 39.27（Cyc −4.68） | ❌（候选保留） | 【已验证】 |
| §42 | max_num 300→600 | 推理 cfg | \|Δ\|<0.02 | ❌（排除瓶颈） | 【记录口径】 |
| §43 | Cyclist oracle/PR 分解 | 只读 | oracle recall 仅 −0.013~−0.016 而 AP −5.16/−8.55 | 诊断 | 【记录口径】 |
| §44 | Cyclist cls 分支 seed0 | cyc_cls_branch | 40.635 @ep12 | ⚠️「方向成功」 | 【已验证】BEST=40.6350 |
| §44.7 | CycCls seed1 | 同 seed1 | 39.710 @ep15（ep23 中止） | ❌ 逐 seed 失败 | 【已验证】22 epoch；seed2 进行中 |

> **checkpoint 现状**【已验证】：按 §17 保留规则，绝大多数 run 只剩 best+last 两个 .pth（如 Run 10 原版只存 ep12=39.26，其 ep14=39.65 权重已丢失）；`cyccls` seed0 全 24 个、KL0 seed0 全 24 个、no_temporal 全 24 个、stage1/ped 系列完整保留。`run10_headv2_multiseed/seed_{0,1,2}` 三套完整 24e checkpoint 在盘（seed0 ep16 / seed1 ep15 / seed2 ep14 = released 候选）。Run 1-8 的 work_dirs 已删除。

---

## 8. 每次训练详解：结果与问题

> 完整逐 epoch 曲线见《训练记录》对应章节；此处给出汇报所需的因果叙事与已验证关键数字。

### 8.1 阶段一：RSSM 首落地 vs GRU（Run 1–5，18e，spg=4，lr 2e-4，Ped 1 anchor 时代）

- **Run 1** GRU baseline 34.50 —— 参考线。
- **Run 2** 无对齐 RSSM 33.54：kl_scale=0.1+free_nats=0 把 KL 压死。【记录口径】
- **Run 3** motion-align v1 30.89：同样的 KL 超参 + 零初始化输出，从头训 —— 全场最低。
- **Run 4** v2 34.01：kl_scale→1.0/free_nats→1.0/warmup→5，且续训自已训 18e 的 v1 —— +3.12 归因不干净（超参+续训混杂）。
- **Run 5** v3 33.77：同 v2 配置从头训 + logstd 上界 —— −0.24，中性。
- **诊断文档**（`work_dirs/rssm_diagnosis.md`，07-31，【已验证：其 stat_* 数据与日志字段名/机制代码一致】）：
  1. KL 死（clamped_ratio=1.0、mu_diff²≈0.04）是**稳态**而非病灶——推理用 μ_q，prior 不在路径上；
  2. **velocity/action 恒为零**（R4Det 调用从不传 velocity；TJ4D 无位姿）——motion-aware 只剩对齐层；
  3. **output_proj 零初始化是起步慢、18e 吃亏的真凶**（GRU 输出层是 Xavier）。
  → 落地两改动：`b64dd1f` Xavier + action_dim=0。此后所有 run 均在此状态上。

### 8.2 阶段二：帧数/容量消融（Run 6–8 + N4hdim64，18e）

- N2→N3：+0.50（Truck +3.83）；N3→N4（hdim128）：−0.22 但 **Car strict 38.49 全库纪录**；
- N3+hdim128：32.87（−1.40，容量过拟合）；N4+hdim64：33.00 → **容量-帧数匹配规律：N4 配 hdim128、N3 配 hdim64**；
- **Pedestrian 3D strict 全部 ≈0（0.01–2.5）**——首次明确「瓶颈在检测头/BEV 分辨率，不在时序融合」。
- 全部【记录口径】（目录已删）。

### 8.3 阶段三：预训练 + head-v2（Run 9 / Run 10 / §19）

- **Run 9**（N4 hdim128 30e + `pretrained_tj4d.pth`）：BEST ep19 37.94【已验证】，+3.23 vs 无预训练，Car strict +12.62——**预训练是全项目最大单点增益**；代价 Cyclist loose −4.85。30e 过长（ep19 后过拟合）。
- **Run 10**（head-v2 24e，唯一差异=检测头）：峰值 ep11 40.60【已验证 40.5950】但 interval=2 未存盘；已存 ep14=39.65；Car strict ep20 53.04。head-v2 带来 loose 口径全面提升（Car +8.71/Cyc +11.32/Trk +5.22），BEV_mod +5.53。**教训：interval=2 丢峰值 → 之后全部 interval=1**。
- **§19 三 seed 复现**（deterministic、3卡 batch12）：39.88 / 40.51 / 40.88，**40.42±0.51**【已验证逐位】；峰值稳定在 ep12-16；原 40.60 非 cherry-pick；seed2 ep14=40.88 为 released checkpoint 候选。逐类别方差集中在 Car loose/Cyc strict/Truck strict；Ped strict 全 seed 0.10–0.15 → 物理瓶颈再次确认。
- **论文报告口径**（§18/§19.9 定稿）：主表报三 seed mean±std；单点 BEST 仅附录。

### 8.4 阶段四：anchor/mask/BPTT 变体（Run 11–15，均 vs Run 10）

| Run | 变量 | 结果 | 教训 |
|---|---|---|---|
| 11 | Truck×3 anchor | Truck strict +1.80（35.03）但 Car −5.01、Overall −1.2~−2.2 | 共享头 max-IoU 分配是**零和跷跷板** |
| 12 | Car-large anchor | Car strict 53.77 纪录但 Truck −5.98、Overall −0.34 | 同上；**且按频率加权口径 Run12 排第 2**——结论依赖口径（§16） |
| 13 | Doppler 动静 mask（β=0.5 velocity-only） | 36.21，**最大单次回退 −3.44**，BEV −5.02 | velocity 失真破坏预训练 PFN/BEV 对齐；Ped loose +2.74 唯一亮点。⚠️ 审计发现：本轮 3卡×spg4=batch12（快照+log 已验证），与 Run10 原 16 batch 口径不同，记录未注明 |
| 14 | N4 BPTT1 | 37.84（−1.81） | 一步 BPTT 无增益、不破坏稳定性 |
| 15 | N2 BPTT1 | 38.08（−1.57） | BPTT 下轻量 N2 不输 N4；Ped strict 尾段首次非噪声异动（ep24=2.70，未复现跟进） |

### 8.5 阶段五：RSSM 机制消融（§20–25，全部 seed0 deterministic、3卡 batch12、24e；窗口=ep12-16 均值）

| 配置 | 窗口 Overall | BEST | 判定 |
|---|---:|---:|---|
| No-Temporal | 34.65 | 35.59 | 时序融合净收益 ≈ **+3.7 窗口 / +4.29 BEST**【已验证】 |
| Deterministic（删随机性） | 37.08 | 38.44 | 确定性传播贡献大部分但**不全部** |
| KL=0 | 39.06 | 40.35 | seed0 单点比完整 RSSM 还好 → 疑问 |
| FixedNoise 0.1 | 36.35 | 38.42 | 固定噪声不是答案（甚至 −0.73 vs Det） |
| Posterior learnable-std | 37.04 | 39.66@ep10 | std 退化到 min_std≈0.1，等效确定性 |
| **KL=0 三 seed** | 37.94 vs RSSM 39.00（Δ=−1.05；seed2 −4.03） | — | **保留完整 RSSM 定稿**（`8c9bc36`） |

> 结论链：确定性传播（h/z+对齐+GRU+重建+残差）解释约 2.4 点，随机建模（posterior sampling 与 prior/KL **共同**）解释约 1.3–1.9 点；单独删任何一个在多 seed 下都不可靠。RSSM 机制就此冻结（§25.6）。

### 8.6 阶段六：Truck 专项三轮（§26–28，seed0 deterministic，基线=clean seed0 窗口 38.39）

| 轮 | 结构 | Truck Δ | Car Δ | Cyc Δ | 判定 |
|---|---|---:|---:|---:|---|
| §26 residual refine | Conv3x3→1x1 残差加 (dx,dy,dl) | **+2.85** | +1.38 | **−4.94** | ❌ Cyc 红线 |
| §27 residual + detach | 同上，输入 detach | −2.79 | −1.38 | −5.11 | ❌ **否证「梯度回流干扰」假设**；Truck 增益恰依赖共享特征改写 |
| §28 独立 tower | 替换式直接预测 | +1.46 | +3.03 | **−5.65** | ❌ |

统一根因（§28.7）：**Truck 回归改善总是抢占 Cyclist 所需的共享 BEV 特征容量（零和）**；Truck 定位差的根因（中心 1.41m、长轴 3.02m，来自 §26 前置诊断）需 anchor 尺寸匹配或类分离解决，残差/tower 均非解。路线停止。

### 8.7 阶段七：Pedestrian 专项（§29–39）

**决定性诊断**（§29，dump `base_seed0_ep16.pkl`=clean seed0 ep16，Ped GT=999【已验证 GT 数】）：
- λ-recall（忽略分类，只问 6 个 Ped anchor 能否摆上 GT）：**IoU≥0.5 只有 4.5%** → strict 上限被 anchor 几何卡死；
- 正确召回样本的分量误差：**w(横向宽度) mean 0.307m / 45.5% 样本 >0.25m**（行人 w 才 0.4–0.7m）、yaw 双峰（33.5% 差 ~90°）、center/l/h 尚可；
- score 分布：91% 样本 max Ped score<0.5 → 分数低是症状不是病因。

**七轮尝试全部未通过**：
1. §29 Ped 7D refine（detach）：Ped strict 0.21 vs 基线 0.22；Cyc −8.54；
2. §30 rotations 2→4：**rotations 跨全部 6 个 size 共享**（实现层面 Ped 专属不可行），Cyc −4.56；Ped strict 仍 0.16；
3. §31 独立 CenterHead（stage1 全冻结隔离）：隔离机制**完美通过**（后三类逐位 0 差异，12 epoch 恒定【已验证】），但 Ped strict ep8-12 均值 0.12 → **0.32m BEV 特征本身不足**；
4. §32 诊断：CenterHead 把 center/l/yaw/h 误差大幅压低但 **w 只降 7.8%、92% 的 strict 失败样本 w 超限**；NMS sweep 排除 NMS；
5. §33/34 零训练先验：硬先验 strict 3.7–7.3 但 loose 掉；**软先验 A(0.655×0.628)×α0.75@ep7：strict 2.405 / loose 28.20 / Overall 39.77**——唯一通过的组合；
6. §35/36/37 训练残差（bounded、delta_xy）：无效；其中 **§36 有 IoU 输入编码 bug（8 维编码向量直接进 IoU 算子），整节作废**；§37 修复+单测后仍不通过；
7. §38 高分辨率 0.16m BEV 支线：经历两次工程事故——**38.6 scatter 布局转置 bug**（H/W 互换致 AP=0，修复后 identity 有召回）与 **38.10 残差死分支**（fusion_conv=Conv+BN+ReLU 且末层零初始化 → ReLU 死区零梯度，权重 6 个 epoch 全零【已验证：记录含权重非零证明表】）；修复后 3e/6e 门控 + 几何 probe（w correlation 为负）+ yaw probe（修正 π 周期后仍 0.49–0.56 rad）全部不通过 → **正式关闭**。

**最终 Ped 工作点**（§39 三 seed）：`clean seed ckpt → stage1 ep7（固定选点）+ 推理期 Aα0.75`：Ped strict **+2.25（均值 2.36）**，Ped loose −1.89，Overall −0.47；seed2 大退化（loose −5.66）。定性：**「Ped strict 优先的可选推理工作点」，不是稳定综合改进**。

### 8.8 阶段八：融合/结构门控与 Cyclist（§40–44，seed0 门控窗口 ep12-16）

| 实验 | 唯一变量 | 窗口结果 vs clean 38.39 | 判定 |
|---|---|---|---|
| §40 恢复预训练 Cross-Modal Fusion | `RCFusion.type: ConcatConv→Cross_Modal`（加载 `cross_attention.*` 权重，missing keys 证明【已验证记录+ckpt 键】） | Overall −0.41；Truck +3.27 但 Car −2.04/Ped −3.23 | ❌ |
| §41 共享残差 stem | `x = x + 0.1·Conv3x3-GN-ReLU(x)` | Overall +0.88、Car +3.15、Truck +3.44、Ped +1.62，**Cyc −4.68** | ❌（Cyc 红线）；stem 保留为候选 |
| §42 max_num 600 | 推理 cfg | \|Δ\|<0.02 | ❌（输出预算非瓶颈） |
| §43 oracle/PR 分解 | 只读 | oracle recall 仅 −0.013/−0.016，AP 却 −5.16/−8.55；高分段 TP 减少、中低分 FP 增多 → **Cyclist 分数排序/分离质量下降是主导** | 诊断 |
| §44 CycCls 分支（保留 stem） | Cyclist 12 个 logit 通道 + 3×3 残差分支（零初始化 identity，接线验证齐全） | **seed0**：Overall +1.18、BEV +0.94、Cyc 仅 −0.72（相对 stem −4.68 大幅修复）、Car +2.93 → 「方向成功」；**seed1**：四项 fail（Ped −3.78/Truck −3.03）→ 逐 seed 复现失败；唯一跨 seed 稳健信号=**Car strict +2.9** | ⚠️ 不能定为可替换主模型；seed2 进行中（09-19 启动） |

§44 还给出重要方法论修正：预注册门控线 47.46 是用**错误的 clean BEV 基线（46.96，实为 KL0 的数字）**写死的；核对后 clean 真值 46.52（`af227ae` 修正），但**坚持不改预注册线、不事后宣称通过**【已验证：clean seed0 ep12-16 BEV=46.5193】。另：CycCls seed0 峰值 40.635 落在平台噪声带内（平台 Overall 逐 epoch σ≈1.5），不能称「刷新历史」。

---

## 9. 失败原因分类与证据链

### 9.1 网络/方法层面（真负结果）
1. **共享 BEV 特征零和**：Truck/Ped/Cyclist 任何「专属回归分支」都压 Cyclist（§26/27/28/29/30/41 六次同模式，Cyc −4.5~−8.5）。
2. **Pedestrian strict = 物理/几何瓶颈**：λ-recall@0.5=4.5%（anchor 几何）+ w 误差（0.32m BEV 下 Ped 宽≈2 体素，换头也压不下 w）→ CenterHead/高分辨率/probe 七轮全败。
3. **随机建模组件不可拆**：KL0/Deterministic/FixedNoise/LearnableStd 单删任一都不可靠（多 seed 方差）。
4. **velocity/action 无信号**：数据集无位姿，action 恒零，移除（中性）。
5. **BPTT1 无增益**（37.84/38.08 vs 39.65）。
6. **Doppler velocity 失真 × 预训练权重不兼容**（Run 13 全线回退）。

### 9.2 工程/配置问题（不是网络问题——均已被识别并修复或绕开）
1. `checkpoint_interval=2` 丢失 Run 10 峰值 ep11 → 全部改 interval=1；
2. §36 IoU 输入编码 bug → 该节作废重跑；
3. §38.6 高分辨率 scatter H/W 转置 → Ped AP 假 0；
4. §38.10 零初始化+ReLU 死区 → 残差分支 6 epoch 零梯度；
5. `reset_for_samples` in-place 破坏 BPTT autograd（`d4406b2` 修复）；
6. FocalLoss per-class alpha 不被 CUDA kernel 支持 → 回退 0.25（`709ac52`）；
7. TJ4D difficulty：三档全不达标对象曾被记 hard → 改 -1 忽略（`7374435`，影响早期评估口径）；
8. 评估脚本曾把多次重启日志叠加读出假 BEST（ep15=35.18 误读）→ 「每 epoch 取最后一次 val」规则（附录两文档均有记载）；
9. /data 磁盘写满致 ep5 checkpoint 损坏（§28，从 ep4 恢复重跑）；会话回收误杀训练一次（§26，从 ep2 续训）；
10. §39 seed1 "ep15 best" 文件不存在 → 固定改用已存在的 ep14。

### 9.3 口径/方法论问题（影响结论解读）
1. **Overall 混合口径**（Car/Trk strict + Ped/Cyc loose）导致 anchor 类实验的排名会随权重翻转（§16：等权 vs 频率加权排名不同）；
2. **单点 BEST vs 平台均值**：Run 10 的 39.65/40.60 是平台上沿；平台 ep12-24=38.56±0.97；后续一律用窗口均值对比；
3. 平台期噪声量级：Overall 逐 epoch σ≈1.5、Car σ≈3-4 → 小于 1.5 的「增益」大多不显著（§44.7.4）；
4. 预注册门控线写死基于错误基线时，宁可记「未通过」也不事后改线（§44.4）。

---

## 10. Baseline / 原时序模块 / RSSM 差异总表

| 维度 | 原始 R4Det（单帧） | TemporalDeformableFusion（Run 1） | BEVRSSMTemporalFusion（Run 2） | MotionAlignedRSSMFusion（现主线） |
|---|---|---|---|---|
| 历史信息 | 无 | 上一帧**真实 BEV 特征** | 自维护 h/z 状态 | 自维护 h/z + **deform 对齐到当前帧** |
| 确定性状态 h | 无 | 无（一次性门控融合） | ConvGRU(h,z,action) | ConvGRU(align(z), align(h)) |
| 随机状态 z / prior / posterior / KL | 无 | 无 | 有（KL 被压死） | 有（kl_scale warmup→1.0, free_nats=1.0） |
| action(velocity) | 无 | 无 | 通道存在但恒零 | 已整体移除（action_dim=0） |
| 输出 | — | `output_layer(h)`（Xavier） | `output_proj(z)+feat`（当年零初始化） | 同左，**Xavier**（`b64dd1f` 起） |
| 辅助损失 | — | 无 | KL+recon | KL+recon（+stat 监控） |
| 推理 | 单帧 | prev 帧真实特征 | posterior 均值 | posterior 均值（prior 不参与） |
| 18e BEST | —（对照 34.50） | **34.50** | 33.54 | 33.77（v3，同 KL 配置） |
| 24e+pretrain+head-v2 | — | 未试 | 未试 | **40.42±0.51（三 seed）** vs No-Temporal 35.59 |

> 注：GRU baseline 与 RSSM 的最终对比只在 18e/ped-1-anchor 时代直接做过（34.50 vs 33.77）；之后的所有增益叠加在 RSSM 线上，**没有在最终配置下重跑 GRU 对照**——严格说「RSSM 优于 GRU」在最终配置下未被直接检验，仅有时序融合本身 +3.7 点的净贡献（vs No-Temporal）是干净的。【诚实标记】

---

## 11. 当前可信结论

**A 级（多源验证，可直接汇报）**
1. 主线模型 = pretrained backbone + N=4/hdim128 Motion-Aligned RSSM（kl1.0/free1.0/warmup10 + BPTT1）+ head-v2，24e、batch12、lr 1.5e-4 AdamW + Cosine。**Overall 3D moderate 40.42±0.51（三 seed deterministic）**，最佳单点 seed2 ep14 = 40.88。【已验证】
2. 时序融合净贡献 ≈ **+3.7~+4.3 点**（vs 同配置 No-Temporal 34.65/35.59）【已验证】；其中确定性传播约占 2.4 点，随机建模（sampling+KL 共同）约占 1.3–1.9 点（§21–25）【已验证】。
3. 预训练 backbone 贡献 **+3.23**（Car strict +12.62）【已验证 Run 9 日志】。
4. head-v2（Ped 3 anchor + ignore_dir + IoU 分支）贡献 ≈ +1.7~+2.7 Overall / +5.5 BEV【已验证 Run 10 日志】。
5. RSSM 动力学：posterior collapse 到 free-nats 平台是普遍稳态；推理只走 posterior 均值；**KL/prior 仍不可删**（三 seed 证据）【已验证代码+日志+多 seed】。
6. Pedestrian 3D strict ≈ 0 是 **BEV 0.32m 分辨率 + 雷达点稀疏下的 anchor 几何上限**（λ-recall@0.5=4.5%）+ 宽度回归上限（换头后 w 仍 ~0.28m）共同决定；不是 NMS/score/anchor 数问题。
7. 共享检测头/共享 BEV 下，任何类别专属回归改动都以牺牲 Cyclist 为代价（六次复现同模式）。

**B 级（单 seed 或单口径证据，汇报需注明）**
8. 频率加权口径下排名会变（Run 12 升第 2）；优先级取决于部署目标。
9. CycCls 分支 seed0「方向成功」（+1.18）未通过 seed1 复现；唯一跨 seed 稳健的是 Car strict +2.9。
10. N=4+hdim128 vs N2+hdim64 的容量-帧数匹配规律（18e 时代）。
11. Ped strict 优先工作点（stage1 ep7 + Aα0.75）：strict 2.36 均值、Overall −0.47（seed2 退化明显）。

**C 级（历史记录，已无法复核）**
12. Run 1–8 的全部数字与部分超参（日志/目录已删；Run 2 超参还与现存配置文件不符）。

---

## 12. 尚未解决的问题与未知项

### 12.1 未解决的性能问题
1. **Pedestrian strict ≈ 0**：全部架构手段试尽（见 §8.7），未解。若必须解决，剩余路线：point/center-based 专属头+真正细分辨率输入（成本高）、或数据侧（LiDAR 蒸馏/更密点云）。
2. **Car–Truck 混淆与 Truck 定位**（长轴 3.02m）：三轮回归支线失败；类分离 tower/检测头在 §28.8 候选清单里，未试。
3. **Cyclist loose 长期平台 ~48–50**：机制已定位到「分数排序/分离质量」而非召回（§43），修复方案未落地。
4. **RSSM 随机性的价值兑现**：posterior collapse 下 z 实际近确定性；「RSSM 理论优势」未被真正利用（多步预测/prior 推理等方向未探索）。

### 12.2 未知 / 待验证清单（本次审计标记）
| # | 事项 | 状态 |
|---|---|---|
| U1 | Run 1–8 的逐 epoch 数字与 Run 2 真实 KL 超参 | 【未知】日志已删，只能引用《训练记录》 |
| U2 | 部分早期 run 的精确 GPU 数/batch 口径差异 | Run 13 已查明（3×4=12）；Run 9/Run 10 原版已查明（4 卡，spg4=16）；其余未逐一核对 |
| U3 | 主线最终配置下 GRU baseline 的成绩 | 【未知】从未跑过（见 §10 注） |
| U4 | KL0 seed1 ep20=41.10（全库最高）的稳健性 | 只此一个 seed 单点，且 KL0 已被多 seed 否决 |
| U5 | `code_weights=[2,2,1,…]` 是否曾在早期版本接入 anchor head | 已查明【否】：全部 configs 中仅作为顶层死变量存在，从未写入 `train_cfg.pts.code_weight`（多源 grep + git -S 检索） |
| U6 | 预训练 ckpt 内 `temporal_fusion.*`（GRU 权重）是否有任何 run 曾成功加载到 RSSM | 判定【否】（结构不匹配必走 missing keys → 随机初始化），未逐 run 核对日志 |
| U7 | 当前仓库主配置含 `shared_stem=True` 的意图 | 疑为 §41 实验后未回退；**复跑 clean 前必须手动改回** |
| U8 | CycCls seed2 结果 | 进行中（09-19 启动），本报告不含其结论 |

### 12.3 仓库状态卫生（建议汇报前处理）
- 主配置 `shared_stem=True` 未回退（见 U7）；
- ~~`BEVRSSMTemporalFusion.forward` 缺 return~~（2026-09-22 已修复，见 §3.3 第 2 条）；
- `baseline_rssm` 配置内容与其历史训练超参不符。

---

## 13. 下一步实验建议

按《训练记录》§44.7.6 与本次审计综合（优先级从高到低）：

1. **等 CycCls seed2 完成**，按既定口径补全三 seed 配对均值与逐 seed 双结论，正式给 CycCls 分支定性（预期：记「仅 Car strict 跨 seed 正向、Overall 未被支持」并停止）。
2. **冻结主线**：把主配置回退成 clean（shared_stem=False）并打 tag/存档快照，避免复现事故；~~修复 `BEVRSSMTemporalFusion` return~~（2026-09-22 已完成）。
3. **论文口径补强**：clean full RSSM 40.42±0.51 为主表；KL0 多 seed 作机制消融附录；No-Temporal/Deterministic 作时序收益分解；频率加权口径作部署讨论。
4. **若继续刷 Overall**：唯一有跨 seed 证据的方向是 **Car**（CycCls 的 Car strict +2.9）；可试「只保留 CycCls 分支收益、去掉 stem」或对 Car 的 anchor/分配器做单变量（注意 Cyclist 红线）。
5. **若解决 Truck**：拆分类专属 prediction tower（§28.8 候选 2）或先做 Car–Truck 混淆专项诊断（分类 vs 定位）——但注意 §26.6 诊断已显示 Truck 首要是定位（长轴/中心）而非分类。
6. **若解决 Ped strict**：跳出 anchor 范式（point/center-based 专属头 + 输入侧稠密化），或在论文中如实报告该物理瓶颈并采用 §39 的 strict 优先工作点作消融。
7. **方法论**：延续「预注册门控 + 窗口均值 + 多 seed」纪律；任何 <1.5 Overall 的单 seed 提升先视为噪声。

---

## 附录 A：一致性核查记录（本次审计执行的独立验证）

| 验证项 | 方法 | 结果 |
|---|---|---|
| 20 个 run 的 BEST/窗口均值/epoch 数 | python 流式解析全部 `*.log.json`（每 epoch 取最后一次 val） | **全部与《训练记录》逐位一致**（含 run10 三 seed、KL0×3、No-Temporal、Det、FixedNoise、LearnStd、CycCls、SharedStem、CrossModal、TruckTower、PedRefine、PedRot4、Run9、Run10 原版） |
| checkpoint 盘点 | 遍历 work_dirs 统计 epoch_*.pth | 与 §17 保留规则一致；cyccls seed1=22 个（中止未清）；Run10 原版仅剩 ep12/ep24（ep14 已失） |
| Overall 聚合公式 | 读 eval.py:925-937 | 与记录口径完全一致 |
| 类别分布/val GT 数 | 解析 infos pkl | Car 48.4%/Cyc 21.6%/Trk 16.5%/Ped 13.5%；val Ped 999、Cyc 2153——与记录全部吻合 |
| 预训练 ckpt 内容 | torch.load 枚举键 | 1421 张量；含 cross_attention（Cross_Modal_Fusion）与 GRU 版 temporal_fusion 权重 |
| `output_proj` 零初始化→Xavier | git show 历史版本 | `b64dd1f`（08-05）确认，与记录「v1/v2/v3 零初始化、之后 Xavier」吻合 |
| `BEVRSSMTemporalFusion` return 缺失 | git log -L / git show | `99c74ff`（08-25）确认误删 |
| 主配置 shared_stem 状态 | 读配置 + git | `b4fd7be` 加入后未回退 |
| Run 13 batch 口径 | work_dir 配置快照 + log 头 | 3 GPU × spg4 = 12（记录未注明，已补） |
| 记录文档一致性 | md5 | work_dirs 副本与 docs/ 正本逐字节相同 |
| pretrained_tj4d.pth 存在性 | ls | 1,091,397,096 字节 |
| 数据集帧数 | pkl 计数 | 5706/2040 |

---

## 附录 B：主线配置关键快照

`configs/r4det/TJ4D-R4Det_motion_align_rssm_det3d_N4_2x4_24e_pretrained_v2_head.py`（**注意含 shared_stem=True，见 §3.3-1**）：

```python
model = dict(type='R4Det', seq_len=4, rssm_bptt_steps=1,
  img_backbone=ResNet50(frozen_stages=1), img_neck=FPN(256),
  depth_net=GeometryDepth_Net(use_radar_depth=False),   # 深度监督关闭
  img_view_transformer=ViewTransformerLSS(downsample=8),  # dbound=[1,73,1]
  pts_voxel_layer=dict(voxel_size=[0.16,0.16,6.0], max_num_points=10),
  pts_voxel_encoder=RadarPillarFeatureNet(in_channels=5, with_velocity_snr_center=True),
  pts_middle_encoder=PointPillarsScatter(64, output_shape=[496,432]),
  pts_backbone=SECOND([64,128,256]), pts_neck=SECONDFPN(→384),
  RCFusion=ConcatConvFusion(256+384→256),
  temporal_fusion=MotionAlignedRSSMFusion(in/out/latent=256, hidden=128,
      action_dim=0, kl_scale=1.0, free_nats=1.0, min_std=0.1, init_std=0.2,
      align_kernel_size=3, align_deform_groups=1, align_z_state=True),
  pts_bbox_head=Anchor3DHead(12 anchors, use_iou_branch=True,
      ignore_dir_classes=[0], anchor_class_mapping=[0,0,0,1,2,3],
      shared_stem=True /*←复现 clean 需改 False*/,
      loss_cls=FocalLoss(0.25,2.0,1.0), loss_bbox=SmoothL1(1/9,2.0),
      loss_dir=CE(0.2)))
train_cfg: MaxIoUAssigner per-size（Ped/Cyc 0.35/0.2, Car/Trk 0.5/0.35）
test_cfg: rotate NMS, nms_pre=1000, nms_thr=0.5, score_thr=0.0, max_num=300
optimizer=AdamW(lr=1.5e-4, betas=(0.95,0.99), wd=0.01), grad_clip=35,
  GradientCumulativeOptimizerHook(cumulative_iters=2), CosineAnnealing(min_lr_ratio=1e-5)
data: RepeatDataset(×2), samples_per_gpu=2, workers=2, val=每 epoch
custom_hooks=[KLScaleSchedulerHook(0→1.0, ep0-10)]
checkpoint_config=dict(interval=1); load_from='checkpoints/pretrained_tj4d.pth'
```

派生配置一览（均已在 git 中）：
- `*_head_kl0.py`（kl_scale=0.0）、`*_deterministic_latent_*.py`、`*_bptt_fixednoise.py`、`*_bptt_learnable_std.py`、`*_bptt_learnable_std_bs4_noaccum.py`
- `*_head_truck.py / _truck_car2.py / _truck_detach.py / _truck_tower.py / _head_confuse.py / _dynmask.py`
- `*_head_pedrefine.py / _pedrot4.py`、`ped_centerhead_stage1_*(seed1/2).py`、`ped_centerhead_stage2_dim/delta*`、`ped_highres_*`
- 早期：`TJ4D-R4Det_baseline_temporal/rssm_det3d_2x4_12e.py`、`motion_align_rssm_det3d_2x4_12e.py`（=v3 后状态）、`N3_2x3 / N3_hdim128 / N4_hdim64 / N4_2x4_12e / N4_2x4_30e(_pretrained / _lr2e4)`、`pretrain_2x4_12e.py`（官方预训练训练配置：Cross_Modal_Fusion、seq_len=2 单帧、lr 4e-4）

---

## 附录 C：复现入口与工具

```bash
# 环境
source .envrc   # conda activate r4det; CUDA 11.8; CUDA_VISIBLE_DEVICES=5,6,7

# 训练（主线三 seed 口径）
bash tools/dist_train.sh configs/r4det/TJ4D-R4Det_motion_align_rssm_det3d_N4_2x4_24e_pretrained_v2_head.py \
  3 --seed 0 --deterministic --work-dir work_dirs/<run_name>

# 评估/复评（支持 --cfg-options 传推理期先验等）
python tools/test_vod.py --config <ckpt目录配置快照> --checkpoint <ckpt> --eval bbox \
  --out <dump.pkl> [--cfg-options model.ped_center_head.test_cfg.size_prior_alpha='[0.655454,0.627535,0.75]']

# 汇总工具
python3 tools/summarize_run.py <work_dir> --tail 5 [--weighted]
bash tools/run_multiseed.sh ...        # 三 seed 编排（commit 1fd3868/047f356，含 --deterministic 与 cfg-options 透传）
# 诊断工具族：dump_predictions / diagnose_errors / ped_error_breakdown / ped_center_diag /
#   ped_nms_sweep / cyc_drop_analysis / cyc_drop_diagnosis / cyc_error_decomposition /
#   class_grad_cosine / ped_highres_geometry_probe
# 单测：tests/test_centerhead_iou_decode.py、tests/test_ped_highres_branch.py、
#   tests/test_ped_highres_geometry_probe.py
```

---

## 附录 D：证据文件索引

| 内容 | 路径 |
|---|---|
| 主实验记录（44 节） | `docs/training_runs_full.md`（254,942 B；work_dirs 副本 md5 相同） |
| 早期 RSSM 诊断 | `work_dirs/rssm_diagnosis.md`（07-31，stat_* 数据） |
| 早期五 run 对比 | `work_dirs/training_comparison.md` |
| RSSM 模块 | `mmdet3d/models/fusion_layers/rssm_fusion.py`（4 个变体类） |
| GRU 时序模块 | `mmdet3d/models/fusion_layers/temporal_r4det_fusion.py` |
| 融合模块 | `fusion_layers/concat_conv_fusion.py`；`necks/BEVCross_modal_attention.py`（Cross_Modal_Fusion） |
| 检测器 | `mmdet3d/models/detectors/R4Det.py`（BPTT/冻结/双头合并） |
| 检测头 | `mmdet3d/models/dense_heads/anchor3d_head.py`（v2 + 6 个可选分支）；`dense_heads/centerpoint_head.py`（CenterHeadkitti + 先验/delta） |
| KL hook | `mmdet3d/core/hook/kl_scale_scheduler.py` |
| Ped 高分辨支线 | `mmdet3d/models/fusion_layers/ped_highres_branch.py` |
| 数据集/时序 | `mmdet3d/datasets/TJ4D_dataset.py`（seq_len 连续帧逻辑） |
| 动静 mask | `mmdet3d/datasets/pipelines/loading.py:661`（RadarStaticDynamicScore） |
| 评估 | `mmdet3d/core/evaluation/kitti_utils/eval.py`（Overall 聚合 :925-937） |
| 全部配置 | `configs/r4det/*.py`；各 run 实际训练快照在 `work_dirs/<run>/*.py` |
| 预训练权重 | `checkpoints/pretrained_tj4d.pth`（1.09 GB，1421 张量） |
| 关键 commit | `256f0c8`(RSSM v1) → `2168bed`(零初始化) → `57da70f`(motion-align) → `c10307f`(KL 超参) → `f39dedd`(logstd) → `b64dd1f`(Xavier+action0) → `094d663`(BPTT) → `d4406b2`(autograd fix) → `d24e1e4`(head-v2) → `7374435`(difficulty fix) → `7643a5b`+`99c74ff`(deterministic 变体+误删 return) → `713dc7a`(learnable-std) → `f36c90c`/`1c95fc3`/`f6a4417`(Truck 三连) → `5a2904f`(ped refine) → `ed70d9d`(CenterHead) → `fa05ea8`/`9d74b8f`(先验) → `b2e4d7e`(highres) → `19b7f81`(布局修复) → `91f3b4c`(残差死分支修复) → `2cd85fc`(cross-modal) → `b4fd7be`(shared stem) → `69e2072`(cyccls) → `a76d194`(HEAD) |

---

*本报告由代码、git 历史、日志、checkpoint 的独立考古生成；所有【已验证】条目均可按附录 D 复查。未标注【已验证】的数字请按【记录口径】对待，引用时注明来源。*
