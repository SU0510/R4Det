# 01 · 架构图谱与信息流（Architecture Map）

> 写作日期：2026-09-19。本文档所有 file:line 均由本次研究在仓库 `goal-test` 分支（HEAD `213cd89`）逐行核验；
> 训练历史结论引自 `docs/R4Det_RSSM_full_report.md`（下称《审计报告》）并标注证据等级。
> 本文是 `me_rssm/` 研究包的公共事实基础：02（问题清单）、03（候选筛选）、04（ME-RSSM 设计）均引用本文编号。

---

## 1. 任务与总体拓扑

单目相机 + 4D 毫米波雷达（TJ4DRadSet）BEV 3D 检测，四类（Pedestrian/Cyclist/Car/Truck），
时序窗口 N=4（当前帧 + 3 历史帧，old→new）。

```
                    ┌─────────────── Camera 分支（semantic/appearance）───────┐
img (B,3,480,640) → ResNet50+FPN(256ch) → GeometryDepth_Net(72 depth bins)
                  → ViewTransformerLSS(downsample=8) → 外积 splat → z 维 mean
                  → camera BEV (B,256,248,216)   ①行=y(248), 列=x(216)
                    └────────────────────────────────────────────────────┘
                    ┌─────────────── Radar 分支（geometry + Doppler）────────┐
pts (N,5)=x,y,z,v_r,SNR → 动态体素化 0.16m pillar(≤10点) → RadarPillarFeatureNet
  (5→12维装饰→64ch, max-pool) → PointPillarsScatter → (B,64,496,432) ②行=y(496),列=x(432)
  → SECOND(64/128/256) → SECONDFPN → radar BEV (B,384,248,216)
                    └────────────────────────────────────────────────────┘
两路 BEV → ConcatConvFusion: cat(256+384)→3×3 conv→(B,256,248,216)
        → permute → (B,256,216,248)   ③画布转置：行=x(216), 列=y(248)
        → MotionAlignedRSSMFusion（时序，自维护 h/z 状态）
        → Anchor3DHead v2（12 anchor × 4 类 + IoU 分支）
```

①②③ 的坐标约定是本项目最容易出错的地方（历史事故《审计报告》§38.6 scatter 转置 bug 即源于此）：
- **scatter 画布**：`PointPillarsScatter` 以 `output_shape=[bev_w_*2, bev_h_*2]=[496,432]` 构造
  （pillar_scatter.py:22-27：`self.ny=output_shape[0]=496, self.nx=output_shape[1]=432`），
  scatter 索引 `coors[:,2]*nx + coors[:,3]`（pillar_scatter.py:110，coors=(b,z,y,x)），
  输出 `(B,C,ny,nx)=(B,C,496,432)`——**行是 y（横向），列是 x（纵向）**，0.16m 分辨率。
- **融合后画布**：`extract_feat` 在融合后执行 `bev_feats.permute(0,1,3,2)`（R4Det.py:863），
  变为 **行=x(216)、列=y(248)**。temporal fusion 与检测头全部工作在这一约定上。
- 因此任何送入 temporal fusion 的辅助空间图（如速度场）必须 `avg_pool(2)`（0.16→0.32m）
  **再 permute(0,1,3,2)**。deform conv 的 offset 通道序为 (Δx_w, Δy_h)，即 W 方向对应 y、H 方向对应 x。

## 2. 数据侧（mmdet3d/datasets/TJ4D_dataset.py）

| 事实 | 证据 |
|---|---|
| `__getitem__` 返回 seq_len 帧序列（old→new，最后一帧=当前帧）；按 `image_idx` 连续性判定有效性，断档帧及更早帧全部标记 invalid 并用空占位帧替代 | TJ4D_dataset.py:260,309（`is_prev_frame_valid`） |
| 雷达点 `LoadPointsFromFile(load_dim=8, use_dim=[0,1,2,3,5])`：x,y,z,**径向速度 v(dim3)**,**SNR(dim4)**，共 5 通道 | 主配置 train_pipeline:347 |
| 数据集无 ego 位姿、无 timestamp、无速度标注 → RSSM 的 action/ego-velocity 分支无真实信号（《审计报告》§1.2【已验证】） | — |
| 训练期每帧独立增广（同一 seed 重置）；当前帧 GT 参与检测 loss | 《审计报告》§4.1 |

**信息特性结论**：相机提供稠密语义/外观，但深度不确定；雷达提供稀疏几何 + **唯一 DIRECT 的运动测量
（径向速度）** + SNR 质量信号 + 占有点数（密度=可靠性代理）。EGO 运动在传感器系中表现为静态物体的
径向流（v_r ≈ −v_ego·cos(az)），因此 Doppler 场同时编码 ego 补偿与目标相对运动。

## 3. 模型侧逐层 tensor 流（R4Det.py 逐行核验）

### 3.1 单帧前向（`extract_feat`，R4Det.py:761-955）

| 步骤 | 代码位置 | 输入→输出 |
|---|---|---|
| 体素化 | `voxelize` R4Det.py:1632-1660 | pts list → voxels(P,10,5), num_points(P,), coors(P,4) |
| 雷达 pillar 编码 | `extract_pts_feat` R4Det.py:720-733 | `pts_voxel_encoder(voxels,num_points,coors)` → (P,64)；`pts_middle_encoder` → (B,64,496,432)；SECOND+FPN → (B,384,248,216) |
| 相机 BEV | R4Det.py:785-801 | ResNet50+FPN → depth_net(context+depth 72bins) → LSS 外积 splat → (B,256,8?,216,248)→z mean → (B,256,248,216) |
| 融合 | R4Det.py:861-865 | `self.cross_attention(img_bev, pts_bev)`（属性名固定为 `cross_attention`，R4Det.py:291）→ (B,256,248,216) → permute → (B,256,216,248) |
| 时序融合 | R4Det.py:875-901 | `self.temporal_fusion(bev_feats, use_posterior, deterministic, detach_state)` → 6 元组 `(output, recon, kl*scale, h_t, z_t, stats)`；当前帧 `output = output_proj(z_t)+feat`；无效历史样本回退 `torch.where(mask, output, feat_cache)` |
| 检测头 | forward_pts_train R4Det.py:1531-1571 | 只吃 `pts_feats[0]`（= 时序融合输出），Anchor3DHead v2 |

### 3.2 时序融合：`MotionAlignedRSSMFusion`（rssm_fusion.py:501-725，逐行核验）

状态：`h_state, z_state ∈ (B,256,216,248)`（hidden=latent=256）；序列开始 `reset_state()` → **零初始化**
（rssm_fusion.py:666-670）；批内样本级重置 `reset_for_samples(mask)`（:299-312，非 in-place）。

单帧计算（:644-725）：

```
h_a  = DeformAlign(h_{t-1}; offset=conv(cat[feat, h_{t-1}]))      # :673-675, offset/mask conv 零初始化=identity 起步
z_a  = DeformAlign(z_{t-1}; 同上, 独立一组 conv)                    # :677-679 (align_z_state=True)
h_t  = ConvGRU(z_a, h_a)                                           # :688, ConvGRUCell: r,u,h̃ 三个 3×3 conv (:44-61)
μ_p, σ_p = prior(h_t)                                              # :691-692, 两个 3×3 conv
e_t  = encoder(feat)                                               # :695, 256→128→256 ConvModule
μ_q, σ_q = posterior(cat[h_t, e_t])                                # :698-699
z_t  = sample(μ_q,σ_q)（训练/当前帧）或 μ_q（burn-in/推理）           # :702-705
KL   = free-bits clamp( KL(q‖p), min=free_nats ) × kl_scale        # :708, kl_loss :327-370（softplus 下界 σ≥0.1, clamp σ≤1）
recon = decoder(cat[h_t, z_t])                                     # :711, 512→128→256
output = output_proj(z_t) + feat                                   # :714-715, 3×3 conv Xavier（b64dbbf 起非零初始化）
状态存储 detach_state=True → detach 后存；False → 保留计算图            # :718-723
```

### 3.3 训练循环的时序编排（`forward_train`，R4Det.py:1254-1390）

```
reset_state()（每个训练 step, :1314-1315）
burn-in: t=0..N-2-bptt 帧   no_grad + detach_state=True           # :1325-1332
BPTT 窗: t=burn_in..N-2 帧   带图 + detach_state=False             # :1334-1346（主线 bptt_steps=1 → 仅最后 1 个历史帧带图）
invalid 历史帧样本: reset_for_samples(~valid_t)                    # :1331-1332,1345-1346
当前帧 t=N-1: 全梯度; is_valid_mask=最后历史帧有效性                # :1353-1356
loss_rssm_recon = mean(窗口内 recon + 当前帧 recon)                 # :1376-1380
loss_rssm_kl    = 同窗口平均 × warmup hook 调度                    # :1381-1386（KLScaleSchedulerHook 直接改 tf.kl_scale）
```

推理（`simple_test`，R4Det.py:1070-1251）：`reset_state()` → 历史 3 帧 no_grad 前向 → 当前帧 z_t=μ_q →
NMS（nms_pre=1000, nms_thr=0.5, max_num=300）。**prior 在推理路径完全不参与**（《审计报告》§4.4【已验证】）。

### 3.4 检测头 v2（anchor3d_head.py；《审计报告》§3.2【已验证】）

12 anchor（Ped×3/Cyc/Car/Truck 尺寸）× 2 朝向；`conv_cls(48)/conv_reg(84)/conv_dir(24)/conv_iou(12)`
全部 1×1；`ignore_dir_classes=[0]`（Ped 无方向分类）；IoU 分支训练期为中心度代理 L1、推理不参与打分。
可选分支（confuse/truck_refine/tower/ped_refine/shared_stem/cyc_cls）全部由 config 开关控制；
**当前主配置仍带 `shared_stem=True`（审计 U7：§41 门控失败后未回退）**。

## 4. 训练配置要点（主配置 `TJ4D-R4Det_motion_align_rssm_det3d_N4_2x4_24e_pretrained_v2_head.py`）

- kl_scale=1.0 + free_nats=1.0 + KLScaleSchedulerHook(0→1.0, ep0-10)；hidden=128；action_dim=0（velocity 分支已移除，:212 注释明确"TJ4D has no ego pose"）。
- AdamW lr 1.5e-4 (0.95,0.99) wd 0.01 + 梯度累积 2 + Cosine；24e；load_from=官方预训练 `pretrained_tj4d.pth`（1.09GB：img/depth/radar 编码器 + Cross_Modal_Fusion 的 cross_attention 权重 + **GRU 版 temporal_fusion 权重**，后者与 RSSM 形状不匹配→RSSM 历来随机初始化）。
- **入口事实**：`tools/dist_train.sh` → `train_vod.py`；`train_vod.py:174` 注入 `meta_info`；`:157-159` 处理 `custom_imports`；`PYTHONPATH=repo root`（dist_train.sh）→ repo 根下的新包可被 import。

## 5. 信息流分析（研究视角的增量结论）

1. **唯一的信息瓶颈汇聚点**：相机/雷达两路 256/384ch 特征在 ConcatConvFusion 处被压成 256ch——
   之后的一切（RSSM 观测编码、KL、recon、检测头）只见混合体。模态身份、Doppler 独有信息、
   各自的可靠性信号在此不可逆丢失。
2. **Doppler 的现有路径**：v_r/SNR → PFN 装饰（pillar_encoder.py:598-603 velocity_snr_center）→
   线性+BN+ReLU+max 混入 64ch → SECOND 两级 stride 混合 → 融合。**在到达 RSSM 之前，
   "运动"已被当作"外观"处理了两次**。RSSM 的 ConvGRU 转移没有任何运动控制信号；
   运动适应完全依赖 deform align（外观驱动的 offset 预测，rssm_fusion.py:623-641）。
3. **先验的语义真空**：KL 自由比特平台（clamped_ratio→99-100%，μ_diff²≈0.008-0.04，
   《审计报告》§5.3【已验证】）说明 p(z|h)≈q(z|h,e)。转移模型不具备运动预测能力时，
   先验只能是后验的模糊镜像——"概率时序建模"的收益（多 seed 证据 +1.3~1.9，《审计报告》§8.5）
   来自正则化/噪声注入，而非"预测-校正"结构本身。
4. **recon 目标=融合外观 BEV**：decoder(cat[h,z]) 重建的是静态占比极高的 fused BEV——
   最容易的解是让 h 记住背景。h 的 256ch 容量没有被引导去编码"会变的东西"。
5. **观测可靠性无表征**：SNR、占有点数、深度不确定性都进了网络，但没有任何显式的
   confidence/reliability 通道参与融合、更新或输出权重决策。

## 6. 与 me_rssm 设计的接口约束汇总（本次逐条验证）

| 约束 | 事实 | 来源 |
|---|---|---|
| temporal_fusion 调用签名 | `(feat, use_posterior=, deterministic=, detach_state=)` → 6 元组 | R4Det.py:881-896 |
| 必须实现的成员 | `reset_state()` / `reset_for_samples(mask)` / 属性 `kl_scale`（hook 改写） | R4Det.py:1108,1315,1332 / kl hook |
| RCFusion 属性名 | 固定为 `self.cross_attention`（预训练键 `cross_attention.*` 可直接加载同名子类） | R4Det.py:291 |
| PFN/scatter 顺序 | `pts_voxel_encoder` → `pts_middle_encoder` → `pts_backbone`；scatter 返回值直接进 SECOND | R4Det.py:724-729 |
| PFN 返回 tuple 有先例 | `PointPillarsScatterRCS` 解包 `(point_features, rcs)` | pillar_scatter.py:140-142 |
| 画布约定 | 见 §1 的 ①②③ | 逐行核验 |
| 训练/测试入口不改 | custom_imports + meta_info 注入均在 train_vod.py | train_vod.py:157,174 |
