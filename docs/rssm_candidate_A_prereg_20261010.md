# 候选 A 预登记：z 的对象语义监督 + 检测分类 residual 读出（2026-10-10）

> **状态：预登记草案。数值门（§5）是建议值，须用户批准后才允许投正式训练；
> 实现与 450-iter 只读冒烟不需要该批准（不产生任何性能宣称）。**
> 本文按 `docs/rssm_next_step_audit_20261010.md` 的排序写成：
> 阶段 0 真实目标审计（§70）已把「重建目标改革」的承重前提否掉
> （`docs/rssm_target_reform_prereg_20261010.md` §3A），差分目标作为**负结果归档**，
> 本文件是排序里的下一步 = 候选 A。
> 差分/标准化/detach/z-only decoder 这四件事**不再同时改动**；本实验不动重建目标。

## 1. 决策与立项理由（测量驱动，不是"再叠一个模块"）

**为什么不再投差分目标**（全部实测，见 §70.6）：真实目标是 56.5% 样本特异的，
「95% 共享模板」不成立；差分目标的特异占比 93.7% **等于纯噪声地板** (S−1)/S；
按现役契约（权重 1.0、不缩放）换差分会把 recon 项目标能量压到 4.0%，
**削弱**而不是增强对 z 的约束。⇒ 目标内容不是瓶颈，立项前提塌了。

**为什么是 A**：审计留下的三个可复核事实都指向"监督→读出"这条断链，而不是
"目标没信息"：

| 事实 | 数值 | 出处 |
|---|---|---|
| recon 项只占总损失 0.18%（ep24），自 ep4 起 <0.5% | 尺度问题，不是内容问题 | §70.2 / §70.6 补正 |
| KL 被 free-bits 完全吸收（raw KL 0.597 < 1.0，clamped 0.9985），`μ_q−μ_p`² 从 1.527 降到 0.0072 | 后验塌到先验，z 不承担观测信息 | §70.2 |
| `zero_z` 使 head 输出变 38.02%，但 `shuffle_z` 只变 4.67% | z 是**公共底色**，不是样本级判别信息 | §49（no2d_igdr seed0 ep14） |
| 标准 RSSM 的 `shuffle_z` 效应**已经**前景集中（fg 0.1888 vs bg_hi 0.0438 = 4.3×，12/12） | 读出的"注意力"不是缺失项；缺的是**类别判别内容** | §69 测量(1)（该 checkpoint 实为标准 `MotionAlignedRSSMFusion`，见 §3.2 纠正） |

⇒ 机制假设（本次要检验的唯一命题）：

> **H-A：给 z 一个只读 z、以对象中心为粒度、类别均衡的语义监督，并把这条语义
> 以 zero-init residual 形式接进检测分类 logits 之后，z 携带的对象类别信息会变成
> 可验证的真实检测决策增量；若做不到，收益不可能来自"修目标/修门控/修对齐"。**

H-A 被否同样是有价值的结论：它把"监督职责由 h 承担"（§70.6 剩下的两个嫌疑之一）
从推断升级为可复现测量，并直接给出论文 Analysis 的负面证据。

## 2. 唯一变量（arm 定义）与不变项

**底座**：标准 `no2d_igdr` 24e（`MotionAlignedRSSMFusion`，全分辨率 RSSM）。
**不改**：重建目标（仍是全分辨率原始 `bev_feats`、不 detach、权重 1.0、
burn_in=2 两帧等权）、posterior/prior/KL/free_nats/kl_scale、`output_proj` 读出、
backbone、检测头几何分支、优化器、日程、数据管线、随机种子协议。

**新增（A 臂 = `sem_supervision_weight>0` 且 `cls_residual=True`）**：

1. `sem_head`：只读 `z_t`（full-res，256ch）→ 3×3conv→ReLU→1×1conv → 4 通道
   对象中心热图。监督 = CenterNet focal（与 §59 route-C 同一实现口径），
   目标 = 当前帧 GT 框中心在 BEV 格上的高斯热图（`bev_centers_to_heatmap`，
   格尺寸 (216,248)，`min_radius` 固定，不按 val 调）。
   ⇒ 梯度**只能**通过 z 反传，`h_t` 不在输入里（z-exclusivity，§6 T3 单测证明）。
2. `z_cls_logits`：只读 `z_t` 的 1×1 conv → **4 通道**类别 logit，
   按 `anchor_class_mapping=[0,0,0,1,2,3]` 广播到 6 个 anchor 槽
   （`conv_cls` 的 `[anchor*4+class]` 布局，见 `anchor3d_head.py:236-249`），
   **zero-init**，加在 `cls_score` 上（几何 `conv_reg`/`conv_dir_cls` 完全不动
   ⇒ 分类 residual 与几何 residual 分离）。

$$\text{logits}_{a,c}(\text{cell}) = \text{conv\_cls}(x)_{a,c} + W_{\text{cls}}\,z_t(\text{cell})\big|_c$$

`feat→head` 主路、`output_proj(z)+feat` 残差、KL/recon 全部保持原样。

**zero-init 的含义（必须逐条核验，§6）**：`W_cls ≡ 0` ⇒ 第 2 项对
`cls_score` 恒等于 0 ⇒ **加载基线 checkpoint 时模型输出与基线逐位相同**，
训练第 1 步之前不改变任何检测行为；`sem_head` 不在检测前向路径上。

## 3. 与已证伪清单的差别（为什么这不是重复）

| 已做过 | 与 A 的差别 |
|---|---|
| §50/§52 低维 bottleneck + 排他未来任务 | 低维 16×16、无重建、FiLM 旁路；A 是 full-res z、重建与主路不动 |
| §53 排名/对比损失 | 粒度是整幅 16×16 correction；A 是对象中心 + 逐类别 logit + 接进检测 logits |
| §59 route-C（`latent_object_*`） | 挂在**低维** fusion 上、只监督不读出、且那条 run 的 fusion 没有 decoder；A 在标准 full-res 家族上，且读出/监督可分别开关 |
| MCI/DC 家族（门控、证据场、ego-warp） | 全部改**状态侧/测量侧**；A 改**监督粒度 + 读出终点**，不碰门控与对齐 |
| §68.9 家族五次复现的副作用 | Ped/Cyc 大跌；A 把 Ped/Cyc 列为硬非劣门（§5），不是看 Overall |

## 4. 实现契约（源码级，实现必须与此一致）

| 项 | 契约 |
|---|---|
| 新增类 | `me_rssm/semantic_readout_rssm.py::SemanticReadoutRSSMFusion(MotionAlignedRSSMFusion)`，不改基类 |
| 新模块 RNG | 构造新模块前后保存/恢复 RNG，保证基线权重逐位不变（否则"off 态等价"无法单测） |
| z 的语义监督作用帧 | **只有当前帧**（`set_object_gt` 只在前向前设置，`reset_state()` 清除）。历史帧不算 sem loss |
| 热图分辨率 | 运行时用 `z_t.shape[-2:]`=(216,248)，不写死；`bev_centers_to_heatmap` 的 H↔x、W↔y 约定 |
| 送进 head 的 residual | 与 `pts_feats` 同样 `permute(0,1,3,2)`（head 侧是 (248,216)）；`is_valid_mask` 为 False 的样本 residual 置 0（与 `bev_feats` 回退一致） |
| 训练/推理一致性 | residual 在 train 与 val **都生效**（val 用 posterior 均值 `mu_q`，与基线一致）；否则 train/test 契约不一致 |
| 关闭态 | `sem_supervision_weight=0` 且 `cls_residual=False` ⇒ 不建参数、前向逐位等价基线（单测 T1） |
| 日志 | `loss_rssm_sem`（真实梯度项）+ `stat_sem_*`（detached 诊断）+ `stat_cls_res_rms`（读出幅度） |

## 5. 判据（**建议值，待批准**）

固定口径：seed0、`ep12–16` 五轮等权均值（窗口不完整时写实际跨度与 n）、
全轮峰值写"值 @ epoch"、best saved 单列；Car/Truck strict、Ped/Cyc loose。
基线 no2d_igdr seed0：Overall **39.2809**、峰值 40.3853@ep14、
Car 47.8399、Truck 29.7246、Ped loose 30.9277、Cyc loose 48.6313。
噪声参照：no2d_igdr 三 seed 的 ep12-16 std = **1.35** ⇒ 单 seed 小于 ~0.7 的差
不得解读为效应。

- **S0 接线/等价（不训练或 450 iter，不作性能宣称）**
  (a) 关闭态逐位等价；(b) 加载基线 `epoch_14.pth` 且 A 模块 zero-init 时，
  同一批 val 帧的 `cls_score/bbox_pred` 与基线**逐位相同**；
  (c) `loss_rssm_sem` 有限且下降、无 NaN/OOM；
  (d) **压力门（§70 的直接教训）**：`‖∂loss_rssm_sem/∂z_t‖ / ‖∂total/∂z_t‖ ≥ 5%`
  （450 iter 内均值；<1% 判"没有压力"，直接停，不靠放大系数救）；
  (e) `stat_cls_res_rms` 必须离开 0（读出活着），但 `cls_score` 相对基线的
  偏移在冒烟结束时不应爆炸（记录中位数/ p98）。
- **S1 ep6 机制门（seed0，决定要不要跑到 ep16）**
  M1 语义头质量：val 上 GT 中心处 `thr50 ≥ 0.30` 且 `hit@1 ≥ 0.60`；
  M2 **z 排他性**：同一权重下把 sem_head 输入换成 `h_t`（或 `feat`）时
  `thr50` 必须**显著低于** z 输入（≥ −30% 相对），否则语义其实来自 h/观测，
  z 只是被穿过的通路；
  M3 读出非装饰：训练后用 `stat_cls_res_rms>0` + 强制 residual=0 的配对
  toggle，正样本 anchor 的 class-CE 变差（≥8/10 clip，符号检验）；
  M4 类别正确性而非幅度：在 GT 前景格把 z 换成**异类样本**的 z，
  分类 logit 必须朝 donor 类移动（`argmax` 命中率 ≥ 随机基线 +20 pt）。
  ⇒ M1–M4 任一不过：**不进 ep16**，按负结果归档并把读数写进 Analysis。
- **S2 ep12–16 任务门（seed0）**
  AP GO（才允许投三 seed）：窗口均值 **≥ 40.00**（= 基线 +0.72 ≈ 半 std）；
  非劣线（任一破线即 FAIL，不看 Overall）：Car ≥ 47.34、Truck ≥ 29.22、
  **Ped ≥ 29.93、Cyc ≥ 47.63**（Ped/Cyc −1.0，因为家族失败模式正是这两类 −5/−6）；
  HARD FAIL：< 38.78（基线 −0.5）。
  落在 (38.78, 40.00) 或类别非劣不过：按 §7 分叉处理，**不做** KL/权重/续训扫描。
- **S3 三 seed（仅 S2 通过后）**：同协议 seed1/seed2 配对；
  四类各自报告均值 + 跨 seed 波动 + 峰值 epoch，不用 Overall 掩盖负类。
- **对照臂（同门同日程，仅在 S2 之后按需投，不同时开）**
  C1 `cls_residual=False`（只监督不读出）；C2 `sem_supervision_weight=0`
  （只读出不监督）；C3 `sem_head_input='feat'`（相同监督换成当前观测，
  读出仍只读 z ⇒ 检验收益是否必须来自"语义进 z"）。

## 6. 单元测试（实现前置，全部必须绿）

T1 关闭态不建参数 + 前向逐位等价基类；T2 `z_cls_logits` zero-init ⇒ head
`cls_score` 与不带 residual 时逐位相同；T3 `loss_rssm_sem` 对 `h_t` 梯度恒为 0、
对 `z_t` 非 0（z-exclusivity）；T4 热图光栅化对齐（已知框中心 → argmax 落在
正确格/正确类通道）；T5 residual 通道布局与 `anchor_class_mapping` 一致
（Ped 3 槽共享同一 logit、Truck 槽独立）；T6 `is_valid_mask` False 样本
residual=0 且 sem loss 不计这些样本；T7 permute 契约（(216,248)↔(248,216)）；
T8 RNG 保存/恢复后基线模块权重逐位不变；T9 `reset_state()` 清 GT 缓存与
pending loss（防止历史帧误监督）。

## 7. 结果分叉（预先写死，不许事后挑解释）

- S1 机制门全过、S2 AP 不过：读出/监督成立但任务不收益 ⇒ 拆召回、中心/尺寸/朝向、
  score/NMS（不统称"分类问题"），转 C 的证据由缺测分组决定；**不**扫 sem 权重/KL。
- S1 M2 不过（语义其实在 h/观测）：**probabilistic state 的贡献主张作废**，
  转"z-exclusive residual"单变量验证或直接写负面 Analysis。
- S2 过、S3 跨 seed 不一致：按 seed 噪声记录，不挑最好 seed 当结论。
- S2 AP 涨但 M3 不过（residual 是装饰）：涨点归因于 sem loss 的隐式正则，
  不得写成"z 进入检测决策"。
- 任何阶段出现 Ped/Cyc ≤ 非劣线：立即停，记录为家族副作用第六次复现。

## 8. 运行与资源

GPU 只用 5/6/7（当前空闲）。冒烟：3 卡 DDP、seed0、deterministic、450 iter、
无 val。正式：24e 日程，ep6 看门狗（先例 `me_rssm/sanity/watch_dcrssm2_ep16.sh`）。
训练记录写进 `docs/training_runs_full.md` §71，按 AGENTS.md 第 5/6 条同时给
"区间均值 + 全轮峰值（值+epoch）"，并保留完整逐 epoch 曲线与逐类别拆解。
