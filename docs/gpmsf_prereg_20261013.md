# 预登记：几何配对多尺度融合（Geometry-Paired Multi-Scale Fusion, GP-MSF）

- 冻结时间：**2026-10-13**（在首次 GPU 运行之前；沿用既有偏好：预登记 + 最小实现 + S0 + 短冒烟 → 开训）
- 臂名：**GP-MSF = `GeometryPairedMultiScaleFusion`**
- 配置：`me_rssm/configs/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_gpmsf.py`
- 冒烟配置：`me_rssm/sanity/gpmsf_smoke_450iter.py`
- 运行目录（待建）：`/data/lurui/work_dirs/fgfull_N4_no2d_igdr_2x4_24e_gpmsf_seed0`
- 设计（权威）：`docs/geometry_multiscale_fusion_design_20261013.md`
- CPU 证据：`docs/geometry_multiscale_fusion_audit_20261013.json`；接线单测：`me_rssm/sanity/test_gpmsf_wiring.py`（7/7 PASS）

---

## 0. 立项证据（实测，非推断）

- 仓库内历史 `crossmodal_fusion_*` 臂：ep12-16 五轮等权均值（n=5）Overall
  **37.9789** vs B0 **39.2809**（Δ = **−1.3019**），其中 Pedestrian −5.24 —— 说明
  「把 concat 换成空间 attention」本身不涨 AP，但**不能**外推为「所有高级 fusion 无效」
  （该模块没有可靠性估计、没有缺测处理、没有新增取证地址）。详见设计 §1.1 与 F1 预登记 §0。
- B0（标准 `no2d_igdr` seed0）现有口径：ep12-16 区间均值 **39.2809**（n=5）、
  全轮峰值 **40.3853 @ ep14**、ep1-6 区间均值 **29.0420**（n=6）；三 seed 样本标准差
  **1.3475** ⇒ **单 seed 的 +0.5 不能当稳定贡献**，也不能把该标准差直接当显著性阈值。
- 当前主线瓶颈的候选假设（设计 §2，证据等级：待证）：细尺度证据在 BEV 混合后不可恢复、
  雷达只在同 cell 控制视觉强弱而没有局部空间取证、跨模态空间对应未被显式约束。
- ⇒ **允许的结论**：GP-MSF 是值得优先实现、**可被反驳**的观测机制候选。
- ⇒ **不允许的结论**：GP-MSF 更复杂所以更准；或它一旦涨点就已证明 RSSM 的随机 latent 必要。
- 排程依据：2026-10-13 的决定（训练记录 §92）——**F1 让位**，GP-MSF 直接接在 A2-R 续训之后；
  F1 实现保留，回答「收益是否只是门控」时再按完整口径训练。

## 1. 假设（单一、可否证）

> 当前前端只用一条「同 cell concat → conv」的通路消费雷达与相机。如果给融合块一条**几何接地、
> 跨源共享米制地址**的第二取证通路——同一地址分别读**原生**雷达多尺度与**原生**图像多尺度，
> 再用「证据在哪（offset）+ 哪个分辨率解释它（level 选择）+ 内容兼容性」决定消息，并按
> **残差**叠加回原 concat 通路——则可以在**不改变 RSSM/head/损失**的前提下提升
> `pts_bbox/KITTI/Overall_3D_moderate`，尤其是小目标与缺测场景（Ped/Cyc、low/zero 组）。

可否证形式：若同容量普通 residual conv 对照与 GP-MSF 同量级、或 GP-MSF 只在 Car/Truck 变好而
Ped/Cyc 或 low/zero 变差，则本假设不成立（§6/§7）。

## 2. 唯一研究因素（公式与几何冻结）

```
C_new = C + W_{R->C}(M_R)        # (B,64) -> (B,256)，零初始化
R_new = R + W_{I->R}(M_I)        # (B,64) -> (B,384)，零初始化
O_t   = fusion_conv(cat([C_new, R_new], dim=1))   # 与基线同一 conv、同一权重
```

- **地址（bounded metric address）**：`a_hk(p) = p + b_par,h * tanh(u_hk) * e_r + b_tan,h * tanh(v_hk) * e_theta`，
  在 **augmentation 后**的 BEV 系内、相对 augmented 原点建立（存在 translation 时不得默认原点为零）；
  每头 4 个地址（k=0 为固定中心），初值 `offset_bounds = ((0.64,0.32),(1.28,0.32),(0.32,0.64),(0.32,1.28))` m。
- **同一地址读三个源**：原生雷达层（SECONDFPN **之前**）+ 原生图像层 P2/P3/P4；层选择用逐候选
  masked softmax（「哪个分辨率解释这个地址」），候选聚合再用一个独立 masked softmax（「证据在哪里」），
  **两个方向各自归一化**，避免密集雷达候选把 camera-only 候选挤掉。每方向保留 value=0 的 **null** 项。
- **缺测处理**：新增 **雷达→相机** 直接证据要求该地址在 0.16 m scatter 网格上有局部回波支持
  （5×5 窗口，来自**输入 xyz**）；**图像→融合**消息在没有回波时仍然可用。
  新增雷达候选全部无支持 ⇒ 雷达消息为 0；图像候选全部无效 ⇒ 图像消息为 0。
- **冻结几何（拷贝自 resolved 主线配置，不随训练变化）**：
  `bev_shape=(248,216)`（行=y/列=x）、`bev_origin=(0.0,-39.68)`、`bev_cell=0.32`；
  `radar_origin=(0.08,-39.60)`、`radar_base_cell=0.16`、`radar_base_shape=(496,432)`、
  `radar_level_channels=(64,128,256)`、`radar_level_steps=(0.32,0.64,1.28)`；
  `image_channels=256`、`image_size=(480,640)`、`image_level_strides=(4,8,16)`、`depth_downsample=8`。
- **train-only 高度 references（覆盖先验，非 per-sample GT、非高度预测）**：
  `(-1.1230436265, -0.3003779650, 0.6601446867)` m（train split 加权 q10/q50/q90）。
- **容量**：`embed_dim=64`、`num_heads=4`（每头 16 维）、`num_offsets=4`、单融合块；
  `query_chunk=4096` + `grad_checkpoint=True`（分块只减前向暂存，反向重算不可省）。
  新增参数 **177,726**（0.1777 M）；`fusion_conv` **1,475,072** 参数、形状与基线逐位相同
  ⇒ 预训练 `cross_attention.fusion_conv.*` 可原样加载。
- **step 0 等价**：两个输出投影零初始化 ⇒ 与**同状态** `ConcatConvFusion` 逐位相同（单测断言）。
- **RNG 隔离**：新参数在 `new_param_seed=20261013` 的独立 RNG 下初始化；**相同 seed 不自动保证**
  新增模块之后 RSSM/head 的初始化一致，故必须由 S0-2/单测机械核验共同模块初始状态。
- **不动**：backbone、neck、head、`temporal_fusion`（仍为标准 `MotionAlignedRSSMFusion`）、KL 日程、
  优化器、数据管线、总 epoch、score/IoU/分组边界。**不引入新 loss、不加对象 decoder、不加二阶段 RoI、
  不加 occupancy loss，也不做全 BEV N² attention。**
- 单变量契约由 S0-2 机械核验：臂与主线配置差异键**必须恰好等于 `['RCFusion']`**。

## 3. 布局正确性（预防静默错位）

- 融合输入布局为 `(B, C, 248, 216)` = **行 y / 列 x**（`bev_h_=216`、`bev_w_=248` 是 (w,h) 命名，
  与张量布局互为转置），因此 `bev_shape` 以 `(y, x)` 记录。
- 原生雷达三层是 SECOND backbone 输出（**SECONDFPN 之前**）0.32/0.64/1.28 m，
  形状 `248×216 / 124×108 / 62×54`；每层用自己的 index bookkeeping（`radar_origin` 为 scatter
  cell 0 的物理**中心** `(0.08, -39.60)`），**不先统一上采样**、不盲用范围归一化。
- 地址→图像：先撤销 BEV augmentation（`bda` 逆）再 `augmented → image` 完整投影（`lidar2img`），
  投影后按 `post_rots/post_trans` 校正；相机深度取齐次分量 **index 2**；采样 `align_corners=False`。
  `_gpmsf_geometry` 对形状**硬断言**（B/N 一致、三层雷达、≥3 层图像、`post_rots` 2×2/3×3、
  `post_trans` 末维 2/3），错接线在第一 iter 就报错，不会静默退化成 concat 基线。
- `require_geo=True`：训练期若检测器不再传 `geo`，**直接 RuntimeError**；eval 下保留 concat 回退。
- 静态核验：CPU gate `t3` 断言三层雷达/三层图像各自的 cell centre 精确读回，`t4` 断言 P3 与三个高度
  reference 的行位置（读回误差 ≤ 2.4e-07），`t9` 断言地址半径 ≤ per-head 上限。
  **布局/投影错误会静默镜像或错行**，因此三者都是硬门。

## 4. 数据、调度与队列（与主线一致）

- seed 0、`--deterministic`、3-rank DDP、GPU **5/6/7**、24 epoch、与 B0 相同的 EvalHook 与数据管线。
- 队列：**替换原 F1 槽位**（`queue_relfus.sh` → `queue_gpmsf.sh`），串联在 **A2-R 续训队列之后**，
  同一时刻只占用 5/6/7；A2-R 未结束前 GP-MSF 不启动。
- 冒烟：450 iter（`IterBasedRunner`），`me_rssm/sanity/gpmsf_smoke_450iter.py`，
  带 `GPMSFWiringStatsHook`（每 50 iter 落 `gpmsf_wiring_stats.json`）与保留的 KL 日程 hook。
- 冒烟与正式 run 均为原 batch / 原 N4 口径；**不得偷偷降 batch 后继续称单变量对照**。

## 5. S0 接线门（先冒烟，后判定）

先跑便宜门：`test_gpmsf_wiring.py`（7 项，接线/配置/step-0）与 `gpmsf_cpu_checks.py`（10 项，含
分块等价、双向梯度、零雷达/无图像退化、边界）；任一失败 ⇒ 不启动冒烟。

| 判据 | 内容 | 判定器 |
| --- | --- | --- |
| S0-1 | 跑到 450 iter、写出 `iter_450.pth`、被记录的 loss/`stat_*` 无 NaN | `gpmsf_s0_verdict.py` |
| S0-2 | 臂/主线差异键恰为 `['RCFusion']`；新建 `proj_r2c`/`proj_i2r` 零初始化；`fusion_conv` 形状与基线一致；`height_refs` 与主线冻结值一致；`require_geo=True` | 同上 + 单测 |
| S0-3 | 训练后 checkpoint 中 `proj_r2c/proj_i2r` 权重**已离开零** ⇒ 几何通路确实跑了（该通路只能由原生多尺度证据到达，闲置参数不可能自发移动） | 同上 |
| S0-4 | 容量冻结：新增 177,726、`fusion_conv` 1,475,072 | 同上 |
| 附加 | 性能对照（时间/显存 vs 既存 baseline 前端冒烟中位数）：任一比 > 1.2× ⇒ 记 `S0_PERF_REVIEW` 供人工复核，**不自动停机** | 同上 |

S0 任一 FAIL ⇒ 修实现后重跑冒烟；连续两次 FAIL ⇒ 停并报硬问题（不调阈值）。

## 6. 判定（三层，遵循 2026-10-12 的附录 B 政策）

1. **ep6 早期检查（只查任务损害）**：`c1valid_curve_check.py --window 1-6`，GP-MSF 的 ep1-6 区间均值
   低于 B0 同窗（**29.0420**）**超过 2.0 点** ⇒ 停并复查。机制读数不参与裁决。
2. **ep12-16 成熟窗口（主对照，n=5）**：按统一口径报**区间均值 + 全轮峰值（值 + epoch）**，
   并与 B0 同窗口逐类比较（Car/Truck/Ped/Cyc）。**必须同时看**四类、high/low/zero、
   FP/FN、径向/切向中心误差与**错误借点率**（框内零雷达点 ≠ 周边没有邻车回波，只能靠离线诊断验证）。
3. **性能目标（预登记，三 seed 配对）**：ep12-16 均值相对 B0 **≥ +2.0 AP points**；
   单 seed 只作为方向性证据，并同时报告全部 val epoch 的峰值及各类代价。

**止损/失败判据（任一成立即判定该形态不成立，需机制归因而非加超参）**：
- ep6 早期检查不通过，且 ep8-10 未见回升；
- ep12-16 区间均值不优于同容量 residual conv 对照；
- 仅 Car/Truck 变好而 Ped/Cyc 或 low/zero 变差 ⇒ **不能宣称解决了当前主要问题**。

## 7. 归因、控制臂与优先级

第一轮只做四个核心臂，目的分别是：

| 对照 | 目的 |
| --- | --- |
| B0：原 `ConcatConvFusion` | 同栈基线 |
| 同容量普通 residual conv | 排除「只是增加容量」 |
| 同 source 的固定几何采样（去 learned address） | 排除「只是重新取证」 |
| 完整 GP-MSF | 检验候选整体能力 |

有成熟正结果后才补：原 BEV-only vs 原生 PV/native radar；单尺度 vs 多尺度；共享地址 vs 独立
offsets；去兼容性/去 null/去 camera fallback；单高度 vs 三高度。

- **已知风险必须写在结论里**：高度 references 只是覆盖先验，坡道、高 Truck、遮挡仍是明确风险；
  「框内零雷达点不等于周边没有邻车回波」，错误借点率须由离线诊断给出。
- **论文宣称边界**：前端 AP 提升**不等于**解决了 RSSM 随机状态 z 的有效性问题。只有在本臂有成熟
  收益后，才做 `ConcatConv × GP-MSF` 与 `RSSM × 原 GRU` 的 2×2（比较 `(B−A)−(D−C)`）并配合
  h/z 排他或确定性对照；只有 reconstruction/KL 更好而 AP 没改善 ⇒ **不算任务成功**。

## 8. 产物索引

- 实现：`mmdet3d/models/fusion_layers/geometry_multiscale_fusion.py`（`GeometryPairedMultiScaleFusion`）
- 接线：`mmdet3d/models/detectors/R4Det.py`（仅当 fusion 声明 `accepts_geometry_sources` 时传
  `geo=self._gpmsf_geometry(...)`；原生雷达层在 `extract_pts_feat` 中于 SECONDFPN 前缓存；
  其它臂签名与行为不变）
- 配置：见文首；冒烟：`me_rssm/sanity/gpmsf_smoke_450iter.py`
- 判定：`me_rssm/sanity/gpmsf_s0_verdict.py`；CPU gate：`me_rssm/sanity/gpmsf_cpu_checks.py`；
  单测：`me_rssm/sanity/test_gpmsf_wiring.py`
- 队列：`me_rssm/sanity/queue_gpmsf.sh`（日志 `queue_gpmsf.log`，早期检查 `gpmsf_ep6_early_check.json`，
  成熟窗口 `gpmsf_ep16_window_check.json`）
- 训练记录：`docs/training_runs_full.md` §91（设计）/ §92（优先级）/ §95（实现 + CPU 门）
