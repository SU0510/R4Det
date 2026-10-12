# 候选 C 最小独立臂预登记：观测有效性条件化的更新（C1）

- 状态：**预登记冻结（2026-10-12 08:20 UTC，在第一次 C1 运行之前）**；本文件冻结后，
  公式、阈值与判定规则不得因结果而修改。
- 立项依据（已完成，不重复追加探针）：`docs/rssm_c_failure_enrichment_contract_20261012.md`
  §78 自然缺测富集成立（Δrecall **+34.78** 点，CI [21.57, 46.55]）；§79 受控屏蔽因果半步成立
  （同批 7157 个 covered GT 的召回 0.5149 → 0.3518，**Δ_mask = −16.31** 点，CI [−19.75, −11.62]，
  identity 全 val 逐位相同）。记录：`docs/training_runs_full.md` §78/§79。
- 训练记录口径（AGENTS 规则 5）：本节所有训练结论必须同时给出
  **区间均值（默认 fixed window ep12-16 五轮等权均值；窗口不完整时写明实际跨度与样本数）**、
  **全轮峰值（指标值 + epoch）**、**best saved（指标值 + epoch，仅 checkpoint 可用性）**。

---

## 1. 实际改动入口（路径图；不给根因定罪）

标准 `no2d_igdr` 的当前路径（均为代码事实，行号为当前 `goal-test` 分支）：

| # | 环节 | 位置 | 现状 |
|---|---|---|---|
| 1 | 雷达点云 → pillar 特征 → 雷达 BEV | `R4Det.py:722-731`（`voxelize` + `pts_voxel_encoder` + `pts_middle_encoder`）→ `pts_backbone/pts_neck` | 空回波区域特征恒为 0 |
| 2 | 相机 BEV（LSS） | `mmdet3d/models/img2bev/forward_projection/ViewTransformerLSS.py:77` → `R4Det.py:830-832` | (B,C,x=216,y=248) → 转置 (y,x) |
| 3 | 融合 | `R4Det.py:892` `ConcatConvFusion`（`concat_conv_fusion.py`：`conv(cat([img_bev, radar_bev]))`） | **concat 保留相机通道**；空雷达区域 = `[img_bev ; 0]`，但网络**无法区分「没有回波」与「回波特征恰为 0」** |
| 4 | RSSM 观测更新 | `rssm_fusion.py:646-728`（`MotionAlignedRSSMFusion.forward`）：`e_t = encoder(feat)` → `mu_q = posterior_mu([h_t, e_t])`，`z_t = mu_q`（测试期 deterministic） | 观测更新**无条件**作用于所有 cell；`mu_p/mu_q/KL/decoder/output` 见 §2 公式 |
| 5 | coverage / 缺失控制 | **当前不存在**：模型内没有任何以输入回波/缺失为条件的通路（唯一近似信号是帧级 `is_valid_mask`，只处理历史帧有效性；`pts_bev_feats` 的空 cell 只有隐式的 0） | ⇒ 本臂命名按事实：**显式保留缺测下的相机更新**，不称「修复已定位的断路」 |

**结论（写进论文口径）**：现有实现既没有「雷达缺失关闭相机」的显式断路，也没有可以利用
缺测的显式通路；C1 是**新增一个显式通路**，不是修复某个已定位的 bug。

## 2. 唯一研究因素：观测有效性条件化的更新（公式冻结）

单变量 vs 标准 `no2d_igdr`（`MotionAlignedRSSMFusion`）：给后验校正加一个**逐 cell 增益**，
增益的输入包含一个**由输入点云构造的雷达回波指示** `valid_t`：

```
valid_t[i,0,r,c] = 1[ 该 cell 内至少有一个已加载点 ]            # 纯输入函数，见 §4
g_t = 1 + tanh( W_v * [h_t ; valid_t] + b_v )                  # (B, latent_dim, H, W)
z_t = mu_q + (g_t - 1) * (mu_q - mu_p)      (≡ mu_p + g_t (mu_q - mu_p))
```

- **不变**：encoder / ConvGRU transition / 状态对齐 / `mu_p, logstd_p` / `mu_q, logstd_q` /
  KL（仍在**未加增益**的 `mu_q, logstd_q` 上计算，`kl_scale=1.0`、`free_nats=1.0` 不变）/
  decoder+重建目标 / `output_proj` + 残差 / 检测头 / 数据集 / 优化器与日程 / seed 协议。
- **不引入**：新协方差、innovation 标准化、新语义头、Doppler 门、额外损失项、额外监督。
- **零初始化恒等契约**：`W_v, b_v = 0` ⇒ `g_t ≡ 1` ⇒ `z_t == mu_q`（逐位相同）。
  CPU 已验证：与主线的 `MotionAlignedRSSMFusion` 前向**逐位相同**（`torch.equal` True）；
  扰动 `W_v` 后输出改变（gate 活性）；`W_v` 梯度非零（不是闲置参数）。
- **参数量增量**：`(256+1)×256 + 256 = 66,048`（1×1 conv，占 RSSM 主体约 0.5%）；
  计算增量：一次 1×1 conv @ 216×248，无额外 loss 前向。
- 布局：`valid_t` 与 RSSM 输入同布局（当前分支实测 rows=x=216, cols=y=248；实现里
  以 `feat.shape[-2:]` 断言，错布局会直接报错而不是静默错位）。

## 3. 控制矩阵（确定性对照并列，不后置）

| 臂 | 更新策略 | 状态 | 作用 |
|---|---|---|---|
| **B0** 标准 `no2d_igdr` | 旧（无 valid 条件） | 随机（采样 + KL） | Q1 的对照；已有 seed0 ep12-16 = 39.2809（n=5） |
| **C1**（本臂，先跑） | **新**（valid 条件增益） | 随机 | Q1：新 vs 旧 |
| **C1-D**（串行第二臂） | 同新策略（同一 gain 模块，同 shape/init） | 确定性（无 prior/logstd/KL、无采样，沿用项目既有 `DeterministicMotionAlignedLatentFusion`） | Q2：同策略下随机 vs 匹配确定性 |

- **只用 C1 优于 B0 不能证明随机状态必要**；Q1、Q2 必须分开报告。
- 容量匹配（**活跃参数**口径，2026-10-12 实测并冻结在
  `test_validity_gain_rssm.py::test_capacity_delta_matches_prereg`）：

  | 臂 | 活跃参数 | 相对 B0 |
  |---|---|---|
  | B0 `MotionAlignedRSSMFusion`(256) | **9,191,510** | — |
  | C1 `ValidityGatedRSSMFusion`(256) | **9,257,558** | +66,048（唯一的 1×1 gain：257×256+256） |
  | C1-D 基线 `DeterministicMotionAlignedLatentFusion`(256) | 6,831,446 | −2,360,064（去掉随机支路） |
  | C1-D `ValidityGatedDeterministicRSSMFusion`(256) | **6,897,494** | 同一 gain +66,048 |

  C1 与 C1-D 相差 **2,360,064** 活跃参数，即随机支路（`prior_mu/prior_logstd/posterior_logstd`
  等 3×3×256×256 级卷积）**整体移除**；**不补任何闲置参数凑数**，差值如实报告。
- 计算量：gain 为单次 1×1 conv @ 216×248，**≈3.53 GMAC/帧**（B=1），相当于 RSSM 里一个
  3×3 256→256 卷积（≈31.7 GMAC）的 ~11%，**无新增 loss 前向**；不加 `forward` 次数、不改 batch。
- 训练预算：epochs / batch / lr / scheduler / seed 协议对三臂**完全相同**；C1-D 按预登记串行执行。
- 底线（写进结论模板）：**若 C1-D 与 C1 同等有效，贡献表述为「缺测可区分的更新机制」，
  不得包装成「概率 RSSM 必要」**。

## 4. 推理门控禁止使用 GT（硬约束）

- `valid_t` 只能由**输入可见量**构造：当前实现 = 同一份已加载点云（`LoadPointsFromFile` +
  `PointsRangeFilter` 之后、网络可见的 `points`）的逐 cell 占用指示。
- **禁止**：GT 框内点数、GT 类别、`num_points_in_gt`、§79 的诊断掩码、任何 `is_valid_mask`
  以外的标签派生量进入推理路径；`frame-level/object-level` 缺测标签只用于**离线分组评估**。
- 回归测试固化：`valid_t` 只依赖 `points[:, :2]`（改动 x/y 之外通道不改变掩码；给定点云
  的掩码可复现）；R4Det 只在 fusion 声明 `accepts_obs_valid` 时传 `obs_points`，其它臂签名不变。

## 5. 三层门（避免重演 A2）

**S0 接线门（450 iter 冒烟，配置 `me_rssm/sanity/c1valid_smoke_450iter.py`）**

| 编号 | 判据 |
|---|---|
| S0-1 | loss 有限、无 NaN/OOM、跑到 iter 450 并写出 `iter_450.pth` |
| S0-2 | 第 0 步 `v_gate_mean == 1.0`（零初始化恒等）；训练启动后 `W_v` 梯度非零 |
| S0-3 | 每帧 `valid_t` 非空、cell 级 invalid 比例被记录（对照 §76 对象级 19.8% zero）；布局断言通过 |
| S0-4 | 非本臂路径不受影响（其它 fusion 调用签名逐字节不变）；主线 450-iter 行为作为对照 |

**早期机制门（ep6，同 epoch / 同 seed / 同 harness）**

- 必测**真实 CE / 召回**，不用梯度范数或特征范数替代：
  在 ep6 checkpoint 上按 §76 冻结分组分开报告 **自然缺测组**（zero / low / mid / high）与
  **frame-level vs object-level**，以及**受控屏蔽组**的三分表：
  `完整输入 / 自然缺测 / 受控屏蔽`。
- 判据（先写死）：`Δrecall(covered, C1 − B0 同窗)` ≥ 0 且 **zero 组召回相对 B0 同窗不降**
  （`≥ −0.5` 点）；分类 CE（zero 组）不劣化（`≤ +0.02`）。二者任一失败 ⇒ 机制门 FAIL，
  归档负结果，不进 ep16（不做超参/增益系数抢救）。
- 附加读数（只报不判）：`v_gate_valid_mean` vs `v_gate_invalid_mean`、gate 的区域分布。

**成熟任务门（正式 EvalHook，ep12-16）**

- 主指标：`pts_bbox/KITTI/Overall_3D_moderate` 的 **ep12-16 五轮等权均值**（n=5），
  对照 B0 同 seed 同窗 `39.2809`。
- 通过线：Overall 区间均值 **≥ 40.00**；四类**非劣**：Car ≥ 47.34、Truck ≥ 29.22、
  Ped ≥ 29.93、Cyc ≥ 47.63（B0 同窗四类均值）；止损：任一类别 ≤ B0 同窗 −2.0 点 ⇒ 停。
- 同时必报：全轮峰值（值 + epoch）、best saved（值 + epoch）、完整逐 epoch 曲线、
  逐类别拆解、以及 §6 的分表。
- 三 seed：**单 seed 过门后**再排 seed1/seed2；不预先承诺多 seed 长训。

## 6. 报告纪律（与既有契约一致）

- 自然缺测 / 受控屏蔽 / 完整输入 **分表**；frame-level 与 object-level 缺测 **分开**。
- 既有 score（0.25 操作点，0.10/0.50 仅敏感性）、类 IoU（Ped/Cyc 0.25，Car/Truck 0.5）、
  分组边界（zero=0、low 1–4、mid 5–10、high ≥11；`v_amp_low ≤2.079`、`mid ≤9.421`）
  **一律不动**。
- 离线 harness 与训练内 EvalHook 的差额（§51.5 的历史 ~2.2 点）只是**口径差**，
  禁止当作可加减的校正常数；同一结论只用同一来源。
- FP 仍进独立桶，不塞进任何 GT 分组。

## 7. 停止规则（写死）

1. S0 任一判据失败 ⇒ 修实现后重跑冒烟；连续两次失败 ⇒ 停，报硬问题（不调阈值）。
2. ep6 机制门 FAIL ⇒ 归档，**不进 ep16**，不扫增益系数/权重/KL。
3. ep12-16 任务门 FAIL 或触发止损 ⇒ 归档负结果，按 §75.15 转对象级读出 / 空间定位 /
   跨场景泛化分支。
4. C1-D 与 C1 同等有效 ⇒ 结论改写为「缺测可区分的更新机制」，不主张随机状态必要性。
5. 任何阶段发现 GT 泄漏、接线不等价、梯度断路、NaN/OOM、实现与本节不一致 ⇒ 立即停并报告。

## 8. 实现与产物索引（冻结时）

- 实现：`me_rssm/validity_gain_rssm.py`（`build_radar_validity_mask`、
  `ValidityGatedRSSMFusion`、`ValidityGatedDeterministicRSSMFusion`）
- 接线：`mmdet3d/models/detectors/R4Det.py`（仅当 fusion 声明 `accepts_obs_valid` 时传
  `obs_points=points`；其它臂签名不变）
- 臂配置：`me_rssm/configs/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_c1valid.py`
- 冒烟配置：`me_rssm/sanity/c1valid_smoke_450iter.py`
- 运行目录：smoke `/data/lurui/work_dirs/c1valid_smoke_450iter`；
  seed0 `/data/lurui/work_dirs/fgfull_N4_no2d_igdr_2x4_24e_c1valid_seed0`
