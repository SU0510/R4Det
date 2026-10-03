# RSSM 框架机制级审查与改进路线（2026-10-02）

> 任务来源：用户要求遍历全部历史训练记录，找出当前 RSSM 框架的问题，
> 为「创新性改造 RSSM + 大幅提升 TJ4D + 发表论文」提供依据。
> 本文只做分析与路线设计，不含任何训练动作。
> 证据指针格式：`§N` = `docs/training_runs_full.md` 第 N 节；
> `file:line` = 当前仓库代码；探针 JSON 指 `/tmp/*seed1_ep16*.json`。
> 写作时运行状态：MAI seed2 训练中（ep13，ep16 验证落盘后链式看门狗自动停），
> 之后自动接力 fgfull18 seed0（§66）。**本文结论不受其影响，但第 7 节实验序列
> 以「不打断当前队列」为前提。**

---

## 0. 审查范围与证据来源

| 来源 | 覆盖内容 |
|---|---|
| `docs/training_runs_full.md`（10646 行，§0–§66） | 全部 66 节逐 run 记录（目录结构全量提取 + 关键节精读） |
| `docs/todo.md`、`me_rssm/STATUS.md`、`me_rssm/README.md` | 当前主线、预注册门控、ME-RSSM 设计包 |
| `me_rssm/docs/01–11` | 架构地图、14 项结构检查、12 候选筛选、ME-RSSM 设计、二线问题、新颖性核查、t0 死状态探针、论文骨架 |
| `work_dirs/`（= `/data/lurui/work_dirs/`，84 个条目） | 全部 run 的 ckpt 状态/日志尾部/队列日志盘点（2026-10-02 快照） |
| 代码 | `mmdet3d/models/fusion_layers/rssm_fusion.py`（2044 行）、`R4Det.py` 时序接线、MAI 配置 |
| 探针数据 | `/tmp/z_util_mai_seed{0,1}_ep16_*.json`、`/tmp/post_source_mai_seed1_ep16_stride700.json` |

---

## 1. 一页结论

1. **当前 RSSM 的"概率"部件是装饰性的**（P2，代码+实证双重确认）：
   `free_nats=1.0` 逐元素钳位使 `clamped_ratio≈0.99`，KL 项对 99% 元素零梯度；
   prior 只作 KL 正则器、推理路径完全不经过（推理恒 `use_posterior+deterministic`）；
   `μ_diff²≈0.03–0.04` 相对 `σ_q≈0.23` 是稳态塌缩。**模型实际是"确定性 ConvGRU +
   一个被称为 z_t 的残差卷积输出"。**
2. **读出通路是机制与 AP 之间的缺失环节**（P1，全线证据）：
   `output = output_proj(z_t) + feat`（rssm_fusion.py:716-717/2023-2024）是全局加性残差，
   无逐像素加权。实测 z 对 head 的因果作用以公共偏置为主：zero_z 26–56%、
   shuffle_z 仅 2.7–8.4%（§62/§64/seed1 探针），shuffle/zero≈0.1。
   路线 A/B/C 证明：修好 correction 的**构成**（innovation 特异占比 62–67%）
   不能自动转化为 head 的**使用**——缺的是"何时信 z"的逐像素结构。
3. **信息流修复是真实的，但 AP 不可跨 seed 复现**（P7）：
   MAI 把 h/e 依赖比从 2.23（标准主线，反向恶化）压到 <1（ep6/12），
   `innovation` 特异占比 47–67%；但 seed0 +0.84 / seed1 −2.51 方向相反，
   且 ep16 机制优势回落（h/e 回升 1.12）。**单机制效应（≤1 点）< 基线 seed 间波动（2.6 点）。**
4. **数据集唯一直接的运动测量（Doppler 径向速度）从未进入时序动态**（P4）：
   这是 14 项结构检查（me_rssm/docs/02）中唯一的"信息已存在但结构到不了"问题。
   ME-RSSM 三机制设计包（无参数 Doppler 证据旁路 + 可靠性门控 Kalman 校正 +
   动态证据调制更新门）**sanity 全绿、等价性三重验证、消融网格齐备，但从未训练**——
   它仍然是未兑现的最大创新资产。
5. **改进主线（本文提议）**：把 MAI（innovation 后验）与 ME-RSSM（Doppler 条件化
   predict + 可靠性增益 correct + 动态门）合并为统一的**测量条件化 predict-correct
   滤波器**，并补上第三个缺失环节——**可靠性门控读出**（P1 的直接解），
   外加三个低成本训练信号修复（观测引导状态初始化、差分重建目标、KL 修理或诚实降级）。
   论文骨架沿用 `me_rssm/docs/11`，素材完成度见第 6 节。

---

## 2. 事实基线

### 2.1 架构事实（代码核验）

- 状态：`h_state, z_state ∈ (B,256,216,248)` 全分辨率画布；ConvGRU hidden=256
  （config 里 `hidden_dim=128` 只是 encoder 中间宽度，rssm_fusion.py:168）。
- 单帧（MotionAlignedRSSMFusion，rssm_fusion.py:644-727）：
  1. deform 对齐：`h_{t-1}, z_{t-1}` 各自经 `ModulatedDeformConv2d`，
     offset 由 `cat[feat, state]` 预测（**纯外观驱动**，:625-643/674-683）；
  2. `h_t = ConvGRU(z_aligned, h_aligned)`（:688-690，无任何运动/位姿输入，
     `action_dim=0`）；
  3. `e_t = encoder(feat)`（2 层 ConvModule，:694-697）；
  4. `μ_p, σ_p = f(h_t)`；`μ_q, σ_q = g([h_t, e_t])`（:692-701）；
  5. `recon = decoder(cat[h_t, z_t])`（:713，目标=当前帧融合 BEV 自身）；
  6. **`output = output_proj(z_t) + feat`**（:716-717，全局加性残差，无门控）。
- MAI（rssm_fusion.py:1865-2044）：`e_hat=obs_prior(h_t)`（仅由
  `0.05·smooth_l1(e_hat, sg(e_t))` 训练）、`innovation = e_t − sg(e_hat)`、
  `μ_q = μ_p + posterior_mu([h_t, innovation])`。其余与基类逐位相同。
- KL（:327-370）：逐元素高斯 KL + free-bits `clamp(min=free_nats)`；
  主线 `free_nats=1.0`。**`clamped_ratio` 是"被钳元素占比"——0.99 即 99% 元素
  贡献常数、梯度为零。**
- 训练接线：BPTT 窗口=1（R4Det.py:1313-1346，前 N−2 帧 no-grad burn-in）；
  推理路径只走 `μ_q`（R4Det.py:1070-1251），prior 从不出现在推理；
  `detach_state=True` 逐帧断图。
- 序列起点：`h_0 = z_0 = 0`（:668-672）；t0 转移输出被实测为逐样本常数
  （me_rssm/docs/10），即每个序列首帧的"状态"是死偏置。

### 2.2 成绩谱系（Overall 3D moderate，ep12-16 窗口均值口径）

| 模型 | 窗口均值 | 备注 |
|---|---|---|
| No-Temporal | 35.59 | 时序净收益 +3.7~4.3（§20） |
| Clean RSSM 主线（Run 10 head-v2，3 seed） | 39.00 ± 0.53 | 峰值 40.42±0.51（§19） |
| FG-FULL N=4（seed0） | 40.02 | §47.1 |
| **no2d_igdr（当前基线，3 seed）** | **40.44 ± 1.35** | seed0 39.28 / seed1 41.92 / seed2 ≈40.12（§48.2）——**seed 间散布 2.6 点** |
| MAI seed0 | 40.12 ± 0.86 | 对同 seed 基线 **+0.84**；峰值 41.42@ep16（§64） |
| MAI seed1 | 39.41 ± 0.81 | 对同 seed 基线 **−2.51**；峰值 42.77@ep6（§65） |
| MAI seed2 | 训练中（ep12 单点 val 41.32） | ep16 后自动停，三 seed 终判（§65） |

类别侧（MAI seed0 ep12-16 vs 同 seed 基线）：Car **+7.09**、Truck **+1.12**、
Ped **−3.40**、Cyc **−1.45**（§64）——增益集中在 Car/Truck，Ped 是主要回归项。

---

## 3. 机制级问题清单

每条按「现象 → 证据 → 后果」给出。P1–P4 是主线问题；P5–P8 是训练信号与
方法论问题；P9–P10 是次级/结构冗余。

### P1 读出通路把 z 退化为公共偏置（本审查的核心新归纳）

- **现象**：对 head 做反事实干预（`tools/test_diagnose_z_utilization.py`），
  零化 z 使 head 输出变化 26–56%，换样本 z 只变化 2.7–8.4%（§62 基线、§64 MAI、
  seed1 探针 w3/w100/w1800 = 4.8/5.1/5.1–8.4%）。z 的因果贡献 ≈ 9:1 的
  公共偏置 : 样本信息。
- **机制**：`output_proj(z_t)+feat` 是全画布共享的加性映射。路线 A 证明单纯放大
  `z_proj` 增益会把偏置与身份**等比**放大（shuffle/zero 比值恒定 0.032–0.046，
  §54）；低维路线进一步证明 `z_proj` 权重停在初始化尺度、检测损失不推大它（§53）。
- **后果**：z 承担"底色"而非"证据"。这解释了为什么 MAI 在状态层把 innovation
  特异占比做到 62–67%，head 响应却只到 0.8–1.4×标准主线（§64）。
  **修好信息流之后，还必须在读出处逐像素决定"信多少"。**

### P2 概率部件失效：KL 被 free-bits 整体短路

- **现象**：主线 `clamped_ratio≈0.99`、`kl_raw≈0.5`、`kl_effective≈1.0`
  （seed2 实时日志、§63/64）。逐元素 free_nats=1.0 下，99% 元素的 KL 梯度为零。
- **代码**：rssm_fusion.py:353-357——`clamp(kl_raw, min=free_nats)` 后取 mean。
  free bits 的本意是"KL 低于阈值时不施加向下压力"，但它同时让低于阈值的元素
  **完全无梯度**；prior 又没有任何其他梯度来源（KL 是它唯一的训练信号，
  MAI 的 obs_pred 训的是 obs_prior 不是 prior_mu），于是 prior 冻结在
  宽而无信息的状态（`σ_p≈0.52` vs `σ_q≈0.23`，μ 几乎重合：`μ_diff²≈0.03–0.04`）。
- **历史佐证**：KL0 单 seed 更好但三 seed 否决（Δ=−1.05，§25）；
  fixed-noise/learnable-std 都滑向确定性（§23/24）。**"采样+KL 共同贡献
  +1.3~1.9"（§20-25）的机制至今未被解释——它更像 warmup 期的噪声正则。**
- **后果**：论文若声称 probabilistic state space，审稿人可以用 clamped_ratio
  一击否决。要么修（第 5 节 E），要么把概率语义迁移到真正被使用的地方
  （可靠性增益），并把现在的失败作为分析节素材。

### P3 predict-correct 不闭环：prior 无推理角色、无降级语义

- 推理恒走 `μ_q`（R4Det.py:1070-1251）；观测退化/缺模态时没有任何
  "退回预测"路径。14 项检查的 #4/#8/#10/#11（me_rssm/docs/02）全部指向这一点。
- 这不是理论洁癖：TJ4D 有连续空帧、雷达点数剧烈波动（docs/08 实测），
  每像素可靠性天然有动态范围，而当前网络对干净像素与退化像素**同权**读出。
- ME-RSSM 的 C2（`z_t = μ_p + g·(z_sel−μ_p)`，g 由模态可靠性驱动）
  正是为此设计；06 P2 又预登记了"g 缺 innovation 输入"——两份设计与 MAI 的
  `delta` 天然拼合（见第 5 节 A）。

### P4 Doppler（唯一直接运动测量）从未进入时序动态

- 证据链：PFN 内两次外观化（me_rssm/docs/01/02）→ ConvGRU 输入只有外观状态 →
  对齐 offset 只由 `cat[feat,state]` 外观预测 → `action_dim=0`。
- Run 13 的 Doppler 动静 mask（−3.44，§13）**不是**反证：它破坏了预训练 PFN
  的特征对齐，属于干预方式错误；ME-RSSM 的无参数旁路（PFN 归一化前取原始
  坐标、7 通道统计、零初始化注入）避开了该失败模式。
- docs/08 的 GT 点级动态分数（Cyclist 0.488 > Car 0.315 > Truck > Ped 0.145）
  给出了"哪里该有增益"的预注册预测——这是现成的论文分析素材。

### P5 序列起点死状态 + 时序信用分配极短

- t0 转移输出逐样本常数（me_rssm/docs/10）；每序列 1/4 帧从零状态起步。
- BPTT 窗口=1：GRU 只通过最后一帧收梯度（Run 14 在旧基座上未获益，
  但 innovation 结构下的梯度流不同，未复测）。
- 现成修复：`state_init='obs'`（me_rssm/docs/07）已实现、sanity 13/13、
  **从未训练**——零风险单变量实验。

### P6 重建目标大部分可由 h_t 预测，对 z 的锚定弱

- §52 实测：当前帧目标相对上一帧 MSE 0.0407 vs 跨样本 0.1691（比值 0.241）；
  低维路线的分解显示重建目标 92% 方差是共享模板（pool 口径）。
- decoder 同时重建"静态背景"浪费容量（14 项检查 #5/#9）。
- 「差分重建」（以帧间残差为目标）在 todo 里已被列为证据最直接的下一步，
  但**只在低维家族试过（方案一，未过门控），全分辨率主线从未执行**。
  注意：低维失败伴随读出/结构多重混淆，不能直接外推为主线结论。

### P7 机制指标与 AP 解耦（本轮研究最重要的方法论发现）

- 三组证据：① 路线 C ep12：机制首次过 2% 门槛、AP 止损 −4.39（§61）；
  ② MAI seed0：机制优势 ep16 回落、AP 却最高（§64）；③ **MAI seed1：
  shuffle_z 4.8–8.4%（不低于 seed0 的 3.9–5.4%）、innovation 特异 61.9%、
  AP 却 −2.51**（本次审查新提取的 seed1 探针数据）。
- 判读：机制读数是必要非充分。它排除"信息没到 z"类失败，但读出结构（P1）、
  优化动力学、seed 敏感度都能独立否决 AP 收益。**预注册门控必须 AP 与机制
  双线并跑（现状正确），不能因机制变好跳过 AP。**

### P8 基线 seed 方差 > 一切单机制效应（硬方法论约束）

- no2d_igdr 三 seed 窗口均值 39.28/41.92/40.12，散布 2.6 点、std 1.35（§48.2）；
  而全部单机制效应在 0.7–1.9 点区间。**任何"大幅提升"必须以三 seed 配对窗口
  为判定口径，且最好把主表做成逐 seed 配对 Δ 而非跨模型绝对值对比。**
- 已有的廉价稳健化手段：checkpoint averaging（§45，ep12+14+16）可提稳；
  18e 完整日程对照（§66 已排队）将澄清"24e 截断"口径问题。

### P9 类别缺口是结构性的，且与 RSSM 改革互相牵制

- Ped strict ≈ 0 是锚点几何物理瓶颈（§29-39 六次闭环），**不再开架构支线**
  （docs/11 明确不做）；Ped loose 是可争取空间，而 MAI 恰好在 Ped 上 −3.4。
- Cyclist 瓶颈在分数排序质量（oracle recall −0.013 vs AP −5.16，§43）；
  CycCls 跨 seed 失败（§44.7）。共享 BEV 改动对 Cyc 是零和红线（§41）。
- **牵制**：任何 RSSM 改动若像 MAI 一样 Ped/Cyc 回退，Overall 增益会被吃掉
  近一半。可靠性门控对 Ped（Doppler 证据最弱类）的行为是合并模型必须
  预注册观察的点。

### P10 结构冗余与模态早融合（二线，记录在案不动手）

- align_h/align_z 双对齐器 ~1.2M 参数重复（06 P3，冻结）；
  ConcatConvFusion 在时序级之前把 256+384 压成 256，模态独有信息不可逆丢失
  （14 项检查 #1）——C4 候选已否决，C1 旁路是既定解法。

---

## 4. 历史路线谱系：已证实资产 vs 已证伪路线

### 4.1 已证实的资产（论文可直接引用的机制事实）

| 事实 | 出处 |
|---|---|
| 时序融合净收益 +3.7~4.3；确定性传播贡献 ~2.4；采样+KL 共同 ~1.3–1.9 | §20-25 |
| posterior collapse（clamped≈1.0, μ_diff² 微小）是稳态；KL0 三 seed 否决 | §22/25/49 |
| z 的 head 因果作用以公共偏置为主（zero 26–56% vs shuffle 2.7–8.4%） | §49/§62/§64 + seed1 探针 |
| 标准主线 posterior 越训越依赖 h_t、越忽略 e_t（h/e 2.23→2.34） | §62 |
| innovation 后验修复信息流：h/e<1、innovation 特异 47–67%、`μ_diff²` 非单调 | §55/62/63 |
| `e_pooled` 的 L2 normalize / `z_prev` normalize 压制样本信息（低维家族定量） | §53/57 |
| 重建目标 76–92% 可被 h_t/模板解释 → 差分重建动机 | §52 |
| t0 死状态定量（转移输出逐样本常数） | me_rssm/docs/10 |
| Doppler 数据事实：ego 主导、v_std 稀疏热点、类别动态分数排序 | docs/08 |
| 反事实探针方法学（posterior_mu hook、specific-energy 分解）+ 58+ 条单测 | tools/ + §52-61 |

### 4.2 已证伪/已止损路线（不要重试）

| 路线 | 结论 | 出处 |
|---|---|---|
| Doppler 动静 mask（改 PFN 特征） | −3.44，干预方式错误 | §13 |
| BPTT>1（旧基座） | 无增益（innovation 基座下未复测） | §14 |
| 0.16m 高分辨率 BEV / Ped 架构系 | 六次闭环全败，物理瓶颈 | §29-39 |
| Truck 回归支线（残差/detach/tower） | 全败，Cyc 零和红线 | §26-28 |
| CycCls / head stem 共享 / 推理 max_num | 跨 seed 失败/未过门控 | §41/42/44 |
| 低维 bottleneck + 排他未来（无重建锚定） | −6.51，结构必然而非调参问题 | §50 |
| 逐 cell 标准化 / 全局去均值作为重建目标 | 3.2%/10.1% 特异占比，无效 | §52 |
| 逐 cell free-bits 之前的「单调 free-bits」 | KL 非瓶颈，反复证实 | §49/53/54/62 |
| KL balancing 单独上（无重建锚定时） | 后置，勿先动 | todo ④ |
| categorical/unimix 全量替换 | 无多峰证据，后置 | todo |

### 4.3 设计完成但从未训练（被搁置的资产）

| 资产 | 状态 |
|---|---|
| **ME-RSSM 三机制**（C1 Doppler 条件化转移 + C2 可靠性门控校正 + C3 动态门） | sanity 13/13、E1-E3 等价性实测、消融 configs 齐备；**零训练** |
| `state_init='obs'`（P1 观测引导初始化） | 已实现、sanity 全绿；零训练 |
| 路线 D（跨样本去塌缩 L_common+L_var）/ 路线 E（KL balancing + 按 cell free-bits） | 设计完成；零训练 |
| 差分重建目标（全分辨率主线版） | 证据充分；仅低维试过 |
| z_proj 高分辨率直连读出 / 几何辅助监督（route C 下一步） | 提议未执行 |

**结论：本项目的瓶颈不是想法供给，而是「验证带宽」——已设计未验证的资产
价值密度高于新想点子。**

---

## 5. 改进方案（按证据强度排序）

### A.（主创新）统一测量条件化 predict-correct：MAI × ME-RSSM 合并

把两条已部分验证的线合并成一个自洽的滤波器，四个环节各对应一个消融开关：

```
Predict   h_t = ConvGRU( align(h_{t-1}; offset + M(e_mot)), z_{t-1} )   ← ME-RSSM C1
          u'  = u + tanh(β)·d·u(1−u)                                    ← ME-RSSM C3
Predict   e_hat_t = obs_prior(h_t)                                       ← MAI（已验证）
Correct   innovation = e_t − sg(e_hat_t)
          delta = posterior([h_t, innovation])                           ← MAI（已验证）
          z_t = μ_p + g ⊙ delta,  g = σ(conv([c_cam, c_mot, |innovation|]))  ← C2 + 06-P2
Readout   output = feat + g ⊙ output_proj(z_t)                           ← 新（见 B）
```

- 为什么合并成立：MAI 的 `delta` 就是 06-P2 里 gain 缺的那个 innovation 输入；
  ME-RSSM 的 gain 语义（观测不可信→降级为预测）恰好补上 MAI 缺的
  "何时信 correct"；C1/C3 解决 P4。初始化上 `g≈0.98`（b=+4）、M/gate 零初始化，
  **开局即 MAI 逐位等价**，训练中被数据推向逐像素差异化——符合本项目
  「零初始化增量 + 等价性证明」的既定工程纪律。
- 消融网格直接继承 ME-RSSM 的四开关（no_motion_offsets / no_reliability_gain /
  no_dynamic_gate / no_modality_obs）+ 新增 no_innovation（退回标准 posterior）
  与 no_gated_readout（退回 MAI）——每个开关对应论文消融表一行。
- 成本：+0.42M 参数 / +44.5G MACs（ME-RSSM 已实测）+ 读出门（可忽略）。
- 风险与缓解：读出门可能重蹈路线 A「等比放大偏置」——缓解：g 的输入是
  reliability 与 |innovation|（特异占比 62–67%），**按构造是样本特异的**，
  不是再放大 z 本身；且先过 450 iter 机制门控（shuffle_z、g 的动态范围、
  Ped/Cyc 早期曲线）再上 24e。

### B.（第二创新）可靠性门控读出——P1 的直接解

- `output = feat + g ⊙ output_proj(z_t)`：干净像素保留观测主导，退化/高
  innovation 像素让状态修正生效。**改动全部在 fusion 模块内**，不触碰历史上
  六连败的检测头。
- 若 A 的完整实现周期长，B 可以先作为 MAI 的单变量增量验证
  （MAI + gated readout，450 iter 机制门控 → ep6 AP 门控）。

### C.（现成零风险）观测引导状态初始化 `state_init='obs'`

单变量、有配置有测试，直接排进下一个 GPU 窗口。预期修复 t0 死状态、
改善序列前段（MAI ep1 明显偏慢：seed0 ep1 13.66 vs 基线 15.26 量级，§60）。

### D.（训练信号修复）全分辨率差分重建

`recon 目标 = 标准化(feat_t − feat_{t-1})` 或「decoder 只吃 correction、
重建帧间残差」（§53 方案一的低维版未过门控，但那次伴随 FiLM 旁路与
16×16 读出混淆）。全分辨率版尚未被检验；若与 A/B 冲突（decoder 输入改变），
作为独立单变量排在 A 之后。

### E.（概率部件：修理或诚实降级，二选一）

- 修理：路线 E（stop-grad 拆分 `L_dyn/L_rep` + 按 cell 聚合 free-bits），
  但**只在 A/B 过门控后做**（todo ④ 的既定纪律）。
- 诚实降级（推荐同步准备）：论文把概率语义放在 gain 上（"逐像素可信度加权的
  predict-correct"），当前高斯 z_t 作为带噪正则保留并在分析节报告
  clamped_ratio 稳态——这与 §20-25 的"采样+KL 有净贡献"实测自洽，
  审稿上无法被 free-bits 证据击穿。

### F.（类别缺口，与 RSSM 解耦但进主表）

- Ped：跟随 MAI 的 Ped −3.4 观察 g 在 Ped 上的取值分布（预期：Doppler 证据
  最弱 → g 低 → 更依赖预测；若 Ped 反而更差，说明预测通路对 Ped 无信息，
  需要在论文里如实归类）。
- Cyc：维持既有结论（排序瓶颈），不开新支线；主表报 loose。
- 主表策略：报 Overall + 逐类别，用逐 seed 配对 Δ 口径（P8）。

---

## 6. 论文故事线（对 docs/11 的更新）

骨架不变（Measurement-Conditioned Recurrent State Spaces for 4D-Radar–Camera
BEV Detection），三条 contribution 修订为：

1. **Innovation-conditioned posterior（全分辨率，已验证）**：
   把 posterior 从"两向量之差"改为 `μ_p + delta([h, e−sg(e_hat)])`，
   实测修复 h/e 依赖方向（2.23→<1）、innovation 特异占比 47–67%
   ——已有 §55/58/62-64 的完整机制证据链。
2. **测量条件化 predict + 可靠性门控 correct/readout（待训练）**：
   Doppler 作为转移控制信号与可靠性证据（ME-RSSM C1/C2/C3）+
   门控读出（B）；与 CRT-Fusion 的差异化叙事不变（docs/09：
   测量 vs 估计、降级语义、无位姿设定）。
3. **机制级分析（素材已齐）**：collapse 稳态、z 公共偏置、机制-AP 解耦、
   t0 死状态、specific-energy 分解方法学；负结果表（4.2）进附录。

风险声明：若 A/B 训练后 AP 无增益，退路仍是 docs/11 预设的
「机制级负结果 + 消融解释」骨架（T-ITS/RA-L 可接受度高于顶会）；
若 AP 有 ≥1.5 点三 seed 配对增益，可冲 CVPR/ICCV 线。

---

## 7. 建议实验序列（不打断当前队列）

当前在跑：MAI seed2（→ep16 自动停）→ 自动接力 fgfull18 seed0（§66）。
建议按以下顺序消费后续 GPU 窗口：

| 步 | 内容 | 判定门 | 成本 |
|---|---|---|---|
| 0 | MAI seed2 ep16 终判（自动） | 三 seed 配对窗口：seed0 +0.84 / seed1 −2.51 / seed2 ? → 决定 innovation posterior 在论文中的表述强度（组件 vs 主线） | 0（等待） |
| 1 | **ME-RSSM seed0**（按 docs/04 原设计，标准 posterior）——隔离 Doppler 三机制的净效应 | 预注册：ep12-16 配对 Δ≥+0.7 且 BEV 同向（STATUS.md 已登记）；机制：stat_gain/conf/dyn | 24e，3 GPU |
| 2 | `state_init='obs'` 单变量（任意基座） | 450 iter 冒烟（t0 探针）→ ep6 AP 门控 | 极低 |
| 3 | **合并模型 A**（MAI + ME-RSSM + gated readout）：实现 + 单测 + 450 iter 机制门控 | shuffle_z ≥2%、g 动态范围非退化、Ped/Cyc 早期不劣化 → ep6 AP 门控（±1 区间续，>2 止损） | 实现约 1-2 天 + 24e |
| 4 | 消融网格（A 的五开关 + stateinit）逐个 450 iter 门控，过线者上 24e | 同上 | 按需 |
| 5 | 胜者三 seed → 主表（逐 seed 配对 Δ + checkpoint averaging 复评） | §18 口径 | 3×24e |

排序理由：先 1 后 3 是为了保留「Doppler 净效应」的干净归因（论文需要
no-motion-offsets 等消融有独立基线）；若 GPU 紧张，1 与 3 可合并为
「直接训合并模型 + 用 no_doppler 开关回溯归因」，但论文消融表会少一层干净对照。

---

## 8. 本次审查的增量发现（此前文档未记录的）

1. **seed1 ep16 机制探针的完整读数**（此前只记录了 AP）：
   shuffle_z head w3/w100/w1800 = 4.8/5.1/5.1–8.4%（不低于 seed0 的 3.9/4.2/5.4%），
   zero_z 41–56%，innovation 特异 61.9%、correction 特异 38.5%、e_pooled 特异 51.1%。
   → **「MAI 机制优势未保持」（§64 的 seed0 结论）在 seed1 上不成立——seed1 机制
   读数良好但 AP −2.51，进一步坐实 P7 解耦，并提示 AP 损失可能来自读出/优化
   而非状态信息。**
2. **P1 的完整表述**（读出门控缺失作为独立问题）：此前文档把「correction 如何被
   检测头放大」（§55）与「head 不依赖 z」（§61）当作两个未解观察；本审查把它们
   统一为读出结构问题，并给出与既有红线（不动 head）兼容的解法（B）。
3. **free-bits 语义澄清**：`clamped_ratio` 的含义是「被钳元素占比=零梯度占比」，
   而非「KL 失效比例」。路线 E 的「按 cell 聚合」设计应把聚合单位从 8192 维
   向量改为空间 cell（~1 nat/cell 沿 channel 求和），使梯度覆盖恢复——
   这一步的预期效果可以直接用现有 stat 口径预注册验证。
4. **MAI ep1 偏慢与 t0 死状态的关联**：MAI seed0/oc 系 ep1 均显著低于基线
   （13.66 vs 15.26，§60），与 t0 死状态 + obs_prior 冷启动一致——
   `state_init='obs'`（C）可能是低成本修复，建议在门控里加 ep1 读数。
