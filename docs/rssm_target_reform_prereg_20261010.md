# 重建目标改革：契约审计与正式预登记（2026-10-10）

> **状态：准备稿，未批准、未实现、未训练、未使用 GPU。**
> 本文按 `docs/rssm_next_step_audit_20261010.md` 的「阶段 0」要求写成：
> 先钉住真实目标/梯度/历史契约，再给出唯一变量与判据。
> **数值门（§4 的阈值）是建议值，须由用户确认后才构成预登记**；在确认前
> 不得据此启动任何训练。DC-RSSM 方向已按 §68.9i 止损，本文与它无关。
>
> **2026-10-10 复核修正（commit `bda2564` 后）**：§1.1 原稿把 burn-in 写成
> 「无」，错误——`rssm_bptt_steps` 未在 config 覆盖，取类签名默认 `1`
> （`R4Det.py:188`）⇒ `burn_in=2`、实际只有最后 1 帧历史 + 当前帧计 recon、
> 等权平均（前 2 帧仅 burn-in、无 loss）。§2 落地规则与 S1 已同步该事实。
> 预登记命令的 checkpoint 路径也已从 `epoch_12.pth`（不存在）改为
> `epoch_14.pth`（基线 best-saved）。§1.2/§1.3/§1.4 日志与源码证据经复核成立。

## 1. 真实目标契约审计（阶段 0.1，全部为源码+日志实测）

被审计对象：标准主线 `no2d_igdr`（`MotionAlignedRSSMFusion`，seed0 24e，
`/data/lurui/work_dirs/fgfull_N4_no2d_igdr_2x4_24e_seed0`），因为它是
用户审计指定的新候选基线。低维家族的契约**不同**，单列在 §1.4。

### 1.1 目标张量

| 项 | 事实 | 证据 |
|---|---|---|
| 目标是什么 | `bev_feats`（`cross_attention` 输出、经 `permute(0,1,3,2)`），即送进 RSSM **之前**的当前帧 BEV 特征 | `R4Det.py:872-875`；`R4Det.py:919` `bev_feats_cache = bev_feats` |
| 分辨率 | 全分辨率（248, 216），**不 pool** | 同上 + `assert bev_feats.shape[2] == self.bev_h_` |
| 标准化 | **无任何标准化**；本家族没有 `_normalized_recon_target`（它只存在于 `LowDimFutureConsistentLatentFusion`，`rssm_fusion.py:1438-1443`） | 全类族 grep |
| detach | **没有 detach**：`F.mse_loss(rssm_recon, bev_feats_cache)` 的第二个参数在计算图内 | `R4Det.py:930`（当前帧）、`R4Det.py:909`（历史帧） |
| 损失权重 | **1.0**（`losses['loss_rssm_recon']` 直接登记，R4Det 无 `loss_weights` 机制） | `R4Det.py:1504` |
| 历史帧 | 窗口内**最后 1 帧历史 + 当前帧**计 recon、等权平均；前 2 帧仅 burn-in、无梯度、无 loss（`rssm_bptt_steps` 默认 `1`、config 未覆盖 ⇒ `burn_in=2`） | `R4Det.py:188`（默认 `1`）→ `1403-1407`（`burn_in=2`）→ `1417-1428`（收集）→ `1493-1504`（平均）；见 `training_runs_full.md` §47 BPTT 表 |
| burn-in | **2 帧**：`rssm_bptt_steps` 未在 config 覆盖，取类签名默认 `1` ⇒ `grad_steps=min(1,3)=1`、`burn_in=2`；只有 t=2（第 3 帧）带 `rssm_detach_state=False` + recon loss，t=0/1 仅前向、不写入 loss | `R4Det.py:188, 1403-1407, 1417-1428` |
| 状态重置 | 无效历史帧的样本 `reset_for_samples(~valid_t)`；基类实现只清 `h_state`/`z_state` | `R4Det.py:1415,1430`；`rssm_fusion.py:299-312` |
| decoder 输入 | `cat([h_t, z_t])`（h 旁路存在）；输出 `output_proj(z_t) + feat` | `rssm_fusion.py:479-481`、`493-495` |
| 配置 | `latent_dim=256, hidden_dim=128, in/out=256, kl_scale=1.0, free_nats=1.0, min_std=0.1, init_std=0.2, seq_len=4` | 运行目录 config 第 211-228 行 |

**结论：真实目标是「全分辨率、原始、未标准化的当前帧 BEV 特征」，
且目标本身带梯度、权重 1.0、窗口四帧等权。**
§69 的 `pool16_globalstd` 是探针候选构造，**不是**该训练目标；
95.5/4.5% 那组数字不能作为「DC 实际优化目标」的证据（用户审计已回填）。

### 1.2 目标项的梯度压力（实测，这是最关键的一组数）

recon 项与 KL 项的逐 epoch 均值（同 run，`*.log.json`，train 记录）：

| epoch | recon MSE | KL | KL clamped 比例 | raw KL 均值 | μ_q−μ_p ² |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.08068 | 0.0000 | 0.5438 | — | 1.526958 |
| 2 | 0.01136 | 0.1407 | 0.9584 | — | 0.342130 |
| 4 | 0.00721 | 0.3125 | 0.9889 | — | 0.089340 |
| 8 | 0.00522 | 0.7066 | 0.9960 | — | 0.026834 |
| 11 | 0.00414 | 1.0027 | 0.9980 | — | 0.012670 |
| 12 | 0.00391 | 1.0024 | 0.9982 | 0.597 | 0.012034 |
| 16 | 0.00322 | 1.0012 | 0.9981 | — | 0.009662 |
| 20 | 0.00277 | 1.0005 | 0.9984 | — | 0.007744 |
| 24 | 0.00263 | 1.0003 | 0.9985 | — | 0.007206 |

末期（ep24 iter350）总损失分解：`loss = 1.45374`，其中
`loss_rssm_kl = 1.00025`（**68.81%**）、`loss_rssm_recon = 0.00256`（**0.18%**）、
`bev_mask_dc_loss_former = 0.17140`（11.79%）、`loss_bbox = 0.12324`、
`loss_cls = 0.05934`、`loss_dir = 0.00393`（0.27%，**比 recon 还大**）。

三条同时成立的机制事实（都在**真实目标**上测得）：

1. **recon 项在总损失里只占 0.18%**（ep24），从 ep4 起就在 0.5% 以下 ——
   它对 z 的锚定压力约等于零。「共享模板」的直觉在真实目标上有**定量对应物**。
2. **KL 被 free-bits 完全吸收**：`free_nats=1.0`，而 raw KL 收敛到 0.597 < 1.0，
   clamped 比例 0.9985，故 KL 项恒等于 1.0（常数，无梯度方向信息）。
3. **后验塌到先验**：`μ_q−μ_p`² 从 1.527（ep1）单调降到 0.0072（ep24，−99.5%），
   即 posterior 不再携带观测特异修正。

⇒ 在真实契约下，「z 不承担观测信息」不是靠探针推断的，而是**损失函数本身
在末期只给了它 0.18% 的说话权**。这解释了为何 MCI/DC/L1 各代都在改状态侧与
读出侧而 AP 不动：**目标项太小，改哪一层都推不动它**。

### 1.3 与「同一几何增强」相关的数据契约

差分目标的头号风险是「窗口内各帧被独立增强，差分被增强差异主导」。实测
**不成立**：`TJ4D_dataset.py:333-335` 先取一个 `seed` 并设进 `np.random`/
`random`，`TJ4D_dataset.py:369-370` 在准备**每一帧**前都把 RNG 重置回同一个
`seed`，因此窗口内各帧走同一条随机数序列 ⇒ **同一套几何增强**。
另：`frame_valid` 由 `image_idx` 连续性决定（`341-350`），不连续的历史帧被
替换为占位符并标记无效（`377-378`），差分监督必须跟随该有效位。

### 1.4 低维家族契约不同（澄清 §53 的历史失败）

`LowDimFutureConsistentLatentFusion` 的契约是另一套：目标是
`_pool_latent(feat)` → 逐样本全局标准化 → **detach**（`rssm_fusion.py:1438-1443`），
decoder 输入可切成只吃 `correction`，且 `latent_loss_is_primary=True`（`:1192`）
走 `loss_rssm_latent` 而不是 `loss_rssm_recon`。

因此 §53.4/53.5 的「帧间残差目标失败」是在
**pool+标准化+detach 目标 + correction-only decoder + latent 损失通路**下得到的，
不能直接外推到「全分辨率、原始、未 detach」的标准家族。
反过来说，标准家族此前**从未**试过差分目标 —— 这正是本次唯一变量的意义。

## 2. 差分目标语义定义（阶段 0.3）

记窗口 `seq_len=4`，顺序 old→new，当前帧为最后一帧；`F_t` 为 `bev_feats`。

```
T_cur_t  = F_t                       # 现役契约（§1.1）
T_del_t  = F_t - F_{t-1}             # 本次唯一变量
```

落地规则（**逐条写在训练之前**）：

1. **帧序与配对**：`F_{t-1}` 取**同一模块实例上一帧 forward 的同一个张量**
   （即上一帧的 `bev_feats`），与 RSSM 的 `h/z` 状态同生命周期，不重新前向。
2. **计损失帧**：`rssm_bptt_steps` 默认 `1` ⇒ `burn_in=2`，实际只有 t=2
   （第 3 帧/最后一个历史帧）与 t=3（当前帧）计 recon，等权平均；
   t=0/1 仅 burn-in、不计差分损失。t=2 无有效 `F_{t-1}`（窗口首帧）的样本
   不计差分损失；历史帧无效的样本按 `reset_for_samples` 语义**清除配对有效位**，
   对齐 `rssm_fusion.py:1418-1433` 已有的 `prev_recon_valid` 模板。
   **禁止**在无有效前缀时强迫拟合「全零差分」。
3. **不改 detach 规则**：目标保持原始梯度契约（不引入 stop-gradient）。
   ⇒ 已知并接受的后果：差分目标的梯度会经由 `F_{t-1}` 回流到**上一帧**的
   cross-attention。原因是「目标 detach 与否」本身就是第二个变量，审计要求
   本阶段单变量；该耦合在消融阶段再单独处理。
4. **尺度约定**：保持原始尺度，**不**改用逐样本标准化或加权分母。
   差分目标的原始能量显著小于当前帧目标（量级见 §3），这会同时压低 recon
   项的绝对幅度 —— 因此**必须**把「recon 项占总损失比例」作为常驻诊断记录，
   而不是事后解释。
5. **几何增强**：沿用 §1.3 的同 seed 机制，不改数据管线。
6. **清空点**：`reset_for_samples()` 必须同时清除配对缓存（新代码，勿漏）。

## 3. 待测的量（阶段 0.2，只读探针，未运行）

`tools/probe_recon_target.py` 已扩展两个**未标准化全分辨率**设计
（`RAW_CANDS`）：`raw_fullres_current`（真实目标）与 `raw_fullres_diff`。
方差拆分与尺度无关，故直接在原始张量上计算；新增打印 `rms / shared rms /
spec rms / diff·current 能量比`，并新增
`--stride`（分散取样）与 `--require-classes N`（要求每类 GT ≥N）以保证
四类都在窗口内 —— 2026-10-09 那个 12 样本窗只有 Car/Cyclist，Ped/Truck
根本没进表。产物另存 `class_coverage` 以便事后核对。

CPU 自检（已跑，无 GPU）：合成张量下 current 的 shared 100%、差分（i.i.d. 噪声）
specific 92.91% ≈ 解析极限 (S−1)/S = 93.75%、**常数差分 specific 恰为 0**
（说明该拆分不是「天然偏向差分」）、全 1 掩码 ≡ 无掩码；选择器在
「每类 ≥3」下正确跳过无 Ped 的帧（3,4,5,6 → 7,14,21,28）。

预登记命令（**需用户批准后才可执行**；只读、不训练、不写 checkpoint）：

```
CUDA_VISIBLE_DEVICES=5 python tools/probe_recon_target.py \
  --config configs/r4det/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr.py \
  --checkpoint /data/lurui/work_dirs/fgfull_N4_no2d_igdr_2x4_24e_seed0/epoch_14.pth \
  --start-index 3 --limit 24 --stride 17 --require-classes 2 \
  --output /data/lurui/work_dirs/fgfull_N4_no2d_igdr_2x4_24e_seed0/recon_target_raw_ep12_w3_24.json
```

要回答的三问（写在这里，避免事后挑解释）：
1. 真实目标（`raw_fullres_current`）的原始 rms 与 shared/specific 拆分是多少？
   它是否与 §1.2 的「recon 只占 0.18%」互为印证（即目标能量本身就低到没压力）？
2. `raw_fullres_diff` 的**原始**特异能量相对 current 是升还是降（标准化版本
   是 6.28× 提升，raw 未测）？
3. 四类都在窗内时，fg / bg_hi / measured 的区域拆分是否改变 §69 的方向性结论。

## 4. 预登记（阶段 0.4；**数值门待用户批准**）

**唯一变量**：重建目标 `T_cur → T_del`（§2）。作用层只有
`loss_rssm_recon` 的**目标张量**：decoder 结构/输入、posterior、KL、
free-bits、读出、`output_proj`、`seq_len`、优化器、日程、数据管线**全部不动**；
不带证据场、不带 L1、不带标准化、不带 detach。

**竞技场**：标准 `no2d_igdr` + `MotionAlignedRSSMFusion`，seed 0，
与基线同日程（24e，峰值 LR、warmup、梯度累积、val_bs=4、checkpoint 每 2 轮）。
基线（同 seed0，固定 ep12-16，n=5）：
Overall **39.2809**、全轮峰值 **40.3853 @ ep14**、
Car strict **47.8399**、Truck strict **29.7246**、
Ped loose **30.9277**、Cyc loose **48.6313**。

**分阶段判据**：

- **S1 接线冒烟（450 iter，不构成 AP 结论）**：目标张量形状/有效位正确、
  只有 t=2/t=3 计差分损失（`burn_in=2`）、t=2 无有效前帧的样本不计、
  梯度有限、`stat_recon_mse` 与 `stat_recon_valid_frac` 合理、
  decoder 确实在学习（recon 有下降）、无 OOM/NaN、显存与基线同量级。
  **不做** AP 裁决、**不**用单样本 shuffle 或方差增大当成功。
- **S2 ep6 检查（决定是否继续）**：
  (a) **反「常数预测」余量**：目标项必须明显优于「样本无关常数预测」这个地板，
      否则它对 z 没有约束。地板 = 目标的共享模板能量（`shared rms`²），
      由 §3 的 0.2 只读测量给出**该目标的实际数值**；因此本项的乘子线
      （例如 recon MSE ≤ 0.5 × 地板）**在 0.2 出数后再定**，不先写死。
      注意：**不能**用「recon 项占总损失比例 ≥1%」当判据 —— 差分目标的原始
      能量本就更小，占比可能更低却不代表更差（见 §5 未知 2）；用占比会
      奖励幅度而非奖励约束力。§53.4 用的正是地板口径（decoder 误差 1.09
      高于恒零预测的 1.00 ⇒ 判为无效），本次沿用同一口径。
  (b) `μ_q−μ_p`² 不低于现役同轮量级（现役 ep4 0.0893 / ep6 0.1003；
      不允许出现更快的塌缩）；
  (c) **h-only vs z-only 重建的条件增益**：固定 `[h,z]` 契约不变，另跑一次
      诊断把 z 置零/换成他样本 z，报告重建误差差 —— 「换目标后仍由 h 承担」
      是本方向最可能的失败模式，必须显式测量，不能只报总 recon 下降。
  未过 (a)/(b) 或 (c) 无差异 ⇒ **止损**，记录负结果并转候选 A。
- **S3 ep12-16 主判据（同 seed 配对窗口，n=5）**：
  - 主门：Overall 窗口均值 **≥ 39.2809**（基线同窗）。
  - 各类非劣（**建议值，待批准**）：Car strict / Truck strict ≥ −1.0；
    Ped loose / Cyc loose ≥ −1.5。若用户要求更严（如四类均 ≥0），按更严执行。
  - 报告口径：Overall、Car/Truck strict、Ped/Cyc loose，另列 Ped strict；
    逐 epoch 曲线、窗口均值、全轮峰值「值+epoch」、best saved 全部保留。
  - **失败即停止该方向**并写机制归因；不做超参扫描、不续 ep24 救活、
    不同时再叠第二条改动。
- **S4 机制门（与 S3 并列，不替代）**：z 的样本特异性必须**同时**在
  ①重建条件增益、②真实任务误差（shuffle/zero 干预对检测输出的影响）
  两处出现；只报方差变大或特征差变大不算通过。

**止损/分叉（承审计阶段 2）**：
- 目标学会、z-only 语义仍差、h-only 重建几乎一样 ⇒ h 旁路/监督职责嫌疑，
  转 A（检测 residual 读出），不在此处继续加码。
- z 有对象语义但真实分类/检测无贡献 ⇒ 优先 A。
- 分类改善但 Ped/Cyc 3D AP 不涨 ⇒ 拆 recall / 中心 / 尺寸 / 朝向 / NMS / score，
  弱/缺测组才触发 C。
- 目标/噪声或几何验证不过 ⇒ 差分候选停止，不进长训。
- 单 seed 无任务级收益或破坏基线对象集中 ⇒ 记录负结果、转 A。

## 5. 明确的边界与未知

- 本文**不是**已批准的预登记：§4 中的非劣界与 S2(a) 的乘子线尚未经用户确认
  （S2(a) 的乘子线还必须等 §3 的 0.2 只读测量出数后才可确定）。
- 本文**未**实现任何模型改动；未训练、未碰 GPU、未产生性能宣称。
- 未知 1：差分目标的**原始**能量尺度与特异占比（§3 要测）。
- 未知 2：差分目标下 recon 项绝对幅度会更小；若因此更没压力，则与
  「给 z 更多责任」的意图相反 —— 故 S2(a) 用「相对常数预测地板」而不是
  「占总损失比例」判定，且地板数值必须来自 0.2 的实测。
- 未知 3：不 detach 使差分把相邻帧特征耦合起来，短期可能通过「让相邻帧更像」
  降损，而不是让 z 学到东西；S2(c) 的 h/z 条件增益是探测该退化的入口。
- 历史边界：帧差分重建是**基础修复/假设验证**，不自动构成论文主贡献
  （审计 §5）；论文贡献仍须落在「检测接地的创新状态学习 + 对象语义与条件
  任务价值」上。
