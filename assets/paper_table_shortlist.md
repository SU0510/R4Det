# 论文表格选表建议（修订版）

> 修订原因：上一版说「Run 1–8 不能用」是**过头了**。记录是够的（有无 checkpoint 不影响），
> 真正的限制是**口径不可比**，不是数据缺失。见第三节。

## 零、口径与角色（按你的定义）

- **本文方法 = FG-FULL 去掉 2D instance 监督 + IGDR**（config `..._no2d_igdr.py`），不是完整 FG-FULL。
- **均值（ep12-16 窗口）= 实验室内选型对比**，用来判断哪个变体更好。
- **BEST = 论文主表 / 汇报的最终结果**。
- 因此下表的「论文列」用 BEST，「内部选型列」用窗口均值。

## 一、当前两个 run 的实时状态（今天 16:00 查）

| run | 身份 | 状态 | 最新 |
|---|---|---|---|
| `fgfull_N4_no2d_igdr_2x4_24e_seed0` | **本文方法** | **训练中**，只到 ep4 | Overall 32.851 @ep4 |
| `fgfull_N4_temporal_baseline_seed0` | GRU 时序控制（补 U3） | **已死**：14:28 启动 → 14:46 停在 ep1 [900/1902]，**0 个 val epoch**，无 traceback，进程已不在 | — |

- 方法 run 还没跑完 → **论文主表现在无法定稿**，BEST 至少要等 ep12-16 窗口过去。
- 控制 run 疑似被抢卡（两个 job 都指向 GPU 5,6,7；现在 6/7 被方法 run 占满 100%），
  日志无报错，属外部中断，需换空闲卡重跑。
- 文档 `docs/*.md` 里**完全没有**这两个 run 的内容（§46 停在「not launched here」），跑完需要补新章节。

## 二、控制 run 是否配得上当对照？——配得上（已核对）

repo 里 `..._temporal_baseline.py` 与 `..._no2d_igdr.py` 逐行 diff，**只差 4 项**：

| 差异 | 值 | 说明 |
|---|---|---|
| `model.temporal_fusion` | `MotionAlignedRSSMFusion` → `TemporalDeformableFusionBaseline` | 唯一实质变量 |
| `rssm_bptt_steps` | 1 → **0** | 保留原 GRU baseline 的「单步 + detach 历史」语义 |
| `custom_hooks` | `KLScaleSchedulerHook` → `[]` | GRU 无 KL，必须移除 |
| `find_unused_parameters` | → `False` | GRU 分支没有 unused param |

两者**同栈**：都 `img_rpn_head=None / img_roi_head=None`、都无 `igdr_fusion`、msk2d/props/depth/rangeview 一致。
→ 这是干净的单变量对照，可用。

⚠️ 一个需要在表注披露的不对称：`bptt=0` 意味着 GRU 的历史回路在 `no_grad` 下，
而 RSSM 走 BPTT=1（历史有梯度）。这是有意的语义保留，但对方法略有利，审稿人可能问。

## 三、Run 1–8 到底能不能用？——能用，但必须单独一块

### 现存的记录（够填一行完整数据）

| 记录 | 内容 | 位置 |
|---|---|---|
| 主指标汇总 | Runs 1–5：BEST/LAST 的 3D easy/mod/hard + BEV_mod | §1；`/data/lurui/work_dirs/training_comparison.md` §1 |
| 逐 epoch 曲线 | Runs 1–5 全 18 epoch 的 Overall 3D_moderate | §2；`training_comparison.md` §2 |
| 四类 @BEST | Runs 1–5 四类 **loose** + Overall 3D/BEV | §3；`training_comparison.md` §3 |
| 四类 @BEST（strict+loose） | baseline_temporal + Run 6/7/8 + N4hdim64 + N2v3 的 **8+1 列** | §7「逐类别 BEST 对比 (loose/strict)」 |
| 逐 epoch（N3/N4 家族） | Run 6/7/8 + N4hdim64 + N2v3 + baseline_temporal | §7「N=3 和 N=4 epoch-by-epoch」 |
| 排名 + 结论 | 9 行 3D_mod 排名 + 9 条核心发现 | §8 |
| 配置/超参/commit | 三个融合模块差异、KL hook、git log | §4/§5；`training_comparison.md` §4 |
| 独立诊断 | KL 压死、velocity 恒零、零初始化 | `work_dirs/rssm_diagnosis.md`（08-05） |

**结论：baseline_temporal 那一行可以完整填出来**——Car_s 32.82 / Car_l 52.18 / Trk_s 29.91 / Trk_l 43.02 /
Ped_s 0.04 / Ped_l 26.64 / Cyc_s 22.52 / Cyc_l 48.63 / Overall 3D 34.50 / Overall BEV 41.68。
Run 6–8 同表也有 strict+loose。所以「记录不足」不成立。

### 真正不能用它们的地方（三条，与 checkpoint 无关）

1. **栈不同，不能与 24e 主线同栏算 Δ**：18e、无预训练、无 head-v2、Ped 还是 1-anchor 时代、
   lr 2e-4、spg=4。把它和 39~40 的 24e 行放一起，读者会以为那是「时序模块的对照」，实际差的是整个栈。
2. **记录本身有已知缺口/冲突**：baseline_temporal **ep1–5 无记录**（曲线从 ep6 起）；
   Run 6 ep10 暴跌到 28.05、Run 7/Run 8 ep18 暴跌，原因无日志可查；
   **Run 2 的超参与现存 config 不符**（报告 U1）；部分 run 的 GPU 数/batch 口径不明（U2）。
3. **方法论上不合格入最终表**：全部单次、非 deterministic。报告 §18 定稿规则写明
   「非 deterministic 的历史 run 只作先导数据，**不进入最终口径表**」。

### 建议的用法

单独一张 **「早期 18e 配置阶段（单次、非 deterministic）」** 表，或放附录，标注栈不同、不跨栏算 Δ。

⚠️ 但要清楚它的叙事风险：这 8 行显示 **18e 时代没有任何 RSSM 变体超过 GRU baseline 34.50**
（RSSM 最好 34.27）。单独列出来会强化「时序模块没赢过简单 GRU」的印象 —— 这正是必须在**最终栈**上
补 GRU 控制（U3）的原因：只有在同栈下证明 RSSM > GRU，这 8 行才变成「早期栈下 RSSM 还没调好」的铺垫，
而不是反证。

## 四、值得进表的 run（对方法=no2d_igdr 栈）

### 主表（BEST 口径，最终结果）

同栈（`no2d_igdr`：无 2D instance / 无 IGDR，24e，pretrained，head-v2）的行：

| 行 | 来源 | 状态 |
|---|---|---|
| No-Temporal 单帧 | `no_temporal_N4_2x4_24e_seed0` | ✅ BEST 35.59 |
| GRU 时序控制 | `fgfull_N4_temporal_baseline_seed0` | ⏳ 待重跑（配得上当对照，见第二节） |
| **Ours：FG-FULL(−2D inst, −IGDR) + N4 Motion-Aligned RSSM** | `fgfull_N4_no2d_igdr_2x4_24e_seed0` | ⏳ 训练中 ep4 |
| 完整 FG-FULL（含 2D inst + IGDR） | `fgfull_N4_2x4_24e_seed0` | ✅ BEST 40.69 —— 作「保留 2D inst/IGDR」的消融行 |

注：`no_temporal` 与完整 `fgfull` 都不是 `no2d_igdr` 栈（前者无时序也无 fg 监督，后者含 2D inst+IGDR）。
严格同栈的行目前**只有方法 run 自己** → 这也是为什么控制 run 必须按第二节的配置重跑。

### 消融表（内部选型用窗口均值，论文用 BEST）

机制消融（seed0）：Deterministic latent 37.08 / FixedNoise 36.35 / Posterior-only 37.04 / KL=0（多 seed 不稳）。
负结果（附录或一段文字）：anchor 变体、Doppler mask、BPTT、Truck 三轮、Ped 五轮、结构门控。

### 不要进表

被 bug 作废的（§36 IoU 编码、§38.1-5 转置布局）、smoke/probe/只读诊断、
截断未跑完的（shared_stem/crossmodal 16ep、fgfull_N3 1ep、cyccls seed2）、
KL=0 seed1 的 41.10（单点不可信，只作反例）。

## 五、`baseline_temporal`（Run 1, GRU）能和谁对照

Run 1 的栈：**head-v1、无预训练、18e、lr 2e-4、spg4、Ped 1 anchor**；BEST **34.50 @ep12**。

### 5.1 完全可比 —— 同栈，唯一变量是 `temporal_fusion`

Run 2 / 3 / 4 / 5（《训练记录》§4 明写：这 5 次 run「骨干、检测头、训练 schedule 完全相同，唯一差异是
`temporal_fusion` 模块及其超参」），以及 Run 6/7/8 + N4hdim64（同 18e 栈，变量是 seq_len/hdim）。

- **最干净的一对：Run 1 (GRU 34.50) ↔ Run 5 (v3 RSSM 33.77)** —— 都从头训、同 lr/spg/head/预训练状态，
  → **18e 时代 RSSM 没赢过 GRU（−0.73）**。这是这 8 行的真实叙事，也正是必须在最终栈上补同栈 GRU
  控制（报告 U3）的原因：只有同栈下证明 RSSM > GRU，这 8 行才不构成反证。
- 使用注意：Run 4（34.01）含续训，+3.12 归因不干净；Run 3（30.89）是踩坑跑；
  Run 2（33.54）现存 config 的超参与当时不符（报告 U1）。

### 5.2 勉强可比 —— 同栈但多一个变量

**`rssm_N4_2x4_30e` 无预训练 34.71 @ep22**。已核验该 config 为 **head-v1**
（`ignore_dir_classes` 与 `anchor_class_mapping` 计数均为 0）、无预训练、同 lr/spg；
相对 Run 1 只有两处差异：① 18e → 30e，② GRU → RSSM。
34.71 vs 34.50 几乎打平，但两处差异方向相反，**不能反推「30e 的 RSSM ≈ 18e 的 GRU」**。
`N4 30e lr2e-4`（30.08）还多改了 lr = 三个变量，不要用。

### 5.3 绝不能对照 —— 栈差会被误读成方法差

凡是 **pretrained**（+3.23）或 **head-v2**（Overall ≈ +1.7~2.7、BEV ≈ +5.5）的 run：
Run 9（37.94）、Run 10 / §19 三 seed（40.42±0.51）、No-Temporal（35.59）、§21–25 全部机制消融。

> **特别警告 `no_temporal_N4_2x4_24e_seed0`（35.59）**：它与 34.50 只差 1.09，看起来是天造地设的
> 「单帧 vs GRU」对照，其实是巧合 —— 它的栈多出预训练（+3.23）与 head-v2（≈+2），同时去掉时序融合
> （−4.29），三项近似抵消。用 35.59 − 34.50 会得出「时序融合只值 +1」的错误结论；真实值应为 §8.5 的
> 同栈对照 **+4.29 BEST / +3.7 窗口**。

### 5.4 必须披露的残留差异

Run 1 的 config 里仍留着数据管线字段差异（`depth2img` / `gt_depths` / `LoadAnnotations3D`，
见《训练记录》§4 脚注）；且报告 §7 U2 未逐一核对早期 run 的 GPU 数/有效 batch。
→ Run 1 ↔ Run 2–5 的对照可用，但表述应写「同 schedule、同 lr，卡数/batch 口径未逐一核验」，
而非「完全受控」。

## 六、`no_temporal_N4_2x4_24e`（单帧对照）能和谁对照

栈：pretrained + head-v2 + 24e + 3 卡 batch12 + deterministic **seed0**，且是**非 FG 栈**（无 fg 监督）。
盘上只有一次（`no_temporal_N4_2x4_24e_seed0`，无 seed1/2）。
数字已从原始 `*.log.json` 重算：**BEST 35.59 @ep20，ep12-16 窗口均值 34.65**。

### 6.1 严格同栈对照 —— 只有 `run10_headv2_multiseed/seed_0`

diff 两份 config，差异只有 `temporal_fusion`（None ↔ `MotionAlignedRSSMFusion`），其余全是记账项
（`custom_imports` 与 `KLScaleSchedulerHook`，在 `temporal_fusion=None` 下不生效；`figures_path`/`work_dir`；
`checkpoint_config interval=1↔2`；hook 列表顺序）。→ **干净的单变量对照**。

| | BEST | ep12-16 窗口 |
|---|---:|---:|
| no_temporal（seed0） | 35.59 @ep20 | 34.65 |
| RSSM（seed_0） | 39.88 @ep16 | 38.39 |
| Δ（时序融合净收益） | **+4.29** | **+3.74** |

若以三 seed 为分母（BEST 40.42±0.51；窗口均值 39.00）：Δ = **+4.83 BEST / +4.35 窗口**。
但 no_temporal 只有 seed0 一次，正式表述应写「no_temporal(seed0) vs RSSM(seed0)」，跨 seed 只作范围。

### 6.2 同表并列（可进同一张消融表）—— §21–25 机制消融

`deterministic_latent` / `rssm_kl0` / `rssm_fixednoise_posterior` / `posterior_only_learnable_std`。
已 diff 核验：它们与 seed_0 的差异同样只在 fusion 模块定义（kl0 仅改 `kl_scale`），**共享同一 38.39 参考线**。
→ no_temporal 是这张表的「下界行」，可并列；但它们的 Δ 都相对同一参考线，**不能彼此相减**。

### 6.3 不能对照

- `fgfull_N4_2x4_24e_seed0`（BEST 40.69 @ep16）及整条 FG-FULL 线：diff 显示 fgfull 相对 seed_0
  一次性多出 `loss_bev_seg`/`loss_range_seg`、depth/props supervision、msk2d(abs/sam2)、
  MRF3Net rangeview、FRPN proposal、`shared_stem`、img_rpn/img_roi —— 十余个变量，绝不能相减。
- `fgfull_N4_temporal_baseline_seed0`（在跑）：它是 **FG 栈上的 GRU 控制**，不是 no_temporal 的同栈替代。
- 18e 的 Run 1–8：栈差（无预训练 / head-v1 / 18e）远超待测效应。

### 6.4 必须知道的缺口

no_temporal 属于**非 FG 栈**，其 +4.29 只在 run10/head-v2 那一代成立。要放进最终 `no2d_igdr`
论文主表，严格说需要 **FG 栈上的单帧 run**（`fgfull_N4_..._no2d_igdr` + `temporal_fusion=None`），
而**盘上不存在该组合**（已 find 全部 24e config 核验）。
→ 在跑的 `fgfull_N4_temporal_baseline_seed0` 补的是 **GRU 那一格**；单帧那一格仍然空缺。

## 七、对照对总表（哪些能配成对）

判断标准：**同栈 + 单变量**才算一对。每对给出被隔离的变量、Δ 与证据等级。

### A. 18e 一代（head-v1、无预训练、lr 2e-4、spg4）—— §4 已证同栈

| # | 对 | 隔离变量 | Δ Overall 3D mod | 证据 |
|---|---|---|---|---|
| A1 | Run 5 RSSM v3 33.77 ↔ Run 1 GRU 34.50 | RSSM ↔ GRU | **−0.73**（18e 下 RSSM 未赢） | §4 同栈声明；【记录口径】 |
| A2 | Run 6 (N3 hdim64) 34.27 ↔ Run 5 (N2 hdim64) 33.77 | 帧数 N2→N3 | **+0.50** | 33.77+0.50=34.27 精确自洽 → §8.2 的 N2 参考线就是 Run 5 |
| A3 | Run 8 (N3 hdim128) 32.87 ↔ Run 6 (N3 hdim64) 34.27 | hdim 64→128 | **−1.40** | 同上，算术自洽 |
| A4 | `rssm_N4_2x4_30e` 34.71 ↔ Run 9（预训练）37.94 | **预训练** | **+3.23** | config diff 仅 `load_from` ✅ |
| — | Run 7 (N4 hdim128) 34.05 / N4 hdim64 33.00 | N 与 hdim 同时变 | 非单变量 | 仅作 A4 的同代旁证 |

A4 是全项目证据最硬的一对（唯一差异：`load_from=None` ↔ `checkpoints/pretrained_tj4d.pth`）。
⚠️ A1–A3 属【记录口径】（work_dirs 已删，只剩汇总记录），不能靠 checkpoint 复评。

### B. 24e / head-v2 一代（pretrained + head-v2，非 FG 栈）

参考线 `run10_headv2_multiseed/seed_0`：BEST 39.88 @ep16 / 窗口 38.39。下 5 行均与之 diff 核验为单变量；
**它们是一族并列行，彼此不能相减**（都与同一参考线比）。

| 对 | 隔离变量 | Δ BEST | Δ 窗口 ep12-16 |
|---|---|---:|---:|
| no_temporal 35.59 ↔ seed_0 | `temporal_fusion=None` | **−4.29** | **−3.74** |
| deterministic_latent 38.44 ↔ seed_0 | 删随机性 | −1.44 | −1.31 |
| fixednoise 38.42 ↔ seed_0 | 固定 0.1 噪声 | −1.46 | −2.04 |
| posterior_only 39.66@ep10 ↔ seed_0 | 删 prior、保留可学 std | −0.22 | −1.35 |
| kl0 40.35 ↔ seed_0 | 关 KL 梯度 | **+0.47** | +0.67 |

kl0 单点为正，但三 seed 翻负（37.94 vs 39.00，Δ −1.05）→ 不成立，只作反例。

### C. 最终栈 `no2d_igdr`（FG-FULL −2D inst −IGDR，24e，pretrained，head-v2）

| 对 | 隔离变量 | 状态 |
|---|---|---|
| `fgfull_N4_temporal_baseline_seed0`（GRU 控制）↔ `fgfull_N4_no2d_igdr_2x4_24e_seed0`（方法） | RSSM ↔ GRU | ⏳ **未出数**；config diff 仅 4 项、同栈 ✅（即报告 U3 的对照） |
| `fgfull_N4_no2d_igdr` ↔ `fgfull_N4_2x4_24e_seed0` 40.69 | 去掉 2D instance + IGDR | ⏳ 方法 run 训练中（ep5） |
| — | **单帧**（`no2d_igdr` + `temporal_fusion=None`） | ❌ **该 run 不存在**，此格空缺 |

### D. 勉强 / 不可用

| 对 | 障碍 |
|---|---|
| Run 1 GRU 34.50 ↔ `rssm_N4_2x4_30e` 34.71 | 两处差异（18e→30e、GRU→RSSM），方向相反 |
| 任何 pretrained / head-v2 run ↔ 18e 一代 | 栈差 +3.23（预训练）或 +1.7~2.7（head-v2），远超效应 |
| `no_temporal` 35.59 ↔ Run 1 34.50 | 假对照：预训练(+3.23) + head-v2(≈+2) − 时序(−4.29) 近似抵消 |
| FG-FULL 40.69 ↔ 非 FG run | 十余个变量（见 §六 6.3） |

**一句话**：能立住的对照共 **10 对**（A1–A4 + B 的 5 行 + 勉强 1 对）。其中证据最硬的只有
**A4 预训练 +3.23** 与 **B 的 no_temporal 时序融合 −4.29**；最终栈上那两对（U3 的 GRU 控制、
2D inst+IGDR 消融）**目前一对都还没有数**。
