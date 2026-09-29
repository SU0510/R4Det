# TODO

## 训练补充

1. 给 `fgfull`（完整 FG-FULL N=4）和 `temporal baseline` 补 seed1/seed2，以支撑消融的配对比较（未开始，需要 GPU 5/6/7 空闲窗口）。
   - 起因：no2d_igdr 三 seed 的 ep12-16 标准差 1.35，与 47.2 节单 seed 得到的 0.74 效应量同量级，配对比较目前的样本量不足以支撑结论。
2. 清理 `no2d_igdr` N4 RSSM seed2 的中间权重（未执行）。
   - 目录 `fgfull_N4_no2d_igdr_2x4_24e_seed2` 保存了 ep2-24 全部偶数 epoch，按 17.5 规则应只留 best saved（ep14）与 last（ep24），其余 10 个可删，约释放 6.4 GiB。
   - 该 run 在 2026-09-24 那轮清理时仍在训练，被整体排除，需并入下一轮。

## 训练效率

1. 加快每个 epoch 的 validation（当前约 61 min/epoch，其中 val 前向 ~10 min / IoU+eval ~1 min / ckpt ~1.6 min）。
   - **val batch size 已做（2026-09-26 完成）**：`R4Det.forward_test`/`simple_test`/`simple_test_pts`/
     `preprocessing_information` 的 batch 支持在 `a18f966` 落地，`data.val.samples_per_gpu` 提到 4（`b250e0d`）。
     完整 2040 样本 A/B：峰值 2.36/3.11/5.92 GiB、Overall 3D mod 38.4516/38.4947/38.5420；
     速度用交错测量 bs=1/2/4 → 271.8/238.6/230.9 ms/sample（1.14x/1.18x）。
     退回产生 ep16 的修订（`c0c53ba`）后同 harness 得到逐位相同的
     3D/BEV 数字，且无 batch 间串扰、DDP per-rank 顺序不变。详见 `docs/training_runs_full.md` 第 51 节。
     **结论：等效但收益有限**（纯前向 1.06x、端到端 workers=2 时 1.18x）。bs=6 OOM，故 4 为上限。
   - **真正的瓶颈是 dataloader，不是 batch size**：loader 单独耗时 workers=2/4/8 =
     111.2/59.9/34.1 ms/sample，占端到端约一半。下一步优先调 `workers_per_gpu` 2→4（约省 51 ms/sample），
     并注意 worker 增加会与训练侧抢 CPU（实测 workers=8 端到端反而回落到 1.14x）。
   - 零风险项（仍未执行）：把 `evaluation.interval` 从 1 改成 2（与 `checkpoint_config.interval=2` 对齐）。
     奇数 epoch 的 val 本来就没有落盘权重、进不了 best saved，改动后每个保留点的数值完全不变，
     只是 ep12-16 窗口从 5 点变 3 点。当前 run 仍在按 interval=1 跑。
   - 附注：离线 eval harness 比训练内 EvalHook 低约 2.2 个 3D 点（2D 完全一致），原因未追，见第 51.5 节。
     横向对照请统一只用一种来源。
   - 口径约束：若启用，seed1 与 seed2 会跑在不同代码/config 版本上；最稳做法是 seed1 跑完后、seed2 起跑前改，并在 `training_runs_full.md` 记录该差异。

## 方法探索

1. 探索原 DreamerV3 的 BlockGRU scaling 方法和超参数使用方法。
2. 解决 `z_t` 失效：冻结 checkpoint 因果诊断已于 2026-09-25 完成，详见 `docs/training_runs_full.md` 第 49 节；继续调 Gaussian、KL/free_nats、fixed noise 或 learnable std 都不会解决。
   - 因果诊断结论：`zero_z` 在 no2d_igdr seed0 ep14 上使 BEV 特征平均变化 82.53%、head 输出平均变化 38.02%，所以 z 不是完全无因果作用；但 `shuffle_z` 只使 head 平均变化 4.67%，prior `mu_p` 只使 head 平均变化 5.87%，clean 主线也得到同构结果。这说明 z 主要是公共偏置/底色，缺少样本级判别信息。
   - **进度（2026-09-26）**：低维 bottleneck + 排他未来任务已实现，但首轮正式训练因 detector 解包接线错误（把 `z_t` 当 loss、丢弃真正的 KL+future 目标）而无效，ep1-4 全部作废，详见 `docs/training_runs_full.md` 第 50 节。接线、未来任务方向、KL 二次归一化、mask 广播、`stat_*loss` 命名共 5 处已修复（commit `f5d900f`），29 个单测 + 450 iter 冒烟通过，随后重启 24e 正式训练（结果见下条）。
   - **结果（2026-09-26，负结果）**：低维 bottleneck（16×16×32）+ 排他未来任务训练到 ep12 后按预设止损终止。ep8-12（n=5，实际跨度，窗口不完整）Overall 3D moderate `33.3887`，低于 no2d_igdr 三 seed 同窗口 `39.8979 ± 0.9554` 达 **6.51 点**；全轮峰值 `35.6352 @ ep9`（该 epoch 无对应权重），best saved `34.4123 @ ep6`。KL 被 free-bits 完全截断（`clamped_ratio → 0.980`），`mu_diff²` 腰斩（0.0256→0.0024），future MSE ep9 后平台化。完整曲线、逐类别拆解与归因见 `docs/training_runs_full.md` 第 50.6 节。**该配置判为负结果，不再继续训练。**
   - **归因（2026-09-26）**：三条已核实——① 本变体删掉了 decoder/重建项（`rssm_fusion.py:1008` 起的类没有 `self.decoder`），而重建是标准 RSSM 里把 `z_t` 锚定到当前观测的唯一梯度源；② 先验/后验被改成只看 `z_prev`（`prior_mu(z_prev)`，标准应为 `h_t`），predict-correct 结构被破坏；③ 输出经 FiLM 门控旁路（`gate_init_bias=-1.0`），z 更易被绕开。对照证据：标准 `MotionAlignedRSSMFusion` 的 KL 同样长期被钳（`clamped_ratio≈0.998`、`loss_rssm_kl≈1.00`），但它靠 `loss_rssm_recon≈0.0037` 锚定 z，指标可用。**结论：KL/free-bits 不是首要病因，优先级下调。**
   - **反事实诊断（2026-09-26，工具已修正）**：旧工具 hook `output_proj`/`decoder`，只覆盖 z 的两条下游分支（漏掉 FiLM/gate 与写回 `z_state`），且本变体没有 `decoder` 属性、会直接 `AttributeError`。已改为 hook `posterior_mu` 输出（确定性重放下 `z_t = mu_q`），一处改写即覆盖全部下游；新增 10 条单测（`tools/test_diagnose_z_utilization.py`），断言 `out[4]` 等于干预后的 z、lowdim/标准 RSSM 都能跑、normal 重复逐位一致、shuffle 来源不得是自己。commit `dd2fb75`。
   - **ep12 重测结果（修正探针，旧结构）**：三个样本窗各 4 条均为「zero 高、shuffle/prior 极低」——`[3,4,5,6]` head 比值 zero `0.0214` / shuffle `0.000097` / prior `0.00065`；`[100..103]` `0.0196` / `0.000052` / `0.00064`；`[1800..1803]` `0.0249` / `0.000088` / `0.00066`。即 z 的因果作用几乎完全是**与样本无关的公共偏置**。修正探针在标准 RSSM（no2d_igdr seed0 ep14）上给出 `0.3802` / `0.0467` / `0.0587`，与第 49.2 节旧数字逐位一致，确认是严格超集而非换口径。
   - **结构修复（2026-09-26，commit `dd2fb75`）**：按标准 RSSM 一次性落地两项，不做残缺中间实验——① prior/posterior 条件从 `z_prev` 改回 `h_t`（`p(z_t|h_t)`、`q(z_t|h_t,e_t)`）；② 加回观测似然 `decoder(cat[h_t,z_t]) -> pooled 且逐样本标准化的当前帧 BEV`，`recon_loss_weight=0.1`，与 KL、future 一起折进 index 2（index 1 保持 `None`，不重复计账）。新增 `stat_recon_mse`。RSSM 全量 31 条单测通过；100 iter 与 450 iter 冒烟通过（recon 1.1714→0.6420，future 1.0388→0.9239 同步下降）。
   - **结果（2026-09-27，负结果）**：`lowdim_z_recon_ht_3x2x2_24e_seed0` 跑到 ep13 验证完成后按**预先登记的 AP 止损**终止（ep8-12 五轮均值比基线低 2 点即停）。ep8-12（n=5，窗口完整）Overall 3D moderate `32.9976`，低于 no2d_igdr 三 seed 同窗口 `39.8979 ± 0.9554` 达 **6.90 点**（上一版为 −6.51，差距反而扩大）；全轮峰值 `34.7784 @ ep13`（无对应权重），best saved `33.6799 @ ep12`。逐类别 ep8-12 3D strict：Car `41.7478`（−9.44）、Truck `18.7115`（−11.17）、Cyclist `23.8772`（+0.81）、Pedestrian `0.0985`（−0.12）。完整曲线、逐类别拆解与归因见 `docs/training_runs_full.md` 第 52 节。
   - **机制门控逐条（未通过）**：`shuffle_z` head 仅 **0.000340**（要求 ≥0.10，差约 294 倍；上一版 0.000097）；`prior_only` head 0.001106（上一版 0.00065，仅 1.7 倍）；recon 0.157→0.163 基本持平；raw KL 有界（0.032-0.371，通过）；`mu_diff²` 0.02816→0.00324 仍单调塌缩。三窗结果一致（`[3,4,5,6]` / `[100..103]` / `[1800..1803]`）。
   - **失败机制（已定量定位）**：重建项**确实被使用**——零化 z 让重建 MSE 从 0.19-0.20 涨到 2.96-3.00（+1385%~+1470%）；但**换用另一个样本的 z，重建损失只变化 −0.09%~+9.07%**。用缓存 BEV 特征分解重建目标：当前实现（pool16 + 逐样本全局标准化）**92.1% 的方差是所有样本共享的模板，仅 7.9% 样本特异**；`pool32` 为 11.2%、全分辨率为 18.6%、逐 cell 标准化只有 3.2%（更差）、按留出集去均值只到 10.1%。另有独立证据：当前帧目标相对上一帧的 MSE 仅 0.0407，而跨样本 MSE 为 0.1691（比值 0.241），即 h_t 光凭上一帧就能解释掉相当一部分目标。**结论：存在一个「样本无关的 z」就能同时满足重建与 future 两项，因此 `shuffle_z` 无感。**
   - 下一步（按代价从低到高）：① 换掉重建目标——改用 `pool32`/全分辨率，或**直接以「与上一帧的差异」为重建目标**（差分重建，证据最直接：被 h_t 解释掉的部分不该由 z 承担）；**不要再试逐 cell 标准化（3.2%）或全局去均值（10.1%），已证无效**。② 给 z 一个当前观测独占且 h_t 无法预测的任务（box-centric/occupancy 监督，或让 future 任务也走差分层）。③ 若前两步仍不改善 shuffle_z，考虑放弃「低维 bottleneck + FiLM/gate 旁路」这一组合，回到输出直接吃 `(h_t,z_t)` 的标准形态。④ **KL balancing/free-bits 仍然后置**：本轮 raw KL 一直有界（≤0.37），已再次证明它不是瓶颈。
   - **方案一（2026-09-27，已完成，门控未通过）**：`correction_t = mu_q − stopgrad(mu_p)`，decoder 只吃 `correction_t`，重建目标改为标准化帧间 BEV 残差，future loss 关闭，correction 直接送检测残差。配置 `..._lowdim_z_delta.py`（commit `12cfc65`）；RSSM 单测 33 条通过。450 iter 冒烟（`/tmp/lowdim_delta450.log`）：recon `1.2204→1.0851`（**−11.1%**，要求 ≥15%）、correction² `0.0155→0.0039`（要求 ≥0.005，且后 150 iter 仍在下滑）、clamped `0.896`（通过）。**未过门控。**
   - **方案二（2026-09-27，已完成，门控未通过）**：加 batch-roll 排名损失（`discr_loss_weight=0.1`、`margin=0.2`，负样本 `roll(1)`，batch<2 跳过），配置 `..._lowdim_z_delta_discr.py`（commit `3d8aefa`），单测 35 条通过。450 iter：`neg−pos` gap 仅 **0.0111**（要求 ≥0.1，差约 9 倍）且最后 50 iter 停滞；correction² 仍滑到 0.0039；recon −10.8%（与方案一几乎相同）。**关键证据：recon `1.09` 已高于「恒输出零」的 `1.00`，`pos`/`neg` 只差 1%，说明 decoder 输出几乎不依赖收到哪个 correction。未过门控。**
   - **方案三（2026-09-27，已完成，门控未通过）**：删 global pool + FiLM/gate，改 `output = feat + h_proj(h_t) + z_proj(correction_t)`（均 `Conv2d(32,256,1)`，16×16 投影后双线性上采样）。`LowDimFutureConsistentLatentFusion` 新增 `readout_mode`（`'film'` 默认、`'spatial'` 新路径，`spatial`+`latent_pool='global'` 报错），commit `9f6c84f`；单测 **41 条通过**。配置 `..._lowdim_z_delta_spatial.py`。450 iter：recon `1.2235→1.0856`、correction² `0.0159→0.0038`、clamped `0.909`，**与方案一在噪声水平内相同**；三窗口 `shuffle_z` head 仅 `2.3–2.8e-5`（要求 ≥1%）。**读出结构不是瓶颈。**
   - **根因定位（2026-09-27，新增两个只读探针，已定量）**：`tools/probe_correction_scale.py` + `tools/probe_posterior_source.py`。① **上游信息在 `F.normalize(e_pooled)` 处被抹掉**：跨场景抽样下 `feat` pool16 的样本特异能量占比 **23.3%** → `encoder` 输出 `e_t` **7.2%** → `normalize(pool(e_t))` 后只剩 **1.28%**（分母被 1.90 RMS 的公共分量主导）→ `h_t` **0.27%** → `mu_q` **0.81%**；被重建和检测共用的 correction（`mu_q−mu_p`）只剩 **0.78% 样本特异能量（即 99.2% 是公共偏置）**，三窗口一致。标准 `MotionAlignedRSSMFusion` 的 posterior 直接吃未归一化的全分辨率 `e_t`，不做这一步。② **下游增益不足**：训练后 `z_proj` 权重仍停在 `std=0.01` 初始化尺度（weight RMS `0.0101`），检测损失没有把它推大；把 `z_proj` 增益 ×10 后 `shuffle_z` head ratio 由 `2.9e-5` 升到 `0.0126`（越过 1% 门槛），×100 到 `0.139`，三窗口同样形状，证明**读出通路是通的，只是信号幅度和上游信息量都不够**。
   - **下一步（按代价从低到高，尚未执行）**：① **去掉 `e_pooled` 的 L2 normalize**（改回标准 RSSM 做法，或换 LayerNorm / 仅对 `z_prev` 保留），这是唯一被定量证明会直接抬高样本特异占比的一步，改动一行、风险最低。② 若 ① 后 `mu_q` 特异占比仍上不去，再考虑放开 `h_t` 的公共偏置（`h_t` 本身只有 0.27% 特异能量，GRU 实际收敛成了常量状态）。③ 只有机制门槛（`shuffle_z` head ≥1%）通过后，才做方案四的 KL balancing 与按 cell 聚合的 free-bits；**当前 raw KL 一直有界（450 iter 时 0.0376），再次确认它不是瓶颈**。
   - **路线 A：2×2 机制矩阵（2026-09-27，已完成，门控未通过）**：新增 `posterior_obs_mode ∈ {l2, scaled_raw}` × `z_proj_init ∈ {small, xavier}` 两轴（commit `48417c1`，单测 45 条通过），四组各 450 iter、其余变量完全固定，详见 `docs/training_runs_full.md` 第 54 节。
     - **上游修复生效**：`scaled_raw`（`pool(e_t)*0.1`）把 `e_pooled` 样本特异占比从 **1.28% 提到 10.31%**（8.1 倍），correction 从 4.79% 提到 6.01%。
     - **下游修复生效**：`z_proj_init='xavier'`（实测 RMS 0.0835，是 `small` 0.0099 的 **8.43 倍**）独立把 head 响应放大 **约 10 倍**（shuffle `2.77e-5 → 2.79e-4`，zero `5.98e-4 → 6.21e-3`）；训练后权重稳在初始尺度（0.0827），说明不是被压小而是没被推大。
     - **两者叠加仍不够**：最好的组 4（raw+xavier）`e_pooled` 14.19%、correction **12.11%**、`z_proj` RMS 0.0827，但 `shuffle_z` head 仅 **3.14e-4**（门槛 1%，差 32 倍），correction² 0.0044（门槛 0.005），recon −11.8%（门槛 15%）。raw KL 有界（0.0424）通过。
     - **关键判读**：`shuffle/zero` 比值在四组间恒定在 **0.036–0.046**——Xavier 把公共偏置和样本身份一起放大了，没有偏向后者；correction 特异占比涨 2.5 倍而 head 只涨 12%。**结论：缺的不是信号幅度而是结构**，即 `correction = mu_q − mu_p` 由两个公共大向量相减得到，绝对幅度小且方向未被约束。这直接对应路线 B 的设计动机。
     - **勘误**：53.6.2 原尺度表把 `z_proj` 的 weight 与 bias 一起放大，读数被高估。分离后：仅 weight ×10 时 head ratio `2.30e-3`，weight×10+bias×10 才 `1.263e-2`；即原「×10 越过 1%」主要来自 bias 注入的样本无关偏置。探针已改为分别报告 `10.0` 与 `10.0_both`，文档已加勘误。
     - **运行事实**：四组均 GPU 5/6/7、3 卡 DDP、seed 0、deterministic、IterBasedRunner 450 iter；显存均 `7436 MiB`，`1.09–1.11 s/iter`。**四组均无 val 记录**，因此无区间均值/全轮峰值/best saved，也不能给出任何 AP 结论。
   - **路线 B：Innovation-Conditioned RSSM（2026-09-27，已完成，机制门槛 3/3 通过、性能门槛未过）**：`e_hat_t = obs_prior(h_t)`（只由 `L_obs_pred` 训练）→ `innovation_t = e_pooled − stopgrad(e_hat_t)` → `delta_mu = posterior_delta(cat[h_t, innovation_t])` → `mu_q = mu_p + delta_mu`，且 **`correction_t = delta_mu`（不再是两个公共大向量之差）**；`L_total += 0.05*L_obs_pred + 0.1*L_recon`。commit `d354232`，RSSM 单测 **55 条通过**，工具单测 12 条通过，详见 `docs/training_runs_full.md` 第 55 节。
     - **机制门槛 3/3 通过**：① `innovation` 特异占比 **14.27% > e_pooled 7.75%**（obs_prior 减掉的主要是公共成分）；② `delta_mu` 特异占比 **13.51%**（≥5%）；③ `prior_only` head 1.8e-3 明显低于 posterior 4.5e-3（0.40×）。
     - **首个系统性改善**：`shuffle/zero` head 比值从路线 A 四组恒定的 **0.032–0.046** 提升到 **0.064–0.092（约 2 倍）**，三窗口一致；`mu_q` 特异占比 1.79%→**3.93%**（+2.2 倍）。`zero_z` 绝对值同时下降（7.0e-3→4.5e-3），即整体公共偏置被削弱。
     - **性能门槛仍未过**：`shuffle_z` head 最好 4.19e-4（门槛 2%，差约 48 倍）；recon −11.5%（门槛 15%）；raw KL 有界 0.0466 通过；`stat_obs_pred` 0.0050→0.0021（−58%）。
     - **工具适配**：两个诊断工具原先硬编码 `posterior_mu` 输出为 `mu_q`，在 innovation 结构下会读错张量；现均检测 `posterior_struct` 并做 `desired − mu_p` 反演 / `mu_q = mu_p + delta` 还原，新增 2 条工具回归测试。
     - **结论**：结构方向被验证（支持 Kalman 式 predict-correct 动机，可作论文机制证据），但路线 B 解决的是「correction 由什么构成」，没有解决「correction 如何被检测头放大」；路线 A 已证明单纯放大 `z_proj` 会等比放大公共偏置与样本身份，所以不能再只调增益。
     - **运行事实**：GPU 5/6/7、3 卡 DDP、seed 0、deterministic、450 iter；显存 `7436 MiB`，`1.095 s/iter`。**无 val 记录**，无区间均值/全轮峰值/best saved，**AP 尚未测量**。
   - **路线 B ep6 门控（2026-09-27/28，已完成，AP 止损、机制通过 ep6 门槛）**：正式训练跑到 ep6 验证落盘后主动停止（保留 24e 余弦日程，避免 LR 对照失真），GPU 5/6/7、seed 0、deterministic。详见 `docs/training_runs_full.md` 第 56 节。
     - **AP 止损**：ep6 Overall 3D moderate `31.4895` vs no2d_igdr **seed0** 同 epoch `34.0243`，差 **−2.5348**，超过 2.0 阈值。附记：交接摘要里那张 ep6 = 37.3254 的对照表实际是 **seed1** 曲线，已在本节勘误；本判定按 seed0 真实曲线执行。
     - **机制门槛通过**：`innovation` specific `43.51% > e_pooled 25.66%`；`delta_mu` specific `39.32%`（≥10%）；`shuffle_z` head `0.73%–1.09%`（三窗口，越过 ep6 继续门槛 0.2%，比 450 iter 高 20–25 倍；论文最终门槛 2% 仍差约 2 倍）。
     - **未过项**：`shuffle/zero` 比值从 450 iter 的 0.064–0.092 回落到 **0.041–0.071**（绝对放大主要来自公共偏置）；`correction²` ep1→ep6 `0.1489→0.0596` 仍单调下降、**未形成平台**；recon `−6.75%`（门槛 15%）。
     - **归因**：ep6 的 `mu_q` 对 `h_t` 的依赖（removal 0.259）强于对观测的依赖（removal 0.165），`h_t` 自身 specific 已升到 12.19%，即 correction 的样本性中混入了递归状态成分，与 `z_prev` 先 L2 normalize 再进 GRU 直接相关。
     - **三口径摘要**（6 条 val，跨度为 ep1-6，标准 ep12-16 窗口不存在）：区间均值 `27.1488`（n=6，实际跨度 1-6）；全轮峰值 `34.8551 @ ep5`；best saved `31.4895 @ ep6`（`interval=2`）。
   - **路线 B 后续：`z_prev` normalize 450 iter 单变量消融（2026-09-28，机制 4/4 正向、绝对量级仍不足）**：新增 `z_prev_normalize` 开关（默认 `True`，与此前所有 run 逐位一致），配置 `..._lowdim_z_innovation_rawnorm.py`（commit `d895f10`），RSSM 单测 **58 条通过**。详见 `docs/training_runs_full.md` 第 57 节。
     - **四个观察点全部正向**：① `correction²` 首次形成平台（最后 100 iter 稳定 0.0054，路线 B 同期仍从 0.0047 滑到 0.0041）；② `mu_q` 对观测的依赖首次超过对 `h_t` 的依赖（removal e 0.315 > h 0.276，路线 B ep6 为 0.165 < 0.259，比值 1.57 → **0.874**）；③ `h_t` specific 0.86% → **1.73%**（+2.0 倍）、`innovation` 14.27% → **36.95%**、`correction` 13.51% → **30.97%**；④ `shuffle/zero` 比值 0.064–0.092 → **0.084–0.129**，三窗口同向。
     - **但绝对量级仍不足**：`shuffle_z` head 1.96e-4–3.12e-4（与 normalize 冒烟同量级，距 2% 最终门槛约两个数量级）；`zero_z` 绝对响应同时减半（4.5e-3 → 2.4e-3），即比值上升是「分子分母同时下降、分子降得更慢」；`correction` RMS 0.0574 → 0.0290（−50%）。recon 轨迹几乎重合（末端 1.0852 vs 1.0812），raw KL 0.0657 有界。
     - **结论**：`z_prev` 的 L2 normalize 确实是压制样本信息的因素之一，**该动且要保留**；但它不是充分条件，不构成「可进论文主表」的证据。**即便机制正向，是否继续仍必须由 AP 决定**——这是 ep6 门控分支的硬约束，不能因为机制读数变好就跳过 AP。
     - **运行事实**：GPU 5/6/7、3 卡 DDP、seed 0、deterministic、450 iter；显存 `7436 MiB`、约 `1.08 s/iter`；`iter_450.pth` 已落盘；**无 val**，故无区间均值/全轮峰值/best saved。
   - **rawnorm ep6 三方对照（2026-09-28，AP 通过止损、机制全面优于路线 B）**：work_dir `/data/lurui/work_dirs/lowdim_z_innovation_rawnorm_3x2x2_24e_seed0`，GPU 5/6/7、seed 0、24e 日程跑到 ep6 验证落盘后停。详见 `docs/training_runs_full.md` 第 58 节。
     - **AP 差距收窄到 −1.1263**：ep6 Overall 3D moderate `32.8980` vs no2d_igdr seed0 `34.0243`；路线 B 同 epoch 为 `31.4895`（−2.5348）。这是低维 z 系列首次把 ep6 差距压到 1.2 点内（历史：第 50 节 −6.51、第 52 节 −6.90、路线 B −2.53）。**落入预登记的 1–2 区间**。
     - **机制全面优于路线 B ep6**：`shuffle_z` head `0.84%–1.38%`（B 为 0.73%–1.09%，论文门槛 2% 仍差约 1.5 倍）；`shuffle/zero` 比值 `0.077–0.125`（B 为 0.041–0.071）；`innovation` specific `46.70% > e_pooled 16.32%`；`delta_mu` specific `38.01%`；`zero_z` 绝对响应同时低于 B（0.11–0.13 vs 0.15–0.18），即分子升、分母降。
     - **两条反向证据（必须与 450 iter 结论并列）**：① `correction²` 在正式训练里仍**逐 epoch 单调下降**（0.2218→0.0685），第 57 节的「450 iter 平台」不是稳定性质；② `mu_q` 对 `h_t` 的依赖**重新超过**对观测的依赖（removal h/e = 1.46，路线 B ep6 为 1.57，450 iter 曾反转到 0.874）。**结论：rawnorm 改善的是 recurrent 通路的幅值尺度，不是信息流向**；「放大信号 ≠ 改变信息结构」。
     - **三口径摘要**（6 条 val，跨度为 ep1-6）：区间均值 `28.4652`（n=6，实际跨度 1-6）；全轮峰值 `33.8487 @ ep5`（无对应权重）；best saved `32.8980 @ ep6`。
     - **运行事实**：显存 `7438 MiB`、约 `1.05 s/iter`；ep1 前半段 `correction²` 冲到 0.5457、`grad_norm` 峰值 15.87 后自行回落，全程无发散；落盘 ep2/4/6；看门狗在 07:04:50 UTC 于 ep6 验证落盘后停止训练。
   - **路线 C 实现 + 450 iter 机制门控（2026-09-28，机制部分通过、绝对身份信号仍不足）**：实现 commit `6d7e865 feat(rssm): add object-centric latent supervision (route C)`，详见 `docs/training_runs_full.md` 第 59 节。
     - **实现**：辅助 head 只吃 `posterior_correction`（`h_t` 不可见，有 pre-hook 回归测试断言 head 输入与 `stat_correction_sq` 逐位一致）；CenterNet 风格 focal loss（alpha=2, gamma=4）；GT 热图由当前帧 `gt_bboxes_3d/gt_labels_3d` 生成（`bev_centers_to_heatmap`，纯 torch）；历史帧不带 target；默认权重 0 时 loss 逐位不变、head 根本不 forward；推理零开销；辅助头在 `init_weights()` 最后初始化以保持旧 checkpoint 逐位可比。**RSSM 单测 74 条通过**（新增 11 条）。
     - **预登记升权有证据**：0.05 跑完 450 iter 后首层 RMS 0.04817 vs Xavier 初值 0.04811（+0.12%）、末层 bias −2.19000 → −2.19207，命中「梯度太弱」条件，按预登记升到 **0.1**；head-only 过拟合探针确认 focal 目标本身可学（40 步内 `prob_at_gt` 0.133 → 0.953），排除接口写错。
     - **机制正向（四项）**：① hit@1 `0.0141 → 0.3988`（约 28 倍，单调）；② `correction²` 最后 150 iter 平台 `0.0077 → 0.0075`，平台值比 rawnorm 450 iter 的 0.0054 高 39%；③ `correction` specific `28.17%`（≥5% 门槛）；④ **上游 `e_pooled` specific 7.62% → 15.33%（+2.0 倍）**，是此前任何配置都没做到的 encoder 侧推进。recon `1.2268 → 1.0682`、raw KL `0.2209 → 0.0764` 有界，未发散。
     - **核心门槛仍未过**：`shuffle_z` head 最好仅 `0.000580`（0.58%），距 2% 论文门槛差约 3.4 倍；`shuffle/zero` 比值三窗口**全部下降**（0.129→0.087、0.120→0.078、0.084→0.061），原因是 `zero_z` 涨得更多（+158%~+174%）而 `shuffle_z` 只涨 72%~88% —— **公共偏置与身份信号同时放大，公共分量放大更多**。结论：路线 C 让 correction 编码了更多观测内容，但没让检测头更依赖样本身份。
     - **诊断口径修正**：首版 recall 用 `prob>0.5`，对 focal 软目标恒为 0（是口径问题不是模型问题）；现改为「每个**有 GT 的类别通道**内 argmax 是否落在该类 GT 中心」的 hit@1，另报 `stat_object_hm_thr50` 作参考。
     - **ep6 门控已完成（2026-09-28/29，AP 通过、身份信号增强、`correction²` 仍塌缩）**：work_dir `/data/lurui/work_dirs/lowdim_z_innovation_rawnorm_oc_3x2x2_24e_seed0`，GPU 5/6/7、seed 0、24e 日程跑到 ep6 验证与权重落盘后停。详见 `docs/training_runs_full.md` 第 60 节。
     - **AP 门控通过，首次单点反超基线**：ep6 Overall 3D moderate `34.2513` vs no2d_igdr seed0 `34.0243`（**+0.2270**），落入预登记的「差距 ≤1」分支。但 ep1-6 均值仍低 `2.2319`（`26.8101` vs `29.0420`），ep1 尤其慢（`13.6636`，比 base 低 1.60），单点反超不能推广为整体追平。
     - **ep6 四类构成不均**：Truck strict `+4.7601`（`26.6900` vs `21.9299`）、Car strict `+0.8743` 为正；Ped loose `−3.9130`（`23.2093` vs `27.1223`）、Cyc loose `−0.8132` 为负。Overall 反超主要由 Truck/Car 贡献。
     - **身份信号相对 450 iter 增强 23–31 倍**：`shuffle_z` head 三窗口 `1.141%–1.364%`（450 iter 为 `0.369%–0.580%`），越过 ep6 继续门槛 `0.2%`；`shuffle/zero` 比值 `0.0753–0.0995`，三窗口均不低于 450 iter，不再退化。论文最终门槛 2% 仍差约 1.5 倍。
     - **source 探针继续正向**：`e_pooled` specific `16.32% → 30.98%`（+90%，450 iter 为 15.33%）、`correction` specific `38.01% → 44.90%`、`innovation` `46.70% → 50.13%`；`correction` RMS 稳定在 `0.25656`（450 iter 时 0.06239），即增益来自方向分化而非幅度膨胀。`delta_mu` specific `44.90% ≥ 10%`，`innovation > e_pooled` 均通过。
     - **两条反向证据（必须与 AP 并列）**：① `correction²` 正式训练里仍**逐 epoch 单调下降** `0.2736 → 0.0793`（−71.0%，逐轮 −20.2%/−33.3%/−24.3%/−18.1%/−12.2%），第 59 节的 450 iter 平台未保持，形态与 rawnorm 几乎相同；② `mu_q` 对 `h_t` 依赖仍**超过**观测（removal h/e = `1.456`，rawnorm ep6 为 `1.458`），posterior 仍由 `h_t` 主导。
     - **三口径摘要**（6 条 val，跨度为 ep1-6）：区间均值 `26.8101`（n=6，实际跨度 1-6）；全轮峰值 `34.2513 @ ep6`；best saved `34.2513 @ ep6`（两者相同）。
     - **执行分支**：按「AP 差距 ≤1 且机制继续增强」预登记，已从 `epoch_6.pth` 续训至 ep12（2026-09-29 02:03:06 UTC 启动，保持 24e 余弦日程与优化器状态，采用 B/C 组合版）。ep12 需复核：Overall 对比 base ep12 `39.2953`、`correction²` 是否出现平台或回升、`shuffle_z` 是否继续向 2% 靠近、`h/e` 是否转向观测侧。
     - **仍未做**：路线 D（跨样本去塌缩 `L_common + L_var`）/ E（KL balancing + 按 cell 聚合 free-bits）未开始；两者都在 identity 门槛通过前不启动。KL 继续后置（raw KL 0.0427 有界）。
   - **本次主线结论**：三项结构修复（差分重建、排名损失、空间读出）全部未过门控，路线 A 的两轴修复均已证实有效但叠加后仍差 32 倍，完整曲线与归因见 `docs/training_runs_full.md` 第 53、54 节。
   - ④ KL 口径改动**仍后置**：待结构有效性过 ep6 门控后再做 stop-gradient 拆分（`L_dyn=KL(sg(q)||p)`、`L_rep=KL(q||sg(p))`，建议 `beta_dyn=1.0`、`beta_rep=0.1`）与按 cell 聚合的 free-bits（沿 channel 求和后 ~1.0 nat/cell，先固定不退火）。**不要在缺重建项时单独调 free-bits。**
   - 低维 bottleneck 与排他未来任务这两个设计方向**尚未被证伪**；本轮证伪的是「在无重建锚定 + prior 脱离 `h_t` 的前提下使用它们」。标准 RSSM 的四条关键要素（prior 依赖 `h_t`、观测似然/重建项、KL balancing、输出吃 `(h_t,z_t)`）与逐条对照见第 50.6.7 节。
   - categorical/unimix 放在上述步骤之后验证，只在低维 z 仍表现出多峰/离散切换表达不足时再引入；不要直接把逐像素 256ch Gaussian 全量替换成 categorical。

## 已完成

以下事项已经收尾，保留结论和结果出处；不再在正文中只标 `[x]`。

1. 已存在的 `no2d_igdr` baseline 的两种口径（2026-09-23 完成）。
   - 已跑完 temporal/GRU 控制组，ep20 val 后按预设判据截断；结果见 `docs/training_runs_full.md` 47.5.2。

2. `no2d_igdr` N4 RSSM 三 seed 补齐（2026-09-25 完成）。
   - seed0 val ep1-23、seed1 val ep1-21、seed2 val ep1-24 均已跑完；逐 epoch 曲线与三 seed 汇总见 `docs/training_runs_full.md` 48.2。
   - 固定窗口均值：ep12-16 Overall 3D moderate `40.4400 ± 1.3475`，ep18-20 `39.5780 ± 1.0651`；全轮峰值均值 `41.8296 ± 1.3843`（三 seed 峰值分别在 ep14/ep15/ep14）。
   - 结论修正：seed 间 std（1.35）与 47.2 节单 seed 判定 no2d_igdr 劣于 FG-FULL N=4 的效应量（0.74）同量级，该单 seed 结论不可靠；后续补配对比较见「训练补充」第 1 项。
   - 附注：seed2 原计划在 ep21 截断，因人工截断未执行而自然跑满 24e，是三个 seed 中唯一训完 24 轮的；其末尾权重清理单列为「训练补充」第 2 项。
