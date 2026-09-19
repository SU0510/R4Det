# 10 · 序列起点状态探针（训练后基线的机制证据）

> 工具：`me_rssm/sanity/diag_state_probe.py`（零训练诊断）。
> 对象：**已训练**的 clean 基线 `run10_headv2_multiseed/seed_0/epoch_16.pth`
> （平台最优 checkpoint），6 个 val 样本（idx 3/10/50/100/200/400）的真实
> eval forward，时序模块按帧内调用次序 t=0..3 分组。
> 原始数据：`me_rssm/figures/state_probe.json`。
> 复现：
> ```bash
> CUDA_VISIBLE_DEVICES=4 python me_rssm/sanity/diag_state_probe.py \
>   --checkpoint work_dirs/run10_headv2_multiseed/seed_0/epoch_16.pth
> ```

## 1. 结果（6 样本均值）

| 位置 | h_in 范数 | h_out 范数 | mu_diff² | off_h 范数 |
|---|---:|---:|---:|---:|
| t0（序列起点） | **None（零状态）** | **109.33** | **0.0239** | 17557 |
| t1 | 109.33 | 500.65 | 0.0128 | 17523 |
| t2 | 500.65 | 578.50 | 0.0080 | 17281 |
| t3 | 578.50 | 624.55 | 0.0085 | 16969 |

## 2. 三条机制结论

1. **死状态被精确证实**：t0 的转移输出 h_t = GRU(0, 0) 在 6 个不同样本间
   范数完全一致（109.3327，逐样本常数）——首帧的历史通路携带**零样本
   信息**，纯 bias 常数。N=4 窗口的 1/4 帧里，"时序记忆"不存在。
2. **起点滤波失配**：t0 的 prior/posterior 均值分歧（mu_diff²=0.0239）是
   稳态（t≥2，≈0.008）的 **~3 倍**——每个窗口开头滤波器都要"重新发现"
   观测与预测的关系，与 05/09.5 观察到的 posterior collapse 稳态并存。
3. **对齐生成器主要读当前特征而非状态**：offset 范数从 t0 到 t3 几乎
   不变（17.5k→17.0k）——deform 对齐的位移场主要由 feat 驱动，历史状态
   对"对齐什么"的贡献很小。这对 ME-RSSM 的 motion-conditioned offsets
   （Doppler 证据项加在 offset 上）是有利的：该通路不与状态依赖竞争。

## 3. 对 P1（state_init='obs'）的意义

- 动机从"静态代码审查"升级为**训练后模型的量化机制证据**（论文
  analysis 小节的素材）：每窗口 1/4 的帧无历史信息 + 起点失配 3×。
- P1 的机制（首观测 bootstrap h_0/z_0）直接作用于这两处；若训练后
  stateinit 变体的 t0 mu_diff 显著低于基线的 0.024，即为机制起效的
  过程证据（连同 AP 一起报告）。
- 探针可在 stateinit 训练完成后以同一脚本复跑（换 checkpoint + 把
  assert 改为 MotionEvidenceRSSMFusion），得到前后对照表。

## 4. 工程注记

- 主线配置 `TJ4D-R4Det_motion_align_rssm_det3d_N4_2x4_24e_pretrained_v2_head.py`
  仍残留 section-41 的 `shared_stem=True`（审计 U7）；Run 10 多 seed 实际
  是 `shared_stem=False`。任何直接加载该 checkpoint 的新脚本都必须像本
  探针与 ME 主线配置那样显式 `shared_stem=False`，否则 head 键全部
  missing。ME 配置已自带该覆盖。
- 探针还顺带确认：eval 路径下基线 stats 含 `mu_diff_sq`（ME 的
  `stat_posterior_std` 等只在 ME 模块里存在），两套监控口径不冲突。
