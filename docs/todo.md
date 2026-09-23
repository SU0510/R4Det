# TODO

## 训练补充

1. 最近需要对 `fgfull` 和 `no2d_igdr` 两种 N4rssm 做 seed1 和 seed2 补充。
   - `no2d_igdr` N4 RSSM 的 seed1/seed2 已于 2026-09-23 起跑（GPU 5/6/7，顺序执行，work_dir `fgfull_N4_no2d_igdr_2x4_24e_seed1/2`）。
2. 最近还需要补充以下 baseline：
   - run10 对应的 GRU 版 baseline。

## 训练效率

1. 加快每个 epoch 的 validation（当前约 61 min/epoch，其中 val 前向 ~10 min / IoU+eval ~1 min / ckpt ~1.6 min）。
   - 零风险项：把 `evaluation.interval` 从 1 改成 2（与 `checkpoint_config.interval=2` 对齐）。奇数 epoch 的 val 本来就没有落盘权重、进不了 BEST，改动后每个保留点的数值完全不变，只是 ep12-16 窗口从 5 点变 3 点。**未执行**，seed1 仍在按 interval=1 跑。
   - val batch size（GPU4 实测 2026-09-23）：bs=1 只占 1.64 GiB、利用率 41-75%，显存有大量余量；但 **bs>1 当前跑不通**，不是 OOM，是两个代码路径问题：
     - `Base3DDetector.forward_test` 把 batch 维与 test-time-augmentation 维混淆，bs>1 被误路由到 `aug_test()`，报 `aug_test() got an unexpected keyword argument 'gt_bboxes_3d'`。
     - 绕过 dispatch 强制走 `simple_test` 也在 `R4Det.py:1105` 越界：collate 出的是 `img=[B,N,3,H,W]`（batch 在前），而该行按 `[frame][batch]` 取 `meta[t]`。
     - `R4Det.simple_test` 注释里的「Test runs with batch≥1」与 train.py 的「Support batch_size > 1 in validation」目前只是声明，实际未实现。
   - 理论判断（待实测确认）：eval 路径 batch 化不应改准确率——`model.eval()`（`mmdet/apis/test.py:22/100`）、BN 走 running stats 且 backbone `norm_eval=True`/`frozen_stages=1`、RSSM 的 `h_state/z_state` 是按 `[B,...]` 逐样本维护、eval 走 `deterministic=True`（`z_t=mu_q`，不采样无随机数）、无 TTA、NMS 逐样本、该 config 无 2D 分支。
   - 两个必须先解决的风险：(a) 要真改 `R4Det.forward_test`（aug_test 误路由 + 帧/批次维转置），写错会静默改结果；(b) 多 batch 下点云 voxelization 的 padding 需实测确认无 batch 间串扰。
   - 验收方式：同一 checkpoint 上跑 bs=1 vs bs=2，逐样本 diff 预测框并比 AP，数值一致才可启用。
   - 口径约束：若启用，seed1 与 seed2 会跑在不同代码/config 版本上；最稳做法是 seed1 跑完后、seed2 起跑前改，并在 `training_runs_full.md` 记录该差异。

## 方法探索

1. 探索原 DreamerV3 的 BlockGRU scaling 方法和超参数使用方法。
2. 解决 `z_t` 失效：当前根因是 z 与 h/feat 高度冗余，没有只有它能完成的预测任务；继续调 Gaussian、KL/free_nats、fixed noise 或 learnable std 都不会解决。
   - 先用冻结 checkpoint 做因果诊断：对比 `z_t=mu_q`、`z_t=0`、batch 内 shuffle `z_t`、prior `mu_p` 四组，确认 z 是否真的影响检测输出。
   - 再做低维 bottleneck：将 z 从全分辨率 256ch 改为全局或粗空间 latent（如 16x16x32），通过 FiLM/门控调制 `h_t`，避免与 h/feat 信息重复。
   - 给 z 增加只有它能做的任务：预测下一帧 BEV/occupancy/Doppler/box latent，用未来一致性训练 prior，而不是只让 prior 拟合 posterior。
   - 重做 KL 平衡与 free-bits 口径：按 latent/spatial 聚合，考虑 DreamerV3 式 KL balancing，并让 free-bits 后期退火。
   - categorical/unimix 放在上述步骤之后验证，只在低维 z 仍表现出多峰/离散切换表达不足时再引入；不要直接把逐像素 256ch Gaussian 全量替换成 categorical。

## 已完成

以下事项已经收尾，保留结论和结果出处；不再在正文中只标 `[x]`。

1. 已存在的 `no2d_igdr` baseline 的两种口径（2026-09-23 完成）。
   - 已跑完 temporal/GRU 控制组，ep20 val 后按预设判据截断；结果见 `docs/training_runs_full.md` 47.5.2。
