# TODO

## 训练补充

1. 最近需要对 `fgfull` 和 `no2d_igdr` 两种 N4rssm 做 seed1 和 seed2 补充。
   - `no2d_igdr` N4 RSSM 的 seed1/seed2 已于 2026-09-23 起跑（GPU 5/6/7，顺序执行，work_dir `fgfull_N4_no2d_igdr_2x4_24e_seed1/2`）。
2. 最近还需要补充以下 baseline：
   - run10 对应的 GRU 版 baseline。
   - [x] 已存在的 `no2d_igdr` baseline 的两种口径：已完成（2026-09-23 跑完 temporal/GRU 控制组，ep20 val 后截断，见 `docs/training_runs_full.md` 47.5.2）。

## 方法探索

1. 探索原 DreamerV3 的 BlockGRU scaling 方法和超参数使用方法。
2. 解决 `z_t` 失效：当前根因是 z 与 h/feat 高度冗余，没有只有它能完成的预测任务；继续调 Gaussian、KL/free_nats、fixed noise 或 learnable std 都不会解决。
   - 先用冻结 checkpoint 做因果诊断：对比 `z_t=mu_q`、`z_t=0`、batch 内 shuffle `z_t`、prior `mu_p` 四组，确认 z 是否真的影响检测输出。
   - 再做低维 bottleneck：将 z 从全分辨率 256ch 改为全局或粗空间 latent（如 16x16x32），通过 FiLM/门控调制 `h_t`，避免与 h/feat 信息重复。
   - 给 z 增加只有它能做的任务：预测下一帧 BEV/occupancy/Doppler/box latent，用未来一致性训练 prior，而不是只让 prior 拟合 posterior。
   - 重做 KL 平衡与 free-bits 口径：按 latent/spatial 聚合，考虑 DreamerV3 式 KL balancing，并让 free-bits 后期退火。
   - categorical/unimix 放在上述步骤之后验证，只在低维 z 仍表现出多峰/离散切换表达不足时再引入；不要直接把逐像素 256ch Gaussian 全量替换成 categorical。
