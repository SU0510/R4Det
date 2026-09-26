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
   - **进度（2026-09-26）**：低维 bottleneck + 排他未来任务已实现，但首轮正式训练因 detector 解包接线错误（把 `z_t` 当 loss、丢弃真正的 KL+future 目标）而无效，ep1-4 全部作废，详见 `docs/training_runs_full.md` 第 50 节。接线、未来任务方向、KL 二次归一化、mask 广播、`stat_*loss` 命名共 5 处已修复（commit `f5d900f`），29 个单测 + 450 iter 冒烟通过，24e 正式训练已重新启动；结论待跑完后补第 50 节。
   - 再做低维 bottleneck：将 z 从全分辨率 256ch 改为全局或粗空间 latent（如 16x16x32），通过 FiLM/门控调制 `h_t`，避免与 h/feat 信息重复。
   - 给 z 增加只有它能做的任务：预测下一帧 BEV/occupancy/Doppler/box latent，用未来一致性训练 prior，而不是只让 prior 拟合 posterior。
   - 重做 KL 平衡与 free-bits 口径：按 latent/spatial 聚合，考虑 DreamerV3 式 KL balancing，并让 free-bits 后期退火。
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
