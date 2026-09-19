# 研究状态快照（中断续接用）

> 最后更新：2026-09-19 15:00 UTC。用途：任何新会话从这里恢复上下文。
> 详细历史：`docs/training_runs_full.md`（Run 1–44）、`me_rssm/README.md`、
> `me_rssm/docs/01–09`。

## 正在运行 / 已排队

> **2026-09-19 15:25 UTC 更新：应用户要求，本轮挂的后台等待任务与两个
> 自动化 tmux（`me_rssm_queue`、`verdict_watch`）已全部拆除。**
> ME-RSSM seed0 **不再自动排队**——seed2 结束后需手动启动：
> ```bash
> tmux new-session -d -s me_rssm_queue 'bash me_rssm/queue_me_rssm_seed0.sh'
> # 或直接：
> CUDA_VISIBLE_DEVICES=5,6,7 bash tools/dist_train.sh \
>   me_rssm/configs/TJ4D-R4Det_me_rssm_N4_24e_pretrained_v2_head.py 3 \
>   --seed 0 --deterministic --work-dir /data/lurui/work_dirs/me_rssm_N4_2x4_24e_seed0
> ```
> 三 seed 终判数字届时手动运行：`python3 tools/verdict_window.py --base
> work_dirs/run10_headv2_multiseed --cand work_dirs/cyccls_branch_N4_2x4_24e_multiseed
> --window 12 16`。

| 项 | 状态 | 位置 |
|---|---|---|
| CycCls seed2（44.7 补跑） | 训练中 ep14/24（会话开始前已在跑，未受拆除影响） | tmux `cyccls_seed2`，GPU 5,6,7，work_dir `/data/lurui/work_dirs/cyccls_branch_N4_2x4_24e_multiseed/seed_2` |
| ME-RSSM 主线 seed0 | **未排队**（自动化已按用户要求拆除，脚本保留待手动触发） | 脚本 `me_rssm/queue_me_rssm_seed0.sh`、日志 `/data/lurui/work_dirs/me_rssm_queue.log`（仅历史） |

## 本会话已完成（全部只新增文件，原项目零修改）

1. **P1 观测引导状态初始化**（`state_init='obs'`）：me_rssm 实现 + 配置 +
   sanity 13/13（docs/07）；提交 6dc4d12。
2. **环境发现**：CUDA 上 deform 对齐逐位比较分配器布局敏感，未改动基线
   同样复现——逐位检查一律移到 CPU（docs/07 §5）。
3. **Doppler 证据可视化**：viz_doppler_evidence.py + 三张样本图；数据事实：
   Doppler 被 ego 运动主导（speed 通道不能单独区分动静）、val 开头有连
   续空帧、v_std 稀疏热点（docs/08）；提交 6bd31b7。
4. **eval 入口验证**：test_entry.py 5/5（test_vod 路径 build + 预训练
   ckpt 契约 + 真样本 4 帧 forward + bus 契约）；提交 3c6437c。
5. **判定/监控工具**：tools/verdict_window.py（配对窗口判定 + stat 曲线
   提取），seed1 数字与 44.7.2 逐位复现；提交 d709119。
6. **新颖性文献核查**：CRT-Fusion/RCFusion/TPFE 等对照表与差异化叙事
   （docs/09）；结论：三要素组合（RSSM+Doppler 证据+BEV 检测滤波）未见
   先例，最近邻 CRT-Fusion 为确定性递归融合。

## 接下来的判定点（按时间顺序）

0. **预注册：ME-RSSM 主线 seed0 的判定阈值**（写于任何 ME-RSSM 验证值
   存在之前，2026-09-19 15:10 UTC）：
   - 口径：ep12-16 窗口均值，与 clean seed0（`run10_headv2_multiseed/seed_0`）
     配对；主指标 Overall 3D moderate，辅 BEV + 四组成项。
   - 背景噪声（2026-09-19 用 run10 多 seed 日志实测）：clean ep12-16 窗口
     均值跨 seed σ=0.534（seed0/1/2 = 38.39/39.20/39.40）；平台期逐 epoch
     跨 seed σ≈0.94。配对 Δ 噪声界 0.42–0.53 ⇒ **+0.7 ≈ 1.3σ**、
     **+1.2 ≈ 2.2σ**，阈值与经验噪声尺度一致。
   - **有效**（进入多 seed）：Overall Δ ≥ +0.7（≈1σ）且 BEV 同向，四组成
     项无一项 < −1.0；**强有效**（直接定主模型候选）：Δ ≥ +1.2 且 ≥3 个
     组成项为正；**无效**：|Δ| < 0.5 或方向不定 → 按失败模式走消融网格
     （`--stats` 曲线先看 stat_gain/conf/dyn 是否偏离初值，机制死了就直接
     读消融，不再burning GPU 于重复 seed）。
   - **按类效应预注册**（依据 docs/08 GT 点级动态分数实测：Cyclist 0.488 >
     Car 0.315 > Truck/背景 > Pedestrian 0.145——Doppler 证据只在运动目标
     上有信息）：若 ME-RSSM 有效，预期 **Cyclist/Car 增益 > Truck > Ped**；
     若有效但增益集中在 Ped（证据最弱的类），反而提示增益来自非运动通路
     （如 camera obs 分支），需要 no_modality_obs 消融甄别。
   - 消融顺序（若主线无效）：no_modality_obs → no_reliability_gain →
     no_dynamic_gate → no_motion_offsets（隔离哪个机制引入噪声）；
     若机制活跃但 AP 平：P2 innovation-gain 优先于加 seed。
1. **seed2 完成 → CycCls 三 seed 终判**（44.7.6 承诺）：
   ```bash
   python3 tools/verdict_window.py \
     --base work_dirs/run10_headv2_multiseed \
     --cand work_dirs/cyccls_branch_N4_2x4_24e_multiseed --window 12 16
   ```
   预注册结论框架：逐 seed 口径 seed0 过/seed1 败；唯一跨 seed 稳健正效应
   是 Car strict（+2.93/+2.86）；若 seed2 完整窗口也失败 → CycCls 记为
   「仅 Car strict 正信号，Overall 不支持」，主模型维持 Run 10 head-v2。
   结果写入 training_runs_full.md 新节。
2. **ME-RSSM seed0 训练完成 → 主线判定**：ep12-16 窗口 vs clean seed0
   （`work_dirs/run10_headv2_multiseed/seed_0`）配对；同时用
   `python3 tools/verdict_window.py --stats <work_dir>` 检查 stat_gain/
   conf/dyn 曲线（决定 06 的 P2 innovation-gain 与 C6 head 不确定性是否
   启动）。若主线有效 → 多 seed；若无效 → 按失败模式走消融网格
   （me_rssm/configs/*_abl_*.py）或转 P1 stateinit 变体。
3. 之后：论文主表口径按第 18 节；related work 骨架用 docs/09。

## 硬约束提醒（每个新实验前重读）

- 原项目零修改；一切新产出进研究目录（me_rssm/、新 tools 文件、docs）。
- 单 seed 的 Overall +0.6~1.2 在平台噪声 ±1.5 内；任何判定用固定窗口
  均值 + 多 seed，不用单点 BEST。
- checkpoint 保留规则：每 run 只留 best+last（第 17 节）。
- **复现事故防护**：主线配置仍残留 `shared_stem=True`（audit U7）——复跑
  clean 基线一律用快照 `me_rssm/configs/TJ4D-R4Det_clean_N4_2x4_24e_
  pretrained_v2_head_snapshot.py`；`BEVRSSMTemporalFusion.forward` 缺
  return（99c74ff 潜伏 bug，report 3.3）——复跑 Run 2 需补丁副本，原类
  不可直接用。
