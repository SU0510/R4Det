# 研究状态快照（中断续接用）

> 最后更新：2026-09-19 15:00 UTC。用途：任何新会话从这里恢复上下文。
> 详细历史：`docs/training_runs_full.md`（Run 1–44）、`me_rssm/README.md`、
> `me_rssm/docs/01–09`。

## 正在运行 / 已排队

| 项 | 状态 | 位置 |
|---|---|---|
| CycCls seed2（44.7 补跑） | 训练中 ep13/24，预计 2026-09-19 ~24:00 UTC 结束 | tmux `cyccls_seed2`，GPU 5,6,7，work_dir `/data/lurui/work_dirs/cyccls_branch_N4_2x4_24e_multiseed/seed_2` |
| ME-RSSM 主线 seed0 | **已排队**（队列脚本轮询，seed2 结束后自动启动，24e，约 22h） | tmux `me_rssm_queue`，脚本 `me_rssm/queue_me_rssm_seed0.sh`，日志 `/data/lurui/work_dirs/me_rssm_queue.log`，work_dir `/data/lurui/work_dirs/me_rssm_N4_2x4_24e_seed0` |

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
