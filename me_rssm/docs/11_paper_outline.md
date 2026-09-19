# 11 · 论文骨架与素材映射

> 目的：把"最终形成论文"拆成可核对的素材清单——每一节对应哪些已有
> 工件、哪些等训练信号。写于 ME-RSSM seed0 训练前；结果落地后按
> 判定结果填充/裁剪。核心故事线假设：**ME-RSSM 有效**；若无效，骨架
> 不变，但 contribution 3 改写为"机制级负结果 + 消融解释"（docs/04 §6
> 已有失败模式分析支撑）。

## 拟定题目方向

Measurement-Conditioned Recurrent State Spaces for 4D-Radar–Camera
BEV Detection（暂定；关键关键词：Doppler evidence / predict-correct /
reliability gain / pose-free temporal alignment）

## Contribution 草案（三条）

1. **测量驱动的时序滤波**：把 4D 雷达唯一直接的运动测量（径向 Doppler）
   作为无参数证据旁路接入 RSSM 转移——运动条件化对齐 offset + 动态证据
   调制的 GRU 更新门（predict）；区别于把它当特征（RCFusion）或当确定性
   对齐引导（CRT-Fusion）的用法。
2. **可靠性门控的 correct 步**：逐像素观测编码 + 学习 Kalman 式 gain，
   z_t = μ_p + g·(z_sel − μ_p)，观测不可信时定义明确地降级为运动传播
   预测（含 modality dropout 训练压力）；对比确定性递归融合无降级语义。
3. **无位姿设定下的时序对齐 + 机制级分析**：TJ4D 无 ego pose/时间戳，
   测量驱动对齐是必需而非选择；给出序列起点死状态的量化证据（docs/10）
   与 posterior collapse 稳态的实测刻画（§9.5/14/15），以及完整消融。

## 章节 → 工件映射

| 论文章节 | 已有素材 | 待补（等训练信号） |
|---|---|---|
| Intro | docs/09 差异化叙事；docs/08 F1（Doppler ego 主导动机图） | 主表数字 |
| Related Work | docs/09 对照表（RCFusion/CRT-Fusion/TPFE/RCBEVDet/Mamba 系/世界模型） | — |
| Method | docs/04 数学形式（7 通道证据、motion offsets、dyn gate u'=u+βdu(1−u)、gain、空 bus 等价三保证） | 最终超参 |
| 实现细节/实验设置 | 训练记录 §18/§19（口径）、snapshot 配置、run_multiseed.sh | — |
| 主表（TJ4D val） | clean 多 seed 40.42±0.51（基准行） | ME-RSSM 多 seed、stateinit（若启动） |
| 消融 | me_rssm/configs/*_abl_* 四开关；KL0 三 seed（附录负证据，§25）；No-Temporal/Deterministic（§20/21，时序收益分解） | 四开关消融行 |
| 分析/可视化 | docs/08（Doppler 画布 4 帧图）、docs/10（死状态探针表）、stat_gain/conf/dyn 曲线（verdict_window --stats）、docs/07 §3 sanity 矩阵 | ME 训练后的 stat 分布图、t0 前后对照 |
| 定性结果 | viz_doppler_evidence.py 图（idx3/400/2000） | 检测结果 BEV 可视化（训练后） |

## 投稿目标候选（按故事完成度决定）

- 主投：IEEE T-ITS / RA-L（雷达-相机检测的常驻地，审稿人熟悉 VoD/TJ4D）；
- 备选：ICRA/IROS（篇幅短，分析小节压缩）；CVPR/ICCV（若 Overall 提升
  幅度足够强且机制消融完整）。

## 写作顺序建议（结果到位后 2–3 周量级）

1. Method + Related Work（已可写 90%，等超参定稿）；
2. 实验：主表 → 消融 → 分析（等 seed0 判定 → 多 seed 排队）；
3. Intro/Conclusion 最后写；图：fig1 动机（docs/08 两图并排）、
   fig2 架构（需新绘制，素材 docs/01）、fig3 stat 曲线、fig4 定性。

## 明确不做

- 不为凑贡献堆第三个模块（P2/P5/C6 全部等 stat 信号，见 06 §3）；
- 不在 Pedestrian strict 物理瓶颈（§29–39 已闭环六次）上再开新支线。
