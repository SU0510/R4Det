# 汇报前核对

- 证据来源：R4Det+RSSM 项目内部实验记录（docs/training_runs_full.md，2026-07 至 2026-09，数字均经 work_dirs 日志与 checkpoint 复核）
- 本次使用的证据图：图A｜TJ4D 类别分布（报告 §1）、图B｜网络总体架构（报告 §4）、图C｜主指标演进链（报告 §6）、图D｜RSSM 机制与代码对应（报告 §5）、图E｜三 seed 复现（报告 §7）、图F｜RSSM 机制消融（报告 §8）、图G｜Truck 专项三轮（报告 §10）
- 所有数字（40.42±0.51、35.59、34.65/37.08/39.00、37.94、+3.23、逐类别 strict/loose 表）均由 work_dirs 日志复算核对，与 docs/training_runs_full.md 一致；逐类别表取各 seed BEST epoch（s0 ep16 / s1 ep15 / s2 ep14）。
- 图 A–G 为本项目实验数据自绘（非论文原图）；每页页脚含来源定位（报告章节 / 代码文件）。
- 结论边界：时序融合收益在『同配置 No-Temporal』对照下成立；18e 与 24e 时代差异已在图C 注明；单 seed 提升不作为结论。
- 已知未解：Pedestrian strict≈0、Car-Truck 混淆、Cyclist 排序质量、CycCls seed2 进行中。
