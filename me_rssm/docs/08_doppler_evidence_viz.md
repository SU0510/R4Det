# 08 · Doppler 证据画布数据检查（viz_doppler_evidence.py）

> 工具：`me_rssm/sanity/viz_doppler_evidence.py`（零训练、零 checkpoint，
> 只依赖数据与无参数统计链 `RadarPillarFeatureNetMotion._motion_stats`
> → `PointPillarsScatterMotion._scatter_motion_batch`）。
> 复现：
> ```bash
> python me_rssm/sanity/viz_doppler_evidence.py --indices 3 400 2000 \
>     --out me_rssm/figures
> ```
> 输出：每样本一张 4 帧 × (原始点云按径向速度着色 + GT 框 | ch3 speed |
> ch4 v_std | ch6 occupancy) 的 PNG。本报告记录 2026-09-19 在 val 集上
> idx=3（ego 运动）、idx=400、idx=2000（ego 静止）的读图结论。

## 1. 数据事实（对 ME-RSSM 设计与论文叙事都重要）

### F1 · Doppler 被 ego 运动主导

- ego 运动帧（idx=3/400）：点的径向速度均值 |v| ≈ 7–13 m/s，
  画布 filled cell 中 **|v|>1 m/s 的比例 ≈ 100%**（idx=3 frame3:
  870/873）。整幅 speed 通道呈现"ego 流场"形态：同向车道为负（接近）、
  对向/被超车为正（远离），与位置 r̂ 的几何强相关。
- ego 静止帧（idx=2000）：全场景 |v| ≈ 0（98 分位 < 2 m/s），
  |v|>1 的 cell 只有 5–8 个。
- **推论 A**：ch3（|v_r| speed）单独**不能**区分动/静目标——它主要编码
  ego 是否在动。目标运动的判别信息在 (a) ch4 v_std（同柱内速度离散，
  混合目标/动态证据），(b) (vx, vy, x, y) 联合模式（ego 流场是位置的
  确定性函数，可被 conv 学出并减掉）。
- **推论 B**：这解释了 Run 13 ego-velocity 拟合（RadarStaticDynamicScore）
  的设计动机，也说明 ME-RSSM 的 `dyn_conv`（1×1 conv 吃全部 7 通道）
  原则上**能**从 (vx, vy, v_r, x/y 相关的 r̂) 学出 ego 补偿——但这是
  一个"留给网络隐式学"的设计选择；显式 ego 补偿通道（Run 13 的残差）
  保留为预注册二线候选（等主线训练信号再决定，P2 同规则）。

### F2 · 4 帧点云结构与空帧

- `dataset[i]` 的 `points` 是 **4 帧雷达点云的 list**（N=4 时序的数据侧
  形态）；GT 也是每帧一份 list。每帧独立过 pillar 统计与 scatter，
  各自发布一张 velocity_bev 画布（时序模块按帧读 bus）。
- val 集开头存在**连续空帧**（idx=0/1/2：0 点、0 GT）。检测器
  `voxelize` 对此有 dummy-point fallback（R4Det.py:1644-1648），
  ME 链路继承该行为；`PointPillarsScatterMotion` 对空柱输出全零画布
  （counts.clamp(min=1) 路径），统计通道不会 NaN。

### F3 · 动态证据（ch4 v_std）非常稀疏

两个运动样本的 v_std 通道整体接近 0（≤0.3），只有少数目标柱被点亮。
这符合"同一 0.16 m 柱内速度混合"的物理直觉，也意味着：
- dyn gate 的输入信号是**稀疏热点**而非稠密场——06 §2 的"稀疏证据
  池化稀释"风险（avg_pool(2) 把单热点稀释 4×）是真实存在的；
  count 通道（ch6）与后续 max-pool 备选（预注册）值得在训练信号到来后
  优先检查 stat_dyn_ratio 的实际分布。

## 2. 工程注记（复现/扩展者读）

- 点云通道序：load 后 (x, y, z, v_r, snr)（use_dim=[0,1,2,3,5]），
  与 PFN/Motion stats 的硬编码索引一致。
- 画布坐标：(7, ny=496, nx=432)，rows=y、cols=x，0.16 m 分辨率；
  `_read_context` 会 avg_pool(2)+permute 到 (7, 216, 248)（rows=x）。
- GT 框张量是 **(x, y, z, l, w, h, yaw)** 顺序（l 在 index 3；由
  Car 4.18/1.79 与 Ped 1.81/0.87 的量级确认），画 BEV 角点时注意。
- 色标：径向速度用 98 分位自适应（±2~±10 m/s 随 ego 状态变化），
  固定色标会饱和。

## 3. 对论文的用法

- idx=3 vs idx=2000 两张图并排 = "Doppler 是 ego 主办的原始测量，
  需要时序/几何上下文才能变成目标运动证据"的 motivation figure 候选。
- 若主线训练有效，stat_gain/stat_dyn 的空间分布可与 ch4 热点对照
  （gain 是否在动态证据热点处更低 = 更信预测），作为解释性 figure。
