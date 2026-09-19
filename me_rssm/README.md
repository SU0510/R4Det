# me_rssm · Motion-Evidence RSSM 研究包

> 目标：让 Radar 不再是"第二种 BEV 外观特征"，而是作为 geometry + velocity +
> motion prior 参与 RSSM state transition；Camera 提供语义观测、Radar 提供运动观测，
> RSSM 建模隐藏世界状态随时间的传播。
>
> 协议：不训练、不验证、不通过指标选结构；不修改任何现有文件；
> 全部产出集中在本文件夹。基于仓库 `goal-test` @ `213cd89`（2026-09-19）。

## 一页结论

当前基线（MotionAlignedRSSMFusion）的概率时序建模有实测净贡献（多 seed +1.3~1.9），
但存在一个结构性断点：**Doppler 径向速度——数据集中唯一直接的运动测量——
在 PFN 与融合层被两次外观化，从未到达时序动态**；转移模型因此没有预测能力，
先验退化为后验的镜像（posterior collapse 是实测稳态），prior/可靠性/模态缺失
fallback 均无从谈起。

本包实现 **ME-RSSM**：一条无参数的 Doppler 证据旁路（7 通道速度统计直接进 BEV）
+ 三个零初始化增量机制——运动条件化状态对齐、可靠性门控的 Kalman 式校正
（z = μ_p + g·(z_sel−μ_p)，观测不可信时降级为运动传播预测）、Doppler 动态证据
调制的 GRU 更新门。参数 +0.42M（+3.9% temporal），MACs +4.4%（temporal 级），
**空 bus / 开关全关时与基线逐位等价**（已实测），预训练加载行为与基线逐键一致（已实测）。

## 文件导航

```
me_rssm/
├── __init__.py                  # 注册全部新模块（FUSION/MIDDLE/VOXEL_ENCODERS）
├── modality_bus.py              # 每帧模态边带通道（scatter/fusion 写，temporal 读）
├── radar_motion_chain.py        # RadarPillarFeatureNetMotion + PointPillarsScatterMotion
├── stash_fusion.py              # ConcatConvFusionStash（基线融合 + 发布输入）
├── motion_evidence_rssm.py      # MotionEvidenceRSSMFusion（核心模块）
├── configs/
│   ├── TJ4D-R4Det_me_rssm_N4_24e_pretrained_v2_head.py   # 主线（继承 clean mainline）
│   └── ..._abl_{no_motion_offsets,no_reliability_gain,no_dynamic_gate,no_modality_obs}.py
├── sanity/                      # 5 个验证脚本（05 报告可一键复现）
│   ├── test_radar_motion_chain.py   # CPU：Doppler 链数值正确性
│   ├── test_equivalence.py          # GPU：E1-E4 基线等价性
│   ├── test_grad_paths.py           # GPU：G1-G6 梯度/状态/回退语义
│   ├── test_config_build.py         # CPU：配置/构建/预训练键兼容/参数计量
│   └── test_flops.py                # GPU：静态 MACs
└── docs/
    ├── 01_architecture_map.md   # 数据流/tensor shape/信息流（file:line 逐条核验）
    ├── 02_problems.md           # 研究目标 14 项结构检查的逐项回答
    ├── 03_candidates.md         # 12 个候选设计的筛选矩阵与取舍论证
    ├── 04_design_me_rssm.md     # 设计规范/数学形式/成本/消融网格/相关工作
    ├── 05_sanity_report.md      # 验证结果与复现方式
    └── 06_second_layer.md       # 第二层审查：遗留问题与后续循环路线
```

## 使用（供后续公平实验；本包自身不执行训练）

```bash
# 训练入口与基线完全一致，仅换配置文件
bash tools/dist_train.sh \
  me_rssm/configs/TJ4D-R4Det_me_rssm_N4_24e_pretrained_v2_head.py 3 \
  --seed 0 --deterministic --work-dir /data/lurui/work_dirs/me_rssm_seed0
```

消融：同命令换 `*_abl_*.py`。方法论沿用《审计报告》§8.8（多 seed、窗口均值、
预注册门控）。

## 与基线的关系

- 基线类（`MotionAlignedRSSMFusion` 等）与本包类并存，config 可切换；主配置与
  全部基线配置文件未做任何修改。
- ME-RSSM 是基线的严格超集：任意基线 RSSM checkpoint 可载入（E1），
  无边带数据时行为逐位等于基线（E2）。
