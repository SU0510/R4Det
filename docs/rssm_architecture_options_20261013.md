# AP 优先的结构级候选：不再围绕同一可靠性门做微调

> 2026-10-13。研究建议，不是预登记，不是训练授权。
> 本次仅检查代码/已有实验记录及原始论文；未修改模型、配置、数据、指标或队列。
> 以下收益均为待验证假设，不承诺 AP 增量；已有方法基础不冒充本项目创新。

## 1. 为什么该换探索空间

可证事实：
- C1 fixed window ep12–16（n=5）Overall 39.8303，对 B0 seed0 +0.5494；
  全轮峰值 40.6389 @ ep16，best saved 同值。Car/Truck +6.3237/+6.7151，
  Ped/Cyc −4.8880/−5.9530。§87 另有 high recall +4.4137、low −6.7401、zero −2.1023。
  这些证明该臂有不利的收益重分配，不证明整个网络只能做零和交换。
- F1 实质仍是 `ConcatConvFusion([camera, alpha * radar])`；
  `alpha = 1 + tanh(gate(occupancy, mean_abs, max_abs))`。
  它改变同位置雷达贡献，不建立跨位置匹配，不是融合范式更新。
  入口：`me_rssm/reliability_fusion_rssm.py:110–127`。
- 原图已有五层 FPN；`R4Det.extract_feat` 在 lifting 前把它们缩放到同一网格并拼接
  （`mmdet3d/models/detectors/R4Det.py:822–825`），之后对 camera volume 的高度维取均值
  （829–831）。现有 radar 也已有 SECOND/SECONDFPN；头已有 IoU 分支与类别锚框。
  因而“加 FPN”“加 IoU head”“再加前景监督”不构成新的候选。
- §29–39 已关闭一批 Ped-only refinement/CenterHead/high-res 支线。
  §38 的 0.16 m 分支为冻结、detach 的 radar scatter + 已融合低分辨率 BEV，
  不是所有类别共同训练、从原始图像多尺度与局部雷达重新取证的架构。

推断而非已证根因：近期修改多在已压缩/混合的 BEV 上改变加权、更新或辅助监督。
值得改的是观测保留、跨模态对应、对象读出与时序建模单位，而不只是门的公式。
这不排除优化/泛化也有瓶颈；注意力、更多参数和模块数量本身都不保证 AP。

## 2. 七种可尝试方向

### 方向 1：几何约束、多尺度的 deformable cross-modal fusion

**替换范围**：替换 ConcatConv；首轮保留现有 RSSM 和 Anchor3DHead。

- camera/radar 保持独立特征流，显式构建多尺度 BEV。
- 以标定投影和 BEV 参考点作初始对应，预测有限局部采样偏移；
  一个位置可以读取另一模态邻近位置/尺度的特征，而非只给同一 cell 乘 alpha。
- 双向交互：相机提供语义与轮廓，雷达提供距离/空间证据；
  不以无回波为由禁用相机分支。稀疏/局部采样，不做全 BEV 二次复杂度注意力。

**假设**：一部分损失来自空间错配/尺度混合，学习“去哪取对应证据”优于只改强弱。
**关键消融**：原 ConcatConv、匹配容量的 conv、无位置偏移、单尺度/多尺度；
额外记录采样偏移的合理性，不以 attention 图替代 AP。
**风险**：相机 BEV 本身可能已丢信息，采样无法找回；稀疏雷达下错配/过拟合。
**工程量**：中；最适合先做的 AP 能力验证，但直接移植已有 fusion 不算论文主创新。
**先例**：RCBEVDet（CVPR 2024）、CRN（ICCV 2023）。

### 方向 2：对象 query 驱动的雷达-相机检测器

**替换范围**：替换“先全图融合、再 dense anchor 分类”的检测组织方式。

- 每个 query 包含 3D reference point/box，从雷达几何候选与图像候选初始化；
  必须保留无雷达回波对象的图像候选通路，不能只以 radar proposals 初始化。
- 直接投影回原始多尺度 image FPN 采样，同时读取邻近 radar points/BEV。
- 迭代更新 reference box 后再次取证；分类、位置、尺寸/朝向共同预测。
- 根据对象尺度选择采样范围，而不是所有类别依赖同一已混合 cell。

**假设**：实例级反复取证比在固定 BEV 上堆分类 residual 更能保住小目标并改进几何。
**关键消融**：BEV-only query vs 原图+雷达 query、图像候选通路开/关、单次/迭代取证。
**风险**：新匹配/优化可能收敛更慢，小数据上过拟合；camera-only 的深度初始化不稳定。
**工程量**：大；是完整新检测器，不是 A2 的 logits residual，不用 ep6 单点判成功。
**先例**：RaCFormer（CVPR 2025）。其 query fusion 本身不是本项目新颖性。

### 方向 3：联合深度—高度表示，而非先平均成二维 BEV

**替换范围**：camera view transformation 与局部三维观测表示。

- 将有效 radar range 作为稀疏几何锚，学习多峰 depth distribution；
  雷达未覆盖像素仍由图像预测深度，不能把“无点”写成深度零。
- 保留多个高度切片或 sparse 3D feature volume，直到局部对象取证后再聚合。
- 对候选周围保留若干深度/高度假设，用图像与雷达一致性更新；
  不要求全场景建立高分辨率 dense 3D volume。

**假设**：depth/height 歧义影响 3D box IoU；明确空间假设可能比改分类分支有效。
**关键消融**：现有 lifting、只加 radar depth、只保留高度、联合 depth-height。
同时报 3D/BEV AP 及匹配对象的中心/高度/尺寸误差，避免把几何错误统称分类错误。
**风险**：显存/计算增长，雷达投影误差污染深度，多峰可能不被训练利用。
**工程量**：大；不是开启现有 `use_radar_depth` 或重复已有相对深度监督。
**先例**：CRN 的 radar-assisted view transformation、RaCFormer 的 depth 机制；
本段 depth-height 联合候选是本项目提案，先例不证明它在本数据上有效。

### 方向 4：所有类别联合训练的多模态 3D RoI 二阶段取证

**替换范围**：保留 coarse detector，加真正读取原始多模态证据的 joint refinement stage。

- 初阶段生成类别无关/宽召回 3D proposals；不得靠下调最终评测阈值伪造 AP 增益。
- 根据候选 3D box 将 RoI/grid points 投影到原图各层 FPN，读取局部 radar points，
  必要时读取短历史，做局部 cross-attention。
- 重估 class、center、size、yaw，并联合训练观测编码与两阶段检测；不是冻结主干的 Ped-only residual。

**假设**：粗网格定位和混合特征造成局部细节不足，重新取证能纠正几何与类别误判。
**关键消融**：BEV-only refinement vs 原图/雷达重采样、冻结 vs 联合训练、各模态去除。
**风险**：proposal recall 限制上限；首阶段完全漏掉的对象不能靠 refinement 恢复。
**工程量**：中到大；复用 coarse detector 比整套 query detector 改写更可控。
**先例**：PV-RCNN（CVPR 2020）的 point-voxel/3D RoI-grid pooling；
其原任务为 LiDAR，不能将其结果当作稀疏 radar-camera 的效果证据。

### 方向 5：对象中心、几何接地的 RSSM，直接生成检测结果

**替换范围**：将每个 BEV cell 的共享时序状态改为对象索引状态与实例观测。

- 对象保留独立 `(h_i, z_i)`，观测来自方向 2 或方向 4 的多模态局部取证。
- prior 根据 ego transform、对象几何/运动与历史预测当前状态；
  posterior 由实际局部观测更新，没有测量时不把速度/位置伪造为零。
- 检测直接读取状态生成 class/box，不仅在最终 logits 上加可忽略 residual。
- 预测下一帧的对象几何与局部观测，取代公共背景模板式全局重建目标。
- 可研究几何/关联的多假设状态：每个假设改变“下一次去哪采样、如何输出 box”，
  而不是随机状态只改变特征范数。

**假设**：对象粒度使 latent 的监督、空间对应和检测读出一致，
随机状态的可能职责是不可观测对象位置/关联歧义，而不是所有类别共享底色。
**关键消融**：同容量 object-GRU/no-z、无 temporal、单假设/多假设、
shuffle/zero z 对实际 box/CE/AP 的影响，部署均值与采样契约必须一致。
**风险**：跨帧关联出错、query 漂移、posterior collapse；不保证随机模型超过确定性。
**工程量**：很大；RSSM 论文主线潜力最高，先有可用实例观测器，再做时序状态替换。
**先例/边界**：StreamPETR（ICCV 2023）已有对象 query 时序；SlotFormer（ICLR 2023）
已有对象中心动态模型。`docs/rssm_paper_directions_20261008.md` §4/5 的候选 B/D
已提出对象中心和未来监督，
本段是实现路线深化，不宣称此前没人做过对象状态/未来预测。

### 方向 6：重写雷达观测编码：点—局部区域层级，而非只在 pillar 后补模块

**替换范围**：radar encoder 的早期观测聚合。

- raw points 保留细粒度几何与已核验物理特征，局部 point-to-region/voxel 交互先于粗 BEV 聚合。
- 局部区域提供多尺度可变范围，区分孤立点、稀疏对象点组与大对象支撑结构。
- 图像语义引导邻域取证，雷达空间约束图像深度；最终保持所有类别联合检测。
- 只使用已核实的数据列：选维后的最后一维是 SNR，不凭名字假定 RCS；
  原始径向速度未 ego compensation，使用前按 §77 的物理约束处理。

**假设**：弱目标证据在早期 pooling 中不够可分，改信息形成过程优于末端可靠性门。
**关键消融**：现有 RadarPillarFeatureNet、同容量早期局部交互、去掉 image guidance；
单独检查稀疏组 recall 与类 AP，不将多帧累计点数当作分类收益。
**风险**：雷达点本身不足时无法凭网络创造信息；噪声关联/计算成本可能增加。
**工程量**：大；不是再开已失败的冻结 Ped 0.16 m 支线。
**先例**：RCBEVDet 的 RadarBEVNet 提供雷达专用编码先例；
其 RCS-aware 分支不能未经通道核实直接移植到本配置。

### 条件方向 7：对象/几何级的跨模态 teacher-student 蒸馏

仅在训练 split 有可用 LiDAR，或已有确实更强的 teacher 时考虑。

- teacher 产生对象 RoI、3D 几何和局部语义分布；student 推理仍只用 radar+camera。
- 蒸馏实例语义/几何与对应关系，不要求弱雷达逐格复制强传感器背景特征。
- 单独报告额外训练模态/数据、teacher 性能与代价，严格禁止 val/test 泄漏。

**假设**：强训练监督改善弱观测表示与跨场景泛化，作用不同于增加普通分类辅助头。
**关键消融**：无蒸馏、仅 feature、对象/几何蒸馏；teacher 强于 student 是必要前提。
**风险**：本地数据/teacher 未必可用，公平性约束不同，不能声称纯同数据同监督增益。
**工程量**：中到大，另有 teacher 训练成本。
**先例**：DistillBEV（ICCV 2023）。单独蒸馏不是 RSSM 创新。

## 3. 建议先后顺序：性能升级与论文主线分开组织

1. **第一优先：方向 1**。做真正的 cross-position、多尺度 fusion，与当前 B0 配对，
   不把 F1 的 reliability gate 改个名字当成完成；性能提升后才加论文层机制。
2. **第二选择：方向 2 或方向 4，二选一**。前者重写 detector，后者复用 coarse detector。
   均从原图/局部雷达重新取证，不把新分类头作为替代。
3. **RSSM 主线：方向 5，与通过的对象观测器组合**。
   可研究“几何接地多模态观测 + 对象多假设随机状态”，但新颖性与必要性都需验证。
4. **方向 3/6 为独立高风险备选，方向 7 为数据条件备选**。
   不同时将七个方向塞进一个实验，不在一次新臂里偷改数据/训练配方。

大架构可以由必要的多个模块构成一个完整方法，不强制唯一变量只能是一个配置键；
先验证完整新方法的 AP，再用受控 ablations 拆贡献。
这不意味着取消 preregistration，也不意味着所有机制探针都得先做完才允许性能实验。
现有 A2-R/F1 的审批与队列状态不由本建议改变。

## 4. 训练与判定建议（尚未预登记）

- 配对主对照：同 no2d_igdr 配方、相同 seed/data/EvalHook；seed0 窗口 39.2809，
  全轮峰值 40.3853 @ ep14。不能拿新 seed0 对 seed1 的挑峰判断因果收益。
- seed1 参考：ep12–16 均值 41.9185、全轮峰值 43.1449 @ ep15（无 ep15 权重）。
  家族三 seed 成熟均值 40.4400 ± 1.3475；成功候选须做多 seed 配对。
- 6 epoch 用于 smoke、灾难/收敛检查，不认定结构成功或慢收敛结构失败。
  默认成熟对照 ep12–16（n=5）+ 所有 val epoch 的峰值及 epoch；完整记录逐 epoch、逐类。
- 新检测器若需要更长训练，两臂预算/窗口在训练前一起定，不事后挑候选有利窗口。
- 用户曾提出同 seed ≥+2.0 且多 seed 一致的“大幅”目标；
  新臂具体主指标与逐类容忍线仍须批准，不覆盖任何已冻结门。
  不承诺每个类、每个 seed 都涨；明显用 Ped/Cyc 大退化换 Overall 小涨不是优先候选。
- 能力对照、随机状态必要性消融用于判论文贡献，不让 feature 范数/辅助 loss 好看取代 AP。

## 5. 评测口径核验与文献边界

- 本仓库 TJ4D `clean_data` 先按距离赋 easy=0（0–50 m）、moderate=1（50–70 m），
  但忽略条件是 `gt_difficulty > difficulty`，不是 `!= difficulty`。
  因此 `moderate` 的有效 GT 是累计 0–70 m，**不是只有 50–70 m**。
  证据：`tools/eval_tools/TJ4D-eval.py:147–233` 与当前
  `mmdet3d/core/evaluation/kitti_utils/eval.py:147–233`；train_vod 会复制 TJ4D evaluator。
  距离使用 evaluator 的 GT location 范数，忽略/类 IoU 等其余协议也须保持。
- 用户提供的 63.60/31.24/62.84/31.46/47.29 出现在 R4Det 原论文
  arXiv:2603.11566v1 的 Table 1，也同值出现在 Table 4。
  Table 1 的文字称 test，§4.3 称 ablations 使用 validation；
  仅凭表格不能断言“47.29 是与当前 val 严格可比的 SOTA”。
  需核实 split、数据处理、训练配方与评测；本次不改评测，也不把该数值换算成收益。
- 文献只证明结构先例，不证明上述候选能提高本项目 AP；未声称完成新颖性全面检索。

原始文献（标题/会议用于后续核验）：
1. RCBEVDet: Radar-camera Fusion in Bird’s Eye View for 3D Object Detection，CVPR 2024。
2. CRN: Camera Radar Object Detection Using Multimodal Feature Fusion，ICCV 2023。
3. RaCFormer: Towards High-Quality 3D Object Detection via Query-based Radar-Camera Fusion，CVPR 2025。
4. PV-RCNN: Point-Voxel Feature Set Abstraction for 3D Object Detection，CVPR 2020。
5. Exploring Object-Centric Temporal Modeling for Efficient Multi-View 3D Object Detection，ICCV 2023。
6. SlotFormer: Unsupervised Visual Dynamics Simulation with Object-Centric Models，ICLR 2023。
7. DistillBEV: Boosting Multi-Camera 3D Object Detection with Cross-Modal Knowledge Distillation，ICCV 2023。
8. R4Det: 4D Radar and Camera Fusion for 3D Object Detection，arXiv:2603.11566v1。
