# 09 · 新颖性定位对照（2026-09-19 文献核查）

> 目的：在 ME-RSSM 主线训练（seed0）出结果前，把"与已有方法的区别"从
> 静态论断升级为**核对过的文献对照**，防撞车、补引用、明确差异化叙事。
> 检索关键词：4D radar camera BEV fusion / Doppler temporal fusion /
> RSSM·SSM detection / probabilistic latent fusion（2023–2026）。

## 1. 最接近的工作与差异

| 工作 | 模态 | 时序形态 | 运动信号用法 | 概率状态? | 与 ME-RSSM 的关键差异 |
|---|---|---|---|---|---|
| RCFusion (T-ITS'23, 233+引用) | radar+cam | **单帧** | Doppler 用于预处理/定位 | 无 | 无时序滤波；Doppler 只是特征，不进转移 |
| TPFE / Kinematic-Comp TF (IEEE'26) | radar(-cam) | pillar flow | flow 补偿切向速度缺失 | 无 | 补偿几何测量缺口；无 prior/posterior、无可靠性门控 |
| **CRT-Fusion (NeurIPS'24)** | radar+cam | 递归融合 | MFE 估计逐像素速度+占据，引导 MGTF 对齐融合 | **无（确定性）** | 最强相关：运动引导的**确定性**递归融合；无 RSSM/KL/prior；无"观测不可信回退预测"机制；用 nuScenes（有 ego pose），ME-RSSM 用无位姿 TJ4D + **原始 Doppler 直接进转移** |
| RecurrentBEV / VideoBEV | camera | 递归 | ego pose warp | 无 | 纯相机、外观对齐 |
| MambaFusion / MambaBEV | lidar/cam | SSM 序列 | 无（外观） | 无（确定性 SSM） | SSM 用于序列建模效率，非滤波语义 |
| DriveWorld / BEV world model (CVPR'25) | cam | 世界模型预训练 | — | 潜变量但用于表征预训练 | 不是检测时的 predict-correct 滤波 |
| **ME-RSSM（本包）** | radar+cam | RSSM 滤波循环 | **无参数 Doppler 证据旁路直接进转移/门控** | **有：prior/posterior + Kalman 式 gain + 可靠性观测** | — |

## 2. 核对结论

1. 未发现"RSSM/概率隐状态 + 雷达 Doppler 证据 + BEV 检测时序融合"三要素
   组合的已发表工作；最接近的 CRT-Fusion 是确定性运动引导递归融合。
2. 差异化叙事主线（论文 intro/related work 骨架）：
   - 已有工作把 Doppler 当**特征**（RCFusion）或当**对齐引导**（CRT-Fusion/TPFE）；
   - 本工作把 Doppler 当**转移模型的控制信号**（predict）+ **可靠性证据**
     （correct 的 gain 输入），使 RSSM 从"第二种外观记忆"变成
     **雷达告知的 predict-correct 滤波器**；
   - 数据集无 ego pose/无时间戳 ⇒ 测量驱动对齐（对比 BEVDet-4D 类位姿 warp）
     本身就是贡献点而非妥协。
3. 需要在论文中正面引用并区分：CRT-Fusion（NeurIPS'24）、RCFusion、
   TPFE/Kinematic-Comp TF、RCBEVDet（CVPR'24）、RecurrentBEV、
   MambaFusion/MambaBEV（SSM 但确定性）。
4. 风险标注：CRT-Fusion 的 MFE 也"估计逐像素速度"——审稿人可能问
   "为什么不直接用他们的 MFE？"答：MFE 是**学习估计**的速度头，ME-RSSM
   用的是**无参数原始 Doppler 统计**（测量而非估计），且服务于概率转移
   而非特征对齐；这一点写进 rebuttal 预案。
5. CRT-Fusion 摘要级核对（2026-09-19 抓取 arXiv 2411.03013v3）补充确认的
   三点差异，rebuttal 直接可用：
   - MFE 需要两条**辅助监督**（速度估计 + BEV 分割）；ME-RSSM 不加任何
     新辅助任务，Doppler 统计是**前向无参数计算**（测量 vs 估计）；
   - MGTF 是**确定性**递归对齐融合——没有 prior/posterior 分解、没有
     观测可靠性门控、没有"观测不可信回退运动预测"的语义；ME-RSSM 的
     Kalman 式 gain 在观测缺失/低可靠时有**定义明确的降级行为**（G5）；
   - CRT-Fusion 在 nuScenes（有 ego pose + 时间戳）上评测；TJ4D 两者皆无，
     时序对齐只能靠测量驱动——问题设定本身不同。

## 3. 来源

- [CRT-Fusion, NeurIPS 2024 (arXiv:2411.03013)](https://arxiv.org/abs/2411.03013)，
  [官方页](https://neurips.cc)、[代码](https://github.com/mjseong0414/CRT-Fusion)
- [RCFusion, IEEE T-ITS 2023](https://ieeexplore.ieee.org/abstract/document/10138035)
- [TPFE / Kinematic Compensation Temporal Fusion, IEEE 2026](https://ieeexplore.ieee.org/iel8/7083369/7339444/11675980.pdf)
- [RCBEVDet, CVPR 2024](https://arxiv.org)（radar-camera BEV，RadarBEVNet）
- [RecurrentBEV, ECCV](https://eccv.ecva.net)；MambaBEV/MambaFusion（arXiv 2025-26）
- [世界模型综述, arXiv 2502.10498](https://arxiv.org/pdf/2502.10498)
