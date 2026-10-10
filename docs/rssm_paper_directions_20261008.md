# RSSM 论文级修复候选：让 z 成为目标证据，而非公共底色

> 2026-10-08；研究建议，未实现、未训练、未宣称性能收益。
> 本文基于训练记录 §49–55、§68.9e/f、`me_rssm/mci_rssm.py` 与原始论文。
> 所有名称是工作描述，不是已确认的新颖性主张。已有论文的方法基础须引用；
> 雷达-相机可观测性、实例随机状态、检测因果读出的结合仍需完整近邻检索。

## 1. 问题边界与已有证据

- z 并非完全失效：MCI-v2 ep16 三窗 shuffle_z 对 head 影响约 3.97–4.86%，
  zero_z 约 19.6–21.9%；目前仍以公共底色作用为主。完整证据见 §68.9e/f。
- MCI-v2 Overall 固定窗口 ep12–16（n=5）39.3786，全轮峰值
  42.9447 @ ep7；Ped loose 同窗 23.0982，全轮峰值 29.0721 @ ep12。
  单 seed 结果不是改进路线已经有效的证据。
- `R4Det.extract_feat` 默认训练采样、验证确定性；不能误说训练也只用均值。
- `mci_rssm.py` 当前 posterior 条件化 innovation，校正门调制 correction，
  读出门调制整条 latent residual。弱 Doppler 与时序校正之间存在负迁移嫌疑，
  但 Ped 区具体因果未证实。
- 历史已经尝试低维 latent、全图差分重建、batch-shuffle 重建排名、FiLM
  改直连、增大 z readout、innovation posterior 等；不能把“加对比损失”
  或“换一个直接读出”重新包装成未经试验的创新。§53.5 的排名项失败于
  低维 correction/decoder 通路，本文候选必须改变监督粒度与检测读出关系。
- “提升分类”与“提升每类 3D AP”分开：后者还包含几何、召回、排序、NMS。
  Ped strict 历史几何瓶颈不会被语义 latent 自动消除。

## 2. 优先候选 A：检测接地的语义 residual latent

**假设**：z 缺少与分类决策直接连接的前景语义监督；全图重建允许背景/共模
占优，通用 residual 也允许 head 绕开样本特异内容。

**最小改变**：保留已验证的 full-resolution RSSM、prior(h)、重建与检测基路，
先只在正样本/候选目标区域读取 latent，不换 backbone、不换整套检测头：

```
v_i = ROI_or_positive_anchor_pool(z_t, proposal_i)
logits_i = logits_base_i + alpha_i * W_cls(v_i)
```

训练加入前景、类别均衡的 z-only 语义预测；对象跨帧一致性只在关联可信时使用。
分类 residual 与几何 residual 分离。最后合并投影可零初始化，但不得让零输出
层再直连零初始化层；新增语义头应单独有可训练通路。

**与旧失败不同**：完整空间 latent；对象/正 anchor 粒度；检测 logits 路径；
有监督类别目标，而非 16×16 latent 经 decoder 拟合全图差分 MSE。

**潜在论文贡献**：任务接地的 innovation state 与语义读出，证明 latent 的
样本/类别信息改变真实检测决策，而非只改变特征范数。

**消融**：相同辅助监督施加给当前观测而非 z；有 z 语义监督但无检测读出；
有读出无监督；同参数量确定性 adapter；随机历史/单帧 latent。

**机制门**：z-only 前景分类；正常 latent 与跨类 shuffle 的真实 CE/检测差；
同类 shuffle 不要求分类必变，应检查实例关联、定位/未来预测。仅表征方差
变大或 shuffle 输出差变大不算成功。

**风险**：z 只变成当前帧分类 encoder 或类别原型；因此必须通过历史打乱与
确定性对照。VICReg 方差/协方差正则仅作可选辅助，不单独充当创新或成功证据。

## 3. 候选 B：对象中心的 RSSM，而非整幅 BEV 随机贴图

**假设**：背景占比与隐状态旁路使 z 难以承担实例身份与遮挡补全；随机状态
应对应目标，而不是所有 BEV cell 共享的大背景表示。

**结构**：保留 dense BEV 当前帧与既有时序基路，增加小规模 proposal/ROI 状态：
每个实例维护 h_i、q(z_i)、p(z_i)，接受相机语义、雷达几何/速度与历史匹配输入；
对象 latent 写回分类/局部几何 residual。第一版只在已有 positive anchors 上
验证对象监督有效，再决定是否升级为跨帧 slot bank，避免直接重写检测器。

**训练**：前景语义、对象局部重建、可信跨帧对象对应。存在可靠 ID 才用 ID；
无 ID 时需显式验证几何/外观关联质量，并屏蔽歧义，不把伪关联当 GT。
GT 正样本只用于训练监督，不作为验证/推理输入。

**潜在贡献**：雷达-相机对象 belief，观测缺失下维持身份并显式表达状态不确定性。
对象中心与 temporal contrast 本身已有 prior art，不能宣称首次。

**消融**：dense z vs 对象 z；相同容量 GRU vs RSSM；仅当前帧对象 encoder；
对象中心但无时间一致性；有一致性但无 latent 检测读出。

**预期**：Ped/Cyc 可获得对象级容量和更少背景污染；Car/Truck 仍用现有 dense
基路。所有收益只是待检验假设，不保证四类同时提升。

**风险**：proposal recall 卡死小目标、关联错误传播、slot ID switch。需保留
新目标 birth 路径与当前帧基路；别只维护高分旧目标。

## 4. 候选 C：可观测性/不确定性校准的多模态 predict-correct

**假设**：当前门把“没有雷达测量/径向速度不可观测”误当成“没有有用时序信息”，
并用未标准化 innovation 幅值调制全部 correction/readout。

**核心改动**：相机与雷达分开形成 innovation 与不确定性，不先合并成一个
误差；区分无测量、测到弱运动、测量与预测冲突三种状态。用有效维度中的
标准化 innovation 作一致性信号：

```
innovation_m = observation_m - predicted_observation_m
S_m = predictive_observation_covariance_m + measurement_covariance_m
normalized_innovation_m = innovation_m.T @ solve(S_m, innovation_m)
```

若采用高斯信息融合，应在共同、对齐的状态空间定义 observation model；
非线性高维 BEV 不能直接冒称精确 Kalman 滤波。参数化 covariance 用正值/PSD
约束并经 NLL/校准评估，不允许靠无限放大方差逃避损失。

无 radar coverage 时雷达 likelihood 不提供信息，不把速度赋零，也不同时关闭
camera correction。当前观测/检测基路保留，但 residual 保留并不构成数学上的
性能不下降保证。count/SNR 可作为 coverage 证据，不自动等于正确率。

**潜在贡献**：缺测与矛盾观测的可区分 belief update，以及跨类别可观测性
差异下的可靠性校准；比“固定物理图乘学得门”更明确的统计语义。

**与 DC-RSSM 的关系**：证据场是基础，但仅活化门输入不直接解决 z 的语义
读取。拟议升级是 modality-specific innovation + 校准不确定性 + camera
fallback，不是立即把五层 DC 设计全部叠上。

**消融**：联合 vs 分模态 innovation；原幅值 vs 标准化 innovation；
unknown coverage 是否区别于 static；保留当前观测 vs 全局门控；
测量条件化确定性滤波 vs 概率 RSSM。

**诊断**：按 class/coverage/运动可观测性分层的 NLL、校准、召回与 AP，
雷达缺失/噪声注入下 camera 贡献，以及 Ped GT 区门图。
标定常数用训练集或训练内 calibration split，不用 val/test 来拟合。

## 5. 候选 D：对象条件化的排他未来目标与反事实验证

**假设**：当前观测和 h 能解释重建目标的大部分内容，z 没有不可替代职责。

保留当前观测重建与 prior(h)，让 z 承担对象未来可见性、局部语义、占据或
位移的不确定信息。预测器第一版只读对象 z；未来观测作为 stop-gradient
训练目标，推理不得访问未来帧。未来任务样本前景均衡，可靠关联/对齐不可省。

“只有预测器读 z”不是绝对排他：z 仍可能复制 h 或当前帧。必须比较
仅当前观测、仅 h、相同容量确定性时序模型，测量 z 在固定观测/h 后的
条件增益；分析的是检测收益和真实预测误差，不只是随机置换特征差。

**潜在贡献**：检测接地的对象 belief 预测，辅以可重复的条件因果检验。
未来预测/latent overshooting 已有 PlaNet 等先例，创新需要落在本任务
的对象级测量条件化与检测读出，而非“首次预测未来”。

**消融**：全图 MSE vs 对象未来语义/可见性；one-step vs multi-step；
纯 h、纯观测、z-only；有未来监督但无 z 检测读出；保持其他目标一致。

**风险**：历史低维排他未来路线已失败；不能再删重建、让 prior 脱离 h，
或同时压低空间分辨率。未来标签/对应不足时，这条路线后置。

## 6. 建议顺序与不做事项

> **2026-10-10 更新**：下面排序是 10-08 的研究建议。结合 §69 和源码复核，
> 先做真实重建目标/梯度契约审计与有限目标改革验证；若目标信息不进入 z 或
> 任务则转 A。A/C 不同时引入，B/D 后置。95.5% pool16 不是现役 DC 实际目标。
> 完整依据见 `docs/rssm_next_step_audit_20261010.md`；未批准新训练。


1. 先完成现有证据场路线的预登记判定；本次不干扰任何正在运行的作业。
2. 新实验先试 A 的最小版本，确定“语义信息可到达 z、z 可改变检测决策”。
3. C 专门解决弱/缺测类别的负迁移；A 与 C 各自通过后才考虑组合。
4. B 是论文架构升级，先做对象 ROI 试验，再做关联/slot bank。
5. D 在对象对应和 prior 可预测性站稳后再引入。

不优先：再调 KL/free-bits、只换 categorical、只增大 z、换 Mamba/Transformer、
叠 SE/CBAM、仅加所有类别分类辅助头。这些不能单独解释已有信息/读出断点；
并非永远无效，而是当前缺少可证伪机制理由。

**候选主线**：对象语义接地的随机状态 + 缺测可识别的测量更新 + 检测 residual
读出。各项独立试验，不一次堆入一个“最终模型”。

## 7. 接受条件（建议，尚未作为新 run 的正式预登记）

- AP：按 no2d_igdr 同 seed 配对；固定 ep12–16、五轮等权，先预登记最小收益
  与各类非劣容忍值。最终至少三 seed；每类均值提升与跨 seed 一致性分开写，
  不要求每个 epoch/seed 均上涨。若目标是严格四类提升，不用 Overall 掩盖负类。
- 报告 Overall、Car/Truck strict、Ped/Cyc loose，另列 Ped strict；完整曲线、
  窗口各项均值、全轮峰值“值+epoch”、best saved 全部保留。
- 分类：固定 GT/正样本或共享 proposal 的诊断评估分类 CE、混淆、置信校准；
  官方 3D AP 另算，不能把定位/召回改善误报成分类改善。
- 机制：正常、shuffle、zero、prior、历史打乱对真实任务误差/AP的作用；
  foreground/object 内进行统计；只在 train 上训练的 z-only probe。
- 概率价值：RSSM 比相同容量确定性对象 GRU 的预测 NLL/校准、缺测鲁棒性、
  检测收益；KL 非零或方差非零都不是“随机状态有用”的充分证据。
- 初始化/接线：一次性核验所有新增参数梯度、loss 接入、对齐、状态重置和
  干预覆盖范围；冒烟通过后再正式训练。GPU 优先 5/6/7。

## 8. 原始文献（用于方法基础，不证明本文候选必然有效）

- Hafner et al., Mastering diverse control tasks through world models, Nature 2025
  (DreamerV3); RSSM 世界模型参考，不是雷达-相机检测收益证据。
  https://www.nature.com/articles/s41586-025-08744-2
- Manasyan et al., Temporally Consistent Object-Centric Learning by Contrasting Slots,
  CVPR 2025; 对象时间一致性与 temporal contrast。
  https://openaccess.thecvf.com/content/CVPR2025/html/Manasyan_Temporally_Consistent_Object-Centric_Learning_by_Contrasting_Slots_CVPR_2025_paper.html
- Bardes et al., VICReg, ICLR 2022; 表征方差/协方差反塌缩，不保证检测与随机状态收益。
  https://arxiv.org/html/2105.04906v3
- Kim et al., CRT-Fusion, NeurIPS 2024; 用对象运动/占据预测进行时序 BEV 对齐融合。
  https://proceedings.neurips.cc/paper_files/paper/2024/file/c49a28241640407b23bba8f2495f4bc9-Paper-Conference.pdf
- Hafner et al., Learning Latent Dynamics for Planning from Pixels, ICML 2019;
  确定性+随机状态、多步 latent overshooting。
  https://proceedings.mlr.press/v97/hafner19a.html
