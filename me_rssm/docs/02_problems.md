# 02 · 结构问题清单（对照研究目标的 14 项检查）

> 每一项给出：结论（是/否/部分）→ 代码证据（引用 01 号文档的行号事实）→ 建模后果。
> 本清单是 03（候选设计筛选）的输入；未被任何候选解决的项目会明确标注"接受为约束"。

| # | 检查项 | 结论 | 证据与后果 |
|---|---|---|---|
| 1 | Camera 与 Radar 融合过早导致 modality 特性丢失？ | **是** | ConcatConvFusion 在时序级之前把 256+384 压成 256（R4Det.py:861-863）。此后 RSSM 观测、KL、recon、检测头只见混合体。Doppler 独有信息与模态可靠性在进入时序模型前不可逆丢失。 |
| 2 | Radar velocity 没有真正进入 temporal dynamics？ | **是（核心问题）** | v_r 进 PFN 后被两次外观化（pillar_encoder.py:598-603 装饰 → 64ch 线性混合；SECOND 再混合）。ConvGRU 转移无任何运动信号；action/ego 分支因数据集无位姿被移除（action_dim=0，主配置:212）。运动适应只剩 deform align——由**外观**预测 offset，而非由**物理测量**驱动。 |
| 3 | RSSM state 只是形式上的 recurrent feature，没有真正的 probabilistic state modeling？ | **部分** | 概率机制齐全且多 seed 证明有净贡献（删任一组件不稳，《审计报告》§8.5）；但 posterior collapse 是实测稳态（clamped_ratio→100%），prior≈posterior，"概率"退化为正则化而非"预测-校正"。根因见 #2/#4。 |
| 4 | prior/posterior 没有明确作用？ | **是** | prior 仅作 KL 正则器；推理路径完全不用 prior（simple_test 只取 μ_q）。转移模型不具备预测能力 → prior 无预测内容可提供 → collapse 是结构必然而非训练偶然。 |
| 5 | deterministic/stochastic state 信息重复？ | **部分** | h 与 z 都是 256ch 全画布外观记忆；recon 目标同为 fused BEV（rssm_fusion.py:711），两者分工（传播 vs 校正）名义存在、实质未引导。h 的容量被静态背景重建占用（见 #9）。 |
| 6 | temporal state 与当前 observation 融合方式不合理？ | **部分** | posterior concat+conv 本身合理；不合理处在于**输出端**：`output=output_proj(z)+feat` 无任何置信加权，检测头无法区分"来自预测"与"来自当前观测"的贡献；观测退化的像素与干净的像素被同等信任。 |
| 7 | sequence reset/state initialization 不合理？ | **是（轻）** | h_0=z_0=0 且不从观测初始化（rssm_fusion.py:666-670）；N=4 下前 1-2 帧状态近死。invalid 历史的处理（reset_for_samples + 当前帧 torch.where 回退未融合 feat，R4Det.py:898-901）是"丢弃"而非"降级"——没有纯预测路径可走（因为先验无预测能力，见 #4）。 |
| 8 | 历史信息与当前信息权重缺乏明确机制？ | **是** | 唯一机制是 GRU update gate，纯学习、无物理/可靠性依据；遮挡、雷达稀疏、深度歧义等退化情形与正常情形同权。 |
| 9 | 动态目标和静态背景没有区分？ | **是** | 单一状态流 + 单一 deform align offset 场：静态背景（本应几乎不 warp）与动态目标共用一个对齐机制；recon 每帧强制重建不变的背景（浪费 h 容量，见 #5）。雷达 Doppler 本可区分二者，但被埋没（#2）。 |
| 10 | 多模态 confidence 没有利用？ | **是** | 全网络无 reliability/confidence 通道。SNR、点密度进入特征但没有显式的"这个观测有多可信"表征参与任何决策。 |
| 11 | 缺失 modality 时模型没有合理 fallback？ | **是** | 仅整帧 invalid 有处理（置零状态）；无单模态退化机制。基线 RSSM 即便想做 prior fallback 也没有意义（prior 无预测内容）。 |
| 12 | detection head 丢失 temporal uncertainty？ | **是（记录，暂不修）** | 检测头只见 `pts_feats[0]`；σ_q、KL、gain 等不确定性信号全部止步于 temporal 模块。修复需改 head 输入维度——历史上六次"类别专属 head 改动"均触发 Cyclist 红线（《审计报告》§9.1-1），收益/风险比差，列为二线候选。 |
| 13 | 计算量集中在低价值模块？ | **部分** | 时序级 ~1006 GMAC/帧（conv-only 实测，见 05）：GRU 190G（有据：三 seed 证明机制有效）；prior 两 conv ~126G 在 collapse 态边际价值低，但删除它已被多 seed 否证（§8.5）→ 冻结不动。新增结构应把计算花在**信息增量**处（Doppler 通路极便宜：+44.5G）。 |
| 14 | 网络存在结构性冗余？ | **是** | (a) align_h 与 align_z 两套独立 deform aligner 处理高度相关的状态（~1.2M 参数重复）；(b) prior/posterior μ conv 形状相同、collapse 态下学近乎同一映射；(c) 相机 256 与雷达 384 均先充分扩容再压缩融合。其中 (b)(c) 属于"已验证机制的可容忍代价"（动它们=动被多 seed 验证的部分），(a) 是二线候选。 |

## 汇总：值得结构干预的排序

1. **#2（velocity→transition）** —— 唯一的"信息已经存在但结构上到不了该到的地方"的案例；
   也是让 #3/#4 从"正则化"升级为"预测-校正"的先决条件。**第一优先。**
2. **#8/#10/#11（reliability 与 fallback）** —— 同一根因：没有可靠性表征。一旦 #2 给了先验
   预测内容，可靠性门控让 prior/posterior 分工自然成立。**第二优先（与 1 组合设计）。**
3. **#9（动静态区分）** —— Doppler 场天然提供 dynamic evidence；用最小代价（调制 GRU update gate）
   拿走大部分收益，避免双状态流的高成本方案。**第三优先（并入 1+2 的实现）。**
4. #1（融合过早）的完全解法（模态独立 BEV 主干）成本极高且与预训练对齐；本研究的答案是
   **给时序级旁路一条未混合的 Doppler 通路**（旁路而非重构），融合主干保持基线。
5. #7（状态初始化）、#12（head 不确定性）、#14(a)（对齐器合并）→ 二线，见 06_second_layer.md。

## 接受为约束（不做无谓对抗）

- **多 seed 验证过的机制不可拆**：KL+free-nits+warmup、posterior sampling、recon、残差输出、
  ConvGRU、deform align 骨架 —— 全部原样保留（历史教训：单删任何一件在多 seed 下不可靠）。
- **检测头共享 BEV 的零和瓶颈**：不做任何类别专属回归分支（六次同模式失败）。
- **数据物理约束**：无 ego pose、无 timestamp → 运动信息只能来自 Doppler 本身；
  Δt 未知 → 运动到位移的尺度必须可学习。
