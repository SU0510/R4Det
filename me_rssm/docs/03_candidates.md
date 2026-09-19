# 03 · 候选设计清单与筛选

> 筛选准则（研究目标给定）：解决 02 清单中的具体问题；与 Camera/Radar/RSSM 信息特性匹配；
> 信息入口/融合点/传播路径清晰；无信息泄漏、无重复建模、无无效参数；参数/计算代价合理；
> 与 RSSM 概率时序逻辑一致；可形成干净 ablation；可成为独立论文贡献。
> **禁止训练/验证/指标搜索**——筛选只依据结构分析与已有多 seed 实验证据（《审计报告》）。

## 候选总表

| # | 候选 | 解决的问题 | 新增参数 | 判定 |
|---|---|---|---|---|
| C1 | Doppler 运动证据旁路 + 运动条件化状态对齐/转移 | #2 | ~11k | **采纳（核心）** |
| C2 | 模态观测 + 可靠性门控 + Kalman 式 gain（prior fallback） | #4/#6/#8/#10/#11 | ~0.4M | **采纳（核心）** |
| C3 | 动态门控 GRU update（u' = u + β·d·u·(1−u)） | #9 | 8 个参数 | **采纳（并入 C2 实现）** |
| C4 | 模态置信门控替代 ConcatConvFusion（上游 gated fusion） | #1 | ~0.6M | 拒绝（与 C2 重复建模） |
| C5 | 静/动双状态流（h_static/h_dynamic 分离 GRU） | #9 | ~4M+ | 推迟（成本/收益劣于 C3） |
| C6 | 检测头注入不确定性通道（σ_q/KL/gain） | #12 | 小 | 推迟（零和红线风险） |
| C7 | 多尺度 BEV 时序（0.16m 支线时序化） | #13 | 大 | 拒绝（§38 已证伪该路线） |
| C8 | 首观测状态初始化（h_1 由 e_1 引导） | #7 | 小 | 推迟（二线，见 06） |
| C9 | 目标级时序轨迹状态 | — | 大 | 拒绝（无 track 标注，粒度不符） |
| C10 | 雷达外观 BEV（384ch）独立进 posterior | #1 | ~0.6M | 拒绝（与 fused feat 冗余；Doppler 旁路才是雷达独有信息） |
| C11 | 重建目标改造（动态残差重建 / depth-weighted recon） | #5 | 0 | 推迟（动被多 seed 验证的 recon 机制，风险不对称） |
| C12 | prior 一致性辅助损失（μ_p(t) vs μ_q(t+1)） | #4 | 0 | 推迟（新增 loss 调参面，BPTT 窗口仅 1 帧收益有限） |

## 采纳组合的论证

### C1：Doppler 运动证据旁路 + 运动条件化转移（对抗 #2）

- **为什么必须是"旁路"**：v_r 在 PFN 内被 64ch 线性混合（可学习装饰把运动与外观绑死），
  之后再想分离需要网络自己重新发现"哪些通道是速度"——这正是当前结构失败的方式。
  物理量应保留物理形态：在 PFN 内、学习装饰之前，按 pillar 聚合原始
  [vx̂, vŷ, v̄_r, |v̄_r|, v_std, SNR, density] 七通道，经**无参数 scatter** 直接成 BEV 证据图。
- **为什么接在转移而不是输出**：RSSM 的转移步是世界模型发生"预测"的地方
  h_t = f(h_{t-1}, z_{t-1})。没有运动信号时 f 只能学"记忆衰减"；有运动信号后，
  f 可以表达"目标沿相对速度移动"。这是 world-model 语义下 motion prior 的正确位置。
- **落点一：deform align 的 offset 加性项**。基线 offset 由外观预测（零初始化起步）；
  ME-RSSM 追加 `offset += conv(e_mot)`（零初始化）。物理含义：位移 ≈ 相对速度 × Δt，
  Δt 未知 → 尺度交给 conv 学。static 物体的 Doppler 流（−ego 径向流）为背景提供
  ego 补偿先验——这是无位姿数据集上唯一可行的 ego 运动补偿信号。
- **落点二：更新门调制（并入 C3）**。见下。
- **泄漏检查**：证据图由原始点云统计构成，不含任何 GT/未来帧信息；`motion_grad=False`
  默认把它当测量值 detach——检测 loss 不能反向改写物理测量（避免"为了检测好看而扭曲
  Doppler 证据"的自欺骗通路）。

### C2：模态观测 + 可靠性门控 + Kalman 式 gain（对抗 #4/#6/#8/#10/#11）

- **观测分工**（研究目标的核心句）：`e_cam = encoder(camera BEV)`（semantic observation），
  `e_mot = encoder(Doppler 证据)`（motion observation），`e_fused = encoder(fused BEV)`
  保留为基线路径。三者在 posterior 前汇合：
  `e_fused' = e_fused + obs_proj(cat[c_cam·e_cam, c_mot·e_mot])`（obs_proj 零初始化）。
- **为什么用零初始化残差而不是替换 posterior 输入**：(a) posterior conv 形状不变 →
  基线权重逐键兼容（E1 实测 0 unexpected）；(b) 训练从"当前已验证行为"出发平滑过渡；
  (c) 消融时可整段旁路。
- **可靠性 c_cam/c_mot**：由各自观测特征经 1×1 conv + sigmoid 产生（零初始化 + 正 bias →
  c≈0.88 起步）。监督信号来自任务本身：某模态在某像素"有用"时，梯度会抬高其 c——
  这是 implicit reliability learning，无需任何标注（对比显式监督 SNR/深度方差等代理，
  免除代理偏差）。
- **Kalman 式 gain**：`z_t = μ_p + g·(z_sel − μ_p)`，`g = σ(conv([c_cam, c_mot]))`
  （bias 初始化 +4 → g≈0.98，等价于先验只在观测差时介入）。这是把 Bayesian filter 的
  "gain 由测量可靠性决定"搬进 latent 空间：观测可信 → z=后验（当前行为）；
  观测不可信/缺失 → z=先验=运动传播的预测（新增的 fallback 能力，G5 已验证）。
- **与 #4 的关系**：gain 使 prior 第一次在推理路径上有角色；KL 把"预测-校正差"
  （innovation）约束在自由比特之上——collapse 不再是纯退化，而是"预测已被接受"的稳态。
- **modality dropout**（训练期随机置零 c_cam/c_mot，默认 0 可开）：给 fallback 通路
  制造训练压力，是 reliability 门控成立性的必要配套（否则 g 恒 ≈1，fallback 分支永远
  没有梯度——冷启动问题的直接解法）。

### C3：动态门控 update（对抗 #9）

- 证据图聚合出 `d = σ(dyn_conv(vel))`（速度散度/幅值高 → 动态）；
  GRU update gate 调制为 `u' = u + tanh(β)·d·u·(1−u)`。
- **为什么这个形式**：(a) u·(1−u) 保证对任意 β∈(−1,1)（tanh 界）u'∈(u²,1)⊂(0,1)——
  门仍是合法概率，数值安全无需要求训练自律；(b) β 零初始化 = 基线行为；
  (c) 语义：动态区加速接受新观测、静态区延长记忆——与 #9 的病灶逐点对应；
  (d) 相比双状态流（C5），1 个标量 + 7 通道 1×1 conv 就完成"区分"的一阶近似。
- **与 C1 的关系**：d 与运动 offset 共用同一 Doppler 证据图（e_mot 的源头），无重复参数。

### 为什么不采纳 C4（上游 gated fusion）

研究目标明确警告重复建模。置信门控在 C2 已于**观测层**实现；再在上游融合层做一遍，
同一 reliability 信号出现两次、两处梯度路径竞争，且上游改写 fused BEV 会破坏
backbone/检测头已对齐的特征统计（§41 shared stem 的 Cyclist 红线教训：改共享特征
分布的代价难以预料）。 ConcatConvFusion 保持原样（连同预训练键兼容），
**模态特异信息以旁路进入时序级**——信息流图上更清晰：融合 BEV 负责"现在长什么样"，
Doppler 旁路负责"现在正在怎么动"。

### 为什么推迟 C5/C6/C8/C11/C12

- C5：双 GRU = 时序级计算翻倍（+~200G MAC），而 C3 已用 <1% 代价占据大部分收益；
  且无动态标注监督分离边界，风险高。列为二线（06）。
- C6：head 侧改动在项目历史中六连败（零和），且本方案的不确定性信号（g、σ_q）
  尚未经训练检验，先观察 stat_* 再决定。列为二线。
- C8/C12：都是好方向但独立成篇贡献不足；在 C1/C2 机制经训练检验后再作为增量。
- C11：recon 是被三 seed 验证过的机制组件，改目标函数的收益不可静态推演，风险不对称。

## 论文贡献切分（对应实现的 ablation 面）

1. **Doppler-conditioned state transition**（C1）——"radar 不再是第二种外观特征，
   而是转移模型的控制信号"。
2. **Reliability-gated probabilistic correction**（C2）——"learned Kalman gain 使
   prior/posterior 分工成立，模态缺失时优雅降级到运动预测"。
3. **Dynamic-gated memory update**（C3）——"Doppler 动态证据区分快慢状态更新"。
   三者共享一条 Doppler 证据通路（实现上是一个模块 + 三个 config 开关），
   ablation 网格：full / −motion offsets / −gain / −dyn gate / −modality obs（= 基线）。
