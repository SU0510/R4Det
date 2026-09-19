# 04 · ME-RSSM 设计规范（Motion-Evidence RSSM）

> 实现位置：`me_rssm/`（全部为新文件，未改动任何现有文件）。
> 模块：`motion_evidence_rssm.py`（核心）、`radar_motion_chain.py`（Doppler 证据链）、
> `stash_fusion.py`（输入发布）、`modality_bus.py`（边带通道）。
> 配置：`me_rssm/configs/TJ4D-R4Det_me_rssm_N4_24e_pretrained_v2_head.py` + 4 个消融配置。

---

## 1. 一句话

**让 Radar 的 Doppler 测量成为 RSSM 转移模型的控制信号与先验的预测依据，
让每像素可靠性决定"信观测还是信预测"——Camera 出语义观测、Radar 出运动观测、
RSSM 建模隐藏世界状态的传播。**

## 2. 数学形式（相对基线 MotionAlignedRSSMFusion 的增量）

基线记号：观测 feat（融合 BEV），状态 h,z ∈ R^{256×216×248}。

```
[Doppler 证据链（无参数）]   m_t = Scatter(pillar_stats(v_r, SNR, xyz)) ∈ R^{7×496×432}
                              m̃_t = permute(avgpool₂(m_t)) ∈ R^{7×216×248}   # 对齐 feat 画布
[证据编码]                   e_mot = MotionEncoder(m̃_t) ∈ R^{32×H×W}
                              e_cam = CameraEncoder(permute(cam_bev)) ∈ R^{64×H×W}
[可靠性]                     c_cam = σ(w_c1 ⊛ e_cam + b_c1),  c_mot = σ(w_c2 ⊛ e_mot + b_c2)
[转移]                       offset'_h = offset_h(feat, h_{t-1}) + M_h(e_mot)     # M 零初始化
                              offset'_z = offset_z(feat, z_{t-1}) + M_z(e_mot)
                              h_a = DeformConv(h_{t-1}; offset'_h),  z_a = DeformConv(z_{t-1}; offset'_z)
                              u' = u + tanh(β)·d·u·(1−u),  d = σ(w_d ⊛ m̃_t)        # 动态门控
                              h_t = GRU(z_a, h_a; u')
[先验]                       p(z_t) = N(μ_p(h_t), σ_p(h_t))                        # 继承
[观测]                       e = encoder(feat) + W_o ⊛ [c_cam⊙e_cam ; c_mot⊙e_mot] # W_o 零初始化
[后验]                       q(z_t) = N(μ_q(h_t, e), σ_q(h_t, e))                  # conv 形状同基线
[gain 校正]                  g = σ(w_g ⊛ [c_cam ; c_mot] + b_g),  b_g=+4 → g≈0.98
                              z_t = μ_p + g·(z_sel − μ_p),  z_sel ~ q（训练采样）/μ_q（推理）
[输出]                       recon = decoder(h_t, z_t);  out = output_proj(z_t) + feat   # 继承
[损失]                       KL(q‖p)（free-bits + warmup）+ MSE(recon, feat)       # 继承，不改
```

## 3. 七通道 Doppler 证据图（radar_motion_chain.py）

| ch | 量 | 含义 |
|---|---|---|
| 0 | vx = v̄_r·x/r | 伪 2D 速度纵向分量（径向速度 × 束方向单位向量） |
| 1 | vy = v̄_r·y/r | 伪 2D 速度横向分量 |
| 2 | v̄_r | 带符号平均径向速度 |
| 3 | \|v̄_r\| | 速度幅值 |
| 4 | v_std | pillar 内速度散度（混合目标/动态证据） |
| 5 | SNR̄ | 原始信噪比均值 |
| 6 | log1p(n/M)/log2 | 归一化占有点数（密度=可靠性代理） |

设计性质：
- **无参数**：PFN 子类只做统计聚合，学习权重与基线逐键相同（预训练直接加载）；
  scatter 子类均值散射，主画布与基线逐位一致（test_radar_motion_chain 验证）。
- **在 PFN 归一化之前计算**：父类 forward 会就地归一化 xyz，统计必须先取原始坐标
  （方向 (x,y)/r 对平移敏感，不可用归一化坐标）。
- **物理完整性**：静态物体在传感器系呈现 −v_ego 的径向流 → 证据图同时携带
  ego 补偿（全局外流场）与目标相对运动（局部异常流），一个 conv 即可读出两种模式。

## 4. 与基线的三重等价保证（"新结构 = 基线 + 三个零初始化开关"）

1. **权重等价**：基线 `MotionAlignedRSSMFusion.state_dict()` 可 strict=False 载入
   ME-RSSM，unexpected=0，missing 恰为新增层（E1：35 个新键）。
2. **空 bus 等价**：无边带上下文时输出/recon/KL/h_t 与基线 max|Δ|<1e-5（E2）。
3. **开关关闭等价**：bus 有数据但三个 use_* 开关全关 → 仍与基线等价（E3）——
   因为每条新通路都以零初始化权重或旁路进入（obs_proj 零、motion offset 零、
   β=0、g 旁路）。

这保证：(a) 即插即用无风险；(b) 消融梯度的最底格就是基线本身；
(c) 未来可从任一训练好的基线 RSSM checkpoint 微调启动。

## 5. 接口与工程契约

- **调用签名/返回值/reset 语义/kl_scale 属性**：与基线逐一相同 → 未修改的
  `R4Det.extract_feat/forward_train/simple_test` 直接可用（含 BPTT、burn-in、
  invalid-mask、KL hook、分布式）。
- **边带通道**（modality_bus.py）：producers（scatter、stash fusion）每帧写、
  temporal fusion 每帧读一次；`_read_context` 带形状/批大小/通道数三重守卫，
  任何不一致 → 安全降级为基线行为（绝不 crash 训练）。
- **画布约定**：证据图 (496,432)=(y,x) → pool(2) → permute → (216,248)=(x,y)
  与 feat 对齐（01 §1 的 ③）；deform offset 的 (Δw,Δh) 对应 (Δy,Δx)。
- **detach 策略**：m̃_t/cam_bev 默认 detach（`motion_grad=False`）——物理测量
  不被检测梯度改写；β、w_c、w_g、M、W_o 全部正常训练。
- **冷启动链（诚实声明）**：camera/motion encoder 与 conf conv 位于零初始化的
  obs_proj/gain 之后，第 1 步反传梯度为 0，第 2 步起激活（G1a/G1b 验证；
  这是零初始化残差的标准性质）。若需一步激活可把 obs_proj 改微小随机初始化，
  默认不这么做（等价性优先）。

## 6. 成本（实测，test_flops / test_config_build）

| 项 | 基线 | ME-RSSM | Δ |
|---|---|---|---|
| temporal 模块参数 | 10.57M | 10.99M | **+0.42M（+3.9%）** |
| 全模型参数 | 48.54M | 48.37M | −0.17M（shared_stem=False 的 clean 回退） |
| temporal conv MACs/帧 | 1006.1 G | 1050.6 G | **+44.5 G（+4.4%）** |
| 开关全关时 MACs | 1006.1 G | 1006.1 G | 0（精确基线） |
| Doppler 证据链 | — | 0 参数 | 纯统计+scatter |

## 7. 消融网格（config 即定义，等待后续公平训练评估）

| 配置 | 文件 | 隔离的变量 |
|---|---|---|
| full | `..._me_rssm_N4_24e_pretrained_v2_head.py` | 全部三机制 |
| −motion offsets (A) | `..._abl_no_motion_offsets.py` | 转移的运动条件化 |
| −gain (C) | `..._abl_no_reliability_gain.py` | prior fallback / 可靠性校正 |
| −dyn gate (D) | `..._abl_no_dynamic_gate.py` | 动静区分 |
| −modality obs (B) | `..._abl_no_modality_obs.py` | 整个边带观测模型（运行时基线） |

配套建议（写给未来实验，不是本次执行）：多 seed ≥3、窗口 ep12-16 均值、
预注册门控线沿用《审计报告》§8.8 方法论；modality_dropout 作为第二层开关
（0 / 0.1）单独消融。

## 8. 相关工作定位（论文叙事素材）

- **BEV 时序对齐**：BEVDet-4D 等用 ego pose 做 warp；本项目数据集无位姿，
  用 Doppler 径向流作为运动先验是**测量驱动**的对齐，区别于外观驱动的
  learnable alignment（基线）。
- **RSSM/World models**：Dreamer 系的 transition 无物理控制信号（agent action
  除外）；此处把"测量到的相对运动"作为 transition 的控制输入，是 RSSM 在
  感知（而非控制）场景下的运动感知改造。
- **Kalman/滤波视角**：learned per-pixel gain 把检测网络与概率滤波的
  predict-correct 结构显式连接；对比 standard deep Kalman filter，
  gain 的输入是**物理可靠性证据**（Doppler/SNR/密度）而非纯学习隐变量。
- **多模态可靠性**：implicit reliability（任务梯度学 c）vs 显式代理监督；
  modality dropout 给 fallback 通路训练压力，属于 robust fusion 的标准做法，
  但与运动感知转移组合后产生新语义（缺失观测时"雷达动哪我信哪"）。

## 9. 明确不做的事

- 不删/不改任何基线类与配置文件（研究协议约束）。
- 不动被多 seed 验证的机制组件（KL/recon/GRU/deform/输出残差）。
- 不做类别专属 head/分支（六次零和失败的教训）。
- 不以任何训练/验证结果选择结构——本文档全部论据为静态分析 + 机制等价性验证。
