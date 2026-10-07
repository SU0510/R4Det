# 13 · DC-RSSM：Doppler-Calibrated Recurrent State Space 设计文档

> 状态（2026-10-07）：设计定稿，MCI-v2（第 0 步）已在训；L1/L2/L3/L4 待实现。
> 证据链：`docs/training_runs_full.md` §68.8i/68.9b（门控空间失明、Ped 死区
> H1、innov 通道死锁）+ §68.10 结论汇总。原则沿用 me_rssm 全套纪律：
> 零初始化/恒等启动、单变量开关、450 iter 机制门 → ep6/ep16 AP 门、
> 多 seed 配对终判。

## 0. 设计原则（从诊断数据推导，非堆料）

三份诊断把设计空间划死：

| 诊断（证据） | 指出的设计缺口 |
|---|---|
| 门控空间失明：c_mot/dyn 塌缩为空间常数，门=全局标量（§68.9b，40 样本探针） | 可靠性必须有**测量推导的空间结构**，不能指望检测梯度自发产生 |
| Ped 死区：动态但 Doppler 测不到（0.145），innovation 后验对它注入噪声（H1 证据链：MAI −3.40 / MCI −4.07 / pdrop10 −3.53） | 需要**相机主导的第三条路由**与**coverage 语义**（无测量 ≠ 静态） |
| 对齐纯外观驱动；静态背景占用全部记忆容量（§52：目标 76–92% 可由 h_t 预测）；docs/08 静态点呈现 −v_ego 流场 | ego 运动可以从 Doppler **解析自标定**；状态需要**测量设定的时间尺度** |
| innov 通道死锁（零乘积形态） | 初始化纪律必须正式化（§4） |

**统一原则：凡是学出来的全局行为，换成一个测量接地的机制；学习只做零初始化的残差细化。**

## 1. 架构总览

```
L1 解析 ego-warp      v_ego = argmin Σ w_p(v_r + v_ego·û_r)²（静态点，闭式解）
                      → 静态分量解析回 warp（detached）→ deformable 只处理残余
L2 双时间尺度状态     h_slow_t = h_slow + (1−d)·ρ·(h_t − h_slow)（d=动态证据，ρ 标量）
                      h_t' = h_t + w_s⊛h_slow（w_s 零初始化 1×1）
L3 证据场             物理层（无参数）：c_mot⁰=sat(log1p(count)·SNR̄)、
                      dyn⁰=sat(v_std·count)、cov=soft-count
                      学习层（零初始化细化）：c_mot = c_mot⁰ + conv⁰([c_mot⁰,dyn⁰,cov])
L4 创新一致性记忆     e_t = ρ_c·e_{t-1} + (1−ρ_c)|inn_t|²（ρ_c sigmoid 有界，0.8 起步）
                      → 进门控输入；可选 offset 搜索范围调制
L5 存量（MCI-v2）     innovation 后验 + [c_cam, c_mot, |inn|↓] 门控 correct/readout
                      + Doppler 对齐偏移 + 动态更新门（innov 通道已复活）
```

数据流不变：`feat → (L1/L2 状态) → L5 forward → 六元组`；全部新路径遵循
恒等启动（见 §4）。

## 2. 各层设计细节

### L1 解析 ego-warp（新设计，旗舰）

- 静态点集：`S = {cell: count≥1, v_std ≤ τ_s, |v̄_r| ≤ τ_v}`（τ 由分位数定，
  无超参扫描）。
- 闭式解：`v_ego* = argmin_{v∈R²} Σ_{p∈S} w_p (v̄_r(p) + v·û_r(p))²`，
  w_p = count·SNR 归一化；2×2 正规方程，逐帧解算，**detached（测量）**。
- 位移场：`Δp = v_ego·Δt`，Δt 为**可学标量**（TJ4D 无时间戳——诚实设计），
  初始化取名义帧间隔；warp 施加于**慢通道 h_slow 与静态分量**（双线性，
  grid_sample，detached），deformable 对齐只看残余。
- 参数：+1（Δt）。novelty：pose-free 雷达-相机 BEV 的检测时自标定 ego 补偿
  （CRT-Fusion 依赖 nuScenes 位姿；本设定下是必需而非选择——docs/09）。
- 恒等启动：`warp_scale` 标量零初始化（乘在 Δp 上），训练中长出。

### L2 双时间尺度状态

- 慢通道是**测量设定寿命**的积分器：d≈0（静态）时 h_slow 向 h_t 缓慢漂移
  （长记忆），d≈1（动态）时 h_slow 冻结（快通道接管）。
- 死锁纪律：ρ 初始化 0.1（非零，见 §4）→ h_slow ≠ 0 → 合并 w_s（零初始化）
  有梯度通路。
- 读出：h_t' = h_t + w_s⊛h_slow（w_s 1×1 零初始化，256 通道），prior/
  posterior/decoder 读 h_t'。+0.03M。
- 预期效应：静态背景记忆长半衰期（重建容量释放）、动态区新鲜（Truck/Car
  持续目标受益——预登记按类假设）。

### L3 证据场（无参数物理层 + 零初始化细化）

- 物理层逐帧从 Doppler canvas 直接计算（无参数、空间结构由构造保证）：
  `c_mot⁰ = σ(α₁·log1p(count)·SNR̄ − β₁)`、`dyn⁰ = σ(α₂·v_std·count − β₂)`、
  `cov = count/log1p(M)`；α/β 为**固定标定常数**（由 val 集分位数一次性
  校准，不训练）。
- 学习细化：`c_mot = c_mot⁰ + conv⁰([c_mot⁰, dyn⁰, cov])`（零初始化）。
- **coverage 语义是 Ped 死区的正面解法入口**：cov≈0 的像素「无测量」，
  路由语义（§5 步 3）据此走相机主导，而不是被当成「静态/低可靠」。
- 开关 `use_physical_evidence`：False 时回退学习版（checkpoint 兼容）；
  DC-RSSM 配置置 True。**此层开启后恒等基准是「物理图」而非学习图**——
  与主线的比较按新设计代际预登记（不与 24e 三 seed 逐位比）。

### L4 创新一致性记忆

- `e_t = ρ_c·e_{t-1} + (1−ρ_c)|inn_t|²`，ρ_c = sigmoid(ρ̂)（ρ̂ init 使
  ρ_c≈0.8）；逐像素 EMA buffer（detach）。
- 用途一：进门控输入（第三动态通道——MCI-v2 的 innov 瞬时值 + 记忆值 =
  「当前意外 + 持续意外」）。
- 用途二（可选开关）：deformable offset 范围调制
  `offset × (1 + κ·tanh(e_t))`，κ 零初始化。
- 类比与新颖性：经典 Kalman 滤波的新息白性/一致性检验（filter health）
  首次移植为学习融合的门控信号。

### L5 存量

MCI-v2 全部保留（innovation 后验、门控 correct/readout、Doppler 对齐
偏移、动态更新门、`state_init` 开关）。MCI-v2 与三 seed 家族的代际差异
（innov 通道死锁修复）见 §68.8i。

## 3. 初始化纪律（正式化，全项目适用）

> **死锁规则**：禁止「输出恒为零的层」直接喂给「零初始化层」——
> 零输入通道的下游权重梯度恒为零，永不解除（§68.8i 的零乘积形态）。
> 二选一即可解锁：上游非零初始化（如 L4 的 Xavier innov conv）或
> 上游输出含非零偏置/物理量（如 L3 的物理层、c_cam/c_mot 的 bias=2）。
> 恒等启动只要求**最后一层**（进入 tanh/残差合并前）零初始化。

每层落地前跑 450 iter「权重活性冒烟」：所有新增 conv 的 weight abs_mean
必须离开精确零（把 §68.8i 的发现变成常态化检查）。

## 4. 实现顺序、配置与门控（全部预登记）

| 步 | 内容 | 机制门（450 iter / 探针） | AP 门 |
|---|---|---|---|
| 0 | **MCI-v2**（在训，09:25 UTC 起） | innov_mag 图非零且空间有结构；shuffle_z ≥2%；h/e <2 | ep12-16 ≥39.6061（主线 −0.5） |
| 1 | **DC-RSSM-1** = v2 + L3 证据场 | c_mot/dyn 图空间 std/mean ≥0.1（对照学习版 ≈0.001）；GT 中心类间区分出现 | ≥ 基线 39.2809 且 ≥ v2 −0.5；**Ped 配对差 ≥ −1.5**（cov+细化的直接检验） |
| 2 | **DC-RSSM-2** = +L1 ego-warp | 静态区跨帧对齐重叠率上升（探针量化） | 同上 + Car/Truck 不回退 |
| 3 | **DC-RSSM-3** = +L2 双速率 + L4 一致性记忆 | 静态/动态区寿命分化（探针）；ρ_c 学到非平凡值 | 同上 |
| 4 | 胜者三 seed → 主表 | — | §18 口径 |

每步一个开关一个配置；450 iter 冒烟全部跑「权重活性检查」（§3）+ 机制门；
AP 门失败即止损记录。多 seed 只对通过 DC-RSSM 全链的配置执行。

## 5. 论文定位（对 docs/11 的第二次更新）

Contribution 重排为四条：

1. **诊断方法学与失败模式分类学**：反事实探针 + specific-energy 分解 +
   权重体检；bias-dominated latent、观测无视、死锁类、空间失明、Ped 死区
   ——全部带三 seed 定量证据（已有）。
2. **测量接地的校准层**（L1+L3）：pose-free 设定下 ego-motion 自标定与
   无参数证据场；与 CRT-Fusion（学速度、有位姿）、RCFusion（Doppler 当
   特征）的差异化轴清晰。
3. **双时间尺度测量门控循环状态**（L2+L4）：测量设定记忆寿命与一致性
   门控对齐。
4. **Innovation-conditioned correct + 门控读出**（MCI 存量，三 seed 机制
   证据 9/9 窗 ≥2%）。

诚实定位不变：AP 主张按数据落点（R 规则复用）；机制主张三 seed 已立。

## 6. 风险与裁剪

- L1 的 Δt 可学标量可能吸收进 deformable——探针必须验证解析 warp 的
  独立贡献（静态对齐重叠率）。
- L2 双速率与 L4 一致性记忆有部分功能重叠（都编码「哪里意外」）——
  消融必须分开归因；GPU 紧张时 L4 先砍。
- L3 的物理标定常数依赖 val 分位数——标定集与评测集分离（前 1020 帧
  标定 / 后 1020 帧 评测），避免泄漏。
- 全部模块一次论文吃不下时的裁剪顺序：L4 → L2 → 保留 L1+L3+存量
  （L1/L3 是差异化最强、风险最低的两件）。
