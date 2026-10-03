# 12 · MCI-RSSM：测量条件化 Innovation RSSM（合并模型）设计与 Runbook

> 状态（2026-10-02）：**代码与配置已备齐，零训练、零测试执行**。本文档给出
> 设计规范、等价性契约、门控定义与完整 runbook；每一个运行操作都需用户批准。
> 动机与证据链：`docs/rssm_review_20261002.md`（P1 读出公共偏置、P2 KL 短路、
> P3 predict-correct 不闭环、P4 Doppler 未进动态、P7 机制-AP 解耦）。
>
> **MAI 三 seed 终判已出（2026-10-03，由 `mai_threeseed_verdict.py` 只读算出，
> seed2 由链式看门狗 ep16 自动停训）**：配对窗口差 seed0 **+0.8433** /
> seed1 **−2.5063** / seed2 **−0.0985**，均值 **−0.59 ± 1.41** —— **MAI 的
> AP 效应不 seed-robust**（预登记口径：|Δ|<0.5 为噪声内）。机制侧三个 seed
> 一致良好（ep16 shuffle_z 5.9/6.7/4.8%、innovation specific 61.4%、
> h/e=1.219 回到 h 主导），即「机制可测且修复、AP 不跟」的 P7 解耦在三
> seed 上坐实。**论文定位相应调整：innovation posterior 是机制组件而非 AP
> 主张；AP 提升押在 MCI 的 Doppler predict + 门控 correct/readout 上**，
> 本文档的合并模型即为该赌注的载体。

## 1. 一句话

把 MAI 的 innovation posterior（已验证的信息流修复）与 ME-RSSM 的三个
Doppler 机制（已设计未训练）合并，并补上二者都缺的第三环——**可靠性门控
读出**，使「预测-校正-使用」在决策层闭环：

```
Predict   h_t = ConvGRU( align(h_{t-1}; offset + M(e_mot)), z_{t-1} )   (C1)
          u'  = u + tanh(β)·d·u·(1−u)                                    (C3)
Predict   e_hat_t = obs_prior(h_t)                                       (MAI)
Correct   innovation = e_obs − sg(e_hat_t)
          delta = posterior_mu([h_t, innovation])                        (MAI)
          z_t   = z_sel + dev_c ⊙ (z_sel − mu_p),  dev_c = tanh(conv(g))
Readout   output = feat + res + dev_r ⊙ res,       dev_r = tanh(conv(g))
          g = [c_cam, c_mot, |innovation|↓]   (06-P2 预登记的 innovation 输入)
```

其中 `e_obs = e_t + obs_proj([c_cam⊙e_cam, c_mot⊙e_mot])`（obs_proj 零初始化，
ME-RSSM C2 的观测汇合）；`z_sel` 是 MAI 的 `mu_q`（或采样）。

## 2. 与 ME-RSSM 的两点关键差异（设计修订）

1. **门控参数化为偏差形式**。ME-RSSM 用 `g = σ(conv+4) ≈ 0.982` 起步，
   初始即有 ~1.8% 的缩偏。MCI 用 `z = z_sel + tanh(·)·(z_sel − mu_p)`、
   `output = feat + res + tanh(·)·res`：初始 `tanh(0)=0` 精确为零，
   **初始行为与 MAI 逐位相同**（不存在浮点重构误差——`mu_p + 1.0*(z_sel−mu_p)`
   不保证逐位回到 `z_sel`，偏差形式绕开它）。语义不变：dev<0 = 更信预测
   （fallback），dev>0 = 放大校正。
2. **门控输入加入 innovation 幅值**（06-P2 预登记）。经零初始化 1×1 conv
   从 `innovation.detach().abs()` 归约；**detach 是设计约束**——门观察分歧度
   但不得成为压缩 innovation 的梯度捷径。c_cam/c_mot 保留梯度（同 ME-RSSM）。

## 3. 等价性契约（单测 `me_rssm/sanity/test_mci_rssm.py`）

| 情形 | 保证 |
|---|---|
| E1 空 bus | 与 MAI **逐位**相同（门控/偏移/动态门/obs 残差全部跳过） |
| E2 bus 存在、开关全关 | 与 MAI **逐位**相同（开关关闭的模块**根本不构造**——DDP 零未用参数） |
| E3 bus 存在、全开、新初始化 | 与 MAI **逐位**相同（零初始化 + 偏差形式） |
| E4 加载任意 MAI checkpoint | 共享键逐位加载，缺失键恰为新模块（零/恒等初始化） |
| RNG 流 | 父类模块初始化 RNG 流不变（同 MAI 的纪律） |
| standard 模式（`posterior_mode='standard'`） | 与 `MotionAlignedRSSMFusion` 逐位相同（消融用）；`obs_prior` 被**删除**（无消费者，否则 DDP 未用参数中止），MAI ckpt 加载时 `obs_prior.*` 成为被忽略的 unexpected 键 |

构造期校验（ValueError）：`use_motion_offsets`/`use_dynamic_gate` 需要 bus；
门控需要至少一个输入源；`innov_in_gain` 需要 innovation posterior；
`modality_dropout` 需要 bus。

**探针兼容性（2026-10-03 补丁）**：`tools/probe_posterior_source.py` 重组
innovation 时默认条件于 `encoder(feat)`，而 MCI 条件于
`e_obs = e_t + obs_proj(...)`。类内已实现工具预留的
`_posterior_observation(e_t)` 钩子（eval 前向暂存 detach 残差，精确重组；
训练前向不暂存，零开销）。`tools/diagnose_z_utilization.py` 的
shuffle_z/zero_z 干预改写 `posterior_mu` 输出，与本差异无关，读数始终有效。
两个工具均为类无关实现（`getattr(fusion, 'posterior_struct', 'standard')`）。

**DDP 硬约束（沿用 docs/06）**：`use_modality_obs=True` 的训练必须用完整
producer 三件套（`RadarPillarFeatureNetMotion` + `PointPillarsScatterMotion`
+ `ConcatConvFusionStash`，主线配置已内置）；空 bus 训练会因未用参数报错。
降级路径仅限 eval/sanity。

已知瓶颈模式（与 ME-RSSM 同、非缺陷）：零初始化门控 conv 使门输入端
（c_cam/c_mot/innovation 幅值）在第一步收到零梯度；门 conv 权重本身第一步
即有梯度（tanh'(0)=1、上游残差通路存活），一步后输入端梯度恢复。
§38.10 的死分支教训（零初始化+ReLU）在此不适用（无 ReLU）。

## 4. 消融网格（7 个单变量配置，全部已写好）

| 配置（前缀 `TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_mci_`） | 关闭的开关 | 隔离的机制 |
|---|---|---|
| `abl_no_motion_offsets` | `use_motion_offsets=False` | C1 Doppler 对齐 |
| `abl_no_dynamic_gate` | `use_dynamic_gate=False` | C3 动态更新门 |
| `abl_no_reliability_gain` | `use_reliability_gain=False` | correct 门（读出门保留） |
| `abl_no_gated_readout` | `use_gated_readout=False` | **P1 修复本体**（若全模型胜 MAI 而此消融不胜，增益来自读出） |
| `abl_no_modality_obs` | `use_modality_obs=False` | 模态观测/置信整体（门退化为纯 innovation 输入） |
| `abl_no_innov_in_gain` | `innov_in_gain=False` | 06-P2 的 innovation 门控输入 |
| `abl_no_innovation` | `posterior_mode='standard'` | innovation posterior（合并模型内部的 MAI-vs-standard） |

## 5. 判定门（预登记）

### 5.1 450 iter 机制门控（冒烟配置 `me_rssm/sanity/mci_smoke_450iter.py`）

| 检查 | 门槛 | 工具 |
|---|---|---|
| 训练健康 | 无 NaN/Inf；`stat_obs_pred` 下降；`stat_innovation_sq` 有界不塌缩；recon/KL 与主线同量级 | `tools/verdict_window.py --stats <work_dir>` |
| correct 门动态范围 | `mean(stat_gain_std) > 0.005`；`spread(stat_gain_mean) > 0.005`；均值 ∈ (0.5, 1.5) | `me_rssm/sanity/mci_gate_report.py <work_dir>` |
| readout 门动态范围 | 同上（`stat_readout_gain_*`） | 同上 |
| 身份信号（与 MAI 450iter 同窗可比） | `shuffle_z` head ≥ MAI 同期（0.369–0.580%）；2% 论文门槛留给正式训练 | `tools/test_diagnose_z_utilization.py`（对 `iter_450.pth`） |

### 5.2 ep6 AP+机制门控（正式 24e 跑到 ep6 停）

沿用 MAI 的预登记规则（§62.9）：
- **AP**：ep6 Overall 3D moderate vs no2d_igdr seed0 同 epoch（34.0243）：
  差距 ≤1 → 续 ep12；1–2 → 记录后续到 ep12 再判；>2 → 止损。
- **机制**：三窗 `shuffle_z` ≥0.2%（继续门槛），`innovation specific > e_pooled`，
  `stat_gain_*/readout` 非退化，**Ped/Cyc 早期不劣化**（ep1-6 逐类别曲线；
  Ped loose 相对同窗基线 −2 以上即亮黄旗——C2 门控对 Ped 的行为是
  review P9 预登记观察点）。
- 工具：`me_rssm/sanity/watch_mci_ep6.sh`（停训 + 三窗/source 探针）。

### 5.3 最终判定

三 seed 配对窗口（seed0/1/2，`tools/verdict_window.py` 或逐 seed 配对），
口径 §18；胜者才进主表（逐 seed 配对 Δ + checkpoint averaging 复评）。

## 6. Runbook（全部操作等待用户批准；每步独立可撤销）

> 以下命令均为**准备好的**操作，未执行。GPU 5/6/7 当前由 MAI seed2（→ep16
> 自动停）与随后的 fgfull18 seed0 接力占用；单测与 CPU 报告不占训练槽。

| # | 操作 | 命令 | 成本 | 依赖 |
|---|---|---|---|---|
| R1 | 跑 MCI 单测套件（35 条） | `python me_rssm/sanity/test_mci_rssm.py` | CPU ~2 min | 无 |
| R2 | 配置构建自检（9 个配置加载 + 全模型构建 + 开关隔离 + 预训练键 + **DDP 未用参数审计**） | `python me_rssm/sanity/test_mci_config_build.py` | CPU ~3 min | 无 |
| R3 | 450 iter 机制冒烟 | **首选 3 卡 DDP**（同时验证 DDP 契约）：`tmux new -d -s mci_smoke 'source .envrc; CUDA_VISIBLE_DEVICES=5,6,7 bash tools/dist_train.sh me_rssm/sanity/mci_smoke_450iter.py 3 --seed 0 --deterministic --work-dir /data/lurui/work_dirs/mci_smoke_450iter'`；GPU 4 单卡仅作数值冒烟（DDP 路径已由 R2 的 M6 静态审计覆盖） | 3 卡 ~25 min | R1/R2 通过 |
| R4 | 机制门控判读 | `python3 tools/verdict_window.py --stats /data/lurui/work_dirs/mci_smoke_450iter` + `python3 me_rssm/sanity/mci_gate_report.py <同>` + 三窗 z-util | CPU/GPU 探针 ~10 min | R3 完成 |
| R5 | MAI 三 seed 终判（seed2 ep16 落盘后，随时可跑） | `python3 me_rssm/sanity/mai_threeseed_verdict.py` | CPU 秒级 | seed2 ep16 val（链式看门狗自动停训后） |
| R6 | MCI seed0 正式 24e | `tmux new -d -s mci_seed0 'bash me_rssm/queue_mci_seed0.sh'`（等整个 GPU 队列排空；若要插队 fgfull18 之前，先 `tmux kill-session -t fgfull18_seed0_watch`，**该取舍由用户决定**） | 3 卡 ~2.5 天 | R4 通过 |
| R7 | ep6 看门狗 | `tmux new -d -s mci_ep6_train '...'`（24e 启动行见 queue 脚本）+ `tmux new -d -s mci_ep6_watch 'bash me_rssm/sanity/watch_mci_ep6.sh'` | 随 R6 | R6 启动 |
| R8 | 消融网格（R4 通过后逐个 450 iter，过线者 24e） | 换 `*_abl_*.py` 配置重复 R3/R4 | 按需 | R4 |
| R9（可选） | `state_init='obs'` 单变量（低成本修复 t0 死状态与 ep1 偏慢，review P5/§8.4） | 配置 `..._mci_stateinit.py` 已备；可与 R6 二选一先跑，或在 R6 判定后叠加（届时须先过单变量门控） | 同 R6 | R4 |

R5 与 R1–R4 相互独立，可并行决定。

## 7. 文件清单（本次新增/修改，全部在 me_rssm/ 内）

```
me_rssm/
├── mci_rssm.py                              # 新类 MeasurementConditionedInnovationRSSMFusion
│                                            #   含 _posterior_observation 探针兼容钩子；
│                                            #   standard 模式删除 obs_prior（DDP）
├── __init__.py                              # 注册新类（一行）
├── configs/
│   ├── TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_mci.py      # 主线
│   ├── ..._mci_stateinit.py                                                   # state_init='obs' 单变量
│   ├── ..._mci_abl_{no_motion_offsets,no_dynamic_gate,no_reliability_gain,
│   │            no_gated_readout,no_modality_obs,no_innov_in_gain,
│   │            no_innovation}.py                                             # 7 消融
├── sanity/
│   ├── test_mci_rssm.py                     # 35 条单测（未运行）
│   ├── test_mci_config_build.py             # 配置构建自检 + DDP 未用参数审计（未运行）
│   ├── mci_smoke_450iter.py                 # 冒烟配置
│   ├── mci_gate_report.py                   # 门动态范围判读（只读）
│   ├── watch_mci_ep6.sh                     # ep6 看门狗（停训+探针，不接续）
│   └── mai_threeseed_verdict.py             # MAI 三 seed 配对终判（只读 CPU）
├── queue_mci_seed0.sh                       # 24e 队列脚本（等待槽位排空）
└── docs/12_mci_rssm.md                      # 本文档
```

口径附注：MCI 继承 no2d_igdr/MAI 的完整基链（经 fgfull→me_rssm clean
snapshot 解析，`shared_stem=False` 与主线一致）。**比较口径只在
no2d_igdr/MAI 家族内部做配对**；与 ME-RSSM（其主线配置走独立基链）的对照
需要先对齐基链才有意义。

论文素材映射（对 docs/11 的修订）：contribution 1 = innovation posterior
（已验证）；contribution 2 = Doppler predict + 可靠性门控 correct/readout
（本文档，待训练）；contribution 3 = 机制级分析（collapse 稳态、z 公共偏置、
机制-AP 解耦、门控动态范围，探针方法学可复用）。
