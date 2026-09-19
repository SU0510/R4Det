# 07 · P1 观测引导的状态初始化（state_init='obs'）

> 实现 06_second_layer.md 的 P1（"状态初始化仍为死状态"）。状态：已实现 +
> sanity 全绿（13/13），**未训练**。排在 ME-RSSM 主线 seed0 判定之后。

## 1. 问题（06 P1 原文摘要）

基线与 ME-RSSM 在序列起点都使用 h_0 = z_0 = 0（rssm_fusion.py 的
reset 语义 + forward 的 `h_state is None` 分支）。N=4 窗口的第一历史帧
转移从零状态出发：deform align 作用于零张量、GRU 候选仅由 z_a=0 驱动，
首帧观测只通过当帧 posterior 进入。滤波器没有"先验引导的初始化"。

## 2. 设计

**可学习初始状态**（06 列出的两个方向中更安全的一个）：

- 两个零初始化 1×1 卷积从当前观测编码生成初始状态：
  - `h_0 = h_init_conv(e_fused)`，`e_fused = encoder(feat)`（in_channels → out_channels）
  - `z_0 = z_init_conv(e_fused)`（in_channels → latent_dim）
- **零初始化 ⇒ 初始行为与主线逐位一致**（h_0=z_0=0），训练中才逐渐
  学出"从首观测 bootstrap 滤波器"的行为——延续整个包的 identity-start 原则。
- bootstrap 用纯外观编码 `encoder(feat)`，不用门控后的模态流：
  与 bus 可用性解耦（空 bus 时同样工作），也避免初始化依赖 conf/gain。

**reset 语义**（关键工程点）：

- `reset_state()` → h/z = None，下一次 forward 走 full-init 分支（'obs' 时
  用 bootstrap）。无需额外簿记。
- `reset_for_samples(mask)` → **保留基线的就地清零**（任何在 reset 与
  forward 之间读状态的代码看到基线语义），同时把清零行记入
  `_pending_obs_init` 掩码；下一次 forward 用该行**自己的当前观测**
  重新 bootstrap。若 pending 掩码因 batch 尺寸/设备不匹配而失效，
  行为退化为基线清零——最坏情况 = 基线，绝不使用陈旧状态。
- forward 里 `e_fused = encoder(feat)` 移到状态初始化之前（encoder 无
  状态依赖，输出不变；test_state_init S1 重新验证了 E2 等价性）。

**成本**：+0.132M 参数（2 × (256×256+256)），temporal 级 +1.2%。
**接口**：6 元组返回、`kl_scale` 契约、E1 权重兼容全部不变。

## 3. 文件

- `motion_evidence_rssm.py`：`state_init` 参数、`h_init_conv`/`z_init_conv`、
  `reset_state`/`reset_for_samples` 覆写、forward 状态初始化分支重构。
- `configs/TJ4D-R4Det_me_rssm_N4_24e_pretrained_v2_head_stateinit.py`：
  唯一改动 `state_init='obs'`，继承主线配置。
- `sanity/test_state_init.py`：S0–S7 共 13 项检查，全绿。

## 4. 验证（test_state_init.py，13/13 green）

| 项 | 内容 | 结果 |
|---|---|---|
| S0 | stateinit 配置构建、仅翻转 state_init、init conv 全零、参数量 131,584 | PASS |
| S1 | 空 bus + state_init='zero' 与基线逐位一致（forward 重排后的 E2 回归守卫） | PASS |
| S2 | 'obs' 零初始化时输出与 'zero' 逐位一致（identity start） | PASS |
| S3/S3b | 权重键契约：zero↔obs 互载的 missing/unexpected 恰为 4 个 init conv 键 | PASS |
| S4 | 扰动 init conv 后序列起点行为改变（机制激活） | PASS |
| S5a-d | B=1 下 reset 语义：reset 即清零（基线语义保留）、pending 一步内消费、用当前帧 re-bootstrap、未 reset 控制组不受影响 | PASS |
| S6/S6b | 首次 backward 梯度到达两个 bootstrap conv（经 alignment→GRU→posterior 路径） | PASS |
| S7 | 参数增量恰为公式值 | PASS |

## 5. 环境发现：deform 对齐的逐位比较在 CUDA 上布局敏感（重要，与 P1 无关）

调查 S5 时发现：同一模块、sample-1 输入逐位相同的两次 forward，仅改
sample-0 的状态（+3.0）并保持 run-1 输出张量的引用存活，sample-1 的输出
会变化（maxdiff ≈ 1.14，逐进程逐位可复现）。排查结论：

- **未改动的基线 `MotionAlignedRSSMFusion` 完全同样复现，maxdiff 逐位相同**
  ——是既有环境属性，不是 P1/ME-RSSM 引入的。
- 隔离的 `ModulatedDeformConv2d`（随机 offset、全零 offset、非连续 offset
  切片、保持引用）全部干净；跨进程手写复刻对齐路径也干净。
- 异常只在"完整模块 + 进程内重复 forward + 特定张量生命周期"下出现，
  加一个 forward hook（改变分配布局）即消失；`CUDA_LAUNCH_BLOCKING=1`
  不消失（非异步竞争）。
- 推断：本环境 mmcv deform conv 调用链在特定分配布局下存在越界读类
  内存安全问题；真实训练（每次 forward 数据都不同、常规分配模式）数月
  未受影响，但**逐位等价类测试在 CUDA 上不可靠**。

**工程决定**：`test_state_init.py` 的逐位检查（S1–S5）全部在 **CPU** 上
执行（确定、布局稳定）；GPU 只做梯度连通性检查（S6）。既有
`test_equivalence.py`（E2 用 max|Δ|<1e-5 容差、生产形状、GPU）不受影响，
维持原样。后续新增逐位检查一律默认 CPU。

## 6. 训练计划（预注册）

- 配置：`..._v2_head_stateinit.py`（唯一变量 = state_init）。
- 排序：**严格排在 ME-RSSM 主线 seed0 判定之后**；若主线 seed0 失败，
  P1 是否继续取决于失败模式（stat_* 显示冷启动问题则优先）。
- 判定：沿用 Run 10 多 seed 口径——ep12-16 窗口均值 vs 主线同 seed
  配对，Overall/BEV ≥ 主线，四组成项降幅 < 1.0；单 seed 通过后需
  seed1/2 复现（CycCls 的教训：单 seed 的 +0.6~1.2 在 ±1.5 噪声内）。
- 预期失败模式：(a) 零初始化 conv 学不动（梯度被 alignment/GRU 稀释），
  stat 上无痕 → 检查 h_init_conv 权重范数随 epoch 的变化；(b) 首帧
  bootstrap 与 posterior 重复计入同一观测 → 过拟合首帧 → 看 ep12-16 的
  Car/Ped 分项。

## 7. 状态初始化的 stat 观测（随下次训练自动记录）

当前 stats dict 不含 init 专用观测量。若 P1 进入训练，临时验证手段是
对比同 seed 主线/stateinit 的 ep1 指标（bootstrap 生效应抬升首 epoch
起点），以及训练结束后 `h_init_conv.weight` 的范数（若仍 ≈ 0 则机制
未被使用）。
