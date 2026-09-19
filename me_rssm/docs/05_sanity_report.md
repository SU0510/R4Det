# 05 · Sanity 验证报告

> 执行日期：2026-09-19。环境：conda `r4det`（torch 2.0.1+cu118 / mmcv 1.7.1 /
> mmdet 2.28.2 / mmdet3d 1.0.0rc4），GPU 4（训练任务占用了其余 GPU，
> 所有验证只使用空闲 GPU 4 的纯 forward/backward，未运行任何训练/评估）。
> 全部脚本：`me_rssm/sanity/`，可一键复现。

## 结果总览

| 测试 | 内容 | 结果 |
|---|---|---|
| `test_radar_motion_chain.py` (CPU) | Doppler 证据链数值正确性 | **PASS** |
| `test_equivalence.py` (GPU) | E1-E4 基线等价性与前向行为 | **PASS** |
| `test_grad_paths.py` (GPU) | G1-G6 梯度通路/状态/回退语义 | **PASS** |
| `test_config_build.py` (CPU) | C1-C5 配置/构建/预训练兼容/参数 | **PASS** |
| `test_flops.py` (GPU) | 静态 MACs 计量 | **PASS** |

## 关键数字

- **E1 权重兼容**：基线 → ME-RSSM 载入 missing=35（全部为新增层的恒等启动键），
  unexpected=0。
- **E2/E3 等价性**：4 帧序列（3 burn-in + 1 采样帧）output/recon/KL/h_t 全部
  max|Δ| < 1e-5（空 bus 与开关全关两种情形）。
- **E4**：全开模式下 4 帧输出全部有限；`stat_gain_mean=0.982`（=σ(4)，恒等启动符合设计）、
  扩展 stat 键齐全。
- **G1a/G1b**：第 1 步反传直达 obs_proj/gain/motion-offset/dyn_conv/β 及全部基线路径；
  第 2 步起 camera/motion encoder 与 conf conv 激活（零初始化残差的标准一步延迟，
  见 04 §5）。
- **G3/G4**：detach_state 语义与 `reset_for_samples` 逐样本行为与基线一致。
- **G5 预测回退**：置信/gain 置 −10 后 z_t→μ_p（max|Δ|=2.18e-4）且输出有限——
  "观测不可信时降级为运动传播预测"的语义成立。
- **C4 预训练兼容**：与基线构建相比，719 个 producer 张量加载行为完全一致；
  808 unexpected + 6 形状不匹配键集合逐键相同（GRU 时代 temporal_fusion 权重 +
  主线未使用的 PaintBEVFusion 等）；新增 missing 恰为 39 个 temporal_fusion 新键。
- **C3 成本**：temporal 参数 +0.416M（+3.93%）；全模型 −0.175M（shared_stem=False
  的 clean 回退）；temporal conv MACs +44.5G/frame（+4.4%），开关全关时 0 增量。

## 过程中发现并修复的问题（对后续工作有价值的）

1. **测试侧坐标约定三次踩坑**（train_vod 数据面无恙，全部是测试脚本的错）：
   - 合成点的 y 坐标符号（`y_idx·v + v/2 + y_min`，y_min=−39.68）；
   - 证据图画布 (ny,nx)=(2·W_feat, 2·H_feat)=(496,432)，写反会触发形状守卫
     安全降级（守卫本身按设计工作，反而证明了守卫的价值）；
   - avg_pool(2) 对稀疏单热点证据有 4× 稀释（5.0→1.25）——这是**设计事实**而非 bug：
     网络可借 count 通道（ch6）补偿稀释；已写入 04 §3。
2. **父类 `__init__` 内调用 `init_weights` 的时序**：ME-RSSM 的额外层在父类
   `__init__` 返回前不存在 → `init_weights` 需 `hasattr` 守卫 + 幂等的
   `_init_me_weights`（与同类变体 Deterministic/PosteriorOnly 的处理一致）。
3. **β 的数值安全**：`u' = u + β·d·u·(1−u)` 在 β<0 且 |β|d 大时可使 u'<0
   （门变成外推）→ 用 `β = tanh(β_raw)` 界定，任意 β 下 u'∈(u²,1)⊂(0,1)。
4. **`load_state_dict(strict=False)` 不豁免形状不匹配**：真实训练用 mmcv 的
   宽容版 load_state_dict（只记录）。兼容性验证必须按 mmcv 语义分类
   （missing/unexpected/mismatch 三集合），test_config_build.py 已复刻。

## 明确声明

- 全部验证为**随机输入的纯 forward/backward、静态参数/MACs 计量、配置/键检查**。
- 未运行任何训练、验证集评估、指标搜索或超参选择；未生成任何用于结构决策的指标。
- 训练类 GPU 上的既有任务（CycCls seed2 等）未受影响（验证仅用空闲 GPU 4）。

## 复现

```bash
cd /home/lurui/workspace/R4Det && source .envrc
python      me_rssm/sanity/test_radar_motion_chain.py          # CPU
CUDA_VISIBLE_DEVICES=4 python me_rssm/sanity/test_equivalence.py
CUDA_VISIBLE_DEVICES=4 python me_rssm/sanity/test_grad_paths.py
python      me_rssm/sanity/test_config_build.py                # CPU, 需 ckpt
CUDA_VISIBLE_DEVICES=4 python me_rssm/sanity/test_flops.py
```
