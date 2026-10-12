# 预登记：可靠性感知前端融合（Reliability-Aware Camera/Radar Fusion, F1）

- 冻结时间：**2026-10-12**（在首次运行之前；用户的既有偏好：预登记 + 最小实现 + S0 + 短冒烟 → 开训）
- 臂名：**F1 = ReliabilityGatedConcatFusion**
- 配置：`me_rssm/configs/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_relfusion.py`
- 冒烟配置：`me_rssm/sanity/relfus_smoke_450iter.py`
- 运行目录（待建）：`/data/lurui/work_dirs/fgfull_N4_no2d_igdr_2x4_24e_relfusion_seed0`

---

## 0. 立项证据（实测，非推断）

仓库内历史 `crossmodal_fusion_N4_2x4_24e_seed0`（把 `RCFusion` 从 `ConcatConvFusion` 换成
`BEVCross_modal_attention` 类模块），ep12-16 五轮等权均值（n=5）对同口径 B0（标准
`no2d_igdr` seed0）：

| 指标 | crossmodal | B0 | Δ |
| --- | --- | --- | --- |
| Overall | 37.9789 | 39.2809 | **−1.3019** |
| Car (strict) | 45.7627 | 47.8399 | **−2.0772** |
| Truck (strict) | 31.4858 | 29.7246 | +1.7612 |
| Pedestrian (loose) | 25.6864 | 30.9277 | **−5.2413** |
| Cyclist (loose) | 48.9808 | 48.6313 | +0.3495 |

- 全轮峰值：crossmodal **39.2835 @ ep16**；B0 **40.3853 @ ep14**。
- 该模块**不是**强意义上的 Transformer cross-attention：每个模态各自做 channel
  mean/max → 生成一张空间 attention → 互相乘 → concat + conv；**没有可靠性估计，
  也没有任何缺测处理**。
- ⇒ **允许的结论**：「不能盲目把 concat 换成 attention 就期待涨 AP」；该模块的失败
  **不能**证明「所有高级 fusion 都无效」。
- ⇒ **不允许的结论**：融合方向已被证伪；雷达缺测就是该模块失败的原因（未测）。

## 1. 假设（单一、可否证）

> 前端 concat 融合缺少「雷达这一路当前是否可靠」的显式信号。给雷达分支一个由**输入可见
> 证据**驱动的**每格可靠性门** α，可以在缺测/弱回波处自动降低雷达权重、保留 camera 信息，
> 从而改善缺测场景下的检出（尤其 Ped/Cyc 这两类在历史 attention 替换中损失最大的类别）。

## 2. 唯一研究因素（公式冻结）

```
occ   = 逐 BEV 格雷达点占用（由 *输入点云* 构造；不读 GT、不读标签、不读诊断掩码）
alpha = 1 + tanh( gate([occ, |radar|_mean_c, |radar|_max_c]) )      # alpha ∈ (0, 2)
out   = fusion_conv( cat([img_bev, alpha * radar_bev], dim=1) )      # conv 与基线同形
```

- `gate` = `Conv2d(3→16, k=3) → ReLU → Conv2d(16→1, k=3)`，**末层零初始化** ⇒
  step 0 时 `alpha ≡ 1` ⇒ **本臂与 `ConcatConvFusion` 逐位相同**（单测断言）。
- **相机通路不被门控**：`img_bev` 原样进入 concat；只有 radar 半部被 α 缩放。
- **α 的取值方向**：弱/无回波 ⇒ α 可低于 1（抑制）；强回波 ⇒ α 可高于 1（增强，上限 2×）。
- **容量**：唯一新增参数 **593**（=3·16·9+16+16·1·9+1），相对基线 1,474,816 → **+0.040%**；
  `fusion_conv` 的形状与基线逐位相同（单测断言）。
- **不动**：backbone、neck、检测头、`temporal_fusion`（仍为标准 `MotionAlignedRSSMFusion`）、
  KL 日程、优化器、数据管线、总 epoch、score/IoU/分组边界。**不引入新 loss、不改 KL、
  不加语义头、不加 Doppler 门。**
- 单变量契约由 S0-2 机械核验：臂与主线配置的差异键**必须恰好等于 `['RCFusion']`**。

## 3. 布局正确性（预防静默错位）

`occ` 必须与它缩放的 radar 张量同格。`PointPillarsScatter` 的源码显示 canvas 以
`coors`（顺序 b,z,y,x）按 `indices = y*nx + x`、`canvas.view(ny, nx)` 组装 ⇒ **行 = y 格，
列 = x 格**，故 `valid_layout='yx'`。单测用「远角/近角各一点」断言角落映射，并用方阵断言
`yx` 与 `xy` 互为转置。**布局错误会静默镜像掩码**，因此这是硬门。

## 4. 数据、调度与队列（与主线一致）

- seed 0、`--deterministic`、3-rank DDP、GPU **5/6/7**、24 epoch、与 B0 相同的 EvalHook。
- 队列：**排在 C1（进行中）→ A2-R 续训 之后**，串行；同一时刻只占用 5/6/7。
- 冒烟：450 iter（`IterBasedRunner`），约 12 分钟。

## 5. S0 接线门（先冒烟，后判定）

| 判据 | 内容 | 判定器 |
| --- | --- | --- |
| S0-1 | 跑到 450 iter、写出 `iter_450.pth`、任何被记录的 loss 无 NaN | `relfus_s0_verdict.py` |
| S0-2 | 臂/主线配置差异键恰为 `['RCFusion']`；新建模块 gate 末层恒零 ⇒ α≡1 | 同上 + 单测 |
| S0-3 | 训练后 gate 权重已离开零（非闲置参数）；`fusion_conv` 形状与基线一致 | 同上 |
| S0-4 | 容量增量 = 593（+0.040%）；`valid_layout='yx'` | 同上 + 布局单测 |

S0 任一 FAIL ⇒ 修实现后重跑冒烟；连续两次 FAIL ⇒ 停并报硬问题（不调阈值）。

## 6. 判定（三层，遵循 2026-10-12 的附录 B 政策）

1. **ep6 早期检查（只查任务损害）**：C/F1 的 ep1-6 区间均值低于 B0 同窗（**29.0420**）
   **> 2.0 点** ⇒ 停并复查。机制读数不参与裁决。
2. **ep12-16 成熟任务门（主对照，n=5）**：Overall 区间均值 ≥ **40.00**；
   四类非劣：Car ≥ 47.34、Truck ≥ 29.22、Ped ≥ 29.93、Cyc ≥ 47.63；
   任一类 ≤ B0 同窗 −2.0 ⇒ 止损归档。**ep14 不设剪枝门**（附录 B）。
3. **三口径必报**：区间均值（写明跨度与 n）、全轮峰值（值 + epoch）、best saved（值 + epoch）。

## 7. 归因与优先级

- **首要目标是提 AP**（用户 2026-10-12 指令）：机制读数不足以否决性能结果；
  但论文 claim 必须等于实际成立的机制。
- 无论 AP 正负，**都要报 α 的机制读数**（ep14 及之后：占用格 vs 空格上的 α 分布、
  α 与占空比的关系、以及 α 是否真的随缺测下降），否则「可靠性感知」只是命名。
- **失败时的归因边界**：若 F1 不及 B0，可归因到「每格标量门不足」或「前端不是瓶颈」，
  但**不能**外推为「所有可靠性感知融合无效」；若 F1 优于 B0，则须区分「门起作用」与
  「多 593 参数 + 训练噪声」，即需要多 seed。

## 8. 产物索引

- 实现：`me_rssm/reliability_fusion_rssm.py`（`ReliabilityGatedConcatFusion`）
- 接线：`mmdet3d/models/detectors/R4Det.py`（仅当 fusion 声明 `accepts_obs_valid` 时传
  `obs_points=points`；其它臂签名不变）
- 配置：见文首；冒烟：`me_rssm/sanity/relfus_smoke_450iter.py`
- 判定：`me_rssm/sanity/relfus_s0_verdict.py`；单测：`me_rssm/sanity/test_reliability_fusion.py`
- 队列：`me_rssm/sanity/queue_relfus.sh`
