# 候选 C 失败富集探针：评分契约（先冻结，后运行）

状态：**已冻结，尚未运行**。本文写于任何 GPU 探针启动之前（2026-10-12），
目的正是让分组边界、匹配规则与判定门槛不可能在看到结果之后再被调整。
本文件只授权**只读探针**；不授权训练、不授权 C 立项、不授权改网络。

配套：
- 分组产物：`/data/lurui/work_dirs/fgfull_N4_no2d_igdr_2x4_24e_seed0/c_coverage_groups_20261012.json`
  （§76，coverage 轴）与 `c_motion_axis_20261012.json`（§77，运动轴）。
- 记录：`docs/training_runs_full.md` §76（数据侧）、§77（运动轴）、§51.5（口径差）。
- 前置审查：§76.5 / §75.15（用户裁决：先 CPU 运动轴，再 GPU 失败富集，不得倒序）。

---

## 1. 冻结的输入

| 项 | 冻结值 | 理由 |
|---|---|---|
| 主臂 | `fgfull_N4_no2d_igdr_2x4_24e_seed0`，`epoch_14.pth` | 该臂全轮峰值 40.3853 @ ep14 且 ep14 有权重；用主基线回答「失败是否富集于缺测」 |
| 次臂 | `semA_seed0` / `semA2S_seed0` / `semA2R_seed0` | 各自单独报告，**不跨臂合并**；A2 两臂的闸门分母与 A 的 10-clip 契约不同（§75.12），因此只用它们的**权重**，不用它们的闸门常数 |
| 分组产物 | 上述两份 JSON 的 sha256 | 写入产物，防止事后移动边界 |
| 评估集 | val 全量 2040 帧（**不抽样、不用 clip 子集**） | clip 契约（10/20）是 A2 机制闸门的口径，与本研究无关；全量避免再引入一个分母 |
| 预测来源 | 该臂 `simple_test` 的 `pts_bbox` 最终输出 | 即网络内部的 NMS 结果，见 §3 |

`probe_semA_bnmode.py` 的 `eval_reproduces_frozen_gate.reproduces` 对 A2 两臂恒为
False（它比对的是 **A** 的冻结常数 0.1167/0.0833）。这是已知地雷，本研究的产物不得
引用该字段作为自检结果；自检只与各臂自己的 gate JSON 比对。

## 2. 被解释的指标，及其在本仓库里的确切定义

主指标 `pts_bbox/KITTI/Overall_3D_moderate`。读 `mmdet3d/core/evaluation/kitti_utils/eval.py`：

- **Overall 聚合**（`kitti_eval`，current_classes = 配置的 `class_names`
  `['Pedestrian','Cyclist','Car','Truck']`）：
  `Overall = mean(Ped@loose, Cyc@loose, Car@strict, Truck@strict)` ——
  即 **Car/Truck 用 0.5、Ped/Cyc 用 0.25 的 3D IoU**（`lap1=0.5`, `lap2=0.25`）。
- **难度分箱**（本 fork 的 `clean_data`）：GT 距离 ≤50 m → easy(0)、50–70 m → moderate(1)、
  其余 → hard(2)。即 **`moderate` 在本仓库里是「50–70 m」距离箱**，不是 KITTI 原版的高度/遮挡规则。
- 因此「moderate」这一列只覆盖 50–70 m 的 GT，且四类的匹配阈值并不相同。

⇒ 富集探针若用一套统一阈值匹配，就会在解释一个**不是**被优化指标的东西。契约如下。

## 3. 冻结的评分/匹配规则

**预测**（每帧，直接取网络输出，离线 harness 不再二次过滤、不再二次 NMS）：

| 项 | 值 | 出处 |
|---|---|---|
| 输出键 | `pts_bbox` | `dataset.evaluate(results, metric='pts_bbox')` |
| score 过滤 | `score_thr=0.0`（网络内已过滤） | 该 run 配置快照 `test_cfg.pts` |
| NMS | `use_rotate_nms=True`, `nms_thr=0.5`, `nms_pre=1000`, `nms_post=300`, `max_num=300` | 同上 |
| 额外后处理 | **无** | 避免与网络内 NMS 重复 |

**TP/FN 匹配**（逐帧、逐类，直接调用仓库自己的匹配代码，不复刻）：

- 匹配函数：`mmdet3d/core/evaluation/kitti_utils/eval.py::clean_data` +
  `compute_statistics_jit`，`metric=2`（**3D IoU**），这样归属与被解释的指标同源。
- 每类 min_overlap 与该类的 Overall 口径一致：**Pedestrian/Cyclist = 0.25，
  Car/Truck = 0.5**（`lap2`/`lap1`）。不做「统一 0.5」的变体作为主口径。
- 难度：主口径用 `difficulty=2`（**全部 GT，不论距离箱**），因为 C 的问题是「缺测时是否更容易漏」，
  不应被 50–70 m 箱预先裁掉样本；同时按预声明分层另报 ≤50 m / 50–70 m / >70 m 三箱。
- **score 操作点：主口径 `score ≥ 0.25`**。另报 0.10 / 0.50 两个点作为敏感性，
  **不用于任何 go/no-go 判定**（三点都在此冻结，取哪个点看结果是不允许的）。
- GT 记 `found` 当且仅当存在同类预测在上述规则下与之匹配；未匹配 GT 记 `fn`，
  并保留其 `image_idx`、组别、距离、尺寸、遮挡、难度以便分层。
- GD 的 `difficulty < 0`（ignored）与 `DontCare` 不参与任何分子分母（与 §76 一致）。

**FP 归属**：

- 未匹配到任何 GT 的预测记入 **独立桶 `fp_unmatched`**，**不得**塞进某个 GT 的缺测组。
  它按自己的口径报告（总数、按分数段、按距离、按类别）。
- 允许额外输出一张**描述性**「FP 最近 GT」表（最近邻 GT 的距离/组别），
  但该表**不进入**主统计、不进入 go/no-go；写在这里以防事后被当作结果使用。

## 4. 对照与口径纪律

- **所有臂、所有组、所有 epoch 一律用同一个离线 harness**（脚本与本契约同一次提交冻结，
  产物里记录脚本 sha256）。离线路径是 `dataset.evaluate` 直接调用、无 `evaluation.pipeline`
  （§51.5）。
- 产物中同时记录该 harness 对该 checkpoint 的 Overall，以便读者**看见**口径差
  （§51.5 约 2.2 个 3D 点）。该 2.2 点是历史观测的**口径差**，
  **不是可以统一加减的校正常数**：任何跨口径比较都禁止，也不会被用来「修正」本研究的数字。
- **自然缺测与受控雷达屏蔽分开报告。** 本探针只做**自然缺测**（数据本身
  `num_points_in_gt == 0` / `n_points_raw == 0`）。受控屏蔽需要改输入做前向，
  属于另一个实验，必须单独预登记；两者绝不合并成一张表。
- 分层必须在**同一份**分组 JSON 的边界下完成（train 冻结、val 只应用）；
  探针不得自行重算阈值。

## 5. 判定规则（先写死）

主统计量：**zero-coverage 组的 GT 召回率** 与**正覆盖组召回率**之差
（Δrecall = recall_covered − recall_zero，单位：个百分点），在 `score ≥ 0.25`、
3D IoU 与类阈值如上、全距离 GT 上计算。

- **富集成立**（→ 才谈 C 立项）当且仅当：
  1. Δrecall ≥ **3.0** 个百分点；且
  2. 在 §76 的混杂控制下（按距离箱分层，≤50 / 50–70 / >70 m）**符号一致且至少 2 个箱的
     Δrecall ≥ 3.0**，箱内 GT 数 ≥ 30；且
  3. 以 **scene 为聚类**的 bootstrap（2000 次重采样，按 §76 的 11 个 val 场景块）得到的
     95% CI 不跨 0。
- **富集不成立**（→ 不立项）：以上任一条不满足。此时按 §75.15 转对象级读出／
  空间定位／跨场景泛化方向，**不得**改阈值再试。
- 次级诊断（**只作描述**，不单独决定 go/no-go）：按组重算 §75.6 同一评分函数下的
  M1 峰漂移与分类 CE；按组的 BEV/3D 误差分解；FP 桶的按组构成。
- 任何格点 GT 数 < 30 一律标注 `low_n`，且不得单独作为证据。

## 6. 已知混杂（写进产物，写进结论）

1. 覆盖与距离/尺寸强相关（§76.3）：zero 组更远更小 ⇒ 分层是必需项，不是可选补充。
2. split 位移了覆盖分布（§76.3）：**不能**把 §74.11「差距来自 data split」读成「val 缺测更多」。
3. 零覆盖里存在**帧级掉线**（§77）：整帧都没回波时，某对象的 0 不代表该对象缺测。
   主统计必须按 `zero_scope`（`frame_level` / `object_level`）**分开报**，
   并额外给出只用 `object_level` 的版本；两者都在运行前定好。
4. 速度通道是**未补偿的径向速度**，帧间自车项 0–27 m/s 不等（§77）⇒ 涉及速度的分组
   一律用 ego-corrected 幅度，且按 `train` 冻结阈值。

## 7. 不做什么

- 不启动训练；不写 C 的预登记；不新增/修改任何 loss、head、readout。
- 不调 score 点、不调 IoU、不调分组边界来「救」富集结论。
- 不把自然缺测与受控屏蔽混表；不把 FP 塞进 GT 组；不做跨口径加减。
- 若探针中途失败，只修工具并**在新提交里记录修改**，不改本契约的数值。
