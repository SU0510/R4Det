# 从“缺证据”到可执行闭环：z 的检测价值与改造决策

日期：2026-10-13。承接 §93/§94；本次新增预测级诊断工具，不修改现役模型、训练配置或队列。

## 1. 先给决策，不再等辅助指标

**不继续盲调 KL，不再拿语义 AUC/张量变化代替检测成功。** 首先完成同权重的最终预测干预，
按结果决定修读出、修几何观测，还是停止把收益归因于随机状态。

“完整证据”在这里不是枚举一切可能原因，而是回答四个可检验问题：

1. 工具是否真的改变目标路径，且其他应固定的东西没有改变？
2. z 当前已经改变哪些真实预测；净新增的是高 IoU 真框，还是错误高分框？
3. 当前观测进入 posterior 后，相比该模型的 prior，是否增加任务价值？
4. 这些收益是否由同容量确定性状态、额外监督或更好的 fusion 就能解释？

前两个是现有 checkpoint 的干预问题；最后一个必须重新训练匹配控制，不能靠 frozen-head
换输入回答。完整链条有明确终点，不需要每失败一次就再叠一个模块。

## 2. 第一阶段：现有 checkpoint，四臂预测级检验

工具：`me_rssm/sanity/paired_z_prediction.py`。

固定一个 checkpoint、同一 val、N4、batch=1、eval 模式与 posterior mean 部署规则。按以下顺序运行：

- `reference`：原 detector，无 wrapper；保存全部最终预测。
- `identity`：安装只读记录 wrapper，但不干预；必须与 reference 输入和最终预测逐位相同、AP key-wise 相同。
- `residual_off`：只在部署 head 调用中去掉直接 z 分类 residual；不改 RSSM 内部、不改历史、不改权重。
- `prior_only`：只在当前帧 RSSM 调用将 `use_posterior=True` 改为 `False`；历史仍正常。

### 2.1 必须先通过的隔离证明

所有臂每帧保存输入指纹；模型参数/持久 buffer 指纹及核心代码、配置、annotation、checkpoint
SHA256 固定。每臂结束复核，任何变化直接报 failed，不输出成功判定。

`residual_off` 必须保持以下张量逐位不变：当前 fusion 输入、h、prior、z、进入检测头的特征、
全部原始 bbox/dir 输出、IoU 辅助分支输出、原 residual。**只允许 cls 输出与后续候选选择变化。**

固定的是 NMS 前所有 anchor 的原始回归输出，不是最终返回的框集合。分数改变后 top-k/NMS
选择不同，本来就是这个实验要测的效应；不得把这种集合变化写成“z 修正了几何”。

`prior_only` 必须保持输入、历史调用数、当前 fusion 输入、h 与 prior 不变，并断言返回的
`z_t` 指纹等于本次 prior 输出，证明两个部署 z 路径都吃到了干预值。

**当前帧识别使用 `extract_feat(feat_or_dict=1)`，不数第 N 次 fusion 调用。**
无效历史会跳过调用；按调用次数干预可能恰好漏掉困难序列。本工具记录历史实际调用数与
last-history-valid，后续分析必须单列无有效历史组。

### 2.2 输出与真实性能口径

每臂保存 `predictions.pkl`；identity/off/prior 另保存每帧 `candidates/*.pt`：

- 各 level 的原始 anchor ID、总候选数；
- 按部署规则实际进入 NMS 的 top-k class scores、box codes、anchors、方向类别；
- 与部署 decoder 相同的 decoded boxes（**方向校正前**）；
- 配对记录中的 sample ID、输入/特征/z/分类/回归指纹。

`report.json` 包含全部臂、隔离检查、源文件 SHA、checkpoint epoch、resolved config、类顺序、
head/test_cfg/方向偏移及全部 AP keys。正式 full-val 调用仓库 `dataset.evaluate`，不自己实现 AP。
以 `cfg.data.val` 建集；本次实测 val/test 配置相同，val 共 **2040** 个样本。

`--limit` 是管线冒烟，**一律不计算 AP**。只通过冒烟不能记作性能成功。
一次完整四臂是 **8160 次 N4 clip 推理**，并非四次训练；不承诺未经实测的耗时。

既有离线 harness 与 EvalHook 的差异（§51.5）仍存在。这里只比较同工具、同 evaluator 的臂；
不将其绝对 AP 与训练日志混算。reference/identity 只证明当前 harness 透明，不证明历史 evaluator
差异已经定位。B0 若参与横向排名也必须以相同口径重评，不能直接塞入历史训练 AP。

### 2.3 两条干预分别能证明什么

- `residual_off`：这份已训练权重中，**直接分类读出**的条件效应；不是“删除全部 z”，也不是该结构重训后的效应。
- `prior_only`：该权重中，当前 posterior 相比 prior 的条件任务效应；**仍有 z、原始视觉/雷达旁路和当前观测**。
  MotionAlignedRSSM 的历史对齐本身也用当前 feat，所以这不是“纯历史预测 vs 完整观测”的信息排他实验。
- 两者都不能独自证明随机采样、KL 或概率状态不可替代。不得将关掉后掉 AP 写成 stochastic 必要性。

## 3. 第二阶段：逐框对账，明确 AP 收益在哪里断掉

第一阶段获得 dump 后，先分析固定的 on/off 配对，不新训、不扫门。

1. **排序与筛选**：按 `(sample, level, anchor ID, class)` 对齐候选；统计进入/退出 top-k 的
   高 IoU 真候选、错误高分候选、重复框，以及 NMS 前后的真目标损失。分 Car/Truck/Ped/Cyc、
   距离、已有覆盖 high/low/zero、有效历史组报告，冻结分组边界，不用结果选组。
2. **匹配与 PR**：复用仓库 evaluator 的 moderate/ignore/DontCare/类别/IoU 规则及 PR；
   不能用“每框最大 IoU”假冒官方 TP，也不能用固定 score 阈值代替 AP。
3. **几何分解**：对固定候选池、固定对应，在分析中分别替换 xy、z、尺寸、yaw，再测 IoU/召回空间。
   oracle 只做诊断，明确使用 GT；不能部署、不能报成模型 AP。各分量有交互，单项修复量不能相加。
4. **只在仍需区分优化冲突时测梯度**：量 detection/semantic/recon/KL 对 z、h、共享参数的
   梯度范数和夹角，按匹配 fg/bg 分开。loss 标量比例不是梯度预算。

**实现边界**：本次工具保存的是部署 top-k 候选，不是完整 dense anchor 池；尚未实现上述
TP/FP、oracle、PR 分解器。两个臂候选的并集可追踪彼此换入/换出的框；两臂都淘汰的框不能
据此宣称“原始候选池没有”。若需要证明 top-k 前上限，另补流式 full-anchor 召回/计数，
不无条件落盘整个 dense map。方向校正必须沿 head 的真实约定还原后再匹配。

决策规则：

- on 的真实 AP 优于 off，且新增合格真框：直接读出已有任务价值；先保住它，再查剩余几何瓶颈。
- on 增加召回但 FP/重复/错框更多，AP 下降：先修候选级质量与排序，不再放大 residual。
- GT 格 CE 好、真实 AP 不改善：不能以局部代理宣布成功；查候选预算/部署目标对齐。
- on/off 均几何差，oracle 显示明确几何修复空间：优先改善观测与定位；几何 fusion 才有具体承重证据。
- posterior/prior 相近：当前 checkpoint 的额外 posterior 路径价值弱；先查信息与监督，不先加 KL。
- 各类抵消：完整列出类别 PR 与几何变化，不能用 Overall 藏掉 Ped/Cyc/Truck 损害。

## 4. 第三阶段：训练控制，回答“为什么必须是 z”

同一 fusion、检测头、参数预算、监督目标、训练时长、数据、seed 与初始化协议：

- 无时序控制：区分更好的单帧观测与时间信息。
- 同容量确定性时序状态：区分状态容量/额外监督与概率 latent。
- 同结构 posterior-mean 训练：尽量固定网络、目标与容量，隔离采样训练效应。
- 完整 RSSM：再比较 prior/posterior 建模及随机训练的额外价值。

这些是分别回答问题的匹配控制，不能把删 prior/删 KL/删 decoder/改旁路全混在一个
“deterministic”臂里。仓库已有 deterministic/fixed-noise/learnable-std 相关类，但旧配方与
本次 A2-R 不完全匹配，不能直接拿旧历史数字当新 matched control。

先 seed0 同配方筛选，若连局部预测与同 seed 成熟窗口净收益都没有就止损；通过后补配对
seed1/2。默认 ep12–16 五轮等权均值、全部 val epoch 峰值（值+epoch）、逐 epoch 全曲线与
逐类均值全部记录。三 seed 提供复现证据，不自动等于统计显著；frame bootstrap 也不能
替代训练 seed 波动。时间相关数据的重采样须按真实片段/序列成块，不假定帧独立。

若确定性控制同样好，应承认收益来自时序/监督/几何，不强行给 z 注入噪声换取论文措辞。

## 5. 怎样让 z 产生真正可用的任务信息

z 不创造额外传感器信息。目标是保存 **相对历史状态尚未被解释的、会改变对象类别/几何/质量
判断的当前观测**，并让部署 decoder 使用它。

### 5.1 先修目标与部署对应，不再另造一张好看的辅助图

当前 A2-R 的主要结构限制已经由源码确认：辅助 `sem_head` 不部署；部署是另一独立 1×1；
同格同类所有 anchor 共用同一个增量；新增直接路径只改 cls，不直接修 bbox/dir。
因此推荐的下一步不是“更强辅助头”，而是**真实候选条件化的同一任务读出**：

- 输入包含局部 z、候选 anchor/object 几何和必要的 h/当前观测 context；候选不同，修正不同。
- 训练与部署用同一个 decoder；分类按实际正负 anchor assignment 监督，回归按实际 bbox coder
  的目标监督，方向沿原 head 约定；如引入质量项，其目标与实际 decoded box IoU 对应。
- 不以 pooled GT 格 softmax CE 替代实际 sigmoid classification，不把每类 max-over-anchor 混成一个对象。
- 输出类别与几何的任务相关修正，而不是四类 broadcast 替所有 anchor 加分；先保留
  零初始化残差和原始视觉 fallback，不为“证明 z 不可绕过”强关旁路。
- 同容量确定性特征输入必须训练匹配；否则仍可能只是更大 decoder 的收益。

这条修改需以第二阶段的断点为启动依据。本次**未实现或开训**；不能无视负结果直接把质量头、
box residual 与新 fusion 一起加上去，导致无法归因。

### 5.2 信息源与表示：几何观测先于潜变量噪声

新的 GP-MSF 应保留原始 image 多尺度/雷达局部几何证据与缺测回退，先证明真框几何收益，
再冻结 fusion 做 RSSM 控制。不能同时改变 fusion 和 z 后把 AP 增益全算给 z。
对零回波小目标，camera 必须能创建/支持候选；雷达缺失不能成为对象不存在的硬门。

真正对象状态 `(h_i,z_i)` 是后续更强的机制路线，不是把 dense heatmap 重命名为 object RSSM：

- 同一个对象/候选 i 有可追踪的历史、局部几何观测和 decoder；
- `p(z_i|h_i)` 给历史条件预测，`q(z_i|h_i,o_i)` 接入局部当前观测；
- 对应、birth/death、无历史与遮挡处理都显式；先验证关联，不假设 TJ4D 有可用真实 ego pose；
- 直接生成/修正实际 class/box，未来任务使用可靠对象对应，不能混入别的对象或共享背景模板；
- 若要宣称概率多假设优势，必须在遮挡/稀疏的歧义样本上验证条件预测质量与校准，并与
  mean-only/确定性控制匹配。不能靠固定噪声、熵变大或 best-of-K oracle 证明它有价值。

这个路线依赖已经过验证的对象观测器与关联（TODO 方向 2/4/5），不能当成本周改一层 conv 就完成。

### 5.3 KL 的位置

先让任务目标奖励有效的信息使用，再让 KL 约束表示成本/预测一致性。
per-element free-bits 下部分 KL 无梯度是目标行为，**KL 变大不是成功条件**。
只在 prior 预测/表示利用已有正任务证据、且梯度诊断显示具体优化问题时，单变量测 balancing
或 free-bits 聚合；不复活此前失败的低维未来模板、路线 B 或独立语义头。

## 6. 本轮完成与下个窗口的执行入口

已完成：

- 预测级四臂 runner，identity/隔离/源文件冻结检查、真实 AP 和部署候选落盘。
- 11 条 CPU 单测（最终结果见训练记录 §96），覆盖缺失历史、干预落点、方法恢复、score-only
  不变量、IoU 分支、top-k/decode 落盘、错误干预拒绝、指纹的 shape/dtype/content。
- 真实 val batch CPU 构建与指纹，2040 样本；真实 ep6 checkpoint strict load；真实 Anchor3DHead
  2×3 CPU score-only 对照：bbox/dir/IoU 分支逐位相同，cls 确实改变。
- 预检 artifact：`docs/rssm_z_prediction_preflight_20261013.json`。

注意：项目 constructor 会初始化 CUDA context；不把本次模型构建写成“完全未接触 GPU”。
模型参数和小尺寸 head 测试在 CPU；**未做真实传感器 GPU 前向、GPU S0、full-val 或新训练**。

执行（只有 GPU5/6/7 确实空闲时；先固定 ep6 补证，再在预登记的 ep16 复核成熟状态）：

```bash
rtk proxy env CUDA_VISIBLE_DEVICES=5 \
  /home/lurui/envs/miniforge3/envs/r4det/bin/python \
  me_rssm/sanity/paired_z_prediction.py \
  --config me_rssm/configs/TJ4D-R4Det_fgfull_N4_2x4_24e_pretrained_v2_head_no2d_igdr_semA2R_24e.py \
  --checkpoint /data/lurui/work_dirs/fgfull_N4_no2d_igdr_2x4_24e_semA2R_seed0/epoch_6.pth \
  --out-dir /data/lurui/work_dirs/z_prediction_ep6_smoke_20261013 --limit 16
```

冒烟全部通过后，同一命令去掉 `--limit 16`、换成独立未存在的 `--out-dir` 才做 full-val。
每个输出目录不覆盖旧证据；失败目录保存 report，不换门槛“复活”失败。
成熟复核固定 ep16，不事后挑 AP 峰值。

推荐排程：A2-R 正常完成后先给这批诊断留窗口，F1 后置；**本轮没有修改 F1 队列或启动新任务**。
当前队列仍可能在 A2-R 后自动启动 F1，不能声称本工具已排上。若批准改变排程，再显式调整队列。

本任务区间均值：N/A（无新增 val，n=0）；全轮峰值：N/A（无新增 epoch/AP）；best saved：N/A。
工具就绪与证据完成是两件事；预测级归因、匹配训练和 z 总目标仍未完成。
