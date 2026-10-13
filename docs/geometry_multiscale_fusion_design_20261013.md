# 方向 1：几何接地、配对取证的原生多尺度融合设计

> 2026-10-13（Asia/Shanghai）。**设计论证 + CPU 数学/数据核验，不是实现、冻结预登记或训练授权。**
> 首轮研究因素只有前端观测融合；保留当前 `MotionAlignedRSSMFusion`、Anchor3DHead、N4 和全部既有损失/配方。
> 不启动/停止训练，不插队，不改变 A2-R/F1 排程。设计完成不等于 TODO 的完整候选验证完成。
> CPU 证据：`docs/geometry_multiscale_fusion_audit_20261013.json`。
> 训练记录：`docs/training_runs_full.md` §91；原始负结果见 §§40、87。
> 本任务：**区间均值 N/A（没有新增 val 曲线）**；**全轮峰值 N/A（没有新增 val 指标/epoch）**；best saved N/A。

## 0. 结论与不做什么

建议实现一个暂名 **Geometry-Paired Multi-Scale Fusion（GP-MSF）** 的单块残差融合器：

1. 保留现有 camera LSS BEV、radar SECONDFPN BEV 与原 ConcatConv 通路。
2. 新取证路径直接访问 **image 原生 FPN P2/P3/P4** 和 **radar SECOND 三层原生 feature**，不是把已融合 BEV 下采样后改名“多尺度”。
3. 在当前 BEV 坐标中预测有界米制地址，将**同一物理地址**映射到各层 radar，以及带有限高度假设的 image 投影位置；跨模态兼容性只能比较这些配对地址。
4. 雷达向相机提供局部空间证据，相机向雷达/最终融合提供重新提取的语义；没有局部回波时允许 radar message 为 null，但不关闭 image message。
5. 新消息通过零初始化的线性输出投影加入两路特征，仍由原 `fusion_conv` 生成 RSSM 观测；step 0 的目标是与同初始化 ConcatConv 完全一致。

**不做**：同时更换 RSSM/head、增加对象 decoder/二阶段 RoI、重做完整 depth-height volume、增加 occupancy/Doppler 辅助任务、重复 F1 的学习可靠性门、堆 3–6 层 attention、无上界全图采样。

核心问题不是“attention 是否更先进”，而是：**当前融合在观测已经错配/混合之后只改强弱；能否通过可追踪的几何地址，重新取得互补的原始尺度证据，同时不把弱雷达目标变成禁用区域？**

这是假设，不是已经证明的根因；若任务 AP 或机制消融不支持它，就不能把该解释写成结论。

## 1. 旧 Cross_Modal_Fusion 到底做了什么

源码：`mmdet3d/models/necks/BEVCross_modal_attention.py:13–50`。

设两路 BEV 为 C、R：

```text
A_C = sigmoid(conv([channel_mean(C), channel_max(C)]))
A_R = sigmoid(conv([channel_mean(R), channel_max(R)]))
F_old = ConvBNReLU_3x3(concat(C * A_R, R * A_C))
```

- 每路只有一个 H×W 标量空间掩码，通道维的信息被 mean/max 压成两张图。
- 互乘发生在相同 cell；掩码卷积和末端卷积能看局部邻域，但没有显式建立另一路的跨位置对应。
- 没有 q/k 内容匹配、米制 reference、标定投影、source-level 选择或 learned sampling offset。
- 文件 import 了 `MultiScaleDeformableAttention`，不等于此类使用了它；同文件 `Virtual_pts_BEV_Warp` / DCN 类是不同实现。
- sigmoid 掩码只能在 0–1 间缩放原特征；最终卷积能重新放大，因此不能断言最终网络只会抑制，但取证位置没有改变。
- 雷达弱/缺测处的掩码仍可能由 bias 或周边上下文产生，并不等于零。可疑机制是“弱雷达控制视觉证据强弱”，不是已测到 A_R 必为零。

当前 `ConcatConvFusion` 则为 `fusion_conv(concat(C,R))`，基础信息不会先被两路 sigmoid 互相裁掉。它也缺少显式位置匹配，但不能因此否认其强基线价值。

### 1.1 历史失败不能跨栈误归因

本次直接重读原 `*.log.json`，每 epoch 取最后一次 val：

| run | fixed window ep12–16，n=5 | 全轮峰值 | 当前 best saved |
|---|---:|---:|---:|
| 旧同栈 ConcatConv：`run10_headv2_multiseed/seed_0` | 38.3902 | 39.8802 @ ep16 | 39.8802 @ ep16 |
| 旧 CrossModal：`crossmodal_fusion_N4_2x4_24e_seed0` | 37.9789 | 39.2835 @ ep16 | 39.2835 @ ep16 |
| 当前 B0：`fgfull_N4_no2d_igdr_2x4_24e_seed0` | 39.2809 | 40.3853 @ ep14 | 40.3853 @ ep14 |

旧 CrossModal 对**旧同栈**基线：Overall −0.4113、Car strict −2.0400、Truck strict +3.2709、Ped loose −3.2296、Cyc loose +0.3537（§40）。
它比**当前 B0**低 1.3019，是历史能力参照，不是当前 no2d_igdr 栈中的单变量复验：FG/depth 训练链不同，不能把 −1.3019 全归因于 fusion。

另外 §40 的换模块同时恢复了预训练 `reduce_mixBEV` 权重；旧 ConcatConv 不加载该命名空间。该实验结论针对“恢复预训练 CrossModal”整个干预，不纯粹识别互乘算子的因果效应。

因此：旧模块失败不能否定几何多尺度取证，也不能作为未经重训的当前同栈 attention 控制。

## 2. 本项目的真实瓶颈与证据等级

### 2.1 已观测事实

1. 当前 B0 的四个主构成项（ep12–16）：Car strict 47.8399、Truck strict 29.7246、Ped loose 30.9277、Cyc loose 48.6313。
   Overall 是这四项的等权平均，不是四类全 strict。
2. C1 的成熟窗口只提升 Overall +0.5494，却使 Car/Truck +6.3237/+6.7151、Ped/Cyc −4.8880/−5.9530；全轮峰值 40.6389 @ ep16。
   §87 同 harness ep14 分层 recall：high +4.4137、low −6.7401、zero −2.1023。这证明该臂的非对称收益重分配，不证明所有融合只能零和。
3. `R4Det.py:822–825` 将 image 的五层 FPN 插值到 stride 8 网格再拼接，depth/context 网络输出一个 LSS BEV。radar 三层 native SECOND 经 SECONDFPN 上采样、拼接成 384 通道单网格。
   多尺度信息并非全消失；雷达通道仍可能保留层级来源。但 fusion 不能独立选择原生分辨率中的采样地址。
4. 当前 resolved `zbound=[-4,2,6.0]`：**只有一个高度 bin**。`mean(-1)` 本身是 no-op；高度表征不足发生于一层高度网格设计，不能错误归因于“最后均值抹掉了多层”。
5. B0 三 seed 的 ep12–16 均值为 39.28088、41.91848、40.12066；均值 40.4400，**样本标准差 1.3475（ddof=1）**，总体标准差 1.1002。
   一个 seed 的 +0.5 不能当稳定研究收益，1.35 也不是单个差值的显著性门槛。

### 2.2 本次 train-only 几何统计

训练信息 5,706 帧，SHA256 见 JSON；按 dataset 的 camera→LiDAR 转换、有效类别及 BEV xy 范围得到 23,029 个框。它们是设计统计总体，不是 AP 分母。

| 类别 | 框数 | 中位长度/宽度/高度，m | 宽度/0.32m |
|---|---:|---|---:|
| Ped | 3,242 | 0.6555 / 0.6273 / 1.6879 | 1.96 cells |
| Cyc | 5,125 | 1.7895 / 0.7560 / 1.5912 | 2.36 cells |
| Car | 11,030 | 4.5212 / 1.8308 / 1.6333 | 5.72 cells |
| Truck | 3,632 | 10.0189 / 2.6735 / 3.7096 | 8.35 cells |

含义：小目标横向只有约两格，粗层/错误邻域的证据容易混入相邻对象；但不能由此断言提高 BEV 分辨率必成功，§38 的相关支线已经是负结果。
本方案首先提高**取证分辨率与对应性**，不同时更换 head 分辨率。

### 2.3 待证假设与可反驳结果

- **H1，空间错配**：同 cell 加权无法修复相机 lifting 与雷达空间证据的局部偏移。预测：有界 learned offsets 优于固定采样，且减少可配对检测的横向/径向中心误差。
- **H2，尺度/语义混叠**：原生 PV/native radar 取证对小目标更有帮助。预测：原生 source 比“同容量、只读旧 BEV 金字塔”强，并提升 low/zero 的小目标 recall，而非只提升 high。
- **H3，弱雷达依赖**：将无支持视作 null 的新增雷达消息，比强制跨模态响应更安全；相机仍能直接取证。预测：移除 camera fallback 会伤害 zero/low，强制取消 null 会增加错误关联/FP。
- **H4，RSSM 观测接口**：更一致的观测能改善状态估计，但不是必然。预测：RSSM×fusion 的配对交互、观测干预和位置诊断支持该机制。

若只有 Truck/Car 增益、Ped/Cyc 或 zero/low 回退，H2/H3 未被支持；若固定采样同样好，不能声称 learned correspondence 是收益来源；若同容量 conv 同样好，不成立“几何优于容量”。

## 3. 论文对比：取哪项能力、不照搬哪些代价

这里只比较机制；论文 TJ4D test mAP、nuScenes NDS 与本项目 validation Overall moderate 不是同指标，不做直接排名。

| 工作 | 原文/作者材料核到的关键机制 | 本项目取舍 |
|---|---|---|
| R4Det [P1] | PDF、pose-free DGTF、IGDR 分别解决深度、时序和缺回波视觉恢复 | 当前主线是 RSSM + no2d_igdr 变体，不等于原文全部模块。保留训练中的 FG/relative-depth，不以“第一次加深度监督”立项；用原图取证而非重启已试过的 2D instance/IGDR |
| SGDet3D [P2] | 双分支几何/语义互补，BEV query 通过 localization-aware cross-attention 读取 image tokens | 学习“回到原图取证”，不将它重新命名为本项目首创；本轮不重写 depth completion / semantic pillar encoder |
| Doracamom [P3] | 几何与语义初始化 voxel query；BEV/voxel 双路时序；CMF 及多任务监督 | 借鉴融合不能只在已混合单表示上调权。暂不引入完整 3D voxel decoder、occupancy GT 和双路 temporal，避免与 fusion 因素混杂 |
| DR-Net [P4] | 已核实 Cao/Ling 的完整题名及 DOI，定位是基于 4D radar 的检测；**未取得原始全文** | 不把它当已经核验的 camera cross-modal fusion 实现；本次不借未核细节设计算子。点级细节/运动编码若另做，是 TODO 方向 6，不混进方向 1 |
| CVFusion [P5] | Stage1 用预测 radar occupancy 加权 camera、逐级构建 BEV；Stage2 在 proposal 上聚合 point/image/BEV 证据 | 取“重采样干净证据”的原则，不同时改两阶段/head。其完整对象级取证属于方向 4，不把 stage2 的收益冒认成单层 attention 收益 |
| RCFusion [P6] | 出版题录核实；FPN/OFT、PillarNet 与 IAM 的组织由 LXL/MSSF 原文交叉核对，未取得 RCFusion 原始正文 | 旧模块属于同类 BEV 交互加权组织，但不能在未逐层对照时声称代码完全复制 IAM；这种能力不足以解决显式跨位置对应 |
| LXL [P7] | sampling-based lifting，depth-assisted 与 predicted radar-occupancy-assisted 两支 | 借鉴以 3D reference 拉取 PV。其 occupancy 是预测的软场，不是 input 点是否存在，不能误说 LXL 会把无雷达处硬置零；本方案的硬约束只约束新增直接雷达证据，不限制 image 通路 |
| RCBEVDet / ++ [P8] | CAMF 可变形对齐；++ 扩展至 sparse image query 与 dense radar 的融合 | 借鉴局部动态对齐，但双向 deformable 本身已存在。首轮保持 dense BEV/RSSM，不能凭 ++ 的 sparse decoder 效果保证当前栈收益 |

**必须补的最近邻**：MSSF [P9] 已有 multiscale deformable feature fusion；SGDet3D++ [P10] 已有 anchor-consistent radar 支持驱动图像检索、几何 refinement 与 Doppler history 验证。
因此“标定投影 + 多尺度 + 双向 attention + RSSM”不是可直接认领的新颖性。P10 已公开，不能因为只比较旧 SGDet3D 就宣称 radar-grounded semantic retrieval 首创。

文献覆盖边界：P1/P3/P5/P7/P8/P9/P10 阅读原文相关 Method；P2 核作者仓库的架构和摘要；P6 核出版社题录并对照 P7/P9 的原文文献回顾，未取得其原始正文；P4 仅题录与出版社存储的元数据。没有把第三方博客的模块解释当原文证据。

**两项反向教训**：CVFusion Stage1 也有 occupancy-weighted camera fusion，不能把其两阶段整网收益都解释成“attention 更强”；MSSF 明确区分 2D 前景投影与真实 3D 前景，几何投影合法不等于对象关联正确。
因此 shared address 只让配对可追踪，不能单独消除遮挡/同 ray 多对象；配对内容、null、FP 与错误关联诊断缺一不可。

## 4. 输入、输出与严格修改边界

基础通路保持：

```text
Image -> ResNet/FPN -> 原 depth/context -> 原 LSS -> C (256,248,216)
Radar -> Pillar/scatter -> SECOND -> 原 SECONDFPN -> R (384,248,216)
                           原 ConcatConv([C,R]) -> (256,248,216)
                           原 permute           -> (256,216,248)
                           MotionAlignedRSSM -> Anchor3DHead
```

额外保留的 source（不修改 backbone 的结构）：

| source | 通道 | H(y)×W(x) / image H×W | 原生步长 |
|---|---:|---:|---:|
| image P2 | 256 | 120×160 | 4 pixels |
| image P3 | 256 | 60×80 | 8 pixels |
| image P4 | 256 | 30×40 | 16 pixels |
| radar R0 | 64 | 248×216 | 0.32 m |
| radar R1 | 128 | 124×108 | 0.64 m |
| radar R2 | 256 | 62×54 | 1.28 m |

图像表按 480×640 的标准输入、现有 FPN 配置推导，实施时必须以实际 shape/assert 为准；P5/P6 原基础 depth 通路仍保留，不为第一臂额外全采样。
新模块只读当前帧 `img_feats`、SECOND outputs、预测 `depth`、标定/augmentation、input radar xyz。**不读 GT box/label/segmentation/my_gt_depth/未来帧**来决定取证地址或 mask。

采用 opt-in 接口，不改旧融合器签名，不通过共享全局 bus 保管当前帧多尺度张量，以免 N4/多 batch/跨帧状态污染。

## 5. 几何契约：这是设计的一部分，不是实施时再猜

### 5.1 参考点与坐标布局

在 fusion 输入布局 `[B,C,y,x]` 中，采用 LSS cell 中心：

```text
x(j) = 0 + (j+0.5)*0.32
y(i) = -39.68 + (i+0.5)*0.32
```

每个 query 是固定 dense cell，不是新的 object query/box。输出仍只在原来的位置 permute 到 RSSM 的 `[B,C,x,y]`，不可额外交换 x/y。

高度 references 只由 train split 拟合：每个有效框取底面 + 0.25/0.50/0.75 倍高度；去掉 z-grid 外 surface，先每对象保留 surface 等权，再每类/类内对象等权；用 left-inverse weighted CDF 的 q10/q50/q90：

```text
z_ref = [-1.1230436265, -0.3003779650, 0.6601446867] m
```

这三个固定常数是 sampling reference 的覆盖先验，不是某类/某实例的 GT 高度、不是真实高度预测，也不是新建三层全局 voxel volume；有明显平地/坡道和 Truck 顶部覆盖限制。
原工作区存在一个未跟踪 CPU 草稿，本次未改/未提交；不得混用其中不同的 quantile 常数，本次算法与统计完全自含在新 JSON。

当前 BEV augmentation 是 yaw/scale/flip/translation。reference height 必须按本帧 BDA 的 z scale/translation 变换，再与 augmented BEV xy 组点；不能把 canonical 高度不加变换直接塞入 augmented grid。

### 5.2 地址到 image：先撤销 BEV augmentation，再完整投影

对 augmented 点 p_aug：

```text
p_native = inverse(A_BEV) @ homogeneous(p_aug)
cam = T_lidar_to_cam @ p_native
h = P_cam @ cam
(u,v) = h.xy / h.depth
(u_aug,v_aug) = A_image_2D @ (u,v,1)
```

等价齐次形式是 `h_aug=A_image_2D @ P_cam @ T_lidar_to_cam @ inverse(A_BEV) @ p_aug` 后除深度。

- camera depth 用相机坐标 z/投影深度，不能用 BEV 欧氏 range 代替。
- 保留 KITTI 3×4 P_cam 的第 4 列；不能只变换内参前三列后默认 stereo shift 为零。
- image 有效性：正深度、真实未 padding 图像 FOV；pad 边界、出界位置不参加 softmax。
- 与现有 `cam_aware` / `final_lidar2img` 的真实 pipeline 一致性必须实测，不能只凭变量名认定等价。
- 无效投影必须先映射到安全 grid 并清零 value；仅 mask logits 但让 NaN 留在 value 中，仍会出现 `0*NaN`，null 本身不能修复它。
- native image feature 的坐标按真实 stride/origin 映射，不用所有层统一 `u/8`；新 sampler 固定一种 align_corners 约定并测中心点。

CPU 本次只证明“真实训练标定 + 合成可逆 augmentation”的分步/复合数学一致性，1,280 点最大误差 4.55e−13 pixels；不等于真实数据 pipeline 已过接线门。

### 5.3 地址到 native radar：不能对所有层盲用范围归一化

原始 0.16m scatter cell 中心是 `(0.08,-39.60)`；SECOND 使用 k3/p1/s2，下采样后的**卷积 receptive-center origin**仍为该点，步长依次 0.32/0.64/1.28m。

```text
xi_l = (x - origin_x_l) / step_l
yi_l = (y - origin_y_l) / step_l
# new sampler chooses align_corners=False:
grid_x = 2*(xi_l+0.5)/W_l - 1
grid_y = 2*(yi_l+0.5)/H_l - 1
```

这些是 native 新采样器的 index bookkeeping，不是证明旧融合/head 已有语义错位；学习特征的语义位置还受训练影响。不能借此未经干预就改旧 neck 或 anchor 定义。
CPU `grid_sample` affine-field 三层最大误差 4.44e−15、3.55e−15、1.78e−15；仍需真实 tensor 的 impulse/四角/transpose 测试。

## 6. 配对取证块：不是两个独立、无限漂移的 attention

### 6.1 有界、跨 source-level 共享的米制地址

所有 source 先各自 1×1 投影至总维度 d=64；4 heads，每 head 16d；第一版只一块，不加全局 self-attention。
query 来自 C、R 和几何位置。以 **augmentation 后的 point-cloud 原点**为基准构造径向/切向基 e_r/e_theta；BDA 带 translation 时原点不再是 (0,0)，否则所谓 ray-aware 约束本身就错了。

```text
a_hk(p) = p + b_parallel_h*tanh(u_hk)*e_r
             + b_tangent_h*tanh(v_hk)*e_theta
```

初始覆盖半径只用原生步长，不冒充传感器噪声协方差：四头 `(b_parallel,b_tangent)` 提议固定为
`(0.64,0.32),(1.28,0.32),(0.32,0.64),(0.32,1.28)` m。
这些是有限支持的结构先验，目的是覆盖细粒度邻域与径向/切向错配；是否优于各向同性必须消融，不在首轮搜索半径。

每头 K=4，一个地址固定为中心；其他三个以径向两侧及该头的切向侧初始化。它们是跨 image/radar、跨 native level 共享的米制地址，而不是每个 image level 自己任意漂移的 pixel offsets。
实际最大欧氏半径也受两个分量共同影响（不是简单声称不超过 1.28m）；实施前枚举边界与 grid 可达性。

### 6.2 分尺度取证再配对，避免尺度 Cartesian-product

对同一 a_hk：

```text
r_hk = sum_l beta_R(l,h,k) * sample(project_R_l(R_l), a_hk)
i_hkj = sum_s beta_I(s,h,k,j) * sample(project_I_s(I_s), pi(a_hk,z_j))
```

beta 使用本 query 与 sampled key 的匹配、level embedding 做 masked softmax。先分别池化三个原生层，再做 r–i 配对；不枚举全部 radar-level × image-level × height 的组合。
reference 相同使层级选择主要回答“用哪种分辨率解释此处”，位置偏移回答“证据在哪”；必须有 shared-address vs independent-offset 的控制来核验这种约束是否真的有益。

允许用**当前预测** depth probability/entropy、投影深度差作为 image key 的输入描述。它们只提供软内容线索，不乘成 image hard mask，也不称为校准的 uncertainty/precision。
这样保留深度消歧机会，避免沿同一光线向多个深度复制语义；若不使用 depth 描述与使用它一样好，不能声称该线索必要。

### 6.3 内容匹配、配对一致性与双向消息

设归一化后的 r/i 兼容性为 `compat_hkj=<r_hk,i_hkj>/sqrt(16)`。第一版明确采用以下 score 组织，避免“配对”只存在于命名里：

```text
q_C = LN(P_C(C) + E_position)
q_R = LN(P_R(R) + E_position)
q_I = q_C + m_parent(p) * q_R
c_hk = logmeanexp_j(compat_hkj over valid image heights), or 0 if none valid
s_R_hk = <q_C, K_R(r_hk)>/sqrt(16) + gamma*c_hk + b_geometry_hk
s_I_hkj = <q_I, K_I(i_hkj)>/sqrt(16)
           + m_hk*gamma*compat_hkj + f_depth_descriptor_hkj + b_geometry_hk
M_R_h = sum_k softmax(s_R_h plus null)_k * r_hk
M_I_h = sum_kj softmax(s_I_h plus null)_kj * i_hkj
M_R = concat_h(M_R_h); M_I = concat_h(M_I_h)
```

式中 q/K、score 按 head 计算，M 最后拼接四头；gamma=softplus(gamma_raw)、初始化为 1，是非负的可学习匹配系数，不是可靠性 gate。
`m_parent` 表示 query 邻域有输入回波，`m_hk` 表示该采样地址邻域支持；二者定义相同但位置不同。
无 radar 支持时 q_I 退化为 q_C，compat 不控制 image；无合法 image 时 c=0，radar 仍可单独提供证据。
b_geometry 使用米制位移/相对径向切向坐标的 embedding，f_depth_descriptor 是预测深度描述的小投影，不给“正确深度”手工加无限权重。
雷达无支持或 source/image 越界的候选在各自 softmax 中 mask 掉；null 始终保留。不得把 source-level attention 的归一化与上述候选归一化混为一次大 softmax。

- **Radar→camera message M_R**：汇聚 r_hk，image 的配对内容可帮助拒绝不相干的邻近回波。
- **Image→radar/融合 message M_I**：汇聚 i_hkj，局部 radar descriptor 在有支持时帮助选择 semantic token；无支持时用 camera query+geometry+预测 depth 描述继续取证。
- 两方向分别归一化，不让密集雷达候选在同一个大 softmax 中把 camera-only 候选全部挤掉。
- 限制兼容性项是 learned score，不能宣称已得到对象身份关联；高度假设和遮挡仍可能造成错误配对。

没有新对象框/类别预测、没有新辅助 loss。首轮由既有 detection/FG/depth/RSSM loss 联合学习 fusion 参数。

### 6.4 input 支持、null 与 camera fallback

新增直接 radar evidence 的有效性来自输入 xyz 的局部回波支持，不用 GT occupancy、feature 幅值或模型预测框来假造“已观测”。

提议在原 0.16m scatter 网格建 occupancy，先以固定 5×5 邻域（中心距离最多两个 cell）得到局部支持，再按上述米制地址采样；这是局部支持窗口，不是 radar source receptive field 或对象归属证明。
输入点不在范围、地址越界或邻域没有输入点时，该新增 radar 候选无支持；每方向都有 value=0 的 null 项，避免全 invalid 的 softmax NaN 或强迫模型读取 padding。

```text
若所有新增 radar 候选无支持：M_R = 0
若所有投影 image 候选无效：M_I = 0
没有 radar 支持但 image 有效：M_I 仍可非零
```

null 包含可学习 score，但 value 恒零，不添加虚构 prior 特征。不使用 null 概率宣称物理测量噪声已经校准。

重要边界：
- “框内零雷达点”不等于“周边采样窗口零雷达点”，邻车回波仍可能被借入；自然 zero 组的效果必须真实评测，结构规则不能保证其 AP 不降。
- SECOND 在空 cell 也能有上下文特征；本约束只限制**新增直接雷达消息**，不删除原 R/ConcatConv 中的上下文。
- 全雷达缺失时保证新增雷达消息为零，**不是证明整个模型已变成纯 camera detector**，因为基线 radar 网络的 bias/上下文通路仍在。
- 不以没有输入回波推断物体不存在，也不以雷达点的径向速度直接取代完整二维运动。

### 6.5 基线保持式输出与初始化

```text
C_new = C + W_RtoC(M_R)        # 64 -> 256
R_new = R + W_ItoR(M_I)        # 64 -> 384
O_t   = fusion_conv(concat(C_new,R_new))
```

W_RtoC / W_ItoR 的 weight+bias 零初始化；其前面的 key/value/query 非零初始化，最后一层为线性，不接会在零点切断梯度的 ReLU，也不同时使用零 gate。
`fusion_conv` 保留名称/权重/norm，step 0 目标逐位等于 ConcatConv。输出仍是 256 通道，RSSM 不接受额外 mask/动作输入。

**初始化公平性**：同 seed 不能自动保证新增类之后的 RSSM/head 随机初始化相同。需从同 seed 基线构造的完整初始 state 映射相同模块，新参数用隔离 RNG 初始化；训练时不加 dropout/随机采样去额外消耗 RSSM 的随机流。
可用已有 checkpoint 做 step-0 identity 验证，但正式对照不能新臂 hot-start 已训练 B0、旧臂却从预训练重新训。
不能偷偷把旧 CrossModal 的 `reduce_mixBEV` 映射到新臂，而 B0 仍随机 `fusion_conv`。

第一步 W 应有梯度，第二/后续步 offsets、source selection 应得到非零梯度；step 0 offsets 因 W=0 暂无梯度是正常的，不把它误判为死分支，也不只测一次 backward 就宣称可学习。

## 7. 为什么有希望，以及为什么仍可能失败

1. **不把门的输出当新信息**：新分支真的改变取证位置、分辨率和 PV/BEV 域，区别于 F1 同位置缩放。
2. **保留再获取通道**：原 LSS 语义错位或弱证据时，真实 3D reference 可以再次读取 image fine feature，而不只能把错位 BEV 放大。
3. **小目标不必依赖粗层**：保留 P2 的边缘/轮廓及 radar R0 的细尺度；粗层只供上下文，由动态 level selection 选择。
4. **密集目标仍有上下文**：R1/R2 与 P3/P4 对 Truck 的大空间结构有帮助，不为 Ped 单独另开一套 head/监督。
5. **缺测不变成视觉禁令**：null 作用在新增雷达证据，image 可以 fallback；这针对 C1/F1 风险，而非把 alpha 换名字。
6. **RSSM 前改善观测而非同时改更新**：让后续状态模型吃更一致的 O_t，能隔离性能来自观测还是时序。

失败风险与最便宜的证伪：

| 风险 | 必看证据 | 若发生该怎么办 |
|---|---|---|
| image 一条 ray 的多个深度共享语义，形成 extruded FP | 无回波/远距离 FP、径向误差与预测 depth 支持 | 不堆更多 attention；判当前取证不足，考虑后续真正 depth-height/对象假设路线 |
| 稀疏 query 借邻车雷达 | GT 仅在离线使用的 sampled-source 同对象率/邻对象率、拥挤分层 AP | 不把局部 occupancy 当身份关联；配对约束无效就否定 H3 或另立对象关联方案 |
| 三个固定 height 在坡道/高 Truck 上不足 | 高度/场景分层、单高度 vs 三高度控制 | 有界追加高度表示需另立唯一因素，不能后台自动扩展 volume |
| offsets 学到远处背景/饱和边界 | 米制 offset 分布、tanh 饱和率、sampling 落点与扰动响应 | 先定位 content/geometry 接线，不能默认再放大采样半径 |
| 只是更多参数/预训练初始化收益 | 同容量 conv、same-init 校验 | 未超过就不能宣称 correspondence 的机制贡献 |
| 改善 detection，却让 RSSM 学重建捷径 | AP 干预、RSSM/GRU 交互及 h/z 排他诊断 | 前端 AP 增益和 RSSM 创新分开写，不用低 reconstruction/KL 代替任务收益 |

## 8. 最小消融矩阵与停止扩大范围的规则

第一轮只实现一个完整块，不先给十几种模块跑高参数网格。下面是待预登记的验证矩阵，不是当前已排队 run。

| arm | 内容 | 要回答的问题 |
|---|---|---|
| B0 | 当前 ConcatConv + 原 RSSM/head | 同栈基线 |
| Bconv | 同位置/局部普通 residual conv，新增参数和实测计算尽量匹配 | 是否只是容量 |
| Bbev | 原 BEV pyramid 上的局部 deformable，预算匹配，无法读取 PV/native 细层 | 原始证据是否必要 |
| Gfixed | 相同 source/高度/null，固定米制采样 pattern，禁用 learned offsets | 学习地址是否必要 |
| Gsingle | 保留相同几何/两域/null，仅一 native level；投影宽度补齐使容量可比 | 多尺度是否必要 |
| Gindependent | 两域独立 offsets，其他相同 | shared metric address / pairing 是否必要 |
| Gno-null | 取消新增雷达 null；只处理真正越界，强制就近特征 | 无支持拒绝是否必要 |
| Gno-fallback | 无局部 radar 支持时关闭新增 image message（仅作负控制） | 缺回波图像通路是否必要 |
| Gno-compat | 地址共享，但移除 r–i compat，保留各自 query-key | 配对内容条件是否必要 |
| G1z | 同方案只有训练高度中位数 | 三高度 reference 的能力代价 |
| Gno-depthdesc | 不给 key 预测 depth/entropy 描述 | 软深度消歧是否必要 |
| Gfull | 全方案 | 任务与各因素效应 |

实践顺序：**B0 / Bconv / Gfixed / Gfull** 先确定新取证是否有效；有正结果再补 Bbev/Gsingle 和其余归因。
其他消融须尽量控制训练预算；不能把只有少量有效 keys 的 no-fallback 称为完全等 FLOPs 的公平容量对照。计算/参数无法严格匹配时报告残余差异。

若 Gfull 成熟窗口没有可复现任务收益，不自动叠加新 loss/new head/new RSSM；先利用现有矩阵分辨实现错接、地址机制不成立或观测信息本身不足，然后停止当前臂或另立项。

## 9. 和 RSSM 论文主线如何建立实质关系

前端单臂的 AP 增益只能证明更好的观测融合，不能证明随机 latent z 必要。

在 Gfull 有成熟正结果后，做配对 2×2：

```text
              ConcatConv     GP-MSF
RSSM               A            B
原 GRU             C            D
interaction = (B-A) - (D-C)
```

若 RSSM 与 GRU 增益相近，结论是一般前端改善；若 interaction 为正，仍需噪声/缺测干预及状态诊断，不能直接称随机状态贡献。
另做 h/z 排他、置换/消融或同容量确定性对照；保持评测 pipeline 一致，报告任务 AP 而不只报告 latent 统计。

**候选机制级贡献**：分析几何观测混叠与弱雷达对象的收益重分配，设计“配对原生取证 + 无支持拒绝 + 非雷达依赖视觉通路”的 dense RSSM 观测构造，并用 AP/位置/错配与 latent 干预闭环。
这与 P10 的 evolving object anchor/return association/history verification 不同，但“不同”不等于“新颖”。投影、MS deformable、残差/null 本身都不是首创。若最终既无稳定 AP 大增、又没有可区分既有融合/时序方法的机制结果，仍不满足项目两大目标。

## 10. 工程约束与 S0（未执行）

### 10.1 接入点

- 新类放在独立 `mmdet3d/models/fusion_layers/geometry_multiscale_fusion.py`，不重写旧 `Cross_Modal_Fusion` 以免历史复现改变。
- `R4Det.extract_pts_feat`（721–736）按 opt-in 返回/传递 neck 前 SECOND outputs；默认路径返回完全不变，不能再次算 backbone。
- `extract_feat`（812–901）传当前帧 `img_feats`、native radar、标定/augmentation、预测 depth、input points 给新融合器；不改变 FG/RSSM loss 和输出 transpose。
- 注册/独立 config 继承 no2d_igdr，唯一研究模块是 RCFusion；原 API caller 的训练/测试兼容性须有 regression。
- 历史帧沿现有训练图策略处理，每帧使用自己的标定/augmentation，多帧不得复用上一帧 native tensor 或 geometry；不假定 TJ4D 有可靠 ego pose。

### 10.2 算力不是“稀疏 attention”四个字

虽然不用 N² attention，dense canvas 仍有 N=248×216=53,568 queries。
K=4、3 radar levels、3 image levels、3 heights、64d、bs=2 时，朴素全 materialize sampled values：

```text
2 * 53568 * 64 * (3*4 + 3*3*4) = 329,121,792 elements
```

仅这一项约 0.613 GiB FP16 / 1.226 GiB FP32，尚不含 gradients、attention logits 和原网络。4 heads×16d 已体现在 64d，不能再把 head 维遗漏或重复计算。
query chunking 只能降低某些 forward 暂存峰值；反向仍会保留中间图，除非采用精确 recomputation/checkpoint/fused operator。不能假称分块自动解决全部训练内存。

设计预算：第一版一块 64d，新增参数目标 <0.5M；实测迭代耗时目标 ≤基线 1.20×、训练峰值内存增加目标 ≤20%。这些都是**待测目标**。
必须先在原训练 bs/梯度累积/N4 上 S0；不能为跑通偷偷降 batch/换分辨率而继续称单变量对照。只用 GPU 5/6/7，且不抢现役队列。

### 10.3 必须过的接线测试

1. 同 seed/初始状态/训练与 eval 两模式，step-0 输出及预测与基线一致；原模块初始化 RNG 检查。
2. 零雷达点、部分 batch 零点、全 image invalid、全采样越界，输出有限，无跨 batch 污染。
3. 单点/affine/边角坐标测试，所有 source 的同一米制位置正确；xy↔yx transpose 严格一次。
4. 真实 train augmentation meta 的投影 round-trip，包括翻转、缩放、translation、P_cam 第 4 列；与 LSS geometry 和现有 metadata 对照。
5. 不注入每样本 GT 的 forward 接口检查；height 常数仅 train 统计，未用 val/test 调整。
6. 两步以上 backward：output W 先学习，随后 offset/source weights 得到非零梯度；历史帧图/当前帧图符合原策略。
7. 原 bs/N4 的 450-iter smoke，新增所有可训练参数 DDP 同步；native features 不是 detach 缓存误接。
8. 运行时 OOM/数值/越界/RSSM API 不一致时停止；不以 ep6 AP 单点提前否定成熟效果（§81 已证明早期门预测弱）。

## 11. 任务判据与记录模板（建议，尚未冻结）

### 11.1 任务效应

- 同 stack、同 seed 配对，主比较 fixed **ep12–16 五轮等权均值**；全部 val epoch 的峰值记“数值 @ epoch”；已存最高点独立列。
- 三 seed 配对平均 ΔOverall **≥+2.0 AP points** 才作为“大幅提升”的候选目标；报告每 seed 的 Δ，不拿最佳 seed/最大单点替代。
- 报告 Ped/Cyc 是否成熟窗口平均回退、各 seed 一致性与 coverage 分层。暂以 Ped/Cyc 平均不回退为希望达到的约束；具体非劣容限须在实施前结合历史波动冻结，不能看完结果再放宽。
- 单 seed +2 也不是统计显著性证明；三 seed 有限，必须如实报告不确定性。不把 1.3475 std 当严格阈值，也不做跨 harness AP 加减。

### 11.2 完整记录，不能只留摘要

正式训练须保存配置/唯一变量/代码及初始状态 hash/训练起止/中断/所有 epoch val/权重，逐 epoch 表至少包括 Overall 3D/BEV 与四个主构成项；窗口逐项均值、全轮峰值和 best saved 同列。

机制诊断分层：
- 四类、range、拥挤/遮挡以及 radar high/mid/low/zero 的 recall、TP/FP/FN（分组定义固定）。
- 每组 sampling 有效率、source-level 权重、null 占比、径向/切向偏移和 saturation；这些只是描述量，不能替代 AP。
- offline GT 仅用于采样来源的同对象/邻对象率及中心误差分析，绝不能回到 forward 作为 mask。
- 基线 FN 转 TP 的 recovery、基线 TP 变 FN 的损害、matched-set 定位误差与总体召回同时报告，避免只筛容易匹配的 TP 造成选择偏差。
- 遮挡反例及 depth-ray duplicated FP 的可视化；错误几何扰动/打乱 calibration/冻结 offsets 的**干预**验证。
- RSSM h/z 排他、任务输出和计算/内存实测，若未做就写未做，不用 attention 图或低 KL 宣称机制成立。

## 12. 当前已完成与还没完成

已完成：源码/同栈负结果核验、原始论文及最近邻对比、完整设计/输入输出/坐标契约、train-only 几何统计、CPU 投影/采样数学测试、消融与失败解释设计。
未完成：实际 GP-MSF 实现、真实 pipeline 几何接线、GPU S0、冻结预登记、正式训练、模型 AP 改善/新颖性确证。不得把这些写成已验证。

仅文档和 CPU 报告入库；现有未跟踪 `docs/artifacts/` 保留不动。

> **状态更新（2026-10-13，§97.6）**：该未跟踪目录与其中唯一文件
> `docs/artifacts/geometry_multiscale_fusion_design_cpu_20261013.json`（1,912 B，
> sha256 `080f6acb…d44487a`）已按用户指示**删除**，原因是它属于**被取代**的中间推导：
> 其高度参考为 `[-1.1266211569, -0.3083836436, 0.6394495964]`，与 arm/config 实际冻结并使用的
> `[-1.1230436265, -0.3003779650, 0.6601446867]`（见 `docs/geometry_multiscale_fusion_audit_20261013.json`
> 的 `reference_heights_m`）不同，两者拟合种群相同（5706 帧 / 23,029 个格内框）。
> 权威数据仍由已入库的完整 audit 提供，本节其余结论不受影响。

## 13. 原始资料与覆盖级别

URL 仅用于复查，固定本次阅读版本；文献表中的机制摘要不冒称复现结果。

```text
[P1] R4Det — Method §3 / Ablation §4.4（原文）
https://arxiv.org/html/2603.11566v1
[P2] SGDet3D — 作者 README/架构说明（未取得完整出版 PDF）
https://github.com/shawnnnkb/SGDet3D
[P3] Doracamom — Method III-C/F（原文）
https://arxiv.org/html/2501.15394v3
[P4] DR-Net — Cao, Ling；原始全文未取到，仅出版社 DOI/题录元数据核验
https://doi.org/10.1109/TCSVT.2026.3651715
https://api.crossref.org/works/10.1109/TCSVT.2026.3651715
[P5] CVFusion — Method §3（原文）
https://arxiv.org/html/2507.04587v1
[P6] RCFusion — 出版题录；粗结构对照 P7/P9 原文回顾，未取原始正文
https://doi.org/10.1109/TIM.2023.3280525
[P7] LXL — III-D/E（原文；depth-assisted 和 occupancy-assisted 是并行两支）
https://arxiv.org/html/2307.00724v3
[P8] RCBEVDet++ — III-B / IV-A（原文；区分原 CAMF 与 ++ sparse fusion）
https://arxiv.org/html/2409.04979
[P9] MSSF — fusion block / MSDFF（原文）
https://arxiv.org/html/2411.15016v1
[P10] SGDet3D++ — AGR/GCR/DVC（原文；新颖性最近邻）
https://arxiv.org/html/2609.27671v1
```

## 附录 A. 旧 CrossModal 完整 Overall 曲线（历史日志复读，非新评测）

| epoch | Overall 3D moderate |
|---:|---:|
| 1 | 16.2131 |
| 2 | 28.4627 |
| 3 | 28.4421 |
| 4 | 31.0702 |
| 5 | 33.1058 |
| 6 | 35.5144 |
| 7 | 38.6989 |
| 8 | 35.4494 |
| 9 | 37.3662 |
| 10 | 38.9072 |
| 11 | 38.3325 |
| 12 | 38.0084 |
| 13 | 37.5784 |
| 14 | 38.9041 |
| 15 | 36.1203 |
| 16 | 39.2835 |

完整逐类曲线与 B0/三 seed 的复读值保留在本次 JSON；旧训练运行事实/对照/逐项均值见 §40、当前主线事实见 §47 及后续多 seed 归档。没有改写旧结论或制造新 run。
