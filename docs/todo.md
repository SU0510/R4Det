# TODO

## 训练补充

1. 给 `fgfull`（完整 FG-FULL N=4）和 `temporal baseline` 补 seed1/seed2，以支撑消融的配对比较（未开始，需要 GPU 5/6/7 空闲窗口）。
   - 起因：no2d_igdr 三 seed 的 ep12-16 标准差 1.35，与 47.2 节单 seed 得到的 0.74 效应量同量级，配对比较目前的样本量不足以支撑结论。
2. 清理 `no2d_igdr` N4 RSSM seed2 的中间权重（未执行）。
   - 目录 `fgfull_N4_no2d_igdr_2x4_24e_seed2` 保存了 ep2-24 全部偶数 epoch，按 17.5 规则应只留 best saved（ep14）与 last（ep24），其余 10 个可删，约释放 6.4 GiB。
   - 该 run 在 2026-09-24 那轮清理时仍在训练，被整体排除，需并入下一轮。

## 训练效率

1. 加快每个 epoch 的 validation（当前约 61 min/epoch，其中 val 前向 ~10 min / IoU+eval ~1 min / ckpt ~1.6 min）。
   - **val batch size 已做（2026-09-26 完成）**：`R4Det.forward_test`/`simple_test`/`simple_test_pts`/
     `preprocessing_information` 的 batch 支持在 `a18f966` 落地，`data.val.samples_per_gpu` 提到 4（`b250e0d`）。
     完整 2040 样本 A/B：峰值 2.36/3.11/5.92 GiB、Overall 3D mod 38.4516/38.4947/38.5420；
     速度用交错测量 bs=1/2/4 → 271.8/238.6/230.9 ms/sample（1.14x/1.18x）。
     退回产生 ep16 的修订（`c0c53ba`）后同 harness 得到逐位相同的
     3D/BEV 数字，且无 batch 间串扰、DDP per-rank 顺序不变。详见 `docs/training_runs_full.md` 第 51 节。
     **结论：等效但收益有限**（纯前向 1.06x、端到端 workers=2 时 1.18x）。bs=6 OOM，故 4 为上限。
   - **真正的瓶颈是 dataloader，不是 batch size**：loader 单独耗时 workers=2/4/8 =
     111.2/59.9/34.1 ms/sample，占端到端约一半。下一步优先调 `workers_per_gpu` 2→4（约省 51 ms/sample），
     并注意 worker 增加会与训练侧抢 CPU（实测 workers=8 端到端反而回落到 1.14x）。
   - 零风险项（仍未执行）：把 `evaluation.interval` 从 1 改成 2（与 `checkpoint_config.interval=2` 对齐）。
     奇数 epoch 的 val 本来就没有落盘权重、进不了 best saved，改动后每个保留点的数值完全不变，
     只是 ep12-16 窗口从 5 点变 3 点。当前 run 仍在按 interval=1 跑。
   - 附注：离线 eval harness 比训练内 EvalHook 低约 2.2 个 3D 点（2D 完全一致），原因未追，见第 51.5 节。
     横向对照请统一只用一种来源。
   - 口径约束：若启用，seed1 与 seed2 会跑在不同代码/config 版本上；最稳做法是 seed1 跑完后、seed2 起跑前改，并在 `training_runs_full.md` 记录该差异。

## 方法探索

### 当前状态快照（2026-10-12 02:28 UTC；历史条目按当时状态保留）

- [x] **A2-R ep1–6 完成、S1 门核验与负结果归档（训练记录 §75）**。Overall 实际 ep1–6
  等权均值 33.9502（n=6），全轮已有 val 峰值 38.4943 @ ep5；best saved 36.3491 @ ep6。
  M1′/M2 PASS，**M1/M3/M4 FAIL**，总 NOT ALL PASS；**不进 ep16**。
  关键机制读数：S0-5 `frac_cells_residual_can_flip` **0.2930**（A 0.2530 / A2-S 0.3555）
  ⇒ §7 的「通道覆盖不是瓶颈」分支成立，分叉指向「分类读出是否应走独立分支」。
- [x] A2-R 只读补测（2026-10-12，未改权重、未启动训练）：z 契约对账（10-clip 与
  本臂 20-clip 两个契约，部署/训练读数**逐位相同**，且 20-clip 复现 M1 实读）、
  train split BN 对照（差距主要来自**数据 split**）；A2-S 的 `semA_bnmode_ep6.json`
  与 `semA_zcontract_ep6.json` 一并补齐。见 §74.11 / §75.13。
- [x] [[A2 生成器最后一个手写常数]]（commit `919e6b8`）：§10 结论句原把 A 的比率当常数
  渲染（4.3×，且是「末值 train ÷ 20-clip val」两个口径混写），会原样复制到 A2-R
  （其自身是 0.4923/0.0591）。现两个操作数各带口径从本包读出、不再打印乘数；
  新增回归测试。修好后 §75.14 的「未手抄任何读数」才成立。
- [x] A2-S ep1–6 完成、S1 门核验与负结果归档（训练记录 §74）。Overall 实际 ep1–6
  等权均值 33.6768（n=6），全轮已有 val 峰值 37.3276 @ ep5；best saved 34.9702 @ ep6。
  M1′/M3 PASS，M1/M4/M1-loc FAIL，总 NOT ALL PASS；不进 ep16。
- [x] A2 看门狗重复 z-util 清理、BN OOM 防护（既有 commit `c25ca30`）。
  旧 §73.6.22 的「待结束后清理」已完成；这不代表缺失的 BN 实测已补跑。
- [x] A2-S 扩展 BN/采样-deploy z 契约对账：**2026-10-12 已补**（GPU5/6/7 空闲期，
  只读探针）：`semA_bnmode_ep6.json`、`semA_bnmode_ep6_trainsplit.json`、
  `semA_zcontract_ep6.json` 均已落盘，读数见 §74.11。原缺口（§74.3 的「BN probe 未写出 /
  z 契约未测」）作废，**不改变** M1/M4/M1-loc 的 FAIL。
  看门狗已修（BN `--batch-size 2`，commit `c25ca30`）——该修复对本 run 的补跑即已验证生效。
- [x] A2-R 正式独立臂：代码/S0/450 iter smoke 完成，**2026-10-11 15:53:23 UTC 已开训**
  （GPU5/6/7、seed0、deterministic、24e、从零；run dir
  `fgfull_N4_no2d_igdr_2x4_24e_semA2R_seed0`），看门狗 `ARM=A2R` 15:54:17 UTC 起等 ep6。
  运行的是已修复版看门狗，故不挂 finisher。详见训练记录 §74.10。
- [x] 生成器两处「有产出、无消费者 / 静默吞行」缺陷修复并归档为 §73.6.23/§73.6.24：
  补充排序读数（`mean_gt_prob`/`mean_bg_prob`/argmax 距离）现已被聚合渲染；
  split「来源归属」兜底行不再被 z 契约分支吞掉。§73.6.22 的重复 z-util 清理已由
  `c25ca30` 执行，§74.9 记录修复后重渲染。
- [ ] ep6 报告器跨窗口基线展示与泛用 A 提示清理：现有输出仍旁列成熟基线，
  不可用于早期性能排名；本次仅记录，未改代码。
- [x] 「产物字段 → 消费者」系统扫描（训练记录 §73.6.26）：结论一 = 不存在
  「预登记要求却没人渲染」的字段（剩余未消费项属诊断，不是违约）。
  另记三处 landmine：M4 门槛基数 `chance_1_over_c` 未渲染（C=4 下数值无害）、
  三组**聚合口径不同**的同名读数（0.740 vs 0.770 等，不是矛盾）、
  `e_t_pooled_normalized` 与 `e_t_pooled_raw` **构造上逐位相同**。
- [ ] 上条 landmine 的去留需拍板：把 M4 门槛基数渲染进共用 M4 块 ⇒
  会**打破 A 的 §72 逐字节一致性**（需连同 A 归档一起改）；或维持现状只留记录。
  未改代码、未改门槛。
- [x] **高优先级**：`residual_class_map` 对 A2-R 的 48 通道残差**静默误读**（训练记录 §73.6.27）。
  会让 M4 门（`own_hit`/`donor_transfer`）与 A2-R 预登记 S0-5
  `frac_cells_residual_can_flip` 变成垃圾数。已于 A2-R ep6 电池**之前**修复并加回归闸；
  A2-S 经内部一致性验证不受影响（pooled own_hit == forced-max head hit = 0.690000）。
- [x] A2-R ep6 归档时须复核：`semA_m4floor_ep6.json` 的 `frac_cells_residual_can_flip`
  实测 **0.2930**（A 0.2530、A2-S 0.3555），已写入 §75.11 并按预登记 §5 S0-5
  「只记录不设线」处理；ep6 电池跑的是**修复后**的 `probe_semreadout_ep6.py`。
  ⇒ §7 的 M3 分叉读作「通道覆盖**不是**瓶颈」（0.2930 只比 0.2530 高 4.0 pt，
  且低于同代 A2-S），**待人工裁决**是否立项「分类读出独立分支」。
- [ ] **人工裁决点（需用户批准，勿自动投训）**：按 §7 分叉，M3 FAIL + S0-5 未明显
  上升指向「分类读出是否应走独立分支」的单变量验证；若采纳「两臂都不过」分支，则论文
  主线转向「为何确定性状态在缺测下不足」（需候选 C 的证据）。任何新臂**先补预登记**
  （唯一变量、门槛、采样契约、早期门与成熟门分离）再问用户；禁权重/KL/增益扫描、
  禁 ep24 抢救、禁 DC 家族继续叠层。
- [x] A2-R 草稿导语缺陷（训练记录 §73.6.28）：导语原写死「A2 臂另跑 M1'/M1-loc」，
  与判定器的 `n/a for A2-R` 自相矛盾。已按臂参数化并加双渲染测试；
  A 的 §72 重渲染仍逐字节相同。合成捆绑**仅作彩排**，不含任何读数。
- [x] M4 隐含前提核验（训练记录 §73.6.29）：无效样本会把残差**整张清零**，
  探针不挡全零；A2-S/A 两个捆绑实测均无全零 clip ⇒ 本代未受污染。
- [ ] **新 landmine（记录，未改探针）**：`probe_semA_bnmode.py` 的
  `eval_reproduces_frozen_gate.reproduces` 比的是 **A 的**冻结常数（0.1167 / 0.0833），
  故对 A2-S/A2-R **恒为 `False`**，与「本臂是否复现自身门读数」无关。
  正确自检是 `eval_running_stats` 与本臂 `semreadout_ep6_gate.json` 的 M1 逐位比较
  （A2-S 0.1458/0.1042 ✓、A2-R 0.0712/0.1292 ✓）。该字段不进任何归档渲染；
  已写入 §74.11 的读法警告。若将来要渲染它，须先改成按臂比对。
- [x] A2-R ep6 归档时用它当哨兵：`semA_m4floor_ep6.json` 的
  `residual_absmax_fg_mean` 最小值 = **0.8867** > 0、**无**零 clip、逐 clip `cells` 全 >0
  ⇒ M4 逐格读数全部有效（§75.11）。
- [ ] 方法成功验收：zt 修复、z 排他性/同容量确定性对照、四类成熟稳定提升、匹配消融、多 seed。
  C/B/D 尚未正式验证。本次仅核查/记录，不启动训练。


- **下一步决策更新（2026-10-12）**：A2 两臂（A2-S 监督侧排序、A2-R 通道覆盖）均已在
  ep6 机制门归档为 **NOT ALL PASS**，两臂都未进 ep16。M3 FAIL + S0-5 flip **0.2930**
  （A 0.2530、A2-S 0.3555）⇒ §7 的「**通道覆盖不是瓶颈**」分支成立。**待用户裁决**：
  (a) 立项「分类读出是否应走独立分支」的单变量验证（须先补预登记：唯一变量/门槛/
  采样契约/早期门与成熟门分离）；(b) 转候选 C（缺测分组证据），论文主线转向
  「为何确定性状态在缺测下不足」；(c) 其他。**本次未自动立项、未投训。**
  另：A2 的 z 契约对账与 split/BN 归因已补齐（§74.11 / §75.13），不改变任何门读数。
- **下一步决策更新（2026-10-10；检查完成，实施未开始）**：DC 停止续训。
  先核对真实 fullres raw 重建目标、梯度与历史契约，补新 run 正式预登记，
  再做标准 no2d_igdr 的有限 current/difference 配对验证；若任务收益不成立转 A，
  C 按缺测类别证据单独验证，B/D 后置。详见 §69.11 与
  `docs/rssm_next_step_audit_20261010.md`。95.5% pool16 是候选探针，不是现役目标；
  区域 baseline 是标准 RSSM，不是 MAI。没有新训练、没有已完成性能修复。
- **契约审计已完成（2026-10-10，只读，见 §70）**：真实目标 = 全分辨率**原始**
  未标准化当前帧 `bev_feats`，**无 detach**、权重 1.0、
  **实际只有最后 1 帧历史 + 当前帧**计 recon、等权平均（`rssm_bptt_steps`
  默认 `1`、config 未覆盖 ⇒ `burn_in=2`；前 2 帧仅 burn-in、无 loss）；
  低维家族的 pool+标准化+detach 契约**不同**（§53 的失败不能外推）。
  实测：recon 项自 ep4 起低于总损失 0.5%、ep24 仅 **0.18%**；KL 被 free-bits
  完全吸收（clamped 0.9985、raw KL 0.597 < 1.0）；`μ_q−μ_p`² 单调降到 0.0072。
  ⇒ **目标项太小是各代改状态侧/读出侧都不动 AP 的直接解释**。
  另实测：窗口内各帧**共享同一几何增强**（`TJ4D_dataset.py:333-335,369-370`），
  差分目标「被增强差异主导」的风险不成立。
- **阶段 0.2 GPU 只读复测已完成（2026-10-10，GPU5，见 §70.6）——目标改革的
  承重前提被否**：① 真实目标在跨场景 16 样本窗口上 **56.5% 是样本特异的**，
  不是「95% 共享模板」；§69 的 95.5%/21×/6.28× 主要由**同场景连续帧窗口**
  + pool16 候选构造产生（同一 checkpoint 换窗口：pool16 5.7%→32.9%、
  fullres 13.5%→59.8%）。② 差分目标特异占比 93.7% **等于纯噪声地板**
  (S−1)/S = 93.75% ⇒ 该统计量对「是否含信息」没有区分力。③ 按现役契约
  （权重 1.0、不缩放）换差分会把目标能量压到 **4.0%**，recon 项从 0.18%
  进一步降到 ~0.007%，**削弱**对 z 的约束，与立项目标相反；补尺度即第二变量。
  另：草案原命令（`--stride 17 --require-classes 2`）只选到 **2 帧**——全 val
  满足「四类同时 ≥2 GT」的帧仅 35 个且集中在同一场景块，S=2 的地板恰是 50%，
  那一次产物已作废。
- **候选 A 已按排序落地（2026-10-10，见 §71）：代码 + 单测 + S0 只读探针 +
  450 iter 机制冒烟全部完成，未做任何 AP 宣称。**
  - 实现（commit `76d2721`）：`SemanticReadoutRSSMFusion`，只动两处 ——
    z 专属 CentreNet 热力监督 `loss_rssm_sem` + 零初始化 `z_cls_logits`
    加到 `conv_cls` 的分类残差；重建目标/KL/对齐/头几何/日程全部沿用标准
    `no2d_igdr`。控制臂 `sem_supervision_weight=0`（C2）、`cls_residual=False`（C1）、
    `sem_head_input='h'|'feat'`（C3）。
  - S0 门（只读，GPU5）：**S0-b 身份 PASS**（分类/回归输出与基线逐位相同，
    max|Δ|=0.0；额外缺失键恰为新模块 6 个）；**S0-d 压力 PASS**
    （语义项占 z 上梯度 90.2%）；**S0-e 存活/分离 PASS**（`∂loss_cls/∂W_cls=0.871`，
    `∂loss_rssm_sem/∂W_cls=0`）。
  - 探针修正一处（commit `44cb425`）：首轮 S0-e 的 FAIL 是**判据选样错误**——
    train split 第一个 batch 是历史帧无效的边界样本，残差被设计性置零
    （只影响 0.75% 样本）；且零初始化下"前向残差恒为 0"本身就是 S0-b 的身份契约，
    不能兼作存活证据。现已改为「选有效历史 batch + 用 `∂loss_cls/∂W_cls`
    与扰动后残差两条独立读数」。
  - 450 iter 冒烟（3 卡 GPU5/6/7、seed0、`iter_450.pth` 落盘、无 NaN/OOM、
    17099 MiB、1.53–1.60 s/iter）：`loss_rssm_sem` 21.40 → 2.27（−89.4%，
    末 5 点均值 2.398，占总损失 39.3%）；`stat_cls_res_rms` 0 → 0.184（读出被激活）；
    **但 `hit@1` 仅 0.120、`thr50` 仅 0.0033**（450 iter ≈1% 日程内的早期读数，
    不是机制判据，S1 判据在 ep6）。另记录一处口径差异：`KLScaleSchedulerHook`
    只实现 `before_train_epoch`，`IterBasedRunner` 从不调用它 ⇒ 冒烟里
    `kl_scale` 恒为 1.0，recon/KL 绝对量不可与 epoch 模式逐点比较。
  - 回归：`test_rssm_fusion` 85 passed、`test_semantic_readout_rssm` 11 passed、
    `test_grad_paths` / `test_config_build` / `test_fgfull_config` /
    `test_mci_config_build` / `test_equivalence` 全绿。
  - 正式 S1 训练已启动（2026-10-10，3 卡 GPU5/6/7、seed0、deterministic、
    24e 完整日程，`--work-dir ..._semA_seed0`），ep6 看门狗 `semA_watch`
    在 `Epoch(val) [6]` + 新鲜 `epoch_6.pth` 同时出现后中断训练并跑
    M1–M4 门 + 三窗反事实 + source 探针；**判定仍是人工步骤**，脚本从不自动
    启动后续训练。
  - ep6 证据链三次预检修复（ep6 未产出时完成，全部有单测；这些若留到
    ep6 后才暴露，会让 6 小时训练的整包证据作废）：
    1. `52fc69d`：M4 donor residual 未随 `pts_feats` 一起搬到 CUDA，
       `_expand_cls_logit_residual` 的 `index_select` 会 device mismatch 抛错。
    2. `6c26bc3`：看门狗 P1/P2 探针是 MCI/DC 专用，semA 上直接
       `AssertionError: target model is not a gated variant`。新增
       `probe_z_utilization_standard.py`（`dz.ZIntervention` 后验重写 +
       **转发部署态 class residual**），`probe_z_util_region_split.py` 同样
       转发（旧配置下为 `None`，§69 读数不受影响）。
    3. `bc0bd9d`：`object_min_radius` 从 config 读出后被丢弃，四个
       `make_target` 调用全部落到硬编码默认值 2（当前 config 恰好也是 2，
       所以只是"碰巧正确"）；现改为必填参数并加拦截测试。
  - 判据口径两处写定（均在 ep6 读数未知时，避免事后调门，门槛数值一律不动）：
    `fdbded2` M2 的「`h_t`（或 `feat`）」按 **z 排他性**严格读法固定为
    **两臂各自** ≤ −30%（与探针既有实现一致）；`f287057` 记录 M1 的
    `hit@1 ≥ 0.60` 是**跨网格外推**（先例 route-C 在 16×16 池化网格、
    `min_radius=1`；A 在 full-res 216×248、`min_radius=2`，随机基线
    1.6e-2 vs 7.5e-5），故新增 `hit1_chance` / `hit1_excess_over_chance`
    作为**不进门的补充读数**；若 M1 因口径擦线，按「证据不足」处理并补
    full-res 同口径对照，不直接当机制否定。
    A 侧训练期同口径（train split）ep1/ep2：`hit@1` 0.159/0.241、
    `thr50` 0.073/0.232。
  - ep6 证据链第四处预检修复（`0b4e0cc`，ep6 未产出时完成）：
    **看门狗没有失败传播**——它是一串不打 `&&` 的探针调用，任一探针死掉
    （OOM / CUDA error / 融合族不匹配）都不写 JSON，而日志仍走到
    `probes done`，缺产物会被误读成"门没过"。现由判定器
    `semA_ep6_verdict.py` 的 `check_evidence()` 逐一校验五份产物的存在性与
    schema，证据包不完整时**即使四门全过也报 `NOT EVALUABLE`**，绝不输出
    `ALL PASS`；同时 gate JSON 新增 config/checkpoint 的
    sha256/size/mtime 指纹，防止读到被覆盖的旧权重。新增
    `test_semA_ep6_verdict.py`（16 条，含生产者—消费者契约测试：判定器要求的
    键必须是探针实际写出的键，任一侧改名会让测试红而非静默 INCOMPLETE）。
    门槛数值、采样窗口、模型与日程均未动；看门狗脚本本身未改（正处等待
    循环，改它须重启会话）。详见 §71.9.6。
  - ep6 证据链**第六处**（最严重一处）预检修复（`118fa2a`，ep6 未产出时
    完成）：M1–M4 的 GT 读错位置。`rollout()` 从 `img_metas[-1][0]` 取
    `gt_bboxes_3d`/`gt_labels_3d`，但 `CustomCollect3D2.meta_keys` **不含 GT**；
    GT 在 sample 顶层且是逐帧列表（`[-1]` 为当前帧，与 `R4Det` 交给
    `set_object_gt` 的当前帧 GT 同源）。后果：每次运行都在**第一个 clip**
    抛 `KeyError` 且**不写 gate JSON**，四门 + 三窗 + source 整包证据全废。
    已抽出唯一入口 `sample_current_gt()` 并加 6 条回归测试（回退修复即红）。
    用 450-iter 冒烟权重实测：旧版 `KeyError` 无 JSON，新版正常写出四门。
    等价性：10/10 clip 标签与 `get_ann_info` 一致，热图与训练同款光栅化器
    逐位相同。同批静态核验其余三件（z-util 三窗索引 4/4、source 探针
    正确选 `transition`、看门狗 offset 无自映射）**均无需改动**。详见 §71.9.7。
  - ep6 证据链**第七处**预检修复（`7d13b19`）：gate JSON 原先只在 M1–M4
    全部走完后一次性落盘，M3/M4 中途崩溃会把算完的 M1/M2 一起丢。现改为
    `write_report()` 原子写（temp + rename），并在 M1/M2 后、M3 后各 flush；
    判定侧同步把「门键缺失」渲染为 **`NOT MEASURED`** 而非 `FAIL`（没测到
    ≠ 测到没过）。新增 4 条测试，`test_probe_semreadout_ep6` 增至 36 条；
    450-iter 冒烟实跑确认四门正常写出且无 `.tmp` 残留。详见 §71.9.8。
  - ep6 证据链**第八处**修复（`e7fbb77`）：§71.9.3(2) 的 M1 例外原先只是
    一句 `NOTE`，聚合行仍报 `NOT ALL PASS`，**预登记的例外从未生效**。
    现写成可执行分支 `m1_grid_limited()`（阈值全部由预登记值导出：
    `excess >= 0.60 − 0.016`），触发需 thr50 过线 + hit@1 掉线 + 超额健康
    + **M1 为唯一未过的门**；命中时输出 `INSUFFICIENT EVIDENCE`，不记为
    机制否定。新增 8 条互斥性测试，三份 probe 测试合跑 64 passed。
    详见 §71.9.9。
  - ep6 证据链预防性加固（`30333b0`）：看门狗加环境预检
    （`command -v python` + `python -c 'import torch, mmcv, mmdet3d'`），
    任一失败即 `ABORT` 退出而不是让每个探针静默失败、日志仍写
    `probes done`。本机现行 `.envrc` 实测可正常激活（该风险当前未触发）。
    正/负向路径均已实跑验证；改脚本后已重启看门狗会话。详见 §71.9.10。
  - ep6 证据链**第九处**修复（`0fae53f`）：`_sem_source()` 对未识别的
    `sem_head_input` **静默返回 `feat`**，而 M2 靠运行时切换该属性构造
    h/feat 控制臂——拼写错误会让 h 臂静默等于 feat 臂，把同一张量的两次
    测量当作两个控制臂。现改为非法值 `raise ValueError`；新增 T13
    （三合法臂 + 7 个非法值），回退即红，`test_semantic_readout_rssm`
    12 passed。详见 §71.9.11。
  - ep6 证据链预防性加固（`f691561`）：看门狗**不能在这里执行**，故参数名
    漂移/窗口缺失只会在 ep6 当场炸成「探针 argparse 退出、不写 JSON」。
    新增 `test_watch_semA_ep6_cli.py`（10 条纯静态测试）：解析看门狗真实
    调用，断言每个 flag 都被目标探针接受、必需 flag 齐全、三窗口与 gate
    clip 计划未变、`--output` 互不重复、判定器仍在最后且脚本不含任何续训
    路径。改坏 flag 即在测试里被发现。五份测试合跑 86 passed。详见 §71.9.12。
  - §72 训练记录**机械生成器**上线（`7abd4ee`）：AGENTS 规则 5/6 要求三口径
    分别写清 + 逐 epoch 曲线 + 逐类别拆解，手抄正是本项目出过错的地方。
    新增 `semA_ep6_record.py`（只读、CPU）：`import` 判定器并复用其
    `check_evidence`/`m1_grid_limited`/`val_by_epoch`/`EVIDENCE`（门槛逻辑
    **单源**，一致性有测试断言），从证据包 + `*.log.json` 渲染 §72 草稿到
    work_dir（**不改正典**），无法自动取得运行事实写成可见 `FILL` 并在
    stderr 计数告警。同时修**证据侧**一处缺陷：M1/M2 逐 clip 条目原先不带
    clip 索引，三臂只能按位置对齐，某 clip 因无 GT 被丢弃时会**错位比较
    不同 clip**；现附上调用索引（纯增量，丢弃时不错位，已单独验证）。
    另加三窗**同一索引警告**：真实看门狗按 `start-index=3/100/1800` 取窗，
    三窗由构造保证不同，故该警告是防探针回归/手工改窗的护栏；已在手工
    同窗**替身包**上验证触发（替身只验告警路径，不作机制读数——冒烟是
    `IterBasedRunner`，没有真实三窗 JSON）。看门狗在判定器之后自动渲染
    草稿，CLI 契约测试扩到 12 条。回归 **117 passed**（含 §72.8 修复新增的
    3 条，见下）；看门狗会话已按偏移
    约定重启（新起点 18:13:58 UTC），训练同 PID 未受影响。详见 §71.9.13。
  - §72.8 只读 `rms` 的缺陷修复（`eeaa32d`）：初版渲染丢掉
    `sample_specific_energy_frac`（审计线依赖量）与 `mu_q_removal_of_h/e`
    比值（跨代对照 2.338/0.8418/0.605），且「只有 `rms`」的**残缺** source
    文件会渲染成半空表而无提示——**等于可以把替身当读数粘进记录**。现渲染
    完整四字段块 + h/e 比值，缺失时明确标注「不是完整读数、不得引用」；
    用真实 MAI ep12 产物校验（`e_pooled` 64.28%、h/e 0.842 复现记录），
    删护栏即测试变红。`test_semA_ep6_record` 增至 29 条、六份合跑 117 passed。
    详见 §71.9.13。
  - 残缺证据包下的**不实渲染**修复（`6594de5`）：H1 标题原先把「什么都没
    测到」也写成「未全门通过」（该措辞意为「测了、有门没过」，属机制结论），
    现为 `NOT EVALUABLE（证据不完整）`；M3 的 `None/None` 与 M4 的 `bar`
    亦改为 `NOT MEASURED`/`n/a`。同时发现上一版断言**可被正文同名字样满足**
    （删掉修复不会变红），新断言只看首行、回退即红。`test_semA_ep6_record`
    36 条、六份合跑 **124 passed**。详见 §71.9.13。
  - 门控采样点在**真实** val 集上的只读预检（§71.9.14）：gate 采样点解析为
    `[3,203,...,1803]` **满 10 个**（M3 的 ≥8 与 10 帧符号检验由构造满足）；
    10 个采样点当前帧 GT 为 3/4/3/11/4/2/6/6/1/2、**无空帧**且主类含
    Car/Truck/Cyclist **三类**，任意 clip 都能找到异类 donor（M4 不会退化到
    兜底分支）；z-util 三窗解析为 `[3..6]/[100..103]/[1800..1803]`
    **互不相同**。**注意**：clip `1603` 只有 1 个 GT 目标，M3/M4 每 clip
    读数只能取 0 或 1，§72 不得由该 clip 单独支撑结论。纯只读、不占 GPU。
- **候选 A 的 S1 ep6 机制门已完成：NOT ALL PASS（M1/M3/M4），按 §7 归档负结果。**
  **已完成**（2026-10-10 停训 + 探针，记录见 `docs/training_runs_full.md` §72）：
  训练在 ep6 由看门狗按预登记中断（SIGINT，预期动作），GPU 5/6/7 已释放。
  四门：**M2 PASS（但证据强度有限）**（h/feat 两臂 `thr50` 均为 0.0000 vs
  z 0.1167 ⇒ 该语义头**依赖它训练时的输入**；但这**不能**证明「语义只存在于
  z」——把 h/feat 喂给一个只被 z 监督过的 head 属于分布外输入，即使其中含有
  同样的对象信息也读不出来。真正的 z 排他性检验在预登记 **C3 臂**
  （`sem_head_input='feat'` 从零训练），**本次未执行**，故「概率状态」这一
  主张在本次测量中**既未被否证、也未被证实**）、
  M1 FAIL（`thr50` 0.1167 / `hit@1` 0.0833）、M3 FAIL（7/10，需 ≥8）、
  M4 FAIL（`own_hit` 0.7379 过线但 `donor_transfer` 0.3818 < 0.45）。
  **M1 的跨网格例外条款（§71.9.3(2)）已判定为「不适用」**，且该判定
  **不依赖口径选择**：门槛先例 §59 route-C 报的是 **train 侧** epoch 均值
  （§60.5/§61.5，38 条 train 记录），本门按 §5 预登记在 **val** 上测量，
  故两种读法并列记录——val 口径 `thr50` 0.1167 / `hit@1` 0.0833；
  A 自身 train 口径（`stat_sem_*` ep6 epoch 均值）`thr50` **0.4923（过线）** /
  `hit@1` **0.3586（未过 0.60）**。要点：该头**确实找得到目标中心**（train
  口径 thr50 过线），但单格 argmax 落不到 GT 格上；`excess <= hit@1`
  给出 train 口径 excess 上界 0.3586 < 0.5840，故两边都不满足例外。
  **不做** ep16 续训、不做权重/KL 扫描、不做 ep24 抢救（§7 预登记）。
  归档范围限于**本次候选 A 的读出/监督设计**，不是「z 无信息」，也不是
  「所有语义监督/残差读出方向」。
  - 机制侧正面读数（供转 A/B 时复用）：z-util 三窗 `shuffle_z` head 为
    近距 **2.6509%** / 中距 **7.4311%** / 远距 **4.2138%**（三窗均越过 2%
    论文线）；posterior source `mu_q` 样本特异 12.45%、`e_pooled` 52.48%、
    h/e 依赖比 1.517（较 MAI 0.8418 / MCI 0.605 更依赖 h）。
  - 早期 AP **仅作上下文、非 S2 宣称**：ep1-6（n=6）均值 34.0459，
    全轮峰值 38.2203 @ ep3，best saved 35.7608 @ ep6；对照基线同跨度
    29.0420，但基线 ep12-16 为 39.2809，四类均高于本 run 的 ep1-6。
  - **归档后审计（2026-10-11，只读，§71.9.17）修正了三处推论口径**，结论方向不变：
    (1) M1 门槛的 train/val 口径混用已按两种读法并列核过（见上）；
    (2) **z 契约错配嫌疑已排除**——训练喂 `mu_q + sigma*eps`、eval 喂 `mu_q`，
    同 checkpoint/同 clip/同评分下切换契约，z 实际移动 **35.26%** 而
    `thr50`/`hit@1` **一点不变（Δ=0.0000）**，故 train/val 的 4.3× 差距不是
    采样契约造成的，**来源仍未定位**；
    (3) M2 的表述已从「未被否证」收紧为「**既未被否证、也未被证实**」。
  - **M4「读出不可用」的机制定位（2026-10-11，只读，§71.9.18）**——把上一轮唯一
    没被质疑过的读数也查了一遍，发现**两处自身口径缺陷 + 一个机械解释**：
    - **bar 比错对象**：预登记的 chance 取 1/C = 0.25，但这份 val 窗口的前景格
    GT 类别计数为 `[5,8,20,6]`（Car 占 51.28%）。**永远投 Car** 的残差就得
    0.5128，已过 0.45 线；per-clip 多数类地板均值 0.6970、最大 1.0000。
    复算 `own_hit` 0.7379 与门控 JSON 逐位一致，**超出 pooled 地板 +0.2251**
    ⇒ 「残差含类别信息」**未被推翻**，但**有效证据量是那 0.225 的余量，
    不是 0.7379**；且 M4 的 FAIL 本就由 `donor_transfer` 0.3818 判定。
    - **`head_swap_to_donor` 0.1167 不能读成「头忽略 z」**：`residual` 全图 RMS
    1.0015 vs `conv_cls` 14.3501（比值 0.0698），前景格 `conv_cls` 的 top1−top2
    间隔均值 2.7782 / 中位 2.1186，而 `residual` absmax 均值只有 1.5961 ⇒
    **仅 25.30% 的前景格上残差在算术上有能力翻转 head argmax**。这是尺度事实。
    - **尺度过增益曲线**（同特征/权重，只乘 residual；按格加权）：关 0.6410 /
      现状 k=1 **0.6923** / k=2,4,8 **均 0.6923** / 强制 1e6 0.5897。
      即：**关掉会掉 5.1 pt**（残差确实参与决策，不是纯装饰），但**放大 2–8 倍
      完全不动** ⇒ 当前**不是靠调增益可救的欠尺度问题**；着力的地方是
      residual 与 `conv_cls` 的**相对尺度/耦合方式**。
    - 强制臂未复现 residual 自身 argmax（0.7379），已逐格定位为**通道投影口径**：
      head 只把每类一个 logit 写进 48 通道中的 12 个（`dst=[0,4,8,...,47]`），
      探针却是「max-over-anchor 再 argmax」，`k→∞` 两者不等价 ⇒ 该强制臂
      **只作内部一致性检查，不作判据**。
    - 产物：`semA_m4floor_ep6.json`；探针 `me_rssm/sanity/probe_semA_m4floor.py`
      （只读，GPU5 一次）；`test_probe_semA_m4floor.py` 8 条。**§72 的 M4 判定
      （FAIL）不变**，变的是它的解释强度与下一步的切入点。
- **下一步（待用户抉择，无训练在跑）**：候选 A 已按预登记归档；机制读数为
  下一步提供两个只有证据支持的分叉（**均未预登记、未获批准，不得自行启动**）：
  (a) 若按「z 有语义但检测不吃」这一读数继续，则转 **A 的分类 residual 读出**
  单变量验证（§71.9.18 已把切入点收窄为「residual×`conv_cls` 的相对尺度/耦合」，
  并给出预登记必须包含的三个对照：**多数类先验地板**、**尺度过增益曲线**、
  **anchor 广播/通道投影口径**）；(b) 若认为 M2 的正面结论值得单独主张，则先补
  **z-exclusive residual** 口径的预登记再做。弃选项：目标內容+尺度的配对设计
  （原 §3A 选项 B，§70.6 已否掉其承重前提）；S2 数值门（ep12-16
  Overall ≥ 40.00 等）随本次归档**不再启用**。**注意**：审计同时发现 M2 的
  弱点（分布外输入），故 (b) 若要做，其预登记必须写明 C3 臂（`sem_head_input=
  'feat'` 从零训练）才算真正的排他性检验；仅凭 M2 不足以支撑该主张。



1. 探索原 DreamerV3 的 BlockGRU scaling 方法和超参数使用方法。
2. 解决 `z_t` 失效：冻结 checkpoint 因果诊断已于 2026-09-25 完成，详见 `docs/training_runs_full.md` 第 49 节；继续调 Gaussian、KL/free_nats、fixed noise 或 learnable std 都不会解决。
   - **完整 18e 主模型对照（2026-10-02 排队 → 2026-10-03 完成）**：MAI seed2 ep16 完成停训和四个 probe 后，watcher 自动从零启动 `MotionAlignedRSSMFusion`（标准主线，非 MAI）seed0、完整 18 epoch。配置 `configs/r4det/TJ4D-R4Det_fgfull_N4_2x4_18e_pretrained_v2_head_no2d_igdr.py`：`max_epochs=18`、完整 18e cosine（峰值 LR 仍 `1.5e-4`）、KL warmup `0->1.0` 缩到 ep0-8；架构、优化器、梯度累积、val_bs=4、checkpoint cadence 与 24e 基线一致。接力保护见 `me_rssm/sanity/watch_after_mai_seed2_run_fgfull18_seed0.sh`。
     - **结果**：ep1-18 连续训练，2026-10-02 19:30:44 UTC 发射、2026-10-03 13:41:54 UTC 完成（`attempt 1`，无崩溃恢复），无 OOM/NaN，显存 16971 MiB。三口径（完整 18e 无 24e 标准 ep12-16，主窗口取最终五轮 ep14-18，n=5）：区间均值 **39.7896 ± 0.4574**；全轮峰值 **40.3094 @ ep16**；best saved **40.3094 @ ep16**（峰值落盘）；last val **39.6617 @ ep18**。补充 ep12-16（n=5）`39.5086 ± 0.7197`、ep1-18 均值 `36.7615`。
     - **与历史 24e seed0 对照（不同日程，非因果）**：ep12-16 `+0.2277`、ep14-18 `+0.5019`、ep18 单点 `+0.6295`，但 18e 峰值 `40.3094` 低于 24e 峰值 `40.3853`；18e 的 LR 在 ep18 到达该日程最低点、KL warmup 为 ep0-8，24e 同点 LR 仍约 `2.9e-5`、warmup 到 ep10，**差值不能当作日程优劣**。可读出的只有「完整 18 轮足以达到 24e 平台量级」。
     - 该 run **未跑** z-util/source 机制探针，本节不给出任何 z 使用程度结论。完整逐 epoch、逐类别、训练诊断与复现命令见 `docs/training_runs_full.md` 第 66 节；跨 run 汇总见第 67 节。
   - 因果诊断结论：`zero_z` 在 no2d_igdr seed0 ep14 上使 BEV 特征平均变化 82.53%、head 输出平均变化 38.02%，所以 z 不是完全无因果作用；但 `shuffle_z` 只使 head 平均变化 4.67%，prior `mu_p` 只使 head 平均变化 5.87%，clean 主线也得到同构结果。这说明 z 主要是公共偏置/底色，缺少样本级判别信息。
   - **进度（2026-09-26）**：低维 bottleneck + 排他未来任务已实现，但首轮正式训练因 detector 解包接线错误（把 `z_t` 当 loss、丢弃真正的 KL+future 目标）而无效，ep1-4 全部作废，详见 `docs/training_runs_full.md` 第 50 节。接线、未来任务方向、KL 二次归一化、mask 广播、`stat_*loss` 命名共 5 处已修复（commit `f5d900f`），29 个单测 + 450 iter 冒烟通过，随后重启 24e 正式训练（结果见下条）。
   - **结果（2026-09-26，负结果）**：低维 bottleneck（16×16×32）+ 排他未来任务训练到 ep12 后按预设止损终止。ep8-12（n=5，实际跨度，窗口不完整）Overall 3D moderate `33.3887`，低于 no2d_igdr 三 seed 同窗口 `39.8979 ± 0.9554` 达 **6.51 点**；全轮峰值 `35.6352 @ ep9`（该 epoch 无对应权重），best saved `34.4123 @ ep6`。KL 被 free-bits 完全截断（`clamped_ratio → 0.980`），`mu_diff²` 腰斩（0.0256→0.0024），future MSE ep9 后平台化。完整曲线、逐类别拆解与归因见 `docs/training_runs_full.md` 第 50.6 节。**该配置判为负结果，不再继续训练。**
   - **归因（2026-09-26）**：三条已核实——① 本变体删掉了 decoder/重建项（`rssm_fusion.py:1008` 起的类没有 `self.decoder`），而重建是标准 RSSM 里把 `z_t` 锚定到当前观测的唯一梯度源；② 先验/后验被改成只看 `z_prev`（`prior_mu(z_prev)`，标准应为 `h_t`），predict-correct 结构被破坏；③ 输出经 FiLM 门控旁路（`gate_init_bias=-1.0`），z 更易被绕开。对照证据：标准 `MotionAlignedRSSMFusion` 的 KL 同样长期被钳（`clamped_ratio≈0.998`、`loss_rssm_kl≈1.00`），但它靠 `loss_rssm_recon≈0.0037` 锚定 z，指标可用。**结论：KL/free-bits 不是首要病因，优先级下调。**
   - **反事实诊断（2026-09-26，工具已修正）**：旧工具 hook `output_proj`/`decoder`，只覆盖 z 的两条下游分支（漏掉 FiLM/gate 与写回 `z_state`），且本变体没有 `decoder` 属性、会直接 `AttributeError`。已改为 hook `posterior_mu` 输出（确定性重放下 `z_t = mu_q`），一处改写即覆盖全部下游；新增 10 条单测（`tools/test_diagnose_z_utilization.py`），断言 `out[4]` 等于干预后的 z、lowdim/标准 RSSM 都能跑、normal 重复逐位一致、shuffle 来源不得是自己。commit `dd2fb75`。
   - **ep12 重测结果（修正探针，旧结构）**：三个样本窗各 4 条均为「zero 高、shuffle/prior 极低」——`[3,4,5,6]` head 比值 zero `0.0214` / shuffle `0.000097` / prior `0.00065`；`[100..103]` `0.0196` / `0.000052` / `0.00064`；`[1800..1803]` `0.0249` / `0.000088` / `0.00066`。即 z 的因果作用几乎完全是**与样本无关的公共偏置**。修正探针在标准 RSSM（no2d_igdr seed0 ep14）上给出 `0.3802` / `0.0467` / `0.0587`，与第 49.2 节旧数字逐位一致，确认是严格超集而非换口径。
   - **结构修复（2026-09-26，commit `dd2fb75`）**：按标准 RSSM 一次性落地两项，不做残缺中间实验——① prior/posterior 条件从 `z_prev` 改回 `h_t`（`p(z_t|h_t)`、`q(z_t|h_t,e_t)`）；② 加回观测似然 `decoder(cat[h_t,z_t]) -> pooled 且逐样本标准化的当前帧 BEV`，`recon_loss_weight=0.1`，与 KL、future 一起折进 index 2（index 1 保持 `None`，不重复计账）。新增 `stat_recon_mse`。RSSM 全量 31 条单测通过；100 iter 与 450 iter 冒烟通过（recon 1.1714→0.6420，future 1.0388→0.9239 同步下降）。
   - **结果（2026-09-27，负结果）**：`lowdim_z_recon_ht_3x2x2_24e_seed0` 跑到 ep13 验证完成后按**预先登记的 AP 止损**终止（ep8-12 五轮均值比基线低 2 点即停）。ep8-12（n=5，窗口完整）Overall 3D moderate `32.9976`，低于 no2d_igdr 三 seed 同窗口 `39.8979 ± 0.9554` 达 **6.90 点**（上一版为 −6.51，差距反而扩大）；全轮峰值 `34.7784 @ ep13`（无对应权重），best saved `33.6799 @ ep12`。逐类别 ep8-12 3D strict：Car `41.7478`（−9.44）、Truck `18.7115`（−11.17）、Cyclist `23.8772`（+0.81）、Pedestrian `0.0985`（−0.12）。完整曲线、逐类别拆解与归因见 `docs/training_runs_full.md` 第 52 节。
   - **机制门控逐条（未通过）**：`shuffle_z` head 仅 **0.000340**（要求 ≥0.10，差约 294 倍；上一版 0.000097）；`prior_only` head 0.001106（上一版 0.00065，仅 1.7 倍）；recon 0.157→0.163 基本持平；raw KL 有界（0.032-0.371，通过）；`mu_diff²` 0.02816→0.00324 仍单调塌缩。三窗结果一致（`[3,4,5,6]` / `[100..103]` / `[1800..1803]`）。
   - **失败机制（已定量定位）**：重建项**确实被使用**——零化 z 让重建 MSE 从 0.19-0.20 涨到 2.96-3.00（+1385%~+1470%）；但**换用另一个样本的 z，重建损失只变化 −0.09%~+9.07%**。用缓存 BEV 特征分解重建目标：当前实现（pool16 + 逐样本全局标准化）**92.1% 的方差是所有样本共享的模板，仅 7.9% 样本特异**；`pool32` 为 11.2%、全分辨率为 18.6%、逐 cell 标准化只有 3.2%（更差）、按留出集去均值只到 10.1%。另有独立证据：当前帧目标相对上一帧的 MSE 仅 0.0407，而跨样本 MSE 为 0.1691（比值 0.241），即 h_t 光凭上一帧就能解释掉相当一部分目标。**结论：存在一个「样本无关的 z」就能同时满足重建与 future 两项，因此 `shuffle_z` 无感。**
   - 下一步（按代价从低到高）：① 换掉重建目标——改用 `pool32`/全分辨率，或**直接以「与上一帧的差异」为重建目标**（差分重建，证据最直接：被 h_t 解释掉的部分不该由 z 承担）；**不要再试逐 cell 标准化（3.2%）或全局去均值（10.1%），已证无效**。② 给 z 一个当前观测独占且 h_t 无法预测的任务（box-centric/occupancy 监督，或让 future 任务也走差分层）。③ 若前两步仍不改善 shuffle_z，考虑放弃「低维 bottleneck + FiLM/gate 旁路」这一组合，回到输出直接吃 `(h_t,z_t)` 的标准形态。④ **KL balancing/free-bits 仍然后置**：本轮 raw KL 一直有界（≤0.37），已再次证明它不是瓶颈。
   - **方案一（2026-09-27，已完成，门控未通过）**：`correction_t = mu_q − stopgrad(mu_p)`，decoder 只吃 `correction_t`，重建目标改为标准化帧间 BEV 残差，future loss 关闭，correction 直接送检测残差。配置 `..._lowdim_z_delta.py`（commit `12cfc65`）；RSSM 单测 33 条通过。450 iter 冒烟（`/tmp/lowdim_delta450.log`）：recon `1.2204→1.0851`（**−11.1%**，要求 ≥15%）、correction² `0.0155→0.0039`（要求 ≥0.005，且后 150 iter 仍在下滑）、clamped `0.896`（通过）。**未过门控。**
   - **方案二（2026-09-27，已完成，门控未通过）**：加 batch-roll 排名损失（`discr_loss_weight=0.1`、`margin=0.2`，负样本 `roll(1)`，batch<2 跳过），配置 `..._lowdim_z_delta_discr.py`（commit `3d8aefa`），单测 35 条通过。450 iter：`neg−pos` gap 仅 **0.0111**（要求 ≥0.1，差约 9 倍）且最后 50 iter 停滞；correction² 仍滑到 0.0039；recon −10.8%（与方案一几乎相同）。**关键证据：recon `1.09` 已高于「恒输出零」的 `1.00`，`pos`/`neg` 只差 1%，说明 decoder 输出几乎不依赖收到哪个 correction。未过门控。**
   - **方案三（2026-09-27，已完成，门控未通过）**：删 global pool + FiLM/gate，改 `output = feat + h_proj(h_t) + z_proj(correction_t)`（均 `Conv2d(32,256,1)`，16×16 投影后双线性上采样）。`LowDimFutureConsistentLatentFusion` 新增 `readout_mode`（`'film'` 默认、`'spatial'` 新路径，`spatial`+`latent_pool='global'` 报错），commit `9f6c84f`；单测 **41 条通过**。配置 `..._lowdim_z_delta_spatial.py`。450 iter：recon `1.2235→1.0856`、correction² `0.0159→0.0038`、clamped `0.909`，**与方案一在噪声水平内相同**；三窗口 `shuffle_z` head 仅 `2.3–2.8e-5`（要求 ≥1%）。**读出结构不是瓶颈。**
   - **根因定位（2026-09-27，新增两个只读探针，已定量）**：`tools/probe_correction_scale.py` + `tools/probe_posterior_source.py`。① **上游信息在 `F.normalize(e_pooled)` 处被抹掉**：跨场景抽样下 `feat` pool16 的样本特异能量占比 **23.3%** → `encoder` 输出 `e_t` **7.2%** → `normalize(pool(e_t))` 后只剩 **1.28%**（分母被 1.90 RMS 的公共分量主导）→ `h_t` **0.27%** → `mu_q` **0.81%**；被重建和检测共用的 correction（`mu_q−mu_p`）只剩 **0.78% 样本特异能量（即 99.2% 是公共偏置）**，三窗口一致。标准 `MotionAlignedRSSMFusion` 的 posterior 直接吃未归一化的全分辨率 `e_t`，不做这一步。② **下游增益不足**：训练后 `z_proj` 权重仍停在 `std=0.01` 初始化尺度（weight RMS `0.0101`），检测损失没有把它推大；把 `z_proj` 增益 ×10 后 `shuffle_z` head ratio 由 `2.9e-5` 升到 `0.0126`（越过 1% 门槛），×100 到 `0.139`，三窗口同样形状，证明**读出通路是通的，只是信号幅度和上游信息量都不够**。
   - **下一步（按代价从低到高，尚未执行）**：① **去掉 `e_pooled` 的 L2 normalize**（改回标准 RSSM 做法，或换 LayerNorm / 仅对 `z_prev` 保留），这是唯一被定量证明会直接抬高样本特异占比的一步，改动一行、风险最低。② 若 ① 后 `mu_q` 特异占比仍上不去，再考虑放开 `h_t` 的公共偏置（`h_t` 本身只有 0.27% 特异能量，GRU 实际收敛成了常量状态）。③ 只有机制门槛（`shuffle_z` head ≥1%）通过后，才做方案四的 KL balancing 与按 cell 聚合的 free-bits；**当前 raw KL 一直有界（450 iter 时 0.0376），再次确认它不是瓶颈**。
   - **路线 A：2×2 机制矩阵（2026-09-27，已完成，门控未通过）**：新增 `posterior_obs_mode ∈ {l2, scaled_raw}` × `z_proj_init ∈ {small, xavier}` 两轴（commit `48417c1`，单测 45 条通过），四组各 450 iter、其余变量完全固定，详见 `docs/training_runs_full.md` 第 54 节。
     - **上游修复生效**：`scaled_raw`（`pool(e_t)*0.1`）把 `e_pooled` 样本特异占比从 **1.28% 提到 10.31%**（8.1 倍），correction 从 4.79% 提到 6.01%。
     - **下游修复生效**：`z_proj_init='xavier'`（实测 RMS 0.0835，是 `small` 0.0099 的 **8.43 倍**）独立把 head 响应放大 **约 10 倍**（shuffle `2.77e-5 → 2.79e-4`，zero `5.98e-4 → 6.21e-3`）；训练后权重稳在初始尺度（0.0827），说明不是被压小而是没被推大。
     - **两者叠加仍不够**：最好的组 4（raw+xavier）`e_pooled` 14.19%、correction **12.11%**、`z_proj` RMS 0.0827，但 `shuffle_z` head 仅 **3.14e-4**（门槛 1%，差 32 倍），correction² 0.0044（门槛 0.005），recon −11.8%（门槛 15%）。raw KL 有界（0.0424）通过。
     - **关键判读**：`shuffle/zero` 比值在四组间恒定在 **0.036–0.046**——Xavier 把公共偏置和样本身份一起放大了，没有偏向后者；correction 特异占比涨 2.5 倍而 head 只涨 12%。**结论：缺的不是信号幅度而是结构**，即 `correction = mu_q − mu_p` 由两个公共大向量相减得到，绝对幅度小且方向未被约束。这直接对应路线 B 的设计动机。
     - **勘误**：53.6.2 原尺度表把 `z_proj` 的 weight 与 bias 一起放大，读数被高估。分离后：仅 weight ×10 时 head ratio `2.30e-3`，weight×10+bias×10 才 `1.263e-2`；即原「×10 越过 1%」主要来自 bias 注入的样本无关偏置。探针已改为分别报告 `10.0` 与 `10.0_both`，文档已加勘误。
     - **运行事实**：四组均 GPU 5/6/7、3 卡 DDP、seed 0、deterministic、IterBasedRunner 450 iter；显存均 `7436 MiB`，`1.09–1.11 s/iter`。**四组均无 val 记录**，因此无区间均值/全轮峰值/best saved，也不能给出任何 AP 结论。
   - **路线 B：Innovation-Conditioned RSSM（2026-09-27，已完成，机制门槛 3/3 通过、性能门槛未过）**：`e_hat_t = obs_prior(h_t)`（只由 `L_obs_pred` 训练）→ `innovation_t = e_pooled − stopgrad(e_hat_t)` → `delta_mu = posterior_delta(cat[h_t, innovation_t])` → `mu_q = mu_p + delta_mu`，且 **`correction_t = delta_mu`（不再是两个公共大向量之差）**；`L_total += 0.05*L_obs_pred + 0.1*L_recon`。commit `d354232`，RSSM 单测 **55 条通过**，工具单测 12 条通过，详见 `docs/training_runs_full.md` 第 55 节。
     - **机制门槛 3/3 通过**：① `innovation` 特异占比 **14.27% > e_pooled 7.75%**（obs_prior 减掉的主要是公共成分）；② `delta_mu` 特异占比 **13.51%**（≥5%）；③ `prior_only` head 1.8e-3 明显低于 posterior 4.5e-3（0.40×）。
     - **首个系统性改善**：`shuffle/zero` head 比值从路线 A 四组恒定的 **0.032–0.046** 提升到 **0.064–0.092（约 2 倍）**，三窗口一致；`mu_q` 特异占比 1.79%→**3.93%**（+2.2 倍）。`zero_z` 绝对值同时下降（7.0e-3→4.5e-3），即整体公共偏置被削弱。
     - **性能门槛仍未过**：`shuffle_z` head 最好 4.19e-4（门槛 2%，差约 48 倍）；recon −11.5%（门槛 15%）；raw KL 有界 0.0466 通过；`stat_obs_pred` 0.0050→0.0021（−58%）。
     - **工具适配**：两个诊断工具原先硬编码 `posterior_mu` 输出为 `mu_q`，在 innovation 结构下会读错张量；现均检测 `posterior_struct` 并做 `desired − mu_p` 反演 / `mu_q = mu_p + delta` 还原，新增 2 条工具回归测试。
     - **结论**：结构方向被验证（支持 Kalman 式 predict-correct 动机，可作论文机制证据），但路线 B 解决的是「correction 由什么构成」，没有解决「correction 如何被检测头放大」；路线 A 已证明单纯放大 `z_proj` 会等比放大公共偏置与样本身份，所以不能再只调增益。
     - **运行事实**：GPU 5/6/7、3 卡 DDP、seed 0、deterministic、450 iter；显存 `7436 MiB`，`1.095 s/iter`。**无 val 记录**，无区间均值/全轮峰值/best saved，**AP 尚未测量**。
   - **路线 B ep6 门控（2026-09-27/28，已完成，AP 止损、机制通过 ep6 门槛）**：正式训练跑到 ep6 验证落盘后主动停止（保留 24e 余弦日程，避免 LR 对照失真），GPU 5/6/7、seed 0、deterministic。详见 `docs/training_runs_full.md` 第 56 节。
     - **AP 止损**：ep6 Overall 3D moderate `31.4895` vs no2d_igdr **seed0** 同 epoch `34.0243`，差 **−2.5348**，超过 2.0 阈值。附记：交接摘要里那张 ep6 = 37.3254 的对照表实际是 **seed1** 曲线，已在本节勘误；本判定按 seed0 真实曲线执行。
     - **机制门槛通过**：`innovation` specific `43.51% > e_pooled 25.66%`；`delta_mu` specific `39.32%`（≥10%）；`shuffle_z` head `0.73%–1.09%`（三窗口，越过 ep6 继续门槛 0.2%，比 450 iter 高 20–25 倍；论文最终门槛 2% 仍差约 2 倍）。
     - **未过项**：`shuffle/zero` 比值从 450 iter 的 0.064–0.092 回落到 **0.041–0.071**（绝对放大主要来自公共偏置）；`correction²` ep1→ep6 `0.1489→0.0596` 仍单调下降、**未形成平台**；recon `−6.75%`（门槛 15%）。
     - **归因**：ep6 的 `mu_q` 对 `h_t` 的依赖（removal 0.259）强于对观测的依赖（removal 0.165），`h_t` 自身 specific 已升到 12.19%，即 correction 的样本性中混入了递归状态成分，与 `z_prev` 先 L2 normalize 再进 GRU 直接相关。
     - **三口径摘要**（6 条 val，跨度为 ep1-6，标准 ep12-16 窗口不存在）：区间均值 `27.1488`（n=6，实际跨度 1-6）；全轮峰值 `34.8551 @ ep5`；best saved `31.4895 @ ep6`（`interval=2`）。
   - **路线 B 后续：`z_prev` normalize 450 iter 单变量消融（2026-09-28，机制 4/4 正向、绝对量级仍不足）**：新增 `z_prev_normalize` 开关（默认 `True`，与此前所有 run 逐位一致），配置 `..._lowdim_z_innovation_rawnorm.py`（commit `d895f10`），RSSM 单测 **58 条通过**。详见 `docs/training_runs_full.md` 第 57 节。
     - **四个观察点全部正向**：① `correction²` 首次形成平台（最后 100 iter 稳定 0.0054，路线 B 同期仍从 0.0047 滑到 0.0041）；② `mu_q` 对观测的依赖首次超过对 `h_t` 的依赖（removal e 0.315 > h 0.276，路线 B ep6 为 0.165 < 0.259，比值 1.57 → **0.874**）；③ `h_t` specific 0.86% → **1.73%**（+2.0 倍）、`innovation` 14.27% → **36.95%**、`correction` 13.51% → **30.97%**；④ `shuffle/zero` 比值 0.064–0.092 → **0.084–0.129**，三窗口同向。
     - **但绝对量级仍不足**：`shuffle_z` head 1.96e-4–3.12e-4（与 normalize 冒烟同量级，距 2% 最终门槛约两个数量级）；`zero_z` 绝对响应同时减半（4.5e-3 → 2.4e-3），即比值上升是「分子分母同时下降、分子降得更慢」；`correction` RMS 0.0574 → 0.0290（−50%）。recon 轨迹几乎重合（末端 1.0852 vs 1.0812），raw KL 0.0657 有界。
     - **结论**：`z_prev` 的 L2 normalize 确实是压制样本信息的因素之一，**该动且要保留**；但它不是充分条件，不构成「可进论文主表」的证据。**即便机制正向，是否继续仍必须由 AP 决定**——这是 ep6 门控分支的硬约束，不能因为机制读数变好就跳过 AP。
     - **运行事实**：GPU 5/6/7、3 卡 DDP、seed 0、deterministic、450 iter；显存 `7436 MiB`、约 `1.08 s/iter`；`iter_450.pth` 已落盘；**无 val**，故无区间均值/全轮峰值/best saved。
   - **rawnorm ep6 三方对照（2026-09-28，AP 通过止损、机制全面优于路线 B）**：work_dir `/data/lurui/work_dirs/lowdim_z_innovation_rawnorm_3x2x2_24e_seed0`，GPU 5/6/7、seed 0、24e 日程跑到 ep6 验证落盘后停。详见 `docs/training_runs_full.md` 第 58 节。
     - **AP 差距收窄到 −1.1263**：ep6 Overall 3D moderate `32.8980` vs no2d_igdr seed0 `34.0243`；路线 B 同 epoch 为 `31.4895`（−2.5348）。这是低维 z 系列首次把 ep6 差距压到 1.2 点内（历史：第 50 节 −6.51、第 52 节 −6.90、路线 B −2.53）。**落入预登记的 1–2 区间**。
     - **机制全面优于路线 B ep6**：`shuffle_z` head `0.84%–1.38%`（B 为 0.73%–1.09%，论文门槛 2% 仍差约 1.5 倍）；`shuffle/zero` 比值 `0.077–0.125`（B 为 0.041–0.071）；`innovation` specific `46.70% > e_pooled 16.32%`；`delta_mu` specific `38.01%`；`zero_z` 绝对响应同时低于 B（0.11–0.13 vs 0.15–0.18），即分子升、分母降。
     - **两条反向证据（必须与 450 iter 结论并列）**：① `correction²` 在正式训练里仍**逐 epoch 单调下降**（0.2218→0.0685），第 57 节的「450 iter 平台」不是稳定性质；② `mu_q` 对 `h_t` 的依赖**重新超过**对观测的依赖（removal h/e = 1.46，路线 B ep6 为 1.57，450 iter 曾反转到 0.874）。**结论：rawnorm 改善的是 recurrent 通路的幅值尺度，不是信息流向**；「放大信号 ≠ 改变信息结构」。
     - **三口径摘要**（6 条 val，跨度为 ep1-6）：区间均值 `28.4652`（n=6，实际跨度 1-6）；全轮峰值 `33.8487 @ ep5`（无对应权重）；best saved `32.8980 @ ep6`。
     - **运行事实**：显存 `7438 MiB`、约 `1.05 s/iter`；ep1 前半段 `correction²` 冲到 0.5457、`grad_norm` 峰值 15.87 后自行回落，全程无发散；落盘 ep2/4/6；看门狗在 07:04:50 UTC 于 ep6 验证落盘后停止训练。
   - **路线 C 实现 + 450 iter 机制门控（2026-09-28，机制部分通过、绝对身份信号仍不足）**：实现 commit `6d7e865 feat(rssm): add object-centric latent supervision (route C)`，详见 `docs/training_runs_full.md` 第 59 节。
     - **实现**：辅助 head 只吃 `posterior_correction`（`h_t` 不可见，有 pre-hook 回归测试断言 head 输入与 `stat_correction_sq` 逐位一致）；CenterNet 风格 focal loss（alpha=2, gamma=4）；GT 热图由当前帧 `gt_bboxes_3d/gt_labels_3d` 生成（`bev_centers_to_heatmap`，纯 torch）；历史帧不带 target；默认权重 0 时 loss 逐位不变、head 根本不 forward；推理零开销；辅助头在 `init_weights()` 最后初始化以保持旧 checkpoint 逐位可比。**RSSM 单测 74 条通过**（新增 11 条）。
     - **预登记升权有证据**：0.05 跑完 450 iter 后首层 RMS 0.04817 vs Xavier 初值 0.04811（+0.12%）、末层 bias −2.19000 → −2.19207，命中「梯度太弱」条件，按预登记升到 **0.1**；head-only 过拟合探针确认 focal 目标本身可学（40 步内 `prob_at_gt` 0.133 → 0.953），排除接口写错。
     - **机制正向（四项）**：① hit@1 `0.0141 → 0.3988`（约 28 倍，单调）；② `correction²` 最后 150 iter 平台 `0.0077 → 0.0075`，平台值比 rawnorm 450 iter 的 0.0054 高 39%；③ `correction` specific `28.17%`（≥5% 门槛）；④ **上游 `e_pooled` specific 7.62% → 15.33%（+2.0 倍）**，是此前任何配置都没做到的 encoder 侧推进。recon `1.2268 → 1.0682`、raw KL `0.2209 → 0.0764` 有界，未发散。
     - **核心门槛仍未过**：`shuffle_z` head 最好仅 `0.000580`（0.58%），距 2% 论文门槛差约 3.4 倍；`shuffle/zero` 比值三窗口**全部下降**（0.129→0.087、0.120→0.078、0.084→0.061），原因是 `zero_z` 涨得更多（+158%~+174%）而 `shuffle_z` 只涨 72%~88% —— **公共偏置与身份信号同时放大，公共分量放大更多**。结论：路线 C 让 correction 编码了更多观测内容，但没让检测头更依赖样本身份。
     - **诊断口径修正**：首版 recall 用 `prob>0.5`，对 focal 软目标恒为 0（是口径问题不是模型问题）；现改为「每个**有 GT 的类别通道**内 argmax 是否落在该类 GT 中心」的 hit@1，另报 `stat_object_hm_thr50` 作参考。
     - **ep6 门控已完成（2026-09-28/29，AP 通过、身份信号增强、`correction²` 仍塌缩）**：work_dir `/data/lurui/work_dirs/lowdim_z_innovation_rawnorm_oc_3x2x2_24e_seed0`，GPU 5/6/7、seed 0、24e 日程跑到 ep6 验证与权重落盘后停。详见 `docs/training_runs_full.md` 第 60 节。
     - **AP 门控通过，首次单点反超基线**：ep6 Overall 3D moderate `34.2513` vs no2d_igdr seed0 `34.0243`（**+0.2270**），落入预登记的「差距 ≤1」分支。但 ep1-6 均值仍低 `2.2319`（`26.8101` vs `29.0420`），ep1 尤其慢（`13.6636`，比 base 低 1.60），单点反超不能推广为整体追平。
     - **ep6 四类构成不均**：Truck strict `+4.7601`（`26.6900` vs `21.9299`）、Car strict `+0.8743` 为正；Ped loose `−3.9130`（`23.2093` vs `27.1223`）、Cyc loose `−0.8132` 为负。Overall 反超主要由 Truck/Car 贡献。
     - **身份信号相对 450 iter 增强 23–31 倍**：`shuffle_z` head 三窗口 `1.141%–1.364%`（450 iter 为 `0.369%–0.580%`），越过 ep6 继续门槛 `0.2%`；`shuffle/zero` 比值 `0.0753–0.0995`，三窗口均不低于 450 iter，不再退化。论文最终门槛 2% 仍差约 1.5 倍。
     - **source 探针继续正向**：`e_pooled` specific `16.32% → 30.98%`（+90%，450 iter 为 15.33%）、`correction` specific `38.01% → 44.90%`、`innovation` `46.70% → 50.13%`；`correction` RMS 稳定在 `0.25656`（450 iter 时 0.06239），即增益来自方向分化而非幅度膨胀。`delta_mu` specific `44.90% ≥ 10%`，`innovation > e_pooled` 均通过。
     - **两条反向证据（必须与 AP 并列）**：① `correction²` 正式训练里仍**逐 epoch 单调下降** `0.2736 → 0.0793`（−71.0%，逐轮 −20.2%/−33.3%/−24.3%/−18.1%/−12.2%），第 59 节的 450 iter 平台未保持，形态与 rawnorm 几乎相同；② `mu_q` 对 `h_t` 依赖仍**超过**观测（removal h/e = `1.456`，rawnorm ep6 为 `1.458`），posterior 仍由 `h_t` 主导。
     - **三口径摘要**（6 条 val，跨度为 ep1-6）：区间均值 `26.8101`（n=6，实际跨度 1-6）；全轮峰值 `34.2513 @ ep6`；best saved `34.2513 @ ep6`（两者相同）。
     - **ep12 门控已完成（2026-09-29，AP 止损、机制首次越 2%、两者解耦）**：从 `epoch_6.pth` 续训至 ep12，保持 24e 日程与优化器状态，看门狗在 ep12 验证与 `epoch_12.pth` 落盘后停止（06:26:57 UTC SIGINT，GPU 5/6/7 已释放）。详见 `docs/training_runs_full.md` 第 61 节。
     - **AP 止损成立**：ep12 Overall `34.9091` vs no2d_igdr seed0 `39.2953`，差 **−4.3862**；ep8-12 窗口均值 `35.0289` vs `38.8906`，差 **−3.8617**，均超 2 点阈值。ep7 冲到 `35.9549` 后 ep8-12 横盘 34.3–35.9，无后期追赶，**不续 ep16**。
     - **机制侧首次部分过论文门槛**：`shuffle_z` head `[3..6]` 窗口 **2.4290%**（ep6 1.3640%、450 iter 0.0580%），4 条样本一致（2.364%–2.499%）；但 `[100..103]` 1.7101%、`[1800..1803]` 1.0703% 仍未达标且远距离偏离散大。**「已过 2%」只能按窗口限定表述**。
     - **h/e 依赖比首次持续转向观测侧**：`1.456 → 1.275`（`removal e` 0.1380→0.1483、`removal h` 0.2010→0.1891），`innovation` specific 升至 `57.39%`（全程最高）。这是低维 z 系列第一次看到跨 epoch 的信息流向改善。
     - **两条反向证据**：① `correction²` 12 轮**全程单调下降** `0.2736 → 0.0537`（−80.4%），从未持平，辅助监督没有阻止 correction 幅值被压缩；② Truck/Cyc 是主要缺口（ep12 `−6.9324` / `−7.0035`，窗口均值 `−6.2554` / `−3.7707`），唯一持平项是 Car（ep12 +0.1878）。ep6 时 Truck +4.7601 的「优势」被证明是早期波动。
     - **联合判定**：本例是 **AP 明确退化 + 机制明确增强**，既不是成功也不是「机制退化被 AP 掩盖」，而是**机制可测但未转化为检测收益**。按预登记「AP 差距 >2 → 路线 C 单独不成立」，**路线 C 单独止损**，不得据此宣布 B+C 组合版有效。
     - **三口径摘要**（12 条 val，跨度为 ep1-12）：区间均值 `35.0289`（标准 ep12-16 窗口不存在，用实际共同区间 ep8-12，n=5）；另附 ep1-12 均值 `30.9967`（n=12）；全轮峰值 `35.9549 @ ep7`（无对应权重）；best saved `34.9091 @ ep12`（落盘 ep2/4/6/8/10/12 中最高）。
     - **下一步（区分实验，不改变 KL 口径）**：① 把 `z_proj` 直连更高分辨率读出（当前 16×16 再上采样到 216×248，可能不足以影响 Truck/Cyc 的尺寸/朝向回归）；② 把 object 监督从「只监督目标中心」换成尺寸/朝向/速度辅助目标（缺口正集中在需要几何回归的类别）。两者可与路线 E 解耦。
     - **仍未做**：路线 D（跨样本去塌缩 `L_common + L_var`）/ E（KL balancing + 按 cell 聚合 free-bits）未开始。注：当前 `clamped_ratio` 0.9440 是**逐元素**口径（`free_nats=0.1`，8192 维），不能据此断言 free-bits 失效，只作路线 E 动机。
   - **本次主线结论**：三项结构修复（差分重建、排名损失、空间读出）全部未过门控，路线 A 的两轴修复均已证实有效但叠加后仍差 32 倍，完整曲线与归因见 `docs/training_runs_full.md` 第 53、54 节。
  - **回主线：Motion-Aligned Innovation RSSM（MAI）（2026-09-29/30，450iter 冒烟 + ep6 门控全部通过、ep12 门控通过，ep16 续训中）**：按用户第 61 节后结论，低维路线止损、回到全分辨率主线，只把 innovation posterior 移植过来。新类 `MotionAlignedInnovationRSSMFusion`（父类 `MotionAlignedRSSMFusion` 全保留：216×248 latent、ConvGRU `transition`、deformable h/z 对齐、`output_proj` 残差读出、KL+recon），唯一新增 `obs_prior(h_t)` 与 `L_obs=0.05*smooth_l1(e_hat, stopgrad(e_t))`；`delta_mu=posterior_mu([h_t, innovation])`、`mu_q=mu_p+delta_mu`。commit `3acbdcb`，RSSM 单测 **85 条通过**（原 75+新 10，含父类 RNG 流不变与主线 checkpoint 可加载），工具单测 12 条通过。详见 `docs/training_runs_full.md` 第 62 节。
     - **标准 RSSM 同 epoch 机制基线（用户步骤 1，已完成）**：seed0 的 ep6/12 权重已被 2026-09-24 清理删除，改用配置相同、权重完整的 **seed1**。216×248 全分辨率三窗口 `shuffle_z` head：ep6 `49.73% / 3.30% / 3.36%`、ep12 `2.71% / 4.71% / 4.14%`；`zero_z` head ep6 `36.97% / 40.87% / 44.29%`、ep12 `26.39% / 47.66% / 54.34%`。**近距 ep6 的 49.73% 是单样本主导的假象**（样本 3 达 161.6%，样本 4/6 仅约 1.1%），须单独标注。
     - **基线关键判读**：标准主线 `e_t` sample-specific 33.74%→66.35%（上游信息充足），但 `mu_q` 特异占比反向从 22.48% 降到 18.75%，h/e 依赖比 2.228→2.338 —— **posterior 越来越依赖 h_t、越来越忽略信息充足的 e_t**，正是 MAI 要修的形态问题。同时标准主线中/远距 shuffle（4.71%/4.14%）**高于路线 C**（1.71%/1.07%），验证了「低维还没超过原主线」的判断。
     - **450 iter 冒烟通过（步骤 3）**：GPU 5/6/7、seed 0、deterministic、无 val、`EXIT=0`、无 NaN/Inf、17.27→17.28 GiB/卡、1.55–1.61 s/iter。recon `0.9537→0.1649`（−82.7%）、`stat_obs_pred 0.1249→0.0484`（−61.2%）、`innovation²` 0.81–0.94 有界、raw KL `1.7304→0.2258` 有界、grad_norm 稳在 16。**唯一下行风险：`mu_diff² 0.3178→0.0417`（−86.9%），仅末 100 iter 持平**，须由 ep6 门控核查是否再次塌缩。
     - **ep6 门控已完成（步骤 4，2026-09-29，三项门控全部通过且 AP 反超）**：work_dir `/data/lurui/work_dirs/fgfull_N4_no2d_igdr_mai_2x4_24e_seed0`，GPU 5/6/7、seed 0、24e 日程跑到 ep6 验证与权重落盘后停（val 19:15:12 UTC 写完，看门狗 `me_rssm/sanity/watch_mai_ep6.sh` 自动停训并跑 probe，SIGINT 退出）。详见 `docs/training_runs_full.md` 第 62.9–62.15 节。
       - **AP 反超基线 +1.9046**：ep6 Overall 3D moderate `35.9289` vs no2d_igdr seed0 `34.0243`；逐类别 delta Car **+8.0021**、Truck +1.5706、Ped +0.0140、Cyc **−1.9681**（唯一负项，仍在 ≤2 内）。ep1-6 均值 `30.1764` vs 基线 `29.0420`（**+1.1343**，n=6）；ep3-4 曾分别落后 0.42/1.18，ep6 才重新反超。
       - **身份信号三窗口全部越过 2% 论文门槛**：`shuffle_z` head 近/中/远 = **14.94% / 8.11% / 5.68%**（标准 RSSM 同 epoch 3.30%/3.36%，即 2.5×/1.7×；路线 C ep6 仅 1.36%/1.16%/1.14%）。`shuffle/zero` 比值 0.302/0.168/0.118，标准为 0.081/0.076。保留项：远距窗口仅近距的 38%，样本离散未消失。
       - **h/e 依赖比首次降到 1.0 以下（0.9996）**：标准主线 2.228、路线 C ep12 最低只到 1.275。`h_t` specific 20.65%→13.51%、`mu_p` 22.25%→11.56%，而 `mu_q` RMS 0.297→0.394、`correction` RMS 0.108→0.165 —— posterior 不再靠 `h_t` 的公共底色撑幅度。`innovation` specific 47.93% > `e_pooled` 27.43%。
       - **`mu_diff²` 首次出现非单调回升**：ep1 1.2663 → ep4 0.0976 → **ep5 0.1627** → ep6 0.0686，直接否定「correction 必然单调塌缩」。`obs_pred` 稳定 0.027–0.034，`innovation²` 0.98→0.71 未塌缩，`kl_raw` ep1 15.9 后稳定 0.62。
       - **三口径**（ep6 截断，n=6）：区间均值（实际跨度 ep1-6）`30.1764`；另附 ep3-6 均值 `33.2634`；全轮峰值 `35.9289 @ ep6`；best saved `35.9289 @ ep6`。
       - **判定**：按预登记「AP 差距 ≤1 且机制继续增强」成立，已从 `epoch_6.pth` 续训至 ep12（`mai_ep12` + `mai_ep12_watch`）。ep12 复核项：ep8-12 均值 ≥ `38.3906`（基线 −0.5，目标 `39.3906`）；`shuffle_z` 是否继续向 2% 靠拢；h/e 比是否守住 ≤1；`mu_diff²` 是否保持非单调。
       - **保留意见**：单 seed、单 epoch 的 +1.90 不能当结论（基线三 seed 在 ep12-16 的 std 为 1.35），且 ep3-4 曾连续落后；判定依据是「按预登记规则可继续」，不是「已统计证实」。
     - **三口径（450 iter 冒烟）**：IterBasedRunner 无 val，**不产生区间均值/全轮峰值/best saved**，不能给 AP 结论。
    - **ep12 门控已完成（步骤 5，2026-09-30）**：2026-09-30 01:34:35 → 07:49:07 UTC，从 `epoch_6.pth` 恢复到 ep12 验证与 `epoch_12.pth` 落盘后停（24e 余弦日程与优化器状态保留，GPU 5/6/7、seed 0）。详见 `docs/training_runs_full.md` 第 63 节。
      - **AP 门控通过、小幅领先**：ep8-12 五轮均值 `39.3392` vs no2d_igdr seed0 `38.8906`，差 **+0.4485**（门槛为 ≥ `38.3906`，目标 `39.3906` 差 0.05 未达）；ep12 单点 `39.2805`（−0.0148）；全轮峰值 `39.6165 @ ep10`；best saved 同为 `39.6165 @ ep10`。逐类别窗口均值 Car **+4.0009**、Truck **+1.4678**、Ped −1.2272、Cyc **−2.4457**（Cyc 已越过 2 点缺口）。
      - **机制优势部分回落**：三窗口 `shuffle_z` head `3.4918% / 4.3686% / 5.9410%`，近距从 ep6 的 14.94% 回落 76.6%；相对标准 RSSM ep12（2.71%/4.71%/4.14%）仅 1.29×/0.93×/1.43×。**「z 样本判别力大幅增强」只在 ep6 成立，不能作为 ep12 结论。**
      - **唯一持续强化的是依赖方向**：h/e 依赖比 `0.9996 → 0.8418`（标准主线同期 2.228 → 2.338，反向移动），`correction` specific `35.76% → 46.28%`、`innovation` specific `47.93% → 66.39%`；但 `innovation²` 0.708→0.287、`mu_diff²` 0.0686→0.0364，correction 绝对幅度仍在收缩。按预登记「AP 恢复到基线 −1.0 内 → 继续 ep16」启动续训；用户第 6 步几何辅助监督的触发条件（AP 保住但无提升且 Cyc 缺口扩大）尚未满足，暂不加新监督。
    - **ep16 续训已完成（步骤 5 延伸，2026-09-30）**：2026-09-30 10:37:10 UTC 从 `epoch_12.pth` 恢复，ep16 val 14:42:29 UTC 写出、`epoch_16.pth` 落盘，看门狗 14:42:51 停训并于 14:44:44 完成三窗口 + source 探针；GPU 5/6/7 已释放。详见 `docs/training_runs_full.md` 第 64 节。
      - **三口径**（16 条 val，跨度 ep1-16）：区间均值（标准 ep12-16，n=5）**40.1242 ± 0.8614**；另附 ep1-16 均值 `36.0257`（n=16）、ep13-16 均值 `40.3351`（n=4）；全轮峰值 **41.4160 @ ep16**；best saved **41.4160 @ ep16**（落盘 ep14/ep16 中最高）。
      - **AP 判定**：对同 seed0 基线 `39.2809` 高 **+0.8434**；对三 seed 基线 `40.4400 ± 1.3475` 低 **−0.3158**，在 seed 间噪声内。ep12-16 逐类别 vs seed0：Car **54.9341 vs 47.8399（+7.0942）**、Truck **30.8444 vs 29.7246（+1.1198）**、Ped **27.5324 vs 30.9277（−3.3953）**、Cyc **47.1859 vs 48.6313（−1.4454）**。Cyc 缺口从 ep8-12 的 −2.4457 收窄到 −1.4454，Ped 缺口从 −1.2272 扩大到 −3.3953；增益仍集中在 Car/Truck。
      - **机制判定（ep12→ep16）**：`shuffle_z` head 近/中/远 `3.4918/4.3686/5.9410 → 3.8653/4.1821/5.4094`，近距小幅回升但中/远回落，仍与标准主线同量级（ep12 标准 2.71/4.71/4.14）；h/e 依赖比 **0.8418 → 1.1236**，**不再保持 <1**（标准主线 2.338）；`correction` specific `46.28% → 38.01%`、`innovation` specific `66.39% → 67.03%`、`obs_prior` specific `60.80% → 55.12%`。**MAI 的早期机制优势没有随训练保持；ep16 的 AP 提升不能归因于 z 身份信号持续增强。**
      - **训练稳定性**：ep13-16 `recon 0.0053→0.0044`、`kl_raw 0.5476→0.5383`、`clamped 0.9924→0.9931`、`mu_diff² 0.0338/0.0345/0.0325/0.0335`（非单调、有平台）、`obs_pred 0.0192→0.0174`、`innovation² 0.2171→0.1595`；无发散，`mu_diff²` 未回到低维路线的单调塌缩，但绝对量级仍小。
      - **下一步判定**：AP 未停滞（ep16 单点 41.4160 为全程最高，窗口均值高于同 seed0 基线、低于三 seed 均值 0.32），且 Cyc 缺口未扩大、反而收窄；按用户第 6 步的触发条件「AP 保住但无提升」尚不完全满足。当前优先补 seed1/2 复现该单 seed 结论，再决定是否单变量加几何辅助监督；路线 E（KL balancing/free-bits）仍最后。
    - **MAI seed1 复现已启动（2026-10-01）**：`seed1`、GPU 5/6/7、deterministic，配置与 seed0 完全相同，仅 `--seed 1` 与 work_dir 不同；work_dir `/data/lurui/work_dirs/fgfull_N4_no2d_igdr_mai_2x4_24e_seed1`，日志 `/tmp/mai_seed1_ep16.log`。tmux `mai_seed1_ep16` + `mai_seed1_ep16_watch`，看门狗 `me_rssm/sanity/watch_mai_ep16.sh` 已参数化，ep16 落盘后自动停训并输出 `/tmp/z_util_mai_seed1_ep16_{w3,w100,w1800}.json` 与 `/tmp/post_source_mai_seed1_ep16_stride700.json`。目的：判定 seed0 的 +0.84（对同 seed 基线）是否为 seed0 对点波动；同时复核 h/e 依赖比是否在所有 seed 都回升到 >1。
      - 运行保护：看门狗新增 `START_EPOCH` 判定，仅当 `epoch_16.pth` 修改时间晚于本次看门狗启动时才触发，防止复用旧日志/旧权重误触发（2026-10-01 03:08 的 seed0 默认参数误触发已修正，probe 数值经核对与原记录一致）。
      - **中期观察（ep1-6，2026-10-01）**：seed1 ep1-6 Overall `17.82 / 28.14 / 35.98 / 37.36 / 36.61 / 42.77`，ep6 已显著高于 seed0 ep6 `35.93` 与 seed0 基线 `34.02`；ep6 逐类别（Car/Truck/Ped/Cyc，moderate lane）`51.64 / 35.31 / 31.92 / 52.23`。**但 seed1 的 no2d_igdr 基线本身就更强（ep12-16 = 41.9185 vs seed0 39.2809），绝对单点不能直接比较，必须以 ep12-16 同 seed 配对窗口为准。**
      - **配对更新（ep7，2026-10-01）**：seed1 MAI 逐轮 vs 同 seed no2d_igdr 基线差值：ep1 `−0.16`、ep2 `+2.97`、ep3 `−0.05`、ep4 `+2.92`、ep5 `+0.39`、ep6 **`+5.45`**、ep7 `+0.29`。早期与 seed0 的 +1.90 ep6 同向；seed1 基线后期更强，最终仍只认 ep12-16 配对窗口。
      - **seed2 自动续接已上线（2026-10-01 17:13 UTC）**：旧的 seed1 专用 watcher 已替换为链式脚本 `me_rssm/sanity/watch_mai_seed1_then_seed2_ep16.sh`。它会在 seed1 ep16 验证与 `epoch_16.pth` 落盘后停训、等待 seed1 四个 probe 完成并确认 GPU 5/6/7 空闲，再以 `seed=2`、同一 MAI 配置、`deterministic`、work_dir `/data/lurui/work_dirs/fgfull_N4_no2d_igdr_mai_2x4_24e_seed2` 启动 seed2，最后在 seed2 ep16 落盘后停训并执行同一套 probe。脚本已做隔离彩排：验证了行号/时间双重 stale-guard、半成品 checkpoint 过滤、seed2 崩溃后清理残留 tmux 并从最近完整权重恢复、探针失败不阻断续接、探针 30 min 硬超时。正式运行时不改 `max_epochs`，仍靠 ep16 闸门停训。
      - **首夜中断与恢复（2026-10-02）**：链式 watcher 在 10-01 19:36:54 正确命中 seed1 ep16 并发 SIGINT，但 torch distributed 退出耗时超过原定 600 s；watcher 在 19:46:55 以 `seed1 did not stop cleanly` 退出，**seed2 未自动启动**。seed1 的 `epoch_16.pth`（19:25 落盘，694 MB）与 ep16 验证均完好，GPU 5/6/7 随后空闲。修复：`STOP_TIMEOUT_SEC` 放宽到 1800 s，且停训超时后只要 GPU 已空闲即视为停止成功；新增 `START_PHASE=seed2` 支持中途接管。10-02 03:01:08 以该模式启动 seed2，03:05 `Epoch [1][100/1902]`，训练正常。seed1 缺失的四个 ep16 probe 已由人工补跑并落盘。
      - **MAI seed1 ep16 结果（2026-10-02，已完成）**：见 `docs/training_runs_full.md` 第 65 节。三口径：区间均值（ep12-16，n=5）**39.4122 ± 0.8133**；全轮峰值 **42.7727 @ ep6**；best saved **42.7727 @ ep6**（ep2/4/6/8/10/12/14/16 中最高）。对同 seed1 no2d_igdr 基线 ep12-16 的 **41.9185** 低 **2.5063**，与 seed0 的 +0.8434 方向相反 —— **MAI 的 seed0 收益未在 seed1 复现，按预登记继续到 seed2 后才能判定三 seed 结论。**
      - **MAI seed2 ep16 与三 seed 终判（2026-10-02，已完成）**：见 `docs/training_runs_full.md` 65.8/65.9 与第 67 节。seed2 三口径：区间均值（ep12-16，n=5）**40.0222 ± 0.8788**；全轮峰值 **41.3163 @ ep12**；best saved **41.3163 @ ep12**；last val 39.7109 @ ep16。对同 seed2 基线 40.1207 差 **−0.0985**。三 seed 配对 ΔAP = `+0.8434 / −2.5063 / −0.0985`，均值 **−0.5872 ± 1.4105**（跨 seed std 1.41 与基线自身 1.35 同量级）；MAI 三 seed 窗口均值 39.8529 ± 0.4178 低于三 seed 基线 40.4400 ± 1.3475。
      - **机制与 AP 解耦（已定量）**：三 seed 都能测到 `shuffle_z` head 4.75%–6.74%、`innovation` specific 高于 `e_t`/`obs_prior`、`correction` specific 稳定 37–39%、h 侧依赖低于标准主线 2.338；但 seed1/seed2 机制读数更好时 AP 反而更低或持平。**判定：innovation posterior 保留为机制级组件（Method/Analysis），不作为 AP 主张；检测收益赌注转到 MCI 合并模型。** 类别上 Cyc loose 是唯一三 seed 同号缺口（−1.4454 / −5.8917 / −1.4660，均值约 −2.93）。
   - ④ KL 口径改动**仍后置**：待结构有效性过 ep6 门控后再做 stop-gradient 拆分（`L_dyn=KL(sg(q)||p)`、`L_rep=KL(q||sg(p))`，建议 `beta_dyn=1.0`、`beta_rep=0.1`）与按 cell 聚合的 free-bits（沿 channel 求和后 ~1.0 nat/cell，先固定不退火）。**不要在缺重建项时单独调 free-bits。**
   - 低维 bottleneck 与排他未来任务这两个设计方向**尚未被证伪**；本轮证伪的是「在无重建锚定 + prior 脱离 `h_t` 的前提下使用它们」。标准 RSSM 的四条关键要素（prior 依赖 `h_t`、观测似然/重建项、KL balancing、输出吃 `(h_t,z_t)`）与逐条对照见第 50.6.7 节。
   - categorical/unimix 放在上述步骤之后验证，只在低维 z 仍表现出多峰/离散切换表达不足时再引入；不要直接把逐像素 256ch Gaussian 全量替换成 categorical。
3. **框架级审查（2026-10-02，纯分析，无训练动作）**：遍历全部训练记录与代码后的 RSSM 机制级问题清单（P1 读出把 z 退化为公共偏置、P2 free-bits 短路 KL、P3 predict-correct 不闭环、P4 Doppler 未进动态、P7 机制-AP 解耦——含 seed1 ep16 探针新读数：机制读数良好但 AP −2.51）与改进路线（MAI×ME-RSSM 合并 + 可靠性门控读出 + state_init/差分重建）、实验序列建议（不打断当前 seed2/fgfull18 队列），见 `docs/rssm_review_20261002.md`。
4. **两项只读测量完成（2026-10-09，GPU4 用户授权窗口；结果入档 §69）**：
   - 测量(1) 区域依赖分解（GT 框前景 / 能量匹配背景 / Doppler 分区）：**DC-RSSM-1 ep16 的 `shuffle_z` 在前景上无集中**（fg−bg_hi = −0.0001，5/12；fg 甚至低于全背景均值），而 **baseline MAI ep14 的前景集中极强**（fg 0.1888 vs bg_hi 0.0438 = 4.3×、vs bg_top10 12.1×，12/12）。⇒ 按预登记判读规则：**瓶颈在读出头而非状态容量，「方法探索 2」里的「前景-only RSSM」方向不立项**。DC 代际（确定性 + innovation 后验）丢失了 MAI 具有的「目标特异」状态内容——这是可直接进 Analysis 的机制级归因。
   - 测量(2) 重建目标 headroom：现役 `pool16_globalstd` 目标在 DC-RSSM-1 ep16 上 **95.5% 的方差是样本无关模板**（特异仅 4.5%）；换成**帧差分**目标后样本特异占比 **97.8%（21×）**，三个 checkpoint 稳定；L1 解析自车全补偿后仍保 **68.8%**。工具校验锚已复现 §51/52 表（偏差 ≤2.3 pt）。⇒ **重建目标改革升为首要候选**，取代「裁剪状态／前景-only」。
   - DC-RSSM-2 ep16 必查项已闭环：`warp_scale` 平台在 ep7 起被打破，但到 ep16 也只到模块满程位移的 **12.93%**（P5 口径，见 §68.9j）。
5. **DC-RSSM-2 ep16 终判完成（2026-10-10，入档 §68.9j）：AP 门 REJECT、L1 机制门 PASS。**
   - AP：ep12-16 窗口 **38.5345**（基线线 39.2809 → **−0.7464**；MCI-v2−0.5 线 38.8786 → **−0.3441**）；Car +5.5747 PASS、Truck +1.0659 PASS、**Ped −6.3493 FAIL**；峰值/best saved 41.1838@ep8、last val 39.2968@ep16。
   - 机制：**P5 L1 toggle 8/10 通过**（learned vs 强制 0 降低静态格跨帧残差；非惰性 0.058 m/frame = 满程 12.93%；**强制 1.0 只赢 5/10** ⇒ 收益幅度敏感、局部于约 13% 工作点）；P1 三窗 `shuffle_z` **5.62/5.39/5.54%**（家族最好）；P2 `correction` 特异 41.77%；P4 稳健解误差 0.1–0.2%（朴素 24–95%）。
   - **判定动作（按预登记）**：**停止 DC-RSSM 方向**，不写 GO、不启动 ep16→ep24 续训，不做超参扫描；机制归因已入档，转论文机制分析（§69.8 + `docs/rssm_paper_directions_20261008.md`）。
   - 家族形态第五次复现：Car/Truck 大涨、Ped/Cyc 大跌、Overall 不过门（MCI 三 seed + DC-RSSM-1/2）；机制读数越好、AP 越不动（P7）仍是未解的核心问题。

## 已完成

- **A2 四项契约已补齐、代码已实现、两臂 S0 全过（2026-10-11，§73.6；A2-S ep6 训练中）**：
  四项审查点已按预登记写死并在实测上定值 —— (1) 排序项 `q=0.999`/`margin=1.0`/
  `λ_rank=0.1` + 逐占用通道归约 + 空通道处理，且 `λ` 与 `margin` 的取值有实测依据
  （train 侧 `margin=0` 只有 1/37 通道有梯度，`margin=1.0` 为 7/37）与梯度预算记录；
  (2) A2-R 的 48 通道展开公式 + 同参数量，`num_anchors` 由 `R4Det` 从检测头读取而非
  在 config 重抄；(3) 采样/统计契约冻结为 20 clip 且逐位核对选样，`top-k`/并列/均值
  中位计数逐门写死，Ped 仅覆盖 5/20 已标注为弱证据；(4) 早期门与 S2 成熟门分离、
  等号统一为 GO `≥` / 停止 `<`。**口径更正**：实测 `get_bboxes_single` 的 reshape 是
  `(cell, anchor, class)`，推理时**不存在跨 anchor 的 max**；故 §71.9.18 的
  `frac_cells_residual_can_flip=0.2530` 是探针代理口径、与推理语义不等价
  （§71.9.18 判定不变）。代码 + 18 条单测（T14–T19 新增）通过；
  两臂 450-iter S0 四项全 PASS（身份 `0.0/0.0`、压力 0.896/0.900、readout live、
  语义项不动 readout、无 NaN、`iter_450.pth` 落盘）。**A2-S 24e 训练进行中**
  （seed0/deterministic/GPU5/6/7，挂 `watch_semA2_ep6.sh`）；**A2-R 按 §6 串行，
  尚未开跑**。S0 只证接线，**不证明机制成立**；A 的 S1 仍 NOT ALL PASS、ep16 未授权。
- **A2 看门狗证据缺失已修复（2026-10-11，§73.6.5；训练中修复，不改判据）**：
  首版 `watch_semA2_ep6.sh` 漏跑 `semA_ep6_verdict.py` 必需的四个家族标准产物
  （`z_util_std_ep6_w3/w100/w1800.json`、`post_source_std_ep6_stride700.json`），
  也不调用 verdict/record —— 按旧版跑完会判 **INCOMPLETE BUNDLE**。三层修复：
  (1) 看门狗补齐四个产物 + verdict/record（窗口/offset/stride 与 A 完全一致）；
  (2) 新增幂等补跑器 `finish_semA2_ep6.sh`（仅在产物缺失时补跑，已在
  `10:48:35 UTC` 挂到 A2-S 补跑，A2-R 开跑后同样要挂）；
  (3) `test_watch_semA_ep6_cli.py` 10 → **19 条**，新增六条把 A2 看门狗/补跑器
  纳入静态契约。负向验证：用 `8ae570d` 的旧脚本跑同一测试，新测试 **3 条失败**
  ⇒ 测试确实拦住该缺陷。**未改判据数值、未改模型/超参、未重启训练**；A 的归档
  判定不变。
- **A2 草案技术审查完成（2026-10-11，§73.5；整稿/训练未批准）**：同意 S/R
  分离和保留 S2 筛选数值；投训前须固定排序项 q/margin/系数/归约、48 通道
  映射及参数量、采样/聚合/有效性契约，限定类别非劣门仅用于 ep12–16 并消除
  ≥/≤ 等号冲突。A 仍 S1 NOT ALL PASS；本次未实现、未重跑测试、未启动训练。


- **建议落实进度检查（2026-10-11，完成）**：真实目标审计/复测及按规则放弃原差分臂、
  候选 A 实现/S0/冒烟/ep6 S1 与后续只读归因均已核验，见训练记录 §73。
  S1 总 NOT ALL PASS，不进 ep16；C/B/D 未进入正式验证。补充公平 ep1–6 配对：
  A Overall **34.0459** vs baseline **29.0420**（+5.0039），仅为早期上下文。
  “z 修复/随机状态必要/四类稳定提升”仍未完成；本次不启动任何训练。

- **语义头 train/val 4.3× 差距归属闭环（2026-10-11，完成；只读，见 §71.9.19）**：
  `docs/training_runs_full.md` §72.10 遗留的「来源仍未定位」已由新探针
  `me_rssm/sanity/probe_semA_bnmode.py` 拆解。**主因是数据 split**：同 pipeline、
  同 eval 模式，只换 info 文件 ⇒ train **0.4739/0.4167**、val **0.1280/0.0417**
  （n=20；train 侧已接近 train logger 自己那组 0.4923/0.3586）。**BN 归一化模式
  是次要因素**（val 侧只切 z 路径 4 个 BN：`thr50` +0.1173、`hit@1` +0.1375），
  z 采样契约此前已排除。校准：eval 臂在 val n=10 复现冻结门控 0.1167/0.0833；
  bs=1 与 bs=2 读数一致（批量中立）；范围对照（同数量 out-of-scope BN）移动≈0；
  GT-shuffle 地板近零 ⇒ 读数确为对象对齐而非位置先验。
  未做：全 split 统计、多 seed、任何训练。**不改变** S1 归档判定，**不**授权续训。
  下一步仍需用户重新预登记：轴已收窄为「留出场景泛化 + anchor/readout 耦合」。

- **失败机制分解完成（2026-10-11，完成；只读，见 §71.9.20）**：M1 的两个读数都是
  **阈值读数**，分不开「排序错」与「置信度低」。新探针
  `me_rssm/sanity/probe_semA_rank.py` 补标度无关读数 + GT-shuffle 对照：
  val `auc` **0.9235** vs 打乱目标地板 **0.4684**（17/20 clip ≥0.8）⇒
  **对象排序确实进了 z，且在留出场景泛化**；但 val **16/20** clip 的
  `max_gt_prob < 0.5`（中位 0.2072，即固定 0.5 阈值**结构上不可能过**），
  argmax 中位偏 **12.2 格**（train 1.0 格）、`recall@top-k` 0.3462→0.1024。
  ⇒ 失败 = **排序泛化、标度与定位不泛化**，两个机制同时成立。判定不变。

- **A2 重新预登记已写好（2026-10-11，草案，见
  `docs/rssm_candidate_A2_prereg_20261011.md`）**：按用户指令把上述两个机制拆成
  两条**串行**单变量对照 —— **A2-S**（监督侧，唯一变量 `sem_target_mode`：
  逐格高斯 vs 对象级池化）先做；**A2-R**（读出侧，唯一变量 `cls_readout_mode`：
  4→4 广播 12/48 通道 vs 按 `anchor_class_mapping` 写满 6 个 anchor 槽）后做。
  明令禁止：放大增益（§71.9.18 已否）、扫 `sem_supervision_weight`/KL、
  续 ep16、叠层。S2 数值与 A 的 S2 **逐字相同**；新增 `auc`/`max_gt_prob`
  只作 A2 自己的 M1'/M1-loc 诊断门，**不得回溯重判 A**。
  **§5 数值门待用户批准；未实现代码、未投训练。**
  - **自我校正（§71.9.21，只读）**：预登记初稿把失败写成「高斯把中心摊开、
    `hit@1` 只认一格」——实测**方向相反**：GT 标记区每通道有 **4–59 格**
    （`min_radius=2` 起步、随框尺寸增长），`hit@1` 比单格更宽容。真实机制是
    **背景峰压过 GT 峰**（val idx403 class2：GT 两格 0.624/0.651 均 >0.5，
    但 `bgmax=0.720` ⇒ `topk_recall=0`）。故 A2-S 的监督侧已从「收窄成一格/ROI」
    改为**排序型**（要求该通道 GT 峰压过自己的背景峰）。另：漂移双峰
    （val ≤2 格 7/20、>20 格 9/20；train >20 格 0/20），配对须逐 clip。
  - **M1-loc 门槛再校正（§71.9.22，只读）**：初稿用「`bgmax > gtmax` 的通道数」
    作门，实测**无区分力**（序统计量必然结果：val 40/42 通道、train 也 21/37）——
    背景有 53,568−n_pos 个候选而 GT 标记区只有 4–59 格。改用 **`recall@top-k`**：
    val 中位 **0.0000**（11/20 clip 为 0）、train 中位 0.3333（4/20 为 0）。
    故 M1-loc = `max_gt_prob ≥ 0.5` 的 clip ≥12/20 **且** `recall@top-k` 中位 ≥0.25。


以下事项已经收尾，保留结论和结果出处；不再在正文中只标 `[x]`。

- 论文级 z 修复/全类别改善方向研究（2026-10-08，**研究已完成，实施未开始**）。
  - 详见 `docs/rssm_paper_directions_20261008.md` 与训练记录 §68.9g。
  - 候选：对象语义 latent 直接分类读出、对象 RSSM、分模态可观测性校准、
    对象条件未来预测。每项含假设、原论文基础、消融、风险和证伪条件。
  - 优先独立验证语义读出与缺测校准，不宣称 z/四类 AP 已修复，不启动训练。


1. 已存在的 `no2d_igdr` baseline 的两种口径（2026-09-23 完成）。
   - 已跑完 temporal/GRU 控制组，ep20 val 后按预设判据截断；结果见 `docs/training_runs_full.md` 47.5.2。

2. `no2d_igdr` N4 RSSM 三 seed 补齐（2026-09-25 完成）。
   - seed0 val ep1-23、seed1 val ep1-21、seed2 val ep1-24 均已跑完；逐 epoch 曲线与三 seed 汇总见 `docs/training_runs_full.md` 48.2。
   - 固定窗口均值：ep12-16 Overall 3D moderate `40.4400 ± 1.3475`，ep18-20 `39.5780 ± 1.0651`；全轮峰值均值 `41.8296 ± 1.3843`（三 seed 峰值分别在 ep14/ep15/ep14）。
   - 结论修正：seed 间 std（1.35）与 47.2 节单 seed 判定 no2d_igdr 劣于 FG-FULL N=4 的效应量（0.74）同量级，该单 seed 结论不可靠；后续补配对比较见「训练补充」第 1 项。
   - 附注：seed2 原计划在 ep21 截断，因人工截断未执行而自然跑满 24e，是三个 seed 中唯一训完 24 轮的；其末尾权重清理单列为「训练补充」第 2 项。
- **ep6 判据口径对齐修复完成（2026-10-11，§73.6.6；训练中修复，不改归档判定）**：
  逐行核对探针/判定器/渲染器后，发现**三处代码与预登记不一致**，全部在 A2-S 的
  ep6 产出**之前**修复，故不构成事后调门：
  (1) **M3 的线是绝对条数**（`min_wins=8`）：A 用 10 clip 是 80%，A2 冻结契约
  改成 20 clip 后同一条门**悄悄变成 8/20 = 40%**。改为**比例 8/10**、
  条数按窗口换算（A2 ⇒ **16/20**），冻结样本量改为**必填参数**，窗口不足读
  NOT MEASURED（不得当更弱的通过）。
  (2) **A2 新增的 M1'/M1-loc 无任何工具评估**：两读数在 `semA_rank_*.json`，
  而判定器只读 M1–M4 ⇒ 一个两门全挂的 A2 臂仍会打印 "S1 ALL PASS"。已新增
  `arm_of_bundle()`/`a2_readings()` 与 **A2 AGGREGATE** 行（只有 M1–M4 **与**
  增补门全过才允许请求 S2），渲染器同步把两门写入 §72 表格、标题跟随全门集合。
  **按 config 路径识别臂**，不按产物是否存在——A 的归档包也含一份事后跑的
  `semA_rank_val_ep6.json`，按文件判会把 A 的归档判定重新打开。
  M1-loc 仅对 A2-S 生效。
  (3) **M4 的 `own_hit` 用错线**：代码查 `1/C + 0.20 = 0.45`，预登记写的是
  **pooled prior floor + 0.20**（A 的窗口地板 0.5128 ⇒ 真线 0.7128）。§71.9.18
  早已诊断但**只写进分析未回改代码**；现把实现对齐预登记，无地板读数时返回
  **None（NOT MEASURED）**，不回退到宽松线。
  **A 的归档逐位不变**：M3 重算仍 7/10 FAIL；M4 的 FAIL 仍由 `donor_transfer`
  0.3818 < 0.45 决定（`own_hit` 由「按错线过」改为「按对线过」）；S1 仍
  **NOT ALL PASS**、**不进 ep16**。新增 17 条回归测试，11 个相关文件合跑
  **218 passed**。**未改模型/超参、未重启训练**；A2-S 仍在 GPU5/6/7 训练中
  （11:43 UTC 检查：ep2 已过、显存约 19.7–20.4 GB/卡、无 OOM/Traceback）。
- **ep6 判定链路的口径与渲染修复完成（2026-10-11，§73.6.6–§73.6.11；训练中修复）**：
  在 A2-S 训练期间完成一轮**只读审计 + 修复**，覆盖 6 个问题：
  (1) **M3 比例化**——绝对条数在 20-clip 窗口会从 80% 悄悄松到 40%，改为比例
  `8/10` + 必填分母（A2 ⇒ 16/20），窗口不足读 NOT MEASURED；
  (2) **A2 增补门（M1'/M1-loc）接进判定器**——此前无人评估，M1–M4 全过即打印
  "ALL PASS"；新增 `A2 AGGREGATE` 行，只有两套门全过才允许请求 S2；
  (3) **M4 用 pooled 地板**——代码查 `1/C+0.2=0.45`，预登记是
  `pooled floor+0.2`（A ⇒ 0.7128）；并写入 `per_clip` 替代口径供核对但**不进门**；
  (4) **渲染器 A 代硬编码**（4 处）——`[watch]` 正则读不到 `[watch-A2S]`、
  日志名/看门狗名/预登记文档写死成 A 代；全部改为臂感知；
  (5) **标题与归档措辞臂感知化**——A2 草稿不再自称"候选 A"，失败原因会列出
  M1-loc，归档范围按 §7 分叉表给出；
  (6) **A2-R 冗余探针**——通用块已跑同一 val 排序探针，A2-R 块是逐字节重复
  且 `_a2r.json` 无消费者；**刻意不在训练中改动在跑脚本**，仅记录，待 A2-R 前清理。
  **关键只读核验**：在跑看门狗 bash 的脚本 fd 停在字节 1951，而新旧版首个差异
  在 **5700**、前 5699 字节逐字节相同 ⇒ **实时运行会执行修复后的尾部**，
  §73.6.5 的第一层修复对本次运行**有效**（合成文件独立复现）。
  **A 的归档判定逐位不变**：M3 重算 7/10 FAIL、M4 仍由 `donor_transfer` 0.3818
  判 FAIL、A 包不被误判为 A2、重渲染的 §72 草稿除预期措辞外逐字节相同。
  新增 24 条回归测试；11 个相关文件合跑 **225 passed**。
  **未改模型/超参/训练流程，未重启训练**；A2-S 正常训练中（12:09 UTC：ep3 val
  已完成、3 个 rank 全部存活）。**A2-R 开跑后仍需挂 `ARM=A2R
  finish_semA2_ep6.sh`**。
- **ep6 判定链路第二轮审计完成（2026-10-11，§73.6.13；训练中修复）**：审出并修复
  4 处，全部只读代码 + 单测，**未触碰在跑的 A2-S 训练**。
  (1) **结论方向级缺陷**——§71.9.3(2) 的跨网格例外**越权适用**：该例外是为 A 的
  M1–M4 写的，原判定式未把 A2 增补门算入，故「M1′/M1-loc 实测 FAIL + M1 恰落例外
  形状」会被软化成「证据不足、需补同口径对照」，即**用仪表口径掩盖已测出的机制
  否定**。新增 `grid_limited_applies()` 作为唯一适用范围判定（M1 未过 + 其余
  M1–M4 全过 + 例外形状 + **本臂增补门全过**），判定器与渲染器**共用**；渲染器
  原先自写的旧二子句式已删除。判定器那句 `PRE-REGISTERED RULE APPLIES` 也改为
  「读数具备形状，是否适用见聚合判定」（原先在未知其余门时就宣布适用）。
  (2) **H-A2 的 train/val 对照在草稿里缺失**——M1′ 只在 val 判，而该臂的命题
  就是「峰预算能否离开训练场景」。看门狗已跑 train split 但无人读；新增
  `a2_split_contrast()` 渲染双 split 的 `auc`/`shuffle`/`max_gt_prob`/`recall@top-k`
  中位与均值/`max_gt_prob≥0.5` clip 数 + `train−val` 差（A 的同口径差并列作参照，
  仅机制定位、不是门槛）。
  (3) **§72.1「唯一变量」写的是 A 代双旋钮句**——会同时描述错 A2-S 与 A2-R，
  且掩盖对照物是 A 底座而非 `no2d_igdr`。新增 `arm_single_variable()` /
  `arm_comparator()` 按臂输出；`变量取值` 行仅对 A2 输出，保证 A 草稿逐字节不变。
  (4) **A2 专属产物刻意「只报告不门槛化」**——新增 `check_a2_evidence()` 列出
  两份 rank 产物的存在性与 schema，但**刻意不并入 `evidence_ok`**：并入会让
  「train/val 对照缺失」把**已实测的 FAIL** 降级成 `NOT EVALUABLE`，方向是错的
  （用缺仪表掩盖负结果）；能否判定增补门由 `a2_readings()` 的 blockers 决定。
  **验证**：新增 19 条单测，11 个相关文件合跑 **243 passed**；A 的归档判定与
  §72 草稿重生成后**逐字节不变**；合成 A2 捆绑（A 产物 + `config_path` 改
  `..._semA2S_24e.py`）实测走通 A2 路径（臂识别、M1p PASS/M1loc FAIL、双 split
  对照、单变量措辞、臂感知标题）。

- **ep6 渲染器第四轮审计完成（2026-10-11，§73.6.17；训练中修复；A2-S ep6 未产出）**：
  本轮改用**「生成器输出 vs 文档 §72 逐行 diff」**的角度，查出三处**反向漂移**
  （文档已更正、生成器仍旧）加一处**放行级**缺陷，全部已修并回归。
  (1) **§72.10 归因只存在于文档**——§71.9.19 的「差距来自数据 split」是手写插入，
  生成器仍打印「来源仍未定位」；再渲染会**删掉**该更正。新增 `split_attribution()`
  从**本包自己的**产物推导该表，支持 n=20（A 的临时探针）与 n=10（A2 看门狗固定
  产出）两条路径并按实际 n 标注；两包皆无时输出 `来源归属：NOT MEASURED`
  （仍可断言契约错配不是原因，但不得断言差距来自泛化）。数字不硬编码，有结构测试
  禁止已发表读数出现在源码里。
  (2) **A2 聚合判定越权（放行级）**——标题已要求「M1–M4 与 A2 增补门」全过，但
  §72.0/§72.9 的 `elif all_pass` 仍只看 M1–M4；实测构造（M1–M4 全绿、M1′/M1-loc
  全红）会渲染出「标题=未全门通过」+「§72.0=S1 聚合判定 ALL PASS」+
  「§72.9=请求批准并续训 ep16」即**同一节自相矛盾且授权一个已被否证的臂进 ep16**。
  两处分支改为 `all_pass and added_all_pass and not grid_limited`，NOT ALL PASS
  原因列表补上增补门名；新增 2 条测试（一致性 1 条 + 防把真通过卡死 1 条）。
  (3) **§72.9 S2 止损线范围更正也只在文档里**——`dbf6bdd` 手写的「该非劣线属
  S2、ep6 不适用」未被生成器输出，再渲染会恢复成 ep6 止损线的误读。现由生成器
  输出，用固定更正日期常量，避免随 run 日期漂移。
  (4) **§73.6.17 初稿自身的一处过度声称（已更正，见 commit `8c955cb`）**：初稿把
  n=10 的 bnmode 对写成「A2 看门狗固定产出」——**不成立**，`watch_semA2_ep6.sh`
  只跑 val 侧（无 `--split-ann-file`），train-split 臂从不执行，故 A2 包**默认就
  渲染 NOT MEASURED**（这是诚实结果，不是缺陷）。**两组产物都只对 A 存在**；要在
  A2 上复现该归因必须显式补跑 train-split 臂，不能回退到 A 的数字。同时补 1 条
  测试钉住 A2 看门狗的真实产物形状（若哪天开始渲染表格即失败）。
  **验证**：把 §72 整节重新渲染后与文档逐行对比 ⇒ **IDEMPOTENT**（文档正文已与
  生成器对齐）；11 个文件合跑 **260 passed**（250 + 10）。A 的门读数与判定逐位不变。

- **ep6 渲染器第五轮审计完成（2026-10-11，§73.6.18；训练中修复）**：扫生成器内部
  字面量，发现 §72.7 的 A2 对照句把 **A 的参照值硬编码**，与 §72.11「未手抄任何
  读数」自相矛盾（数值当前正确，但会陈旧、且在 A 归档不可读时越权引用未测参照）。
  新增 `A_REFERENCE_RUN` / `a_rank_gap()`，从 A 自己的归档推导同一组聚合量，
  不可用时显式输出「A 的同口径参照不可用」。推导值与旧硬编码**逐位一致**，
  不改任何已发表读数。新增 2 条测试（禁字面量 + 断言不可用分支）；
  `test_semA_ep6_record` 107 passed，§72 仍 IDEMPOTENT。

- **ep6 渲染器第六轮审计完成（2026-10-11，§73.6.19；训练中修复）**：把 §73.6.17
  缺陷二的判据从「草稿」延伸到**判定器控制台**——人工裁决实际先看到的输出。
  构造 M1–M4 全绿 + M1′ FAIL 的 A2-S 捆，控制台会连打 `S1 AGGREGATE: ALL PASS`
  与 `A2 AGGREGATE: NOT ALL PASS … does NOT enter ep16`，**同一臂两行相反**且第一行
  是放行。修为：A2 臂若增补门未全过，`S1 AGGREGATE` 直接输出 `NOT ALL PASS`；
  A 代分支逐字节不变。新增 2 条 subprocess 测试（真跑交付脚本）；11 个文件合跑
  **264 passed**，A 的判定与 §72 草稿仍不变。

- **ep6 渲染器第七轮审计完成（2026-10-11，§73.6.20；训练中修复）**：A2 预登记 §5
  把「类别覆盖」写成**必须输出的报告字段**（并实测 val 分布 1 类×4/2 类×11/3 类×4/
  4 类×1），但生成器从未输出过。后果：归档里读不到「某类只覆盖少数 clip」这一
  会削弱类别专属结论的事实；A 的 M4 窗口（n=10）**一个 Car 都没有**，pooled floor
  0.5128 由 Cyc 主导（counts [5,8,20,6]），而报告只给一个总体 `own_hit`。
  新增 `class_coverage()` 从本包 rank 产物推导 val/train 覆盖并在 §72.6 输出；
  A 代不输出（其 rank 产物不属冻结契约）。推导值与预登记手记**逐位一致**，
  不产生新读数。新增 2 条测试；`test_semA_ep6_record` 109 passed，11 文件合跑
  **266 passed**，A 的 §72 仍 IDEMPOTENT。

- **ep6 渲染器第八轮审计完成（2026-10-11，§73.6.21；训练中修复）**：发现
  `semA_m4floor_ep6.json` **没有任何消费者**（判定器只声明常量、渲染器从未读），
  而它承载两项被判据依赖的读数：M4 的 `own_hit` 线基准 `pooled_prior_floor`
  （A=0.5128）与 A2-R 的 S0-5 `frac_cells_residual_can_flip`（§5 明写要记录）。
  测了却没有报告携带 ⇒ 记录上与「没测」不可区分。新增 `m4floor_readings()` 与
  §72.6 地板/读出存活表（pooled floor、margin、flip 比例、前景间隔、尺度曲线），
  **仅 A2 臂渲染**。新增 2 条测试；`test_semA_ep6_record` 111 passed，11 文件合跑
  **268 passed**，A 的 §72 逐字节不变。

- **A2 看门狗 z-util 重复块（2026-10-11 记录，§73.6.22；**待 A2-S 结束后清理**）**：
  `watch_semA2_ep6.sh` 把 `probe_z_utilization_standard.py` 调 5 次但只有 3 个不同
  输出——w3 与 w100 各被跑两遍（参数完全相同、仅 GPU 设备号不同），成因是修
  「证据不全」时整段追加了 A 家族的三窗，与早期的 A2 专属块叠加。探针种子固定，
  第二次只是**覆盖同值**，故非正确性问题，仅浪费每个重复窗约 1–2 min GPU。
  **A2-S 看门狗仍在运行时不得改脚本**（bash 按字节偏移读取在执行中的脚本，
  §73.6.5 已踩过），故本轮只记录。清理时机：A2-S 结束、**A2-R 开跑之前**，
  与 §73.6.10 的 `_a2r.json` 冗余块一并删除。
