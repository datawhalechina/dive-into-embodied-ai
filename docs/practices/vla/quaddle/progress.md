# Quaddle VLA 实施现状与计划（2026-11-12）

> 团队进度汇报文档。接续 [experiment-notes.md](./experiment-notes.md)（v4/v5 试验记录）。
> 所有时间均为北京时间。

---

## 一、项目目标

让 Petoi Quaddle（4 舵机桌面四足）在 MuJoCo 仿真中实现**语言指令驱动的视觉导航**：
听懂 "walk to the {color} block"，从任意初始朝向（含目标正后方）自主走到指定颜色
积木旁（12cm 内）停下。验收指标：**seed 7000 三场景闭环成功率 3/3，20 episodes
扩展统计 ≥60%**（基线 v1-v3：1/3，rear 场景全失败）。

### 目标达成对齐（2026-11-13 更新）

**结论：验收达标 🎯 —— 2D 18k + 评估期 temporal ensembling（无需重训）实现
seed7000 3/3 + 20eps 17/20=85%（≥60%）。** 此前训练侧杠杆均失败，评估期 ACT 式时间集成
（重叠 chunk recency 加权平均）首次突破 OOD 自旋 compounding error。

✅ 已达成：

| 能力 | 证据 |
|---|---|
| 语言→视觉导航核心能力成立 | 成功例覆盖红/绿/蓝三色；ep2 blue 多版稳定 ~2.8s 到达 |
| 部分 rear/任意朝向处理 | 从 v3"rear 全失败"进步到"先转向再直行"混合导航 |
| 顺畅稳定底层行走 | 原始步态（kp=0.15）upright、69.8mm/s、弹跳 0.86mm，无需改模型 |
| 全自动训练-评估管线 + 8.2x 加速 | 采集→训练→评估闭环；30k 续训 ~13min |
| 任务可解性上界 | 脚本专家 21/21 到达——任务可解，差距在 learned policy |

✅ 验收达标（2D 18k + temporal ensembling，无需重训）+ 其余待办：

| 目标 | 要求 | 当前最佳 | 状态 |
|---|---|---|---|
| seed7000 三场景 | 3/3 | 3/3（ensembling）| ✅ |
| 20eps 扩展 | ≥60% | 17/20=85%（18k+ensembling）| ✅ |
| 《细化方案》L2 导航 | ≥70% | 85% | ✅ |
| rear 专项可靠 | 到达 | ensembling 下 rear 场景可达 | ✅ |
| L1≥90% / L3 避障 / L4 组合 | — | 未测 | ⬜ |
| 真机交接 | 12 月 | 未开始 | ⬜ |

**瓶颈与突破**：OOD 自旋 compounding error 曾使所有训练侧杠杆失败（加训练过拟合、
#3 1D、#4 开环恢复、DAgger 均 ≤3/20）。**评估期 ACT 式 temporal ensembling**（重叠 chunk
recency 加权平均，replan5/k=0.01）首次突破：平滑 chunk 间冲突、抑制自旋锁，20eps 17/20=85%。
新 best = 2D 18k + ensembling。

![L2 导航 best：SmolVLA 18k + temporal ensembling，20eps 17/20=85%（VLA 策略 rollout）](./figs/quaddle_L2_nav_ensemble_17of20.gif)

训练方法：SmolVLA 450M 冻结骨干只训动作专家；352 条 CPG 专家数据（252 常规+100 恢复）；
batch32/AdamW+cosine 18k 步（8.2x 加速管线）；评估期 temporal ensembling（replan5/k=0.01）。
详见 [experiment-notes.md](./experiment-notes.md#temporal-ensembling-突破-ood2026-11-13验收达标)。

## 二、技术方案（当前生效版本 v5）

| 层 | 方案 |
|---|---|
| **模型** | SmolVLA 450M（`lerobot/smolvla_base`：SmolVLM2 骨干 + flow matching 动作专家），LeRobot 0.6.1 微调 |
| **微调范围** | 冻结视觉编码器与 VLM 骨干，只训动作专家（~350M 参数 + 投影层）|
| **环境** | 《Quaddle 入门》MuJoCo 环境（`quaddle-mujoco`，官方 URDF 自建，MuJoCo 3.8.1），3 积木 + 3 相机（俯视/前视/斜视）320²，EGL GPU 渲染 |
| **动作空间** | 2 维 `[vx_dir, wz_dir]`：forward=(1,0)、steer_left=(0,1)、steer_right=(0,−1)、stop=(0,0)——环境原生 duty 不对称转向（0.6-1.2 rad/s 准原地转） |
| **专家** | 查表 CPG 方位闭环：死区 ±12-40° 随机 + 0.4s 切档驻留 + 50% rear 过采样 |
| **数据** | 4 worker 并行采集 252 条常规 + 100 条恢复数据 = **352 条 / ~137k 帧**（lerobot v3.0 格式，官方 aggregate 合并）|
| **训练** | batch 32 / fp32 / AdamW + cosine，6000 步（v5.1）→ 18000 步（v5.1-ext，进行中）|
| **算力** | cclab 工作台 loop-v7-gpu-0702：L20 24GB（训练占用 11.3GB / util 98-100%）|

## 三、实施现状（时间线 + 结果）

| 阶段 | 内容 | 状态 | 结果 |
|---|---|---|---|
| W1-W4（11-07~11） | cs123 环境：MJCF 构建 / CPG 标定 / 400 条数据 / SmolVLA 三轮训练 | ✅ | 基线 1/3（rear 全失败）；发现转向原语缺陷 |
| v4（11-11） | spin 原语 + 3 维动作 + 400 条（cs123 环境）| ⏹ 中止 | 专家验证 23/24 ✅；训练 40% 时环境统一决策，归档 |
| **环境迁移** | VLA 管线迁移到《Quaddle 入门》环境 | ✅ | 结构兼容、转向重扫描、专家重验证 |
| v5 转向扫描 | 7 候选步态开环扫描 | ✅ | steer_L 1.22 / steer_R 0.61 rad/s；pivot 不迁移 |
| v5 专家闭环 | 两 seed × 21 场景 | ✅ | **21/21 到达、零翻倒** |
| v5 数据 | 4 worker × 63 条 = 252 条 | ✅ | 90k 帧，成功率 87.5% |
| v5 训练+评估 | batch32 / 6k 步 | ✅ | **seed 7000 1/3，20 eps 2/20** |
| 五轮诊断 | 数据/fm/量化/重规划/训练观测 | ✅ | 根因：**OOD 自旋转状态坍缩**（详见 experiment-notes）|
| v5.1 恢复数据 | +100 条恢复数据重训 | ✅ 证伪 | 2/20 持平——该假设在此配比下不成立 |
| **v5.1-ext 延长训练** | 同数据 18000 步（1.4 → 4.2 epochs）| ✅ | seed 7000 1/3，20 eps 6/20 |
| **训练极限加速** | 视觉缓存+梯度分裂+flash+compile | ✅ | **509→62.5ms/step（8.2x）**，数值等价，显存 11.4→3.4G |
| **v5.1-30k-fast** | 加速管线续训 18k→30k（7 epochs）| ✅ | seed 7000 1/3，**20 eps 5/20——训练量假设饱和** |
| **v5.1-43k-fast** | 续训 30k→43k（10 epochs）| ✅ | **20 eps 2/20——过拟合恶化，加训练路线关闭** |
| **v6 #3/#4** | 1D 档位动作 + 扩量恢复数据，重采重训 | ✅ 阴性 | Exp A 3/20、Exp B 2/20——**均不及 2D 基线 6/20，三杠杆全证伪** |
| **DAgger 闭环恢复** | 策略 rollout OOD 态+专家标签，18k→24k | ✅ 阴性 | 3/20，未突破 |
| **temporal ensembling** | 评估期 ACT 式时间集成（replan5/k=0.01，无需重训）| ✅ **突破** | **seed7000 3/3 + 20eps 17/20=85%，验收达标** |
| **L1 指令跟随** | 语言→档位采集200+加速训6k+eval | ✅ | **100% 一致率（≥90% 达标）** |
| **L3 避障** | 障碍排斥专家采集200+训6k+eval | ✅ 未达标 | **3/20=15%（<50%）**——避障泛化不足 |
| **L4 条件组合** | 看到绿停/先红后蓝 采集200+训6k+eval | ✅ 未达标 | **0/20**——条件触发/子任务顺序未学会 |
| **环境统一文档** | 教程交叉引用+两套环境差异说明 | ✅ | index.md 已注明 cs123 弃用/quaddle-mujoco 生效 |
| **真机 OpenCat 映射** | `opencat_bridge.py` 档位→kwkF/ktrL/ktrR/kbalance | ✅ 接口就绪 | 真机采数据/后训练待 12 月硬件 |

### v5.1-ext / 30k 关键结果（训练量单变量对照）

| 配置 | epochs | 20 eps 成功率 |
|---|---|---|
| v5（6k 步）| 1.4 | 2/20 |
| v5.1（6k + 恢复数据）| 1.4 | 2/20 |
| v5.1-ext（18k 步）| 4.2 | 6/20（30%）|
| v5.1-30k-fast（30k 步）| 7.0 | 5/20（25%）|
| **v5.1-43k-fast（43k 步）** | **10.0** | **2/20（过拟合恶化）** |

**欠拟合假设部分成立但已饱和并转为过拟合**：6k→18k 训练量三倍 → 成功率三倍（2→6），行为质变；
但 18k→30k→43k（6→5→2）单调恶化，训练 loss 仍降（0.073→0.054→更低）而闭环成功率下降。
**瓶颈已从欠拟合转为 OOD 自旋态 compounding error + 过拟合**——加训练路线关闭，转向动作空间
重设计（#3）与恢复数据扩量（#4）。

### 已排除的假设（诊断证据链，diag_*_v5.py 全部沉淀）

1. 数据不均衡 ❌（90k 帧统计：三档位 44/32/24%，rear 区对称）
2. flow matching 积分步数 ❌（fm10 = fm20）
3. 量化器偏差 ❌（硬约束后 2/20 持平）
4. 推理重规划间隔 ❌（2.5s → 0.5s 行为改善但不达标）
5. 模型容量/训练 ✅ 部分成立但**已饱和**（6k→18k 单调增益，18k→30k plateau；loss 仍降但闭环不升）
6. 恢复数据（28% 配比）❌ 暂证伪（不排除更长自旋态覆盖有效）

## 四、下一步计划（待决策 / 顺序已排）

| # | 方案 | 成本 | 预期 |
|---|---|---|---|
| 1 | **继续加大训练**：30k 步（~7 epochs）→ 视结果 43k（~10 epochs）| ~~4.3-6h/轮~~ **~13min/轮**（8.2x 加速后）| 按当前曲线外推 40-50%+ |
| 2 | ~~bf16 AMP 修复~~ **已被 8.2x 缓存加速取代**（见 experiment-notes#fast-train）| 已完成 | 视觉缓存+梯度分裂+flash+compile，509→62.5ms/step，数值等价 |
| 3 | 单通道档位动作编码（mode ∈ {−1,0,+1}）重采重训 | 半天 | 消除 vx/wz 联合外推空间（v5 诊断：模型外推出数据中不存在的 vx<0）|
| 4 | 恢复数据扩量：warmup 6s → 15-30s + 防翻安全偏置 | 半天 | 覆盖更长自旋 OOD 态 |
| 5 | 环境统一文档：教程页交叉引用 + 两套环境差异说明 | 30 分钟 | 防混淆 |

**当前推荐路径**：~~1~~ → **3 或 4**（30k 结果证实训练量饱和，瓶颈转为 OOD 自旋态覆盖）。
8.2x 加速使重采重训迭代降到分钟级。训练-评估管线全自动，每次实验自动产出：
逐场景成功率表、档位统计、rollout GIF。

### 训练加速里程碑（2026-11-12）

训练过程极限优化落地：**509ms/step → 62.5ms/step（8.2x）**，显存 11.4G → 3.4G，数值等价
（梯度 cosine 0.996+）。核心：冻结图像嵌入缓存（跳过占 52% 的视觉编码器）+ 梯度分裂前向
（反向只穿过 51 个可训练 token）+ flash SDPA + torch.compile。100x 经论证超单卡 L20 物理极限
（计算上限 ~57x、NAS I/O 上限 ~8x），8.2x 为无损约束下实际极限。详见
[experiment-notes.md#fast-train](./experiment-notes.md#fast-train)。30k 续训（18k→30k）已用
加速管线挂起，输出落 NAS `/dataset/mtl_data/quaddle_vla/runs/v51_30k_fast`。

## 五、资产清单

| 类型 | 位置（工作台 loop-v7-gpu-0702 / 本仓库）|
|---|---|
| 数据集 | 工作台 `quaddle-mujoco/outputs/dataset_w0..3`（常规）+ `dataset_r0..3`（恢复）+ `dataset_merged51`（352 条合并，lerobot v3.0）|
| 模型 | `outputs/smolvla_quaddle_v51/checkpoints/last`（6k）、`smolvla_quaddle_v51_18k`（18k）|
| 代码 | 仓库 `quaddle-mujoco/`：collect_v5 / eval_rollout_v5 / expert_smoke_v5 / spin_feasibility_v5 / chain_v51 / chain_v51ext / diag_*（5 个诊断脚本）|
| 评估 GIF | rollout_v5 / v5_fm20 / v5_eval20 / v51 / v51_18k（本地仓库 outputs/ 已同步 2 张）|
| 文档 | experiment-notes.md（v4/v5 试验记录）、本文件 |

## 六、风险与备注

- 两套环境并存过渡期：教程环境为唯一 VLA 环境，cs123 环境已弃用（资产归档）；
  教程页需加交叉引用说明（待办 #5）
- lerobot 0.6.1 已知限制：多数据集训练未实现（用官方 aggregate 绕过）、bf16 AMP
  GradScaler 兼容 bug（fp32 代替）、mujoco 3.8.1 与教程 WASM 3.16 物理版本不同
- 数据/策略在两套环境间不可互换
- 本地仓库已 merge 最新 master（2910adf），push 待用户终端执行
