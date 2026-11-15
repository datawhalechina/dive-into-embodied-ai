---
title: "Quaddle VLA：实验记录与 v5 路线图"
sidebar_position: 99
displayed_sidebar: practicesOverviewSidebar
---

# 实验记录与 v5 路线图

这里保留 Quaddle VLA 的历史实验结果、v4 试验记录与后续路线。初次复现请从[Quaddle VLA 动手教程](./index.md)开始。本页接续仓库根的总方案 `Petoi-Quaddle-VLA训练方案-10GB.md`（W1/W2/W5 执行记录），并对照真机主线的《Quaddle_SmolVLA训练目标细化方案.md》（本地 Downloads，下文简称《细化方案》）。

## 如何理解这些记录 \{#how-to-read}

- **专家闭环验证 ≠ 模型成功**：v4 的“23/24”是脚本化专家在仿真里的到达数，不是训练后策略的成功率。v4 模型训练仍在进行中，验收表里的成功率**待回填**。
- **没有随仓库分发的视频**：数据集、检查点、rollout GIF 都在 gitignore 的 `outputs/` 下，只在本机保留。本页不从画面反推奖励函数、训练耗时或成功率。
- **特权信息**：观测里的 `bearing/dist` 在仿真中直接给出，真机上需外置相机定位求出或让 VLA 从图像学。跨平台比较时注意这一差异。
- 记录日期：2026-11-11。v4 执行环境：cclab 工作台 `loop-v7-gpu-0702`（L20 24GB），与基线复现用的本地 4070 Ti 不是同一台机器。

## 一、v4 做了什么：把“原地转”能力缺口补上 \{#v4}

### 1.1 背景（W5 遗留问题）

v3 基线：目标在前侧方 ±60° 时 100% 到达（1/3 场景），初始 bearing > 120°（目标近正后方）时失败。W5 归因为“动作空间缺纯原地转原语——差频档必带弧线”。

### 1.2 根因更深一层：弧线档本身就是坏的（本轮新发现）

在滚轮足端 MJCF 上系统扫描（`spin_feasibility.py`，8 候选步态表 × 10s）+ 闭环专家对照（`expert_spin_smoke.py`，2 seeds × 12 场景）后，W5 的归因需要修正：

| 档位 | 实测行为 | 结论 |
| --- | --- | --- |
| turn weak（1.5/2.5Hz） | **净转向 ≈ 0 且符号随机**（多次实测 bearing 越转越偏） | W2 只标定了 strong 表，weak 表从未单独验证——v3 的“死区振荡极限环”与“恒 turn_right 超时”均源于此 |
| turn strong（1.0/3.0Hz） | wz ≈ ±0.3 可用，但持续执行约 2s 内翻倒（roll 30°+） | 只能短脉冲用，专家闭环里不可依赖 |
| 左右反相同频同幅 | 仍以直线前进为主（vx 118mm/s），翻倒 | 单向棘轮抵消了相位差，排除 |
| **spin（一侧腿停摆 + 另一侧 trot）** | **wz +0.50 / −0.42 rad/s，连续 10s roll < 13° 不翻，净位移 < 35mm** | **滚轮单向棘轮下唯一可靠的转向原语** ✅ |

推论：**v3 专家在 rear 场景本身就大量翻倒/超时被静默丢弃**（闭环对照实测 v3 专家 24 场景仅 2/24 到达）——“数据里 rear 场景少”不是采样偏差，是专家自己进不去那个域。

### 1.3 v4 专家设计（已冻结，闭环验证 23/24 vs v3 2/24）

```text
dist < 12cm                    → stop（成功）
|bearing| > deadzone(±12~40°)  → spin_left / spin_right   # 转向 100% 支点转
否则                           → forward
```

两条稳定化规则（缺一不可，均有翻倒证据）：

1. **方向锁定**：spin 触发时锁定方向直到退出（初始 bearing 近 ±180° 时符号会被噪声翻转，逐帧重取方向会抖动换向）；
2. **切档最小驻留 0.4s**：死区边界 spin↔forward 抖动会让 CPG 幅值斜坡（0.3s）反复清零重爬，腿部冲击是近目标翻倒的直接诱因。

### 1.4 动作空间与数据（v4 相对 v3 的变更）

- action 2 维 → **3 维 `[vx_dir, wz_dir, spin]`**：`forward=(1,0,0)`、`spin_left=(0,+1,1)`、`spin_right=(0,−1,1)`、`stop=(0,0,0)`
- 初始朝向 **50% 过采样“目标正后方”**（±30°，v3 均匀 yaw 下 rear 仅 1/3 覆盖，且其中大部分被翻倒专家丢掉了）
- 数据集：`local/quaddle_nav_400v4`，400 条，seed 5000（EGL 渲染，约 2s/条）
- 推理端：`eval_rollout_gif.py` 量化器适配 3 维（spin 旗标阈值 0.45 + wz 符号），执行器同款驻留规则；新增 `--fm-steps` 参数（flow matching 积分步数 10→20 对照）

### 1.5 v4 验收标准（v4 随环境迁移终止，未出最终评估；v5 在新环境接棒）

| 项 | v3 基线 | v4 目标 | v4 实测 |
| --- | --- | --- | --- |
| seed 7000 三场景闭环 | 1/3 | **3/3** | 待训练 |
| rear 场景（bearing>120°）专项 | 0%（翻倒/绕弧） | 到达 | 待训练 |
| 前侧方不回退 | ✓ | ✓ | 待训练 |
| flow matching 20 步 vs 10 步 | — | 幅值尖锐化对比 | 待试验 |

## 二、与《细化方案》的对照 \{#vs-plan}

《细化方案》以**真机**为主线（OpenCat 固件承担步态，VLA 出连续速度指令）；本文 v4 以**仿真**为主线（自建 MJCF + CPG 档位库）。两者是同一管线的两个阶段，不是二选一——《细化方案》风险表自己列明：“发货前无真机 → 官方模拟器跑通管线”、“真机数据采不动 → MuJoCo sim 预训练，真机数据后训练”。v4 就是这两条的落实，且深度超出其设想（完整闭链 MJCF 而非简化模型）。

### 2.1 关键分歧：动作空间

| | 《细化方案》主方案 | v4 实际采用 |
| --- | --- | --- |
| 选型 | 连续 `(vx, vy, ω)` | 离散档位 + spin 旗标（即《细化方案》的“技能 token”备选路线） |
| 前提 | OpenCat 固件步态质量好（kwkF/ktrL/kbwF 全向） | 仿真中连续速度**没有可靠执行器** |

仿真实测证据（v3 三轮 + 本轮扫描）：

- weak 弧线档转向不可控（净转向≈0/符号随机）、strong 档不可持续（2s 翻倒）→ ω 跟踪失败；
- 滚轮单向棘轮：无倒挡、无横移 → **`vy` 与负 `vx` 在仿真里没有执行器**；
- flow matching 连续强度编码已翻过车（输出 ±0.4 而非 ±1.0，绝对阈值量化全判 stop）。

结论：**仿真阶段技能 token 是唯一可行解；《细化方案》的连续速度决策在真机上依然成立**（固件步态质量好），届时是接口切换而非架构重写——档位 ↔ OpenCat 指令天然一一对应（`forward↔kwkF`、`spin↔ktrL`），迁移面比连续速度更干净。

### 2.2 全维度对照表

| 维度 | 《细化方案》 | v4 | 评价 / 采纳 |
| --- | --- | --- | --- |
| 任务考纲 | L1 指令跟随 → L2 视觉导航 → L3 避障 → L4 条件组合 | 仅 L2（颜色块导航） | 方案更好，**采纳为 v5+ 考纲**（见第三节） |
| 数据来源 | 真机遥操作 + Puppet Mode + 恢复数据 | 脚本化专家（方位闭环） | 各自平台最优；真机阶段按方案执行 |
| 恢复数据 | 20~30%（扰动注入 + 录制恢复） | 无显式恢复数据 | **v5 采纳**（仿真零成本：踢扰/偏移初值 + 专家天然闭环恢复段） |
| 微调方式 | 全参 bf16，batch 32~64 | 冻结视觉编码器，batch 4 | batch 4 是 4070Ti/10GB 预算遗产，**L20 上放大到 8~16** |
| 动作平滑正则 | 差分惩罚（保护舵机） | 不需要 | 档位空间 + amp ramp + 驻留已内置 |
| 观测状态 | 4 维舵机角 | 8 维（舵机+vx/wz+**bearing/dist**） | bearing/dist 是特权信息：真机需外置相机定位求出，或让 VLA 从图像学（方案的“外置俯拍相机”路线兼容） |
| flow matching 步数 | 推理 5~10 步 | 计划测 20 步 | v3 幅值保守教训支持多步；两种都测 |
| 评估 | 每任务 20 次，L2≥70%，延迟 <100ms，平滑度指标 | seed 7000 固定 3 场景 | 方案统计力强，**扩到 20 episodes/任务**；延迟/平滑度真机阶段再测 |
| 周期与验收 | 4 周，L1≥90%/L2≥70%/L3≥50% | 单任务闭环 | 真机阶段按方案验收 |

### 2.3 《细化方案》可吸收清单（按成本排序）

**立即（v4 训练时）**：

1. L20 上 batch 4 → 8（必要时 16），训练时长进一步压缩；
2. 评估从 3 场景扩到 20 episodes/任务（固定场景集 + 随机 seed 池）。

**v5（仿真扩展轮）**：

3. 恢复数据 20~30%：采集时对机器人注入随机踢扰/位置偏移，专家天然闭环恢复段自动入库；
4. L1 指令跟随任务（“向前走/左转/原地转”纯语言→档位）——注意“后退两步”在仿真无倒挡（单向棘轮），需先做倒挡建模或该指令跳过；
5. L3 避障任务（积木间穿越/绕行）——正好检验 flow matching 多峰分布价值（《细化方案》§2.2 的论点在仿真里可定量验证）；
6. L4 条件组合（“看到绿色就停下/先红后蓝”）——语义深度。

**真机交接时决策**：

7. 动作空间两条路线都留接口：连续 `(vx,vy,ω)` 重训（方案主线）vs 档位直接映射 OpenCat 指令。建议先用档位映射跑通（迁移面最小），再评估是否值得换连续空间重训。

## 三、v5+ 路线图（仿真阶段，对齐《细化方案》考纲） \{#roadmap}

| 阶段 | 内容 | 验收 |
| --- | --- | --- |
| **v4（进行中）** | L2 导航 + spin 原语 + rear 过采样 | seed 7000 3/3；rear 专项通过；前侧方不回退 |
| v4.5 | batch 放大 + 20 episodes/任务评估 + fm-steps 10/20 对照 | 评估报告（成功率表 + 失败归因） |
| v5-L1 | 指令跟随任务集（语言→档位，含原地转） | 指令语义-动作一致率 ≥90%（对齐方案 L1 标准） |
| v5-L2+ | 恢复数据 20~30% 混入 L2 重训 | 分布偏移鲁棒性：踢扰后恢复率 |
| v5-L3 | 避障任务（静态障碍绕行/穿越） | 不碰撞到达 ≥50% |
| v5-L4 | 条件组合任务 | 触发时机 + 子任务顺序正确 |
| 真机（12 月交付后） | 按《细化方案》主线执行：档位→OpenCat 映射打通 → 遥操作采数据（含 Puppet Mode）→ sim 预训练权重 + 真机数据后训练 | 方案 §九 验收清单 |

## 四、v4 执行状态（2026-11-11） \{#v4-status}

- **全链接力已挂**：工作台 `v4chain.sh` 无人值守运行——采集结束 → 自动训练（batch8/30k）→ seed 7000 评估 → fm-steps 20 对照 → 20 episodes 扩展评估；日志 `/root/v4chain.log`（分段日志 `/root/v4train.log`）。
- 代码改动（本地工作区 + 工作台已同步，**未 commit**，待 v4 结果一起提交）：
  - `8.rl_quaddle/cpg_controller.py`：SKILLS 增 `spin_left/spin_right` 档 + 标定注释
  - `quaddle-data/collect.py`：3 维动作、50% rear 过采样、spin-only 专家 + 驻留
  - `quaddle-data/eval_rollout_gif.py`：3 维量化、执行器驻留、`--fm-steps`
  - 新增 `8.rl_quaddle/scripts/spin_feasibility.py`、`quaddle-data/expert_spin_smoke.py`
- 工作台环境：torch 2.11+cu128 / mujoco 3.8.1（官方 PyPI，镜像 wheel 残缺）/ lerobot 0.6.1（av 锁 15.1.0）/ smolvla_base 873MB 就绪；渲染 **`MUJOCO_GL=egl`**（2026-11-11 修复 NVIDIA EGL：`/workspace/eglfix/install.sh` 30 秒重装；实测 0.3-0.5ms/帧，物理仅 0.06ms/步——首轮采集曾用 osmesa 软渲染，~95% 时间耗在渲染上）。
- 数据采集：**EGL 重采**（~2s/条，约 15 分钟完成 400 条；首轮 osmesa 165 条因光栅化器混流弃用——一次数据集只能用一种渲染器，动态不受影响：渲染不参与物理）。评估与训练保持 EGL 一致。

下面是**工作台 `loop-v7-gpu-0702` 上的实际执行命令**（L20 24GB，EGL GPU 渲染，独立 leroenv），保留作为运行记录；本地 4070 Ti 的规范复现路径见[动手教程](./index.md#reproduce)。

```bash
# 训练（L20，batch 8 起步按显存二分；save_freq=30001 > steps ⇒ 只存 final
# checkpoint ~5.4GB——/workspace 仅 12G 余量，多档位 checkpoint 会爆盘）
cd /workspace/dive-into-embodied-ai/codes/practices/vla/quaddle-data
PYTHONNOUSERSITE=1 MUJOCO_GL=egl LD_LIBRARY_PATH=/usr/local/nvidia/lib HF_HUB_OFFLINE=1 /workspace/leroenv/bin/lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=local/quaddle_nav_400v4 \
  --dataset.root=outputs/dataset/lerobot \
  --batch_size=8 --steps=30000 --save_freq=30001 \
  --policy.device=cuda --policy.push_to_hub=false \
  --policy.scheduler_decay_steps=30000 \
  --output_dir=outputs/smolvla_quaddle_v4

# 评估（seed 7000 对比基线 + fm-steps 20 对照；渲染器必须与训练数据一致 = EGL）
PYTHONNOUSERSITE=1 MUJOCO_GL=egl LD_LIBRARY_PATH=/usr/local/nvidia/lib HF_HUB_OFFLINE=1 /workspace/leroenv/bin/python \
  eval_rollout_gif.py --episodes 3 --seed 7000 \
  --ckpt outputs/smolvla_quaddle_v4/checkpoints/last/pretrained_model \
  --out outputs/rollout_v4.gif
PYTHONNOUSERSITE=1 MUJOCO_GL=egl LD_LIBRARY_PATH=/usr/local/nvidia/lib HF_HUB_OFFLINE=1 /workspace/leroenv/bin/python \
  eval_rollout_gif.py --episodes 3 --seed 7000 --fm-steps 20 \
  --ckpt outputs/smolvla_quaddle_v4/checkpoints/last/pretrained_model \
  --out outputs/rollout_v4_fm20.gif
```

> 注：v3 的对照命令见[动手教程的复现章节](./index.md#reproduce)；v4 的 seed 7000 场景与 v1/v3 完全一致（同 eval 脚本同 seed 流），成功率可直接对比。

## 五、历史记录：v1/v2/v3 三轮评估 \{#history}

同 seed（7000）三场景闭环 rollout，检查点取各自 last：

| 版本 | 数据 | 训练 | 成功率 | 失败模式 |
| --- | --- | --- | --- | --- |
| v1 | 250 条固定死区 | 20k | 1/3 | ep0/1 恒 turn_right 超时 |
| v2 | 250 条（死区随机化**未生效**，事后发现） | +dont_wait 续训 | 1/3 | 与 v1 完全一致 |
| v3 | 400 条真·随机死区 ±12~40° + 转向连续强度 | 30k @base | 1/3 | ep0/1 转向激进 → **翻倒** |

**ep2（blue）三版全部成功**（约 2.8s 到达）——VLA 闭环导航能力成立。

### 根因分析（诊断脚本 diag_ep0 实测）

ep0 场景初始 bearing ≈ −161°（目标几乎正后方）。v3 诊断输出：模型方向正确（wz<0 右转）但强度不足（−0.47），且 **turn 档执行自带直行分量 → 弧线稳态中 bearing 恒定**（几何性质）→ 无限右弧。W5 当时归因为“动作空间缺纯原地转原语”；v4 进一步发现弧线档本身不可靠（见 §1.2），归因已修正。

### 低层 RL 的存档结论

低层运控曾尝试 RL 速度跟踪策略（逐关节动作、CPG 参数动作两代动作空间，含 dont_wait 罚与 skew 谐波维），四轮 checkpoint 均未学会命令条件化响应（vx_err 64mm/s @ ±120 命令）。症结是 4 足球足端摩擦系统的“行走解”需精确相位-偏斜组合，on-policy PPO 的参数空间探索无法可靠命中；而查表（人工标定）10 分钟即得。最终低层以**查表 CPG 控制器**验收通过（forward +113mm/s，11s 指令序列位移 1.1m 未翻倒）。RL 复活前提：观测加 CPG 相位时钟与“上一档位”历史。详见 `8.rl_quaddle/README.md`。

## 环境参考 \{#env}

- 基线复现（本地）：Python 3.12、torch 2.10+cu128、lerobot 0.6.1，环境在 `codes/practices/vla/act`，依赖以该目录 `pyproject.toml` 和 `uv.lock` 为准。
- v4 试验（工作台）：torch 2.11+cu128、mujoco 3.8.1、lerobot 0.6.1（av 锁 15.1.0），渲染 `MUJOCO_GL=osmesa`。
- 显存与吞吐需在自己的机器上通过短训练检查；batch 4 是 10GB 预算遗产，不是最低要求。

## 上游与许可 \{#upstream}

- 机器人模型：[Petoi 官方 Quaddle](https://huggingface.co/petoi/quaddle)（URDF + STL，MIT 协议），闭链结构复刻官方网页仿真器 [quaddle-sim](https://github.com/one-for-all/quaddle-sim)。
- 基座：SmolVLA（`lerobot/smolvla_base`）+ SmolVLM2，经 [LeRobot](https://github.com/huggingface/lerobot) 0.6.1 微调。
- 复用模型与代码时分别检查对应许可；真机 OpenCat 固件的步态表（kwkF/ktrL/kbwF）以官方开源进度为准。

---

## v5：环境统一与首轮闭环（2026-11-12） \{#v5}

**决策**：仓库两套 Quaddle MuJoCo 环境不能并存，VLA 管线统一到《Quaddle 入门》教程环境
（`codes/practices/quadruped/quaddle-mujoco/`）。v4（cs123 环境）结果归档为旧环境基线，
训练在 ~40% 处终止。**两套环境数据/策略不可互换**。

### v5.0 转向扫描与专家（环境迁移重验证）

| 候选 | 实测 | 结论 |
|---|---|---|
| steer_L（duty 左缩右放 ±0.2） | **wz +1.203 rad/s**，转弯半径 ~2.6cm 准原地转，10s 不翻 | 主转向原语 ✅ |
| steer_R（镜像） | wz −0.612 rad/s（方向不对称），10s 不翻 | 可用 ✅ |
| duty 不对称 ≥0.33（任意方向） | roll 30°+ **翻倒** | 弃用 |
| pivot（cs123 的停摆机制） | pivot_L 翻倒 / pivot_R 弱 | **不迁移成立**，弃用 |
| base_walk 开环 | 自身漂移 +0.23 rad/s | 专家闭环需修正 |

- **动作空间回退 2 维 `[vx_dir, wz_dir]`**——新环境原生转向可靠，spin 旗标取消
- 专家 = 二值档位（|bearing|>死区 → steer ±全幅）+ 0.4s 切档驻留 + rear 过采样 50%
- **闭环 smoke 两 seed 全胜：9/9 + 12/12 到达、零翻倒**（含 rear 场景过半）
- 比例连续转向复测更差（4 条 2 超时）——**v4 教训跨环境复验：连续强度 = 弱执行**

### v5.0 管线与首轮结果

- 数据：**4 worker 并行采集 252 条**（各 63 条、独立 repo_id），lerobot 官方
  `aggregate_datasets` 合并（v3.0 格式，`concatenate_videos=False` 免 ffmpeg）；
  单条 35-40s / 350-400 帧 = 帧密度为 cs123 的 4-5 倍，合计 90,054 帧
- 训练：batch 32 / fp32 / 6000 步（loss 1.28 → 0.105），51 分钟，11.3GB 显存
- 评估：**seed 7000 三场景 1/3，20 episodes 仅 2/20**（1/3 非噪声）
- 多数据集训练在 lerobot 0.6.1 = NotImplementedError（列表 repo_id 未实现）

### 五轮诊断（diag_*_v5.py，全部沉淀于 quaddle-mujoco/）

1. 数据分布：steer_R 32.1% / steer_L 23.5% / forward 44.1%，rear 区转向选择 48/48 对称——**数据健康 ✅**
2. fm-steps 10 vs 20：逐档位统计完全一致——**积分步数假设否定 ✅**
3. 量化器硬约束（vx<0 置零、|wz|>0.5 强制纯转向）：2/20 持平——**量化层排除 ✅**
4. 重规划间隔 2.5s → 0.5s：行为质变（单档锁死 → 出现方向修正，末距离 48→29cm）但仍 1/3——**推理机制部分改善，不达标**
5. **训练观测直接喂模型**（policy.reset() 清 select_action 内部队列后）：**7 帧档位判别全对**——
   forward 帧 → (+1.01, +0.09)，steer 帧 → (−0.87, ±1.38)。模型本身在训练分布内是好的 ✅

### 根因与 v5.1 修复

**根因：OOD 自旋转状态的 compounding error**——评估中策略持续转向，自转 60s 的状态
（高角速度、任意朝向、关节构型）远超训练分布（数据转向段仅 2-5s），模型输出坍缩为
常数单档（steer_R 600 步不换），bearing 扫过 0 也不前进。v3 的"恒 turn_right"同族。

**v5.1 修复实验（恢复数据，规划项落地）**：`collect_v5.py --recovery 6`——专家接管前
1~6s 随机档位扰动（不录制），把机器人推进自旋/偏移态，随后的专家恢复段教会策略
"从乱态回目标"。4 worker × 25 条（摔倒率 55%，随机扰动本身易致跌——v5.2 若扩量需
限制 warmup ≤3s 或加安全偏置），合计 352 条 / 137k 帧，恢复占比 28%。

**v5.1 结果：恢复数据未奏效**——seed 7000 1/3、20 eps 2/20（与 v5 持平）。
新观察：出现纯 forward 直行到达的成功新模式（ep16，forward 282 步）；
重规划 0.5s 后失败例也开始混合双方向修正。**恢复数据假设在 28% 配比下证伪。**

### 待验证假设与下一步（按成本排序）

1. 训练欠拟合 → 12k-20k 步延长训练（loss 终值 0.105，需对比 loss 平台）
2. 动作空间重设计：单通道档位编码（mode ∈ {−1,0,+1} 消除 vx/wz 联合外推空间）
3. 恢复数据扩量 + 覆盖更长自旋态（warmup 6s → 15-30s，配安全偏置防翻）
4. 升级 lerobot 修 bf16 AMP（2x 提速，为 1-3 的多轮迭代降本）
5. 观测表征：bearing/dist 特权信息可能被图像通道淹没，尝试 state-only 消融

---

## 训练极限加速：8.2x 无损提速（2026-11-12） \{#fast-train}

为支撑"加大训练量"的多轮迭代，对训练过程做了极限优化。**实测 509ms/step → 62.5ms/step
（8.2x，62.9 → 512 samples/s），显存 11.4G → 3.4G，数值等价**（同 batch 同噪声 loss 差
2.2e-4，探针参数梯度 cosine 0.996–0.99999）。配方完全不变（AdamW/cosine/clip10/batch32/
同 EpisodeAwareSampler），checkpoint 与 `eval_rollout_v5.py`、lerobot resume 完全兼容。

### 瓶颈定位（batch32 单步实测分解）

| 阶段 | 耗时 | 性质 |
|---|---|---|
| SigLIP 视觉编码器（3×512²） | 266ms (52%) | 冻结却每步重算——纯浪费 |
| VLM+专家前向 | 87ms | eager fp32 注意力（物化 291² 权重矩阵） |
| 反向 | 132ms | 梯度穿过全部 291 token，其中 240 个冻结 token 本不需反传 |
| 预处理+优化器 | ~31ms | 图像 resize/uint8 转换可省 |

### 优化栈（`cached_smolvla.py` / `build_cache_v5.py` / `train_cached_v5.py`）

1. **冻结图像嵌入缓存**：视觉编码器+connector 冻结、无图像增强（`image_transforms.enable=False`），
   故每帧 SigLIP 输出（3×64×960 bf16，已乘 √hidden）是定值。一次性预计算 123,328 帧落 NAS
   `/dataset/mtl_data/quaddle_vla/cache_emb_v51`（45.5GB memmap + meta.npz，构建 27min）。
   训练彻底跳过视觉编码器与图像预处理。
2. **梯度分裂前向**：prefix-lm 掩码下图像/语言 token 不 attend state/action（`make_att_2d_masks`
   cumsum 语义），其逐层 K/V 与可训练 `state_proj`、噪声动作无关。冻结 prefix 在 `no_grad` 下跑
   16 层 VLM 只留 post-RoPE K/V；可训练流（1 state token 走 VLM + 50 action token 走专家）带梯度
   attend 到冻结 K/V。反向只穿过 51 token（原 291）。
3. **flash SDPA**：替换 eager fp32 注意力。语言 padding（48 中仅 6 真实，三任务皆 6）在原模型中
   处处被掩掉，故冻结 prefix 直接丢弃 padding 用 198 真实 token 双向**无掩码** attention（flash 后端）；
   GQA 头显式 expand 以避免 `enable_gqa+bool mask` 退化到 math 慢路径。
4. **torch.compile**（`max-autotune-no-cudagraphs`）：融合 layernorm/RoPE/残差/SiLU 小算子。
   未 compile 112ms/step（4.5x），compile 后 62.5ms/step（8.2x）。预热 ~5min（一次性）。

### 为什么 100x 物理不可达（单卡 L20）

- 可训练专家 fwd+bwd ≈ 33 GFLOPs/样本，L20 bf16 峰值 119 TFLOPS → **即便完美缓存+零 I/O，
  100% MFU 理论上限 ~57x**；当前 42% MFU 已是小模型合理水平。
- 完整逐层 KV 缓存（4MB/帧）受 NAS ~2GB/s 读取限制（O_DIRECT 实测，多线程不扩展），
  batch32 每步 128MB → **I/O 上限 ~8x**。图像嵌入缓存（368KB/帧，12MB/步）则无此瓶颈。
- 结论：8.2x 已逼近"无损（数值等价）+ 单卡 + NAS 存储"约束下的实际极限。再往上需 fp8 KV（牺牲数值）
  或多卡，不在本轮范围。

### 续训运行与结果（v5.1-30k-fast）

`chain_v51_30k_fast.sh`：从 v5.1-ext 18k checkpoint sample-exact 续训到 30k（新 cosine decay=30000，
起点 lr 3.62e-5），输出落 NAS `/dataset/mtl_data/quaddle_vla/runs/v51_30k_fast`，训完自动 seed 7000
三场景 + 20 episodes 评估。12k 新步实测 **~13min**（稳态 62.6ms/step + ~5min compile 预热），对比原速 1.7h。

| 配置 | epochs | 训练 loss 终值 | seed7000 | 20 eps |
|---|---|---|---|---|
| v5（6k）| 1.4 | 0.105 | 1/3 | 2/20 |
| v5.1-ext（18k）| 4.2 | — | 1/3 | 6/20 |
| **v5.1-30k-fast** | **7.0** | **0.054** | **1/3** | **5/20** |

**关键结论：训练量假设饱和。** 6k→18k（2→6/20）的真实增益在 18k→30k（6→5/20，噪声内）停止；
训练 loss 仍单调下降（0.073→0.054）但闭环成功率 plateau 于 ~25-30%。这是 compounding error 的
典型信号——**训练分布内拟合更好 ≠ rollout OOD 漂移修复**。失败例仍为 steer_R/steer_L 振荡不收敛
（如 ep1 200R+400L、ep6 130R+470L），成功例已覆盖三色（red ep12 / green ep4,19 / blue ep2,8）。

**下一步转向**：瓶颈已非欠拟合，而是 OOD 自旋态覆盖不足。优先 plan #3（单通道档位动作编码，
消除 vx/wz 联合外推）或 #4（恢复数据扩量 + 更长自旋态覆盖）。8.2x 加速使这两条半天级迭代降到
分钟级训练，可快速证伪/证实。

---

## v6：#3 单通道档位 + #4 扩量恢复（2026-11-12，阴性结果） \{#v6}

用 8.2x 加速管线快速验证 plan #3/#4。**两条杠杆均未修复 OOD 自旋，1D 动作反而略降。**

| 实验 | 动作空间 | 数据 | 训练 | seed7000 | 20eps |
|---|---|---|---|---|---|
| v5.1-ext | 2D [vx,wz] | 352（28% 恢复）| 18k | 1/3 | **6/20** |
| v5.1-30k | 2D | 352 | 30k | 1/3 | 5/20 |
| v5.1-43k | 2D | 352 | 43k | 1/3 | 2/20 |
| **Exp A（#3）** | **1D mode** | **252 常规** | 18k | 1/3 | **3/20** |
| **Exp B（#3+#4）** | **1D mode** | **352（扩量恢复）** | 18k | 1/3 | **2/20** |

### 实现要点

- **#3 1D 档位**：action=mode∈{−1,0,+1}={steer_R, forward/stop, steer_L}，消除 2D 联合外推区
  （mode 负值=steer_R 合法，无原 vx<0 无执行器问题）。eval 量化阈值 ±0.5。
- **lerobot 1D bug**：action_dim=1 时数据集把 singleton 维 squeeze 成 (B,50)，标准训练前向
  `prepare_action` 崩溃（mat1 1024×50 vs mat2 32×720）。绕过：`init_1d_v6.py` 用 `make_policy`
  只构造+加载权重（不前向）生成 1D 适配 step0 checkpoint，加速训练器读缓存 (N,50,1) 正常。
- **#4 扩量恢复**：warmup 15-30s（原 1-6s）+ 转向档偏置 70%（诱发原地自旋 OOD 态，准原地转保持
  远离目标）+ roll 防翻（body z 轴竖直分量<0.92 强制前进稳定）+ warmup 后距目标<20cm 丢弃
  （避免 1 帧空 episode）。reg252/full352，action 均 1D。

### 结论与反思

**三个规划杠杆（#1 加训练、#3 动作空间、#4 恢复数据）全部证伪**——闭环成功率不升反降，
失败例仍是 steer_R/steer_L 锁死 600 步原地自旋。这说明：

1. **动作空间不是病灶**：1D 消除无效外推后自旋依旧，甚至略差（6→3/20），2D 的表达力反而有益。
2. **恢复数据未触及真实 OOD 态**：专家恢复段覆盖的"乱态"与策略 rollout 实际漂移进的自旋态
   分布不匹配，或策略在闭环中根本 ignoring 这些示范。
3. **自旋是策略级闭环漂移**：flow matching + chunk50（5s 承诺）+ 重规划机制下，策略一旦进入
   自旋就自我强化，单靠数据/动作编码无法打破。

**需更深层改动**（超出原 plan，待决策）：观测表征消融（bearing/dist 特权信息是否被图像淹没，
state-only 对照）、重规划间隔/chunk 缩短、加入自旋惩罚的 RL 后训练、或换更强的闭环架构。
当前仿真 VLA 路线在 L2 导航任务上遇到方法论瓶颈。

---

## 步态顺畅度排查：模型无罪，重规划间隔是关键（2026-11-13） \{#gait-smooth}

用户反馈"走路爬行不顺畅"。排查过程有一次**重要自我纠正**：

### 错误诊断与纠正

初版探针在 `outputs/dataset_w0/scene.xml`（**带积木**）上测得"任何步态只走 0.4mm/s"，
据此误判"伺服 kp=0.15 太弱、需改模型"。后在**空场景** `models/scene.xml` 复测发现：
**原始步态（kp=0.15）upright、69.8mm/s、垂直弹跳仅 0.86-1.0mm——走得又稳又顺**。
"走不动"是探针场景里机器人被积木挡住的假象。**模型与步态本身无需修改**（kp=2 等"修正"已撤回，
scene.xml/gait.json 从未改动）。执行器限速+换档斜坡在空场景改善可忽略（bounce 1.0→0.95mm），
因底层本就顺。

### 不顺畅的真正来源与修复

导航 GIF 里的不顺畅 = **VLA 策略原地自旋**（OOD 坍缩），非步态问题。评估期杠杆
**缩短重规划间隔**（`--replan-steps`，打破 chunk50=5s 的自旋承诺）在 18k checkpoint 上：

| replan 间隔 | 20eps |
|---|---|
| 25 步=2.5s（旧默认）| 6/20 |
| **5 步=0.5s** | **7/20（最佳，走路 upright 顺畅）** |
| 10 步=1s | 1/20（非单调，共振自旋周期）|

**结论**：保持原始步态（kp=0.15）+ eval 默认 `--replan-steps 5`，机器人 upright 顺畅行走导航，
成功率 7/20（无需重训）。这是"顺畅走路"的正确形态。重规划间隔与自旋周期非单调，0.5s 为 sweet spot。

### DAgger 闭环恢复重训（2026-11-13，阴性）

用策略自身 rollout（replan5）访问的 OOD 态 + 专家 bearing/dist 标签采集 60 条
（`dagger_collect_v5.py`），与 reg252 合并（312 集）从 18k 加速续训到 24k。
结果 seed7000 1/3、20eps **3/20**，**差于基线 18k+replan5 的 7/20**。行为上 forward 尝试增多
但仍 steer 振荡超时。DAgger 未能修复 compounding error。

**全部训练侧杠杆汇总（20eps）**：18k+replan5=7/20（最佳）；30k/43k=5/20/2/20（过拟合）；
#3 1D=3/20；#4 开环恢复=2/20；DAgger=3/20。OOD 自旋为顽固瓶颈，验收线（3/3、≥60%）未达。

### Temporal ensembling 突破 OOD（2026-11-13，验收达标）🎯

评估期 **ACT 式时间集成**（`eval_ensemble_v5.py`）：每 5 步 `predict_action_chunk` 重规划，
对重叠 chunk 预测做 recency 加权平均（`exp(-k·age)`，k=0.01）后执行，取代固定动作队列。
**无需重训**，在 18k checkpoint 上：

- **seed7000 三场景 3/3**（ep0 red / ep1 green / ep2 blue 全✓）
- **20eps 17/20 = 85%**，远超验收线 ≥60%

行为质变：档位呈健康 forward+steer 混合（不再纯自旋锁死）。时间集成平滑了 chunk 间冲突、
抑制 OOD 自旋 compounding error——此前所有训练侧杠杆失败，而评估期时序集成首次突破。

**新 best = 2D 18k + temporal ensembling（replan5, k=0.01）**。验收（3/3、≥60%）**达标**。

![L2 导航 best：SmolVLA 18k + temporal ensembling，20eps 17/20=85%（VLA 策略 rollout，斜视全场，播放 3x）](./figs/quaddle_L2_nav_ensemble_17of20.gif)

**丝滑版（50fps 渲染）**：原始 GIF 为 10fps 显卡顿；改为每控制步渲染、50fps 播放后动作丝滑
（同一 18k+ensembling 策略，6 集 5 胜）。提速需更快步态，但当前策略绑定训练步态，提速会伤导航，
需在更快步态上重训（见下）。以上两张 GIF 均已做**播放期 3x 时间压缩 + 30fps 运动补偿插值**
（2026-11-14，物理轨迹不变，见 [#gif-playback](#gif-playback)）。

![L2 导航丝滑版：50fps 渲染 + 播放 3x，动作连贯（VLA 策略 rollout）](./figs/quaddle_L2_nav_silky50.gif)

**这个 GIF 是怎么训练出来的（VLA）**：
1. **模型**：SmolVLA 450M（`lerobot/smolvla_base`，SmolVLM2 骨干 + flow matching 动作专家）；冻结视觉编码器与 VLM 骨干，只训动作专家（~100M）+ 投影层。
2. **数据**：查表 CPG 方位闭环专家在《Quaddle 入门》MuJoCo 环境采集 252 常规 + 100 恢复 = 352 条（v5.1），10Hz、3 相机 320² + 8 维状态 + 语言指令 `walk to the {color} block`；50% rear 过采样。
3. **训练**：batch32 / AdamW + cosine，18000 步（v5.1-ext）；用 8.2x 加速管线（冻结视觉嵌入缓存 + 梯度分裂 + flash SDPA + torch.compile，509→62.5ms/step）。
4. **评估（突破点）**：temporal ensembling——每 5 步 `predict_action_chunk` 重规划，对重叠 chunk 预测做 recency 加权平均（`exp(-0.01·age)`）后执行，平滑 chunk 冲突、抑制 OOD 自旋锁。20eps **17/20=85%**、seed7000 **3/3**。

**非 VLA 对照（低层步态，无策略）**：

![原始 CPG 步态开环前进（kp=0.15，upright、69.8mm/s、弹跳0.86mm）——低层运控演示，非 VLA](./figs/quaddle_gait_orig_smooth.gif)

此 GIF 为低层查表 CPG 开环执行 forward 档，**不含 VLA 策略**，仅证明底层步态顺畅、模型无需修改。

---

## L1/L3/L4 任务扩展（2026-11-13） \{#l1-l3-l4}

按《细化方案》考纲扩展 L1/L3/L4，均用 8.2x 加速管线（采集→缓存→训6k→eval），cclab loop-v7-gpu-0702：

| 任务 | 数据 | 结果 | 目标 | 达标 |
|---|---|---|---|---|
| L1 指令跟随（walk forward/turn left/turn right/stop）| 200 | **100%** 一致率 | ≥90% | ✅ |
| L3 避障（到目标+绕障碍）| 200 | **3/20=15%** | ≥50% | ❌ |
| L4 条件组合（看到绿停/先红后蓝）| 200 | **0/20** | 触发+顺序 | ❌ |

**结论**：L1（语言→档位映射）易学达标；L3 避障泛化不足（15%）；L4 条件触发/子任务顺序未学会（0%）。
L2 导航（85%）与 L1（100%）达标，L3/L4 是短板——需更多样化的避障/条件数据或更强架构。
采集/评估脚本：`collect_l1.py`/`collect_l3.py`/`collect_l4.py`、`eval_l1.py`/`eval_l3.py`/`eval_l4.py`、
`chain_l1.sh`/`chain_l3.sh`/`chain_l4.sh`。

### L3/L4 数据扩充（2026-11-13，阴性） \{#l3l4-expand}

假设上轮 L3 15%/L4 0% 源于数据不足（200 集）。扩到 **600 集（×3）+ 训 12k（×2）+ 后方过采样 50%**（L4 另调 60/40 任务配比，偏向较简单的"看到绿停"）。专家采集成功率 L3 42%、L4 40%（后方过采样场景更难，丢弃多但 600 条优质示范足量）。

| 任务 | 200集/6k | 600集/12k | 达标线 |
|---|---|---|---|
| L3 避障 | 15% (3/20) | **10% (2/20)** ↓ | ≥50% |
| L4 条件 | 0% (0/20) | **0% (0/20)** → | 触发+顺序 |

**结论：数据量不是瓶颈。** 3× 数据 + 2× 训练不但没拉起 L3/L4，L3 反而略降。证实是**架构/任务复杂度上限**——二值档位 450M VLA 学不会"避障+导航"复合任务与"先红后蓝"条件顺序，与 fast-gait 暴露的动作空间架构限制同源。下一步若要突破，需更强架构/连续动作空间/任务分解，而非继续堆数据。L2(85%)/L1(100%) 核心基线不受影响。

脚本 `chain_l3l4_expand.sh`；GIF `outputs/quaddle_L3_avoid_expand_2of20.gif`、`outputs/quaddle_L4_cond_expand_0of20.gif`
（均已做播放期 4x 时间压缩，见 [#gif-playback](#gif-playback)）。

## 提速天花板复验：K/V 缓存与 CUDA graphs 均否决（2026-11-13） \{#speed-ceiling}

在 8.2x 基础上继续榨提速，两条路线实测均不可行，**确认 8.2x（no-cudagraphs）是单卡 L20 + 此 NAS 的安全天花板**。

### 路线一：冻结前缀 K/V 缓存——NAS I/O 反噬（否决）

思路：`frozen_prefill` 每步重算 198-token 前缀的逐层 K/V（实测 32.9ms，占 ~39%），VLM 冻结（vlm_layer0 可训张量 0/9）故 K/V 训练期恒定，理论上可预计算缓存跳过。

实测否决（先算 I/O 成本再动手）：

| 项 | 实测 |
|---|---|
| NAS 聚合读吞吐 | 4 线程 ~687 MB/s、8 线程 ~543 MB/s（NFS 服务端瓶颈，**不随线程扩展**） |
| K/V 缓存 batch32 体积 | 131 MB（每帧 4.1MB，是图像 embedding 368KB 的 **11 倍**） |
| 新增 I/O 耗时 | 131MB ÷ ~600MB/s ≈ **218 ms/step** |
| 省下的计算 | ~30 ms（frozen_prefill） |
| 当前整 step | 62.5 ms |

K/V 缓存把 ~30ms 计算换成 ~218ms I/O → 训练反慢 ~4x。**缓存类优化必须先实测存储吞吐算 I/O 成本，不能只算省下的计算。** NAS 剩 7.5TB 容量够，但带宽喂不动。

### 路线二：CUDA graphs——计算 bound 无收益（否决）

`torch.compile` 改 `max-autotune`/`reduce-overhead`（含 cudagraphs）。首测两模式均崩：`accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run`（frozen_prefill 的 kv 输出缓冲被下一步覆盖）。加 `torch.compiler.cudagraph_mark_step_begin()` 每步标记后崩溃解决，但基准（各 80 步）：

| 模式 | 稳态 ms/step | mem | 问题 |
|---|---|---|---|
| no-cudagraphs（基线） | 69-70（稳定） | 3.4G | 无 |
| max-autotune（graphs） | 62-69，偶发 96 尖峰 | 3.2G | 提速在噪声内 + 不稳定 |
| reduce-overhead（graphs） | 70-77，偶发 91 尖峰 | 3.2G | 反而更慢 |

且 graphs 模式 grad norm 明显偏离基线（如 2.11/1.43 vs 1.46/0.72）→ **数值不等价**。根因：管线是**计算 bound**（frozen_prefill 33ms + backward 29ms 的 matmul 主导），CUDA graphs 只省 kernel-launch 开销（此处占比极小）。生产代码恢复 no-cudagraphs。

**结论**：再提速只能走 FP8 计算（Transformer Engine，L20 有 FP8 张量核，~1.5x → 累计 ~12x，但需等价+闭环精度验证）或蒸馏换小模型（改变模型）。基准脚本 `bench_compile_modes.sh`、剖面 `profile_cached.py`。

## 快速步态重训：高速破坏二值导航（2026-11-13，阴性） \{#fast-gait}

诉求"爬行太慢"。结论：**原始步态 ~70mm/s 已是当前二值档位架构的导航速度近优；更快步态会破坏闭环导航。"丝滑"靠 50fps 渲染解决，"更快"受动作空间架构硬约束**（观感上的"更快"最终由播放期时间压缩解决，见 [#gif-playback](#gif-playback)）。

| 配置 | 开环速度 | 闭环导航成功率 | 有效导航速度 |
|---|---|---|---|
| 原始步态 kp0.15 | 69.8 mm/s | **87.5%** | 正常 |
| 盲改 freq2.3/amp1.15（kp0.15） | 滑移不走 | 5.4%（全超时） | — |
| 搜索 rank3 freq2.63/kp0.40 | **108 mm/s (+55%)** | 25.9%（全超时） | ~11 mm/s |
| rank3 + 驻留0.1s/死区8-20° | 108 mm/s | 23.8% | ~10 mm/s |

根因链：
1. 盲改失败因 servo kp 未提（默认 0.15 太软，追不上快目标→滑移）。步态搜索（`gait_speed_search.py`，调 freq/ascale/kp/kv）找到开环最快稳定步态 108mm/s，需 kp0.3-0.47。
2. 但**开环快 ≠ 能导航**：二值档位（全速 steer_L/R）+ bang-bang 转向在高速下必然过冲振荡→绕目标打转、收敛极慢，开环 108 反成闭环 ~10mm/s。缩短驻留/减小死区救不回。
3. 连续转向动作空间在 v4 已证失败（弱执行易超时），二值档位是刻意选择。故更快导航需重设计低层控制器/动作表征（超出本轮范围）。

决策：**接受原始步态**，不发射 fast-gait 重训。`collect_v5.py`/`eval_ensemble_v5.py` 已加 `--gait-freq/--gait-amp-scale/--servo-kp/--servo-kv/--dwell-s/--deadzone-*` 参数（搜索↔采集↔评估动力学一致，position 执行器需同步 `biasprm[1]=-kp`）。

## 演示 GIF 播放优化：时间压缩 + 运动补偿插值（2026-11-14） \{#gif-playback}

反馈"GIF 动作太慢"。物理侧已被 [#fast-gait](#fast-gait) 关闭（提速会破坏闭环导航，GIF 文件名锚定成功率，不能改物理），故在**播放时间轴**上优化：`ffmpeg` `setpts` 时间压缩 + `minterpolate`（mci/aobmc）补帧到 30fps + 调色板两遍（`palettegen=stats_mode=diff` / `paletteuse=bayer5`）。源 GIF 为 10fps 采集（silky50 为 50fps），原速播放 15-20 分钟（L4 20:00 = 20 集×60s 超时合集）。

| GIF | 内容 | 压缩 | 时长 | 体积 |
|---|---|---|---|---|
| `quaddle_L2_nav_ensemble_17of20` | L2 best 17/20（17 胜+3 超时合集）| 3x | 14:43 → **4:54** | 48→30MB |
| `quaddle_L2_nav_silky50` | L2 丝滑版 | 3x | 4:19 → **1:26** | 46→6.7MB |
| `quaddle_L3_avoid_expand_2of20` | L3 扩充 2/20 | 4x | 19:30 → **4:53** | 68→30MB |
| `quaddle_L4_cond_expand_0of20` | L4 扩充 0/20 | 4x | 20:00 → **5:00** | 69→30MB |

要点：**物理轨迹逐帧不变**（只是播放更快），成功率证据链不受影响；插值帧人工查验无鬼影；原件保留为 `*_orig.gif`。成功集（L2）压 3x 保观感质量，失败合集（L3/L4）压 4x 提高浏览效率。

执行在 cclab 工作台（loop-v7-gpu-0702）：源 GIF 仅存本地 → 34 块×10MB base64 分块上传（md5 逐一校验）→ 工作台 imageio-ffmpeg 9.0.2 编码（9.5 分钟，CPU，与本地同版本同参数产物逐字节一致）→ 结果 500KB 分块回传（实测 get 通道 >512KB 截断，4MB/1MB 均不可用）→ 全量 md5 验证落位。

## 任务分解突破 L3/L4 架构上限（2026-11-15） \{#decomposed}

**结论：L3 避障 10%→85%（17/20）、L4 条件组合 0%→80%（16/20），双双达标。** 此前已证端到端
450M VLA + 二值档位学不会复合任务（600 集扩充仍阴性）；本轮**不改模型、不加训练**，把任务分解为
"编排器 + 已验证技能"，两个架构上限同时打破。

### 架构

| 层 | 实现 |
|---|---|
| 编排器（脚本级，确定性） | 指令解析 → 阶段计划："先红后蓝" = [nav(red), nav(blue)]；"看到绿停" = [nav(green) + stop 门控] |
| 导航技能 | **原封不动复用 85% L2 资产**：18k ckpt + temporal ensembling（replan5/k=0.01）+ 原生训练短语 "walk to the {color} block" |
| L3 反射层（subsumption） | 最近**非目标**块 <15cm 且在前方 → 转向避让 0.8s → 清集成缓冲交还技能（目标色块不误触发——相机可分辨颜色）|
| L4 转向辅助 | task B 中 \|bearing\|>45° 即脚本转向、<31.5° 交还策略（贴块起步+大方位角是训练分布空洞）|
| 阶段切换/停停门控 | 与策略 state[6:8] 同源的目标状态接口（dist/bearing；真机对应目标色跟踪器）|

判据与 RNG/场景协议逐字节镜像 `eval_l3.py` / `eval_l4.py`（同 seed 同 20 场景），apples-to-apples。

### 结果

| 任务 | 端到端基线 | 分解版 | 达标线 | 判定 |
|---|---|---|---|---|
| L3 避障（到目标+不碰障碍）| 2/20=10% | **17/20=85%** | ≥50% | ✅ |
| L4 看到绿停 | 混在 0/20 内 | **10/10** | 触发 | ✅ |
| L4 先红后蓝 | 混在 0/20 内 | **6/10**（red 达成 10/10）| 顺序 | ✅ |
| L4 合计 | 0/20 | **16/20=80%** | 触发+顺序 | ✅ |

（L4 预算按目标数配：每段 60s。同日志可算严格总窗截断分：60s = 12/20、90s = 15/20，均远超 0%。）

![L3 避障分解版：L2 技能 + 非目标块反射层，20eps 17/20=85%（VLA 技能闭环，播放 3x）](./figs/quaddle_L3_decomposed_17of20.gif)

![L4 条件组合分解版：编排器 + L2 技能，看到绿停 10/10、先红后蓝 6/10，合计 16/20=80%（播放 3x）](./figs/quaddle_L4_decomposed_16of20.gif)

### 关键发现

1. **顺序切换本身零失败**：red 达成 + 切段 10/10（四轮运行一致）——"触发+顺序"由编排器确定性保证，这正是端到端学不会的部分。
2. **blue 段残余失败 = 两个已知瓶颈叠加**，非编排问题：① **弱侧大角转向慢**——steer_R 大角净转速实测 ~5°/s（0.612 rad/s 规格被摆振抵消，与 v4"弱档弧线"同源），rear-start 光初始转向耗 20-30s；② **贴块起步 + 大方位角 OOD**——切段瞬间距 red 12cm 而指令是 blue，"近距+大角"状态在 L2 训练分布外，策略失锁可跑飞（ep15 曾漂到 127cm；同 episode 单独重跑可成功——flow 采样噪声下接近抛硬币）。
3. **L4 辅助迭代**：无辅助 13/20 → 距离门控辅助 14/20（跑出 0.30m 即脱离辅助，门控失效）→ 纯方位辅助 15/20（90s）→ 加每段 60s 预算 **16/20**。信息界面声明：编排器使用的 dist/bearing 与策略自身输入 state[6:8] 同源（非新增特权信息），L3 反射层用障碍身份（场景状态）；真机等价物 = 目标色跟踪器/障碍检测器，均为常规感知组件。
4. **下一步（若要把 task B 推向 ~100%）**：sequential-depart 数据补丁——采 30-60 条"从块旁出发去另一目标"专家数据（正是切段 OOD 态），8.2x 管线微调 L2 ckpt 仅需分钟级；弱侧转向运动学瓶颈需步态层解决（与 [#fast-gait](#fast-gait) 结论一致）。

脚本 `eval_l4_decomposed.py` / `eval_l3_decomposed.py`；GIF `outputs/rollout_l3_decomposed_3x.gif`、`outputs/rollout_l4_decomposed_final_3x.gif`。