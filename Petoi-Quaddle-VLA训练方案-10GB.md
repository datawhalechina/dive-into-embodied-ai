# Petoi Quaddle：四舵机 MuJoCo 步行仿真 + 10GB 显存 VLA 训练方案

> 姊妹篇：《四足机器人VLA仿真训练方案》（12GB / Isaac Lab / Go2）。
> 本方案针对 **Petoi Quaddle（4 舵机桌面四足）+ MuJoCo + 本地 RTX 4070 Ti 12GB（实际可用约 11.5GB，目标控制在 10GB 内）** 重新设计，结论与 12GB 方案有一处关键不同：**SmolVLA（450M）让 VLA 微调完全可以在本地完成，云端租卡从"必选"降级为"可选项"。**
>
> 调研日期：2026-11-06。不纳入 git 跟踪。

---

## 0. 结论速览

| 问题 | 结论 |
|---|---|
| Quaddle 有官方 MuJoCo 模型吗？ | **没有 MJCF，但有官方 URDF**（HuggingFace `petoi/quaddle`，MIT 协议，46 links / 45 joints）。官方自带的是 Rust/WASM 网页仿真器（非 MuJoCo）。需要自行 URDF→MuJoCo |
| 10GB 显存能训 VLA 吗？ | **能**。SmolVLA 450M 是唯一为消费级显卡设计的 VLA：官方算力指南 batch 8 约 10–16GB，本机 batch 2–4 + bf16 AMP 可压到 **5–10GB**，单轮训练一晚完成 |
| 低层运控怎么做？ | 4 舵机 + 被动连杆，**比 Go2 12DOF 简单一个量级**。复用本仓库 `6.rl_pupper_v2_5` 管线改 4 维动作即可，CPU 就够，不需要 Isaac Lab |
| 需要花钱吗？ | 仿真训练全程本地，0 元。云端 L4/A40（¥2–6/h）仅作显存兜底与加速备选，单轮几十元 |
| 周期估计 | 5 周：W1 仿真环境 → W2 低层策略 → W3 数据管线+冒烟 → W4 正式 VLA 训练 → W5 评测迭代 |

---

## 1. 背景盘点

### 1.1 Quaddle 是什么

[Quaddle](https://www.petoi.com/pages/quaddle-educational-robot-kit) 是 Petoi（OpenCat 生态，Bittle/Nybble 厂商）最新的开源桌面四足：

- **仅 4 个舵机**（每条腿 1 个髋关节舵机），替代常规 8–12 舵机方案；全向移动（前进/横移/原地转），宣传指标 ~1.8 体长/秒
- **复古机械连杆腿**：单个舵机驱动四连杆机构，把旋转转化为"抬膝-落脚"步态（被动连杆，无膝关节电机）
- **滚轮足端**：每条腿末端有 2 个 `continuous` 滚轮关节，实现无额外执行器的全向移动
- **位置反馈舵机**（可读位置），由此支持 **Puppet Mode**：手扶腿部录制动作，无需写代码——这是未来真机数据采集的重要通道
- OpenCat 指令层兼容（`ksit`、`kwkF`、`ktrL`、`m0 <channel> <deg>` 等），预置 90+ 动作
- 2026-09-02 Kickstarter 上线（起价 $99），**2026 年 12 月初交付**；固件源码交付前开源（尚未发布）

### 1.2 现有可复用资产

**官方资产（已公开、可立即使用）：**

| 资产 | 内容 | 用途 |
|---|---|---|
| [HF `petoi/quaddle`](https://huggingface.co/petoi/quaddle) | `robot.urdf` + `assets/*.stl`，MIT 协议。4 个主动舵机关节（`*_thigh`）+ 每腿 3 个被动连杆关节（`motor_arm`/`leg`/`spring`）+ 每腿 2 个滚轮关节（`wheel`/`wheel_tip`），`closing_*` 为连杆闭合 dummy 链 | **MuJoCo 环境的模型来源** |
| [`one-for-all/quaddle-sim`](https://github.com/one-for-all/quaddle-sim)（官方网页仿真器的开发仓） | Rust + WASM；物理用 `gorilla-physics`，且内置 `esp32rs`（ESP32-S3 模拟器）直接跑真实固件；`docs/` 内含同一份 `robot.urdf`、mesh、esp32 ROM | 行为参考基准：`kwkF`/`ktrL` 的步态相位差、速度-指令映射可在此观察 |
| [在线仿真器](https://www.petoi.com/pages/quaddle-robot-dog-simulator) | 浏览器直接操作，支持 WASD、OpenCat 指令、C++ 编码 | 无需装环境即可采集指令-行为对照 |

**本仓库资产（已验证可跑）：**

| 路径 | 内容 | 本方案的复用点 |
|---|---|---|
| `codes/practices/quadruped/cs123/6.rl_pupper_v2_5/` | Pupper 12DOF 速度条件 PPO（SB3 + MuJoCo CPU，48/53 维观测，奖励工程完整，含课程/续训/迁移机制） | **直接改造为 Quaddle 低层策略**：换 MJCF、动作 12→4 维、观测裁剪 |
| `codes/practices/quadruped/cs123/1~5.*` | PID/FK/IK/MJCF/CPG 步态 | 连杆运动学等效建模、CPG 步态先验 |
| `codes/practices/vla/act/` | LeRobot 0.6.1 ACT：ALOHA 仿真数据训练（RTX 4080 SUPER 39min/50k steps），`MUJOCO_GL=egl` 下 `lerobot-eval` 闭环 20 回合成功 50% | **证明本机 MuJoCo + LeRobot 数据/训练/评测链路已通**；ACT 作为轻量对照组 |
| `docs/practices/amd/pupper-control/vla.md` | 分层 Skill 接口设计（observe/walk/turn/stop） | VLA 高层接口的既有设计共识 |

**硬件：** RTX 4070 Ti 12GB（12282 MiB，桌面占用约 445 MiB → 训练可用约 11.5GB，预算按 10GB 留余量）；MuJoCo 走 CPU + EGL 渲染。

### 1.3 与《四足机器人VLA仿真训练方案》的差异（为什么结论变了）

| 维度 | 12GB 方案（Go2/Isaac Lab） | 本方案（Quaddle/MuJoCo） | 原因 |
|---|---|---|---|
| 仿真平台 | Isaac Lab（GPU 并行） | **MuJoCo CPU** | 4DOF 机器人不需要千级并行；本仓库 MuJoCo 管线全套已通；Isaac Lab 的安装重量对 4 舵机机器人不划算 |
| 低层策略 | 必须训 locomotion PPO（12DOF 平衡难） | 仍训 PPO，但**难度骤降**（连杆被动产生步态、滚轮自稳），CPG/查表也可兜底 | Quaddle 的"复古机械"设计把步态难点转移到机械结构 |
| VLA 基座 | π0 / GR00T（2.2–3.3B），12GB 不可行 → 云端租卡 | **SmolVLA 450M 本地微调**；ACT 做对照 | LeRobot 官方算力指南将策略按显存分级：SmolVLA 是 Small VLA 档（batch 8 ≈ 10–16GB），π0/GR00T 是 24–40GB 档 |
| 云端依赖 | 必选（唯一付费环节） | **可选**（兜底/加速） | 450M vs 3.3B，量级差 7 倍 |
| 评测 | SIMPLER/LIBERO 等 | 自建 MuJoCo 闭环 rollout | 无现成 Quaddle benchmark |

---

## 2. 总体架构

与 12GB 方案相同的分层原则（VLA 只出高层指令，不碰 50Hz 控制）：

```
┌────────────────────────────────────────────┐
│ SmolVLA 高层策略（~5–10 Hz 重规划）           │
│ 输入：overhead/前视 RGB + 语言指令            │
│      + 本体状态（舵机角、IMU）                │
│ 输出：速度指令 (vx, vy, wz) 动作块（chunk）    │
└──────────────────┬─────────────────────────┘
                   ▼
┌────────────────────────────────────────────┐
│ 低层运控（50 Hz）                            │
│ 选项A：PPO 速度跟踪策略（复用 v2.5 管线）      │
│ 选项B：CPG/相位振荡器（贴近真机 OpenCat）      │
│ 输出：4 个舵机目标角 → 位置伺服（MuJoCo）      │
└────────────────────────────────────────────┘
```

**高层动作空间选速度指令 `(vx, vy, wz)`**（而非关节角/目标点）：
- 与低层策略的命令空间天然对齐（v2.5 就是速度条件策略）；
- 真机侧可无损映射到 OpenCat 指令（`kwkF`+`m0` 微调 = 前进 + 修向）；
- 3 维动作让 SmolVLA 的 flow-matching 头训练非常容易收敛。
- 备选（若速度指令表达力不够）：技能 token（`walk`/`strafe`/`spin`/`stand`）+ 参数。

---

## 3. 阶段一：Quaddle MuJoCo 环境（MJCF）

### 3.1 模型获取与转换

```bash
# 官方 URDF（MIT 协议，可入仓库 assets）
uv run huggingface-cli download petoi/quaddle --local-dir codes/practices/quadruped/cs123/assets/quaddle_urdf
```

两条路线，**建议先简后精**：

**路线 A（推荐起步）：运动学等效简化模型**

把每条腿简化为「1-DOF 舵机关节 + 运动学映射的等效足端」：
- 舵机 `thigh` 保留为主动关节；被动四连杆不逐杆建模，而是**用连杆正运动学算出足端轨迹**，以一个球形足端 geom（挂在随动 body 上，位置由 `thigh` 角度决定）替代；
- 滚轮焊接固定，足端用低摩擦球面近似；
- 连杆 FK 计算可仿照 `2.forward-kinematics`/`3.inverse-kinematics` 的既有工具。

优点：1 周内可训、闭环稳定性问题归零；缺点：丢失连杆动力学细节。**先用它打通 VLA 全链路，再升级路线 B。**

**路线 B（精确版）：URDF 直接加载 + 闭环约束**

1. MuJoCo 原生支持 URDF：在 URDF 头部插入 `<mujoco><compiler discardvisual="false" strippath="false"/></mujoco>`（保留 visual mesh 供相机渲染；`package://assets/` 重写为相对路径 `assets/`），`mujoco.MjModel.from_xml_path("robot.urdf")` 后另存为 MJCF 再手工完善；
2. **闭环四连杆**：URDF 的 `closing_*` dummy 链在 MuJoCo 里要用 `<equality><connect>` 把连杆两端点焊接回闭合环，被动关节（`motor_arm`/`leg`/`spring`）加小阻尼（1e-3~1e-2）与限位；这是本阶段最大的技术坑，先在单腿模型上调试；
3. **滚轮**：`continuous` 关节加阻尼作为被动滚动；若数值不稳定则临时焊死；
4. 物理参数：`timestep 2–4ms`、物理 250Hz / 控制 50Hz（与 v2.5 对齐）、舵机用 `position` actuator（kp 按舵机规格 + 限位取 URDF `limit`）。

### 3.2 验收标准（阶段关卡）

- [ ] 静止站立 30s 不倒、不穿模、不漂移；
- [ ] 四舵机正弦摆动（固定相位差）产生可观察的前进/转向位移；
- [ ] 键盘 WASD 驱动 4 个舵机目标角，机器人行为可预期；
- [ ] `MUJOCO_GL=egl` 无头渲染 320×320 相机帧正常（已在 ACT 管线验证过链路）；
- [ ] （路线 B）`kwkF`-style 开环步态表能走出稳定步行。

---

## 4. 阶段二：低层运控策略（50Hz）

| 选项 | 做法 | 何时选 |
|---|---|---|
| **A. PPO 速度策略（推荐主线）** | 复制 `6.rl_pupper_v2_5` → 新建 `7.rl_quaddle/`：换 MJCF、动作 12→4 维舵机位置残差（±30° 限幅 + 5.5Hz 低通）、观测裁剪到 ~26–33 维（重力方向 3、body 角速度 3、线速度 3、命令 3、舵机位置/速度 8、上步动作 4），奖励沿用 v2.5（速度跟踪、姿态、能耗、足端腾空、防抖），命令回合内重采样 | 默认路径；SB3 + CPU（n_envs 16–32），4DOF 预计数百万步收敛，一两天内出结果 |
| B. CPG / 开环步态表 | 相位振荡器 4 路相位差 → 舵机角；从官方在线仿真器观察 `kwkF`/`ktrL`/`kstrF` 的相位差做先验（固件开源后直接读 SkillLibrary） | RL 不收敛时的兜底；也是最贴近真机的执行层 |
| C. 混合 | B 起步 + RL 学残差 | 折中，v2.5 的 `bootstrap + finetune` 课程思路可平移 |

显存：本阶段 < 1GB（MuJoCo CPU + 百万参数 MLP 策略）。**验收：** 指令速度 vs 实际速度跟踪误差 < ±20%，`vx∈[-0.3, 0.3] m/s` 级别、`wz` 原地转可控、踢扰后 2s 内恢复。

> 说明：Quaddle 车轮足端 + 被动连杆的组合意味着"步态"可能更像混合轮腿滚动，RL 奖励里的足端腾空/接触时序项（v2.5 的 `foot_clearance`/`feet_stance_time`）需要按实际接触模式重标定，不要照抄。

---

## 5. 阶段三：VLA 数据采集（脚本化专家，本地）

**无需人工遥操作**——四足导航任务的专家可以直接参数化生成（12GB 方案同思路，但在 MuJoCo 里更简单）：

1. **场景**：平地 + 2–4 个彩色物体（红/蓝/绿积木或球），位置随机；机器人随机初始朝向；
2. **任务集与语言模板**（起步 5 个任务，每任务递增难度）：
   - "Walk to the red block" / "走到红色积木旁"
   - "Go around the blue ball to the green block" / "绕过蓝色球，到绿色方块旁"
   - "Spin around in place" / "原地转一圈"
   - "Walk forward then turn right and stop"
3. **专家策略**：目标点已知 → 直线/圆弧绕行路径 → 离散成 `(vx, vy, wz)` 指令序列 → 低层策略执行并记录。执行失败（翻倒/超时）的 episode 丢弃并重采样；
4. **观测记录**：`observation.images.overhead`（320×320，30fps；可加第二机位 `front`，单相机优先省显存）、`observation.state`（舵机角/角速度 + IMU + 速度估计）、`action`（`(vx,vy,wz)`，按控制频率写入）、`task`（语言指令）；
5. **格式**：`LeRobotDataset`（与 `vla/act` 管线直接兼容，微调命令零改动）；
6. **规模**：冒烟 5 任务 × 50 episodes（250 条）；正式 **2k–5k episodes**。320×320 视频编码后每条约几 MB，5k 条约 10–25GB，注意磁盘预留；
7. **吞吐**：MuJoCo CPU 步进（4 关节很轻）+ EGL 渲染，预计每 episode（10–20s 仿真）数秒到十几秒，千条一夜。

**验收：** 数据集可通过 `LeRobotDataset` 加载、视频帧与 action 时间戳对齐（`tolerance_s` 内）、每任务各变体覆盖均匀（SmolVLA 官方经验：**每种变体至少 10 条，25 条以下会明显欠拟合**）。

---

## 6. 阶段四：VLA 训练（10GB 显存核算）

### 6.1 基座选型（2026-11 调研）

LeRobot 官方算力指南按显存分级（batch 8 + AdamW，量级估计 ±50%）：

| 档位 | 模型 | batch 8 显存 | 本机 10–11GB 可行性 | 语言理解 |
|---|---|---|---|---|
| Light BC | ACT / VQ-BeT / TDMPC | 2–6 GB | ✅✅（`vla/act` 已跑通） | ❌ 需自行接入（one-hot 任务 embedding） |
| **Small VLA** | **SmolVLA** | **10–16 GB** | ✅ **batch 2–4 + bf16 AMP** | ✅ SmolVLM2 骨干，预训练 450M |
| Diffusion | diffusion / multi_task_dit | 8–14 GB | ✅（无语言，作对照可） | ❌ |
| Large VLA | π0 / π0.5 / π0-fast | 24–40 GB | ❌ 本地不可行；云端 LoRA（A100/A40） | ✅ |
| Large VLA | GR00T N1 / OpenVLA 7B | 24–40 GB+ | ❌（OpenVLA QLoRA 12–16GB 勉强，吞吐极低不推荐） | ✅ |

**主线：SmolVLA**（`lerobot/smolvla_base`）。它是目前唯一"为单卡消费级 GPU 训练而生"的 VLA：
- 结构：SmolVLM2（SigLIP 视觉编码器 + SmolLM2 解码器，只用前 16 层）+ ~100M flow-matching 动作专家；
- 每帧仅 64 个视觉 token（PixelShuffle 降采样 512×512），天然省激活显存；
- 配置默认 `freeze_vision_encoder=True`、`train_expert_only=True`——**视觉编码器冻结、只训动作专家，显存直接砍半**；
- 动作块 `chunk_size=50`，非自回归流匹配输出，与速度指令连续动作空间匹配。

**对照组：ACT + 语言条件**（复用 `vla/act`，把任务 one-hot/CLIP 文本特征拼入观测）。意义：如果 ACT+语言 已能满足 Quaddle 的封闭任务集，说明 SmolVLA 的开放词汇能力在本项目里是"超前投资"；反之则验证 VLA 预训练的价值。两者同数据同评测，直接产出对比结论。

### 6.2 显存核算（4070 Ti 12GB / 预算 10GB）

| 项 | 估算 |
|---|---|
| 权重（bf16，450M） | ~0.9 GB |
| 优化器状态（只训 ~350M 专家+投影，AdamW fp32） | ~4 GB |
| 激活（batch 4，1 相机，64 token/帧 + 文本） | ~3–5 GB |
| CUDA 上下文/碎片 | ~0.5–1 GB |
| **合计（batch 4 + AMP）** | **~8–10 GB** ✅ |
| 保险档（batch 2 + AMP） | ~5–7 GB ✅✅ |

依据：官方档位 batch 8 ≈ 10–16GB，显存近似随 batch 线性缩放；`use_amp` bf16 再降激活（0.6.1 源码确认支持 `--policy.use_amp`，Ada 卡走 bf16 无需梯度缩放）。视觉编码器冻结与专家独训是默认配置，无需额外操作。

### 6.3 训练命令（LeRobot 0.6.1 本地验证过的形态）

```bash
cd codes/practices/vla/act   # 复用现有 uv 环境（lerobot 0.6.1 已锁版本）
uv run lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=local/quaddle_nav \
  --batch_size=4 \
  --steps=20000 \
  --save_freq=5000 \
  --policy.device=cuda \
  --policy.use_amp=true \
  --policy.scheduler_decay_steps=20000 \
  --output_dir=outputs/smolvla_quaddle_v1
```

**踩坑清单（来自官方文档与社区实测）：**
1. 官方示例命令是 `batch_size=64`（为 A100/多卡写的），**直接粘贴必 OOM**；先 `batch_size=4` 冒烟，OOM 则降 2。0.6.1 未暴露梯度累积参数，等效大 batch 靠更多步数替代（或升级 lerobot 新版，其训练配置已加入 mixed precision/gradient accumulation 运行时字段）；
2. **缩短 `steps` 必须同步缩短 `scheduler_decay_steps`**（默认 30000），否则学习率全程停在峰值、训练不收敛——这是短训最常见翻车点；
3. 先跑 **2k steps 冒烟**：loss 下降 + rollout 出现趋标行为，再上 2 万步正式轮；
4. SmolVLA 默认按 512×512 重缩放图像，数据侧给 320×320 即可（内部有 padding resize），不必喂 512 原图；
5. 单相机优先；确需双机位时 batch 降 2；
6. 显存兜底：本地反复 OOM 时，同一命令原样搬到云端 L4（HF Jobs 约 $0.8/h，AutoDL/A40 约 ¥2–6/h），batch 提回 8–16，单轮 ¥10–30。

### 6.4 时长估算

官方参考：4×H100 上 5k steps@batch32 ≈ 28 分钟（SmolVLA）。折算 4070 Ti 单卡 batch 4、20k steps：**约 3–8 小时，一晚一轮**。整个"补数据→微调→评测"迭代完全在本地闭环。

---

## 7. 阶段五：评测闭环（回本地 MuJoCo rollout）

SmolVLA 没有 Quaddle 现成 benchmark，自建闭环评测（参照 `vla/act` 的 `lerobot-eval` 思路，自定义 env 用自写 rollout 循环）：

- **协议**：每任务 ≥ 20 个随机种子 episode，固定评测场景集；成功 = 目标距离 < 10cm 且稳定停止 2s；同时记录碰撞率、路径效率（实际路径/直线距离）、中途翻倒次数；
- **对比矩阵**：低层直控（专家上界）/ ACT+语言 / SmolVLA；单任务 vs 多任务混合训练；
- **失败归因三分类**（沿用 12GB 方案）：感知错误（没看到目标）→ 补场景渲染多样性；语言理解错误（指令映射错）→ 补任务模板与变体；动作精度不足（方向对但停不住）→ 调 chunk/增加近目标减速数据；
- **渲染与推理同卡**：EGL 渲染 ~0.5GB + SmolVLA 推理 bf16 ~1.5GB，与训练不同时进行即可，无显存冲突。

**验收：** 5 个任务平均成功率 ≥ 60%（冒烟轮 ≥ 30% 即进入迭代），输出成功率表 + 失败案例视频。

---

## 8. Sim2Real 展望（真机 2026-12 交付后，非本方案范围）

- **指令层对齐**：VLA 的 `(vx, vy, wz)` → OpenCat 指令（前进 `kwkF`、转向 `ktrL`、微调用 `m0` 舵机增量），先在官方 Rust 仿真器验证映射，真机交付后直接复用；
- **低层一致化**：真机用 OpenCat 固件步态（交付前开源），MuJoCo 低层策略仅作仿真内 VLA 数据引擎，sim2real 只迁移高层 VLA + 指令接口，**这大幅缩小迁移面**（不需要动量策略上真机）；
- **真机数据补充**：Puppet Mode 手导录制 → 转 LeRobotDataset，与仿真数据混训（比例先 1:4）；
- **部署**：SmolVLA 支持 CPU/Mac 推理 + 异步推理栈（官方数据：快 30%、吞吐 2×）；真机推理可跑在笔记本 GPU（bf16 ~1GB 级）或 RPi5 + 量化（int8）；
- **动力学对齐**：连杆阻尼/滚轮摩擦用真机开环动作 vs 仿真对照标定。

---

## 9. 里程碑与时间预算

| 周 | 内容 | 产出/验收 |
|---|---|---|
| W1 | 阶段一：简化 MJCF 环境（路线 A） | 站立/WASD/正弦步态/EGL 渲染四项验收 |
| W2 | 阶段二：RL 低层策略（改造 v2.5） | 速度跟踪 < ±20%，抗扰恢复 |
| W3 | 阶段三：数据管线 + 250 条冒烟 + SmolVLA 2k steps 冒烟 | loss 下降、rollout 有趋标行为 |
| W4 | 阶段四：2k–5k episodes + 20k steps 正式训练（一晚/轮） | 多任务成功率 ≥ 60% |
| W5 | 阶段五：评测矩阵、失败归因、补数据迭代（1–2 轮） | 对比报告 + demo 视频 |
| 追加 | 路线 B 精确连杆 MJCF（可选，替换数据重跑 W3–W4） | 开环 kwkF 步态可稳定行走 |

预算：**仿真训练 0 元**；云端兜底预算 ~¥50–100（大概率用不上）。

---

## 10. 风险与对策

| 风险 | 概率 | 对策 |
|---|---|---|
| URDF 闭环连杆在 MuJoCo 数值不稳（路线 B） | 高 | 先行路线 A 简化模型；单腿调试 equality 约束；仍不行则 A 版直接作为最终仿真（对 VLA 高层影响有限） |
| 滚轮足端接触行为怪异（抖动/打滑） | 中 | 焊死滚轮 + 球面足端近似；调整 `condim`/摩擦锥 |
| SmolVLA 10GB 内 OOM | 中 | batch 4→2、单相机、AMP 必开；极端情况云端 L4（同命令搬家，¥10–30/轮） |
| 4 舵机速度指令表达力不足（全向+绕障不够用） | 低中 | 高层动作空间升级为技能 token + 参数（架构已预留） |
| RL 低层在连杆+滚轮构型上不收敛 | 中 | CPG 开环步态兜底（选项 B）；参照官方仿真器相位差 |
| Quaddle 固件/技能库未开源（12 月前） | 确定 | 仿真内自洽完成全部 VLA 工作；交付后对齐 SkillLibrary 做指令层校准 |
| MuJoCo 与官方 Rust 仿真行为不一致 | 中 | 以官方 sim 的 `kwkF` 轨迹做行为校准参照；本方案 sim2real 只迁移高层，影响可控 |

---

## 11. 参考资料

**Quaddle 官方：**
- URDF 模型（MIT）：https://huggingface.co/petoi/quaddle
- Quaddle 产品页：https://www.petoi.com/pages/quaddle-educational-robot-kit
- 在线仿真器：https://www.petoi.com/pages/quaddle-robot-dog-simulator
- 仿真器开发仓（Rust/WASM，含 URDF/mesh/ESP32 ROM）：https://github.com/one-for-all/quaddle-sim
- OpenCat 主仓（Quaddle 公告，4 舵机/Puppet Mode 描述）：https://github.com/PetoiCamp/OpenCat-Quadruped-Robot
- OpenCat ESP32 仓（Quaddle 代码结构参照）：https://github.com/PetoiCamp/OpenCatEsp32-Quadruped-Robot

**VLA 与低显存训练：**
- SmolVLA 文档（LeRobot）：https://huggingface.co/docs/lerobot/smolvla
- SmolVLA 发布博客（架构/异步推理）：https://huggingface.co/blog/smolvla
- SmolVLA 论文：https://huggingface.co/papers/2506.01844
- 基座权重：https://huggingface.co/lerobot/smolvla_base
- LeRobot 训练算力指南（按显存分档，batch 8）：见 https://swiftwand.com/en/smolvla-finetune-local-gpu-2026-en/ 的转述表格（Light BC 2–6GB / Small VLA 10–16GB / Large VLA 24–40GB）
- SmolVLA 仿真环境微调教程（Correll Lab）：https://medium.com/correll-lab/fine-tuning-smolvla-for-new-environments-code-included-af266c56d632

**云端备选（仅兜底）：**
- HF Jobs 定价（L4 $0.8/h 级）：https://huggingface.co/docs/huggingface_jobs
- 12GB 方案的 A40/L40 国内租价参考见《四足机器人VLA仿真训练方案》第五节

**本仓库相关：**
- 低层策略改造基座：`codes/practices/quadruped/cs123/6.rl_pupper_v2_5/`
- VLA 训练/评测链路（LeRobot 0.6.1 + MuJoCo EGL）：`codes/practices/vla/act/`
- 分层 Skill 接口设计：`docs/practices/amd/pupper-control/vla.md`

---

## 附：执行记录

### W1 · 阶段一：Quaddle MuJoCo 环境（2026-11-07 完成 ✅）

产物：`codes/practices/quadruped/cs123/7.quaddle-mjcf/`（未跟踪新目录，零改动既有代码）

- **模型**：官方 URDF → MJCF 构建器（`scripts/build_quaddle_mjcf.py`）。每腿 6 刚体
  （thigh/motor_arm/spring/leg/wheel/wheel_tip），thigh↔leg 双点 `connect`（site 对）闭链；
  碰撞几何照抄官方 builder.rs；视觉 mesh 16 个 STL（剔除 53MB 弹簧）。
- **关键发现 1（闭链）**：URDF 零位在 y-z 平面天然闭合，x 向 ~1.9mm 为 CAD 间隙；
  装配解解析可得（ψ=θ 平行曲柄，σ=λ=0），间隙由约束焊接吸收。
- **关键发现 2（平衡站姿）**：官方 power-on 是 rest 蹲姿（前腿悬空）；
  FK 扫描出前腿补偿角 +6.95° 实现四足等高 → `q=0` 即平衡站姿（全关节 ref 技巧）。
- **关键发现 3（数值）**：扭簧刚度 0.6 N·m/rad（0.15 塌陷 / 2.0 与 2ms 步长冲突爆炸）；
  timestep 1ms + Newton(100, 1e-10) + 舵机目标低通 τ=30ms 后全部扫描无 NaN。
- **验收**：30s 站立 PASS（漂移 0.0mm，roll 峰 0.71°）；EGL 渲染正常；pytest 5/5。
- **步态基线**：trot ±8°@1.5Hz 站稳爬行仅 3~5mm/s——球足端对称激励的物理上限，
  直线行走需 W2 路线 A（官方 kwkF 轨迹回放）或 B（RL）。
- 详见 `7.quaddle-mjcf/README.md`。

### 待办（按方案顺序）

- [ ] W2：低层运控——路线 A：从官方仿真器录 kwkF 关节轨迹；路线 B：复制 v2.5 管线建 `8.rl_quaddle`（4 维动作）
- [ ] W3：数据管线（LeRobotDataset + 脚本化专家）
- [ ] W4：SmolVLA 冒烟 → 正式微调
- [ ] W5：评测闭环

### W2 · 低层运控（2026-11-07 ~ 11-08 完成 ✅，路线调整：RL → 查表 CPG）

产物：`codes/practices/quadruped/cs123/8.rl_quaddle/`

- **走过的弯路（有价值的证据链）**：RL 四轮（关节动作 500k+1.5M、CPG 参数动作
  1.5M、修 bug 后 1.5M）均未学会命令条件化响应。期间修复两个环境级 bug：
  ① 评估脚本接口（ctrl 被 step 覆盖）；② reset 沉降不足致悬空开局翻倒。
- **物理结论**：球足端模型稳定行走包线 ~10-20mm/s，RL 收敛值即物理真值。
  修正足端为滚轮圆柱（7.quaddle-mjcf 同步更新，站立复验 PASS 35.4mm）后，
  forward 实测 +113mm/s——仿真追上真机量级的关键是**足端滚动方向性**，
  而非控制器智能。
- **交付**：`cpg_controller.py` 查表 CPG（forward/back/turn_left/turn_right/stop
  五档，对齐真机 OpenCat SkillLibrary 模型），11s 指令序列验收 PASS
  （总位移 1.1m 未翻倒）。W3 的脚本化专家直接调用此控制器。
- RL checkpoint 全部保留；复活要点：观测加相位时钟+档位历史。

### W5 · 评估闭环（2026-11-08/09 完成，基线 1/3 → 迭代受挫但根因清晰）

- 冒烟：300 步 GPU 训练通过（14.7 steps/s，显存 1.69GB@batch1 fp32），端到端
  推理验证（3 路 RGB + state + 指令 → 动作块）。
- 正式训练 v1（250 条 20k 步）：loss 1.51→0.32，同 seed 三场景 rollout 1/3。
- 迭代 v2/v3（边界密集数据、连续转向强度、30k 步）：仍 1/3。根因经诊断脚本
  定位：**弧线几何死锁 + 动作空间缺纯原地转原语**（详见 quaddle-data/README
  W5 评估记录）。改进路线四条已列出，超出本轮范围。
- 产物：rollout.gif（v1/v3）、rollout_ep2_strip.png、评估日志、全套检查点。
