<div align="center">
    <img src="static/img/career.webp" width="100%" alt="Dive into Embodied AI 横幅" />
</div>

<h1 align="center">Dive into Embodied AI</h1>
<p align="center"><b>具身智能入门与求职开源教程</b></p>

<p align="center">
  <a href="https://datawhalechina.github.io/dive-into-embodied-ai/"><img alt="在线阅读" src="https://img.shields.io/badge/%E5%9C%A8%E7%BA%BF%E9%98%85%E8%AF%BB-datawhalechina-blue" /></a>
  <a href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="许可协议" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a>
  <img alt="状态" src="https://img.shields.io/badge/status-Alpha-orange" />
</p>

<p align="center">
  <sub>合作支持</sub><br />
  <a href="docs/practices/amd/intro.md">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="assets/logo/logo_amd_wht.svg" />
      <source media="(prefers-color-scheme: light)" srcset="assets/logo/logo_amd.svg" />
      <img src="assets/logo/logo_amd.svg" width="220" alt="AMD University Program" />
    </picture>
  </a>
</p>

> [!CAUTION]
> **Alpha 内测版本**:仍在迁移和重构中,部分章节是占位页,欢迎提 Issue 反馈问题或建议。

## 项目定位

从零到一搭建一台具身智能机器人：深入强化学习、World-Model、VLA 等智能决策方法的工程落地，贯穿仿真环境、控制器、运动规划、感知系统等技能树模块,并在真实项目中跑通"决策—控制—感知"完整链路。

## 学习地图

先看全局，再找到你的第一步。下面按「概念 → 知识模块 → 入门路线 → 实践方向」组织学习入口。

[认识具身智能](#认识具身智能) · [知识全景](#知识全景) · [从零开始](#从零开始) · [选择实践方向](#选择实践方向)

也可以打开 [交互版学习地图](https://datawhalechina.github.io/dive-into-embodied-ai/learning-map)，切换机器人任务，或观看可旋转、可调速的 [3D 机器人足球演示](https://datawhalechina.github.io/dive-into-embodied-ai/learning-map#what-is-embodied-ai)。

### 认识具身智能

具身智能（Embodied AI）关注智能体如何依托身体，通过 **感知、决策与行动闭环** 与环境交互并完成任务。本项目以机器人等物理系统为主线，连接概念、算法和工程实践。

机器人是具身智能的重要载体。机器人学研究机器人的设计、构建与控制；具身智能强调智能、身体与环境之间的协同。

以机器人踢足球为例，目标是把球踢进球门，同时保持身体平衡：

```mermaid
flowchart LR
    perception[感知与理解] --> decision[学习与决策]
    decision --> control[运动与控制]
    control --> environment[环境与结果]
    environment -->|新的观测与反馈| perception
```

| 环节 | 要回答的问题 | 踢足球时发生了什么 |
| :--- | :--- | :--- |
| 感知 | 世界和自己，现在是什么状态？ | 识别足球、球门和守门员，估计位置与身体姿态。 |
| 决策 | 为了完成任务，下一步做什么？ | 观察防守位置，选择射门空当，安排靠近和踢球动作。 |
| 行动 | 怎样把动作准确、稳定地做出来？ | 调整站位与重心，协调关节，驱动腿部踢球。 |
| 反馈 | 行动之后发生了什么？ | 跟踪球的运动、检查射门结果，用新的观测调整下一步。 |

这是按功能理解系统的方式。实际系统可以由多个模块协作，也可以用一个模型承担多种功能。

### 知识全景

前三个模块串起行动闭环，后三个模块支撑整个系统的开发、训练与验证。

| 模块 | 核心知识 | 从哪里开始 |
| :--- | :--- | :--- |
| **感知与理解** | 视觉与三维几何、状态估计、视觉语言模型（VLM） | [传感器、坐标系与标定](docs/foundations/perception/1.sensor-calibration-sim2real.md)、[VLM 基础](docs/foundations/vlm/0.intro.md) |
| **学习与决策** | 模仿学习、强化学习、视觉语言动作模型（VLA）、世界模型 | [模仿学习](docs/foundations/rl-for-robotics/12.imitation-learning.md)、[强化学习](docs/foundations/rl-for-robotics/1.intro.md)、[VLA](docs/foundations/vla/vla-intro.md)、[世界模型](docs/foundations/world-model/0.intro.md) |
| **运动与控制** | 运动学与动力学、运动规划、反馈控制 | [坐标变换与正逆运动学](docs/foundations/robotics-and-ros2/2.kinematics_transform.md)、[PID 到 MPC](docs/foundations/controllers/intro.md)、[MoveIt 2](docs/foundations/robotics-and-ros2/10.moveit2_basics.md) |
| **仿真与评测** | 物理仿真、评测基准、仿真到真实（Sim2Real） | [仿真工具](docs/foundations/simulation/1.intro.md)、[数据集与评测基准](docs/information/datasets.md) |
| **数据与训练** | 遥操作、示教数据、数据质量、策略训练与验证 | [LeRobot 数据与工具链](docs/practices/robot-arm/data-collection/lerobot-course/index.md)、[从示范中学习动作](docs/foundations/rl-for-robotics/12.imitation-learning.md) |
| **本体与系统** | 机械结构、传感器与电机、ROS2、嵌入式 | [机器人系统全景](docs/foundations/robotics-and-ros2/1.course_introduction.md)、[SO-101 真机连接与调试](docs/practices/robot-arm/data-collection/so101-lerobot-real/index.md) |

算法在地图里的位置：**模仿学习（IL）** 从示范中学动作，**强化学习（RL）** 用奖励改进策略，**VLA** 把视觉与语言连接到动作，**世界模型** 预测动作的后果。规划与控制也可以独立解决任务，或与学习方法组合。

### 从零开始

先体验，再补基础，再完成一个项目。每一步都留下一份可以检查的学习成果。

| 步骤 | 学习入口 | 完成标志 |
| :--- | :--- | :--- |
| **1. 建立系统概念** | 阅读 [机器人系统全景](docs/foundations/robotics-and-ros2/1.course_introduction.md)，无需设备。 | 能用踢球、拿杯子或行走的例子，说明观测、动作、目标和反馈。 |
| **2. 亲手体验一个原理** | 打开 [PD 控制实验](https://datawhalechina.github.io/dive-into-embodied-ai/cs123/pd-playground)，在浏览器中调整参数。 | 能解释响应、超调和稳定性怎样随参数变化。 |
| **3. 跑通最小仿真** | 按 [MuJoCo 教程](docs/foundations/simulation/3.mujoco.md) 加载模型、推进仿真并施加控制。 | 保存一份自己能运行、能修改的仿真实验。 |
| **4. 沿一条主线完成项目** | 跟随 [从零到一搭建四足机器人](docs/practices/quadruped/cs123/0.intro.md)，从单关节控制走到步态与策略训练。 | 记录实验条件、结果与失败原因，完成一次可复现的项目实践。 |

准备写代码时，按需补 **Python、Linux / Git、线性代数**；进入模型训练后，再补 **PyTorch、概率与深度学习**。已有基础可以直接跳到对应阶段。

### 选择实践方向

机器人形态、任务和算法是不同维度，同一台机器人可以组合多种能力。下面是建议的学习顺序，可以按已有基础调整。

| 方向与任务 | 建议学习顺序 | 实践入口 |
| :--- | :--- | :--- |
| **抓取与操作**：抓取、推拉、装配或协调双臂 | 运动学 → 示教数据 → 模仿学习 → ACT → VLA | [ACT 双臂实验](docs/practices/vla/act/index.md)，建议先有 Python / PyTorch 基础。 |
| **足式运动**：保持平衡，行走、转向与起身 | 反馈控制 → 运动学 → 仿真 → 强化学习 → 迁移验证 | [四足机器人课程](docs/practices/quadruped/cs123/0.intro.md)，再进入独立机器人项目。 |
| **导航与移动操作**：到达目标、绕开障碍，再完成操作 | 定位与建图 → 路径规划 → 导航 → 移动与操作协同 | [机器人学与 ROS2 路线](docs/overview/robotics-and-ros2-roadmap.md)；完整导航与移动操作项目仍待补充。 |

继续深入前，先写清楚：机器人看到什么、输出什么动作、什么条件下算成功。更多背景与方向选择见 [进阶学习路径](docs/overview/learning-path.md)，完整章节见下方 [内容大纲](#内容大纲)。

### 参考

1. [NVIDIA：什么是具身智能？](https://www.nvidia.cn/glossary/embodied-ai/)

## 内容大纲

网站分为「理论基础、系列教程、项目实战、具身信息」四个栏目：理论基础讲解原理，系列教程沿章节主线学习，项目实战提供独立 Demo 与复现案例，具身信息整理论文、数据集、开源项目和工具。项目实战继续按「AMD 专区、仿真实战、真机实战」归类。

状态标记说明:**✅ 可用** = 章节内容完整,可直接阅读;**🚧 部分可用** = 一部分章节有内容、一部分仍是占位;**🚧 占位中** = 目录已建但只有占位页;**⏳ 待补充** = 暂未开工。

### 系列教程

从 [系列教程总览](docs/tutorials/intro.md) 选择学习主线，原课程文章与代码路径保持不变。

| 教程 | 学习主线 | 进度 |
| :--- | :--- | :--- |
| [从零到一搭建四足机器人](docs/practices/quadruped/cs123/0.intro.md) | MuJoCo、PD、运动学、策略训练、语言控制与感知 | 总览与 8 章可读 |
| [LeRobot 中文课程讲义](docs/practices/robot-arm/data-collection/lerobot-course/index.md) | 机器人学习、数据集、工具链与经典机器人学 | 已整理 Unit 0–2 |
| [Flamingo 轮足课程](docs/practices/wheel-legged/flamingo-isaaclab/preview.md) | Isaac Lab 训练与跨仿真验证 | 课程预告 |

### 具身信息

从 [具身信息总览](docs/information/intro.md) 查找公开学习与研究资源：

- [论文与研究](docs/information/papers.md)：原论文、项目主页与站内解读。
- [具身数据集](docs/information/datasets.md)：机器人示教数据与评测基准。
- [开源项目](docs/information/open-source.md)：模型代码与训练框架。
- [仿真与工具](docs/information/tools.md)：仿真引擎、学习环境与官方文档。

### 项目实战

| 分类 | 章节 | 简介 | 状态 |
| :--- | :--- | :--- | :--- |
| AMD 专区 | [AUP Learning Cloud 云算力](docs/practices/amd/aup-learning-cloud.md) | Ryzen AI APU、ROCm、JupyterHub 与 Code Server | ✅ 可用 |
| AMD 专区 | [MicroDuck RL｜AMD ROCm](docs/practices/amd/microduck-rl/index.md) | R9700、ROCm MuJoCo Warp、动态接触回归与双足 PPO | ✅ 可用 |
| AMD 专区 | [玩转 Pupper 四足机器人](docs/practices/amd/pupper-control/intro.md) | AMD 平台上的强化学习运动策略与 VLA 实验 | ✅ 可用 |
| 仿真实战 | [MicroDuck RL 小黄鸭双足机器人](docs/practices/humanoid/microduck-rl/index.md) | mjlab + MuJoCo Warp：GPU 并行 PPO 与双足步态训练 | ✅ 可用 |
| 仿真实战 | [MuJoCo 机械臂与 DDPG](docs/practices/robot-arm/mujoco-arm-pick-place/index.md) | MuJoCo 环境与连续控制实验 | ✅ 可用 |
| 仿真实战 | [ACT 双臂操作训练](docs/practices/vla/act/index.md) | ACT + ALOHA：训练、评估与结果复现 | ✅ 可用 |
| 真机实战 | [SO-101 + LeRobot 真机教程](docs/practices/robot-arm/data-collection/so101-lerobot-real/index.md) | 硬件连通、安全测试与动作回放 | ✅ 可用 |
| 真机实战 | [Sim2Real 指南](docs/practices/quadruped/sim2real-guide/placeholder.md) | 仿真策略部署与真机验证 | 🚧 占位中 |

### 最新 Demo：MicroDuck RL 小黄鸭

<p align="center">
  <a href="docs/practices/humanoid/microduck-rl/index.md">
    <img src="docs/practices/humanoid/microduck-rl/figs/microduck-training-1500.gif" width="640" alt="MicroDuck 小黄鸭双足机器人稳定步态回放" />
  </a>
  <br/>
  <sub>✅ <b><a href="docs/practices/humanoid/microduck-rl/index.md">MicroDuck RL · 小黄鸭双足稳定步态</a></b><br/>mjlab + MuJoCo Warp · PPO GPU 并行训练（iteration 1500）</sub>
</p>

<table align="center">
  <tr>
    <td align="center" width="25%">
      <a href="https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/intro">
        <img src="assets/lab5_forward_gait_comparison.gif" height="220" alt="CS123 四足步态对比" />
      </a>
      <br/><sub>✅ <b><a href="https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/intro">从 0 到 1 搭建四足机器人</a></b><br/>CS123 仿真版 · MuJoCo + PPO + LLM 控制</sub>
    </td>
    <td align="center" width="25%">
      <a href="docs/practices/wheel-legged/flamingo-isaaclab/preview.md">
        <img src="assets/hero_swarm.gif" height="220" alt="Flamingo 两轮足在 Isaac Lab 中训练" />
      </a>
      <br/><sub>🔜 <b><a href="docs/practices/wheel-legged/flamingo-isaaclab/preview.md">两轮足 Flamingo · Isaac Lab</a></b><br/>新章预告 · Isaac Lab + PPO / CaT 训练</sub>
    </td>
    <td align="center" width="25%">
      <img src="assets/rebot_act_training.gif" height="220" alt="ReBot-Act 机械臂 ACT 策略训练效果" />
      <br/><sub>✅ <b>ReBot-Act · ACT 训练效果</b><br/>真机视觉模仿学习 · 方块抓取与放置</sub>
    </td>
    <td align="center" width="25%">
      <a href="docs/practices/vla/act/index.md">
        <img src="docs/practices/vla/act/figs/act_50k_success.gif" height="220" alt="ACT 在 ALOHA 仿真中完成双臂方块交接" />
      </a>
      <br/><sub>✅ <b><a href="docs/practices/vla/act/index.md">ACT · ALOHA 双臂交接</a></b><br/>50k 训练 · MuJoCo 20 回合成功率 50%</sub>
    </td>
  </tr>
</table>

### 理论基础

理论基础按当前导航的四列组织:大脑、小脑、感知系统、工程底座。当前优先把已有内容并入技能树,空缺模块先保留占位。

#### 大脑：智能决策

| 章节 | 简介 | 状态 |
| :--- | :--- | :--- |
| [强化学习决策](docs/foundations/rl-for-robotics/1.intro.md) | MDP、DQN、PPO、SAC、DDPG/TD3 与模仿学习 | ✅ 可用 |
| [视觉-语言-动作大模型(VLA)](docs/foundations/vla/vla-intro.md) | RT-1/RT-2、OpenVLA、ACT、Diffusion Policy、π 系列 | ✅ 可用 |
| [World-Model](docs/foundations/world-model/0.intro.md) | 世界模型概念、技术路线与具身场景应用 | ✅ 可用 |

#### 小脑：运动控制

| 章节 | 简介 | 状态 |
| :--- | :--- | :--- |
| [强化学习控制](docs/foundations/rl-for-robotics/10.ppo.md) | 把策略学习接到连续控制和机器人任务上 | ✅ 可用 |
| [控制器](docs/foundations/controllers/intro.md) | PID、LQR、MPC、阻抗控制与系统集成教程 | ✅ 可用 |
| [运动规划](docs/foundations/robotics-and-ros2/10.moveit2_basics.md) | Motion Planning 与 MoveIt 2 规划闭环 | ✅ 可用 |

#### 感官：感知系统

本体感知：机器人必须知道自己在哪里、姿态如何、速度如何、是否失稳。

外部感知：相机、雷达、触觉、电机电流、IMU、足端接触、机身姿态、末端位置。

| 章节 | 简介 | 状态 |
| :--- | :--- | :--- |
| [视觉感知与 VLM](docs/foundations/vlm/0.intro.md) | Transformer、ViT、视觉编码器与多模态融合 | ✅ 可用 |
| [定位、触觉与传感器标定](docs/foundations/perception/placeholder.md) | SLAM、足端接触、触觉传感、多传感器融合和 sim2real 标定 | 🚧 部分可用 |
| [传感器标定与 sim2real](docs/foundations/perception/1.sensor-calibration-sim2real.md) | 坐标系、时间同步、外参误差放大和在线标定监控 | ✅ 可用 |

#### 工程底座

| 章节 | 简介 | 状态 |
| :--- | :--- | :--- |
| [仿真工具](docs/foundations/simulation/1.intro.md) | Isaac Sim、MuJoCo、Gymnasium、PyBullet 快速上手 | ✅ 可用 |
| [ROS2](docs/foundations/robotics-and-ros2/0.intro.md) | 坐标变换、FK/IK、tf2、URDF 与 MoveIt 2 | ✅ 可用 |
| [CAN 与 MCU 通信](docs/foundations/communication/can-mcu.md) | 底层通信、执行器协议和上下位机链路 | 🚧 占位中 |
| [机械结构](docs/foundations/hardware/placeholder.md) | 连杆、关节、电机、减速器和末端执行器 | 🚧 占位中 |
| [数据工程与模仿学习](docs/foundations/rl-for-robotics/12.imitation-learning.md) | 从遥操作数据到模仿学习、LeRobot 工具链和策略训练 | ✅ 可用 |

## 组队学习

Datawhale 会围绕本教程组织组队学习。历史与在筹备中的组队学习计划文档会集中放在 `docs/team-learning/`(施工中),包括每期的学习路线、打卡要求和对应章节的导读。

- 最新一期报名入口:施工中
- 往期学习资料归档:施工中

## 本地预览

仓库使用 **Git LFS** 存放视频和 GIF。clone 之后必须先装 `git-lfs` 再 `git lfs pull`,否则本地看到的图/视频是 pointer 文本而不是真内容。完整步骤见 [CONTRIBUTING.md](CONTRIBUTING.md#首次克隆必读)。

```bash
# 1. 装 git-lfs(每台机器只需一次)
# brew install git-lfs        # macOS
# sudo apt install git-lfs    # Ubuntu / Debian
# choco install git-lfs       # Windows

# 2. 初始化并拉取 LFS 文件
git lfs install
git lfs pull

# 3. 装依赖、起本地预览
npm install
npm run dev
```

## Star History

<p align="center">
  <a href="assets/star-history.svg">
    <img src="assets/star-history.svg" width="900" alt="Dive into Embodied AI Star History Chart" />
  </a>
  <br />
  <sub>由 GitHub Actions 自动更新</sub>
</p>

## 贡献者名单

| 姓名 | 职责 | 简介 |
| :--- | :--- | :--- |
| 江季  | 项目负责人 | [蘑菇书](https://github.com/datawhalechina/easy-rl)作者，强化学习算法研究员 |
| 康博 | 项目负责人 | nobl.ai 联合创始人 & 比利时根特大学访问教授|
| 黄潇 | 项目负责人 | 同济大学毕业，智驾算法工程师 |
| 罗如意 | 项目负责人 | 智能汽车竞赛国奖&FunRec开源项目负责人 |

## 关注我们

<div align=center>
<p>扫描下方二维码关注公众号:Datawhale</p>
<img src="https://raw.githubusercontent.com/datawhalechina/pumpkin-book/master/res/qrcode.jpeg" width = "180" height = "180">
</div>

## LICENSE

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="知识共享许可协议" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a><br />本作品采用<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议</a>进行许可。
