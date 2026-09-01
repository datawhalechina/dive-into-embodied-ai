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

## 内容大纲

教程只保留「理论基础 + 项目实战」两大模块。项目实战再按「AMD 专区、仿真实战、真机实战」归类，章节名后的链接直接指向当前文档或对应目录。

状态标记说明:**✅ 可用** = 章节内容完整,可直接阅读;**🚧 部分可用** = 一部分章节有内容、一部分仍是占位;**🚧 占位中** = 目录已建但只有占位页;**⏳ 待补充** = 暂未开工。

### 项目实战

| 分类 | 章节 | 简介 | 状态 |
| :--- | :--- | :--- | :--- |
| AMD 专区 | [AUP Learning Cloud 云算力](docs/practices/amd/aup-learning-cloud.md) | Ryzen AI APU、ROCm、JupyterHub 与 Code Server | ✅ 可用 |
| AMD 专区 | [MicroDuck RL｜AMD ROCm](docs/practices/amd/microduck-rl/index.md) | R9700、ROCm MuJoCo Warp、动态接触回归与双足 PPO | ✅ 可用 |
| AMD 专区 | [玩转 Pupper 四足机器人](docs/practices/amd/pupper-control/intro.md) | AMD 平台上的强化学习运动策略与 VLA 实验 | ✅ 可用 |
| 仿真实战 | [从 0 到 1 搭建四足机器人](docs/practices/quadruped/cs123/0.intro.md) | MuJoCo、PD、运动学、PPO 与 LLM 控制 | ✅ 可用 |
| 仿真实战 | [MicroDuck RL 小黄鸭双足机器人](docs/practices/humanoid/microduck-rl/index.md) | mjlab + MuJoCo Warp：GPU 并行 PPO 与双足步态训练 | ✅ 可用 |
| 仿真实战 | [MuJoCo 机械臂与 DDPG](docs/practices/robot-arm/mujoco-arm-pick-place/index.md) | MuJoCo 环境与连续控制实验 | ✅ 可用 |
| 仿真实战 | [ACT 双臂操作训练](docs/practices/vla/act/index.md) | ACT + ALOHA：训练、评估与结果复现 | ✅ 可用 |
| 仿真实战 | [两轮足 Flamingo · Isaac Lab](docs/practices/wheel-legged/flamingo-isaaclab/preview.md) | PPO / CaT 训练与跨仿真验证 | 🔜 预告 |
| 真机实战 | [SO-101 + LeRobot 真机教程](docs/practices/robot-arm/data-collection/so101-lerobot-real/index.md) | 硬件连通、安全测试与动作回放 | ✅ 可用 |
| 真机实战 | [LeRobot 中文课程讲义](docs/practices/robot-arm/data-collection/lerobot-course/index.md) | 数据集、工具链与真机学习流程的前置知识 | ✅ 可用 |
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
| [World-Model](docs/foundations/world-model/placeholder.md) | 世界模型在具身场景下的落地路径 | 🚧 占位中 |

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
| 罗如意 | 项目负责人 | 智能汽车竞赛国奖&FunRec开源项目负责人 |
| 江季  | 项目负责人 | [蘑菇书](https://github.com/datawhalechina/easy-rl)作者 |
| 康博 | 项目负责人 | nobl.ai 联合创始人 & 比利时根特大学访问教授|

## 关注我们

<div align=center>
<p>扫描下方二维码关注公众号:Datawhale</p>
<img src="https://raw.githubusercontent.com/datawhalechina/pumpkin-book/master/res/qrcode.jpeg" width = "180" height = "180">
</div>

## LICENSE

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="知识共享许可协议" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a><br />本作品采用<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议</a>进行许可。
