<div align="center">
    <img src="static/img/embodied-ai-learning.webp" width="100%" alt="机器人学习工作台：仿真、机械臂、四足和人形机器人" />
</div>

<h1 align="center">Dive into Embodied AI</h1>
<p align="center"><b>具身智能入门与实践开源教程</b></p>

<p align="center"><b>中文</b> · <a href="README.en.md" lang="en">English</a> · <a href="README.it.md" lang="it">Italiano</a></p>

<p align="center">
  <a href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="许可协议" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a>
</p>

<h3 align="center">🤝 合作支持</h3>

<p align="center">
  <a href="docs/practices/amd/intro.md">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="assets/logo/logo_amd_wht.svg" />
      <source media="(prefers-color-scheme: light)" srcset="assets/logo/logo_amd.svg" />
      <img src="assets/logo/logo_amd.svg" width="220" alt="AMD University Program" />
    </picture>
  </a>
</p>

> [!TIP]
> **📖 在线阅读：[点击进入完整教程 →](https://datawhalechina.github.io/dive-into-embodied-ai/)**

<a id="项目定位"></a>

## 🎯 项目定位

一套面向入门与工程进阶的具身智能开源教程：围绕机器人在真实世界中完成**感知—决策—控制**闭环，系统串联感知与定位、强化学习 / VLA / World-Model、规划与运动控制，以及仿真、数据与部署等核心技术栈，并通过仿真实验与真机项目将原理落到可运行的代码上。

<a id="新手入门"></a>

## 🧭 新手入门

具身智能跨学科、模块多，**不必等所有理论都学完再动手**，建议按下面三步循序推进：

1. **先建学习地图**：快速浏览 [具身导论](docs/introduction/intro.md)，先摸清核心概念、常见技能、载体形态与技术栈分工，建立起全局视野（也可以先到 [在线仿真](https://datawhalechina.github.io/dive-into-embodied-ai/simulation) 直接动手体验）。
2. **带着问题补理论**：从最感兴趣的方向切入，带着具体问题进入 [理论基础](https://datawhalechina.github.io/dive-into-embodied-ai/foundations) 查漏补缺，需要什么学什么。
3. **用实战解决真问题**：选一个 [项目实战](https://datawhalechina.github.io/dive-into-embodied-ai/practices) 在仿真或真机中跑通闭环，在复现、调参与排错中真正解决实际问题。

<a id="精选实战-demo"></a>

## 🎬 精选实战 Demo

<table align="center">
  <tr>
    <td align="center" width="33%">
      <a href="docs/practices/humanoid/unitree-g1/index.md">
        <img src="docs/practices/humanoid/unitree-g1/figs/unitree-g1-push.gif" height="200" alt="Unitree G1 12-DoF 人形机器人在 MuJoCo 中行走并抵抗 0.55 m/s 侧向推撞" />
      </a>
      <br/><sub><b><a href="docs/practices/humanoid/unitree-g1/index.md">Unitree G1 · 12-DoF 人形运控</a></b><br/>LSTM 步态 · Sim2Sim 与 0.55 m/s 抗推撞恢复</sub>
    </td>
    <td align="center" width="33%">
      <a href="docs/practices/amd/microduck-rl/index.md">
        <img src="docs/practices/amd/microduck-rl/figs/microduck-amd-r9700-model750.gif" height="200" alt="MicroDuck 小黄鸭双足稳定行走回放" />
      </a>
      <br/><sub><b><a href="docs/practices/amd/microduck-rl/index.md">MicroDuck RL · 双足稳定行走</a></b><br/>mjlab + MuJoCo Warp · GPU 并行 PPO（ROCm / CUDA）</sub>
    </td>
    <td align="center" width="33%">
      <a href="docs/practices/humanoid/flamingo-isaaclab/0.intro.md">
        <img src="assets/hero_swarm.gif" height="200" alt="Flamingo 两轮足机器人在 Isaac Lab 中并行仿真" />
      </a>
      <br/><sub><b><a href="docs/practices/humanoid/flamingo-isaaclab/0.intro.md">Flamingo · 两轮足并行训练</a></b><br/>Isaac Lab · 动态平衡与 PPO / CaT 控制</sub>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="docs/practices/quadruped/cs123/0.intro.md">
        <img src="docs/practices/quadruped/cs123/figs/lab8_ball_chase.gif" height="200" alt="CS123 四足视觉感知与闭环追球" />
      </a>
      <br/><sub><b><a href="docs/practices/quadruped/cs123/0.intro.md">从 0 到 1 搭建四足机器人</a></b><br/>MuJoCo + RL 步态 + 视觉闭环追球</sub>
    </td>
    <td align="center" width="33%">
      <a href="docs/practices/vla/quaddle/index.md">
        <img src="docs/practices/vla/quaddle/figs/quaddle_L2_nav_silky50.gif" height="200" alt="Quaddle 四舵机机器人基于 SmolVLA 执行视觉语言端到端导航" />
      </a>
      <br/><sub><b><a href="docs/practices/vla/quaddle/index.md">Quaddle · SmolVLA 端到端导航</a></b><br/>10 GB 显存复现 · 视觉语言控制与 CPG 步态</sub>
    </td>
    <td align="center" width="33%">
      <a href="docs/practices/amd/pupper-control/intro.md">
        <img src="docs/practices/amd/pupper-control/figs/rocm-command-demo.gif" height="200" alt="AMD ROCm 平台上的 Pupper 四足速度指令跟踪演示" />
      </a>
      <br/><sub><b><a href="docs/practices/amd/pupper-control/intro.md">AMD ROCm · 四足运控实战</a></b><br/>Radeon GPU 原生加速 · 速度指令跟踪实测</sub>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="docs/practices/vla/recap/index.md">
        <img src="docs/practices/vla/recap/videos/state_030_success_value.gif" height="200" alt="π0.5 + RECAP 在 LIBERO 中执行长时程操作并输出价值曲线" />
      </a>
      <br/><sub><b><a href="docs/practices/vla/recap/index.md">π₀.₅ + RECAP · 长时程 VLA</a></b><br/>LIBERO 评测 · 价值模型与 RL 后训练</sub>
    </td>
    <td align="center" width="33%">
      <a href="docs/practices/vla/act/index.md">
        <img src="docs/practices/vla/act/figs/act_50k_success.gif" height="200" alt="ACT 在 ALOHA 仿真中完成双臂方块交接" />
      </a>
      <br/><sub><b><a href="docs/practices/vla/act/index.md">ACT · ALOHA 双臂交接</a></b><br/>动作分块模仿学习 · MuJoCo 闭环评测</sub>
    </td>
    <td align="center" width="33%">
      <img src="assets/rebot_act_training.gif" height="200" alt="ReBot-Act 机械臂 ACT 策略真机抓取与放置" />
      <br/><sub><b>ReBot-Act · 真机视觉模仿学习</b><br/>真实机械臂 · 方块抓取与放置</sub>
    </td>
  </tr>
</table>

> 更多四足、双足、机械臂与 AMD ROCm 实战请浏览 **[项目实战目录 →](https://datawhalechina.github.io/dive-into-embodied-ai/practices)**；无需配置环境即可在 **[在线仿真 →](https://datawhalechina.github.io/dive-into-embodied-ai/simulation)** 试玩 Unitree Go2、Unitree G1 / H1 / H1-2 人形、Pupper v3、MicroDuck 与 Quaddle（7 款机器人），并体验实时外力推撞恢复。

<a id="内容大纲"></a>

## 🗂️ 内容大纲

教程按网站顶部的四个栏目组织，推荐在网页版阅读以获得全文搜索、交互演示与一键复制 Markdown：

| 栏目 | 适合什么时候看 | 核心内容 | 入口 |
| :--- | :--- | :--- | :--- |
| **🧭 具身导论** | 刚入门，先建全局地图 | 什么是具身智能、发展脉络、技能与载体、核心挑战、技术栈、就业岗位与领域资源 | [开始阅读](https://datawhalechina.github.io/dive-into-embodied-ai/docs/introduction/intro) · [本地目录](docs/introduction/intro.md) |
| **📐 理论基础** | 带着问题查原理、补底座 | 大脑（RL / VLA / World-Model）、小脑（控制器 / 运动规划）、感官（VLM / 定位触觉 / 标定）、工程（仿真 / ROS 2 / 机械结构 / CAN & MCU） | [理论目录](https://datawhalechina.github.io/dive-into-embodied-ai/foundations) · [技能树](docs/foundations/intro.md) |
| **🛠️ 项目实战** | 跑通代码、解决真实问题 | **Unitree G1 人形运控**、四足（CS123 / Go2 MJX / Quaddle）、双足与轮腿（MicroDuck / Flamingo）、**移动操作（SLAM / Nav2 / VLN）**、机械臂与 VLA、SO-101 真机与 AMD ROCm 专区 | [项目目录](https://datawhalechina.github.io/dive-into-embodied-ai/practices) · [实战总览](docs/practices/intro.md) |
| **🎮 在线仿真** | 零配置直接上手体验 | Go2 / G1・H1・H1-2 人形 / Pupper v3 / MicroDuck 强化学习行走与外力推撞测试、Quaddle 步态对照（见 [模型库说明](simulation/README.md)） | [立即体验](https://datawhalechina.github.io/dive-into-embodied-ai/simulation) |

<details open>
<summary>重点章节与全量实战直达链接</summary>

- **具身导论**：[1. 什么是具身智能](docs/introduction/1.what-is-embodied-ai.md) · [2. 发展脉络](docs/introduction/2.history.md) · [3. 机器人的技能](docs/introduction/3.tasks-and-skills.md) · [4. 具身智能的载体](docs/introduction/4.embodiments.md) · [5. 核心挑战](docs/introduction/5.challenges.md) · [6. 技术栈](docs/introduction/6.tech-stack.md) · [7. 就业岗位](docs/introduction/7.careers.md) · [领域资源](docs/introduction/resources/index.md)
- **理论基础**：[强化学习](docs/foundations/rl-for-robotics/1.intro.md) · [VLA 大模型](docs/foundations/vla/vla-intro.md) · [World-Model](docs/foundations/world-model/0.intro.md) · [控制器](docs/foundations/controllers/intro.md) · [VLM 与视觉](docs/foundations/vlm/0.intro.md) · [定位与触觉感知](docs/foundations/perception/placeholder.md) · [传感器标定与 sim2real](docs/foundations/perception/1.sensor-calibration-sim2real.md) · [仿真工具](docs/foundations/simulation/1.intro.md) · [ROS 2](docs/foundations/robotics-and-ros2/0.intro.md) · [机械结构入门](docs/foundations/hardware/placeholder.md) · [CAN 与 MCU 通信](docs/foundations/communication/can-mcu.md)
- **人形与双足**：[Unitree G1 人形运控实战（4 篇）](docs/practices/humanoid/unitree-g1/index.md) · [MicroDuck RL & 跳墙课程](docs/practices/humanoid/microduck-rl/index.md) · [Flamingo 两轮足](docs/practices/humanoid/flamingo-isaaclab/0.intro.md) · [双足平衡与抗扰](docs/practices/humanoid/balance-control/placeholder.md) · [全身运动跟踪](docs/practices/humanoid/motion-tracking/placeholder.md) · [双足任务规划](docs/practices/humanoid/task-planning-demo/placeholder.md)
- **四足与移动操作**：[CS123 四足（8 章）](docs/practices/quadruped/cs123/0.intro.md) · [Go2 运控 MJX（ROCm / CUDA）](docs/practices/amd/unitree-rl-mjx/index.md) · [Quaddle 仿真与 VLA 导航](docs/practices/quadruped/quaddle-mujoco/index.md) · [Sim2Sim 指南](docs/practices/quadruped/sim2sim/placeholder.md) · [Sim2Real 指南](docs/practices/quadruped/sim2real-guide/placeholder.md) · [移动操作（SLAM / Nav2 / VLN / 协同）](docs/practices/mobile-manipulation/placeholder.md)
- **机械臂、VLA 与 AMD 专区**：[π₀.₅ + RECAP](docs/practices/vla/recap/index.md) · [ACT 双臂](docs/practices/vla/act/index.md) · [ROS 2 机械臂控制](docs/practices/robot-arm/ros2-arm-control/placeholder.md) · [自定义模仿学习](docs/practices/robot-arm/imitation-learning/placeholder.md) · [LeRobot + SO-101 真机](docs/practices/robot-arm/data-collection/so101-lerobot-real/index.md) · [AMD ROCm 专区](docs/practices/amd/intro.md)

</details>

<a id="组队学习"></a>

## 👥 组队学习

Datawhale 会围绕本教程定期组织公开组队学习与共学打卡，每期路线、导读与资料会陆续沉淀到仓库中。

- **获取开营通知**：可先 **Star / Watch** 本仓库，或关注文末 **Datawhale 公众号**获取最新一期报名信息。
- **现在就开始自学**：直接按上方三步路线推进，遇到问题或想建议新专题，欢迎在 [GitHub Issues](https://github.com/datawhalechina/dive-into-embodied-ai/issues) 交流。

<a id="本地预览"></a>

## 💻 本地预览

```bash
sh local-build.sh
```

打开终端显示的地址即可预览，按 `Ctrl+C` 停止。详细说明见[贡献指南](CONTRIBUTING.md#本地预览)。

<a id="star-history"></a>

## ⭐ Star 曲线

<p align="center">
  <a href="assets/star-history.svg">
    <img src="assets/star-history.svg" width="900" alt="Dive into Embodied AI Star History Chart" />
  </a>
</p>

<a id="如何贡献"></a>

## 🙌 如何贡献

欢迎任何形式的参与——无论你是刚入门的读者，还是有仿真、算法或真机经验的开发者：

- **纠错与建议**：发现公式、代码、链接错误或表述不清，欢迎直接提 [Issue](https://github.com/datawhalechina/dive-into-embodied-ai/issues) 或发起 PR。
- **补充教程与实战**：欢迎认领占位章节，或提交新的仿真实验、论文复现、真机部署案例与在线仿真模型。
- **完善多语言翻译**：协助同步英文（`i18n/en/`）与意大利语（`i18n/it/`）译文。

### 提交流程

1. Fork 本仓库并克隆到本地，按 [贡献指南](CONTRIBUTING.md) 安装 Git LFS 并拉取大文件（`git lfs install && git lfs pull`）。
2. 使用 `.nvmrc` 指定的 Node.js 版本，运行 `sh local-build.sh` 或 `npm run dev` 本地预览修改。
3. 新增位图请转为 WebP（`npm run assets:webp`），提交前运行 `npm run assets:webp:check` 与 `npm run build` 确保构建和检查通过。
4. 向 `master` 分支发起 Pull Request，简要说明改动内容与验证结果；完整规范与环境配置见 [CONTRIBUTING.md](CONTRIBUTING.md)。

<a id="贡献者名单"></a>

## 🤝 贡献者名单

### 核心贡献者

* 江季 [@johnjim0816](https://github.com/johnjim0816)，[蘑菇书](https://github.com/datawhalechina/easy-rl)作者，强化学习算法研究员。
* 黄潇 [@huang-xiao-01](https://github.com/huang-xiao-01)，同济大学毕业，智驾算法工程师。
* 孙成超 [@scchy](https://github.com/scchy)，Quaddle VLA 导航训练与仿真教程贡献者。

### 其他贡献者

* 康博 [@bokang-ugent](https://github.com/bokang-ugent)，nobl.ai 联合创始人 & 比利时根特大学访问教授。
* 罗如意 [@ruyiluo](https://github.com/ruyiluo)，智能汽车竞赛国奖 & FunRec 开源项目负责人。
* YirongZheng [@Yirzzzz](https://github.com/Yirzzzz)，LIBERO 长时程任务上的 RECAP 复现与奖励函数可视化贡献者。
* Ariestar [@Ariestar](https://github.com/Ariestar)，MDP 章节公式排版与 Docusaurus 提示块修复贡献者。

<a id="关注我们"></a>

## 📣 关注我们

<div align=center>
<p>扫描下方二维码关注公众号:Datawhale</p>
<img src="https://raw.githubusercontent.com/datawhalechina/pumpkin-book/master/res/qrcode.jpeg" width = "180" height = "180">
</div>

<a id="license"></a>

## 📄 LICENSE

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="知识共享许可协议" style="border-width:0" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a><br />本作品采用<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">知识共享署名-非商业性使用-相同方式共享 4.0 国际许可协议</a>进行许可。
