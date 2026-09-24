<div align="center">
    <img src="static/img/career.webp" width="100%" alt="Dive into Embodied AI banner" />
</div>

<h1 align="center">Dive into Embodied AI</h1>
<p align="center"><b>An open course for getting started and building a career in embodied AI</b></p>
<p align="center"><a href="README.md" lang="zh-Hans">中文</a> · <b>English</b></p>

> [!TIP]
> **📖 Read online: [Open the course →](https://datawhalechina.github.io/dive-into-embodied-ai/en/)**
>
> The website offers full-text search, chapter navigation, and interactive demos. **[Get started with the introduction →](https://datawhalechina.github.io/dive-into-embodied-ai/en/docs/introduction/intro)**.
>
> The README, homepage navigation, and introduction are available in English. Other tutorial chapters and standalone playgrounds are currently in Chinese. Use the language menu to switch between 中文 and English on the same page.

<p align="center">
  <a href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="License" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a>
  <img alt="Status" src="https://img.shields.io/badge/status-Alpha-orange" />
</p>

<h3 align="center">🤝 Supported by</h3>

<p align="center">
  <a href="docs/practices/amd/intro.md">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="assets/logo/logo_amd_wht.svg" />
      <source media="(prefers-color-scheme: light)" srcset="assets/logo/logo_amd.svg" />
      <img src="assets/logo/logo_amd.svg" width="220" alt="AMD University Program" />
    </picture>
  </a>
</p>

> [!CAUTION]
> **Alpha release:** Migration and restructuring are ongoing, and some chapters are placeholders. Please open an issue with feedback or suggestions.

## 🎯 About this project

Build an embodied AI robot from scratch. Explore practical implementations of reinforcement learning, world models, and vision-language-action models (VLA), alongside simulation, controllers, motion planning, and perception. Connect decision making, control, and perception in real projects.

🧭 **Get started: [Introduction](i18n/en/docusaurus-plugin-content-docs/current/introduction/intro.md)** — explore embodied AI concepts, tasks, platforms, and methods.

<a id="course-outline"></a>

## 🗂️ Course outline

The website has four sections: **Introduction**, **Foundations**, **Tutorials**, and **Projects**. The introduction gives an overview of the field and collects papers, datasets, open source projects, and tools; foundations explain the principles; tutorials follow a structured sequence of chapters; projects offer standalone demos and reproduction studies. Projects are grouped into AMD projects, simulation projects, and real robot projects.

Status labels: **✅ Available** means the content is ready to read; **🚧 Partially available** means some chapters are complete and others are placeholders; **🚧 Placeholder** means only a placeholder page exists; **⏳ Planned** means work has not started. These labels describe the Chinese source content.

<a id="introduction"></a>

### 🧭 Introduction

Start with the [introduction to embodied AI](i18n/en/docusaurus-plugin-content-docs/current/introduction/intro.md) to see the big picture, then move on to the foundations and tutorials for specific techniques:

| Chapter | Contents |
| :--- | :--- |
| [1. What is embodied AI](i18n/en/docusaurus-plugin-content-docs/current/introduction/1.what-is-embodied-ai.md) | Definition, the perception–decision–action loop, and why the body matters |
| [2. A brief history](i18n/en/docusaurus-plugin-content-docs/current/introduction/2.history.md) | Four periods: programmed rules, model-based control, deep reinforcement learning, and foundation models |
| [3. Robot skills](i18n/en/docusaurus-plugin-content-docs/current/introduction/3.tasks-and-skills.md) | Defining tasks; grasping, manipulation, locomotion, and navigation |
| [4. Embodied AI platforms](i18n/en/docusaurus-plugin-content-docs/current/introduction/4.embodiments.md) | Humanoids, arms, wheeled and legged robots, mobile manipulators, and simulated bodies |
| [5. Key challenges](i18n/en/docusaurus-plugin-content-docs/current/introduction/5.challenges.md) | Data, sim-to-real transfer, generalization, real-time execution, safety, and evaluation |
| [6. Technology stack](i18n/en/docusaurus-plugin-content-docs/current/introduction/6.tech-stack.md) | Introductions and learning links for VLMs, VLAs, world models, manipulation, motion control, navigation, and more |
| [7. Career roles](i18n/en/docusaurus-plugin-content-docs/current/introduction/7.careers.md) | Responsibilities and interview preparation for brain algorithms, motion control, perception and navigation, engineering platforms, and hardware |

The introduction's [resources](i18n/en/docusaurus-plugin-content-docs/current/introduction/resources/index.md) collect public learning and research materials:

- [Papers and research](i18n/en/docusaurus-plugin-content-docs/current/introduction/resources/papers.md): original papers, project pages, and explanations.
- [Embodied AI datasets](i18n/en/docusaurus-plugin-content-docs/current/introduction/resources/datasets.md): robot demonstrations and evaluation benchmarks.
- [Open source projects](i18n/en/docusaurus-plugin-content-docs/current/introduction/resources/open-source.md): model code and training frameworks.
- [Simulation and tools](i18n/en/docusaurus-plugin-content-docs/current/introduction/resources/tools.md): simulation engines, learning environments, and official documentation.

### 📚 Tutorials

Choose a learning path from the [tutorial overview](docs/tutorials/intro.md). Existing article and code paths remain unchanged.

| Tutorial | Learning path | Progress |
| :--- | :--- | :--- |
| [Build a quadruped from scratch](docs/practices/quadruped/cs123/0.intro.md) | MuJoCo, PD control, kinematics, policy training, language control, and perception | Overview and 8 chapters available |
| [LeRobot course notes in Chinese](docs/practices/robot-arm/data-collection/lerobot-course/index.md) | Robot learning, datasets, toolchains, and classical robotics | Units 0–2 compiled |
| [Flamingo wheeled-biped course](docs/practices/wheel-legged/flamingo-isaaclab/preview.md) | Isaac Lab training and cross-simulator validation | Course preview |

### 🛠️ Projects

| Category | Project | Description | Status |
| :--- | :--- | :--- | :--- |
| AMD | [AUP Learning Cloud](docs/practices/amd/aup-learning-cloud.md) | Ryzen AI APUs, ROCm, JupyterHub, and Code Server | ✅ Available |
| AMD | [MicroDuck RL · AMD ROCm](docs/practices/amd/microduck-rl/index.md) | R9700, ROCm MuJoCo Warp, dynamic-contact regression tests, and biped PPO | ✅ Available |
| AMD | [Explore the Pupper quadruped](docs/practices/amd/pupper-control/intro.md) | RL locomotion policies and VLA experiments on AMD hardware | ✅ Available |
| Simulation | [MicroDuck RL biped](docs/practices/humanoid/microduck-rl/index.md) | mjlab + MuJoCo Warp: parallel PPO and biped gait training on GPUs | ✅ Available |
| Simulation | [MuJoCo robot arms and DDPG](docs/practices/robot-arm/mujoco-arm-pick-place/index.md) | MuJoCo environments and continuous-control experiments | ✅ Available |
| Simulation | [ACT bimanual training](docs/practices/vla/act/index.md) | ACT + ALOHA: training, evaluation, and reproduction | ✅ Available |
| Real robots | [SO-101 + LeRobot hardware tutorial](docs/practices/robot-arm/data-collection/so101-lerobot-real/index.md) | Hardware connectivity, safety tests, and action playback | ✅ Available |
| Real robots | [Sim2Real guide](docs/practices/quadruped/sim2real-guide/placeholder.md) | Deploying simulation policies and validating on hardware | 🚧 Placeholder |

### 🦆 Latest demo: MicroDuck RL

<p align="center">
  <a href="docs/practices/humanoid/microduck-rl/index.md">
    <img src="docs/practices/humanoid/microduck-rl/figs/microduck-training-1500.webp" width="640" alt="MicroDuck biped stable gait playback" />
  </a>
  <br/>
  <sub>✅ <b><a href="docs/practices/humanoid/microduck-rl/index.md">MicroDuck RL · Stable biped locomotion</a></b><br/>mjlab + MuJoCo Warp · Parallel PPO training on GPUs (iteration 1500)</sub>
</p>

<table align="center">
  <tr>
    <td align="center" width="25%">
      <a href="https://datawhalechina.github.io/dive-into-embodied-ai/en/docs/practices/quadruped/cs123/intro">
        <img src="assets/lab5_forward_gait_comparison.gif" height="220" alt="CS123 quadruped gait comparison" />
      </a>
      <br/><sub>✅ <b><a href="https://datawhalechina.github.io/dive-into-embodied-ai/en/docs/practices/quadruped/cs123/intro">Build a quadruped from scratch</a></b><br/>CS123 simulation course · MuJoCo + PPO + LLM control</sub>
    </td>
    <td align="center" width="25%">
      <a href="docs/practices/wheel-legged/flamingo-isaaclab/preview.md">
        <img src="assets/hero_swarm.gif" height="220" alt="Flamingo wheeled biped training in Isaac Lab" />
      </a>
      <br/><sub>🔜 <b><a href="docs/practices/wheel-legged/flamingo-isaaclab/preview.md">Flamingo · Isaac Lab</a></b><br/>Course preview · Isaac Lab + PPO / CaT training</sub>
    </td>
    <td align="center" width="25%">
      <img src="assets/rebot_act_training.gif" height="220" alt="ReBot-Act robot arm using an ACT policy" />
      <br/><sub>✅ <b>ReBot-Act · ACT training results</b><br/>Visual imitation learning on hardware · Block pick-and-place</sub>
    </td>
    <td align="center" width="25%">
      <a href="docs/practices/vla/act/index.md">
        <img src="docs/practices/vla/act/figs/act_50k_success.gif" height="220" alt="ACT transferring a block between two arms in ALOHA simulation" />
      </a>
      <br/><sub>✅ <b><a href="docs/practices/vla/act/index.md">ACT · ALOHA bimanual transfer</a></b><br/>50k training · 50% success over 20 MuJoCo episodes</sub>
    </td>
  </tr>
</table>

### 📐 Foundations

The foundations follow the four groups in the site navigation: decision making, motion control, perception, and engineering foundations. Existing content is being integrated into this knowledge map; missing modules remain placeholders.

#### Decision making

| Topic | Description | Status |
| :--- | :--- | :--- |
| [Reinforcement learning for decisions](docs/foundations/rl-for-robotics/1.intro.md) | MDPs, DQN, PPO, SAC, DDPG/TD3, and imitation learning | ✅ Available |
| [Vision-language-action models (VLA)](docs/foundations/vla/vla-intro.md) | RT-1/RT-2, OpenVLA, ACT, Diffusion Policy, and the π family | ✅ Available |
| [World models](docs/foundations/world-model/0.intro.md) | Applying world models to embodied tasks | ✅ Available |

#### Motion control

| Topic | Description | Status |
| :--- | :--- | :--- |
| [Reinforcement learning for control](docs/foundations/rl-for-robotics/10.ppo.md) | Connecting policy learning to continuous control and robotics | ✅ Available |
| [Controllers](docs/foundations/controllers/intro.md) | PID, LQR, MPC, impedance control, and system integration | ✅ Available |
| [Motion planning](docs/foundations/robotics-and-ros2/10.moveit2_basics.md) | Motion planning and closed-loop planning with MoveIt 2 | ✅ Available |

#### Perception

Robots need to estimate their position, orientation, velocity, and stability. Relevant observations come from cameras, LiDAR, touch, motor currents, IMUs, foot contacts, body pose, and end-effector position.

| Topic | Description | Status |
| :--- | :--- | :--- |
| [Visual perception and VLMs](docs/foundations/vlm/0.intro.md) | Transformers, ViT, vision encoders, and multimodal fusion | ✅ Available |
| [Localization, touch, and sensor calibration](docs/foundations/perception/placeholder.md) | SLAM, foot contact, tactile sensing, multisensor fusion, and sim2real calibration | 🚧 Partially available |
| [Sensor calibration and sim2real](docs/foundations/perception/1.sensor-calibration-sim2real.md) | Coordinate frames, synchronization, extrinsic-error amplification, and online calibration monitoring | ✅ Available |

#### Engineering foundations

| Topic | Description | Status |
| :--- | :--- | :--- |
| [Simulation tools](docs/foundations/simulation/1.intro.md) | Getting started with Isaac Sim, MuJoCo, Gymnasium, and PyBullet | ✅ Available |
| [ROS2](docs/foundations/robotics-and-ros2/0.intro.md) | Coordinate transforms, FK/IK, tf2, URDF, and MoveIt 2 | ✅ Available |
| [CAN and MCU communication](docs/foundations/communication/can-mcu.md) | Low-level communication, actuator protocols, and host–controller links | 🚧 Placeholder |
| [Mechanical design](docs/foundations/hardware/placeholder.md) | Links, joints, motors, gearboxes, and end effectors | 🚧 Placeholder |
| [Data engineering and imitation learning](docs/foundations/rl-for-robotics/12.imitation-learning.md) | Teleoperation data, imitation learning, LeRobot toolchains, and policy training | ✅ Available |

## 👥 Study groups

Datawhale organizes study groups around this course. Past and upcoming study plans will be collected in `docs/team-learning/` (under construction), including learning paths, check-in requirements, and chapter introductions.

- Registration for the next group: under construction
- Previous study materials: under construction

## 💻 Local preview

This repository uses **Git LFS** for videos and GIFs. Install `git-lfs` and run `git lfs pull` after cloning; otherwise, media files will contain pointer text instead of the actual content. See [CONTRIBUTING.md](CONTRIBUTING.md#首次克隆必读) for the full setup guide in Chinese.

```bash
# 1. Install git-lfs once per machine
# brew install git-lfs        # macOS
# sudo apt install git-lfs    # Ubuntu / Debian
# choco install git-lfs       # Windows

# 2. Initialize LFS and download media
git lfs install
git lfs pull

# 3. Install dependencies and start the default Chinese site
npm install
npm run dev

# Start the English site instead
npm run start -- --locale en
```

The development server runs one language at a time. To test switching between both languages, build and serve the full site:

```bash
npm run build
npm run serve
```

Chinese remains at `/dive-into-embodied-ai/`; English is at `/dive-into-embodied-ai/en/`. UI translations live in `i18n/en/code.json`, and navigation/footer translations live in `i18n/en/docusaurus-theme-classic/`.

To translate a chapter later, add its English Markdown under `i18n/en/docusaurus-plugin-content-docs/current/`, keeping the source directory structure, document ID, and slug. Docusaurus displays the Chinese source with a translation notice when no English version exists.

## ⭐ Star history

<p align="center">
  <a href="assets/star-history.svg">
    <img src="assets/star-history.svg" width="900" alt="Dive into Embodied AI star history chart" />
  </a>
  <br />
  <sub>Updated automatically by GitHub Actions</sub>
</p>

## 🤝 Contributors

| Name | Role | Background |
| :--- | :--- | :--- |
| Jiang Ji (江季) | Project lead | Author of [Easy RL](https://github.com/datawhalechina/easy-rl), reinforcement learning researcher |
| Kang Bo (康博) | Project lead | Co-founder of nobl.ai and visiting professor at Ghent University, Belgium |
| Huang Xiao (黄潇) | Project lead | Tongji University graduate, intelligent-driving algorithm engineer |
| Luo Ruyi (罗如意) | Project lead | National award winner in the smart car competition and lead of the FunRec open source project |

## 📣 Follow us

<div align="center">
  <p>Scan the QR code to follow Datawhale on WeChat.</p>
  <img src="https://raw.githubusercontent.com/datawhalechina/pumpkin-book/master/res/qrcode.jpeg" width="180" height="180" alt="Datawhale WeChat QR code" />
</div>

## 📄 License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons license" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a>

This work is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/).
