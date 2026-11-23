<div align="center">
    <img src="static/img/embodied-ai-learning.webp" width="100%" alt="Embodied AI workbench with a simulation, robot arm, quadruped, and humanoid" />
</div>

<h1 align="center">Dive into Embodied AI</h1>
<p align="center"><b>An open course for learning embodied AI through hands-on projects</b></p>
<p align="center"><a href="README.md" lang="zh-Hans">中文</a> · <b>English</b> · <a href="README.it.md" lang="it">Italiano</a></p>

The README, homepage and navigation, introduction (including interactive demos and resources), learning map, project overview, and terms and formulas quick reference are available in English and Italian. Some other chapters, including MDPs and temporal difference, are also available in English; untranslated chapters and standalone playgrounds show a Chinese-content notice. Use the language menu to switch between 中文, English, and Italiano on the same page.

<p align="center">
  <a href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="License" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a>
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

> [!TIP]
> **📖 Read online: [Open the course →](https://datawhalechina.github.io/dive-into-embodied-ai/en/)**

## 🎯 About this project

An open-source course taking you from beginner concepts to real engineering in embodied AI. Centered on the real-world **perception–decision–control** closed loop, it connects perception and localization, RL / VLA / world models, planning and motion control, and simulation, data, and deployment—bringing theory into runnable code across simulated and physical robots.

## 🧭 Getting started

Embodied AI spans many disciplines—**you do not need to finish all the theory before building**. Follow three practical steps:

1. **Build a learning map first**: skim the [Introduction](i18n/en/docusaurus-plugin-content-docs/current/introduction/intro.md) to grasp core concepts, robot skills, platforms, and how the technology stack fits together (or jump straight into [Simulation](https://datawhalechina.github.io/dive-into-embodied-ai/en/simulation) in your browser).
2. **Study foundations with concrete questions**: pick an area that interests you and explore the matching chapters in [Foundations](https://datawhalechina.github.io/dive-into-embodied-ai/en/foundations) on demand.
3. **Solve real problems through projects**: run a full loop in [Projects](https://datawhalechina.github.io/dive-into-embodied-ai/en/practices)—reproducing, tuning, and debugging in simulation or on real hardware.

## 🎬 Featured project demos

<table align="center">
  <tr>
    <td align="center" width="33%">
      <a href="docs/practices/amd/microduck-rl/index.md">
        <img src="docs/practices/amd/microduck-rl/figs/microduck-amd-r9700-model750.gif" height="200" alt="MicroDuck biped stable walking gait playback" />
      </a>
      <br/><sub><b><a href="docs/practices/amd/microduck-rl/index.md">MicroDuck RL · Stable biped walking</a></b><br/>mjlab + MuJoCo Warp · Parallel PPO (ROCm / CUDA)</sub>
    </td>
    <td align="center" width="33%">
      <a href="docs/practices/humanoid/flamingo-isaaclab/0.intro.md">
        <img src="assets/hero_swarm.gif" height="200" alt="Flamingo wheel-legged robots running in parallel in Isaac Lab" />
      </a>
      <br/><sub><b><a href="docs/practices/humanoid/flamingo-isaaclab/0.intro.md">Flamingo · Wheel-legged swarm RL</a></b><br/>Isaac Lab · Dynamic balance &amp; PPO / CaT control</sub>
    </td>
    <td align="center" width="33%">
      <a href="docs/practices/quadruped/cs123/0.intro.md">
        <img src="docs/practices/quadruped/cs123/figs/lab8_ball_chase.gif" height="200" alt="CS123 quadruped visual perception and closed-loop ball chasing" />
      </a>
      <br/><sub><b><a href="docs/practices/quadruped/cs123/0.intro.md">Build a quadruped from scratch</a></b><br/>MuJoCo + RL gaits + closed-loop visual chase</sub>
    </td>
  </tr>
  <tr>
    <td align="center" width="33%">
      <a href="docs/practices/vla/recap/index.md">
        <img src="docs/practices/vla/recap/videos/state_030_success_value.gif" height="200" alt="π0.5 + RECAP long-horizon manipulation and value curve in LIBERO" />
      </a>
      <br/><sub><b><a href="docs/practices/vla/recap/index.md">π₀.₅ + RECAP · Long-horizon VLA</a></b><br/>LIBERO benchmark · Value modeling &amp; RL post-training</sub>
    </td>
    <td align="center" width="33%">
      <a href="docs/practices/vla/act/index.md">
        <img src="docs/practices/vla/act/figs/act_50k_success.gif" height="200" alt="ACT transferring a block between two arms in ALOHA simulation" />
      </a>
      <br/><sub><b><a href="docs/practices/vla/act/index.md">ACT · ALOHA bimanual transfer</a></b><br/>Action-chunking imitation · MuJoCo evaluation</sub>
    </td>
    <td align="center" width="33%">
      <img src="assets/rebot_act_training.gif" height="200" alt="ReBot-Act robot arm using an ACT policy on real hardware" />
      <br/><sub><b>ReBot-Act · Real-robot imitation</b><br/>Physical arm · Visual block pick-and-place</sub>
    </td>
  </tr>
</table>

> Browse more quadruped, biped, arm, and AMD ROCm tutorials in **[Projects →](https://datawhalechina.github.io/dive-into-embodied-ai/en/practices)**, or try Go2, Pupper v3, Quaddle, and MicroDuck right in your browser in **[Simulation →](https://datawhalechina.github.io/dive-into-embodied-ai/en/simulation)**.

<a id="course-outline"></a>

## 🗂️ Course outline

The course is organized around four top-level website sections (use the web version for full-text search, interactive demos, and one-click Markdown copy):

| Section | When to use it | What it covers | Links |
| :--- | :--- | :--- | :--- |
| **🧭 Introduction** | Start here for the big picture | Concepts, history, skills, platforms, challenges, technology stack, career roles, and curated resources | [Read online](https://datawhalechina.github.io/dive-into-embodied-ai/en/docs/introduction/intro) · [Repo index](i18n/en/docusaurus-plugin-content-docs/current/introduction/intro.md) |
| **📐 Foundations** | Look up principles as questions arise | Brain (RL / VLA / World Models), Cerebellum (controllers / planning), Sensing (VLM / calibration), Engineering (simulation / ROS 2 / imitation) | [Foundations hub](https://datawhalechina.github.io/dive-into-embodied-ai/en/foundations) · [Skill tree](docs/foundations/intro.md) |
| **🛠️ Projects** | Run code and solve concrete tasks | Quadruped, biped, and wheel-legged control, arm &amp; VLA reproductions, LeRobot + SO-101 hardware, and AMD ROCm projects | [Projects hub](https://datawhalechina.github.io/dive-into-embodied-ai/en/practices) · [Overview](i18n/en/docusaurus-plugin-content-docs/current/practices/intro.md) |
| **🎮 Simulation** | Try robots with zero setup | Go2 &amp; Pupper v3 PPO gaits, Quaddle periodic reference, MicroDuck free joint control (see [model guide](simulation/README.en.md)) | [Launch in browser](https://datawhalechina.github.io/dive-into-embodied-ai/en/simulation) |

<details>
<summary>Expand quick links to key chapters</summary>

- **Introduction**: [1. What is embodied AI](i18n/en/docusaurus-plugin-content-docs/current/introduction/1.what-is-embodied-ai.md) · [2. Brief history](i18n/en/docusaurus-plugin-content-docs/current/introduction/2.history.md) · [3. Robot skills](i18n/en/docusaurus-plugin-content-docs/current/introduction/3.tasks-and-skills.md) · [4. Platforms](i18n/en/docusaurus-plugin-content-docs/current/introduction/4.embodiments.md) · [5. Key challenges](i18n/en/docusaurus-plugin-content-docs/current/introduction/5.challenges.md) · [6. Technology stack](i18n/en/docusaurus-plugin-content-docs/current/introduction/6.tech-stack.md) · [7. Career roles](i18n/en/docusaurus-plugin-content-docs/current/introduction/7.careers.md) · [Resources](i18n/en/docusaurus-plugin-content-docs/current/introduction/resources/index.md)
- **Foundations**: [RL for robotics](docs/foundations/rl-for-robotics/1.intro.md) · [VLA models](docs/foundations/vla/vla-intro.md) · [World models](docs/foundations/world-model/0.intro.md) · [Controllers](docs/foundations/controllers/intro.md) · [Visual perception &amp; VLMs](docs/foundations/vlm/0.intro.md) · [Calibration &amp; sim2real](docs/foundations/perception/1.sensor-calibration-sim2real.md) · [Simulation tools](docs/foundations/simulation/1.intro.md) · [ROS 2](docs/foundations/robotics-and-ros2/0.intro.md) ([code](codes/foundations/robotics-and-ros2/README.en.md))
- **Projects**: [MicroDuck RL](docs/practices/humanoid/microduck-rl/index.md) · [Flamingo wheel-legged](docs/practices/humanoid/flamingo-isaaclab/0.intro.md) · [CS123 quadruped](docs/practices/quadruped/cs123/0.intro.md) · [Quaddle optimization](i18n/en/docusaurus-plugin-content-docs/current/practices/quadruped/quaddle-mujoco/optimization.md) · [π₀.₅ + RECAP](docs/practices/vla/recap/index.md) · [ACT bimanual](docs/practices/vla/act/index.md) · [SO-101 + LeRobot](docs/practices/robot-arm/data-collection/so101-lerobot-real/index.md) · [AMD zone](docs/practices/amd/intro.md)

</details>

## 👥 Study groups

Datawhale organizes study groups around this course. Past and upcoming study plans will be collected in `docs/team-learning/` (under construction), including learning paths, check-in requirements, and chapter introductions.

- Registration for the next group: under construction
- Previous study materials: under construction

## 💻 Local preview

```bash
sh local-build.sh
```

Open the URL printed in the terminal to preview the site; press `Ctrl+C` to stop. See the [contributing guide](CONTRIBUTING.md#本地预览) (in Chinese) for details.

## ⭐ Star history

<p align="center">
  <a href="assets/star-history.svg">
    <img src="assets/star-history.svg" width="900" alt="Dive into Embodied AI star history chart" />
  </a>
</p>

## 🙌 How to contribute

Contributions of all kinds are welcome—whether you are learning embodied AI for the first time or building simulation, algorithm, and hardware systems:

- **Fixes &amp; feedback**: spot an error in equations, code, links, or wording? Open an [Issue](https://github.com/datawhalechina/dive-into-embodied-ai/issues) or submit a PR.
- **Tutorials &amp; projects**: fill in placeholder chapters or contribute new simulation experiments, paper reproductions, real-robot guides, or browser simulation models.
- **Translations**: help keep the English (`i18n/en/`) and Italian (`i18n/it/`) docs synchronized.

### Contribution workflow

1. Fork and clone the repository, then install Git LFS and pull binary assets (`git lfs install && git lfs pull`).
2. Switch to the Node.js version in `.nvmrc` and preview locally with `sh local-build.sh` or `npm run dev`.
3. Use WebP for new raster images (`npm run assets:webp`), and run `npm run assets:webp:check` and `npm run build` before committing.
4. Open a Pull Request against `master` summarizing your changes and checks. See [CONTRIBUTING.md](CONTRIBUTING.md) (in Chinese) for full guidelines.

## 🤝 Contributors

### Core contributors

* Jiang Ji (江季) [@johnjim0816](https://github.com/johnjim0816), author of [Easy RL](https://github.com/datawhalechina/easy-rl), reinforcement learning researcher.
* Huang Xiao (黄潇) [@huang-xiao-01](https://github.com/huang-xiao-01), Tongji University graduate, intelligent-driving algorithm engineer.
* Sun Chengchao (孙成超) [@scchy](https://github.com/scchy), contributor to Quaddle VLA navigation training and simulation tutorials.

### Other contributors

* Kang Bo (康博) [@bokang-ugent](https://github.com/bokang-ugent), co-founder of nobl.ai and visiting professor at Ghent University, Belgium.
* Luo Ruyi (罗如意) [@ruyiluo](https://github.com/ruyiluo), national award winner in the smart car competition and lead of the FunRec open source project.
* YirongZheng [@Yirzzzz](https://github.com/Yirzzzz), contributor to RECAP reproduction on LIBERO long-horizon tasks and reward-function visualization.
* Ariestar [@Ariestar](https://github.com/Ariestar), contributor to MDP formula formatting and Docusaurus admonition fixes.

## 📣 Follow us

<div align="center">
  <p>Scan the QR code to follow Datawhale on WeChat.</p>
  <img src="https://raw.githubusercontent.com/datawhalechina/pumpkin-book/master/res/qrcode.jpeg" width="180" height="180" alt="Datawhale WeChat QR code" />
</div>

## 📄 License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons license" src="https://img.shields.io/badge/license-CC%20BY--NC--SA%204.0-lightgrey" /></a>

This work is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](http://creativecommons.org/licenses/by-nc-sa/4.0/).
