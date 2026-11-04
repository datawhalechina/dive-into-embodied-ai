---
title: "MicroDuck RL：教小黄鸭学会一个动作"
sidebar_position: 2
displayed_sidebar: practicesHumanoidSidebar
description: "看动作、玩奖励实验，再亲手启动一次 PPO 训练，让小黄鸭从试错中学习。"
---

import {ActionGallery, TrainingLoop, RewardLab} from '@site/src/components/docs/practices/MicroDuckLesson';

# MicroDuck RL：教小黄鸭学会一个动作

如果不给小黄鸭写好每一步动作，只告诉它“这样做可以得分”，它能学会站起来、转向，甚至踢球吗？先选一段视频，看看它的答案。

[先看动作 ↓](#先看你想教它哪一招) · [动手训练 ↓](#动手启动你的第一次训练) · [玩奖励实验 ↓](#你来设计奖励)

## 先看：你想教它哪一招？

下面是仿真中训练后策略的回放。点击播放，再展开视频下的问题。三个动作来自各自独立的策略。

<ActionGallery />

这一课，我们选择**平地走路**作为动手任务：目标容易理解，也方便改变速度、观察效果。上面的起身和踢球，是理解了训练流程之后可以继续探索的技能。

| 你的路线 | 需要什么 | 完成后你能做到 |
| --- | --- | --- |
| 先体验 | 浏览器即可 | 看懂训练闭环，完成奖励互动与观察题 |
| 动手训练 | Linux、NVIDIA GPU、已安装 Git 和 uv | 启动一次 PPO 训练，保存并回放自己的策略 |
| 继续挑战 | 完成第一次训练 | 修改一个奖励项，用视频与指标比较变化 |

不需要先掌握 PPO 的全部公式。先弄清楚三个问题：**它能感到什么？它能控制什么？什么行为能得分？**

## 小黄鸭不是在照着视频跳舞

想象你在教一个刚学走路的小朋友：给出“向前走”的目标，让它尝试，再根据结果反馈。这里，反馈由仿真环境中的奖励函数计算。

| 概念 | 在走路任务里是什么 | 你可以这样理解 |
| --- | --- | --- |
| 观测 observation | 身体姿态相关信息、关节角度和速度、目标命令等 | 我现在怎么样，要往哪里走？ |
| 动作 action | 给关节设置目标位置，由执行器模型产生运动 | 接下来腿和身体怎么动？ |
| 奖励 reward | 跟上目标速度、保持姿态、减少抖动等信号 | 这次尝试离目标更近了吗？ |
| 策略 policy | 将观测映射到动作的神经网络 | 小黄鸭逐渐学会的动作习惯 |
| checkpoint | 保存下来的模型及训练状态 | 把当前学到的东西存个档 |

这里的走路策略不读取视频画面来模仿。视频是训练后的展示；MuJoCo 计算物理运动，mjlab 组织环境，PPO 利用尝试得到的经验更新策略。

<TrainingLoop />

**注意“走一步”和“学一次”的区别：**策略会连续输出很多步动作；PPO 收集一批经验后才更新网络。多个并行环境，相当于让许多小黄鸭同时练习，再一起贡献经验。

## 你来设计奖励

先猜一下：如果只奖励“球走得远”，一个把球撞远、自己却倒地的小黄鸭，会不会得高分？下面用三个假想结果做个小实验。

<RewardLab />

试着完成两件事：让“踢球后站稳”得分最高；再把动作代价加大，看看什么时候“不动”反而更划算。

真实训练还涉及探索、整段轨迹的累计回报和终止条件。这个实验帮助你理解评分规则，不是在浏览器里训练完整机器人。

## 动手：启动你的第一次训练

以下命令使用 Linux Bash，在教程仓库中运行。本课使用 NVIDIA CUDA 路线；没有对应 GPU，可以先完成上面的体验路线。首次安装需要下载较大的依赖包，请预留网络和磁盘空间。

### 1. 准备环境：确认能找到小黄鸭

先获取[教程仓库](https://github.com/datawhalechina/dive-into-embodied-ai)，进入仓库根目录，再执行：

```bash
cd codes/practices/humanoid/microduck-rl
uv sync --locked
uv run python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
uv run list-envs
```

**成功信号：**CUDA 检查输出 `True`，任务列表里能找到 `Mjlab-Velocity-Flat-MicroDuck`。后续命令都在这个代码目录下执行。依赖版本已经写入锁文件，首次运行先保持原配置。

:::info[我能直接下载视频里的模型吗？]

当前教程仓库提供代码和演示视频，没有随页面分发这些精选视频对应的 checkpoint。你可以先看视频，再按下面的步骤训练自己的模型。如果课程提供了模型文件，需要确认它对应的任务和代码版本后再播放。

:::

### 2. 先跑 5 轮，检查整条链路

```bash
WANDB_MODE=offline uv run train Mjlab-Velocity-Flat-MicroDuck \
  --env.scene.num-envs 64 \
  --env.seed 42 --agent.seed 42 \
  --agent.max-iterations 5 \
  --agent.logger tensorboard \
  --agent.experiment-name tutorial_smoke \
  --agent.upload-model False
```

这一步会创建仿真环境、采样经验并执行 PPO 更新。**成功信号：**终端出现训练迭代信息，程序正常结束，并在 `logs/rsl_rl/tutorial_smoke/` 下留下训练记录。5 轮是检查能否运行，还不足以期待漂亮的步态。

| 你遇到的现象 | 先检查什么 |
| --- | --- |
| 找不到 `uv` | 先安装 uv，再执行环境同步 |
| CUDA 检查为 `False` | GPU 是否对当前环境可见、驱动与 Torch 是否匹配 |
| 显存不足 / OOM | 将 `num-envs` 从 64 降到 32 或 16，再做短训练检查 |
| 找不到任务 | 是否在正确代码目录、是否完成 `uv sync --locked` |
| 出现 NaN / Inf | 先保留报错和配置，检查模型、观测及奖励中的异常值 |

### 3. 正式练习：给它更多尝试机会

确认短训练通过后，再运行下面的单任务训练。它会**从零开始一个新实验**，并不自动接着前面的 5 轮训练。

```bash
WANDB_MODE=offline uv run train Mjlab-Velocity-Flat-MicroDuck \
  --env.scene.num-envs 64 \
  --env.seed 42 --agent.seed 42 \
  --agent.max-iterations 1500 \
  --agent.save-interval 100 \
  --agent.logger tensorboard \
  --agent.experiment-name tutorial_walk \
  --agent.run-name first_walk \
  --agent.upload-model False
```

1500 轮是这次练习的观察预算，不是保证学会的门槛。实际效果和耗时取决于配置、硬件及训练过程。每 100 轮保存一次模型，方便之后回看它的成长。

想看训练曲线，在另一个终端进入相同代码目录运行：

```bash
uv run tensorboard --logdir logs/rsl_rl/tutorial_walk --port 6006
```

在运行训练的机器上打开 `http://localhost:6006`；远程训练时先转发 6006 端口。先看平均回报是否变化、episode 长度是否变化，以及是否出现数值异常。**奖励变高之后，还要看视频里它到底学会了什么。**

### 4. 打开自己的模型存档

训练结束后，先找到最近一次练习里编号最大的模型：

```bash
CKPT=$(find logs/rsl_rl/tutorial_walk -type f -name 'model_*.pt' | sort -V | tail -n 1)
printf '%s\n' "$CKPT"
test -f "$CKPT" && echo "模型文件已找到"
```

**成功信号：**输出一个真实存在的 `.pt` 文件路径和“模型文件已找到”。如果为空，先查看训练是否成功保存；不要继续执行回放命令。下面的命令需在同一个终端运行，以保留 `CKPT` 变量。

有图形桌面时，打开交互窗口：

```bash
uv run play Mjlab-Velocity-Flat-MicroDuck \
  --checkpoint-file "$CKPT" \
  --num-envs 1
```

远程机器没有桌面时，可以输出一段视频：

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
uv run python scripts/render_mjlab_checkpoint.py \
  Mjlab-Velocity-Flat-MicroDuck \
  --checkpoint "$CKPT" \
  --mp4 /tmp/my-first-walk.mp4 \
  --frames 200 --lin-vel-x 0.15
```

**成功信号：**生成 `/tmp/my-first-walk.mp4`。将视频下载到自己的电脑播放，观察它是否向前移动、是否频繁倒地、动作是否抖动。离屏渲染仍需要可用的 GPU/EGL 环境；“没有桌面”不等于“不需要显卡”。

## 给小黄鸭做一本成长相册

别只保存最好看的那一段。把同一次训练的早期、中期、后期模型分别回放，才能看到学习发生在哪里。

1. 到 `logs/rsl_rl/tutorial_walk/` 中找到本次运行目录，选择三个实际存在的 `model_*.pt`，记录各自编号。
2. 将 `CKPT` 设置成其中一个文件的路径，用同一渲染命令生成视频；输出文件分别命名为 `early.mp4`、`middle.mp4`、`late.mp4`，避免覆盖。
3. 保持任务、速度、seed 和镜头一致，填写下面的观察表。轮数以真实文件为准，不假定特定编号一定存在。

| 观察问题 | 早期 | 中期 | 后期 |
| --- | --- | --- | --- |
| 能否持续保持身体直立？ | 记录现象 | 记录现象 | 记录现象 |
| 是否向目标方向移动？ | 记录现象 | 记录现象 | 记录现象 |
| 是否有明显抖动或倒地重置？ | 记录现象 | 记录现象 | 记录现象 |

本页现有精选素材来自不同任务，不把它们拼成同一个任务的“成长过程”。你的三段同条件视频，才是这次练习自己的成长相册。

<details>
<summary>为什么不能只比较总奖励？</summary>

奖励由多个项组合而成。修改权重之后，总分的刻度也变了，两个实验的总奖励可能不再能直接比较。保留固定命令下的位移、速度误差、倒地情况等任务指标，再配合视频解释结果。

</details>

## 闯关：从会运行，到会设计

### 第 1 关：倒着走会怎样？

将离屏回放命令中的 `--lin-vel-x 0.15` 改成 `--lin-vel-x -0.1`，换一个输出文件名。先预测动作，再比较视频。

<details>
<summary>提示：这里有没有重新训练？</summary>

没有。你使用的是同一份 checkpoint，改变的是输入命令。它能否后退，取决于训练经历和策略的泛化能力。

</details>

### 第 2 关：少抖一点，还是走快一点？

在 `src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py` 中查找 `action_rate_l2` 以及相关 curriculum（课程调度）。看清这个奖励项和它的权重如何变化，再设计一个只改变该项的实验。

<details>
<summary>提示：为什么不能只改一个数字就宣布实验成立？</summary>

课程调度可能在训练中再次修改权重。先确认实际生效的配置；新实验使用不同的 `experiment-name`，保留原版，其他条件尽量相同。改变奖励需要重新训练，不能仅通过回放旧模型检验。

</details>

### 第 3 关：交一份小黄鸭实验报告

交付两段同条件视频、一张比较表和三句话：**改了什么？观察到什么？下一步如何验证？** 如果动作退步，解释失败也算一次有价值的实验。

## 社区进阶案例：离地就算跳跃成功了吗？

先暂停在起跳前，预测接下来会发生什么；再播放视频，观察身体翻转、落地和后续运动。

<video
  controls
  playsInline
  preload="none"
  style={{width: '100%', maxWidth: '960px'}}
  poster={require('./figs/community/sun-chengchao-obstacle-poster.webp').default}
  aria-label="孙成超提供的 MicroDuck 障碍场景动作演示"
>
  <source src={require('./figs/community/sun-chengchao-obstacle-demo.mp4').default} type="video/mp4" />
  你的浏览器不支持视频播放，请使用下方 MP4 链接。
</video>

**作者：孙成超，经作者许可转载。** [打开 MP4](./figs/community/sun-chengchao-obstacle-demo.mp4)。视频约 24 秒，来自作者的训练演示，供本节观察与讨论；不是上方单任务命令的运行结果。教学副本保留完整视频画面，移除音轨，未改变播放速度。

### 把“动作好看”拆成可检查的阶段

| 阶段 | 看视频时记录什么 | 要验证结论，还需要什么 |
| --- | --- | --- |
| 起跳 | 离地前后的姿态，以及画面中的速度变化 | 足端接触、竖直速度的定义和采样时刻 |
| 越障 / 翻转 | 机器人相对障碍的位置，身体是否发生翻转 | 障碍几何、完整轨迹和碰撞记录 |
| 落地 | 是否恢复支撑，是否出现明显失衡 | 接触状态、姿态和终止原因 |
| 继续运动 | 落地后是否继续运动，画面是否出现跳变 | episode/reset 记录，以及预先定义的完成条件 |

画面叠加的 `v`、`vz`、`z` 和 `AIR` 是作者的调试信息。字段的坐标系、单位和 `AIR` 判定逻辑需要结合作者代码确认；仅凭标签出现，不能认定越障或落地成功。当前案例提供视频，尚未附任务代码、checkpoint 与训练配置，因此不从画面反推奖励函数、训练耗时或成功率。

练习：挑选一次动作，记录“起跳前—空中—落地后”三个时刻。分别写下**能从画面确认的现象**和**还需要日志才能判断的事项**，再定义一条可测量的成功条件，例如“通过指定障碍、无碰撞，落地后保持支撑并持续前进指定时长”。条件中的距离、姿态容差与时长应在评估前确定。

<details>
<summary>只看 AIR 标签和最终站立画面，够不够？</summary>

不够。离地只说明满足了某种状态判定，最终站立也可能来自重置。要判断完整动作是否成功，需要把接触、碰撞、终止/reset 记录与视频时间对齐，并检查预先定义的任务条件。

</details>

## 继续深入：找到代码里的每一层

| 想了解什么 | 从哪里开始 |
| --- | --- |
| 任务如何注册 | `src/mjlab_microduck/tasks/__init__.py` |
| 走路的观测、奖励和课程 | `src/mjlab_microduck/tasks/microduck_velocity_env_cfg.py` |
| 奖励和观测如何计算 | `src/mjlab_microduck/tasks/mdp.py` |
| 电机如何响应目标位置 | `src/mjlab_microduck/actuator/` |
| 模型如何变成视频 | `scripts/render_mjlab_checkpoint.py` |
| 动作如何串成链（跑→跳→越墙→滚翻） | [跳墙课程](./wall-curriculum.md) + `codes/practices/humanoid/microduck-wall-curriculum/` |

<details>
<summary>进阶：PPO 在优化什么？</summary>

策略不是只追求当前一步的奖励，而是学习获得更好的长期回报。PPO 根据采样经验估计动作的优势：这个动作的结果，比当时预期的好多少？更新时，它限制策略变化过大，以减少一次更新把已有行为破坏掉的风险。

在代码里可以沿着“环境配置 → rollout 采样 → rsl_rl 的 PPO 更新”继续阅读。先完成一组对照实验，再带着问题看算法细节会更容易理解。

</details>

需要查模型来源、完整模式表、验收字段或创意动作？查看[实验记录与创意动作](./experiment-notes.md)。准备把单个动作串成动作链？进入[跳墙课程](./wall-curriculum.md)：12 步课程带小黄鸭从平地翻滚跑到 5cm 墙全链。这些材料供复核和拓展使用，不是完成本课的前置要求。
