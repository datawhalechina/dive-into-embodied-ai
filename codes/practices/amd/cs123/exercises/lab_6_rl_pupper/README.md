# Lab 6：你的第一只 RL Pupper (Your First RL Pupper)

教程 [§6.7](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/rl-gait#67-速度奖励实验) / [§6.8](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/rl-gait#68-能量惩罚实验) 各跑了一次"暴走"和"优雅"奖励对比，但**没把训练真正跑完、没保存 checkpoint、没录变速命令 demo**。Lab 6 的增量：在 CPU 上用 `stable-baselines3` PPO 端到端训出一个能跟随变速命令的策略，拿到自己的 `pupper_ppo.zip`。

训完之后你手里第一次真正持有"自己训出来的策略文件"——Lab 7 的 LLM agent 会把它当 `walk` 工具调用。

## 为什么选这个任务

| 候选 | 教程覆盖 | 门槛 | 作品价值 |
|---|---|---|---|
| Reward tuning 遍历 | [§6.4](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/rl-gait#64-奖励设计)–[§6.8](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/rl-gait#68-能量惩罚实验) 已覆盖 | 低 | 中（图表） |
| Domain randomization | [§6.5](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/rl-gait#65-域随机化) | 中 | 中 |
| **端到端跑通 + 变速命令演示** | ❌ | 低 | **高（1 zip + 1 GIF）** |
| MJX 加速 | 提过 | 高（JAX） | 中 |

端到端跑通赢在：它是前面所有 RL 讨论的必经终点。跑到保存模型 + 能做变速 demo，才算真正"过了 RL 这一关"。

## 你交什么

- `portfolio/pupper_ppo.zip`：SB3 checkpoint，约 1–3 MB
- `portfolio/rl_pupper_commands.gif`：命令序列 demo（仅你训出来的策略），1280×480，12 fps，≤ 2.0 MB
- `portfolio/comparison.gif`：side-by-side 对比——左：你训的 PPO；右：上游 `test_policy.json`（GPU 训了数十亿步的 RTNeural 策略，作为"成功 baseline"对照）
- `portfolio/reward_curve.png`：训练曲线三栏图
- `portfolio/velocity_tracking.png`：3 档命令的速度跟踪曲线
- `portfolio/notes.md`：50–100 字反思

## 起点与 TODO map

当前仓库保留教师版 `envs/pupper_env.py` 和 `starter.py`，方便直接运行测试与生成素材。学生练习版只需要补环境里的三处 TODO；对应提示保留在 `starter_todo.py` 的 TODO 1–3。

| TODO | task | what to write |
|---|---|---|
| TODO 1 | `_compute_reward()` | 18 项加权和（tracking 3 + 稳定性 3 + 能耗 4 + 步态 4 + 终止/接触 4），penalty 项 term 返回正幅值、权重写负；最终 `reward = sum(scale × term) × dt` |
| TODO 2 | `_get_obs()` | 36 维 / step × 15 帧 stack = 540 维：IMU(6) + cmd(3) + desired_z(3) + 关节角差(12) + last_act(12)，含 IMU latency 与观测噪声 |
| TODO 3 | `_sample_command()` | `vx ∈ [-0.75, 0.75]`，`vy ∈ [-0.5, 0.5]`，`wz ∈ [-2.0, 2.0]`，1% 概率回零命令 |

## 分步任务

1. **环境拼接（30 min）**：读懂 `PupperEnv.__init__` 里的 PD 残差设定，补 `_get_obs`。（对应教程 [§6.2](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/rl-gait#62-gymnasium-环境)）
2. **奖励设计（30 min）**：补 `_compute_reward`，理解为什么 `r_torque` 权重是 `-2e-4`、`r_termination` 是 `-100`。（对应教程 [§6.4](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/rl-gait#64-奖励设计)）
3. **命令分布（10 min）**：补 `_sample_command`，注意要让 `wz` 范围足够宽，否则学不出转向。（对应教程 [§6.2](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/rl-gait#62-gymnasium-环境)）
4. **训练（数小时 CPU）**：`uv run python lab_6_rl_pupper/train_ppo.py`，看 tensorboard。150M 步在 16-core CPU 上约 8 小时是个起点；要看到学习信号，2-5M 步就够。如果训完 150M `ep_len_mean` 仍在涨，再续 50-150M。（对应教程 [§6.6](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/rl-gait#66-ppo-训练脚本)）
5. **评估录屏（20 min）**：`uv run python lab_6_rl_pupper/eval_commands.py`。（对应教程 [§6.9](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/rl-gait#69-评估与录屏)）
6. **画图（10 min）**：reward_curve + velocity_tracking 自动生成。

## MuJoCo scene

复用 Lab 4 / Lab 5 验证过的 `shared/models/pupper_v3_floating.xml`（浮基 + 棋盘地板 + skybox + spotlight + tracking_cam）。不另起 MJCF。

## Rubric

- `observation_space.shape == (540,)`，`action_space.shape == (12,)`
- `reset()` 后 obs 全部 finite
- `step(zeros)` 时所有 18 项 `r_*` 都在 info 里；tracking 项 ≥ 0，penalty 项 ≤ 0
- `REWARD_WEIGHTS` 18 个 key 齐全；`_sample_command()` 200 次采样落在声明范围内
- GIF 视觉判据：命令切换时身体方向 / 速度肉眼能跟上

## 常见坑

- 权重的**符号约定**容易混淆：本 lab 的 penalty term 函数返回正幅值（如 `r_torques = Σ τ²`），权重需要写负数才能起惩罚作用；tracking term 用 `exp(-err²/σ²)` 返回 [0, 1]，权重写正。把符号弄反等于"奖励翻车"。
- 不要忘了 `last_action` 进 obs，少了它策略高频抖动。
- 命令分布太窄学不出转向（比如 `wz ∈ [-0.6, 0.6]` 训出来的策略在 ±2.0 命令下完全跟不上）。
- `SubprocVecEnv` 在 macOS 起进程要 `if __name__ == "__main__":` 包好。
- 训练前几百万步 `ep_rew_mean` 看起来很差是正常的；真正的指标是 `eval/avg_episode_length` 何时饱和到 1000（不再跌倒）。

## 教程衔接

- **复用**：教程 [§6.2](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/rl-gait#62-gymnasium-环境) 环境封装、[§6.3](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/rl-gait#63-动作设计) PD 残差、[§6.4](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/rl-gait#64-奖励设计) 奖励配置、[§6.6](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/rl-gait#66-ppo-训练脚本) PPO 训练脚本、[§6.9](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/rl-gait#69-评估与录屏) 评估结构。
- **扩展**：把"训练一个会走的策略"完整跑完一次；第一次获得可被 Lab 7 调用的 checkpoint。
- **不重复**：教程 [§6.7](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/rl-gait#67-速度奖励实验) 暴走 vs [§6.8](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/rl-gait#68-能量惩罚实验) 优雅对比作为前置练习。

## 不做什么

- 不做完整地形组奖励（foot clearance 等）
- 作品集提交不要求 MJX / JAX / Brax——主线是纯 CPU 的 SB3，无 GPU 也能完成；想试 GPU 加速见文末「可选进阶」
- 不做 sim2real / 真机 / 真相机
- 不开 mujoco viewer 录 GIF（统一 offscreen）
- 不重训 multiple seeds 做曲线带

## Run

从 `exercises/` 运行。**首次运行前先拉取上游 RTNeural policy**（约 38 MB，已 `.gitignore`，仅需一次；用来生成 `comparison.gif` 那个"成功 baseline"对照）：

```bash
bash shared/rl/fetch_policies.sh                   # 下载 test_policy.json 到 shared/rl/policies/
```

然后：

```bash
uv run python lab_6_rl_pupper/tests.py             # 4 条断言
uv run python lab_6_rl_pupper/train_ppo.py         # CPU 训练（默认 150M 步，~8 小时；
                                                   #  想先看是否学得动可以 --total-steps 5_000_000）
uv run python lab_6_rl_pupper/eval_commands.py     # 加载 ckpt 录 GIF + 画图（含上游对比）
uv run python lab_6_rl_pupper/make_artifacts.py    # 一键串起训练、GIF、对比图和曲线
```

## 可选进阶：MJX/brax GPU 训练

主线到此为止。如果你有 GPU、想跑到 Stanford 量级（8192 envs、200M 步、几小时
而非几十小时），可以走 `train_brax_ppo.py` + `envs/pupper_env_mjx.py` 的 MJX 路径。
这条路**完全可选**，不影响作品集，也不要求你用它产出任何东西。

jax / brax / mujoco-mjx 的版本互相牵制，曾经很难凑齐（jax 0.10 删了 brax 0.14.2
还在用的 `device_put_replicated`；降 jax 又和 flax/optax 冲突；torch 拉进的 CUDA 13
运行库在只支持 CUDA 12 的驱动上让 jax 回退 CPU）。一组锁好版本、在三类机器上验过
的依赖固化在 [`requirements-mjx.txt`](requirements-mjx.txt)，照它走即可，**装在独立
venv，不和主线 `.venv` 混淆**。下面所有命令和主线一样**在 `exercises/` 下执行**
（`.venv-mjx` 会落在 `exercises/.venv-mjx`）。

**NVIDIA 和 AMD 都支持**，用一键脚本自动识别，不用自己记该装哪套插件：

```bash
bash lab_6_rl_pupper/setup_mjx_env.sh               # 自动识别后端并建 .venv-mjx
.venv-mjx/bin/python lab_6_rl_pupper/train_brax_ppo.py --output portfolio/pupper_mjx
```

脚本会打印识别结果，并在装完自检 `jax.default_backend()`——回退到 CPU 时直接报错，
而不是让你训到一半才发现慢得离谱：

```
  detected: ROCm / gfx1151  -> rocm
  creating exercises/.venv-mjx (py3.12)
  self-check:
    jax.default_backend() = gpu  [rocm:0]
```

识别不准或想强制某条路时用 `--backend`：

| `--backend` | 适用 | 装的插件 |
| --- | --- | --- |
| `auto`（默认） | 按 `/dev/kfd`、`nvidia-smi` 探测 | 见下 |
| `rocm` | AMD GPU + 系统 ROCm 7.x | `jax-rocm7-pjrt` + `jax-rocm7-plugin` |
| `cuda12` | NVIDIA，驱动支持 CUDA 12 | `jax[cuda12]` + pjrt + plugin |
| `cuda13` | NVIDIA，驱动支持 CUDA 13（驱动 >=580） | `jax[cuda13]` + pjrt + plugin |
| `cpu` | 纯 CPU / macOS | 不装加速器插件 |

插件版本一律从 `requirements-mjx.txt` 里的 `jax==` 读出来，四个组件版本永远一致。
手动装法仍完整保留在 [`requirements-mjx.txt`](requirements-mjx.txt) 头部注释里。

**AMD/ROCm 两个和 CUDA 不对称的坑**：jax **没有** `rocm` extra（写 `jax[rocm]` 会被
静默忽略并告警，只能显式装 pjrt + plugin 两个包）；ROCm 插件只有 ~24 MB、**不自带
运行库**，依赖系统已装好的 ROCm 7.x（`/opt/rocm`），而 CUDA 那边是 pip 把整套
`nvidia-*` 运行库拉进来。系统 ROCm 缺失或大版本不符时 jax 会静默回退 CPU——这正是
脚本自检要拦的情况。

启动时的 `Failed to import warp` 是无害告警（mjx 探测可选 warp 后端、退回 JAX 后端），
不用理会；ROCm 上另有 `Could not determine PCIe bandwidth` 告警，核显没有独立 PCIe
链路，同样无害。

验证覆盖：**Ubuntu** 上跑通了 NVIDIA GPU（CUDA 12）与纯 CPU 两条路，**macOS** 上跑通了
纯 CPU 路（`reset`/`step` 均正常）；**AMD Ryzen AI MAX+ 395 / Radeon 8060S（gfx1151,
ROCm 7.13）** 上跑通了 ROCm 路——`train_brax_ppo.py` 无需任何改动即可在 GPU 上训练。
CUDA 13 路依赖可正常解析，但需驱动 >=580 的机器实跑确认。
