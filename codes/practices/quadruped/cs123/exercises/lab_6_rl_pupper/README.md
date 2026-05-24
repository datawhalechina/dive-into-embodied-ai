# Lab 6：你的第一只 RL Pupper (Your First RL Pupper)

教程 [§6.7](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/rl-gait#67-速度奖励实验) / [§6.8](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/rl-gait#68-能量惩罚实验) 各跑了一次"暴走"和"优雅"奖励对比，但**没把训练真正跑完 2M 步、没保存 checkpoint、没录变速命令 demo**。这就是 Lab 6 的增量：在 CPU 上用 `stable-baselines3` PPO 端到端训出一个能跟随变速命令的策略，拿到你自己的 `pupper_ppo.zip`。

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
- `portfolio/comparison.gif`：side-by-side 对比 GIF——左：你训的 PPO，右：上游 `test_policy.json`
- `portfolio/reward_curve.png`：训练曲线三栏图
- `portfolio/velocity_tracking.png`：3 档命令的速度跟踪曲线
- `portfolio/notes.md`：50–100 字反思

## 起点与 TODO map

当前仓库保留教师版 `envs/pupper_env.py` 和 `starter.py`，方便直接运行测试与生成素材。学生练习版只需要补环境里的三处 TODO；对应提示保留在 `starter_todo.py` 的 TODO 1–3。

| TODO | task | what to write |
|---|---|---|
| TODO 1 | `_compute_reward()` | 6 项加权和：vel / alive / torque / action_rate / ori / height |
| TODO 2 | `_get_obs()` | 49 维向量拼接，顺序和 `observation_space` 对齐 |
| TODO 3 | `_sample_command()` | 命令分布 `vx / vy / ωz` 三个均匀分布范围 |

## 分步任务

1. **环境拼接（30 min）**：读懂 `PupperEnv.__init__` 里的 PD 残差设定，补 `_get_obs`。（对应教程 [§6.2](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/rl-gait#62-gymnasium-环境)）
2. **奖励设计（30 min）**：补 `_compute_reward`，理解为什么 `r_torque` 系数是 2e-4。（对应教程 [§6.4](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/rl-gait#64-奖励设计)）
3. **命令分布（10 min）**：补 `_sample_command`，选合理范围。（对应教程 [§6.2](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/rl-gait#62-gymnasium-环境)）
4. **训练（30–60 min）**：`uv run python lab_6_rl_pupper/train_ppo.py`，看 tensorboard。（对应教程 [§6.6](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/rl-gait#66-ppo-训练脚本)）
5. **评估录屏（20 min）**：`uv run python lab_6_rl_pupper/eval_commands.py`。（对应教程 [§6.9](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/rl-gait#69-评估与录屏)）
6. **画图（10 min）**：reward_curve + velocity_tracking 自动生成。

## MuJoCo scene

复用 Lab 4 / Lab 5 验证过的 `shared/models/pupper_v3_floating.xml`（浮基 + 棋盘地板 + skybox + spotlight + tracking_cam）。不另起 MJCF。

## Rubric

- `observation_space.shape == (49,)`，`action_space.shape == (12,)`
- `reset()` 后 obs 全部 finite；gravity z 分量 ≤ -0.9
- `step(zeros)` 时 `r_alive > 0`、`r_vel ∈ (0, 1]`、`r_torque ≤ 0`、`r_action_rate ≤ 0`
- `REWARD_WEIGHTS` 6 个 key 齐全且为正数；`_sample_command()` 100 次采样落在声明范围内
- GIF 视觉判据：命令切换时身体方向 / 速度肉眼能跟上

## 常见坑

- 不要把 `r_torque` 系数从 2e-4 调到 1e-2——PPO 探索期立刻趴下。
- 不要忘了 `last_action` 进 obs，少了它策略高频抖动。
- 命令分布太大（如 vx ∈ [-2, 2]）训不出来。
- `SubprocVecEnv` 在 macOS 起进程要 `if __name__ == "__main__":` 包好。

## 教程衔接

- **复用**：教程 [§6.2](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/rl-gait#62-gymnasium-环境) 环境封装、[§6.3](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/rl-gait#63-动作设计) PD 残差、[§6.4](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/rl-gait#64-奖励设计) 奖励配置、[§6.6](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/rl-gait#66-ppo-训练脚本) PPO 训练脚本、[§6.9](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/rl-gait#69-评估与录屏) 评估结构。
- **扩展**：把"训练一个会走的策略"完整跑完一次；第一次获得可被 Lab 7 调用的 checkpoint。
- **不重复**：教程 [§6.7](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/rl-gait#67-速度奖励实验) 暴走 vs [§6.8](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/rl-gait#68-能量惩罚实验) 优雅对比作为前置练习。

## 不做什么

- 不做 [§6.5](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/rl-gait#65-域随机化) 域随机化（Stretch 留给学生自己加）
- 不做完整稳定性 / 接触 / 地形组奖励
- 不做 MJX / JAX / Brax
- 不做 sim2real / 真机 / 真相机
- 不开 mujoco viewer 录 GIF（统一 offscreen）
- 不做 `vy` / `ωz` 的奖励项（命令值进 obs 但不做惩罚）
- 不重训 multiple seeds 做曲线带

## Run

从 `exercises/` 运行。**首次运行前先拉取上游 RTNeural policy**（约 38 MB，已 `.gitignore`，仅需一次）：

```bash
bash shared/rl/fetch_policies.sh                   # 下载 test_policy.json 到 shared/rl/policies/
```

然后：

```bash
uv run python lab_6_rl_pupper/tests.py             # 4 条断言
uv run python lab_6_rl_pupper/train_ppo.py         # 30–60 min CPU 训练
uv run python lab_6_rl_pupper/eval_commands.py     # 加载 ckpt 录 GIF + 画图
uv run python lab_6_rl_pupper/make_artifacts.py    # 一键串起训练、GIF、对比图和曲线
```
