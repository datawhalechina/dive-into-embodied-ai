# Pupper PPO 本机 ROCm 训练报告

日期：2026-08-11

## 结论

训练已在本机 AMD Radeon AI PRO R9700 上跑通，PPO 网络和张量计算使用 ROCm；
MuJoCo 物理仿真按预期保留在 CPU。最终选择 `v3_precision`：它在固定命令基准中
得到 **84.48 分**，15/15 个 10 秒回合全部完成，平移命令 RMSE 为
**0.080 m/s**，偏航命令 RMSE 为 **0.151 rad/s**。

最佳模型：
`outputs/experiments/v3_precision/pupper_ppo.zip`

同一模型也已复制到默认加载位置 `outputs/pupper_ppo.zip`；两者 SHA-256 均为
`aa22277d87fb1942fc7fbb887fa1c7ba876725434d114c2f3e030f7a42789685`。

## 运行环境

| 项目 | 值 |
| --- | --- |
| GPU | AMD Radeon AI PRO R9700（gfx1201，32 GiB） |
| ROCm / HIP | 7.0.51831 |
| PyTorch | 2.10.0+rocm7.0 |
| Stable-Baselines3 | PPO / MlpPolicy |
| GPU 设备名 | `cuda:0`（PyTorch 的 ROCm 设备 API 沿用 `cuda` 名称） |
| 仿真 | MuJoCo 3.7.0，CPU |

训练和评估入口现在都会执行 HIP 张量自检。缺少 ROCm wheel 或 AMD GPU 时会直接
报错，不会静默回退到 CPU。训练日志也确认策略参数位于 `cuda:0`。

## 训练过程与选择

为了避免把训练步数不足误判成参数问题，稳定配置完整训练到 2000 万步后才作调整；
精度微调也完整跑到追加的 1000 万步，并比较了中间 checkpoint。

| 阶段 | 环境步 | 处理 | 固定基准结果 |
| --- | ---: | --- | --- |
| v1 冒烟基线 | 300 万 | 较高探索度，只用于管线诊断 | 58.97 分，存活率 83.4% |
| v2 stable | 2000 万（完整） | 稳定 PPO 参数，从头训练 | 81.09 分，存活率 100% |
| v3 precision | 追加 1000 万（完整） | 从 v2 热启动，收紧线速度跟踪奖励 | **84.48 分，存活率 100%** |

正式两阶段训练墙钟时间约 77 分 40 秒：v2 约 53 分 32 秒，v3 约 24 分
08 秒。v3 的 500 万步和 1000 万步基准分别为 82.28 与 84.48 分，最终仍在改善，
因此保留完整的 1000 万步结果，而不是过早停止。

主要 PPO 配置：8 个并行环境、`n_steps=2048`、`batch_size=512`、4 个 epoch、
学习率 `1e-4`、熵系数 `1e-3`、`gamma=0.97`、`gae_lambda=0.95`、
`target_kl=0.02`，策略网络为两层 256 单元 MLP。

环境侧增加了机身线速度观测、零命令采样和诊断项。最终微调把线速度跟踪核宽从
0.25 收紧到 0.10，并增加连续线速度误差惩罚；其余 PPO 超参数保持保守配置。

## 固定基准结果

基准包含站立、前进、后退、侧移和偏航五种命令；每种命令使用种子 11、22、33，
每回合运行 10 秒。推理为 deterministic，v2 和 v3 使用完全相同的环境与评测脚本。

| 指标 | v2 stable | v3 precision | 变化 |
| --- | ---: | ---: | ---: |
| 综合分数 | 81.09 | **84.48** | +3.39 |
| 存活 / 完成率 | 100% / 100% | **100% / 100%** | 持平 |
| 全场景线速度 RMSE | 0.101 m/s | **0.062 m/s** | -38.1% |
| 平移命令 RMSE | 0.139 m/s | **0.080 m/s** | -42.3% |
| 偏航命令 RMSE | **0.130 rad/s** | 0.151 rad/s | +16.4% |
| 平均机身倾角 | **2.47°** | 3.45° | +0.98° |

v3 牺牲了少量偏航误差和姿态平稳度，换取明显更好的平移速度精度，因此综合分数更高。
最终策略在三个种子上的稳态均值大致为：

| 场景 | 目标命令 | 实际均值 |
| --- | --- | --- |
| 站立 | `(0, 0, 0)` | `vx=-0.001, vy=0.005, wz=-0.023` |
| 前进 | `vx=0.5` | `vx=0.507 m/s` |
| 后退 | `vx=-0.3` | `vx=-0.336 m/s` |
| 侧移 | `vy=0.3` | `vy=0.350 m/s` |
| 偏航 | `wz=0.8` | `wz=0.736 rad/s` |

完整逐回合数据见
[`outputs/experiments/v3_precision/benchmark_final.json`](outputs/experiments/v3_precision/benchmark_final.json)。

## 产物

- 最佳 checkpoint：[`outputs/experiments/v3_precision/pupper_ppo.zip`](outputs/experiments/v3_precision/pupper_ppo.zip)
- 默认 checkpoint：[`outputs/pupper_ppo.zip`](outputs/pupper_ppo.zip)
- 动画演示：[`outputs/final_v3/demo.gif`](outputs/final_v3/demo.gif)
- 速度跟踪：[`outputs/final_v3/velocity_tracking.png`](outputs/final_v3/velocity_tracking.png)
- 训练曲线：[`outputs/final_v3/training_curves.png`](outputs/final_v3/training_curves.png)
- 基准进展：[`outputs/final_v3/benchmark_progress.png`](outputs/final_v3/benchmark_progress.png)
- 训练配置：[`outputs/experiments/v3_precision/training_config.json`](outputs/experiments/v3_precision/training_config.json)

`outputs/` 被 `.gitignore` 排除，模型和图片是本机训练产物，不会意外进入提交。

## 复现命令

```bash
cd codes/practices/amd/cs123
uv sync --frozen

uv run python 6.rl_pupper/train.py \
  --timesteps 20000000 --n-envs 8 --seed 42 --tensorboard \
  --out 6.rl_pupper/outputs/experiments/v2_stable

uv run python 6.rl_pupper/train.py \
  --timesteps 10000000 --n-envs 8 --seed 84 --tensorboard \
  --checkpoint 6.rl_pupper/outputs/experiments/v2_stable/pupper_ppo.zip \
  --out 6.rl_pupper/outputs/experiments/v3_precision

uv run python 6.rl_pupper/benchmark.py \
  6.rl_pupper/outputs/experiments/v3_precision/pupper_ppo.zip \
  --output 6.rl_pupper/outputs/experiments/v3_precision/benchmark_final.json

MUJOCO_GL=egl uv run python 6.rl_pupper/evaluate.py \
  --checkpoint 6.rl_pupper/outputs/experiments/v3_precision/pupper_ppo.zip \
  --out 6.rl_pupper/outputs/final_v3
```

## 已知限制

- 最终策略能稳定跟踪 `vx=0.4` 和 `0.6 m/s`，但 `vx=0.2 m/s` 时仍倾向原地站立；
  下一轮更适合增加低速命令采样，而不是继续无差别堆训练步数。
- 后退和侧移仍有小幅超调，偏航略有欠跟踪。
- 当前评估是确定性 MuJoCo 仿真，尚未加入地面摩擦、质量、传感噪声、时延等域随机化，
  不能直接视为真机可部署策略。
- SB3 的 MLP PPO 计算量较小，虽然策略确实运行在 ROCm 上，GPU 利用率不会像大型网络
  或 GPU 物理仿真那样高。
