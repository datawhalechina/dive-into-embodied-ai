# 在 AMD ROCm 上训练 ACT：ALOHA Transfer Cube 完整示例

这个示例参考 [`codes/practices/vla/act`](../../../vla/act)，实现了只使用 AMD ROCm
GPU 的 ACT 训练、断点续训、评测和教程素材导出。实测在 AMD Radeon AI PRO R9700
上训练 10,001 step，用 20 个固定种子进行闭环 MuJoCo 评测，成功率为 **50%（10/20）**。

站点版完整教程见
[`docs/practices/amd/vla-act/index.md`](../../../../../docs/practices/amd/vla-act/index.md)。

![训练曲线](reports/training_curves.webp)

## 最终效果

下面是评测 episode 1 的完整成功回合。策略在 4.44 秒内完成抓取、抬起和双臂移交，
获得环境满分奖励 4。GIF 适合直接嵌入文档；原始 H.264 MP4 可从
[这里下载](reports/act_aloha_transfer_demo.mp4)。

![ACT ALOHA Transfer Cube 成功演示](reports/act_aloha_transfer_demo.gif)

![成功回合关键帧](reports/act_aloha_transfer_keyframes.webp)

20 个评测回合中，绿色柱表示完成任务，蓝色柱表示没有完成；橙线是该回合达到的最高
阶段奖励。

![20 回合闭环评测结果](reports/evaluation_results.webp)

| 指标 | 实测值 |
| --- | ---: |
| 优化器步数 | 10,001 |
| 训练样本 / batch size | 20,000 / 32 |
| 数值精度 | BF16 |
| 训练用时 | 3,028 秒（0.84 小时） |
| 稳态吞吐 | 3.35 step/s（约 107 sample/s） |
| PyTorch 峰值显存 | 6.10 GiB |
| 最终 / 最低训练 loss | 0.1214 / 0.0860 |
| 闭环成功率 | **50%（10/20）** |
| 评测用时 | 185 秒 |

逐回合原始结果和更完整的实验说明见 [TRAINING_REPORT.md](TRAINING_REPORT.md)。

## 1. 环境准备

目标环境是 Linux x86_64、Python 3.12/3.13、可访问 `/dev/kfd`，并已安装兼容当前
GPU 的 AMD 驱动。进入示例目录后用 lockfile 创建环境：

```bash
cd codes/practices/amd/vla/act
uv sync --frozen
```

项目固定使用 LeRobot 0.6.1、PyTorch 2.10.0 + ROCm 7.0、torchvision 0.25.0、
TorchCodec 0.10.0 和 `triton-rocm`。检查 GPU：

```bash
uv run python -c \
  "import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

正常输出应包含 `+rocm7.0`、非空 HIP 版本、`True` 和 AMD GPU 名称。PyTorch 的
ROCm 后端沿用 `torch.cuda` API，因此日志里的 `cuda:0` 仍然表示 AMD GPU，并非
NVIDIA CUDA。

### 视频解码依赖

最快配置使用 TorchCodec，它需要系统中有完整 FFmpeg shared libraries。在 Ubuntu
24.04 上可以安装：

```bash
sudo apt update
sudo apt install ffmpeg
```

然后验证：

```bash
uv run python -c "import torchcodec; print(torchcodec.__version__)"
```

若机器上不能安装系统 FFmpeg，把下面命令中的 `--video-backend torchcodec` 改成
`--video-backend pyav` 即可，训练逻辑和结果口径不变，只是数据解码稍慢。

## 2. 一步冒烟测试

第一次运行会从 Hugging Face 下载 `lerobot/aloha_sim_transfer_cube_human@v3.0`。
先只加载一个 episode，验证视频解码、预处理、ROCm 前后向和 checkpoint 保存：

```bash
uv run python train_act.py \
  --episodes 0 \
  --steps 1 \
  --batch-size 2 \
  --precision bf16
```

脚本会强制检查 HIP wheel、可见 GPU 和一次真实的 ROCm 张量计算；检查失败会直接报错，
不会静默回退到 CPU。

## 3. 快速完整训练

本教程实测使用下面的配置。R9700 有 32 GiB 显存，而 batch 32 的 PyTorch 峰值只约
6.1 GiB，可以同时提高 GPU 利用率并减少达到可用效果所需的墙钟时间。

```bash
uv run python train_act.py \
  --steps 10001 \
  --batch-size 32 \
  --chunk-size 100 \
  --num-workers 4 \
  --video-backend torchcodec \
  --precision bf16 \
  --learning-rate 2e-5 \
  --log-every 250 \
  --metrics-every 10 \
  --save-every 5000 \
  --output-dir outputs/act_aloha_transfer_rocm_full
```

训练流程保持 ACT 的完整教学路径：根据数据统计量创建归一化处理器，读取相机图像和
14 维关节状态，构造 100 步未来动作 chunk，用 CVAE + Transformer 预测动作，反向
传播 `L1 + 10 × KL` 损失，并保存策略、处理器和优化器状态。

输出目录中的关键文件：

- `model.safetensors`：可直接评测或部署的策略权重；
- `config.json`：ACT 网络和输入输出特征配置；
- `policy_preprocessor*` / `policy_postprocessor*`：训练数据对应的归一化处理器；
- `training_state.pt`：包含模型、优化器和随机数状态的滚动断点；
- `training_metrics.csv`：绘制曲线所需的逐步指标；
- `training_run.json` / `training_summary.json`：环境、参数和最终摘要。

中断后继续到更大的目标步数：

```bash
uv run python train_act.py \
  --steps 20000 \
  --batch-size 32 \
  --chunk-size 100 \
  --num-workers 4 \
  --video-backend torchcodec \
  --precision bf16 \
  --learning-rate 2e-5 \
  --save-every 5000 \
  --resume \
  --output-dir outputs/act_aloha_transfer_rocm_full
```

## 4. 闭环评测并录制视频

无显示器的 Linux 机器用 EGL 渲染。LeRobot 0.6.1 的 ALOHA Gym 注册表在异步 worker
中可能丢失，因此这里显式使用同步向量环境；4 个环境仍会组成一个 batch 送入 GPU。

```bash
MUJOCO_GL=egl uv run lerobot-eval \
  --policy.path=outputs/act_aloha_transfer_rocm_full \
  --env.type=aloha \
  --env.task=AlohaTransferCube-v0 \
  --eval.n_episodes=20 \
  --eval.batch_size=4 \
  --eval.use_async_envs=false \
  --output_dir=outputs/eval_act_aloha_transfer_rocm_10k \
  --seed=1000
```

成功的判定不是人工看视频，而是环境在一个回合内达到最高阶段奖励 4。上述固定种子
得到 10 个成功回合，`pc_success = 50.0`。LeRobot 会把前 10 个回合自动录制成 H.264
MP4；`eval_info.json` 保存所有 20 个回合的奖励和成功标志。

## 5. 生成训练曲线和教程素材

```bash
uv run python plot_training.py
uv run python make_eval_artifacts.py
```

两个脚本会在 `reports/` 下生成 WebP 训练与评测图、成功回合 MP4、GIF、关键帧
拼图和机器可读的 JSON 摘要。`make_eval_artifacts.py` 会从已录制的成功回合里自动选择
完成用时最短的样例，而不是手工挑结果。

## 6. 为什么这版训练比较快

1. **BF16**：R9700 的 BF16 路径显著快于 FP32，并保持本实验的收敛稳定性。
2. **batch 32**：把单步吞吐提高到约 107 sample/s，同时显存仍有很大余量。
3. **TorchCodec + 4 个持久 worker**：避免每个数据 epoch 重建解码进程；worker 再多会
   增加主机内存压力，实测不更快。
4. **先到验收线就停**：10k 步已经达到教程设定的 50% 成功率，因此无需盲目跑满
   100k 步；断点仍然保留，需要更高成功率时可以继续训练。

这次结果用于验证“AMD ROCm 上可以快速跑通完整 ACT 教学闭环”，不是不同硬件或算法
之间的严格 benchmark。成功率会受 checkpoint、评测种子和依赖版本影响，复现实验时
应同时报告这些条件。
