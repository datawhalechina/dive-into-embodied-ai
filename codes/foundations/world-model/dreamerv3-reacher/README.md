# DreamerV3 Reacher：状态输入实验

配套[教程](../../../../docs/foundations/world-model/2.dreamerv3-reacher.md)。视频先看左侧随机动作与右侧策略的区别，再看完整回合评估表。This folder contains the pinned patch, evaluation script and per-condition results for the state-input Reacher tutorial. It does not contain trained weights; reproducing the policy requires training.

## 文件与结果

- `tutorial.patch`：作用于作者源码 `e01491fad6434b2245a3b8ca201dd7faedcc458c`。仅增加 DMC 环境种子、教程配置、结束时保存模型；不改算法损失。
- `evaluate.py`：独立加载 checkpoint，对相同初态分别执行随机动作与 Dreamer；默认5组、每组1000步。输出逐步JSON、汇总及首组视频。
- `results/*.json`：三阶段开发评估和最终10初态原始汇总。`manifest.json`记录来源、模型步数及视频处理方式。
- 训练与评估使用一个模型种子。开发初态100–104参与结果比较；最终200–209独立保留。模型权重和大型经验池未包含在教程仓库。

## 环境

作者安装说明见 https://github.com/danijar/dreamerv3/tree/e01491fad6434b2245a3b8ca201dd7faedcc458c 。请在独立 Python 环境中安装对应版本的训练依赖，不要使用网站的 Node 环境。

本次实际组合：Linux、Python3.11、JAX0.4.33、DMControl1.0.34、MuJoCo3.3.6、NumPy1.26.4、单张RTX4090。CUDA共享库需要与JAX匹配。作者依赖包含其他任务的包；本实验只使用DMC。以下命令假定已按作者说明安装依赖并确认JAX能看到GPU；它们不是一键安装脚本，也不保证不同驱动和依赖版本逐位一致。

```bash
# 在本 README 所在目录运行；本地生成的源码和日志已忽略。
git clone https://github.com/danijar/dreamerv3.git source
git -C source checkout e01491fad6434b2245a3b8ca201dd7faedcc458c
git -C source apply ../tutorial.patch
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
python -c 'import jax; print(jax.devices())'
```

首次先验证可离屏渲染，再训练：

```bash
python - <<'PY'
from dm_control import suite
env = suite.load('reacher', 'easy', task_kwargs={'random': 0})
env.reset()
frame = env.physics.render(64, 64, camera_id=0)
assert frame.shape == (64, 64, 3)
print('DMC rendering OK')
PY
```

## 训练与独立评估

第二轮先10万步评估，再从同目录恢复到30万步。始终从`source/`启动训练，确保作者源码包可导入。预期耗时为小时量级；最终数字以日志与 checkpoint 为准，步数可能按采样批次略超预算。

```bash
cd source
python dreamerv3/main.py --configs dmc_proprio tutorial_reacher \
  --run.train_ratio 1024 --run.steps 100000 --logdir ../runs/reacher-ratio1024
cd ..
python evaluate.py --run runs/reacher-ratio1024 \
  --output runs/reacher-ratio1024/evaluation-100000

cd source
python dreamerv3/main.py --configs dmc_proprio tutorial_reacher \
  --run.train_ratio 1024 --run.steps 300000 --logdir ../runs/reacher-ratio1024
cd ..
python evaluate.py --run runs/reacher-ratio1024 \
  --output runs/reacher-ratio1024/evaluation-300000
python evaluate.py --run runs/reacher-ratio1024 --start-index 200 --count 10 \
  --output runs/reacher-ratio1024/final-evaluation
```

第一轮基线用**新目录**从零训练，设置`--run.train_ratio 128 --run.steps 100000`。不要复用已训练1024配置的目录来伪装成独立对照。

本脚本首组视频为64×64的原始日志画面，适合机器核对；教程展示视频使用保存的动作重新以640×640渲染两侧，各逐步核对后合成1280×800，前5秒以0.5倍速播放。仅提高输出渲染分辨率，不改变训练观测或动作。

## 验证与判读

本次训练和独立评估已实际完成，结果见JSON。原始代码的有限损失、checkpoint存在，仅证明流程工作；到点与保持还要看回报、末100步目标占用比例及回放。二值奖励下，整回合占用比例等于回报/1000，两者不构成独立验证。

最终10个新初态平均回报954.8，对照随机11.7；最差回报851。不能据此声称多seed鲁棒性或真机安全。若训练日志好、评估差，先检查种子、模型加载、观测、动作包装与reset/历史状态，再决定是否增加预算。
