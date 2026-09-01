
## 环境准备

用 [uv](https://docs.astral.sh/uv/) 管理依赖。在本目录下执行：

```bash
uv sync
```

`uv sync` 会按 `.python-version` 自动准备 Python 3.12，创建 `.venv` 并安装 `pyproject.toml` 里锁定的依赖（mujoco、gymnasium、matplotlib、pillow）。

之后用 `uv run` 执行脚本，无需手动激活环境：

```bash
uv run python xxx.py
```

macOS 上跑交互式 viewer 必须用 `mjpython`，Linux / Windows 用 `python` 即可：

```bash
uv run mjpython xxx.py
```

## 运行指南

所有命令都在 `cs123` 目录下执行。

### 1.pid-control

单摆 PD 位置控制，杆摆到目标角并稳住（交互窗口）：

```bash
uv run python 1.pid-control/pd_single_joint.py
# MacOS 上开窗口必须用 mjpython，Linux / Windows 换成 python
# uv run mjpython 1.pid-control/pd_single_joint.py
```

离屏渲染单摆 PD 响应，导出 GIF：

```bash
uv run python 1.pid-control/render_pd_single_joint_gif.py
```

### 2.forward-kinematics

手写 NumPy 正运动学与 MuJoCo 对拍，打印最大误差：

```bash
uv run python 2.forward-kinematics/fk_numpy_mujoco_check.py
```

### 3.inverse-kinematics

DLS 数值 IK 跟踪三角轨迹，打印跟踪误差：

```bash
uv run python 3.inverse-kinematics/ik_dls_triangle.py
```

交互查看 DLS 收敛过程，看末端实时追目标：

```bash
uv run python 3.inverse-kinematics/viewer_dls_convergence.py
# MacOS 上开窗口必须用 mjpython，Linux / Windows 换成 python
# uv run mjpython 3.inverse-kinematics/viewer_dls_convergence.py
```

离屏渲染 DLS 收敛过程，导出 GIF：

```bash
uv run python 3.inverse-kinematics/render_dls_convergence_gif.py
```

### 4.quadruped-mjcf

静态查看固定基座模型，机器人不动：

```bash
uv run python 4.quadruped-mjcf/run_view_pupper_fixed.py
# MacOS 上开窗口必须用 mjpython，Linux / Windows 换成 python
# uv run mjpython 4.quadruped-mjcf/run_view_pupper_fixed.py
```

浮动基座自由落地，位置伺服把腿拉回 home（纯观察，不打印）：

```bash
uv run python 4.quadruped-mjcf/run_view_pupper.py
# MacOS 上开窗口必须用 mjpython，Linux / Windows 换成 python 
# uv run mjpython 4.quadruped-mjcf/run_view_pupper.py
```

同上，但站姿锁到可改的 STAND_POSE，关窗打印稳定性判据（std<5mm 算站稳）：

```bash
uv run python 4.quadruped-mjcf/run_stand_pupper.py
# MacOS 上开窗口必须用 mjpython，Linux / Windows 换成 python
# uv run mjpython 4.quadruped-mjcf/run_stand_pupper.py
```

PD 调参对比扫描，出 CSV / 图 / GIF（无窗口）：

```bash
uv run python 4.quadruped-mjcf/run_gain_sweep.py
```

生成 `original`、`long-leg` 和 `heavy` 三种形态，搜索对应站姿并验证稳定性：

```bash
uv run python 4.quadruped-mjcf/pupper_variants/run_pupper_variants.py
uv run python 4.quadruped-mjcf/pupper_variants/test_pupper_variants.py
```

### 5.gait-control

渲染原地 / 前进步态 GIF（默认 trot，可选 walk / pace / bound / gallop）：

```bash
uv run python 5.gait-control/run_gait_control.py
uv run python 5.gait-control/run_gait_control.py --gait walk
uv run python 5.gait-control/run_gait_control.py --gait pace
uv run python 5.gait-control/run_gait_control.py --gait bound
uv run python 5.gait-control/run_gait_control.py --gait gallop
```

在 MuJoCo 中交互预览：

```bash
# macOS 必须用 mjpython；Linux / Windows 把 mjpython 换成 python
uv run mjpython 5.gait-control/run_gait_control.py --gait walk --viewer inplace
uv run mjpython 5.gait-control/run_gait_control.py --gait pace --viewer forward
uv run mjpython 5.gait-control/run_gait_control.py --gait gallop --viewer forward
```

### 6.rl_pupper

运行环境和奖励函数冒烟测试：

```bash
uv run pytest -q 6.rl_pupper/tests
uv run python 6.rl_pupper/pupper_env.py
```

训练 PPO 速度跟踪策略，先用短训练确认流程：

```bash
uv run python 6.rl_pupper/train.py --timesteps 100000 --n-envs 4
```

加载 checkpoint，生成命令演示 GIF 和速度跟踪图：

```bash
uv run python 6.rl_pupper/evaluate.py
```

训练参数和控制设计见 [`6.rl_pupper/README.md`](6.rl_pupper/README.md)。

### 6.rl_pupper_v2 系列（优化迭代）

`6.rl_pupper` 的逐步优化迭代，每个目录只引入一组改动、结构与 v1 一一对应，v1 保持原样作为对照基线：

- [`6.rl_pupper_v2`](6.rl_pupper_v2/README.md)：观测追加机身线速度（45 → 48 维）、补齐防抖/防打滑/站立静止/dont_wait 奖励、命令重采样与随机扰动、两阶段课程训练。演示步态显著更平稳（前进速度波动 -64%、机身弹跳 -79%、命令切换零摔倒）。
- [`6.rl_pupper_v2_1`](6.rl_pupper_v2_1/README.md)：在 v2 上仅新增钉腿超时惩罚 `feet_stance_time`，修复 v2 的"三腿跛行"（一条腿 92% 时间钉地当锚，看起来像只有前腿驱动）。
- [`6.rl_pupper_v2_2`](6.rl_pupper_v2_2/README.md)：在 v2.1 上仅新增动作低通滤波（EMA α=0.5），用手写步态"频带受限"的先验消除高频抖动，保留 RL 闭环平衡。
- [`6.rl_pupper_v2_3`](6.rl_pupper_v2_3/README.md)：在 v2.2 上新增第三阶段 trot 步态条件化微调（48 → 53 维迁移 + 对角接触时序奖励），给涌现步态加"节拍器"先验；演示渲染升级为 20 fps + 相机平滑。
- [`6.rl_pupper_v2_4`](6.rl_pupper_v2_4/README.md)：在 v2.3 上做摆动整形——腾空目标 0.25 秒治高频小碎步、foot_clearance 奖励产生可见抬腿（目标 3.5cm），并新增与手写演示同速档的 0.15 m/s 巡航 GIF。
- [`6.rl_pupper_v2_5`](6.rl_pupper_v2_5/README.md)：v2.4 的纯配置稳定性打磨——加强竖直/横滚阻尼、提高速度跟踪权重、低熵收尾，治摆动整形带来的机身晃动回升。

训练与评估命令以各自 README 为准。
