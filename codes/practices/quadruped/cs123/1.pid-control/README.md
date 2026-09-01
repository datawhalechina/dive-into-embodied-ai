# PID / PD 控制实验

本目录用于存放 PID、PD 控制相关的独立实验。不同实验可能使用纯 Python 数学模型、MuJoCo 仿真或其他被控对象，每个实验的具体用法放在对应的说明文档中。

## 环境准备

项目要求 Python 3.12 或更高版本，并使用 `uv` 管理依赖。

从仓库根目录进入项目目录：

```bash
cd codes/practices/quadruped/cs123
```

安装或同步依赖：

```bash
uv sync
```

项目依赖中已经包含本目录实验所需的 Matplotlib、MuJoCo、NumPy 和 Pillow。

后续命令均默认在 `codes/practices/quadruped/cs123` 目录中执行。

## 实验列表

### Python 实时 PID 曲线实验

使用二阶数学模型开展 PID 控制实验，支持实时调整 `Kp`、`Ki`、`Kd` 和目标值，并显示过程值与控制输出曲线。

详细说明：[docs/run_pid_demo.md](docs/run_pid_demo.md)

运行：

```bash
uv run python 1.pid-control/run_pid_demo.py
```

### MuJoCo 单关节 PD 控制实验

在 MuJoCo 中控制单关节摆模型，既可以使用手写电机 PD 控制，也可以切换到 MuJoCo 的位置执行器。

详细说明：[docs/pd_single_joint.md](docs/pd_single_joint.md)

运行：

```bash
uv run python 1.pid-control/pd_single_joint.py
```
