# MuJoCo 单关节 PD 控制实验

本实验使用 MuJoCo 搭建单关节摆模型，通过 PD 控制器将关节转动到指定目标角度。实验支持手写电机 PD 控制和 MuJoCo 位置执行器两种控制方式。

环境准备请参见上级目录的 [README.md](../README.md)。

## 启动实验

在 `codes/practices/quadruped/cs123` 目录中运行：

```bash
uv run python 1.pid-control/pd_single_joint.py
```

仿真窗口会运行约 8 秒。程序结束后，终端会输出最终仿真时间、关节角度、关节速度和执行器控制量。

## 控制方法

脚本通过 `use_motor` 选择控制方式：

```python
use_motor = True
```

### 手写电机 PD 控制

当 `use_motor = True` 时，程序根据关节角度误差和角速度计算力矩：

\[
\tau = K_p(q_{des}-q)+K_d(0-\dot q)
\]

默认参数为：

- 目标角度 `q_des = 0.8 rad`，约为 `45.8°`。
- 比例增益 `Kp = 20.0`。
- 微分增益 `Kd = 1.0`。
- 电机力矩限制为 `-5.0～5.0`。

使用手写 PD 时，脚本会关闭 MuJoCo 位置执行器的增益，避免两个执行器同时输出。

### MuJoCo 位置执行器

将脚本中的配置改为：

```python
use_motor = False
```

此时程序不再手动计算 PD 力矩，而是直接把目标角度发送给 MuJoCo 的位置执行器：

```python
data.ctrl[0] = q_des
```

## 相关文件

- `pd_single_joint.py`：交互式仿真和 PD 控制代码。
- `pendulum.xml`：单关节摆的 MuJoCo 模型。
- `render_pd_single_joint_gif.py`：离屏渲染脚本。
- `pd_single_joint.gif`：渲染生成的实验动画。

## 生成实验动画

在 `codes/practices/quadruped/cs123` 目录中运行：

```bash
uv run python 1.pid-control/render_pd_single_joint_gif.py
```

生成的动画会保存为 `1.pid-control/pd_single_joint.gif`。
