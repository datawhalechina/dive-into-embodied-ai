---
title: 3. ONNX 导出与官方栈 Sim2Sim
sidebar_position: 4
displayed_sidebar: practicesAmdSidebar
description: "将 MJX 训练出的 Go2 策略导出为 ONNX，装入未改动的 Unitree 官方 unitree_mujoco 仿真桥与 C++ 控制器完成 Sim2Sim 闭环与跟踪误差评测。"
---

# 3. ONNX 导出与官方栈 Sim2Sim

训练在 MJX 中完成，但策略最终要跑在 Unitree 官方的部署栈上。本章将 [上一节](./training.md) 训练出的 `params.bin` 导出为 ONNX，装进**未做任何修改的**官方 `unitree_mujoco` 仿真桥 + Go2 C++ 控制器，走完 `Passive → FixStand → Velocity` 状态机并执行速度指令。

## 第一步：导出并安装策略 \{#export-policy}

在 `codes/practices/amd/unitree-rl-mjx/` 下执行：

```bash
uv run python -m unitree_rl_mjx.export.install \
  --robot go2 \
  --params runs/go2-full/params.bin \
  --dest <unitree_rl_mjlab 检出目录> \
  --version v1
```

:::tip[直接复用仓库附带的成品策略]
不想先花数小时训练？仓库附带同一任务的成品策略 `policies/go2/velocity/v0/`（ONNX + `deploy.yaml`）。把整个 `v0/` 目录拷到上游检出目录的 `deploy/robots/go2/config/policy/velocity/` 下，即可直接跳过本步。
:::

导出脚本自动完成三件事：

1. 将观测归一化烘焙进 ONNX 图（`Sub` / `Div` 节点），固定 `batch = 1` 以匹配官方运行时张量分配；
2. 按官方 `deploy.yaml` 契约生成配置（关节映射、PD 增益、观测顺序、指令范围全部从训练常量推导）；
3. 执行数值合同验证——JAX 策略与 ONNX 在代表性观测上需满足逐元素 `|Δ| < 1e-5`，否则拒绝安装。

安装结果会写入 `deploy/robots/go2/config/policy/velocity/v1/`，控制器自动选取字典序最新且包含 `exported/` 的版本。

## 第二步：构建官方 C++ 栈 \{#build-cpp-stack}

需要 Linux（部署栈捆绑 `onnxruntime-linux` 与 DDS）。依赖包括：`eigen >= 3.4`、`yaml-cpp`（静态库）、[`unitree_sdk2`](https://github.com/unitreerobotics/unitree_sdk2)、`fmt`、`spdlog`、`glfw3`。要点如下：

- `unitree_sdk2` 需要系统 `eigen`；不编译示例可加 `-DBUILD_EXAMPLES=OFF`。
- `spdlog` 若以外部 `fmt` 构建，消费方编译需加 `-DSPDLOG_FMT_EXTERNAL`。

```bash
P=<依赖安装前缀>
REPO=<unitree_rl_mjlab 检出目录>

# Go2 控制器
cmake -S "$REPO/deploy/robots/go2" -B build/go2_controller \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$P" \
  -DCMAKE_CXX_FLAGS="-I$P/include -I$P/include/eigen3 -I$P/include/ddscxx -DSPDLOG_FMT_EXTERNAL" \
  -DCMAKE_EXE_LINKER_FLAGS="-L$P/lib -Wl,-rpath,$P/lib"
cmake --build build/go2_controller -j

# MuJoCo 仿真桥（unitree_mujoco）
M="$REPO/simulate/mujoco/lib"
cmake -S "$REPO/simulate" -B build/unitree_mujoco \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$P" \
  -DCMAKE_CXX_FLAGS="-I$P/include -I$P/include/ddscxx -DSPDLOG_FMT_EXTERNAL" \
  -DCMAKE_EXE_LINKER_FLAGS="-L$P/lib -Wl,-rpath,$P/lib -Wl,-rpath,$M"
cmake --build build/unitree_mujoco -j
```

两个二进制按**自身所在路径**查找配置：

- `unitree_mujoco` 读取上一级目录的 `config.yaml`（机器人、场景 XML、DDS 网卡、手柄设备）；场景选 `go2`，路径建议写绝对路径。
- `go2_ctrl` 读取同目录的 `config.yaml` 与 `config/` 策略树——将它们符号链接到检出目录的 `deploy/robots/go2/config` 即可。

## 第三步：闭环运行 \{#closed-loop-run}

1. 启动仿真桥（有显示器直接运行；无头服务器可用 `Xvfb :99` + `DISPLAY=:99`）。
2. 启动控制器：`./go2_ctrl -n lo`（DDS 走本机回环）。
3. 手柄操作：`LT + ↑` 进入 `FixStand`（起立），`RT + A` 进入 `Velocity`（策略接管），左摇杆控制平移、右摇杆控制转向，`LT + B` 退回 `Passive`。

:::info[无实体手柄时如何脚本化测试]
仿真桥只从 `joystick_device` 读取 8 字节 Linux `js_event` 结构，不做 `ioctl`。可以用 `mkfifo /tmp/js0.fifo` 创建命名管道、将 `joystick_device` 指向该管道，再用脚本写入事件即可全自动复现（Xbox 布局：按钮 `A=0`、`B=1`；轴 `LT=2`、`RT=5`、十字键纵轴 `7` 上为负；左摇杆 `0/1`、右摇杆 `3/4`，量程 `±32768`）。
:::

## 实测参考：MJX 与官方栈跟踪对比 \{#sim2sim-results}

用同一份训练检查点在两边实测（去除前 2 s 瞬态后的稳态均值）：

| 指令段 | 指令 | MJX 内 | 官方栈 Sim2Sim |
| :--- | :--- | :---: | :---: |
| 前进 | `vx = 1.0 m/s` | `0.93` | `0.85` |
| 横移 | `vy = 0.4 m/s` | `0.30` | `0.18` |
| 原地转 | `wz = 0.8 rad/s` | `0.81` | `0.41` |

- **前进迁移良好**；侧移与转向在官方栈中衰减约一半，主要源自仿真器与模型差异（官方场景为全碰撞模型、求解器参数不同、以及 1 kHz PD 执行路径）。
- **已知语义差异**：训练环境的偏航指令由航向伺服生成，而部署栈直接把摇杆输入作为 `wz` 指令——持续满速旋转属于训练分布的边缘输入。

---

上一步：[← 2. 训练 Go2 速度策略](./training.md) · 返回：[Go2 四足运控总览](./index.md) · 延伸：[在线仿真试玩 Go2](/simulation)
