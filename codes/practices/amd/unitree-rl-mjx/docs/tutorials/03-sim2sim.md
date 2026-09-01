# 教程 3：导出策略，在官方仿真栈里闭环

训练在 MJX 里完成，但策略最终要跑在 Unitree 官方的部署栈上。
本章把教程 2 训练出的 `params.bin` 导出为 ONNX，装进**未做任何修改的**
官方 `unitree_mujoco` 仿真桥 + Go2 C++ 控制器，走完
Passive → FixStand → Velocity 状态机并执行速度指令。

## 第一步：导出并安装策略

```bash
uv run python -m unitree_rl_mjx.export.install \
  --robot go2 \
  --params runs/go2-full/params.bin \
  --dest <unitree_rl_mjlab 检出目录> \
  --version v1
```

> 不想先花数小时训练？仓库附带同一任务的成品策略
> `policies/go2/velocity/v0/`（ONNX + `deploy.yaml`）。把整个 `v0/` 目录拷到
> 检出目录的 `deploy/robots/go2/config/policy/velocity/` 下，即可跳过本步。

导出做三件事：把观测归一化烘焙进 ONNX 图（Sub/Div 节点）、
按官方 `deploy.yaml` 契约生成配置（关节映射、增益、观测顺序、指令范围
全部从训练常量推导）、并做合同验证——JAX 策略与 ONNX 在代表性观测上
逐元素 |Δ| < 1e-5，不通过就拒绝安装。安装结果是
`deploy/robots/go2/config/policy/velocity/v1/` 一个版本文件夹，
控制器会自动选择字典序最新且含 `exported/` 的版本。

注意：官方运行时按 ONNX 声明的形状分配张量，导出图固定 batch = 1；
这由导出器保证，无需手工处理。

## 第二步：构建官方 C++ 栈

需要 Linux（部署栈捆绑 onnxruntime-linux 与 DDS）。依赖：
eigen ≥ 3.4、yaml-cpp（静态库）、[unitree_sdk2]、fmt、spdlog、glfw3。
要点：

- unitree_sdk2 需要系统 eigen；若不装示例可加 `-DBUILD_EXAMPLES=OFF`。
- spdlog 若以外部 fmt 构建，消费方编译需加 `-DSPDLOG_FMT_EXTERNAL`。

```bash
P=<依赖安装前缀>
REPO=<unitree_rl_mjlab 检出目录>

# Go2 控制器
cmake -S $REPO/deploy/robots/go2 -B build/go2_controller \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$P" \
  -DCMAKE_CXX_FLAGS="-I$P/include -I$P/include/eigen3 -I$P/include/ddscxx -DSPDLOG_FMT_EXTERNAL" \
  -DCMAKE_EXE_LINKER_FLAGS="-L$P/lib -Wl,-rpath,$P/lib"
cmake --build build/go2_controller -j

# MuJoCo 仿真桥（unitree_mujoco）
M=$REPO/simulate/mujoco/lib
cmake -S $REPO/simulate -B build/unitree_mujoco \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$P" \
  -DCMAKE_CXX_FLAGS="-I$P/include -I$P/include/ddscxx -DSPDLOG_FMT_EXTERNAL" \
  -DCMAKE_EXE_LINKER_FLAGS="-L$P/lib -Wl,-rpath,$P/lib -Wl,-rpath,$M"
cmake --build build/unitree_mujoco -j
```

两个二进制按**自身所在路径**找配置：

- `unitree_mujoco` 读上一级目录的 `config.yaml`（机器人、场景 XML、DDS 网卡、
  手柄设备）；场景选 go2，路径写绝对路径最稳。
- `go2_ctrl` 读同目录的 `config.yaml` 与 `config/`
  策略树——把它们符号链接到检出目录的 `deploy/robots/go2/config` 即可。

## 第三步：闭环运行

1. 启动仿真桥（有显示器直接跑；无头机器用 `Xvfb :99` + `DISPLAY=:99`）。
2. 启动控制器：`./go2_ctrl -n lo`（DDS 走本机回环）。
3. 手柄操作：`LT + ↑` 进入 FixStand（起立），`RT + A` 进入 Velocity
   （策略接管），左摇杆平移 / 右摇杆转向，`LT + B` 退回 Passive。

没有手柄时可用命名管道模拟：仿真桥的手柄读取只是从
`joystick_device` 读 8 字节的 Linux `js_event` 结构，不做任何 ioctl，
所以 `mkfifo /tmp/js0.fifo`、把 `joystick_device` 指过去，再用脚本写入
事件即可脚本化整个流程（Xbox 布局：按钮 A=0、B=1；轴 LT=2、RT=5、
十字键纵轴=7 上为负；左摇杆 0/1、右摇杆 3/4，量程 ±32768）。

## 参考预期：跟踪表现

用同一份训练检查点在两边实测（去除 2 s 瞬态后的均值）：

| 指令段 | 指令 | MJX 内 | 官方栈 sim2sim |
|---|---|---|---|
| 前进 | vx = 1.0 m/s | 0.93 | 0.85 |
| 横移 | vy = 0.4 m/s | 0.30 | 0.18 |
| 原地转 | wz = 0.8 rad/s | 0.81 | 0.41 |

前进迁移良好；侧移与转向在官方栈里衰减约一半，来自仿真器与模型差异
（官方场景为全碰撞模型、求解器参数不同、1 kHz PD 执行路径）。
一个已知语义差异值得注意：训练环境的偏航指令由航向伺服生成，
而部署栈直接把摇杆值作为 wz 指令——持续满速旋转对策略是分布边缘输入。
