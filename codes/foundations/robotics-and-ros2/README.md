# ROS2 教程配套代码

[English](README.en.md) | [Italiano](README.it.md)

代码由本仓库 [ROS2 教程第 2–11 讲](../../../docs/foundations/robotics-and-ros2/0.intro.md) 的示例整理而来，沿用仓库的 CC BY-NC-SA 4.0 许可。旧版文档中的 `ros_learning` 不再是前置目录，克隆本仓库即可取得这些文件。

## 目录与环境

```text
ros_ws/
├── scripts/                 # 第 2–3 讲：不依赖 ROS 的 Python 示例
├── src/                     # 自己动手创建的包（初始为空）
└── reference/
    ├── COLCON_IGNORE        # 防止 colcon 同时发现参考包和练习包
    ├── moveit/              # 第 10 讲的 demo_sim.launch.py
    └── src/
        ├── my_robot_basics/       # 第 4–8 讲：节点、通信、参数、tf2
        ├── my_robot_interfaces/   # 第 6 讲：SolveIK.srv、MoveTo.action
        ├── my_robot_description/  # 第 9–10 讲：URDF、可视化、mock hardware
        ├── my_robot_planner/      # 第 10–11 讲：规划、目标订阅、障碍物
        └── my_robot_bringup/      # 第 11 讲：完整系统启动
```

ROS 实验目标环境是 **Ubuntu 24.04 + ROS2 Jazzy + 系统 Python 3.12 + Bash**。先按[第 1 讲](../../../docs/foundations/robotics-and-ros2/1.course_introduction.md)安装 ROS、colcon 和 rosdep。第 2–3 讲只需 Python 和 NumPy，可在 macOS / Windows / Linux 单独运行；网站的 npm 依赖不包含这些 Python / ROS 依赖。

在仓库根目录运行：

```bash
export ROS_WS="$(pwd)/codes/foundations/robotics-and-ros2/ros_ws"
echo "$ROS_WS"
```

后续新终端也需要设置这个变量；Bash 用户可将上面输出的绝对路径写入 `~/.bashrc` 的 `export ROS_WS="..."` 行。请勿删除 `reference/COLCON_IGNORE`。练习代码及 `build/`、`install/`、`log/` 已忽略，不会混入仓库源码。

## 先运行运动学脚本

在代码目录创建独立环境（只运行数学脚本和下面的检查，不用于编译 ROS 包）：

```bash
cd codes/foundations/robotics-and-ros2
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python "$ROS_WS/scripts/lesson_02_transforms.py"
python "$ROS_WS/scripts/lesson_03_two_link_kinematics.py"
python -m unittest discover -s tests -v
deactivate
```

## 第 4–9 讲：练习与参考

推荐按章节在 `$ROS_WS/src` 中运行 `ros2 pkg create`，再对照 `reference/src` 中的同名文件。参考包是后续章节合并后的版本：发布器包含第 7 讲的参数与动态频率更新，Action 服务包含启动参数，URDF 已含第 10 讲的 `ros2_control` 标签；复制该 URDF 后不要重复插入标签。`SolveIK` 的连杆长度来自每次请求，不使用 YAML 中的默认值。Action 示例演示目标、反馈和结果，尚未实现取消处理。

若想先运行参考包，在**尚无同名练习包**的工作空间中复制这三个包。已有练习时只对照或复制需要的文件，保留自己的改动。

```bash
source /opt/ros/jazzy/setup.bash
cd "$ROS_WS"
cp -rn reference/src/my_robot_interfaces reference/src/my_robot_basics reference/src/my_robot_description src/
rosdep install --from-paths src --ignore-src --rosdistro jazzy -r -y
colcon build --symlink-install --packages-up-to my_robot_basics my_robot_description
source install/setup.bash
ros2 run my_robot_basics hello_node
```

`hello_node` 会持续运行，用 Ctrl+C 退出。其他入口：

```bash
ros2 launch my_robot_basics demo.launch.py publish_rate:=2.0
ros2 launch my_robot_description display.launch.py
```

按需分别运行；新终端先加载 `/opt/ros/jazzy/setup.bash` 和 `"$ROS_WS/install/setup.bash"`。tf2 与 URDF 实验按第 8–9 讲切换，避免手写 TF 广播器和 `robot_state_publisher` 同时发布同一坐标系。

## 第 10–11 讲：生成 MoveIt 配置后继续

`my_robot_moveit_config` **不预置**。先按[第 10 讲](../../../docs/foundations/robotics-and-ros2/10.moveit2_basics.md)用 MoveIt Setup Assistant 生成到 `$ROS_WS/src/my_robot_moveit_config`，并完成该讲列出的 SRDF、规划器、关节限位和控制器配置。随后复制参考启动文件和另外两个包（同样仅适用于尚无同名练习包的空间）：

```bash
cd "$ROS_WS"
cp -n reference/moveit/demo_sim.launch.py src/my_robot_moveit_config/launch/
cp -rn reference/src/my_robot_planner reference/src/my_robot_bringup src/
rosdep install --from-paths src --ignore-src --rosdistro jazzy -r -y
colcon build --symlink-install
source install/setup.bash
ros2 launch my_robot_bringup full_system.launch.py
```

在另一个已加载环境的终端发送第 11 讲的目标：

```bash
ros2 topic pub --once /target_pose geometry_msgs/msg/Point "{x: 1.2, y: 0.8, z: 0.0}"
```

## 验证范围

`tests/` 检查数学结果、FK/IK 双解与边界、Python 语法、包入口和资源路径、URDF 结构。它不加载 ROS，也不代表 ROS 节点、RViz2 或 MoveIt 运行验证；完整实验仍需在上述 Ubuntu / Jazzy 环境中按章节执行。
