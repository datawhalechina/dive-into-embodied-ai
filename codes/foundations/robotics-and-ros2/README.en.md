# ROS2 companion code

[中文](README.md) | [Italiano](README.it.md)

These files assemble the examples from [lessons 2–11](../../../docs/foundations/robotics-and-ros2/0.intro.md), under the repository's CC BY-NC-SA 4.0 license. The old `ros_learning` directory is no longer a prerequisite: clone this repository to get the code.

## Layout and environment

```text
ros_ws/
├── scripts/                 # Lessons 2–3: standalone Python examples
├── src/                     # Your exercise packages; initially empty
└── reference/
    ├── COLCON_IGNORE        # Prevent duplicate discovery of reference packages
    ├── moveit/              # Lesson 10: demo_sim.launch.py
    └── src/
        ├── my_robot_basics/       # Lessons 4–8: nodes, communication, parameters, tf2
        ├── my_robot_interfaces/   # Lesson 6: SolveIK.srv and MoveTo.action
        ├── my_robot_description/  # Lessons 9–10: URDF, visualization, mock hardware
        ├── my_robot_planner/      # Lessons 10–11: planning, targets, obstacles
        └── my_robot_bringup/      # Lesson 11: system launch
```

The ROS experiments target **Ubuntu 24.04, ROS2 Jazzy, system Python 3.12, and Bash**. Install ROS, colcon, and rosdep following [lesson 1](../../../docs/foundations/robotics-and-ros2/1.course_introduction.md) (Chinese). Lessons 2–3 only need Python and NumPy and can run on macOS, Windows, or Linux. The website's npm dependencies do not install Python or ROS dependencies.

From the repository root:

```bash
export ROS_WS="$(pwd)/codes/foundations/robotics-and-ros2/ros_ws"
echo "$ROS_WS"
```

Set this variable in each new terminal, or add `export ROS_WS="..."` with the printed absolute path to `~/.bashrc`. Keep `reference/COLCON_IGNORE`. Exercise files and `build/`, `install/`, and `log/` are ignored by Git.

## Run the kinematics scripts

Create a separate environment for the math scripts and checks; do not use it to build ROS packages:

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

## Lessons 4–9: exercises and reference packages

Create your packages in `$ROS_WS/src` using `ros2 pkg create` as each lesson explains, then compare individual files with `reference/src`. The reference version incorporates later changes: lesson 7 parameters and dynamic publication frequency, startup parameters for the Action server, and lesson 10 `ros2_control` tags in the URDF. Do not insert those tags again after copying the reference URDF. `SolveIK` uses link lengths from each request, rather than YAML defaults. The Action example demonstrates goals, feedback, and results; cancellation is not implemented.

To try the reference packages first, copy these three packages into a workspace **without exercise packages of the same names**. If you have existing exercises, compare or copy individual files to preserve your work.

```bash
source /opt/ros/jazzy/setup.bash
cd "$ROS_WS"
cp -rn reference/src/my_robot_interfaces reference/src/my_robot_basics reference/src/my_robot_description src/
rosdep install --from-paths src --ignore-src --rosdistro jazzy -r -y
colcon build --symlink-install --packages-up-to my_robot_basics my_robot_description
source install/setup.bash
ros2 run my_robot_basics hello_node
```

Stop `hello_node` with Ctrl+C. Other entry points, to run separately as needed:

```bash
ros2 launch my_robot_basics demo.launch.py publish_rate:=2.0
ros2 launch my_robot_description display.launch.py
```

In a new terminal, source `/opt/ros/jazzy/setup.bash` and `"$ROS_WS/install/setup.bash"`. Follow lessons 8–9 when switching between manual TF and URDF experiments; avoid publishing the same transforms from both your broadcaster and `robot_state_publisher`.

## Lessons 10–11: generate the MoveIt configuration

`my_robot_moveit_config` is **not bundled**. Use MoveIt Setup Assistant in [lesson 10](../../../docs/foundations/robotics-and-ros2/10.moveit2_basics.md) to generate it at `$ROS_WS/src/my_robot_moveit_config`, and complete the SRDF, planner, joint-limit, and controller configuration described there. Then copy the launch file and remaining packages, again only when no exercise packages of those names exist:

```bash
cd "$ROS_WS"
cp -n reference/moveit/demo_sim.launch.py src/my_robot_moveit_config/launch/
cp -rn reference/src/my_robot_planner reference/src/my_robot_bringup src/
rosdep install --from-paths src --ignore-src --rosdistro jazzy -r -y
colcon build --symlink-install
source install/setup.bash
ros2 launch my_robot_bringup full_system.launch.py
```

From another terminal with the environment sourced:

```bash
ros2 topic pub --once /target_pose geometry_msgs/msg/Point "{x: 1.2, y: 0.8, z: 0.0}"
```

## Validation scope

`tests/` checks numerical results, both IK branches and workspace boundaries, Python syntax, package entry points and resource paths, and the URDF structure. These checks do not load ROS or validate nodes, RViz2, or MoveIt at runtime. Run the full experiments on Ubuntu / Jazzy following the lessons.
