# Lab 2：Pupper 单腿混合轴 FK——拖一拖就动

教程 [§2.6](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/forward-kinematics#26-平面-3-dof-手推) / [§2.7](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/forward-kinematics#27-numpy-与-mujoco) 里练的是 3-DoF **平面**臂：三个关节轴都是 $z$ 轴，链式乘法最干净。真实 Pupper 单腿也是 3-DoF，但语义上是混合轴：`HAA` 绕 $x$，`HFE` / `KFE` 绕 $y$。MJCF 是从 URDF 自动导出的，把这件事编码成了"局部 hinge `axis="0 0 1"` + 父 body 一个固定四元数"，所以你写 FK 时实际乘的还是 `rot_z`，混合轴信息全部塞在我们提前算好的 `T_*_FIXED` 矩阵里。

本 Lab 把 [§2](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/forward-kinematics) 的平面 FK 搬到 Pupper 真腿上：照着 MJCF 的 body 树写四段 `T_parent_child`，用三条滑杆拖动关节，再像教程 [§2.10](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/forward-kinematics#210-末端叠加可视化) 一样把自写 FK 算出的红球叠到 viewer 里。红球贴着 foot site，说明你的乘法顺序和那几个固定四元数都没读反。

## 为什么是这件事

| 候选任务 | 可在仿真完成？ | 出片程度 | 留下的作品 |
|---|---|---|---|
| ROS2 topic + RViz（原 Lab 2） | 需 ROS，不行 | 中 | RViz 截图 |
| **Pupper 单腿 FK + MuJoCo 滑杆遥操作** | 可以 | **高** | **GIF + workspace 散点** |
| 加 IK（提前做 [§3](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/inverse-kinematics) 内容） | 可以 | 高 | 但重复 [§3](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/inverse-kinematics) |

滑杆遥操作赢在代码少、反馈快。Lab 1 已经把 Pupper 单腿画出来了，只是 `HAA` 和 `KFE` 被 `<equality>` 锁住；这里删掉那一块，三关节就变成可动的 FK 链。你第一次会真切看到：轴向或乘法顺序错一点，红球马上飞走。

## 你交什么

- `portfolio/deliverable.png` = `leg_workspace.png`：20000 个 Monte Carlo 点的 3D foot 工作空间散点，alpha=0.1
- `portfolio/deliverable.gif` = `leg_teleop.gif`：10 秒 / 720×400 / 15 fps；前 8.5 秒三关节脚本扫描、红球贴 foot site，最后 1.5 秒做反例段：故意把 HFE 写成 `rot_x`，红球应当明显飞离 foot
- `portfolio/notes.md`：50–100 字反思，写清楚你最容易写错的轴、符号或平移顺序

## 起点

打开 `starter.py`：MJCF 加载、关节 / site / actuator 名字解析、`Renderer` 包装、viewer 主循环、tkinter 滑杆骨架、workspace 出图和 GIF 录制都已经写好。你只填三处算法空白：

| TODO | 对应任务 | 干什么 |
|---|---|---|
| TODO 1 | 任务 B | 写 `fk_leg(theta)` 的 4 段齐次变换链，返回 foot 世界坐标 |
| TODO 2 | 任务 C | 在 `fk_validate` 里调用 `mj_forward`，取 `data.site_xpos[foot]`，返回最大误差 |
| TODO 3 | 任务 E | 在关节限位内随机采样，调用 `fk_leg` 得到 $n\times3$ 工作空间点 |

本目录不写 `solution.py`。

## 分步任务

1. **任务 A · 解锁单腿（10 min）**：从 Lab 1 的 `scene.xml` 起步，删掉 `<equality>` 整块；三路 motor 已在 `shared/models/leg.xml` 里，跑 `mj_forward` 确认 `HAA/HFE/KFE` 都能动。
2. **任务 B · 写 FK chain（40 min）**：按父子链写 `T_world_HAA`、`T_HAA_HFE`、`T_HFE_KFE`、`T_KFE_foot`，连乘后读第 4 列前三行。（对应教程 [§2.6](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/forward-kinematics#26-平面-3-dof-手推)）
3. **任务 C · 对齐 MuJoCo（20 min）**：随机 100 组关节角，用 `mj_forward` 作为真值，要求 `max_err < 1e-10`。（对应教程 [§2.7](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/forward-kinematics#27-numpy-与-mujoco)）
4. **任务 D · 滑杆遥操作（30 min）**：`mujoco.viewer.launch_passive` 跑 viewer，tkinter 线程读三条滑杆，主线程更新 `data.qpos` 并叠红球。（对应教程 [§2.10](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/forward-kinematics#210-末端叠加可视化)）
5. **任务 E · 工作空间葫芦（20 min）**：Monte Carlo 采样 20000 组关节角，画出 3D foot 点云；它是教程 [§2.9](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/forward-kinematics#29-工作空间散点) 平面甜甜圈的 Pupper 版。

## MuJoCo 场景

`models/scene.xml` 继续引用共享单腿和唯一 mesh 源，不复制任何 STL：

```xml
<compiler angle="radian" meshdir="../../shared/models/meshes/"/>
<include file="../../shared/models/leg.xml"/>
<asset>
  <texture name="lab2_grid" type="2d" builtin="checker" .../>
  <material name="lab2_grid" texture="lab2_grid" texrepeat="1 1"/>
</asset>
<worldbody>
  <geom name="floor" type="plane" size="0.7 0.7 0.02" material="lab2_grid"/>
  <camera name="iso" pos="0.55 -0.9 0.72"/>
</worldbody>
```

和 Lab 1 相比，这里就是删掉 `<equality>` 整块；`HAA/HFE/KFE` 的 motor 已经在 `shared/models/leg.xml` 里。Lab 3 会在这份 XML 上**再加一行 `<weld>` 把 hip 锁到世界**变成踏步支架，Lab 4 会把它 `<include>` 四次组成整机。

## 评分点

- `fk_validate(seed=0, n=100)` 的 `max_err < 1e-10`
- 故意把 HFE 那段写成绕 $x$ 时，`fk_validate` 的 `max_err > 1e-3`
- `sample_workspace(20000, seed=0)` 返回 `(20000, 3)`，且 foot 的 $z$ 范围覆盖约 `0.35 m` 到 `0.65 m`
- GIF 前 8.5 秒红球始终贴在 foot site 上，右侧字幕显示当前 `(HAA, HFE, KFE)`；最后 1.5 秒反例段红球离开 foot site，字幕变红提示 "HFE 误写成 rot_x"

## 常见坑

1. **MJCF 不是平面臂**。共享 MJCF 里 `HAA/HFE/KFE` 都是 `axis="0 0 1"`，真实的 $R_x$ / $R_y$ 语义被父 body 的 `quat="..."` 吸收了。所以你乘的是 `rot_z`，但每段前面要先乘一个我们提前算好的 `T_*_FIXED`——读错或漏掉那一段，红球马上飞。
2. **顺序敏感**：`T_world_HAA = T_WORLD_BASE @ T_BASE_HIP_FIXED @ rot_z(haa)`，不是 `rot_z(haa) @ T_BASE_HIP_FIXED`。父 body 的固定姿态永远在 hinge 角左边。
3. `T_parent_child` 不要倒着乘。FK 的位置永远从最终 `T_world_foot[:3, 3]` 读。
4. tkinter 只负责滑杆数值，viewer 主线程负责 `mj_forward` 和 `viewer.sync()`；两个 GUI 混在一个线程里容易卡住。

## 与教程的衔接

- **复用**：Lab 1 的 `scene.xml` 骨架；教程 [§2.7](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/forward-kinematics#27-numpy-与-mujoco) 的 `rot_*` / `trans` 小工具、[§2.9](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/forward-kinematics#29-工作空间散点) 的 Monte Carlo 散点、[§2.10](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/forward-kinematics#210-末端叠加可视化) 的 viewer 叠加红球。
- **扩展**：第一次处理混合轴；第一次把 FK 接到交互式 UI；第一次把一条 Pupper 腿的三个关节都解锁出来。
- **不重复**：教程的平面 3-DoF 练习作为前置作业，本 Lab 不再复刻。

## 运行

命令都从 `exercises/` 目录里跑：

```bash
uv run python lab_2_fk_teleop/starter_todo.py        # 学生起点：会在 TODO 1 处报 NotImplementedError
uv run python lab_2_fk_teleop/starter.py             # 参考答案：直接跑通，打印零位 / 100 组随机姿态的 max_err
uv run python lab_2_fk_teleop/starter.py --viewer    # 任务 D：tkinter 三滑杆 + viewer 实时叠红球
uv run python lab_2_fk_teleop/make_artifacts.py      # 写出 leg_workspace.png / leg_teleop.gif 和 deliverable.*
uv run python lab_2_fk_teleop/tests.py               # 三条 assert 全过
```
