# Lab 3：Pupper 单腿 Raibert 悬空踏步

教程 [§3.8](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/inverse-kinematics#38-实验末端走三角形) 让一个平面机械臂末端走抽象三角形。三个点只是几何点，还没有 touchdown、standing、liftoff 这些物理含义。真实四足的一步也像三角形，但 stance 是脚贴地向后扫，swing 是脚抬起回到前方，中间要有 mid-swing 高点。

这个 Lab 把 [§3](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/inverse-kinematics) IK 第一次装到真 Pupper 腿上：Lab 2 的单腿被 `<weld>` 挂在空中，`raibert_foot_traj` 生成足端目标，DLS IK 把目标换成关节角，PD 再把关节角变成 actuator torque。Lab 5 的 trot 会把同一件事复制到四条腿并加相位偏移。

## 为什么是这件事

| 候选任务 | 教程是否已覆盖 | 出片程度 | 作品价值 |
|---|---|---|---|
| 画圆实验 | [§3.7](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/inverse-kinematics#37-实验末端画圆) 已讲 | 中 | 重复 |
| **Raibert 三角踏步** | 只在步态理论里提过 | 高 | Lab 5 的原子动作 |
| 6-DoF 机械臂 DLS | [§3.4](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/inverse-kinematics#34-dls-数值法) 已够用 | 中 | 离 Pupper 远 |

Raibert 三角赢在增量刚好：它不重复画圆和抽象三角形，只把三角形顶点物理化，再接到 Lab 2 的真腿和 Lab 1 的 PD 上。

## 你交什么

- `portfolio/deliverable.gif` = `leg_stepping.gif`：8 秒 / 720x400 / 15 fps，0.5 Hz 共 4 步，步幅 10 cm / 抬腿 5 cm（节奏对齐 `lab4` 的 `slow` 预设）。前 6 秒正常 Raibert 三角，最后 2 秒反例段去掉 mid-swing。
- `portfolio/deliverable.png` = `raibert_vs_triangle.png`：左边是教程 [§3.8](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/inverse-kinematics#38-实验末端走三角形) 抽象三角形，右边是 Raibert 三角，横轴 x，纵轴 z。
- `portfolio/leg_stepping_sweep.gif`：Stretch。12 秒内 `step_length` 从 0 扫到 0.12 m，足端轨迹从原地抬脚变成 D 字形。
- `portfolio/notes.md`：50–100 字反思，写清楚你最容易写错的是 stance/swing 边界、DLS 阻尼，还是 mid-swing 顶点高度。

## 起点

打开 `starter.py`：MJCF 加载、`mj_jacSite` context、viewer 绿/红球、GIF 字幕、trail、plot 和测试入口都已经写好。你只填三处算法空白：

| TODO | 对应任务 | 干什么 |
|---|---|---|
| TODO 1 | 任务 B | `raibert_foot_traj` 内写 stance / swing 分支和 mid-swing 抛物线 |
| TODO 2 | 任务 C | `dls_ik_step` 内写一次 DLS 迭代：`mj_forward`、`mj_jacSite`、阻尼伪逆、更新 q |
| TODO 3 | 任务 D | 把 PD 算出的 `torques` 写进 `data.ctrl[...]` |

本目录不写 `solution.py`。

## 分步任务

1. **任务 A · 把腿挂起来（15 min）**：在 `scene.xml` 里用 `<weld body1="hip" body2="world"/>` 把 hip 固定到世界，跑 `check_scene_mounted(model)` 确认只有这一条 equality。（对应教程 [§4.1](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/quadruped-mjcf#41-pupper-结构) 的 MJCF 语法，复用 Lab 1 的 `<equality>` 直觉）
2. **任务 B · Raibert 三角轨迹（40 min）**：实现 `raibert_foot_traj(t)`，stance 沿 -x 后扫，swing 用抛物线抬起再落下，支持 `step_length=0` 原地踏步。（对应教程 [§3.8](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/inverse-kinematics#38-实验末端走三角形) 的三角形骨架）
3. **任务 C · DLS IK（40 min）**：用教程 [§3.4](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/inverse-kinematics#34-dls-数值法) 的 $\Delta q = J^\top(JJ^\top+\lambda^2I)^{-1}e$，雅可比从 `mj_jacSite` 取 foot site 的 3x3 平移部分。（对应教程 [§3.4](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/inverse-kinematics#34-dls-数值法) / [§3.6](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/inverse-kinematics#36-ik--pd-控制)）
4. **任务 D · 闭环踏步（30 min）**：每帧目标位置 → DLS IK → PD torque → `mj_step`，并在 viewer 叠绿色目标球和红色 IK foot 预测。（对应教程 [§3.9](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/inverse-kinematics#39-viewer-看-dls-收敛)）
5. **任务 E · step_length 扫描（Stretch）**：把 `step_length` 从 0 线性扫到 0.12 m，录一段从原地踏步变成迈步的 GIF。（预告 Lab 5 phase/duty）

## MuJoCo 场景

`models/scene.xml` 继续引用共享单腿和唯一 mesh 源，不复制 STL。Lab 3 只多一条 mounted 约束：

```xml
<compiler angle="radian" meshdir="../../shared/models/meshes/"/>
<include file="../../shared/models/leg.xml"/>
<equality>
  <weld body1="hip" body2="world"
        relpose="-0.075 -0.0835 0.5 0 0 0.70710828 -0.70710528"/>
</equality>
```

> 关于这条 `<weld>`：`leg.xml` 里 `base` body 没有任何 joint，本身就被 MJCF 默认刚性挂在世界 (0,0,0.5)。Lab 2 没加 weld 也能让腿"悬空"。所以这条约束在物理上其实是**冗余**的——它和"`base` 无 joint" 表达的是同一件事，只是落在了 `<equality>` 块里。它真正的作用是**教学对照**：Lab 1 用 `<equality joint polycoef>` 锁两个关节、Lab 2 整块删掉解锁 3 DoF、Lab 3 用 `<equality weld>` 锁 body——同一段 XML 三种语义。`relpose` 写的是 HAA=0 时 hip 在世界系的实际 pose，所以 HAA 名义可动，但只要 HAA 偏离 0，weld 就会以 soft 等式的形式跟 PD 抢一点力矩。本 Lab 的目标轨迹整段在矢状面（`FOOT_Y` 固定），HAA 始终≈0，因此这点抢力矩看不出来。如果你想做"侧向 swing"或者后续 Lab 想真的 free-base，记得改成 `<freejoint/>` 而不是再加 weld。

Lab 4 会把这份 `leg.xml` `<include>` 四次组成整机，base 也加上 `<freejoint/>` 真的浮起来，weld 自然消失，整机改靠地面接触和 PD 站住。

## 评分点

- `dls_ik(q0, target=fk_leg(q_truth))` 收敛后 `||fk_leg(q) - target|| < 1e-3`
- 把 DLS 阻尼写成 `lam=0` 后，在奇异附近应跑到 `max_iter`，且残差 `> 1e-2`
- `raibert_foot_traj(0.0)` 和 `raibert_foot_traj(period)` 的闭合误差 `< 1e-9`
- 跑 8 秒闭环无 viewer，`base z` 标准差 `< 1e-6`

## 常见坑

1. DLS 阻尼太小会在奇异点附近震荡，`lam=0.05` 是这条腿的稳妥起点。
2. `mj_jacSite` 前必须先 `mj_forward`，否则雅可比对应的是上一帧的姿态。
3. mid-swing 顶点过高会让目标点接近工作空间边缘，IK 残差会变大。
4. `weld relpose` 的 quat 要归一；写错会让腿一加载就歪。
5. 教程 [§3.4](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/inverse-kinematics#34-dls-数值法) 给的 DLS 默认 `step=0.3, max_iter=200, tol=1e-4`。本 Lab 的评分点是 50 步内残差 < 1e-3，这个预算下 `step=0.3` 在 `q_truth=(0,-0.9,1.3)` 这条目标上经常停在 ~1.1e-3。所以 `starter.py` 里把 `step` 抬到 0.5——记得这个改动只换迭代步长，没换公式。
6. 红球画的是 `fk_leg(q_target)`，也就是 IK 解出的足端预测，不是 `data.site_xpos[foot]`。这和教程 [§3.9](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/inverse-kinematics#39-viewer-看-dls-收敛) 的红球语义不同：教程红球是"实际末端"，跟绿球比可以同时看 IK 和 PD；本 Lab 红球只查 IK 是否自洽，PD 跟踪是否够紧请看 `tests.py` 里的 `base_z` 标准差和 `q_actual - q_target` 的差值。

## 与教程的衔接

- **复用**：教程 [§3.4](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/inverse-kinematics#34-dls-数值法) 的 DLS、[§3.6](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/inverse-kinematics#36-ik--pd-控制) 的 IK+PD 串联、[§3.8](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/inverse-kinematics#38-实验末端走三角形) `interpolate_triangle` 骨架、[§3.9](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/inverse-kinematics#39-viewer-看-dls-收敛) viewer 绿/红球。
- **扩展**：把三个顶点改成 touchdown / standing / liftoff；引入 mid-swing 抛物线；第一次在 Pupper 真腿上跑 IK + PD。
- **不重复**：教程 [§3.7](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/inverse-kinematics#37-实验末端画圆) 画圆和 [§3.8](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/inverse-kinematics#38-实验末端走三角形) 抽象三角形是前置作业，本 Lab 不再复刻。

## 运行

命令都从 `exercises/` 目录里跑：

```bash
uv run python lab_3_stepping/starter_todo.py    # 学生起点：会在 TODO 2 处报 NotImplementedError
uv run python lab_3_stepping/starter.py         # 参考答案：直接跑通，打印 IK 自检 residual / iters
uv run python lab_3_stepping/make_artifacts.py  # 写出 leg_stepping.gif / raibert_vs_triangle.png / sweep GIF
uv run python lab_3_stepping/tests.py           # 四条 assert 全过
```
