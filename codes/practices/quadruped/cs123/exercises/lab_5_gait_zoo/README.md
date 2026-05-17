# Lab 5：步态动物园 (The Gait Zoo)

教程 [§5.1](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/gait-control#51-常见步态) 列了 walk / trot / pace / bound / gallop 五种步态，但 [§5.6](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/gait-control#56-原地踏步实验) / [§5.7](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/gait-control#57-前进实验) 只实现了 trot。其它步态还停在表格里。这是教程自己留下来的悬念：同一套 `leg_phase`、`foot_trajectory`、IK、`PD`，换几组相位数字会发生什么？

本 Lab 把这件事做成一张作品集动图。三只 welded Pupper 并排原地踏步：左边 trot，中间 pace，右边 bound。你会看到数学上只差 `offsets` 和 `duty`，视觉上却像三种动物。

## 为什么选这个任务

| 候选 | 教程覆盖 | 作品价值 |
|---|---|---|
| 调 `v_cmd` 走 1 米 | [§5.7](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/gait-control#57-前进实验) 已讲 | 容易变成调参 |
| **trot / pace / bound 三联画** | 只讲了 trot | **一眼看懂 phase pattern** |
| yaw 反馈修走偏 | [§5.8](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/gait-control#58-常见失败模式) 提到 | 留给 Lab 6 前后对比 |

三联画赢在变量少。你不写 CPG，不训练 RL，不做 ground race，只把"哪几条腿同相"画出来、跑出来、量出来。

## 你交什么

- `portfolio/deliverable.gif`：`gait_zoo.gif`，1280×480，12 秒，约 150 帧 @ 12 fps。它是本 Lab 头牌。
- `portfolio/deliverable.png`：`gait_gantts.png`，0–2 秒 Gantt 图，trot 呈 X，pace 呈 =，bound 呈 Z。
- `portfolio/base_z_fft.png` 和 `portfolio/notes.md`：FFT Stretch + 50–100 字观察。

## 起点与 TODO map

学生版 `starter.py` 已经接好 MJCF 加载、三只 Pupper 并行控制、`Kp/Kd`、offscreen render、GIF 压缩和画图。你只补三处算法。

| TODO | task | what to write |
|---|---|---|
| TODO 1 | `leg_phase()` | `(t / T_cycle + offset) % 1.0`，判断 stance/swing 并归一化 `s` |
| TODO 2 | `foot_trajectory()` | stance 直线，swing 用 `step_height * sin(pi*s)` |
| TODO 3 | `gait_step()` | 四条腿循环：phase → traj → `ik_pupper_leg` → 12 维目标角 |

## 分步任务

1. **gait factory（20 min）**：读 `GAITS` 字典，确认 trot / pace / bound 只在 `offsets` 和 `duty` 上不同。（对应教程 [§5.1](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/gait-control#51-常见步态) / [§5.2](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/gait-control#52-步态节奏图)）
2. **leg_phase（30 min）**：把全局时间映射成单腿局部相位。（对应教程 [§5.2](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/gait-control#52-步态节奏图)）
3. **foot_trajectory（40 min）**：stance 贴地走线段，swing 用 `sin(pi*s)` 抬脚，端点垂直速度为 0。（对应教程 [§5.3](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/gait-control#53-足端轨迹)）
4. **gait_step（30 min）**：把四条腿的足端目标交给 `shared.kinematics.ik_pupper_leg`。（对应教程 [§5.4](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/gait-control#54-轨迹到关节角)）
5. **zoo 闭环（30 min）**：三只 Pupper 在同一个 `pupper_zoo.xml` 里并行跑，控制器按 prefix 写入 12 维 actuator 块。（对应教程 [§5.6](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/gait-control#56-原地踏步实验)）
6. **Gantt 图（30 min）**：用数据图验证 [§5.1](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/gait-control#51-常见步态) 的分类，不从 GIF 里猜。（对应教程 [§5.2](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/gait-control#52-步态节奏图)）
7. **Stretch（可选）**：画 `base_z_fft.png`，比较 trot / pace / bound 的频谱。

## MuJoCo scene

本 Lab 第二次复用 Lab 4 的 `<asset><model>` + `<attach prefix>` 套路。Lab 4 用它并排展示三只 URDF 变体；这里用它并排展示三种 gait。

```xml
<asset>
  <model name="pupper_on_stand" file="../../shared/models/pupper_v3_on_stand.xml"/>
</asset>
<worldbody>
  <frame pos="-0.6 0 0"><attach model="pupper_on_stand" body="base" prefix="trot/"/></frame>
  <frame pos="0 0 0"><attach model="pupper_on_stand" body="base" prefix="pace/"/></frame>
  <frame pos="0.6 0 0"><attach model="pupper_on_stand" body="base" prefix="bound/"/></frame>
</worldbody>
```

`shared/models/pupper_v3_on_stand.xml` 在 `skeleton.xml` 上补了一条 base-world `<weld>`。删掉这条 weld，再接 ground 模式，就是后面 Lab 6 要面对的真实欠驱动问题。

## Rubric

- `leg_phase(0, "FL", trot)` 返回 stance，`s=0`；trot 的对角腿同相。
- `foot_trajectory(..., in_stance=False)` 在 `s=0` 和 `s=1` 的 z 都等于 `-stand_height`。
- welded 场景下 trot 的 `base z` 标准差 < 5 mm；pace 的侧滚激励明显大于 trot。
- Gantt offset 关系正确：trot 对角同相，pace 同侧同相，bound 前后同相。

## 常见坑

- 不要把 swing 写成普通抛物线；端点速度不为 0，会复现教程 [§5.8](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/gait-control#58-常见失败模式) 的"滑步"。
- 不要重写 IK；Lab 3 已经做过 DLS，本 Lab 直接调 `ik_pupper_leg`。
- 不要在 ground 上录头牌 GIF；pace / bound 12 秒内会把注意力从 phase pattern 带到摔倒姿态。
- 不要开 viewer 录 GIF；本 Lab 统一走 offscreen renderer 和 `gif_utils.write_gif`。

## 教程衔接

- **复用**：教程 [§5.2](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/gait-control#52-步态节奏图) 的 `leg_phase`、[§5.3](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/gait-control#53-足端轨迹) 的 `foot_trajectory`、[§5.4](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/gait-control#54-轨迹到关节角) 的 `gait_step`、[§5.6](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/gait-control#56-原地踏步实验) 的原地踏步控制循环。
- **扩展**：把"一种 trot"推广成 gait config 空间，并用 Gantt 图把 [§5.1](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/gait-control#51-常见步态) 的分类落到数据上。
- **不重复**：教程 [§5.7](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/gait-control#57-前进实验) 的按 `v_cmd` 走 1 米是前置练习；本 Lab 不做 ground displacement。

## 不做什么

本 Lab 不实现 CPG。教程 [§5.5](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/gait-control#55-cpg-简述) 只是解释相位概念从哪里来，工程上手写 trot 不需要 CPG。

本 Lab 不做 yaw 反馈，也不做 RL 训练。教程 [§5.7](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/gait-control#57-前进实验) 已经预告手写 trot 注定走不直；Lab 6 会把"何时抬腿"端到端学出来，这就是 Lab 5 → Lab 6 的 forward-ref。

## Run

从 `exercises/` 运行：

```bash
uv run python shared/kinematics/test_leg_kinematics.py   # shared 4-leg IK 自检
uv run python lab_5_gait_zoo/tests.py                    # 四条数值断言
uv run python lab_5_gait_zoo/starter_todo.py             # 学生起点：调用 leg_phase / gait_step 时会触发 NotImplementedError
uv run python lab_5_gait_zoo/starter.py                  # 参考答案 / 快速环境检查
uv run python lab_5_gait_zoo/make_artifacts.py           # 写 portfolio GIF / PNG / FFT
```
