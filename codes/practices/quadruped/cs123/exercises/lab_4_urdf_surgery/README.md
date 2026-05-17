# Lab 4：URDF 手术，给 Pupper 换条腿

教程 [§4.5](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/quadruped-mjcf#45-让-pupper-站起来) 让原版 Pupper 第一次站住。那是起点，不是终点。你现在要主动改机器人本体：一只原版、一只 long-leg、一只 heavy，三只都要重新找 `stand_pose`，再扫一遍 `PD` 甜点。

这件事是教程 [§4.5](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/quadruped-mjcf#45-让-pupper-站起来) 站立脚本和 [§4.6](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/quadruped-mjcf#46-调-gainprm-与-biasprm) `Kp/Kd` 对照的延伸：不再只是跑别人调好的 URDF，而是改完 MJCF 之后自己承担后果。Lab 5 的 trot 会继续用这三只 Pupper，它们会有完全不同的步频甜点。

## 为什么是这件事

| 候选任务 | 教程是否已覆盖 | 是否主动改机器人 | 作品价值 |
|---|---|---|---|
| 4-leg FK | 教程 [§4.1](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/quadruped-mjcf#41-pupper-结构) / [§4.5](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/quadruped-mjcf#45-让-pupper-站起来) 已覆盖结构与站姿 | 是 | 中，但不出片 |
| 原版 Pupper 站立 | [§4.5](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/quadruped-mjcf#45-让-pupper-站起来) 已演示 | 否 | 重复 |
| `Kp/Kd` 四格对照 | [§4.6](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/quadruped-mjcf#46-调-gainprm-与-biasprm) 已演示 | 否 | 重复 |
| **URDF 手术：三只 Pupper 变体** | 没覆盖 | **是** | **heatmap + 三联静帧** |

它赢在因果链清楚：改 `fromto` / `mass` 之后，原来的 `stand_pose` 和 `PD` 不再可靠。你必须把 URDF 五件套、站姿和控制器放在同一张桌上处理。

## 你交什么

- `portfolio/deliverable.png`：三张 4x4 `PD` 甜点 heatmap，颜色是最后 1 秒 base z 标准差。
- `portfolio/pupper_zoo.png`：三只 Pupper 的最终站姿静帧，用来检查 original / long-leg / heavy 的结构差异。
- `portfolio/stand_z_vs_t.png`：Stretch，三只调好 `PD` 后的 base z 时间序列。
- `portfolio/notes.md`：50–100 字反思，写 long-leg 为什么常把甜点推向更高 `Kp`。

## 起点

打开 `starter.py`：MJCF include、三变体写出、zoo 静帧渲染、heatmap、pickle 都已接好。你只填三处算法空白：

| TODO | 对应任务 | 干什么 |
|---|---|---|
| TODO 1 | 任务 A | `make_variant()` 里决定 `fromto`、`mass`、foot site 怎么随 `leg_scale` / `torso_mass_scale` 变 |
| TODO 2 | 任务 B | `find_stand_pose()` 给定腿长，求 HFE/KFE，HAA 保持 0 |
| TODO 3 | 任务 C | `pd_sweet_spot()` 写 `PD` torque，跑网格，选 `(Kp, Kd)` |

本目录不写 `solution.py`。

## 分步任务

1. **任务 A · 变体 factory（30 min）**：从 `shared/models/skeleton.xml` 注入 default class，写出 `pupper_v3.xml`、`pupper_longleg.xml`、`pupper_heavy.xml`。（对应教程 [§4.2.3](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/quadruped-mjcf#423-加载-mesh) 的 `compiler` / `asset` / `mesh` 装配）
2. **任务 B · stand_pose 求解（40 min）**：把单腿看成二维链，搜索 HFE/KFE，让 foot 回到合理站高。（对应教程 [§4.5](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/quadruped-mjcf#45-让-pupper-站起来) 的站立脚本）
3. **任务 C · PD 甜点扫描（50 min）**：在 $K_p \in \{10,30,60,120\}$、$K_d \in \{0.5,1,2,5\}$ 上跑 6 秒，记录最后 1 秒 base z 标准差。（对应教程 [§4.6](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/quadruped-mjcf#46-调-gainprm-与-biasprm) 的 `Kp/Kd` 崩坏模式）
4. **任务 D · zoo 静帧（20 min）**：用每只 Pupper 的甜点 `PD` 跑到稳定后，渲染三只并排最终站姿。（对应教程 [§4.5](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/quadruped-mjcf#45-让-pupper-站起来) 的站立脚本，把 `launch_passive` 换成离屏 `Renderer`）
5. **任务 E · base z 曲线（Stretch）**：把三条调好后的 base z 叠在一张图上，检查是否都稳。

## MuJoCo 场景

三份变体文件不复制整棵树，只在 include 前注入 class default：

```xml
<compiler angle="radian" meshdir="../../shared/models/meshes/" autolimits="true"/>
<default>
  <default class="variant_thigh"><geom fromto="0 0 0 0 0 -0.12" mass="0.279"/></default>
  <default class="variant_calf"><geom fromto="0 0 0 0 0 -0.165" mass="0.075"/></default>
</default>
<include file="../../shared/models/skeleton.xml"/>
```

`pupper_zoo.xml` 用 MuJoCo `asset/model` + `attach prefix` 装三只 Pupper。这样三份变体仍然各自 include 同一份 skeleton，但 zoo 里不会出现重复的 joint / body 名字。Lab 4 → Lab 5 的解锁路径是：同一份整机 MJCF 不再只站立，下一章会给四条腿加相位和步频。

## 评分点

- `make_variant("longleg", leg_scale=1.5)` 写出的 MJCF 能被 `mujoco.MjModel.from_xml_path` 编译。
- long-leg 的 thigh `fromto` 长度相对原版约为 1.5 倍。
- 三只 Pupper 用各自甜点 `PD` 站立时，最后 1 秒 base z 标准差 `< 5 mm`。
- 原版 Pupper 的 heatmap 不应选择过软格，`Kp >= 30` 且 `Kd >= 1.0`。
- 三份变体 include `shared/models/skeleton.xml` 后结构一致，heavy 总质量相对原版明显变大。

## 常见坑

1. `qpos` 前 7 维是 free base，`qvel` 前 6 维是 free base，PD 只能作用在后面 12 个关节上。
2. `Kp` 过小可能“稳稳地趴着”，所以只看最终高度不够，要看扰动下最后 1 秒 z 的标准差。
3. long-leg 改了 `fromto` 还要改 foot site，否则视觉变长但接触点没跟着走。
4. 三只 Pupper 放进一个 MJCF 时要 namespacing；直接 `<include>` 三份会撞名。

## 与教程的衔接

- **复用**：教程 [§4.2](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/quadruped-mjcf#42-搭建-pupper-仿真模型) 五件套、[§4.2.3](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/quadruped-mjcf#423-加载-mesh) `compiler` / `asset`、[§4.5](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/quadruped-mjcf#45-让-pupper-站起来) 站立脚本、[§4.6](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/quadruped-mjcf#46-调-gainprm-与-biasprm) `Kp/Kd` 对照。
- **扩展**：把“读别人的 URDF”变成“改 URDF 后还要负责把它调回能用的状态”。
- **不重复**：不做 4-leg FK；不重复原版 Pupper 站立；不重画教程 [§4.6](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/quadruped-mjcf#46-调-gainprm-与-biasprm) 那张 4 行 PD 表，本 Lab 自己跑 16 格 heatmap。

## 运行

命令都从 `exercises/` 目录里跑：

```bash
uv run python lab_4_urdf_surgery/starter_todo.py    # 学生起点：打印提示语；TODO 1/2/3 在 tests.py 时会触发 NotImplementedError
uv run python lab_4_urdf_surgery/starter.py         # 参考答案：直接跑通，打印三只 Pupper 的甜点 (Kp, Kd, z_std)
uv run python lab_4_urdf_surgery/make_artifacts.py  # 写出 heatmap / pupper_zoo.png / stand_z_vs_t.png
uv run python lab_4_urdf_surgery/tests.py           # 四条 assert 全过
```
