# Lab 4：URDF 手术，给 Pupper 换条腿

教程 [§4.5](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/quadruped-mjcf#45-让-pupper-站起来) 让原版 Pupper 第一次站住。这个目录演示下一步：从同一份 `skeleton.xml` 派生出一只原版、一只 long-leg、一只 heavy，三只分别重新找 `stand_pose`，再扫描一遍 `PD` 参数。

这件事是教程 [§4.5](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/quadruped-mjcf#45-让-pupper-站起来) 站立脚本和 [§4.6](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/quadruped-mjcf#46-调-gainprm-与-biasprm) `Kp/Kd` 对照的延伸：不再只是跑别人调好的 URDF，而是改完 MJCF 之后自己承担后果。Lab 5 的 trot 会继续用这三只 Pupper，它们会需要不同的步频和控制参数。

## 为什么看这个例子

| 主题 | 教程是否已覆盖 | 这个例子补上的内容 |
|---|---|---|
| 4-leg FK | 教程 [§4.1](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/quadruped-mjcf#41-pupper-结构) / [§4.5](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/quadruped-mjcf#45-让-pupper-站起来) 已覆盖结构与站姿 | 不重复写 FK，而是看整机 MJCF 改动后的连锁反应 |
| 原版 Pupper 站立 | [§4.5](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/quadruped-mjcf#45-让-pupper-站起来) 已演示 | 不只站一只，而是三只变体都重新求站姿 |
| `Kp/Kd` 四格对照 | [§4.6](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/quadruped-mjcf#46-调-gainprm-与-biasprm) 已演示 | 从手动对比扩展成 4x4 heatmap |
| **URDF 手术：三只 Pupper 变体** | 没覆盖 | **把几何、质量、站姿、PD 参数和可视化串成一条链路** |

它赢在因果链清楚：改 `fromto` / `mass` 之后，原来的 `stand_pose` 和 `PD` 不再可靠。你必须把 URDF 五件套、站姿和控制器放在同一张桌上处理。

## 运行后看什么

- `portfolio/pd_heatmap.png`：三张 4x4 `PD` 参数 heatmap，颜色是最后 1 秒 base z 标准差。
- `portfolio/pupper_zoo.png`：三只 Pupper 的最终站姿静帧，用来观察 original / long-leg / heavy 的结构差异。
- `portfolio/stand_z_vs_t.png`：三只调好 `PD` 后的 base z 时间序列，用来观察是否持续抖动。

## 起点

打开 `starter.py`。当前仓库把完整实现直接放在 `starter.py`，学习方式是：先运行脚本确认结果，再按三个改动点读代码，最后自己改参数做二次实验。

| 阅读顺序 | 函数 | 看懂什么 |
|---|---|---|
| 1 | `make_variant()` | 如何用 `default class` 注入 `fromto`、`mass`、foot site，而不是复制整棵 body 树 |
| 2 | `find_stand_pose()` | 腿长变化后，为什么要重新搜索 HFE/KFE 站姿 |
| 3 | `_pd_step()` / `find_stable_pd_gains()` | floating base 模型里只对后 12 个关节做 PD，并用 4x4 网格找较优参数 |
| 4 | `render_zoo_still()` / `save_heatmap()` | 如何把仿真结果变成三联静帧和 heatmap |

## 阅读路径

1. **变体 factory**：`make_variant()` 从 `shared/models/skeleton.xml` 注入 default class，写出 `pupper_v3.xml`、`pupper_longleg.xml`、`pupper_heavy.xml`。重点看 thigh/calf 的 `fromto`、质量缩放、foot geom/site 位置，以及 heavy 的视觉配重块。
2. **stand_pose 求解**：`find_stand_pose()` 把单腿看成二维链，固定 `HAA=0`，搜索 HFE/KFE，让 foot 接近目标站高。long-leg 不能复用 original 的站姿。
3. **PD 参数扫描**：`_pd_step()` 取出 12 个关节索引，避开 `qpos` 前 7 维和 `qvel` 前 6 维的 floating base；`find_stable_pd_gains()` 在 $K_p \in \{10,30,60,120\}$、$K_d \in \{0.5,1,2,5\}$ 上跑 6 秒，记录最后 1 秒 base z 标准差。
4. **zoo 静帧**：`render_zoo_still()` 用每只 Pupper 的较优 `PD` 参数跑到稳定后，渲染三只并排最终站姿。
5. **base z 曲线（Stretch）**：`save_stand_z_plot()` 把三条调好后的 base z 叠在一张图上，检查是否都稳。

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
uv run python lab_4_urdf_surgery/starter.py         # 主脚本：直接跑通，打印三只 Pupper 的较优参数 (Kp, Kd, z_std)
uv run python lab_4_urdf_surgery/make_artifacts.py  # 写出 heatmap / pupper_zoo.png / stand_z_vs_t.png
```

想更深入时再运行 `uv run python lab_4_urdf_surgery/tests.py`，它只是帮你确认比例、质量、站姿稳定性这些数值细节。
