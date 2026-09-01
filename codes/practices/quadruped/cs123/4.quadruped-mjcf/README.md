# 4. 搭建四足机器人 · 配套代码

第 4 章 [搭建四足机器人](https://datawhalechina.github.io/dive-into-embodied-ai/docs/practices/quadruped/cs123/quadruped-mjcf) 的配套代码：Pupper v3 的 MJCF 模型，以及在 MuJoCo 里查看模型和保持站立的脚本。

## 目录结构

```
4.quadruped-mjcf/
├── run_view_pupper_fixed.py    # §4.4 看固定基座（mj_forward 静态）
├── run_view_pupper.py          # §4.5 看浮动基座（mj_step 落地 + 伺服拉回）
├── run_stand_pupper.py         # §4.6 让浮动基座站住并判稳
├── run_gain_sweep.py           # 可选扩展：扫描 gainprm / biasprm
└── pupper_variants/            # Pupper 形态与站姿探索
```

模型和 STL 网格统一存放在 `../assets/mjcfs/`，供第 4、5、6 章共用。

两份模型只差一个 `<freejoint/>`：`pupper_v3.xml` 给 `base_link` 加了浮动关节，机身有 6 个被动 DoF、会受重力（`nq=19, nu=12`）；`pupper_v3_fixed.xml` 把机身焊在世界里，只留 12 个驱动关节，方便先把腿单独调通。

## 环境准备

依赖由 `cs123` 目录下的 `uv` 环境统一管理，首次先同步：

```bash
uv sync            # 在 cs123 目录下执行
```

MuJoCo 的交互 viewer 在 macOS 上必须用 `mjpython` 启动（随 `mujoco` 一起装好）；Linux / Windows 直接用 `python`。

## 运行说明

所有命令都在 `cs123` 目录下执行。

### 观察固定基座

看机器人摆出 `home` keyframe 的静态姿态：

```bash
uv run python 4.quadruped-mjcf/run_view_pupper_fixed.py
# macOS 换成 mjpython：
# uv run mjpython 4.quadruped-mjcf/run_view_pupper_fixed.py
```

### 观察浮动基座

主循环用 `mj_step` 推进物理，能看到机身从高处落下、四脚触地、再被位置伺服拉回 `home` 目标角：

```bash
uv run python 4.quadruped-mjcf/run_view_pupper.py
# macOS 换成 mjpython：
# uv run mjpython 4.quadruped-mjcf/run_view_pupper.py
```

### 让 Pupper 站住

每帧把 `STAND_POSE` 写进 `data.ctrl`，关窗后打印 base z 的末段标准差作为判稳标准（< 5 mm 即通过）。想换站姿改脚本顶部的 `STAND_POSE` 即可：

```bash
uv run python 4.quadruped-mjcf/run_stand_pupper.py
# macOS 换成 mjpython：
# uv run mjpython 4.quadruped-mjcf/run_stand_pupper.py
```

### 可选：扫增益看刚度变化

批量模式不开窗口（用 Agg 后端），为每组 `Kp / Kd` 重新加载模型、只在内存里改 actuator 参数，结果写到 `4.quadruped-mjcf/outputs/`（CSV + 曲线 + GIF）：

```bash
uv run python 4.quadruped-mjcf/run_gain_sweep.py
# 加 --no-push 去掉 t=2s 的外力扰动，只看自由落体后站稳
```

macOS 上想开窗口单独看某一组参数：

```bash
uv run mjpython 4.quadruped-mjcf/run_gain_sweep.py --viewer default
```

### 探索形态与站姿

从同一份骨架派生 `original`、`long-leg` 和 `heavy` 三种模型，分别搜索站姿并验证稳定性：

```bash
uv run python 4.quadruped-mjcf/pupper_variants/run_pupper_variants.py
uv run python 4.quadruped-mjcf/pupper_variants/test_pupper_variants.py
```

生成结果写入 `4.quadruped-mjcf/outputs/pupper_variants/`。详细说明见 [`pupper_variants/README.md`](pupper_variants/README.md)。

> macOS 上不要走 `mjpython -m mujoco.viewer --mjcf=...`：mjpython 启动时已经 import 过一次 `mujoco.viewer` 来占用 GUI 主线程，再用 `-m` 经 runpy 跑第二次会撞出 `RuntimeError: Caught an unknown exception!`。用脚本入口 `mjpython <script>.py` 绕开它。
