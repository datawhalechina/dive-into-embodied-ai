# exercises/shared

`shared/` 放多个配套 Lab 会复用的资产。它是资源树，不是 Python 私有包，
所以目录名前面不加下划线。

## 内容

- `models/leg.xml`: 一条语义命名的 Pupper 单腿，关节名是 `HAA`、`HFE`、`KFE`。Lab 1 用 `<equality>` 锁住 `HAA` 和 `KFE`；Lab 2 删除这些锁之后，同一条腿就能作为 3 DoF 链复用。
- `models/skeleton.xml`: 完整 12-DoF Pupper skeleton（torso `<freejoint/>` + 4 条腿 × HAA/HFE/KFE）。Lab 4–8 用它做整机 include，变体只改 default class，不复制 mesh 或整棵 body tree。
- `models/pupper_v3_on_stand.xml`: 在 `skeleton.xml` 上补一条 base-world `<weld>`，给 Lab 5 做三只 Pupper 原地踏步的基础场景。
- `models/meshes/`: Pupper V3 STL 文件的唯一副本，从 `lab1/models/meshes/` 复制而来。
- `controllers/pd_controller.py`: PD 跟踪 Lab 共用的 dataclass 和数值工具函数。真正的 PD 公式仍然留在每个 `starter.py` 的 TODO 里。
- `kinematics/fk.py`: `rot_x` / `rot_y` / `rot_z` / `homogeneous_transform` 这类纯几何积木，以及 Lab 2/3 会复用的 Pupper 单腿 `fk_leg` 与 `T_*_FIXED` 常量。
- `kinematics/ik.py`: `dls_ik` 迭代框架，以及 `damped_pinv` / `pose_error` 这类 DLS IK 几何工具。Lab 3 的 DLS 一步迭代仍留在 `starter.py` 的 TODO 里。
- `kinematics/leg_kinematics.py`: Lab 5 起使用的 4-leg `ik_pupper_leg` wrapper。FR/FL/RL/RR 的 hip offset 和 HAA 镜像都在 helper 内部处理。
- `trajectories/raibert.py`: stance / swing 相位 helper。Raibert 三角轨迹方程不放在 shared，留给 Lab 3 的 TODO 1。
- `viz/gif_utils.py`: `imageio` 包装、GIF 字幕、MuJoCo 画面居中，以及 `max_frames` / `width` 压缩写出。
- `viz/plot_utils.py`: Matplotlib 主题工具函数、Bode 绘图函数和 3D workspace 散点样式。

## import 写法

在任意 `lab_*` 目录里，先把 `exercises/` 加进 `sys.path`，再 import：

```python
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.controllers.pd_controller import PDGains
from shared.kinematics import Transform, fk_leg
from shared.trajectories import stance_swing_phase
from shared.viz import gif_utils, plot_utils
```

MJCF 文件用相对路径 include 共享资产：

```xml
<compiler meshdir="../../shared/models/meshes/"/>
<include file="../../shared/models/leg.xml"/>
```

不要把 mesh 复制到单个 Lab 的 `models/` 目录里。后续 Lab 如果需要新的共用
工具函数，就放在这里，再从 Lab 里 import。

## `rl/` 子包

Lab 6 起使用的 RL 相关 helper：

- `rl/obs_helpers.py`：`base_local_gravity` / `foot_contact_indicator` / `joint_qpos_qvel_ids`，三个轻量函数，把 MuJoCo data 里的原始数据转成 RL 观测向量的常用分量。
- `rl/pd_residual_actuator.py`：`override_pd(model, kp, kv)`，把 `<general>` actuator 的 gainprm/biasprm 改写成 PD 控制器。沿 `lab_4_viewer._override_pd` 的写法，放到 shared 方便 Lab 6+ 复用。
- `rl/test_obs_helpers.py`：用 `lab4/models/pupper_v3_floating.xml` 做 fixture 的数值自检。
