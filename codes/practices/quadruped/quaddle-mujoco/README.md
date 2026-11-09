# Quaddle MuJoCo 仿真与残差强化学习

已在 Apple Silicon macOS、Python 3.12.13、MuJoCo 3.16.0 上运行并验证。

这是基于 Petoi 官方 URDF 的**本地适配工程**。保留官方网格、连杆位置和实体惯性，补齐物理仿真配置。支持持续向前步行、停止站立和小幅关节摆动。当前默认使用四腿错相交替的步态，优先减小机身起伏和滑移，并用方向反馈修正左右步幅。没有使用强化学习或 Petoi 官方步态。

控制器只写四个关节的位置目标。机身位置由 MuJoCo 的动力学和地面接触推进，没有平移机体、传送坐标或施加外部推动力。本模型的弹簧、接触与舵机动态仍有近似，未经过实机标定。

## 网页交互

仓库根目录运行 `npm install` 和 `npm run dev`，进入网站的“项目实战 → 仿真实战 → Petoi Quaddle 四足步行仿真”。加载后可以行走、停止站立、暂停物理、重置和调整视角，无需本地 Python 服务。

网页用官方 `@mujoco/mujoco@3.16.0` 在 Web Worker 内模拟相同的 MJCF，Three.js 只渲染引擎给出的几何体变换。`web/controller.mjs` 对应 Python 控制器，`web/simulation.mjs` 同时供浏览器和 Node 验证使用。网站构建插件 `scripts/docusaurus-quaddle.cjs` 从这里的模型、步态、Worker 源码以及锁定的 npm 包生成同源静态资源，不依赖 CDN。生成的 `static/quaddle/` 不进 Git；修改源文件后重新构建即可。

在仓库根目录运行 `npm run check:quaddle-web` 可用真实 WASM 引擎验证 60 秒行走、停止、重置和无外加力。`reports/web_validation.json` 保存一次结果；60 秒包含控制器的 2 秒起步等待，与下方原生验证的“额外站立 2 秒 + 行走 60 秒”计时口径不同。

## 残差 PPO 实验

`rl_env.py`、`train_ppo.py` 和 `evaluate_ppo.py` 提供独立的 Python 训练链路。策略以 100 Hz 读取 32 维观测，输出四个 ±0.12 rad 的关节目标修正量。网页默认仍使用周期控制器，没有加载短训练权重。

```bash
uv venv --python 3.12 .venv-rl
uv pip sync --python .venv-rl/bin/python requirements-rl.lock
.venv-rl/bin/python check_rl.py
.venv-rl/bin/python train_ppo.py --smoke --steps 2048 --envs 2 --output outputs/ppo-smoke
.venv-rl/bin/python evaluate_ppo.py --model outputs/ppo-smoke --episodes 2

# 从头开始的正式实验示例；不保证收敛或优于基线。
.venv-rl/bin/python train_ppo.py --steps 1000000 --envs 4 --seed 0 --output outputs/ppo-seed0
.venv-rl/bin/python evaluate_ppo.py --model outputs/ppo-seed0 --episodes 10 --randomize
```

`reports/rl_smoke.json` 记录接口检查、短训练及重新加载后的对照评估。短训练后的名义场景回报低于零残差基线，不能作为已优化策略发布。模型权重、观测归一化统计量和版本元数据保存在同一个 `outputs/ppo-*/` 目录，评估时必须配套使用。教程包含观测顺序、奖励项、PPO 更新过程、随机化范围和接回网页的部署约束。

## 启动

在仓库根目录首次安装（Python 3.12，使用独立虚拟环境）：

```bash
git lfs pull  # 确保 STL 网格已经下载为真实文件
cd codes/practices/quadruped/quaddle-mujoco
uv venv --python 3.12 .venv
uv pip sync --python .venv/bin/python requirements.lock
```

之后双击本目录中的 `run.command`，或在本目录运行：

```bash
./run.command --mode walk
```

默认 `./run.command` 即为步行模式。先站立 2 秒，再用 1.5 秒逐渐启动步态。

| 按键 | 功能 |
| --- | --- |
| W | 启动步行 |
| S | 平滑停止，保持站立 |
| M | 轮换站立 / 小幅摆动 / 步行 |
| R | 重置位置和仿真时间 |
| Space | 暂停 / 继续 |
| + / - | 调节小幅摆动模式的幅度，默认 0.10 rad；不改变步态参数 |
| 鼠标拖动 / 滚轮 | 旋转视角 / 缩放 |
| Tab / Shift+Tab | 显示 MuJoCo 原生侧栏 |

关闭窗口即可停止。使用 `--duration 10` 可让窗口运行 10 秒后退出。

地面显示已修正：`ground` 使用 `size="0 0 0.1"` 无限平面，棋盘设置 `texuniform="true"`，按世界坐标重复。机器人走出原先 4×4 米的显示范围后仍能看到地面，不需要重置位置。物理碰撞和步态参数不变。参数含义见 [MuJoCo plane 文档](https://mujoco.readthedocs.io/en/stable/XMLreference.html#body-geom-type)。

网站教程位于 `docs/practices/quadruped/quaddle-mujoco/index.md`（相对于仓库根目录）。`reports/` 保存随工程附带的验证结果；每次运行产生的文件写到 `outputs/`，虚拟环境、缓存和临时输出不进入 Git。

macOS 的被动查看器需要 `mjpython`。`launch.py` 为 uv Python 的动态库设置了**仅对当前进程生效**的查找路径，解决其 `libpython3.12.dylib` 加载错误；没有修改系统配置。

## 已验证的结果

旧版主要奖励前进距离，出现前腿小幅摆动、后腿大幅蹬地的动作。新版保持物理模型不变，调整了控制和搜索目标：

- 频率从 2.975 Hz 降至 1.936 Hz；前/后腿标称摆幅从 10.1°/47.6° 改为 23.2°/27.6°。
- 左前、右前、右后、左后的控制相位固定为 0°、180°、90°、270°。左右腿相差半个周期，四腿依次错开；实际落脚时刻仍取决于被动连杆和地面接触，并非严格理想的四拍落脚。
- 使用分段余弦轨迹，两个摆动方向的时间占比为 62.3%/37.7%，端点速度连续为零；目标角度的变化率限制为 8 rad/s。它限制控制目标，不是物理舵机转速的硬限制。
- 搜索同时惩罚足底接触点切向速度、机身高度变化、垂直速度、倾斜、偏航和触地；达到适中速度后，不再奖励更快的前进。
- 同时尝试了严格对角相位的小跑。该模型上四相位方案的高度波动和稳定性更好，因此作为默认值。

`reports/gait_comparison_2026-10-17.json` 保存步态调整时的 20 秒新旧比较；质量指标在各自启动渐变结束 0.5 秒后统计。双方使用相同模型和接触参数，早于本次仅涉及显示的修正：

| 指标 | 旧版 | 当前 |
| --- | --- | --- |
| 稳态机身高度标准差 | 1.668 mm | 0.661 mm（降低约 60%） |
| 机身垂直速度 RMS | 0.0823 m/s | 0.0147 m/s（降低约 82%） |
| 平均接触点切向速度 | 0.0586 m/s | 0.0469 m/s（降低约 20%） |
| 平均向前速度（含启动） | 0.1532 m/s | 0.0736 m/s |

这里的滑移指标是接触点的绝对切向速度，包含轮端转动的运动学贡献，不把轮心移动直接当作滑移。它不是单位前进距离的滑移损失，也不表示完全无滑移。身体倾角 RMS 约为旧版 8.34°、新版 8.47°，没有声称所有姿态指标都改善。

`reports/walk_validation.json` 记录修正启动前的 60 秒步行和停止检查（步行前另有 2 秒站立），下表为该存档结果。修正后的结果见下文 `startup_fix_2026-11-09.json`；重新运行会写入 `outputs/walk_validation.json`：

| 指标 | 默认地面结果 |
| --- | --- |
| 向前位移 | 4.5692 m |
| 平均前进速度（含启动渐变） | 0.0762 m/s |
| 最大横向偏离 | 0.0430 m |
| 最大航向偏差 | 12.84° |
| 机身原点高度 | 41.0–48.0 mm |
| 机身触地 / 数值警告 / 外加推动力 | 均为 0 |
| 闭环最大位置残差 | 0.920 mm |
| 最大执行器力矩 | 约 0.0540 N·m，小于 0.245 N·m 限制 |
| 最大驱动关节速度 | 8.76 rad/s |

四条腿都检测到反复离地、着地；接触状态经 20 ms 去抖，但仍不把接触变化次数等同于完整步数。脚轮与下腿网格均会接触地面，运动包含滚动与滑动，并不是理想的无滑移点足步态。8 秒步行后切换站立，3 秒后机体线速度约为 0.000311 m/s。

`reports/validation.json` 另记录站立和默认小幅摆动各 30 秒的检查。模型总质量 0.188253 kg，31 个位置变量、30 个速度自由度、4 个执行器、4 个闭环连接约束。

**适用范围：当前默认平地和初始状态。** 地面与脚轮的滑动摩擦参数为 0.8，下腿等未显式指定的碰撞几何体使用 MuJoCo 默认值 1.0；并非所有几何体都设为 0.8。`reports/walk_sensitivity_2026-10-17.json` 将所有几何体的滑动摩擦统一改为 0.6 或 1.0，两个 12 秒用例都未通过：均无机身触地，但出现偏航/姿态或横向偏离超限。额外方向增益搜索也没有找到同时满足三种场景的方案。不能把默认场景通过理解成跨地面稳健或已经适合实机。

网站教程附有当前步行视频和走远后的地面截图。使用下方命令可生成 `outputs/quaddle-coordinated.mp4` 和并排对比 `outputs/gait-before-after.mp4`；左右均为原速、相同仿真时刻。旧步态配置保存在 `reference/previous-gait.json`。原始实验目录中的候选视频和调参日志不打包进仓库。

```bash
# 重跑默认场景的步行、停止检查
.venv/bin/python validate_walk.py --seconds 60

# 附加摩擦参数敏感性检查；目前两个额外用例均失败，命令返回非零
.venv/bin/python validate_walk.py --seconds 60 --sensitivity

# 检查站立及默认小幅摆动
.venv/bin/python run_sim.py --validate --duration 30

# 重新生成 14 秒步行录像，包含启动阶段
.venv/bin/python render_demo.py --mode walk --seconds 14 --output quaddle-coordinated
.venv/bin/python render_demo.py --mode walk --seconds 14 --config reference/previous-gait.json --output quaddle-previous
.venv/bin/python make_comparison_video.py

# 重跑新旧步态的同模型质量比较
.venv/bin/python search_natural.py --compare

# 从固定随机种子搜索四相位步态；输出候选，不自动覆盖 gait.json
.venv/bin/python search_natural.py --pattern crawl --generations 25
```

## 坡面与地面适应检查

先复现平地，再只改变地面条件。以下坡度为探索档位，不代表机器人已具备爬坡能力：

```bash
.venv/bin/python check_terrain.py
.venv/bin/python validate_walk.py --seconds 60 --sensitivity
.venv/bin/python validate_walk.py --seconds 20 --slope-deg 3
.venv/bin/python validate_walk.py --seconds 20 --slope-deg 5
# Linux 无窗口录像；需要可用的 EGL 实现
MUJOCO_GL=egl .venv/bin/python render_demo.py --seconds 10 --slope-deg 3 --output uphill3
```

正坡度沿机器人初始前方（世界 -Y）上升。`terrain.py` 同时旋转地面与初始机身，保持初始离地距离；修改地面姿态后调用 `mj_setConst` 更新编译时的坐标系快捷标记，否则地面可能仍按水平姿态参与计算。前进距离、离地高度和姿态倾斜均在坡面坐标系测量，避免把沿坡上升算作额外离地高度。航向控制器仍读取世界坐标真值，尚未改成坡面反馈或实机状态估计。

`reports/ground_readiness_2026-11-09.json` 是 Linux / Python 3.12.13 / MuJoCo 3.16.0 的单次确定性复跑：平地60秒与停止检查通过；统一摩擦0.6/1.0均未通过；3°与5°上坡均出现机身触地和姿态超限。不同用例时长不同，不用于直接比较速度排名。默认场景使用混合摩擦参数，统一摩擦覆盖所有碰撞几何体，二者不是同一条件。

测试仅验证当前模型和固定初始状态，不是多seed鲁棒性结论，也未进行真机验证。下一步应先做单变量控制对照，再引入初始姿态、接触和舵机响应扰动。

### 启动站姿修复（2026-11-09）

默认配置已增加 `startup_bias_rad=[0.2,0.2,-0.4,-0.4]`（乘电机方向符号前），用 `startup_ramp_seconds=0.05` 的平滑曲线建立支撑，继续受原8 rad/s目标变化率限制。2秒支撑阶段后，在原1.5秒起步窗口内从支撑姿态过渡到周期步态；没有改质量、弹簧、摩擦、地面角度或外加力。

`reports/startup_fix_2026-11-09.json` 保存修复前后同条件复跑。新增前3.5秒的相对坡面俯仰与机身触地检查，分别要求小于15°和零触地。3°用例最大俯仰由40.35°降至3.93°，20秒行走检查通过；平地60秒和停止也通过。5°启动最大俯仰降至4.02°且无机身触地，但后续速度、足端支撑与连续前进未通过。统一摩擦用例也仍有失败。参数经过这些开发条件筛选，不能视为独立测试或实机稳健性证明。

```bash
.venv/bin/python check_startup.py
.venv/bin/python validate_walk.py --seconds 20 --slope-deg 3
.venv/bin/python validate_walk.py --seconds 20 --slope-deg 3 --config reference/pre-slope-startup-gait.json
```

先另存输出再跑下一组。旧基线配置不包含启动支撑参数，控制器保持其原有零位启动行为。`render_demo.py --config` 可用相同配置生成完整前后录像，画面标注坡度和相对坡面的俯仰。

## 多场景实验入口

`scenarios.py` 定义平地、3°/5°上坡、3°下坡、4mm×3级上下台阶、±3mm坑洼、已知地图绕障和8mm浅水近似。`scenario_matrix.py` 对行走/跑步候选分别进行实步动力学校验，生成可筛选的视频总览、每组JSON指标与CSV轨迹：

```bash
.venv/bin/python check_scenarios.py
.venv/bin/python scenario_matrix.py --scene stairs_up --mode walk --seconds 20
MUJOCO_GL=egl .venv/bin/python scenario_matrix.py --scene all --mode both --seconds 20 --video --workers 2
.venv/bin/python -m http.server 8012 --bind 127.0.0.1 --directory outputs/scenario_matrix
```

浏览器打开 `http://127.0.0.1:8012`。失败任务也会保存，主命令不把物理失败当成程序崩溃；逐项检查`summary.json`中的`passed`与`checks`。录像与无录像共用5ms更新时刻，录像不额外调用动力学求解器，避免改变warm-start状态。6秒平地和绕障的有/无录像结果已做1e-12绝对容差对照。

运行约束与解释：

- 台阶尺寸仅为探索档位；坑洼底面实际降低6mm，凹坑不是被原平面填平的视觉效果。
- 下台阶初始机身抬高12mm，使它真实站在起始平台。运行中不写机身位姿、不施加推进力。
- 绕障由已知地图A*与仿真真值定位闭环完成；导航单独使用5rad/s目标变化率与0.45转向限幅，保留原10rad/s实际关节速度门槛。它尚未包含感知/建图。
- `run_attempt`读取`reference/run-candidate.json`，仅是周期候选，尚未实现合格跑步。验收要求至少3个持续10ms的完整无地形接触区间、均速≥0.1m/s和全部其他检查。
- 浅水显式施加局部浸没球冠浮力与二次阻力，并单独记录力与耗散。地面接触优先级设为1，使实际接触滑动摩擦采用0.55，避免被机器人几何体的0.8/1.0覆盖；测试直接检查求解器接触系数。假设体积/阻力/湿地摩擦未标定，无CFD或防水验证；水面视觉几何不碰撞。
- 每条件只跑单一初态；源代码哈希和所有参数保存在报告中，不作为多seed稳健性结论。

## 官方模型及必要补充

来源：https://huggingface.co/petoi/quaddle

固定版本：`d5ed8ccfd238f98c2c47eb47658ed23232b1a74d`，下载日期：2026-10-17。原始文件保存在 `original/`，未改动；官方 MIT 许可保留在 `original/LICENSE`。`source.json` 和 `source_manifest.json` 记录版本及本地文件校验值。

原始 URDF 有 46 个 link、45 个 joint，但没有 collision、驱动器配置或可执行闭环约束。用 MuJoCo 默认方式直接加载时得到 24 个关节位置变量、**0 个几何体、0 个执行器**，因此“能解析”不代表已经可以进行接触运动仿真。

`build_model.py` 做了以下适配：

1. 将 URDF 层级、关节坐标、视觉网格和实体惯性转换为 MJCF；机体增加自由关节。
2. 将每条腿的两处 `closing_*` 标记连接。原始标记沿销轴相差约 1.89 mm，选取同一世界位置作为连接点，避免强迫两个不同的轴向位置重合。
3. 根据官方模型说明，将四个 `*_thigh` 关节设为驱动关节；其余关节保持被动。位置执行器 `kp=0.15`、`kv=0.006`，力矩限制约 ±0.245 N·m，后者换算自官方约 2.5 kg·cm 的标称值。
4. 添加地面、机身盒体、大腿胶囊体、下腿和脚部官方网格的凸包接触。早期下腿胶囊体偏大，现已改为官方网格凸包。接触掩码只启用机器人与地面之间的碰撞。
5. 对两份各约 106 万三角面的弹簧网格及部分其他高面数网格进行简化。原始网格保留；简化不重新计算 URDF 惯性。脚部凸包取自对应的适配网格。

## 物理近似与适用范围

官方目前提供了弹簧尺寸，但没有完整的刚度、预紧和舵机动态标定。本工程中：

- 官方弹簧零件仍作为刚体连杆处理，**没有模拟真实拉簧的轴向伸缩**。
- 采用被动关节扭转弹簧作支撑近似：`*_motor_arm` 刚度 0.012 N·m/rad，阻尼 0.0004 N·m·s/rad；`*_leg` 刚度 0.08，阻尼 0.001。均为调试参数，不是实测值。
- 舵机 PD 增益、转子惯量、摩擦系数、阻尼、碰撞外形均有工程假设。增加了目标角度变化率限制，仍未实现完整的舵机物理速度饱和、死区和延迟；旧基线 60 秒步行测得峰值关节速度约 8.76 rad/s，不能视作实物舵机可达到的指标；未验证真实驱动映射。
- 关闭机器人自碰撞，保留地面接触；验证覆盖给定初始状态、默认平地和所记录的模式。
- 官方未提供的外观部件不会自动添加，模型外观以该 URDF 为准。
- 没有加载 Petoi 网页仿真器、OpenCat 固件或官方步态。步行使用约 1.936 Hz 周期目标、固定相位关系、前后腿不同的偏置/幅度、100 Hz 控制更新及方向反馈。方向反馈读取仿真真值中的航向、横向位置和角速度，调整左右步幅；实机还需要状态估计。

可以从这里继续研究控制、添加 Gymnasium 环境或训练策略。做可信的 sim-to-real 前，需要核对传动结构、弹簧模型、接触参数和舵机实测响应。

## 文件

| 文件 | 用途 |
| --- | --- |
| `original/robot.urdf`、`original/assets/` | 官方原始模型 |
| `build_model.py` | 可重复执行的 URDF → MJCF 适配 |
| `models/scene.xml` | 可独立交给 MuJoCo 加载的场景；附属网格在 `models/assets/` |
| `models/build_report.json` | 模型构建统计 |
| `models/mesh_report.json` | 网格简化面数记录 |
| `run_sim.py` | 控制、交互查看器和无界面验证 |
| `gait.json`、`gait_controller.py` | 步态参数与关节控制器 |
| `validate_walk.py` | 步行、离地/着地、停止及敏感性验证 |
| `search_gait.py`、`search_feedback.py` | 旧版以位移为主的搜索，保留探索过程 |
| `search_natural.py` | 相位约束搜索、接触点滑移和机身起伏度量、新旧比较 |
| `tune_heading.py` | 默认和统一摩擦参数下的方向反馈试验 |
| `launch.py`、`run.command` | 本机启动入口 |
| `render_demo.py`、`make_comparison_video.py` | 新步态视频、预览与同速并排比较 |
| `requirements.txt`、`requirements.lock` | 直接依赖及实际安装版本 |
| `reference/` | 新旧比较需要的旧步态参数 |
| `reports/` | 验证报告，包括长距离地面显示检查 |

重建环境与模型（原始文件已包含在项目内）：

```bash
uv venv --python 3.12 .venv
uv pip sync --python .venv/bin/python requirements.lock
.venv/bin/python build_model.py
./run.command
```

参考：[Petoi 官方模型](https://huggingface.co/petoi/quaddle)、[MuJoCo 模型说明](https://mujoco.readthedocs.io/en/stable/modeling.html)、[macOS 查看器说明](https://mujoco.readthedocs.io/en/stable/python.html#passive-viewer)。

## 失败定位与分场景训练

`diagnose_terrain.py` 记录原步态首次倾斜、触地、偏航和关节超速的时刻。先看失效顺序，再决定是改步态参数还是训练状态反馈。`search_terrain_gait.py` 用有边界的 CEM 搜索频率、幅度、偏置和方向反馈；它是步态参数优化，不能称作 PPO。

楼梯、坑洼使用新增的 `terrain_rl.py` 和 `train_terrain_ppo.py`。旧 `rl_env.py` 是独立的平地示例，启动方式与本轮多场景评测不同，不能直接混用其 checkpoint。

新环境保持 1 ms 物理步进、100 Hz 动作、原支撑启动姿态与关节限制。PPO 输出四个 ±0.25 rad 残差，叠加到周期目标后，再限制目标变化速率为 8 rad/s；这不是对实际关节速度的硬钳位，实际峰值仍按原 10 rad/s 阈值独立验收。训练目标包含前进、速度跟踪、偏航、实际关节超速和动作平滑代价，机身触地或倾倒会结束训练回合。独立评测仍运行完整 22 秒，保留失败后续状态。

36 维观测包含姿态、速度、关节状态、上一动作、周期目标、相位与前方五个地形高度。高度和速度来自仿真真值；目前没有实现真机地形感知。训练只随机化初始横向位置 ±5 mm，不能据此宣称对未知地形、摩擦或模型误差鲁棒。

本轮 CPU 环境使用 Python 3.12、MuJoCo 3.16.0、PyTorch 2.15.0+cpu、SB3 2.9.2、Gymnasium 1.4.0。在已有基础依赖环境中安装：

```bash
uv pip install --python .venv/bin/python torch==2.15.0+cpu --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python stable-baselines3==2.9.2 gymnasium==1.4.0
OPENBLAS_NUM_THREADS=1 .venv/bin/python check_terrain_rl.py
OPENBLAS_NUM_THREADS=1 .venv/bin/python train_terrain_ppo.py \
  --scene stairs_up --steps 32768 --envs 4 --seed 42 --output outputs/terrain_ppo/stairs_up/r1
OPENBLAS_NUM_THREADS=1 .venv/bin/python train_terrain_ppo.py \
  --scene stairs_up --steps 65536 --envs 4 --seed 42 \
  --resume outputs/terrain_ppo/stairs_up/r1 --output outputs/terrain_ppo/stairs_up/r2
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 .venv/bin/python evaluate_terrain_stages.py \
  --scene stairs_up --policy outputs/terrain_ppo/stairs_up/r2 \
  --output outputs/terrain_ppo/stairs_up/r2/evaluation --video
```

每轮同时保存 `policy.zip`、`vecnormalize.pkl`、推理用的 `obs_stats.npz` 和记录训练步数、耗时、依赖版本、源码哈希的 `metadata.json`。这些文件必须成套使用。续训保留优化器和归一化统计；推理使用冻结统计。

参数搜索与回放示例：

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python search_terrain_gait.py \
  --scene downhill3 --output outputs/terrain_ppo/search/downhill3
MUJOCO_GL=egl .venv/bin/python scenario_matrix.py --scene downhill3 --mode walk \
  --config outputs/terrain_ppo/search/downhill3/selected.json --video
```

本轮未修改默认 `gait.json`，每个通过的场景使用显式配置或 checkpoint；不存在一个已经验证能解决所有地形的统一策略。跑步还要通过原来的速度、至少三段完整腾空等检查，奖励提高或步频提高都不等于任务完成。

2026-11-09实跑结果见 [`reports/terrain_iteration_2026-11-09.json`](reports/terrain_iteration_2026-11-09.json)：76组参数试验中，`reference/terrain-downhill3.json`、`terrain-stairs_down.json`、`terrain-uphill5.json` 在原场景通过，±5mm初始横向偏移也分别通过。上楼梯与坑洼各完成两轮、累计98,304步PPO，尚未通过；坑洼第二轮虽走得更远，但触地/超速退步，保留第一轮作为稳定候选。`reference/run-bound-candidate.json` 有完整腾空，但速度与关节限制仍不合格。未将这些场景配置或策略设为默认，也未发布“已会跑”的结论。模型与视频保存在本次实验本地归档中，未作为教程仓库二进制文件提交。
