# 7. Quaddle MJCF：四舵机 MuJoCo 步行仿真

Petoi Quaddle（4 舵机全向桌面四足）的 MuJoCo 环境。从官方 URDF
[HuggingFace `petoi/quaddle`](https://huggingface.co/petoi/quaddle)（MIT 协议）
生成，闭链结构与参数复刻官方网页仿真器
[`quaddle-sim`](https://github.com/one-for-all/quaddle-sim) 的 builder.rs。

属于 [VLA 训练方案](../../../../../../Petoi-Quaddle-VLA训练方案-10GB.md) 的阶段一产物；
不改任何既有课程代码。

## 机器人机构

每条腿是**平行曲柄四连杆 + 闭链**：

- `thigh`：舵机驱动关节（主动）；`motor_arm`：平行曲柄（被动，带舵机转子 armature）
- `spring`：扭簧关节（闭链柔性）；`leg`：足端连杆；`wheel`/`wheel_tip`：滚轮足端（视觉为主）
- thigh↔leg 用两点 `connect` 等等约束闭合，等效转动副

**关键约定：所有关节 `ref` 设为装配角 → `q=0` 即"官方上电平衡站姿"**
（前腿 6.95°、后腿 ±40°，四足等高，roll/pitch ≈ 0）。
控制目标就是舵机相对站姿的偏转角，正负号沿 URDF 子坐标系 z 轴。

## 目录

```
quaddle.xml              生成产物（勿手改，跑 scripts/build_quaddle_mjcf.py 重新生成）
assets/quaddle_urdf/     官方 URDF + STL mesh（MIT）
scripts/build_quaddle_mjcf.py   URDF → MJCF 构建器（闭链解算 + 平衡站姿搜索）
run_stand.py             验收 1：30s 站立（漂移/姿态阈值判定 PASS/FAIL）
run_gait_sine.py         验收 2：正弦步态基线扫描（净位移/翻倒统计）
run_view_keyboard.py     交互：WASD 开环步态控制（需图形环境）
render_smoke.py          验收 3：EGL 无头渲染 → render_smoke.png（入库时转 render_smoke.webp）
tests/                   模型结构/站立/舵机/步态稳定性冒烟测试
```

## 用法（在 `codes/practices/quadruped/cs123` 下）

```bash
# 重建模型（改了 builder 或 mesh 后）
uv run python 7.quaddle-mjcf/scripts/build_quaddle_mjcf.py

# 验收
uv run python 7.quaddle-mjcf/run_stand.py
uv run python 7.quaddle-mjcf/run_gait_sine.py
MUJOCO_GL=egl uv run python 7.quaddle-mjcf/render_smoke.py

# 测试
uv run pytest -q 7.quaddle-mjcf/tests
```

## 阶段一验收结果（2026-11-07）

| 项 | 结果 |
|---|---|
| 30s 站立 | **PASS**：漂移 0.0mm，高度波动 0.0mm，roll 峰 0.71°，pitch 峰 0.03° |
| 四足等高 | foot 球底 -0.7~-0.05mm 全接触，body 高 49.5mm |
| 步态基线 | trot 相位（±8°，1.5Hz）站稳爬行 **3~5mm/s**、伴随偏航漂移；大幅（≥15°）或 wave 相位翻倒 |
| EGL 渲染 | 480×360 正常出图，mesh/光照/地面完整 |
| 测试 | pytest 5/5 通过 |

**核心结论**：球面足端 + 各向同性摩擦下，对称正弦激励只能产生 ~5mm/s
漂移爬行——真实 Quaddle 的直线行走依赖**非对称步态表（固件 kwkF 轨迹）
与滚轮足端的方向性**。这就是阶段二的两条路线：
A) 从官方仿真器录制 kwkF/ktrL 关节轨迹回放（贴真机）；
B) RL 速度跟踪策略（复用 `6.rl_pupper_v2_5` 管线改 4 维动作），由奖励函数
   自己发现非对称轨迹。

## 踩坑记录（供后续阶段参考）

1. **URDF 零位 ≠ 闭合零位**：闭链在 y-z 平面天然闭合，x 向有 ~1.9mm CAD
   间隙（绕 body-x 的关节转不动 x 坐标，数值求解器无解）。装配解是解析的
   `ψ=θ, σ=λ=0`，间隙由 connect 约束焊接吸收。
2. **平行曲柄**：thigh 与 motor_arm 转角相同即整腿刚性随动（官方 builder.rs
   注释证实）；σ 是唯一的弹性自由度，扭簧刚度 0.6 N·m/rad（0.15 塌陷、
   2.0 在 2ms 步长下 QACC 爆炸）。
3. **闭链约束用 site 对**：`connect` 不能混用 body/anchor 与 site1/site2
   语义；site 版无需关心 anchor 坐标系。
4. **数值稳定性**：timestep 1ms + Newton(iterations=100, tol=1e-10) +
   舵机目标一阶低通（τ=30ms，模拟真实舵机带宽）后，扫描不再 NaN。
5. **mesh 下载**：hf-mirror 偶发 SSL 错误页存成 .stl——下载后按
   `84 + 50*n == size` 校验二进制 STL。
