# Quaddle VLA 数据管线（W3）：脚本化专家 → LeRobotDataset

阶段三产物：为 SmolVLA 微调生成导航数据集。

## 任务设计

桌面散布 3 个彩色积木（red/green/blue，各 5cm 立方），指令指定目标颜色，
机器人从随机初始朝向出发走到目标积木旁（<12cm）停下：

> **"walk to the {red|green|blue} block"**

- 干扰物：非目标颜色积木同时在场景内（强迫 VLA 学颜色分辨）
- 专家：查表 CPG 控制器（W2 交付物）+ 方位闭环（死区 ±25°，到达半径 12cm）
- 失败 episode（翻倒/超时 45s）整段丢弃不入库；`--episodes` 语义 = **成功数**（失败自动补采）

## 数据规格（fps=10）

| 字段 | 形状 | 内容 |
|---|---|---|
| `observation.image` | 320×320×3 video | overhead 固定俯视相机（机器人+3 积木同框） |
| `observation.state` | 8 | [4 舵机角, vx, wz, bearing, dist] |
| `action` | 2 | [vx_dir, wz_dir] ∈ {-1,0,+1}² 档位指令 |
| `task` | str | 语言指令 |

物理 1kHz / 控制 50Hz / 每 5 控制步记 1 帧。单条 episode 30~50 帧。

## 用法（必须用 `codes/practices/vla/act` 的 lerobot 0.6.1 环境）

```bash
cd codes/practices/vla/act
MUJOCO_GL=egl uv run python ../quaddle-data/collect.py \
    --episodes 50 --repo-id local/quaddle_nav_50 --seed 2000
```

输出：`quaddle-data/outputs/dataset/lerobot/`（LeRobotDataset v2 + mp4 视频）+
`scene.xml`（注入积木与相机的场景，mesh 用绝对路径引用 `7.quaddle-mjcf/assets`）。

## SmolVLA 微调（W4 冒烟命令，同环境运行）

```bash
export HF_ENDPOINT=https://hf-mirror.com   # HF 直连不通时的镜像
cd codes/practices/vla/act
uv run lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=local/quaddle_nav_50 \
  --dataset.root=../../vla/quaddle-data/outputs/dataset/lerobot \
  --batch_size=4 \
  --steps=2000 \
  --save_freq=1000 \
  --policy.device=cuda \
  --policy.use_amp=true \
  --policy.scheduler_decay_steps=2000 \
  --output_dir=outputs/smolvla_quaddle_smoke
```

要点：短训必须同步缩 `scheduler_decay_steps`（默认 30000），否则 lr 不衰减不收敛；
`batch_size` 从 4 起步按显存二分（4070 Ti 预算 10GB，官方 batch8≈10-16GB）。

## 版本记录

- 2026-11-08：管线打通。首个成功 episode 4.4s/44 帧；专家决策与 state 一致性
  人工校验通过（bearing −1.17 ↔ action=turn_right）。
- 踩坑：`CpgController` 相位钟曾全局共享（差频转向从未生效，W2 的 turn 档
  实为直线冲）→ 已修为每腿独立相位钟并重标定（turn_left 1.0/3.0Hz 差频
  wz+0.31，turn_right 2.5/1.5Hz wz−0.37，转弯半径 ~220-230mm 对称）。
- 滚轮模型无倒挡（单向棘轮），专家用弧线转向绕行即可到达任意方位目标。

## W5 评估记录（2026-11-08/09，三轮迭代）

同 seed（7000）三场景闭环 rollout，检查点取各自 last：

| 版本 | 数据 | 训练 | 成功率 | 失败模式 |
|---|---|---|---|---|
| v1 | 250 条固定死区 | 20k | 1/3 | ep0/1 恒 turn_right 超时 |
| v2 | 250 条（死区随机化**未生效**，事后发现） | +dont_wait 续训 | 1/3 | 与 v1 完全一致 |
| v3 | 400 条真·随机死区 ±12~40° + 转向连续强度 | 30k @base | 1/3 | ep0/1 转向激进 → **翻倒** |

**ep2（blue）三版全部成功**（~2.8s 到达）——VLA 闭环导航能力成立。

### 根因分析（诊断脚本 diag_ep0 实测）

ep0 场景初始 bearing ≈ −161°（目标几乎正后方）。v3 诊断输出：模型方向正确
（wz<0 右转）但强度不足（−0.47），且 **turn 档执行自带直行分量 → 弧线稳态中
bearing 恒定**（几何性质）→ 无限右弧。**动作空间缺"纯原地转"原语**——差频
档必带弧线，是能力缺口而非数据缺口。

### 改进路线（按优先级，供后续执行）

1. 动作空间增加原地转原语（左右腿反相摆动，需先在 MuJoCo 验证可行性）
2. 长转向序列的行为一致性：观测加步态相位/已转角度记忆，或决策频率降至 1Hz
3. 数据侧：初始 bearing > 120° 的场景过采样（当前均匀分布仅 1/3 覆盖）
4. flow matching 积分步数 10 → 20（输出幅值尖锐化）

### 复现

```bash
cd codes/practices/vla/act
MUJOCO_GL=egl HF_HUB_OFFLINE=1 uv run --no-sync python ../quaddle-data/eval_rollout_gif.py \
  --episodes 3 --seed 7000 \
  --ckpt ../quaddle-data/outputs/smolvla_quaddle_v3/checkpoints/last/pretrained_model \
  --out ../quaddle-data/outputs/rollout_v3.gif
```
