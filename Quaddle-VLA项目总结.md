# Quaddle VLA 项目总结（收尾版）

> Petoi Quaddle（4 舵机桌面四足）在 10GB 显存约束下的 VLA 训练：从零到可复现基线。
> 收尾日期：2026-11-09。本文档为独立总结，接续《Petoi-Quaddle-VLA训练方案-10GB.md》。
> 全部产物未纳入 git 跟踪。

---

## 一、项目目标与最终状态

**目标**：在本地 RTX 4070 Ti（12GB，预算 10GB）上，为 Petoi Quaddle 建立
"SmolVLA 高层决策 + 低层运动控制"的完整 VLA 训练-评估闭环。

| 里程碑 | 状态 | 结果 |
|---|---|---|
| Quaddle MuJoCo 仿真环境 | ✅ | 官方 URDF→MJCF，闭链+滚轮足端，站立/步态/渲染验收通过 |
| 低层运控（50Hz 舵机控制） | ✅ | 查表 CPG，forward +113mm/s，左右弧线转向对称（R≈220mm） |
| 数据管线（LeRobotDataset） | ✅ | 400 条三相机成功 episodes，专家方位闭环自动采集 |
| SmolVLA 微调管道 | ✅ | GPU 14.7 steps/s（vs CPU 395 倍），显存 1.69GB@batch1 |
| **VLA 闭环导航** | ⚠️ 部分 | **前侧方目标可靠到达（ep2 类三版全成功）；目标正后方（bearing>120°）失败** |

**一句话结论**：10GB 显存训 VLA 完全可行（SmolVLA 450M 本地微调打通全流程）；
当前模型是"前侧方导航"基线（1/3 → 目标在前侧方时 100%），正后方场景需补
"原地转"运动原语后重训（路线已明确，见第五节）。

---

## 二、系统架构（分层）

```
SmolVLA 高层（10Hz 重规划，450M，模仿学习 + flow matching）
  输入：3 路相机 RGB（320²）+ 状态 8 维 + 语言指令 "walk to the {color} block"
  输出：动作 (vx_dir, wz_dir) —— 量化为档位
    ↓
查表 CPG 低层（50Hz，免学习，W2 标定）
  forward / turn_left_weak|strong / turn_right_weak|strong / stop
  差频转向强度插值（1.0↔3.0Hz），每腿独立相位钟
    ↓
MuJoCo 物理（1kHz，闭链四连杆 + 滚轮足端，q=0 = 平衡站姿）
```

## 三、关键交付物路径

### 代码（全部新增，零改动既有文件）

| 路径 | 内容 |
|---|---|
| `codes/practices/quadruped/cs123/7.quaddle-mjcf/` | MJCF 构建（官方 URDF→MuJoCo）+ 站立/步态/渲染脚本 + 测试 |
| `codes/practices/quadruped/cs123/8.rl_quaddle/` | CPG 控制器（W2）+ RL 环境存档 + 标定数据 |
| `codes/practices/vla/quaddle-data/` | **数据采集 + 训练 + rollout 评估（W3-W5）** |
| `codes/practices/vla/act/` | lerobot 0.6.1 环境（pyproject 已锁定 GPU 可用依赖栈） |

### 模型与数据

| 路径 | 内容 |
|---|---|
| `vla/quaddle-data/outputs/smolvla_quaddle_v{1,2,3}/` | 三轮正式训练检查点（各 4+1 个） |
| `vla/quaddle-data/outputs/dataset/lerobot/` | 400 条数据集（camera1/2/3 视频 + parquet + scene.xml） |
| `vla/quaddle-data/outputs/rollout*.gif` | v1/v3 闭环行为 GIF |
| `~/.cache/huggingface/hub/` | smolvla_base（873MB）+ SmolVLM2（6.7GB），离线训练就绪 |

### 文档

| 路径 | 内容 |
|---|---|
| `Petoi-Quaddle-VLA训练方案-10GB.md`（仓库根） | 总方案 + W1/W2/W5 执行记录 |
| `7.quaddle-mjcf/README.md` | MJCF 构建、验收、踩坑 |
| `8.rl_quaddle/README.md` | RL 三轮迭代、CPG 控制器、评估报告 |
| `vla/quaddle-data/README.md` | 数据管线、SmolVLA 训练命令、W5 三轮评估记录 + 根因分析 |

## 四、关键技术发现（每条都花了真实迭代成本）

1. **闭链装配解是解析的**：Quaddle 四连杆在 y-z 平面天然闭合，x 向 1.9mm 为
   CAD 间隙；平行曲柄（ψ=θ）即整腿刚性——不要对闭链盲跑数值求解器。
2. **官方 power-on 是蹲姿不是站姿**：前腿悬空前倾 5.7°；FK 扫描出前腿 +6.95°
   补偿实现四足等高，配合全关节 `ref` 技巧让 **ctrl=0 恰好是平衡站姿**。
3. **足端方向性决定行走能力**：球足端（各向同性滑动摩擦）稳定行走上限
   ~10-20mm/s，RL 三轮收敛到 3→6→14mm/s 就是在逼近这个物理上限；换成
   滚轮圆柱（真机 omni 轮机制）后 forward 实测 +113mm/s。**仿真保真度 > 控制器智能**。
4. **滚轮是单向棘轮**：正/倒相位、skew 反号全部产生 +x 位移——仿真模型无倒挡
   （真机 OpenCat 有 kbwF，无此限制）；后方目标用 180° 弧线转向替代。
5. **CPG 相位钟必须每腿独立**：全局共享时差频转向从未生效（W2 的 turn 档
   实为直线冲刺），修复后左右转向半径对称（~220mm）。
6. **uv run 会按 uv.lock 静默恢复环境**：所有依赖变更必须进 pyproject 锁定
   （torch 2.10+cu128、transformers 5.4-5.6、cusparselt 补丁），否则 pip 装的
   会被下一次 `uv run` 悄悄还原。
7. **flow matching 输出幅值保守**：one-hot 档位训练下模型输出 ±0.4 而非 ±1.0，
   绝对阈值量化会全判成 stop——用 argmax 相对判别 + 连续强度编码。
8. **弧线几何死锁**（当前能力边界根因）：turn 档自带直行分量 → 弧线稳态中
   bearing 恒定 → 目标在正后方时无限绕弧。缺"纯原地转"原语，是能力缺口
   而非数据缺口（三轮数据迭代证明）。

## 五、当前能力边界与后续路线

**可靠能力**：目标位于机器人前侧方 ±60° 扇区时，VLA 闭环导航稳定到达
（ep2 类场景三版模型全部成功，2.8s/0.5m）。

**失败模式**：初始 bearing > 120°（目标近正后方）时陷入转向-弧线循环或翻倒。

**后续路线（按优先级）**：

1. **动作空间补原地转原语**：左右腿反相摆动（OpenCat ktrL 机制），需先在
   MuJoCo 验证滚轮模型可行性；验证通过后控制器加档、重采重训（~半天）
2. **决策频率与记忆**：观测增加已转角度/步态相位，或将 VLA 决策降至 1Hz +
   低层插值，解决长转向序列的一致性
3. **数据侧**：初始 bearing > 120° 场景过采样（均匀 yaw 下仅 1/3 覆盖）
4. **推理端**：flow matching 积分步数 10 → 20，输出幅值尖锐化
5. **（真机后）**：OpenCat 固件开源后用 SkillLibrary 真步态表替换查表档位

## 六、复现入口

```bash
# 环境：codes/practices/vla/act（python 3.12，torch 2.10+cu128，lerobot 0.6.1）
cd /media/scc/sccDisk/dive-into-embodied-ai/codes/practices/vla/act

# 采集 400 条
MUJOCO_GL=egl uv run --no-sync python ../quaddle-data/collect.py \
  --episodes 400 --repo-id local/quaddle_nav_400b --seed 5000

# 训练 30k 步（离线，权重已缓存）
HF_HUB_OFFLINE=1 MUJOCO_GL=egl uv run --no-sync lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=local/quaddle_nav_400b --dataset.revision=v3.0 \
  --dataset.root=../quaddle-data/outputs/dataset/lerobot \
  --batch_size=4 --steps=30000 --save_freq=7500 \
  --policy.device=cuda --policy.use_amp=false --policy.push_to_hub=false \
  --policy.scheduler_decay_steps=30000 \
  --output_dir=../quaddle-data/outputs/smolvla_quaddle_v3

# 评估 + GIF
MUJOCO_GL=egl HF_HUB_OFFLINE=1 uv run --no-sync python \
  ../quaddle-data/eval_rollout_gif.py --episodes 3 --seed 7000 \
  --ckpt ../quaddle-data/outputs/smolvla_quaddle_v3/checkpoints/last/pretrained_model \
  --out ../quaddle-data/outputs/rollout_v3.gif

# RL 低层/模型构建/测试（cs123 环境）
cd ../quadruped/cs123 && uv run pytest -q 7.quaddle-mjcf/tests 8.rl_quaddle/tests
```

## 七、资源占用实测

| 阶段 | 显存 | 时长 |
|---|---|---|
| SmolVLA 训练（batch1 fp32） | 1.69 GB | 300 步 30s（14.7 steps/s） |
| SmolVLA 训练（batch4 fp32，v1 正式） | ~5 GB | 20k 步 51 min |
| 数据采集（400 条，EGL 渲染） | CPU 为主 | ~60 min |
| 推理 rollout（每 0.5s 一次） | ~1.5 GB | 30s 场景 ≈ 1 min |

唯一注意：显存与用户其他进程（如 algo-pi-agent）共享时，batch 按剩余量调整。
