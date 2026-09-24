import {translate} from '@docusaurus/Translate';
import React from 'react';
import Link from '@docusaurus/Link';
import type {Skill} from './types';

type RobotSkillContent = {
  skill: Skill;
  id: string;
  name: string;
  english: string;
  question: string;
  paragraphs: React.ReactNode[];
  input: string;
  output: string;
  frequency: React.ReactNode;
  methods: string;
  progress: string;
};

export const robotSkills: RobotSkillContent[] = [
  {
    skill: 'grasping', id: 'ability-grasping', name: translate({message: "抓取"}), english: 'Grasping', question: translate({message: "怎样把一个物体拿稳？"}),
    paragraphs: [
      <>{translate({message: "用夹爪、灵巧手或吸盘等末端执行器，建立并维持对物体的稳定约束。典型的抓取检测（grasp detection）以点云或 RGB-D 图像为输入，为平行夹爪预测抓取位姿：三维位置、三维朝向（6-DoF）与夹爪开度，再结合可达性和碰撞检查执行。"})}</>,
      <>{translate({message: "几何与力学方法分析形封闭（form closure）、力封闭（force closure）等条件；数据驱动方法学习抓取质量或直接生成抓取位姿。代表项目包括"})}<Link href="https://berkeleyautomation.github.io/dex-net/">Dex-Net</Link>{translate({message: "、作为数据集与评测基准的"})}<Link href="https://graspnet.net/">GraspNet-1Billion</Link>{translate({message: "，以及"})}<Link href="https://graspnet.net/anygrasp.html">AnyGrasp</Link>{translate({message: "。解析模型也可以用于生成学习所需的数据。"})}</>,
      <>{translate({message: "抓取可以先规划再执行，也可以通过视觉、力觉或触觉持续修正；例如 AnyGrasp 支持抓取位姿跟踪。受控工位中的规则刚体抓取已有较成熟方案，灵巧手、透明或柔性物体、严重遮挡与复杂堆叠仍对可靠性提出挑战。"})}</>,
    ],
    input: translate({message: "点云 / RGB-D；也可加入力觉、触觉"}),
    output: translate({message: "抓取位姿、夹爪开度或接触参数"}),
    frequency: <>{translate({message: "按任务触发，或连续更新抓取目标"})}</>,
    methods: translate({message: "几何与力学分析、监督学习"}),
    progress: translate({message: "受控工位较成熟；复杂物体与接触仍有挑战"}),
  },
  {
    skill: 'manipulation', id: 'ability-manipulation', name: translate({message: "操作"}), english: 'Manipulation', question: translate({message: "怎样通过接触改变物体状态？"}),
    paragraphs: [
      <>{translate({message: "操作的范围包括抓取、搬运、推、拨、翻转、手内操作（in-hand manipulation），也包括叠衣服、整理绳索、开门和抽屉、使用工具、双臂协作与长时序任务。抓取是其中一个子问题；推物体或踢足球属于不需要先抓住物体的非抓取式操作。"})}</>,
      <>{translate({message: "复杂操作常需要根据图像、本体感知，以及可选的语言目标，持续决定下一步动作。模仿学习（IL）与视觉语言动作模型（VLA）是重要研究路线："})}<Link href="https://tonyzhaozh.github.io/aloha/">ACT</Link> {translate({message: "预测动作片段，"})}<Link href="https://diffusion-policy.cs.columbia.edu/">Diffusion Policy</Link> {translate({message: "结合动作生成与滚动执行，"})}<Link href="https://www.pi.website/blog/pi0">π0</Link>{translate({message: "、"})}<Link href="https://openvla.github.io/">OpenVLA</Link> {translate({message: "则探索多任务的视觉语言动作策略。"})}</>,
      <>{translate({message: "难点包括丰富且易变化的接触（contact-rich）、无法直接观测的状态、同一目标对应多种合理动作，以及误差在长序列中的累积。闭环反馈对复杂操作很重要，抓取同样可以使用闭环。判断能力时，应区分固定任务复现与新物体、新环境下的泛化。"})}</>,
    ],
    input: translate({message: "图像、本体感知；可加入语言、力觉与触觉"}),
    output: translate({message: "末端或关节动作、动作片段"}),
    frequency: <><Link href="https://tonyzhaozh.github.io/aloha/">{translate({message: "ACT 示例"})}</Link>{translate({message: "：50 Hz 动作执行"})}</>,
    methods: translate({message: "规划与控制、IL、RL、VLA"}),
    progress: translate({message: "任务间差异大；通用、长时序操作仍有挑战"}),
  },
  {
    skill: 'locomotion', id: 'ability-locomotion', name: translate({message: "运动 / 行走"}), english: 'Locomotion', question: translate({message: "怎样让身体稳定地移动？"}),
    paragraphs: [
      <>{translate({message: "关注机器人自身的移动能力。足式机器人需要处理步态、平衡、台阶、斜坡和跌倒恢复；轮式机器人也属于这一范畴，需要应对转向约束、打滑和不平地形。全身协调控制还可以把移动与操作连接起来。"})}</>,
      <>{translate({message: "观测常包括关节位置与速度、惯性测量单元（IMU）提供的姿态和角速度，以及可选的深度图或高程图。策略可输出关节目标，再由底层控制器跟踪，也可以直接输出力矩。步态策略的更新频率与关节伺服频率需要分开看。"})}</>,
      <>{translate({message: "模型预测控制（MPC）和仿真到真实的强化学习（sim-to-real RL）都是常见路线。"})}<Link href="https://proceedings.mlr.press/v164/rudin22a.html">Learning to Walk in Minutes</Link> {translate({message: "用 Isaac Gym 中的大规模并行仿真训练 ANYmal；域随机化帮助应对现实差异，教师—学生训练可把带有额外信息的技能迁移到机载观测。"})}<Link href="https://robot-parkour.github.io/">Robot Parkour</Link> {translate({message: "等工作进一步研究依靠视觉选择与执行越障技能。"})}</>,
      <>{translate({message: "四足机器人在多种已验证场景中已有很强的运动能力；极端地形、低摩擦接触、长期可靠性，以及人形机器人和负载变化下的全身协调，仍需要针对具体条件验证。"})}</>,
    ],
    input: translate({message: "关节状态、IMU；可加入地形感知"}),
    output: translate({message: "关节目标、力矩或足端目标"}),
    frequency: <>{translate({message: "50 Hz 策略 / 200 Hz 仿真控制"})}<small>{translate({message: "按"})}<Link href="https://github.com/leggedrobotics/legged_gym/blob/master/legged_gym/envs/base/legged_robot_config.py">{translate({message: "legged_gym 基础配置"})}</Link>{translate({message: "折算"})}</small></>,
    methods: translate({message: "MPC、全身控制、sim-to-real RL"}),
    progress: translate({message: "多种四足场景已验证；复杂地形与全身协同仍有挑战"}),
  },
  {
    skill: 'navigation', id: 'ability-navigation', name: translate({message: "导航"}), english: 'Navigation', question: translate({message: "去哪里、走哪条路、何时停下？"}),
    paragraphs: [
      <>{translate({message: "让机器人从当前位置到达目标，涉及定位、建图、路径规划与避障。经典系统通常组合定位 / SLAM、全局规划（如 A*）与局部规划或控制（如 DWA、MPC）；已知地图与受控环境中已有成熟的工程方案。"})}</>,
      <>{translate({message: "具身导航任务还可以按目标形式区分：PointNav 前往指定坐标，ObjectNav 寻找某类物体，ImageNav 寻找图像对应的地点或物体，VLN 按自然语言指令行进。输入可包含 RGB-D、激光雷达、里程计和目标描述，输出可以是速度指令，也可以是前进、转向、停止等离散动作。"})}</>,
      <>{translate({message: "导航常把移动能力作为执行基础：先决定目标或路径，再由运动控制把指令落到身体上。部分仿真基准简化了底层移动；真实部署还要处理定位误差、打滑、动态障碍和执行失败。比较结果时要同时看任务定义、传感器、成功条件与路径效率，不能把某个 PointNav 基准的高成功率推广到所有导航任务。"})}</>,
    ],
    input: translate({message: "视觉 / 激光雷达、位姿估计、目标描述"}),
    output: translate({message: "路径、航点、速度或离散动作"}),
    frequency: <>{translate({message: "高层按需更新；"})}<Link href="https://docs.nav2.org/rolling/configuration_and_development/configuration_guide/core_servers/controller_server/">{translate({message: "Nav2 局部控制默认 20 Hz"})}</Link></>,
    methods: translate({message: "定位 / SLAM、规划与控制、学习方法"}),
    progress: translate({message: "已知地图较成熟；开放环境与语义目标仍有挑战"}),
  },
];

export const robotSkillAnchors = robotSkills.map(({id}) => id);
