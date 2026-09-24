import {translate} from '@docusaurus/Translate';
import {BrainCircuit, Database, Eye, FlaskConical, Move3D, Settings2} from 'lucide-react';
import {robotSkills} from './skills/content';

type Chapter = {
  id: string;
  label: string;
  quickLabel: string;
  children?: {id: string; label: string}[];
};

export const chapters: Chapter[] = [
  {id: 'what-is-embodied-ai', label: translate({message: "什么是具身智能"}), quickLabel: translate({message: "认识具身智能"})},
  {
    id: 'robot-interaction', label: translate({message: "机器人的技能"}), quickLabel: translate({message: "机器人的技能"}),
    children: robotSkills.map(({id, name}) => ({id, label: name})),
  },
  {id: 'embodied-platforms', label: translate({message: '具身智能的载体'}), quickLabel: translate({message: '认识具身载体'})},
  {id: 'knowledge-map', label: translate({message: "知识全景"}), quickLabel: translate({message: "看知识全景"})},
  {id: 'first-steps', label: translate({message: "入门路线"}), quickLabel: translate({message: "从零开始学"})},
  {id: 'directions', label: translate({message: "实践方向"}), quickLabel: translate({message: "选择实践方向"})},
  {id: 'references', label: translate({message: "参考"}), quickLabel: translate({message: "参考"})},
];

export const scenarios = [
  {
    name: translate({message: "机器人踢足球"}),
    goal: translate({message: "观察防守位置，把足球踢进球门。"}),
    success: translate({message: "足球进入球门，踢球过程中保持身体平衡。"}),
    stages: [translate({message: "从相机图像中识别足球、球门和守门员，估计相对位置。"}), translate({message: "根据防守位置选择射门空当，安排靠近足球与踢球的动作。"}), translate({message: "协调腿部关节和身体重心，执行站位调整与踢球动作。"})],
    feedback: translate({message: "球向哪里运动？是否进入球门？根据新的观测判断结果并调整下一步。"}),
  },
  {
    name: translate({message: "机械臂拿杯子"}),
    goal: translate({message: "把桌上的杯子拿起，放进指定区域。"}),
    success: translate({message: "杯子放对位置，保持直立，手臂不碰撞。"}),
    stages: [translate({message: "从相机图像中定位杯子，估计它与机械臂的相对位置。"}), translate({message: "根据目标选择抓取位置，安排靠近、夹取和放置动作。"}), translate({message: "把动作转成关节指令，让手臂移动、夹爪闭合。"})],
    feedback: translate({message: "杯子移动了吗？抓稳了吗？用新的观测修正下一步动作。"}),
  },
  {
    name: translate({message: "四足向前走"}),
    goal: translate({message: "让四足机器人按给定速度向前行走。"}),
    success: translate({message: "跟上目标速度，保持平衡，并减少滑倒。"}),
    stages: [translate({message: "读取关节角度、机身姿态与接触信息，估计运动状态。"}), translate({message: "根据速度目标和当前状态，生成步态或选择运动策略。"}), translate({message: "跟踪关节目标，驱动电机完成抬腿、落脚和支撑。"})],
    feedback: translate({message: "速度偏了多少？身体是否倾斜？根据反馈调整下一步。"}),
  },
  {
    name: translate({message: "导航到门口"}),
    goal: translate({message: "从房间里出发，绕过障碍物到达门口。"}),
    success: translate({message: "到达目标区域，途中不碰撞，并记录耗时。"}),
    stages: [translate({message: "用相机或激光雷达观察障碍物，估计自身位置。"}), translate({message: "确定门口的位置，规划路线，并随环境变化重新规划。"}), translate({message: "把路径转成底盘的速度与转向指令，跟踪局部轨迹。"})],
    feedback: translate({message: "前方出现障碍了吗？位置有偏差吗？持续更新路径与动作。"}),
  },
];

export const capabilities = [
  {
    title: translate({message: "感知与理解"}), question: translate({message: "世界和自己，现在是什么状态？"}), icon: Eye,
    concepts: translate({message: "视觉与三维几何 · 状态估计 · 视觉语言模型"}),
    links: [
      {label: translate({message: "传感器、坐标系与标定"}), to: '/docs/foundations/perception/sensor-calibration-sim2real'},
      {label: translate({message: "多模态模型基础（VLM）"}), to: '/docs/foundations/vlm/intro'},
    ],
  },
  {
    title: translate({message: "学习与决策"}), question: translate({message: "为了完成任务，下一步做什么？"}), icon: BrainCircuit,
    concepts: translate({message: "模仿学习 · 强化学习 · VLA · 世界模型"}),
    links: [
      {label: translate({message: "模仿学习与强化学习"}), to: '/docs/foundations/rl-for-robotics/intro'},
      {label: translate({message: "视觉、语言到动作（VLA）"}), to: '/docs/foundations/vla/vla-intro'},
      {label: translate({message: "世界模型：预测动作后果"}), to: '/docs/foundations/world-model/intro'},
    ],
  },
  {
    title: translate({message: "运动与控制"}), question: translate({message: "怎样把动作准确、稳定地做出来？"}), icon: Move3D,
    concepts: translate({message: "运动学与动力学 · 运动规划 · 反馈控制"}),
    links: [
      {label: translate({message: "坐标变换与正逆运动学"}), to: '/docs/foundations/robotics-and-ros2/kinematics_transform'},
      {label: translate({message: "从 PID 到模型预测控制"}), to: '/docs/foundations/controllers/intro'},
      {label: translate({message: "MoveIt 2 运动规划"}), to: '/docs/foundations/robotics-and-ros2/moveit2_basics'},
    ],
  },
];

export const foundations = [
  {
    title: translate({message: "仿真与评测"}), icon: FlaskConical,
    description: translate({message: "在可重复的环境里验证方法，用成功率、误差和失败案例判断效果。"}),
    concepts: translate({message: "物理仿真 · 评测基准 · 仿真到真实（Sim2Real）"}),
    links: [
      {label: translate({message: "认识仿真工具"}), to: '/docs/foundations/simulation/intro'},
      {label: translate({message: "数据集与评测基准"}), to: '/docs/introduction/resources/datasets'},
    ],
  },
  {
    title: translate({message: "数据与训练"}), icon: Database,
    description: translate({message: "把示教或交互记录变成训练数据，检查观测、动作与时间是否对齐。"}),
    concepts: translate({message: "遥操作 · 数据质量 · 策略训练与验证"}),
    links: [
      {label: translate({message: "LeRobot 数据与工具链"}), to: '/docs/practices/robot-arm/data-collection/lerobot-course'},
      {label: translate({message: "从示教数据学习动作"}), to: '/docs/foundations/rl-for-robotics/imitation-learning'},
    ],
  },
  {
    title: translate({message: "本体与系统"}), icon: Settings2,
    description: translate({message: "让传感器、计算机、通信与执行器协同工作，把算法接到机器人身体上。"}),
    concepts: translate({message: "机械结构 · 传感器与电机 · ROS2 · 嵌入式"}),
    links: [
      {label: translate({message: "ROS2 系统导览"}), to: '/docs/foundations/robotics-and-ros2/course_introduction'},
      {label: translate({message: "SO-101 真机连接与调试"}), to: '/docs/practices/robot-arm/data-collection/so101-lerobot-real'},
    ],
  },
];

export const firstSteps = [
  {
    title: translate({message: "建立系统概念"}), preparation: translate({message: "从这里开始 · 无需设备"}),
    description: translate({message: "用一个机器人任务，分清观测、动作、目标和反馈。"}),
    outcome: translate({message: "能说明一个动作背后有哪些模块在协作。"}),
    link: {label: translate({message: "阅读具身导论"}), to: '/docs/introduction/what-is-embodied-ai'},
  },
  {
    title: translate({message: "亲手体验一个原理"}), preparation: translate({message: "浏览器即可"}),
    description: translate({message: "调整 PD 控制器的参数，观察响应、超调与稳定性。"}),
    outcome: translate({message: "能解释参数变化为什么会让关节抖动或变慢。"}),
    link: {label: translate({message: "打开 PD 控制实验"}), to: '/cs123/pd-playground'},
  },
  {
    title: translate({message: "跑通最小仿真"}), preparation: translate({message: "准备 Python 基础"}),
    description: translate({message: "安装 MuJoCo，加载模型，用代码推进仿真并施加控制。"}),
    outcome: translate({message: "保存一份自己能运行、能修改的仿真实验。"}),
    link: {label: translate({message: "开始 MuJoCo 教程"}), to: '/docs/foundations/simulation/mujoco'},
  },
  {
    title: translate({message: "沿一条主线完成项目"}), preparation: translate({message: "按章节补齐数学与算法"}),
    description: translate({message: "从单关节控制走到四足建模、步态与策略训练。"}),
    outcome: translate({message: "记录实验条件、结果和失败原因，形成完整复现。"}),
    link: {label: translate({message: "从零到一搭建四足机器人"}), to: '/docs/practices/quadruped/cs123/intro'},
  },
];

export const directions = [
  {
    title: translate({message: "抓取与操作"}), english: 'Manipulation',
    description: translate({message: "让机器人抓取、推拉、装配或协调双臂，改变物体的状态。"}),
    path: translate({message: "运动学 → 示教数据 → 模仿学习 → ACT → VLA"}),
    to: '/docs/practices/vla/act', label: translate({message: "做一次 ACT 双臂实验"}),
    note: translate({message: "建议先有 Python / PyTorch 基础，再按教程准备训练环境。"}),
  },
  {
    title: translate({message: "足式运动"}), english: 'Locomotion',
    description: translate({message: "让四足、双足机器人保持平衡，学会行走、转向和起身。"}),
    path: translate({message: "反馈控制 → 运动学 → 仿真 → 强化学习 → 迁移验证"}),
    to: '/docs/practices/quadruped/cs123/intro', label: translate({message: "进入四足机器人课程"}),
    note: translate({message: "先用仿真建立直觉，再继续 MicroDuck 等独立项目。"}),
  },
  {
    title: translate({message: "导航与移动操作"}), english: 'Navigation',
    description: translate({message: "找到目标、绕开障碍，再把移动与抓取连接成一个任务。"}),
    path: translate({message: "定位与建图 → 路径规划 → 导航 → 移动与操作协同"}),
    to: '/docs/overview/robotics-and-ros2-roadmap', label: translate({message: "先补机器人学与 ROS2"}),
    note: translate({message: "本站已有基础路线；完整导航与移动操作项目仍待补充。"}),
  },
];
