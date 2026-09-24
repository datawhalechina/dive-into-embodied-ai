import {translate} from '@docusaurus/Translate';
export type MegaMenuItem = {
  icon?: string;
  title: string;
  description: string;
  to: string;
  activeBasePath?: string;
  keywords?: string[];
  featured?: boolean;
};

export type MegaMenuColumn = {
  title: string;
  items: MegaMenuItem[];
};

export type MegaMenuFooter = {
  text: string;
  ctaLabel: string;
  to: string;
};

export type MegaMenuConfig = {
  id: string;
  label: string;
  activeBasePaths: string[];
  excludedBasePaths?: string[];
  panelWidth: number;
  columns: MegaMenuColumn[];
  footer: MegaMenuFooter;
};

// Keep existing course URLs while giving each course a single navigation home.
const tutorialBasePaths = [
  '/docs/practices/quadruped/cs123',
  '/docs/practices/robot-arm/data-collection/lerobot-course',
  '/docs/practices/wheel-legged/flamingo-isaaclab',
];

export const megaMenus: MegaMenuConfig[] = [
  {
    id: 'foundations',
    label: translate({message: "理论基础"}),
    activeBasePaths: ['/docs/foundations'],
    panelWidth: 1120,
    columns: [
      {
        title: translate({message: "大脑：智能决策"}),
        items: [
          {
            icon: '🎮',
            title: translate({message: "强化学习决策"}),
            description: translate({message: "从 MDP 到 PPO/SAC，建立序列决策直觉。"}),
            to: '/docs/foundations/rl-for-robotics/intro',
            activeBasePath: '/docs/foundations/rl-for-robotics',
            keywords: ['RL', 'PPO', 'SAC'],
            featured: true,
          },
          {
            icon: '🤖',
            title: translate({message: "视觉-语言-动作大模型(VLA)"}),
            description: translate({message: "RT、OpenVLA、ACT、Diffusion Policy 与 π 系列。"}),
            to: '/docs/foundations/vla/vla-intro',
            activeBasePath: '/docs/foundations/vla',
            keywords: ['VLA', 'RT-2', 'OpenVLA'],
          },
          {
            icon: '🌍',
            title: 'World-Model',
            description: translate({message: "世界模型在具身场景下的落地路径。"}),
            to: '/docs/foundations/world-model/intro',
            activeBasePath: '/docs/foundations/world-model',
            keywords: ['World Model'],
          },
        ],
      },
      {
        title: translate({message: "小脑：运动控制"}),
        items: [
          {
            icon: '🎛️',
            title: translate({message: "强化学习控制"}),
            description: translate({message: "把策略学习接到连续控制和机器人任务上。"}),
            to: '/docs/foundations/rl-for-robotics/ppo',
            activeBasePath: '/docs/foundations/rl-for-robotics',
            keywords: [translate({message: "控制"}), translate({message: "策略训练"})],
          },
          {
            icon: '🎚️',
            title: translate({message: "控制器"}),
            description: translate({message: "从 PID、LQR 到 MPC 与阻抗控制的连续教程。"}),
            to: '/docs/foundations/controllers/intro',
            activeBasePath: '/docs/foundations/controllers',
            keywords: ['PID', 'MPC', 'LQR'],
          },
          {
            icon: '🧭',
            title: translate({message: "运动规划"}),
            description: translate({message: "从模型、坐标树到 MoveIt 2 规划闭环。"}),
            to: '/docs/foundations/robotics-and-ros2/moveit2_basics',
            activeBasePath: '/docs/foundations/robotics-and-ros2',
            keywords: ['Motion Planning', 'MoveIt 2'],
          },
        ],
      },
      {
        title: translate({message: "感知系统"}),
        items: [
          {
            icon: '👁️',
            title: translate({message: "视觉感知与 VLM"}),
            description: translate({message: "Transformer、ViT、视觉编码器与多模态融合。"}),
            to: '/docs/foundations/vlm/intro',
            activeBasePath: '/docs/foundations/vlm',
            keywords: [translate({message: "视觉"}), 'VLM'],
          },
          {
            icon: '🦶',
            title: translate({message: "定位与触觉感知"}),
            description: translate({message: "SLAM、足端接触、触觉传感和多传感器融合。"}),
            to: '/docs/foundations/perception/placeholder',
            activeBasePath: '/docs/foundations/perception',
            keywords: ['SLAM', translate({message: "触觉"})],
          },
        ],
      },
      {
        title: translate({message: "工程底座"}),
        items: [
          {
            icon: '🧪',
            title: translate({message: "仿真工具"}),
            description: translate({message: "MuJoCo / Isaac Sim / Gymnasium / PyBullet 快速上手。"}),
            to: '/docs/foundations/simulation/intro',
            activeBasePath: '/docs/foundations/simulation',
            keywords: [translate({message: "仿真"}), 'MuJoCo'],
          },
          {
            icon: '🦾',
            title: 'ROS2',
            description: translate({message: "坐标变换、FK/IK、tf2、URDF 与 MoveIt 2。"}),
            to: '/docs/foundations/robotics-and-ros2/intro',
            activeBasePath: '/docs/foundations/robotics-and-ros2',
            keywords: ['FK', 'IK', 'ROS2'],
            featured: true,
          },
          {
            icon: '🔌',
            title: translate({message: "CAN 与 MCU 通信"}),
            description: translate({message: "底层通信、执行器协议和上下位机链路。"}),
            to: '/docs/foundations/communication/can-mcu',
            activeBasePath: '/docs/foundations/communication',
            keywords: ['CAN', 'MCU'],
          },
          {
            icon: '🦿',
            title: translate({message: "机械结构"}),
            description: translate({message: "连杆、关节、电机、减速器和末端执行器。"}),
            to: '/docs/foundations/hardware/placeholder',
            activeBasePath: '/docs/foundations/hardware',
            keywords: [translate({message: "硬件"}), translate({message: "本体"})],
          },
          {
            icon: '📦',
            title: translate({message: "数据工程与模仿学习"}),
            description: translate({message: "从遥操作数据到模仿学习和策略训练。"}),
            to: '/docs/foundations/rl-for-robotics/imitation-learning',
            activeBasePath: '/docs/foundations/rl-for-robotics',
            keywords: [translate({message: "数据"}), 'Imitation'],
          },
        ],
      },
    ],
    footer: {
      text: translate({message: "先用理论技能树定位缺口，再进入对应专题。"}),
      ctaLabel: translate({message: "查看理论技能树"}),
      to: '/docs/foundations/intro',
    },
  },
  {
    id: 'tutorials',
    label: translate({message: "系列教程"}),
    activeBasePaths: ['/docs/tutorials', '/cs123', ...tutorialBasePaths],
    panelWidth: 760,
    columns: [
      {
        title: translate({message: "按章节系统学习"}),
        items: [
          {
            title: translate({message: "从零到一搭建四足机器人"}),
            description: translate({message: "沿着 8 章主线，从 PD 控制与运动学走到策略训练和感知。"}),
            to: '/docs/practices/quadruped/cs123/intro',
            activeBasePath: '/docs/practices/quadruped/cs123',
            keywords: ['CS123', 'MuJoCo', translate({message: "8 章"})],
            featured: true,
          },
          {
            title: translate({message: "LeRobot 中文课程讲义"}),
            description: translate({message: "按课程顺序学习机器人数据、工具链与经典机器人学。"}),
            to: '/docs/practices/robot-arm/data-collection/lerobot-course',
            keywords: ['LeRobot', translate({message: "中文讲义"})],
          },
        ],
      },
      {
        title: translate({message: "学习准备与课程预告"}),
        items: [
          {
            title: translate({message: "选择一套教程"}),
            description: translate({message: "对照前置知识、学习目标和内容进度，选择学习主线。"}),
            to: '/docs/tutorials/intro',
          },
          {
            title: translate({message: "两轮足 Flamingo · 课程预告"}),
            description: translate({message: "了解 Isaac Lab 训练与跨仿真验证课程的规划。"}),
            to: '/docs/practices/wheel-legged/flamingo-isaaclab/preview',
            activeBasePath: '/docs/practices/wheel-legged/flamingo-isaaclab',
            keywords: ['Isaac Lab', translate({message: "预告"})],
          },
        ],
      },
    ],
    footer: {
      text: translate({message: "围绕一条主线，按章节逐步完成一个系统。"}),
      ctaLabel: translate({message: "查看系列教程"}),
      to: '/docs/tutorials/intro',
    },
  },
  {
    id: 'practices',
    label: translate({message: "项目实战"}),
    activeBasePaths: ['/docs/practices'],
    excludedBasePaths: tutorialBasePaths,
    panelWidth: 1180,
    columns: [
      {
        title: translate({message: "AMD 专区"}),
        items: [
          {
            icon: '☁️',
            title: translate({message: "AUP Learning Cloud 云算力"}),
            description: translate({message: "浏览器直连 Ryzen AI APU，体验 ROCm、JupyterHub 与 Code Server。"}),
            to: '/docs/practices/amd/aup-learning-cloud',
            activeBasePath: '/docs/practices/amd/aup-learning-cloud',
            keywords: ['AMD', 'AUP', 'Ryzen AI', 'ROCm', translate({message: "云算力"})],
          },
          {
            icon: '🦾',
            title: translate({message: "ACT 双臂操作训练"}),
            description: translate({message: "Radeon GPU 上的 ACT BF16 训练、闭环评测与成功视频。"}),
            to: '/docs/practices/amd/vla-act',
            activeBasePath: '/docs/practices/amd/vla-act',
            keywords: ['AMD', 'ROCm', 'ACT', 'ALOHA', 'LeRobot'],
          },
          {
            icon: '🐕',
            title: translate({message: "玩转 Pupper 四足机器人"}),
            description: translate({message: "AMD 旗舰项目：强化学习运动策略与 VLA 具身智能。"}),
            to: '/docs/practices/amd/pupper-control/intro',
            activeBasePath: '/docs/practices/amd/pupper-control',
            keywords: ['AMD', 'Pupper', 'RL Locomotion', 'VLA'],
            featured: true,
          },
        ],
      },
      {
        title: translate({message: "仿真实战"}),
        items: [
          {
            icon: '🐥',
            title: translate({message: "MicroDuck RL 小黄鸭双足机器人"}),
            description: translate({message: "mjlab + MuJoCo Warp GPU 并行 PPO 与稳定步态训练。"}),
            to: '/docs/practices/humanoid/microduck-rl',
            activeBasePath: '/docs/practices/humanoid/microduck-rl',
            keywords: ['MicroDuck', 'mjlab', 'PPO'],
            featured: true,
          },
          {
            icon: '🦾',
            title: translate({message: "MuJoCo 机械臂与 DDPG"}),
            description: translate({message: "从环境搭建到 InvertedPendulum、Reacher 与 Pusher 连续控制。"}),
            to: '/docs/practices/robot-arm/mujoco-arm-pick-place',
            activeBasePath: '/docs/practices/robot-arm/ddpg-mujoco',
            keywords: ['MuJoCo', 'DDPG', translate({message: "机械臂"})],
          },
          {
            icon: '🧠',
            title: translate({message: "ACT 双臂操作训练"}),
            description: translate({message: "用 ALOHA 仿真数据训练 ACT，并完成多回合评估。"}),
            to: '/docs/practices/vla/act',
            activeBasePath: '/docs/practices/vla/act',
            keywords: ['ACT', 'ALOHA', translate({message: "模仿学习"})],
          },
        ],
      },
      {
        title: translate({message: "真机实战"}),
        items: [
          {
            icon: '🦾',
            title: translate({message: "SO-101 + LeRobot 真机教程"}),
            description: translate({message: "从硬件连通、安全测试到真机动作回放。"}),
            to: '/docs/practices/robot-arm/data-collection/so101-lerobot-real',
            activeBasePath: '/docs/practices/robot-arm/data-collection/so101-lerobot-real',
            keywords: ['SO-101', 'LeRobot', translate({message: "真机"})],
            featured: true,
          },
          {
            icon: '🔁',
            title: translate({message: "Sim2Real 指南"}),
            description: translate({message: "从仿真策略走向真机部署的验证入口。"}),
            to: '/docs/practices/quadruped/sim2real-guide/placeholder',
            activeBasePath: '/docs/practices/quadruped/sim2real-guide',
            keywords: ['Sim2Real', translate({message: "部署"}), translate({message: "验证"})],
          },
        ],
      },
    ],
    footer: {
      text: translate({message: "按平台专区、仿真验证与真机部署选择项目。"}),
      ctaLabel: translate({message: "查看项目实战"}),
      to: '/docs/practices/intro',
    },
  },
  {
    id: 'information',
    label: translate({message: "具身信息"}),
    activeBasePaths: ['/docs/information'],
    panelWidth: 760,
    columns: [
      {
        title: translate({message: "论文与数据"}),
        items: [
          {
            title: translate({message: "论文与研究"}),
            description: translate({message: "按研究主题收集原论文、项目主页和站内解读。"}),
            to: '/docs/information/papers',
            keywords: ['ACT', 'Diffusion Policy', 'VLA'],
            featured: true,
          },
          {
            title: translate({message: "具身数据集"}),
            description: translate({message: "整理机器人示教数据、任务类型与评测基准。"}),
            to: '/docs/information/datasets',
            keywords: ['Open X-Embodiment', 'LIBERO'],
            featured: true,
          },
        ],
      },
      {
        title: translate({message: "代码与工具"}),
        items: [
          {
            title: translate({message: "开源项目"}),
            description: translate({message: "查找模型代码、训练框架和相关实战入口。"}),
            to: '/docs/information/open-source',
            keywords: ['LeRobot', 'OpenVLA', 'ACT'],
          },
          {
            title: translate({message: "仿真与工具"}),
            description: translate({message: "收集仿真引擎、学习环境与官方文档。"}),
            to: '/docs/information/tools',
            keywords: ['MuJoCo', 'Isaac Lab', 'Gymnasium'],
          },
        ],
      },
    ],
    footer: {
      text: translate({message: "找到论文、数据与工具，再连接到学习和实践。"}),
      ctaLabel: translate({message: "查看具身信息"}),
      to: '/docs/information/intro',
    },
  },
];

export function getMegaMenuById(id: string): MegaMenuConfig | undefined {
  return megaMenus.find((menu) => menu.id === id);
}
