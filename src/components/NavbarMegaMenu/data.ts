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
  panelWidth: number;
  columns: MegaMenuColumn[];
  footer: MegaMenuFooter;
};

export const megaMenus: MegaMenuConfig[] = [
  {
    id: 'foundations',
    label: '理论基础',
    activeBasePaths: ['/docs/foundations'],
    panelWidth: 1120,
    columns: [
      {
        title: '大脑：智能决策',
        items: [
          {
            icon: '🎮',
            title: '强化学习决策',
            description: '从 MDP 到 PPO/SAC，建立序列决策直觉。',
            to: '/docs/foundations/rl-for-robotics/intro',
            activeBasePath: '/docs/foundations/rl-for-robotics',
            keywords: ['RL', 'PPO', 'SAC'],
            featured: true,
          },
          {
            icon: '🤖',
            title: '视觉-语言-动作大模型(VLA)',
            description: 'RT、OpenVLA、ACT、Diffusion Policy 与 π 系列。',
            to: '/docs/foundations/vla/vla-intro',
            activeBasePath: '/docs/foundations/vla',
            keywords: ['VLA', 'RT-2', 'OpenVLA'],
          },
          {
            icon: '🌍',
            title: 'World-Model',
            description: '世界模型在具身场景下的落地路径。',
            to: '/docs/foundations/world-model/intro',
            activeBasePath: '/docs/foundations/world-model',
            keywords: ['World Model'],
          },
        ],
      },
      {
        title: '小脑：运动控制',
        items: [
          {
            icon: '🎛️',
            title: '强化学习控制',
            description: '把策略学习接到连续控制和机器人任务上。',
            to: '/docs/foundations/rl-for-robotics/ppo',
            activeBasePath: '/docs/foundations/rl-for-robotics',
            keywords: ['控制', '策略训练'],
          },
          {
            icon: '🎚️',
            title: '控制器',
            description: '从 PID、LQR 到 MPC 与阻抗控制的连续教程。',
            to: '/docs/foundations/controllers/intro',
            activeBasePath: '/docs/foundations/controllers',
            keywords: ['PID', 'MPC', 'LQR'],
          },
          {
            icon: '🧭',
            title: '运动规划',
            description: '从模型、坐标树到 MoveIt 2 规划闭环。',
            to: '/docs/foundations/robotics-and-ros2/moveit2_basics',
            activeBasePath: '/docs/foundations/robotics-and-ros2',
            keywords: ['Motion Planning', 'MoveIt 2'],
          },
        ],
      },
      {
        title: '感知系统',
        items: [
          {
            icon: '👁️',
            title: '视觉感知与 VLM',
            description: 'Transformer、ViT、视觉编码器与多模态融合。',
            to: '/docs/foundations/vlm/intro',
            activeBasePath: '/docs/foundations/vlm',
            keywords: ['视觉', 'VLM'],
          },
          {
            icon: '🦶',
            title: '定位与触觉感知',
            description: 'SLAM、足端接触、触觉传感和多传感器融合。',
            to: '/docs/foundations/perception/placeholder',
            activeBasePath: '/docs/foundations/perception',
            keywords: ['SLAM', '触觉'],
          },
        ],
      },
      {
        title: '工程底座',
        items: [
          {
            icon: '🧪',
            title: '仿真工具',
            description: 'MuJoCo / Isaac Sim / Gymnasium / PyBullet 快速上手。',
            to: '/docs/foundations/simulation/intro',
            activeBasePath: '/docs/foundations/simulation',
            keywords: ['仿真', 'MuJoCo'],
          },
          {
            icon: '🦾',
            title: 'ROS2',
            description: '坐标变换、FK/IK、tf2、URDF 与 MoveIt 2。',
            to: '/docs/foundations/robotics-and-ros2/intro',
            activeBasePath: '/docs/foundations/robotics-and-ros2',
            keywords: ['FK', 'IK', 'ROS2'],
            featured: true,
          },
          {
            icon: '🔌',
            title: 'CAN 与 MCU 通信',
            description: '底层通信、执行器协议和上下位机链路。',
            to: '/docs/foundations/communication/can-mcu',
            activeBasePath: '/docs/foundations/communication',
            keywords: ['CAN', 'MCU'],
          },
          {
            icon: '🦿',
            title: '机械结构',
            description: '连杆、关节、电机、减速器和末端执行器。',
            to: '/docs/foundations/hardware/placeholder',
            activeBasePath: '/docs/foundations/hardware',
            keywords: ['硬件', '本体'],
          },
          {
            icon: '📦',
            title: '数据工程与模仿学习',
            description: '从遥操作数据到模仿学习和策略训练。',
            to: '/docs/foundations/rl-for-robotics/imitation-learning',
            activeBasePath: '/docs/foundations/rl-for-robotics',
            keywords: ['数据', 'Imitation'],
          },
        ],
      },
    ],
    footer: {
      text: '先用理论技能树定位缺口，再进入对应专题。',
      ctaLabel: '查看理论技能树',
      to: '/docs/foundations/intro',
    },
  },
  {
    id: 'practices',
    label: '项目实战',
    activeBasePaths: ['/docs/practices'],
    panelWidth: 1180,
    columns: [
      {
        title: 'AMD 专区',
        items: [
          {
            icon: '☁️',
            title: 'AUP Learning Cloud 云算力',
            description: '浏览器直连 Ryzen AI APU，体验 ROCm、JupyterHub 与 Code Server。',
            to: '/docs/practices/amd/aup-learning-cloud',
            activeBasePath: '/docs/practices/amd/aup-learning-cloud',
            keywords: ['AMD', 'AUP', 'Ryzen AI', 'ROCm', '云算力'],
          },
          {
            icon: '🦾',
            title: 'ACT 双臂操作训练',
            description: 'Radeon GPU 上的 ACT BF16 训练、闭环评测与成功视频。',
            to: '/docs/practices/amd/vla-act',
            activeBasePath: '/docs/practices/amd/vla-act',
            keywords: ['AMD', 'ROCm', 'ACT', 'ALOHA', 'LeRobot'],
          },
          {
            icon: '🐕',
            title: '玩转 Pupper 四足机器人',
            description: 'AMD 旗舰项目：强化学习运动策略与 VLA 具身智能。',
            to: '/docs/practices/amd/pupper-control/intro',
            activeBasePath: '/docs/practices/amd/pupper-control',
            keywords: ['AMD', 'Pupper', 'RL Locomotion', 'VLA'],
            featured: true,
          },
        ],
      },
      {
        title: '仿真实战',
        items: [
          {
            icon: '🐕',
            title: '从零到一搭建四足机器人',
            description: 'CS123 四足仿真课程，8 章从 PD 走到 LLM 控制。',
            to: '/docs/practices/quadruped/cs123/intro',
            activeBasePath: '/docs/practices/quadruped/cs123',
            keywords: ['CS123', 'MuJoCo', 'PPO'],
            featured: true,
          },
          {
            icon: '🐥',
            title: 'MicroDuck RL 小黄鸭双足机器人',
            description: 'mjlab + MuJoCo Warp GPU 并行 PPO 与稳定步态训练。',
            to: '/docs/practices/humanoid/microduck-rl',
            activeBasePath: '/docs/practices/humanoid/microduck-rl',
            keywords: ['MicroDuck', 'mjlab', 'PPO'],
            featured: true,
          },
          {
            icon: '🦾',
            title: 'MuJoCo 机械臂与 DDPG',
            description: '从环境搭建到 InvertedPendulum、Reacher 与 Pusher 连续控制。',
            to: '/docs/practices/robot-arm/mujoco-arm-pick-place',
            activeBasePath: '/docs/practices/robot-arm/ddpg-mujoco',
            keywords: ['MuJoCo', 'DDPG', '机械臂'],
          },
          {
            icon: '🧠',
            title: 'ACT 双臂操作训练',
            description: '用 ALOHA 仿真数据训练 ACT，并完成多回合评估。',
            to: '/docs/practices/vla/act',
            activeBasePath: '/docs/practices/vla/act',
            keywords: ['ACT', 'ALOHA', '模仿学习'],
          },
          {
            icon: '🛞',
            title: '两轮足 Flamingo · Isaac Lab',
            description: 'Isaac Lab + PPO/CaT 训练与跨仿真验证。',
            to: '/docs/practices/wheel-legged/flamingo-isaaclab/preview',
            activeBasePath: '/docs/practices/wheel-legged/flamingo-isaaclab',
            keywords: ['Flamingo', 'Isaac Lab', 'Sim2Sim'],
          },
        ],
      },
      {
        title: '真机实战',
        items: [
          {
            icon: '🦾',
            title: 'SO-101 + LeRobot 真机教程',
            description: '从硬件连通、安全测试到真机动作回放。',
            to: '/docs/practices/robot-arm/data-collection/so101-lerobot-real',
            activeBasePath: '/docs/practices/robot-arm/data-collection/so101-lerobot-real',
            keywords: ['SO-101', 'LeRobot', '真机'],
            featured: true,
          },
          {
            icon: '🤗',
            title: 'LeRobot 中文课程讲义',
            description: '补齐机器人数据、工具链与真机学习流程的前置知识。',
            to: '/docs/practices/robot-arm/data-collection/lerobot-course',
            activeBasePath: '/docs/practices/robot-arm/data-collection/lerobot-course',
            keywords: ['LeRobot', '数据采集', '机器人学习'],
          },
          {
            icon: '🔁',
            title: 'Sim2Real 指南',
            description: '从仿真策略走向真机部署的验证入口。',
            to: '/docs/practices/quadruped/sim2real-guide/placeholder',
            activeBasePath: '/docs/practices/quadruped/sim2real-guide',
            keywords: ['Sim2Real', '部署', '验证'],
          },
        ],
      },
    ],
    footer: {
      text: '按平台专区、仿真验证与真机部署选择项目。',
      ctaLabel: '查看项目实战',
      to: '/docs/practices/intro',
    },
  },
];

export function getMegaMenuById(id: string): MegaMenuConfig | undefined {
  return megaMenus.find((menu) => menu.id === id);
}
