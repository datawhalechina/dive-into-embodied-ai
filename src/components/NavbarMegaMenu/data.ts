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
    id: 'overview',
    label: '零基础入门',
    activeBasePaths: ['/docs/overview'],
    panelWidth: 780,
    columns: [
      {
        title: '路径选择',
        items: [
          {
            icon: '🧭',
            title: '学习路径',
            description: '从零基础到项目和求职的主线安排。',
            to: '/docs/overview/learning-path',
            keywords: ['入门', '路径'],
            featured: true,
          },
        ],
      },
      {
        title: '第一项目',
        items: [
          {
            icon: '🤖',
            title: '从0到1搭建四足机器人',
            description: '从一个完整项目入手理解具身智能系统。',
            to: '/docs/practices/quadruped/cs123/intro',
            activeBasePath: '/docs/practices/quadruped/cs123',
            keywords: ['四足', '项目'],
            featured: true,
          },
        ],
      },
    ],
    footer: {
      text: '先定路径，再用一个项目建立整体感。',
      ctaLabel: '查看入门路径',
      to: '/docs/overview/learning-path',
    },
  },
  {
    id: 'foundations',
    label: '技能树进阶',
    activeBasePaths: ['/docs/foundations'],
    panelWidth: 1280,
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
            to: '/docs/foundations/world-model/placeholder',
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
        title: '仿真环境',
        items: [
          {
            icon: '🧪',
            title: '仿真工具基础',
            description: 'MuJoCo / Isaac Sim / Gymnasium / PyBullet 快速上手。',
            to: '/docs/foundations/simulation/intro',
            activeBasePath: '/docs/foundations/simulation',
            keywords: ['仿真', 'MuJoCo'],
          },
        ],
      },
      {
        title: '神经 / 硬件 / 数据 / 安全',
        items: [
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
          {
            icon: '🛡️',
            title: '安全防护',
            description: '限位、急停、碰撞检测和实机部署检查。',
            to: '/docs/foundations/safety/placeholder',
            activeBasePath: '/docs/foundations/safety',
            keywords: ['安全', '真机'],
          },
        ],
      },
    ],
    footer: {
      text: '先用技能树定位缺口，再进入对应专题。',
      ctaLabel: '查看技能树进阶',
      to: '/docs/foundations/intro',
    },
  },
  {
    id: 'practices',
    label: '项目实战',
    activeBasePaths: ['/docs/practices'],
    panelWidth: 940,
    columns: [
      {
        title: '课程入口',
        items: [
          {
            icon: '🐕',
            title: '从零到一搭建四足机器人',
            description: 'CS123 四足仿真课程，8 章从 PD 走到 LLM 控制。',
            to: '/docs/practices/quadruped/cs123/intro',
            activeBasePath: '/docs/practices/quadruped/cs123',
            keywords: ['CS123', '四足', 'Pupper'],
            featured: true,
          },
          {
            icon: '🤗',
            title: 'LeRobot 中文课程讲义',
            description: '基于 Hugging Face Robotics Course 整理的中文主线。',
            to: '/docs/practices/robot-arm/data-collection/lerobot-course',
            activeBasePath: '/docs/practices/robot-arm/data-collection/lerobot-course',
            keywords: ['LeRobot', '课程', '机器人学习'],
            featured: true,
          },
        ],
      },
      {
        title: '本体方向',
        items: [
          {
            icon: '🦾',
            title: '机械臂方向',
            description: '抓取、数据采集、VLA 控制与真机部署入口。',
            to: '/docs/practices/robot-arm/placeholder',
            activeBasePath: '/docs/practices/robot-arm/placeholder',
            keywords: ['MuJoCo', 'VLA', '模仿学习'],
          },
          {
            icon: '🐕',
            title: '四足机器人方向',
            description: '四足课程、sim2sim 与 sim2real 实践入口。',
            to: '/docs/practices/quadruped/placeholder',
            activeBasePath: '/docs/practices/quadruped/placeholder',
            keywords: ['CS123', '四足', 'sim2real'],
          },
          {
            icon: '🚶',
            title: '双足 / 人形机器人',
            description: '平衡控制、运动跟踪与任务规划。',
            to: '/docs/practices/humanoid/placeholder',
            activeBasePath: '/docs/practices/humanoid',
            keywords: ['Humanoid', '平衡', '跟踪'],
          },
        ],
      },
      {
        title: '部署与综合',
        items: [
          {
            icon: '🦾',
            title: 'SO-101 + LeRobot 真机教程',
            description: '从硬件连接到策略回放的最小真机流程。',
            to: '/docs/practices/robot-arm/data-collection/so101-lerobot-real',
            activeBasePath: '/docs/practices/robot-arm/data-collection/so101-lerobot-real',
            keywords: ['SO-101', 'LeRobot', '真机'],
            featured: true,
          },
          {
            icon: '📦',
            title: '移动操作',
            description: '导航 + 操作 + 视觉语言导航闭环。',
            to: '/docs/practices/mobile-manipulation/placeholder',
            activeBasePath: '/docs/practices/mobile-manipulation',
            keywords: ['Navigation', 'VLN', 'Mobile Manip'],
          },
        ],
      },
    ],
    footer: {
      text: '先仿真跑通，再做真机验证。',
      ctaLabel: '查看项目实战',
      to: '/docs/practices/intro',
    },
  },
  {
    id: 'career',
    label: '求职面试',
    activeBasePaths: ['/docs/career'],
    panelWidth: 620,
    columns: [
      {
        title: '核心准备',
        items: [
          {
            icon: '🗺️',
            title: '岗位技能拆解',
            description: '把岗位 JD 拆成可学习的技能项。',
            to: '/docs/career/job-skill-map/placeholder',
            activeBasePath: '/docs/career/job-skill-map',
            keywords: ['岗位', '技能图谱'],
            featured: true,
          },
          {
            icon: '📝',
            title: '面经与八股',
            description: '常见面试题与高频考点。',
            to: '/docs/career/interview-questions/placeholder',
            activeBasePath: '/docs/career/interview-questions',
            keywords: ['面试', '八股'],
          },
          {
            icon: '📌',
            title: '招聘信息',
            description: '在招岗位与社招/校招通道。',
            to: '/docs/career/job-listings/placeholder',
            activeBasePath: '/docs/career/job-listings',
            keywords: ['招聘', '内推'],
          },
        ],
      },
    ],
    footer: {
      text: '先看岗位拆解，再补面经、招聘信息和简历表达。',
      ctaLabel: '查看求职面试',
      to: '/docs/career/intro',
    },
  },
];

export function getMegaMenuById(id: string): MegaMenuConfig | undefined {
  return megaMenus.find((menu) => menu.id === id);
}
