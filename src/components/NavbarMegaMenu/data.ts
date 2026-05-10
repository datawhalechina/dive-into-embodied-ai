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
    label: '总览',
    activeBasePaths: ['/docs/overview'],
    panelWidth: 1020,
    columns: [
      {
        title: '路径选择',
        items: [
          {
            icon: '🧭',
            title: '学习路径总览',
            description: '先选路线再开始学习，避免低效试错。',
            to: '/docs/overview/learning-path',
            keywords: ['路径', '规划'],
            featured: true,
          },
          {
            icon: '🤖',
            title: '机器人学和 ROS2 路线图',
            description: '按模块系统推进，建立可落地的工程闭环。',
            to: '/docs/overview/robotics-and-ros2-roadmap',
            keywords: ['ROS2', '工程闭环'],
          },
        ],
      },
      {
        title: '背景迁移',
        items: [
          {
            icon: '🧠',
            title: '哪些人适合转具身算法？',
            description: '判断你的既有经验是否适合切入具身算法。',
            to: '/docs/overview/algorithm-engineer-transition',
            keywords: ['转型', '岗位匹配'],
          },
          {
            icon: '🏢',
            title: '具身公司全景',
            description: '梳理国内外典型公司与产品方向。',
            to: '/docs/overview/company-landscape',
            keywords: ['公司', '产品方向'],
          },
        ],
      },
      {
        title: '入门方向',
        items: [
          {
            icon: '🌱',
            title: '具身智能路线图',
            description: '一图看懂这个方向在做什么、要补什么。',
            to: '/docs/overview/embodied-ai-roadmap',
            keywords: ['路线图', '入门'],
          },
        ],
      },
    ],
    footer: {
      text: '先定路径，再进基础和实践，整体效率更高。',
      ctaLabel: '查看总览',
      to: '/docs/overview/intro',
    },
  },
  {
    id: 'foundations',
    label: '基础篇',
    activeBasePaths: ['/docs/foundations'],
    panelWidth: 1020,
    columns: [
      {
        title: '系统主线',
        items: [
          {
            icon: '🤖',
            title: '机器人运动学与 ROS2 基础',
            description: '从系统认知到 MoveIt 2 工程闭环。',
            to: '/docs/foundations/robotics-and-ros2/intro',
            activeBasePath: '/docs/foundations/robotics-and-ros2',
            keywords: ['URDF', 'tf2', 'MoveIt 2'],
            featured: true,
          },
          {
            icon: '🧪',
            title: '仿真工具基础',
            description: 'MuJoCo / Isaac Sim / Gazebo + RViz / RQT 可视化。',
            to: '/docs/foundations/simulation/intro',
            activeBasePath: '/docs/foundations/simulation',
            keywords: ['MuJoCo', 'Isaac Sim', 'Gazebo'],
          },
        ],
      },
      {
        title: '算法主线',
        items: [
          {
            icon: '🎮',
            title: '强化学习与控制',
            description: '从 MDP 到 PPO/SAC，建立算法直觉。',
            to: '/docs/foundations/rl-for-robotics/intro',
            activeBasePath: '/docs/foundations/rl-for-robotics',
            keywords: ['DQN', 'PPO', 'SAC'],
          },
          {
            icon: '📦',
            title: '数据采集与模仿学习',
            description: '从遥操作数据到 Diffusion Policy。',
            to: '/docs/foundations/rl-for-robotics/imitation-learning',
            activeBasePath: '/docs/foundations/rl-for-robotics',
            keywords: ['Imitation', 'Diffusion'],
          },
          {
            icon: '🧠',
            title: '视觉语言大模型(VLM)',
            description: 'Transformer、ViT、视觉编码器与多模态融合。',
            to: '/docs/foundations/vlm/intro',
            activeBasePath: '/docs/foundations/vlm',
            keywords: ['VLM', 'Transformer', 'ViT'],
          },
          {
            icon: '🤖',
            title: '视觉-语言-动作大模型(VLA)',
            description: 'RT-1/RT-2、OpenVLA、ACT、Diffusion Policy、π 系列。',
            to: '/docs/foundations/vla/vla-intro',
            activeBasePath: '/docs/foundations/vla',
            keywords: ['VLA', 'RT-2', 'OpenVLA'],
          },
          {
            icon: '🌍',
            title: '世界模型(World Model)',
            description: '世界模型在具身场景下的落地(施工中)。',
            to: '/docs/foundations/world-model/placeholder',
            activeBasePath: '/docs/foundations/world-model',
            keywords: ['World Model'],
          },
        ],
      },
      {
        title: '入门',
        items: [
          {
            icon: '🌱',
            title: '基础篇概述',
            description: '看懂代码、跑通环境、读懂论文的最小集。',
            to: '/docs/foundations/intro',
            keywords: ['入门', '基础'],
          },
        ],
      },
    ],
    footer: {
      text: '建议顺序：系统主线 → 算法主线。',
      ctaLabel: '进入基础篇',
      to: '/docs/foundations/intro',
    },
  },
  {
    id: 'practices',
    label: '实践篇',
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
        title: '方向总览',
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
      ctaLabel: '查看全部项目',
      to: '/docs/practices/intro',
    },
  },
  {
    id: 'career',
    label: '求职篇',
    activeBasePaths: ['/docs/career'],
    panelWidth: 1020,
    columns: [
      {
        title: '准备',
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
            icon: '📄',
            title: '简历与作品集',
            description: '怎么把项目写进简历和作品集。',
            to: '/docs/career/resume-portfolio/placeholder',
            activeBasePath: '/docs/career/resume-portfolio',
            keywords: ['简历', '作品集'],
          },
        ],
      },
      {
        title: '信息',
        items: [
          {
            icon: '🏢',
            title: '公司技术栈',
            description: '主流具身公司在用什么栈。',
            to: '/docs/career/company-tech-stacks/placeholder',
            activeBasePath: '/docs/career/company-tech-stacks',
            keywords: ['公司', '技术栈'],
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
      {
        title: '路径',
        items: [
          {
            icon: '🔁',
            title: '转岗路径',
            description: '不同背景的转岗路线参考。',
            to: '/docs/career/transition-paths/intro',
            activeBasePath: '/docs/career/transition-paths',
            keywords: ['转岗', '路线'],
          },
          {
            icon: '🤝',
            title: '社区与内推',
            description: '社群、活动与内推机会。',
            to: '/docs/career/community/placeholder',
            activeBasePath: '/docs/career/community',
            keywords: ['社区', '内推'],
          },
        ],
      },
    ],
    footer: {
      text: '先看岗位拆解，再补面经和简历。',
      ctaLabel: '查看求职篇',
      to: '/docs/career/intro',
    },
  },
];

export function getMegaMenuById(id: string): MegaMenuConfig | undefined {
  return megaMenus.find((menu) => menu.id === id);
}
