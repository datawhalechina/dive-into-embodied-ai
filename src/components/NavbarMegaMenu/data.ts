import {translate} from '@docusaurus/Translate';

export type MegaMenuItem = {
  title: string;
  to: string;
  activeBasePath?: string;
};

export type MegaMenuColumn = {
  title: string;
  items: MegaMenuItem[];
};

export type MegaMenuConfig = {
  id: string;
  label: string;
  to: string;
  activeBasePaths: string[];
  columns: MegaMenuColumn[];
};

export const megaMenus: MegaMenuConfig[] = [
  {
    id: 'foundations',
    label: translate({message: "理论基础"}),
    to: '/docs/foundations/intro',
    activeBasePaths: ['/docs/foundations'],
    columns: [
      {
        title: translate({message: "大脑：智能决策"}),
        items: [
          {
            title: translate({message: "强化学习决策"}),
            to: '/docs/foundations/rl-for-robotics/intro',
            activeBasePath: '/docs/foundations/rl-for-robotics',
          },
          {
            title: translate({message: "视觉-语言-动作大模型(VLA)"}),
            to: '/docs/foundations/vla/vla-intro',
            activeBasePath: '/docs/foundations/vla',
          },
          {
            title: 'World-Model',
            to: '/docs/foundations/world-model/intro',
            activeBasePath: '/docs/foundations/world-model',
          },
        ],
      },
      {
        title: translate({message: "小脑：运动控制"}),
        items: [
          {
            title: translate({message: "强化学习控制"}),
            to: '/docs/foundations/rl-for-robotics/ppo',
          },
          {
            title: translate({message: "控制器"}),
            to: '/docs/foundations/controllers/intro',
            activeBasePath: '/docs/foundations/controllers',
          },
          {
            title: translate({message: "运动规划"}),
            to: '/docs/foundations/robotics-and-ros2/moveit2_basics',
          },
        ],
      },
      {
        title: translate({message: "感知系统"}),
        items: [
          {
            title: translate({message: "视觉感知与 VLM"}),
            to: '/docs/foundations/vlm/intro',
            activeBasePath: '/docs/foundations/vlm',
          },
          {
            title: translate({message: "定位与触觉感知"}),
            to: '/docs/foundations/perception/placeholder',
            activeBasePath: '/docs/foundations/perception',
          },
        ],
      },
      {
        title: translate({message: "工程底座"}),
        items: [
          {
            title: translate({message: "仿真工具"}),
            to: '/docs/foundations/simulation/intro',
            activeBasePath: '/docs/foundations/simulation',
          },
          {
            title: 'ROS2',
            to: '/docs/foundations/robotics-and-ros2/intro',
            activeBasePath: '/docs/foundations/robotics-and-ros2',
          },
          {
            title: translate({message: "CAN 与 MCU 通信"}),
            to: '/docs/foundations/communication/can-mcu',
            activeBasePath: '/docs/foundations/communication',
          },
          {
            title: translate({message: "机械结构"}),
            to: '/docs/foundations/hardware/placeholder',
            activeBasePath: '/docs/foundations/hardware',
          },
          {
            title: translate({message: "数据工程与模仿学习"}),
            to: '/docs/foundations/rl-for-robotics/imitation-learning',
          },
        ],
      },
    ],
  },
  {
    id: 'practices',
    label: translate({message: "项目实战"}),
    to: '/docs/practices/intro',
    activeBasePaths: ['/docs/practices', '/cs123'],
    columns: [
      {
        title: translate({message: "AMD 专区"}),
        items: [
          {
            title: translate({message: "AUP Learning Cloud 云算力"}),
            to: '/docs/practices/amd/aup-learning-cloud',
            activeBasePath: '/docs/practices/amd/aup-learning-cloud',
          },
          {
            title: translate({message: "ACT 双臂操作训练"}),
            to: '/docs/practices/amd/vla-act',
            activeBasePath: '/docs/practices/amd/vla-act',
          },
          {
            title: translate({message: "玩转 Pupper 四足机器人"}),
            to: '/docs/practices/amd/pupper-control/intro',
            activeBasePath: '/docs/practices/amd/pupper-control',
          },
        ],
      },
      {
        title: translate({message: "仿真实战"}),
        items: [
          {
            title: translate({message: "四足系列"}),
            to: '/docs/practices/quadruped/placeholder',
            activeBasePath: '/docs/practices/quadruped',
          },
          {
            title: translate({message: "MicroDuck RL 小黄鸭双足机器人"}),
            to: '/docs/practices/humanoid/microduck-rl',
            activeBasePath: '/docs/practices/humanoid/microduck-rl',
          },
          {
            title: translate({message: "MuJoCo 机械臂与 DDPG"}),
            to: '/docs/practices/robot-arm/mujoco-arm-pick-place',
            activeBasePath: '/docs/practices/robot-arm/ddpg-mujoco',
          },
          {
            title: translate({message: "ACT 双臂操作训练"}),
            to: '/docs/practices/vla/act',
            activeBasePath: '/docs/practices/vla/act',
          },
          {
            title: translate({message: "两轮足 Flamingo · 课程预告"}),
            to: '/docs/practices/wheel-legged/flamingo-isaaclab/preview',
            activeBasePath: '/docs/practices/wheel-legged/flamingo-isaaclab',
          },
        ],
      },
      {
        title: translate({message: "真机实战"}),
        items: [
          {
            title: translate({message: "LeRobot 中文课程讲义"}),
            to: '/docs/practices/robot-arm/data-collection/lerobot-course',
          },
          {
            title: translate({message: "SO-101 + LeRobot 真机教程"}),
            to: '/docs/practices/robot-arm/data-collection/so101-lerobot-real',
            activeBasePath: '/docs/practices/robot-arm/data-collection/so101-lerobot-real',
          },
          {
            title: translate({message: "Sim2Real 指南"}),
            to: '/docs/practices/quadruped/sim2real-guide/placeholder',
            activeBasePath: '/docs/practices/quadruped/sim2real-guide',
          },
        ],
      },
    ],
  },
];

export function getMegaMenuById(id: string): MegaMenuConfig | undefined {
  return megaMenus.find((menu) => menu.id === id);
}
