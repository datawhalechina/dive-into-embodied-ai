import {translate} from '@docusaurus/Translate';

export type EmbodiedCarrier = {
  id: string;
  name: string;
  body: string;
  motion: string;
  official: {
    model: string;
    motion: string;
    source: {name: string; href: string};
  };
  platforms: {name: string; href: string}[];
  skills: string;
  skillNote?: string;
};

export const embodiedCarriers: EmbodiedCarrier[] = [
  {
    id: 'humanoid',
    name: translate({message: '人形机器人'}),
    body: translate({message: '双足、躯干与双臂'}),
    motion: translate({message: '双足交替迈步，手臂与身体协调运动。'}),
    official: {
      model: 'Unitree G1',
      motion: translate({message: '双足交替迈步，躯干与手臂协调完成行走。'}),
      source: {name: 'Unitree', href: 'https://www.unitree.com/g1/'},
    },
    platforms: [
      {name: 'Unitree G1', href: 'https://www.unitree.com/g1/'},
      {name: 'Figure 03', href: 'https://www.figure.ai/'},
    ],
    skills: translate({message: '抓取、操作、运动、导航'}),
    skillNote: translate({message: '可组合研究，依具体配置而定'}),
  },
  {
    id: 'arm',
    name: translate({message: '固定基座机械臂'}),
    body: translate({message: '单臂或双臂，基座固定'}),
    motion: translate({message: '基座保持固定，机械臂夹取并移动物体。'}),
    official: {
      model: 'ALOHA 2',
      motion: translate({message: '双臂在桌面上抓取并转移物体。遥操作演示，原片为 4× 速度。'}),
      source: {name: 'ALOHA 2', href: 'https://aloha-2.github.io/'},
    },
    platforms: [
      {name: 'Franka Research 3', href: 'https://franka.de/franka-research-3'},
      {name: 'ALOHA 2', href: 'https://aloha-2.github.io/'},
      {name: 'SO-101', href: 'https://huggingface.co/docs/lerobot/so101'},
    ],
    skills: translate({message: '抓取、操作'}),
  },
  {
    id: 'wheeled',
    name: translate({message: '轮式移动机器人'}),
    body: translate({message: '依靠轮式底盘移动'}),
    motion: translate({message: '车轮驱动底盘，沿路径行进并转向。'}),
    official: {
      model: 'Husky A300',
      motion: translate({message: '车轮驱动底盘，在户外地面行进并通过起伏地形。'}),
      source: {name: 'Clearpath Robotics', href: 'https://clearpathrobotics.com/husky-a300-unmanned-ground-vehicle-robot/'},
    },
    platforms: [
      {name: 'TurtleBot 4', href: 'https://turtlebot.github.io/turtlebot4-user-manual/'},
      {name: 'Jackal', href: 'https://clearpathrobotics.com/jackal-small-unmanned-ground-vehicle/'},
      {name: 'Husky A300', href: 'https://clearpathrobotics.com/husky-a300-unmanned-ground-vehicle-robot/'},
    ],
    skills: translate({message: '运动、导航'}),
  },
  {
    id: 'quadruped',
    name: translate({message: '四足机器人'}),
    body: translate({message: '用四条腿支撑与移动'}),
    motion: translate({message: '四条腿交替支撑，协调落脚与身体平衡。'}),
    official: {
      model: 'Unitree Go2',
      motion: translate({message: '跟随行走时，四条腿交替支撑并配合身体转向。'}),
      source: {name: 'Unitree', href: 'https://www.unitree.com/go2/'},
    },
    platforms: [
      {name: 'Unitree Go2', href: 'https://www.unitree.com/go2/'},
      {name: 'Spot', href: 'https://bostondynamics.com/products/spot/'},
    ],
    skills: translate({message: '运动、导航'}),
    skillNote: translate({message: '加装机械臂后可扩展操作技能'}),
  },
  {
    id: 'mobile-manipulator',
    name: translate({message: '移动操作平台'}),
    body: translate({message: '移动底座与操作机构组合'}),
    motion: translate({message: '底盘靠近工作台，机械臂伸出并取回物体。'}),
    official: {
      model: 'Mobile ALOHA',
      motion: translate({message: '移动底座靠近橱柜，机械臂伸出并打开柜门。'}),
      source: {name: 'Mobile ALOHA', href: 'https://mobile-aloha.github.io/'},
    },
    platforms: [
      {name: 'Stretch 3', href: 'https://docs-arch.hello-robot.com/0.3/hardware/hardware_guide_stretch_3/'},
      {name: 'Mobile ALOHA', href: 'https://mobile-aloha.github.io/'},
      {name: 'ALMA', href: 'https://rsl.ethz.ch/robots-media/alma.html'},
    ],
    skills: translate({message: '抓取、操作、运动、导航'}),
    skillNote: translate({message: '强调移动与操作的配合'}),
  },
  {
    id: 'simulation',
    name: translate({message: '仿真载体'}),
    body: translate({message: '由模型定义身体与传感器'}),
    motion: translate({message: '在虚拟环境中设置机器人、传感器与动作。'}),
    official: {
      model: 'MuJoCo · ALOHA 2',
      motion: translate({message: '虚拟双臂抓取并传递工具。来自 ALOHA 2 的 MuJoCo 遥操作仿真。'}),
      source: {name: 'ALOHA 2', href: 'https://aloha-2.github.io/'},
    },
    platforms: [
      {name: 'Habitat', href: 'https://aihabitat.org/'},
      {name: 'Isaac Lab', href: 'https://isaac-sim.github.io/IsaacLab/main/'},
      {name: 'MuJoCo', href: 'https://mujoco.org/'},
    ],
    skills: translate({message: '取决于模拟的身体与任务'}),
  },
];
