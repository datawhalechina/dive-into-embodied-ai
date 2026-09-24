import {translate} from '@docusaurus/Translate';

export type EmbodiedCarrier = {
  id: string;
  name: string;
  motion: string;
  official: {
    model: string;
    motion: string;
    source: {name: string; href: string};
  };
};

export const embodiedCarriers: EmbodiedCarrier[] = [
  {
    id: 'humanoid',
    name: translate({message: '人形机器人'}),
    motion: translate({message: '双足交替迈步，手臂与身体协调运动。'}),
    official: {
      model: 'Unitree G1',
      motion: translate({message: '双足交替迈步，躯干与手臂协调完成行走。'}),
      source: {name: 'Unitree', href: 'https://www.unitree.com/g1/'},
    },
  },
  {
    id: 'arm',
    name: translate({message: '固定基座机械臂'}),
    motion: translate({message: '基座保持固定，机械臂夹取并移动物体。'}),
    official: {
      model: 'ALOHA 2',
      motion: translate({message: '双臂在桌面上抓取并转移物体。遥操作演示，原片为 4× 速度。'}),
      source: {name: 'ALOHA 2', href: 'https://aloha-2.github.io/'},
    },
  },
  {
    id: 'wheeled',
    name: translate({message: '轮式移动机器人'}),
    motion: translate({message: '车轮驱动底盘，沿路径行进并转向。'}),
    official: {
      model: 'Husky A300',
      motion: translate({message: '车轮驱动底盘，在户外地面行进并通过起伏地形。'}),
      source: {name: 'Clearpath Robotics', href: 'https://clearpathrobotics.com/husky-a300-unmanned-ground-vehicle-robot/'},
    },
  },
  {
    id: 'quadruped',
    name: translate({message: '四足机器人'}),
    motion: translate({message: '四条腿交替支撑，协调落脚与身体平衡。'}),
    official: {
      model: 'Unitree Go2',
      motion: translate({message: '跟随行走时，四条腿交替支撑并配合身体转向。'}),
      source: {name: 'Unitree', href: 'https://www.unitree.com/go2/'},
    },
  },
  {
    id: 'mobile-manipulator',
    name: translate({message: '移动操作平台'}),
    motion: translate({message: '底盘靠近工作台，机械臂伸出并取回物体。'}),
    official: {
      model: 'Mobile ALOHA',
      motion: translate({message: '移动底座靠近橱柜，机械臂伸出并打开柜门。'}),
      source: {name: 'Mobile ALOHA', href: 'https://mobile-aloha.github.io/'},
    },
  },
  {
    id: 'simulation',
    name: translate({message: '仿真载体'}),
    motion: translate({message: '在虚拟环境中设置机器人、传感器与动作。'}),
    official: {
      model: 'MuJoCo · ALOHA 2',
      motion: translate({message: '虚拟双臂抓取并传递工具。来自 ALOHA 2 的 MuJoCo 遥操作仿真。'}),
      source: {name: 'ALOHA 2', href: 'https://aloha-2.github.io/'},
    },
  },
];
