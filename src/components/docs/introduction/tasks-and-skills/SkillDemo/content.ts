import {translate} from '@docusaurus/Translate';
import type {Skill} from './types';

export const skillDemos = {
  grasping: {
    title: translate({message: "拿稳，再提起"}),
    scene: translate({message: "机械臂用平行夹爪对准橙色方块，闭合夹爪，再将方块提离桌面。"}),
    focus: translate({message: "看夹爪与物体之间的关系：从尚未接触，到稳定夹持。"}),
    phases: [
      {name: translate({message: "观察物体"}), start: 0, sample: .05, detail: translate({message: "确定方块的位置与朝向，选择两侧的夹持区域。"})},
      {name: translate({message: "对准夹爪"}), start: .22, sample: .42, detail: translate({message: "张开夹爪，移动到方块两侧，保持接近方向。"})},
      {name: translate({message: "闭合夹持"}), start: .48, sample: .62, detail: translate({message: "两侧指爪闭合，与物体接触并建立稳定夹持。"})},
      {name: translate({message: "提起确认"}), start: .64, sample: 1, detail: translate({message: "保持夹持并向上提起；物体随夹爪移动，完成抓取。"})},
    ],
  },
  manipulation: {
    title: translate({message: "抓住把手，把抽屉拉开"}),
    scene: translate({message: "机械臂夹住抽屉把手，持续保持接触，沿滑轨方向拉开抽屉。"}),
    focus: translate({message: "抓住只是开始。操作还要持续改变抽屉的状态。"}),
    phases: [
      {name: translate({message: "靠近把手"}), start: 0, sample: .26, detail: translate({message: "观察把手与滑轨方向，将夹爪移动到把手附近。"})},
      {name: translate({message: "建立接触"}), start: .34, sample: .46, detail: translate({message: "夹住把手，为后续拉动建立稳定接触。"})},
      {name: translate({message: "沿轨拉动"}), start: .47, sample: .7, detail: translate({message: "保持接触并沿滑轨移动；真实系统还需要根据视觉与力反馈修正动作。"})},
      {name: translate({message: "检查状态"}), start: .92, sample: 1, detail: translate({message: "抽屉到达目标开度，机械臂停止拉动并保持位置。"})},
    ],
  },
  locomotion: {
    title: translate({message: "保持平衡，迈上台阶"}),
    scene: translate({message: "四足机器人交替抬腿、落脚和支撑，逐步登上三级台阶。"}),
    focus: translate({message: "看身体怎么移动：协调落脚位置、关节与重心。"}),
    phases: [
      {name: translate({message: "准备支撑"}), start: 0, sample: 0, detail: translate({message: "四足落地支撑身体，准备向前移动。"})},
      {name: translate({message: "交替迈步"}), start: .12, sample: .28, detail: translate({message: "支撑腿承重，摆动腿向前迈出，为下一步落脚做准备。"})},
      {name: translate({message: "抬腿越阶"}), start: .42, sample: .62, detail: translate({message: "提高摆腿轨迹，把脚落在台阶上，逐步抬高身体。"})},
      {name: translate({message: "站稳收步"}), start: .9, sample: 1, detail: translate({message: "到达平台后收步，四足重新建立稳定支撑。"})},
    ],
  },
  navigation: {
    title: translate({message: "绕开障碍，到达目标"}),
    scene: translate({message: "轮式机器人从起点出发，沿规划路径绕过两个障碍物，到达终点后停下。"}),
    focus: translate({message: "看目标与路线：决定去哪里、如何绕行、何时停止。"}),
    phases: [
      {name: translate({message: "定位与目标"}), start: 0, sample: .08, detail: translate({message: "确定当前位置、目标位置，以及场地中障碍物的位置。"})},
      {name: translate({message: "规划绕行"}), start: .22, sample: .3, detail: translate({message: "规划绕开障碍的路线，同时为机器人身体留出通行空间。"})},
      {name: translate({message: "跟踪路径"}), start: .38, sample: .68, detail: translate({message: "根据路径输出移动与转向指令，由底盘控制执行。"})},
      {name: translate({message: "到达停止"}), start: .94, sample: 1, detail: translate({message: "进入目标区域后停止移动，并确认到达。"})},
    ],
  },
} satisfies Record<Skill, {title: string; scene: string; focus: string; phases: {name: string; start: number; sample: number; detail: string}[]}>;
