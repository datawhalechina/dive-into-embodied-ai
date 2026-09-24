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
  // Methods, examples, and limitations live in the introduction chapter.
  introduction: {label: string; to: string};
};

export const robotSkills: RobotSkillContent[] = [
  {
    skill: 'grasping', id: 'ability-grasping', name: translate({message: "抓取"}), english: 'Grasping', question: translate({message: "怎样把一个物体拿稳？"}),
    paragraphs: [
      <><strong>{translate({message: "抓取（Grasping）"})}</strong>{translate({message: "是末端执行器与物体建立一组接触，使物体的运动受到约束、能够随末端一起运动，并在预期扰动下不滑脱的过程及其结果。"})}</>,
      <>{translate({message: "通俗来讲，就是让机器人用夹爪、灵巧手或吸盘把物体拿稳，为后续移动和操作做好准备。例如，把桌面上的小方块放进指定盒子时，需要先夹住并提起它；其中，建立并保持稳定夹持是抓取，后续还要完成搬运、放置和松开，才构成完整的拾取与放置操作。"})}</>,
    ],
    introduction: {label: translate({message: "深入了解抓取"}), to: '/docs/introduction/tasks-and-skills#grasping'},
  },
  {
    skill: 'manipulation', id: 'ability-manipulation', name: translate({message: "操作"}), english: 'Manipulation', question: translate({message: "怎样通过接触改变物体状态？"}),
    paragraphs: [
      <><strong>{translate({message: "操作（Manipulation）"})}</strong>{translate({message: "是机器人通过施加力或运动，有目的地改变物体的位置、姿态、形状或其他任务相关状态的过程。"})}</>,
      <>{translate({message: "通俗来讲，就是让机器人通过推、拉、搬、转等动作，把物体变成任务要求的状态。例如，机器人先抓住抽屉把手，再沿导轨向外拉，直到抽屉打开到指定位置。"})}</>,
    ],
    introduction: {label: translate({message: "深入了解操作"}), to: '/docs/introduction/tasks-and-skills#manipulation'},
  },
  {
    skill: 'locomotion', id: 'ability-locomotion', name: translate({message: "运动"}), english: 'Locomotion', question: translate({message: "怎样让身体稳定地移动？"}),
    paragraphs: [
      <><strong>{translate({message: "运动（Locomotion）"})}</strong>{translate({message: "是机器人通过腿、轮等机构与环境的相互作用，实现自身移动，并协调身体姿态与运动稳定性的过程。"})}</>,
      <>{translate({message: "通俗来讲，就是解决机器人“身体怎么动起来、怎样动得稳”。例如，四足机器人迈上台阶时，需要协调抬腿、落脚和重心转移，避免绊倒或失去平衡。"})}{translate({message: "要让它进一步自主选择路线并到达门口，还需要与"})}<Link href="#ability-navigation">{translate({message: "导航"})}</Link>{translate({message: "配合。"})}</>,
    ],
    introduction: {label: translate({message: "深入了解运动"}), to: '/docs/introduction/tasks-and-skills#locomotion'},
  },
  {
    skill: 'navigation', id: 'ability-navigation', name: translate({message: "导航"}), english: 'Navigation', question: translate({message: "去哪里、走哪条路、何时停下？"}),
    paragraphs: [
      <><strong>{translate({message: "导航（Navigation）"})}</strong>{translate({message: "是机器人依据目标、环境观测和自身状态，决定并调整移动方向或路径，避开障碍并到达目标位置或区域的过程。"})}</>,
      <>{translate({message: "通俗来讲，就是解决“我在哪里、要去哪里、应该怎么走”。例如，机器人从房间一端出发，绕过地上的障碍物，到达门口的指定区域。"})}</>,
    ],
    introduction: {label: translate({message: "深入了解导航"}), to: '/docs/introduction/tasks-and-skills#navigation'},
  },
];

export const robotSkillAnchors = robotSkills.map(({id}) => id);
