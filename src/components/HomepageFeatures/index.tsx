import {translate} from '@docusaurus/Translate';
import React from 'react';
import Link from '@docusaurus/Link';
import {ArrowRight, BookOpen, FlaskConical, GraduationCap, Telescope} from 'lucide-react';
import styles from './styles.module.css';

const tracks = [
  {title: translate({message: "具身导论"}), lead: translate({message: "看清全局，认识具身智能。"}), description: translate({message: "了解具身智能的概念、发展、任务与载体，查找论文、数据集与开源工具。"}), to: '/docs/introduction/intro', label: translate({message: "阅读具身导论"}), icon: Telescope, topics: [translate({message: "什么是具身智能"}), translate({message: "任务与载体"}), translate({message: "论文与数据集"})]},
  {title: translate({message: "理论基础"}), lead: translate({message: "先理解，机器人如何思考。"}), description: translate({message: "从智能决策、运动控制与感知，建立具身智能的知识框架。"}), to: '/docs/foundations/intro', label: translate({message: "查看学习路线"}), icon: BookOpen, topics: [translate({message: "机器人学与 ROS2"}), translate({message: "强化学习"}), translate({message: "视觉语言动作模型"})]},
  {title: translate({message: "系列教程"}), lead: translate({message: "沿着主线，搭建完整系统。"}), description: translate({message: "跟随有章节顺序的课程，从基础准备逐步走到系统实现。"}), to: '/docs/tutorials/intro', label: translate({message: "选择系列教程"}), icon: GraduationCap, topics: [translate({message: "从零到一搭建四足机器人"}), translate({message: "LeRobot 中文课程"})]},
  {title: translate({message: "项目实战"}), lead: translate({message: "动手实验，让算法跑起来。"}), description: translate({message: "选择一个独立项目或 Demo，在仿真和真机中验证具体方法。"}), to: '/docs/practices/intro', label: translate({message: "选择实践项目"}), icon: FlaskConical, topics: [translate({message: "MicroDuck 小黄鸭"}), translate({message: "ACT 双臂操作"}), translate({message: "AMD 实践专区"})]},
];

export default function HomepageFeatures(): React.JSX.Element {
  return (
    <section className={styles.features} aria-labelledby="learning-title">
      <div className="home-container">
        <div className="home-section-heading">
          <div><p className="section-kicker">{translate({message: "学习与探索"})}</p><h2 id="learning-title">{translate({message: "从理解原理，到动手探索。"})}</h2></div>
          <p>{translate({message: "先看全局，再补理论、跟课程、做项目。"})}</p>
        </div>
        <div className={styles.grid}>
          {tracks.map(({title, lead, description, to, label, icon: Icon, topics}) => (
            <Link className={styles.card} to={to} key={to}>
              <div className={styles.topline}><span><Icon size={20} strokeWidth={1.7} aria-hidden="true" />{title}</span><ArrowRight size={20} aria-hidden="true" /></div>
              <h3>{lead}</h3><p>{description}</p>
              <ul className={styles.topics}>{topics.map(topic => <li key={topic}>{topic}</li>)}</ul>
              <span className="site-text-link">{label} <ArrowRight size={16} aria-hidden="true" /></span>
            </Link>
          ))}
        </div>
      </div>
    </section>
  );
}
