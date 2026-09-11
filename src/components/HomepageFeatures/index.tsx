import React from 'react';
import Link from '@docusaurus/Link';
import {ArrowRight, BookOpen, FlaskConical, GraduationCap, Library} from 'lucide-react';
import styles from './styles.module.css';

const tracks = [
  {title: '理论基础', lead: '先理解，机器人如何思考。', description: '从智能决策、运动控制与感知，建立具身智能的知识框架。', to: '/docs/foundations/intro', label: '查看学习路线', icon: BookOpen, topics: ['机器人学与 ROS2', '强化学习', '视觉语言动作模型']},
  {title: '系列教程', lead: '沿着主线，搭建完整系统。', description: '跟随有章节顺序的课程，从基础准备逐步走到系统实现。', to: '/docs/tutorials/intro', label: '选择系列教程', icon: GraduationCap, topics: ['从零到一搭建四足机器人', 'LeRobot 中文课程']},
  {title: '项目实战', lead: '动手实验，让算法跑起来。', description: '选择一个独立项目或 Demo，在仿真和真机中验证具体方法。', to: '/docs/practices/intro', label: '选择实践项目', icon: FlaskConical, topics: ['MicroDuck 小黄鸭', 'ACT 双臂操作', 'AMD 实践专区']},
  {title: '具身信息', lead: '找到研究，也找到资源。', description: '按主题整理具身智能论文、数据集、开源项目与开发工具。', to: '/docs/information/intro', label: '浏览具身信息', icon: Library, topics: ['论文与研究', '具身数据集', '开源与工具']},
];

export default function HomepageFeatures(): React.JSX.Element {
  return (
    <section className={styles.features} aria-labelledby="learning-title">
      <div className="home-container">
        <div className="home-section-heading">
          <div><p className="section-kicker">学习与探索</p><h2 id="learning-title">从理解原理，到动手探索。</h2></div>
          <p>补理论、跟课程、做项目，或寻找下一篇论文和数据集。</p>
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
