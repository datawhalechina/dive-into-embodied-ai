import React from 'react';
import Link from '@docusaurus/Link';
import {ArrowRight, BookOpen, FlaskConical} from 'lucide-react';
import styles from './styles.module.css';

const tracks = [
  {title: '理论基础', lead: '先理解，机器人如何思考。', description: '从智能决策、运动控制与感知，建立具身智能的知识框架。', to: '/docs/foundations/intro', label: '查看学习路线', icon: BookOpen, topics: ['机器人学与 ROS2', '强化学习', '视觉语言动作模型']},
  {title: '项目实战', lead: '再动手，让算法走进现实。', description: '跟着完整项目，在仿真和真机上跑通训练、控制与部署。', to: '/docs/practices/intro', label: '选择实践项目', icon: FlaskConical, topics: ['MuJoCo 仿真', '机器人控制', 'AMD 实践专区']},
];

export default function HomepageFeatures(): React.JSX.Element {
  return (
    <section className={styles.features} aria-labelledby="learning-title">
      <div className="home-container">
        <div className="home-section-heading">
          <div><p className="section-kicker">你的学习路线</p><h2 id="learning-title">理解原理，也亲手实现。</h2></div>
          <p>刚入门，从理论开始。已有基础，直接动手做项目。</p>
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
