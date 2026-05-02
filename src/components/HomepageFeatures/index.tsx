import React from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  description: string;
  link: string;
  emoji: string;
};

const FeatureList: FeatureItem[] = [
  {
    title: '总览',
    description: '项目介绍、学习路径和公司图谱',
    link: '/docs/overview/intro',
    emoji: '\uD83D\uDDFA\uFE0F',
  },
  {
    title: '基础篇',
    description: '机器人学、ROS2、仿真、强化学习、VLM/VLA 等基础模块',
    link: '/docs/foundations/intro',
    emoji: '\uD83D\uDCDA',
  },
  {
    title: '实践篇',
    description: '机械臂、四足、人形、移动操作四类项目',
    link: '/docs/practices/intro',
    emoji: '\uD83D\uDD27',
  },
  {
    title: '求职篇',
    description: '岗位拆解、面经、简历和公司技术栈',
    link: '/docs/career/intro',
    emoji: '\uD83C\uDFAF',
  },
];

function Feature({ title, description, link, emoji }: FeatureItem) {
  return (
    <div className={clsx('col col--3')}>
      <a href={link} className="feature-card" style={{ textDecoration: 'none', color: 'inherit', display: 'block' }}>
        <div className="text--center" style={{ fontSize: '3rem' }}>
          {emoji}
        </div>
        <div className="text--center padding-horiz--md">
          <h3>{title}</h3>
          <p>{description}</p>
        </div>
      </a>
    </div>
  );
}

export default function HomepageFeatures(): React.JSX.Element {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
