import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import { Bot, BrainCircuit, BriefcaseBusiness, Map, type LucideIcon } from 'lucide-react';
import styles from './styles.module.css';

type FeatureItem = {
  title: string;
  description: string;
  link: string;
  eyebrow: string;
  icon: LucideIcon;
  accent: 'cyan' | 'green' | 'amber' | 'blue';
};

const FeatureList: FeatureItem[] = [
  {
    title: '总览',
    description: '项目定位、学习路径、公司图谱和不同背景的切入方式。',
    link: '/docs/overview/intro',
    eyebrow: 'Start here',
    icon: Map,
    accent: 'cyan',
  },
  {
    title: '基础篇',
    description: '机器人学、ROS2、仿真、强化学习、VLM/VLA 等基础模块。',
    link: '/docs/foundations/intro',
    eyebrow: 'Core stack',
    icon: BrainCircuit,
    accent: 'blue',
  },
  {
    title: '实践篇',
    description: '机械臂、四足、人形、移动操作四类可展示项目。',
    link: '/docs/practices/intro',
    eyebrow: 'Build labs',
    icon: Bot,
    accent: 'green',
  },
  {
    title: '求职篇',
    description: '岗位拆解、面经、简历表达和公司技术栈。',
    link: '/docs/career/intro',
    eyebrow: 'Career loop',
    icon: BriefcaseBusiness,
    accent: 'amber',
  },
];

function Feature({ title, description, link, eyebrow, icon: Icon, accent }: FeatureItem) {
  return (
    <div className={clsx('col col--3', styles.featureCol)}>
      <Link to={link} className={clsx(styles.featureCard, styles[`featureCard--${accent}`])}>
        <div className={styles.featureTopline}>
          <span>{eyebrow}</span>
          <Icon size={20} aria-hidden="true" />
        </div>
        <div>
          <h3>{title}</h3>
          <p>{description}</p>
        </div>
        <span className={styles.featureLink}>进入模块</span>
      </Link>
    </div>
  );
}

export default function HomepageFeatures(): React.JSX.Element {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className={styles.sectionHeading}>
          <p>LEARNING MAP</p>
          <h2>四条路线，围绕开源内容持续生长</h2>
        </div>
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
