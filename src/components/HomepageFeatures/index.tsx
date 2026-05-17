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
    title: '零基础入门',
    description: '学习路径、方向认知和从0到1搭建四足机器人的入门项目。',
    link: '/docs/overview/learning-path',
    eyebrow: 'Start here',
    icon: Map,
    accent: 'cyan',
  },
  {
    title: '项目实战',
    description: '机械臂、四足、人形、移动操作四类可展示项目。',
    link: '/docs/practices/intro',
    eyebrow: 'Build labs',
    icon: Bot,
    accent: 'green',
  },
  {
    title: '求职面试',
    description: '岗位技能拆解、面经八股和招聘信息。',
    link: '/docs/career/intro',
    eyebrow: 'Career loop',
    icon: BriefcaseBusiness,
    accent: 'amber',
  },
  {
    title: '理论技能树',
    description: '按智能决策、运动控制、感知系统和工程底座补齐能力。',
    link: '/docs/foundations/intro',
    eyebrow: 'Skill tree',
    icon: BrainCircuit,
    accent: 'blue',
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
          <h2>四个阶段，围绕入门、项目、求职和理论补齐持续生长</h2>
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
