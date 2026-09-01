import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import { Bot, BrainCircuit, type LucideIcon } from 'lucide-react';
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
    title: '理论基础',
    description: '按智能决策、运动控制、感知系统和工程底座补齐能力。',
    link: '/docs/foundations/intro',
    eyebrow: 'Skill tree',
    icon: BrainCircuit,
    accent: 'blue',
  },
  {
    title: '项目实战',
    description: '按 AMD 专区、仿真实战和真机实战进入可复现项目。',
    link: '/docs/practices/intro',
    eyebrow: 'Build labs',
    icon: Bot,
    accent: 'green',
  },
];

function Feature({ title, description, link, eyebrow, icon: Icon, accent }: FeatureItem) {
  return (
    <div className={clsx('col col--6', styles.featureCol)}>
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
          <h2>两大模块，从理论基础走向项目实战</h2>
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
