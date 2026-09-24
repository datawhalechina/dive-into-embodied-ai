import {translate} from '@docusaurus/Translate';
import React from 'react';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Link from '@docusaurus/Link';
import {ArrowRight} from 'lucide-react';
import SkillDemo from '../demos/skills';
import {robotSkills} from './content';
import styles from './styles.module.css';

export default function RobotSkills({className}: {className?: string}) {
  const isEnglish = useDocusaurusContext().i18n.currentLocale === 'en';
  return (
    <section id="robot-interaction" className={className} aria-labelledby="interaction-title">
      <header className={styles.heading}>
        <p className={styles.kicker}>{translate({message: "先定义任务，再认识技能"})}</p>
        <h2 id="interaction-title">{translate({message: "机器人的技能"})}</h2>
      </header>

      <div className={styles.introduction}>
        <p>{translate({message: "想要学习并研究具身智能，需要先把任务定义清楚。"})}<strong>{translate({message: "如何合理地建模任务、确定目标与评价指标，往往比模型选择与优化更为关键。"})}</strong></p>
        <p>{translate({message: "对机器人技能的研究，可以先从四类常见任务入手："})}<strong>{translate({message: "抓取（Grasping）、操作（Manipulation）、运动（Locomotion）和导航（Navigation）"})}</strong>{translate({message: "。抓取与操作关注改变物体的状态，运动与导航关注移动机器人自身。"})}</p>
      </div>

      <div className={styles.abilities}>
        {robotSkills.map(({skill, id, name, english, question, paragraphs, introduction}) => (
          <article id={id} key={id} className={styles.ability} aria-labelledby={`${id}-title`}>
            <header className={styles.abilityHeading}>
              {!isEnglish && <p>{english}</p>}
              <h3 id={`${id}-title`}>{name}</h3>
              <span>{question}</span>
            </header>
            <div className={styles.explanation}>
              {paragraphs.map((paragraph, index) => <p key={index}>{paragraph}</p>)}
            </div>
            <SkillDemo skill={skill} />
            <Link to={introduction.to} className={`site-text-link ${styles.readMore}`}>
              {introduction.label}<ArrowRight size={16} aria-hidden="true" />
            </Link>
          </article>
        ))}
      </div>
    </section>
  );
}
