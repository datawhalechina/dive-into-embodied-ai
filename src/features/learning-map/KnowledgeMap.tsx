import React, {useState} from 'react';
import Link from '@docusaurus/Link';
import {ArrowRight, ArrowUpRight, Check, RotateCcw} from 'lucide-react';
import {capabilities, foundations, scenarios} from './content';
import styles from './styles.module.css';

function TopicLinks({links}: {links: {label: string; to: string}[]}) {
  return (
    <ul className={styles.topicLinks}>
      {links.map(({label, to}) => (
        <li key={to}><Link to={to}>{label}<ArrowUpRight size={15} aria-hidden="true" /></Link></li>
      ))}
    </ul>
  );
}

export default function KnowledgeMap() {
  const [scenarioIndex, setScenarioIndex] = useState(0);
  const scenario = scenarios[scenarioIndex];

  return (
    <section id="knowledge-map" className={styles.section} aria-labelledby="map-title">
      <div className={styles.sectionHeading}>
        <div><p className={styles.kicker}>知识全景</p><h2 id="map-title">从一个任务，看懂整个系统。</h2></div>
        <p>上层串起行动闭环，下层支撑开发与验证。<br />选择知识点，就能进入对应教程。</p>
      </div>
      <div className={styles.map}>
        <div className={styles.scenarioBar}>
          <span id="scenario-label">换个任务理解</span>
          <div className={styles.scenarioOptions} role="group" aria-labelledby="scenario-label">
            {scenarios.map(({name}, index) => (
              <button key={name} type="button" aria-pressed={scenarioIndex === index}
                aria-controls="task-example" onClick={() => setScenarioIndex(index)}>
                {scenarioIndex === index && <Check size={14} aria-hidden="true" />}{name}
              </button>
            ))}
          </div>
        </div>
        <div id="task-example" className={styles.taskDefinition}>
          <p><span>任务目标</span>{scenario.goal}</p>
          <p><span>怎样算完成</span>{scenario.success}</p>
        </div>
        <p className={styles.srOnly} role="status">
          {scenario.name}。{scenario.goal}{scenario.stages.join('')}{scenario.feedback}完成标准：{scenario.success}
        </p>
        <div className={styles.loop}>
          {capabilities.map(({title, question, icon: Icon, concepts, links}, index) => (
            <article className={styles.capability} key={title}>
              <div className={styles.capabilityHeading}><Icon size={23} strokeWidth={1.7} aria-hidden="true" /><h3>{title}</h3></div>
              <p className={styles.question}>{question}</p>
              <p className={styles.example}>{scenario.stages[index]}</p>
              <p className={styles.concepts}>{concepts}</p>
              <TopicLinks links={links} />
              {index < capabilities.length - 1 && <ArrowRight className={styles.flowArrow} size={21} aria-hidden="true" />}
            </article>
          ))}
        </div>
        <div className={styles.feedback}><RotateCcw size={18} aria-hidden="true" /><p><strong>行动改变环境，新的观测再次进入闭环。</strong><span>{scenario.feedback}</span></p></div>
        <p className={styles.modelNote}>这是按功能理解系统的方式；实际系统可以用多个模块协作，也可以用一个模型承担多种功能。</p>
        <div className={styles.foundationLabel}><span>支撑整个闭环</span></div>
        <div className={styles.foundations}>
          {foundations.map(({title, icon: Icon, description, concepts, links}) => (
            <article key={title}>
              <div className={styles.capabilityHeading}><Icon size={21} strokeWidth={1.7} aria-hidden="true" /><h3>{title}</h3></div>
              <p>{description}</p><p className={styles.concepts}>{concepts}</p>
              <TopicLinks links={links} />
            </article>
          ))}
        </div>
      </div>
      <div className={styles.methodNote}>
        <strong>算法在地图里的位置</strong>
        <p>模仿学习（IL）从示范中学动作，强化学习（RL）用奖励改进策略；VLA 把视觉与语言连接到动作，世界模型预测动作的后果。控制和规划同样可以独立解决任务，也可以与学习方法组合。</p>
      </div>
    </section>
  );
}
