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
        <p>{translate({message: "在选择方法之前，先明确：希望机器人学会什么技能？在什么环境和约束下完成任务？使用哪些传感器，输入的观测与输出的动作分别是什么？如何判断任务完成，并用成功率、完成时间等指标评价表现？"})}</p>
        <p>{translate({message: "同样需要审视任务设定是否合理，以及能否通过调整设定改善问题。例如，当顶部相机对环境的观测不够充分时，可以考虑加装腕部相机，补充近距离视角。传感器配置的改变也意味着观测条件发生了变化，实验中应明确记录，并在一致的设定下比较方法。"})}</p>
        <p>{translate({message: "对机器人技能的研究，可以先从四类常见任务入手："})}<strong>{translate({message: "抓取（Grasping）、操作（Manipulation）、运动（Locomotion）和导航（Navigation）"})}</strong>{translate({message: "。抓取与操作关注改变物体的状态，运动与导航关注移动机器人自身。"})}</p>
      </div>

      <div className={styles.groups}>
        <div className={styles.group}>
          <h3>{translate({message: "改变物体"})}</h3>
          <div className={styles.relationship}>
            <a href="#ability-grasping"><strong>Grasping</strong>{!isEnglish && <span>{translate({message: "抓取"})}</span>}</a>
            <span className={styles.connector}>{translate({message: "属于"})}<ArrowRight size={18} aria-hidden="true" /></span>
            <a href="#ability-manipulation"><strong>Manipulation</strong>{!isEnglish && <span>{translate({message: "操作"})}</span>}</a>
          </div>
          <p>{translate({message: "把杯子拿稳是抓取；搬运、推移或倾倒杯子，都属于操作。"})}</p>
        </div>
        <div className={styles.group}>
          <h3>{translate({message: "移动自身"})}</h3>
          <div className={styles.relationship}>
            <a href="#ability-navigation"><strong>Navigation</strong>{!isEnglish && <span>{translate({message: "导航"})}</span>}</a>
            <span className={styles.connector}>{translate({message: "调用"})}<ArrowRight size={18} aria-hidden="true" /></span>
            <a href="#ability-locomotion"><strong>Locomotion</strong>{!isEnglish && <span>{translate({message: "运动"})}</span>}</a>
          </div>
          <p>{translate({message: "导航决定去哪里、走哪条路；运动负责把这一步走出来。"})}</p>
        </div>
      </div>
      <p className={styles.bodyNote}>{translate({message: "可以借助“手和臂操作、腿或轮带动身体”建立直觉。分类依据是任务：脚也能踢球或推物体，轮式平台也能搭载机械臂完成操作。"})}</p>

      <div className={styles.abilities}>
        {robotSkills.map(({skill, id, name, english, question, paragraphs}) => (
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
          </article>
        ))}
      </div>

      <div className={styles.comparison}>
        <h3 id="interaction-comparison-title">{translate({message: "放在一起看"})}</h3>
        <p id="interaction-frequency-note">{translate({message: "频率描述的是某一层的更新速度。下面列出具体系统的示例；策略推理、动作执行、仿真步进与关节伺服频率应分别核对。"})}</p>
        <div className={styles.tableScroll} role="region" aria-labelledby="interaction-comparison-title" aria-describedby="interaction-frequency-note" tabIndex={0}>
          <table className={styles.table}>
            <caption className={styles.srOnly}>{translate({message: "四类机器人技能的输入、输出、时间尺度、方法与适用边界"})}</caption>
            <thead><tr>{[translate({message: "技能"}), translate({message: "典型输入"}), translate({message: "典型输出"}), translate({message: "时间尺度示例"}), translate({message: "常见方法"}), translate({message: "进展与边界"})].map(label => <th key={label} scope="col">{label}</th>)}</tr></thead>
            <tbody>
              {robotSkills.map(({id, name, english, input, output, frequency, methods, progress}) => (
                <tr key={id}>
                  <th scope="row"><a href={`#${id}`}>{name}</a>{!isEnglish && <span>{english}</span>}</th>
                  <td>{input}</td><td>{output}</td><td>{frequency}</td><td>{methods}</td><td>{progress}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className={styles.tableHint}>{translate({message: "窄屏可左右滚动查看完整表格。"})}</p>
      </div>

      <div className={styles.combinations}>
        <h3>{translate({message: "组合起来，完成一个真实任务"})}</h3>
        <dl>
          <div><dt>Mobile manipulation{!isEnglish && <span>{translate({message: "移动操作"})}</span>}</dt><dd>{translate({message: "把移动与操作连接起来，通常需要导航找到工作位置，再协调底盘和机械臂完成任务。例如“去厨房，把杯子拿回来”。"})}</dd></div>
          <div><dt>Loco-manipulation{!isEnglish && <span>{translate({message: "运动与操作协同"})}</span>}</dt><dd>{translate({message: "强调移动、平衡与接触操作之间的耦合，例如四足带臂搬运物体，或人形机器人一边保持平衡一边开门。"})}<Link href="https://rsl.ethz.ch/robots-media/alma.html">ALMA</Link> {translate({message: "是足式全身操作的一个例子。"})}</dd></div>
        </dl>
        <p>{translate({message: "人形机器人执行日常任务时，可能同时涉及这四类能力；具体能做到什么，取决于硬件、感知、控制与训练，而不只取决于外形。"})}</p>
      </div>

      <div className={styles.hierarchy}>
        <h3>{translate({message: "为什么系统常采用分层协作？"})}</h3>
        <p>{translate({message: "这些任务的观测类型、动作空间和反馈时限不同。导航或任务规划给出目标，操作或步态策略生成动作，底层控制器跟踪关节与接触状态；新的反馈再回到上层。"})}</p>
        <ol className={styles.controlLayers} aria-label={translate({message: "一种常见的控制分工，反馈会逐层返回"})}>
          <li><strong>{translate({message: "导航 / 任务规划"})}</strong><span>{translate({message: "决定去哪里、做什么"})}</span></li>
          <li><strong>{translate({message: "操作 / 运动策略"})}</strong><span>{translate({message: "生成可执行的动作"})}</span></li>
          <li><strong>{translate({message: "关节 / 全身控制"})}</strong><span>{translate({message: "跟踪目标，协调力与平衡"})}</span></li>
        </ol>
        <p>{translate({message: "VLA 可以连接视觉、语言与操作动作，RL 或 MPC 可以承担运动控制，全身控制器（WBC）可以协调臂、腿与身体。系统也可以学习统一的全身策略。理解一种方法时，先看它负责哪一层、接收什么观测、输出什么动作，以及由谁处理执行误差。"})}</p>
      </div>
    </section>
  );
}
