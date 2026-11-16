import React, {useEffect, useState} from 'react';
import Link from '@docusaurus/Link';
import useBaseUrl from '@docusaurus/useBaseUrl';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import {ArrowDown, ArrowRight, ArrowUpRight, PawPrint, Footprints, Bot, Cpu, Brain, ScanEye, Activity, Layers} from 'lucide-react';
import {foundationGroups, practiceGroups, type Course, type PracticeGroup, type FoundationGroup} from './data';
import {hubText} from './messages';
import ParticleScene from './ParticleScene';
import ModuleDiagram from './ModuleDiagram';
import CategoryMotion from './CategoryMotion';
import styles from './styles.module.css';

const icons = {quadruped: PawPrint, humanoid: Footprints, 'robot-arm': Bot, amd: Cpu, decision: Brain, control: Activity, perception: ScanEye, engineering: Layers};

function CourseList({courses}: {courses: Course[]}) {
  return <ul className={styles.courseList}>{courses.map(course => <li key={course.to}>
    <Link className={styles.courseLink} to={course.to}>
      <div><h3>{course.title}</h3>{course.description && <p>{course.description}</p>}</div>
      <ArrowUpRight size={19} aria-hidden="true" />
    </Link>
    {course.children && <div className={styles.subcourses}>{course.children.map(child => <Link key={child.to} to={child.to}>{child.title}<ArrowUpRight size={13} aria-hidden="true" /></Link>)}</div>}
  </li>)}</ul>;
}

function PracticeSection({group}: {group: PracticeGroup}) {
  const text = hubText();
  const [feature, ...courses] = group.courses;
  return <section id={group.id} className={styles.practiceSection} aria-labelledby={`${group.id}-title`} data-hub-section>
    <header className={styles.sectionHeader}>
      <div><p className={styles.eyebrow}>{group.title}</p><h2 id={`${group.id}-title`}>{group.lead}</h2><p className={styles.sectionDescription}>{group.description}</p></div>
      <Link className={styles.textLink} to={group.overview}>{text.allSeries}<ArrowRight size={17} aria-hidden="true" /></Link>
    </header>
    <div className={styles.seriesContent}>
      <Link className={styles.feature} to={feature.to}>
        <div className={styles.featureImage}><img src={useBaseUrl(group.image)} alt={group.alt} loading="lazy" width="720" height="540" /></div>
        <div className={styles.featureBody}><span className={styles.tags}>{feature.tags}</span><h3>{feature.title}</h3><p>{feature.description}</p><span className={styles.textLink}>{text.startCourse}<ArrowUpRight size={17} aria-hidden="true" /></span></div>
      </Link>
      <CourseList courses={courses} />
    </div>
  </section>;
}

function FoundationSection({group}: {group: FoundationGroup}) {
  const text = hubText();
  const Icon = icons[group.id];
  return <section id={group.id} className={styles.module} aria-labelledby={`${group.id}-title`} data-hub-section>
    <div className={styles.moduleHeader}><Icon size={20} aria-hidden="true" /><p>{group.title}</p></div>
    <ModuleDiagram kind={group.id} />
    <h2 id={`${group.id}-title`}>{group.lead}</h2><p className={styles.sectionDescription}>{group.description}</p>
    <CourseList courses={group.courses} />
    <Link className={styles.textLink} to={group.project}>{text.apply}<ArrowRight size={17} aria-hidden="true" /></Link>
  </section>;
}

export default function LearningHub({kind}: {kind: 'practices' | 'foundations'}) {
  const text = hubText();
  const isPractice = kind === 'practices';
  const groups = isPractice ? practiceGroups() : foundationGroups();
  const {i18n} = useDocusaurusContext();
  const [active, setActive] = useState('');
  const [paused, setPaused] = useState(false);
  const title = isPractice ? text.practices : text.foundations;
  const description = isPractice ? text.practiceDescription : text.foundationDescription;

  useEffect(() => {
    const sections = Array.from(document.querySelectorAll<HTMLElement>('[data-hub-section]'));
    const update = () => {
      const threshold = window.innerWidth < 768 ? 100 : 210;
      const eligible = sections.map(section => ({id: section.id, top: section.getBoundingClientRect().top})).filter(section => section.top <= threshold);
      const nearestTop = Math.max(...eligible.map(section => section.top));
      const row = eligible.filter(section => Math.abs(section.top - nearestTop) < 2);
      // Two knowledge modules share a row on desktop. Keep the selected module
      // highlighted rather than always selecting the last DOM sibling.
      setActive(previous => row.some(section => section.id === previous) ? previous : row[0]?.id ?? '');
    };
    let frame = 0;
    const onScroll = () => {if (!frame) frame = requestAnimationFrame(() => {frame = 0; update();});};
    const hash = window.location.hash.slice(1);
    if (sections.some(section => section.id === hash)) setActive(hash);
    update();
    window.addEventListener('scroll', onScroll, {passive: true});
    window.addEventListener('resize', onScroll);
    return () => {cancelAnimationFrame(frame); window.removeEventListener('scroll', onScroll); window.removeEventListener('resize', onScroll);};
  }, [kind]);

  return <Layout title={title} description={description}>
    <main className={styles.hub} data-learning-hub={kind} data-motion-paused={paused}>
      <nav className={styles.seriesNav} aria-label={isPractice ? text.quickNavigation : text.modulesNavigation}>
        <div className={styles.navInner}><div className={styles.navLinks}>
          {groups.map(group => <a href={`#${group.id}`} key={group.id} onClick={() => setActive(group.id)} aria-current={active === group.id ? 'location' : undefined}>
            <CategoryMotion kind={group.id} /><span>{group.title}</span>
          </a>)}
        </div></div>
      </nav>
      <header className={styles.hero}>
        <div className={styles.heroCopy}><p className={styles.eyebrow}>{title}</p><h1>{isPractice ? text.practiceTitle : text.foundationTitle}</h1><p className={styles.heroDescription}>{description}</p>
          <div className={styles.heroActions}><a className="site-button site-button--primary" href={`#${groups[0].id}`}>{isPractice ? text.explore : text.exploreTheory}<ArrowDown size={16} aria-hidden="true" /></a>
            <Link className={styles.textLink} to={isPractice ? '/simulation' : '/practices'}>{isPractice ? text.simulation : text.viewPractices}<ArrowUpRight size={16} aria-hidden="true" /></Link>
          </div>
        </div><ParticleScene kind={kind} paused={paused} onToggle={() => setPaused(value => !value)} />
      </header>
      <div className={styles.content}>
        {!isPractice && <div className={styles.loop} role="group" aria-label={text.loop}>
          {[text.observation, text.policy, text.action, text.feedback].map((label, index) => <React.Fragment key={label}>
            {index > 0 && <ArrowRight size={18} aria-hidden="true" />}<span>{label}</span>
          </React.Fragment>)}
        </div>}
        <div className={isPractice ? styles.practiceGroups : styles.moduleGrid}>
          {isPractice ? (groups as PracticeGroup[]).map(group => <PracticeSection key={group.id} group={group} />)
            : (groups as FoundationGroup[]).map(group => <FoundationSection key={group.id} group={group} />)}
        </div>
        {isPractice && <aside className={styles.planning}><div><h2>{text.planning}</h2><p>{text.planningDescription}</p></div><div>
          <Link to="/docs/practices/mobile-manipulation/placeholder">{text.mobile}<ArrowUpRight size={16} aria-hidden="true" /></Link>
          <Link to="/docs/practices/quadruped/sim2real-guide/placeholder">{text.sim2real}<ArrowUpRight size={16} aria-hidden="true" /></Link>
        </div></aside>}
        <section className={styles.bridge}>
          <h2>{text.bridgeTitle}</h2><p>{isPractice ? text.bridgePractice : text.bridgeTheory}</p>
          <Link className="site-button site-button--primary" to={isPractice ? '/foundations' : '/practices'}>{isPractice ? text.viewFoundations : text.viewPractices}<ArrowRight size={16} aria-hidden="true" /></Link>
        </section>
        {i18n.currentLocale !== i18n.defaultLocale && <p className={styles.scope}>{text.scope}</p>}
      </div>
    </main>
  </Layout>;
}
