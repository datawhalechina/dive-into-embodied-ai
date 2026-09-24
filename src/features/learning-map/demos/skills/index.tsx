import React, {useEffect, useId, useRef, useState} from 'react';
import {ArrowRight, Pause, Play, RotateCcw} from 'lucide-react';
import {skillDemos} from './content';
import type {Skill} from '../../skills/types';
import type {SkillScene} from './scene';
import styles from './styles.module.css';

const DURATION = 8000;

export default function SkillDemo({skill}: {skill: Skill}) {
  const demo = skillDemos[skill];
  const id = useId().replace(/:/g, '');
  const figure = useRef<HTMLElement>(null);
  const host = useRef<HTMLDivElement>(null);
  const scene = useRef<SkillScene | null>(null);
  const progressRef = useRef(0);
  const [progress, setProgress] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [rate, setRate] = useState(1);
  const [nearby, setNearby] = useState(false);
  const [visible, setVisible] = useState(false);
  const [reducedMotion, setReducedMotion] = useState(false);
  const [sceneState, setSceneState] = useState<'loading' | 'ready' | 'unavailable'>('loading');
  const stage = demo.phases.reduce((index, phase, i) => progress >= phase.start ? i : index, 0);
  const complete = progress >= 1;

  useEffect(() => {
    const preference = window.matchMedia('(prefers-reduced-motion: reduce)');
    const update = () => {setReducedMotion(preference.matches); if (preference.matches) setPlaying(false);};
    update(); preference.addEventListener('change', update);
    const pause = () => {if (document.hidden) setPlaying(false);};
    document.addEventListener('visibilitychange', pause);
    const preload = new IntersectionObserver(([entry]) => setNearby(entry.isIntersecting), {rootMargin: '300px'});
    const visibility = new IntersectionObserver(([entry]) => {
      setVisible(entry.isIntersecting);
      if (!entry.isIntersecting) setPlaying(false);
    });
    if (figure.current) {preload.observe(figure.current); visibility.observe(figure.current);}
    return () => {preload.disconnect(); visibility.disconnect(); preference.removeEventListener('change', update); document.removeEventListener('visibilitychange', pause);};
  }, []);

  useEffect(() => {
    if (!nearby) return;
    let cancelled = false;
    setSceneState('loading');
    // Only nearby demos own a WebGL context; long pages remain inexpensive to browse.
    import('./scene').then(({createSkillScene}) => {
      if (cancelled || !host.current) return;
      try {
        scene.current = createSkillScene(host.current, skill, () => {setSceneState('unavailable'); setPlaying(false);});
        scene.current.update(progressRef.current);
        setSceneState('ready');
      } catch {setSceneState('unavailable'); setPlaying(false);}
    }).catch(() => {if (!cancelled) {setSceneState('unavailable'); setPlaying(false);}});
    return () => {cancelled = true; scene.current?.dispose(); scene.current = null;};
  }, [nearby, skill]);

  useEffect(() => {scene.current?.update(progress);}, [progress]);

  useEffect(() => {
    if (!playing || !visible || sceneState !== 'ready') return;
    let frame = 0;
    let previous = performance.now();
    const tick = (now: number) => {
      const delta = Math.min(now - previous, 100) * rate / DURATION;
      previous = now;
      progressRef.current = Math.min(1, progressRef.current + delta);
      setProgress(progressRef.current);
      if (progressRef.current < 1) frame = requestAnimationFrame(tick);
      else setPlaying(false);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [playing, visible, rate, sceneState]);

  const seek = (value: number) => {progressRef.current = value; setProgress(value);};
  const selectStage = (index: number) => {setPlaying(false); seek(demo.phases[index].sample);};
  const toggle = () => {
    if (reducedMotion || sceneState === 'unavailable') {selectStage((stage + 1) % demo.phases.length); return;}
    if (complete) seek(0);
    setPlaying(!playing);
  };
  const stepMode = reducedMotion || sceneState === 'unavailable';
  const PlaybackIcon = stepMode ? ArrowRight : playing ? Pause : complete ? RotateCcw : Play;

  return (
    <figure ref={figure} className={styles.demo} aria-labelledby={`${id}-title`} data-skill={skill}>
      <header className={styles.header}>
        <div><p>3D 技能示意</p><h4 id={`${id}-title`}>{demo.title}</h4></div>
        <div className={styles.controls}>
          <button type="button" className={styles.play} onClick={toggle} disabled={sceneState === 'loading'}>
            <PlaybackIcon size={16} aria-hidden="true" />{stepMode ? '下一阶段' : playing ? '暂停' : complete ? '重播' : '播放'}
          </button>
          {!stepMode && <div className={styles.rates} role="group" aria-label={`${demo.title}播放速度`}>
            {[.5, 1, 2].map(value => <button type="button" key={value} aria-label={`${value} 倍速`} aria-pressed={value === rate} onClick={() => setRate(value)}>{value}×</button>)}
          </div>}
        </div>
      </header>
      <div className={styles.world}>
        <div className={styles.canvas} ref={host} />
        {sceneState !== 'ready' && <div className={styles.sceneMessage} role="status">
          <strong>{sceneState === 'loading' ? '正在准备 3D 场景…' : '暂时无法显示 3D 场景'}</strong>
          <span>{demo.scene}</span>
          {sceneState === 'unavailable' && <span>可以使用下方阶段按钮继续查看动作说明。</span>}
        </div>}
        {sceneState === 'ready' && <div className={styles.cameraControls}>
          <span>拖动旋转 · 方向键调整视角</span>
          <button type="button" onClick={() => scene.current?.resetView()}><RotateCcw size={14} aria-hidden="true" />复位视角</button>
        </div>}
      </div>
      <div className={styles.timeline}>
        <label className={styles.srOnly} htmlFor={`${id}-progress`}>{demo.title}演示进度</label>
        <input id={`${id}-progress`} type="range" min="0" max="100" step="1" value={Math.round(progress * 100)}
          aria-valuetext={`${Math.round(progress * 100)}%，${demo.phases[stage].name}`}
          onChange={event => {setPlaying(false); seek(Number(event.target.value) / 100);}} />
        <div className={styles.phases} role="group" aria-label={`${demo.title}动作阶段`}>
          {demo.phases.map((phase, index) => <button type="button" key={phase.name} aria-pressed={stage === index} onClick={() => selectStage(index)}><span aria-hidden="true">{index + 1}</span>{phase.name}</button>)}
        </div>
        <p className={styles.status} role="status" aria-live="polite" aria-atomic="true"><strong>{demo.phases[stage].name}</strong><span>{demo.phases[stage].detail}</span></p>
      </div>
      <figcaption className={styles.caption}>{demo.focus}<span>{reducedMotion ? '已按减少动态效果的偏好改为逐步查看。' : '可拖动进度条或点击阶段逐步查看。'}动作已简化，用于理解技能。</span></figcaption>
    </figure>
  );
}
