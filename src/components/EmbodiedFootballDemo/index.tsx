import React, {useEffect, useId, useRef, useState} from 'react';
import {ArrowRight, BrainCircuit, Eye, Move3D, Pause, Play, RotateCcw, Shuffle} from 'lucide-react';
import FootballScene3D from './FootballScene3D';
import styles from './styles.module.css';

const DURATION = 14000;
// Keep the scene's choreography timeline while shortening normal playback to eight seconds.
const NORMAL_PLAYBACK_DURATION = 8000;
const playbackRates = [0.5, 1, 2];
const stages = [
  {name: '感知', summary: '看见足球、球门和守门员', icon: Eye, sample: 0},
  {name: '决策', summary: '找到空当，选择射门方向', icon: BrainCircuit, sample: 4500},
  {name: '行动', summary: '调整站位，驱动关节踢球', icon: Move3D, sample: 8750},
  {name: '反馈', summary: '跟踪足球，检查射门结果', icon: RotateCcw, sample: 12500},
];

export default function EmbodiedFootballDemo(): React.JSX.Element {
  const [elapsed, setElapsed] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [playbackRate, setPlaybackRate] = useState(1);
  const [keeperNear, setKeeperNear] = useState(false);
  const [hasMoved, setHasMoved] = useState(false);
  const [reducedMotion, setReducedMotion] = useState(false);
  const elapsedRef = useRef(0);
  const figureRef = useRef<HTMLElement>(null);
  const id = useId().replace(/:/g, '');
  const stageIndex = elapsed < 3000 ? 0 : elapsed < 6000 ? 1 : elapsed < 11000 ? 2 : 3;
  const complete = elapsed >= DURATION;

  useEffect(() => {
    const preference = window.matchMedia('(prefers-reduced-motion: reduce)');
    const update = () => {
      setReducedMotion(preference.matches);
      if (preference.matches) setPlaying(false);
    };
    update();
    preference.addEventListener('change', update);
    return () => preference.removeEventListener('change', update);
  }, []);

  useEffect(() => {
    if (!playing) return;
    let frame: number;
    let previous = performance.now();
    const tick = (now: number) => {
      const delta = Math.min(now - previous, 100) * (DURATION / NORMAL_PLAYBACK_DURATION) * playbackRate;
      elapsedRef.current = Math.min(DURATION, elapsedRef.current + delta);
      previous = now;
      setElapsed(elapsedRef.current);
      if (elapsedRef.current < DURATION) frame = requestAnimationFrame(tick);
      else setPlaying(false);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [playing, playbackRate]);

  useEffect(() => {
    const pauseWhenHidden = () => {if (document.hidden) setPlaying(false);};
    document.addEventListener('visibilitychange', pauseWhenHidden);
    const observer = new IntersectionObserver(([entry]) => {
      if (!entry.isIntersecting) setPlaying(false);
    });
    if (figureRef.current) observer.observe(figureRef.current);
    return () => {
      document.removeEventListener('visibilitychange', pauseWhenHidden);
      observer.disconnect();
    };
  }, []);

  const seek = (time: number) => {
    elapsedRef.current = time;
    setElapsed(time);
  };
  const selectStage = (index: number) => {
    setPlaying(false);
    seek(stages[index].sample);
  };
  const togglePlayback = () => {
    if (reducedMotion) {
      selectStage((stageIndex + 1) % stages.length);
      return;
    }
    if (complete) seek(0);
    setPlaying(!playing);
  };
  const moveKeeper = () => {
    setKeeperNear(!keeperNear);
    setHasMoved(true);
    seek(0);
  };

  const details = [
    hasMoved ? '守门员换了位置，机器人重新观察足球和防守区域。' : '摄像头捕捉场上信息，识别足球、球门和守门员的位置。',
    `守门员挡住${keeperNear ? '近' : '远'}侧，选择${keeperNear ? '远' : '近'}侧空当作为射门目标。`,
    '根据射门方向调整站位，控制身体平衡，再驱动腿部关节踢球。',
    '球进入球门。新的视觉观测帮助机器人判断结果，并决定下一步行动。',
  ];
  const PlaybackIcon = reducedMotion ? ArrowRight : playing ? Pause : complete ? RotateCcw : Play;

  return (
    <figure ref={figureRef} className={styles.demo} aria-labelledby={`${id}-title`}>
      <div className={styles.header}>
        <div><p className={styles.eyebrow}>机器人足球 · 3D 闭环演示</p><h3 id={`${id}-title`}>让机器人把球踢进球门</h3></div>
        <div className={styles.controls}>
          <button type="button" className={styles.playButton} onClick={togglePlayback}>
            <PlaybackIcon size={17} aria-hidden="true" />{reducedMotion ? '下一阶段' : playing ? '暂停演示' : complete ? '重播射门' : '播放射门'}
          </button>
          <button type="button" onClick={moveKeeper}><Shuffle size={17} aria-hidden="true" />移动守门员</button>
          {!reducedMotion && <div className={styles.playbackRates} role="group" aria-label="播放速度">
            {playbackRates.map(rate => (
              <button type="button" key={rate} aria-label={`${rate} 倍速`} aria-pressed={playbackRate === rate} onClick={() => setPlaybackRate(rate)}>{rate}×</button>
            ))}
          </div>}
        </div>
      </div>
      <div className={styles.content}>
        <FootballScene3D elapsed={elapsed} keeperNear={keeperNear} stage={stageIndex} />
        <div className={styles.explanation}>
          <div className={styles.stages} role="group" aria-label="逐步查看具身智能闭环">
            {stages.map(({name, summary, icon: Icon}, index) => (
              <button type="button" key={name} aria-pressed={index === stageIndex} onClick={() => selectStage(index)}>
                <Icon size={21} strokeWidth={1.7} aria-hidden="true" /><span><strong>{name}</strong><span>{summary}</span></span>
              </button>
            ))}
          </div>
          <p className={styles.returnLoop}><RotateCcw size={16} aria-hidden="true" />新的观测，再次进入感知</p>
        </div>
      </div>
      <div className={styles.status} role="status" aria-live="polite" aria-atomic="true"><strong>{stages[stageIndex].name}</strong><span>{details[stageIndex]}</span></div>
      <figcaption className={styles.caption}>{reducedMotion ? '已按减少动态效果的偏好改为逐步查看。' : '可播放或点击阶段逐步查看。'}移动守门员会重新开始演示，观察射门方向如何变化。动作已简化，用来理解闭环。</figcaption>
    </figure>
  );
}
