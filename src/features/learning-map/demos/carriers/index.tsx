import React, {useEffect, useRef, useState} from 'react';
import {translate} from '@docusaurus/Translate';
import useBaseUrl from '@docusaurus/useBaseUrl';
import {ArrowUpRight, Play, Square} from 'lucide-react';
import {embodiedCarriers} from '../../carriers/content';
import type {EmbodiedCarrier} from '../../carriers/content';
import styles from './styles.module.css';

type DemoMode = 'official' | 'illustration';

function CarrierDemo({carrier, mode}: {carrier: EmbodiedCarrier; mode: DemoMode}) {
  const {id, name, official} = carrier;
  const isOfficial = mode === 'official';
  const motion = isOfficial ? official.motion : carrier.motion;
  const figure = useRef<HTMLElement>(null);
  const player = useRef<HTMLVideoElement>(null);
  const [playing, setPlaying] = useState(false);
  const [visible, setVisible] = useState(false);
  const [pageVisible, setPageVisible] = useState(true);
  const [failed, setFailed] = useState(false);
  const directory = `/img/learning-map/carriers/${isOfficial ? 'official/' : ''}`;
  const poster = useBaseUrl(`${directory}${id}.webp`);
  const gif = useBaseUrl(`/img/learning-map/carriers/${id}.gif`);
  const video = useBaseUrl(`/img/learning-map/carriers/official/${id}.mp4`);
  const animated = playing && visible && pageVisible && !failed;

  useEffect(() => {
    const preference = window.matchMedia('(prefers-reduced-motion: reduce)');
    const updatePreference = () => setPlaying(!preference.matches);
    const updatePageVisibility = () => setPageVisible(!document.hidden);
    updatePreference();
    updatePageVisibility();
    preference.addEventListener('change', updatePreference);
    document.addEventListener('visibilitychange', updatePageVisibility);
    const observer = new IntersectionObserver(([entry]) => setVisible(entry.isIntersecting));
    if (figure.current) observer.observe(figure.current);
    return () => {
      observer.disconnect();
      preference.removeEventListener('change', updatePreference);
      document.removeEventListener('visibilitychange', updatePageVisibility);
    };
  }, []);

  useEffect(() => {
    if (!animated || !isOfficial) return;
    let active = true;
    player.current?.play().catch(() => {if (active) setPlaying(false);});
    return () => {active = false;};
  }, [animated, isOfficial]);

  return (
    <figure ref={figure} className={styles.demo} aria-labelledby={`carrier-gif-${id}`}>
      <div className={styles.image}>
        {isOfficial && animated ? <video ref={player} src={video} poster={poster} width="640" height="400"
          autoPlay loop muted playsInline preload="none"
          aria-label={translate({message: '{name}：{motion}'}, {name, motion})}
          onError={() => setFailed(true)} /> : <img src={animated ? gif : poster} width="640" height="400" loading="lazy" decoding="async"
          alt={translate({message: '{name}：{motion}'}, {name, motion})}
          onError={() => {if (animated) setFailed(true);}} />}
      </div>
      <figcaption>
        <div className={styles.captionHeader}>
          <strong id={`carrier-gif-${id}`}>{name}</strong>
          <button type="button" onClick={() => setPlaying(!playing)} disabled={failed}
            aria-label={playing ? translate({message: '停止{name}动画'}, {name}) : translate({message: '播放{name}动画'}, {name})}>
            {playing ? <Square size={14} aria-hidden="true" /> : <Play size={14} aria-hidden="true" />}
            {playing ? translate({message: '停止'}) : translate({message: '播放'})}
          </button>
        </div>
        <p><span className={styles.model}>{isOfficial ? official.model : translate({message: '通用结构示意'})}</span>{motion}</p>
        {isOfficial && <div className={styles.links}>
          <a href={official.source.href} target="_blank" rel="noopener noreferrer">
            {translate({message: '来源：{name}'}, {name: official.source.name})}<ArrowUpRight size={13} aria-hidden="true" />
          </a>
        </div>}
        {failed && <span className={styles.error} role="status">{translate({message: '动画暂时无法加载，已显示静态封面。'})}</span>}
      </figcaption>
    </figure>
  );
}

export default function CarrierDemos() {
  const [mode, setMode] = useState<DemoMode>('official');

  return (
    <div className={styles.gallery}>
      <div className={styles.toolbar}>
        <div className={styles.modes} role="group" aria-label={translate({message: '载体演示类型'})}>
          <button type="button" aria-pressed={mode === 'official'} onClick={() => setMode('official')}>{translate({message: '官方演示'})}</button>
          <button type="button" aria-pressed={mode === 'illustration'} onClick={() => setMode('illustration')}>{translate({message: '3D 示意'})}</button>
        </div>
        <p className={styles.hint}>{mode === 'official'
          ? translate({message: '观看真实机器人动作，仿真载体对应官方仿真片段。'})
          : translate({message: '通过通用模型看清身体结构与动作方式。'})}</p>
      </div>
      <div className={styles.grid}>
        {embodiedCarriers.map(carrier => <CarrierDemo key={`${carrier.id}-${mode}`} carrier={carrier} mode={mode} />)}
      </div>
      <p className={styles.note}>{mode === 'official'
        ? translate({message: '官方演示为原片节选，保留原有速度与画面标注；完整演示见来源链接。'})
        : translate({message: '3D 结构与动作示意，不对应特定产品。'})}</p>
    </div>
  );
}
