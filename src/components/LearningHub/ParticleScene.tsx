import React, {useEffect, useRef, useState} from 'react';
import {Pause, Play} from 'lucide-react';
import {hubText} from './messages';
import styles from './styles.module.css';

type Point = {x: number; y: number; z: number; leg?: number; weight?: number};

/** A deliberately illustrative point cloud, independent of the physics demos. */
function robotPoints(): Point[] {
  const points: Point[] = [];
  const box = (cx: number, cy: number, cz: number, w: number, h: number, d: number) => {
    const steps = 10;
    for (let i = 0; i <= steps; i++) for (let j = 0; j <= steps; j++) {
      const a = i / steps - .5;
      const b = j / steps - .5;
      for (const side of [-.5, .5]) {
        points.push({x: cx + a * w, y: cy + b * h, z: cz + side * d});
        points.push({x: cx + a * w, y: cy + side * h, z: cz + b * d});
      }
    }
  };
  box(0, .22, 0, 1.35, .4, .55);
  box(.78, .38, 0, .35, .32, .45);
  for (let leg = 0; leg < 4; leg++) {
    const x = leg < 2 ? .5 : -.5;
    const z = leg % 2 ? .34 : -.34;
    const joints = [[x, .12], [x + .16, -.32], [x - .05, -.8]];
    for (let segment = 0; segment < 2; segment++) for (let j = 0; j < 22; j++) {
      const t = j / 21;
      for (let k = 0; k < 6; k++) {
        const a = k * Math.PI / 3;
        points.push({x: joints[segment][0] * (1 - t) + joints[segment + 1][0] * t + Math.cos(a) * .045,
          y: joints[segment][1] * (1 - t) + joints[segment + 1][1] * t,
          z: z + Math.sin(a) * .045, leg, weight: (segment + t) / 2});
      }
    }
  }
  return points;
}

function networkPoints(): Point[] {
  return Array.from({length: 1100}, (_, i) => {
    const theta = i * 2.399963;
    const y = 1 - (i / 1099) * 2;
    const radius = Math.sqrt(1 - y * y);
    return {x: Math.cos(theta) * radius, y: y * .9, z: Math.sin(theta) * radius};
  });
}

export default function ParticleScene({kind, paused, onToggle}: {kind: 'practices' | 'foundations'; paused: boolean; onToggle: () => void}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const timeRef = useRef(0);
  const [reduced, setReduced] = useState(false);
  const text = hubText();

  useEffect(() => {
    const media = window.matchMedia('(prefers-reduced-motion: reduce)');
    const update = () => setReduced(media.matches);
    update();
    media.addEventListener('change', update);
    return () => media.removeEventListener('change', update);
  }, []);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;
    const points = kind === 'practices' ? robotPoints() : networkPoints();
    const motion = !paused && !reduced && !window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    let width = 0, height = 0, raf = 0, last = 0, time = timeRef.current;
    let visible = false;
    let pointer = {x: -1000, y: -1000};
    let blue = '', muted = '';
    const colors = () => {
      const css = getComputedStyle(canvas);
      blue = css.getPropertyValue('--site-blue').trim();
      muted = css.getPropertyValue('--site-muted').trim();
    };
    const paint = () => {
      ctx.clearRect(0, 0, width, height);
      const scale = Math.min(width * .32, height * .43);
      const angle = -.52 + Math.sin(time * .00015) * .18;
      const project = (x: number, y: number, z: number) => {
        const rx = x * Math.cos(angle) - z * Math.sin(angle);
        const rz = x * Math.sin(angle) + z * Math.cos(angle);
        return {x: width * .5 + rx * scale, y: height * .49 - (y * .94 - rz * .34) * scale, z: rz};
      };
      // Ground samples give the cloud a spatial reference.
      for (let i = -10; i <= 10; i++) for (let j = -7; j <= 7; j++) {
        const p = project(i * .16, -1, j * .16);
        ctx.globalAlpha = .13;
        ctx.fillStyle = muted;
        ctx.fillRect(p.x, p.y, 1.4, 1.4);
      }
      const stride = width < 420 ? 2 : 1;
      const projected = points.filter((_, i) => i % stride === 0).map((p, index) => {
        const gait = p.leg !== undefined ? Math.sin(time * .0025 + (p.leg === 0 || p.leg === 3 ? 0 : Math.PI)) : 0;
        const wave = kind === 'foundations' ? 1 + Math.sin(time * .0007 + p.y * 4) * .045 : 1;
        const v = project(p.x * wave + gait * .16 * (p.weight ?? 0), p.y * wave + Math.max(gait, 0) * .09 * (p.weight ?? 0), p.z * wave);
        if (motion) {
          const dx = v.x - pointer.x, dy = v.y - pointer.y;
          const distance = Math.hypot(dx, dy);
          if (distance < 95 && distance > 0) {
            const force = (1 - distance / 95) * 16;
            v.x += dx / distance * force;
            v.y += dy / distance * force;
          }
        }
        return {...v, index};
      }).sort((a, b) => a.z - b.z);
      for (const p of projected) {
        ctx.globalAlpha = .35 + (p.z + 1.5) / 4;
        ctx.fillStyle = blue;
        ctx.beginPath();
        ctx.arc(p.x, p.y, (p.index % 17 === 0 ? 2 : 1.15) * Math.max(.8, width / 650), 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.globalAlpha = 1;
    };
    const tick = (now: number) => {
      raf = 0;
      if (!visible || document.hidden || !motion) return;
      if (now - last >= 32) {
        time += Math.min(now - last, 50);
        last = now;
        paint();
      }
      raf = requestAnimationFrame(tick);
    };
    const sync = () => {
      cancelAnimationFrame(raf);
      raf = 0;
      if (visible && !document.hidden) {
        paint();
        last = performance.now();
        if (motion) raf = requestAnimationFrame(tick);
      }
    };
    const resize = () => {
      const rect = canvas.getBoundingClientRect();
      width = rect.width; height = rect.height;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      paint();
    };
    const move = (event: PointerEvent) => {
      if (event.pointerType !== 'mouse') return;
      const rect = canvas.getBoundingClientRect();
      pointer = {x: event.clientX - rect.left, y: event.clientY - rect.top};
    };
    const leave = () => {pointer = {x: -1000, y: -1000};};
    colors();
    const resizer = new ResizeObserver(resize);
    resizer.observe(canvas);
    const observer = new IntersectionObserver(([entry]) => {visible = entry.isIntersecting; sync();});
    observer.observe(canvas);
    const theme = new MutationObserver(() => {colors(); paint();});
    theme.observe(document.documentElement, {attributes: true, attributeFilter: ['data-theme']});
    document.addEventListener('visibilitychange', sync);
    canvas.addEventListener('pointermove', move);
    canvas.addEventListener('pointerleave', leave);
    return () => {
      timeRef.current = time;
      cancelAnimationFrame(raf);
      resizer.disconnect(); observer.disconnect(); theme.disconnect();
      document.removeEventListener('visibilitychange', sync);
      canvas.removeEventListener('pointermove', move);
      canvas.removeEventListener('pointerleave', leave);
    };
  }, [kind, paused, reduced]);

  return <div className={styles.scene}>
    <canvas ref={canvasRef} aria-hidden="true" />
    <div className={styles.sceneCaption}>
      <span>{kind === 'practices' ? text.robotCloud : text.knowledgeCloud}</span>
      <button type="button" onClick={onToggle} disabled={reduced} aria-pressed={paused || reduced}>
        {paused || reduced ? <Play size={14} aria-hidden="true" /> : <Pause size={14} aria-hidden="true" />}
        {reduced ? text.staticView : paused ? text.play : text.pause}
      </button>
    </div>
  </div>;
}
