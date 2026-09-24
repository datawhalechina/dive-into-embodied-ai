import React, {useEffect, useState} from 'react';
import {chapters} from './content';
import styles from './styles.module.css';

export default function ChapterNavigation() {
  const [activeId, setActiveId] = useState(chapters[0].id);

  useEffect(() => {
    const sections = chapters
      .map(({id}) => document.getElementById(id))
      .filter((section): section is HTMLElement => section !== null);
    if (!sections.length) return undefined;

    let frame = 0;
    const updateActiveChapter = () => {
      frame = 0;
      const navbarBottom = document.querySelector('.navbar')?.getBoundingClientRect().bottom ?? 0;
      const readingLine = Math.max(0, navbarBottom) + 32;
      let current = sections[0];
      for (const section of sections) {
        if (section.getBoundingClientRect().top <= readingLine + 1) current = section;
      }
      // The final section may be too short to reach the top of the viewport.
      if (window.scrollY + window.innerHeight >= document.documentElement.scrollHeight - 2) {
        current = sections[sections.length - 1];
      }
      setActiveId(current.id);
    };
    const scheduleUpdate = () => {
      if (!frame) frame = window.requestAnimationFrame(updateActiveChapter);
    };

    updateActiveChapter();
    window.addEventListener('scroll', scheduleUpdate, {passive: true});
    window.addEventListener('resize', scheduleUpdate);
    window.addEventListener('hashchange', scheduleUpdate);
    const resizeObserver = new ResizeObserver(scheduleUpdate);
    resizeObserver.observe(sections[0].parentElement!);

    return () => {
      window.cancelAnimationFrame(frame);
      window.removeEventListener('scroll', scheduleUpdate);
      window.removeEventListener('resize', scheduleUpdate);
      window.removeEventListener('hashchange', scheduleUpdate);
      resizeObserver.disconnect();
    };
  }, []);

  return (
    <nav className={styles.chapterNav} aria-labelledby="chapter-nav-title">
      <p id="chapter-nav-title" className={styles.chapterNavTitle}>本页目录</p>
      <ol className={styles.chapterLinks}>
        {chapters.map(({id, label}) => (
          <li key={id}>
            <a href={`#${id}`} aria-current={activeId === id ? 'location' : undefined}>{label}</a>
          </li>
        ))}
      </ol>
    </nav>
  );
}
