import {translate} from '@docusaurus/Translate';
import React, {useEffect, useState} from 'react';
import {chapters} from './content';
import styles from './styles.module.css';

export default function ChapterNavigation() {
  const [activeId, setActiveId] = useState(chapters[0].id);

  useEffect(() => {
    const sections = chapters
      .flatMap(({id, children = []}) => [
        {id, nested: false},
        ...children.map(child => ({id: child.id, nested: true})),
      ])
      .flatMap(({id, nested}) => {
        const element = document.getElementById(id);
        return element ? [{element, nested}] : [];
      });
    if (!sections.length) return undefined;

    let frame = 0;
    const updateActiveChapter = () => {
      frame = 0;
      const navbarBottom = document.querySelector('.navbar')?.getBoundingClientRect().bottom ?? 0;
      const readingLine = Math.max(0, navbarBottom) + 32;
      let current = sections[0].element;
      for (const {element, nested} of sections) {
        const {top, bottom} = element.getBoundingClientRect();
        // Return to the parent chapter after leaving a child section.
        if (top <= readingLine + 1 && (!nested || bottom > readingLine + 1)) current = element;
      }
      // The final section may be too short to reach the top of the viewport.
      if (window.scrollY + window.innerHeight >= document.documentElement.scrollHeight - 2) {
        current = sections[sections.length - 1].element;
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
    resizeObserver.observe(sections[0].element.parentElement!);

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
      <p id="chapter-nav-title" className={styles.chapterNavTitle}>{translate({message: "本页目录"})}</p>
      <ol className={styles.chapterLinks}>
        {chapters.map(({id, label, children}) => (
          <li key={id}>
            <a
              href={`#${id}`}
              className={children?.some(child => child.id === activeId) ? styles.chapterAncestor : undefined}
              aria-current={activeId === id ? 'location' : undefined}>
              {label}
            </a>
            {children && (
              <ol className={styles.chapterChildren}>
                {children.map(child => (
                  <li key={child.id}>
                    <a href={`#${child.id}`} aria-current={activeId === child.id ? 'location' : undefined}>{child.label}</a>
                  </li>
                ))}
              </ol>
            )}
          </li>
        ))}
      </ol>
    </nav>
  );
}
