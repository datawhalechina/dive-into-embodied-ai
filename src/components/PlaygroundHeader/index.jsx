import React, { useEffect, useRef, useState } from 'react';
import Link from '@docusaurus/Link';
import PageMarkdown from '@site/src/components/PageMarkdown';
import { ArrowLeft, ChevronDown, ChevronLeft, ChevronRight } from 'lucide-react';

/**
 * 所有 playground 的登记表。新增交互页时只需在此加一条。
 * 条目顺序决定 prev/next 翻页顺序(建议与课程章节顺序一致)。
 */
export const PLAYGROUNDS = [
  {
    key: 'pd',
    title: 'PD Playground',
    subtitle: 'PID 控制',
    chapterNumber: 1,
    playgroundHref: '/cs123/pd-playground',
    readingHref: '/docs/practices/quadruped/cs123/pid-control',
  },
  {
    key: 'fk',
    title: 'FK Playground',
    subtitle: '正运动学',
    chapterNumber: 2,
    playgroundHref: '/cs123/fk-playground',
    readingHref: '/docs/practices/quadruped/cs123/forward-kinematics',
  },
  {
    key: 'ik',
    title: 'IK Playground',
    subtitle: '逆运动学',
    chapterNumber: 3,
    playgroundHref: '/cs123/ik-playground',
    readingHref: '/docs/practices/quadruped/cs123/inverse-kinematics',
  },
];

const barStyle = {
  position: 'sticky',
  top: 'var(--ifm-navbar-height, 60px)',
  zIndex: 20,
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'space-between',
  gap: 12,
  padding: '8px 16px',
  background: 'var(--ifm-background-surface-color)',
  borderBottom: '1px solid var(--ifm-color-emphasis-200)',
  fontSize: 13,
  flexWrap: 'wrap',
};

const returnBtnStyle = {
  display: 'inline-flex',
  alignItems: 'center',
  gap: 6,
  padding: '4px 10px',
  borderRadius: 6,
  color: 'var(--ifm-color-emphasis-700)',
  textDecoration: 'none',
  background: 'var(--ifm-color-emphasis-100)',
  fontWeight: 500,
  flexShrink: 0,
};

const pagerGroupStyle = {
  display: 'inline-flex',
  alignItems: 'center',
  justifyContent: 'center',
  flexWrap: 'wrap',
  gap: 8,
};

const sideSlotStyle = {
  flex: '1 0 auto',
  display: 'flex',
  alignItems: 'center',
  minWidth: 0,
};

const arrowBtnBase = {
  display: 'inline-flex',
  alignItems: 'center',
  justifyContent: 'center',
  gap: 4,
  height: 28,
  padding: '0 10px',
  borderRadius: 6,
  border: '1px solid var(--ifm-color-emphasis-200)',
  background: 'var(--ifm-background-color)',
  color: 'var(--ifm-color-emphasis-700)',
  fontSize: 12,
  fontWeight: 500,
  textDecoration: 'none',
  transition: 'background 0.15s ease, color 0.15s ease',
};

const titleBtnStyle = {
  display: 'inline-flex',
  alignItems: 'center',
  gap: 6,
  padding: '5px 14px',
  borderRadius: 8,
  border: '1px solid var(--ifm-color-emphasis-200)',
  background: 'var(--ifm-background-color)',
  color: 'var(--ifm-color-emphasis-800)',
  fontWeight: 600,
  fontSize: 13,
  cursor: 'pointer',
  fontFamily: 'inherit',
  lineHeight: 1.4,
};

const dropdownStyle = {
  position: 'absolute',
  top: 'calc(100% + 6px)',
  left: '50%',
  transform: 'translateX(-50%)',
  zIndex: 50,
  minWidth: 260,
  maxHeight: 320,
  overflowY: 'auto',
  padding: 6,
  margin: 0,
  listStyle: 'none',
  background: 'var(--ifm-background-surface-color)',
  border: '1px solid var(--ifm-color-emphasis-200)',
  borderRadius: 10,
  boxShadow: '0 12px 28px -8px rgba(15, 23, 42, 0.25), 0 2px 6px -1px rgba(15, 23, 42, 0.1)',
};

const dropdownItemBase = {
  display: 'flex',
  alignItems: 'center',
  gap: 8,
  padding: '8px 12px',
  borderRadius: 6,
  fontSize: 13,
  textDecoration: 'none',
};

function NavArrow({ target, direction }) {
  const isPrev = direction === 'prev';
  const Icon = isPrev ? ChevronLeft : ChevronRight;
  const label = isPrev ? '上一章' : '下一章';
  const children = isPrev ? (
    <>
      <Icon size={14} /> {label}
    </>
  ) : (
    <>
      {label} <Icon size={14} />
    </>
  );
  if (!target) {
    return (
      <span
        style={{ ...arrowBtnBase, opacity: 0.35, cursor: 'not-allowed' }}
        aria-label={`没有${label}`}
        aria-disabled="true"
      >
        {children}
      </span>
    );
  }
  const hint = `${label}:第 ${target.chapterNumber} 章 · ${target.subtitle}`;
  return (
    <Link to={target.playgroundHref} style={arrowBtnBase} aria-label={hint} title={hint}>
      {children}
    </Link>
  );
}

/**
 * 顶部 slim 导航条:返回阅读版 + 上一章/目录/下一章。
 * 用法:<PlaygroundHeader currentKey="ik" />
 */
export default function PlaygroundHeader({ currentKey }) {
  const currentIdx = PLAYGROUNDS.findIndex((p) => p.key === currentKey);
  const current = PLAYGROUNDS[currentIdx];
  const prev = currentIdx > 0 ? PLAYGROUNDS[currentIdx - 1] : null;
  const next = currentIdx >= 0 && currentIdx < PLAYGROUNDS.length - 1 ? PLAYGROUNDS[currentIdx + 1] : null;

  const [open, setOpen] = useState(false);
  const wrapperRef = useRef(null);

  useEffect(() => {
    if (!open) return;
    const onDocClick = (e) => {
      if (wrapperRef.current && !wrapperRef.current.contains(e.target)) {
        setOpen(false);
      }
    };
    const onEsc = (e) => {
      if (e.key === 'Escape') setOpen(false);
    };
    document.addEventListener('mousedown', onDocClick);
    document.addEventListener('keydown', onEsc);
    return () => {
      document.removeEventListener('mousedown', onDocClick);
      document.removeEventListener('keydown', onEsc);
    };
  }, [open]);

  if (!current) return null;

  return (
    <div style={barStyle} data-markdown-exclude>
      <div style={{ ...sideSlotStyle, justifyContent: 'flex-start' }}>
        <Link to={current.readingHref} style={returnBtnStyle}>
          <ArrowLeft size={14} />
          返回阅读版
        </Link>
      </div>

      <div style={pagerGroupStyle}>
        <NavArrow target={prev} direction="prev" />
        <div ref={wrapperRef} style={{ position: 'relative' }}>
          <button
            type="button"
            onClick={() => setOpen((v) => !v)}
            style={titleBtnStyle}
            aria-haspopup="listbox"
            aria-expanded={open}
          >
            <span>
              第 {current.chapterNumber} 章 · {current.subtitle}
            </span>
            <ChevronDown
              size={14}
              style={{
                transition: 'transform 0.2s',
                transform: open ? 'rotate(180deg)' : 'none',
              }}
            />
          </button>

          {open && (
            <ul role="listbox" style={dropdownStyle} aria-label="交互 Playground 目录">
              {PLAYGROUNDS.map((p) => {
                const active = p.key === currentKey;
                return (
                  <li key={p.key} role="option" aria-selected={active}>
                    {active ? (
                      <span
                        style={{
                          ...dropdownItemBase,
                          background: 'var(--ifm-color-primary)',
                          color: '#ffffff',
                          fontWeight: 600,
                          cursor: 'default',
                        }}
                      >
                        <span style={{ fontFamily: 'ui-monospace, monospace', fontSize: 11, opacity: 0.85 }}>
                          Ch {p.chapterNumber}
                        </span>
                        {p.title}
                        <span style={{ opacity: 0.75, fontSize: 12 }}>· {p.subtitle}</span>
                      </span>
                    ) : (
                      <Link
                        to={p.playgroundHref}
                        onClick={() => setOpen(false)}
                        style={{
                          ...dropdownItemBase,
                          color: 'var(--ifm-color-emphasis-800)',
                          fontWeight: 500,
                        }}
                      >
                        <span
                          style={{
                            fontFamily: 'ui-monospace, monospace',
                            fontSize: 11,
                            color: 'var(--ifm-color-emphasis-500)',
                          }}
                        >
                          Ch {p.chapterNumber}
                        </span>
                        {p.title}
                        <span style={{ color: 'var(--ifm-color-emphasis-500)', fontSize: 12 }}>· {p.subtitle}</span>
                      </Link>
                    )}
                  </li>
                );
              })}
            </ul>
          )}
        </div>
        <NavArrow target={next} direction="next" />
      </div>

      <div style={{ ...sideSlotStyle, flex: '1 0 auto', justifyContent: 'flex-end' }}>
        <PageMarkdown />
      </div>
    </div>
  );
}
