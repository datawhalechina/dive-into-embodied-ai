import React, {type ReactNode} from 'react';
import clsx from 'clsx';
import {translate} from '@docusaurus/Translate';
import {ThemeClassNames} from '@docusaurus/theme-common';
import {useDoc} from '@docusaurus/plugin-content-docs/client';
import {useLocation} from '@docusaurus/router';
import Link from '@docusaurus/Link';
import TOCItems from '@theme/TOCItems';
import {Gamepad2} from 'lucide-react';
// Single source of truth for playground pages (shared with PlaygroundHeader).
import {PLAYGROUNDS} from '@site/src/components/PlaygroundHeader';
import styles from './styles.module.css';

function normalizePath(pathname: string): string {
  return pathname.replace(/\/$/, '');
}

function getInteractiveHref(pathname: string): string | undefined {
  const currentPath = normalizePath(pathname);
  return PLAYGROUNDS.find((playground) => {
    const readingPath = normalizePath(playground.readingHref);
    return currentPath === readingPath || currentPath.endsWith(readingPath);
  })?.playgroundHref;
}

function InteractiveModeButton({href}: {href: string}) {
  return (
    <Link
      to={href}
      className={styles.interactiveModeLink}>
      <Gamepad2 size={16} aria-hidden="true" />
      <span>{translate({id: 'theme.DocItem.interactiveMode', message: '交互模式'})}</span>
    </Link>
  );
}

export default function DocItemTOCDesktop(): ReactNode {
  const {toc, frontMatter} = useDoc();
  const {pathname} = useLocation();
  const interactiveHref = getInteractiveHref(pathname);
  const label = translate({id: 'theme.DocItem.tocTitle', message: '本页目录'});

  return (
    <nav
      aria-label={label}
      className={clsx(ThemeClassNames.docs.docTocDesktop, styles.desktopToc, 'thin-scrollbar')}>
      {interactiveHref && <InteractiveModeButton href={interactiveHref} />}
      <div className={styles.tocTitle}>{label}</div>
      <TOCItems
        toc={toc}
        minHeadingLevel={frontMatter.toc_min_heading_level}
        maxHeadingLevel={frontMatter.toc_max_heading_level}
        linkClassName="table-of-contents__link toc-highlight"
        linkActiveClassName="table-of-contents__link--active"
      />
    </nav>
  );
}
