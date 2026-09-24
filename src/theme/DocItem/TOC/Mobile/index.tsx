import React, {type ReactNode, useMemo} from 'react';
import clsx from 'clsx';
import {useLocation} from '@docusaurus/router';
import {translate} from '@docusaurus/Translate';
import {ThemeClassNames} from '@docusaurus/theme-common';
import {useDoc, useDocsSidebar} from '@docusaurus/plugin-content-docs/client';
import DocSidebarItems from '@theme/DocSidebarItems';
import TOCItems from '@theme/TOCItems';
import Link from '@docusaurus/Link';
import {Gamepad2} from 'lucide-react';
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

function InteractiveModeLink({href}: {href: string}) {
  return (
    <Link to={href} className={styles.interactiveModeLink}>
      <Gamepad2 size={16} aria-hidden="true" />
      <span>{translate({id: 'theme.DocItem.interactiveMode', message: '交互模式'})}</span>
    </Link>
  );
}

function DirectorySection({
  label,
  className,
  children,
}: {
  label: string;
  className?: string;
  children: ReactNode;
}) {
  return (
    <details className={clsx(styles.directorySection, className)}>
      <summary className={styles.directorySummary}>{label}</summary>
      <nav aria-label={label} className={styles.directoryContent}>{children}</nav>
    </details>
  );
}

function useMobileDocDirectory() {
  const sidebar = useDocsSidebar();

  return useMemo(() => {
    if (!sidebar || sidebar.items.length === 0) {
      return null;
    }

    const [firstItem] = sidebar.items;

    if (
      sidebar.items.length === 1 &&
      firstItem &&
      firstItem.type === 'category'
    ) {
      return {
        label: firstItem.label,
        items: firstItem.items,
      };
    }

    return {
      label: translate({
        id: 'theme.DocItem.mobileDirectory.defaultLabel',
        message: '章节目录',
      }),
      items: sidebar.items,
    };
  }, [sidebar]);
}

export default function DocItemTOCMobile(): ReactNode {
  const {pathname} = useLocation();
  const {toc, frontMatter} = useDoc();
  const directory = useMobileDocDirectory();
  const interactiveHref = getInteractiveHref(pathname);
  const canRenderToc = !frontMatter.hide_table_of_contents && toc.length > 0;

  if (!directory && !canRenderToc && !interactiveHref) {
    return null;
  }

  return (
    <div
      key={pathname}
      className={clsx(
        ThemeClassNames.docs.docTocMobile,
        styles.mobileDirectory,
      )}>
      {interactiveHref && <InteractiveModeLink href={interactiveHref} />}
      {directory && (
        <DirectorySection label={directory.label} className={styles.chapterDirectory}>
          <ul
            className={clsx(
              ThemeClassNames.docs.docSidebarMenu,
              'menu__list',
              styles.mobileDirectoryList,
            )}>
            <DocSidebarItems items={directory.items} activePath={pathname} level={1} />
          </ul>
        </DirectorySection>
      )}
      {canRenderToc && (
        <DirectorySection label={translate({id: 'theme.DocItem.tocTitle', message: '本页目录'})}>
          <TOCItems
            toc={toc}
            minHeadingLevel={frontMatter.toc_min_heading_level}
            maxHeadingLevel={frontMatter.toc_max_heading_level}
          />
        </DirectorySection>
      )}
    </div>
  );
}
