import React from 'react';
import DocBreadcrumbs from '@theme-original/DocBreadcrumbs';
import PageMarkdown from '@site/src/components/PageMarkdown';
import styles from './styles.module.css';

export default function DocPageToolbar() {
  return (
    <div className={styles.toolbar} data-markdown-exclude>
      <DocBreadcrumbs />
      <PageMarkdown />
    </div>
  );
}
