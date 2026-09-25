import React from 'react';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import {useDoc} from '@docusaurus/plugin-content-docs/client';
import DocItemContent from '@theme-original/DocItem/Content';
import type {Props} from '@theme/DocItem/Content';
import TranslationNotice from '@site/src/components/TranslationNotice';

export default function LocalizedDocItemContent(props: Props) {
  const {i18n} = useDocusaurusContext();
  const {metadata} = useDoc();
  // Docusaurus falls back to docs/ when a translated Markdown file is absent.
  const isChineseFallback = i18n.currentLocale === 'en' && metadata.source.startsWith('@site/docs/');

  return (
    <div data-markdown-content>
      {isChineseFallback && <TranslationNotice />}
      {isChineseFallback ? (
        <div lang="zh-Hans"><DocItemContent {...props} /></div>
      ) : <DocItemContent {...props} />}
    </div>
  );
}
