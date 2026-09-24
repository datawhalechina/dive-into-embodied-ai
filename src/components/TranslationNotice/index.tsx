import React from 'react';
import Link from '@docusaurus/Link';
import {useLocation} from '@docusaurus/router';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import {useAlternatePageUtils} from '@docusaurus/theme-common/internal';

/** Keep the scope of the first English edition visible at its entry points. */
export default function TranslationNotice({translated = false}: {translated?: boolean}) {
  const {i18n} = useDocusaurusContext();
  const {createUrl} = useAlternatePageUtils();
  const {search, hash} = useLocation();
  if (i18n.currentLocale !== 'en') return null;

  return (
    <aside className="translation-notice" lang="en" aria-label="Translation status">
      {translated ? (
        <>
          The <Link href="https://github.com/datawhalechina/dive-into-embodied-ai/blob/master/README.en.md">README</Link> and learning map are available in English.
          {' '}Tutorial chapters and standalone playgrounds are currently in Chinese.
        </>
      ) : (
        <>
          This page is currently available in Chinese.
          {' '}Explore the <Link to="/learning-map">English learning map</Link> or{' '}
          <Link autoAddBaseUrl={false} to={`pathname://${createUrl({locale: i18n.defaultLocale, fullyQualified: false})}${search}${hash}`}>
            read the Chinese version
          </Link>.
        </>
      )}
    </aside>
  );
}
