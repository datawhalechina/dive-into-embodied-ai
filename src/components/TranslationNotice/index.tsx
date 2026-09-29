import React from 'react';
import Link from '@docusaurus/Link';
import Translate, {translate} from '@docusaurus/Translate';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import {useHistorySelector} from '@docusaurus/theme-common';
import {useAlternatePageUtils} from '@docusaurus/theme-common/internal';

/** Describe translation coverage without presenting Chinese fallback as translated. */
export default function TranslationNotice({translated = false}: {translated?: boolean}) {
  const {i18n} = useDocusaurusContext();
  const {createUrl} = useAlternatePageUtils();
  // Match the locale menu: Docusaurus freezes useLocation during hydration.
  const search = useHistorySelector(history => history.location.search);
  const hash = useHistorySelector(history => history.location.hash);
  if (i18n.currentLocale === i18n.defaultLocale) return null;

  return (
    <aside className="translation-notice" lang={i18n.currentLocale}
      data-translation-notice={translated ? 'overview' : 'fallback'}
      aria-label={translate({id: 'translationNotice.label', message: '翻译状态'})}>
      {translated ? (
        <Translate id="translationNotice.overview" values={{
          readme: <Link href={`https://github.com/datawhalechina/dive-into-embodied-ai/blob/master/README.${i18n.currentLocale}.md`}>README</Link>,
          introduction: <Link to="/docs/introduction/intro">{translate({id: 'translationNotice.introduction', message: '具身导论'})}</Link>,
          projects: <Link to="/docs/practices/intro">{translate({id: 'translationNotice.projects', message: '项目总览'})}</Link>,
          roadmap: <Link to="/docs/overview/embodied-ai-roadmap">{translate({id: 'translationNotice.roadmap', message: '学习地图'})}</Link>,
        }}>
          {'本语言版本已提供 {readme}、{introduction}、{projects}和{roadmap}。其他教程章节与独立实验页目前仍为中文。'}
        </Translate>
      ) : (
        <Translate id="translationNotice.fallback" values={{
          introduction: <Link to="/docs/introduction/intro">{translate({id: 'translationNotice.introduction', message: '具身导论'})}</Link>,
          original: <Link target="_self" autoAddBaseUrl={false} to={`pathname://${createUrl({locale: i18n.defaultLocale, fullyQualified: false})}${search}${hash}`}>
            {translate({id: 'translationNotice.original', message: '阅读中文原文'})}
          </Link>,
        }}>
          {'本页目前仅提供中文。你可以阅读已翻译的{introduction}，或{original}。'}
        </Translate>
      )}
    </aside>
  );
}
