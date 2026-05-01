import React from 'react';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';

function HomepageHeader() {
  const { siteConfig } = useDocusaurusContext();
  return (
    <header className="hero">
      <div className="container">
        <h1 className="hero__title">{siteConfig.title}</h1>
        <p className="hero__subtitle">{siteConfig.tagline}</p>
        <p style={{ fontSize: '1rem', color: 'var(--ifm-color-emphasis-600)' }}>
          面向求职与转行的具身智能开源实践教程
          <br />
          基础知识打底 · 项目实践贯穿 · 求职能力收口
        </p>
        <div style={{ marginTop: '1.5rem' }}>
          <a
            className="button button--primary button--lg"
            href="/dw-dive-into-embodied-ai/docs/overview/intro"
          >
            开始学习
          </a>
          <a
            className="button button--secondary button--lg"
            href="https://github.com/datawhalechina/dw-dive-into-embodied-ai"
            style={{ marginLeft: '1rem' }}
          >
            GitHub
          </a>
        </div>
      </div>
    </header>
  );
}

export default function Home(): React.JSX.Element {
  const { siteConfig } = useDocusaurusContext();
  return (
    <Layout title={siteConfig.title} description={siteConfig.tagline}>
      <HomepageHeader />
      <main>
        <HomepageFeatures />
      </main>
    </Layout>
  );
}
