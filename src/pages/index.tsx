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
          给想入门、转行或找相关工作的同学准备
          <br />
          先补基础，再做项目，最后整理到简历和面试
        </p>
        <div style={{ marginTop: '1.5rem' }}>
          <a
            className="button button--primary button--lg"
            href="/dive-into-embodied-ai/docs/overview/intro"
          >
            开始学习
          </a>
          <a
            className="button button--secondary button--lg"
            href="https://github.com/datawhalechina/dive-into-embodied-ai"
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
