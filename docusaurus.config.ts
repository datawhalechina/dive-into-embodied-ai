import type { Config } from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';
import remarkMath from 'remark-math';
import rehypeMathjax from 'rehype-mathjax';

const config: Config = {
  title: 'Dive into Embodied AI',
  tagline: '动手学具身智能',
  favicon: 'img/favicon.ico',

  url: 'https://datawhalechina.github.io',
  baseUrl: '/dw-dive-into-embodied-ai/',
  trailingSlash: false,

  organizationName: 'datawhalechina',
  projectName: 'dw-dive-into-embodied-ai',

  onBrokenLinks: 'throw',

  future: {
    faster: true,
    v4: true,
  },

  i18n: {
    defaultLocale: 'zh-Hans',
    locales: ['zh-Hans'],
  },

  markdown: {
    mermaid: true,
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeMathjax],
          showLastUpdateTime: true,
          showLastUpdateAuthor: true,
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themes: ['@docusaurus/theme-mermaid'],

  themeConfig: {
    colorMode: {
      defaultMode: 'light',
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: 'Dive into Embodied AI',
      logo: {
        alt: 'Logo',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'overviewSidebar',
          position: 'left',
          label: '总览',
        },
        {
          type: 'docSidebar',
          sidebarId: 'foundationsSidebar',
          position: 'left',
          label: '基础篇',
        },
        {
          type: 'docSidebar',
          sidebarId: 'practicesSidebar',
          position: 'left',
          label: '实践篇',
        },
        {
          type: 'docSidebar',
          sidebarId: 'careerSidebar',
          position: 'left',
          label: '求职篇',
        },
        {
          href: 'https://github.com/datawhalechina/dw-dive-into-embodied-ai',
          position: 'right',
          className: 'header-github-link',
          'aria-label': 'GitHub repository',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: '内容',
          items: [
            { label: '总览', to: '/docs/overview/intro' },
            { label: '基础篇', to: '/docs/foundations/intro' },
            { label: '实践篇', to: '/docs/practices/intro' },
            { label: '求职篇', to: '/docs/career/intro' },
          ],
        },
        {
          title: '社区',
          items: [
            {
              label: 'GitHub Discussions',
              href: 'https://github.com/datawhalechina/dw-dive-into-embodied-ai/discussions',
            },
            {
              label: 'Datawhale',
              href: 'https://datawhale.club',
            },
          ],
        },
        {
          title: '更多',
          items: [
            {
              label: 'GitHub',
              href: 'https://github.com/datawhalechina/dw-dive-into-embodied-ai',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Datawhale. Built with Docusaurus.`,
    },
    prism: {
      theme: require('prism-react-renderer').themes.github,
      darkTheme: require('prism-react-renderer').themes.dracula,
      additionalLanguages: ['python', 'bash', 'yaml', 'json'],
    },
    docs: {
      sidebar: {
        hideable: true,
        autoCollapseCategories: true,
      },
    },
  } satisfies Preset.ThemeConfig,
};

export default config;
