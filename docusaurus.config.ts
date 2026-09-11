import type { Config } from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';
import remarkMath from 'remark-math';
import rehypeMathjax from './src/rehype-mathjax-twopass.mjs';

const mathjaxConfig = {
  tex: {
    tags: 'ams',
    tagSide: 'right',
  },
};

const config: Config = {
  title: 'Dive into Embodied AI',
  tagline: '动手学具身智能',
  favicon: 'img/favicon.svg',

  url: 'https://datawhalechina.github.io',
  baseUrl: '/dive-into-embodied-ai/',
  trailingSlash: false,

  organizationName: 'datawhalechina',
  projectName: 'dive-into-embodied-ai',

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

  plugins: [
    function nativeCssLayers() {
      return {
        name: 'native-css-layers',
        configurePostCss(options) {
          // Stylesheets are processed separately. Polyfilling layers adds ID-level
          // specificity to Infima but not to our overrides. Keep native layers so
          // the browser can resolve the cascade across the complete stylesheet.
          options.plugins = options.plugins.map((plugin) => {
            if (Array.isArray(plugin) && String(plugin[0]).includes('postcss-preset-env')) {
              return [plugin[0], {...plugin[1], features: {...plugin[1]?.features, 'cascade-layers': false}}];
            }
            return plugin;
          });
          return options;
        },
      };
    },
  ],

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          remarkPlugins: [remarkMath],
          rehypePlugins: [[rehypeMathjax, mathjaxConfig]],
          showLastUpdateTime: true,
          showLastUpdateAuthor: true,
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themes: [
    '@docusaurus/theme-mermaid',
    [
      require.resolve('@easyops-cn/docusaurus-search-local'),
      {
        indexDocs: true,
        indexBlog: false,
        indexPages: true,
        language: ['zh', 'en'],
        hashed: true,
        highlightSearchTermsOnTargetPage: true,
        explicitSearchResultPath: true,
        searchResultLimits: 10,
        searchResultContextMaxLength: 80,
        searchBarShortcut: true,
        searchBarShortcutHint: true,
        searchBarShortcutKeymap: 'mod+k',
        searchBarPosition: 'right',
      },
    ],
  ],

  themeConfig: {
    colorMode: {
      defaultMode: 'light',
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: 'Dive into Embodied AI',
      logo: {
        alt: 'Embodied AI',
        src: 'img/logo.svg',
      },
      items: [
        {
          to: '/',
          label: '首页',
          position: 'left',
          className: 'navbar-center-start',
        },
        {
          type: 'custom-navbarMegaMenu',
          menuId: 'foundations',
          position: 'left',
        },
        {
          type: 'custom-navbarMegaMenu',
          menuId: 'practices',
          position: 'left',
        },
        {
          type: 'search',
          position: 'right',
        },
        {
          href: 'https://github.com/datawhalechina/dive-into-embodied-ai',
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
            { label: '理论基础', to: '/docs/foundations/intro' },
            { label: '项目实战', to: '/docs/practices/intro' },
          ],
        },
        {
          title: '社区',
          items: [
            {
              label: 'GitHub Discussions',
              href: 'https://github.com/datawhalechina/dive-into-embodied-ai/discussions',
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
              href: 'https://github.com/datawhalechina/dive-into-embodied-ai',
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
