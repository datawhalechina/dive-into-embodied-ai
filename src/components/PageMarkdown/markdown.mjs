import TurndownService from 'turndown';
import {gfm} from 'turndown-plugin-gfm';

/** Convert a copy of the reading surface, never the live interactive page. */
export function pageToMarkdown(element, pageUrl, title = '') {
  const root = element.cloneNode(true);
  const doc = root.ownerDocument;
  const absoluteUrl = value => {
    try { return new URL(value, pageUrl).href; } catch { return value; }
  };

  // MathJax renders SVG paths, so retain the original TeX before removing graphics.
  for (const math of Array.from(root.querySelectorAll('[data-markdown-math]'))) {
    const replacement = doc.createElement(math.getAttribute('display') === 'true' ? 'div' : 'span');
    replacement.setAttribute('data-markdown-math', math.getAttribute('data-markdown-math'));
    replacement.textContent = math.getAttribute('data-markdown-math');
    math.replaceWith(replacement);
  }

  root.querySelectorAll([
    '[data-markdown-exclude]', '[hidden]', '[aria-hidden="true"]',
    'script', 'style', 'nav', 'button', 'select', 'textarea',
    'input:not([type="checkbox"])', '.hash-link', '[role="tablist"]',
    '.theme-doc-footer', '.pagination-nav', '.table-of-contents',
  ].join(',')).forEach(node => node.remove());

  // Highlighted code is split into token lines without literal newline characters.
  for (const pre of Array.from(root.querySelectorAll('pre'))) {
    const code = pre.querySelector('code');
    if (!code) continue;
    const language = [code.className, pre.className, pre.closest('[class*="language-"]')?.className]
      .join(' ').match(/\blanguage-([\w+-]+)/)?.[1] ?? '';
    pre.setAttribute('data-language', language);
    const lines = code.querySelectorAll('.token-line');
    if (lines.length) code.textContent = Array.from(lines, line => line.textContent).join('\n');
  }

  for (const link of Array.from(root.querySelectorAll('a[href]'))) {
    link.setAttribute('href', absoluteUrl(link.getAttribute('href')));
  }
  for (const img of Array.from(root.querySelectorAll('img[src]'))) {
    img.setAttribute('src', absoluteUrl(img.getAttribute('src')));
  }
  for (const graphic of Array.from(root.querySelectorAll('svg, canvas'))) {
    const label = graphic.getAttribute('aria-label') || graphic.querySelector('title')?.textContent;
    graphic.replaceWith(doc.createTextNode(label || ''));
  }

  const converter = new TurndownService({
    headingStyle: 'atx', codeBlockStyle: 'fenced', bulletListMarker: '-', emDelimiter: '*',
  });
  converter.use(gfm);
  converter.addRule('math', {
    filter: node => node.hasAttribute('data-markdown-math'),
    replacement: (_content, node) => {
      const tex = node.getAttribute('data-markdown-math');
      return node.nodeName === 'DIV' ? `\n\n$$\n${tex}\n$$\n\n` : `$${tex}$`;
    },
  });
  converter.addRule('code', {
    filter: node => node.nodeName === 'PRE' && !!node.querySelector('code'),
    replacement: (_content, node) => {
      const code = node.querySelector('code').textContent.replace(/\n$/, '');
      const fence = '`'.repeat(Math.max(3, ...Array.from(code.matchAll(/`+/g), match => match[0].length + 1)));
      return `\n\n${fence}${node.getAttribute('data-language')}\n${code}\n${fence}\n\n`;
    },
  });
  converter.addRule('tableCell', {
    filter: ['th', 'td'],
    replacement: (content, node) => {
      const value = content.trim().replace(/\|/g, '\\|').replace(/ *\n+/g, '<br>');
      return `${node.previousElementSibling ? ' ' : '| '}${value} |`;
    },
  });
  converter.addRule('strikethrough', {
    filter: ['del', 's', 'strike'],
    replacement: content => `~~${content}~~`,
  });
  converter.addRule('media', {
    filter: ['video', 'audio', 'iframe'],
    replacement: (_content, node) => {
      const src = node.getAttribute('src') || node.querySelector('source')?.getAttribute('src');
      return src ? `\n\n[${node.getAttribute('title') || node.getAttribute('aria-label') || node.nodeName.toLowerCase()}](${absoluteUrl(src)})\n\n` : '';
    },
  });

  const body = converter.turndown(root).trim();
  if (!body) throw new Error('No page content');
  const heading = !root.querySelector('h1') && title ? `# ${converter.escape(title)}\n\n` : '';
  return `${heading}${body}\n\n---\n\n<${pageUrl}>\n`;
}
