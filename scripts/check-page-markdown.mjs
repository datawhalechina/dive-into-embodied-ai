import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import domino from '@mixmark-io/domino';
import {pageToMarkdown} from '../src/components/PageMarkdown/markdown.mjs';

const url = 'https://datawhalechina.github.io/dive-into-embodied-ai/en/docs/example';
const parse = html => domino.createDocument(`<main>${html}</main>`).querySelector('main');
const content = parse(`
  <nav>Site navigation</nav><div data-markdown-exclude>Copy menu</div>
  <aside>This page is currently available in Chinese.</aside>
  <h1>Markdown 导出</h1><h2 id="details">Details<a class="hash-link" href="#details">#</a></h2>
  <p>Read <a href="next#section">next</a> and <a href="#details">details</a>.</p>
  <img src="/dive-into-embodied-ai/img/robot.webp" alt="Robot">
  <ul><li>One</li><li><strong>Two</strong><ul><li>Nested</li></ul></li></ul>
  <blockquote>Keep this note.</blockquote>
  <div class="language-python"><pre><code><div class="token-line">if x:</div><div class="token-line">    print(1)</div><div class="token-line"><br></div><div class="token-line">print(2)</div></code></pre><button>Copy code</button></div>
  <pre><code>const fence = \`\`\`;</code></pre>
  <p>Inline <mjx-container data-markdown-math="x_i + \\alpha"><svg><text>SVG noise</text></svg></mjx-container>.</p>
  <mjx-container display="true" data-markdown-math="\\begin{aligned}a&amp;=b\\\\c&amp;=d\\end{aligned}"><svg></svg></mjx-container>
  <table><thead><tr><th>Key</th><th>Value</th></tr></thead><tbody><tr><td>a | b</td><td>First<br>Second</td></tr></tbody></table>
  <p><del>Old</del> <code>a_b</code></p><p hidden>Hidden panel</p><svg aria-hidden="true"><text>Icon noise</text></svg>
`);
const original = content.outerHTML;
const markdown = pageToMarkdown(content, url);
assert.equal(content.outerHTML, original, 'Export must not mutate the live page');
for (const expected of [
  '# Markdown 导出', '## Details', 'This page is currently available in Chinese.',
  '[next](https://datawhalechina.github.io/dive-into-embodied-ai/en/docs/next#section)',
  `[details](${url}#details)`, '![Robot](https://datawhalechina.github.io/dive-into-embodied-ai/img/robot.webp)',
  '-   One', '**Two**', 'Nested', '> Keep this note.',
  '```python\nif x:\n    print(1)\n\nprint(2)\n```',
  '````\nconst fence = ```;\n````', '$x_i + \\alpha$',
  '$$\n\\begin{aligned}a&=b\\\\c&=d\\end{aligned}\n$$',
  '| Key | Value |\n| --- | --- |', 'a \\| b', 'First<br>Second', '~~Old~~', '`a_b`', `<${url}>`,
]) assert(markdown.includes(expected), `Missing Markdown content: ${expected}\n${markdown}`);
for (const excluded of ['Site navigation', 'Copy menu', 'Copy code', 'SVG noise', 'Icon noise', 'Hidden panel', 'hash-link']) {
  assert(!markdown.includes(excluded), `Export leaked page chrome: ${excluded}`);
}
assert(pageToMarkdown(parse('<p>Playground description</p>'), url, 'Playground').startsWith('# Playground\n'));
assert.throws(() => pageToMarkdown(parse('<button>Only chrome</button>'), url));

// Exercise the actual generated HTML, including MDX tables and MathJax output.
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const pages = [
  'build/index.html', 'build/en/index.html',
  'build/docs/introduction/intro.html', 'build/en/docs/introduction/intro.html',
  'build/en/docs/foundations/controllers/intro.html',
];
for (const file of pages) {
  const document = domino.createDocument(fs.readFileSync(path.join(root, file), 'utf8'));
  const surface = document.querySelector('[data-markdown-content]');
  assert(surface, `${file}: missing reading surface`);
  const result = pageToMarkdown(surface, url);
  assert(result.includes('# '), `${file}: missing heading`);
  assert(!result.includes('Markdown 选项') && !result.includes('Markdown options'), `${file}: included copy controls`);
  if (file.includes('controllers')) assert(result.includes('This page is currently available in Chinese.'));
  if (file === 'build/en/index.html') assert(!/\p{Script=Han}/u.test(result));
}
console.log('Markdown export checks passed (formatting, code, math, links, page chrome, locales, and live DOM preservation).');
