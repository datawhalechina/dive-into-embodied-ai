const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const ts = require('typescript');
const domino = require('@mixmark-io/domino');

const root = path.resolve(__dirname, '..');
const read = file => fs.readFileSync(path.join(root, file), 'utf8');
const walk = dir => fs.readdirSync(path.join(root, dir), {withFileTypes: true})
  .flatMap(entry => entry.isDirectory() ? walk(`${dir}/${entry.name}`) : [`${dir}/${entry.name}`]);
const placeholders = text => [...new Set([...text.matchAll(/\{(\w+)\}/g)].map(match => match[1]))].sort();
const parse = file => domino.createDocument(read(file));
const elements = (node, selector) => Array.from(node.querySelectorAll(selector));
const attributes = (node, selector, name) => elements(node, selector).map(element => element.getAttribute(name));
const anchors = (node, selector = '[id]') => attributes(node, selector, 'id').sort();
// React useId values label interactive demos; only heading permalinks are shared URLs.
const sectionAnchors = node => elements(node, 'h2[id], h3[id], h4[id], h5[id], h6[id]')
  .filter(heading => heading.querySelector('.hash-link')).map(heading => heading.id).sort();
const hasHan = text => /\p{Script=Han}/u.test(text);

// Follow the configured locales so a future language cannot silently escape checks.
let locales;
let defaultLocale;
const config = ts.createSourceFile('docusaurus.config.ts', read('docusaurus.config.ts'), ts.ScriptTarget.Latest, true);
function readConfig(node) {
  if (ts.isPropertyAssignment(node)) {
    if (node.name.getText(config) === 'locales' && ts.isArrayLiteralExpression(node.initializer)) {
      locales = node.initializer.elements.map(element => element.text);
    }
    if (node.name.getText(config) === 'defaultLocale') defaultLocale = node.initializer.text;
  }
  ts.forEachChild(node, readConfig);
}
readConfig(config);
assert.equal(defaultLocale, 'zh-Hans');
assert.deepEqual(locales, ['zh-Hans', 'en', 'es', 'it']);
const translatedLocales = locales.filter(locale => locale !== defaultLocale);
const catalogs = Object.fromEntries(translatedLocales.map(locale => [locale, JSON.parse(read(`i18n/${locale}/code.json`))]));

const sources = [
  'src/pages/index.tsx',
  'src/components/HomepageFeatures/index.tsx',
  'src/components/HomepageProjects/index.tsx',
  'src/components/NavbarMegaMenu/data.ts',
  'src/components/PageMarkdown/index.tsx',
  'src/components/TranslationNotice/index.tsx',
  ...walk('src/theme/DocItem/TOC').filter(file => /\.tsx?$/.test(file)),
  ...walk('src/components/docs/introduction').filter(file => /\.tsx?$/.test(file)),
];
let checked = 0;
function checkMessage(file, key, message) {
  for (const locale of translatedLocales) {
    const translation = catalogs[locale][key]?.message;
    assert(translation?.trim(), `${file}: missing ${locale} translation for ${key}`);
    assert(!hasHan(translation), `${file}: untranslated ${locale} message: ${key}`);
    assert.deepEqual(placeholders(translation), placeholders(message), `${file}: mismatched ${locale} placeholders: ${key}`);
  }
  checked++;
}
for (const file of sources) {
  const source = ts.createSourceFile(file, read(file), ts.ScriptTarget.Latest, true);
  function visit(node) {
    if (ts.isCallExpression(node) && node.expression.getText(source) === 'translate') {
      const descriptor = node.arguments[0];
      assert(ts.isObjectLiteralExpression(descriptor), `${file}: translation must be statically extractable`);
      const fields = Object.fromEntries(descriptor.properties.filter(ts.isPropertyAssignment)
        .map(prop => [prop.name.getText(source), prop.initializer.text]));
      checkMessage(file, fields.id ?? fields.message, fields.message ?? '');
    }
    if (ts.isJsxElement(node) && node.openingElement.tagName.getText(source) === 'Translate') {
      const id = node.openingElement.attributes.properties.find(prop => prop.name?.getText(source) === 'id');
      const message = node.children.find(child => ts.isJsxExpression(child) && child.expression && ts.isStringLiteral(child.expression));
      assert(id && message, `${file}: Translate needs a static id and message`);
      checkMessage(file, id.initializer.text, message.expression.text);
    }
    ts.forEachChild(node, visit);
  }
  visit(source);
}

// Catch catalog drift, including navigation/footer/sidebar messages outside React.
for (const relative of ['code.json', 'docusaurus-theme-classic/navbar.json', 'docusaurus-theme-classic/footer.json', 'docusaurus-plugin-content-docs/current.json']) {
  const english = JSON.parse(read(`i18n/en/${relative}`));
  for (const locale of translatedLocales) {
    const messages = JSON.parse(read(`i18n/${locale}/${relative}`));
    for (const [key, value] of Object.entries(english)) {
      assert(messages[key]?.message?.trim(), `${locale}/${relative}: missing ${key}`);
      assert(!hasHan(messages[key].message), `${locale}/${relative}: untranslated ${key}`);
      assert.deepEqual(placeholders(messages[key].message), placeholders(value.message), `${locale}/${relative}: placeholders for ${key}`);
    }
  }
}
// Local search has no bundled Spanish or Italian catalog.
const searchKeys = Object.keys(JSON.parse(read('node_modules/@easyops-cn/docusaurus-search-local/dist/locales/zh-Hans.json')));
for (const locale of ['es', 'it']) {
  for (const key of searchKeys) assert(catalogs[locale][key]?.message, `${locale}: missing search translation ${key}`);
}

const baseUrl = '/dive-into-embodied-ai/';
const prefix = locale => locale === defaultLocale ? '' : `${locale}/`;
const chinese = parse('build/index.html');
const chineseMain = chinese.querySelector('main');
assert(chineseMain, 'The Chinese homepage must render on the server');
assert(!chinese.querySelector('[data-translation-notice]'), 'Chinese must not show a translation notice');
const docs = walk('build/docs').filter(file => file.endsWith('.html'));
const translatedPages = [
  ...walk('build/docs/introduction').filter(file => file.endsWith('.html')),
  'build/docs/practices/intro.html',
  'build/docs/overview/embodied-ai-roadmap.html',
];
assert(translatedPages.length > 2, 'Missing introduction pages');
for (const locale of locales) {
  const page = parse(`build/${prefix(locale)}index.html`);
  const main = page.querySelector('main');
  assert.equal(page.documentElement.lang, locale, `${locale}: wrong page language`);
  assert(main, `${locale}: homepage must render on the server`);
  assert.deepEqual(anchors(main), anchors(chineseMain), `${locale}: homepage section anchors differ`);
  for (const alternate of locales) {
    assert(attributes(page.querySelector('nav.navbar'), 'a[href]', 'href').includes(`${baseUrl}${prefix(alternate)}`), `${locale}: missing ${alternate} language link`);
  }
  const label = locale === defaultLocale ? '新手入门' : catalogs[locale]['新手入门'].message;
  const beginner = elements(main, 'a').find(link => link.textContent === label);
  assert(beginner, `${locale}: missing beginner entry`);
  assert.equal(beginner.getAttribute('href'), `${baseUrl}${prefix(locale)}docs/introduction/intro`);
  if (locale === defaultLocale) continue;
  assert(!hasHan(main.textContent), `${locale}: homepage contains Chinese text`);
  const notice = main.querySelector('[data-translation-notice="overview"]');
  assert(notice, `${locale}: missing translation scope notice`);
  assert.equal(notice.lang, locale);
  const scopeMessage = catalogs[locale]['translationNotice.overview'].message.replace(/\{(\w+)\}/g,
    (_, key) => key === 'readme' ? 'README' : catalogs[locale][`translationNotice.${key}`].message);
  assert.equal(notice.textContent, scopeMessage, `${locale}: incomplete scope notice`);

  for (const file of docs) {
    const alternate = file.replace('build/', `build/${locale}/`);
    assert(fs.existsSync(path.join(root, alternate)), `${locale}: missing tutorial route ${alternate}`);
  }
  const fallbackPath = 'docs/foundations/controllers/intro';
  const fallback = parse(`build/${locale}/${fallbackPath}.html`);
  const fallbackNotice = fallback.querySelector('[data-translation-notice="fallback"]');
  assert(fallbackNotice, `${locale}: missing fallback notice`);
  assert.equal(fallbackNotice.lang, locale);
  assert(!hasHan(fallbackNotice.textContent), `${locale}: fallback notice must use the chosen language`);
  assert(fallback.querySelector('[data-markdown-content] > div[lang="zh-Hans"]'), `${locale}: Chinese content needs its actual language for screen readers`);
  const original = elements(fallbackNotice, 'a').find(link => link.textContent === catalogs[locale]['translationNotice.original'].message);
  assert(original, `${locale}: missing original-language link`);
  assert.equal(original.getAttribute('href'), `${baseUrl}${fallbackPath}`);

  for (const file of translatedPages) {
    const source = parse(file);
    const target = parse(file.replace('build/', `build/${locale}/`));
    // Other entries in the learning-map sidebar can still be Chinese-only.
    const content = target.querySelector('[data-markdown-content]');
    assert(content, `${locale}/${file}: missing translated reading surface`);
    assert(!content.querySelector('[data-translation-notice="fallback"]'), `${locale}/${file}: translated page falls back to Chinese`);
    assert(!hasHan(content.textContent), `${locale}/${file}: translated page contains Chinese text`);
    assert.deepEqual(sectionAnchors(content), sectionAnchors(source.querySelector('[data-markdown-content]')), `${locale}/${file}: section anchors differ`);
    for (const href of attributes(content, 'a[href]', 'href')) {
      if (href.startsWith(baseUrl)) assert(href.startsWith(`${baseUrl}${locale}/`), `${locale}/${file}: link loses locale: ${href}`);
    }
    const sourcePath = new URL(source.querySelector('link[rel="canonical"]').href).pathname;
    for (const alternate of locales) {
      const expected = `${baseUrl}${prefix(alternate)}${sourcePath.slice(baseUrl.length)}`;
      assert(attributes(target.querySelector('nav.navbar'), 'a[href]', 'href').includes(expected), `${locale}/${file}: missing language switch to ${alternate}`);
    }
  }
  for (const href of attributes(main, 'a[href]', 'href')) {
    if (href.startsWith(baseUrl)) assert(href.startsWith(`${baseUrl}${locale}/`), `${locale}: homepage link loses locale: ${href}`);
  }
}
console.log(`i18n checks passed: ${locales.length} locales, ${checked} UI messages per translated locale, ${translatedPages.length} translated pages per locale, matching anchors and language links, and ${docs.length} tutorial routes per locale.`);
