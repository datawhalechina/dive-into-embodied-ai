const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const ts = require('typescript');

const root = path.resolve(__dirname, '..');
const read = file => fs.readFileSync(path.join(root, file), 'utf8');
const translations = JSON.parse(read('i18n/en/code.json'));
const walk = dir => fs.readdirSync(path.join(root, dir), {withFileTypes: true})
  .flatMap(entry => entry.isDirectory() ? walk(`${dir}/${entry.name}`) : [`${dir}/${entry.name}`]);
const placeholders = text => [...new Set([...text.matchAll(/\{(\w+)\}/g)].map(match => match[1]))].sort();

// Catch untranslated additions, including controls that only appear after interaction.
const sources = [
  'src/pages/index.tsx',
  'src/components/HomepageFeatures/index.tsx',
  'src/components/HomepageProjects/index.tsx',
  'src/components/NavbarMegaMenu/data.ts',
  ...walk('src/features/learning-map').filter(file => /\.tsx?$/.test(file)),
];
let checked = 0;
for (const file of sources) {
  const source = ts.createSourceFile(file, read(file), ts.ScriptTarget.Latest, true);
  function visit(node) {
    if (ts.isCallExpression(node) && node.expression.getText(source) === 'translate') {
      const descriptor = node.arguments[0];
      assert(ts.isObjectLiteralExpression(descriptor), `${file}: translation must be statically extractable`);
      const fields = Object.fromEntries(descriptor.properties
        .filter(ts.isPropertyAssignment)
        .map(prop => [prop.name.getText(source), prop.initializer.text]));
      const key = fields.id ?? fields.message;
      assert(translations[key]?.message !== undefined, `${file}: missing English translation for ${key}`);
      assert(!/\p{Script=Han}/u.test(translations[key].message), `${file}: untranslated English message: ${key}`);
      assert.deepEqual(placeholders(translations[key].message), placeholders(fields.message ?? ''), `${file}: mismatched placeholders: ${key}`);
      checked++;
    }
    ts.forEachChild(node, visit);
  }
  visit(source);
}

const chinese = read('build/learning-map.html');
const english = read('build/en/learning-map.html');
const main = html => html.match(/<main\b[^>]*>([\s\S]*?)<\/main>/)?.[1] ?? '';
const text = html => html.replace(/<script\b[^>]*>[\s\S]*?<\/script>/g, '').replace(/<[^>]*>/g, '');
// Production minification may omit quotes around HTML attribute values.
const attributes = (html, name) => [...html.matchAll(new RegExp(`\\s${name}=(?:"([^"]*)"|'([^']*)'|([^\\s>]+))`, 'g'))]
  .map(match => match[1] ?? match[2] ?? match[3]);
assert.equal(attributes(chinese.match(/<html[^>]*>/)[0], 'lang')[0], 'zh-Hans');
assert.equal(attributes(english.match(/<html[^>]*>/)[0], 'lang')[0], 'en');
assert(chinese.includes('具身智能学习地图'));
assert(english.includes('Embodied AI learning map'));
assert(main(english), 'The English learning map must render on the server');
assert(!/\p{Script=Han}/u.test(text(main(english))), 'The rendered English learning map contains Chinese text');
assert(!chinese.includes('Translation status'), 'The Chinese page should not show an English translation notice');
assert(english.includes('Other tutorial chapters and standalone playgrounds are currently in Chinese.'));

const anchors = html => attributes(main(html), 'id').sort();
assert.deepEqual(anchors(english), anchors(chinese), 'Language switching must preserve section anchors');
assert(attributes(english, 'href').includes('/dive-into-embodied-ai/learning-map'), 'Missing Chinese language link');
assert(attributes(chinese, 'href').includes('/dive-into-embodied-ai/en/learning-map'), 'Missing English language link');

// Every Chinese tutorial must remain reachable after switching to English.
const docs = walk('build/docs').filter(file => file.endsWith('.html'));
for (const file of docs) {
  const alternate = file.replace('build/', 'build/en/');
  assert(fs.existsSync(path.join(root, alternate)), `Missing alternate route: ${alternate}`);
}
const fallback = read('build/en/docs/foundations/controllers/intro.html');
assert(fallback.includes('This page is currently available in Chinese.'));
assert(/<div lang=(?:"zh-Hans"|zh-Hans)>/.test(fallback), 'Fallback content needs its actual language for screen readers');
const originalLink = fallback.match(/<a\b[^>]*>read the Chinese version<\/a>/)?.[0];
assert(originalLink, 'Missing original-language link in the fallback notice');
assert.equal(attributes(originalLink, 'href')[0], '/dive-into-embodied-ai/docs/foundations/controllers/intro');

// The introduction is translated, so its English pages must not fall back to Chinese.
const headingIds = html => [...main(html).matchAll(/<h[2-6]\b[^>]*>/g)].flatMap(([tag]) => attributes(tag, 'id')).sort();
const introductionPages = walk('build/docs/introduction').filter(file => file.endsWith('.html'));
assert(introductionPages.length > 0, 'Missing introduction pages');
for (const file of introductionPages) {
  const chinesePage = read(file);
  const englishPage = read(file.replace('build/', 'build/en/'));
  assert(!englishPage.includes('This page is currently available in Chinese.'), `${file}: the English introduction falls back to Chinese`);
  assert(!/\p{Script=Han}/u.test(text(main(englishPage))), `${file}: the English introduction contains Chinese text`);
  assert.deepEqual(headingIds(englishPage), headingIds(chinesePage), `${file}: section anchors differ between languages`);
}

for (const file of ['build/en/index.html', 'build/en/learning-map.html']) {
  const html = read(file);
  // Internal content links should stay in the selected locale.
  for (const href of attributes(main(html), 'href')) {
    if (href.startsWith('/dive-into-embodied-ai/')) {
      assert(href.startsWith('/dive-into-embodied-ai/en/'), `${file}: link loses the selected language: ${href}`);
    }
  }
}

console.log(`i18n checks passed: ${checked} UI messages, matching learning-map anchors, ${introductionPages.length} translated introduction pages, and ${docs.length} alternate tutorial routes.`);
