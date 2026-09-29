const fs = require('node:fs');
const path = require('node:path');
const domino = require('@mixmark-io/domino');

const rootDir = path.resolve(__dirname, '..');
const buildDir = path.join(rootDir, 'build');

function listHtmlFiles(dir) {
  return fs.readdirSync(dir, {withFileTypes: true}).flatMap((entry) => {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      return listHtmlFiles(fullPath);
    }
    return entry.isFile() && entry.name.endsWith('.html') ? [fullPath] : [];
  });
}

if (!fs.existsSync(buildDir)) {
  console.error('Missing build directory. Run `npm run build` before `npm run check:math-refs`.');
  process.exit(1);
}

const failures = [];
for (const filePath of listHtmlFiles(buildDir)) {
  const html = fs.readFileSync(filePath, 'utf8');
  const document = domino.createDocument(html);
  const brokenRefs = document.querySelectorAll('a[href="#"] .MathJax_ref');
  if (brokenRefs.length > 0) {
    failures.push({
      file: path.relative(rootDir, filePath),
      count: brokenRefs.length,
    });
  }
}

if (failures.length > 0) {
  console.error('Broken MathJax equation references found:');
  for (const failure of failures) {
    console.error(`- ${failure.file}: ${failure.count} unresolved reference(s)`);
  }
  process.exit(1);
}

console.log('MathJax equation references are resolved.');
