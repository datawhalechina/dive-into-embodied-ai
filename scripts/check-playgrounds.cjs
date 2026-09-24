const fs = require('node:fs');
const path = require('node:path');

const root = path.resolve(__dirname, '..');

const requiredFiles = [
  'src/components/PlaygroundHeader/index.jsx',
  'src/components/docs/projects/cs123/pid-control/PdPlayground/index.jsx',
  'src/components/docs/projects/cs123/forward-kinematics/FkPlayground/index.jsx',
  'src/components/docs/projects/cs123/inverse-kinematics/IkPlayground/index.jsx',
  'src/pages/cs123/pd-playground.jsx',
  'src/pages/cs123/fk-playground.jsx',
  'src/pages/cs123/ik-playground.jsx',
  'src/theme/DocItem/TOC/Desktop/index.tsx',
  'src/theme/DocItem/TOC/Mobile/index.tsx',
  'src/theme/DocItem/TOC/Mobile/styles.module.css',
];

const expectedText = [
  {
    file: 'src/components/PlaygroundHeader/index.jsx',
    snippets: [
      "readingHref: '/docs/practices/quadruped/cs123/pid-control'",
      "readingHref: '/docs/practices/quadruped/cs123/forward-kinematics'",
      "readingHref: '/docs/practices/quadruped/cs123/inverse-kinematics'",
    ],
  },
  {
    file: 'docs/practices/quadruped/cs123/1.pid-control.mdx',
    snippets: ['/cs123/pd-playground'],
  },
  {
    file: 'docs/practices/quadruped/cs123/2.forward-kinematics.md',
    snippets: ['/cs123/fk-playground'],
  },
  {
    file: 'docs/practices/quadruped/cs123/3.inverse-kinematics.md',
    snippets: ['/cs123/ik-playground'],
  },
  {
    file: 'src/theme/DocItem/TOC/Desktop/index.tsx',
    snippets: ['function getInteractiveHref', 'currentPath.endsWith(readingPath)', "id: 'theme.DocItem.interactiveMode'"],
  },
  {
    file: 'src/theme/DocItem/TOC/Mobile/index.tsx',
    snippets: ['function getInteractiveHref', 'currentPath.endsWith(readingPath)', "id: 'theme.DocItem.interactiveMode'"],
  },
];

const forbiddenText = [
  {
    file: 'src/pages/cs123/pd-playground.jsx',
    snippets: ['FloatingOrbs'],
  },
  {
    file: 'src/pages/cs123/fk-playground.jsx',
    snippets: ['FloatingOrbs'],
  },
  {
    file: 'src/pages/cs123/ik-playground.jsx',
    snippets: ['FloatingOrbs'],
  },
  {
    file: 'src/components/PlaygroundHeader/index.jsx',
    snippets: ['/docs/projects/cs123', '/tutorial/rl/dqn-playground'],
  },
];

const failures = [];

for (const relativeFile of requiredFiles) {
  const absoluteFile = path.join(root, relativeFile);
  if (!fs.existsSync(absoluteFile)) {
    failures.push(`missing file: ${relativeFile}`);
  }
}

for (const {file, snippets} of expectedText) {
  const absoluteFile = path.join(root, file);
  if (!fs.existsSync(absoluteFile)) {
    failures.push(`missing text target: ${file}`);
    continue;
  }
  const content = fs.readFileSync(absoluteFile, 'utf8');
  for (const snippet of snippets) {
    if (!content.includes(snippet)) {
      failures.push(`missing snippet in ${file}: ${snippet}`);
    }
  }
}

for (const {file, snippets} of forbiddenText) {
  const absoluteFile = path.join(root, file);
  if (!fs.existsSync(absoluteFile)) {
    continue;
  }
  const content = fs.readFileSync(absoluteFile, 'utf8');
  for (const snippet of snippets) {
    if (content.includes(snippet)) {
      failures.push(`forbidden snippet in ${file}: ${snippet}`);
    }
  }
}

if (failures.length > 0) {
  console.error('Playground migration check failed:');
  for (const failure of failures) {
    console.error(`- ${failure}`);
  }
  process.exit(1);
}

console.log('Playground migration check passed.');
