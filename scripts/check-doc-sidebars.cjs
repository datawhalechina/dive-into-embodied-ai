const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const ts = require('typescript');

const rootDir = path.resolve(__dirname, '..');

function loadSidebars() {
  const sidebarsPath = path.join(rootDir, 'sidebars.ts');
  const source = fs.readFileSync(sidebarsPath, 'utf8');
  const compiled = ts.transpileModule(source, {
    compilerOptions: {
      module: ts.ModuleKind.CommonJS,
      esModuleInterop: true,
      target: ts.ScriptTarget.ES2020,
    },
  }).outputText;
  const module = {exports: {}};
  const requireFromRoot = (specifier) => {
    if (specifier.startsWith('.')) {
      return require(path.join(path.dirname(sidebarsPath), specifier));
    }
    return require(specifier);
  };
  const fn = new Function('require', 'module', 'exports', compiled);
  fn(requireFromRoot, module, module.exports);
  return module.exports.default ?? module.exports;
}

function collectSidebarRefs(items, refs = []) {
  for (const item of items) {
    if (typeof item === 'string') {
      refs.push(item);
      continue;
    }
    if (!item || typeof item !== 'object') {
      continue;
    }
    if (item.type === 'doc' && item.id) {
      refs.push(item.id);
    }
    if (item.type === 'link' && item.href) {
      refs.push(item.href);
    }
    if (item.link?.type === 'doc' && item.link.id) {
      refs.push(item.link.id);
    }
    if (Array.isArray(item.items)) {
      collectSidebarRefs(item.items, refs);
    }
  }
  return refs;
}

function listDocs(dir) {
  return fs.readdirSync(dir, {withFileTypes: true}).flatMap((entry) => {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      return listDocs(fullPath);
    }
    if (/\.mdx?$/.test(entry.name)) {
      return [fullPath];
    }
    return [];
  });
}

function assertDisplayedSidebar(docPath, sidebarId) {
  const source = fs.readFileSync(docPath, 'utf8');
  assert.match(
    source,
    new RegExp(`^displayed_sidebar:\\s*${sidebarId}\\s*$`, 'm'),
    `${path.relative(rootDir, docPath)} must render with ${sidebarId}`,
  );
}

function assertSidebarCase(sidebars, sidebarCase) {
  const sidebar = sidebars[sidebarCase.sidebarId];
  assert.ok(sidebar, `${sidebarCase.sidebarId} should exist`);

  const refs = collectSidebarRefs(sidebar);
  assert.ok(
    refs.includes(sidebarCase.entryRef),
    `${sidebarCase.sidebarId} should link to ${sidebarCase.entryRef}`,
  );
  for (const forbiddenRef of sidebarCase.forbiddenRefs) {
    assert.ok(
      refs.every((ref) => !ref.includes(forbiddenRef)),
      `${sidebarCase.sidebarId} must not include ${forbiddenRef}: ${refs.join(', ')}`,
    );
  }
  for (const docPath of listDocs(path.join(rootDir, sidebarCase.docDir))) {
    assertDisplayedSidebar(docPath, sidebarCase.sidebarId);
  }
}

const sidebars = loadSidebars();

assert.ok(sidebars.foundationsOverviewSidebar, 'foundationsOverviewSidebar should exist');

assertDisplayedSidebar(
  path.join(rootDir, 'docs/foundations/intro.md'),
  'foundationsOverviewSidebar',
);

const foundationsCases = [
  {
    sidebarId: 'foundationsEmbodiedAiIntroSidebar',
    docDir: 'docs/foundations/embodied-ai-intro',
    entryRef: 'foundations/embodied-ai-intro/placeholder',
    forbiddenRefs: ['foundations/robotics-and-ros2', 'foundations/simulation', 'foundations/vla'],
  },
  {
    sidebarId: 'foundationsRoboticsAndRos2Sidebar',
    docDir: 'docs/foundations/robotics-and-ros2',
    entryRef: 'foundations/robotics-and-ros2/intro',
    forbiddenRefs: ['foundations/simulation', 'foundations/rl-for-robotics', 'foundations/vla'],
  },
  {
    sidebarId: 'foundationsSimulationSidebar',
    docDir: 'docs/foundations/simulation',
    entryRef: 'foundations/simulation/intro',
    forbiddenRefs: ['foundations/robotics-and-ros2', 'foundations/rl-for-robotics', 'foundations/vla'],
  },
  {
    sidebarId: 'foundationsRlForRoboticsSidebar',
    docDir: 'docs/foundations/rl-for-robotics',
    entryRef: 'foundations/rl-for-robotics/intro',
    forbiddenRefs: ['foundations/robotics-and-ros2', 'foundations/simulation', 'foundations/vla'],
  },
  {
    sidebarId: 'foundationsVlmSidebar',
    docDir: 'docs/foundations/vlm',
    entryRef: 'foundations/vlm/intro',
    forbiddenRefs: ['foundations/robotics-and-ros2', 'foundations/vla', 'foundations/world-model'],
  },
  {
    sidebarId: 'foundationsVlaSidebar',
    docDir: 'docs/foundations/vla',
    entryRef: 'foundations/vla/vla-intro',
    forbiddenRefs: ['foundations/robotics-and-ros2', 'foundations/vlm', 'foundations/world-model'],
  },
  {
    sidebarId: 'foundationsWorldModelSidebar',
    docDir: 'docs/foundations/world-model',
    entryRef: 'foundations/world-model/intro',
    forbiddenRefs: ['foundations/robotics-and-ros2', 'foundations/vlm', 'foundations/vla'],
  },
];

for (const sidebarCase of foundationsCases) {
  assertSidebarCase(sidebars, sidebarCase);
}

console.log('Foundation sidebars are isolated.');
