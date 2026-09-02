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

const sidebars = loadSidebars();

assert.ok(sidebars.practicesOverviewSidebar, 'practicesOverviewSidebar should exist');
assert.ok(sidebars.practicesCs123CourseSidebar, 'practicesCs123CourseSidebar should exist');
assert.ok(sidebars.practicesLerobotCourseSidebar, 'practicesLerobotCourseSidebar should exist');
assert.ok(sidebars.practicesSo101CourseSidebar, 'practicesSo101CourseSidebar should exist');
assert.ok(sidebars.practicesAmdSidebar, 'practicesAmdSidebar should exist');

const overviewCategories = sidebars.practicesOverviewSidebar.filter(
  (item) => item && typeof item === 'object' && item.type === 'category',
);
assert.deepEqual(
  overviewCategories.map((category) => category.label),
  ['AMD 专区', '仿真实战', '真机实战'],
  'Practice overview should use the three top-level project groups',
);

const cs123Refs = collectSidebarRefs(sidebars.practicesCs123CourseSidebar);
assert.ok(
  cs123Refs.includes('practices/quadruped/cs123/intro'),
  'CS123 course sidebar should link to its course overview',
);
assert.ok(
  cs123Refs.every((ref) => !ref.includes('practices/robot-arm')),
  `CS123 sidebar must not include robot-arm docs: ${cs123Refs.join(', ')}`,
);

const lerobotRefs = collectSidebarRefs(sidebars.practicesLerobotCourseSidebar);
assert.ok(
  lerobotRefs.includes('practices/robot-arm/data-collection/lerobot-course/index'),
  'LeRobot course sidebar should include its course document',
);
assert.ok(
  lerobotRefs.every((ref) => !ref.includes('practices/quadruped')),
  `LeRobot sidebar must not include quadruped docs: ${lerobotRefs.join(', ')}`,
);

const so101Refs = collectSidebarRefs(sidebars.practicesSo101CourseSidebar);
assert.deepEqual(
  so101Refs,
  ['practices/robot-arm/data-collection/so101-lerobot-real/index'],
  'SO-101 course sidebar should only include its own document',
);

const amdRefs = collectSidebarRefs(sidebars.practicesAmdSidebar);
assert.deepEqual(
  amdRefs,
  [
    'practices/amd/intro',
    'practices/amd/aup-learning-cloud',
    'practices/amd/microduck-rl/index',
    'practices/amd/vla-act/index',
    'practices/amd/pupper-control/intro',
    'practices/amd/pupper-control/locomotion',
    'practices/amd/pupper-control/vla',
  ],
  'AMD sidebar should only include its overview, cloud guide, MicroDuck, ACT tutorial, and Pupper project',
);

for (const docPath of listDocs(path.join(rootDir, 'docs/practices/quadruped/cs123'))) {
  assertDisplayedSidebar(docPath, 'practicesCs123CourseSidebar');
}

assertDisplayedSidebar(
  path.join(rootDir, 'docs/practices/robot-arm/data-collection/lerobot-course/index.md'),
  'practicesLerobotCourseSidebar',
);

assertDisplayedSidebar(
  path.join(rootDir, 'docs/practices/robot-arm/data-collection/so101-lerobot-real/index.md'),
  'practicesSo101CourseSidebar',
);

for (const docPath of listDocs(path.join(rootDir, 'docs/practices/amd'))) {
  assertDisplayedSidebar(docPath, 'practicesAmdSidebar');
}

console.log('Practice course sidebars are isolated.');
