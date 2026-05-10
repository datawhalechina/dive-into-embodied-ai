const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const rootDir = path.resolve(__dirname, '..');
const cs123Dir = path.join(rootDir, 'docs/practices/quadruped/cs123');
const cs123FigsDir = path.join(cs123Dir, 'figs');
const cs123PidDocPath = path.join(cs123Dir, '1.pid-control.mdx');
const cs123FkDocPath = path.join(cs123Dir, '2.forward-kinematics.md');
const requiredCs123Assets = [
  'actuator-workflow.webp',
  'bang-bang-control.webp',
  'bang-bang-limit-cycle.webp',
  'closed-loop-control.webp',
  'coordinate-frames.webp',
  'integral-windup.webp',
  'open-loop-control.webp',
  'pd-spring-damper.webp',
  'pid-three-terms.webp',
  'joint-link-ee.webp',
];

const customCssPath = path.join(rootDir, 'src/css/custom.css');
const figureComponentPath = path.join(rootDir, 'src/components/Figure/index.tsx');
const docTableComponentPath = path.join(rootDir, 'src/components/DocTable/index.tsx');

function listFiles(dir, predicate = () => true) {
  return fs.readdirSync(dir, {withFileTypes: true}).flatMap((entry) => {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      return listFiles(fullPath, predicate);
    }
    return predicate(fullPath) ? [fullPath] : [];
  });
}

function isGitLfsPointer(bytes) {
  return bytes
    .subarray(0, 48)
    .toString('utf8')
    .startsWith('version https://git-lfs.github.com/spec/v1');
}

function assertImageBytes(filePath) {
  const ext = path.extname(filePath).toLowerCase();
  const bytes = fs.readFileSync(filePath);

  assert.ok(!isGitLfsPointer(bytes), `${path.relative(rootDir, filePath)} is a Git LFS pointer, not an image`);

  if (ext === '.gif') {
    const header = bytes.subarray(0, 6).toString('ascii');
    assert.ok(header === 'GIF87a' || header === 'GIF89a', `${path.relative(rootDir, filePath)} is not a valid GIF`);
    return;
  }

  if (ext === '.png') {
    assert.deepEqual(
      Array.from(bytes.subarray(0, 8)),
      [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a],
      `${path.relative(rootDir, filePath)} is not a valid PNG`,
    );
    return;
  }

  if (ext === '.webp') {
    assert.equal(bytes.subarray(0, 4).toString('ascii'), 'RIFF', `${path.relative(rootDir, filePath)} is not a RIFF file`);
    assert.equal(bytes.subarray(8, 12).toString('ascii'), 'WEBP', `${path.relative(rootDir, filePath)} is not a valid WebP`);
    return;
  }

  if (ext === '.svg') {
    assert.match(bytes.toString('utf8', 0, 256), /<svg|<\?xml/, `${path.relative(rootDir, filePath)} is not a valid SVG`);
  }
}

function collectImageRefs(docPath) {
  const source = fs.readFileSync(docPath, 'utf8');
  const refs = [];

  assert.doesNotMatch(
    source,
    /TODO:\s*.*图待补充/,
    `${path.relative(rootDir, docPath)} still has placeholder figure TODOs`,
  );

  for (const match of source.matchAll(/require\(['"](\.\/figs\/[^'"]+)['"]\)/g)) {
    refs.push(match[1]);
  }

  for (const match of source.matchAll(/!\[[^\]]*]\(([^)]+)\)/g)) {
    const ref = match[1].trim();
    if (!ref.startsWith('http') && ref.includes('figs/')) {
      refs.push(ref);
    }
  }

  return refs;
}

for (const assetName of requiredCs123Assets) {
  assert.ok(
    fs.existsSync(path.join(cs123FigsDir, assetName)),
    `Missing migrated CS123 reference asset: docs/practices/quadruped/cs123/figs/${assetName}`,
  );
}

const customCss = fs.readFileSync(customCssPath, 'utf8');
const cs123PidDoc = fs.readFileSync(cs123PidDocPath, 'utf8');
const cs123FkDoc = fs.readFileSync(cs123FkDocPath, 'utf8');
const figureComponent = fs.readFileSync(figureComponentPath, 'utf8');
const docTableComponent = fs.readFileSync(docTableComponentPath, 'utf8');
assert.match(
  customCss,
  /\.doc-figure\s*{[^}]*text-align:\s*center;[^}]*margin:\s*1\.75rem auto;[^}]*}/s,
  'Figure wrapper should center document figures',
);
assert.match(
  customCss,
  /\.doc-figure img,\s*\.doc-figure svg\s*{[^}]*display:\s*block;[^}]*margin:\s*0 auto;[^}]*}/s,
  'Figure images and SVGs should be block-centered',
);
assert.match(
  customCss,
  /\.markdown\s+table\s*{[^}]*display:\s*block\s*!important;[^}]*width:\s*max-content;[^}]*max-width:\s*100%;[^}]*margin:\s*1\.5em auto\s*!important;[^}]*overflow-x:\s*auto;[^}]*}/s,
  'Markdown tables should shrink to content width and center when they fit',
);
assert.match(
  customCss,
  /@media\s+print\s*{[\s\S]*?\.markdown\s+table\s*{[^}]*display:\s*table\s*!important;[^}]*width:\s*100%;[^}]*max-width:\s*100%;[^}]*overflow:\s*visible\s*!important;[^}]*margin:\s*1rem 0\s*!important;[^}]*}/s,
  'Print tables should keep full-width table layout',
);
assert.match(
  customCss,
  /\.markdown\s*{[^}]*counter-reset:\s*figure table;[^}]*}/s,
  'Markdown counters should include figures and tables',
);
assert.match(
  customCss,
  /\.doc-table\s*{[^}]*counter-increment:\s*table;[^}]*}/s,
  'Document table wrappers should increment the table counter',
);
assert.match(
  customCss,
  /\.doc-table figcaption::before\s*{[^}]*content:\s*"表 " counter\(table\) ": ";[^}]*}/s,
  'Document table captions should be automatically numbered',
);
assert.match(
  cs123PidDoc,
  /import DocTable from '@site\/src\/components\/DocTable';[\s\S]*按能量来源，常见执行器可以分成三大类，如\[表 1\]\(#tab-actuator-types\) 所示：[\s\S]*<DocTable id="tab-actuator-types" caption="常见执行器按能量来源的三大类">[\s\S]*<\/DocTable>/,
  'CS123 PID actuator table should use an anchored numbered table wrapper',
);
assert.match(
  cs123FkDoc,
  /import DocTable from '@site\/src\/components\/DocTable';[\s\S]*两种约定的对照见\[表 1\]\(#tab-dh-urdf\)。[\s\S]*<DocTable id="tab-dh-urdf" caption="DH 参数与 URDF \/ MJCF 父-子链式约定的对照">[\s\S]*<\/DocTable>/,
  'CS123 FK DH/URDF table should use an anchored numbered table wrapper',
);
assert.match(
  figureComponent,
  /id\?:\s*string;/,
  'Figure component should accept optional id anchors',
);
assert.match(
  figureComponent,
  /<figure className="doc-figure" id={id}>/,
  'Figure component should render optional id anchors',
);
assert.match(
  figureComponent,
  /useBrokenLinks\(\)\.collectAnchor\(id\);/,
  'Figure component should register optional id anchors for Docusaurus broken-anchor checks',
);
assert.match(
  docTableComponent,
  /useBrokenLinks\(\)\.collectAnchor\(id\);/,
  'DocTable component should register table anchors for Docusaurus broken-anchor checks',
);

for (const assetPath of listFiles(cs123FigsDir, (filePath) => /\.(gif|png|svg|webp)$/i.test(filePath))) {
  assertImageBytes(assetPath);
}

for (const docPath of listFiles(cs123Dir, (filePath) => /\.mdx?$/.test(filePath))) {
  for (const ref of collectImageRefs(docPath)) {
    const imagePath = path.resolve(path.dirname(docPath), ref);
    assert.ok(fs.existsSync(imagePath), `${path.relative(rootDir, docPath)} references missing image ${ref}`);
    assertImageBytes(imagePath);
  }
}

console.log('CS123 document image assets are present and readable.');
