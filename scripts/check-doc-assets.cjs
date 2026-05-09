const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const rootDir = path.resolve(__dirname, '..');
const cs123Dir = path.join(rootDir, 'docs/practices/quadruped/cs123');
const cs123FigsDir = path.join(cs123Dir, 'figs');
const requiredCs123Assets = [
  'actuator-workflow.webp',
  'bang-bang-control.webp',
  'bang-bang-limit-cycle.webp',
  'closed-loop-control.webp',
  'coordinate-frames.webp',
  'open-loop-control.webp',
];

const customCssPath = path.join(rootDir, 'src/css/custom.css');

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
