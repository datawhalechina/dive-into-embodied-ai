const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');

// This deliberately reads only the scalar arrays in the pinned deployment
// contract. Fail on changes instead of guessing how a new policy should run.
function readArray(yaml, key) {
  const match = yaml.match(new RegExp(`^${key}:[^\\n]*\\n((?:- [^\\n]+\\n)+)`, 'm'));
  if (!match) throw new Error(`Missing deployment array: ${key}`);
  const values = match[1].trim().split('\n').map(line => Number(line.slice(2)));
  if (values.length !== 12 || !values.every(Number.isFinite)) throw new Error(`Invalid deployment array: ${key}`);
  return values;
}

function generateSimulationAssets(root) {
  const output = path.join(root, 'static/simulation-assets');
  const catalog = JSON.parse(fs.readFileSync(path.join(root, 'simulation/models.json')));
  fs.rmSync(output, {recursive: true, force: true});
  fs.mkdirSync(output, {recursive: true});
  const copy = (from, to) => {
    const bytes = fs.readFileSync(from);
    if (bytes.subarray(0, 80).toString().startsWith('version https://git-lfs.github.com/')) throw new Error(`${from}: run git lfs pull`);
    const dest = path.join(output, to);
    fs.mkdirSync(path.dirname(dest), {recursive: true});
    fs.writeFileSync(dest, bytes);
  };
  for (const name of fs.readdirSync(path.join(root, 'simulation/runtime'))) {
    if (name.endsWith('.mjs')) copy(path.join(root, 'simulation/runtime', name), name);
  }
  const ort = path.join(root, 'node_modules/onnxruntime-web');
  for (const name of ['ort.wasm.min.mjs', 'ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm']) copy(path.join(ort, 'dist', name), `ort/${name}`);
  copy(path.join(root, 'simulation/licenses/ONNX-Runtime-LICENSE'), 'ort/LICENSE');
  for (const entry of catalog) {
    if (entry.adapter === 'quaddle') continue; // Canonical assets from the Quaddle plugin.
    if (['pupper', 'microduck'].includes(entry.adapter)) {
      const source = path.join(root, entry.source);
      const dir = path.dirname(path.join(source, entry.scene));
      const files = ['scene.xml'];
      let xml = fs.readFileSync(path.join(source, entry.scene), 'utf8');
      if (entry.adapter === 'microduck') {
        const robot = fs.readFileSync(path.join(dir, 'robot_allcollisions.xml'), 'utf8');
        const supported = robot.replace('<equality/>', '<equality><weld name="manual_support" body1="trunk_base" solref="0.02 1"/></equality>');
        fs.mkdirSync(path.join(output, entry.id), {recursive: true});
        fs.writeFileSync(path.join(output, entry.id, 'robot_allcollisions.xml'), supported);
        files.push('robot_allcollisions.xml');
        for (const [, name] of robot.matchAll(/<mesh\s[^>]*file="([^"]+)"/g)) {
          copy(path.join(dir, 'assets', name), `${entry.id}/assets/${name}`); files.push(`assets/${name}`);
        }
        for (const name of ['LICENSE', 'UPSTREAM_README.md']) copy(path.join(source, name), `${entry.id}/${name}`);
      } else {
        for (const [, name] of xml.matchAll(/<mesh\s[^>]*file="([^"]+)"/g)) {
          copy(path.join(dir, 'meshes/stl', name), `${entry.id}/meshes/stl/${name}`); files.push(`meshes/stl/${name}`);
        }
        copy(path.join(source, entry.policy), `${entry.id}/policy.onnx`);
        copy(path.join(source, entry.config), `${entry.id}/config.json`);
        copy(path.join(path.dirname(path.join(source, entry.policy)), 'provenance.json'), `${entry.id}/provenance.json`);
        copy(path.join(source, 'README.md'), `${entry.id}/SOURCE_README.md`);
      }
      fs.writeFileSync(path.join(output, entry.id, 'scene.xml'), xml);
      fs.writeFileSync(path.join(output, entry.id, 'manifest.json'), JSON.stringify({files}, null, 2));
      continue;
    }
    if (entry.adapter !== 'go2') throw new Error(`No asset builder for ${entry.adapter}`);
    const source = path.join(root, entry.source);
    const yaml = fs.readFileSync(path.join(source, entry.deploy), 'utf8');
    const stepDt = Number(yaml.match(/^step_dt: ([\d.]+)/m)?.[1]);
    const period = Number(yaml.match(/period: ([\d.]+)/)?.[1]);
    if (stepDt !== .02 || period !== .6 || !yaml.includes('    - 0.25')) throw new Error('Go2 deployment contract changed; update and revalidate the adapter.');
    const config = {
      defaults: readArray(yaml, 'default_joint_pos'),
      stiffness: readArray(yaml, 'stiffness'), damping: readArray(yaml, 'damping'),
      stepDt, period, actionScale: .25,
      joints: ['FL', 'FR', 'RL', 'RR'].flatMap(leg => ['hip', 'thigh', 'calf'].map(joint => `${leg}_${joint}_joint`)),
    };
    const original = fs.readFileSync(path.join(source, entry.scene), 'utf8');
    // Keep the original inertias, joints, actuators and every collision shape.
    // Only visual meshes are omitted, so builds never fetch 27 MB of upstream OBJ
    // assets. The viewer renders the retained collision geometry explicitly.
    const xml = original.replace(/<mesh\b[^>]*\/>/g, '').replace(/<geom\b[^>]*\bmesh="[^"]+"[^>]*\/>/g, '');
    fs.mkdirSync(path.join(output, entry.id), {recursive: true});
    fs.writeFileSync(path.join(output, entry.id, 'scene.xml'), xml);
    fs.writeFileSync(path.join(output, entry.id, 'config.json'), JSON.stringify(config, null, 2));
    copy(path.join(source, entry.policy), `${entry.id}/policy.onnx`);
    copy(path.join(source, 'LICENSE'), `${entry.id}/LICENSE`);
    const assets = path.join(source, 'src/unitree_rl_mjx/assets/robots/unitree_go2');
    for (const name of ['LICENSE-mjlab', 'PROVENANCE.md']) copy(path.join(assets, name), `${entry.id}/${name}`);
    fs.writeFileSync(path.join(output, entry.id, 'provenance.json'), JSON.stringify({
      sceneSha256: crypto.createHash('sha256').update(original).digest('hex'),
      policySha256: crypto.createHash('sha256').update(fs.readFileSync(path.join(source, entry.policy))).digest('hex'),
      visualMode: 'collision-geometry',
    }, null, 2));
  }
  fs.writeFileSync(path.join(output, 'models.json'), JSON.stringify(catalog, null, 2));
}
module.exports = {generateSimulationAssets};
