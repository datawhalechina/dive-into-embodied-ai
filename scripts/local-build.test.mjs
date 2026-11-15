import assert from 'node:assert/strict';
import {spawn} from 'node:child_process';
import {copyFileSync, mkdirSync, mkdtempSync, realpathSync, rmSync, writeFileSync} from 'node:fs';
import {createServer} from 'node:net';
import {tmpdir} from 'node:os';
import {dirname, join} from 'node:path';
import {setTimeout as delay} from 'node:timers/promises';
import {test} from 'node:test';

// Exercise the real shell script and OS process inspection with lightweight
// fixture builds/servers; never terminate the developer's running preview.
function fixture(t) {
  const root = realpathSync(mkdtempSync(join(tmpdir(), 'local preview test ')));
  for (const dir of ['bin', 'scripts', 'node_modules/.bin']) mkdirSync(join(root, dir), {recursive: true});
  copyFileSync(new URL('../local-build.sh', import.meta.url), join(root, 'local-build.sh'));
  copyFileSync(new URL('./with-build-lock.mjs', import.meta.url), join(root, 'scripts/with-build-lock.mjs'));
  writeFileSync(join(root, '.nvmrc'), process.versions.node);
  writeFileSync(join(root, 'package.json'), JSON.stringify({engines: {npm: '99.0.0'}}));
  writeFileSync(join(root, 'package-lock.json'), '{}');
  writeFileSync(join(root, 'node_modules/.package-lock.json'), '{}');
  writeFileSync(join(root, 'bin/npm'), `#!/usr/bin/env node
    const {spawnSync} = require('node:child_process');
    const [command, script] = process.argv.slice(2);
    if (command === '--version') console.log('99.0.0');
    else if (command === 'ls') process.exit(0);
    else if (command === 'run' && script === 'build') {
      const result = spawnSync(process.execPath, ['scripts/with-build-lock.mjs', 'build.mjs'], {stdio: 'inherit'});
      process.exit(result.status ?? 1);
    } else throw new Error('Unexpected npm command: ' + process.argv.slice(2));
  `, {mode: 0o755});
  writeFileSync(join(root, 'build.mjs'), `
    console.log('FIXTURE_BUILD');
    if (process.env.FAIL_BUILD) process.exit(7);
    await new Promise(resolve => setTimeout(resolve, Number(process.env.BUILD_DELAY || 0)));
  `);
  writeFileSync(join(root, 'node_modules/.bin/docusaurus'), `#!/usr/bin/env node
    const port = Number(process.argv[process.argv.indexOf('--port') + 1]);
    require('node:http').createServer((req, res) => res.end(String(process.pid)))
      .listen(port, '127.0.0.1', () => console.log('FIXTURE_SERVING ' + port));
  `, {mode: 0o755});
  t.after(() => rmSync(root, {recursive: true, force: true}));
  return root;
}

function launch(t, root, port, {env = {}, dev = false} = {}) {
  const child = spawn(dev ? process.execPath : 'sh', dev
    ? ['node_modules/.bin/docusaurus', 'start', '--port', String(port)]
    : ['local-build.sh', String(port)], {
    cwd: root, detached: true,
    env: {...process.env, ...env, PATH: `${join(root, 'bin')}:${dirname(process.execPath)}:${process.env.PATH}`},
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  let output = '';
  child.stdout.on('data', chunk => { output += chunk; });
  child.stderr.on('data', chunk => { output += chunk; });
  const done = new Promise((resolve, reject) => {
    child.once('error', reject);
    child.once('exit', (code, signal) => resolve({code, signal}));
  });
  t.after(async () => {
    try { process.kill(-child.pid, 'SIGKILL'); } catch (error) {
      if (error.code !== 'ESRCH') throw error;
    }
    await done;
  });
  return {
    child, done, output: () => output,
    async waitFor(text) {
      const deadline = Date.now() + 15000;
      while (!output.includes(text)) {
        assert.equal(child.exitCode, null, output);
        assert.equal(child.signalCode, null, output);
        assert.ok(Date.now() < deadline, output);
        await delay(50);
      }
    },
  };
}

async function freePortPair() {
  const first = createServer();
  await new Promise(resolve => first.listen(0, '127.0.0.1', resolve));
  const port = first.address().port;
  const second = createServer();
  try {
    await new Promise((resolve, reject) => {
      second.once('error', reject);
      second.listen(port + 1, '127.0.0.1', resolve);
    });
    return port;
  } finally {
    await Promise.all([first, second].map(server => new Promise(resolve => server.close(resolve))));
  }
}

test('restarting replaces the existing preview on the same port', {timeout: 25000}, async t => {
  const root = fixture(t);
  const port = await freePortPair();
  const first = launch(t, root, port);
  await first.waitFor(`FIXTURE_SERVING ${port}`);
  const next = launch(t, root, port);
  await next.waitFor(`FIXTURE_SERVING ${port}`);
  assert.equal((await first.done).signal, 'SIGTERM');
  assert.match(next.output(), /Stopping this checkout's preview/);
  assert.equal(await (await fetch(`http://127.0.0.1:${port}`)).text(), String(next.child.pid));
});

for (const dev of [false, true]) {
  test(`restart preserves ${dev ? 'a dev server in this checkout' : 'another checkout preview'}`, {timeout: 25000}, async t => {
    const root = fixture(t);
    const port = await freePortPair();
    const existing = launch(t, dev ? root : fixture(t), port, {dev});
    await existing.waitFor(`FIXTURE_SERVING ${port}`);
    const next = launch(t, root, port);
    await next.waitFor(`FIXTURE_SERVING ${port + 1}`);
    assert.doesNotMatch(next.output(), /Stopping this checkout's preview/);
    assert.equal(await (await fetch(`http://127.0.0.1:${port}`)).text(), String(existing.child.pid));
  });
}

test('a failed rebuild exits without serving incomplete output', {timeout: 25000}, async t => {
  const root = fixture(t);
  const port = await freePortPair();
  const first = launch(t, root, port);
  await first.waitFor(`FIXTURE_SERVING ${port}`);
  const failed = launch(t, root, port, {env: {FAIL_BUILD: '1'}});
  assert.equal((await failed.done).code, 7, failed.output());
  assert.equal((await first.done).signal, 'SIGTERM');
  assert.doesNotMatch(failed.output(), /FIXTURE_SERVING/);
  await assert.rejects(fetch(`http://127.0.0.1:${port}`));
});

test('a second invocation cannot interrupt an active build', {timeout: 25000}, async t => {
  const root = fixture(t);
  const port = await freePortPair();
  const first = launch(t, root, port, {env: {BUILD_DELAY: '4000'}});
  await first.waitFor('FIXTURE_BUILD');
  const duplicate = launch(t, root, port);
  assert.equal((await duplicate.done).code, 1, duplicate.output());
  assert.match(duplicate.output(), /already running/);
  assert.doesNotMatch(duplicate.output(), /FIXTURE_BUILD|FIXTURE_SERVING/);
  await first.waitFor(`FIXTURE_SERVING ${port}`);
});

test('invalid ports are rejected before any build', {timeout: 15000}, async t => {
  const root = fixture(t);
  for (const port of ['abc', '0', '-1', '65536']) {
    const invalid = launch(t, root, port);
    assert.equal((await invalid.done).code, 1);
    assert.match(invalid.output(), /Port must be an integer/);
    assert.doesNotMatch(invalid.output(), /FIXTURE_BUILD|FIXTURE_SERVING/);
  }
});
