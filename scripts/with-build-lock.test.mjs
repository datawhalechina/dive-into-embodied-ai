import assert from 'node:assert/strict';
import {fork} from 'node:child_process';
import {copyFileSync, mkdirSync, mkdtempSync, readFileSync, rmSync, symlinkSync, writeFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {test} from 'node:test';

function fixture(t) {
  const root = mkdtempSync(join(tmpdir(), 'docusaurus-build-lock-'));
  mkdirSync(join(root, 'scripts'));
  copyFileSync(new URL('./with-build-lock.mjs', import.meta.url), join(root, 'scripts/with-build-lock.mjs'));
  writeFileSync(join(root, 'work.cjs'), `
    const fs = require('node:fs');
    if (process.env.FAIL_BUILD) throw new Error('fixture build failed');
    if (process.env.DELETE_INDEX) fs.rmSync(process.env.DELETE_INDEX);
    process.send({ready: true, args: process.argv.slice(2)});
    process.on('message', code => process.exit(code));
  `);
  t.after(() => rmSync(root, {recursive: true, force: true}));
  return root;
}

function start(t, root, {args = ['work.cjs'], env = {}, cwd = root} = {}) {
  const child = fork(join(root, 'scripts/with-build-lock.mjs'), args, {
    cwd, env: {...process.env, ...env}, execArgv: [], silent: true,
  });
  let output = '';
  child.stdout.on('data', chunk => { output += chunk; });
  child.stderr.on('data', chunk => { output += chunk; });
  const done = new Promise((resolve, reject) => {
    child.once('error', reject);
    child.once('exit', (code, signal) => resolve({code, signal}));
  });
  const ready = Promise.race([
    new Promise(resolve => child.once('message', resolve)),
    done.then(result => ({exited: result})),
  ]);
  t.after(async () => {
    if (child.exitCode === null && child.signalCode === null) child.kill('SIGKILL');
    await done;
  });
  return {child, done, ready, output: () => output};
}

async function expectReady(process) {
  const message = await process.ready;
  assert.equal(message.ready, true, process.output());
  return message;
}

test('a competing build exits before touching the active build output', {timeout: 15000}, async t => {
  const root = fixture(t);
  const index = join(root, 'search-index.json');
  writeFileSync(index, 'active build output');
  const first = start(t, root);
  await expectReady(first);
  const second = start(t, root, {env: {DELETE_INDEX: index}, cwd: tmpdir()});
  assert.equal((await second.done).code, 1);
  assert.match(second.output(), /already running/);
  assert.equal(readFileSync(index, 'utf8'), 'active build output');
  first.child.send(0);
  assert.equal((await first.done).code, 0);
  const next = start(t, root, {args: ['work.cjs', '--locale', 'en', '--out-dir', 'custom-build']});
  assert.deepEqual((await expectReady(next)).args, ['--locale', 'en', '--out-dir', 'custom-build']);
  next.child.send(7);
  assert.equal((await next.done).code, 7);
});

for (const signal of ['SIGINT', 'SIGTERM', 'SIGKILL']) {
  test(`a build can restart immediately after ${signal}`, {timeout: 15000}, async t => {
    const root = fixture(t);
    const first = start(t, root);
    await expectReady(first);
    first.child.kill(signal);
    await first.done;
    const next = start(t, root);
    await expectReady(next);
    next.child.send(0);
    assert.equal((await next.done).code, 0);
  });
}

test('an exception releases the lock and preserves failure status', {timeout: 15000}, async t => {
  const root = fixture(t);
  const failed = start(t, root, {env: {FAIL_BUILD: '1'}});
  assert.equal((await failed.done).code, 1);
  assert.match(failed.output(), /fixture build failed/);
  const next = start(t, root);
  await expectReady(next);
  next.child.send(0);
  await next.done;
});

test('simultaneous starts admit exactly one process', {timeout: 15000}, async t => {
  const root = fixture(t);
  const processes = Array.from({length: 8}, () => start(t, root));
  const results = await Promise.all(processes.map(process => process.ready));
  assert.equal(results.filter(result => result.ready).length, 1);
  for (const [i, result] of results.entries()) {
    if (result.ready) processes[i].child.send(0);
    else {
      assert.equal(result.exited.code, 1);
      assert.match(processes[i].output(), /already running/);
    }
  }
  await Promise.all(processes.map(process => process.done));
});

test('automatic dependency installation and builds share the lock', {timeout: 15000}, async t => {
  const root = fixture(t);
  const installer = start(t, root, {args: ['--npm-ci'], env: {npm_execpath: join(root, 'work.cjs')}});
  assert.deepEqual((await expectReady(installer)).args, ['ci']);
  const build = start(t, root);
  assert.equal((await build.done).code, 1);
  assert.match(build.output(), /already running/);
  installer.child.send(0);
  await installer.done;
  const next = start(t, root);
  await expectReady(next);
  const secondInstaller = start(t, root, {args: ['--npm-ci'], env: {npm_execpath: join(root, 'work.cjs')}});
  assert.equal((await secondInstaller.done).code, 1);
  next.child.send(0);
  await next.done;
});

test('independent checkouts can build but a symlink cannot bypass a lock', {timeout: 15000}, async t => {
  const root = fixture(t);
  const other = fixture(t);
  const first = start(t, root);
  const independent = start(t, other);
  await Promise.all([expectReady(first), expectReady(independent)]);
  const alias = join(other, 'alias');
  symlinkSync(root, alias, 'junction');
  const duplicate = start(t, alias);
  assert.equal((await duplicate.done).code, 1);
  assert.match(duplicate.output(), /already running/);
  first.child.send(0);
  independent.child.send(0);
  await Promise.all([first.done, independent.done]);
});
