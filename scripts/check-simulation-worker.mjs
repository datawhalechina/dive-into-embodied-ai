// Exercise the browser worker's message protocol using the same assets and WASM
// engines in Node. This does not replace visual/browser interaction checks.
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {Worker, isMainThread, parentPort} from 'node:worker_threads';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import quaddleAssets from './docusaurus-quaddle.cjs';

if (!isMainThread) {
  globalThis.onmessage = null;
  globalThis.postMessage = (message, transfer) => parentPort.postMessage(message, transfer);
  globalThis.fetch = async url => {
    assert.equal(new URL(url).protocol, 'file:');
    return new Response(await fs.readFile(new URL(url)));
  };
  parentPort.on('message', data => globalThis.onmessage?.({data}));
  await import('../static/simulation-assets/worker.mjs');
  parentPort.postMessage({type: 'boot'});
} else {
  const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
  await quaddleAssets({siteDir: root}).loadContent();
  for (const id of ['go2', 'pupper-v3', 'microduck', 'quaddle']) {
    const worker = new Worker(new URL(import.meta.url));
    const wait = predicate => new Promise((resolve, reject) => {
      const timer = setTimeout(() => done(new Error(`${id}: worker timed out`)), 45000);
      function done(error, value) {
        clearTimeout(timer); worker.off('message', message); worker.off('error', failed);
        if (error) reject(error); else resolve(value);
      }
      const failed = error => done(error);
      const message = value => {
        if (value.type === 'error') done(new Error(value.detail));
        else if (predicate(value)) done(null, value);
      };
      worker.on('message', message); worker.on('error', failed);
    });
    const send = message => worker.postMessage(message);
    try {
      await wait(m => m.type === 'boot');
      const loaded = wait(m => m.type === 'ready');
      send({type: 'init', id, mode: 'showcase'});
      const ready = await loaded;
      assert(ready.geometry.length > 0);
      const moving = await wait(m => m.type === 'state' && m.time > .1);
      assert.equal(moving.mode, id === 'microduck' ? 'manual' : 'showcase');
      send({type: 'pause', value: true});
      const paused = await wait(m => m.type === 'state' && m.paused);
      const next = await wait(m => m.type === 'state');
      assert.equal(next.time, paused.time);
      send({type: 'reset'});
      const reset = await wait(m => m.type === 'state' && m.time === 0);
      assert(reset.paused, 'Reset must preserve pause');
      if (id === 'microduck') {
        assert.equal(ready.joints.length, 14);
        send({type: 'mode', value: 'showcase'});
        send({type: 'joint', index: 7, value: .3});
        const joint = await wait(m => m.type === 'state' && m.jointTargets?.[7] === .3);
        assert.equal(joint.mode, 'manual');
        send({type: 'support', value: false});
        await wait(m => m.type === 'state' && m.supported === false);
        send({type: 'support', value: true});
        await wait(m => m.type === 'state' && m.supported && m.jointTargets?.[7] === 0);
      }
      send({type: 'mode', value: 'manual'});
      send({type: 'command', value: [.2, 0, 0]});
      await wait(m => m.type === 'state' && m.command[0] === .2);
      send({type: 'visibility', value: false});
      send({type: 'pause', value: false});
      const hidden = await wait(m => m.type === 'state' && !m.paused && m.command.every(v => v === 0));
      const hiddenNext = await wait(m => m.type === 'state');
      assert.equal(hidden.time, hiddenNext.time);
      send({type: 'visibility', value: true});
      await wait(m => m.type === 'state' && m.time > hidden.time);
      console.log(`${id}: worker load, modes, pause/reset, visibility and input clearing passed.`);
    } finally { await worker.terminate(); }
  }
}
