// Run the CLI in this process so the OS lock lives exactly as long as the work.
// Node's built-in SQLite provides an exclusive file lock without a dependency,
// heartbeat, stale PID file, or cleanup race after Ctrl+C / SIGKILL.
import {realpathSync} from 'node:fs';
import {dirname, resolve} from 'node:path';
import {DatabaseSync} from 'node:sqlite';
import {fileURLToPath, pathToFileURL} from 'node:url';

const root = realpathSync(resolve(dirname(fileURLToPath(import.meta.url)), '..'));
const [target, ...args] = process.argv.slice(2);
const entry = target === '--npm-ci' ? process.env.npm_execpath : target;

if (!entry) {
  console.error('Use npm run build or npm run install:locked.');
  process.exit(1);
}

let lock;
try {
  lock = new DatabaseSync(resolve(root, '.docusaurus-build-lock.sqlite'));
  lock.exec('PRAGMA busy_timeout = 0; BEGIN EXCLUSIVE;');
} catch (error) {
  lock?.close();
  // SQLite's primary result codes: SQLITE_BUSY and SQLITE_LOCKED.
  if ([5, 6].includes(error.errcode & 0xff)) {
    console.error('Another build or dependency installation is already running in this checkout. Wait for it to finish, then retry.');
    process.exit(1);
  }
  throw error;
}

// Keep the file: deleting a lock file can let a second process lock a new inode.
// Closing the connection (also done by the OS on exit) releases the lock.
process.once('exit', () => lock.close());
const cli = resolve(root, entry);
process.argv = [process.execPath, cli, ...(target === '--npm-ci' ? ['ci'] : args)];
await import(pathToFileURL(cli).href);
