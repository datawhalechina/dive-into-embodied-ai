import {spawnSync} from 'node:child_process';
import {copyFile, mkdir, readdir, stat} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
import {kinds} from './scene.mjs';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const output = path.join(root, 'static/img/introduction/carriers');
await mkdir(output, {recursive: true});
for (const kind of kinds) {
  const frames = path.join(root, '.agents/carrier-gifs/frames', kind);
  const count = (await readdir(frames)).filter(name => /^\d{4}\.png$/.test(name)).length;
  if (count !== 80) throw new Error(`${kind}: expected 80 frames, got ${count}`);
  const destination = path.join(output, `${kind}.gif`);
  const result = spawnSync('ffmpeg', ['-hide_banner', '-loglevel', 'error', '-y', '-framerate', '20', '-i', path.join(frames, '%04d.png'), '-filter_complex', '[0:v]split[a][b];[a]palettegen=max_colors=192:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=3', '-loop', '0', destination], {stdio: 'inherit'});
  if (result.status !== 0) throw new Error(`ffmpeg failed for ${kind}`);
  await copyFile(path.join(frames, 'poster.webp'), path.join(output, `${kind}.webp`));
  process.stdout.write(`${kind}: ${Math.round((await stat(destination)).size / 1024)} KB\n`);
}
