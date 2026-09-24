import {spawnSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {mkdir, readFile, stat, unlink, writeFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import path from 'node:path';

const directory = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(directory, '../..');
const sources = path.join(root, '.agents/carrier-gifs/sources');
const output = path.join(root, 'static/img/introduction/carriers/official');
const clips = JSON.parse(await readFile(path.join(directory, 'official-sources.json'), 'utf8'));
await mkdir(sources, {recursive: true});
await mkdir(output, {recursive: true});

function ffmpeg(args) {
  const result = spawnSync('ffmpeg', ['-hide_banner', '-loglevel', 'error', '-y', ...args], {stdio: 'inherit'});
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(`ffmpeg exited with ${result.status}`);
}

function saveWebP(input, output) {
  const result = spawnSync('python3', ['-c',
    'import sys; from PIL import Image; Image.open(sys.argv[1]).convert("RGB").save(sys.argv[2], "WEBP", quality=85, method=6)',
    input, output], {stdio: 'inherit'});
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(`WebP conversion exited with ${result.status}; install Python Pillow`);
}

for (const clip of clips) {
  const input = path.join(sources, clip.sourceFile);
  let bytes;
  try {
    bytes = await readFile(input);
  } catch (error) {
    if (error.code !== 'ENOENT') throw error;
    const response = await fetch(clip.url);
    if (!response.ok) throw new Error(`${clip.id}: source returned HTTP ${response.status}`);
    bytes = Buffer.from(await response.arrayBuffer());
    await writeFile(input, bytes);
  }
  const checksum = createHash('sha256').update(bytes).digest('hex');
  if (checksum !== clip.sha256) throw new Error(`${clip.id}: source changed; review the footage and manifest before encoding`);

  // Fit the entire original frame so source labels and watermarks stay visible.
  const fit = (width, height) => `scale=${width}:${height}:force_original_aspect_ratio=decrease:force_divisible_by=2:flags=lanczos,pad=${width}:${height}:(ow-iw)/2:(oh-ih)/2:color=0x142132,setsar=1`;
  const video = path.join(output, `${clip.id}.mp4`);
  ffmpeg(['-ss', String(clip.start), '-t', String(clip.duration), '-i', input,
    '-vf', `fps=24,${fit(640, 400)}`, '-c:v', 'libx264', '-preset', 'slow', '-crf', '25', '-pix_fmt', 'yuv420p', '-an', '-movflags', '+faststart', video]);
  const frame = path.join(sources, `${clip.id}-poster.png`);
  ffmpeg(['-ss', String(clip.start), '-i', input, '-vf', fit(640, 400), '-frames:v', '1', frame]);
  try {
    saveWebP(frame, path.join(output, `${clip.id}.webp`));
  } finally {
    await unlink(frame);
  }
  process.stdout.write(`${clip.id}: playback ${Math.round((await stat(video)).size / 1024)} KB\n`);
}
