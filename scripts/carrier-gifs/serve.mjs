import http from 'node:http';
import {readFile, mkdir, writeFile} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import path from 'node:path';

const directory = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(directory, '../..');
const port = Number(process.env.CARRIER_RENDER_PORT || 3026);
const origin = `http://127.0.0.1:${port}`;
const files = new Map([
  ['/', [path.join(directory, 'index.html'), 'text/html']],
  ['/render.mjs', [path.join(directory, 'render.mjs'), 'text/javascript']],
  ['/scene.mjs', [path.join(directory, 'scene.mjs'), 'text/javascript']],
  ['/vendor/three.module.js', [path.join(root, 'node_modules/three/build/three.module.js'), 'text/javascript']],
  ['/vendor/three.core.js', [path.join(root, 'node_modules/three/build/three.core.js'), 'text/javascript']],
  ['/vendor/addons/geometries/RoundedBoxGeometry.js', [path.join(root, 'node_modules/three/examples/jsm/geometries/RoundedBoxGeometry.js'), 'text/javascript']],
]);
http.createServer(async (request, response) => {
  try {
    const url = new URL(request.url, origin);
    if (request.method === 'GET' && files.has(url.pathname)) {
      const [file, type] = files.get(url.pathname);
      response.writeHead(200, {'Content-Type': type, 'Cache-Control': 'no-store'});
      response.end(await readFile(file)); return;
    }
    const frame = url.pathname.match(/^\/frames\/(humanoid|arm|wheeled|quadruped|mobile-manipulator|simulation)\/(\d{4}\.png|poster\.webp)$/);
    if (request.method === 'POST' && frame && request.headers.origin === origin) {
      let size = 0; const chunks = [];
      for await (const chunk of request) {size += chunk.length; if (size > 2_000_000) {response.writeHead(413); response.end(); return;} chunks.push(chunk);}
      const destination = path.join(root, '.agents/carrier-gifs/frames', frame[1]);
      await mkdir(destination, {recursive: true});
      await writeFile(path.join(destination, frame[2]), Buffer.concat(chunks));
      response.writeHead(204); response.end(); return;
    }
    response.writeHead(404); response.end('Not found');
  } catch (error) {response.writeHead(500); response.end(error.message);}
}).listen(port, '127.0.0.1', () => process.stdout.write(`Carrier renderer: ${origin}\n`));
