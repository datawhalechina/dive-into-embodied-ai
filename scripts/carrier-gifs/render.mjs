import * as THREE from 'three';
import {createCarrierScene, kinds} from './scene.mjs';

const names = ['人形机器人', '固定基座机械臂', '轮式移动机器人', '四足机器人', '移动操作平台', '仿真载体'];
const selector = document.querySelector('#kind');
const status = document.querySelector('#status');
const progress = document.querySelector('#progress');
const playButton = document.querySelector('#play');
const exportButton = document.querySelector('#export');
const renderer = new THREE.WebGLRenderer({antialias: true, preserveDrawingBuffer: true});
renderer.setSize(640, 400);
renderer.setPixelRatio(1);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.12;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
document.querySelector('#stage').appendChild(renderer.domElement);
for (let i = 0; i < kinds.length; i++) selector.add(new Option(names[i], kinds[i]));
let current = createCarrierScene(kinds[0]);
let playing = true;
let exporting = false;
let elapsed = 0;
let previous = performance.now();
const draw = t => {current.update(t); renderer.render(current.scene, current.camera);};
const select = kind => {current.dispose(); current = createCarrierScene(kind); elapsed = 0; draw(0);};
selector.addEventListener('change', () => select(selector.value));
playButton.addEventListener('click', () => {playing = !playing; playButton.textContent = playing ? '停止预览' : '播放预览';});
progress.addEventListener('input', () => {playing = false; playButton.textContent = '播放预览'; elapsed = Number(progress.value) / 100 * 4000; draw(elapsed / 4000);});
function tick(now) {
  if (playing && !exporting) {elapsed = (elapsed + Math.min(100, now - previous)) % 4000; draw(elapsed / 4000); progress.value = String(elapsed / 40);}
  previous = now; requestAnimationFrame(tick);
}
requestAnimationFrame(tick);
status.textContent = '已就绪 · 每段 4 秒 / 20 FPS / 640 × 400';

const encodeCanvas = (mime) => new Promise((resolve, reject) => renderer.domElement.toBlob(blob => blob ? resolve(blob) : reject(new Error('Canvas encoding failed')), mime, .92));
async function save(path, blob) {
  const response = await fetch(path, {method: 'POST', body: blob});
  if (!response.ok) throw new Error(`Save failed: ${response.status}`);
}
exportButton.addEventListener('click', async () => {
  exporting = true; selector.disabled = playButton.disabled = exportButton.disabled = progress.disabled = true;
  try {
    for (let n = 0; n < kinds.length; n++) {
      const kind = kinds[n]; selector.value = kind; select(kind);
      for (let frame = 0; frame < 80; frame++) {
        draw(frame / 80);
        if (frame === 0) await save(`/frames/${kind}/poster.webp`, await encodeCanvas('image/webp'));
        await save(`/frames/${kind}/${String(frame).padStart(4, '0')}.png`, await encodeCanvas('image/png'));
        status.textContent = `${names[n]}：${frame + 1} / 80 帧（${n + 1} / 6）`;
      }
    }
    status.textContent = '完成：6 组帧与静态封面已保存，可以运行 encode.mjs。';
  } catch (error) {status.textContent = `失败：${error.message}`;}
  finally {exporting = false; playing = false; playButton.textContent = '播放预览'; selector.disabled = playButton.disabled = exportButton.disabled = progress.disabled = false;}
});
