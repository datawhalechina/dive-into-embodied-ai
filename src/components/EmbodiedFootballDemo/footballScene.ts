import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {RoundedBoxGeometry} from 'three/addons/geometries/RoundedBoxGeometry.js';

export type FootballFrame = {elapsed: number; keeperNear: boolean; stage: number};
export type FootballScene = {update: (frame: FootballFrame) => void; resetView: () => void; dispose: () => void};

const smooth = (time: number, start: number, end: number) => {
  const t = THREE.MathUtils.clamp((time - start) / (end - start), 0, 1);
  return t * t * (3 - 2 * t);
};

function turfTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = 2048;
  canvas.height = 1024;
  const context = canvas.getContext('2d')!;
  for (let i = 0; i < 16; i++) {
    context.fillStyle = i % 2 ? '#237754' : '#29825c';
    context.fillRect(i * 128, 0, 128, 1024);
  }
  let seed = 71;
  for (let i = 0; i < 48000; i++) {
    seed = (seed * 16807) % 2147483647;
    const x = seed % 2048;
    seed = (seed * 16807) % 2147483647;
    context.fillStyle = i % 2 ? 'rgba(235,255,223,.10)' : 'rgba(0,25,12,.10)';
    context.fillRect(x, seed % 1024, 1, 3);
  }
  const x = (value: number) => (value / 15 + .5) * canvas.width;
  const z = (value: number) => (value / 9 + .5) * canvas.height;
  context.strokeStyle = 'rgba(236,251,235,.88)';
  context.lineWidth = 5;
  context.strokeRect(x(-6), z(-3.7), x(6) - x(-6), z(3.7) - z(-3.7));
  context.beginPath(); context.moveTo(x(-3), z(-3.7)); context.lineTo(x(-3), z(3.7)); context.stroke();
  context.beginPath(); context.ellipse(x(-3), z(0), 1.2 / 15 * 2048, 1.2 / 9 * 1024, 0, 0, Math.PI * 2); context.stroke();
  context.strokeRect(x(3.5), z(-2.8), x(6) - x(3.5), z(2.8) - z(-2.8));
  context.strokeRect(x(5), z(-1.8), x(6) - x(5), z(1.8) - z(-1.8));
  context.fillStyle = '#ebf7e8';
  context.beginPath(); context.arc(x(4.4), z(0), 5, 0, Math.PI * 2); context.fill();
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = 4;
  return texture;
}

export function createFootballScene(host: HTMLDivElement, onUnavailable: () => void): FootballScene {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color('#101c2b');
  scene.fog = new THREE.Fog('#101c2b', 23, 48);
  const camera = new THREE.PerspectiveCamera(38, 1, .1, 70);
  const renderer = new THREE.WebGLRenderer({antialias: true, alpha: false, powerPreference: 'low-power'});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.75));
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.2;
  renderer.domElement.tabIndex = 0;
  renderer.domElement.setAttribute('role', 'img');
  renderer.domElement.setAttribute('aria-label', '可旋转的三维机器人足球场。拖动或使用方向键旋转视角，Home 键复位。');
  host.appendChild(renderer.domElement);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.target.set(.7, .7, 0);
  controls.enablePan = false;
  controls.enableZoom = false;
  controls.minPolarAngle = .32;
  controls.maxPolarAngle = Math.PI / 2.25;
  controls.minAzimuthAngle = -Math.PI * .8;
  controls.maxAzimuthAngle = Math.PI * .8;
  renderer.domElement.style.touchAction = 'pan-y';
  const render = () => renderer.render(scene, camera);

  const material = (color: string, metalness = .12, roughness = .4) => new THREE.MeshStandardMaterial({color, metalness, roughness});
  const ceramic = material('#e9f1f4', .25, .28);
  const joint = material('#253446', .65, .32);
  const blue = material('#1a6deb', .35, .25);
  const orange = material('#ed7a2e', .25, .3);
  const visor = material('#061c2b', .55, .16);
  const light = new THREE.MeshStandardMaterial({color: '#92f1ff', emissive: '#35c7ef', emissiveIntensity: 1.7});
  const white = material('#eff6fa', .3, .35);
  const addMesh = (parent: THREE.Object3D, geometry: THREE.BufferGeometry, surface: THREE.Material, position: [number, number, number]) => {
    const mesh = new THREE.Mesh(geometry, surface);
    mesh.position.set(...position); mesh.castShadow = true; mesh.receiveShadow = true;
    parent.add(mesh); return mesh;
  };
  const box = (parent: THREE.Object3D, size: [number, number, number], position: [number, number, number], surface: THREE.Material, radius = .07) =>
    addMesh(parent, new RoundedBoxGeometry(...size, 3, radius), surface, position);
  const sphere = (parent: THREE.Object3D, radius: number, position: [number, number, number], surface: THREE.Material) =>
    addMesh(parent, new THREE.SphereGeometry(radius, 20, 16), surface, position);
  const rod = (parent: THREE.Object3D, a: THREE.Vector3, b: THREE.Vector3, radius: number, surface: THREE.Material) => {
    const mesh = addMesh(parent, new THREE.CylinderGeometry(radius, radius, a.distanceTo(b), 12), surface, [0, 0, 0]);
    mesh.position.copy(a).add(b).multiplyScalar(.5);
    mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), b.clone().sub(a).normalize());
    return mesh;
  };

  scene.add(new THREE.HemisphereLight('#d5eeff', '#254c3d', 1.7));
  const sun = new THREE.DirectionalLight('#fff0de', 3.2);
  sun.position.set(-3, 10, 6); sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  Object.assign(sun.shadow.camera, {left: -11, right: 11, top: 9, bottom: -9, near: .5, far: 30});
  sun.shadow.normalBias = .025;
  sun.shadow.bias = -.0002;
  scene.add(sun);
  const rim = new THREE.DirectionalLight('#91bfff', 2);
  rim.position.set(5, 5, -7); scene.add(rim);
  box(scene, [80, .2, 70], [0, -.8, 0], material('#101c2b', .1, .9));
  box(scene, [15.2, .5, 9.2], [0, -.31, 0], material('#182b3a', .25, .55), .18);
  const grass = turfTexture();
  const pitch = addMesh(scene, new THREE.PlaneGeometry(15, 9), new THREE.MeshStandardMaterial({map: grass, roughness: .95}), [0, -.045, 0]);
  pitch.rotation.x = -Math.PI / 2; pitch.castShadow = false;
  const rail = material('#263c51', .4, .4);
  for (const z of [-4.8, 4.8]) {
    box(scene, [15.7, .42, .22], [0, -.08, z], rail);
    box(scene, [15.5, .035, .03], [0, .14, z - Math.sign(z) * .12], light, .01);
  }
  for (let i = 0; i < 3; i++) {
    box(scene, [15.6, .25 + i * .25, .8], [0, -.35 + i * .13, -5.65 - i * .85], material(i % 2 ? '#25344b' : '#1c2b3c', .1, .7));
  }

  function robot(surface: THREE.Material) {
    const root = new THREE.Group(); scene.add(root);
    box(root, [.55, .24, .36], [0, .94, 0], joint);
    box(root, [.77, .63, .42], [0, 1.43, 0], ceramic, .13);
    box(root, [.60, .38, .09], [0, 1.5, .24], surface, .06);
    box(root, [.35, .085, .03], [0, 1.52, .295], light, .02);
    box(root, [.44, .10, .06], [0, 1.19, .24], joint, .025);
    sphere(root, .15, [0, 1.83, 0], joint);
    const head = new THREE.Group(); head.position.set(0, 2.05, 0); root.add(head);
    box(head, [.67, .49, .52], [0, 0, 0], ceramic, .15);
    box(head, [.55, .23, .10], [0, -.015, .255], visor, .07);
    box(head, [.13, .045, .025], [-.125, -.005, .314], light, .02);
    box(head, [.13, .045, .025], [.125, -.005, .314], light, .02);
    box(head, [.3, .07, .33], [0, .245, -.045], surface, .03);
    for (const side of [-1, 1]) {
      const ear = addMesh(head, new THREE.CylinderGeometry(.105, .105, .075, 20), joint, [side * .34, -.005, 0]);
      ear.rotation.z = Math.PI / 2;
    }
    const legs = [-1, 1].map(side => {
      const hip = new THREE.Group(); hip.position.set(side * .22, .91, 0); root.add(hip);
      sphere(hip, .135, [0, 0, 0], joint);
      box(hip, [.27, .36, .29], [0, -.20, 0], ceramic, .09);
      box(hip, [.19, .22, .055], [0, -.18, .16], surface, .04);
      const knee = new THREE.Group(); knee.position.y = -.4; hip.add(knee);
      sphere(knee, .14, [0, 0, 0], joint);
      box(knee, [.23, .33, .27], [0, -.20, 0], ceramic, .075);
      box(knee, [.08, .23, .055], [0, -.19, .145], surface, .025);
      box(knee, [.33, .16, .49], [0, -.42, .11], surface, .055);
      box(knee, [.34, .035, .5], [0, -.507, .11], joint, .015);
      return {hip, knee};
    });
    const arms = [-1, 1].map(side => {
      const shoulder = new THREE.Group(); shoulder.position.set(side * .49, 1.69, 0); root.add(shoulder);
      sphere(shoulder, .17, [0, 0, 0], surface);
      box(shoulder, [.23, .33, .25], [0, -.2, 0], ceramic, .07);
      const elbow = new THREE.Group(); elbow.position.y = -.4; shoulder.add(elbow);
      sphere(elbow, .115, [0, 0, 0], joint);
      box(elbow, [.22, .28, .23], [0, -.17, 0], ceramic, .07);
      box(elbow, [.24, .19, .25], [0, -.36, .025], surface, .07);
      shoulder.rotation.z = side * .13;
      return {shoulder, elbow};
    });
    return {root, head, legs, arms};
  }
  const striker = robot(blue);
  const keeper = robot(orange);
  striker.root.rotation.y = Math.PI / 2;
  keeper.root.rotation.y = -Math.PI / 2;
  keeper.root.position.x = 5.35;
  keeper.arms.forEach(({shoulder, elbow}, i) => {shoulder.rotation.z = i ? 1.02 : -1.02; elbow.rotation.x = -.25;});

  const goal = new THREE.Group(); scene.add(goal);
  const v = (x: number, y: number, z: number) => new THREE.Vector3(x, y, z);
  for (const z of [-1.95, 1.95]) {
    rod(goal, v(6, 0, z), v(6, 2.35, z), .06, white);
    rod(goal, v(6, 2.35, z), v(7.15, 2.35, z), .035, white);
    rod(goal, v(7.15, 2.35, z), v(7.15, 0, z), .035, white);
    rod(goal, v(6, 0, z), v(7.15, 0, z), .035, white);
  }
  rod(goal, v(6, 2.35, -1.95), v(6, 2.35, 1.95), .06, white);
  rod(goal, v(7.15, 2.35, -1.95), v(7.15, 2.35, 1.95), .035, white);
  const netPoints: THREE.Vector3[] = [];
  for (let z = -1.95; z <= 1.96; z += .195) {
    netPoints.push(v(7.15, 0, z), v(7.15, 2.35, z), v(6, 2.35, z), v(7.15, 2.35, z));
  }
  for (let y = 0; y <= 2.36; y += .196) {
    netPoints.push(v(7.15, y, -1.95), v(7.15, y, 1.95));
    for (const z of [-1.95, 1.95]) netPoints.push(v(6, y, z), v(7.15, y, z));
  }
  for (let x = 6; x <= 7.16; x += .192) {
    for (const z of [-1.95, 1.95]) netPoints.push(v(x, 0, z), v(x, 2.35, z));
    netPoints.push(v(x, 2.35, -1.95), v(x, 2.35, 1.95));
  }
  goal.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(netPoints), new THREE.LineBasicMaterial({color: '#d3e5ef', transparent: true, opacity: .4})));

  const ball = new THREE.Group(); scene.add(ball);
  sphere(ball, .23, [0, 0, 0], material('#f4f6f1', .05, .7));
  const patchMaterial = material('#10202c', .05, .65);
  const phi = (1 + Math.sqrt(5)) / 2;
  const patchCenters: THREE.Vector3[] = [];
  for (const a of [-1, 1]) for (const b of [-phi, phi]) patchCenters.push(v(0, a, b), v(a, b, 0), v(b, 0, a));
  for (const center of patchCenters) {
    const normal = center.normalize();
    const tangent = v(0, 1, 0).cross(normal).normalize();
    const bitangent = normal.clone().cross(tangent);
    const polygon = Array.from({length: 5}, (_, i) => normal.clone().addScaledVector(tangent, .38 * Math.cos(i * Math.PI * 2 / 5)).addScaledVector(bitangent, .38 * Math.sin(i * Math.PI * 2 / 5)).normalize().multiplyScalar(.232));
    // Project subdivided patches onto the sphere so their interiors stay above its surface.
    const vertices: number[] = [];
    const triangle = (a: THREE.Vector3, b: THREE.Vector3, c: THREE.Vector3, depth: number) => {
      if (!depth) {vertices.push(...a.toArray(), ...b.toArray(), ...c.toArray()); return;}
      const ab = a.clone().add(b).normalize().multiplyScalar(.232);
      const bc = b.clone().add(c).normalize().multiplyScalar(.232);
      const ca = c.clone().add(a).normalize().multiplyScalar(.232);
      triangle(a, ab, ca, depth - 1); triangle(ab, b, bc, depth - 1);
      triangle(ca, bc, c, depth - 1); triangle(ab, bc, ca, depth - 1);
    };
    polygon.forEach((point, i) => triangle(normal.clone().multiplyScalar(.232), point, polygon[(i + 1) % 5], 2));
    const geometry = new THREE.BufferGeometry(); geometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3)); geometry.computeVertexNormals();
    addMesh(ball, geometry, patchMaterial, [0, 0, 0]);
  }

  const sensing = new THREE.Group(); scene.add(sensing);
  const sensorMaterial = new THREE.MeshBasicMaterial({color: '#88eeff', transparent: true, opacity: .85, side: THREE.DoubleSide, depthWrite: false});
  const ring = (parent: THREE.Object3D, radius: number, surface: THREE.Material) => {
    const mesh = addMesh(parent, new THREE.RingGeometry(radius, radius + .024, 64), surface, [0, .025, 0]);
    mesh.rotation.x = -Math.PI / 2; mesh.castShadow = false; return mesh;
  };
  const ballRing = ring(sensing, .4, sensorMaterial);
  const keeperRing = ring(sensing, .7, sensorMaterial);
  const plan = new THREE.Mesh(new THREE.BufferGeometry(), new THREE.MeshBasicMaterial({color: '#135df2', toneMapped: false}));
  scene.add(plan);
  const targetRing = ring(scene, .42, sensorMaterial);
  const successMaterial = new THREE.MeshBasicMaterial({color: '#b0ffad', transparent: true, opacity: .85, side: THREE.DoubleSide});
  const success = ring(scene, .55, successMaterial);
  const visionGeometry = new THREE.BufferGeometry();
  const vision = new THREE.Mesh(visionGeometry, new THREE.MeshBasicMaterial({color: '#77dfff', transparent: true, opacity: .055, side: THREE.DoubleSide, depthWrite: false}));
  sensing.add(vision);
  let previousSide: boolean | undefined;

  const update = ({elapsed, keeperNear, stage}: FootballFrame) => {
    const walk = smooth(elapsed, 6000, 7800);
    const swing = elapsed > 6000 && elapsed < 7800 ? Math.sin(walk * Math.PI * 6) : 0;
    const kick = elapsed < 8800 ? smooth(elapsed, 8000, 8800) : 1 - smooth(elapsed, 8800, 9800);
    const flight = smooth(elapsed, 8800, 11000);
    const targetZ = keeperNear ? -1.22 : 1.22;
    striker.root.position.set(-3.4 + walk * 1.27, Math.abs(swing) * .025, 0);
    striker.root.rotation.y = THREE.MathUtils.lerp(Math.PI / 2, Math.atan2(8.3, targetZ + .22), walk);
    striker.legs.forEach(({hip, knee}, i) => {
      hip.rotation.x = i ? -kick * .68 + swing * .23 : -swing * .23;
      knee.rotation.x = Math.max(0, (i ? -swing : swing)) * .3;
    });
    striker.arms.forEach(({shoulder, elbow}, i) => {
      shoulder.rotation.x = (i ? -.5 : .5) * swing - kick * .22;
      shoulder.rotation.z = (i ? 1 : -1) * (.13 + kick * .23);
      elbow.rotation.x = -.15 - kick * .24;
    });
    striker.head.rotation.y = stage === 1 ? (keeperNear ? .2 : -.2) : 0;
    keeper.root.position.z = keeperNear ? 1.1 : -1.1;
    ball.position.set(-1.5 + flight * 8.3, .23 + Math.sin(flight * Math.PI) * .68, -.22 + flight * (targetZ + .22));
    ball.rotation.set(flight * 4, 0, -flight * 18);
    sensing.visible = stage === 0 || stage === 3;
    ballRing.position.set(ball.position.x, .025, ball.position.z);
    keeperRing.position.set(5.35, .025, keeper.root.position.z);
    plan.visible = stage === 1 || stage === 2;
    targetRing.visible = stage === 1 || stage === 2;
    targetRing.position.set(6.2, .025, targetZ);
    success.visible = stage === 3;
    success.position.set(6.8, .028, targetZ);
    if (previousSide !== keeperNear) {
      previousSide = keeperNear;
      const points = Array.from({length: 65}, (_, i) => {const t = i / 64; return v(-1.5 + t * 8.3, .23 + Math.sin(t * Math.PI) * .68, -.22 + t * (targetZ + .22));});
      plan.geometry.dispose(); plan.geometry = new THREE.TubeGeometry(new THREE.CatmullRomCurve3(points), 64, .022, 6, false);
      visionGeometry.setAttribute('position', new THREE.Float32BufferAttribute([-3.3, .018, 0, 6, .018, -2.9, 6, .018, 2.9], 3));
    }
    visionGeometry.attributes.position.setXYZ(0, striker.root.position.x + .1, .018, striker.root.position.z);
    visionGeometry.attributes.position.needsUpdate = true;
    render();
  };

  const resetView = () => {
    const scale = Math.max(1, 1.5 / Math.max(camera.aspect, .7));
    controls.target.set(.7, .7, 0);
    camera.position.set(-6.8 * scale, .7 + 7.1 * scale, 10.5 * scale);
    controls.update(); render();
  };
  const resize = () => {
    const {width, height} = host.getBoundingClientRect();
    if (!width || !height) return;
    camera.aspect = width / height; camera.updateProjectionMatrix(); renderer.setSize(width, height);
    resetView();
  };
  const onKey = (event: KeyboardEvent) => {
    if (event.key === 'Home') {event.preventDefault(); resetView(); return;}
    if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown'].includes(event.key)) return;
    event.preventDefault();
    const orbit = new THREE.Spherical().setFromVector3(camera.position.clone().sub(controls.target));
    orbit.theta += event.key === 'ArrowLeft' ? -.12 : event.key === 'ArrowRight' ? .12 : 0;
    orbit.phi = THREE.MathUtils.clamp(orbit.phi + (event.key === 'ArrowUp' ? -.1 : event.key === 'ArrowDown' ? .1 : 0), controls.minPolarAngle, controls.maxPolarAngle);
    camera.position.setFromSpherical(orbit).add(controls.target); controls.update(); render();
  };
  const onContextLost = (event: Event) => {event.preventDefault(); onUnavailable();};
  const observer = new ResizeObserver(resize); observer.observe(host);
  renderer.domElement.addEventListener('keydown', onKey);
  renderer.domElement.addEventListener('webglcontextlost', onContextLost);
  controls.addEventListener('change', render);
  resize(); update({elapsed: 0, keeperNear: false, stage: 0});
  return {
    update, resetView,
    dispose() {
      observer.disconnect(); controls.removeEventListener('change', render); controls.dispose();
      renderer.domElement.removeEventListener('keydown', onKey);
      renderer.domElement.removeEventListener('webglcontextlost', onContextLost);
      const geometries = new Set<THREE.BufferGeometry>(); const materials = new Set<THREE.Material>();
      scene.traverse(object => {
        if (object instanceof THREE.Mesh || object instanceof THREE.Line) {
          geometries.add(object.geometry);
          (Array.isArray(object.material) ? object.material : [object.material]).forEach(item => materials.add(item));
        }
      });
      geometries.forEach(item => item.dispose()); materials.forEach(item => item.dispose()); grass.dispose();
      sun.shadow.map?.dispose(); renderer.dispose(); renderer.domElement.remove();
    },
  };
}
