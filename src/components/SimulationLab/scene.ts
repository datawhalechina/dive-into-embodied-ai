import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';

export type Geometry = {id: number; type: number; size?: Float64Array; mesh?: number; vertices?: Float32Array; faces?: Int32Array; color: Float32Array};
export type Snapshot = {time: number; paused: boolean; mode: 'manual' | 'showcase'; position: Float64Array; positions: Float64Array; rotations: Float64Array; speed: number; upright: number};

export function createScene(host: HTMLElement, distance: number, label: string, onLost: () => void) {
  const renderer = new THREE.WebGLRenderer({antialias: true});
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFShadowMap;
  renderer.domElement.setAttribute('role', 'img');
  renderer.domElement.setAttribute('aria-label', label);
  host.appendChild(renderer.domElement);
  const lost = (event: Event) => { event.preventDefault(); onLost(); };
  renderer.domElement.addEventListener('webglcontextlost', lost);
  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(42, 1, .001, 500);
  camera.up.set(0, 0, 1);
  const controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = !matchMedia('(prefers-reduced-motion: reduce)').matches;
  controls.enablePan = false;
  controls.minDistance = distance * .4;
  controls.maxDistance = distance * 4;
  controls.maxPolarAngle = Math.PI / 2 - .02;
  const ambient = new THREE.HemisphereLight(0xffffff, 0x52657d, 2);
  ambient.up.set(0, 0, 1);
  scene.add(ambient);
  const sun = new THREE.DirectionalLight(0xfff5e4, 2.6);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  Object.assign(sun.shadow.camera, {left: -distance * 2, right: distance * 2, top: distance * 2, bottom: -distance * 2, near: .001, far: distance * 12});
  sun.shadow.bias = -.0001;
  sun.shadow.normalBias = distance * .002;
  scene.add(sun, sun.target);
  const floorMaterial = new THREE.MeshStandardMaterial({roughness: 1});
  const floor = new THREE.Mesh(new THREE.PlaneGeometry(2000, 2000), floorMaterial);
  floor.position.z = -.0002; floor.receiveShadow = true; scene.add(floor);
  const grid = new THREE.GridHelper(distance * 80, 160, '#8395aa', '#bcc7d4');
  grid.rotation.x = Math.PI / 2; grid.position.z = .0001; scene.add(grid);
  const meshes = new Map<number, THREE.Mesh>();
  const geometries = new Set<THREE.BufferGeometry>();
  const materials = new Set<THREE.Material>();
  let body = new THREE.Vector3(0, 0, distance * .15);
  let first = true, active = true, follow = true, frame = 0;
  const rotation = new THREE.Matrix4();
  function resetView() {
    controls.target.copy(body);
    camera.position.copy(body).add(new THREE.Vector3(distance * .85, -distance, distance * .6));
    controls.update();
  }
  function setDark(dark: boolean) {
    const color = dark ? '#1b2433' : '#edf2f7';
    renderer.setClearColor(color);
    scene.fog = new THREE.Fog(color, distance * 8, distance * 35);
    floorMaterial.color.set(dark ? '#293548' : '#e4ebf3');
    (grid.material as THREE.LineBasicMaterial).opacity = dark ? .18 : .35;
    (grid.material as THREE.LineBasicMaterial).transparent = true;
  }
  resetView(); setDark(false);
  const resize = new ResizeObserver(() => {
    const {clientWidth: width, clientHeight: height} = host;
    if (!width || !height) return;
    renderer.setSize(width, height, false); camera.aspect = width / height; camera.updateProjectionMatrix();
  });
  resize.observe(host);
  function render() {
    frame = requestAnimationFrame(render);
    if (active && !document.hidden) { controls.update(); renderer.render(scene, camera); }
  }
  render();
  return {
    resetView, setDark,
    setFollow(value: boolean) { follow = value; if (value) resetView(); },
    setVisible(value: boolean) { active = value; },
    load(items: Geometry[]) {
      for (const item of items) {
        const s = item.size ?? [1, 1, 1];
        let geometry: THREE.BufferGeometry;
        if (item.type === 7) {
          geometry = new THREE.BufferGeometry();
          geometry.setAttribute('position', new THREE.BufferAttribute(item.vertices!, 3));
          geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(item.faces!), 1));
          geometry.computeVertexNormals();
        } else if (item.type === 6) geometry = new THREE.BoxGeometry(s[0] * 2, s[1] * 2, s[2] * 2);
        else if (item.type === 5) geometry = new THREE.CylinderGeometry(s[0], s[0], s[1] * 2, 24).rotateX(Math.PI / 2);
        else if (item.type === 3) geometry = new THREE.CapsuleGeometry(s[0], s[1] * 2, 6, 16).rotateX(Math.PI / 2);
        else geometry = new THREE.SphereGeometry(1, 20, 12).scale(s[0], item.type === 4 ? s[1] : s[0], item.type === 4 ? s[2] : s[0]);
        const material = new THREE.MeshStandardMaterial({
          color: new THREE.Color(item.color[0], item.color[1], item.color[2]), roughness: .6, metalness: .15,
          opacity: item.color[3], transparent: item.color[3] < 1,
        });
        const mesh = new THREE.Mesh(geometry, material);
        mesh.castShadow = true; mesh.receiveShadow = true;
        scene.add(mesh); meshes.set(item.id, mesh); geometries.add(geometry); materials.add(material);
      }
    },
    update(state: Snapshot) {
      const next = new THREE.Vector3().fromArray(state.position);
      if (first) { body = next; resetView(); first = false; }
      else if (follow) { const delta = next.clone().sub(body); camera.position.add(delta); controls.target.add(delta); }
      body = next;
      sun.position.copy(body).add(new THREE.Vector3(distance, -distance, distance * 3));
      sun.target.position.copy(body);
      const tile = distance / 2;
      grid.position.x = Math.round(body.x / tile) * tile;
      grid.position.y = Math.round(body.y / tile) * tile;
      floor.position.x = body.x; floor.position.y = body.y;
      for (const [id, mesh] of meshes) {
        const r = id * 9, m = state.rotations;
        mesh.position.fromArray(state.positions, id * 3);
        rotation.set(m[r], m[r+1], m[r+2], 0, m[r+3], m[r+4], m[r+5], 0, m[r+6], m[r+7], m[r+8], 0, 0, 0, 0, 1);
        mesh.quaternion.setFromRotationMatrix(rotation);
      }
    },
    dispose() {
      cancelAnimationFrame(frame); resize.disconnect(); controls.dispose();
      renderer.domElement.removeEventListener('webglcontextlost', lost);
      geometries.forEach(g => g.dispose()); materials.forEach(m => m.dispose());
      floor.geometry.dispose(); floorMaterial.dispose(); grid.geometry.dispose();
      (grid.material as THREE.Material).dispose(); sun.shadow.dispose();
      renderer.dispose(); renderer.forceContextLoss(); renderer.domElement.remove();
    },
  };
}
