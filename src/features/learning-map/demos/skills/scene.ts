import {translate} from '@docusaurus/Translate';
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {RoundedBoxGeometry} from 'three/addons/geometries/RoundedBoxGeometry.js';
import type {Skill} from '../../skills/types';
import {skillDemos} from './content';
import {drawerPose, graspPose, navigationPose, navObstacles, routePoint, walkPose} from './motion';
import type {Point} from './motion';

export type SkillScene = {update: (progress: number) => void; resetView: () => void; dispose: () => void};

export function createSkillScene(host: HTMLDivElement, skill: Skill, onUnavailable: () => void): SkillScene {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color('#142132');
  scene.fog = new THREE.Fog('#142132', 14, 32);
  const camera = new THREE.PerspectiveCamera(36, 1, .1, 80);
  const renderer = new THREE.WebGLRenderer({antialias: true, powerPreference: 'low-power'});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.25;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  const canvas = renderer.domElement;
  canvas.tabIndex = 0;
  canvas.setAttribute('role', 'img');
  canvas.setAttribute('aria-label', translate({message: '{scene}拖动或用方向键旋转，Home 键复位视角。'}, {scene: skillDemos[skill].scene}));
  canvas.style.touchAction = 'pan-y';
  host.appendChild(canvas);
  const controls = new OrbitControls(camera, canvas);
  controls.enableZoom = false;
  controls.enablePan = false;
  controls.minPolarAngle = .3;
  controls.maxPolarAngle = Math.PI / 2.15;
  controls.minAzimuthAngle = -Math.PI;
  controls.maxAzimuthAngle = Math.PI;
  const render = () => renderer.render(scene, camera);
  const vector = (point: Point) => new THREE.Vector3(...point);
  const materials: THREE.Material[] = [];
  const geometries: THREE.BufferGeometry[] = [];
  const textures: THREE.Texture[] = [];
  const surface = (color: string, metalness = .15, roughness = .4) => {
    const mat = new THREE.MeshStandardMaterial({color, metalness, roughness});
    materials.push(mat);
    return mat;
  };
  const white = surface('#e1ebef', .3, .3);
  const blue = surface('#2069d8', .4, .3);
  const dark = surface('#203144', .55, .34);
  const rubber = surface('#101d28', .05, .85);
  const silver = surface('#93a9b8', .65, .27);
  const orange = surface('#fba54e', .05, .55);
  const floor = surface('#34485d', .1, .8);
  const cyan = surface('#6fe9ff', .1, .3);
  cyan.emissive.set('#23a9d3'); cyan.emissiveIntensity = .6;
  const green = surface('#83e5b4', .1, .4);
  const mesh = (parent: THREE.Object3D, geometry: THREE.BufferGeometry, material: THREE.Material, point: Point) => {
    geometries.push(geometry);
    const item = new THREE.Mesh(geometry, material);
    item.position.set(...point);
    item.castShadow = true; item.receiveShadow = true;
    parent.add(item);
    return item;
  };
  const box = (parent: THREE.Object3D, size: Point, point: Point, material: THREE.Material, radius = .04) =>
    mesh(parent, new RoundedBoxGeometry(...size, 2, radius), material, point);
  const cylinder = (parent: THREE.Object3D, radius: number, height: number, point: Point, material: THREE.Material) =>
    mesh(parent, new THREE.CylinderGeometry(radius, radius, height, 32), material, point);
  const ball = (parent: THREE.Object3D, radius: number, point: Point, material: THREE.Material) =>
    mesh(parent, new THREE.SphereGeometry(radius, 20, 12), material, point);
  const placeRod = (item: THREE.Object3D, a: THREE.Vector3, b: THREE.Vector3) => {
    item.position.copy(a).add(b).multiplyScalar(.5);
    item.scale.y = a.distanceTo(b);
    item.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), b.clone().sub(a).normalize());
  };
  const rod = (parent: THREE.Object3D, a: Point, b: Point, radius: number, material: THREE.Material) => {
    const item = cylinder(parent, radius, 1, [0, 0, 0], material);
    placeRod(item, vector(a), vector(b));
    return item;
  };
  const ring = (parent: THREE.Object3D, radius: number, point: Point, material: THREE.Material) => {
    const item = mesh(parent, new THREE.TorusGeometry(radius, .024, 8, 64), material, point);
    item.rotation.x = Math.PI / 2;
    return item;
  };
  function label(text: string, point: Point, color = '#dfeeff') {
    const textureCanvas = document.createElement('canvas');
    textureCanvas.width = 512; textureCanvas.height = 128;
    const context = textureCanvas.getContext('2d')!;
    context.fillStyle = '#162535';
    context.beginPath(); context.roundRect(4, 4, 504, 120, 24); context.fill();
    context.strokeStyle = '#56728b'; context.lineWidth = 2; context.stroke();
    context.fillStyle = color; context.font = '500 48px system-ui, sans-serif';
    // English labels can be longer than Chinese; fit them inside the badge.
    const textWidth = context.measureText(text).width;
    if (textWidth > 464) context.font = `500 ${48 * 464 / textWidth}px system-ui, sans-serif`;
    context.textAlign = 'center'; context.textBaseline = 'middle'; context.fillText(text, 256, 66);
    const texture = new THREE.CanvasTexture(textureCanvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    textures.push(texture);
    const material = new THREE.SpriteMaterial({map: texture, depthTest: true, toneMapped: false});
    materials.push(material);
    const sprite = new THREE.Sprite(material);
    sprite.position.set(...point); sprite.scale.set(1.55, .3875, 1); scene.add(sprite);
    return sprite;
  }

  scene.add(new THREE.HemisphereLight('#d7edff', '#304054', 2.2));
  const light = new THREE.DirectionalLight('#fff2df', 3.3);
  light.position.set(-3, 8, 6); light.castShadow = true;
  light.shadow.mapSize.set(1024, 1024);
  Object.assign(light.shadow.camera, {left: -6, right: 6, top: 5, bottom: -5, near: .1, far: 25});
  light.shadow.normalBias = .015; light.shadow.bias = -.0001;
  scene.add(light);
  const rim = new THREE.DirectionalLight('#96baff', 2.3);
  rim.position.set(5, 4, -5); scene.add(rim);
  const background = box(scene, [100, .2, 100], [0, -.7, 0], surface('#142132', .05, .9));
  background.castShadow = false;

  const mobile = skill === 'navigation' || skill === 'locomotion';
  const boardWidth = mobile ? 8.8 : 5.6;
  const boardDepth = skill === 'navigation' ? 5.5 : mobile ? 3.9 : 3.5;
  box(scene, [boardWidth + .12, .23, boardDepth + .12], [0, -.22, 0], dark, .09);
  box(scene, [boardWidth, .14, boardDepth], [0, -.055, 0], floor, .06);
  const gridPoints: THREE.Vector3[] = [];
  for (let x = -Math.floor(boardWidth / 2); x <= boardWidth / 2; x += .5) gridPoints.push(vector([x, .018, -boardDepth / 2 + .05]), vector([x, .018, boardDepth / 2 - .05]));
  for (let z = -Math.floor(boardDepth / 2); z <= boardDepth / 2; z += .5) gridPoints.push(vector([-boardWidth / 2 + .05, .018, z]), vector([boardWidth / 2 - .05, .018, z]));
  const gridGeometry = new THREE.BufferGeometry().setFromPoints(gridPoints); geometries.push(gridGeometry);
  const gridMaterial = new THREE.LineBasicMaterial({color: '#7897ae', transparent: true, opacity: .17}); materials.push(gridMaterial);
  scene.add(new THREE.LineSegments(gridGeometry, gridMaterial));
  box(scene, [boardWidth - .35, .025, .027], [0, -.1, boardDepth / 2 + .066], cyan, .008);

  // The two links keep fixed lengths; only their joint angles change with the target.
  function elbowBetween(a: THREE.Vector3, b: THREE.Vector3, l1: number, l2: number, bend: THREE.Vector3) {
    const direction = b.clone().sub(a);
    const distance = Math.max(.001, direction.length());
    direction.normalize();
    const along = (l1 * l1 - l2 * l2 + distance * distance) / (2 * distance);
    const height = Math.sqrt(Math.max(0, l1 * l1 - along * along));
    const perpendicular = bend.clone().addScaledVector(direction, -bend.dot(direction)).normalize();
    return a.clone().addScaledVector(direction, along).addScaledVector(perpendicular, height);
  }

  function arm(horizontal: boolean) {
    const baseX = horizontal ? -1.85 : -1.55;
    box(scene, [.9, .12, .85], [baseX, .08, 0], silver, .06);
    for (const x of [-.32, .32]) for (const z of [-.29, .29]) cylinder(scene, .035, .02, [baseX + x, .15, z], dark);
    cylinder(scene, .3, .28, [baseX, .27, 0], white);
    cylinder(scene, .31, .09, [baseX, .45, 0], blue);
    const shoulder = vector([baseX, .56, 0]);
    const upper = box(scene, [.26, 1, .3], [0, 0, 0], white, .08);
    const lower = box(scene, [.22, 1, .26], [0, 0, 0], white, .065);
    const upperAccent = box(upper, [.12, .68, .035], [0, 0, .155], blue, .015);
    upperAccent.castShadow = false;
    box(lower, [.1, .65, .032], [0, 0, .136], blue, .01);
    const joints = [shoulder, shoulder, shoulder].map(position => {
      const joint = new THREE.Group(); scene.add(joint);
      const axle = cylinder(joint, .19, .37, [0, 0, 0], dark); axle.rotation.x = Math.PI / 2;
      for (const z of [-.2, .2]) {
        const cap = cylinder(joint, .135, .035, [0, 0, z], blue); cap.rotation.x = Math.PI / 2;
        const bolt = cylinder(joint, .046, .038, [0, 0, z * 1.1], silver); bolt.rotation.x = Math.PI / 2;
      }
      joint.position.copy(position); return joint;
    });
    const gripper = new THREE.Group(); scene.add(gripper);
    if (horizontal) gripper.rotation.z = Math.PI / 2;
    cylinder(gripper, .14, .15, [0, .53, 0], silver);
    box(gripper, [.65, .15, .28], [0, .38, 0], dark, .035);
    box(gripper, [.4, .07, .035], [0, .4, .155], blue, .016);
    box(gripper, [.07, .025, .025], [0, .4, .18], cyan, .009);
    const fingers = [-1, 1].map(side => {
      const finger = new THREE.Group(); gripper.add(finger);
      box(finger, [.085, .44, .18], [0, .06, 0], silver, .015);
      box(finger, [.022, .25, .17], [-side * .052, -.015, 0], rubber, .01);
      box(finger, [.13, .1, .25], [0, .27, 0], white, .014);
      return finger;
    });
    return {
      update(grip: Point, opening: number) {
        gripper.position.set(...grip);
        fingers.forEach((finger, index) => {finger.position.x = (index ? 1 : -1) * opening;});
        gripper.updateMatrixWorld(true);
        const wrist = gripper.localToWorld(new THREE.Vector3(0, .58, 0));
        const elbow = elbowBetween(shoulder, wrist, 1.45, 1.35, new THREE.Vector3(0, 1, 0));
        placeRod(upper, shoulder, elbow); placeRod(lower, elbow, wrist);
        joints[1].position.copy(elbow); joints[2].position.copy(wrist);
        joints[2].scale.setScalar(.72);
      },
    };
  }

  let animate: (p: number) => void;
  if (skill === 'grasping') {
    const robot = arm(false);
    const cube = box(scene, [.46, .46, .46], [.9, .24, .25], orange, .045);
    box(cube, [.27, .007, .27], [0, .234, 0], white, .025);
    const marker = ring(scene, .48, [.9, .035, .25], cyan);
    label(translate({message: "目标物体"}), [1.5, .3, 1.05], '#ffcf91');
    label(translate({message: "平行夹爪"}), [-.65, .6, -1.15]);
    animate = p => {
      const pose = graspPose(p); robot.update(pose.grip, pose.opening);
      cube.position.set(...pose.object);
      marker.material = p >= .94 ? green : cyan;
    };
  } else if (skill === 'manipulation') {
    const robot = arm(true);
    const cabinet = new THREE.Group(); scene.add(cabinet);
    // An open-front cabinet and a separate drawer make the sliding constraint visible.
    box(cabinet, [1.48, .13, 1.5], [1.02, 1.59, 0], white);
    box(cabinet, [1.48, .12, 1.5], [1.02, .13, 0], white);
    box(cabinet, [.12, 1.5, 1.5], [1.72, .84, 0], white);
    for (const z of [-.69, .69]) {
      box(cabinet, [1.42, 1.35, .12], [1.02, .86, z], white);
      box(cabinet, [1.18, .055, .04], [.89, .69, z * .86], silver, .005);
    }
    for (const x of [.45, 1.59]) for (const z of [-.56, .56]) cylinder(cabinet, .06, .12, [x, .05, z], dark);
    const drawer = new THREE.Group(); scene.add(drawer);
    box(drawer, [.12, .72, 1.25], [.35, .94, 0], blue);
    box(drawer, [1.16, .08, 1.15], [.97, .59, 0], silver);
    for (const z of [-.53, .53]) box(drawer, [1.15, .43, .06], [.97, .82, z], silver, .01);
    box(drawer, [.07, .43, 1.12], [1.53, .82, 0], silver, .01);
    for (const z of [-.25, .25]) rod(drawer, [.28, .94, z], [.17, .94, z], .033, silver);
    rod(drawer, [.17, .94, -.25], [.17, .94, .25], .041, silver);
    box(drawer, [.28, .2, .26], [1.05, .73, .14], orange);
    label(translate({message: "沿滑轨拉开"}), [.9, .24, 1.18], '#9feeff');
    animate = p => {
      const pose = drawerPose(p);
      drawer.position.x = -pose.distance;
      robot.update(pose.grip, pose.opening);
    };
  } else if (skill === 'locomotion') {
    for (const [from, to, height] of [[-.2, .75, .2], [.75, 1.7, .4], [1.7, 3.65, .6]]) {
      box(scene, [to - from, height, 2.75], [(from + to) / 2, height / 2 + .02, 0], silver, .025);
      box(scene, [.035, .013, 2.6], [from + .055, height + .028, 0], cyan, .004);
    }
    const body = new THREE.Group(); scene.add(body);
    box(body, [1.42, .38, .64], [0, 0, 0], white, .12);
    box(body, [1.02, .06, .51], [-.06, .215, 0], blue, .035);
    box(body, [1.08, .16, .51], [0, -.2, 0], dark, .06);
    const sensor = box(body, [.12, .23, .48], [.735, .025, 0], dark, .045);
    for (const z of [-.14, .14]) ball(sensor, .045, [.07, .015, z], cyan);
    cylinder(body, .13, .08, [.37, .25, 0], dark);
    for (const side of [-1, 1]) {
      for (let i = 0; i < 4; i++) box(body, [.075, .05, .022], [-.3 + i * .13, .05, side * .327], dark, .008);
    }
    const legs = Array.from({length: 4}, (_, i) => {
      const upper = box(scene, [.14, 1, .18], [0, 0, 0], blue, .04);
      const lower = box(scene, [.105, 1, .12], [0, 0, 0], white, .035);
      const hip = ball(scene, .14, [0, 0, 0], dark);
      const knee = ball(scene, .11, [0, 0, 0], dark);
      const foot = box(scene, [.2, .12, .22], [0, 0, 0], rubber, .045);
      const hipCap = cylinder(hip, .092, .2, [0, 0, 0], silver); hipCap.rotation.x = Math.PI / 2;
      return {upper, lower, hip, knee, foot, offset: vector([i < 2 ? .55 : -.55, -.08, i % 2 ? .36 : -.36])};
    });
    label(translate({message: "落脚与支撑"}), [-2.65, .25, 1.42]);
    label(translate({message: "逐级抬高身体"}), [2.15, .94, -1.3], '#9feeff');
    animate = p => {
      const pose = walkPose(p);
      body.position.set(...pose.body); body.rotation.z = pose.pitch; body.updateMatrixWorld(true);
      legs.forEach((leg, i) => {
        const hip = body.localToWorld(leg.offset.clone());
        const foot = vector(pose.feet[i]);
        const knee = elbowBetween(hip, foot, .64, .66, new THREE.Vector3(i < 2 ? -1 : 1, 0, 0));
        leg.hip.position.copy(hip); leg.knee.position.copy(knee); leg.foot.position.copy(foot);
        placeRod(leg.upper, hip, knee); placeRod(leg.lower, knee, foot);
      });
    };
  } else {
    navObstacles.forEach(({x, z, width, depth}, i) => {
      box(scene, [width, .95, depth], [x, .5, z], i ? white : silver, .07);
      box(scene, [width + .025, .08, depth + .025], [x, .99, z], orange, .025);
    });
    const robot = new THREE.Group(); scene.add(robot);
    cylinder(robot, .35, .24, [0, .29, 0], white);
    cylinder(robot, .32, .065, [0, .445, 0], blue);
    cylinder(robot, .29, .07, [0, .16, 0], dark);
    cylinder(robot, .115, .12, [0, .54, 0], dark);
    cylinder(robot, .12, .03, [0, .61, 0], cyan);
    box(robot, [.25, .075, .075], [0, .31, .33], dark, .022);
    for (const x of [-.065, .065]) ball(robot, .022, [x, .32, .376], cyan);
    const wheels = [-1, 1].map(side => {
      const wheel = new THREE.Group(); robot.add(wheel); wheel.position.set(side * .33, .16, 0);
      const tire = cylinder(wheel, .17, .075, [0, 0, 0], rubber); tire.rotation.z = Math.PI / 2;
      const hub = cylinder(wheel, .1, .085, [0, 0, 0], silver); hub.rotation.z = Math.PI / 2;
      box(wheel, [.088, .035, .16], [0, 0, 0], dark, .008); return wheel;
    });
    const start = routePoint(0), end = routePoint(1);
    ring(scene, .45, [start[0], .042, start[2]], silver);
    const goal = ring(scene, .47, [end[0], .042, end[2]], orange);
    label(translate({message: "起点"}), [start[0], .35, start[2] + .7]);
    label(translate({message: "目标"}), [end[0], .5, end[2] - .55], '#ffcf91');
    class Route extends THREE.Curve<THREE.Vector3> {
      constructor() {super();}
      getPoint(t: number, target = new THREE.Vector3()) {return target.set(...routePoint(t));}
      getPointAt(t: number, target = new THREE.Vector3()) {return this.getPoint(t, target);}
      getTangentAt(t: number, target = new THREE.Vector3()) {return this.getTangent(t, target);}
    }
    const curve = new Route();
    const pathGeometry = new THREE.TubeGeometry(curve, 180, .029, 6, false);
    const path = mesh(scene, pathGeometry, cyan, [0, 0, 0]); path.castShadow = false;
    const completedGeometry = new THREE.TubeGeometry(curve, 180, .034, 6, false);
    const completedPath = mesh(scene, completedGeometry, green, [0, 0, 0]); completedPath.castShadow = false;
    animate = p => {
      const pose = navigationPose(p);
      robot.position.set(pose.point[0], 0, pose.point[2]); robot.rotation.y = pose.heading;
      wheels.forEach(wheel => {wheel.rotation.x = pose.travel * 48;});
      path.visible = p >= .22; completedPath.visible = p >= .38;
      // TubeGeometry's segment index groups preserve a continuous, growing trail.
      completedGeometry.setDrawRange(0, Math.floor(pose.travel * 180) * 6 * 6);
      goal.material = p >= .94 ? green : orange;
    };
  }

  const defaultPosition = skill === 'navigation' ? new THREE.Vector3(6.5, 8.5, 9.5)
    : skill === 'locomotion' ? new THREE.Vector3(5.4, 4.5, 7.5)
    : skill === 'manipulation' ? new THREE.Vector3(-4.8, 3.7, 6.8) : new THREE.Vector3(4.5, 3.7, 6.5);
  const target = new THREE.Vector3(0, skill === 'locomotion' ? -.25 : mobile ? .25 : .6, 0);
  let disposed = false;
  let contextLost = false;
  const resetView = () => {camera.position.copy(defaultPosition); controls.target.copy(target); controls.update(); render();};
  const resize = () => {
    if (disposed || contextLost) return;
    const width = host.clientWidth || 640, height = host.clientHeight || 380;
    camera.aspect = width / height;
    // Expand the vertical field on narrow screens so the complete work area still fits.
    camera.fov = camera.aspect < 1.35 ? 49 : 36;
    camera.updateProjectionMatrix(); renderer.setSize(width, height, false); render();
  };
  const keydown = (event: KeyboardEvent) => {
    if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home'].includes(event.key)) return;
    event.preventDefault();
    if (event.key === 'Home') {resetView(); return;}
    const spherical = new THREE.Spherical().setFromVector3(camera.position.clone().sub(controls.target));
    if (event.key === 'ArrowLeft') spherical.theta -= .12;
    if (event.key === 'ArrowRight') spherical.theta += .12;
    if (event.key === 'ArrowUp') spherical.phi -= .1;
    if (event.key === 'ArrowDown') spherical.phi += .1;
    spherical.phi = THREE.MathUtils.clamp(spherical.phi, controls.minPolarAngle, controls.maxPolarAngle);
    camera.position.copy(controls.target).add(new THREE.Vector3().setFromSpherical(spherical)); controls.update(); render();
  };
  const lost = (event: Event) => {event.preventDefault(); contextLost = true; onUnavailable();};
  canvas.addEventListener('keydown', keydown);
  canvas.addEventListener('webglcontextlost', lost);
  controls.addEventListener('change', render);
  const observer = new ResizeObserver(resize); observer.observe(host);
  animate(0); resetView(); resize();
  return {
    update(progress) {if (!disposed && !contextLost) {animate(progress); render();}},
    resetView,
    dispose() {
      disposed = true; observer.disconnect(); controls.dispose();
      canvas.removeEventListener('keydown', keydown); canvas.removeEventListener('webglcontextlost', lost);
      controls.removeEventListener('change', render);
      geometries.forEach(geometry => geometry.dispose()); materials.forEach(material => material.dispose()); textures.forEach(texture => texture.dispose());
      renderer.dispose(); renderer.forceContextLoss(); canvas.remove();
    },
  };
}
