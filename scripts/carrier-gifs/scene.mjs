import * as THREE from 'three';
import {RoundedBoxGeometry} from 'three/addons/geometries/RoundedBoxGeometry.js';

export const kinds = ['humanoid', 'arm', 'wheeled', 'quadruped', 'mobile-manipulator', 'simulation'];
const tau = Math.PI * 2;
const mix = (a, b, t) => a + (b - a) * t;
const ease = (t, a, b) => {const x = THREE.MathUtils.clamp((t - a) / (b - a), 0, 1); return x * x * (3 - 2 * x);};
const point = (x, y, z) => new THREE.Vector3(x, y, z);

// Offline illustration geometry: no product-specific model or trained policy is implied.
export function createCarrierScene(kind) {
  const scene = new THREE.Scene();
  scene.background = new THREE.Color('#142132');
  const camera = new THREE.OrthographicCamera(-3.68, 3.68, 2.3, -2.3, .1, 100);
  camera.position.set(5.2, 4.4, 7.4);
  camera.lookAt(0, .72, 0);
  const resources = new Set();
  const surface = (color, metalness = .25, roughness = .34) => {
    const material = new THREE.MeshStandardMaterial({color, metalness, roughness});
    resources.add(material); return material;
  };
  const white = surface('#e5eff4');
  const blue = surface('#2675e4', .45);
  const dark = surface('#1a2b40', .45);
  const rubber = surface('#101923', 0, .85);
  const silver = surface('#9bafc1', .7, .3);
  const orange = surface('#ffb258', .05, .45);
  const floor = surface('#30465b', .1, .9);
  const cyan = surface('#79e4ff', .1);
  cyan.emissive.set('#236f9a');
  const add = (parent, geometry, material, position = [0, 0, 0]) => {
    resources.add(geometry);
    const mesh = new THREE.Mesh(geometry, material);
    mesh.position.set(...position); mesh.castShadow = true; mesh.receiveShadow = true;
    parent.add(mesh); return mesh;
  };
  const box = (parent, size, position, material, radius = .04) => add(parent, new RoundedBoxGeometry(...size, 2, radius), material, position);
  const ball = (parent, radius, position, material) => add(parent, new THREE.SphereGeometry(radius, 24, 16), material, position);
  const cylinder = (parent, radius, height, position, material) => add(parent, new THREE.CylinderGeometry(radius, radius, height, 32), material, position);
  const group = (parent, position = [0, 0, 0]) => {const root = new THREE.Group(); root.position.set(...position); parent.add(root); return root;};
  const link = (mesh, a, b) => {
    mesh.position.copy(a).add(b).multiplyScalar(.5);
    mesh.scale.y = a.distanceTo(b);
    mesh.quaternion.setFromUnitVectors(point(0, 1, 0), b.clone().sub(a).normalize());
  };
  const elbow = (a, b, l1, l2, bend) => {
    const axis = b.clone().sub(a);
    const d = Math.max(.001, axis.length()); axis.normalize();
    const x = (l1 * l1 - l2 * l2 + d * d) / (2 * d);
    const height = Math.sqrt(Math.max(0, l1 * l1 - x * x));
    const perpendicular = bend.clone().addScaledVector(axis, -bend.dot(axis)).normalize();
    return a.clone().addScaledVector(axis, x).addScaledVector(perpendicular, height);
  };
  const lines = (parent, points, color, opacity = 1) => {
    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const material = new THREE.LineBasicMaterial({color, transparent: opacity < 1, opacity});
    resources.add(geometry); resources.add(material);
    const item = new THREE.LineSegments(geometry, material); parent.add(item); return item;
  };
  const ring = (parent, radius, position, material) => {
    const mesh = add(parent, new THREE.TorusGeometry(radius, .018, 8, 64), material, position);
    mesh.rotation.x = Math.PI / 2; return mesh;
  };

  scene.add(new THREE.HemisphereLight('#d9efff', '#354b68', 2.5));
  const key = new THREE.DirectionalLight('#fff2dd', 3.5);
  key.position.set(-3, 7, 5); key.castShadow = true;
  key.shadow.mapSize.set(1024, 1024);
  Object.assign(key.shadow.camera, {left: -5, right: 5, top: 5, bottom: -5, near: .1, far: 20});
  key.shadow.normalBias = .02; scene.add(key);
  const rim = new THREE.DirectionalLight('#82b7ff', 2.3); rim.position.set(5, 3, -4); scene.add(rim);
  box(scene, [100, .1, 100], [0, -.38, 0], surface('#142132', 0, 1)).castShadow = false;
  box(scene, [5.8, .23, 3.65], [0, -.2, 0], dark, .09);
  box(scene, [5.7, .12, 3.55], [0, -.045, 0], floor, .06);
  box(scene, [5.4, .025, .025], [0, -.1, 1.835], cyan, .005);
  const grid = [];
  for (let x = -2.5; x <= 2.5; x += .5) grid.push(point(x, .022, -1.7), point(x, .022, 1.7));
  for (let z = -1.5; z <= 1.5; z += .5) grid.push(point(-2.75, .022, z), point(2.75, .022, z));
  lines(scene, grid, '#8bbad2', .18);

  function humanoid(parent) {
    const root = group(parent);
    const torso = group(root);
    box(torso, [.57, .25, .38], [0, .97, 0], dark, .075);
    box(torso, [.77, .65, .44], [0, 1.46, 0], white, .12);
    box(torso, [.57, .35, .08], [0, 1.52, .255], blue, .065);
    box(torso, [.32, .055, .025], [0, 1.54, .303], cyan, .02);
    ball(torso, .13, [0, 1.83, 0], dark);
    const head = group(torso, [0, 2.04, 0]);
    box(head, [.59, .44, .49], [0, 0, 0], white, .12);
    box(head, [.5, .21, .08], [0, -.01, .245], dark, .06);
    for (const side of [-1, 1]) {
      box(head, [.12, .035, .025], [side * .115, 0, .293], cyan, .014);
      const ear = cylinder(head, .09, .055, [side * .3, 0, 0], silver); ear.rotation.z = Math.PI / 2;
    }
    box(head, [.27, .04, .26], [0, .223, 0], blue, .02);
    const arms = [-1, 1].map(side => {
      const shoulder = group(torso, [side * .49, 1.69, 0]);
      ball(shoulder, .16, [0, 0, 0], blue);
      box(shoulder, [.23, .33, .25], [0, -.21, 0], white, .07);
      const forearm = group(shoulder, [0, -.41, 0]);
      ball(forearm, .115, [0, 0, 0], dark);
      box(forearm, [.2, .29, .22], [0, -.18, 0], white, .06);
      box(forearm, [.23, .18, .24], [0, -.38, .03], blue, .055);
      shoulder.rotation.z = side * .14;
      return {shoulder, forearm};
    });
    const legs = [-1, 1].map(side => ({
      side, upper: box(root, [.25, 1, .27], [0, 0, 0], white, .07),
      lower: box(root, [.21, 1, .23], [0, 0, 0], white, .06),
      hip: ball(root, .135, [0, 0, 0], dark), knee: ball(root, .13, [0, 0, 0], blue),
      foot: box(root, [.33, .13, .49], [0, 0, 0], blue, .045),
    }));
    return {root, update(t) {
      const cycle = tau * 2 * t;
      const bob = .018 * Math.cos(cycle * 2);
      torso.position.y = bob; torso.rotation.z = .028 * Math.sin(cycle);
      head.rotation.y = .08 * Math.sin(cycle * .5);
      legs.forEach((leg, i) => {
        const swing = Math.max(0, Math.sin(cycle + i * Math.PI));
        const hip = point(leg.side * .22, .95 + bob, 0);
        const foot = point(leg.side * .23, .09 + .22 * swing, .08 + .19 * swing);
        const knee = elbow(hip, foot, .46, .46, point(0, 0, 1));
        leg.hip.position.copy(hip); leg.knee.position.copy(knee); leg.foot.position.copy(foot);
        link(leg.upper, hip, knee); link(leg.lower, knee, foot);
        arms[i].shoulder.rotation.x = .32 * Math.sin(cycle + i * Math.PI);
        arms[i].forearm.rotation.x = -.2 - .1 * swing;
      });
    }};
  }

  function quadruped(parent) {
    const root = group(parent);
    const body = group(root);
    box(body, [1.48, .36, .64], [0, 0, 0], white, .11);
    box(body, [1.08, .055, .5], [-.05, .2, 0], blue, .035);
    box(body, [1.08, .16, .5], [0, -.19, 0], dark, .04);
    box(body, [.13, .22, .46], [.75, .02, 0], dark, .035);
    for (const side of [-1, 1]) {
      ball(body, .045, [.823, .03, side * .13], cyan);
      for (let n = 0; n < 4; n++) box(body, [.07, .055, .016], [-.3 + n * .13, .02, side * .326], dark, .007);
    }
    cylinder(body, .12, .07, [.4, .23, 0], dark);
    const legs = Array.from({length: 4}, (_, i) => ({
      offset: point(i < 2 ? .53 : -.53, -.07, i % 2 ? .37 : -.37),
      upper: box(root, [.14, 1, .18], [0, 0, 0], blue, .04),
      lower: box(root, [.1, 1, .12], [0, 0, 0], white, .035),
      hip: ball(root, .14, [0, 0, 0], dark), knee: ball(root, .1, [0, 0, 0], silver),
      foot: box(root, [.19, .1, .2], [0, 0, 0], rubber, .04),
    }));
    return {root, update(t) {
      const cycle = tau * 2 * t;
      body.position.set(0, 1.04 + .025 * Math.cos(cycle * 2), 0);
      body.rotation.z = .018 * Math.sin(cycle);
      body.updateMatrix();
      legs.forEach((leg, i) => {
        const phase = cycle + (i === 0 || i === 3 ? 0 : Math.PI);
        const lift = Math.max(0, Math.sin(phase));
        const hip = leg.offset.clone().applyMatrix4(body.matrix);
        const foot = point(leg.offset.x + .15 * lift, .08 + .23 * lift, leg.offset.z);
        const knee = elbow(hip, foot, .56, .58, point(i < 2 ? -1 : 1, 0, 0));
        leg.hip.position.copy(hip); leg.knee.position.copy(knee); leg.foot.position.copy(foot);
        link(leg.upper, hip, knee); link(leg.lower, knee, foot);
      });
    }};
  }

  function wheeled(parent) {
    const root = group(parent);
    box(root, [.86, .3, .94], [0, .34, 0], white, .1);
    box(root, [.76, .05, .84], [0, .52, 0], blue, .055);
    box(root, [.82, .11, .94], [0, .18, 0], dark, .04);
    box(root, [.32, .1, .055], [0, .35, .49], dark, .024);
    for (const side of [-1, 1]) ball(root, .026, [side * .09, .36, .523], cyan);
    const scan = group(root, [0, .62, 0]);
    cylinder(scan, .13, .14, [0, 0, 0], dark);
    cylinder(scan, .132, .025, [0, .065, 0], cyan);
    const wheels = [];
    for (const side of [-1, 1]) for (const z of [-.31, .31]) {
      const wheel = group(root, [side * .45, .22, z]);
      const tire = cylinder(wheel, .21, .1, [0, 0, 0], rubber); tire.rotation.z = Math.PI / 2;
      const hub = cylinder(wheel, .125, .112, [0, 0, 0], silver); hub.rotation.z = Math.PI / 2;
      box(wheel, [.118, .035, .18], [0, 0, 0], blue, .008);
      wheels.push(wheel);
    }
    return {root, scan, wheels};
  }

  function arm(parent, origin = point(-1.15, .49, 0), l1 = 1.16, l2 = 1.1) {
    const base = group(parent, [origin.x, origin.y - .26, origin.z]);
    cylinder(base, .28, .4, [0, 0, 0], white);
    cylinder(base, .29, .08, [0, .17, 0], blue);
    const upper = box(parent, [.25, 1, .29], [0, 0, 0], white, .07);
    const lower = box(parent, [.21, 1, .25], [0, 0, 0], white, .06);
    box(upper, [.12, .7, .025], [0, 0, .153], blue, .018);
    box(lower, [.1, .66, .025], [0, 0, .132], blue, .014);
    const joints = Array.from({length: 3}, () => {
      const joint = group(parent);
      const axle = cylinder(joint, .17, .33, [0, 0, 0], dark); axle.rotation.x = Math.PI / 2;
      for (const side of [-1, 1]) {
        const cap = cylinder(joint, .12, .025, [0, 0, side * .18], blue);
        cap.rotation.x = Math.PI / 2;
      }
      return joint;
    });
    const gripper = group(parent);
    cylinder(gripper, .12, .13, [0, .5, 0], silver);
    box(gripper, [.7, .15, .25], [0, .36, 0], dark, .03);
    box(gripper, [.31, .055, .035], [0, .37, .14], blue, .016);
    const fingers = [-1, 1].map(side => {
      const finger = group(gripper);
      box(finger, [.07, .43, .17], [0, .09, 0], silver, .014);
      box(finger, [.022, .24, .15], [-side * .045, .035, 0], rubber, .008);
      return finger;
    });
    return {update(target, opening) {
      const wrist = target.clone().add(point(0, .56, 0));
      const mid = elbow(origin, wrist, l1, l2, point(0, 1, 0));
      link(upper, origin, mid); link(lower, mid, wrist);
      [origin, mid, wrist].forEach((position, i) => joints[i].position.copy(position));
      joints[2].scale.setScalar(.75);
      gripper.position.copy(target);
      fingers.forEach((finger, i) => {finger.position.x = (i ? 1 : -1) * opening;});
    }};
  }

  let update;
  if (kind === 'humanoid') {
    const robot = humanoid(scene); robot.root.rotation.y = -.35;
    ring(scene, 1.12, [0, .03, 0], silver);
    update = t => robot.update(t);
  } else if (kind === 'quadruped') {
    const robot = quadruped(scene); robot.root.scale.setScalar(1.28);
    camera.lookAt(0, .6, 0);
    update = t => robot.update(t);
  } else if (kind === 'wheeled') {
    const robot = wheeled(scene);
    const route = [];
    for (let i = 0; i < 100; i++) {
      const a = i / 100 * tau, b = (i + 1) / 100 * tau;
      route.push(point(1.85 * Math.cos(a), .035, 1.04 * Math.sin(a)), point(1.85 * Math.cos(b), .035, 1.04 * Math.sin(b)));
    }
    lines(scene, route, '#79e4ff', .8);
    box(scene, [.52, .63, .52], [0, .335, 0], silver, .045);
    box(scene, [.55, .075, .55], [0, .68, 0], orange, .025);
    camera.lookAt(0, .2, 0);
    update = t => {
      const phase = t * tau;
      robot.root.position.set(1.85 * Math.cos(phase), 0, 1.04 * Math.sin(phase));
      robot.root.rotation.y = Math.atan2(-1.85 * Math.sin(phase), 1.04 * Math.cos(phase));
      robot.wheels.forEach(wheel => {wheel.rotation.x = t * tau * 8;});
    };
  } else if (kind === 'arm') {
    box(scene, [.84, .1, .8], [-1.15, .07, 0], silver, .05);
    const robot = arm(scene);
    const start = point(.8, .25, .35);
    const cube = box(scene, [.42, .42, .42], start.toArray(), orange, .035);
    box(cube, [.25, .008, .25], [0, .214, 0], white, .018);
    ring(scene, .45, [.8, .03, .35], cyan);
    update = t => {
      const approach = ease(t, 0, .15), retreat = ease(t, .9, 1);
      const lift = ease(t, .25, .4) * (1 - ease(t, .7, .83));
      const transfer = ease(t, .4, .55) * (1 - ease(t, .55, .7));
      const grip = start.clone().add(point(0, .7 * (1 - approach + retreat + lift), -1.15 * transfer));
      const held = t >= .25 && t < .84;
      const close = ease(t, .15, .25) * (1 - ease(t, .83, .9));
      robot.update(grip, mix(.34, .266, close));
      cube.position.copy(held ? grip : start);
    };
  } else if (kind === 'mobile-manipulator') {
    const robot = wheeled(scene); robot.scan.visible = false;
    robot.root.rotation.y = Math.PI / 2;
    cylinder(robot.root, .19, .4, [0, .74, 0], silver);
    const manipulator = arm(robot.root, point(0, 1.16, 0), 1.05, 1.02);
    box(scene, [1.1, .13, 1.25], [1.52, .71, 0], white, .04);
    for (const x of [1.1, 1.94]) for (const z of [-.48, .48]) box(scene, [.08, .67, .08], [x, .36, z], silver, .012);
    const start = point(1.23, .98, 0);
    const cube = box(scene, [.36, .4, .36], start.toArray(), orange, .035);
    update = t => {
      const approach = ease(t, 0, .15) * (1 - ease(t, .9, 1));
      const baseX = mix(-1.65, -.53, approach);
      robot.root.position.x = baseX;
      const reach = ease(t, .15, .3) * (1 - ease(t, .8, .9));
      const take = ease(t, .4, .55) * (1 - ease(t, .62, .75));
      const lift = ease(t, .35, .4) * (1 - ease(t, .7, .75));
      const home = point(baseX + .43, 1.43, 0);
      const grip = home.clone().lerp(start, reach);
      grip.y += lift * .38;
      grip.x = mix(grip.x, baseX + .35, take);
      const close = ease(t, .3, .35) * (1 - ease(t, .75, .8));
      robot.root.updateWorldMatrix(true, false);
      manipulator.update(robot.root.worldToLocal(grip.clone()), mix(.32, .235, close));
      cube.position.copy(t >= .35 && t < .79 ? grip : start);
      robot.wheels.forEach(wheel => {wheel.rotation.x = approach * 5.6;});
    };
  } else {
    const robots = [];
    for (let i = 0; i < 4; i++) {
      const robot = quadruped(scene);
      robot.root.position.set(i % 2 ? 1.3 : -1.3, .025, i < 2 ? -.82 : .82);
      robot.root.scale.setScalar(.67);
      const material = surface(i % 2 ? '#61dcd5' : '#7eb3ff', .1, .5);
      material.transparent = true; material.opacity = .42;
      const meshes = [];
      robot.root.traverse(item => {if (item.isMesh) meshes.push(item);});
      meshes.forEach(item => {
        item.material = material;
        const geometry = new THREE.EdgesGeometry(item.geometry, 22);
        const edgesMaterial = new THREE.LineBasicMaterial({color: i % 2 ? '#acfff1' : '#b0d7ff', transparent: true, opacity: .7});
        resources.add(geometry); resources.add(edgesMaterial);
        item.add(new THREE.LineSegments(geometry, edgesMaterial));
      });
      const sensor = point(.82, 1.05, 0);
      const rays = [];
      for (const z of [-.55, .55]) for (const y of [.03, .85]) rays.push(sensor, point(1.8, y, z));
      lines(robot.root, rays, '#81efff', .65);
      robots.push(robot);
    }
    lines(scene, [point(0, .035, -1.68), point(0, .035, 1.68), point(-2.7, .035, 0), point(2.7, .035, 0)], '#8fe6ff', .75);
    camera.lookAt(0, .22, 0);
    update = t => robots.forEach((robot, i) => robot.update(t + i / 8));
  }
  update(0);
  return {scene, camera, update, dispose() {resources.forEach(resource => resource.dispose());}};
}
