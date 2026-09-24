export type Point = [number, number, number];
export const clamp = (value: number, min = 0, max = 1) => Math.max(min, Math.min(max, value));
export const mix = (a: number, b: number, t: number) => a + (b - a) * t;
export const ease = (p: number, start: number, end: number) => {
  const t = clamp((p - start) / (end - start));
  return t * t * (3 - 2 * t);
};

export function graspPose(p: number) {
  const approach = ease(p, 0, .22);
  const descend = ease(p, .22, .48);
  const lift = ease(p, .64, .94);
  const object: Point = [.9, .24 + lift * 1.05, .25];
  const grip: Point = [mix(.05, .9, approach), mix(1.5, 1.1, approach) - descend * .86 + lift * 1.05, mix(-.35, .25, approach)];
  return {object, grip, opening: mix(.39, .294, ease(p, .48, .64))};
}

export function drawerPose(p: number) {
  const approach = ease(p, 0, .34);
  const distance = ease(p, .47, .92) * .95;
  const grip: Point = [mix(-.55, .17, approach) - distance, mix(1.55, .94, approach), mix(.4, 0, approach)];
  return {grip, distance, opening: mix(.28, .104, ease(p, .34, .47))};
}

export const stairHeight = (x: number) => x < -.2 - 1e-8 ? 0 : x < .75 - 1e-8 ? .2 : x < 1.7 - 1e-8 ? .4 : .6;
const supportHeight = (x: number) => .2 * (ease(x, -.6, .2) + ease(x, .35, 1.15) + ease(x, 1.3, 2.1));
const landingX = (x: number) => {
  const edge = [-.2, .75, 1.7].find(value => x < value - 1e-8 && x > value - .13);
  return edge === undefined ? x : edge - .13;
};

export function walkPose(p: number) {
  const walking = clamp((p - .12) / .78);
  const x = mix(-2.1, 2.4, walking);
  const settle = ease(p, .9, 1);
  const cycles = walking * 6;
  const stride = .75;
  const feet: Point[] = [];
  for (let i = 0; i < 4; i++) {
    const front = i < 2;
    const side = i % 2 ? 1 : -1;
    const phase = i === 0 || i === 3 ? 0 : .5;
    const cycle = cycles + phase;
    const step = Math.floor(cycle);
    const fraction = cycle - step;
    const nominalFrom = -2.1 + (front ? .55 : -.55) + stride * (step - phase + .3);
    const from = landingX(nominalFrom), to = landingX(nominalFrom + stride);
    const swingPhase = clamp((fraction - .6) / .4);
    // Lift before crossing the riser, then lower only after reaching the landing.
    const swing = ease(swingPhase, .25, .75);
    const fromHeight = stairHeight(from), toHeight = stairHeight(to);
    const clearanceHeight = Math.max(fromHeight, toHeight) + .2;
    const swingHeight = mix(fromHeight, clearanceHeight, ease(swingPhase, 0, .25))
      - (clearanceHeight - toHeight) * ease(swingPhase, .75, 1);
    const footX = mix(mix(from, to, swing), x + (front ? .55 : -.55), settle);
    const footY = mix(swingHeight, stairHeight(x + (front ? .55 : -.55)), settle);
    feet.push([footX, footY + .06, side * .4]);
  }
  return {
    body: [x, 1.02 + supportHeight(x), 0] as Point,
    pitch: (supportHeight(x + .55) - supportHeight(x - .55)) * .6 * (1 - settle),
    feet,
  };
}

export const navObstacles = [
  {x: -1.15, z: .15, width: 1.15, depth: 1.45},
  {x: 1.05, z: -.45, width: 1.2, depth: 1.3},
];
const waypoints = [[-3.1, 1.55], [-2.1, 1.55], [-.1, 1.55], [1.85, 1.2], [2.65, .25], [2.7, -1.5], [3.15, -1.5]];

// A deterministic smooth route lets animation, path rendering and clearance checks agree.
export function routePoint(t: number): Point {
  const scaled = clamp(t) * (waypoints.length - 1);
  const i = Math.min(Math.floor(scaled), waypoints.length - 2);
  const u = scaled - i;
  const points = [waypoints[Math.max(0, i - 1)], waypoints[i], waypoints[i + 1], waypoints[Math.min(waypoints.length - 1, i + 2)]];
  const coordinate = (axis: number) => {
    const [a, b, c, d] = points.map(point => point[axis]);
    return .5 * (2 * b + (-a + c) * u + (2 * a - 5 * b + 4 * c - d) * u * u + (-a + 3 * b - 3 * c + d) * u * u * u);
  };
  return [coordinate(0), .06, coordinate(1)];
}

export function navigationPose(p: number) {
  const travel = ease(p, .38, .94);
  const point = routePoint(travel);
  const before = routePoint(Math.max(0, travel - .001));
  const after = routePoint(Math.min(1, travel + .001));
  return {point, travel, heading: Math.atan2(after[0] - before[0], after[2] - before[2])};
}
