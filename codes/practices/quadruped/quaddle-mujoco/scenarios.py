"""Small, explicit terrain tasks. All dimensions are metres; no hardware claims."""
from dataclasses import asdict, dataclass, field
import heapq
import math

import mujoco
import numpy as np

from run_sim import ROOT, SIDES
from terrain import configure_ground


@dataclass
class Scene:
    name: str
    label: str
    slope_deg: float = 0.
    goal_m: float = .72
    plane_z: float = 0.
    initial_height_m: float = 0.
    boxes: list = field(default_factory=list)
    water_depth_m: float = 0.
    seed: int = 7

    def box(self, name, xmin, xmax, s0, s1, top, bottom=-.01, obstacle=False):
        self.boxes.append(dict(name=name, xmin=xmin, xmax=xmax, s0=s0, s1=s1,
                              top=top, bottom=bottom, obstacle=obstacle))

    def surface_height(self, x, y):
        """Top of the actual support geometry below a vertical probe."""
        s = -y
        height = self.plane_z + s * math.tan(math.radians(self.slope_deg))
        for box in self.boxes:
            if not box['obstacle'] and box['xmin'] <= x <= box['xmax'] and box['s0'] <= s <= box['s1']:
                height = max(height, box['top'])
        return height

    def metadata(self):
        result = asdict(self)
        if self.water_depth_m:
            result['water_model'] = dict(type='uncalibrated submerged-sphere buoyancy and quadratic drag',
                density_kg_m3=1000., drag_coefficient=1., foot_radius_m=.0045,
                lower_leg_radius_m=.005, region_s_m=[.15, 1.05], region_x_m=[-.22, .22],
                floor_sliding_friction=.55, floor_contact_priority=1, waves=False, fluid_solver=False,
                notes='Equivalent sealed volumes are assumptions; no waterproofing or leakage model.')
        return result


def make_scene(name):
    labels = dict(flat='Flat reference', uphill3='Uphill +3 deg', uphill5='Uphill +5 deg',
                  downhill3='Downhill -3 deg', stairs_up='Up stairs / 4 mm x 3',
                  stairs_down='Down stairs / 4 mm x 3', rough='Uneven / +/-3 mm',
                  avoid='Known-map A* avoidance', water='Shallow water / 8 mm / APPROXIMATION')
    if name not in labels:
        raise ValueError(f'Unknown scene: {name}')
    scene = Scene(name, labels[name])
    if name in ('uphill3', 'uphill5', 'downhill3'):
        scene.slope_deg = dict(uphill3=3., uphill5=5., downhill3=-3.)[name]
    elif name == 'stairs_up':
        for j in range(3):
            scene.box(f'step_{j}', -.3, .3, .18 + j*.12, .30+j*.12 if j<2 else 1.8, .004*(j+1))
    elif name == 'stairs_down':
        scene.initial_height_m = .012
        for j in range(3):
            scene.box(f'step_{j}', -.3, .3, -.4 if j==0 else .06+j*.12, .18+j*.12, .004*(3-j))
    elif name == 'rough':
        scene.plane_z = -.006
        scene.box('start', -.3, .3, -.4, .18, 0.)
        rng = np.random.default_rng(scene.seed)
        for j in range(12):
            for i in range(8):
                scene.box(f'tile_{j}_{i}', -.24+i*.06, -.18+i*.06,
                          .18+j*.06, .24+j*.06, float(rng.uniform(-.003, .003)))
        scene.box('finish', -.3, .3, .90, 1.8, 0.)
        scene.goal_m = 1.04
    elif name == 'avoid':
        scene.box('obstacle', -.04, .04, .43, .53, .06, obstacle=True)
        scene.goal_m = 1.
    elif name == 'water':
        scene.water_depth_m = .008
        scene.goal_m = 1.15  # Exit the water region, not just enter it.
    return scene


SCENES = ('flat', 'uphill3', 'uphill5', 'downhill3', 'stairs_up', 'stairs_down', 'rough', 'avoid', 'water')


def load_scene(scene):
    spec = mujoco.MjSpec.from_file(str(ROOT/'models/scene.xml'))
    spec.compiler.meshdir = str(ROOT/'models/assets')
    for box in scene.boxes:
        x = (box['xmin']+box['xmax'])/2
        s = (box['s0']+box['s1'])/2
        z = (box['top']+box['bottom'])/2
        spec.worldbody.add_geom(name='scene_'+box['name'], type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[x, -s, z], size=[(box['xmax']-box['xmin'])/2, (box['s1']-box['s0'])/2, (box['top']-box['bottom'])/2],
            contype=1, conaffinity=2, group=4 if box['obstacle'] else 0,
            friction=[.8, .002, .0001], rgba=[.9, .3, .18, 1] if box['obstacle'] else [.42, .48, .56, 1])
    if scene.water_depth_m:
        spec.worldbody.add_geom(name='water_visual_ONLY', type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[0, -.6, scene.water_depth_m], size=[.22, .45, .0002],
            rgba=[.10, .60, .95, .4], contype=0, conaffinity=0, group=5)
    model = spec.compile()
    data = mujoco.MjData(model)
    model.geom_pos[model.geom('ground').id, 2] = scene.plane_z
    if scene.water_depth_m:
        model.geom_friction[model.geom('ground').id, 0] = .55
        # Higher priority makes this the actual contact friction, rather than
        # max(.55, the robot geom's .8/1.0) under the default mixing rule.
        model.geom_priority[model.geom('ground').id] = 1
    rotation = configure_ground(model, data, scene.slope_deg)
    data.qpos[2] += scene.initial_height_m
    mujoco.mj_forward(model, data)
    return model, data, rotation


def plan_path(scene, spacing=.025, clearance=.09):
    """A* in a known static map, with inflated obstacles; not visual perception."""
    obstacles = [box for box in scene.boxes if box['obstacle']]
    start, goal = (0, 0), (0, round(scene.goal_m/spacing))
    def free(node):
        x, s = node[0]*spacing, node[1]*spacing
        return -.3 <= x <= .3 and -.05 <= s <= scene.goal_m+.1 and not any(
            b['xmin']-clearance <= x <= b['xmax']+clearance and b['s0']-clearance <= s <= b['s1']+clearance for b in obstacles)
    costs, previous, queue = {start: 0.}, {}, [(0., start)]
    while queue:
        _, node = heapq.heappop(queue)
        if node == goal:
            break
        for dx in (-1, 0, 1):
            for ds in (-1, 0, 1):
                nxt = (node[0]+dx, node[1]+ds)
                if (dx==ds==0) or not free(nxt) or not free((node[0]+dx, node[1])) or not free((node[0], node[1]+ds)):
                    continue
                cost = costs[node] + math.hypot(dx, ds)
                if cost < costs.get(nxt, float('inf')):
                    costs[nxt] = cost; previous[nxt] = node
                    heapq.heappush(queue, (cost+math.dist(nxt, goal), nxt))
    if goal not in costs:
        raise ValueError('No collision-free route in the known map')
    route = [goal]
    while route[-1] != start:
        route.append(previous[route[-1]])
    route.reverse()
    # Keep a corner only when a straight segment is not free in the inflated map.
    simplified, i = [route[0]], 0
    while i < len(route)-1:
        j = len(route)-1
        while j>i+1:
            delta = np.array(route[j])-route[i]
            count = max(2, int(np.linalg.norm(delta)*4))
            if all(free(tuple(np.rint(np.array(route[i])+t*delta).astype(int))) for t in np.linspace(0, 1, count)):
                break
            j -= 1
        simplified.append(route[j]); i = j
    return [(node[0]*spacing, -node[1]*spacing) for node in simplified]


def submerged_volume(radius, centre_z, surface_z):
    h = float(np.clip(surface_z-centre_z+radius, 0., 2*radius))
    return math.pi*h*h*(radius-h/3.)


class ShallowWater:
    """Prescribed still-water forces on assumed equivalent spheres, not CFD."""
    def __init__(self, model, scene):
        self.model, self.scene = model, scene
        self.samples = [(model.body(s+'_wheel_tip').id, s, .0045, 0.) for s in SIDES]
        self.samples += [(model.body(s+'_leg').id, s, .005, .01) for s in SIDES]
        self.jacp = np.zeros((3, model.nv)); self.jacr = np.zeros_like(self.jacp)
        self.max_force = 0.; self.max_drag_power = 0.; self.wet_steps = 0
        self.drag_energy = 0.

    def apply(self, data):
        data.qfrc_applied.fill(0.)
        wet = False
        for body, side, radius, dz in self.samples:
            point = data.body(side+'_wheel_tip').xpos.copy()
            point[2] += dz
            if not (-.22 <= point[0] <= .22 and .15 <= -point[1] <= 1.05):
                continue
            volume = submerged_volume(radius, point[2], self.scene.water_depth_m)
            if volume <= 0:
                continue
            wet = True
            mujoco.mj_jac(self.model, data, self.jacp, self.jacr, point, body)
            velocity = self.jacp @ data.qvel
            fraction = volume / (4*math.pi*radius**3/3)
            drag = -.5*1000.*math.pi*radius**2*fraction*np.linalg.norm(velocity)*velocity
            force = drag - 1000.*volume*self.model.opt.gravity
            mujoco.mj_applyFT(self.model, data, force, np.zeros(3), point, body, data.qfrc_applied)
            self.max_force = max(self.max_force, float(np.linalg.norm(force)))
            self.max_drag_power = max(self.max_drag_power, float(drag@velocity))
            self.drag_energy -= float(drag@velocity)*self.model.opt.timestep
        self.wet_steps += int(wet)
