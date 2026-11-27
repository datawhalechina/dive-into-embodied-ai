"""Adapt Petoi's pinned URDF to an explicitly approximate MuJoCo model.

The source files remain untouched. This is a local adapter, not a Petoi simulator.
"""
from pathlib import Path
import json
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent
SIDES = ("left_front", "right_front", "right_back", "left_back")


def fmt(value):
    return " ".join(f"{float(x):.10g}" for x in np.asarray(value).ravel())


def origin(element):
    result = np.eye(4)
    if element is not None:
        result[:3, 3] = np.fromstring(element.get("xyz", "0 0 0"), sep=" ")
        result[:3, :3] = Rotation.from_euler(
            "xyz", np.fromstring(element.get("rpy", "0 0 0"), sep=" ")
        ).as_matrix()
    return result


def pose(transform):
    q = Rotation.from_matrix(transform[:3, :3]).as_quat()
    return dict(pos=fmt(transform[:3, 3]), quat=fmt(q[[3, 0, 1, 2]]))


def prepare_meshes():
    out = ROOT / "models" / "assets"
    out.mkdir(parents=True, exist_ok=True)
    report = []
    for source in sorted((ROOT / "original" / "assets").glob("*.stl")):
        dest = out / source.name
        if dest.exists():
            continue
        mesh = trimesh.load_mesh(source)
        n = len(mesh.faces)
        # The two spring meshes each contain > 1 million triangles.
        # Simplification changes only rendering; inertia comes from the URDF.
        target = 12000 if "spring" in source.stem else 16000
        if n > target:
            mesh = mesh.simplify_quadric_decimation(face_count=target)
        mesh.export(dest)
        report.append(dict(file=source.name, original_faces=n, rendered_faces=len(mesh.faces)))
        print(f"mesh {source.name}: {n} -> {len(mesh.faces)}", flush=True)
    if report:
        (ROOT / "models" / "mesh_report.json").write_text(json.dumps(report, indent=2) + "\n")


def build():
    prepare_meshes()
    urdf = ET.parse(ROOT / "original" / "robot.urdf").getroot()
    links = {link.get("name"): link for link in urdf.findall("link")}
    children = {name: [] for name in links}
    for joint in urdf.findall("joint"):
        children[joint.find("parent").get("link")].append(joint)

    root = ET.Element("mujoco", model="Quaddle - local approximate adaptation")
    ET.SubElement(root, "compiler", angle="radian", meshdir="assets", autolimits="true")
    ET.SubElement(root, "option", timestep="0.001", integrator="implicitfast", iterations="80", tolerance="1e-10")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1280", offheight="960")
    ET.SubElement(visual, "headlight", ambient="0.35 0.35 0.35", diffuse="0.6 0.6 0.6")
    ET.SubElement(visual, "map", znear="0.005")
    defaults = ET.SubElement(root, "default")
    ET.SubElement(defaults, "geom", solref="0.008 1", solimp="0.95 0.99 0.001")
    assets = ET.SubElement(root, "asset")
    for file in sorted((ROOT / "models" / "assets").glob("*.stl")):
        ET.SubElement(assets, "mesh", name=file.stem, file=file.name)
    ET.SubElement(assets, "texture", name="grid", type="2d", builtin="checker", rgb1="0.12 0.16 0.22", rgb2="0.18 0.23 0.30", width="512", height="512")
    # World-space texture density also works when the plane has no visual edge.
    ET.SubElement(assets, "material", name="floor", texture="grid", texuniform="true", texrepeat="16 16", reflectance="0.12")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0 -0.4 0.8", dir="0 0.4 -0.8", directional="true")
    # Zero XY sizes render an infinite plane; collision was already infinite.
    ET.SubElement(world, "geom", name="ground", type="plane", size="0 0 0.1", material="floor", contype="1", conaffinity="2", friction="0.8 0.002 0.0001")
    bodies, transforms = {}, {}

    def visit(name, parent, transform, joint=None):
        local = origin(joint.find("origin")) if joint is not None else np.eye(4)
        transforms[name] = transform @ local
        body = ET.SubElement(parent, "body", name=name, **pose(local))
        bodies[name] = body
        if joint is None:
            body.set("pos", "0 0 0.065")
            ET.SubElement(body, "freejoint", name="root")
        elif joint.get("type") != "fixed":
            kwargs = dict(name=joint.get("name"), type="hinge", axis=joint.find("axis").get("xyz"), damping="0.00002", armature="0.00000002")
            if name.endswith("_thigh"):
                kwargs.update(damping="0.0005", armature="0.000002", range="-1.4 1.4")
            elif name.endswith("_motor_arm"):
                # A tunable torsional regularizer for the passive branch.
                # It is NOT a measured model of the physical tension spring.
                kwargs.update(damping="0.0004", stiffness="0.012", springref="0")
            elif name.endswith("_leg"):
                # Knee support is also a proxy for the unmodeled spring compliance.
                kwargs.update(damping="0.001", stiffness="0.08", springref="0")
            elif name.endswith("_wheel"):
                # Self-collision is disabled (floor-only contact mask), so bound the
                # passive wheel rocker bracket against the rubber foot tip and lower leg.
                # Stiff limit parameters prevent light wheel inertia from overshooting.
                kwargs.update(range="-0.4 0.85", solreflimit="0.004 1", solimplimit="0.99 0.999 0.0001")
            ET.SubElement(body, "joint", **kwargs)
        inertia = links[name].find("inertial")
        mass = float(inertia.find("mass").get("value"))
        if mass > 1e-8:
            data = inertia.find("inertia")
            it = origin(inertia.find("origin"))
            # Official inertial frames all have zero rotation.
            assert np.allclose(it[:3, :3], np.eye(3))
            ET.SubElement(body, "inertial", mass=str(mass), pos=fmt(it[:3, 3]), fullinertia=" ".join(data.get(k) for k in ("ixx", "iyy", "izz", "ixy", "ixz", "iyz")))
        for i, v in enumerate(links[name].findall("visual")):
            mesh = Path(v.find("geometry/mesh").get("filename")).stem
            color = v.find("material/color")
            ET.SubElement(body, "geom", name=f"{name}_visual_{i}", type="mesh", mesh=mesh, rgba=color.get("rgba") if color is not None else "0.7 0.7 0.7 1", contype="0", conaffinity="0", group="2", mass="0", **pose(origin(v.find("origin"))))
        for j in children[name]:
            visit(j.find("child").get("link"), body, transforms[name], j)

    visit("body", world, np.eye(4))

    # Contacts are deliberately separate from detailed render meshes.
    # Floor-only contact masks avoid spurious collisions in overlapping CAD parts.
    ET.SubElement(bodies["body"], "geom", name="body_contact", type="box", pos="-0.00526647 -0.000744244 0.0021", size="0.025 0.0525 0.0121", contype="2", conaffinity="1", group="3", rgba="0.2 0.5 0.9 0.3", mass="0")
    equality = ET.SubElement(root, "equality")
    actuators = ET.SubElement(root, "actuator")
    closure_report = []
    for side in SIDES:
        a, b = f"closing_{side}_leg_1", f"closing_{side}_leg_2"
        p1, p2 = transforms[a][:3, 3], transforms[b][:3, 3]
        # CAD frames lie at different points on the same pin axis (~1.89 mm).
        # Use one common point, rather than forcing those different origins equal.
        anchor = (p1 + p2) / 2
        for name in (a, b):
            local_anchor = transforms[name][:3, :3].T @ (anchor - transforms[name][:3, 3])
            ET.SubElement(bodies[name], "site", name=name + "_anchor", pos=fmt(local_anchor), size="0.001", rgba="1 0.2 0.2 1", group="4")
        ET.SubElement(equality, "connect", name=side + "_closure", site1=a + "_anchor", site2=b + "_anchor", solref="0.004 1", solimp="0.99 0.999 0.0001")
        closure_report.append(dict(side=side, source_frame_offset_m=float(np.linalg.norm(p1 - p2))))
        ET.SubElement(actuators, "position", name=side + "_servo", joint=side + "_thigh", kp="0.15", kv="0.006", ctrlrange="-1.2 1.2", forcerange="-0.245 0.245")
        thigh_end = origin(next(j for j in children[side + "_thigh"] if j.find("child").get("link") == b).find("origin"))[:3, 3]
        ET.SubElement(bodies[side + "_thigh"], "geom", name=side + "_thigh_contact", type="capsule", fromto="0 0 0 " + fmt(thigh_end), size="0.003", contype="2", conaffinity="1", group="3", rgba="0.2 0.9 0.5 0.3", mass="0")
        # The former knee-to-foot capsule extended below the actual foot.
        # Use the official lower-leg mesh hull instead, avoiding false shin support.
        leg_visual = links[side + "_leg"].find("visual")
        ET.SubElement(bodies[side + "_leg"], "geom", name=side + "_leg_contact", type="mesh",
                      mesh=Path(leg_visual.find("geometry/mesh").get("filename")).stem,
                      contype="2", conaffinity="1", group="3", rgba="0.2 0.9 0.5 0.3", mass="0",
                      **pose(origin(leg_visual.find("origin"))))
        # Use the wheel CAD shapes as convex collision hulls; no visual spring hulls.
        for part in ("wheel", "wheel_tip"):
            name = side + "_" + part
            for i, v in enumerate(links[name].findall("visual")):
                mesh = Path(v.find("geometry/mesh").get("filename")).stem
                ET.SubElement(bodies[name], "geom", name=name + f"_contact_{i}", type="mesh", mesh=mesh, contype="2", conaffinity="1", group="3", rgba="0.2 0.9 0.5 0.3", mass="0", friction="0.8 0.001 0.0001", **pose(origin(v.find("origin"))))

    ET.indent(root)
    model_path = ROOT / "models" / "scene.xml"
    ET.ElementTree(root).write(model_path, encoding="utf-8", xml_declaration=True)
    m = mujoco.MjModel.from_xml_path(str(model_path))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    report = dict(mujoco_version=mujoco.__version__, source_links=len(links), source_joints=len(urdf.findall("joint")), nq=m.nq, nv=m.nv, actuators=m.nu, equality_constraints=m.neq, total_mass_kg=float(m.body_mass.sum()), closures=closure_report)
    (ROOT / "models" / "build_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    build()
