"""Offline checks; these do not replace a ROS 2 Jazzy build or runtime test."""

import ast
import importlib.util
import math
import os
from pathlib import Path
import random
import re
import runpy
import types
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np


WORKSPACE = Path(__file__).resolve().parents[1] / 'ros_ws'
PACKAGES = WORKSPACE / 'reference/src'


def load_module(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


transforms = load_module(WORKSPACE / 'scripts/lesson_02_transforms.py')
implementations = [
    load_module(WORKSPACE / 'scripts/lesson_03_two_link_kinematics.py'),
    load_module(PACKAGES / 'my_robot_basics/my_robot_basics/two_link_kinematics.py'),
]


class KinematicsTests(unittest.TestCase):
    def test_rotation_and_translation(self):
        np.testing.assert_allclose(transforms.rotation_matrix_2d(90) @ [1, 0], [0, 1], atol=1e-12)
        np.testing.assert_allclose(
            transforms.homogeneous_transform_2d(90, 1, 0.5) @ [2, 0, 1],
            [1, 2.5, 1], atol=1e-12,
        )

    def test_known_forward_positions(self):
        for module in implementations:
            for angles, expected in [((0, 0), (1.8, 0)), ((90, -90), (0.8, 1)), ((0, 90), (1, 0.8))]:
                with self.subTest(module=module.__name__, angles=angles):
                    np.testing.assert_allclose(module.forward_kinematics(*angles, 1.0, 0.8), expected, atol=1e-12)

    def test_matrix_and_analytic_fk_agree_and_both_ik_branches_return_target(self):
        rng = random.Random(18)
        for _ in range(100):
            angles = (rng.uniform(-180, 180), rng.uniform(-170, 170))
            lengths = (rng.uniform(0.2, 2), rng.uniform(0.2, 2))
            target = transforms.fk_via_matrix(*lengths, *angles)
            for module in implementations:
                np.testing.assert_allclose(module.forward_kinematics(*angles, *lengths), target, atol=1e-10)
                solutions = module.inverse_kinematics(*target, *lengths)
                self.assertEqual(set(solutions), {'elbow_down', 'elbow_up'})
                for solution in solutions.values():
                    np.testing.assert_allclose(module.forward_kinematics(*solution, *lengths), target, atol=1e-10)

    def test_workspace_boundaries_and_invalid_targets(self):
        for module in implementations:
            for target in [(1.8, 0), (0.2, 0), (0, 1.8), (-1.8, 0)]:
                for angles in module.inverse_kinematics(*target, 1.0, 0.8).values():
                    np.testing.assert_allclose(module.forward_kinematics(*angles, 1.0, 0.8), target, atol=1e-10)
            for args in [(2, 0, 1, 0.8), (0.1, 0, 1, 0.8), (1, 0, 0, 1), (math.nan, 0, 1, 1), (1, 0, -1, 1)]:
                with self.assertRaises(ValueError):
                    module.inverse_kinematics(*args)


class WorkspaceTests(unittest.TestCase):
    def test_documented_reference_paths_exist(self):
        docs = Path(__file__).resolve().parents[4] / 'docs/foundations/robotics-and-ros2'
        for document in docs.glob('*.md'):
            paths = re.findall(r'\$ROS_WS/((?:reference|scripts)/[A-Za-z0-9_./-]*)', document.read_text())
            for path in paths:
                with self.subTest(document=document.name, path=path):
                    self.assertTrue((WORKSPACE / path).exists(), path)

    def test_python_sources_parse(self):
        for file in WORKSPACE.rglob('*.py'):
            with self.subTest(file=file):
                ast.parse(file.read_text(), filename=str(file))

    def test_manifests_and_reference_isolation(self):
        self.assertTrue((WORKSPACE / 'reference/COLCON_IGNORE').is_file())
        manifests = list(PACKAGES.glob('*/package.xml'))
        self.assertEqual(len(manifests), 5)
        for manifest in manifests:
            root = ET.parse(manifest).getroot()
            self.assertEqual(root.findtext('name'), manifest.parent.name)
            build_type = root.findtext('export/build_type')
            self.assertIn(build_type, ('ament_python', 'ament_cmake'))
            if build_type == 'ament_cmake':
                self.assertTrue((manifest.parent / 'CMakeLists.txt').is_file())

    def test_installed_python_entry_points_and_assets_exist(self):
        for package in ('my_robot_basics', 'my_robot_planner'):
            directory = PACKAGES / package
            captured = {}
            setuptools = types.ModuleType('setuptools')
            setuptools.setup = lambda **kwargs: captured.update(kwargs)
            setuptools.find_packages = lambda **kwargs: [package]
            old_cwd = Path.cwd()
            try:
                os.chdir(directory)
                with patch.dict('sys.modules', {'setuptools': setuptools}):
                    runpy.run_path(str(directory / 'setup.py'))
            finally:
                os.chdir(old_cwd)
            for entry in captured['entry_points']['console_scripts']:
                _, target = entry.split('=')
                module, function = target.strip().split(':')
                source = directory / (module.replace('.', '/') + '.py')
                tree = ast.parse(source.read_text())
                self.assertIn(function, [node.name for node in tree.body if isinstance(node, ast.FunctionDef)])
            installed = {file for _, files in captured['data_files'] for file in files}
            for file in installed:
                self.assertTrue((directory / file).is_file(), file)
            for folder in ('launch', 'config'):
                for file in (directory / folder).glob('*'):
                    self.assertIn(str(file.relative_to(directory)), installed)
            self.assertIn('script_dir=$base/lib/' + package, (directory / 'setup.cfg').read_text())

    def test_urdf_tree_and_control_joint_names(self):
        robot = ET.parse(PACKAGES / 'my_robot_description/urdf/two_link_arm.urdf').getroot()
        links = {link.attrib['name'] for link in robot.findall('link')}
        children = set()
        for joint in robot.findall('joint'):
            parent = joint.find('parent').attrib['link']
            child = joint.find('child').attrib['link']
            self.assertIn(parent, links)
            self.assertIn(child, links)
            self.assertNotIn(child, children)
            children.add(child)
        self.assertEqual(links - children, {'base_link'})
        self.assertEqual(len(robot.findall('ros2_control')), 1)
        self.assertEqual({joint.attrib['name'] for joint in robot.findall('ros2_control/joint')}, {'joint1', 'joint2'})


if __name__ == '__main__':
    unittest.main()
