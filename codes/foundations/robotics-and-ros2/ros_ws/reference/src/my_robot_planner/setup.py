from glob import glob
from setuptools import find_packages, setup

package_name = 'my_robot_planner'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Dive into Embodied AI contributors',
    maintainer_email='maintainers@example.com',
    description='Reference examples for the robotics and ROS 2 lessons',
    license='CC-BY-NC-SA-4.0',
    entry_points={'console_scripts': [
        'planner_node = my_robot_planner.planner_node:main',
        'scene_manager_node = my_robot_planner.scene_manager_node:main',
        'topic_planner_node = my_robot_planner.topic_planner_node:main',
    ]},
)
