from glob import glob
from setuptools import find_packages, setup

package_name = 'my_robot_basics'

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
        'hello_node = my_robot_basics.hello_node:main',
        'ik_client_node = my_robot_basics.ik_client_node:main',
        'ik_service_node = my_robot_basics.ik_service_node:main',
        'kinematics_node = my_robot_basics.kinematics_node:main',
        'kinematics_publisher = my_robot_basics.kinematics_publisher:main',
        'kinematics_subscriber = my_robot_basics.kinematics_subscriber:main',
        'move_to_action_client = my_robot_basics.move_to_action_client:main',
        'move_to_action_server = my_robot_basics.move_to_action_server:main',
        'tf_broadcaster_node = my_robot_basics.tf_broadcaster_node:main',
        'tf_listener_node = my_robot_basics.tf_listener_node:main',
    ]},
)
