# launch/demo.launch.py
# 运行方式：ros2 launch my_robot_basics demo.launch.py

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('my_robot_basics'),
        'config',
        'robot_params.yaml'
    )

    # 声明 Launch 参数：名称、默认值、描述
    # 用户可以在 ros2 launch 命令后用 key:=value 覆盖默认值
    rate_arg = DeclareLaunchArgument(
        'publish_rate',
        default_value='1.0',
        description='关节角发布频率（Hz）'
    )

    return LaunchDescription([
        rate_arg,   # 必须放在 LaunchDescription 里才会生效

        Node(
            package='my_robot_basics',
            executable='kinematics_publisher',
            name='kinematics_publisher',
            parameters=[
                config,
                # LaunchConfiguration('publish_rate') 在运行时读取命令行传入的值
                {'publish_rate': LaunchConfiguration('publish_rate')},
            ],
            output='screen',
        ),

        Node(
            package='my_robot_basics',
            executable='ik_service_node',
            name='ik_service_node',
            parameters=[config],
            output='screen',
        ),

        Node(
            package='my_robot_basics',
            executable='move_to_action_server',
            name='move_to_action_server',
            parameters=[config],
            output='screen',
        ),
    ])
