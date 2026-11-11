# launch/display.launch.py
# 运行方式：ros2 launch my_robot_description display.launch.py

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration


def generate_launch_description():
    pkg_dir = get_package_share_directory('my_robot_description')
    urdf_file = os.path.join(pkg_dir, 'urdf', 'two_link_arm.urdf')

    # 读取 URDF 文件内容为字符串
    # robot_state_publisher 需要的是 robot_description 参数的字符串值，
    # 而不是文件路径——它不会自己去读文件，必须把内容传给它
    # 注意：第 10 讲 MoveIt Setup Assistant 生成的 launch 文件会用
    # Command(['xacro ', urdf_xacro_file]) 的方式展开 xacro 模板，
    # 两种方式等价，只是写法不同。纯 URDF 用 open() 读取即可。
    with open(urdf_file, 'r') as f:
        robot_description = f.read()

    return LaunchDescription([
        # robot_state_publisher：ROS2 官方节点
        # 职责：读取 robot_description 参数（URDF 字符串）+ 订阅 /joint_states
        #       根据 URDF 里的关节结构和收到的关节角，自动计算并发布整棵 TF 树
        #       替代了第 8 讲里手写的 tf_broadcaster_node
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            # parameters 传字典：key 是参数名，value 是参数值
            # robot_description 必须是 URDF 的完整 XML 字符串
            parameters=[{'robot_description': robot_description}],
            output='screen',
        ),

        # joint_state_publisher_gui：ROS2 官方节点
        # 职责：读取 robot_description，为每个可动关节生成一个滑块
        #       拖动滑块时发布 /joint_states，驱动 robot_state_publisher 更新 TF
        #       仅用于调试，真实系统里由控制器或算法节点发布 /joint_states
        Node(
            package='joint_state_publisher_gui',
            executable='joint_state_publisher_gui',
            name='joint_state_publisher_gui',
            output='screen',
        ),

        # RViz2：可视化工具
        # 职责：订阅 /robot_description、/tf、/joint_states 等 Topic
        #       把机器人模型和坐标系树渲染出来
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
        ),
    ])
