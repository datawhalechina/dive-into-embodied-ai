from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_move_group_launch, generate_moveit_rviz_launch
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    moveit_config = (
        MoveItConfigsBuilder("two_link_arm", package_name="my_robot_moveit_config")
        .to_moveit_configs()
    )

    # robot_state_publisher：发布 URDF 和 TF 树
    rsp_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[moveit_config.robot_description],
    )

    # ros2_control_node：加载 mock hardware，接管关节状态
    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[
            moveit_config.robot_description,
            str(moveit_config.package_path / "config/ros2_controllers.yaml"),
        ],
    )

    # spawner：等 controller_manager 就绪后自动激活控制器
    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster"],
    )

    arm_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["arm_controller"],
    )

    # move_group：MoveIt 2 核心节点
    move_group = generate_move_group_launch(moveit_config)

    # RViz2 + MotionPlanning 插件
    rviz = generate_moveit_rviz_launch(moveit_config)

    return LaunchDescription(
        [rsp_node, ros2_control_node, joint_state_broadcaster_spawner, arm_controller_spawner]
        + move_group.entities
        + rviz.entities
    )
