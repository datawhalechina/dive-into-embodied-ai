from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_move_group_launch, generate_moveit_rviz_launch
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import TimerAction


def generate_launch_description():
    moveit_config = (
        MoveItConfigsBuilder("two_link_arm", package_name="my_robot_moveit_config")
        .to_moveit_configs()
    )

    # robot_state_publisher
    rsp_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        parameters=[moveit_config.robot_description],
    )

    # ros2_control_node：加载 mock hardware
    ros2_control_node = Node(
        package="controller_manager",
        executable="ros2_control_node",
        parameters=[
            moveit_config.robot_description,
            str(moveit_config.package_path / "config/ros2_controllers.yaml"),
        ],
    )

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

    # move_group + RViz2
    move_group = generate_move_group_launch(moveit_config)
    rviz = generate_moveit_rviz_launch(moveit_config)

    # 构造 pipeline 参数（MoveItPy 底层需要 planning_pipelines.pipeline_names）
    pp = moveit_config.planning_pipelines
    pipeline_params = {
        "planning_pipelines.pipeline_names": ["ompl"],
        "default_planning_pipeline": "ompl",
    }
    for k, v in pp.get("ompl", {}).items():
        if k != "planner_configs":
            pipeline_params[f"ompl.{k}"] = v

    # topic_planner_node 延迟 5 秒，等待 move_group 和控制器完全就绪
    topic_planner = TimerAction(
        period=5.0,
        actions=[
            Node(
                package='my_robot_planner',
                executable='topic_planner_node',
                name='topic_planner_node',
                output='screen',
                parameters=[
                    moveit_config.robot_description,
                    moveit_config.robot_description_semantic,
                    moveit_config.robot_description_kinematics,
                    moveit_config.joint_limits,
                    moveit_config.trajectory_execution,
                    pipeline_params,
                    {'l1': 1.0, 'l2': 0.8},
                ],
            )
        ]
    )

    # scene_manager 直接发 /collision_object topic，不需要 MoveItPy
    scene_manager = TimerAction(
        period=6.0,
        actions=[
            Node(
                package='my_robot_planner',
                executable='scene_manager_node',
                name='scene_manager_node',
                output='screen',
            )
        ]
    )

    return LaunchDescription(
        [rsp_node, ros2_control_node, joint_state_broadcaster_spawner, arm_controller_spawner]
        + move_group.entities
        + rviz.entities
        + [topic_planner, scene_manager]
    )
