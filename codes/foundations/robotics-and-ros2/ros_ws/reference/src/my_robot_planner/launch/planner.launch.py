from moveit_configs_utils import MoveItConfigsBuilder
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    moveit_config = (
        MoveItConfigsBuilder("two_link_arm", package_name="my_robot_moveit_config")
        .to_moveit_configs()
    )

    # MoveItPy 底层（MoveItCpp）读取的是 planning_pipelines.pipeline_names，
    # 而 moveit_config.planning_pipelines 传的 key 是 planning_pipelines（列表），
    # 两者不匹配，需要手动构造正确格式的参数字典。
    pp = moveit_config.planning_pipelines
    pipeline_params = {
        "planning_pipelines.pipeline_names": ["ompl"],
        "default_planning_pipeline": "ompl",
    }
    for k, v in pp.get("ompl", {}).items():
        if k != "planner_configs":
            pipeline_params[f"ompl.{k}"] = v

    planner_node = Node(
        package="my_robot_planner",
        executable="planner_node",
        output="screen",
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.joint_limits,
            moveit_config.trajectory_execution,
            pipeline_params,
        ],
    )

    return LaunchDescription([planner_node])
