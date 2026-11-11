import rclpy
from rclpy.node import Node
from moveit.planning import MoveItPy, PlanRequestParameters
from moveit.core.robot_state import RobotState
import math


class PlannerNode(Node):
    def __init__(self):
        super().__init__('planner_node')
        self.get_logger().info('规划节点启动，初始化 MoveIt...')

        self.moveit = MoveItPy(node_name='planner_node')
        self.arm = self.moveit.get_planning_component('arm')

        self.get_logger().info('MoveIt 初始化完成，2 秒后开始规划...')
        self.timer = self.create_timer(2.0, self.plan_and_execute)

    def plan_and_execute(self):
        self.timer.cancel()

        self.arm.set_start_state_to_current_state()

        robot_state = RobotState(self.moveit.get_robot_model())
        robot_state.set_joint_group_positions(
            'arm',
            [math.radians(30), math.radians(45)]
        )
        self.arm.set_goal_state(robot_state=robot_state)

        self.get_logger().info('开始规划...')
        # 必须显式指定规划 pipeline，否则报 "No planning pipeline available"
        plan_params = PlanRequestParameters(self.moveit, 'ompl')
        plan_params.planning_pipeline = 'ompl'
        plan_params.planner_id = 'RRTConnect'
        plan_result = self.arm.plan(single_plan_parameters=plan_params)

        if plan_result:
            self.get_logger().info('规划成功，执行中...')
            # 必须指定控制器名，否则报 "No active controllers configured"
            self.moveit.execute(plan_result.trajectory, controllers=["arm_controller"])
            self.get_logger().info('执行完成')
        else:
            self.get_logger().warn('规划失败')


def main(args=None):
    rclpy.init(args=args)
    node = PlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
