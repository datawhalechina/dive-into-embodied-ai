# my_robot_planner/topic_planner_node.py
# 运行方式：ros2 run my_robot_planner topic_planner_node
#
# ============================================================
# 节点目标：订阅 /target_pose（目标末端位置），收到后自动完成
#           IK 求解 → MoveIt 规划 → 执行的完整流程
# 执行逻辑：
#   1. __init__ 初始化 MoveItPy，订阅 /target_pose
#   2. 收到目标点 → target_callback
#   3. 调用 IK 求解关节角（不可达则警告并忽略）
#   4. 调用 _plan_to_joint_angles 规划并执行
#   5. _planning 标志位防止并发规划
# ============================================================

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from moveit.planning import MoveItPy, PlanRequestParameters
from moveit.core.robot_state import RobotState

from my_robot_basics.two_link_kinematics import inverse_kinematics


class TopicPlannerNode(Node):
    def __init__(self):
        super().__init__('topic_planner_node')

        self.declare_parameter('l1', 1.0)
        self.declare_parameter('l2', 0.8)
        self.L1 = self.get_parameter('l1').value
        self.L2 = self.get_parameter('l2').value

        self.get_logger().info('初始化 MoveIt...')
        self.moveit = MoveItPy(node_name='topic_planner_node')
        self.arm = self.moveit.get_planning_component('arm')
        self.get_logger().info('MoveIt 就绪，等待目标点...')

        self.subscription = self.create_subscription(
            Point,
            '/target_pose',
            self.target_callback,
            10,
        )

        # 规划重入保护标志：规划和执行是阻塞操作，期间可能收到新目标
        # _planning=True 时忽略新目标，避免并发规划导致状态混乱
        self._planning = False

    def target_callback(self, msg):
        if self._planning:
            self.get_logger().warn('上一次规划还未完成，忽略本次目标')
            return

        x, y = msg.x, msg.y
        self.get_logger().info(f'收到目标点: ({x:.3f}, {y:.3f})')

        # 先用 IK 检查目标是否可达，不可达直接返回，不进入规划流程
        try:
            solutions = inverse_kinematics(x, y, self.L1, self.L2)
        except ValueError as e:
            self.get_logger().warn(f'目标不可达: {e}')
            return

        t1_deg, t2_deg = solutions['elbow_down']
        self.get_logger().info(f'IK 解: θ1={t1_deg:.2f}°, θ2={t2_deg:.2f}°')

        # 设置标志位，开始规划
        self._planning = True
        try:
            self._plan_to_joint_angles(
                math.radians(t1_deg),
                math.radians(t2_deg),
            )
        finally:
            # finally 确保无论规划成功还是抛出异常，标志位都会被清除
            # 如果不用 finally，规划失败时 _planning 永远为 True，节点就"卡死"了
            self._planning = False

    def _plan_to_joint_angles(self, theta1_rad, theta2_rad):
        # 每次规划前都要重新设置起始状态，确保从当前实际位置出发
        self.arm.set_start_state_to_current_state()

        robot_state = RobotState(self.moveit.get_robot_model())
        robot_state.set_joint_group_positions('arm', [theta1_rad, theta2_rad])
        self.arm.set_goal_state(robot_state=robot_state)

        plan_params = PlanRequestParameters(self.moveit, 'ompl')
        plan_params.planning_pipeline = 'ompl'
        plan_params.planner_id = 'RRTConnect'
        plan_result = self.arm.plan(single_plan_parameters=plan_params)
        if not plan_result:
            self.get_logger().warn('规划失败，目标可能被障碍物阻挡')
            return

        self.get_logger().info('规划成功，执行中...')
        self.moveit.execute(plan_result.trajectory, controllers=['arm_controller'])
        self.get_logger().info('执行完成')


def main(args=None):
    rclpy.init(args=args)
    node = TopicPlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
