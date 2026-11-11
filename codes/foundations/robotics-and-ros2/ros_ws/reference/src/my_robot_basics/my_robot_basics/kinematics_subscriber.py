# my_robot_basics/kinematics_subscriber.py
# 运行方式：ros2 run my_robot_basics kinematics_subscriber
#
# ============================================================
# 节点目标：订阅 /kinematics/joint_states Topic，收到关节角后做 FK + IK
#           round-trip 验证，确认发布者的数据是正确的
# 执行逻辑：
#   1. __init__ 注册订阅者，绑定回调函数
#   2. spin() 进入事件循环，等待消息到来
#   3. 每当 /kinematics/joint_states 有新消息 → callback 自动被调用
#   4. callback 里：弧度→度 → FK → IK → 打印对比结果
# ============================================================

import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

from my_robot_basics.two_link_kinematics import forward_kinematics, inverse_kinematics


class KinematicsSubscriber(Node):
    def __init__(self):
        super().__init__('kinematics_subscriber')

        # create_subscription(消息类型, Topic名称, 回调函数, 队列深度)
        # 队列深度 10：如果回调处理慢，最多缓存 10 条待处理消息
        # 回调函数 self.callback 会在每次收到消息时被 spin() 调度执行
        # 注意：回调函数签名必须是 def callback(self, msg)，msg 类型与订阅类型一致
        self.subscription = self.create_subscription(
            JointState,
            '/kinematics/joint_states',
            self.callback,
            10,
        )

        self.L1 = 1.0
        self.L2 = 0.8

        self.get_logger().info('kinematics_subscriber 启动，等待关节角消息...')

    def callback(self, msg):
        # msg 是收到的 JointState 消息对象
        # 防御性检查：确保消息里有至少 2 个关节角
        if len(msg.position) < 2:
            return

        # JointState.position 单位是弧度，转换为度供 FK/IK 函数使用
        theta1_deg = math.degrees(msg.position[0])
        theta2_deg = math.degrees(msg.position[1])

        # FK：用收到的关节角算出末端位置
        x, y = forward_kinematics(theta1_deg, theta2_deg, self.L1, self.L2)

        # IK：把末端位置反算回关节角，和原始值对比（round-trip 验证）
        try:
            solutions = inverse_kinematics(x, y, self.L1, self.L2)
            # 取肘下解做对比（发布者随机生成的角度不一定是肘下解，但末端位置应一致）
            ik_t1, ik_t2 = solutions['elbow_down']
            self.get_logger().info(
                f'收到: θ1={theta1_deg:6.1f}°, θ2={theta2_deg:6.1f}° | '
                f'FK→({x:.3f},{y:.3f}) | '
                f'IK验证: θ1={ik_t1:.1f}°, θ2={ik_t2:.1f}°'
            )
        except ValueError as e:
            self.get_logger().warn(f'IK 失败: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = KinematicsSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
