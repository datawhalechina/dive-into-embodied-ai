# my_robot_basics/kinematics_publisher.py
# 运行方式：ros2 run my_robot_basics kinematics_publisher
#
# ============================================================
# 节点目标：每秒随机生成一组关节角，计算 FK，把关节角发布到
#           /kinematics/joint_states Topic，供其他节点（订阅者、RViz2、
#           robot_state_publisher 等）使用
# 执行逻辑：
#   1. __init__ 创建发布者和 1 秒定时器
#   2. 每次定时器触发 → timer_callback 执行
#   3. 随机生成 θ1、θ2 → 调用 FK 算末端位置（仅用于日志）
#   4. 构造 JointState 消息 → publish() 发出去
# ============================================================

import math
import random

import rclpy
from rclpy.node import Node
# JointState 是 ROS2 标准消息，描述关节状态
# 包含 header（时间戳）、name（关节名列表）、position（角度，弧度）等字段
from sensor_msgs.msg import JointState
from rcl_interfaces.msg import SetParametersResult

from my_robot_basics.two_link_kinematics import forward_kinematics


class KinematicsPublisher(Node):
    def __init__(self):
        super().__init__('kinematics_publisher')

        # create_publisher(消息类型, Topic名称, 队列深度)
        # 队列深度 10：如果消费者处理慢，最多缓存 10 条消息，超出则丢弃最旧的
        # Topic 名称以 '/' 开头表示绝对路径，不受命名空间影响
        self.publisher_ = self.create_publisher(JointState, '/kinematics/joint_states', 10)

        # 第 7 讲：参数可由 YAML 或 launch 覆盖。
        self.declare_parameter('l1', 1.0)
        self.declare_parameter('l2', 0.8)
        self.declare_parameter('publish_rate', 1.0)
        self.L1 = self.get_parameter('l1').value
        self.L2 = self.get_parameter('l2').value
        rate = self.get_parameter('publish_rate').value
        if not all(math.isfinite(v) and v > 0 for v in (self.L1, self.L2, rate)):
            raise ValueError('l1, l2 and publish_rate must be finite and positive')
        self.timer = self.create_timer(1.0 / rate, self.timer_callback)
        self.add_on_set_parameters_callback(self.validate_parameters)
        self.add_post_set_parameters_callback(self.apply_parameters)
        self.get_logger().info(f'参数: L1={self.L1}, L2={self.L2}, 频率={rate}Hz')

    def validate_parameters(self, params):
        for param in params:
            if param.name in ('l1', 'l2', 'publish_rate'):
                if not isinstance(param.value, (int, float)) or not math.isfinite(param.value) or param.value <= 0:
                    return SetParametersResult(successful=False, reason='Parameters must be finite and positive')
        return SetParametersResult(successful=True)

    def apply_parameters(self, params):
        # 仅在整批参数通过校验后更新状态，拒绝的修改不会影响定时器。
        for param in params:
            if param.name == 'l1':
                self.L1 = param.value
            elif param.name == 'l2':
                self.L2 = param.value
            elif param.name == 'publish_rate':
                self.destroy_timer(self.timer)
                self.timer = self.create_timer(1.0 / param.value, self.timer_callback)

    def timer_callback(self):
        # 随机生成关节角（度）
        theta1_deg = random.uniform(-90.0, 90.0)
        theta2_deg = random.uniform(-120.0, 120.0)

        # 计算末端位置（仅用于日志，不发布）
        x, y = forward_kinematics(theta1_deg, theta2_deg, self.L1, self.L2)

        # 构造 JointState 消息
        msg = JointState()

        # header.stamp：消息的时间戳，tf2 和 RViz2 依赖它做时间同步
        # get_clock().now() 获取节点当前时间，to_msg() 转为消息格式
        # 不填时间戳会导致后续 tf2 坐标系同步出问题
        msg.header.stamp = self.get_clock().now().to_msg()

        # header.frame_id：消息所在的参考坐标系，JointState 通常填 'base_link'
        msg.header.frame_id = 'base_link'

        # name：关节名称列表，必须和 URDF 里的 <joint name="..."> 完全一致
        # robot_state_publisher 靠这个名字匹配 URDF 里的关节
        msg.name = ['joint1', 'joint2']

        # position：关节角度列表，单位是弧度（ROS2 约定）
        # 我们的 FK/IK 函数用度，发布前必须转换
        msg.position = [math.radians(theta1_deg), math.radians(theta2_deg)]

        # publish() 把消息发送到 /kinematics/joint_states Topic
        # 所有订阅了这个 Topic 的节点都会收到这条消息
        self.publisher_.publish(msg)

        self.get_logger().info(
            f'发布: θ1={theta1_deg:6.1f}°, θ2={theta2_deg:6.1f}° → 末端=({x:.3f}, {y:.3f})'
        )


def main(args=None):
    rclpy.init(args=args)
    node = KinematicsPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
