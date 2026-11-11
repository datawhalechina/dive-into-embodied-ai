# my_robot_basics/tf_broadcaster_node.py
# 运行方式：ros2 run my_robot_basics tf_broadcaster_node
#
# ============================================================
# 节点目标：订阅 /joint_states，把关节角转换为坐标系变换并广播
#           到 tf2，让整个系统共享机械臂各连杆的坐标系关系
# 执行逻辑：
#   1. __init__ 创建动态/静态广播器，广播固定变换（world→base_link）
#   2. 订阅 /joint_states，收到关节角后触发 joint_state_callback
#   3. callback 里根据 θ1、θ2 构造 TransformStamped 消息
#   4. sendTransform() 把变换发布到 /tf Topic
#   5. RViz2、tf_listener 等节点从 /tf 读取并使用这些变换
# 注意：本节点是第 9 讲 robot_state_publisher 的手动实现版本
#       理解这里的逻辑，才能明白 RSP 背后在做什么
# ============================================================

import math
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TransformStamped
# TransformBroadcaster：发布动态变换到 /tf Topic（随时间变化的坐标系关系）
# StaticTransformBroadcaster：发布静态变换到 /tf_static Topic（固定不变的关系）
# 两者的区别：静态变换只需广播一次，会被持久缓存；动态变换需要持续广播
from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster
from sensor_msgs.msg import JointState


class TFBroadcasterNode(Node):
    def __init__(self):
        super().__init__('tf_broadcaster_node')

        self.declare_parameter('l1', 1.0)
        self.declare_parameter('l2', 0.8)
        self.L1 = self.get_parameter('l1').value
        self.L2 = self.get_parameter('l2').value

        # 动态变换广播器：用于随关节角变化的坐标系（link1、link2）
        self.tf_broadcaster = TransformBroadcaster(self)

        # 静态变换广播器：用于固定不变的坐标系（world→base_link）
        # 静态变换只需广播一次，tf2 会永久缓存它
        self.static_broadcaster = StaticTransformBroadcaster(self)

        # 立即广播静态变换（节点启动时执行一次即可）
        self._broadcast_static_transforms()

        # 订阅关节角，每次收到新关节角就更新动态变换
        # [注意] 订阅 '/kinematics/joint_states' 而不是 '/joint_states'，与 kinematics_publisher 保持一致
        self.subscription = self.create_subscription(
            JointState,
            '/kinematics/joint_states',
            self.joint_state_callback,
            10,
        )

        self.get_logger().info('TF Broadcaster 启动，等待关节角...')

    def _broadcast_static_transforms(self):
        """广播固定不变的坐标系关系：world → base_link"""
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = 'world'       # 父坐标系
        t.child_frame_id = 'base_link'    # 子坐标系
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0  # 单位四元数（x=0,y=0,z=0,w=1）表示无旋转
        self.static_broadcaster.sendTransform(t)

    def joint_state_callback(self, msg):
        if len(msg.position) < 2:
            return

        theta1 = msg.position[0]  # 弧度，joint1 的角度
        theta2 = msg.position[1]  # 弧度，joint2 的角度
        now = self.get_clock().now().to_msg()

        # 广播 base_link → link1（由 θ1 决定旋转）
        # 平面绕 z 轴旋转 θ 的四元数：(x=0, y=0, z=sin(θ/2), w=cos(θ/2))
        # 这是从欧拉角到四元数的标准转换公式（绕单轴旋转的简化形式）
        t1 = TransformStamped()
        t1.header.stamp = now
        t1.header.frame_id = 'base_link'
        t1.child_frame_id = 'link1'
        t1.transform.translation.x = 0.0  # link1 原点与 base_link 原点重合
        t1.transform.translation.y = 0.0
        t1.transform.translation.z = 0.0
        t1.transform.rotation.x = 0.0
        t1.transform.rotation.y = 0.0
        t1.transform.rotation.z = math.sin(theta1 / 2)
        t1.transform.rotation.w = math.cos(theta1 / 2)

        # 广播 link1 → link2（由 θ2 决定旋转，原点在 link1 末端）
        t2 = TransformStamped()
        t2.header.stamp = now
        t2.header.frame_id = 'link1'
        t2.child_frame_id = 'link2'
        # link2 的原点在 link1 坐标系下的 x=L1 处（即 link1 的末端）
        t2.transform.translation.x = self.L1
        t2.transform.translation.y = 0.0
        t2.transform.translation.z = 0.0
        t2.transform.rotation.x = 0.0
        t2.transform.rotation.y = 0.0
        t2.transform.rotation.z = math.sin(theta2 / 2)
        t2.transform.rotation.w = math.cos(theta2 / 2)

        # 广播 link2 → end_effector（固定偏移，link2 末端，无旋转）
        t3 = TransformStamped()
        t3.header.stamp = now
        t3.header.frame_id = 'link2'
        t3.child_frame_id = 'end_effector'
        t3.transform.translation.x = self.L2  # end_effector 在 link2 末端
        t3.transform.translation.y = 0.0
        t3.transform.translation.z = 0.0
        t3.transform.rotation.x = 0.0
        t3.transform.rotation.y = 0.0
        t3.transform.rotation.z = 0.0
        t3.transform.rotation.w = 1.0  # 无旋转

        # 一次调用发送多个变换，比分三次调用更高效，时间戳也保持一致
        self.tf_broadcaster.sendTransform([t1, t2, t3])


def main(args=None):
    rclpy.init(args=args)
    node = TFBroadcasterNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
