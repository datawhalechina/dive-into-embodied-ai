# my_robot_basics/kinematics_node.py
# 运行方式：ros2 run my_robot_basics kinematics_node
#
# ============================================================
# 节点目标：把第 3 讲的 FK/IK 函数接入 ROS2，验证运动学计算
#           在 ROS2 节点环境下能正常运行
# 执行逻辑：
#   1. 节点启动时（__init__）用固定关节角计算一次 FK
#   2. 把 FK 结果传给 IK，做 round-trip 验证
#   3. 把结果打印到日志
#   4. 之后节点进入 spin() 等待状态（本讲暂无定时器/订阅）
# 注意：本节点只在启动时计算一次，不持续发布数据
#       第 5 讲会升级为持续发布关节角到 Topic
# ============================================================

import rclpy
from rclpy.node import Node

# 用完整包路径导入，不要用相对导入（.two_link_kinematics）
# 原因：ROS2 通过 entry_points 启动节点时，相对导入可能失效
from my_robot_basics.two_link_kinematics import forward_kinematics, inverse_kinematics


class KinematicsNode(Node):
    def __init__(self):
        # 'kinematics_node' 是节点名，会出现在 ros2 node list 的输出里
        super().__init__('kinematics_node')

        # 连杆长度暂时硬编码，第 7 讲会改为从参数系统读取
        # 这样改连杆长度时不需要重新编译，只需修改参数文件
        self.L1 = 1.0
        self.L2 = 0.8

        # 节点启动时计算一次 FK 并打印
        # __init__ 里的代码在节点创建时立即执行，不需要等 spin()
        theta1, theta2 = 30.0, 45.0
        x, y = forward_kinematics(theta1, theta2, self.L1, self.L2)
        self.get_logger().info(
            f'FK: θ1={theta1}°, θ2={theta2}° → 末端=({x:.4f}, {y:.4f})'
        )

        # 把 FK 结果传给 IK，验证 round-trip 是否正确
        # inverse_kinematics 返回字典：{'elbow_down': (t1, t2), 'elbow_up': (t1, t2)}
        solutions = inverse_kinematics(x, y, self.L1, self.L2)
        for name, (t1, t2) in solutions.items():
            self.get_logger().info(
                f'IK {name}: θ1={t1:.2f}°, θ2={t2:.2f}°'
            )


def main(args=None):
    rclpy.init(args=args)
    node = KinematicsNode()
    # spin() 让节点持续存活，即使本节点没有定时器/订阅
    # 没有 spin()，节点会在 __init__ 执行完后立即退出
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
