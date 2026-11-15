# my_robot_basics/hello_node.py
# 运行方式：ros2 run my_robot_basics hello_node
#
# ============================================================
# 节点目标：最小 ROS2 节点示例，验证工程环境是否搭建正确
# 执行逻辑：
#   1. rclpy.init() 初始化 ROS2 运行时
#   2. 创建节点实例，节点名注册到 ROS2 系统
#   3. __init__ 里打印一条日志
#   4. rclpy.spin() 进入事件循环，节点持续存活
#   5. Ctrl+C 后清理退出
# ============================================================

import rclpy                    # ROS2 Python 客户端库，提供 init/spin/shutdown 等全局函数
from rclpy.node import Node     # 所有节点的基类，提供日志、定时器、发布/订阅等能力


class HelloNode(Node):
    def __init__(self):
        # 向 ROS2 系统注册节点，传入的字符串就是节点名
        # 节点名决定了 ros2 node list 里显示的名字，以及日志前缀
        # 同一系统里节点名必须唯一，否则会有警告
        super().__init__('hello_node')

        # get_logger() 返回该节点专属的日志器
        # 日志会带上时间戳和节点名，比 print() 更适合 ROS2 调试
        # 级别从低到高：debug < info < warn < error < fatal
        self.get_logger().info('Hello from ROS2!')


def main(args=None):
    # 初始化 rclpy 运行时，必须在创建任何节点之前调用
    # args=args 允许从命令行传入 ROS2 参数（如 --ros-args -p key:=value）
    rclpy.init(args=args)

    # 创建节点实例，__init__ 在这里执行
    node = HelloNode()

    # 进入事件循环：rclpy.spin() 会阻塞在这里，持续监听并分发回调
    # 包括：订阅回调、定时器回调、服务请求回调等
    # 没有 spin()，节点虽然创建了，但所有回调都不会被执行
    # 按 Ctrl+C 会触发 KeyboardInterrupt，spin() 随之返回
    rclpy.spin(node)

    # spin() 返回后执行清理：注销节点、释放资源
    node.destroy_node()
    # 关闭 rclpy 运行时，释放底层 DDS 通信资源
    rclpy.shutdown()
