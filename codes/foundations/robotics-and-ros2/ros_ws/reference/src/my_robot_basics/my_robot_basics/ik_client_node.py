# my_robot_basics/ik_client_node.py
# 运行方式：ros2 run my_robot_basics ik_client_node
#
# ============================================================
# 节点目标：向 /solve_ik 服务发送一次 IK 请求，打印结果后退出
# 执行逻辑：
#   1. 创建 Service 客户端，连接 /solve_ik
#   2. 等待服务端上线（最多 5 秒）
#   3. 构造请求 → call_async() 异步发送
#   4. spin_until_future_complete() 阻塞等待响应
#   5. 打印结果，节点退出
# ============================================================

import sys
import rclpy
from rclpy.node import Node

from my_robot_interfaces.srv import SolveIK


class IKClientNode(Node):
    def __init__(self):
        super().__init__('ik_client_node')
        # create_client(接口类型, 服务名称)
        # 客户端和服务端的服务名称必须完全一致（包括前缀 '/'）
        self.client = self.create_client(SolveIK, '/solve_ik')

    def send_request(self, x, y, l1=1.0, l2=0.8):
        # wait_for_service()：阻塞等待服务端上线，超时返回 False
        # 如果服务端还没启动就直接 call，会立即失败
        if not self.client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('服务不可用，请先启动 ik_service_node')
            return

        # 构造请求对象，填写 .srv 文件 --- 上方定义的字段
        request = SolveIK.Request()
        request.x = x
        request.y = y
        request.l1 = l1
        request.l2 = l2

        # call_async()：异步发送请求，立即返回一个 Future 对象
        # Future 是"未来某时刻会有结果"的占位符，现在还没有值
        future = self.client.call_async(request)

        # spin_until_future_complete()：让事件循环运转，直到 Future 有结果
        # 这是在单线程节点里"同步等待异步结果"的标准写法
        # 不能用 future.result() 直接取值，因为结果还没到
        rclpy.spin_until_future_complete(self, future)

        # Future 完成后，.result() 才有值，即服务端返回的 response 对象
        response = future.result()
        if response.success:
            self.get_logger().info(
                f'IK 成功: θ1={response.theta1_deg:.2f}°, θ2={response.theta2_deg:.2f}°'
            )
        else:
            self.get_logger().warn(f'IK 失败: {response.message}')


def main(args=None):
    rclpy.init(args=args)
    node = IKClientNode()

    # 从命令行读取目标点，默认 (1.2, 0.8)
    x = float(sys.argv[1]) if len(sys.argv) > 1 else 1.2
    y = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8

    node.send_request(x, y)
    node.destroy_node()
    rclpy.shutdown()
