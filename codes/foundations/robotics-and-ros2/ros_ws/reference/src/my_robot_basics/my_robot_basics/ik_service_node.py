# my_robot_basics/ik_service_node.py
# 运行方式：ros2 run my_robot_basics ik_service_node
#
# ============================================================
# 节点目标：提供 /solve_ik 服务，客户端发来目标点 (x, y) 后
#           立即返回 IK 求解结果（关节角），类似一次函数调用
# 执行逻辑：
#   1. __init__ 注册服务，绑定 handle_request 回调
#   2. spin() 等待客户端请求
#   3. 收到请求 → handle_request 被调用
#   4. 调用 inverse_kinematics → 填写 response → 返回
#   5. rclpy 自动把 response 发回给客户端
# 与 Topic 的区别：Service 是一问一答，客户端会阻塞等待响应
# ============================================================

import rclpy
from rclpy.node import Node

# SolveIK 是我们在 my_robot_interfaces/srv/SolveIK.srv 里定义的接口
# 编译后自动生成 Python 类，包含 SolveIK.Request 和 SolveIK.Response
from my_robot_interfaces.srv import SolveIK
from my_robot_basics.two_link_kinematics import inverse_kinematics


class IKServiceNode(Node):
    def __init__(self):
        super().__init__('ik_service_node')

        # create_service(接口类型, 服务名称, 回调函数)
        # 服务名称 '/solve_ik' 是绝对路径，客户端用同样的名称连接
        # 回调函数签名必须是 def handle_request(self, request, response)
        #   request：客户端发来的请求对象（对应 .srv 文件 --- 上方的字段）
        #   response：要填写并返回的响应对象（对应 .srv 文件 --- 下方的字段）
        self.srv = self.create_service(SolveIK, '/solve_ik', self.handle_request)
        self.get_logger().info('IK Service 已启动，等待请求...')

    def handle_request(self, request, response):
        # request 对象的字段对应 SolveIK.srv 里 --- 上方的定义：x, y, l1, l2
        self.get_logger().info(
            f'收到请求: x={request.x:.3f}, y={request.y:.3f}, '
            f'L1={request.l1}, L2={request.l2}'
        )

        try:
            solutions = inverse_kinematics(request.x, request.y, request.l1, request.l2)
            # 默认返回肘下解（theta2 > 0 的那组）
            t1, t2 = solutions['elbow_down']
            # 填写 response 对象的字段（对应 .srv 文件 --- 下方的定义）
            response.success = True
            response.theta1_deg = t1
            response.theta2_deg = t2
            response.message = f'elbow_down 解: θ1={t1:.2f}°, θ2={t2:.2f}°'
        except ValueError as e:
            response.success = False
            response.theta1_deg = 0.0
            response.theta2_deg = 0.0
            response.message = str(e)

        self.get_logger().info(f'响应: {response.message}')
        # 返回 response，rclpy 会自动把它序列化并发回给客户端
        return response


def main(args=None):
    rclpy.init(args=args)
    node = IKServiceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
