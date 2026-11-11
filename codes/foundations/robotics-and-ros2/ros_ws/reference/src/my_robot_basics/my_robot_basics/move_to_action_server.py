# my_robot_basics/move_to_action_server.py
# 运行方式：ros2 run my_robot_basics move_to_action_server
#
# ============================================================
# 节点目标：提供 /move_to Action 服务，接收目标末端位置后模拟
#           机械臂运动过程，期间持续发送进度反馈，完成后返回结果
# 执行逻辑：
#   1. __init__ 注册 ActionServer，绑定 execute_callback
#   2. 客户端发来 Goal → executor 调度 execute_callback
#   3. 先做 IK 可达性检查，不可达则 abort()
#   4. 可达则循环模拟运动，每步 publish_feedback() 发送进度
#   5. 完成后调用 succeed()，返回最终 Result
# 与 Service 的区别：Action 支持长时间执行 + 中途反馈 + 取消
# ============================================================

import math
import time
import rclpy
from rclpy.node import Node
# ActionServer：在服务端注册动作，收到 Goal 后执行 execute_callback
from rclpy.action import ActionServer

# MoveTo 由 .action 文件生成：内含 Goal / Result / Feedback 三种消息类型
from my_robot_interfaces.action import MoveTo
from my_robot_basics.two_link_kinematics import inverse_kinematics


class MoveToActionServer(Node):
    def __init__(self):
        super().__init__('move_to_action_server')

        self.declare_parameter('l1', 1.0)
        self.declare_parameter('l2', 0.8)
        self.declare_parameter('simulate_steps', 5)
        self.declare_parameter('step_duration', 0.5)
        self.L1 = self.get_parameter('l1').value
        self.L2 = self.get_parameter('l2').value
        self.steps = self.get_parameter('simulate_steps').value
        self.step_duration = self.get_parameter('step_duration').value
        if self.steps <= 0 or not math.isfinite(self.step_duration) or self.step_duration < 0:
            raise ValueError('simulate_steps must be positive and step_duration finite and nonnegative')

        # ActionServer(节点, 动作类型, 动作名, 执行回调)
        # 动作名 '/move_to' 需与客户端 ActionClient 里的名称一致
        # execute_callback 在 executor 中运行；此演示用阻塞等待模拟运动。
        # [注意] 此最小示例未实现取消回调；真实系统应异步执行并处理取消。
        self._action_server = ActionServer(
            self,
            MoveTo,
            '/move_to',
            self.execute_callback,
        )
        self.get_logger().info('MoveTo Action Server 已启动...')

    def execute_callback(self, goal_handle):
        # goal_handle：本次目标的句柄，提供三个关键能力：
        #   goal_handle.request   → 读取客户端发来的 Goal 字段
        #   goal_handle.publish_feedback(msg) → 向客户端发送进度
        #   goal_handle.succeed() / abort() / canceled() → 标记最终状态
        x = goal_handle.request.x
        y = goal_handle.request.y
        self.get_logger().info(f'收到目标: ({x:.3f}, {y:.3f})')

        # 先构造 Result 对象，成功和失败分支都要返回同一类型
        result = MoveTo.Result()

        # 可达性检查：不可达则 abort，客户端会收到"被中止"的状态
        try:
            solutions = inverse_kinematics(x, y, self.L1, self.L2)
        except ValueError as e:
            self.get_logger().warn(f'目标不可达: {e}')
            # abort() 表示任务无法完成（区别于 canceled()，后者是客户端主动取消）
            goal_handle.abort()
            result.success = False
            result.message = str(e)
            return result

        t1, t2 = solutions['elbow_down']

        # 模拟运动过程：循环发送进度反馈
        # 真实机器人里，这里会等待控制器到位信号，而不是 sleep
        feedback_msg = MoveTo.Feedback()
        steps = self.steps
        for i in range(1, steps + 1):
            # progress 是 0.0~1.0 的浮点数，客户端可用来显示进度条
            feedback_msg.progress = float(i) / steps
            feedback_msg.status = f'执行中 {i}/{steps}...'
            # publish_feedback() 把 Feedback 消息推送给所有订阅了反馈的客户端
            goal_handle.publish_feedback(feedback_msg)
            self.get_logger().info(f'进度: {feedback_msg.progress:.0%}')
            time.sleep(self.step_duration)

        # succeed() 标记任务成功完成，客户端的 result_callback 会被触发
        # 必须调用 succeed()/abort()/canceled() 之一，否则客户端会一直等待
        goal_handle.succeed()
        result.success = True
        result.final_theta1_deg = t1
        result.final_theta2_deg = t2
        result.message = f'到达目标: θ1={t1:.2f}°, θ2={t2:.2f}°'
        self.get_logger().info(result.message)
        return result


def main(args=None):
    rclpy.init(args=args)
    node = MoveToActionServer()
    rclpy.spin(node)  # 阻塞：处理 Action 请求、定时器、Topic 等回调
    node.destroy_node()
    rclpy.shutdown()
