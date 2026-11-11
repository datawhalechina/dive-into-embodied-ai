# my_robot_basics/move_to_action_client.py
# 运行方式：ros2 run my_robot_basics move_to_action_client
#
# ============================================================
# 节点目标：向 /move_to Action 服务发送一个运动目标，实时打印
#           进度反馈，等待执行完成后打印最终结果
# 执行逻辑（全异步，靠回调链驱动）：
#   1. send_goal_async() → 服务端响应"接不接受" → goal_response_callback
#   2. Goal 被接受 → get_result_async() → 等待执行完成 → result_callback
#   3. 执行期间服务端每次 publish_feedback → feedback_callback 被调用
#   4. result_callback 里打印结果，调用 rclpy.shutdown() 退出
# 注意：所有回调都由 spin() 调度，没有 spin() 则回调永远不会执行
# ============================================================

import sys
import rclpy
from rclpy.node import Node
# ActionClient：发 Goal、收 Feedback/Result；所有 API 均为异步（Future + 回调）
from rclpy.action import ActionClient

from my_robot_interfaces.action import MoveTo


class MoveToActionClient(Node):
    def __init__(self):
        super().__init__('move_to_action_client')
        # ActionClient(节点, 动作类型, 动作名)
        # 动作名必须与 ActionServer 里的名称完全一致
        self._client = ActionClient(self, MoveTo, '/move_to')

    def send_goal(self, x, y):
        # 等待 Action Server 上线，最多 5 秒
        if not self._client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error('Action Server 不可用')
            return

        # 构造 Goal 对象，填写 .action 文件第一段（--- 之前）的字段
        goal = MoveTo.Goal()
        goal.x = x
        goal.y = y

        self.get_logger().info(f'发送目标: ({x}, {y})')

        # send_goal_async() 立即返回 Future，不阻塞主线程
        # feedback_callback：服务端每次 publish_feedback 时由 rclpy 自动调用
        # Future 完成的含义：服务端已响应"接不接受这个 Goal"（不是执行完成）
        send_goal_future = self._client.send_goal_async(
            goal,
            feedback_callback=self.feedback_callback,
        )
        # add_done_callback：Future 完成时自动调用 goal_response_callback
        # 这是 Action 客户端的标准异步链式写法
        send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        # future.result() 返回 ClientGoalHandle，包含 accepted 字段
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().warn('Goal 被拒绝')
            return
        self.get_logger().info('Goal 已接受，等待结果...')

        # get_result_async() 返回另一个 Future，在任务执行完成时触发
        # 任务执行期间，feedback_callback 仍会持续被调用
        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self.result_callback)

    def feedback_callback(self, feedback_msg):
        # feedback_msg 是包装类型，.feedback 才是 MoveTo.Feedback 对象
        # 包含 .progress（0.0~1.0）和 .status（字符串）字段
        fb = feedback_msg.feedback
        self.get_logger().info(f'[反馈] {fb.status} ({fb.progress:.0%})')

    def result_callback(self, future):
        # future.result() 是 GetResult 响应对象
        # .result 才是 MoveTo.Result（success、final_theta1_deg 等字段）
        result = future.result().result
        if result.success:
            self.get_logger().info(
                f'执行完成: θ1={result.final_theta1_deg:.2f}°, '
                f'θ2={result.final_theta2_deg:.2f}°'
            )
        else:
            self.get_logger().warn(f'执行失败: {result.message}')
        # 本示例发完一个 Goal 就退出；常驻节点通常不在这里 shutdown
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = MoveToActionClient()

    x = float(sys.argv[1]) if len(sys.argv) > 1 else 1.2
    y = float(sys.argv[2]) if len(sys.argv) > 2 else 0.8

    node.send_goal(x, y)
    # spin() 是必须的：Future 完成事件、Feedback 消息、Result 回调
    # 都需要事件循环来调度，没有 spin() 所有回调都不会执行
    rclpy.spin(node)
