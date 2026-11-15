# my_robot_basics/tf_listener_node.py
# 运行方式：ros2 run my_robot_basics tf_listener_node
#
# ============================================================
# 节点目标：每秒查询一次末端执行器在 world 坐标系下的位置，
#           验证 tf2 广播的坐标系变换与 FK 计算结果一致
# 执行逻辑：
#   1. __init__ 创建 Buffer 和 TransformListener（开始缓存 tf2 数据）
#   2. 创建 1 秒定时器，每秒触发一次查询
#   3. timer_callback 调用 lookup_transform() 查询变换
#   4. 从变换结果中提取末端位置并打印
# ============================================================

import rclpy
from rclpy.node import Node
# Buffer：在内存中缓存一段时间内的所有 tf2 变换数据
# TransformListener：订阅 /tf 和 /tf_static，把收到的变换存入 Buffer
# 两者必须配合使用：Listener 负责收数据，Buffer 负责存和查
from tf2_ros import TransformListener, Buffer
from rclpy.duration import Duration


class TFListenerNode(Node):
    def __init__(self):
        super().__init__('tf_listener_node')

        # Buffer 默认缓存最近 10 秒的变换数据
        # 查询历史时刻的变换时，数据必须在缓存窗口内
        self.tf_buffer = Buffer()

        # TransformListener(buffer, node)：
        # 内部会创建对 /tf 和 /tf_static 的订阅
        # 收到变换消息后自动存入 tf_buffer，不需要手动处理
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # 每秒查询一次末端位置
        self.timer = self.create_timer(1.0, self.timer_callback)
        self.get_logger().info('TF Listener 启动，每秒查询末端位置...')

    def timer_callback(self):
        try:
            # lookup_transform(目标坐标系, 源坐标系, 时间点)
            # 含义：返回"把源坐标系的点变换到目标坐标系"所需的变换矩阵
            # 即：end_effector 原点在 world 坐标系下的位置和姿态
            # rclpy.time.Time()：查询最新时刻的变换（时间戳为 0 表示"最新"）
            # 如果 tf2 还没收到数据，会抛出异常（见错误处理章节）
            t = self.tf_buffer.lookup_transform(
                'world',           # 目标坐标系（想要结果表示在哪个坐标系下）
                'end_effector',    # 源坐标系（想要查询哪个坐标系的位置）
                rclpy.time.Time(), # 查询最新时刻
            )
            # t.transform.translation 就是 end_effector 原点在 world 坐标系下的坐标
            x = t.transform.translation.x
            y = t.transform.translation.y
            self.get_logger().info(f'末端位置 (world 坐标系): x={x:.4f}, y={y:.4f}')

        except Exception as e:
            # 常见异常：
            # - LookupException：坐标系不存在（广播节点未启动）
            # - ExtrapolationException：查询时间超出缓存范围
            self.get_logger().warn(f'查询失败: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = TFListenerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
