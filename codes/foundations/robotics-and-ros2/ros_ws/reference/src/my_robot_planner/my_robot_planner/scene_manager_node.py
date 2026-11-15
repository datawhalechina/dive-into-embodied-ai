import rclpy
from rclpy.node import Node
from moveit_msgs.msg import CollisionObject
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose


class SceneManagerNode(Node):
    def __init__(self):
        super().__init__('scene_manager_node')

        # 直接向 move_group 的 /collision_object topic 发布障碍物
        # 不需要 MoveItPy，move_group 会自动更新 Planning Scene
        self.publisher = self.create_publisher(CollisionObject, '/collision_object', 10)

        # 延迟 2 秒，等待 move_group 完全启动并订阅 /collision_object
        self.timer = self.create_timer(2.0, self.setup_scene)

    def setup_scene(self):
        self.timer.cancel()

        obj = CollisionObject()
        obj.id = 'wall'
        obj.header.frame_id = 'base_link'

        box = SolidPrimitive()
        box.type = SolidPrimitive.BOX
        box.dimensions = [0.02, 0.3, 0.5]  # 2cm 厚，30cm 宽，50cm 高

        pose = Pose()
        pose.position.x = 0.7
        pose.position.y = 0.3
        pose.position.z = 0.25
        pose.orientation.w = 1.0

        obj.primitives = [box]
        obj.primitive_poses = [pose]
        obj.operation = CollisionObject.ADD

        self.publisher.publish(obj)
        self.get_logger().info('障碍物 "wall" 已发布到规划场景')


def main(args=None):
    rclpy.init(args=args)
    node = SceneManagerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
