"""
kitchen_node.py — the ROS 2 face of the kitchen assistant.

    ros2 run within_reach kitchen_node
    ros2 topic pub --once /within_reach/command std_msgs/String "{data: 'medicine next to water'}"
    ros2 topic echo /within_reach/plan
    ros2 topic echo /joint_states

Topics
    sub  /within_reach/command   std_msgs/String        what to deliver
    pub  /within_reach/plan      std_msgs/String        one line per goal the planner issues
    pub  /within_reach/event     std_msgs/String        monitor verdicts and the final outcome
    pub  /joint_states           sensor_msgs/JointState streamed while the arm executes

/joint_states is the standard name on purpose: rviz, rqt_plot and `ros2 bag`
all expect it there, so the node is inspectable with stock tools instead of
needing its own.

No hardware is involved. The executor's backend here is FakeBackend, the same
one its self-test uses — this node is a fourth consumer of the executor's
interface alongside the Isaac backend and the taught-pose arm, not a rewrite of
any of them.
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from within_reach.bridge import KitchenBridge


class KitchenNode(Node):
    def __init__(self):
        super().__init__("within_reach_kitchen")
        self.plan_pub = self.create_publisher(String, "within_reach/plan", 10)
        self.event_pub = self.create_publisher(String, "within_reach/event", 10)
        # A deep queue: a single command emits several hundred joint states in
        # a burst, and a short one silently drops most of the trajectory.
        self.joint_pub = self.create_publisher(JointState, "joint_states", 1000)
        self.create_subscription(String, "within_reach/command", self.on_command, 10)
        self.bridge = KitchenBridge(
            publish_plan=self._say(self.plan_pub),
            publish_event=self._say(self.event_pub),
            publish_joints=self.publish_joints,
        )
        self.get_logger().info(
            "ready — publish a std_msgs/String on within_reach/command")

    def _say(self, publisher):
        def publish(text: str) -> None:
            publisher.publish(String(data=text))
            self.get_logger().info(text)
        return publish

    def publish_joints(self, names, positions) -> None:
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = list(names)
        msg.position = [float(v) for v in positions]
        self.joint_pub.publish(msg)

    def on_command(self, msg: String) -> None:
        self.get_logger().info(f"command: {msg.data!r}")
        try:
            result = self.bridge.handle_command(msg.data)
        except Exception as exc:      # a bad command must not kill the node
            self.get_logger().error(f"command failed: {exc}")
            self.event_pub.publish(String(data=f"ERROR {exc}"))
            return
        self.get_logger().info(f"done: delivered={result['ok']}")


def main(args=None):
    rclpy.init(args=args)
    node = KitchenNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
