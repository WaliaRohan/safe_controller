#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
import numpy as np

"""
This ROS Node takes position data from pure python sim (cbf_lite) and
 publishes it to "lane_position" to run with "control_lane" node.
"""

class Spoofer(Node):

    def __init__(self):
        super().__init__("spoofer")  # node name

        # -----------------------
        # Parameters
        # -----------------------

        file_path = "/home/ubuntu/ros_ws/src/safe_controller/resource/sim_sinusoidal_EKF.npz"
        topic     = "/lane_position"
        hz        = 5000
        self.key  = "x_meas"
        self.loop = False

        # -----------------------
        # Load NPZ file
        # -----------------------
        self.get_logger().info(f"[spoofer] Loading .npz file: {file_path}")
        try:
            self.data = np.load(file_path, allow_pickle=True)
        except Exception as e:
            raise RuntimeError(f"[spoofer] Failed to load {file_path}: {e}")

        if self.key not in self.data:
            raise ValueError(
                f"[spoofer] Key '{self.key}' not found. Available: {list(self.data.keys())}"
            )

        # Extract array
        y_obs = (self.data[self.key])[:, 1]
        self.array = y_obs/40 + 0.5
        self.idx = 0
        self.length = len(self.array)

        # Publisher
        self.pub = self.create_publisher(Float32, topic, 10)

        # Timer to publish at rate Hz
        self.timer = self.create_timer(1.0 / hz, self.publish_step)
        self.get_logger().info(
            f"[spoofer] Publishing key '{self.key}' to '{topic}' @ {hz} Hz"
        )

    def publish_step(self):
        """Publish next vector/row from the NPZ array."""
        if self.idx >= self.length:
            if self.loop:
                self.idx = 0
            else:
                self.get_logger().info("[spoofer] End of data reached. Stopping.")
                self.timer.cancel()
                return

        msg = Float32()
        msg.data = (self.array[self.idx]).astype(Float32)
        self.pub.publish(msg)

        self.idx += 1


def main(args=None):
    rclpy.init(args=args)
    node = Spoofer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
