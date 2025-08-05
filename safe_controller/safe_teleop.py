import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import sys
import termios
import tty
import select
import signal
import atexit

MAX_LINEAR = 0.2
MAX_ANGULAR = 2.0
STEP_LINEAR = 0.05
STEP_ANGULAR = 0.1

KEY_BINDINGS = {
    'w': (STEP_LINEAR, 0.0),
    's': (-STEP_LINEAR, 0.0),
    'a': (0.0, STEP_ANGULAR),
    'd': (0.0, -STEP_ANGULAR),
    ' ': (0.0, 0.0),
    'q': 'quit'
}

settings = termios.tcgetattr(sys.stdin)

def restore_terminal():
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)

atexit.register(restore_terminal)

class SafeTeleop(Node):
    def __init__(self):
        super().__init__('safe_teleop')
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.linear = 0.0
        self.angular = 0.0
        self.timer = self.create_timer(0.1, self.publish_cmd)
        print("Use keys: W/S = forward/backward, A/D = left/right, SPACE = stop, q = Quit. Ctrl+C to exit.")

    def get_key(self, timeout=0.1):
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        key = sys.stdin.read(1) if rlist else None
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        return key

    def publish_cmd(self):
        try:
            key = self.get_key()
            if key:
                if key == 'q':
                    print("Quit key pressed. Shutting down...")
                    # self.destroy_node()
                    rclpy.shutdown()
                    return
                if key in KEY_BINDINGS:
                    lin_delta, ang_delta = KEY_BINDINGS[key]
                    self.linear += lin_delta
                    self.angular += ang_delta

                    clamped = False

                    if abs(self.linear) > MAX_LINEAR:
                        clamped = True
                        self.linear = max(min(self.linear, MAX_LINEAR), -MAX_LINEAR)
                        self.get_logger().warn(f"Linear velocity capped to ±{MAX_LINEAR}")

                    if abs(self.angular) > MAX_ANGULAR:
                        clamped = True
                        self.angular = max(min(self.angular, MAX_ANGULAR), -MAX_ANGULAR)
                        self.get_logger().warn(f"Angular velocity capped to ±{MAX_ANGULAR}")

                    if key == ' ':
                        self.linear = 0.0
                        self.angular = 0.0

                    twist = Twist()
                    twist.linear.x = self.linear
                    twist.angular.z = self.angular
                    self.publisher.publish(twist)

                    print(f"[vel] linear: {self.linear:.2f}  angular: {self.angular:.2f}" + (" (filtered)" if clamped else ""))

        except Exception as e:
            self.get_logger().error(f"Error: {e}")

def main():
    rclpy.init()
    node = SafeTeleop()

    # Ensure Ctrl+C works
    signal.signal(signal.SIGINT, lambda s, f: rclpy.shutdown())

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
