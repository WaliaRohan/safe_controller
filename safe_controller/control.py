import safe_controller.safety as safety
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation as R

TOPIC_NOM_CTRL = "/nominal_control"
TOPIC_SAFE_CTRL = "/cmd_vel"
TOPIC_ODOM = "/odom"

MAX_LINEAR = 0.2
MAX_ANGULAR = 2.0

class Control(Node):
    def __init__(self):
        super().__init__('control')

        qos_profile_depth = 10 # This is the message queue size
        
        self.nominal_vel_subscriber_ = self.create_subscription(Twist,
                                                    TOPIC_NOM_CTRL,
                                                    self.nominal_vel_subscriber_callback,
                                                    qos_profile_depth)
        
        self.odom_subscriber_ = self.create_subscription(Odometry,
                                                         TOPIC_ODOM,
                                                         self.odom_subscriber_callback,
                                                         qos_profile_depth)
        
        self.publisher_ = self.create_publisher(Twist,
                                                TOPIC_SAFE_CTRL,
                                                qos_profile_depth)

        self.lin_vel = 0.0
        self.ang_vel = 0.0

        self.state = np.array([0.0, 0.0, 0.0, 0.0])
        
    def odom_subscriber_callback(self, msg):

        # "msg" contains covariance - figure out how to extract that!

        pose = msg.pose.pose # you can extract covariance from this as well
        twist = msg.twist.twist# you can extract covariance from this as well

        x = pose.position.x
        y = pose.position.y
        v = twist.linear.x

        q = msg.pose.pose.orientation  # This is a geometry_msgs.msg.Quaternion
        quat = [q.x, q.y, q.z, q.w]    # Extract to list of floats
        r = R.from_quat(quat)
        roll, pitch, yaw = r.as_euler('xyz') 
        theta = yaw

        # print(f"{type(x)}, {type(y)}, {type(v)}, {type(theta)}")

        self.state = np.array([x, y, v, theta])
    
    def nominal_vel_subscriber_callback(self, msg):
        self.lin_vel = msg.linear.x
        self.ang_vel = msg.angular.z

        self.safety_filter()

    def publisher_callback(self):
        twist = Twist()
        twist.linear.x = self.lin_vel
        twist.angular.z = self.ang_vel
        self.publisher_.publish(twist)

    def safety_filter(self):

        clamped = False

        if abs(self.lin_vel) > MAX_LINEAR:
            clamped = True
            self.lin_vel = max(min(self.lin_vel, MAX_LINEAR), -MAX_LINEAR)
            self.get_logger().warn(f"Linear velocity capped to ±{MAX_LINEAR}")

        if abs(self.ang_vel) > MAX_ANGULAR:
            clamped = True
            self.ang_vel = max(min(self.ang_vel, MAX_ANGULAR), -MAX_ANGULAR)
            self.get_logger().warn(f"Angular velocity capped to ±{MAX_ANGULAR}")

        self.publisher_callback()

        sol, H = safety.solve_qp(self.state, 0.01*np.ones((4, 4))) # Replace zeros with proper covariance (Look at odom callback)
        u_sol = sol.primal[0][:2]

        print("[state]:", np.array2string(self.state, precision=2))
        print(f"[vel] linear: {self.lin_vel:.2f}  angular: {self.ang_vel:.2f}" + (" (filtered)" if clamped else ""))
        
        print(f"[ctrl]: {u_sol}, [cbf_value]: {H}")


def main(args=None):
    rclpy.init(args=args)
    node = Control()

    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()