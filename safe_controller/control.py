
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

TOPIC_NOM_CTRL = "/nominal_control"
TOPIC_SAFE_CTRL = "/cmd_vel"

class Control(Node):
    def __init__(self):
        super().__init__('control')

        qos_profile_depth = 10 # This is the message queue size
        
        self.subscriber_ = self.create_subscription(Twist,
                                                    TOPIC_NOM_CTRL,
                                                    self.subscriber_callback,
                                                    qos_profile_depth)
        
        self.publisher_ = self.create_publisher(Twist,
                                                TOPIC_SAFE_CTRL,
                                                qos_profile_depth)

        self.lin_vel = 0.0
        self.ang_vel = 0.0
        
    
    def subscriber_callback(self, msg):
        self.lin_vel = msg.linear.x
        self.ang_vel = msg.angular.z

        self.publisher_callback()        

    def publisher_callback(self):
        twist = Twist()
        twist.linear.x = self.lin_vel
        twist.angular.z = self.ang_vel
        self.publisher_.publish(twist)

        print(f"[vel] linear: {self.lin_vel:.2f}  angular: {self.ang_vel:.2f}")
    

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