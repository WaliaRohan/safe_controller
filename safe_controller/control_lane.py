import safe_controller.safety as safety
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32
from scipy.spatial.transform import Rotation as R

TOPIC_NOM_CTRL = "/nominal_control"
TOPIC_SAFE_CTRL = "/cmd_vel"
TOPIC_ODOM = "/vicon_pose"
TOPIC_LANE_POSE = "/lane_position"

MAX_LINEAR = 0.5
MAX_ANGULAR = 0.25
U_MAX = np.array([MAX_LINEAR, MAX_ANGULAR])

class Control(Node):
    def __init__(self):
        super().__init__('control')

        topics = self.get_topic_names_and_types()
        topic_names = [t[0] for t in topics]
        print(topic_names)

        qos_profile_depth = 10 # This is the message queue size
        
        # self.nominal_vel_subscriber_ = self.create_subscription(Twist,
        #                                             TOPIC_NOM_CTRL,
        #                                             self.nominal_vel_subscriber_callback,
        #                                             qos_profile_depth)
        
        # self.odom_subscriber_ = self.create_subscription(PoseStamped,
        #                                                  TOPIC_ODOM,
        #                                                  self.odom_subscriber_callback,
        #                                                  qos_profile_depth)
        
        self.perception_subscriber_callback_ = self.create_subscription(Float32,
                                                                        TOPIC_LANE_POSE,
                                                                        self.perception_subscriber_callback,
                                                                        qos_profile_depth)
        
        self.publisher_ = self.create_publisher(Twist,
                                                TOPIC_SAFE_CTRL,
                                                qos_profile_depth)

        self.nom_lin_vel = 0.0
        self.nom_ang_vel = 0.0

        self.lin_vel_cmd = 0.0
        self.ang_vel_cmd = 0.0

        self.wall_y = 0.61 # Lane width in m (24 inches)

        self.state = np.array([0.5, self.wall_y/2, 0.5, 0.0])
        self.state_initialized = False
        self.stepper_initialized = False

        print("[Control] Trying to initialize state")

        while not self.state_initialized:
            rclpy.spin_once(self, timeout_sec=0.1)

        self.stepper = safety.Stepper(t_init=self.get_time(),
                                      x_initial_measurement=self.state) # CHANGE THIS LATER!
        self.stepper_initialized = True

        rate = 100.0 # Hz
        self.safety_timer = self.create_timer((1/rate), self.safety_filter)
        self.publisher_timer = self.create_timer((1/rate), self.publisher_callback)
        self.publisher_timer = self.create_timer((1/rate), self.nominal_vel_loop)

        print("[Control] Safe Controller Initialized")

    def get_time(self):
        t = self.get_clock().now().seconds_nanoseconds()
        secs, nsecs = t
        time_in_seconds = secs + nsecs * 1e-9
        return time_in_seconds

    def perception_subscriber_callback(self, msg):
        # 

        if not self.state_initialized:
            print("[Control_lane] Got lane_pos, initializing state")
            self.state_initialized = True
        
        if self.stepper_initialized:
             # Adding NaN to ensure x, v, and theta aren't used anywhere else
            # measurement = np.array([np.NAN,
            #                         msg.data,
            #                         np.NAN,
            #                         np.NAN])
            
            self.stepper.step_measure(msg.data)
            self.state, cov = self.stepper.estimator.get_belief()
            # print(f"{np.array2string(np.asarray(self.state), precision=3)}, {np.trace(np.asarray(cov)):.3f}")

    def nominal_vel_loop(self):
        """
        Sets the nominal velocity to strictly forward
        """
        self.nom_lin_vel = 0.5
        self.nom_ang_vel = 0.0

    def publisher_callback(self):
        """
        Publishes filtered control command
        """
        twist = Twist()
        twist.linear.x = self.lin_vel_cmd
        twist.angular.z = self.ang_vel_cmd
        self.publisher_.publish(twist)

        # if self.state_initialized and self.stepper_initialized:
        #     self.stepper.step_predict(self.get_time(), np.array([self.lin_vel_cmd, self.ang_vel_cmd]))
        #     self.state, _ = self.stepper.estimator.get_belief()

    def safety_filter(self):
        """
            Calls minimally invasive CBF QP to calculate safety commands
        """
        u_nom = np.array([self.nom_lin_vel, self.nom_ang_vel])
        sol, h, h_2 = self.stepper.solve_qp_ref_lane(self.state, 0.01*np.ones((4, 4)), U_MAX, u_nom)  # Replace zeros with proper covariance (Look at odom callback)
        u_sol = sol.primal[0][:2]
        u_opt = np.clip(u_sol, -U_MAX, U_MAX)

        self.lin_vel_cmd = np.float64(u_opt[0])
        self.ang_vel_cmd = np.float64(u_opt[1])

        self.state = self.state.at[2].set(self.lin_vel_cmd)

        # self.publisher_callback()

        # print("[state]:", np.array2string(self.state, precision=2))
        # print(f"[vel_cmd] linear: {self.lin_vel_cmd:.2f}  angular: {self.ang_vel_cmd:.2f}")
        
        # print(f"[ctrl]: {u_opt}, [cbf 1 value (y < 1)]: {h}, [cbf 2 value (y > -1)]: {h_2}")


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