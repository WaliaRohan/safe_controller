import safe_controller.safety as safety
import numpy as np
import jax.numpy as jnp
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Float32
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Float32MultiArray
from tf2_msgs.msg import TFMessage

TOPIC_NOM_CTRL = "/nominal_control"
TOPIC_SAFE_CTRL = "/cmd_vel"
TOPIC_ODOM = "/vicon_pose"
TOPIC_LANE_POSE = "/lane_position"
TOPIC_CONTROL_STATS = "/control_stats"

MAX_LINEAR = 1.0
MAX_ANGULAR = 1.0
U_MAX = np.array([MAX_LINEAR, MAX_ANGULAR])

class Control(Node):
    def __init__(self):
        super().__init__('control') # STEADY TIME IS REQUIRED FOR POSITIVE dt FOR PREDICTION STEP

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

        self.control_stats_publisher_ = self.create_publisher(Float32MultiArray,
                                                              TOPIC_CONTROL_STATS,
                                                              10)

        self.nom_lin_vel = 0.0
        self.nom_ang_vel = 0.0

        self.lin_vel_cmd = 0.0
        self.ang_vel_cmd = 0.0

        self.wall_y = 24.0 # Lane width (24 in / 0.61 m)

        self.state = jnp.array([0.0, self.wall_y/2, 0.25, 0.0]) # x y v theta
        self.covariance = jnp.eye(len(self.state)) * 1.0
        self.state_initialized = False
        self.stepper_initialized = False

        print("[Control] Trying to initialize state")

        while not self.state_initialized:
            rclpy.spin_once(self, timeout_sec=0.1)

        self.stepper = safety.Stepper(t_init=self.get_time(),
                                      x_initial_measurement=self.state,
                                      P_init=self.covariance,
                                      wall_y = self.wall_y) # CHANGE THIS LATER!
        self.stepper_initialized = True

        rate = 5000.0 # Hz
        self.safety_timer = self.create_timer((1/rate), self.safety_filter)
        self.pub_timer = self.create_timer(1/rate, self.publisher_callback)
        self.nominal_timer = self.create_timer(1/rate, self.nominal_vel_loop)

        # Lists for logging values
        self.P_list = []
        self.x_hat_list = []
        self.K_list = []
        self.z_obs_list = []
        self.cbf_left_list = []
        self.cbf_right_list = []
        
        self.u_opt_list = []
        self.ground_truth_list = []
        self.right_lglfh_list = []
        self.right_rhs_list = []
        self.right_l_f_h_list = []
        self.right_l_f_2_h_list = []
        self.left_lglfh_list = []
        self.left_rhs_list = []
        self.left_l_f_h_list = []
        self.left_l_f_2_h_list = []

        self.state_time_list = []
        self.control_time_list = []
        
        self.origin_rotated = None   # Will store first rotated (x, y) from tf as origin

        self.tf_sub = self.create_subscription(
            TFMessage,
            '/tf',
            self.tf_callback,
            10
        )

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
            
            z_obs = self.stepper.step_measure(msg.data)

            self.P_list.append(self.stepper.estimator.P)
            self.x_hat_list.append(self.stepper.estimator.x_hat)
            self.K_list.append(self.stepper.estimator.K)
            self.z_obs_list.append(z_obs)
            self.state_time_list.append(self.stepper.t)

            self.state, cov = self.stepper.estimator.get_belief()
            # print(f"{np.array2string(np.asarray(self.state), precision=3)}, {np.trace(np.asarray(cov)):.3f}")
            # print(f"{np.array2string(np.asarray(self.state), precision=3)}, {cov[1, 1]:.3f}, {self.stepper.estimator.K[1, 1]:.3f}")

    def nominal_vel_loop(self):
        """
        Sets the nominal velocity to strictly forward
        """
        self.nom_lin_vel = 0.25
        self.nom_ang_vel = 0.0

    def publisher_callback(self):
        """
        Publishes filtered control command
        """
        twist = Twist()
        twist.linear.x = self.lin_vel_cmd
        twist.angular.z = self.ang_vel_cmd
        self.publisher_.publish(twist)

        if self.state_initialized and self.stepper_initialized:
            self.stepper.step_predict(self.get_time(), np.array([self.lin_vel_cmd, self.ang_vel_cmd]))
            self.state, self.covariance = self.stepper.estimator.get_belief()

    def safety_filter(self):
        """
            Calls minimally invasive CBF QP to calculate safety commands
        """
        u_nom = np.array([self.nom_lin_vel, self.nom_ang_vel])
        sol, h, L_f_h, L_f_2_h, Lg_Lf_h, rhs, h_2, L_f_h_2, L_f_2_h_2, Lg_Lf_h_2, rhs2 = self.stepper.solve_qp_ref_lane(self.state, self.covariance, U_MAX, u_nom)  # Replace zeros with proper covariance (Look at odom callback)

        u_sol = sol.primal[0][:2]
        u_opt = np.clip(u_sol, -U_MAX @ np.array([[1.0, 0.0], [0.0, 1.0]]), U_MAX)

        self.control_time_list.append(self.stepper.t)
        self.cbf_left_list.append(h_2)
        self.cbf_right_list.append(h)
        self.u_opt_list.append(u_opt)
        self.right_lglfh_list.append(Lg_Lf_h)
        self.left_lglfh_list.append(Lg_Lf_h_2)
        self.right_rhs_list.append(rhs)
        self.left_rhs_list.append(rhs2)
        self.right_l_f_h_list.append(L_f_h)
        self.right_l_f_2_h_list.append(L_f_2_h)
        self.left_l_f_h_list.append(L_f_h_2)
        self.left_l_f_2_h_list.append(L_f_2_h_2)

        self.lin_vel_cmd = np.float64(u_opt[0])
        self.ang_vel_cmd = np.float64(u_opt[1])

        self.state = self.state.at[2].set(self.lin_vel_cmd)
        
        # print(f"[{u_sol[0]: .3f}, {u_sol[1]: .3f}], [Left CBF (wall_y > y)]: {h_2:.3f}, [Right CBF (y > 0)]: {h:.3f}")

        msg = Float32MultiArray()
        msg.data = [float(u_sol[0]), float(u_sol[1]), float(h_2), float(h)]

        self.control_stats_publisher_.publish(msg)

    def save_logs(self):
        np.savez(
            "/home/ubuntu/ros_ws/src/safe_controller/safe_controller/logs.npz",
            P=np.array(self.P_list),
            x_hat=np.array(self.x_hat_list),
            K=np.array(self.K_list),
            z_obs=np.array(self.z_obs_list),
            cbf_left = np.array(self.cbf_left_list),
            cbf_right = np.array(self.cbf_right_list),
            u_opt = np.array(self.u_opt_list),
            ground_truth = np.array(self.ground_truth_list),
            
            left_lglfh=np.array(self.left_lglfh_list),
            right_lglfh=np.array(self.right_lglfh_list),
            right_rhs=np.array(self.right_rhs_list),
            left_rhs=np.array(self.left_rhs_list),

            right_l_f_h=np.array(self.right_l_f_h_list),
            right_l_f_2_h=np.array(self.right_l_f_2_h_list),
            left_l_f_h=np.array(self.left_l_f_h_list),
            left_l_f_2_h=np.array(self.left_l_f_2_h_list),

            state_time = np.array(self.state_time_list),
            control_time = np.array(self.control_time_list)
            )

    def tf_callback(self, msg):
        for transform in msg.transforms:
            if transform.child_frame_id == "base_link":

                # Original TF translation
                x = transform.transform.translation.x
                y = transform.transform.translation.y

                # ------------------------------------------
                # Apply 90° clockwise rotation:
                # (x', y') = (y, -x)
                # ------------------------------------------
                x_rot = y
                y_rot = -x

                # ------------------------------------------
                # Establish origin on FIRST measurement
                # ------------------------------------------
                if self.origin_rotated is None:
                    self.origin_rotated = (x_rot, y_rot)
                    self.get_logger().info(
                        f"Set origin (rotated): {self.origin_rotated}"
                    )

                # ------------------------------------------
                # Translate so first sample becomes (0,0)
                # ------------------------------------------
                x_zeroed = x_rot - self.origin_rotated[0]
                y_zeroed = y_rot - self.origin_rotated[1]

                # Store result
                self.ground_truth_list.append((x_zeroed, y_zeroed))

                # Optional debug print
                self.get_logger().info(
                    f"Recorded GT: x={x_zeroed:.3f}, y={y_zeroed:.3f}"
                )


def main(args=None):
    rclpy.init(args=args)
    node = Control()

    try:
        rclpy.spin(node)
    finally:
        node.save_logs()
        node.destroy_node()
        rclpy.shutdown()
        

if __name__ == '__main__':
    main()