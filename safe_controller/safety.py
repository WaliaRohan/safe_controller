# CBF
import sys
sys.path.append("/home/ubuntu/cbf_lite")

import jax.numpy as jnp
from jax import grad, jit
from jaxopt import BoxOSQP as OSQP

from cbfs import BeliefCBF
from cbfs import vanilla_clf_dubins as clf
from dynamics import *
from estimators import *
from functools import partial

class Stepper():
    def __init__(self, t_init, x_initial_measurement):

        # Sim Params
        self.t = t_init
        self.dynamics = DubinsDynamics()

        # Sensor Params
        # mu_u = 0.1
        mu_u = -0.76917669977005
        sigma_u = jnp.sqrt(0.01) # Standard deviation
        mu_v = 0.335
        sigma_v = jnp.sqrt(0.0001) # Standard deviation

        # State initialization, goal and constraints
        # wall_y = 1.0
        wall_y = 0.61
        self.obstacle = jnp.array([wall_y])  # Wall

        # Estimator Initialization
        scale_factor = 0.61 # m (value used for normalization: y_normalized = y_true/scale_factor)
        h = lambda x: jnp.array([x[1]/(scale_factor)])
        # h = None
        Q = jnp.square(1.0)*jnp.eye(self.dynamics.state_dim)
        self.estimator = GEKF(self.dynamics, mu_u, sigma_u, mu_v, sigma_v,
                              h=h,
                              x_init=x_initial_measurement,
                              Q=Q,
                              P_init=jnp.eye(self.dynamics.state_dim) * 0.5)
        # estimator = EKF(dynamics, dt, h=h, x_init=x_initial_measurement, R=jnp.square(sigma_v)*jnp.eye(dynamics.state_dim))

        self.x_estimated, self.p_estimated = self.estimator.get_belief()

        # Define first CBF (y < 1)
        n = self.dynamics.state_dim
        alpha = jnp.array([0.0, -1.0, 0.0, 0.0])
        beta = jnp.array([-wall_y])
        delta = 0.001  # Probability of failure threshold
        self.cbf = BeliefCBF(alpha, beta, delta, n)

        # Define second CBF (y > 1)
        n = self.dynamics.state_dim
        alpha2 = jnp.array([0.0, 1.0, 0.0, 0.0])
        beta = jnp.array([-wall_y])
        delta = 0.001  # Probability of failure threshold
        self.cbf2 = BeliefCBF(alpha2, beta, delta, n)

        # Dynamic constraints
        wheelbase = 0.165 # Wheelbase - measured directly from deepracer
        max_steering_angle = 0.8 # rad (little over 45 deg)
        r_min = wheelbase/jnp.tan(max_steering_angle) # min turning radius for this angle

        # Control params
        self.clf_gain = 20.0 # CLF linear gain
        self.clf_slack_penalty = 100.0
        self.cbf_gain = 10.0  # CBF linear gain
        CBF_ON = True

        # Autodiff: Compute Gradients for CLF
        self.grad_V = grad(clf, argnums=0)  # ∇V(x)

        # OSQP solver instance
        self.solver = OSQP()

    @partial(jit, static_argnums=0)
    def solve_qp_ref_lane(self, x_estimated, covariance, u_max, u_nom):
        """
            Minimally invasive lane centering control
        """

        m = len(u_max)
        var_dim = m + 1

        b = self.cbf.get_b_vector(x_estimated, covariance)

        # Compute CBF components
        h = self.cbf.h_b(b)
        L_f_hb, L_g_hb, L_f_2_h, Lg_Lf_h, grad_h_b, f_b = self.cbf.h_dot_b(b, self.dynamics) # ∇h(x)

        L_f_h = L_f_hb

        rhs, L_f_h, h_gain = self.cbf.h_b_r2_RHS(h, L_f_h, L_f_2_h, self.cbf_gain)

        # Compute CBF2 components
        h_2 = self.cbf2.h_b(b)
        L_f_hb_2, L_g_hb_2, L_f_2_h_2, Lg_Lf_h_2, _, _ = self.cbf2.h_dot_b(b, self.dynamics) # ∇h(x)

        L_f_h_2 = L_f_hb_2

        rhs2, L_f_h2, _ = self.cbf2.h_b_r2_RHS(h_2, L_f_h_2, L_f_2_h_2, self.cbf_gain)

        A = jnp.vstack([
            jnp.concatenate([-Lg_Lf_h, jnp.array([0.0])]), # -LgLfh u       <= -[alpha1 alpha2].T @ [Lfh h] + Lf^2h
            jnp.concatenate([-Lg_Lf_h_2, jnp.array([0.0])]), # 2nd CBF
            jnp.eye(var_dim)
        ])

        u = jnp.hstack([
            (rhs).squeeze(),                            # CBF constraint: rhs = -[alpha1 alpha2].T [Lfh h] + Lf^2h
            (rhs2).squeeze(),                           # 2nd CBF constraint
            u_max, 
            jnp.inf # no upper limit on slack
        ])

        l = jnp.hstack([
            -jnp.inf, # No lower limit on CBF condition
            -jnp.inf, # 2nd CBF
            -u_max,
            0.0 # slack can't be negative
        ])

        # Define Q matrix: Minimize ||u||^2 and slack (penalty*delta^2)
        Q = jnp.eye(var_dim)
        Q = Q.at[-1, -1].set(2*self.clf_slack_penalty)

        c = jnp.append(-2.0*u_nom.flatten(), 0.0)

        # Solve the QP using jaxopt OSQP
        sol = self.solver.run(params_obj=(Q, c), params_eq=A, params_ineq=(l, u)).params
        return sol, h, h_2

    @partial(jit, static_argnums=0)
    def solve_qp_ref(self, x_estimated, covariance, u_max, u_nom):
        """
            Minimally invasive control for avoiding an edge
        """
        m = len(u_max)

        b = self.cbf.get_b_vector(x_estimated, covariance)

        # # Compute CBF components
        h = self.cbf.h_b(b)
        L_f_hb, L_g_hb, L_f_2_h, Lg_Lf_h, grad_h_b, f_b = self.cbf.h_dot_b(b, self.dynamics) # ∇h(x)

        L_f_h = L_f_hb
        L_g_h = L_g_hb

        rhs, L_f_h, h_gain = self.cbf.h_b_r2_RHS(h, L_f_h, L_f_2_h, self.cbf_gain)

        var_dim = m + 1

        # Define Q matrix: Minimize ||u||^2 and slack (penalty*delta^2)
        Q = jnp.eye(var_dim)
        Q = Q.at[-1, -1].set(2*self.clf_slack_penalty)

        # This accounts for reference trajectory
        c = jnp.append(-2.0*u_nom.flatten(), 0.0)

        A = jnp.vstack([
            jnp.concatenate([-Lg_Lf_h, jnp.array([0.0])]), # -LgLfh u       <= -[alpha1 alpha2].T @ [Lfh h] + Lf^2h
            jnp.eye(var_dim)
        ])

        u = jnp.hstack([
            (rhs).squeeze(),                            # CBF constraint: rhs = -[alpha1 alpha2].T [Lfh h] + Lf^2h
            u_max, 
            jnp.inf # no upper limit on slack
        ])

        l = jnp.hstack([
            -jnp.inf, # No lower limit on CBF condition
            -u_max,
            0.0 # slack can't be negative
        ])

        # Solve the QP using jaxopt OSQP
        sol = self.solver.run(params_obj=(Q, c), params_eq=A, params_ineq=(l, u)).params
        return sol, h
    
    # @partial(jit, static_argnums=0)
    def step_predict(self, t, u):

        dt = self.t - t
        self.t = t

        # belief = self.cbf.get_b_vector(self.x_estimated, self.p_estimated)

        # sol, _ = self.solve_qp_ref(belief)

        # u_sol = sol.primal[0][:2]
        # u_opt = jnp.clip(u_sol, -u_max, u_max)

        # Clip ang_vel based on min turning radius
        # theta_est = x_estimated[-1]
        # vel_des = u_opt[0]
        # w_max = vel_des*jnp.tan(max_steering_angle)/(wheelbase) # Comes from time derivative of r_min = L/tan(max_steering_angle)
        # u_opt = u_opt.at[1].set(jnp.clip(u_opt[1], -w_max, w_max))

        # Apply control to the true state (x_true)
        self.estimator.predict(u, dt)

    # @partial(jit, static_argnums=0)
    def step_measure(self, x_measured):
        # update measurement and estimator belief
    
        if self.estimator.name == "GEKF":
            self.estimator.update(x_measured)

        if self.estimator.name == "EKF":
            self.estimator.update(x_measured)
