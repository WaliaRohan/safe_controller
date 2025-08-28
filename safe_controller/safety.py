# CBF
import sys
sys.path.append("/root/cbf_lite")

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
        mu_u = 0.1
        sigma_u = jnp.sqrt(0.01) # Standard deviation
        mu_v = 0.001
        sigma_v = jnp.sqrt(0.0005) # Standard deviation
        sensor_update_frequency = 0.1 # Hz

        # State initialization, goal and constraints
        wall_y = 1.0
        self.goal = 1.5*jnp.array([1.0, 1.0])  # Goal position
        self.obstacle = jnp.array([wall_y])  # Wall

        # Sensor and estimator initialization
        # x_initial_measurement = sensor(x_true, 0, mu_u, sigma_u, mu_v, sigma_v) # mult_noise
        # x_initial_measurement = sensor(x_true, t=0, std=sigma_v) # unbiased_fixed_noise
        # Observation function: Return second and 4rth element of the state vector
        # h = lambda x: x[jnp.array([1, 3])]
        # h = lambda x: jnp.array([x[1]])
        h = None
        Q = jnp.square(0.5)*jnp.eye(self.dynamics.state_dim)
        self.estimator = GEKF(self.dynamics, mu_u, sigma_u, mu_v, sigma_v, h=h, x_init=x_initial_measurement, Q=Q)
        # estimator = EKF(dynamics, dt, h=h, x_init=x_initial_measurement, R=jnp.square(sigma_v)*jnp.eye(dynamics.state_dim))

        self.x_estimated, self.p_estimated = self.estimator.get_belief()

        # Define belief CBF parameters
        n = self.dynamics.state_dim
        alpha = jnp.array([0.0, -1.0, 0.0, 0.0])
        beta = jnp.array([-wall_y])
        delta = 0.001  # Probability of failure threshold
        self.cbf = BeliefCBF(alpha, beta, delta, n)

        # Dynamic constraints
        wheelbase = 0.165 # Wheelbase - measured directly from deepracer
        max_steering_angle = 0.8 # rad (little over 45 deg)
        r_min = wheelbase/jnp.tan(max_steering_angle) # min turning radius for this angle

        # Control params
        lin_vel = 1.0 # m/s
        ang_vel = 0.2 # rad/s -> This is filtered later based on min turning radius
        # u_max = jnp.array([lin_vel, ang_vel])
        # m = len(u_max)
        self.clf_gain = 20.0 # CLF linear gain
        self.clf_slack_penalty = 100.0
        self.cbf_gain = 10.0  # CBF linear gain
        CBF_ON = True

        # Autodiff: Compute Gradients for CLF
        self.grad_V = grad(clf, argnums=0)  # ∇V(x)

        # OSQP solver instance
        self.solver = OSQP()

    @partial(jit, static_argnums=0)
    def solve_qp_ref(self, x_estimated, covariance, u_max, u_nom):
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

    @jit
    def solve_qp(self, x_estimated, covariance, u_max, CLF_ON=False):
        m = len(u_max)

        b = self.cbf.get_b_vector(x_estimated, covariance)

        """Solve the CLF-CBF-QP using JAX & OSQP"""
        # Compute CLF components
        V = clf(x_estimated, self.goal)
        grad_V_x = self.grad_V(x_estimated, self.goal)  # ∇V(x)

        L_f_V = jnp.dot(grad_V_x.T, self.dynamics.f(x_estimated))
        L_g_V = jnp.dot(grad_V_x.T, self.dynamics.g(x_estimated))
        
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

        c = jnp.zeros(var_dim)  # No linear cost term

        A = jnp.vstack([
            jnp.concatenate([L_g_V, jnp.array([-1.0])]), #  LgV u - delta <= -LfV - gamma(V) 
            jnp.concatenate([-Lg_Lf_h, jnp.array([0.0])]), # -LgLfh u       <= -[alpha1 alpha2].T @ [Lfh h] + Lf^2h
            jnp.eye(var_dim)
        ])


        u = jnp.hstack([
            (-L_f_V - self.clf_gain * V).squeeze(),          # CLF constraint
            (rhs).squeeze(),                            # CBF constraint: rhs = -[alpha1 alpha2].T [Lfh h] + Lf^2h
            u_max, 
            jnp.inf # no upper limit on slack
        ])

        l = jnp.hstack([
            -jnp.inf, # No lower limit on CLF condition
            -jnp.inf, # No lower limit on CBF condition
            -u_max,
            0.0 # slack can't be negative
        ])

        if not self.CBF_ON:
            A = jnp.delete(A, 1, axis=0)  # Remove 2nd row
            u = jnp.delete(u, 1)          # Remove corresponding element in u
            l = jnp.delete(l, 1)          # Remove corresponding element in l

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

        x_estimated, p_estimated = self.estimator.get_belief()
