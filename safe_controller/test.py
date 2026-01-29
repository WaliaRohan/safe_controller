import jax.numpy as jnp
from jaxopt import BoxOSQP as OSQP
import matplotlib.pyplot as plt

def solve(Lg_Lf_h, Lg_Lf_h_2, rhs, rhs2):

    solver = OSQP()

    var_dim=3
    slack_penalty = 100.0

    u_nom =  jnp.array([0.25, 0.0])

    MAX_LINEAR = 0.25
    MAX_ANGULAR = 10.0
    u_max = jnp.array([MAX_LINEAR, MAX_ANGULAR])


    A = jnp.vstack([
        jnp.concatenate([-Lg_Lf_h, jnp.array([0.0])]), # -LgLfh u <= [alpha1 alpha2].T @ [Lfh h] + Lf^2h
        jnp.concatenate([-Lg_Lf_h_2, jnp.array([0.0])]), # 2nd CBF
        jnp.eye(var_dim)
    ])

    u = jnp.hstack([
        (rhs),                            # rhs = [alpha1 alpha2].T [Lfh h] + Lf^2h
        (rhs2),                           # 2nd CBF
        u_max, 
        jnp.inf # no upper limit on slack
    ])

    l = jnp.hstack([
        -jnp.inf, # No lower limit on CBF condition
        -jnp.inf, # 2nd CBF
        -u_max @ jnp.array([[1.0, 0.0], [0.0, 1.0]]), # Cap lower lin speed at 25 % of high linear speed value
        0.0 # slack can't be negative
    ])

    # Define Q matrix: Minimize ||u||^2 and slack (penalty*delta^2)
    Q = jnp.eye(var_dim)
    Q = Q.at[-1, -1].set(2*slack_penalty)

    c = jnp.append(-2.0*u_nom.flatten(), 0.0)

    # Solve the QP using jaxopt OSQP
    sol = solver.run(params_obj=(Q, c), params_eq=A, params_ineq=(l, u)).params
    return sol


def run(Lg_Lf_h, Lg_Lf_h_2, rhs, rhs2):

    print(f"Right: u >= {-rhs/Lg_Lf_h}")

    u_list = []

    MAX_LINEAR = 0.25
    MAX_ANGULAR = 10.0
    u_max = jnp.array([MAX_LINEAR, MAX_ANGULAR])

    sol = solve(Lg_Lf_h, Lg_Lf_h_2, rhs, rhs2)
    u_sol = sol.primal[0][:2]
    u_opt = jnp.clip(u_sol, -u_max @ jnp.array([[0.05, 0.0], [0.0, 1.0]]), u_max)

    print(u_opt)
    
    # for _ in range(1, 10):
    #     sol = solve(Lg_Lf_h, Lg_Lf_h_2, rhs, rhs2)
    #     u_sol = sol.primal[0][:2]
    #     u_opt = jnp.clip(u_sol, -u_max @ jnp.array([[0.05, 0.0], [0.0, 1.0]]), u_max)
    #     u_list.append(u_opt)

    # fig_u, ax_u = plt.subplots(figsize=(8, 4))

    # time_ctrl = jnp.arange(len(u_list))

    # u_list = jnp.array(u_list)

    # ax_u.plot(time_ctrl, u_list[:, 0], label="velocity", color="red")
    # ax_u.plot(time_ctrl, u_list[:, 1], label="heading", color="purple")

    # ax_u.set_ylabel("u_opt")
    # ax_u.set_xlabel("Time step")
    # ax_u.grid(True)
    # ax_u.legend()

    # fig_u.tight_layout()
    # plt.show()

if __name__ == '__main__':
    
    run(
        Lg_Lf_h   = jnp.array([0.993739, 0.05535129], dtype=jnp.float32),      # right
        Lg_Lf_h_2 = jnp.array([-0.9941349, -0.05454381], dtype=jnp.float32),   # left
        rhs       = -1.30436,                                                  # right RHS
        rhs2      = 5.1878104                                                  # left RHS
    )

    run(
    Lg_Lf_h   = jnp.array([0.9996651, -0.01283219], dtype=jnp.float32),     # right
    Lg_Lf_h_2 = jnp.array([-0.9996845, 0.01266034], dtype=jnp.float32),     # left
    rhs       = -1.29069891,                                                # right RHS
    rhs2      = 5.1742644                                                   # left RHS
    )
    