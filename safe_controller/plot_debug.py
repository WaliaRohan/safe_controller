import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Load the logs
# -----------------------------
data = np.load("/home/ubuntu/ros_ws/src/safe_controller/safe_controller/logs.npz", allow_pickle=True)

P_list = data["P"]            # shape (T, 4, 4)
x_hat_list = data["x_hat"]    # shape (T, n)
K_list = data["K"]            # shape (T, n, obs_dim)
z_obs_list = data["z_obs"]    # shape (T,)
cbf_left_list = data["cbf_left"]
cbf_right_list = data["cbf_right"]
u_opt_list = data["u_opt"]
ground_truth_list = data["ground_truth"]

T = P_list.shape[0]
time = np.arange(T)

scale_factor = 24  # Normalization scale (24 in = 0.61 m)

ground_truth = np.array(ground_truth_list) # in m
gt_x = ground_truth[:, 0] * scale_factor 
gt_y = ground_truth[:, 1] * scale_factor + scale_factor/2

# Set origin and scale to inches
# gt_y[0] = 12.0
# gt_x = gt_x*12

# ============================================================
#                     FIGURE 1
# ============================================================

fig1, axs1 = plt.subplots(4, 1, figsize=(12, 16), sharex=True)

# 1. Diagonal elements of P
ax = axs1[0]
ax.plot(time, P_list[:, 0, 0], label="P[0,0]")
ax.plot(time, P_list[:, 1, 1], label="P[1,1]")
ax.plot(time, P_list[:, 2, 2], label="P[2,2]")
ax.plot(time, P_list[:, 3, 3], label="P[3,3]")
ax.set_title("Diagonal Elements of Covariance Matrix P")
ax.set_ylabel("Variance")
ax.legend()
ax.grid(True)

# 2. x_hat trajectory (x vs y)
ax = axs1[1]
x_vals = x_hat_list[:, 0]
y_vals = x_hat_list[:, 1]
ax.plot(x_vals, y_vals, marker='o', markersize=2, linewidth=1)
ax.plot(gt_x, gt_y, color="black", linestyle="--", linewidth=2, label="gt")

ax.set_title("State Trajectory (x vs y)")
ax.set_ylabel("y")
ax.set_xlabel("x")
ax.grid(True)

# 3. Kalman Gain K
ax = axs1[2]
K_arr = np.array(K_list)
n, obs_dim = K_arr.shape[1], K_arr.shape[2]
for i in range(n):
    for j in range(obs_dim):
        ax.plot(time, K_arr[:, i, j], label=f"K[{i},{j}]")
ax.set_title("Kalman Gain K Elements")
ax.set_ylabel("Gain Value")
ax.legend(loc="upper right", fontsize=8)
ax.grid(True)

# 4. y_pred vs y_obs
ax = axs1[3]
ax.plot(time, x_hat_list[:, 1], label="y_pred")
ax.plot(time, z_obs_list * scale_factor, label="y_obs")
ax.set_title("x_hat[1] vs Observations z_obs")
ax.set_xlabel("Time Step")
ax.set_ylabel("Value")
ax.legend()
ax.grid(True)

fig1.tight_layout()

# ============================================================
#                     FIGURE 2
# ============================================================

fig2, ax2 = plt.subplots(4, 4, figsize=(12, 12), sharex=True)

for i in range(4):
    for j in range(4):
        ax2[i, j].plot(time, P_list[:, i, j])
        ax2[i, j].set_title(f"P[{i},{j}]")

# ============================================================
#                     FIGURE 3
# ============================================================

time_ctrl = np.arange(len(cbf_left_list))

fig3, axs3 = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# CBFs on same plot
ax_cbf = axs3[0]
ax_cbf.plot(time_ctrl, cbf_left_list,  label="cbf_left",  color="blue")
ax_cbf.plot(time_ctrl, cbf_right_list, label="cbf_right", color="green")
ax_cbf.set_ylabel("CBF value")
ax_cbf.set_title("Figure 3: CBFs and Optimal Control Inputs")
ax_cbf.grid(True)
ax_cbf.legend()

# u_opt (velocity and heading)
ax_u = axs3[1]
ax_u.plot(time_ctrl, u_opt_list[:, 0], label="velocity", color="red")
ax_u.plot(time_ctrl, u_opt_list[:, 1], label="heading", color="purple")
ax_u.set_ylabel("u_opt")
ax_u.set_xlabel("Time step")
ax_u.grid(True)
ax_u.legend()

fig3.tight_layout()



# ============================================================
#                     FIGURE 4
#             Trajectory with Heading (θ)
# ============================================================

fig4, ax4 = plt.subplots(figsize=(10, 8))

x = x_hat_list[:, 0]
y = x_hat_list[:, 1]
theta = x_hat_list[:, 3]

# Arrow directions
u = np.cos(theta)
v = np.sin(theta)

# Quiver thinning
step = max(1, len(x) // 200)

# Trajectory
ax4.plot(x, y, color="black", linewidth=1.5, label="Trajectory")

# Heading arrows
ax4.quiver(
    x[::step], y[::step],
    u[::step], v[::step],
    angles="xy",
    scale_units="xy",
    scale=1.0,
    width=0.004,
    color="blue",
    alpha=0.9
)

# Labels + styling
ax4.set_xlabel("x position")
ax4.set_ylabel("y position")
ax4.set_title("Figure 4: Trajectory with Heading (θ)")
ax4.grid(True)
ax4.set_aspect("auto")   # prevents squashing
ax4.legend()

fig4.tight_layout()


# ============================================================
#                     SHOW ALL FIGURES
# ============================================================
plt.show()
