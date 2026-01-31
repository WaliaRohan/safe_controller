import numpy as np
import matplotlib.pyplot as plt

"""
This script plots logged data from control_lane, including:

State (ground_truth_list)

State estimation (P, K, x_hat, z_obs)

Right/Left CBF Data (h, Lfh, Lf_2_h, Lg_Lf_h, u_opt)

"""

# -----------------------------
# Load the logs
# -----------------------------
data = np.load("/home/ubuntu/ros_ws/src/safe_controller/safe_controller/pure_sim_replicate.npz", allow_pickle=True)

P_list = data["P"]            # shape (T, 4, 4)
x_hat_list = data["x_hat"]    # shape (T, n)
K_list = data["K"]            # shape (T, n, obs_dim)
z_obs_list = data["z_obs"]    # shape (T,)
cbf_left_list = data["cbf_left"]
cbf_right_list = data["cbf_right"]
u_opt_list = data["u_opt"]
ground_truth_list = data["ground_truth"]

if len(ground_truth_list) == 0:
    ground_truth_list = np.zeros((100, 2))

left_lglfh = data["left_lglfh"]
right_lglfh = data["right_lglfh"]
left_rhs = data["left_rhs"]
right_rhs = data["right_rhs"]

right_l_f_h   = data["right_l_f_h"]
right_l_f_2_h = data["right_l_f_2_h"]
left_l_f_h    = data["left_l_f_h"]
left_l_f_2_h  = data["left_l_f_2_h"]

T = P_list.shape[0]
time = np.arange(T)

scale_factor = 24  # Normalization scale (24 in = 0.61 m)

ground_truth = np.array(ground_truth_list) # in m
gt_x = ground_truth[:, 0] * scale_factor 
gt_y = ground_truth[:, 1] * scale_factor + scale_factor/2

plt.close("all") 

# Set origin and scale to inches
# gt_y[0] = 12.0
# gt_x = gt_x*12

# ============================================================
#                     FIGURE 1
# ============================================================

fig1 = plt.figure(figsize=(12, 16), constrained_layout=True)

# 1. x_hat trajectory (x vs y)
ax1 = fig1.add_subplot(4, 1, 1) 
x_vals = x_hat_list[:, 0]
y_vals = x_hat_list[:, 1]
ax1.plot(x_vals, y_vals, marker='o', markersize=2, linewidth=1)
ax1.plot(gt_x, gt_y, color="black", linestyle="--", linewidth=2, label="gt")

ax1.set_title("State Trajectory (x vs y)")
ax1.set_ylabel("y")
ax1.set_xlabel("x")
ax1.grid(True)

# 2. Diagonal elements of P
ax2 = fig1.add_subplot(4, 1, 2)
ax2.plot(time, P_list[:, 0, 0], label="P[0,0]")
ax2.plot(time, P_list[:, 1, 1], label="P[1,1]")
ax2.plot(time, P_list[:, 2, 2], label="P[2,2]")
ax2.plot(time, P_list[:, 3, 3], label="P[3,3]")
ax2.set_title("Diagonal Elements of Covariance Matrix P")
ax2.set_ylabel("Variance")
ax2.legend()
ax2.grid(True)

# 3. Kalman Gain K
ax3 = fig1.add_subplot(4, 1, 3, sharex=ax2)
K_arr = np.array(K_list)
n, obs_dim = K_arr.shape[1], K_arr.shape[2]
for i in range(n):
    for j in range(obs_dim):
        ax3.plot(time, K_arr[:, i, j], label=f"K[{i},{j}]")
ax3.set_title("Kalman Gain K Elements")
ax3.set_ylabel("Gain Value")
ax3.legend(loc="upper right", fontsize=8)
ax3.grid(True)

# 4. y_pred vs y_obs
ax4 = fig1.add_subplot(4, 1, 4, sharex=ax2)
ax4.plot(time, x_hat_list[:, 1], label="y_pred")
ax4.plot(time, z_obs_list * scale_factor, label="y_obs")
ax4.set_title("x_hat[1] vs Observations z_obs")
ax4.set_xlabel("Time Step")
ax4.set_ylabel("Value")
ax4.legend()
ax4.grid(True)

# ============================================================
#                     FIGURE 2
# ============================================================

# fig2, ax2 = plt.subplots(4, 4, figsize=(12, 12), sharex=True)

# for i in range(4):
#     for j in range(4):
#         ax2[i, j].plot(time, P_list[:, i, j])
#         ax2[i, j].set_title(f"P[{i},{j}]")

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

# fig3.tight_layout()



# # ============================================================
# #                     FIGURE 4
# #             Trajectory with Heading (θ)
# # ============================================================

# fig4, ax4 = plt.subplots(figsize=(10, 8))

# x = x_hat_list[:, 0]
# y = x_hat_list[:, 1]
# theta = x_hat_list[:, 3]

# # Arrow directions
# u = np.cos(theta)
# v = np.sin(theta)

# # Quiver thinning
# step = max(1, len(x) // 200)

# # Trajectory
# ax4.plot(x, y, color="black", linewidth=1.5, label="Trajectory")

# # Heading arrows
# ax4.quiver(
#     x[::step], y[::step],
#     u[::step], v[::step],
#     angles="xy",
#     scale_units="xy",
#     scale=1.0,
#     width=0.004,
#     color="blue",
#     alpha=0.9
# )

# # Labels + styling
# ax4.set_xlabel("x position")
# ax4.set_ylabel("y position")
# ax4.set_title("Figure 4: Trajectory with Heading (θ)")
# ax4.grid(True)
# ax4.set_aspect("auto")   # prevents squashing
# ax4.legend()

# fig4.tight_layout()

# # ============================================================
# #                     FIGURE 5
# #             Higher Order Control Data
# # ============================================================

# T = len(left_lglfh)
# time = np.arange(T)

# # --- Create NEW figure only for these plots ---
# fig_cbf, axs_cbf = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

# # ============================================================
# #   Subplot 1 — Left CBF: L_g L_f h and RHS
# # ============================================================
# axs_cbf[0].plot(time, -np.einsum("ij,ij->i", left_lglfh, u_opt_list),
#                 label="LHS (left)")
# axs_cbf[0].plot(time, left_rhs, '--', label="RHS (left)")
# axs_cbf[0].set_ylabel("Left CBF")
# axs_cbf[0].legend()
# axs_cbf[0].grid(True)

# # ============================================================
# #   Subplot 2 — Right CBF: L_g L_f h and RHS
# # ============================================================
# axs_cbf[1].plot(time, -np.einsum("ij,ij->i", right_lglfh, u_opt_list),
#                 label="LHS right)")
# axs_cbf[1].plot(time, right_rhs, '--', label="RHS (right)")
# axs_cbf[1].set_ylabel("Right CBF")
# axs_cbf[1].legend()
# axs_cbf[1].grid(True)

# # ============================================================
# #   Subplot 3 — Left: h, h_dot, h_ddot
# # ============================================================

# left_h = cbf_left_list
# left_h_dot = left_l_f_h
# left_h_ddot = left_l_f_2_h.squeeze() + np.einsum("ij,ij->i", left_lglfh, u_opt_list)

# axs_cbf[2].plot(time, left_h,      label="h_left")
# axs_cbf[2].plot(time, left_h_dot,  label="ḣ_left")
# axs_cbf[2].plot(time, left_h_ddot, markevery=50, label="ḧ_left")
# axs_cbf[2].set_ylabel("Left h / ḣ / ḧ")
# axs_cbf[2].legend()
# axs_cbf[2].grid(True)

# # ============================================================
# #   Subplot 4 — Right: h, h_dot, h_ddot
# # ============================================================

# right_h = cbf_right_list
# right_h_dot = right_l_f_h
# right_h_ddot = right_l_f_2_h.squeeze() + np.einsum("ij,ij->i", right_lglfh, u_opt_list)

# axs_cbf[3].plot(time, right_h,      label="h_right")
# axs_cbf[3].plot(time, right_h_dot,  label="ḣ_right")
# axs_cbf[3].plot(time, right_h_ddot, label="ḧ_right")
# axs_cbf[3].set_ylabel("Right h / ḣ / ḧ")
# axs_cbf[3].set_xlabel("Time Index")
# axs_cbf[3].legend()
# axs_cbf[3].grid(True)

# # ============================================================
# #                     FIGURE 6
# #     Right HOCBF | h, h_dot, h_ddot | u_opt (v, ω)
# # ============================================================

T = len(right_lglfh)
time = np.arange(T)

fig6, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

# Unpack axes for clarity
ax_hocbf   = axes[0]   # Top
ax_uopt    = axes[1]   # Middle
ax_horders = axes[2]   # Bottom

# ------------------------------------------------------------
# Subplot 1 — Right HOCBF inequality
# ------------------------------------------------------------
right_LgLf_u = -np.einsum("ij,ij->i", right_lglfh, u_opt_list)

ax_hocbf.plot(time, right_LgLf_u, label="LHS: -L_g L_f h · u")
ax_hocbf.plot(time, right_rhs, "--", label="RHS")

ax_hocbf.set_ylabel("HOCBF Inequality")
ax_hocbf.set_title("Right HOCBF:  -L_g L_f h · u  ≤  RHS")
ax_hocbf.grid(True)
ax_hocbf.legend()

# ------------------------------------------------------------
# Subplot 2 — u_opt (v, ω)
# ------------------------------------------------------------
ax_uopt.plot(time_ctrl, u_opt_list[:, 0], label="velocity (v)", color="red")
ax_uopt.plot(time_ctrl, u_opt_list[:, 1], label="heading (ω)",  color="purple")

ax_uopt.set_ylabel("Control Input")
ax_uopt.set_xlabel("Time Step")
ax_uopt.set_title("Optimal Control Inputs u = [v, ω]")
ax_uopt.grid(True)
ax_uopt.legend()

# ------------------------------------------------------------
# Subplot 3 — Right h, h_dot, h_ddot
# ------------------------------------------------------------
right_h      = np.array(cbf_right_list)
right_h_dot  = np.array(right_l_f_h)
right_h_ddot = right_l_f_2_h.squeeze() + np.einsum("ij,ij->i", right_lglfh, u_opt_list)

ax_horders.plot(time, right_h,      label="h")
# ax_horders.plot(time, right_h_dot,  label="ḣ")
ax_horders.plot(time, right_h_ddot, label="ḧ")

ax_horders.set_ylabel("CBF Value")
ax_horders.set_xlabel("Time Step")
ax_horders.set_title("Right CBF: h, ḣ, ḧ")
ax_horders.grid(True)
ax_horders.legend()

# fig6.tight_layout()

# ============================================================
#                     FIGURE 7
#       Compare ω = u_opt[:,1]  vs  -RHS / LgLfh[:,1]
# ============================================================

T = len(right_lglfh)
time = np.arange(T)

# Extract the second control input (ω)
u2 = u_opt_list[:, 1]

# Extract the second coefficient of Lg_Lf_h for the right CBF
LgLfh_right_ang = right_lglfh[:, 1]

# Compute the HOCBF-implied upper bound on u2
hocbf_bound = -right_rhs.squeeze() / LgLfh_right_ang   # elementwise division

# Create 2-subplot figure
fig7, axes7 = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# -----------------------------
# Subplot 1 — Existing ω plot
# -----------------------------
ax7 = axes7[0]

MAX_ANGULAR = 10.0
u2_clipped = np.clip(u2, -MAX_ANGULAR, MAX_ANGULAR)

ax7.plot(time, u2, label="u_opt ω (heading)", color="red")
ax7.plot(time, hocbf_bound, label="-RHS / LgLf_h[1] (bound)", linestyle="--", color="gray", alpha=0.4)

ax7.set_title("Figure 7: Heading Control vs HOCBF-implied Bound")
ax7.set_ylabel("ω (rad/s)")
ax7.set_ylim(-MAX_ANGULAR - 5, MAX_ANGULAR + 5)
ax7.grid(True)
ax7.legend()

# -----------------------------
# Subplot 2 — alpha-c1-c2 terms + Lf^2 h
# -----------------------------
ax_terms = axes7[1]

alpha = 50.0
roots = np.array([-0.75]) # Manually select root to be in left half plane
coeff = alpha*np.poly(roots)

h_ddot  = right_l_f_2_h.squeeze()

term_c1 = -alpha * coeff[0] * right_l_f_h
term_c2 = -alpha * coeff[1] * cbf_right_list

ax_terms.plot(time, term_c1, label="-α c₁ ḣ", color="blue")
ax_terms.plot(time, term_c2, label="-α c₂ h", color="green")
ax_terms.plot(time, -right_l_f_2_h, label="-L_f² h", color="black")
ax_terms.plot(time, LgLfh_right_ang, label="LgLf_h [1] (ang)")
ax_terms.plot(time, right_rhs, label="rhs")

ax_terms.set_title("Right CBF Higher-Order Terms")
ax_terms.set_ylabel("Value")
ax_terms.set_xlabel("Time Step")
# ax_terms.set_ylim(-1.2, 0.8)
ax_terms.grid(True)
ax_terms.legend()

### New figure

fig8, axes = plt.subplots(5, 1, figsize=(12, 12), sharex=True)

ax1, ax2, ax3, ax4, ax5 = axes

# 1. u_opt_list[:, 1]  (angular control)
ax1.plot(time, u_opt_list[:, 1], label="u₂ (ang control)", color='tab:blue')
ax1.set_ylabel("u₂")
ax1.grid(True)
ax1.legend()

# 2. right_rhs
ax2.plot(time, right_rhs, label="right RHS", color='tab:orange')
ax2.set_ylabel("RHS")
ax2.grid(True)
ax2.legend()

# 3. right_h
ax3.plot(time, right_h, label="h", color='tab:green')
ax3.set_ylabel("h")
ax3.grid(True)
ax3.legend()

# 4. right_h_dot
ax4.plot(time, right_h_dot, label="ḣ", color='tab:red')
ax4.set_ylabel("ḣ")
ax4.grid(True)
ax4.legend()

# 5. right_h_ddot
ax5.plot(time, right_h_ddot, label="ḧ", color='tab:purple')
ax5.set_ylabel("ḧ")
ax5.set_xlabel("Time Step")
ax5.grid(True)
ax5.legend()

fig8.tight_layout()

# ============================================================
#                     SHOW ALL FIGURES
# ============================================================

plt.show()
