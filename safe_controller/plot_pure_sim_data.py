import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint

# -----------------------------
# Load data
# -----------------------------
npz_path = "/home/ubuntu/ros_ws/src/safe_controller/resource/sim_sinusoidal_GEKF.npz"  # <-- change path if needed
data = np.load(npz_path)

# pprint(data.files)

# -----------------------------
# Extract signals
# -----------------------------
u_traj = data["u_traj"]                  # shape (T, m)
cbf_values = data["cbf_values"]          # shape (T, k)
right_rhs = data["right_rhs"]            # shape (T,)
right_l_f_h = data["right_l_f_h_full"]   # shape (T,)
right_l_f_2_h = data["right_l_f_2_h_full"]  # shape (T,)
right_lg_lf_h = data["right_lglfh"]
right_h_ddot = right_l_f_2_h.squeeze() + np.einsum("ij,ij->i", right_lg_lf_h, u_traj.squeeze())
time = data["time"] if "time" in data else np.arange(u_traj.shape[0])

# -----------------------------
# Create subplots
# -----------------------------
fig, axs = plt.subplots(5, 1, figsize=(10, 12), sharex=True)

# Subplot 1: angular velocity (2nd component of u_traj)
axs[0].plot(time, u_traj[:, 1])
axs[0].set_ylabel("u_traj[1]")
axs[0].set_title("Angular Velocity")

# Subplot 2: 2nd column of cbf_values
axs[1].plot(time, cbf_values[:, 1])
axs[1].set_ylabel("cbf_values[:, 1]")
axs[1].set_title("CBF Value (2nd column)")

# Subplot 3: right_rhs
axs[2].plot(time, right_rhs)
axs[2].set_ylabel("right_rhs")
axs[2].set_title("Right RHS")

# Subplot 4: right_l_f_h
axs[3].plot(time, right_l_f_h)
axs[3].set_ylabel("L_f h (right)")
axs[3].set_title("Right L_f h")

# Subplot 5: h_ddot
axs[4].plot(time, right_h_ddot)
axs[4].set_ylabel("h_ddot")
axs[4].set_title("h_ddot")
axs[4].set_xlabel("Time")

for ax in axs:
    ax.grid(True)

plt.tight_layout()
plt.show()
