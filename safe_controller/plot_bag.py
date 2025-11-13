#!/usr/bin/env python3
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # Use "Agg" for headless; "TkAgg" for live GUI
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import rosbag2_py
import rclpy.serialization as serialization
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Float32MultiArray

bag_path = "/home/ubuntu/ros_ws/experiment_bag"

# --- Detect storage backend ---
storage_id = "mcap" if any(f.endswith(".mcap") for f in os.listdir(bag_path)) else "sqlite3"

reader = rosbag2_py.SequentialReader()
storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id=storage_id)
converter_options = rosbag2_py.ConverterOptions("", "")
reader.open(storage_options, converter_options)

lane_vals, ctrl_vals, cbf_vals, rgb_imgs, gray_imgs = [], [], [], [], []

# --- Read entire bag ---
while reader.has_next():
    topic, data, t = reader.read_next()

    if topic == "/lane_position":
        msg = serialization.deserialize_message(data, Float32)
        lane_vals.append(msg.data)

    elif topic == "/control_stats":
        msg = serialization.deserialize_message(data, Float32MultiArray)
        arr = np.array(msg.data)
        if len(arr) >= 4:
            ctrl_vals.append(arr[:2])   # [linear, angular]
            cbf_vals.append(arr[-2:])   # [cbf_left, cbf_right]

    elif topic == "/rgb":
        msg = serialization.deserialize_message(data, Image)
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        rgb_imgs.append(img)

    elif topic == "/gray_pub":
        msg = serialization.deserialize_message(data, Image)
        gray = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
        gray_imgs.append(gray)

# --- Sync and report ---
n_frames = min(len(rgb_imgs), len(gray_imgs), len(ctrl_vals), len(lane_vals))
print(f"Loaded {len(lane_vals)} lanes, {len(ctrl_vals)} controls, {len(rgb_imgs)} RGB, {len(gray_imgs)} gray frames.")
print(f"Using {n_frames} synchronized frames.")

# --- Create figure layout ---
fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1])

ax_rgb  = fig.add_subplot(gs[0, 0])
ax_gray = fig.add_subplot(gs[0, 1])
ax_lane = fig.add_subplot(gs[0, 2])
ax_ctrl = fig.add_subplot(gs[1, :])
ax_cbf  = fig.add_subplot(gs[2, :])
fig.tight_layout(pad=3.0)

# --- Image panels ---
im_rgb = ax_rgb.imshow(rgb_imgs[0])
ax_rgb.set_title("RGB Image")
ax_rgb.axis("off")

im_gray = ax_gray.imshow(gray_imgs[0], cmap="gray", vmin=0, vmax=255)
ax_gray.set_title("Gray Image (/gray_pub)")
ax_gray.axis("off")

# --- Lane position plot ---
(line_lane,) = ax_lane.plot([], [], "tab:blue")
ax_lane.set_title("Lane Position (0=left, 1=right)")
ax_lane.set_xlim(0, n_frames)
ax_lane.set_ylim(0, 1)
ax_lane.grid(True)

# --- Control vector (arrow) ---
ax_ctrl.set_xlim(-1.5, 1.5)
ax_ctrl.set_ylim(-1.5, 1.5)
ax_ctrl.set_aspect("equal")
ax_ctrl.grid(True)
ax_ctrl.set_title("Control Vector (Linear vs Angular)")

# Draw a faint unit circle to visualize max control range
circle = plt.Circle((0, 0), 1.0, color='gray', fill=False, linestyle='--', alpha=0.5)
ax_ctrl.add_patch(circle)

# Initialize one static arrow at (0,0)
quiv = ax_ctrl.quiver(0, 0, 0, 0, angles='xy', scale_units='xy', scale=1, color='tab:red')

# --- CBF plots ---
(line_cbfL,) = ax_cbf.plot([], [], label="CBF Left")
(line_cbfR,) = ax_cbf.plot([], [], label="CBF Right")
ax_cbf.legend()
ax_cbf.set_title("Control Barrier Functions")
ax_cbf.grid(True)

# --- Data buffers ---
lane_x, lane_y, cbf_x, cbfL_y, cbfR_y = [], [], [], [], []

# --- Update function ---
def update(frame):
    # Images
    im_rgb.set_data(rgb_imgs[frame])
    im_gray.set_data(gray_imgs[frame])
    ax_rgb.set_title(f"RGB Image (Frame {frame+1}/{n_frames})")

    # Lane
    lane_x.append(frame)
    lane_y.append(lane_vals[frame])
    line_lane.set_data(lane_x, lane_y)
    ax_lane.set_ylim(min(0, min(lane_y)), max(1, max(lane_y)))

    # Control vector (updated in place)
    lin, ang = ctrl_vals[frame]
    norm = np.linalg.norm([lin, ang])
    if norm > 1e-6:
        lin /= norm
        ang /= norm
    quiv.set_UVC(lin, ang)
    ax_ctrl.set_title(f"Control Vector (Linear={ctrl_vals[frame][0]:.2f}, Angular={ctrl_vals[frame][1]:.2f})")

    # CBF
    cbf_x.append(frame)
    cbfL_y.append(cbf_vals[frame][0])
    cbfR_y.append(cbf_vals[frame][1])
    line_cbfL.set_data(cbf_x, cbfL_y)
    line_cbfR.set_data(cbf_x, cbfR_y)
    ax_cbf.relim(); ax_cbf.autoscale_view()

    return [im_rgb, im_gray, line_lane, quiv, line_cbfL, line_cbfR]

# --- Animate ---
ani = FuncAnimation(fig, update, frames=n_frames, interval=50, blit=False)
plt.show()
