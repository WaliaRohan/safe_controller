#!/usr/bin/env python3
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")   # use Agg backend for Docker/headless
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import rosbag2_py
import rclpy.serialization as serialization
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Float32MultiArray

bag_path = "experiment_bag"

# Detect MCAP vs SQLite3
storage_id = "mcap" if any(f.endswith(".mcap") for f in os.listdir(bag_path)) else "sqlite3"

reader = rosbag2_py.SequentialReader()
storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id=storage_id)
converter_options = rosbag2_py.ConverterOptions("", "")
reader.open(storage_options, converter_options)

lane_vals, ctrl_vals, cbf_vals, rgb_imgs = [], [], [], []

# Read entire bag into memory
while reader.has_next():
    topic, data, t = reader.read_next()
    if topic == "/lane_position":
        msg = serialization.deserialize_message(data, Float32)
        lane_vals.append(msg.data)
    elif topic == "/control_stats":
        msg = serialization.deserialize_message(data, Float32MultiArray)
        arr = np.array(msg.data)
        if len(arr) >= 4:
            ctrl_vals.append(arr[:2])   # lin, ang
            cbf_vals.append(arr[-2:])   # left, right
    elif topic == "/rgb":
        msg = serialization.deserialize_message(data, Image)
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rgb_imgs.append(gray)

n_frames = min(len(rgb_imgs), len(ctrl_vals), len(lane_vals))
print(f"Loaded {len(lane_vals)} lanes, {len(ctrl_vals)} controls, {len(rgb_imgs)} images.")
print(f"Using {n_frames} synchronized frames.")

# --- Create layout ---
fig = plt.figure(figsize=(10, 8))
gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1])
ax_img = fig.add_subplot(gs[:, 0])
ax_lane = fig.add_subplot(gs[0, 1])
ax_ctrl = fig.add_subplot(gs[1, 1])
ax_cbf = fig.add_subplot(gs[2, 1])
fig.tight_layout(pad=3.0)

# Image panel
im = ax_img.imshow(np.zeros_like(rgb_imgs[0]), cmap="gray", vmin=0, vmax=255)
ax_img.set_title("Gray Image")
ax_img.axis("off")

# Lane position
(line_lane,) = ax_lane.plot([], [], "tab:blue")
ax_lane.set_title("Lane Position (0=left, 1=right)")
ax_lane.set_xlim(0, n_frames)
ax_lane.set_ylim(0, 1)
ax_lane.grid(True)

# Control
(line_lin,) = ax_ctrl.plot([], [], label="Linear vel")
(line_ang,) = ax_ctrl.plot([], [], label="Angular vel")
ax_ctrl.legend()
ax_ctrl.set_title("Control Vector")
ax_ctrl.set_xlim(0, n_frames)
ax_ctrl.grid(True)

# CBF
(line_cbfL,) = ax_cbf.plot([], [], label="CBF Left")
(line_cbfR,) = ax_cbf.plot([], [], label="CBF Right")
ax_cbf.legend()
ax_cbf.set_title("Control Barrier Functions")
ax_cbf.set_xlim(0, n_frames)
ax_cbf.grid(True)

lane_x, lane_y, ctrl_x, lin_y, ang_y, cbf_x, cbfL_y, cbfR_y = [], [], [], [], [], [], [], []

def update(frame):
    im.set_data(rgb_imgs[frame])
    ax_img.set_title(f"Frame {frame+1}/{n_frames}")

    lane_x.append(frame)
    lane_y.append(lane_vals[frame])
    line_lane.set_data(lane_x, lane_y)
    ax_lane.set_ylim(min(0, min(lane_y)), max(1, max(lane_y)))

    ctrl_x.append(frame)
    lin_y.append(ctrl_vals[frame][0])
    ang_y.append(ctrl_vals[frame][1])
    line_lin.set_data(ctrl_x, lin_y)
    line_ang.set_data(ctrl_x, ang_y)
    ax_ctrl.relim(); ax_ctrl.autoscale_view()

    cbf_x.append(frame)
    cbfL_y.append(cbf_vals[frame][0])
    cbfR_y.append(cbf_vals[frame][1])
    line_cbfL.set_data(cbf_x, cbfL_y)
    line_cbfR.set_data(cbf_x, cbfR_y)
    ax_cbf.relim(); ax_cbf.autoscale_view()

    return [im, line_lane, line_lin, line_ang, line_cbfL, line_cbfR]

ani = FuncAnimation(fig, update, frames=n_frames, interval=50, blit=False)
plt.show()

# Save MP4 if no display (Docker)
# out_file = os.path.join(bag_path, "bag_combined.mp4")
# ani.save(out_file, fps=20, dpi=150)
# print(f"✅ Saved combined video to {out_file}")
