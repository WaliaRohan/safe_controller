#!/usr/bin/env python3
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import rosbag2_py
import rclpy.serialization as serialization
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Float32MultiArray

# ----------------------------
# CONFIG
# ----------------------------
bag_path = "experiment_bag"

# auto-detect storage type
if any(f.endswith('.mcap') for f in os.listdir(bag_path)):
    storage_id = 'mcap'
else:
    storage_id = 'sqlite3'

# ----------------------------
# READ BAG
# ----------------------------
reader = rosbag2_py.SequentialReader()
storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id=storage_id)
converter_options = rosbag2_py.ConverterOptions('', '')
reader.open(storage_options, converter_options)

lane_vals = []
ctrl_vals = []
cbf_vals = []
rgb_imgs = []

while reader.has_next():
    topic, data, t = reader.read_next()

    if topic == "/lane_position":
        msg = serialization.deserialize_message(data, Float32)
        lane_vals.append(msg.data)

    elif topic == "/control_stats":
        msg = serialization.deserialize_message(data, Float32MultiArray)
        arr = np.array(msg.data)
        if len(arr) >= 4:
            ctrl_vals.append(arr[:2])   # linear, angular
            cbf_vals.append(arr[-2:])   # left, right

    elif topic == "/rgb":
        msg = serialization.deserialize_message(data, Image)
        img_np = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        rgb_imgs.append(gray)

print(f"Loaded {len(lane_vals)} lane positions, {len(ctrl_vals)} control stats, {len(rgb_imgs)} images.")

# ----------------------------
# PLOTS
# ----------------------------
fig, axs = plt.subplots(2, 2, figsize=(12, 8))
ax_img, ax_lane, ax_ctrl, ax_cbf = axs.flatten()

# 1. Gray image (just show last one)
if rgb_imgs:
    ax_img.imshow(rgb_imgs[-1], cmap='gray')
    ax_img.set_title("Gray image (last frame)")
    ax_img.axis('off')

# 2. Lane position (normalized 0–1)
if lane_vals:
    ax_lane.plot(np.linspace(0, 1, len(lane_vals)), lane_vals, color='tab:blue')
    ax_lane.set_title("Lane position (0=left, 1=right)")
    ax_lane.set_xlabel("Normalized time")
    ax_lane.set_ylabel("Position")

# 3. Control stats: lin, ang velocity
if ctrl_vals:
    ctrl_vals = np.array(ctrl_vals)
    ax_ctrl.plot(ctrl_vals[:, 0], label="Linear vel")
    ax_ctrl.plot(ctrl_vals[:, 1], label="Angular vel")
    ax_ctrl.set_title("Control commands")
    ax_ctrl.legend()
    ax_ctrl.set_xlabel("Step")

# 4. CBF Left / Right
if cbf_vals:
    cbf_vals = np.array(cbf_vals)
    ax_cbf.plot(cbf_vals[:, 0], label="CBF Left (wall_y > y)")
    ax_cbf.plot(cbf_vals[:, 1], label="CBF Right (y > 0)")
    ax_cbf.set_title("Control Barrier Functions")
    ax_cbf.legend()
    ax_cbf.set_xlabel("Step")

plt.tight_layout()
plt.show()
