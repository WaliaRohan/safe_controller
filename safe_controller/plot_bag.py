#!/usr/bin/env python3
import os
from tqdm import tqdm
import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # Use "Agg" for headless; "TkAgg" for live GUI
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
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

# --- Store (timestamp, data) for each topic ---
lane_msgs, ctrl_msgs, cbf_msgs, rgb_msgs, gray_msgs = [], [], [], [], []

while reader.has_next():
    topic, data, t = reader.read_next()
    if topic == "/lane_position":
        msg = serialization.deserialize_message(data, Float32)
        lane_msgs.append((t, msg.data))
    elif topic == "/control_stats":
        msg = serialization.deserialize_message(data, Float32MultiArray)
        arr = np.array(msg.data)
        if len(arr) >= 4:
            ctrl_msgs.append((t, arr[:2]))   # [linear, angular]
            cbf_msgs.append((t, arr[-2:]))   # [cbf_left, cbf_right]
    elif topic == "/rgb":
        msg = serialization.deserialize_message(data, Image)
        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        rgb_msgs.append((t, img))
    elif topic == "/gray_pub":
        msg = serialization.deserialize_message(data, Image)
        gray = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
        gray_msgs.append((t, gray))

print(f"Loaded {len(lane_msgs)} lane, {len(ctrl_msgs)} control, {len(rgb_msgs)} RGB, {len(gray_msgs)} gray messages.")

# --- Helper: find nearest message by timestamp ---
def find_nearest(target_t, msg_list):
    if not msg_list:
        return None
    times = [t for (t, _) in msg_list]
    idx = np.searchsorted(times, target_t)
    if idx == 0:
        return msg_list[0][1]
    if idx >= len(times):
        return msg_list[-1][1]
    before_t, before_val = msg_list[idx - 1]
    after_t, after_val = msg_list[idx]
    return before_val if abs(target_t - before_t) < abs(target_t - after_t) else after_val

# --- Synchronize by control timestamps ---
sync_rgb, sync_gray, sync_lane, sync_ctrl, sync_cbf = [], [], [], [], []
for t_ctrl, ctrl in ctrl_msgs:
    rgb = find_nearest(t_ctrl, rgb_msgs)
    gray = find_nearest(t_ctrl, gray_msgs)
    lane = find_nearest(t_ctrl, lane_msgs)
    cbf = find_nearest(t_ctrl, cbf_msgs)
    if rgb is not None and gray is not None and lane is not None and cbf is not None:
        sync_rgb.append(rgb)
        sync_gray.append(gray)
        sync_lane.append(lane)
        sync_ctrl.append(ctrl)
        sync_cbf.append(cbf)

n_frames = len(sync_ctrl)
print(f"Synchronized {n_frames} frames across all topics.")

# --- Figure layout ---
fig = plt.figure(figsize=(13, 8))
gs = fig.add_gridspec(2, 4, height_ratios=[1.5, 1]) #, width_ratios=[1., 1])

# Top row: images
ax_rgb  = fig.add_subplot(gs[1, 0])
ax_gray = fig.add_subplot(gs[1, 1])
ax_lane = fig.add_subplot(gs[1, 2])
ax_ctrl = fig.add_subplot(gs[1, 3])
ax_cbf  = fig.add_subplot(gs[0, :])

fig.tight_layout(pad=3.0)

# --- Images ---
pos = ax_rgb.get_position()
ax_rgb.set_position([pos.x0 - 0.035, pos.y0 - 0.07, pos.width * 1.4, pos.height * 1.4])
im_rgb = ax_rgb.imshow(sync_rgb[0])
ax_rgb.set_title("Raw Camera Feed")
ax_rgb.axis("off")

im_gray = ax_gray.imshow(sync_gray[0], cmap="gray", vmin=0, vmax=255)
ax_gray.set_title("Processed Frames")
ax_gray.axis("off")

# --- Lane localization plot (rotated view) ---
ax_lane.set_xlim(-0.1, 1.1)
ax_lane.set_ylim(-1, 1)
ax_lane.grid(True)
ax_lane.set_title("Lane Localization (0 = Left, 1 = Right)")
ax_lane.axvline(0, color='blue', linestyle='--', label='Left Lane')
ax_lane.axvline(1, color='green', linestyle='--', label='Right Lane')
lane_marker, = ax_lane.plot([], [], 'ro', markersize=8, label='Vehicle')
ax_lane.legend(loc="upper center")

# --- Control vector plot ---
ax_ctrl.set_xlim(-1.5, 1.5)
ax_ctrl.set_ylim(-1.5, 1.5)
ax_ctrl.set_aspect("equal")
ax_ctrl.grid(True)
ax_ctrl.set_title("Control Vector")
circle = plt.Circle((0, 0), 1.0, color='gray', fill=False, linestyle='--', alpha=0.5)
ax_ctrl.add_patch(circle)
quiv = ax_ctrl.quiver(0, 0, 0, 0, angles='xy', scale_units='xy', scale=1, color='tab:red')

# --- CBF plot ---
(line_cbfL,) = ax_cbf.plot([], [], label="CBF Left")
(line_cbfR,) = ax_cbf.plot([], [], label="CBF Right")
ax_cbf.legend()
ax_cbf.set_xlabel("Frame Index")
ax_cbf.set_ylabel("Value")
ax_cbf.set_title("Control Barrier Functions")
ax_cbf.grid(True)

# Buffers
cbf_x, cbfL_y, cbfR_y = [], [], []

# --- Update ---
def update(frame):
    im_rgb.set_data(sync_rgb[frame])
    im_gray.set_data(sync_gray[frame])
    # ax_rgb.set_title(f"RGB Image (Frame {frame+1}/{n_frames})")

    # Lane position marker
    lane_val = sync_lane[frame]
    lane_marker.set_data(lane_val, 0)
    
    # Control vector
    lin, ang = sync_ctrl[frame]
    lin_rot, ang_rot = -ang, lin
    quiv.set_UVC(lin_rot, ang_rot)

    # CBF
    cbf_x.append(frame)
    cbfL_y.append(sync_cbf[frame][0])
    cbfR_y.append(sync_cbf[frame][1])
    line_cbfL.set_data(cbf_x, cbfL_y)
    line_cbfR.set_data(cbf_x, cbfR_y)
    ax_cbf.relim(); ax_cbf.autoscale_view()

    return [im_rgb, im_gray, lane_marker, quiv, line_cbfL, line_cbfR]

# --- Animate (adjust speed here) ---
ani = FuncAnimation(fig, update, frames=n_frames, interval=1, blit=False)  # smaller interval = faster

# Display
# plt.show()

# Save
pbar = tqdm(total=n_frames, desc="Saving animation")

def update_progress(i, n):
    pbar.update(1)

# writer = FFMpegWriter(fps=30, codec="libx264", bitrate=1800)

writer = FFMpegWriter(
    fps=30,
    codec="libx264",
    bitrate=3000,
    extra_args=["-threads", str(os.cpu_count()), "-preset", "ultrafast", "-crf", "23", "-pix_fmt", "yuv420p"]
)

ani.save(
    "/home/ubuntu/ros_ws/experiment_bag/bag_video.mp4",
    writer=writer,
    dpi=300,
    progress_callback=update_progress
)

pbar.close()
print("Saved video successfully.")

# To speed up video 4x, use this: ffmpeg -i bag_video.mp4 -filter:v "setpts=0.25*PTS" -an bag_animation_4x.mp4
