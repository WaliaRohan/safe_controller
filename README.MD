# ROS Package Overview

This package contains the following ROS nodes:

## Nodes

### teleop
Implements a nominal controller using keyboard teleop. It publishes `cmd_vel` data to a nominal control topic (`/nominal_control` by default).

### control
Filters the control commands from `teleop` using a Belief CBF. It relies on odometry to update system state. It also relies on the `safety` script (`safety` is NOT a ROS node, but implements the minimally invasive CBF QP).

**Default Topics:**
- `TOPIC_NOM_CTRL = "/nominal_control"`
- `TOPIC_SAFE_CTRL = "/cmd_vel"`
- `TOPIC_ODOM = "/vicon_pose"`

### control_lane
Filters the control commands from a `teleop` topic using a Belief CBF. It updates system state based on lane position values. Meant to be used with DeepRacer experiments to drive it around a track. It also relies on the `safety` script.

**Default Topics:**
- `TOPIC_NOM_CTRL = "/nominal_control"`
- `TOPIC_SAFE_CTRL = "/cmd_vel"`
- `TOPIC_ODOM = "/vicon_pose"`
- `TOPIC_LANE_POSE = "/lane_position"`

## Scripts

### safety
Implements minimally invasive Belief CBF QP. **Note:** This is a script, not a ROS node.