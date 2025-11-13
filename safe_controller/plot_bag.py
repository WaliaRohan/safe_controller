#!/usr/bin/env python3
import rosbag2_py
from std_msgs.msg import Float32, Float32MultiArray
from sensor_msgs.msg import Image
import rclpy.serialization as serialization

bag_path = "experiment_bag"  # path to your bag folder

def main():
    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=bag_path, storage_id='mcap')
    converter_options = rosbag2_py.ConverterOptions('', '')
    reader.open(storage_options, converter_options)

    while reader.has_next():
        (topic, data, t) = reader.read_next()

        if topic == "/lane_position":
            msg = serialization.deserialize_message(data, Float32)
            print(f"[lane_position] {msg.data:.3f}")

        elif topic == "/control_stats":
            msg = serialization.deserialize_message(data, Float32MultiArray)
            print(f"[control_stats] {msg.data}")

        elif topic == "/rgb":
            msg = serialization.deserialize_message(data, Image)
            print(f"[rgb] Image {msg.width}x{msg.height}, encoding={msg.encoding}")

if __name__ == "__main__":
    main()
