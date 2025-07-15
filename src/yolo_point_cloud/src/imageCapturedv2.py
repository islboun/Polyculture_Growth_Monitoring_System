#!/usr/bin/python3

import rospy
from std_msgs.msg import String
import os
import subprocess

class ImageCapturedNode:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('image_captured_node', anonymous=True)

        # Directory where images will be saved locally after SCP
        self.image_directory = 'plantImages'

        # Publishers for local RGB and depth paths after SCP
        self.rgb_path_pub = rospy.Publisher('rgb_path', String, queue_size=10)
        self.depth_path_pub = rospy.Publisher('depth_path', String, queue_size=10)

        # Subscribers to the topics that provide full remote image paths
        self.rgb_path_sub = rospy.Subscriber('rgbFullPath', String, self.rgb_callback)
        self.depth_path_sub = rospy.Subscriber('depthFullPath', String, self.depth_callback)

    def rgb_callback(self, msg):
        # Callback for receiving the full RGB image path
        rgb_remote_path = msg.data
        rospy.loginfo(f"Received remote RGB image path: {rgb_remote_path}")

        # Copy the RGB image from the remote path using SCP
        #local_rgb_path = self.get_image_via_scp(rgb_remote_path)
        #if local_rgb_path:
            # Publish the local RGB image path after SCP
        self.publish_image_path(rgb_remote_path, self.rgb_path_pub)

    def depth_callback(self, msg):
        # Callback for receiving the full Depth image path
        depth_remote_path = msg.data
        rospy.loginfo(f"Received remote Depth image path: {depth_remote_path}")

        # Copy the Depth image from the remote path using SCP
        #local_depth_path = self.get_image_via_scp(depth_remote_path)
        #if local_depth_path:
            # Publish the local Depth image path after SCP
        self.publish_image_path(depth_remote_path, self.depth_path_pub)

    def get_image_via_scp(self, remote_path):
        # Extract the filename from the remote path
        filename = os.path.basename(remote_path)
        local_path = os.path.join(self.image_directory, filename)

        # Execute the SCP command to retrieve the image from the remote computer
        scp_command = f"scp jetson@10.42.0.1:{remote_path} {local_path}"
        try:
            subprocess.run(scp_command, shell=True, check=True)
            rospy.loginfo(f"SCP transfer completed for: {local_path}")
            return local_path
        except subprocess.CalledProcessError as e:
            rospy.logerr(f"SCP command failed with error: {e}")
            return None

    def publish_image_path(self, file_path, publisher):
        # Publish the local image path
        path_msg = String()
        path_msg.data = file_path
        publisher.publish(path_msg)
        rospy.loginfo(f"Published local image path: {file_path}")

if __name__ == '__main__':
    try:
        # Start the node
        ImageCapturedNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
