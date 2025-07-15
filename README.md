This repository contains the ROS package source codes for the full structural phenotyping pipeline and the trained YOLO models.

There are sample input images under the input_sample folder.

You can use plant_growth.sh to run the pipeline.

imageCaptured_node is a dummy version of imageCapturedv2 (for Jetson). You can trigger it with:

rostopic pub /image_captured std_msgs/Bool "data: true"

You can also find this command inside the bash file.

There are two files for measurement: pcl_measurements_node and measurements_node.
The first one (pcl_measurements_node) shows PCL versions of the outputs.
The automated pipeline only measures features without viewing PCLs (using measurements_node).
