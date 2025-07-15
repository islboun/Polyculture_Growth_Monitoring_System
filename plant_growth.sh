source devel/setup.bash

catkin_make

source devel/setup.bash


export DISPLAY=:0

rosnode kill -a
killall -9 rosmaster 

roscore &

sleep 5

gnome-terminal --tab --title="Dummy ImageCaptured Node" -- bash -c 'source devel/setup.bash; rosrun yolo_point_cloud imageCaptured_node'

#sleep 5

#gnome-terminal --tab --title="Yolov8 Node" -- bash -c 'source devel/setup.bash; rosrun yolo_point_cloud imageCapturedv2.py'

sleep 5

gnome-terminal --tab --title="Yolov8 Segmentation" -- bash -c 'source devel/setup.bash; rosrun yolo_point_cloud yolov8_segmentation_node.py'

#sleep 5

#gnome-terminal --tab --title="Yolov7 Leaf Detection" -- bash -c 'source devel/setup.bash; rosrun yolo_point_cloud yolov7_leaf_detection.py'

sleep 5

gnome-terminal --tab --title="Yolov8 Leaf Detection" -- bash -c 'source devel/setup.bash; rosrun yolo_point_cloud yolov8_leaf_detection.py'

sleep 5

#Visualizers. 
gnome-terminal --tab --title="Point Cloud Measurements" -- bash -c 'source devel/setup.bash; rosrun yolo_point_cloud pcl_measurements_node'

sleep 5

gnome-terminal --tab --title="Point Cloud Measurements" -- bash -c 'source devel/setup.bash; rosrun yolo_point_cloud measurements_node'

sleep 5

gnome-terminal --tab --title="Store Features" -- bash -c 'source devel/setup.bash; rosrun yolo_point_cloud store_features_node'

#sleep 5


#gnome-terminal --tab --title="ImageCaptured Node Trigger" -- bash -c 'source devel/setup.bash; rostopic pub /image_captured std_msgs/Bool "data: true"'


