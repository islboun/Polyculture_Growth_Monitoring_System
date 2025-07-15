#!/usr/bin/python3

import rospy
from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray
import json
from pathlib import Path
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import numpy as np


class YOLOv8SegmentationNode:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('plant_segmentation_node', anonymous=False)

        # Subscriber to RGB path topics
        rospy.Subscriber('rgb_path', String, self.rgb_path_callback)

        # Publishers for BBox and SegMask topics
        self.bbox_pub = rospy.Publisher('bbox', String, queue_size=10)
        self.segmask_pub = rospy.Publisher('seg_mask', String, queue_size=10)

        # Load the YOLO model once
        self.model = YOLO("src/yolo_point_cloud/yolo_weights/PlantSegmentation/yolov8m-seg_JuneF.pt")
        self.rgb_path = None

    def rgb_path_callback(self, msg):
        self.rgb_path = msg.data
        self.process_image()
        # Reset state
        self.rgb_path = None

    def process_image(self):
        if self.rgb_path is None:
            return

        image_path = Path(self.rgb_path)
        if not image_path.exists():
            rospy.logerr(f"Image path does not exist: {self.rgb_path}")
            return

        img = cv2.imread(str(image_path))
        if img is None:
            rospy.logerr(f"Failed to load image {image_path}")
            return

        # Predict using the loaded model
        results = self.model.predict(img, conf=0.6)
        annotator = Annotator(img, line_width=2)

        # Show the most centered box
        height, width, _ = img.shape
        image_center = np.array([width / 2, height / 2])

        if results[0].masks is not None:
            clss = results[0].boxes.cls.cpu().tolist()
            masks = results[0].masks.xy
            boxes = results[0].boxes.xyxy.cpu().tolist()

            # Calculate the center of each box and find the most centered one
            min_distance = float('inf')
            most_centered_idx = None
            for i, box in enumerate(boxes):
                box_center = np.array([(box[2] + box[0]) / 2, (box[3] + box[1]) / 2])
                distance = np.linalg.norm(box_center - image_center)
                if distance < min_distance:
                    min_distance = distance
                    most_centered_idx = i

            if most_centered_idx is not None:
                mask = masks[most_centered_idx]
                box = boxes[most_centered_idx]
                cls = clss[most_centered_idx]
                color = colors(int(cls), True)
                annotator.seg_bbox(mask=mask, mask_color=color)
                annotator.box_label(box=box, color=color)
                rospy.loginfo(f"This is the box = {box}")
                #rospy.loginfo(f"This is the mask = {mask}")
                size_mask = len(mask)
                rospy.loginfo(f"Size of the mask is {size_mask}")

                # Publish the box coordinates
                bbox_msg = String(data=json.dumps(box))
                self.bbox_pub.publish(bbox_msg)

                # Publish the mask coordinates as JSON string
                mask_list = mask.tolist()
                segmask_msg = String(data=json.dumps(mask_list))
                self.segmask_pub.publish(segmask_msg)

            # Display the annotated image
            #cv2.imshow("Instance Segmentation", img)
            #if cv2.waitKey(1000):  # Press ESC to close the window
            #  cv2.destroyAllWindows()

if __name__ == '__main__':
    try:
        plant_segmentation_node = YOLOv8SegmentationNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    cv2.destroyAllWindows()

