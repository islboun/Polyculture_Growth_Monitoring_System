#!/usr/bin/python3

import sys
import os
import rospy
from std_msgs.msg import String, Int32
import json
from pathlib import Path
import cv2
import numpy as np
import torch

# Add YOLOv7 directory to Python path
yolo_v7_dir = "Add Yolov7 path"
sys.path.insert(0, yolo_v7_dir)

from models.experimental import attempt_load
from utils.general import non_max_suppression, scale_coords, set_logging
from utils.torch_utils import select_device
from utils.plots import plot_one_box

class YOLOv7LeafDetectionNode:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('yolov7_leaf_detection_node', anonymous=True)

        # Subscribers to SegMask and rgbPath topics
        rospy.Subscriber('seg_mask', String, self.segmask_callback)
        rospy.Subscriber('rgb_path', String, self.rgb_path_callback)

        # Publishers for LeafBBoxes and LeafCount topics
        self.leaf_bboxes_pub = rospy.Publisher('leaf_bboxes_v7', String, queue_size=10)
        self.leaf_count_pub = rospy.Publisher('leaf_count_v7', Int32, queue_size=10)

        # Load the YOLOv7 model once
        self.weights = "src/yolo_point_cloud/yolo_weights/LeafDetectors/yolov7-FT_LeafOuterDetect.pt"
        self.device = select_device('')
        self.model = attempt_load(self.weights, map_location=self.device)
        self.model.eval()
        self.img_size = 640  # Define the input image size

        self.segmask = None
        self.rgb_path = None

    def segmask_callback(self, msg):
        self.segmask = json.loads(msg.data)
        self.process_image()

    def rgb_path_callback(self, msg):
        self.rgb_path = msg.data
        self.process_image()

    def process_image(self):
        if self.segmask is None or self.rgb_path is None:
            return

        image_path = Path(self.rgb_path)
        if not image_path.exists():
            rospy.logerr(f"Image path does not exist: {self.rgb_path}")
            return

        img = cv2.imread(str(image_path))
        if img is None:
            rospy.logerr(f"Failed to load image {image_path}")
            return

        # Convert segmask to a list of points
        points = np.array(self.segmask, dtype=np.int32).reshape((-1, 2))

        # Create a mask image with the same resolution as the original image
        mask_img = np.zeros_like(img)

        # Draw the polygon on the mask
        cv2.fillPoly(mask_img, [points], (255, 255, 255))

        # Apply the mask to the original image to retain the segmented region and black out the rest
        segmented_img = cv2.bitwise_and(img, mask_img)

        # Run YOLOv7 detection on the segmented image
        detections = self.detect_leaves(segmented_img)

        # Publish the detection results
        leaf_count = len(detections)
        self.leaf_count_pub.publish(leaf_count)
        
        if leaf_count > 0:
            bboxes_list = [[xyxy[0], xyxy[1], xyxy[2], xyxy[3]] for *xyxy, conf, cls in detections]
            bboxes_msg = json.dumps(bboxes_list)
            self.leaf_bboxes_pub.publish(bboxes_msg)

        rospy.loginfo(f"Number of leaves detected: {leaf_count}")

        # Draw bounding boxes on the segmented image
        for *xyxy, conf, cls in detections:
            label = f'leaf {conf:.2f}'
            plot_one_box(xyxy, segmented_img, label=label, color=(255, 0, 0), line_thickness=2)
        

        # Reset state
        self.segmask = None
        self.rgb_path = None

    def detect_leaves(self, img):
        # Resize image
        img_resized = cv2.resize(img, (self.img_size, self.img_size))
        img_resized = img_resized[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img_resized = np.ascontiguousarray(img_resized)
        img_resized = torch.from_numpy(img_resized).to(self.device)
        img_resized = img_resized.float()  # uint8 to fp16/32
        img_resized /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img_resized.ndimension() == 3:
            img_resized = img_resized.unsqueeze(0)

        # Inference
        with torch.no_grad():
            pred = self.model(img_resized)[0]
        pred = non_max_suppression(pred, 0.5, 0.45)

        # Process detections
        detections = []
        for det in pred:
            if len(det):
                # Rescale boxes from img_size to original image size
                det[:, :4] = scale_coords(img_resized.shape[2:], det[:, :4], img.shape).round()
                for *xyxy, conf, cls in det:
                    detections.append([xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item(), conf.item(), cls.item()])
        return detections

if __name__ == '__main__':
    try:
        os.chdir(yolo_v7_dir)  # Ensure the working directory is set to YOLOv7 directory for model dependencies
        set_logging()
        yolo_node = YOLOv7LeafDetectionNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

