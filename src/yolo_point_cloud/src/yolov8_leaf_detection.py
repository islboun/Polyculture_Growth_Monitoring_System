#!/usr/bin/python3

import rospy
from std_msgs.msg import String, Int32, Float32
import json
from pathlib import Path
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import numpy as np

class YOLOv8DetectionNode:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('leaf_detection_node', anonymous=False)

        # Subscribers for RGB path, segmentation mask, bounding box, and plant area topics
        rospy.Subscriber('rgb_path', String, self.rgb_path_callback)
        rospy.Subscriber('seg_mask', String, self.seg_mask_callback)
        rospy.Subscriber('bbox', String, self.bbox_callback)
        rospy.Subscriber('plant_area', Float32, self.plant_area_callback)

        # Publishers for bounding boxes and leaf count
        self.leaf_bbox_pub = rospy.Publisher('leaf_bboxes', String, queue_size=10)
        self.leaf_count_pub = rospy.Publisher('leaf_count', Int32, queue_size=10)
        self.avg_leaf_area_pub = rospy.Publisher('avg_leaf_area', Float32, queue_size=10)

        # Load the YOLO model once
        self.model = YOLO("src/yolo_point_cloud/yolo_weights/LeafDetectors/yolov8m_FT_LeafOuterDetect.pt")
        self.rgb_path = None
        self.seg_mask = None
        self.plant_bbox = None
        self.plant_area = None

    def rgb_path_callback(self, msg):
        self.rgb_path = msg.data.strip()
        rospy.loginfo(f"Received new RGB path: {self.rgb_path}")
        self.try_process_image()

    def bbox_callback(self, msg):
        # Ensure the bounding box is always a list of lists (even if it's a single box)
        self.plant_bbox = json.loads(msg.data)
        if isinstance(self.plant_bbox, list) and len(self.plant_bbox) == 4 and isinstance(self.plant_bbox[0], float):
            self.plant_bbox = [self.plant_bbox]  # Convert single bounding box to list of lists
        rospy.loginfo(f"Received Bounding Box: {self.plant_bbox}")
        self.try_process_image()

    def seg_mask_callback(self, msg):
        self.seg_mask = json.loads(msg.data)
        rospy.loginfo("Received new Segmentation Mask")
        self.try_process_image()

    def plant_area_callback(self, msg):
        self.plant_area = msg.data
        rospy.loginfo(f"Received new plant area: {self.plant_area}")
        self.try_process_image()

    def try_process_image(self):
        if self.rgb_path and self.seg_mask and self.plant_bbox and self.plant_area is not None:
            self.process_image()

    def process_image(self):
        image_path = Path(self.rgb_path)
        if not image_path.exists():
            rospy.logerr(f"Image path does not exist: {self.rgb_path}")
            return

        # Load the original RGB image
        img = cv2.imread(str(image_path))
        if img is None:
            rospy.logerr(f"Failed to load image {image_path}")
            return

        # Resize image to 40% of the original size
        img_resized = cv2.resize(img, None, fx=0.6, fy=0.6)

        ### First Image: Original Image
        original_img = img_resized.copy()

        ### Second Image: Plant Detection with Bounding Box and Segmentation Contour
        plant_det_img = img_resized.copy()

        # Draw bounding boxes on the original image
        if isinstance(self.plant_bbox, list):
            for box in self.plant_bbox:
                if len(box) == 4:
                    x1, y1, x2, y2 = [int(coord * 0.6) for coord in box]  # Scale coordinates
                    cv2.rectangle(plant_det_img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red bbox
                else:
                    rospy.logerr(f"Invalid bounding box format: {box}")
        else:
            rospy.logerr(f"Invalid bounding box data: {self.plant_bbox}")

        # Apply segmentation mask as a contour (no filled mask)
        if self.seg_mask:
            points = np.array(self.seg_mask, dtype=np.int32).reshape((-1, 2)) * 0.6  # Scale mask coordinates
            points = points.astype(np.int32)
            cv2.polylines(plant_det_img, [points], isClosed=True, color=(0, 0, 255), thickness=2)  # Red contour

        ### Third Image: Leaf Detection Result (from the second script you provided)
        img_third = cv2.imread(str(image_path))  # Load the original image for the third image (unresized)

        points = np.array(self.seg_mask, dtype=np.int32).reshape((-1, 2))
        mask_img = np.zeros_like(img_third)
        cv2.fillPoly(mask_img, [points], (255, 255, 255))
        segmented_img = cv2.bitwise_and(img_third, mask_img)

        results = self.model.predict(segmented_img, conf=0.6)
        annotator = Annotator(segmented_img, line_width=2)
        boxes = results[0].boxes.xyxy.cpu().tolist()

        for box in boxes:
            annotator.box_label(box, label=str(results[0].boxes.cls[boxes.index(box)].cpu().item()))
        annotated_img = annotator.result()

        # Replace all black (0,0,0) pixels in the annotated_img with white (255,255,255)
        annotated_img[np.all(annotated_img == [0, 0, 0], axis=-1)] = [255, 255, 255]

        # Resize all images to ensure they have the same dimensions before concatenation
        height, width = original_img.shape[:2]
        plant_det_img = cv2.resize(plant_det_img, (width, height))
        annotated_img = cv2.resize(annotated_img, (width, height))

        ### Concatenate the three images (original, plant detection & segmentation contour, leaf detection)
        concatenated_image = cv2.hconcat([original_img, plant_det_img, annotated_img])

        # Display the concatenated images in one window
        #cv2.imshow('Original + Plant Detection & Segmentation Contour + Leaf Detection', concatenated_image)

        # Wait for 2000ms (2 seconds) before updating the window with new images
        #cv2.waitKey(300)  # Refresh every 2 seconds

        ### Publish Results
        leaf_count = len(self.plant_bbox) if isinstance(self.plant_bbox, list) else 0
        self.leaf_count_pub.publish(leaf_count)

        leaf_bbox_msg = String(data=json.dumps(self.plant_bbox) if isinstance(self.plant_bbox, list) else "[]")
        self.leaf_bbox_pub.publish(leaf_bbox_msg)

        if leaf_count > 0:
            avg_leaf_area = self.plant_area / leaf_count
            self.avg_leaf_area_pub.publish(Float32(avg_leaf_area))

        # Reset values for the next image processing
        self.rgb_path = None
        self.seg_mask = None
        self.plant_bbox = None
        self.plant_area = None

if __name__ == '__main__':
    try:
        leaf_detection_node = YOLOv8DetectionNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
