#include <ros/ros.h>
#include <std_msgs/String.h>
#include <std_msgs/Float32.h>
#include <opencv2/opencv.hpp>
#include <json/json.h>
#include <limits>

class MeasurementsNode
{
public:
    MeasurementsNode()
    {
        rgb_path_sub_ = nh_.subscribe("rgb_path", 10, &MeasurementsNode::rgbPathCallback, this);
        depth_path_sub_ = nh_.subscribe("depth_path", 10, &MeasurementsNode::depthPathCallback, this);
        segmask_sub_ = nh_.subscribe("seg_mask", 10, &MeasurementsNode::segmaskCallback, this);
        bbox_sub_ = nh_.subscribe("bbox", 10, &MeasurementsNode::bboxCallback, this);

        plant_width_pub_ = nh_.advertise<std_msgs::Float32>("plant_width", 10);
        plant_length_pub_ = nh_.advertise<std_msgs::Float32>("plant_length", 10);
        plant_height_pub_ = nh_.advertise<std_msgs::Float32>("plant_height", 10);
        plant_area_pub_ = nh_.advertise<std_msgs::Float32>("plant_area", 10);

        std::cout << "MeasurementsNode initialized." << std::endl;
    }

    void rgbPathCallback(const std_msgs::String::ConstPtr& msg)
    {
        rgb_path_ = msg->data;
        processImages();
    }

    void depthPathCallback(const std_msgs::String::ConstPtr& msg)
    {
        depth_path_ = msg->data;
        processImages();
    }

    void segmaskCallback(const std_msgs::String::ConstPtr& msg)
    {
        Json::CharReaderBuilder reader;
        std::string errs;
        std::istringstream s(msg->data);
        Json::Value root;
        if (!Json::parseFromStream(reader, s, &root, &errs))
        {
            ROS_ERROR("Failed to parse SegMask JSON: %s", errs.c_str());
            return;
        }

        segmask_.clear();
        for (const auto& point : root)
        {
            segmask_.emplace_back(cv::Point(point[0].asInt(), point[1].asInt()));
        }
        processImages();
    }

    void bboxCallback(const std_msgs::String::ConstPtr& msg)
    {
        Json::CharReaderBuilder reader;
        std::string errs;
        std::istringstream s(msg->data);
        Json::Value root;
        if (!Json::parseFromStream(reader, s, &root, &errs))
        {
            ROS_ERROR("Failed to parse BBox JSON: %s", errs.c_str());
            return;
        }

        bbox_.x = root[0].asInt();
        bbox_.y = root[1].asInt();
        bbox_.width = root[2].asInt() - root[0].asInt();
        bbox_.height = root[3].asInt() - root[1].asInt();
        processImages();
    }

private:

    // Closing Function
    void applyClosing(cv::Mat& image, int kernelSize = 5) {
        cv::Mat element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(kernelSize, kernelSize));
        cv::morphologyEx(image, image, cv::MORPH_CLOSE, element);
    }

    void processImages()
    {
        if (rgb_path_.empty() || depth_path_.empty() || segmask_.empty() || bbox_.width == 0 || bbox_.height == 0)
        {
            std::cout << "Waiting for all inputs..." << std::endl;
            return;
        }

        cv::Mat rgb_image = cv::imread(rgb_path_, cv::IMREAD_COLOR);
        cv::Mat depth_image = cv::imread(depth_path_, cv::IMREAD_UNCHANGED);

        // Apply closing to depth image
        applyClosing(depth_image);

        if (rgb_image.empty() || depth_image.empty())
        {
            ROS_ERROR("Failed to load images");
            return;
        }

        // Undistort points
        std::vector<cv::Point2f> imagePoints;
        for (int i = 0; i < depth_image.rows; i++) {
            for (int j = 0; j < depth_image.cols; j++) {
                imagePoints.push_back(cv::Point2f(j, i));
            }
        }

        std::vector<cv::Point2f> undistortedPoints;
        cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << fx_, 0, cx_, 0, fy_, cy_, 0, 0, 1);
        cv::Mat distCoeffs = (cv::Mat_<double>(5, 1) << k1, k2, p1, p2, k3);
        cv::undistortPoints(imagePoints, undistortedPoints, cameraMatrix, distCoeffs, cv::noArray(), cameraMatrix);

        double ratio = visualizeSegmentation(rgb_image);
        measureDimensions(undistortedPoints, depth_image, ratio);

        // Reset the paths and flags
        rgb_path_.clear();
        depth_path_.clear();
        segmask_.clear();
        bbox_ = cv::Rect();
    }

    double visualizeSegmentation(const cv::Mat& rgb_image)
    {
        if (segmask_.empty()) {
            std::cerr << "Segmentation mask is empty." << std::endl;
            return 0.0; // Return a default value indicating an error or inappropriate condition
        }



        // Create a mask image with the same resolution as the original image
        cv::Mat mask(rgb_image.size(), CV_8UC1, cv::Scalar(0));

        // Convert segmask_ vector to a vector of cv::Point
        std::vector<cv::Point> contour;
        for (const auto& point : segmask_) {
            contour.push_back(cv::Point(point.x, point.y));
        }

        // Draw and fill the polygon on the mask
        std::vector<std::vector<cv::Point>> contours = {contour};
        cv::fillPoly(mask, contours, cv::Scalar(255));

        // Apply the mask to the original image to retain the segmented region
        cv::Mat visualized;
        rgb_image.copyTo(visualized, mask);

        // Crop the visualized image using the bounding box coordinates
        cv::Rect bbox(cv::Point(bbox_.x, bbox_.y), cv::Size(bbox_.width, bbox_.height));
        cv::Mat croppedVisualized = visualized(bbox);


        cv::Mat gray;
        cv::cvtColor(croppedVisualized, gray, cv::COLOR_BGR2GRAY);

        double nonblack_pixels = cv::countNonZero(gray);
        double total_pixels = croppedVisualized.rows * croppedVisualized.cols;
        double ratio = nonblack_pixels / total_pixels;

        std::cout << "bbox_width = " << bbox_.width << " bbox_height = " << bbox_.height <<  " Non-black pixels = " << nonblack_pixels << ", Total pixels in bbox = " << total_pixels << ", Ratio = " << ratio << std::endl;

        return ratio; // Return the computed ratio
    }

    void measureDimensions(const std::vector<cv::Point2f>& undistortedPoints, const cv::Mat& depth_image, double ratio)
    {
        float maxReasonableDepth = 0.55f;
        float minMeasurableDepth = 0.3f;

        // Measure width and length from segmentation mask
        float min_x = std::numeric_limits<float>::max(), max_x = -std::numeric_limits<float>::max();
        float min_y = std::numeric_limits<float>::max(), max_y = -std::numeric_limits<float>::max();

        for (const auto& point : segmask_)
        {
            float depth = depth_image.at<float>(point.y, point.x);

            if (depth > maxReasonableDepth || depth < minMeasurableDepth || std::isinf(depth) || std::isnan(depth)) {
                depth = 0;
            }
            if (depth != 0) {
                int idx = point.y * depth_image.cols + point.x;
                float real_x = (undistortedPoints[idx].x - cx_) * depth / fx_;
                float real_y = (undistortedPoints[idx].y - cy_) * depth / fy_;

                if (real_x < min_x) min_x = real_x;
                if (real_x > max_x) max_x = real_x;
                if (real_y < min_y) min_y = real_y;
                if (real_y > max_y) max_y = real_y;
            }
        }

        std::cout << "max_x = " << max_x << " min_x = " << min_x << " max_y = " << max_y << " min_y = " << min_y << "\n";

        if (min_x == std::numeric_limits<float>::max() || max_x == -std::numeric_limits<float>::max() ||
            min_y == std::numeric_limits<float>::max() || max_y == -std::numeric_limits<float>::max()) {
            ROS_ERROR("No valid depth data found in the segmented area");
            return;
        }

        float width = max_x - min_x;
        float length = max_y - min_y;

        std::cout << "Calculated Width: " << width << ", Length: " << length << std::endl;

        // Measure height from bounding box using depth image
        float min_z = std::numeric_limits<float>::max();
        float max_z = -std::numeric_limits<float>::max();
        // Measure width and length from bounding box
        float min_x_bb = std::numeric_limits<float>::max(), max_x_bb = -std::numeric_limits<float>::max();
        float min_y_bb = std::numeric_limits<float>::max(), max_y_bb = -std::numeric_limits<float>::max();

         for (int y = bbox_.y; y < bbox_.y + bbox_.height; ++y)
        {
            for (int x = bbox_.x; x < bbox_.x + bbox_.width; ++x)
            {
                float depth = depth_image.at<float>(y, x);
                if (depth > maxReasonableDepth || depth < minMeasurableDepth || std::isinf(depth) || std::isnan(depth)) {
                    depth = 0;
                }
                if (depth != 0) {
                    int idx = y * depth_image.cols + x;
                    float real_x = (undistortedPoints[idx].x - cx_) * depth / fx_;
                    float real_y = (undistortedPoints[idx].y - cy_) * depth / fy_;

                    if (real_x < min_x_bb) min_x_bb = real_x;
                    if (real_x > max_x_bb) max_x_bb = real_x;
                    if (real_y < min_y_bb) min_y_bb = real_y;
                    if (real_y > max_y_bb) max_y_bb = real_y;

                    if (depth < min_z) min_z = depth;
                    if (depth > max_z) max_z = depth;
                }
            }
        }

        if (min_z == std::numeric_limits<float>::max() || max_z == -std::numeric_limits<float>::max()) {
            ROS_ERROR("No valid depth data found in the bounding box area");
            return;
        }
        std::cout << "max_z = " << max_z << " min_z = " << min_z << std::endl;

        float height = max_z - min_z;

        std::cout << "Calculated Height: " << height << std::endl;
        double plant_area = ratio * (max_x_bb-min_x_bb) * (max_y_bb-min_y_bb);

        // Publish measurements
        std_msgs::Float32 width_msg;
        width_msg.data = width;
        plant_width_pub_.publish(width_msg);

        std_msgs::Float32 length_msg;
        length_msg.data = length;
        plant_length_pub_.publish(length_msg);

        std_msgs::Float32 height_msg;
        height_msg.data = height;
        plant_height_pub_.publish(height_msg);

        std_msgs::Float32 area_msg;
        area_msg.data = plant_area;
        plant_area_pub_.publish(area_msg);

        std::cout << "Published plant dimensions - Width: " << width << ", Length: " << length << ", Height: " << height << ", Area: " << plant_area  << std::endl;
    }

    ros::NodeHandle nh_;
    ros::Subscriber rgb_path_sub_;
    ros::Subscriber depth_path_sub_;
    ros::Subscriber segmask_sub_;
    ros::Subscriber bbox_sub_;

    ros::Publisher plant_width_pub_;
    ros::Publisher plant_length_pub_;
    ros::Publisher plant_height_pub_;
    ros::Publisher plant_area_pub_;

    std::string rgb_path_;
    std::string depth_path_;
    std::vector<cv::Point> segmask_;
    cv::Rect bbox_;

    const float fx_ = 528.272, fy_ = 528.272, cx_ = 489.542, cy_ = 267.570;
    float k1 = -0.0361431, k2 = 0.00486255, p1 = -0.000249107, p2 = 4.91599e-06, k3 = -0.00301238;
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "measurements_node");
    MeasurementsNode node;
    ros::spin();
    return 0;
}
