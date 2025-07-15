#include <ros/ros.h>
#include <thread>
#include <std_msgs/String.h>
#include <std_msgs/Float32.h>
#include <opencv2/opencv.hpp>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/common_headers.h>
#include <pcl/point_cloud.h>
#include <json/json.h>

class PCLMeasurementsNode
{
public:
    PCLMeasurementsNode()
    {
        rgb_path_sub_ = nh_.subscribe("rgb_path", 10, &PCLMeasurementsNode::rgbPathCallback, this);
        depth_path_sub_ = nh_.subscribe("depth_path", 10, &PCLMeasurementsNode::depthPathCallback, this);
        segmask_sub_ = nh_.subscribe("seg_mask", 10, &PCLMeasurementsNode::segmaskCallback, this);
        bbox_sub_ = nh_.subscribe("bbox", 10, &PCLMeasurementsNode::bboxCallback, this);

        plant_width_pub_ = nh_.advertise<std_msgs::Float32>("plant_width_pcl", 10);
        plant_length_pub_ = nh_.advertise<std_msgs::Float32>("plant_length_pcl", 10);
        plant_height_pub_ = nh_.advertise<std_msgs::Float32>("plant_height_pcl", 10);

        std::cout << "PCLMeasurementsNode initialized." << std::endl;
    }

    void rgbPathCallback(const std_msgs::String::ConstPtr& msg)
    {
        rgb_path_ = msg->data;
        std::cout << "Received RGB path: " << rgb_path_ << std::endl;
        processImages();
    }

    void depthPathCallback(const std_msgs::String::ConstPtr& msg)
    {
        depth_path_ = msg->data;
        std::cout << "Received depth path: " << depth_path_ << std::endl;
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
        std::cout << "Received SegMask with " << segmask_.size() << " points." << std::endl;
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
        std::cout << "Received BBox: " << bbox_ << std::endl;
        processImages();
    }

private:

    void applyClosing(cv::Mat& image, int kernelSize = 5)
    {
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

        std::cout << "Processing images with:\nRGB Path: " << rgb_path_ << "\nDepth Path: " << depth_path_ << std::endl;

        cv::Mat rgb_image = cv::imread(rgb_path_, cv::IMREAD_COLOR);
        cv::Mat depth_image = cv::imread(depth_path_, cv::IMREAD_UNCHANGED);

        if (rgb_image.empty() || depth_image.empty())
        {
            ROS_ERROR("Failed to load images");
            return;
        }

        std::cout << "Loaded images successfully." << std::endl;

        applyClosing(depth_image);

        // Undistort points
        std::vector<cv::Point2f> imagePoints;
        for (int i = 0; i < depth_image.rows; i++)
        {
            for (int j = 0; j < depth_image.cols; j++)
            {
                imagePoints.push_back(cv::Point2f(j, i));
            }
        }

        std::vector<cv::Point2f> undistortedPoints;
        cv::Mat cameraMatrix = (cv::Mat_<double>(3, 3) << fx_, 0, cx_, 0, fy_, cy_, 0, 0, 1);
        cv::Mat distCoeffs = (cv::Mat_<double>(5, 1) << k1, k2, p1, p2, k3);
        cv::undistortPoints(imagePoints, undistortedPoints, cameraMatrix, distCoeffs, cv::noArray(), cameraMatrix);

        createPointCloudFromBBox(rgb_image, depth_image, undistortedPoints);
        createPointCloudFromSegMask(rgb_image, depth_image, undistortedPoints);

        rgb_path_.clear();
        depth_path_.clear();
        segmask_.clear();
        bbox_ = cv::Rect();
    }

    void calculateDimensionsAndVisualize(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud, pcl::visualization::PCLVisualizer::Ptr& viewer)
    {
        float min_x = std::numeric_limits<float>::max(), max_x = -std::numeric_limits<float>::max();
        float min_y = std::numeric_limits<float>::max(), max_y = -std::numeric_limits<float>::max();
        float min_z = std::numeric_limits<float>::max(), max_z = std::numeric_limits<float>::min();

        for (const auto& point : cloud->points)
        {
            if (point.x < min_x) min_x = point.x;
            if (point.x > max_x) max_x = point.x;
            if (point.y < min_y) min_y = point.y;
            if (point.y > max_y) max_y = point.y;
            if (point.z < min_z) min_z = point.z;
            if (point.z > max_z) max_z = point.z;
        }

        pcl::PointXYZ minPt(min_x, min_y, min_z);
        pcl::PointXYZ maxPt(max_x, max_y, max_z);
        viewer->addLine(minPt, pcl::PointXYZ(max_x, min_y, min_z), 1, 0, 0, "line_x");
        viewer->addLine(minPt, pcl::PointXYZ(min_x, max_y, min_z), 0, 1, 0, "line_y");
        viewer->addLine(minPt, pcl::PointXYZ(min_x, min_y, max_z), 0, 0, 1, "line_z");
    }

    // Update for BBox viewer
    void createPointCloudFromBBox(const cv::Mat& rgb_image, const cv::Mat& depth_image, const std::vector<cv::Point2f>& undistortedPoints)
    {
        std::cout << "Creating point cloud from BBox." << std::endl;

        cv::Mat bbox_mask = cv::Mat::zeros(rgb_image.size(), CV_8UC1);
        cv::rectangle(bbox_mask, bbox_, cv::Scalar(255), cv::FILLED);

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        for (int y = 0; y < rgb_image.rows; ++y)
        {
            for (int x = 0; x < rgb_image.cols; ++x)
            {
                if (bbox_mask.at<uchar>(y, x) == 255)
                {
                    float depth = depth_image.at<float>(y, x);
                    if (depth > 0.55f || depth < 0.3f || std::isinf(depth) || std::isnan(depth))
                    {
                        depth = 0;
                    }
                    if (depth != 0)
                    {
                        pcl::PointXYZRGB point;
                        point.z = depth;
                        int idx = y * rgb_image.cols + x;
                        point.x = (undistortedPoints[idx].x - cx_) * depth / fx_;
                        point.y = (undistortedPoints[idx].y - cy_) * depth / fy_;
                        cv::Vec3b color = rgb_image.at<cv::Vec3b>(y, x);
                        uint32_t rgb = (static_cast<uint32_t>(color[2]) << 16 |
                                        static_cast<uint32_t>(color[1]) << 8 |
                                        static_cast<uint32_t>(color[0]));
                        point.rgb = *reinterpret_cast<float*>(&rgb);
                        if (!(color[0] == 0 && color[1] == 0 && color[2] == 0))
                        {
                            cloud->points.push_back(point);
                        }
                    }
                }
            }
        }

        std::cout << "Generated point cloud from BBox with " << cloud->points.size() << " points." << std::endl;

        static pcl::visualization::PCLVisualizer::Ptr viewerBBox(new pcl::visualization::PCLVisualizer("BBox Viewer"));
        viewerBBox->setBackgroundColor(1.0, 1.0, 1.0);  // Set background to white

        // Remove previous point cloud if it exists
        if (viewerBBox->contains("bbox cloud"))
        {
            viewerBBox->removePointCloud("bbox cloud");
        }

        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
        viewerBBox->addPointCloud<pcl::PointXYZRGB>(cloud, rgb, "bbox cloud");

        // Set the point size and custom camera view
        viewerBBox->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "bbox cloud");
        viewerBBox->setCameraPosition(0.15, -0.1, 1.2, 0, 0, 0, 0, 1, 0);

        // Update dimensions and visualize
        calculateDimensionsAndVisualize(cloud, viewerBBox);

        // Resize the viewer window
        viewerBBox->setSize(480, 360); // Set window size

        viewerBBox->spinOnce(100);
    }

    // Segmentation mask point cloud viewer (no change)
    void createPointCloudFromSegMask(const cv::Mat& rgb_image, const cv::Mat& depth_image, const std::vector<cv::Point2f>& undistortedPoints)
    {
        std::cout << "Creating point cloud from SegMask." << std::endl;

        cv::Mat mask = cv::Mat::zeros(rgb_image.size(), CV_8UC1);
        std::vector<std::vector<cv::Point>> contours = {segmask_};
        cv::fillPoly(mask, contours, cv::Scalar(255));

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        for (int y = 0; y < rgb_image.rows; ++y)
        {
            for (int x = 0; x < rgb_image.cols; ++x)
            {
                if (mask.at<uchar>(y, x) == 255)
                {
                    float depth = depth_image.at<float>(y, x);
                    if (depth > 0.6f || depth < 0.3f || std::isinf(depth) || std::isnan(depth))
                    {
                        depth = 0;
                    }
                    if (depth != 0)
                    {
                        pcl::PointXYZRGB point;
                        point.z = depth;
                        int idx = y * rgb_image.cols + x;
                        point.x = (undistortedPoints[idx].x - cx_) * depth / fx_;
                        point.y = (undistortedPoints[idx].y - cy_) * depth / fy_;
                        cv::Vec3b color = rgb_image.at<cv::Vec3b>(y, x);
                        uint32_t rgb = (static_cast<uint32_t>(color[2]) << 16 |
                                        static_cast<uint32_t>(color[1]) << 8 |
                                        static_cast<uint32_t>(color[0]));
                        point.rgb = *reinterpret_cast<float*>(&rgb);
                        if (!(color[0] == 0 && color[1] == 0 && color[2] == 0))
                        {
                            cloud->points.push_back(point);
                        }
                    }
                }
            }
        }

        std::cout << "Generated point cloud from SegMask with " << cloud->points.size() << " points." << std::endl;

        static pcl::visualization::PCLVisualizer::Ptr viewerSegMask(new pcl::visualization::PCLVisualizer("SegMask Viewer"));
        viewerSegMask->setBackgroundColor(1.0, 1.0, 1.0);

        // Remove previous point cloud if it exists
        if (viewerSegMask->contains("seg cloud"))
        {
            viewerSegMask->removePointCloud("seg cloud");
        }

        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
        viewerSegMask->addPointCloud<pcl::PointXYZRGB>(cloud, rgb, "seg cloud");

        viewerSegMask->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "seg cloud");
        viewerSegMask->setCameraPosition(0.15, -0.1, 1.2, 0, 0, 0, 0, 1, 0);

        calculateDimensionsAndVisualize(cloud, viewerSegMask);

        viewerSegMask->setSize(480, 360); // Set window size

        viewerSegMask->spinOnce(100);
    }

    ros::NodeHandle nh_;
    ros::Subscriber rgb_path_sub_;
    ros::Subscriber depth_path_sub_;
    ros::Subscriber segmask_sub_;
    ros::Subscriber bbox_sub_;

    ros::Publisher plant_width_pub_;
    ros::Publisher plant_length_pub_;
    ros::Publisher plant_height_pub_;

    std::string rgb_path_;
    std::string depth_path_;
    std::vector<cv::Point> segmask_;
    cv::Rect bbox_;

    const float fx_ = 528.272, fy_ = 528.272, cx_ = 489.542, cy_ = 267.570;
    float k1 = -0.0361431, k2 = 0.00486255, p1 = -0.000249107, p2 = 4.91599e-06, k3 = -0.00301238;
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "pcl_measurements_node");
    PCLMeasurementsNode node;
    ros::spin();
    return 0;
}
