#include <ros/ros.h>
#include <std_msgs/Bool.h>
#include <std_msgs/String.h>
#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>

class ImageCapturedNode {
public:
    ImageCapturedNode() {
        ros::NodeHandle nh;
        rgb_path_pub_ = nh.advertise<std_msgs::String>("rgb_path", 10);
        depth_path_pub_ = nh.advertise<std_msgs::String>("depth_path", 10);
        image_captured_sub_ = nh.subscribe("image_captured", 10, &ImageCapturedNode::imageCapturedCallback, this);
        image_directory_ = "Add your path here/input_sample";// Change this string with your input images directory
        loadFiles();
    }

    void loadFiles() {
        namespace fs = boost::filesystem;
        fs::path directory(image_directory_);
        if (fs::exists(directory) && fs::is_directory(directory)) {
            fs::directory_iterator end_iter;
            for (fs::directory_iterator dir_iter(directory); dir_iter != end_iter; ++dir_iter) {
                if (fs::is_regular_file(dir_iter->status())) {
                    std::string path = dir_iter->path().string();
                    if (boost::filesystem::extension(path) == ".png") {
                        rgb_files.push_back(path);
                    } else if (boost::filesystem::extension(path) == ".tiff") { // depth files
                        depth_files.push_back(path);
                    }
                }
            }
            // Sort files to ensure they match by name
            std::sort(rgb_files.begin(), rgb_files.end());
            std::sort(depth_files.begin(), depth_files.end());
            iter_rgb_files = rgb_files.begin();
            iter_depth_files = depth_files.begin();
            ROS_INFO("RGB and Depth files loaded and sorted.");
        } else {
            ROS_WARN("Specified directory does not exist or is not a directory.");
        }
    }
    /*
    void imageCapturedCallback(const std_msgs::Bool::ConstPtr& msg) {
    if (msg->data && iter_rgb_files != rgb_files.end() && iter_depth_files != depth_files.end()) {
            publishImagePath(*iter_rgb_files, rgb_path_pub_);
            ++iter_rgb_files;
            publishImagePath(*iter_depth_files, depth_path_pub_);
            ++iter_depth_files;
        }
    }
    */
    void imageCapturedCallback(const std_msgs::Bool::ConstPtr& msg) {
        if (msg->data && iter_depth_files != depth_files.end()) {
            std::string depth_name = boost::filesystem::path(*iter_depth_files).stem().string().substr(5);
            
            while (iter_rgb_files != rgb_files.end()) {
                std::string rgb_name = boost::filesystem::path(*iter_rgb_files).stem().string().substr(3);
                std::cout << "This is rgb: " << rgb_name << " and this is depth: " << depth_name << std::endl;

                if (rgb_name == depth_name) {
                    publishImagePath(*iter_rgb_files, rgb_path_pub_);
                    publishImagePath(*iter_depth_files, depth_path_pub_);
                    ++iter_depth_files;
                    ++iter_rgb_files;
                    break;
                }
                ++iter_rgb_files;
            }
        }
    }

        

    void publishImagePath(const std::string& file_path, ros::Publisher& publisher) {
        std_msgs::String path_msg;
        path_msg.data = file_path;
        publisher.publish(path_msg);
        ROS_INFO("Published image path: %s", file_path.c_str());
    }

private:
    ros::NodeHandle nh;
    ros::Publisher rgb_path_pub_;
    ros::Publisher depth_path_pub_;
    ros::Subscriber image_captured_sub_;

    std::string image_directory_;
    std::vector<std::string> rgb_files;
    std::vector<std::string> depth_files;
    std::vector<std::string>::iterator iter_rgb_files, iter_depth_files;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "imageCaptured_node");
    ImageCapturedNode node;
    ros::spin();
    return 0;
}
