#include <fstream>
#include <iostream>
#include <string>
#include <ros/ros.h>
#include <std_msgs/String.h>
#include <std_msgs/Float32.h>
#include <std_msgs/Int32.h>
#include <std_msgs/Bool.h>

class PlantFeatureNode {
public:
    PlantFeatureNode() : received_height(false), received_width(false), received_length(false), received_leaf_count(false), received_avg_leaf_area(false), received_plant_area(false) {
        nh = ros::NodeHandle();
        rgb_path_sub_ = nh.subscribe("rgb_path", 10, &PlantFeatureNode::rgbPathCallback, this);
        depth_path_sub_ = nh.subscribe("depth_path", 10, &PlantFeatureNode::depthPathCallback, this);
        plant_height_sub_ = nh.subscribe("plant_height", 10, &PlantFeatureNode::heightCallback, this);
        plant_width_sub_ = nh.subscribe("plant_width", 10, &PlantFeatureNode::widthCallback, this);
        plant_length_sub_ = nh.subscribe("plant_length", 10, &PlantFeatureNode::lengthCallback, this);
        leaf_count_sub_ = nh.subscribe("leaf_count", 10, &PlantFeatureNode::leafCountCallback, this);
        avg_leaf_area_sub_ = nh.subscribe("avg_leaf_area", 10, &PlantFeatureNode::avgLeafAreaCallback, this);
        plant_area_sub_ = nh.subscribe("plant_area", 10, &PlantFeatureNode::plantAreaCallback, this);

        image_captured_pub_ = nh.advertise<std_msgs::Bool>("image_captured", 10);
        std::cout << "Store Features Node initialized." << std::endl;
    }

    std::string extractTimestamp(const std::string& path) {
        size_t pos = path.find_last_of("/");
        if (pos != std::string::npos && pos + 1 < path.length()) {
            std::string filename = path.substr(pos + 1);
            // Check if the filename starts with "rgb_" and has the expected format
            if (filename.substr(0, 4) == "rgb_" && filename.length() >= 19) {
                std::string timestamp = filename.substr(4, 15);  // Extract "yyyyMMdd_HHMMSS"
                // Format the timestamp
                return timestamp.substr(0, 4) + "-" + timestamp.substr(4, 2) + "-" + timestamp.substr(6, 2) + " " +
                    timestamp.substr(9, 2) + ":" + timestamp.substr(11, 2) + ":" + timestamp.substr(13, 2);
            }
        }
        return "";  // Return empty string if timestamp couldn't be extracted
    }

    void checkAndStoreFeatures() {

        if (received_height && received_width && received_length && received_leaf_count && received_avg_leaf_area && received_plant_area && received_rgb_path) {
            appendToCSV();
            std::cout << "Appended to CSV." << std::endl;
            resetFeatureFlags();
            std::cout << "Flags were resetted." << std::endl;
        }
    }

    void resetFeatureFlags() {
        received_height = received_width = received_length = received_leaf_count = received_avg_leaf_area = received_plant_area = received_rgb_path = false;
    }

    void appendToCSV() {
        std::string filename = "20_broccoli.csv"; // Example csv for a broccoli dataset
    std::ofstream file(filename, std::ios::app | std::ios::out);

    if (file.tellp() == 0) {
        file << "plant_name,plant_id,plant_height,plant_width,plant_length,plant_area,leaf_count,avg_leaf_area,rgb_path,time\n";
    }

    file << feature.plant_name << "," << feature.plant_id << "," << feature.plant_height << "," << feature.plant_width << ","
         << feature.plant_length << "," << feature.plant_area << "," << feature.leaf_count << ","
         << feature.avg_leaf_area << "," << feature.rgb_path << "," << feature.timestamp << "\n";
    file.close();

    std_msgs::Bool msg;
    msg.data = true;
    image_captured_pub_.publish(msg);
    std::cout << "Capture Published for the new image." << std::endl;
    }

private:
    ros::NodeHandle nh;
    ros::Subscriber rgb_path_sub_, depth_path_sub_, plant_height_sub_, plant_width_sub_, plant_length_sub_, leaf_count_sub_, avg_leaf_area_sub_, plant_area_sub_;
    ros::Publisher image_captured_pub_;

    struct PlantFeature {
        std::string plant_name = "broccoli";
        int plant_id = 20;
        float plant_height = 0;
        float plant_width = 0;
        float plant_length = 0;
        float plant_area = 0;
        int leaf_count = 0;
        float avg_leaf_area = 0;
        std::string rgb_path = "";
        std::string timestamp = "";  // New field for timestamp
    } feature;

    bool received_rgb_path,received_height, received_width, received_length, received_leaf_count, received_avg_leaf_area, received_plant_area;

    // Callbacks for each feature
    void rgbPathCallback(const std_msgs::String::ConstPtr& msg) {
    feature.rgb_path = msg->data;
    feature.timestamp = extractTimestamp(msg->data);
    received_rgb_path = true;
    checkAndStoreFeatures();  }
    void depthPathCallback(const std_msgs::String::ConstPtr& msg) { /* Handle Depth Path */ }
    void heightCallback(const std_msgs::Float32::ConstPtr& msg) { feature.plant_height = msg->data; received_height = true; checkAndStoreFeatures(); }
    void widthCallback(const std_msgs::Float32::ConstPtr& msg) { feature.plant_width = msg->data; received_width = true; checkAndStoreFeatures(); }
    void lengthCallback(const std_msgs::Float32::ConstPtr& msg) { feature.plant_length = msg->data; received_length = true; checkAndStoreFeatures(); }
    void leafCountCallback(const std_msgs::Int32::ConstPtr& msg) { feature.leaf_count = static_cast<int>(msg->data); received_leaf_count = true; checkAndStoreFeatures(); }
    void avgLeafAreaCallback(const std_msgs::Float32::ConstPtr& msg) { feature.avg_leaf_area = msg->data; received_avg_leaf_area = true; checkAndStoreFeatures(); }
    void plantAreaCallback(const std_msgs::Float32::ConstPtr& msg) { feature.plant_area = msg->data; received_plant_area = true; checkAndStoreFeatures(); }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "store_features_node");
    PlantFeatureNode node;
    ros::spin();
    return 0;
}