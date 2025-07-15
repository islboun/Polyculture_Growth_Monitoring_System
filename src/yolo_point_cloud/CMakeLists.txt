cmake_minimum_required(VERSION 3.0.2)
project(yolo_point_cloud)

## Compile as C++14, required for PCL
add_compile_options(-std=c++14)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  roscpp
  rospy
  sensor_msgs
  std_msgs
)

## Find system libraries
find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)
find_package(jsoncpp REQUIRED)

## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
# catkin_python_setup()

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
catkin_package(
  CATKIN_DEPENDS cv_bridge roscpp rospy sensor_msgs std_msgs
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${jsoncpp_INCLUDE_DIRS}
)

## Declare a C++ executable
add_executable(imageCaptured_node src/imageCaptured.cpp)
add_executable(measurements_node src/measurements_node.cpp)
add_executable(pcl_measurements_node src/pcl_measurements_node.cpp)
add_executable(store_features_node src/store_features_node.cpp)

## Add cmake target dependencies of the executable
add_dependencies(imageCaptured_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(measurements_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(pcl_measurements_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
add_dependencies(store_features_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
target_link_libraries(imageCaptured_node
  ${catkin_LIBRARIES}
)


target_link_libraries(pcl_measurements_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}
  jsoncpp_lib
)

target_link_libraries(measurements_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}
  jsoncpp_lib
)

target_link_libraries(store_features_node
  ${catkin_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}
  jsoncpp_lib
)
