cmake_minimum_required(VERSION 2.8.3)
project(animated_obstacle_plugin)

find_package(roscpp REQUIRED)
find_package(message_generation REQUIRED)
find_package(tf REQUIRED)
find_package(std_msgs REQUIRED)
include_directories(${roscpp_INCLUDE_DIRS})
include_directories(${obstacle_manager_INCLUDE_DIRS})
include_directories(${tf_INCLUDE_DIRS})
include_directories(${std_msgs_INCLUDE_DIRS})
include_directories(/user/include/sdformat-5.2)
include_directories(./include)
find_package(gazebo REQUIRED)
include_directories(${GAZEBO_INCLUDE_DIRS})
link_directories(${GAZEBO_LIBRARY_DIRS})
list(APPEND CMAKE_CXX_FLAGS "${GAZEBO_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GAZEBO_CXX_FLAGS}")

## Generate services in the 'srv' folder
add_service_files(
  DIRECTORY
  srv
  FILES
  SetPose.srv
  SetVel.srv
  GetPose.srv
  GetVel.srv
  Init.srv
)
## Generate added messages and services with any dependencies listed here
generate_messages(
   DEPENDENCIES
   std_msgs  # Or other packages containing msgs
)
catkin_package(CATKIN_DEPENDS message_runtime std_msgs)

add_library(animated_obstacle_plugin_ros SHARED src/animated_obstacle_plugin_ros.cc)
target_link_libraries(animated_obstacle_plugin_ros ${GAZEBO_LIBRARIES}  ${roscpp_LIBRARIES} ${tf_LIBRARIES})
set_target_properties(animated_obstacle_plugin_ros PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)

