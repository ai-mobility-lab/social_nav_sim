cmake_minimum_required(VERSION 2.8.3)
project(assets)

find_package(catkin REQUIRED COMPONENTS
  gazebo_ros
)

catkin_package()

install(DIRECTORY models worlds
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

