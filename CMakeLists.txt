cmake_minimum_required(VERSION 2.8.3)
project(navigation_foundation)

# Default to C++14
if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 14)
endif()

find_package(catkin REQUIRED COMPONENTS)

catkin_package(
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS
)
