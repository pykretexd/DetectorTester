cmake_minimum_required(VERSION 3.10)

project(DetectorTester)

find_package(OpenCV REQUIRED)
find_package(OpenVINO REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(DetectorTester src/main.cpp)

target_link_libraries(DetectorTester openvino::runtime)
target_link_libraries(DetectorTester ${OpenCV_LIBS})