cmake_minimum_required (VERSION 3.3)
project (VideoTrainer)

find_package (OpenCV REQUIRED)
find_package (Boost REQUIRED COMPONENTS filesystem system)

include_directories(${Boost_INCLUDE_DIRS})

add_executable (hogtrain hogtrain.cpp)
target_link_libraries (hogtrain ${OpenCV_LIBS} ${Boost_LIBRARIES})

add_executable (svmtrain svmtrain.cpp)
target_link_libraries (svmtrain ${OpenCV_LIBS} ${Boost_LIBRARIES})
