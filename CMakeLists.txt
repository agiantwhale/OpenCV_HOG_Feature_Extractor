cmake_minimum_required (VERSION 3.3)
project (VideoTrainer)

include_directories (/usr/local/include/)
link_directories (/usr/local/lib/)

add_executable (hogtrain hogtrain.cpp)
target_link_libraries (hogtrain boost_system boost_filesystem opencv_core opencv_objdetect opencv_imgproc opencv_videoio)

add_executable (svmtrain svmtrain.cpp)
target_link_libraries (svmtrain boost_system boost_filesystem opencv_core opencv_objdetect opencv_imgproc opencv_videoio)
