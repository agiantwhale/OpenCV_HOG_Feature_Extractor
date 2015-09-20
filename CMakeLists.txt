cmake_minimum_required (VERSION 3.3)
project (VideoTrainer)

find_package (OpenCV REQUIRED)
find_package (Boost REQUIRED COMPONENTS filesystem system program_options)

include_directories(${Boost_INCLUDE_DIRS})

add_library (svm libsvm/svm.cpp)

add_executable (svmtrain svmtrain.cpp)
target_link_libraries (svmtrain ${OpenCV_LIBS} ${Boost_LIBRARIES})

add_executable (svmexport svmexport.cpp)
target_link_libraries (svmexport svm ${OpenCV_LIBS} ${Boost_LIBRARIES})

add_executable (svmdetector svmdetector.cpp)
target_link_libraries (svmdetector svm ${OpenCV_LIBS} ${Boost_LIBRARIES})
