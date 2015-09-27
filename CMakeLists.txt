cmake_minimum_required (VERSION 3.3)
project (VideoTrainer)

find_package (OpenCV 3.0 REQUIRED)
find_package (Boost REQUIRED COMPONENTS filesystem system program_options)

include_directories(${Boost_INCLUDE_DIRS})

add_library (svm libsvm/svm.cpp)
add_library (svmlight svmlight/svm_common.c)
target_link_libraries (svmlight m)

add_executable (svmtrain svmtrain.cpp)
target_link_libraries (svmtrain ${OpenCV_LIBS} ${Boost_LIBRARIES})

add_executable (svmtrainhog svmtrainhog.cpp)
target_link_libraries (svmtrainhog ${OpenCV_LIBS} ${Boost_LIBRARIES})

add_executable (libsvmexport libsvmexport.cpp)
target_link_libraries (libsvmexport svm ${OpenCV_LIBS} ${Boost_LIBRARIES})

add_executable (svmlightexport svmlightexport.cpp)
target_link_libraries (svmlightexport svmlight ${OpenCV_LIBS} ${Boost_LIBRARIES})

add_executable (svmdetector svmdetector.cpp)
target_link_libraries (svmdetector ${OpenCV_LIBS} ${Boost_LIBRARIES})
