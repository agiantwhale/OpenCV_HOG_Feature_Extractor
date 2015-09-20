cmake_minimum_required (VERSION 3.3)
project (VideoTrainer)

find_package (OpenCV REQUIRED)
find_package (Boost REQUIRED COMPONENTS filesystem system program_options)

include_directories(${Boost_INCLUDE_DIRS})

add_library (svm libsvm/svm.cpp)
add_library (svm_learn svmlight/svm_learn.c)
add_library (svm_common svmlight/svm_common.c)

add_executable (svmtrain svmtrain.cpp)
target_link_libraries (svmtrain ${OpenCV_LIBS} ${Boost_LIBRARIES})

add_executable (libsvmexport libsvmexport.cpp)
target_link_libraries (libsvmexport svm ${OpenCV_LIBS} ${Boost_LIBRARIES})

add_executable (svmlightexport svmlightexport.cpp)
target_link_libraries (svmlightexport svm_learn svm_common ${OpenCV_LIBS} ${Boost_LIBRARIES})

add_executable (svmdetector svmdetector.cpp)
target_link_libraries (svmdetector ${OpenCV_LIBS} ${Boost_LIBRARIES})
