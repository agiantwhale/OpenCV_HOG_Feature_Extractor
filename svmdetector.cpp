/*
 * =====================================================================================
 *
 *       Filename:  svmdetector.cpp
 *
 *    Description:  Test SVM detector
 *
 *        Version:  1.0
 *        Created:  2015/09/19 22시 22분 46초
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Il Jae Lee (iljae), iljae@umich.edu
 *   Organization:  University of Michigan
 *
 * =====================================================================================
 */
#include <vector>
#include <iostream>
#include <fstream>
#include <boost/program_options.hpp>
#include <opencv2/opencv.hpp>

void draw_locations(cv::Mat & img, const std::vector<cv::Rect> & locations, const cv::Scalar & color  ) {
  if(!locations.empty()) {
    std::vector<cv::Rect>::const_iterator loc = locations.begin();
    std::vector<cv::Rect>::const_iterator end = locations.end();
    for( ; loc != end ; ++loc  ) {
      rectangle( img, *loc, color, 2  );
    }
  }
}

int main ( int argc, const char * argv[] ) {
  int width, height;
  std::string source_file;
  try {
    namespace po=boost::program_options;
    po::options_description desc("Options");
    desc.add_options()
    ("help,h", "Print help messages")
    ("width,w", po::value<int>(&width)->default_value(128), "Specify train window width")
    ("height,h", po::value<int>(&height)->default_value(72), "Specify train window height")
    ("source,o", po::value<std::string>(&source_file)->required(), "Specify an source file");

    po::positional_options_description p;
    p.add("source",-1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc,argv).options(desc).positional(p).run(), vm);

    if (vm.count("help")) {
      std::cout << "Usage: " << argv[0] << " source" << std::endl;
      std::cout << desc;
      return 0;
    }

    po::notify(vm);
  }
  catch(std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }
  catch(...) {
    std::cerr << "Exception of unknown type!" << std::endl;
    return 1;
  }

  std::vector<float> single_detector_vector;
  {
    std::fstream detector_file(source_file.c_str());

    if(!detector_file) {
      std::cerr << "Error opening source file" << std::endl;
      return 1;
    }

    float feature;
    while(detector_file >> feature) {
      single_detector_vector.push_back(feature);
    }
  }

  cv::HOGDescriptor hog;
  hog.winSize=cv::Size(width, height);
  hog.setSVMDetector(single_detector_vector);

  cv::VideoCapture cam(0);
  if(!cam.isOpened()) {
    std::cerr << "Error opening a video camera source" << std::endl;
    return 1;
  }

  cv::Mat img, draw;
  char key;
  std::vector<cv::Rect> locations;
  bool end_of_process=false;
  while(!end_of_process)
  {
    cam >> img;
    if(img.empty()) break;

    draw = img.clone();

    locations.clear();
    hog.detectMultiScale( draw, locations );
    draw_locations( draw, locations, cv::Scalar(0, 0, 255));

    imshow("cam", draw);
    key = (char)cv::waitKey(10);
    if(27==key) end_of_process = true;
  }

  return 0;
}
