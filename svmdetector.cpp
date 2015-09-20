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

int main ( int argc, const char * argv[] ) {
  std::string source_file;
  try {
    namespace po=boost::program_options;
    po::options_description desc("Options");
    desc.add_options()
    ("help,h", "Print help messages")
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
  hog.setSVMDetector(single_detector_vector);

  cv::VideoCapture cam(0);
  if(!cam.isOpened()) {
    std::cerr << "Error opening a video camera source" << std::endl;
    return 1;
  }

  cv::Mat img;

  while(cam.read(img)) {
    std::vector<cv::Rect> found, found_filtered;
    hog.detectMultiScale(img, found, 0, cv::Size(8,8), cv::Size(32,32), 1.05, 2);

    size_t i, j;
    for (i=0; i<found.size(); i++) {
      cv::Rect r = found[i];
      for (j=0; j<found.size(); j++)
        if (j!=i && (r & found[j])==r) break;

      if (j==found.size()) found_filtered.push_back(r);
    }

    for (i=0; i<found_filtered.size(); i++) {
      cv::Rect r = found_filtered[i];
      r.x += cvRound(r.width*0.1);
      r.width = cvRound(r.width*0.8);
      r.y += cvRound(r.height*0.06);
      r.height = cvRound(r.height*0.9);
      rectangle(img, r.tl(), r.br(), cv::Scalar(0,255,0), 2);
    }

    cv::Mat smaller_image;
    cv::resize(img,smaller_image,cv::Size(640,480));

    imshow("result", smaller_image);

    if(cv::waitKey(30) >= 0) break;
  }

  cv::waitKey(0);

  return 0;
}
