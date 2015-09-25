/*
 * =====================================================================================
 *
 *       Filename:  hog.cpp
 *
 *    Description:  HOG trainer & detector
 *
 *        Version:  1.0
 *        Created:  2015/09/21 22시 56분 00초
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Il Jae Lee (iljae), iljae@umich.edu
 *   Organization:  University of Michigan
 *
 * =====================================================================================
 */
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <ctime>

#include <opencv/cv.hpp>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include <boost/function.hpp>

typedef std::vector<float> Features;

void ComputeFeatures(const cv::Mat & image,
                     Features & features,
                     const cv::Size & size) {
  cv::HOGDescriptor hog;
  hog.winSize = size;
  cv::Mat gray;
  cv::cvtColor( image, gray, cv::COLOR_BGR2GRAY );
  hog.compute( gray, features, cv::Size(8,8), cv::Size(0,0) );
}

void BuildPyramid(const cv::Mat & image,
                  std::vector<cv::Mat> & pyramid,
                  const cv::Size & min_size) {
  for(int i=1;;i++) {
    const cv::Size scaled_size( image.cols/std::pow(2,i),
                                image.rows/std::pow(2,i));
    if(scaled_size.width<min_size.width||scaled_size.height<min_size.height) break;

    cv::Mat scaled_image;
    cv::resize(image, scaled_image, scaled_size);

    pyramid.push_back(scaled_image.clone());
  }
}

void ExtractFeaturesFromEachFrame(const std::string & video_source,
                                  std::vector<Features> features_collection,
                                  const cv::Size & window_size,
                                  bool scale) {
  std::srand(std::time(0));

  cv::VideoCapture video(video_source);
  if(!video.isOpened()) return;

  cv::HOGDescriptor hog;
  hog.winSize=window_size;

  cv::Mat frame;
  cv::Mat extracted_frame;
  Features features;
  while(video.read(frame)) {
    if(scale) cv::resize(frame, extracted_frame, window_size);
    else {
      // Extract frame
      cv::Rect patch;
      patch.width=window_size.width;
      patch.height=window_size.height;
      patch.x=std::rand()%(frame.cols-window_size.width);
      patch.y=std::rand()%(frame.rows-window_size.height);
      extracted_frame=((frame)(patch)).clone();

      features_collection.push_back(Features((features_collection.size()==0?0:features_collection.front().size())));
      ComputeFeatures(extracted_frame, features_collection.back(), window_size);
    }
    features.clear();
  }
}

void ExtractFeaturesFromEachWindow(const std::string & video_source,
                                   std::vector<Features> features_collection,
                                   const cv::Size & window_size,
                                   bool scale) {
}

void ApplySlidingWindow(const cv::Mat & image,
                        std::vector<cv::Mat> & windows,
                        const boost::function<bool (const cv::Mat &)> & detect_func,
                        const cv::Size & window_size,
                        const unsigned int max_horizontal_steps,
                        const unsigned int max_vertical_steps) {
  const unsigned int horizontal_padding=std::max(image.cols/std::max((int)max_horizontal_steps,1)-window_size.width,0);
  const unsigned int vertical_padding=std::max(image.rows/std::max((int)max_vertical_steps,1)-window_size.height,0);

  cv::Rect box;
  box.width=window_size.width;
  box.height=window_size.height;
  cv::Mat sliding_window;
  int x=0,y=0;
  while(y<image.rows) {
    while(x<image.cols) {
      box.x=x;
      box.y=y;
      sliding_window=(image)(box);
      if(detect_func(sliding_window)) windows.push_back(sliding_window.clone());
      x+=(horizontal_padding+window_size.width);
    }

    x=0;
    y+=(vertical_padding+window_size.height);
  }
}

int main( int argc, char** argv ) {
  bool test_only;
  int width, height, video_source;
  std::string output_file;
  std::string positive_source_directory;
  std::string negative_source_directory;

  try {
    std::string current_path=boost::filesystem::current_path().string<std::string>();
    namespace po=boost::program_options;
    po::options_description desc("Options");
    desc.add_options()
    ("help,h", "Print help messages")
    ("test,t", po::value<bool>(&test_only)->default_value(false), "Specify whether to train")
    ("camera,c", po::value<int>(&video_source)->default_value(0), "Specify camera to retrieve test feed")
    ("width,w", po::value<int>(&width)->default_value(72), "Specify train window width")
    ("height,h", po::value<int>(&height)->default_value(128), "Specify train window height")
    ("positive,p", po::value<std::string>(&positive_source_directory)->default_value(current_path+"/positive"), "Specify positive video files directory")
    ("negative,n", po::value<std::string>(&negative_source_directory)->default_value(current_path+"/negative"), "Specify negative video files direcotry")
    ("output,o", po::value<std::string>(&output_file)->default_value(current_path+"/feature.data"), "Specify an output file");

    po::variables_map vm;
    po::store(po::command_line_parser(argc,argv).options(desc).run(), vm);

    if (vm.count("help")) {
      std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
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
  const cv::Size win_size(width,height);

  if(!test_only) {
    std::vector<Features> positive_features_collection;
    std::vector<Features> negative_features_collection;
  }

  // std::cout << "Testing..." << std::endl;
  // test_it( output_file, video_source, win_size );

  return 0;
}
