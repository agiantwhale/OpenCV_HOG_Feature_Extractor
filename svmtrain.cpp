/*
 * =====================================================================================
 *
 *       Filename:  svmtrain.cpp
 *
 *    Description:  SVM Trainer from Video Files
 *
 *        Version:  1.0
 *        Created:  2015/09/18 08시 41분 43초
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Il Jae Lee (iljae), iljae@umich.edu
 *   Organization:  University of Michigan
 *
 * =====================================================================================
 */
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <opencv/cv.hpp>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>

const cv::Size kTrainingPadding = cv::Size(0, 0);
const cv::Size kWinStride = cv::Size(8,8);

void CalculateFeaturesFromInput(const cv::Mat &image_data, std::vector<float>& feature_vector, cv::HOGDescriptor& hog) {
  if (image_data.empty()) {
    feature_vector.clear();
    std::cerr << "Error: HOG image is empty, features calculation skipped!" << std::endl;
    return;
  }
  // Check for mismatching dimensions
  if (image_data.cols != hog.winSize.width || image_data.rows != hog.winSize.height) {
    feature_vector.clear();
    std::cerr << "Error: Image dimensions (" << image_data.cols << " x " << image_data.rows << ") do not match HOG window size (" << hog.winSize.width << " x "<< hog.winSize.height <<")!" << std::endl;
    return;
  }
  std::vector<cv::Point> locations;
  hog.compute(image_data, feature_vector, kWinStride, kTrainingPadding, locations);
}

void PopulateWithVideoPath(const std::string &folder_name, std::vector<std::string> &videos) {
  boost::filesystem::path current_dir(boost::filesystem::current_path().string<std::string>()
                                      +"/"+folder_name);
  if(!boost::filesystem::is_directory(current_dir)) return;

  boost::filesystem::directory_iterator dir_iter(current_dir), eod;

  BOOST_FOREACH(boost::filesystem::path const &file_path, std::make_pair(dir_iter, eod)) {
    videos.push_back(file_path.string<std::string>());
  }
}

int main ( int argc, const char * argv[] ) {
  int width, height;
  std::string output_file;
  std::string positive_source_directory;
  std::string negative_source_directory;

  try {
    namespace po=boost::program_options;
    po::options_description desc("Options");
    desc.add_options()
    ("help,h", "Print help messages")
    ("width,w", po::value<int>(&width)->default_value(384), "Specify train window width")
    ("height,h", po::value<int>(&height)->default_value(216), "Specify train window height")
    ("positive,p", po::value<std::string>(&positive_source_directory)->default_value(boost::filesystem::current_path().string<std::string>()+"/positive"), "Specify positive video files directory")
    ("negative,n", po::value<std::string>(&negative_source_directory)->default_value(boost::filesystem::current_path().string<std::string>()+"/negative"), "Specify negative video files direcotry")
    ("output,o", po::value<std::string>(&output_file)->default_value(boost::filesystem::current_path().string<std::string>()+"/feature.data"), "Specify an output file");

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
  cv::HOGDescriptor hog;
  hog.winSize=cv::Size(width, height);
  // hog.blockSize=cv::Size(16,16);
  // hog.blockStride=cv::Size(8,8);
  // hog.cellSize=cv::Size(8,8);

  // Get the files to train from somewhere
  std::vector<std::string> positive_training_sample_videos;
  std::vector<std::string> negative_training_sample_videos;

  PopulateWithVideoPath("positive",positive_training_sample_videos);
  PopulateWithVideoPath("negative",negative_training_sample_videos);

  std::vector<std::string> videos;
  videos.insert(videos.end(), positive_training_sample_videos.begin(), positive_training_sample_videos.end());
  videos.insert(videos.end(), negative_training_sample_videos.begin(), negative_training_sample_videos.end());

  std::ofstream feature_data;
  feature_data.open(output_file.c_str(), std::ios::out|std::ios::app);

  std::vector<double> results;
  typedef std::vector<float> FeatureSet;

  int current_frame=0;
  for(int video_index=0; video_index<(int)videos.size(); video_index++) {
    const std::string &video_path=(videos[video_index]);
    std::cout << "Processing video " << video_path << std::endl;
    cv::VideoCapture video(video_path);
    if(!video.isOpened()) continue;

    cv::Mat frame;
    FeatureSet features;
    while(video.read(frame)) {
      cv::Mat resized_frame;
      cv::resize(frame, resized_frame, hog.winSize);

      features.clear();
      CalculateFeaturesFromInput(resized_frame, features, hog);

      static bool report_features=false;
      if(!report_features) {
        std::cout << "Number of features: " << features.size() << std::endl;
        report_features=true;
      }

      feature_data << (video_index<(int)positive_training_sample_videos.size() ? "+1" : "-1");

      for(int feature_index=0; feature_index<(int)features.size(); feature_index++) {
        feature_data << " " << (feature_index+1) << ":" << features[feature_index];
      }
      feature_data << std::endl;

      std::cout << ++current_frame << " frames processed..." << std::endl;
    }
  }

  return 0;
}
