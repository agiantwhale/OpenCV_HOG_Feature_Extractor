/*
 * =====================================================================================
 *
 *       Filename:  hog.cpp
 *
 *    Description:  Hog trainer
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
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>

const cv::Size kTrainingPadding = cv::Size(0, 0);
const cv::Size kWinStride = cv::Size(60, 60);

void calculateFeaturesFromInput(const cv::Mat &image_data, std::vector<float>& feature_vector, cv::HOGDescriptor& hog) {
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

void populateWithVideoPath(const std::string &folder_name, std::vector<std::string> &videos) {
  boost::filesystem::path current_dir(boost::filesystem::current_path().string<std::string>()
                                      +"/"+folder_name);
  if(!boost::filesystem::is_directory(current_dir)) return;

  boost::filesystem::directory_iterator dir_iter(current_dir), eod;

  BOOST_FOREACH(boost::filesystem::path const &file_path, std::make_pair(dir_iter, eod)) {
    videos.push_back(file_path.string<std::string>());
  }
}

int main ( int argc, const char * argv[] ) {
  cv::HOGDescriptor hog;
  hog.winSize=cv::Size(1920,1080);
  // hog.winSize=cv::Size(320,180);
  // hog.blockSize=cv::Size(8,8);
  // hog.blockStride=cv::Size(4,4);
  // hog.cellSize=cv::Size(4,4);

  // Get the files to train from somewhere
  std::vector<std::string> positive_training_sample_videos;
  std::vector<std::string> negative_training_sample_videos;

  populateWithVideoPath("positive",positive_training_sample_videos);
  populateWithVideoPath("negative",negative_training_sample_videos);

  std::vector<std::string> videos;
  videos.insert(videos.end(), positive_training_sample_videos.begin(), positive_training_sample_videos.end());
  videos.insert(videos.end(), negative_training_sample_videos.begin(), negative_training_sample_videos.end());

  std::ofstream feature_data;
  feature_data.open("feature.data", std::ios::in | std::ios::out | std::ios::trunc);

  int total_frames=0;
  bool file_written=false;
  BOOST_FOREACH(const std::string &video_path, videos) {
    total_frames+=cv::VideoCapture(video_path).get(cv::CAP_PROP_FRAME_COUNT);
  }

  int current_frame=0;
  for(int video_index=0; video_index<(int)videos.size(); video_index++) {
    const std::string &video_path=(videos[video_index]);
    std::cout << "Processing video " << video_path << std::endl;
    cv::VideoCapture video(video_path);
    if(!video.isOpened()) continue;

    cv::Mat frame;
    std::vector<float> features;
    while(video.retrieve(frame)) {
      cv::resize(frame, frame, hog.winSize);

      features.clear();
      calculateFeaturesFromInput(frame, features, hog);

      if(!file_written) {
        std::cout << "Number of features: " << features.size() << std::endl;
        file_written=true;
        feature_data << total_frames
                     << " "
                     << features.size()
                     << " "
                     << 2
                     << std::endl;
      }

      for(int feature_index=0; feature_index<(int)features.size(); feature_index++) {
        feature_data << features[feature_index];
        if((feature_index+1)==(int)features.size()) feature_data << std::endl;
        else feature_data << " ";
      }

      if(video_index<(int)positive_training_sample_videos.size()) feature_data << "1 0";
      else feature_data << "0 1";

      feature_data << std::endl;

      std::cout << ++current_frame << "/" << total_frames << " frames processed..." << std::endl;
    }
  }

  feature_data.close();

  return 0;
}
