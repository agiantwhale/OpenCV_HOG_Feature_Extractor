/*
 * =====================================================================================
 *
 *       Filename:  svmtrainhog.cpp
 *
 *    Description:
 *
 *        Version:  1.0
 *        Created:  2015/09/20 19시 29분 48초
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Il Jae Lee (iljae), iljae@umich.edu
 *   Organization:  University of Michigan
 *
 * =====================================================================================
 */
#include <opencv2/opencv.hpp>

#include <string>
#include <iostream>
#include <fstream>
#include <vector>

#include <time.h>

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>

using namespace cv;
using namespace cv::ml;
using namespace std;

void get_svm_detector(const Ptr<SVM>& svm, vector< float > & hog_detector );
void convert_to_ml(const std::vector< cv::Mat > & train_samples, cv::Mat& trainData );
void load_images( const string & directory, vector< Mat > & img_lst, const Size & size=Size(0,0) );
void sample_neg( const vector< Mat > & full_neg_lst, vector< Mat > & neg_lst, const Size & size );
Mat get_hogdescriptor_visu(const Mat& color_origImg, vector<float>& descriptorValues, const Size & size );
void compute_hog( const vector< Mat > & img_lst, vector< Mat > & gradient_lst, const Size & size );
void train_svm( const vector< Mat > & gradient_lst, const vector< int > & labels, const string & output_file );
void draw_locations( Mat & img, const vector< Rect > & locations, const Scalar & color );
void test_it( const string & output_file, int video_source, const Size & size );

void get_svm_detector(const Ptr<SVM>& svm, vector< float > & hog_detector )
{
    // get the support vectors
    Mat sv = svm->getSupportVectors();
    const int sv_total = sv.rows;
    // get the decision function
    Mat alpha, svidx;
    double rho = svm->getDecisionFunction(0, alpha, svidx);

    CV_Assert( alpha.total() == 1 && svidx.total() == 1 && sv_total == 1 );
    CV_Assert( (alpha.type() == CV_64F && alpha.at<double>(0) == 1.) ||
               (alpha.type() == CV_32F && alpha.at<float>(0) == 1.f) );
    CV_Assert( sv.type() == CV_32F );
    hog_detector.clear();

    hog_detector.resize(sv.cols + 1);
    memcpy(&hog_detector[0], sv.ptr(), sv.cols*sizeof(hog_detector[0]));
    hog_detector[sv.cols] = (float)-rho;
}


/*
* Convert training/testing set to be used by OpenCV Machine Learning algorithms.
* TrainData is a matrix of size (#samples x max(#cols,#rows) per samples), in 32FC1.
* Transposition of samples are made if needed.
*/
void convert_to_ml(const std::vector< cv::Mat > & train_samples, cv::Mat& trainData )
{
    //--Convert data
    const int rows = (int)train_samples.size();
    const int cols = (int)std::max( train_samples[0].cols, train_samples[0].rows );
    cv::Mat tmp(1, cols, CV_32FC1); //< used for transposition if needed
    trainData = cv::Mat(rows, cols, CV_32FC1 );
    vector< Mat >::const_iterator itr = train_samples.begin();
    vector< Mat >::const_iterator end = train_samples.end();
    for( int i = 0 ; itr != end ; ++itr, ++i ) {
        CV_Assert( itr->cols == 1 ||
            itr->rows == 1 );
        if( itr->cols == 1 )
        {
            transpose( *(itr), tmp );
            tmp.copyTo( trainData.row( i ) );
        }
        else if( itr->rows == 1 )
        {
            itr->copyTo( trainData.row( i ) );
        }
    }
}

void load_images( const string & directory, vector< Mat > & img_lst, const Size & size )
{
  boost::filesystem::path current_dir(directory);
  if(!boost::filesystem::is_directory(current_dir)) return;

  boost::filesystem::directory_iterator dir_iter(current_dir), eod;

  vector<string> videos;
  BOOST_FOREACH(boost::filesystem::path const &file_path, std::make_pair(dir_iter, eod)) {
    videos.push_back(file_path.string<string>());
  }

  for(vector<string>::iterator iter=videos.begin();
      iter!=videos.end();
      iter++) {
    const string &video_path=(*iter);
    cout << "Loading " << video_path << "..." << endl;
    cv::VideoCapture video(video_path);
    if(!video.isOpened()) continue;
    cv::Mat frame;
    int frame_count=0;
    while(video.read(frame)) {
      if(frame.empty()) break;

      Mat cloned_img;
      if(size.width==0||size.height==0) {
        cloned_img=frame.clone();
      } else resize(frame, cloned_img, size);
#ifdef _DEBUG
      imshow( "image", cloned_img );
      waitKey( 10 );
#endif
      img_lst.push_back( cloned_img );
      cout << "Loaded " << ++frame_count << " frames." << endl;
    }
  }
}

void sample_neg( const vector< Mat > & full_neg_lst, vector< Mat > & neg_lst, const Size & size )
{
  Rect box;
  box.width = size.width;
  box.height = size.height;

  const int size_x = box.width;
  const int size_y = box.height;

  srand( (unsigned int)time( NULL ) );

  vector< Mat >::const_iterator img = full_neg_lst.begin();
  vector< Mat >::const_iterator end = full_neg_lst.end();

  int sampled_frames=0;
  for( ; img != end ; ++img )
  {
    box.x = rand() % (img->cols - size_x);
    box.y = rand() % (img->rows - size_y);
    Mat roi = (*img)(box);
    neg_lst.push_back( roi.clone() );
#ifdef _DEBUG
    imshow( "img", roi.clone() );
    waitKey( 10 );
#endif
    cout << "Sampled " << ++sampled_frames << " frames." << endl;
  }
}

// From http://www.juergenwiki.de/work/wiki/doku.php?id=public:hog_descriptor_computation_and_visualization
Mat get_hogdescriptor_visu(const Mat& color_origImg, vector<float>& descriptorValues, const Size & size )
{
    const int DIMX = size.width;
    const int DIMY = size.height;
    float zoomFac = 3;
    Mat visu;
    resize(color_origImg, visu, Size( (int)(color_origImg.cols*zoomFac), (int)(color_origImg.rows*zoomFac) ) );

    int cellSize        = 8;
    int gradientBinSize = 9;
    float radRangeForOneBin = (float)(CV_PI/(float)gradientBinSize); // dividing 180 into 9 bins, how large (in rad) is one bin?

    // prepare data structure: 9 orientation / gradient strenghts for each cell
    int cells_in_x_dir = DIMX / cellSize;
    int cells_in_y_dir = DIMY / cellSize;
    float*** gradientStrengths = new float**[cells_in_y_dir];
    int** cellUpdateCounter   = new int*[cells_in_y_dir];
    for (int y=0; y<cells_in_y_dir; y++)
    {
        gradientStrengths[y] = new float*[cells_in_x_dir];
        cellUpdateCounter[y] = new int[cells_in_x_dir];
        for (int x=0; x<cells_in_x_dir; x++)
        {
            gradientStrengths[y][x] = new float[gradientBinSize];
            cellUpdateCounter[y][x] = 0;

            for (int bin=0; bin<gradientBinSize; bin++)
                gradientStrengths[y][x][bin] = 0.0;
        }
    }

    // nr of blocks = nr of cells - 1
    // since there is a new block on each cell (overlapping blocks!) but the last one
    int blocks_in_x_dir = cells_in_x_dir - 1;
    int blocks_in_y_dir = cells_in_y_dir - 1;

    // compute gradient strengths per cell
    int descriptorDataIdx = 0;
    int cellx = 0;
    int celly = 0;

    for (int blockx=0; blockx<blocks_in_x_dir; blockx++)
    {
        for (int blocky=0; blocky<blocks_in_y_dir; blocky++)
        {
            // 4 cells per block ...
            for (int cellNr=0; cellNr<4; cellNr++)
            {
                // compute corresponding cell nr
                cellx = blockx;
                celly = blocky;
                if (cellNr==1) celly++;
                if (cellNr==2) cellx++;
                if (cellNr==3)
                {
                    cellx++;
                    celly++;
                }

                for (int bin=0; bin<gradientBinSize; bin++)
                {
                    float gradientStrength = descriptorValues[ descriptorDataIdx ];
                    descriptorDataIdx++;

                    gradientStrengths[celly][cellx][bin] += gradientStrength;

                } // for (all bins)


                // note: overlapping blocks lead to multiple updates of this sum!
                // we therefore keep track how often a cell was updated,
                // to compute average gradient strengths
                cellUpdateCounter[celly][cellx]++;

            } // for (all cells)


        } // for (all block x pos)
    } // for (all block y pos)


    // compute average gradient strengths
    for (celly=0; celly<cells_in_y_dir; celly++)
    {
        for (cellx=0; cellx<cells_in_x_dir; cellx++)
        {

            float NrUpdatesForThisCell = (float)cellUpdateCounter[celly][cellx];

            // compute average gradient strenghts for each gradient bin direction
            for (int bin=0; bin<gradientBinSize; bin++)
            {
                gradientStrengths[celly][cellx][bin] /= NrUpdatesForThisCell;
            }
        }
    }

    // draw cells
    for (celly=0; celly<cells_in_y_dir; celly++)
    {
        for (cellx=0; cellx<cells_in_x_dir; cellx++)
        {
            int drawX = cellx * cellSize;
            int drawY = celly * cellSize;

            int mx = drawX + cellSize/2;
            int my = drawY + cellSize/2;

            rectangle(visu, Point((int)(drawX*zoomFac), (int)(drawY*zoomFac)), Point((int)((drawX+cellSize)*zoomFac), (int)((drawY+cellSize)*zoomFac)), Scalar(100,100,100), 1);

            // draw in each cell all 9 gradient strengths
            for (int bin=0; bin<gradientBinSize; bin++)
            {
                float currentGradStrength = gradientStrengths[celly][cellx][bin];

                // no line to draw?
                if (currentGradStrength==0)
                    continue;

                float currRad = bin * radRangeForOneBin + radRangeForOneBin/2;

                float dirVecX = cos( currRad );
                float dirVecY = sin( currRad );
                float maxVecLen = (float)(cellSize/2.f);
                float scale = 2.5; // just a visualization scale, to see the lines better

                // compute line coordinates
                float x1 = mx - dirVecX * currentGradStrength * maxVecLen * scale;
                float y1 = my - dirVecY * currentGradStrength * maxVecLen * scale;
                float x2 = mx + dirVecX * currentGradStrength * maxVecLen * scale;
                float y2 = my + dirVecY * currentGradStrength * maxVecLen * scale;

                // draw gradient visualization
                line(visu, Point((int)(x1*zoomFac),(int)(y1*zoomFac)), Point((int)(x2*zoomFac),(int)(y2*zoomFac)), Scalar(0,255,0), 1);

            } // for (all bins)

        } // for (cellx)
    } // for (celly)


    // don't forget to free memory allocated by helper data structures!
    for (int y=0; y<cells_in_y_dir; y++)
    {
        for (int x=0; x<cells_in_x_dir; x++)
        {
            delete[] gradientStrengths[y][x];
        }
        delete[] gradientStrengths[y];
        delete[] cellUpdateCounter[y];
    }
    delete[] gradientStrengths;
    delete[] cellUpdateCounter;

    return visu;

} // get_hogdescriptor_visu

void compute_hog( const vector< Mat > & img_lst, vector< Mat > & gradient_lst, const Size & size )
{
    HOGDescriptor hog;
    hog.winSize = size;
    Mat gray;
    vector< Point > location;
    vector< float > descriptors;

    vector< Mat >::const_iterator img = img_lst.begin();
    vector< Mat >::const_iterator end = img_lst.end();
    for( ; img != end ; ++img ) {
        cvtColor( *img, gray, COLOR_BGR2GRAY );
        hog.compute( gray, descriptors, Size( 8, 8 ), Size( 0, 0 ), location );
        gradient_lst.push_back( Mat( descriptors ).clone() );
#ifdef _DEBUG
        imshow( "gradient", get_hogdescriptor_visu( img->clone(), descriptors, size ) );
        waitKey( 10 );
#endif
    }
}

void train_svm( const vector< Mat > & gradient_lst, const vector< int > & labels , const string & output_file )
{

    Mat train_data;
    convert_to_ml( gradient_lst, train_data );

    clog << "Start training...";
    Ptr<SVM> svm = SVM::create();
    /* Default values to train SVM */
    svm->setCoef0(0.0);
    svm->setDegree(3);
    svm->setTermCriteria(TermCriteria( CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 1000, 1e-3 ));
    svm->setGamma(0);
    svm->setKernel(SVM::LINEAR);
    svm->setNu(0.5);
    svm->setP(0.1); // for EPSILON_SVR, epsilon in loss function?
    svm->setC(0.01); // From paper, soft classifier
    svm->setType(SVM::EPS_SVR); // C_SVC; // EPSILON_SVR; // may be also NU_SVR; // do regression task
    svm->train(train_data, ROW_SAMPLE, Mat(labels));
    clog << "...[done]" << endl;

    svm->save( output_file );
}

void draw_locations( Mat & img, const vector< Rect > & locations, const Scalar & color )
{
    if( !locations.empty() )
    {
        vector< Rect >::const_iterator loc = locations.begin();
        vector< Rect >::const_iterator end = locations.end();
        for( ; loc != end ; ++loc )
        {
            rectangle( img, *loc, color, 2 );
        }
    }
}

void test_it( const string & output_file, int video_source, const Size & size )
{
    char key = 27;
    Scalar reference( 0, 255, 0 );
    Scalar trained( 0, 0, 255 );
    Mat img, draw;
    Ptr<SVM> svm;
    HOGDescriptor hog;
    hog.winSize = size;
    VideoCapture video;
    vector< Rect > locations;

    // Load the trained SVM.
    svm = StatModel::load<SVM>( output_file );
    // Set the trained svm to hog
    vector< float > hog_detector;
    get_svm_detector( svm, hog_detector );
    hog.setSVMDetector( hog_detector );
    // Open the camera.
    video.open(1);
    if( !video.isOpened() )
    {
        cerr << "Unable to open the device 0" << endl;
        exit( -1 );
    }

    bool end_of_process = false;
    while( !end_of_process )
    {
        video >> img;
        if( img.empty() )
            break;

        draw = img.clone();

        locations.clear();
        hog.detectMultiScale( img, locations );
        draw_locations( draw, locations, trained );

        imshow( "Video", draw );
        key = (char)waitKey( 10 );
        if( 27 == key )
            end_of_process = true;
    }
}

int main( int argc, char** argv )
{
  bool test_only;
  int width, height, video_source;
  std::string output_file;
  std::string positive_source_directory;
  std::string negative_source_directory;

  try {
    namespace po=boost::program_options;
    po::options_description desc("Options");
    desc.add_options()
    ("help,h", "Print help messages")
    ("test,t", po::value<bool>(&test_only)->default_value(false), "Specify whether to train")
    ("camera,c", po::value<int>(&video_source)->default_value(0), "Specify camera to retrieve test feed")
    ("width,w", po::value<int>(&width)->default_value(72), "Specify train window width")
    ("height,h", po::value<int>(&height)->default_value(128), "Specify train window height")
    ("positive,p", po::value<std::string>(&positive_source_directory)->default_value(boost::filesystem::current_path().string<string>()+"/positive"), "Specify positive video files directory")
    ("negative,n", po::value<std::string>(&negative_source_directory)->default_value(boost::filesystem::current_path().string<string>()+"/negative"), "Specify negative video files direcotry")
    ("output,o", po::value<std::string>(&output_file)->default_value(boost::filesystem::current_path().string<string>()+"/feature.data"), "Specify an output file");

    po::variables_map vm;
    po::store(po::command_line_parser(argc,argv).options(desc).run(), vm);

    if (vm.count("help")) {
      cout << "Usage: " << argv[0] << " [options]" << endl;
      cout << desc;
      return 0;
    }

    po::notify(vm);
  }
  catch(std::exception& e) {
    cerr << "Error: " << e.what() << endl;
    return 1;
  }
  catch(...) {
    cerr << "Exception of unknown type!" << endl;
    return 1;
  }

  const Size win_size=Size(width,height);

  if(!test_only) {
  vector< Mat > pos_lst;
  vector< Mat > full_neg_lst;
  vector< Mat > neg_lst;
  vector< Mat > gradient_lst;
  vector< int > labels;

  load_images( positive_source_directory, pos_lst, win_size );
  labels.assign( pos_lst.size(), +1 );
  const unsigned int old = (unsigned int)labels.size();
  load_images( negative_source_directory, full_neg_lst );
  sample_neg( full_neg_lst, neg_lst, win_size );
  labels.insert( labels.end(), neg_lst.size(), -1 );
  CV_Assert( old < labels.size() );

  cout << "Computing HOG for positive samples..." << endl;
  compute_hog( pos_lst, gradient_lst, win_size );
  cout << "Computing HOG for negative samples..." << endl;
  compute_hog( neg_lst, gradient_lst, win_size );

  cout << "Training..." << endl;
  train_svm( gradient_lst, labels, output_file );

  pos_lst.clear();
  full_neg_lst.clear();
  neg_lst.clear();
  gradient_lst.clear();
  labels.clear();
  }

  cout << "Testing..." << endl;
  test_it( output_file, video_source, win_size );

  return 0;
}
