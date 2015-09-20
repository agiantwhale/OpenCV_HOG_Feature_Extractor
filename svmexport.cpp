/*
 * =====================================================================================
 *
 *       Filename:  svmexport.cpp
 *
 *    Description:  Exports trained SVM to a single detecting vector.
 *
 *        Version:  1.0
 *        Created:  2015/09/19 20시 39분 01초
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
#include <boost/filesystem.hpp>
#include "libsvm/svm.h"

int main(int argc, char** argv) {
  std::string source_file;
  std::string output_file;
  try {
    namespace po=boost::program_options;
    po::options_description desc("Options");
    desc.add_options()
    ("help,h", "Print help messages")
    ("source,o", po::value<std::string>(&source_file)->required(), "Specify an source file")
    ("output,o", po::value<std::string>(&output_file)->default_value(boost::filesystem::current_path().string<std::string>()+"/detector.data"), "Specify an output file");

    po::positional_options_description p;
    p.add("source",-1);

    po::variables_map vm;
    po::store(po::command_line_parser(argc,argv).options(desc).positional(p).run(), vm);

    if (vm.count("help")) {
      std::cout << "Usage: " << argv[0] << " [options] source" << std::endl;
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
  std::vector<unsigned int> single_detector_vector_indices;

  svm_model *model=svm_load_model(source_file.c_str());

  for (int ssv = 0; ssv < model->l; ++ssv) {
    svm_node* single_support_vector = model->SV[ssv];
    double alpha = model->sv_coef[0][ssv];
    int single_vector_component = 0;
    while (single_support_vector[single_vector_component].index != -1) {
      if (ssv == 0) {
        single_detector_vector.push_back(single_support_vector[single_vector_component].value * alpha);
        single_detector_vector_indices.push_back(single_support_vector[single_vector_component].index); // Holds the indices for the corresponding values in single_detector_vector, mapping from single_vector_component to single_support_vector[single_vector_component].index!
      } else {
        if (single_vector_component > (int)single_detector_vector.size()) { // Catch oversized vectors (maybe from differently sized images?)
          std::cerr << "Warning: Component " << single_vector_component << " out of range, should have the same size as other/first vector" << std::endl;
        } else single_detector_vector.at(single_vector_component) += (single_support_vector[single_vector_component].value * alpha);
      }
      single_vector_component++;
    }
  }

  std::ofstream result_data;
  result_data.open(output_file, std::ios::out|std::ios::app);

  for(auto iter=single_detector_vector.begin(); iter!=single_detector_vector.end(); iter++) {
    result_data << *iter << std::endl;
  }

  svm_free_and_destroy_model(&model);
}
