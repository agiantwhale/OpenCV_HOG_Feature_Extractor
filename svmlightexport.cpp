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
#include "svmlight/svm_common.h"
#include "svmlight/svm_learn.h"

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

  char model_file[source_file.size()+1];
  strcpy(model_file, source_file.c_str());
  MODEL *model=read_model(model_file);

  DOC** supveclist = model->supvec;
  single_detector_vector.clear();
  single_detector_vector.resize(model->totwords, 0.);

  for (long ssv = 1; ssv < model->sv_num; ++ssv) {
    DOC* single_support_vector = supveclist[ssv];
    SVECTOR* single_support_vector_values = single_support_vector->fvec;
    WORD single_support_vector_component;
    for (long singleFeature = 0; singleFeature < model->totwords; ++singleFeature) {
      single_support_vector_component = single_support_vector_values->words[singleFeature];
      single_detector_vector.at(single_support_vector_component.wnum-1) += (single_support_vector_component.weight * model->alpha[ssv]);
    }
  }

  free_model(model,1);

  std::ofstream result_data;
  result_data.open(output_file, std::ofstream::out|std::ofstream::app);

  for(std::vector<float>::iterator iter=single_detector_vector.begin(); iter!=single_detector_vector.end(); iter++) {
    result_data << *iter << std::endl;
  }
}
