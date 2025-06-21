#pragma once
#include "input_parser.hpp"
#include "util/iteration_data.hpp"
#include <Eigen/Dense>
class Output {
private:
  std::string folder;

public:
  Output();
  Output(std::string folder_);
  void matrixToFile(std::string fileName, Eigen::MatrixXd matrix);
  void shellsToFile(std::string fileName,
                    std::pair<Eigen::MatrixXcd, Eigen::VectorXd> neutronShells,
                    std::pair<Eigen::MatrixXcd, Eigen::VectorXd> protonShells,
                    std::shared_ptr<IterationData> iterationData,
                    InputParser input, const Grid &grid);
};
