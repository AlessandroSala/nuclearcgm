#pragma once
#include "input_parser.hpp"
#include <Eigen/Dense>
class IterationData;
class Output
{
private:
  std::string folder;

public:
  Output();
  Output(std::string folder_);
  double x2(std::shared_ptr<IterationData> data, const Grid &grid, char dir);
  void swapAxes(Eigen::VectorXd &vec, int axis1, int axis2);
  void matrixToFile(std::string fileName, Eigen::MatrixXd matrix);
  void shellsToFile(std::string fileName,
                    std::pair<Eigen::MatrixXcd, Eigen::VectorXd> neutronShells,
                    std::pair<Eigen::MatrixXcd, Eigen::VectorXd> protonShells,
                    std::shared_ptr<IterationData> iterationData,
                    InputParser input, int iterations,
                    std::vector<double> energies, double cpuTime,
                    const Grid &grid);
};
