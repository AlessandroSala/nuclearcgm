#pragma once
#include "input_parser.hpp"
#include <Eigen/Dense>
class IterationData;
class Constraint;
class Output {
private:
  std::string folder;

public:
  Output();
  Output(std::string folder_);
  double x2(IterationData *data, const Grid &grid, char dir);
  void swapAxes(Eigen::VectorXd &vec, int axis1, int axis2);
  void matrixToFile(std::string fileName, Eigen::MatrixXd matrix);
  void
  shellsToFile(std::pair<Eigen::MatrixXcd, Eigen::VectorXd> neutronShells,
               std::pair<Eigen::MatrixXcd, Eigen::VectorXd> protonShells,
               IterationData *iterationData, InputParser &input, int iterations,
               std::vector<double> energies, std::vector<double> HFEnergies,
               double cpuTime, char mode,
               const std::vector<std::unique_ptr<Constraint>> &constraints);
};
