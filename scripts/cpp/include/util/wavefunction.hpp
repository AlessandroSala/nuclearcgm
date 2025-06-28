#pragma once
#include "grid.hpp"
#include "input_parser.hpp"
#include "math.h"
#include <Eigen/Dense>

namespace Wavefunction {
Eigen::VectorXd density(const Eigen::MatrixXcd &psi, const Grid &grid);
Eigen::VectorXd field(Eigen::VectorXd &rho, Eigen::VectorXd &rhoN,
                      const Grid &grid, SkyrmeParameters params);
Eigen::Matrix3Xd spinDensity(const Eigen::MatrixXcd &psi, const Grid &grid);
Eigen::VectorXd kineticDensity(const Eigen::MatrixXcd &psi, const Grid &grid);
Eigen::Matrix<double, Eigen::Dynamic, 9> soDensity(const Eigen::MatrixXcd &psi,
                                                   const Grid &grid);
void normalize(Eigen::MatrixXcd &psi, const Grid &grid);
std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXcd>
hfVectors(const Eigen::MatrixXcd &psi, const Grid &grid);
void printShells(std::pair<Eigen::MatrixXcd, Eigen::VectorXd> levels,
                 const Grid &grid);
void printShellsToFile(std::pair<Eigen::MatrixXcd, Eigen::VectorXd> levels,
                       const Grid &grid, std::ofstream &stream);
double roundHI(double x);
double momFromSq(double x);
} // namespace Wavefunction
