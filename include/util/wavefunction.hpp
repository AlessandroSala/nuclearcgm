#pragma once
#include "grid.hpp"
#include "input_parser.hpp"
#include "math.h"
#include <Eigen/Dense>

class EDF;

namespace Wavefunction {
Eigen::VectorXd density(const Eigen::MatrixXcd &psi, const Grid &grid);

Eigen::MatrixXcd timeReverse(const Eigen::MatrixXcd &psi);
Eigen::MatrixXcd TROrder(Eigen::MatrixXcd &psi);

Eigen::VectorXd coulombField(Eigen::VectorXd &rho, const Grid &grid);
Eigen::VectorXd coulombFieldPoisson(Eigen::VectorXd &rho, const Grid &grid,
                                    int Z, std::shared_ptr<Eigen::VectorXd> U);

Eigen::VectorXd exchangeCoulombField(Eigen::VectorXd &rho, const Grid &grid);
Eigen::VectorXd field(Eigen::VectorXd &rho, Eigen::VectorXd &rhoQ,
                      Eigen::VectorXd &tau, Eigen::VectorXd &tauQ,
                      Eigen::VectorXd &nabla2rho, Eigen::VectorXd &nabla2rhoQ,
                      Eigen::VectorXd &divJ, Eigen::VectorXd &divJQ,
                      const Grid &grid, std::shared_ptr<EDF> interaction);
Eigen::Matrix3Xd spinDensity(const Eigen::MatrixXcd &psi, const Grid &grid);

Eigen::VectorXd kineticDensity(const Eigen::MatrixXcd &psi, const Grid &grid);
Eigen::VectorXd kineticDensityLagrange(const Eigen::MatrixXcd &psi);

Eigen::Matrix<double, Eigen::Dynamic, 9> soDensity(const Eigen::MatrixXcd &psi,
                                                   const Grid &grid);
Eigen::Matrix<double, Eigen::Dynamic, 9>
soDensityLagrange(const Eigen::MatrixXcd &psi);

Eigen::VectorXd divJ(const Eigen::MatrixXcd &psi, const Grid &grid);
void normalize(Eigen::MatrixXcd &psi, const Grid &grid);

void printShells(std::pair<Eigen::MatrixXcd, Eigen::VectorXd> levels,
                 const Grid &grid);
void printShellsToFile(std::pair<Eigen::MatrixXcd, Eigen::VectorXd> levels,
                       const Grid &grid, std::ofstream &stream);
double roundHI(double x);
double momFromSq(double x);
} // namespace Wavefunction
