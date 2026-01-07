#pragma once
#include "grid.hpp"
#include <Eigen/Dense>
#include <complex>

namespace Operators {
Eigen::VectorXcd dv(const Eigen::VectorXcd &psi, const Grid &grid, char dir);
Eigen::VectorXcd dv2(const Eigen::VectorXcd &psi, const Grid &grid, char dir);
Eigen::VectorXcd dvNoSpin(const Eigen::VectorXcd &psi, const Grid &grid,
                          char dir);
Eigen::VectorXd dvNoSpin(const Eigen::VectorXd &psi, const Grid &grid,
                         char dir);
Eigen::VectorXd dv2NoSpin(const Eigen::VectorXd &psi, const Grid &grid,
                          char dir);
Eigen::VectorXcd divNoSpin(const Eigen::MatrixX3cd &psi, const Grid &grid);
Eigen::VectorXd divNoSpin(const Eigen::MatrixX3d &psi, const Grid &grid);
Eigen::Matrix<double, Eigen::Dynamic, 3> gradNoSpin(const Eigen::VectorXd &psi,
                                                    const Grid &grid);
Eigen::MatrixX3cd grad(const Eigen::VectorXcd &psi, const Grid &grid);
Eigen::MatrixX3cd gradLagrange(const Eigen::VectorXcd &psi);

Eigen::MatrixX3d gradLagrangeNoSpin(const Eigen::VectorXd &psi);

Eigen::VectorXd lapLagrangeNoSpin(const Eigen::VectorXd &psi);

std::complex<double> derivative(const Eigen::VectorXcd &psi, int i, int j,
                                int k, int s, const Grid &grid, char axis);

std::complex<double> derivativeNoSpin(const Eigen::VectorXcd &psi, int i, int j,
                                      int k, const Grid &grid, char axis);
double derivativeNoSpin(const Eigen::VectorXd &psi, int i, int j, int k,
                        const Grid &grid, char axis);
double derivative2NoSpin(const Eigen::VectorXd &psi, int i, int j, int k,
                         const Grid &grid, char axis);
std::complex<double> derivative2(const Eigen::VectorXcd &psi, int i, int j,
                                 int k, int s, const Grid &grid, char axis);
Eigen::VectorXd lapNoSpin(const Eigen::VectorXd &psi, const Grid &grid);
Eigen::VectorXcd lap(const Eigen::VectorXcd &psi, const Grid &grid);

} // namespace Operators
