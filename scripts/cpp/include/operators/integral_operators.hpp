#pragma once
#include "grid.hpp"
#include <Eigen/Dense>
#include <complex>

namespace Operators {
std::complex<double> integral(const Eigen::VectorXcd &psi, const Grid &grid);
std::complex<double> integralNoSpin(const Eigen::VectorXcd &psi, const Grid &grid);
double integral(const Eigen::VectorXd &psi, const Grid &grid);
} // namespace Operators
