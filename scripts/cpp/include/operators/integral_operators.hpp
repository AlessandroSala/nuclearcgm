#pragma once
#include <Eigen/Dense>
#include <complex>
#include "grid.hpp"

namespace Operators {
    std::complex<double> integral(const Eigen::VectorXcd& psi, const Grid& grid);
}