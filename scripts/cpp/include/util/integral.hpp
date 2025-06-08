#pragma once
#include "grid.hpp"
#include <Eigen/Dense>

namespace Integral {
    double wholeSpace(const Eigen::VectorXd& f, const Grid& grid);
}