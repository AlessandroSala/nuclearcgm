#pragma once
#include <Eigen/Dense>
#include "grid.hpp"

namespace Operators {
    Eigen::VectorXcd P(const Eigen::VectorXcd& x, const Grid& grid);
}