#pragma once
#include "grid.hpp"
#include "types.hpp"
#include <Eigen/Dense>

namespace Operators {
Eigen::VectorXcd P(const Eigen::VectorXcd &x, const Grid &grid);
Eigen::MatrixX3d leviCivita(Real2Tensor x);
Eigen::VectorXd dot(Eigen::MatrixX3d x, Eigen::MatrixX3d y);
Eigen::VectorXd mod2(Eigen::MatrixX3d x);
} // namespace Operators
