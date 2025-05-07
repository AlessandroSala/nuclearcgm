#include <Eigen/Dense>
#include "grid.hpp"

namespace Operators {
    double dvx(const Eigen::VectorXd& psi, int i, const Grid& grid);
}