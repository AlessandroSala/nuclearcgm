#include <Eigen/Dense>
#include "grid.hpp"

namespace Operators {
    Eigen::VectorXd Jz(const Eigen::VectorXd& x, const Grid& grid);
}