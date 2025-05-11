#include <Eigen/Dense>
#include "grid.hpp"

namespace Operators {
    Eigen::VectorXcd Jz(const Eigen::VectorXcd& x, const Grid& grid);
    Eigen::VectorXcd J(const Eigen::VectorXcd& x, const Grid& grid);
    double JzExp(const Eigen::VectorXcd& x, const Grid& grid);
    
}