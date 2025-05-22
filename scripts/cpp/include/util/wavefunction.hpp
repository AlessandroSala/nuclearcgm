#pragma once
#include <Eigen/Dense>
#include "grid.hpp"

namespace Wavefunction {
    Eigen::VectorXd density(const Eigen::MatrixXcd& psi, const Grid& grid);
    Eigen::VectorXd kineticDensity(const Eigen::MatrixXcd& psi, const Grid& grid);
    Eigen::MatrixXcd soDensity(const Eigen::MatrixXcd& psi, const Grid& grid);
}
    
