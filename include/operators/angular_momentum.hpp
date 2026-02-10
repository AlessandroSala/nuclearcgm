#pragma once
#include <Eigen/Dense>
#include "grid.hpp"

namespace Operators {
    Eigen::VectorXcd Jz(const Eigen::VectorXcd& x, const Grid& grid);
    Eigen::VectorXcd Lz(const Eigen::VectorXcd& x, const Grid& grid);
    Eigen::VectorXcd Ly(const Eigen::VectorXcd& x, const Grid& grid);
    Eigen::VectorXcd Lx(const Eigen::VectorXcd& x, const Grid& grid);
    Eigen::VectorXcd Sz(const Eigen::VectorXcd& x, const Grid& grid);
    Eigen::VectorXcd J(const Eigen::VectorXcd& x, const Grid& grid);
    Eigen::VectorXcd S2(const Eigen::VectorXcd& x, const Grid& grid);
    Eigen::VectorXcd L2(const Eigen::VectorXcd& x, const Grid& grid);
    Eigen::VectorXcd LS(const Eigen::VectorXcd& x, const Grid& grid);
    Eigen::VectorXcd J2(const Eigen::VectorXcd& x, const Grid& grid);
    double JzExp(const Eigen::VectorXcd& x, const Grid& grid);
    
}