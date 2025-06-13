#pragma once
#include <Eigen/Dense>
#include <complex>
#include "grid.hpp"

namespace Operators {
    Eigen::VectorXcd dv(const Eigen::VectorXcd& psi, const Grid& grid, char dir);
    Eigen::VectorXcd dvNoSpin(const Eigen::VectorXcd& psi, const Grid& grid, char dir);
    Eigen::VectorXd dvNoSpin(const Eigen::VectorXd& psi, const Grid& grid, char dir);
    Eigen::VectorXcd divNoSpin(const Eigen::MatrixXcd& psi, const Grid& grid);
    Eigen::Matrix<double, Eigen::Dynamic, 3> gradNoSpin(const Eigen::VectorXd& psi, const Grid& grid);
    Eigen::MatrixX3cd grad(const Eigen::VectorXcd& psi, const Grid& grid);
    std::complex<double> derivative(const Eigen::VectorXcd& psi, int i, int j, int k, int s, const Grid& grid, char axis); 
    std::complex<double> derivativeNoSpin(const Eigen::VectorXcd& psi, int i, int j, int k, const Grid& grid, char axis); 
    double derivativeNoSpin(const Eigen::VectorXd& psi, int i, int j, int k, const Grid& grid, char axis); 
    std::complex<double> derivative2(const Eigen::VectorXcd& psi, int i, int j, int k, int s, const Grid& grid, char axis); 
    
}