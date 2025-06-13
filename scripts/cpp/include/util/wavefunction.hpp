#pragma once
#include <Eigen/Dense>
#include "grid.hpp"
#include "math.h"
namespace Wavefunction {
    Eigen::VectorXd density(const Eigen::MatrixXcd& psi, const Grid& grid);
    Eigen::VectorXd kineticDensity(const Eigen::MatrixXcd& psi, const Grid& grid);
    Eigen::MatrixXcd soDensity(const Eigen::MatrixXcd& psi, const Grid& grid);
    void normalize(Eigen::MatrixXcd& psi, const Grid& grid);
    std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXcd> hfVectors(const Eigen::MatrixXcd& psi, const Grid& grid);
    void printShells(std::pair<Eigen::MatrixXcd, Eigen::VectorXd> levels, const Grid& grid);
    double roundHI(double x);
    double momFromSq(double x);
}
    
