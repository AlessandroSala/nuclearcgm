#pragma once
#include <Eigen/Dense>
#include <complex>
#include <memory>
#include "grid.hpp"

class Shell {
public:
    Shell(std::shared_ptr<Grid> grid_ptr, std::shared_ptr<Eigen::VectorXcd> psi_, double energy_);
    std::shared_ptr<Grid> grid;
    std::shared_ptr<Eigen::VectorXcd> psi; 

    double energy;
    double l();
    double j();
    double mj();
    double P();
};