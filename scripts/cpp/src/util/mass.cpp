#include "util/mass.hpp"
#include <cmath>
#include "constants.hpp"
#include "operators/differential_operators.hpp"
#include <iostream>

Mass::Mass(Eigen::VectorXd rho, std::shared_ptr<Grid> grid_ptr, double t1, double t2)
     {
        using namespace nuclearConstants;
        mVec = m/(1 + 2*m/(h_bar*h_bar)*(3*t1 + 5*t2)*rho.array());
        grid_ptr_ = grid_ptr;
    }

double Mass::getMass(size_t i, size_t j, size_t k) const noexcept {
    return mVec(grid_ptr_->idxNoSpin(i, j, k));
}
Eigen::Vector3d Mass::getGradient(size_t i, size_t j, size_t k) const noexcept {
    Eigen::Vector3d grad(3);
    grad(0) = Operators::derivativeNoSpin(mVec, i, j, k, *grid_ptr_, 'x');
    grad(1) = Operators::derivativeNoSpin(mVec, i, j, k, *grid_ptr_, 'y');
    grad(2) = Operators::derivativeNoSpin(mVec, i, j, k, *grid_ptr_, 'z');
    return grad;
}
