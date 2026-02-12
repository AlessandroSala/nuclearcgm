#include "skyrme/ycm_constraint.hpp"
#include "operators/integral_operators.hpp"
#include "spherical_harmonics.hpp"
#include "util/fields.hpp"
#include "util/iteration_data.hpp"
#include <iostream>

YCMConstraint::YCMConstraint(double target_)
    : target(target_), C(0.05), lambda(0.0), firstIter(true) {
  value = 0.0;
  residuals.clear();
}
double YCMConstraint::error() const { return 0.0; }
Eigen::VectorXd YCMConstraint::getField(IterationData *data) {
  using Eigen::VectorXd;
  using Operators::integral;
  auto grid = *Grid::getInstance();

  VectorXd O(grid.get_total_spatial_points());
  for (int k = 0; k < grid.get_n(); ++k) {
    for (int j = 0; j < grid.get_n(); ++j) {
      for (int i = 0; i < grid.get_n(); ++i) {
        int idx = grid.idxNoSpin(i, j, k);
        O(idx) = grid.get_ys()[j];
      }
    }
  }

  Eigen::VectorXd rho = *(data->rhoN) + *(data->rhoP);
  double Q22 = 0.0;

  Q22 += integral((VectorXd)(O.array() * rho.array()), grid);

  // std::cout << "q: " << Q22 << std::endl;
  // std::cout << "mu20: " << mu20 << std::endl;

  auto constraintEnergy = evaluate(data);
  // std::cout << "Constraint energy: " << constraintEnergy << std::endl;

  double mu20 = target;
  if (firstIter) {
    firstIter = false;
    // return Eigen::VectorXd::Zero(data->rhoN->rows());
    return 2.0 * C * (Q22 - mu20) * O;
  }
  double gamma = 0.2;

  // if(residuals.size() > 1 &&
  // std::abs(residuals.back()/residuals[residuals.size()-2]-1) < 1e-1) {
  if (data->energyDiff < data->constraintTol) {
    lambda += gamma * 2.0 * C * (Q22 - mu20);
    // std::cout << "Updated lambda: " << lambda << std::endl;
  }
  double mu = mu20 - lambda / (2.0 * C);
  double alpha = lambda + 2.0 * C * (Q22 - mu20);

  double residual = (Q22 - mu20);
  value = Q22;

  residuals.push_back(residual);

  // for(int i = 0; i < residuals.size(); ++i) {
  //     std::cout << residuals[i] << ", ";
  // }
  // std::cout << std::endl;

  return 2.0 * C * (Q22 - mu) * O;
}

double YCMConstraint::evaluate(IterationData *data) const {
  using Eigen::VectorXd;
  using Operators::integral;
  auto grid = *Grid::getInstance();

  Eigen::VectorXd rho = *(data->rhoN) + *(data->rhoP);
  Eigen::VectorXd O(grid.get_total_spatial_points());
  for (int k = 0; k < grid.get_n(); ++k) {
    for (int j = 0; j < grid.get_n(); ++j) {
      for (int i = 0; i < grid.get_n(); ++i) {
        int idx = grid.idxNoSpin(i, j, k);
        O(idx) = grid.get_ys()[j];
      }
    }
  }
  double Qc = integral((VectorXd)(O.array() * rho.array()), grid);

  double mu20 = target;
  return C * pow(Qc - mu20, 2) + lambda * (Qc - mu20);
}
