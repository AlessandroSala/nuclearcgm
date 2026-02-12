#include "skyrme/quadrupole_constraint.hpp"
#include "operators/integral_operators.hpp"
#include "spherical_harmonics.hpp"
#include "util/fields.hpp"
#include "util/iteration_data.hpp"
#include <iostream>

QuadrupoleConstraint::QuadrupoleConstraint(double target_)
    : C(0.005), lambda(0.0), firstIter(true) {

  value = 0.0;
  target = target_;

  residuals.clear();
}
QuadrupoleConstraint::QuadrupoleConstraint(double target_, double lambda_)
    : C(0.001), lambda(lambda_), firstIter(true) {
  target = target_;
  residuals.clear();
}
Eigen::VectorXd QuadrupoleConstraint::getField(IterationData *data) {

  using Operators::integral;

  auto grid = *Grid::getInstance();
  Eigen::VectorXd rho = *(data->rhoN) + *(data->rhoP);

  Eigen::VectorXd x(grid.get_total_spatial_points()),
      y(grid.get_total_spatial_points()), z(grid.get_total_spatial_points());
  for (int k = 0; k < grid.get_n(); ++k) {
    for (int j = 0; j < grid.get_n(); ++j) {
      for (int i = 0; i < grid.get_n(); ++i) {
        int idx = grid.idxNoSpin(i, j, k);
        x(idx) = grid.get_xs()[i];
        y(idx) = grid.get_ys()[j];
        z(idx) = grid.get_zs()[k];
      }
    }
  }
  Eigen::VectorXd O =
      0.25 * sqrt(5 / (M_PI)) *
      (2.0 * z.array().pow(2) - y.array().pow(2) - x.array().pow(2));

  // double Q20 = SphericalHarmonics::Q(2, 0, rho).real();
  // Eigen::VectorXd Y20 = SphericalHarmonics::Y(2, 0).real();
  // Eigen::VectorXd pos = Fields::position().array().abs().pow(2);
  // Eigen::VectorXd O = Y20.array() * pos.array();
  double Q20 = integral((Eigen::VectorXd)(O.array() * rho.array()), grid);

  std::cout << "q: " << Q20 << std::endl;
  std::cout << "target: " << target << std::endl;

  std::cout << "Constraint energy: " << evaluate(data) << std::endl;
  if (firstIter) {
    firstIter = false;
    // return Eigen::VectorXd::Zero(data->rhoN->rows());
    double mu = target - lambda / (2.0 * C);
    return 2.0 * C * (Q20 - mu) * O;
  }

  double gamma = 0.3;

  // if(residuals.size() > 1 &&
  // std::abs(residuals.back()/residuals[residuals.size()-2]-1) < 1e-2) {
  // if(data->energyDiff < data->constraintTol) {
  std::cout << "Updated lambda Q20, previous: " << lambda;
  lambda += gamma * 2.0 * C * (Q20 - target);
  std::cout << ", new: " << lambda << std::endl;
  //}
  double mu = target - lambda / (2.0 * C);
  double alpha = lambda + 2.0 * C * (Q20 - target);

  double residual = (Q20 - target);

  residuals.push_back(residual);
  value = Q20;
  std::cout << "New residual: " << residuals.back();

  // for(int i = 0; i < residuals.size(); ++i) {
  //     std::cout << residuals[i] << ", ";
  // }

  std::cout << std::endl;

  return 2.0 * C * (Q20 - mu) * O;
}

double QuadrupoleConstraint::evaluate(IterationData *data) const {
  Eigen::VectorXd rho = *(data->rhoN) + *(data->rhoP);
  auto grid = *Grid::getInstance();
  using Operators::integral;

  Eigen::VectorXd x(grid.get_total_spatial_points()),
      y(grid.get_total_spatial_points()), z(grid.get_total_spatial_points());
  for (int i = 0; i < grid.get_n(); ++i) {
    for (int j = 0; j < grid.get_n(); ++j) {
      for (int k = 0; k < grid.get_n(); ++k) {
        int idx = grid.idxNoSpin(i, j, k);
        x(idx) = grid.get_xs()[i];
        y(idx) = grid.get_ys()[j];
        z(idx) = grid.get_zs()[k];
      }
    }
  }
  Eigen::VectorXd O =
      0.25 * sqrt(5 / (M_PI)) *
      (2.0 * z.array().pow(2) - y.array().pow(2) - x.array().pow(2));

  double Qc = integral((Eigen::VectorXd)(O.array() * rho.array()), grid);
  return C * pow(Qc - target, 2) + lambda * (Qc - target);
}
