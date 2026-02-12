#include "skyrme/x2my2_constraint.hpp"
#include "operators/integral_operators.hpp"
#include "spherical_harmonics.hpp"
#include "util/fields.hpp"
#include "util/iteration_data.hpp"
#include <iostream>

X2MY2Constraint::X2MY2Constraint(double mu20)
    : target(mu20), C(0.005), lambda(0.0), firstIter(true) {
  value = 0.0;
  residuals.clear();
}
Eigen::VectorXd X2MY2Constraint::getField(IterationData *data) {
  using Eigen::VectorXd;
  using Operators::integral;
  auto grid = *Grid::getInstance();

  VectorXd x(grid.get_total_spatial_points());
  VectorXd y(grid.get_total_spatial_points());

  for (int k = 0; k < grid.get_n(); ++k) {
    for (int j = 0; j < grid.get_n(); ++j) {
      for (int i = 0; i < grid.get_n(); ++i) {
        int idx = grid.idxNoSpin(i, j, k);
        x(idx) = grid.get_xs()[i];
        y(idx) = grid.get_ys()[j];
      }
    }
  }

  Eigen::VectorXd rho = *(data->rhoN) + *(data->rhoP);
  Eigen::VectorXd O = x.array().pow(2) - y.array().pow(2);
  double Q22 = integral((VectorXd)(O.array() * rho.array()), grid);
  // std::cout << "XCM: " << integral((VectorXd)(x.array()*rho.array()), grid)
  // << std::endl; std::cout << "YCM: " <<
  // integral((VectorXd)(y.array()*rho.array()), grid) << std::endl;

  // std::cout << "q: " << Q22 << std::endl;
  // std::cout << "mu20: " << mu20 << std::endl;

  auto constraintEnergy = evaluate(data);
  std::cout << "X2mY2 Constraint energy: " << constraintEnergy << std::endl;

  double mu20 = target;
  if (firstIter) {
    firstIter = false;
    // return Eigen::VectorXd::Zero(data->rhoN->rows());
    return 2.0 * C * (Q22 - mu20) * O;
  }
  double gamma = 0.3;

  // if(residuals.size() > 1 &&
  // std::abs(residuals.back()/residuals[residuals.size()-2]-1) < 1e-2) {
  // if(data->energyDiff < data->constraintTol) {
  std::cout << "Updated lambda X^2-Y^2, previous: " << lambda;
  lambda += gamma * 2.0 * C * (Q22 - mu20);
  std::cout << ", new: " << lambda << std::endl;
  //}
  double mu = mu20 - lambda / (2.0 * C);
  double alpha = lambda + 2.0 * C * (Q22 - mu20);

  double residual = (Q22 - mu20);

  residuals.push_back(residual);
  value = Q22;

  // for(int i = 0; i < residuals.size(); ++i) {
  //     std::cout << residuals[i] << ", ";
  // }
  // std::cout << std::endl;

  return 2.0 * C * (Q22 - mu) * O;
}

double X2MY2Constraint::evaluate(IterationData *data) const {
  using Eigen::VectorXd;
  using Operators::integral;
  auto grid = *Grid::getInstance();

  VectorXd x(grid.get_total_spatial_points());
  VectorXd y(grid.get_total_spatial_points());

  for (int k = 0; k < grid.get_n(); ++k) {
    for (int j = 0; j < grid.get_n(); ++j) {
      for (int i = 0; i < grid.get_n(); ++i) {
        int idx = grid.idxNoSpin(i, j, k);
        x(idx) = grid.get_xs()[i];
        y(idx) = grid.get_ys()[j];
      }
    }
  }

  Eigen::VectorXd rho = *(data->rhoN) + *(data->rhoP);
  Eigen::VectorXd O = x.array().pow(2) - y.array().pow(2);
  double Qc = integral((VectorXd)(O.array() * rho.array()), grid);

  double mu20 = target;
  return C * pow(Qc - mu20, 2) + lambda * (Qc - mu20);
}
