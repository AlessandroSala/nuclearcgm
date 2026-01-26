#include "skyrme/octupole_constraint.hpp"
#include "operators/integral_operators.hpp"
#include "spherical_harmonics.hpp"
#include "util/fields.hpp"
#include "util/iteration_data.hpp"
#include <iostream>

OctupoleConstraint::OctupoleConstraint(double mu20)
    : mu20(mu20), C(0.001), lambda(0.0), firstIter(true) {
  value = 0.0;
  target = mu20;
  residuals.clear();
}
Eigen::VectorXd OctupoleConstraint::getField(IterationData *data) {

  using Operators::integral;

  auto grid = *Grid::getInstance();
  Eigen::VectorXd rho = *(data->rhoN) + *(data->rhoP);
  Eigen::VectorXd pos = Fields::position().array().abs().pow(3);
  Eigen::VectorXd O = pos.array() * SphericalHarmonics::Y(3, 0).real().array();

  // double Q20 = SphericalHarmonics::Q(2, 0, rho).real();
  // Eigen::VectorXd Y20 = SphericalHarmonics::Y(2, 0).real();
  // Eigen::VectorXd pos = Fields::position().array().abs().pow(2);
  // Eigen::VectorXd O = Y20.array() * pos.array();
  double Q20 = integral((Eigen::VectorXd)(O.array() * rho.array()), grid);

  std::cout << "q: " << Q20 << std::endl;
  std::cout << "mu30: " << mu20 << std::endl;

  if (firstIter) {
    firstIter = false;
    // return Eigen::VectorXd::Zero(data->rhoN->rows());
    return 2.0 * C * (Q20 - mu20) * O;
  }

  double gamma = 0.3;

  lambda += gamma * 2.0 * C * (Q20 - mu20);

  double mu = mu20 - lambda / (2.0 * C);

  double residual = (Q20 - mu20);

  residuals.push_back(residual);
  value = Q20;

  // for(int i = 0; i < residuals.size(); ++i) {
  //     std::cout << residuals[i] << ", ";
  // }
  // std::cout << std::endl;

  return 2.0 * C * (Q20 - mu) * O;
}

double OctupoleConstraint::evaluate(IterationData *data) const {
  Eigen::VectorXd rho = *(data->rhoN) + *(data->rhoP);
  auto grid = *Grid::getInstance();
  using Operators::integral;

  Eigen::VectorXd pos = Fields::position().array().abs().pow(3);
  Eigen::VectorXd O = pos.array() * SphericalHarmonics::Y(3, 0).real().array();

  double Qc = integral((Eigen::VectorXd)(O.array() * rho.array()), grid);
  return C * pow(Qc - mu20, 2) + lambda * (Qc - mu20);
}
