#include "skyrme/multipole_constraint.hpp"
#include "operators/integral_operators.hpp"
#include "spherical_harmonics.hpp"
#include "util/fields.hpp"
#include "util/iteration_data.hpp"

MultipoleConstraint::MultipoleConstraint(double target_, int l_, int m_,
                                         IterationData *data_)
    : lambda(0.0), l(l_), m(m_) {

  value = 0.0;
  target = target_;
  C = 1.0;
  double en = evaluate(data_);
  C = 0.01 / en;
}

Eigen::VectorXd MultipoleConstraint::getField(IterationData *data) {

  using Operators::integral;

  auto grid = *Grid::getInstance();
  Eigen::VectorXd rho = *(data->rhoN) + *(data->rhoP);

  Eigen::VectorXd r_l = Fields::position().array().abs().pow(l);

  Eigen::VectorXd O = SphericalHarmonics::X(l, m).cwiseProduct(r_l);

  double Q_exp = integral((Eigen::VectorXd)(O.array() * rho.array()), grid);

  double gamma = 0.3;

  lambda += gamma * 2.0 * C * (Q_exp - target);

  double mu = target - lambda / (2.0 * C);

  value = Q_exp;

  return 2.0 * C * (Q_exp - mu) * O;
}

double MultipoleConstraint::evaluate(IterationData *data) const {
  using Operators::integral;

  Eigen::VectorXd rho = *(data->rhoN) + *(data->rhoP);
  auto grid = *Grid::getInstance();

  Eigen::VectorXd r_l = Fields::position().array().abs().pow(l);

  Eigen::VectorXd O = SphericalHarmonics::X(l, m).cwiseProduct(r_l);

  double Q_exp = integral((Eigen::VectorXd)(O.array() * rho.array()), grid);

  double Qc = integral((Eigen::VectorXd)(O.array() * rho.array()), grid);
  return C * pow(Qc - target, 2) + lambda * (Qc - target);
}
