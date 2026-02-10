#include "constraint.hpp"

double Constraint::error() const {

  if (std::abs(target) <= 1e-12) {
    return std::abs(value);
  } else {
    return std::abs(value / target - 1.0);
  }
}

Eigen::VectorXd Constraint::getField(IterationData *data) {
  throw std::runtime_error("Constraint::getField not implemented");
}

double Constraint::evaluate(IterationData *data) const {
  throw std::runtime_error("Constraint::evaluate not implemented");
}
