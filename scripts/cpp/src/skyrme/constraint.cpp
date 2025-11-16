#include "constraint.hpp"

double Constraint::error() const {
  if (target == 0.0) {
    return 0.0;
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
