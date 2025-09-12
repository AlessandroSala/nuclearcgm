#include "constraint.hpp"

Eigen::VectorXd Constraint::getField(IterationData* data)  {
    throw std::runtime_error("Constraint::getField not implemented");
}

double Constraint::evaluate(IterationData* data) const  {
    throw std::runtime_error("Constraint::evaluate not implemented");
}
