#pragma once
#include "functional_term.hpp"
#include <Eigen/Dense>
class Constraint : public FunctionalTerm {
public:
  double evaluate(IterationData *data) const override;
  virtual double error() const;
  virtual Eigen::VectorXd getField(IterationData *data);
  virtual ~Constraint() = default;

  double value;
  double target;
  double lambda;
  double gamma;
};
