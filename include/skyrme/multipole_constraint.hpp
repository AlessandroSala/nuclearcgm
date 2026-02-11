#pragma once
#include "constraint.hpp"
#include <Eigen/Dense>

class MultipoleConstraint : public Constraint {
public:
  MultipoleConstraint(double mu20, int l, int m, IterationData *data);

  Eigen::VectorXd getField(IterationData *data) override;

  double evaluate(IterationData *data) const override;

  double lambda;
  int l, m;

private:
  double C;
};
