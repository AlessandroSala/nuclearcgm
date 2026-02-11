#pragma once
#include "constraint.hpp"
#include <Eigen/Dense>

class YCMConstraint : public Constraint {
public:
  YCMConstraint(double mu20);
  double error() const override;

  Eigen::VectorXd getField(IterationData *data) override;

  double evaluate(IterationData *data) const override;

  double target;
  double lambda;

private:
  double C;
  bool firstIter;
  std::vector<double> residuals;
};
