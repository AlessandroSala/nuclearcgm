#pragma once
#include "constraint.hpp"
#include <Eigen/Dense>

class XCMConstraint : public Constraint {
public:
  XCMConstraint(double mu20);
  Eigen::VectorXd getField(IterationData *data) override;
  double evaluate(IterationData *data) const override;
  double error() const override;

  double target;
  double lambda;

private:
  double C;
  bool firstIter;
  std::vector<double> residuals;
};
