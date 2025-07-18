#pragma once
#include "local_potential.hpp"
#include <Eigen/Dense>
#include <memory>

class LocalCoulombPotential : public Potential {

public:
  LocalCoulombPotential(std::shared_ptr<Eigen::VectorXd> field);
  double getValue(double x, double y, double z) const override;

  std::complex<double> getElement5p(int i, int j, int k, int s, int i1, int j1,
                                    int k1, int s1,
                                    const Grid &grid) const override;

private:
  std::shared_ptr<Eigen::VectorXd> field;
};
