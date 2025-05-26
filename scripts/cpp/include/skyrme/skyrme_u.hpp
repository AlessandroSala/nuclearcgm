
#pragma once
#include "local_potential.hpp"
#include <Eigen/Dense>
#include <memory>
/**
 */
class SkyrmeU : public Potential {
public:
  /**
   */
  SkyrmeU(std::shared_ptr<Eigen::VectorXd> rho,
          std::shared_ptr<Eigen::VectorXd> nabla2rho,
          std::shared_ptr<Eigen::VectorXd> tau,
          std::shared_ptr<Eigen::VectorXcd> divJ, double t0, double t1,
          double t2, double t3, double W0);
  double getValue(double x, double y, double z) const override;
  std::complex<double> getElement(int i, int j, int k, int s, int i1, int j1,
                                  int k1, int s1,
                                  const Grid &grid) const override;
  std::complex<double> getElement5p(int i, int j, int k, int s, int i1, int j1,
                                    int k1, int s1,
                                    const Grid &grid) const override;

public:
  double t0, t1, t2, t3, W0;
  std::shared_ptr<Eigen::VectorXd> rho;
  std::shared_ptr<Eigen::VectorXd> nabla2rho;
  std::shared_ptr<Eigen::VectorXd> tau;
  std::shared_ptr<Eigen::VectorXcd> divJ;
};
