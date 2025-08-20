
#pragma once
#include "potential.hpp"
#include <Eigen/Dense>
class IterationData;
/**
 * @brief Implements an isotropic local kinetic energy term.
 */
class IsoKineticPotential : public Potential {
public:
  /**
   * @brief Implements a local kinetic energy term.
   * @param m Local mass
   */
  IsoKineticPotential();

  double getValue(double x, double y, double z) const override;
  Eigen::VectorXd getFactor(double x, double y, double z) const;
  std::complex<double> getElement(int i, int j, int k, int s, int i1, int j1,
                                  int k1, int s1, const Grid &grid) const;
  std::complex<double> getElement5p(int i, int j, int k, int s, int i1, int j1,
                                    int k1, int s1,
                                    const Grid &grid) const override;
};
