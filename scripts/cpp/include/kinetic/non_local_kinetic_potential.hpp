
#pragma once
#include "constants.hpp"
#include "potential.hpp"
#include <Eigen/Dense>
#include <memory>
class IterationData;
/**
 * @brief Implements a non local kinetic energy term, related to the Skyrme
 * interaction.
 */
class NonLocalKineticPotential : public Potential {
public:
  /**
   * @brief Implements a non local kinetic energy term, related to the Skyrme
   * interaction.
   * @param m Mass instance
   */
  NonLocalKineticPotential(std::shared_ptr<IterationData> d, NucleonType n);

  double getValue(double x, double y, double z) const override;
  Eigen::VectorXd getFactor(double x, double y, double z) const;
  std::complex<double> getElement(int i, int j, int k, int s, int i1, int j1,
                                  int k1, int s1, const Grid &grid) const;
  std::complex<double> getElement5p(int i, int j, int k, int s, int i1, int j1,
                                    int k1, int s1,
                                    const Grid &grid) const override;

public:
  std::shared_ptr<IterationData> data;
  NucleonType nucleon;
};
