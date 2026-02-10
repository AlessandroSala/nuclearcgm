#pragma once
#include "potential.hpp"
#include "util/iteration_data.hpp"
class LocalPotential : public Potential {
public:
  /**
   * @brief Base class for potentials which are local in the Hamiltonian matrix
   */

  virtual double getValue(double x, double y, double z) const = 0;
  std::complex<double> getElement(int i, int j, int k, int s, int i1, int j1,
                                  int k1, int s1, const Grid &grid) const {
    if (i != i1 || j != j1 || k != k1 || s != s1) {
      return std::complex<double>(0.0, 0.0);
    }
    return getValue(grid.get_xs()[i], grid.get_ys()[j], grid.get_zs()[k]);
  };
  std::complex<double> getElement5p(int i, int j, int k, int s, int i1, int j1,
                                    int k1, int s1, const Grid &grid) const {
    if (i != i1 || j != j1 || k != k1 || s != s1) {
      return std::complex<double>(0.0, 0.0);
    }
    return std::complex<double>(
        getValue(grid.get_xs()[i], grid.get_ys()[j], grid.get_zs()[k]), 0.0);
  };

public:
  virtual ~LocalPotential() = default;
};
