#include "kinetic/local_kinetic_potential.hpp"
#include "constants.hpp"
#include <cmath>
#include "util/effective_mass.hpp"
#include "util/iteration_data.hpp"

LocalKineticPotential::LocalKineticPotential(std::shared_ptr<IterationData> d,
                                             NucleonType n)
    : data(d), nucleon(n) {}
double LocalKineticPotential::getValue(double x, double y, double z) const {
  return 1;
}
// TODO: not really, implement 3p derivatives
std::complex<double> LocalKineticPotential::getElement(int i, int j, int k,
                                                       int s, int i1, int j1,
                                                       int k1, int s1,
                                                       const Grid &grid) const {

  return 0;
}

std::complex<double>
LocalKineticPotential::getElement5p(int i, int j, int k, int s, int i1, int j1,
                                    int k1, int s1, const Grid &grid) const {
  double val = 0.0;
  double hh = grid.get_h() * grid.get_h();

  using nuclearConstants::h_bar;
  int idx = grid.idxNoSpin(i, j, k);

  double C = nucleon == NucleonType::N ? -data->massN->vector(idx)
                                       : -data->massP->vector(idx);

  if (i == i1 && j == j1 && k == k1 && s == s1) {
    val = -C * (90.0 / 12.0) / hh;
  } else if (s == s1 && ((i == i1 && j == j1 && std::abs(k1 - k) == 1) ||
                         (i == i1 && k == k1 && std::abs(j1 - j) == 1) ||
                         (j == j1 && k == k1 && std::abs(i1 - i) == 1))) {
    val = C * (16.0 / 12.0) / hh;
  } else if (s == s1 && ((i == i1 && j == j1 && std::abs(k1 - k) == 2) ||
                         (i == i1 && k == k1 && std::abs(j1 - j) == 2) ||
                         (j == j1 && k == k1 && std::abs(i1 - i) == 2))) {
    val = -C * (1.0 / 12.0) / hh;
  }
  return std::complex<double>(val, 0.0);
}
