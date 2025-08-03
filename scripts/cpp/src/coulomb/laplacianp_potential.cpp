#include "coulomb/laplacian_potential.hpp"
#include <cmath>

LaplacianPotential::LaplacianPotential() {}
double LaplacianPotential::getValue(double x, double y, double z) const {
  return 1;
}
// TODO: not really, implement 3p derivatives
std::complex<double> LaplacianPotential::getElement(int i, int j, int k, int s,
                                                    int i1, int j1, int k1,
                                                    int s1,
                                                    const Grid &grid) const {

  return 0;
}

std::complex<double> LaplacianPotential::getElement5p(int i, int j, int k,
                                                      int s, int i1, int j1,
                                                      int k1, int s1,
                                                      const Grid &grid) const {
  double val = 0.0;
  double hh = grid.get_h() * grid.get_h();

  if (s != s1) {
    return std::complex<double>(0.0, 0.0);
  }

  // if (i == i1 && j == j1 && k == k1) {
  //   val = -(490.0 / 180.0) / hh;
  // } else if (((i == i1 && j == j1 && std::abs(k1 - k) == 1) ||
  //             (i == i1 && k == k1 && std::abs(j1 - j) == 1) ||
  //             (j == j1 && k == k1 && std::abs(i1 - i) == 1))) {
  //   val = (270.0 / 180.0) / hh;
  // } else if (((i == i1 && j == j1 && std::abs(k1 - k) == 2) ||
  //             (i == i1 && k == k1 && std::abs(j1 - j) == 2) ||
  //             (j == j1 && k == k1 && std::abs(i1 - i) == 2))) {
  //   val = -(27.0 / 180.0) / hh;
  // } else if (((i == i1 && j == j1 && std::abs(k1 - k) == 3) ||
  //             (i == i1 && k == k1 && std::abs(j1 - j) == 3) ||
  //             (j == j1 && k == k1 && std::abs(i1 - i) == 3))) {
  //   val = (2.0 / 180.0) / hh;
  // }
  if (i == i1 && j == j1 && k == k1 && s == s1) {
    val = -(90.0 / 12.0) / hh;
  } else if (s == s1 && ((i == i1 && j == j1 && std::abs(k1 - k) == 1) ||
                         (i == i1 && k == k1 && std::abs(j1 - j) == 1) ||
                         (j == j1 && k == k1 && std::abs(i1 - i) == 1))) {
    val = (16.0 / 12.0) / hh;
  } else if (s == s1 && ((i == i1 && j == j1 && std::abs(k1 - k) == 2) ||
                         (i == i1 && k == k1 && std::abs(j1 - j) == 2) ||
                         (j == j1 && k == k1 && std::abs(i1 - i) == 2))) {
    val = -(1.0 / 12.0) / hh;
  }
  return std::complex<double>(val, 0.0);
}
