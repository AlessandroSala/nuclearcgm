
#include "kinetic/iso_kinetic_potential.hpp"
#include "constants.hpp"
#include <cmath>
#include <iostream>

IsoKineticPotential::IsoKineticPotential() {}
double IsoKineticPotential::getValue(double x, double y, double z) const {
  return 1;
}
// TODO: not really, implement 3p derivatives
std::complex<double> IsoKineticPotential::getElement(int i, int j, int k, int s,
                                                     int i1, int j1, int k1,
                                                     int s1,
                                                     const Grid &grid) const {

  return 0;
}

std::complex<double> IsoKineticPotential::getElement5p(int i, int j, int k,
                                                       int s, int i1, int j1,
                                                       int k1, int s1,
                                                       const Grid &grid) const {
  double val = 0.0;
  double hh = grid.get_h() * grid.get_h();

  double C = 1 / nuclearConstants::C;

  if (s == s1) {
    // Punto centrale
    if (i == i1 && j == j1 && k == k1) {
      val = -C * (64.0 / 15.0) / hh;
    }

    // Vicini sugli assi (±1,0,0), (0,±1,0), (0,0,±1)
    else if ((i == i1 && j == j1 && std::abs(k1 - k) == 1) ||
             (i == i1 && k == k1 && std::abs(j1 - j) == 1) ||
             (j == j1 && k == k1 && std::abs(i1 - i) == 1)) {
      val = C * (7.0 / 15.0) / hh;
    }

    // Vicini diagonali "edge" (±1,±1,0) e permutazioni
    else if ((std::abs(i1 - i) == 1 && std::abs(j1 - j) == 1 && k1 == k) ||
             (std::abs(i1 - i) == 1 && std::abs(k1 - k) == 1 && j1 == j) ||
             (std::abs(j1 - j) == 1 && std::abs(k1 - k) == 1 && i1 == i)) {
      val = C * (1.0 / 10.0) / hh;
    }

    // Vicini "corner" (±1,±1,±1)
    else if (std::abs(i1 - i) == 1 && std::abs(j1 - j) == 1 &&
             std::abs(k1 - k) == 1) {
      val = C * (1.0 / 30.0) / hh;
    }
  }
  return std::complex<double>(val, 0.0);
}
