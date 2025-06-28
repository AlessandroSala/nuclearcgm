#include "spin_orbit/deformed_spin_orbit.hpp"
#include "constants.hpp"
#include <cmath>

DeformedSpinOrbitPotential::DeformedSpinOrbitPotential(double V0_,
                                                       Radius radius_,
                                                       double diff_)
    : V0(V0_), R(radius_), diff(diff_) {}
DeformedSpinOrbitPotential::DeformedSpinOrbitPotential(
    Parameters::SpinOrbitParameters p, int A)
    : V0(p.V0), R(Radius(p.beta, p.r0, A)), diff(p.diff) {}
Eigen::VectorXd DeformedSpinOrbitPotential::getFactor(double x, double y,
                                                      double z) const {
  Eigen::VectorXd gradr(3);
  Eigen::VectorXd res(3);
  double r2 = x * x + y * y + z * z;
  double r = std::sqrt(r2);
  gradr(0) = x / r;
  gradr(1) = y / r;
  gradr(2) = z / r;

  if (r > 1e-12) {
    double t = (r - R.getRadius(x, y, z)) / diff;
    double expt = exp(t);
    res = (V0 / diff) * pow(1 + expt, -2) * expt *
          (gradr - R.getGradient(x, y, z));
  }
  return res;
}

double DeformedSpinOrbitPotential::getValue(double x, double y,
                                            double z) const {
  using namespace nuclearConstants;
  return -lambda * pow(h_bar * 0.5 / m, 2);
}

// TODO: REDO, NOT WORKING
std::complex<double>
DeformedSpinOrbitPotential::getElement(int i, int j, int k, int s, int i1,
                                       int j1, int k1, int s1,
                                       const Grid &grid) const {
  if (i1 == i && j1 == j && k1 == k && s1 == s)
    return std::complex<double>(0.0, 0.0);
  SpinMatrix spin(2, 2);
  spin.setZero();
  auto pauli = nuclearConstants::getPauli();

  double h = grid.get_h();
  double x = grid.get_xs()[i], y = grid.get_ys()[j], z = grid.get_zs()[k];
  double ls = getValue(x, y, z);
  if ((i + 1 == i1 || i - 1 == i1) && j == j1 && k == k1)
    spin += (i1 - i) * (pauli[1] * z - pauli[2] * y);

  else if (i == i1 && (j + 1 == j1 || j - 1 == j1) && k == k1)
    spin += (j1 - j) * (-pauli[0] * z + pauli[2] * x);

  else if (i == i1 && j == j1 && (k + 1 == k1 || k - 1 == k1))
    spin += (k1 - k) * (pauli[0] * y - pauli[1] * x);
  else
    return std::complex<double>(0.0, 0.0);
  // ls = 0;
  using namespace nuclearConstants;
  spin = -pow(2 * h, -1) * std::complex<double>(0, 1.0) * 0.5 * h_bar * h_bar *
         spin * ls;

  return spin(s, s1);
}
std::complex<double>
DeformedSpinOrbitPotential::getElement5p(int i, int j, int k, int s, int i1,
                                         int j1, int k1, int s1,
                                         const Grid &grid) const {
  if (i1 == i && j1 == j && k1 == k && s1 == s)
    return std::complex<double>(0.0, 0.0);
  SpinMatrix spin(2, 2);
  spin.setZero();
  auto pauli = nuclearConstants::getPauli();

  double h = grid.get_h();
  double x = grid.get_xs()[i], y = grid.get_ys()[j], z = grid.get_zs()[k];
  double ls = getValue(x, y, z);
  auto gradV = getFactor(x, y, z);

  if (std::abs(i - i1) == 1 && j == j1 && k == k1)
    spin +=
        (2.0 / 3.0) * (i1 - i) * (pauli[2] * gradV(1) - pauli[1] * gradV(2));
  else if (std::abs(i - i1) == 2 && j == j1 && k == k1)
    spin +=
        -(i1 - i) * (1.0 / 24.0) *
        (pauli[2] * gradV(1) - pauli[1] * gradV(2)); // i1-i carries factor 2

  else if (i == i1 && std::abs(j - j1) == 1 && k == k1)
    spin +=
        (2.0 / 3.0) * (j1 - j) * (-pauli[2] * gradV(0) + pauli[0] * gradV(2));
  else if (i == i1 && std::abs(j - j1) == 2 && k == k1)
    spin +=
        -(j1 - j) * (1.0 / 24.0) *
        (-pauli[2] * gradV(0) + pauli[0] * gradV(2)); // j1-j carries factor 2

  else if (i == i1 && j == j1 && std::abs(k - k1) == 1)
    spin +=
        (2.0 / 3.0) * (k1 - k) * (pauli[1] * gradV(0) - pauli[0] * gradV(1));
  else if (i == i1 && j == j1 && std::abs(k - k1) == 2)
    spin +=
        -(k1 - k) * (1.0 / 24.0) *
        (pauli[1] * gradV(0) - pauli[0] * gradV(1)); // k1-k carries factor 2
  else
    return std::complex<double>(0.0, 0.0);
  // ls = 0;
  using namespace nuclearConstants;
  std::complex<double> img = std::complex<double>(0, 1.0);
  spin = (1.0 / h) * img * ls * spin;

  return spin(s, s1);
}
