#include "skyrme/skyrme_so.hpp"
#include "constants.hpp"
#include "util/iteration_data.hpp"
#include <cmath>
#include <complex>
#include <memory>

SkyrmeSO::SkyrmeSO(std::shared_ptr<IterationData> data, NucleonType n)
    : data(data), n(n) {}
double SkyrmeSO::getValue(double x, double y, double z) const { return 0.0; }

std::complex<double> SkyrmeSO::getElement5p(int i, int j, int k, int s, int i1,
                                            int j1, int k1, int s1,
                                            const Grid &grid) const {
  if (i1 == i && j1 == j && k1 == k && s1 == s)
    return std::complex<double>(0.0, 0.0);
  SpinMatrix spin(2, 2);
  spin.setZero();
  auto pauli = nuclearConstants::getPauli();
  int idx = grid.idxNoSpin(i, j, k);

  double h = grid.get_h();

  Eigen::Vector3d BP = (*data->BP).row(idx);
  Eigen::Vector3d BN = (*data->BN).row(idx);

  // Spin-orbit field
  auto B = n == NucleonType::N ? BN : BP;

  // Spin-orbit components
  double Wx = B(0);
  double Wy = B(1);
  double Wz = B(2);

  if (std::abs(i - i1) == 1 && j == j1 && k == k1)
    spin += (2.0 / 3.0) * (i1 - i) * (pauli[1] * Wz - pauli[2] * Wy);
  else if (std::abs(i - i1) == 2 && j == j1 && k == k1)
    spin += -(i1 - i) * (1.0 / 24.0) *
            (pauli[1] * Wz - pauli[2] * Wy); // i1-i carries factor 2

  else if (i == i1 && std::abs(j - j1) == 1 && k == k1)
    spin += (2.0 / 3.0) * (j1 - j) * (-pauli[0] * Wz + pauli[2] * Wx);
  else if (i == i1 && std::abs(j - j1) == 2 && k == k1)
    spin += -(j1 - j) * (1.0 / 24.0) *
            (-pauli[0] * Wz + pauli[2] * Wx); // j1-j carries factor 2

  else if (i == i1 && j == j1 && std::abs(k - k1) == 1)
    spin += (2.0 / 3.0) * (k1 - k) * (pauli[0] * Wy - pauli[1] * Wx);
  else if (i == i1 && j == j1 && std::abs(k - k1) == 2)
    spin += -(k1 - k) * (1.0 / 24.0) *
            (pauli[0] * Wy - pauli[1] * Wx); // k1-k carries factor 2
  else
    return std::complex<double>(0.0, 0.0);
  using namespace nuclearConstants;
  spin = -img * spin / h;

  return spin(s, s1);
}
