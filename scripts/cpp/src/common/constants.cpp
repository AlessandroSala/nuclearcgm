#include "constants.hpp"
#include <iostream>

std::vector<SpinMatrix, Eigen::aligned_allocator<SpinMatrix>>
nuclearConstants::getPauli() {
  std::vector<SpinMatrix, Eigen::aligned_allocator<SpinMatrix>> pauli;

  SpinMatrix sigma_x(2, 2); // Initialize with dimensions
  sigma_x << 0, 1, 1, 0;
  pauli.push_back(sigma_x);

  SpinMatrix sigma_y(2, 2);
  sigma_y << 0, std::complex<double>(0, -1), std::complex<double>(0, 1), 0;
  pauli.push_back(sigma_y);

  SpinMatrix sigma_z(2, 2);

  sigma_z << 1, 0, 0, -1;

  pauli.push_back(sigma_z);

  return pauli;
}
void nuclearConstants::printConstants() {}
