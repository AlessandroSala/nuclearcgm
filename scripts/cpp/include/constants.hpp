#pragma once
#define EIGEN_USE_BLAS

#define EIGEN_USE_LAPACKE

#include "types.hpp"
#include <Eigen/StdVector>
#include <vector>

namespace nuclearConstants {
constexpr double h_bar = 197.327053;
constexpr double ev = 1.60218e-19;
constexpr double m = 938.91869;
constexpr double C = -2 * m / (h_bar * h_bar);
constexpr double c = 299792458;
constexpr double lambda = 35;
constexpr double e2 = 1.43996446;
constexpr double e = 1.602176565e-19;
constexpr double rp = 0.64;
constexpr double rn = -0.11;
// constexpr double magneton = 0.5 * e * c * h_bar / m;

constexpr double muN = -1.913043;
constexpr double muP = 2.792847;
constexpr std::complex<double> img = std::complex<double>(0, 1.0);

void printConstants();

std::vector<SpinMatrix, Eigen::aligned_allocator<SpinMatrix>> getPauli();
} // namespace nuclearConstants
typedef enum NucleonType { N, P } NucleonType;
