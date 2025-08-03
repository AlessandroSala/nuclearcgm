#pragma once
#define EIGEN_USE_BLAS // Dice a Eigen di usare BLAS per certe operazioni (es.
                       // Matrice*Matrice)
#define EIGEN_USE_LAPACKE // Dice a Eigen di usare LAPACKE per decomposizioni
                          // (es. SVD, Eigensolver densi)
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
constexpr std::complex<double> img = std::complex<double>(0, 1.0);

void printConstants();

std::vector<SpinMatrix, Eigen::aligned_allocator<SpinMatrix>> getPauli();
} // namespace nuclearConstants
typedef enum e_NucleonType { N, P } NucleonType;
