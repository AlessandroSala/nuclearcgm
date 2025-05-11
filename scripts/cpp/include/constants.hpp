#pragma once
#define EIGEN_USE_BLAS       // Dice a Eigen di usare BLAS per certe operazioni (es. Matrice*Matrice)
#define EIGEN_USE_LAPACKE    // Dice a Eigen di usare LAPACKE per decomposizioni (es. SVD, Eigensolver densi)
#include "types.hpp"
#include <vector>
#include <Eigen/StdVector>

namespace nuclearConstants {
    constexpr double h_bar = 197.326971;
    constexpr double ev = 1.60218e-19;
    constexpr double m = 938.272;
    constexpr double C = -2*m/(h_bar*h_bar);
    constexpr double diff = 0.67;
    constexpr double c = 299792458;
    constexpr double lambda = 35;
    

    std::vector<SpinMatrix, Eigen::aligned_allocator<SpinMatrix>> getPauli();
}