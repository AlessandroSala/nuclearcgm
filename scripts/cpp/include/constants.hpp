#pragma once
#include "types.hpp"
#include <vector>
#include <Eigen/StdVector>

namespace nuclearConstants {
    constexpr double h_bar = 197.326971;
    constexpr double m = 939;
    constexpr double ev = 1.60218e-19;
    constexpr double C = -2*m/(h_bar*h_bar);
    constexpr double diff = 0.67;
    

    std::vector<SpinMatrix, Eigen::aligned_allocator<SpinMatrix>> getPauli();
}