#pragma once
namespace nuclearConstants {
    constexpr double h_bar = 197.326971;
    constexpr double m = 939;
    constexpr double ev = 1.60218e-19;
    constexpr double C = -2*m/(h_bar*h_bar);
    Eigen::Matrix<std::complex<double>, 2, 2> pauli[3] = {(Eigen::Matrix<std::complex<double>, 2, 2>() << 0, 1, 1, 0).finished(), (Eigen::Matrix<std::complex<double>, 2, 2>() << 0, std::complex<double>(0, -1), std::complex<double>(0, 1), 0).finished(), (Eigen::Matrix<std::complex<double>, 2, 2>() << 1, 0, 0, -1).finished()};

}