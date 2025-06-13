#include "operators/integral_operators.hpp"
#include <omp.h> // Include OpenMP header

std::complex<double> Operators::integral(const Eigen::VectorXcd& psi, const Grid& grid) {
    double h = grid.get_h();
    double hhh = h*h*h;

    return psi.sum()*hhh;
}