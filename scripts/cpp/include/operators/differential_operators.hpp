#include <Eigen/Dense>
#include <complex>
#include "grid.hpp"

namespace Operators {
    std::complex<double> dvx(const Eigen::VectorXcd& psi, int i, int j, int k, int s, const Grid& grid);
    std::complex<double> dvy(const Eigen::VectorXcd& psi, int i, int j, int k, int s, const Grid& grid);
    std::complex<double> dvz(const Eigen::VectorXcd& psi, int i, int j, int k, int s, const Grid& grid);
    std::complex<double> dv(std::complex<double> f2, std::complex<double> f1, std::complex<double> f_2, std::complex<double> f_1, std::complex<double> h);
    std::complex<double> derivative_along_axis(const Eigen::VectorXcd& psi, int i, int j, int k, int s, const Grid& grid, char axis); 
    double integral(const Eigen::VectorXd& psi, const Grid& grid);
}