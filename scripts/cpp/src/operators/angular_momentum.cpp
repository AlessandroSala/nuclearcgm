#include "operators/angular_momentum.hpp"
#include "constants.hpp"
#include "operators/differential_operators.hpp"
#include <complex>
#include <bits/stdc++.h>

//TODO: completare
Eigen::VectorXcd Operators::J(const Eigen::VectorXcd& psi, const Grid& grid) {
    using namespace nuclearConstants;
    Eigen::VectorXcd res(grid.get_total_points());
    int n = grid.get_n();
    auto pauli = getPauli();

    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < n; ++j) {
            for(int k = 0; k < n; ++k) {
                Eigen::VectorXcd chi(2);
                chi(0) = psi(grid.idx(i, j, k, 0));
                chi(1) = psi(grid.idx(i, j, k, 1));
                Eigen::MatrixXcd spinPsi(3, 2);
                spinPsi.row(0) = pauli[0] * chi;
                spinPsi.row(1) = pauli[1] * chi;
                spinPsi.row(2) = pauli[2] * chi;
                for(int s = 0; s < 2; ++s) {
                    int idx = grid.idx(i, j, k, s);
                    std::complex<double> derX = Operators::derivative_along_axis(psi, i, j, k, s, grid, 'x');
                    std::complex<double> derY = Operators::derivative_along_axis(psi, i, j, k, s, grid, 'y');
                    std::complex<double> derZ = Operators::derivative_along_axis(psi, i, j, k, s, grid, 'z');
                    //std::complex<double> derY = Operators::dvy(psi, i, j, k, s, grid);
                    double x = grid.get_xs()[i]; 
                    double y = grid.get_ys()[j];
                    double z = grid.get_ys()[k];
                    std::complex<double> spinZ = s == 0 ? psi(idx) : -psi(idx);
                    auto img = std::complex<double>(0, 1.0);
                    auto h_bar = nuclearConstants::h_bar;

                    //res(idx) = h_bar*(img*(y*derX - x*derY) + 0.5*spinPsi);
                    //res(idx) = h_bar*spinPart;

                }
            }
        }
    }
    return res;

}
Eigen::VectorXcd Operators::Jz(const Eigen::VectorXcd& psi, const Grid& grid) {
    Eigen::VectorXcd res(grid.get_total_points());
    int n = grid.get_n();

    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < n; ++j) {
            for(int k = 0; k < n; ++k) {
                for(int s = 0; s < 2; ++s) {
                    int idx = grid.idx(i, j, k, s);
                    std::complex<double> derX = Operators::derivative_along_axis(psi, i, j, k, s, grid, 'x');
                    std::complex<double> derY = Operators::derivative_along_axis(psi, i, j, k, s, grid, 'y');
                    //std::complex<double> derY = Operators::dvy(psi, i, j, k, s, grid);
                    double x = grid.get_xs()[i]; 
                    double y = grid.get_ys()[j];
                    std::complex<double> spinPart = s == 0 ? psi(idx) : -psi(idx);
                    spinPart *= 0.5;
                    auto img = std::complex<double>(0, 1.0);
                    auto h_bar = nuclearConstants::h_bar;

                    res(idx) = h_bar*(img*(y*derX - x*derY) + spinPart);
                    //res(idx) = h_bar*spinPart;

                }
            }
        }
    }
    return res;

}
double Operators::JzExp(const Eigen::VectorXcd& psi, const Grid& grid) {
    int n = grid.get_n();
    DenseVector f(grid.get_total_spatial_points());
    ComplexDenseVector Jzpsi = Jz(psi, grid);
    double res = 0;
    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < n; ++j) {
            for(int k = 0; k < n; ++k) {
                res += ((norm(psi(grid.idx(i, j, k, 0))) + norm(psi(grid.idx(i, j, k, 1))))* norm(Jzpsi(grid.idx(i, j, k, 0))) * norm(Jzpsi(grid.idx(i, j, k, 1))))*pow(grid.get_h(), 3);
            }

        }
    }

    return res;

}

