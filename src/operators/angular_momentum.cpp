#include "operators/angular_momentum.hpp"
#include "constants.hpp"
#include "operators/differential_operators.hpp"
#include <complex>
#include <bits/stdc++.h>

// TODO: completare
Eigen::VectorXcd Operators::J(const Eigen::VectorXcd &psi, const Grid &grid)
{
    using namespace nuclearConstants;
    Eigen::VectorXcd res(grid.get_total_points());
    int n = grid.get_n();
    auto pauli = getPauli();

    for (int k = 0; k < n; ++k)
    {
        for (int j = 0; j < n; ++j)
        {
            for (int i = 0; i < n; ++i)
            {
                Eigen::VectorXcd chi(2);
                chi(0) = psi(grid.idx(i, j, k, 0));
                chi(1) = psi(grid.idx(i, j, k, 1));
                Eigen::MatrixXcd spinPsi(3, 2);
                spinPsi.row(0) = pauli[0] * chi;
                spinPsi.row(1) = pauli[1] * chi;
                spinPsi.row(2) = pauli[2] * chi;
                for (int s = 0; s < 2; ++s)
                {
                    int idx = grid.idx(i, j, k, s);
                    std::complex<double> derX = Operators::derivative(psi, i, j, k, s, grid, 'x');
                    std::complex<double> derY = Operators::derivative(psi, i, j, k, s, grid, 'y');
                    std::complex<double> derZ = Operators::derivative(psi, i, j, k, s, grid, 'z');
                    // std::complex<double> derY = Operators::dvy(psi, i, j, k, s, grid);
                    double x = grid.get_xs()[i];
                    double y = grid.get_ys()[j];
                    double z = grid.get_ys()[k];
                    std::complex<double> spinZ = s == 0 ? psi(idx) : -psi(idx);
                    auto img = std::complex<double>(0, 1.0);
                    auto h_bar = nuclearConstants::h_bar;

                    // res(idx) = h_bar*(img*(y*derX - x*derY) + 0.5*spinPsi);
                    // res(idx) = h_bar*spinPart;
                }
            }
        }
    }
    return res;
}
Eigen::VectorXcd Operators::J2(const Eigen::VectorXcd &psi, const Grid &grid)
{
    return L2(psi, grid) + S2(psi, grid) + 2 * LS(psi, grid);
}
Eigen::VectorXcd Operators::S2(const Eigen::VectorXcd &psi, const Grid &grid)
{
    using nuclearConstants::h_bar;
    return (3.0 / 4.0) * (h_bar * h_bar) * psi;
}
Eigen::VectorXcd Operators::L2(const Eigen::VectorXcd &psi, const Grid &grid)
{
    using Operators::Lx;
    using Operators::Ly;
    using Operators::Lz;
    return (Lx(Lx(psi, grid), grid) + Ly(Ly(psi, grid), grid) + Lz(Lz(psi, grid), grid));
    using nuclearConstants::h_bar;
    Eigen::VectorXcd res(grid.get_total_points());

    int n = grid.get_n();
    auto dX = Operators::dv(psi, grid, 'x');
    auto dY = Operators::dv(psi, grid, 'y');
    auto dZ = Operators::dv(psi, grid, 'z');
    Eigen::VectorXcd xPsi(grid.get_total_points());
    Eigen::VectorXcd yPsi(grid.get_total_points());
    Eigen::VectorXcd zPsi(grid.get_total_points());
    for(int i = 0; i < n; ++i) {
        for(int j = 0; j < n; ++j) {
            for(int k = 0; k < n; ++k) {
                for(int s = 0; s < 2; ++s) {
                    int idx = grid.idx(i, j, k, s);
                    xPsi(idx) = psi(idx) * grid.get_xs()[i];
                    yPsi(idx) = psi(idx) * grid.get_ys()[j];
                    zPsi(idx) = psi(idx) * grid.get_zs()[k];
                }
            }
        }
    }
    for (int k = 0; k < n; ++k)
    {
        for (int j = 0; j < n; ++j)
        {
            for (int i = 0; i < n; ++i)
            {
                for (int s = 0; s < 2; ++s)
                {
                    int idx = grid.idx(i, j, k, s);
                    std::complex<double> der2X = Operators::derivative2(psi, i, j, k, s, grid, 'x');
                    std::complex<double> der2Y = Operators::derivative2(psi, i, j, k, s, grid, 'y');
                    std::complex<double> der2Z = Operators::derivative2(psi, i, j, k, s, grid, 'z');
                    std::complex<double> dXY = Operators::derivative(dX, i, j, k, s, grid, 'y');
                    std::complex<double> dXZ = Operators::derivative(dX, i, j, k, s, grid, 'z');
                    std::complex<double> dYZ = Operators::derivative(dY, i, j, k, s, grid, 'z');

                    double x = grid.get_xs()[i];
                    double y = grid.get_ys()[j];
                    double z = grid.get_zs()[k];
                    using Operators::derivative;
                    auto Lx2 = -h_bar * h_bar * (y*y*der2Z + der2Y*z*z - z*derivative(yPsi.cwiseProduct(dZ), i, j, k, s, grid, 'y') - y*derivative(zPsi.cwiseProduct(dY), i, j, k, s, grid, 'z'));
                    auto Ly2 = -h_bar * h_bar * (z*z*der2X + der2Z*x*x - x*derivative(zPsi.cwiseProduct(dX), i, j, k, s, grid, 'x') - z*derivative(xPsi.cwiseProduct(dZ), i, j, k, s, grid, 'z'));
                    auto Lz2 = -h_bar * h_bar * (x*x*der2Y + der2X*y*y - y*derivative(xPsi.cwiseProduct(dY), i, j, k, s, grid, 'y') - x*derivative(yPsi.cwiseProduct(dX), i, j, k, s, grid, 'x'));
                    res(idx) = Lx2 + Ly2 + Lz2; 
                }
            }
        }
    }
    return res;
}

Eigen::VectorXcd Operators::LS(const Eigen::VectorXcd &psi, const Grid &grid)
{
    using nuclearConstants::h_bar;
    using nuclearConstants::img; // Assuming img is std::complex<double>(0.0, 1.0)

    Eigen::VectorXcd res(grid.get_total_points());
    int n = grid.get_n(); // Assuming this is the number of points in each dimension

    // Prefactor for L.S operator: -i * h_bar^2 / 2
    std::complex<double> prefactor = -img * h_bar * h_bar * 0.5;

    for (int k = 0; k < n; ++k)
    {
        for (int j = 0; j < n; ++j)
        {
            for (int i = 0; i < n; ++i)
            {
                // Coordinates of the current grid point
                double x = grid.get_xs()[i];
                double y = grid.get_ys()[j];
                double z = grid.get_zs()[k];

                // Indices for spin-up (s=0) and spin-down (s=1) components
                // Ensure grid.idx(i,j,k,s) returns the correct global index for psi
                // And that psi for spin down is at idx_up + 1 if that's the convention.
                int idx_up = grid.idx(i, j, k, 0);   // Index for psi_up
                int idx_down = grid.idx(i, j, k, 1); // Index for psi_down
                                                     // Or, if spin components are contiguous: int idx_down = idx_up + 1;

                // Derivatives of the spin-up component (psi_0 or psi_up)
                std::complex<double> d_psi0_dx = Operators::derivative(psi, i, j, k, 0, grid, 'x');
                std::complex<double> d_psi0_dy = Operators::derivative(psi, i, j, k, 0, grid, 'y');
                std::complex<double> d_psi0_dz = Operators::derivative(psi, i, j, k, 0, grid, 'z');

                // Derivatives of the spin-down component (psi_1 or psi_down)
                std::complex<double> d_psi1_dx = Operators::derivative(psi, i, j, k, 1, grid, 'x');
                std::complex<double> d_psi1_dy = Operators::derivative(psi, i, j, k, 1, grid, 'y');
                std::complex<double> d_psi1_dz = Operators::derivative(psi, i, j, k, 1, grid, 'z');

                // L'_k psi_s = (r x grad)_k psi_s terms
                // For psi_up (psi_0)
                std::complex<double> Lx_psi0 = y * d_psi0_dz - z * d_psi0_dy; // (y*pz - z*py) acting on psi_0 (missing -i*h_bar from L_k proper)
                std::complex<double> Ly_psi0 = z * d_psi0_dx - x * d_psi0_dz; // (z*px - x*pz) acting on psi_0
                std::complex<double> Lz_psi0 = x * d_psi0_dy - y * d_psi0_dx; // (x*py - y*px) acting on psi_0

                // For psi_down (psi_1)
                std::complex<double> Lx_psi1 = y * d_psi1_dz - z * d_psi1_dy;
                std::complex<double> Ly_psi1 = z * d_psi1_dx - x * d_psi1_dz;
                std::complex<double> Lz_psi1 = x * d_psi1_dy - y * d_psi1_dx;

                // Resulting components (L.S psi)_s = Prefactor * [ (L_op . sigma_vec)_s_s' psi_s' ]
                // (L.S psi)_up = Prefactor * [ Lz_psi_up + (Lx - i*Ly)_psi_down ]
                // (L.S psi)_down = Prefactor * [ (Lx + i*Ly)_psi_up - Lz_psi_down ]

                std::complex<double> res_psi_up = prefactor * (Lz_psi0 + (Lx_psi1 - img * Ly_psi1));
                std::complex<double> res_psi_down = prefactor * ((Lx_psi0 + img * Ly_psi0) - Lz_psi1);
                
                res(idx_up) = res_psi_up;
                res(idx_down) = res_psi_down;



            }
        }
    }
    return res;
}
Eigen::VectorXcd Operators::Sz(const Eigen::VectorXcd &psi, const Grid &grid) {
    Eigen::VectorXcd res(grid.get_total_points());
    using nuclearConstants::h_bar;
    auto sigma_z = nuclearConstants::getPauli()[2];
    int n = grid.get_n();
    for (int k = 0; k < n; ++k) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                Eigen::Vector2cd spinPart(2);
                int idx = grid.idx(i, j, k, 0);
                spinPart(0) = psi(idx);
                spinPart(1) = psi(idx+1);
                spinPart = 0.5*h_bar*spinPart;
                spinPart = sigma_z*spinPart;
                res(idx) = spinPart(0);
                res(idx + 1) = spinPart(1);
            }
        }
    }
    return res;
}
Eigen::VectorXcd Operators::Lx(const Eigen::VectorXcd &psi, const Grid &grid) {
    Eigen::VectorXcd res(grid.get_total_points());
    using nuclearConstants::h_bar;
    using nuclearConstants::img;
    auto dZ = Operators::dv(psi, grid, 'z');
    auto dY = Operators::dv(psi, grid, 'y');
    int n = grid.get_n();
    for (int k = 0; k < n; ++k) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                double z = grid.get_zs()[k];
                double y = grid.get_ys()[j];
                for (int s = 0; s < 2; ++s) {
                    int idx = grid.idx(i, j, k, s);
                    res(idx) = -img*h_bar*(y*dZ(idx) - z*dY(idx));
                }
            }
        }
    }
    return res;
}
Eigen::VectorXcd Operators::Ly(const Eigen::VectorXcd &psi, const Grid &grid) {
    Eigen::VectorXcd res(grid.get_total_points());
    using nuclearConstants::h_bar;
    using nuclearConstants::img;
    auto dX = Operators::dv(psi, grid, 'x');
    auto dZ = Operators::dv(psi, grid, 'z');
    int n = grid.get_n();
    for (int k = 0; k < n; ++k) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                double x = grid.get_xs()[i];
                double z = grid.get_zs()[k];
                for (int s = 0; s < 2; ++s) {
                    int idx = grid.idx(i, j, k, s);
                    res(idx) = -img*h_bar*(z*dX(idx) - x*dZ(idx));
                }
            }
        }
    }
    return res;
}
Eigen::VectorXcd Operators::Lz(const Eigen::VectorXcd &psi, const Grid &grid) {
    Eigen::VectorXcd res(grid.get_total_points());
    using nuclearConstants::h_bar;
    using nuclearConstants::img;
    auto dX = Operators::dv(psi, grid, 'x');
    auto dY = Operators::dv(psi, grid, 'y');
    int n = grid.get_n();
    for (int k = 0; k < n; ++k) {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                double x = grid.get_xs()[i];
                double y = grid.get_ys()[j];
                for (int s = 0; s < 2; ++s) {
                    int idx = grid.idx(i, j, k, s);
                    res(idx) = -img*h_bar*(x*dY(idx) - y*dX(idx));
                }
            }
        }
    }
    return res;
}
Eigen::VectorXcd Operators::Jz(const Eigen::VectorXcd &psi, const Grid &grid)
{
    using Operators::Lz;
    using Operators::Sz;
    return Lz(psi, grid) + Sz(psi, grid);
    Eigen::VectorXcd res(grid.get_total_points());
    int n = grid.get_n();

    for (int k = 0; k < n; ++k)
    {
        for (int j = 0; j < n; ++j)
        {
            for (int i = 0; i < n; ++i)
            {
                std::complex<double> derX = Operators::derivative(psi, i, j, k, 0, grid, 'x');
                std::complex<double> derY = Operators::derivative(psi, i, j, k, 0, grid, 'y');
                for (int s = 0; s < 2; ++s)
                {
                    int idx = grid.idx(i, j, k, s);
                    // std::complex<double> derY = Operators::dvy(psi, i, j, k, s, grid);
                    double x = grid.get_xs()[i];
                    double y = grid.get_ys()[j];
                    std::complex<double> spinPart = s == 0 ? psi(idx) : -psi(idx);
                    spinPart *= 0.5;
                    auto img = std::complex<double>(0, 1.0);
                    auto h_bar = nuclearConstants::h_bar;

                    res(idx) = h_bar * (img * (y * derX - x * derY) + spinPart);
                }
            }
        }
    }
    return res;
}
double Operators::JzExp(const Eigen::VectorXcd &psi, const Grid &grid)
{
    int n = grid.get_n();
    DenseVector f(grid.get_total_spatial_points());
    ComplexDenseVector Jzpsi = Jz(psi, grid);
    double res = 0;
    for (int k = 0; k < n; ++k)
    {
        for (int j = 0; j < n; ++j)
        {
            for (int i = 0; i < n; ++i)
            {
                res += ((norm(psi(grid.idx(i, j, k, 0))) + norm(psi(grid.idx(i, j, k, 1)))) * norm(Jzpsi(grid.idx(i, j, k, 0))) * norm(Jzpsi(grid.idx(i, j, k, 1)))) * pow(grid.get_h(), 3);
            }
        }
    }

    return res;
}
