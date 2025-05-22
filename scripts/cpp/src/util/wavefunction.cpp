#include "util/wavefunction.hpp"
#include "operators/differential_operators.hpp"
#include <complex>

Eigen::VectorXd Wavefunction::density(const Eigen::MatrixXcd &psi, const Grid &grid)
{
    int n = grid.get_n();
    using std::complex;
    Eigen::VectorXd rho(grid.get_total_spatial_points());
    rho.setZero();
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            for (int k = 0; k < n; ++k)
            {
                int idx = grid.idx(i, j, k, 0);
                for (int col = 0; col < psi.cols(); ++col)
                {
                    rho(grid.idxNoSpin(i, j, k)) += norm(psi(idx, col)) + norm(psi(idx + 1, col));
                }
            }
        }
    }
    return rho;
}

//TODO: Non so se questa è corretta
Eigen::VectorXd Wavefunction::kineticDensity(const Eigen::MatrixXcd &psi, const Grid &grid)
{
    using std::complex;
    Eigen::VectorXd tau(grid.get_total_spatial_points());
    tau.setZero();
    for (int i = 0; i < grid.get_n(); ++i)
    {
        for (int j = 0; j < grid.get_n(); ++j)
        {
            for (int k = 0; k < grid.get_n(); ++k)
            {
                for (int s = 0; s < 2; ++s)
                {
                    for (int col = 0; col < psi.cols(); ++col)
                    {
                        std::complex<double> dx = Operators::derivative(psi, i, j, k, s, grid, 'x');
                        std::complex<double> dy = Operators::derivative(psi, i, j, k, s, grid, 'y');
                        std::complex<double> dz = Operators::derivative(psi, i, j, k, s, grid, 'z');
                        tau(grid.idxNoSpin(i, j, k)) += norm(dx) + norm(dy) + norm(dz);
                    }
                }
            }
        }
    }
    return tau;
}
