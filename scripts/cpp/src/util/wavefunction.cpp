#include "util/wavefunction.hpp"
#include "operators/differential_operators.hpp"
#include <complex>
#include "constants.hpp"

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

// TODO: Non so se questa è corretta
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
// TODO: Terminare
Eigen::MatrixXcd Wavefunction::soDensity(const Eigen::MatrixXcd &psi, const Grid &grid)
{

    using std::complex;
    Eigen::MatrixXcd J(grid.get_total_spatial_points(), 3);
    J.setZero();
    auto pauli = nuclearConstants::getPauli();
    for (int i = 0; i < grid.get_n(); ++i)
    {
        for (int j = 0; j < grid.get_n(); ++j)
        {
            for (int k = 0; k < grid.get_n(); ++k)
            {
                for (int col = 0; col < psi.cols(); ++col)
                {
                    Eigen::Matrix<complex<double>, 2, 3> chiGrad;
                    Eigen::Vector2cd chi(2);


                    for (int s = 0; s < 2; ++s)
                    {
                        Eigen::Vector3cd grad(3);
                        grad(0) = Operators::derivative(psi, i, j, k, s, grid, 'x');
                        grad(1) = Operators::derivative(psi, i, j, k, s, grid, 'y');
                        grad(2) = Operators::derivative(psi, i, j, k, s, grid, 'z');

                        chiGrad.row(s) = grad.transpose();
                        chi(s) = psi(grid.idx(i, j, k, s), col);

                    }
                    J(grid.idxNoSpin(i, j, k), 0) = chi.dot(pauli[2] * chiGrad.col(1) - pauli[1] * chiGrad.col(2));
                    J(grid.idxNoSpin(i, j, k), 1) = chi.dot(pauli[0] * chiGrad.col(2) - pauli[2] * chiGrad.col(0));
                    J(grid.idxNoSpin(i, j, k), 2) = chi.dot(pauli[1] * chiGrad.col(0) - pauli[0] * chiGrad.col(1));
                    // tau(grid.idxNoSpin(i, j, k)) += norm(dx) + norm(dy) + norm(dz);
                }
            }
        }
    }
    return -nuclearConstants::img * J;
}
