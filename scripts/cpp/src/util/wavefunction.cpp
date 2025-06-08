#include <omp.h> // Required for OpenMP
#include "operators/differential_operators.hpp" // Assuming this is where Operators::derivative is
#include "constants.hpp" // Assuming this is where nuclearConstants are
#include "operators/angular_momentum.hpp"
#include "operators/common_operators.hpp"
#include <iostream>

// Forward declaration of Grid class if not fully included,
// or include the necessary header for Grid.
// class Grid; 

// Assuming Wavefunction is a class or namespace
namespace Wavefunction {

Eigen::VectorXd density(const Eigen::MatrixXcd &psi, const Grid &grid)
{
    int n = grid.get_n();
    // using std::complex; // Already included via Eigen headers or <complex>
    Eigen::VectorXd rho(grid.get_total_spatial_points());
    // rho.setZero(); // Eigen vectors/matrices are default-initialized (often to zero for numeric types, but explicit is safer)
                     // However, since we assign directly `rho(rho_idx) = point_density;`, setZero is not strictly needed if all points are covered.
                     // For safety, especially if some points might not be touched by the loop (not the case here), setZero is good.
    rho.setZero();


    // Parallelize the loops over spatial grid points (i, j, k)
    // The collapse(3) clause treats the three nested loops as a single larger loop for parallelization.
    // This can be beneficial if 'n' is not excessively large, providing more chunks of work.
    // If 'n' is very large, #pragma omp parallel for on the 'i' loop alone might be sufficient.
    #pragma omp parallel for collapse(3)
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            for (int k = 0; k < n; ++k)
            {
                int rho_idx = grid.idxNoSpin(i, j, k);
                // Assuming grid.idx(i, j, k, 0) gives the index for spin component 0
                // and grid.idx(i, j, k, 1) gives the index for spin component 1.
                // The original code used idx and idx+1, which implies grid.idx(i,j,k,1) = grid.idx(i,j,k,0) + 1.
                // Using explicit calls to grid.idx for each spin is safer if this assumption isn't guaranteed.
                int psi_idx_s0 = grid.idx(i, j, k, 0);
                int psi_idx_s1 = grid.idx(i, j, k, 1);

                double point_density = 0.0;
                for (int col = 0; col < psi.cols(); ++col)
                {
                    point_density += std::norm(psi(psi_idx_s0, col)); // std::norm(complex) = |complex|^2
                    point_density += std::norm(psi(psi_idx_s1, col));
                }
                rho(rho_idx) = point_density; // Each thread writes to a unique element of rho
            }
        }
    }
    return rho;
}

// TODO: Non so se questa è corretta (The user's original comment)
Eigen::VectorXd kineticDensity(const Eigen::MatrixXcd &psi, const Grid &grid)
{
    // using std::complex;
    Eigen::VectorXd tau(grid.get_total_spatial_points());
    tau.setZero();

    #pragma omp parallel for collapse(3)
    for (int i = 0; i < grid.get_n(); ++i)
    {
        for (int j = 0; j < grid.get_n(); ++j)
        {
            for (int k = 0; k < grid.get_n(); ++k)
            {
                int tau_idx = grid.idxNoSpin(i, j, k);
                double point_tau = 0.0;
                for (int s = 0; s < 2; ++s) // Loop over spin components
                {
                    for (int col = 0; col < psi.cols(); ++col) // Loop over orbitals/states
                    {
                        // psi.col(col) passes a column vector (or an expression representing it)
                        // to the derivative function. Operators::derivative is assumed to be thread-safe.
                        point_tau += std::norm(Operators::derivative(psi.col(col), i, j, k, s, grid, 'x'));
                        point_tau += std::norm(Operators::derivative(psi.col(col), i, j, k, s, grid, 'y'));
                        point_tau += std::norm(Operators::derivative(psi.col(col), i, j, k, s, grid, 'z'));
                    }
                }
                tau(tau_idx) = point_tau; // Each thread writes to a unique element of tau
            }
        }
    }
    return tau;
}

// TODO: Terminare (The user's original comment - implies the logic might be incomplete, but parallelizing as is)
Eigen::MatrixXcd soDensity(const Eigen::MatrixXcd &psi, const Grid &grid)
{
    using std::complex; // Explicitly using std::complex for clarity
    Eigen::MatrixXcd J(grid.get_total_spatial_points(), 3);
    J.setZero();
    
    // nuclearConstants::getPauli() should be thread-safe or pauli matrices copied if stateful.
    // Assuming it returns const objects or is thread-safe.
    auto pauli_matrices = nuclearConstants::getPauli(); // Call once outside the parallel region

    #pragma omp parallel for collapse(5)
    for (int i = 0; i < grid.get_n(); ++i)
    {
        for (int j = 0; j < grid.get_n(); ++j)
        {
            for (int k = 0; k < grid.get_n(); ++k)
            {
                int J_row_idx = grid.idxNoSpin(i, j, k);
                complex<double> Jx_point(0.0, 0.0);
                complex<double> Jy_point(0.0, 0.0);
                complex<double> Jz_point(0.0, 0.0);

                for (int col = 0; col < psi.cols(); ++col)
                {
                    // These are local to each (i,j,k,col) part of the iteration within a thread
                    Eigen::Matrix<complex<double>, 2, 3> chiGrad; // Stores nabla_x chi, nabla_y chi, nabla_z chi for spin up/down
                    Eigen::Vector2cd chi;                         // Spinor for current point and orbital

                    for (int s = 0; s < 2; ++s) // s=0 (spin up), s=1 (spin down)
                    {
                        chiGrad(s, 0) = Operators::derivative(psi.col(col), i, j, k, s, grid, 'x'); // d/dx psi_s
                        chiGrad(s, 1) = Operators::derivative(psi.col(col), i, j, k, s, grid, 'y'); // d/dy psi_s
                        chiGrad(s, 2) = Operators::derivative(psi.col(col), i, j, k, s, grid, 'z'); // d/dz psi_s
                        
                        chi(s) = psi(grid.idx(i, j, k, s), col);
                    }

                    // J_alpha = sum_orbitals chi_dagger * (sigma_beta * nabla_gamma - sigma_gamma * nabla_beta) * chi (with Levi-Civita for alpha,beta,gamma)
                    // The .dot() method for Eigen complex vectors computes v1.adjoint() * v2 (v1^dagger * v2)
                    Jx_point += chi.dot(pauli_matrices[2] * chiGrad.col(1) - pauli_matrices[1] * chiGrad.col(2));
                    Jy_point += chi.dot(pauli_matrices[0] * chiGrad.col(2) - pauli_matrices[2] * chiGrad.col(0));
                    Jz_point += chi.dot(pauli_matrices[1] * chiGrad.col(0) - pauli_matrices[0] * chiGrad.col(1));
                }
                J(J_row_idx, 0) = Jx_point;
                J(J_row_idx, 1) = Jy_point;
                J(J_row_idx, 2) = Jz_point;
            }
        }
    }
    // The final multiplication by -nuclearConstants::img is a scalar-matrix operation
    // applied to the fully computed J matrix. This is fine outside the parallel region.
    return -nuclearConstants::img * J;
}
std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXcd> hfVectors(const Eigen::MatrixXcd& psi, const Grid& grid) {
    using std::complex;
    Eigen::VectorXd rho(grid.get_total_spatial_points());
    Eigen::VectorXd tau(grid.get_total_spatial_points());
    Eigen::MatrixXcd J(grid.get_total_spatial_points(), 3);
    J.setZero();
    rho.setZero();
    tau.setZero();
    int n = grid.get_n();
    auto pauli_matrices = nuclearConstants::getPauli();

    #pragma omp parallel for collapse(3)
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            for (int k = 0; k < n; ++k)
            {
                int rho_idx = grid.idxNoSpin(i, j, k);
                int psi_idx_s0 = grid.idx(i, j, k, 0);
                int psi_idx_s1 = grid.idx(i, j, k, 1);

                double point_density = 0.0;
                complex<double> Jx_point(0.0, 0.0);
                complex<double> Jy_point(0.0, 0.0);
                complex<double> Jz_point(0.0, 0.0);
                for (int col = 0; col < psi.cols(); ++col)
                {
                    Eigen::Matrix<complex<double>, 2, 3> chiGrad; // Stores nabla_x chi, nabla_y chi, nabla_z chi for spin up/down
                    Eigen::Vector2cd chi;                         // Spinor for current point and orbital
                    point_density += std::norm(psi(psi_idx_s0, col)); // std::norm(complex) = |complex|^2
                    point_density += std::norm(psi(psi_idx_s1, col));
                    for(int s = 0; s < 2; ++s) {
                        std::complex<double> dx = Operators::derivative(psi.col(col), i, j, k, s, grid, 'x');
                        std::complex<double> dy = Operators::derivative(psi.col(col), i, j, k, s, grid, 'y');
                        std::complex<double> dz = Operators::derivative(psi.col(col), i, j, k, s, grid, 'z');
                        tau(rho_idx) += std::norm(dx) + std::norm(dy) + std::norm(dz);
                        chiGrad(s, 0) = Operators::derivative(psi.col(col), i, j, k, s, grid, 'x'); // d/dx psi_s
                        chiGrad(s, 1) = Operators::derivative(psi.col(col), i, j, k, s, grid, 'y'); // d/dy psi_s
                        chiGrad(s, 2) = Operators::derivative(psi.col(col), i, j, k, s, grid, 'z'); // d/dz psi_s
                        
                        chi(s) = psi(grid.idx(i, j, k, s), col);
                    }
                    Jx_point += chi.dot(pauli_matrices[2] * chiGrad.col(1) - pauli_matrices[1] * chiGrad.col(2));
                    Jy_point += chi.dot(pauli_matrices[0] * chiGrad.col(2) - pauli_matrices[2] * chiGrad.col(0));
                    Jz_point += chi.dot(pauli_matrices[1] * chiGrad.col(0) - pauli_matrices[0] * chiGrad.col(1));
                }
                J(rho_idx, 0) = Jx_point;
                J(rho_idx, 1) = Jy_point;
                J(rho_idx, 2) = Jz_point;
                rho(rho_idx) = point_density; // Each thread writes to a unique element of rho

            }
        }
    }

return std::tuple< Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXcd>(rho, tau, (-nuclearConstants::img)*J);
} 
    double roundHI(double x) {
        int c = ceil(x*2);
        int f = floor(x*2);

        if (c % 2 == 0) {
            return f / 2.0;
        }
        return c / 2.0;
    }
    double momFromSq(double x) {
        return((0.5*(-1 + sqrt(1 + 4*x))));
    }

    void printShells(const Eigen::MatrixXcd& psi, const Grid& grid) {
        using namespace std;
        using namespace nuclearConstants;
        roundHI(1.0);


  for (int i = 0; i < psi.cols(); ++i) {
    auto L2 = Operators::L2(psi.col(i), grid)
                  .dot(psi.col(i)) /
              (h_bar * h_bar);
    cout << "L2: " << round(momFromSq( L2.real())) << " | ";
    double J2exp = psi.col(i).dot(
             Operators::J2(psi.col(i), grid)).real()/
                (h_bar * h_bar);
    cout << "J2: "
         << roundHI(momFromSq(J2exp))
         << " | ";
    auto par = (psi.col(i).adjoint() *
                Operators::P(psi.col(i), grid));
    cout << "P: " << (par / par.norm()).real() << " | ";
    cout << "mz: "
         << roundHI(Operators::Jz(psi.col(i), grid).norm() / (h_bar))
         << endl;
  }
    }
}