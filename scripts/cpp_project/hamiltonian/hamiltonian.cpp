#include "hamiltonian.hpp"
#include "grid/grid.hpp"
#include "potentials/potential.hpp"
#include "common/constants.hpp"
#include <Eigen/Sparse>
#include <vector>
#include <cmath>     // For std::abs
#include <stdexcept> // For error checking
#include <omp.h>     // For OpenMP parallelization

// Constructor implementation
Hamiltonian::Hamiltonian(std::shared_ptr<Grid> grid,
                         const std::vector<std::shared_ptr<Potential>> &potential_terms)
    : grid_ptr_(grid), potential_terms_(potential_terms)
{
    if (!grid_ptr_)
    {
        throw std::invalid_argument("Hamiltonian error: Grid pointer cannot be null.");
    }
}

ComplexSparseMatrix Hamiltonian::build_matrix()
{
    if (!grid_ptr_)
    {
        throw std::runtime_error("Hamiltonian error: Grid pointer is null during build.");
    }

    // Get grid parameters
    const size_t n = grid_ptr_->get_n();
    const double h = grid_ptr_->get_h();
    if (n == 0 || std::abs(h) < 1e-15)
    {
        throw std::runtime_error("Hamiltonian error: Invalid grid parameters (n=0 or h=0).");
    }

    // Get physical constant C (related to mass and hbar)
    const double C_const = nuclearConstants::C;
    if (std::abs(C_const) < 1e-15)
    {
        throw std::runtime_error("Hamiltonian error: Constant C is zero.");
    }
    double Chh = C_const * h * h;
    if (std::abs(Chh) < 1e-15)
    {
        throw std::runtime_error("Hamiltonian error: C*h*h is zero.");
    }

    // Pre-calculate kinetic term coefficients
    ComplexScalar diag_kinetic_term = -6.0 / Chh;
    ComplexScalar offdiag_kinetic_term = 1.0 / Chh;

    size_t N_total = grid_ptr_->get_total_points();

    // Vector to store non-zero elements (triplets)
    std::vector<Eigen::Triplet<ComplexScalar>> tripletList;

// Parallelize the outer spatial loops
#pragma omp parallel for collapse(3) schedule(static)
    for (size_t k = 0; k < n; ++k)
    { // Loop over source k
        for (size_t j = 0; j < n; ++j)
        { // Loop over source j
            for (size_t i = 0; i < n; ++i)
            { // Loop over source i

                // Determine neighborhood boundaries for target indices (i1, j1, k1)
                size_t i1_min = (i > 0) ? i - 1 : 0;
                size_t i1_max = std::min(n - 1, i + 1);
                size_t j1_min = (j > 0) ? j - 1 : 0;
                size_t j1_max = std::min(n - 1, j + 1);
                size_t k1_min = (k > 0) ? k - 1 : 0;
                size_t k1_max = std::min(n - 1, k + 1);

                for (size_t k1 = k1_min; k1 <= k1_max; ++k1)
                { // Loop over target k1
                    for (size_t j1 = j1_min; j1 <= j1_max; ++j1)
                    { // Loop over target j1
                        for (size_t i1 = i1_min; i1 <= i1_max; ++i1)
                        { // Loop over target i1

                            // Original code looped s, s1 inside here. Let's do the same.
                            for (size_t s = 0; s < 2; ++s)
                            { // Loop over source spin s
                                for (size_t s1 = 0; s1 < 2; ++s1)
                                { // Loop over target spin s1

                                    // --- Calculate total value for H(n0, n1) ---
                                    // Equivalent to the original A(i,j,k,s, i1,j1,k1,s1) function

                                    // 1. Calculate Kinetic Term Contribution
                                    ComplexScalar kinetic_term(0.0, 0.0);
                                    if (i == i1 && j == j1 && k == k1 && s == s1)
                                    {
                                        kinetic_term = diag_kinetic_term;
                                    }
                                    else if (s == s1 && (std::abs(static_cast<int>(i1) - static_cast<int>(i)) +
                                                             std::abs(static_cast<int>(j1) - static_cast<int>(j)) +
                                                             std::abs(static_cast<int>(k1) - static_cast<int>(k)) ==
                                                         1))
                                    {
                                        // Check if it's a nearest spatial neighbour with same spin
                                        kinetic_term = offdiag_kinetic_term;
                                    }

                                    // 2. Calculate Total Potential Term Contribution
                                    ComplexScalar potential_term(0.0, 0.0);
                                    // Loop through the list of potential objects provided
                                    for (const auto &term_ptr : potential_terms_)
                                    {
                                        if (term_ptr)
                                        { // Check if pointer is valid
                                            potential_term += term_ptr->getElement(i, j, k, s, i1, j1, k1, s1, *grid_ptr_);
                                        }
                                    }

                                    // 3. Combine terms
                                    ComplexScalar total_val = kinetic_term + potential_term;

                                    size_t n0 = grid_ptr_->idx(i, j, k, s); // Calculate indices just before storing
                                    size_t n1 = grid_ptr_->idx(i1, j1, k1, s1);
                                    #pragma omp critical
                                    {
                                        tripletList.emplace_back(n0, n1, total_val);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // 5. Construct the sparse matrix from triplets
    ComplexSparseMatrix H(N_total, N_total);
    H.setFromTriplets(tripletList.begin(), tripletList.end());

    // Optional: Compress the matrix representation
    H.makeCompressed();

    return H;
}