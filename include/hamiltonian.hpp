#pragma once

#include "potential.hpp"
#include "types.hpp"
#include <memory> // For std::shared_ptr
#include <vector>

class Grid;

/**
 * @brief Builds the sparse Hamiltonian matrix
 *
 * Combines the kinetic energy term (finite difference Laplacian) with
 * various potential energy terms provided through the PotentialTerm interface.
 */
class Hamiltonian {
public:
  /**
   * @brief Constructs the Hamiltonian builder.
   *
   * @param grid A shared pointer to the Grid object defining the spatial
   * discretization.
   * @param potential_terms A vector of shared pointers to PotentialTerm objects
   * representing interactions (e.g., Woods-Saxon, Spin-Orbit).
   */
  Hamiltonian(std::shared_ptr<Grid> grid,
              const std::vector<std::shared_ptr<Potential>> &potential_terms);

  /**
   * @brief Builds and returns the sparse Hamiltonian matrix.
   * @return ComplexSparseMatrix The constructed Hamiltonian matrix.
   */
  // ComplexSparseMatrix build_matrix();
  ComplexSparseMatrix buildMatrix();
  Eigen::SparseMatrix<double> buildMatrixNoSpin();
  ComplexSparseMatrix build_matrix5p();
  ComplexSparseMatrix build_matrix5p_symm();
  ComplexSparseMatrix build_matrix7p();

  inline double getzero() { return 0.0; }

private:
  std::shared_ptr<Grid> grid_ptr_; // Pointer to the grid configuration
  std::vector<std::shared_ptr<Potential>>
      potential_terms_; // List of potential terms
};
