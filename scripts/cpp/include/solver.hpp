#pragma once
#include "types.hpp"
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Sparse>
#include <omp.h>
#include <utility>

typedef enum EigenpairsOrdering { MATCH_PREVIOUS, ASCENDING_ENERGIES } ord;

double f_constrained(const Eigen::VectorXd &x, const Eigen::VectorXd &Ax);
Eigen::VectorXd g_constrained(const Eigen::VectorXd &x,
                              const Eigen::VectorXd &Ax);

std::pair<double, Eigen::VectorXd>
find_eigenpair_constrained(const Eigen::SparseMatrix<double> &A,
                           const Eigen::VectorXd &X0, int max_iter, double tol);

void b_modified_gram_schmidt_complex(ComplexDenseMatrix &V,
                                     const ComplexSparseMatrix &B);
void b_modified_gram_schmidt_complex(ComplexDenseMatrix &V,
                                     const RealSparseMatrix &B);
void b_modified_gram_schmidt_complex_no_B(ComplexDenseMatrix &V);
std::pair<ComplexDenseMatrix, DenseVector>
rayleighRitz_complex(const ComplexDenseMatrix &V, const ComplexSparseMatrix &A,
                     const RealSparseMatrix &B, int nev);
std::pair<ComplexDenseMatrix, DenseVector>
rayleighRitz_complex_no_B(const ComplexDenseMatrix &V,
                          const ComplexSparseMatrix &A, int nev);
std::pair<ComplexDenseMatrix, DenseVector>
gcgm_complex(const ComplexSparseMatrix &A, const RealSparseMatrix &B,
             const ComplexDenseMatrix &X_initial, int nev, double shift,
             int max_iter, double tolerance, int cg_steps, double cg_tol,
             bool benchmark);

std::pair<ComplexDenseMatrix, DenseVector>
gcgm_complex_no_B(const ComplexSparseMatrix &A,
                  const ComplexDenseMatrix &X_initial, int nev, double shift,
                  int max_iter, double tolerance, int cg_steps, double cg_tol,
                  bool benchmark, int blockSize);

std::pair<ComplexDenseMatrix, DenseVector> gcgm_complex_no_B_lock(
    const ComplexSparseMatrix &A, const ComplexDenseMatrix &X_initial,
    const ComplexDenseMatrix &ConjDir, int nev, double shift, int max_iter,
    double tolerance, int cg_steps, double cg_tol, bool benchmark,
    EigenpairsOrdering ordering);
