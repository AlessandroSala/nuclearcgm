#pragma once
#include <vector>
#include <iostream>
#include <utility>
#include <cmath>
#include <algorithm>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/Core>
#include <iostream>
#include <thread>
#include <omp.h>
#include "types.hpp"

double f_constrained(const Eigen::VectorXd &x, const Eigen::VectorXd &Ax);
Eigen::VectorXd g_constrained(const Eigen::VectorXd &x, const Eigen::VectorXd &Ax);

std::pair<double, Eigen::VectorXd> find_eigenpair_constrained(const Eigen::SparseMatrix<double> &A, const Eigen::VectorXd &X0, int max_iter , double tol);

/**
 * @brief Ortogonalizzazione di Gram-Schmidt Modificata Complessa rispetto a B reale.
 * Modifica V sul posto. Assume B Reale Simmetrica Definitia Positiva (SPD).
 *
 * @param V Matrice complessa le cui colonne verranno B-ortogonalizzate.
 * @param B Matrice reale SPD che definisce il prodotto scalare.
 */
void b_modified_gram_schmidt_complex(ComplexDenseMatrix& V, const RealSparseMatrix& B);
void b_modified_gram_schmidt_complex_no_B(ComplexDenseMatrix& V);
/**
 * @brief Procedura di Rayleigh-Ritz Complessa per Ax = lambda Bx (A Hermitiana, B Reale SPD).
 *
 * @param V Matrice complessa la cui base (colonne) Spannano il sottospazio. DEVE essere B-ortonormale.
 * @param A Matrice Complessa Hermitiana A.
 * @param B Matrice Reale SPD B.
 * @param nev Numero di autovalori/autovettori desiderati.
 * @return std::pair<ComplexDenseMatrix, DenseVector> Pair contenente:
 * - Matrice C complessa dei coefficienti (colonne sono autovettori proiettati).
 * - Vettore Lambda reale dei corrispondenti autovalori.
 */
std::pair<ComplexDenseMatrix, DenseVector> rayleighRitz_complex(
    const ComplexDenseMatrix &V,
    const ComplexSparseMatrix &A, // A è complessa Hermitiana
    const RealSparseMatrix &B,    // B è reale SPD
    int nev);
std::pair<ComplexDenseMatrix, DenseVector> rayleighRitz_complex_no_B(
    const ComplexDenseMatrix &V,
    const ComplexSparseMatrix &A, // A è complessa Hermitiana
    int nev);
/**
 * @brief Algoritmo GCGM Complesso per Ax = lambda Bx (A Hermitiana, B Reale SPD).
 *
 * @param A Matrice Complessa Hermitiana A.
 * @param B Matrice Reale SPD B.
 * @param X_initial Stima iniziale Complessa per gli autovettori.
 * @param nev Numero di autocoppie da calcolare.
 * @param shift Shift iniziale (reale).
 * @param max_iter Max iterazioni GCGM.
 * @param tolerance Tolleranza sul residuo relativo.
 * @param cg_steps Numero di passi del solver iterativo (es. BiCGSTAB) per W.
 * @return std::pair<ComplexDenseMatrix, DenseVector> Pair contenente:
 * - Matrice X complessa degli autovettori.
 * - Vettore Lambda reale degli autovalori.
 */
std::pair<ComplexDenseMatrix, DenseVector> gcgm_complex(
    const ComplexSparseMatrix &A, // A Complessa Hermitiana
    const RealSparseMatrix &B,    // B Reale SPD
    const ComplexDenseMatrix &X_initial,
    int nev,
    double shift, // Shift rimane reale
    int max_iter,
    double tolerance,
    int cg_steps, // Passi BiCGSTAB (o altro solver complesso)
    double cg_tol
);

std::pair<ComplexDenseMatrix, DenseVector> gcgm_complex_no_B(
    const ComplexSparseMatrix &A, // A Complessa Hermitiana
    const ComplexDenseMatrix &X_initial,
    int nev,
    double shift, // Shift rimane reale
    int max_iter,
    double tolerance,
    int cg_steps, // Passi BiCGSTAB (o altro solver complesso)
    double cg_tol
);

Eigen::MatrixXd random_orthonormal_matrix(int n, int k);