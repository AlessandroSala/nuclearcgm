#include "hamiltonian.hpp"
#include "Eigen/src/SparseCore/SparseMatrix.h"
#include "constants.hpp"
#include "grid.hpp"
#include "potential.hpp"
#include "types.hpp"
#include <Eigen/Sparse>
#include <cmath>
#include <omp.h>
#include <stdexcept>
#include <vector>

Hamiltonian::Hamiltonian(
    std::shared_ptr<Grid> grid,
    const std::vector<std::shared_ptr<Potential>> &potential_terms)
    : grid_ptr_(grid), potential_terms_(potential_terms) {
  if (!grid_ptr_) {
    throw std::invalid_argument(
        "Hamiltonian error: Grid pointer cannot be null.");
  }
}

ComplexSparseMatrix Hamiltonian::build_matrix7p() {
  if (!grid_ptr_) {
    throw std::runtime_error(
        "Hamiltonian error: Grid pointer is null during build.");
  }

  const int n = grid_ptr_->get_n();
  const double h = grid_ptr_->get_h();
  if (n == 0 || std::abs(h) < 1e-15) {
    throw std::runtime_error(
        "Hamiltonian error: Invalid grid parameters (n=0 or h=0).");
  }

  const double C_const = nuclearConstants::C;
  if (std::abs(C_const) < 1e-15) {
    throw std::runtime_error("Hamiltonian error: Constant C is zero.");
  }
  double Chh = C_const * h * h;
  if (std::abs(Chh) < 1e-15) {
    throw std::runtime_error("Hamiltonian error: C*h*h is zero.");
  }

  ComplexScalar diag_kinetic_term = -3 * (245.0 / 90.0) / Chh;
  ComplexScalar offdiag_kinetic_term = (135.0 / 90.0) / Chh;
  ComplexScalar offdiag_kinetic_term_2 = -(13.5 / 90.0) / Chh;
  ComplexScalar offdiag_kinetic_term_3 = (1.0 / 90.0) / Chh;

  size_t N_total = grid_ptr_->get_total_points();

  std::vector<std::vector<Eigen::Triplet<ComplexScalar>>> threadTriplets(
      omp_get_max_threads());

#pragma omp parallel for collapse(3)
  for (int k = 0; k < n; ++k) {
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < n; ++i) {
        auto &localTriplets = threadTriplets[omp_get_thread_num()];

        const int k1_min = std::max(0, k - 3);
        const int k1_max = std::min(n - 1, k + 3);
        const int j1_min = std::max(0, j - 3);
        const int j1_max = std::min(n - 1, j + 3);
        const int i1_min = std::max(0, i - 3);
        const int i1_max = std::min(n - 1, i + 3);

        for (size_t s = 0; s < 2; ++s) {
          for (size_t s1 = 0; s1 < 2; ++s1) {
            for (int k1 = k1_min; k1 <= k1_max; ++k1) {
              for (int j1 = j1_min; j1 <= j1_max; ++j1) {
                for (int i1 = i1_min; i1 <= i1_max; ++i1) {
                  ComplexScalar val(0.0, 0.0);
                  if (i == i1 && j == j1 && k == k1 && s == s1) {
                    val = diag_kinetic_term;
                  } else if (s == s1 &&
                             ((i == i1 && j == j1 && std::abs(k1 - k) == 1) ||
                              (i == i1 && k == k1 && std::abs(j1 - j) == 1) ||
                              (j == j1 && k == k1 && std::abs(i1 - i) == 1))) {
                    val = offdiag_kinetic_term;
                  } else if (s == s1 &&
                             ((i == i1 && j == j1 && std::abs(k1 - k) == 2) ||
                              (i == i1 && k == k1 && std::abs(j1 - j) == 2) ||
                              (j == j1 && k == k1 && std::abs(i1 - i) == 2))) {
                    val = offdiag_kinetic_term_2;
                  } else if (s == s1 &&
                             ((i == i1 && j == j1 && std::abs(k1 - k) == 3) ||
                              (i == i1 && k == k1 && std::abs(j1 - j) == 3) ||
                              (j == j1 && k == k1 && std::abs(i1 - i) == 3))) {
                    val = offdiag_kinetic_term_3;
                  }

                  for (const auto &term_ptr : potential_terms_) {
                    if (term_ptr) {
                      val += term_ptr->getElement5p(i, j, k, s, i1, j1, k1, s1,
                                                    *grid_ptr_);
                    }
                  }

                  if (val != ComplexScalar(0, 0)) {
                    const size_t n0 = grid_ptr_->idx(i, j, k, s);
                    const size_t n1 = grid_ptr_->idx(i1, j1, k1, s1);
                    localTriplets.emplace_back(n0, n1, val);
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  std::vector<Eigen::Triplet<ComplexScalar>> tripletList;
  for (auto &local : threadTriplets) {
    tripletList.insert(tripletList.end(), local.begin(), local.end());
  }

  ComplexSparseMatrix H(N_total, N_total);
  H.setFromTriplets(tripletList.begin(), tripletList.end());
  H.makeCompressed();

  return H;
}

ComplexSparseMatrix Hamiltonian::build_matrix5p_symm() {

  const int n = grid_ptr_->get_n() / 2;
  const double h = grid_ptr_->get_h();

  const double C_const = nuclearConstants::C;
  double Chh = C_const * h * h;

  ComplexScalar diag_kinetic_term = -(90.0 / 12.0) / Chh;
  ComplexScalar offdiag_kinetic_term = (16.0 / 12.0) / Chh;
  ComplexScalar offdiag_kinetic_term_2 = -(1.0 / 12.0) / Chh;

  std::vector<std::vector<Eigen::Triplet<ComplexScalar>>> threadTriplets(
      omp_get_max_threads());

#pragma omp parallel for collapse(3)
  for (int k = 0; k < n; ++k) {
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < n; ++i) {
        auto &localTriplets = threadTriplets[omp_get_thread_num()];

        const int k1_min = std::max(0, k - 2);
        const int k1_max = std::min(n - 1, k + 2);
        const int j1_min = std::max(0, j - 2);
        const int j1_max = std::min(n - 1, j + 2);
        const int i1_min = std::max(0, i - 2);
        const int i1_max = std::min(n - 1, i + 2);

        for (size_t s = 0; s < 2; ++s) {
          for (size_t s1 = 0; s1 < 2; ++s1) {
            for (int k1 = k1_min; k1 <= k1_max; ++k1) {
              for (int j1 = j1_min; j1 <= j1_max; ++j1) {
                for (int i1 = i1_min; i1 <= i1_max; ++i1) {
                  ComplexScalar val(0.0, 0.0);
                  if (i == i1 && j == j1 && k == k1 && s == s1) {
                    val = diag_kinetic_term;
                  } else if (s == s1 &&
                             ((i == i1 && j == j1 && std::abs(k1 - k) == 1) ||
                              (i == i1 && k == k1 && std::abs(j1 - j) == 1) ||
                              (j == j1 && k == k1 && std::abs(i1 - i) == 1))) {
                    val = offdiag_kinetic_term;
                  } else if (s == s1 &&
                             ((i == i1 && j == j1 && std::abs(k1 - k) == 2) ||
                              (i == i1 && k == k1 && std::abs(j1 - j) == 2) ||
                              (j == j1 && k == k1 && std::abs(i1 - i) == 2))) {
                    val = offdiag_kinetic_term_2;
                  }

                  for (const auto &term_ptr : potential_terms_) {
                    if (term_ptr) {
                      val += term_ptr->getElement5p(i, j, k, s, i1, j1, k1, s1,
                                                    *grid_ptr_);
                    }
                  }

                  if (val != ComplexScalar(0, 0)) {
                    const size_t n0 = grid_ptr_->idx(i, j, k, s);
                    const size_t n1 = grid_ptr_->idx(i1, j1, k1, s1);
                    localTriplets.emplace_back(n0, n1, val);
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  std::vector<Eigen::Triplet<ComplexScalar>> tripletList;
  for (auto &local : threadTriplets) {
    tripletList.insert(tripletList.end(), local.begin(), local.end());
  }

  int N_total = n * n * n * n * 2;
  ComplexSparseMatrix H(N_total, N_total);
  H.setFromTriplets(tripletList.begin(), tripletList.end());
  H.makeCompressed();

  return H;
}
ComplexSparseMatrix Hamiltonian::build_matrix5p() {
  if (!grid_ptr_) {
    throw std::runtime_error(
        "Hamiltonian error: Grid pointer is null during build.");
  }

  const int n = grid_ptr_->get_n();
  const double h = grid_ptr_->get_h();
  if (n == 0 || std::abs(h) < 1e-15) {
    throw std::runtime_error(
        "Hamiltonian error: Invalid grid parameters (n=0 or h=0).");
  }

  const double C_const = nuclearConstants::C;
  if (std::abs(C_const) < 1e-15) {
    throw std::runtime_error("Hamiltonian error: Constant C is zero.");
  }
  double Chh = C_const * h * h;
  if (std::abs(Chh) < 1e-15) {
    throw std::runtime_error("Hamiltonian error: C*h*h is zero.");
  }

  ComplexScalar diag_kinetic_term = -(90.0 / 12.0) / Chh;
  ComplexScalar offdiag_kinetic_term = (16.0 / 12.0) / Chh;
  ComplexScalar offdiag_kinetic_term_2 = -(1.0 / 12.0) / Chh;

  size_t N_total = grid_ptr_->get_total_points();

  std::vector<std::vector<Eigen::Triplet<ComplexScalar>>> threadTriplets(
      omp_get_max_threads());

#pragma omp parallel for collapse(3)
  for (int k = 0; k < n; ++k) {
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < n; ++i) {
        auto &localTriplets = threadTriplets[omp_get_thread_num()];

        const int k1_min = std::max(0, k - 2);
        const int k1_max = std::min(n - 1, k + 2);
        const int j1_min = std::max(0, j - 2);
        const int j1_max = std::min(n - 1, j + 2);
        const int i1_min = std::max(0, i - 2);
        const int i1_max = std::min(n - 1, i + 2);

        for (size_t s = 0; s < 2; ++s) {
          for (size_t s1 = 0; s1 < 2; ++s1) {
            for (int k1 = k1_min; k1 <= k1_max; ++k1) {
              for (int j1 = j1_min; j1 <= j1_max; ++j1) {
                for (int i1 = i1_min; i1 <= i1_max; ++i1) {
                  ComplexScalar val(0.0, 0.0);
                  if (i == i1 && j == j1 && k == k1 && s == s1) {
                    val = diag_kinetic_term;
                  } else if (s == s1 &&
                             ((i == i1 && j == j1 && std::abs(k1 - k) == 1) ||
                              (i == i1 && k == k1 && std::abs(j1 - j) == 1) ||
                              (j == j1 && k == k1 && std::abs(i1 - i) == 1))) {
                    val = offdiag_kinetic_term;
                  } else if (s == s1 &&
                             ((i == i1 && j == j1 && std::abs(k1 - k) == 2) ||
                              (i == i1 && k == k1 && std::abs(j1 - j) == 2) ||
                              (j == j1 && k == k1 && std::abs(i1 - i) == 2))) {
                    val = offdiag_kinetic_term_2;
                  }

                  for (const auto &term_ptr : potential_terms_) {
                    if (term_ptr) {
                      val += term_ptr->getElement5p(i, j, k, s, i1, j1, k1, s1,
                                                    *grid_ptr_);
                    }
                  }

                  if (val != ComplexScalar(0, 0)) {
                    const size_t n0 = grid_ptr_->idx(i, j, k, s);
                    const size_t n1 = grid_ptr_->idx(i1, j1, k1, s1);
                    localTriplets.emplace_back(n0, n1, val);
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  std::vector<Eigen::Triplet<ComplexScalar>> tripletList;
  for (auto &local : threadTriplets) {
    tripletList.insert(tripletList.end(), local.begin(), local.end());
  }

  ComplexSparseMatrix H(N_total, N_total);
  H.setFromTriplets(tripletList.begin(), tripletList.end());
  H.makeCompressed();

  return H;
}
ComplexSparseMatrix Hamiltonian::buildMatrix() {
  if (!grid_ptr_) {
    throw std::runtime_error(
        "Hamiltonian error: Grid pointer is null during build.");
  }

  const int n = grid_ptr_->get_n();
  const double h = grid_ptr_->get_h();
  if (n == 0 || std::abs(h) < 1e-15) {
    throw std::runtime_error(
        "Hamiltonian error: Invalid grid parameters (n=0 or h=0).");
  }

  size_t N_total = grid_ptr_->get_total_points();

  std::vector<std::vector<Eigen::Triplet<ComplexScalar>>> threadTriplets(
      omp_get_max_threads());

#pragma omp parallel for collapse(3)
  for (int k = 0; k < n; ++k) {
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < n; ++i) {
        auto &localTriplets = threadTriplets[omp_get_thread_num()];

        const int k1_min = std::max(0, k - 2);
        const int k1_max = std::min(n - 1, k + 2);
        const int j1_min = std::max(0, j - 2);
        const int j1_max = std::min(n - 1, j + 2);
        const int i1_min = std::max(0, i - 2);
        const int i1_max = std::min(n - 1, i + 2);

        for (size_t s = 0; s < 2; ++s) {
          for (size_t s1 = 0; s1 < 2; ++s1) {
            for (int k1 = k1_min; k1 <= k1_max; ++k1) {
              for (int j1 = j1_min; j1 <= j1_max; ++j1) {
                for (int i1 = i1_min; i1 <= i1_max; ++i1) {
                  ComplexScalar val(0.0, 0.0);

                  for (const auto &term_ptr : potential_terms_) {
                    if (term_ptr) {
                      val += term_ptr->getElement5p(i, j, k, s, i1, j1, k1, s1,
                                                    *grid_ptr_);
                    }
                  }

                  if (val != ComplexScalar(0.0, 0.0)) {
                    const size_t n0 = grid_ptr_->idx(i, j, k, s);
                    const size_t n1 = grid_ptr_->idx(i1, j1, k1, s1);
                    localTriplets.emplace_back(n0, n1, val);
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  std::vector<Eigen::Triplet<ComplexScalar>> tripletList;
  for (auto &local : threadTriplets) {
    tripletList.insert(tripletList.end(), local.begin(), local.end());
  }

  ComplexSparseMatrix H(N_total, N_total);
  H.setFromTriplets(tripletList.begin(), tripletList.end());
  H.makeCompressed();

  return H;
}

Eigen::SparseMatrix<double> Hamiltonian::buildMatrixNoSpin() {
  if (!grid_ptr_) {
    throw std::runtime_error(
        "Hamiltonian error: Grid pointer is null during build.");
  }

  const int n = grid_ptr_->get_n();
  const double h = grid_ptr_->get_h();
  if (n == 0 || std::abs(h) < 1e-15) {
    throw std::runtime_error(
        "Hamiltonian error: Invalid grid parameters (n=0 or h=0).");
  }

  size_t N_total = grid_ptr_->get_total_spatial_points();

  std::vector<std::vector<Eigen::Triplet<double>>> threadTriplets(
      omp_get_max_threads());

#pragma omp parallel for collapse(3)
  for (int k = 0; k < n; ++k) {
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < n; ++i) {
        auto &localTriplets = threadTriplets[omp_get_thread_num()];

        const int k1_min = std::max(0, k - 2);
        const int k1_max = std::min(n - 1, k + 2);
        const int j1_min = std::max(0, j - 2);
        const int j1_max = std::min(n - 1, j + 2);
        const int i1_min = std::max(0, i - 2);
        const int i1_max = std::min(n - 1, i + 2);

        for (int k1 = k1_min; k1 <= k1_max; ++k1) {
          for (int j1 = j1_min; j1 <= j1_max; ++j1) {
            for (int i1 = i1_min; i1 <= i1_max; ++i1) {
              double val = 0.0;

              for (const auto &term_ptr : potential_terms_) {
                if (term_ptr) {
                  val +=
                      term_ptr
                          ->getElement5p(i, j, k, 0, i1, j1, k1, 0, *grid_ptr_)
                          .real();
                }
              }

              if (val != 0.0) {
                const size_t n0 = grid_ptr_->idxNoSpin(i, j, k);
                const size_t n1 = grid_ptr_->idxNoSpin(i1, j1, k1);
                localTriplets.emplace_back(n0, n1, val);
              }
            }
          }
        }
      }
    }
  }

  std::vector<Eigen::Triplet<double>> tripletList;
  for (auto &local : threadTriplets) {
    tripletList.insert(tripletList.end(), local.begin(), local.end());
  }

  Eigen::SparseMatrix<double> H(N_total, N_total);
  H.setFromTriplets(tripletList.begin(), tripletList.end());
  H.makeCompressed();

  return H;
}
