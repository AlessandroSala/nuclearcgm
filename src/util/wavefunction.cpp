#include "EDF.hpp"
#include "VariadicTable.h"
#include "constants.hpp"
#include "coulomb/laplacian_potential.hpp"
#include "hamiltonian.hpp"
#include "input_parser.hpp"
#include "operators/differential_operators.hpp"
#include "operators/integral_operators.hpp"
#include "spherical_harmonics.hpp"
#include "util/shell.hpp"
#include <cmath>
#include <complex>
#include <fstream>
#include <iostream>
#include <memory>
#include <omp.h>
#include <vector>

namespace Wavefunction {

Eigen::VectorXd density(const Eigen::MatrixXcd &psi, const Grid &grid) {
  int n = grid.get_n();
  Eigen::VectorXd rho(grid.get_total_spatial_points());
  rho.setZero();

#pragma omp parallel for collapse(3)
  for (int k = 0; k < n; ++k) {
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < n; ++i) {
        int rho_idx = grid.idxNoSpin(i, j, k);
        int psi_idx_s0 = grid.idx(i, j, k, 0);
        int psi_idx_s1 = grid.idx(i, j, k, 1);

        double point_density = 0.0;
        for (int col = 0; col < psi.cols(); ++col) {
          point_density += std::norm(psi(psi_idx_s0, col));
          point_density += std::norm(psi(psi_idx_s1, col));
        }
        rho(rho_idx) = point_density;
      }
    }
  }
  return rho;
}
Eigen::VectorXd exchangeCoulombField(Eigen::VectorXd &rhoP, const Grid &grid) {

  Eigen::VectorXd res = Eigen::VectorXd::Zero(grid.get_total_spatial_points());
  using namespace nuclearConstants;

  res -= e2 * pow(3.0 / M_PI, 1.0 / 3.0) * rhoP.array().pow(1.0 / 3.0).matrix();

  return res;
}

Eigen::VectorXd coulombFieldPoisson(Eigen::VectorXd &rhoP, const Grid &grid,
                                    int Z, std::shared_ptr<Eigen::VectorXd> U) {

  using namespace Eigen;
  using nuclearConstants::e2;
  using Operators::integral;
  using SphericalHarmonics::Y20;
  using SphericalHarmonics::Y22;
  using std::make_shared;
  using std::shared_ptr;
  using std::sqrt;
  using std::vector;

  double pi = M_PI;

  double eps0inv = e2 * 4 * M_PI; // Mev fm
  Eigen::VectorXd Q20int(grid.get_total_spatial_points());
  Eigen::VectorXd Q22int(grid.get_total_spatial_points());

#pragma omp parallel for collapse(3)
  for (int i = 0; i < grid.get_n(); i++) {
    for (int j = 0; j < grid.get_n(); j++) {
      for (int k = 0; k < grid.get_n(); k++) {
        int idx = grid.idxNoSpin(i, j, k);
        double x = grid.get_xs()[i];
        double y = grid.get_ys()[j];
        double z = grid.get_zs()[k];

        Q20int(idx) = rhoP(idx) * (2.0 * z * z - x * x - y * y);
        Q22int(idx) = rhoP(idx) * (x * x - y * y);
      }
    }
  }

  Q20int *= sqrt(5.0 / (16.0 * pi));
  Q22int *= sqrt(15.0 / (32.0 * pi));
  double Q20 = integral(Q20int, grid);
  double Q22 = integral(Q22int, grid);

  vector<shared_ptr<Potential>> pots;
  pots.push_back(make_shared<LaplacianPotential>());
  Hamiltonian ham(make_shared<Grid>(grid), pots);

  int n = grid.get_n();
  double a = grid.get_a();
  double h = grid.get_h();
  int nnew = n + 4;
  double hh = h * h;

  Eigen::VectorXd Uquad(nnew * nnew * nnew);

  auto idxNoSpin = [&](int i, int j, int k, int dim) {
    return i * dim * dim + j * dim + k;
  };

#pragma omp parallel for collapse(3)
  for (int k = 0; k < nnew; k++) {
    for (int j = 0; j < nnew; j++) {
      for (int i = 0; i < nnew; i++) {

        int idx = idxNoSpin(i, j, k, nnew);
        double x = -a - 2 * h + i * h;
        double y = -a - 2 * h + j * h;
        double z = -a - 2 * h + k * h;

        double r = sqrt(x * x + y * y + z * z);
        if (r < 1e-9)
          r = 1e-9;

        Uquad(idx) = (e2 * Z) / r;
        Uquad(idx) += e2 * (Y20(x, y, z) * Q20 + 2.0 * Y22(x, y, z) * Q22) / r;
      }
    }
  }

  SparseMatrix<double> mat = ham.buildMatrixNoSpin().real();
  VectorXd rhs = -eps0inv * rhoP;

// i = 0 (contributi da -1, -2)
#pragma omp parallel for collapse(2)
  for (int k = 0; k < n; ++k) {
    for (int j = 0; j < n; ++j) {
      int rhs_idx = idxNoSpin(0, j, k, n);
      rhs(rhs_idx) -=
          (16.0 / 12.0) * Uquad(idxNoSpin(1, j + 2, k + 2, nnew)) / hh;
      rhs(rhs_idx) +=
          (1.0 / 12.0) * Uquad(idxNoSpin(0, j + 2, k + 2, nnew)) / hh;
      int rhs_idx1 = idxNoSpin(1, j, k, n);
      rhs(rhs_idx1) +=
          (1.0 / 12.0) * Uquad(idxNoSpin(1, j + 2, k + 2, nnew)) / hh;
    }
  }

// i = n-1 (contributi da n, n+1)
#pragma omp parallel for collapse(2)
  for (int k = 0; k < n; ++k) {
    for (int j = 0; j < n; ++j) {
      int rhs_idx = idxNoSpin(n - 1, j, k, n);
      rhs(rhs_idx) -=
          (16.0 / 12.0) * Uquad(idxNoSpin(n + 2, j + 2, k + 2, nnew)) / hh;
      rhs(rhs_idx) +=
          (1.0 / 12.0) * Uquad(idxNoSpin(n + 3, j + 2, k + 2, nnew)) / hh;
      int rhs_idx1 = idxNoSpin(n - 2, j, k, n);
      rhs(rhs_idx1) +=
          (1.0 / 12.0) * Uquad(idxNoSpin(n + 2, j + 2, k + 2, nnew)) / hh;
    }
  }

// j = 0
#pragma omp parallel for collapse(2)
  for (int k = 0; k < n; ++k) {
    for (int i = 0; i < n; ++i) {
      int rhs_idx = idxNoSpin(i, 0, k, n);
      rhs(rhs_idx) -=
          (16.0 / 12.0) * Uquad(idxNoSpin(i + 2, 1, k + 2, nnew)) / hh;
      rhs(rhs_idx) +=
          (1.0 / 12.0) * Uquad(idxNoSpin(i + 2, 0, k + 2, nnew)) / hh;
      int rhs_idx1 = idxNoSpin(i, 1, k, n);
      rhs(rhs_idx1) +=
          (1.0 / 12.0) * Uquad(idxNoSpin(i + 2, 1, k + 2, nnew)) / hh;
    }
  }

// j = n-1
#pragma omp parallel for collapse(2)
  for (int k = 0; k < n; ++k) {
    for (int i = 0; i < n; ++i) {
      int rhs_idx = idxNoSpin(i, n - 1, k, n);
      rhs(rhs_idx) -=
          (16.0 / 12.0) * Uquad(idxNoSpin(i + 2, n + 2, k + 2, nnew)) / hh;
      rhs(rhs_idx) +=
          (1.0 / 12.0) * Uquad(idxNoSpin(i + 2, n + 3, k + 2, nnew)) / hh;
      int rhs_idx1 = idxNoSpin(i, n - 2, k, n);
      rhs(rhs_idx1) +=
          (1.0 / 12.0) * Uquad(idxNoSpin(i + 2, n + 2, k + 2, nnew)) / hh;
    }
  }

// k = 0
#pragma omp parallel for collapse(2)
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      int rhs_idx = idxNoSpin(i, j, 0, n);
      rhs(rhs_idx) -=
          (16.0 / 12.0) * Uquad(idxNoSpin(i + 2, j + 2, 1, nnew)) / hh;
      rhs(rhs_idx) +=
          (1.0 / 12.0) * Uquad(idxNoSpin(i + 2, j + 2, 0, nnew)) / hh;
      int rhs_idx1 = idxNoSpin(i, j, 1, n);
      rhs(rhs_idx1) +=
          (1.0 / 12.0) * Uquad(idxNoSpin(i + 2, j + 2, 1, nnew)) / hh;
    }
  }

// k = n-1
#pragma omp parallel for collapse(2)
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < n; ++i) {
      int rhs_idx = idxNoSpin(i, j, n - 1, n);
      rhs(rhs_idx) -=
          (16.0 / 12.0) * Uquad(idxNoSpin(i + 2, j + 2, n + 2, nnew)) / hh;
      rhs(rhs_idx) +=
          (1.0 / 12.0) * Uquad(idxNoSpin(i + 2, j + 2, n + 3, nnew)) / hh;
      int rhs_idx1 = idxNoSpin(i, j, n - 2, n);
      rhs(rhs_idx1) +=
          (1.0 / 12.0) * Uquad(idxNoSpin(i + 2, j + 2, n + 2, nnew)) / hh;
    }
  }

  BiCGSTAB<SparseMatrix<double>> solver;
  solver.compute(mat);
  if (U != nullptr)
    return solver.solveWithGuess(rhs, *U);
  else
    return solver.solve(rhs);
}

Eigen::VectorXd coulombField(Eigen::VectorXd &rho, const Grid &grid) {
  Eigen::VectorXd res = Eigen::VectorXd::Zero(grid.get_total_spatial_points());
  double h = grid.get_h();
  int n = grid.get_n();

#pragma omp parallel for collapse(3)
  for (int ii = 0; ii < n; ++ii) {
    for (int jj = 0; jj < n; ++jj) {
      for (int kk = 0; kk < n; ++kk) {
        int iNS = grid.idxNoSpin(ii, jj, kk);
        double potential_sum = 0.0;

        for (int k = 0; k < n; ++k) {
          for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
              if (ii == i && jj == j && kk == k) {
                continue;
              }
              int iNSP = grid.idxNoSpin(i, j, k);
              potential_sum += h * h * rho(iNSP) /
                               (sqrt((ii - i) * (ii - i) + (jj - j) * (jj - j) +
                                     (kk - k) * (kk - k)));
            }
          }
        }
        res(iNS) = potential_sum + rho(iNS) * h * h * h * 1.939285;
      }
    }
  }

  return res * nuclearConstants::e2;
}

Eigen::VectorXd field(Eigen::VectorXd &rho, Eigen::VectorXd &rhoQ,
                      Eigen::VectorXd &tau, Eigen::VectorXd &tauQ,
                      Eigen::VectorXd &nabla2rho, Eigen::VectorXd &nabla2rhoQ,
                      Eigen::VectorXd &divJ, Eigen::VectorXd &divJQ,
                      const Grid &grid, std::shared_ptr<EDF> interaction) {
  auto params = interaction->params;

  Eigen::VectorXd field(grid.get_total_spatial_points());
  field.setZero();

  // Un = U0 + U1, Up = U0 - U1
  // implies field expression is always U = (A0-A1)r + 2A1rQ
  // In terms of C0, C1: Ut = 2Ctr + 2Ctdr + Ctt + CtnJ

  // t0 * ((1.0 + 0.5 * x0) * rho - (x0 + 0.5) * rhoQ);
  field += (2 * params.C0rr - 2 * params.C1rr) * rho + 4 * params.C1rr * rhoQ;

  //  0.125 * (t1 * (2 + x1) + t2 * (2 + x2)) * tau;
  //  0.125 * (t2 * (2 * x2 + 1) - t1 * (2 * x1 + 1)) * tauQ;
  field += (params.C0rt - params.C1rt) * tau + 2 * params.C1rt * tauQ;

  // (1.0 / 16.0) * (t2 * (2 + x2) - 3 * t1 * (2 + x1)) * nabla2rho;
  // (1.0 / 16.0) * (3 * t1 * (2 * x1 + 1) + t2 * (2 * x2 + 1)) * nabla2rhoQ;
  field += (2 * params.C0rdr - 2 * params.C1rdr) * nabla2rho +
           4 * params.C1rdr * nabla2rhoQ;

  //  -0.5 * W0 * divJJQ, here the expression simplifies!!
  field += (params.C0nJ - params.C1nJ) * divJ + 2 * params.C1nJ * divJQ;

  // TODO: In case of strange coupling different from standard skyrme, check
  // again this expression, I'm too tired right now, this version should be good
  // for numerical stability
  double t3 = 48 * params.C0Drr / 3;
  double x3 = -0.5 - 24 * params.C1Drr / t3;

  double sigma = params.sigma;
  field +=
      ((t3 / 12.0) * pow(rho.array(), sigma - 1.0) *
       ((1.0 + 0.5 * x3) * (sigma + 2.0) * pow(rho.array(), 2.0) -
        (x3 + 0.5) *
            (sigma * (rhoQ.array().square() + (rho - rhoQ).array().square()) +
             2.0 * rho.array() * rhoQ.array())))

          .matrix();

  return field;
}

Eigen::Matrix3Xd spinDensity(const Eigen::MatrixXcd &psi, const Grid &grid) {
  Eigen::Matrix3Xd res =
      Eigen::Matrix3Xd::Zero(3, grid.get_total_spatial_points());
  auto pauli = nuclearConstants::getPauli();
  for (int col = 0; col < psi.cols(); ++col) {
    for (int k = 0; k < grid.get_n(); ++k) {
      for (int j = 0; j < grid.get_n(); ++j) {
        for (int i = 0; i < grid.get_n(); ++i) {
          int idx = grid.idx(i, j, k, 0);
          int idxNoSpin = grid.idxNoSpin(i, j, k);
          Eigen::Vector2cd chi = psi(Eigen::seq(idx, idx + 1), col);
          res(idxNoSpin, 0) += chi.dot(pauli[0] * chi).real();
          res(idxNoSpin, 1) += chi.dot(pauli[1] * chi).real();
          res(idxNoSpin, 2) += chi.dot(pauli[2] * chi).real();
        }
      }
    }
  }

  return res;
}

Eigen::VectorXd kineticDensityLagrange(const Eigen::MatrixXcd &psi) {

  auto grid = *Grid::getInstance();
  Eigen::VectorXd res = Eigen::VectorXd::Zero(grid.get_total_spatial_points());
  for (int i = 0; i < psi.cols(); ++i) {
    Eigen::MatrixX3cd grad = Operators::gradLagrange(psi.col(i));
    Eigen::VectorXd mult = (grad * grad.adjoint()).diagonal().real();
    for (int l = 0; l < grid.get_total_spatial_points(); ++l) {
      res(l) += mult(2 * l) + mult(2 * l + 1);
    }
  }

  return res;
}
Eigen::VectorXd kineticDensity(const Eigen::MatrixXcd &psi, const Grid &grid) {
  // using std::complex;
  Eigen::VectorXd res = Eigen::VectorXd::Zero(grid.get_total_spatial_points());
  for (int i = 0; i < psi.cols(); ++i) {
    Eigen::MatrixX3cd grad = Operators::grad(psi.col(i), grid);
    Eigen::VectorXd mult = (grad * grad.adjoint()).diagonal().real();
    for (int l = 0; l < grid.get_total_spatial_points(); ++l) {
      res(l) += mult(2 * l) + mult(2 * l + 1);
    }
  }

  return res;
}

void normalize(Eigen::MatrixXcd &psi, const Grid &grid) {

  for (int c = 0; c < psi.cols(); ++c) {
    Eigen::VectorXd density = Wavefunction::density(psi.col(c), grid);
    auto integral =
        Operators::integral(Wavefunction::density(psi.col(c), grid), grid);
    psi.col(c) /= sqrt(integral);
  }
}

Eigen::Matrix<double, Eigen::Dynamic, 9>
soDensityLagrange(const Eigen::MatrixXcd &psi) {
  using std::complex;
  auto grid = *Grid::getInstance();

  Eigen::Matrix<complex<double>, Eigen::Dynamic, 9> J(
      grid.get_total_spatial_points(), 9);
  J.setZero();

  auto pauli = nuclearConstants::getPauli(); // Call once outside the
                                             // parallel region

  using nuclearConstants::img;
  for (int col = 0; col < psi.cols(); ++col) {
    Eigen::MatrixX3cd grad = Operators::gradLagrange(psi.col(col));

#pragma omp parallel for collapse(3)
    for (int k = 0; k < grid.get_n(); ++k) {
      for (int j = 0; j < grid.get_n(); ++j) {
        for (int i = 0; i < grid.get_n(); ++i) {
          int idx = grid.idx(i, j, k, 0);

          for (int mu = 0; mu < 3; ++mu) {
            for (int nu = 0; nu < 3; nu++) {
              Eigen::Vector2cd chi(2), chiPsi(2);
              chi(0) = grad(idx, mu);
              chi(1) = grad(idx + 1, mu);
              chiPsi(0) = psi(idx, col);
              chiPsi(1) = psi(idx + 1, col);
              auto prod = pauli[nu] * chi;
              if (std::norm(chiPsi.dot(prod)) > 1e20) {
                std::cout << "Anomaly " << std::endl;
                std::cout << "Chi: " << chi << std::endl;
              }

              J(grid.idxNoSpin(i, j, k), mu * 3 + nu) +=
                  chiPsi(0).real() * prod(0).imag() -
                  chiPsi(0).imag() * prod(0).real() +
                  chiPsi(1).real() * prod(1).imag() -
                  chiPsi(1).imag() * prod(1).real();
            }
          }
        }
      }
    }
  }

  return J.real();
}
Eigen::Matrix<double, Eigen::Dynamic, 9> soDensity(const Eigen::MatrixXcd &psi,
                                                   const Grid &grid) {
  using std::complex;

  Eigen::Matrix<complex<double>, Eigen::Dynamic, 9> J(
      grid.get_total_spatial_points(), 9);
  J.setZero();

  auto pauli = nuclearConstants::getPauli(); // Call once outside the
                                             // parallel region

  using nuclearConstants::img;
  for (int col = 0; col < psi.cols(); ++col) {
    Eigen::MatrixX3cd grad = Operators::grad(psi.col(col), grid);

#pragma omp parallel for collapse(3)
    for (int k = 0; k < grid.get_n(); ++k) {
      for (int j = 0; j < grid.get_n(); ++j) {
        for (int i = 0; i < grid.get_n(); ++i) {
          int idx = grid.idx(i, j, k, 0);

          for (int mu = 0; mu < 3; ++mu) {
            for (int nu = 0; nu < 3; nu++) {
              Eigen::Vector2cd chi(2), chiPsi(2);
              chi(0) = grad(idx, mu);
              chi(1) = grad(idx + 1, mu);
              chiPsi(0) = psi(idx, col);
              chiPsi(1) = psi(idx + 1, col);
              auto prod = pauli[nu] * chi;

              J(grid.idxNoSpin(i, j, k), mu * 3 + nu) +=
                  chiPsi(0).real() * prod(0).imag() -
                  chiPsi(0).imag() * prod(0).real() +
                  chiPsi(1).real() * prod(1).imag() -
                  chiPsi(1).imag() * prod(1).real();
            }
          }
        }
      }
    }
  }

  return J.real();
}

Eigen::VectorXd divJ(const Eigen::MatrixXcd &psi, const Grid &grid) {
  int N = grid.get_total_spatial_points();
  Eigen::VectorXd W = Eigen::VectorXd::Zero(N);

  auto pauli = nuclearConstants::getPauli();
  using std::complex;

  for (int col = 0; col < psi.cols(); ++col) {
    Eigen::MatrixXcd grad = Operators::grad(psi.col(col), grid);

    for (int k = 0; k < grid.get_n(); ++k)
      for (int j = 0; j < grid.get_n(); ++j)
        for (int i = 0; i < grid.get_n(); ++i) {
          int idx_up = grid.idx(i, j, k, 0);
          int idx_down = grid.idx(i, j, k, 1);
          int idx_spatial = grid.idxNoSpin(i, j, k);

          complex<double> sum = 0.0;

          for (int mu = 0; mu < 3; ++mu)
            for (int nu = 0; nu < 3; ++nu)
              for (int lambda = 0; lambda < 3; ++lambda) {
                // Levi-Civita symbol ε_{μνλ}
                double eps = 0;
                if ((mu == 0 && nu == 1 && lambda == 2) ||
                    (mu == 1 && nu == 2 && lambda == 0) ||
                    (mu == 2 && nu == 0 && lambda == 1))
                  eps = +1.0;
                else if ((mu == 2 && nu == 1 && lambda == 0) ||
                         (mu == 0 && nu == 2 && lambda == 1) ||
                         (mu == 1 && nu == 0 && lambda == 2))
                  eps = -1.0;

                if (eps == 0.0)
                  continue;

                Eigen::Vector2cd dmu, dnu;
                dmu(0) = grad(idx_up, mu);
                dmu(1) = grad(idx_down, mu);
                dnu(0) = grad(idx_up, nu);
                dnu(1) = grad(idx_down, nu);

                complex<double> term = dnu.adjoint() * pauli[lambda] * dmu;
                sum += eps * term;
              }

          W(idx_spatial) += sum.imag();
        }
  }

  return -W;
}

std::string pretty(double x) {
  return "" + std::to_string(x * 2).substr(0, 1) + "/2";
}
std::string prettyP(double p) { return p > 0 ? "+" : "-"; }
void printShells(std::pair<Eigen::MatrixXcd, Eigen::VectorXd> levels,
                 const Grid &grid) {

  VariadicTable<double, std::string, std::string, std::string, double> vt(
      {"l", "j", "mj", "parity", "energy_mev"});
  for (int i = 0; i < levels.first.cols(); ++i) {
    Shell shell(std::make_shared<Grid>(grid),
                std::make_shared<Eigen::VectorXcd>(levels.first.col(i)),
                levels.second(i));
    vt.addRow(shell.l(), pretty(shell.j()), pretty(shell.mj()),
              prettyP(shell.P()), shell.energy);
  }
  vt.print(std::cout);
}

void printShellsToFile(std::pair<Eigen::MatrixXcd, Eigen::VectorXd> levels,
                       const Grid &grid, std::ofstream &stream) {

  VariadicTable<double, std::string, std::string, std::string, double> vt(
      {"l", "j", "mj", "parity", "energy_mev"});
  for (int i = 0; i < levels.first.cols(); ++i) {
    Shell shell(std::make_shared<Grid>(grid),
                std::make_shared<Eigen::VectorXcd>(levels.first.col(i)),
                levels.second(i));
    vt.addRow(shell.l(), pretty(shell.j()), pretty(shell.mj()),
              prettyP(shell.P()), shell.energy);
  }
  vt.print(stream);
}

// TODO: spostare
Eigen::MatrixXcd timeReverse(const Eigen::MatrixXcd &psi) {
  auto grid = *Grid::getInstance();

  Eigen::MatrixXcd res(grid.get_total_points(), psi.cols());

  using nuclearConstants::img;
  int n = grid.get_n();
  auto pauli = nuclearConstants::getPauli();
#pragma omp parallel for collapse(3)
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      for (int k = 0; k < n; k++) {

        for (int col = 0; col < psi.cols(); col++) {
          int up = grid.idx(i, j, k, 0);
          int down = grid.idx(i, j, k, 1);
          Eigen::Vector2cd chi;
          chi << psi(up, col), psi(down, col);
          auto prod = img * pauli[1] * chi.conjugate();
          res(up, col) = prod(0);
          res(down, col) = prod(1);
        }
      }
    }
  }
  return res;
}
Eigen::MatrixXcd TROrder(Eigen::MatrixXcd &psi) {
  int nStates = psi.cols();
  int rows = psi.rows();

  Eigen::MatrixXcd ord = psi.adjoint() * timeReverse(psi);

  Eigen::MatrixXcd sortedPsi(rows, nStates);

  std::vector<bool> used(nStates, false);
  std::vector<int> ordering(nStates);

  int currentSlot = 0;

  for (int i = 0; i < nStates; ++i) {
    if (used[i])
      continue;

    sortedPsi.col(currentSlot) = psi.col(i);
    used[i] = true;
    ordering[currentSlot] = i;

    int best_j = -1;
    double max_overlap = -1.0;

    for (int j = 0; j < nStates; ++j) {
      if (used[j])
        continue;

      double overlap = std::abs(ord(i, j));
      if (overlap > max_overlap) {
        max_overlap = std::abs(overlap);
        best_j = j;
      }
    }

    if (best_j != -1) {

      sortedPsi.col(currentSlot + 1) = psi.col(best_j);
      used[best_j] = true;
      ordering[currentSlot + 1] = best_j;

    } else {
      std::cerr << "Error in Kramer pairs ordering: Could not find a TR "
                   "partner for state "
                << i << std::endl;
      return psi;
    }

    currentSlot += 2;
    if (currentSlot >= nStates)
      break;
  }

  return sortedPsi;
}
} // namespace Wavefunction
