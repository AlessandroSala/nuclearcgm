#include "VariadicTable.h"
#include "constants.hpp" // Assuming this is where nuclearConstants are
#include "input_parser.hpp"
#include "operators/differential_operators.hpp" // Assuming this is where Operators::derivative is
#include "types.hpp"
#include "util/shell.hpp"
#include <cmath>
#include <complex>
#include <fstream>
#include <iostream>
#include <memory>
#include <omp.h> // Required for OpenMP

// Forward declaration of Grid class if not fully included,
// or include the necessary header for Grid.
// class Grid;

// Assuming Wavefunction is a class or namespace
namespace Wavefunction {

Eigen::VectorXd density(const Eigen::MatrixXcd &psi, const Grid &grid) {
  int n = grid.get_n();
  Eigen::VectorXd rho(grid.get_total_spatial_points());
  rho.setZero();

#pragma omp parallel for collapse(3)
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      for (int k = 0; k < n; ++k) {
        int rho_idx = grid.idxNoSpin(i, j, k);
        int psi_idx_s0 = grid.idx(i, j, k, 0);
        int psi_idx_s1 = grid.idx(i, j, k, 1);

        double point_density = 0.0;
        for (int col = 0; col < psi.cols(); ++col) {
          point_density += std::norm(
              psi(psi_idx_s0, col)); // std::norm(complex) = |complex|^2
          point_density += std::norm(psi(psi_idx_s1, col));
        }
        rho(rho_idx) =
            point_density; // Each thread writes to a unique element of rho
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

        for (int i = 0; i < n; ++i) {
          for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
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

/**
 * @brief Calculates the Mean field
 * @param rho Isoscalar density.
 * @param rhoQ nucleon density.
 * @param tau Isoscalar kinetic density.
 * @param tauQ nucleon kinetic density.
 * @param nabla2rho Laplacian of the isoscalar density.
 * @param nabla2rhoQ Laplacian of the nucleon density.
 * @param nablaJJQ Divergence of the spin-orbit density current.
 * @param grid The spatial grid.
 * @param params Parameters of the Skyrme interaction.
 * @return Eigen::VectorXd The calculated field.
 */
Eigen::VectorXd field(Eigen::VectorXd &rho, Eigen::VectorXd &rhoQ,
                      Eigen::VectorXd &tau, Eigen::VectorXd &tauQ,
                      Eigen::VectorXd &nabla2rho, Eigen::VectorXd &nabla2rhoQ,
                      Eigen::VectorXd &divJJQ, const Grid &grid,
                      SkyrmeParameters params) {
  double t0 = params.t0, t1 = params.t1, t2 = params.t2, t3 = params.t3;
  double x0 = params.x0, x1 = params.x1, x2 = params.x2, x3 = params.x3;
  double W0 = params.W0;
  double sigma = params.sigma; // Corresponds to 'alpha' in the formula

  Eigen::VectorXd field(grid.get_total_spatial_points());
  field.setZero();

  // t0 * [(1 + x0/2) * rho - (x0 + 1/2) * rhoQ]
  field += t0 * ((1.0 + 0.5 * x0) * rho - (x0 + 0.5) * rhoQ);

  // Chabant
  field += 0.125 * (t1 * (2 + x1) + t2 * (2 + x2)) * tau;

  field += 0.125 * (t2 * (2 * x2 + 1) - t1 * (2 * x1 + 1)) * tauQ;

  // Chabant
  // + 1/8 * (t2 - 3*t1 - (3*t1*x1 + t2*x2)/2) * nabla^2 rho
  field += (1.0 / 16.0) * (t2 * (2 + x2) - 3 * t1 * (2 + x1)) * nabla2rho;

  // + 1/16 * (t2 + 3*t1 + 6*t1*x1 + 2*t2*x2) * nabla^2 rhoQ
  // //Uguale C e S?
  field +=
      (1.0 / 16.0) * (3 * t1 * (2 * x1 + 1) + t2 * (2 * x2 + 1)) * nabla2rhoQ;

  field += -0.5 * W0 * divJJQ;

  field +=
      ((t3 / 12.0) * pow(rho.array(), sigma - 1.0) *
       ((1.0 + 0.5 * x3) * (sigma + 2.0) * pow(rho.array(), 2.0) -
        (x3 + 0.5) *
            (sigma * (rhoQ.array().square() + (rho - rhoQ).array().square()) +
             2.0 * rho.array() * rhoQ.array())))
          .matrix();

  return field;
}
// Eigen::VectorXd kineticDensity(const Eigen::MatrixXcd &psi, const Grid &grid)
//{
//// using std::complex;
// Eigen::VectorXd tau(grid.get_total_spatial_points());
// tau.setZero();

// #pragma omp parallel for collapse(3)
// for (int i = 0; i < grid.get_n(); ++i)
//{
// for (int j = 0; j < grid.get_n(); ++j)
//{
// for (int k = 0; k < grid.get_n(); ++k)
//{
// int tau_idx = grid.idxNoSpin(i, j, k);
// double point_tau = 0.0;
// for (int s = 0; s < 2; ++s) // Loop over spin components
//{
// for (int col = 0; col < psi.cols(); ++col) // Loop over orbitals/states
//{
//// psi.col(col) passes a column vector (or an expression representing it)
//// to the derivative function. Operators::derivative is assumed to be
/// thread-safe.
// point_tau += std::norm(Operators::derivative(psi.col(col), i, j, k, s, grid,
// 'x')); point_tau += std::norm(Operators::derivative(psi.col(col), i, j, k, s,
// grid, 'y')); point_tau += std::norm(Operators::derivative(psi.col(col), i, j,
// k, s, grid, 'z'));
//}
//}
// tau(tau_idx) = point_tau; // Each thread writes to a unique element of tau
//}
//}
//}
// return tau;
//}
Eigen::Matrix3Xd spinDensity(const Eigen::MatrixXcd &psi, const Grid &grid) {
  Eigen::Matrix3Xd res =
      Eigen::Matrix3Xd::Zero(3, grid.get_total_spatial_points());
  auto pauli = nuclearConstants::getPauli();
  for (int col = 0; col < psi.cols(); ++col) {
    for (int i = 0; i < grid.get_n(); ++i) {
      for (int j = 0; j < grid.get_n(); ++j) {
        for (int k = 0; k < grid.get_n(); ++k) {
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
    psi.col(c) /= sqrt(density.sum() * pow(grid.get_h(), 3));
  }
}

Eigen::Matrix<double, Eigen::Dynamic, 9> soDensity(const Eigen::MatrixXcd &psi,
                                                   const Grid &grid) {
  using std::complex;

  Eigen::Matrix<complex<double>, Eigen::Dynamic, 9> J(
      grid.get_total_spatial_points(), 9);
  J.setZero();

  auto pauli =
      nuclearConstants::getPauli(); // Call once outside the parallel region

  using nuclearConstants::img;
  for (int col = 0; col < psi.cols(); ++col) {
    Eigen::MatrixX3cd grad = Operators::grad(psi.col(col), grid);

#pragma omp parallel for collapse(3)
    for (int i = 0; i < grid.get_n(); ++i) {
      for (int j = 0; j < grid.get_n(); ++j) {
        for (int k = 0; k < grid.get_n(); ++k) {
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
                // std::cout << "Chi psi: " << chiPsi << std::endl;
                //  std::cout << "Prod: " << prod << std::endl;
              }

              J(grid.idxNoSpin(i, j, k), mu * 3 + nu) +=
                  chiPsi(0).real() * prod(0).imag() -
                  chiPsi(0).imag() * prod(0).real() +
                  chiPsi(1).real() * prod(1).imag() -
                  chiPsi(1).imag() * prod(1).real();

              // auto prod = (pauli[nu] * chi).dot(chiPsi);
              //  if (std::norm(prod) > 1e-1) {
              //    prod = {0.0, 0.0};
              //  }

              // J(grid.idxNoSpin(i, j, k), mu * 3 + nu) += prod;
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

    for (int i = 0; i < grid.get_n(); ++i)
      for (int j = 0; j < grid.get_n(); ++j)
        for (int k = 0; k < grid.get_n(); ++k) {
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

std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXcd>
hfVectors(const Eigen::MatrixXcd &psi, const Grid &grid) {
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
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      for (int k = 0; k < n; ++k) {
        int rho_idx = grid.idxNoSpin(i, j, k);
        int psi_idx_s0 = grid.idx(i, j, k, 0);
        int psi_idx_s1 = grid.idx(i, j, k, 1);

        double point_density = 0.0;
        complex<double> Jx_point(0.0, 0.0);
        complex<double> Jy_point(0.0, 0.0);
        complex<double> Jz_point(0.0, 0.0);
        for (int col = 0; col < psi.cols(); ++col) {
          Eigen::Matrix<complex<double>, 2, 3>
              chiGrad; // Stores nabla_x chi, nabla_y chi, nabla_z chi for
                       // spin up/down
          Eigen::Vector2cd chi; // Spinor for current point and orbital
          point_density += std::norm(
              psi(psi_idx_s0, col)); // std::norm(complex) = |complex|^2
          point_density += std::norm(psi(psi_idx_s1, col));
          for (int s = 0; s < 2; ++s) {
            std::complex<double> dx =
                Operators::derivative(psi.col(col), i, j, k, s, grid, 'x');
            std::complex<double> dy =
                Operators::derivative(psi.col(col), i, j, k, s, grid, 'y');
            std::complex<double> dz =
                Operators::derivative(psi.col(col), i, j, k, s, grid, 'z');
            tau(rho_idx) += std::norm(dx) + std::norm(dy) + std::norm(dz);
            chiGrad(s, 0) = Operators::derivative(psi.col(col), i, j, k, s,
                                                  grid, 'x'); // d/dx psi_s
            chiGrad(s, 1) = Operators::derivative(psi.col(col), i, j, k, s,
                                                  grid, 'y'); // d/dy psi_s
            chiGrad(s, 2) = Operators::derivative(psi.col(col), i, j, k, s,
                                                  grid, 'z'); // d/dz psi_s

            chi(s) = psi(grid.idx(i, j, k, s), col);
          }
          Jx_point += chi.dot(pauli_matrices[2] * chiGrad.col(1) -
                              pauli_matrices[1] * chiGrad.col(2));
          Jy_point += chi.dot(pauli_matrices[0] * chiGrad.col(2) -
                              pauli_matrices[2] * chiGrad.col(0));
          Jz_point += chi.dot(pauli_matrices[1] * chiGrad.col(0) -
                              pauli_matrices[0] * chiGrad.col(1));
        }
        J(rho_idx, 0) = Jx_point;
        J(rho_idx, 1) = Jy_point;
        J(rho_idx, 2) = Jz_point;
        rho(rho_idx) =
            point_density; // Each thread writes to a unique element of rho
      }
    }
  }

  return std::tuple<Eigen::VectorXd, Eigen::VectorXd, Eigen::MatrixXcd>(
      rho, tau, (-nuclearConstants::img) * J);
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
} // namespace Wavefunction
