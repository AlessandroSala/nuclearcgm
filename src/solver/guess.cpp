#include "guess.hpp"
#include "constants.hpp"
#include "spherical_harmonics.hpp"
#include <cmath>
#include <iomanip>
#include <iostream>
#include <tuple>
#include <vector>

ComplexDenseMatrix harmonic_oscillator_guess(const Grid &grid, int nev,
                                             double omega, // omega unit of hbar
                                             double beta, bool useSpinOrbit) {
  int grid_n = grid.get_n();
  int total_points = grid.get_total_points();

  ComplexDenseMatrix guess(total_points, nev);
  guess.setZero();

  // Hermite polynomials
  auto hermite = [](int order, double x) {
    if (order == 0)
      return 1.0;
    if (order == 1)
      return 2.0 * x;
    double Hnm2 = 1.0;
    double Hnm1 = 2.0 * x;
    double Hn = 0.0;
    for (int k = 2; k <= order; ++k) {
      Hn = 2.0 * x * Hnm1 - 2.0 * (k - 1) * Hnm2;
      Hnm2 = Hnm1;
      Hnm1 = Hn;
    }
    return Hn;
  };

  auto ho_1d = [omega, &hermite](int order, double x) {
    double xi = x / omega;
    double norm = 1.0 / (std::sqrt((1u << order) * std::tgamma(order + 1) *
                                   std::sqrt(M_PI) * omega));
    return norm * hermite(order, xi) * std::exp(-0.5 * xi * xi);
  };

  // Generalized Laguerre (recurrence)
  auto laguerre = [](int n, double alpha, double x) {
    if (n == 0)
      return 1.0;
    if (n == 1)
      return 1.0 + alpha - x;
    double Lnm2 = 1.0;
    double Lnm1 = 1.0 + alpha - x;
    double Ln = 0.0;
    for (int k = 2; k <= n; ++k) {
      Ln = ((2.0 * k - 1.0 + alpha - x) * Lnm1 - (k - 1.0 + alpha) * Lnm2) / k;
      Lnm2 = Lnm1;
      Lnm1 = Ln;
    }
    return Ln;
  };

  auto ho_radial = [omega, &laguerre](int nr, int l, double r) {
    double rho = r / omega;
    double norm =
        std::sqrt(2.0 * std::tgamma(nr + 1.0) /
                  (omega * omega * omega * std::tgamma(nr + l + 1.5)));
    return norm * std::pow(rho, l) * laguerre(nr, l + 0.5, rho * rho) *
           std::exp(-0.5 * rho * rho);
  };

  int state_index = 0;

  if (!useSpinOrbit) {

    for (int nx = 0; nx < grid_n && state_index < nev; ++nx) {
      for (int ny = 0; ny < grid_n && state_index < nev; ++ny) {
        for (int nz = 0; nz < grid_n && state_index < nev; ++nz) {

          if (state_index < nev) {
            for (int i = 0; i < grid_n; ++i) {
              double x = grid.get_xs()[i];
              double psi_x = ho_1d(nx, x);
              for (int j = 0; j < grid_n; ++j) {
                double y = grid.get_ys()[j];
                double psi_y = ho_1d(ny, y);
                for (int k = 0; k < grid_n; ++k) {
                  double z = grid.get_zs()[k];
                  double psi_z = ho_1d(nz, z);
                  double val = psi_x * psi_y * psi_z;
                  guess(grid.idx(i, j, k, 0), state_index) =
                      ComplexScalar(val, 0.0);
                }
              }
            }
            guess.col(state_index).normalize();
            ++state_index;
          }

          if (state_index < nev) {
            for (int i = 0; i < grid_n; ++i) {
              double x = grid.get_xs()[i];
              double psi_x = ho_1d(nx, x);
              for (int j = 0; j < grid_n; ++j) {
                double y = grid.get_ys()[j];
                double psi_y = ho_1d(ny, y);
                for (int k = 0; k < grid_n; ++k) {
                  double z = grid.get_zs()[k];
                  double psi_z = ho_1d(nz, z);
                  double val = psi_x * psi_y * psi_z;
                  guess(grid.idx(i, j, k, 1), state_index) =
                      ComplexScalar(val, 0.0);
                }
              }
            }
            guess.col(state_index).normalize();
            ++state_index;
          }
        }
      }
    }

  } else {
    std::vector<std::tuple<int, int, int, int>> orbitals;

    for (int N = 0; static_cast<int>(orbitals.size()) < nev; ++N) {
      for (int nr = 0; nr <= N / 2 && static_cast<int>(orbitals.size()) < nev;
           ++nr) {
        int l = N - 2 * nr;

        // j = l + 1/2  -> j2 = 2*l + 1
        int j2_plus = 2 * l + 1;
        // mj2 runs over odd integers: ±1, ±3, ..., ±j2_plus
        for (int mj2_abs = 1;
             mj2_abs <= j2_plus && static_cast<int>(orbitals.size()) < nev;
             mj2_abs += 2) {
          orbitals.emplace_back(nr, l, j2_plus, mj2_abs); // +|mj|
          if (static_cast<int>(orbitals.size()) < nev)
            orbitals.emplace_back(nr, l, j2_plus, -mj2_abs); // -|mj|
        }

        // j = l - 1/2 (exists when l>0)
        if (l > 0) {
          int j2_minus = 2 * l - 1;
          for (int mj2_abs = 1;
               mj2_abs <= j2_minus && static_cast<int>(orbitals.size()) < nev;
               mj2_abs += 2) {
            orbitals.emplace_back(nr, l, j2_minus, mj2_abs);
            if (static_cast<int>(orbitals.size()) < nev)
              orbitals.emplace_back(nr, l, j2_minus, -mj2_abs);
          }
        }
      }
    }

    // std::cout << "=== HO states ===" << std::endl;
    // std::cout << std::setw(5) << "STATE" << "   " << std::setw(2) << "nr"
    //           << "   " << std::setw(2) << "l" << "   " << std::setw(3) << "j"
    //           << "   " << std::setw(4) << "mj" << "   " << std::setw(5)
    //           << "ml_up" << "   " << std::setw(7) << "ml_down" << "   "
    //           << std::setw(7) << "have_up" << "   " << std::setw(9)
    //           << "have_down" << "\n";
    for (size_t idx = 0; idx < orbitals.size(); ++idx) {
      int nr, l, j2, mj2;
      std::tie(nr, l, j2, mj2) = orbitals[idx];
      double j = j2 / 2.0;
      double mj = mj2 / 2.0;

      int ml_up2 = mj2 - 1;
      int ml_down2 = mj2 + 1;
      int ml_up = ml_up2 / 2;
      int ml_down = ml_down2 / 2;
      bool have_up = (std::abs(ml_up) <= l);
      bool have_down = (std::abs(ml_down) <= l);

      // std::cout << std::setw(5) << idx << "   " << std::setw(2) << nr << " "
      //           << std::setw(2) << l << "   " << std::setw(3) << j << "   "
      //           << std::setw(4) << mj << "   " << std::setw(5) << ml_up << "
      //           "
      //           << std::setw(7) << ml_down << "   " << std::setw(7) <<
      //           have_up
      //           << "   " << std::setw(9) << have_down << "\n";
    }
    for (int state_index = 0;
         state_index < static_cast<int>(orbitals.size()) && state_index < nev;
         ++state_index) {
      int nr, l, j2, mj2;
      std::tie(nr, l, j2, mj2) = orbitals[state_index];

      // mj = mj2 / 2.0 (half int)
      double mj = mj2 * 0.5;
      double denom = 2.0 * l + 1.0;

      // CG coefficients
      double C_up = 0.0, C_down = 0.0;
      bool j_is_plus = (j2 == 2 * l + 1);
      if (j_is_plus) {
        C_up = std::sqrt((l + mj + 0.5) / denom);
        C_down = std::sqrt((l - mj + 0.5) / denom);
      } else { // j = l - 1/2
        C_up = std::sqrt((l - mj + 0.5) / denom);
        C_down = -std::sqrt((l + mj - 0.5) / denom);
      }

      int ml_up2 = mj2 - 1;
      int ml_down2 = mj2 + 1;
      int ml_up = ml_up2 / 2;
      int ml_down = ml_down2 / 2;

      bool have_up = (std::abs(ml_up) <= l);
      bool have_down = (std::abs(ml_down) <= l);

      for (int i = 0; i < grid_n; ++i) {
        double x = grid.get_xs()[i];
        for (int jg = 0; jg < grid_n; ++jg) {
          double y = grid.get_ys()[jg];
          for (int k = 0; k < grid_n; ++k) {
            double z = grid.get_zs()[k];

            double r = std::sqrt(x * x + y * y + z * z);
            double theta = (r > 1e-12) ? std::acos(z / r) : 0.0;
            double phi = std::atan2(y, x);

            double Rnl = ho_radial(nr, l, r);

            ComplexScalar spinor_up(0.0, 0.0);
            ComplexScalar spinor_down(0.0, 0.0);

            if (have_up) {
              ComplexScalar Ylm_up =
                  SphericalHarmonics::Y(l, ml_up, theta, phi);
              spinor_up = (C_up * Rnl) * Ylm_up;
            }
            if (have_down) {
              ComplexScalar Ylm_down =
                  SphericalHarmonics::Y(l, ml_down, theta, phi);
              spinor_down = (C_down * Rnl) * Ylm_down;
            }

            guess(grid.idx(i, jg, k, 0), state_index) = spinor_up;
            guess(grid.idx(i, jg, k, 1), state_index) = spinor_down;
          }
        }
      }

      guess.col(state_index).normalize();
    }
  }

  return guess;
}
