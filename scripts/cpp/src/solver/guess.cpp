#include "guess.hpp"
#include "constants.hpp"
#include "spherical_harmonics.hpp"
#include <iostream>

std::pair<int, int> getNumbers(int stateIndex) {
    int part = stateIndex +1;
    if(part <= 2) return {0, 0};
    

}
ComplexDenseMatrix gaussian_guess(const Grid &grid, int nev, double a) {
  int n = grid.get_n();

  ComplexDenseMatrix guess(grid.get_total_points(), nev);
  for (int ev = 0; ev < nev; ++ev) {
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        for (int k = 0; k < n; ++k) {
          double r = grid.get_xs()[i] * grid.get_xs()[i] +
                     grid.get_ys()[j] * grid.get_ys()[j] +
                     grid.get_zs()[k] * grid.get_zs()[k];
          r = (sqrt(r)) / (5 * a);
          for (int s = 0; s < 2; ++s) {
            guess(grid.idx(i, j, k, s), ev) =
                ComplexScalar(exp(-pow(r, ev)), 0);
          }
        }
      }
    }
  }
  for (int ev = 0; ev < nev; ++ev) {
    guess.col(ev).normalize();
  }
  return guess;
}

ComplexDenseMatrix harmonic_oscillator_guess(const Grid &grid, int nev,
                                             double a, bool useSpinOrbit) {
  int n = grid.get_n();
  int total_points = grid.get_total_points();

  ComplexDenseMatrix guess(total_points, nev);

  // ---------------------------
  // Hermite polynomials (recurrence)
  // ---------------------------
  auto hermite = [](int n, double x) {
    if (n == 0)
      return 1.0;
    if (n == 1)
      return 2.0 * x;
    double Hnm2 = 1.0;
    double Hnm1 = 2.0 * x;
    double Hn = 0.0;
    for (int k = 2; k <= n; ++k) {
      Hn = 2.0 * x * Hnm1 - 2.0 * (k - 1) * Hnm2;
      Hnm2 = Hnm1;
      Hnm1 = Hn;
    }
    return Hn;
  };

  // ---------------------------
  // Normalized 1D HO wavefunction
  // ---------------------------
  auto ho_1d = [a, &hermite](int n, double x) {
    double xi = x / a;
    double norm =
        1.0 / (std::sqrt((1 << n) * std::tgamma(n + 1) * std::sqrt(M_PI) * a));
    return norm * hermite(n, xi) * std::exp(-0.5 * xi * xi);
  };

  // ---------------------------
  // Radial HO wavefunction (spherical)
  // R_{nl}(r) ∝ r^l L_n^{l+1/2}(r^2/a^2) e^{-r^2/2a^2}
  // ---------------------------
  auto laguerre = [](int n, int alpha, double x) {
    // Generalized Laguerre polynomial L_n^alpha(x)
    if (n == 0)
      return 1.0;
    if (n == 1)
      return 1.0 + alpha - x;
    double Lnm2 = 1.0;
    double Lnm1 = 1.0 + alpha - x;
    double Ln = 0.0;
    for (int k = 2; k <= n; ++k) {
      Ln = ((2 * k - 1 + alpha - x) * Lnm1 - (k - 1 + alpha) * Lnm2) / k;
      Lnm2 = Lnm1;
      Lnm1 = Ln;
    }
    return Ln;
  };

  auto ho_radial = [a, &laguerre](int n, int l, double r) {
    double rho = r / a;
    double norm = std::sqrt(2.0 * std::tgamma(n + 1) /
                            (a * a * a * std::tgamma(n + l + 1.5)));
    return norm * std::pow(rho, l) * laguerre(n, l + 1 / 2.0, rho * rho) *
           std::exp(-0.5 * rho * rho);
  };

  // ---------------------------
  // Spherical harmonics (real form for simplicity)
  // ---------------------------
  auto Ylm = [](int l, int m, double theta, double phi) {
    // Normalization
    double norm = std::sqrt((2.0 * l + 1) / (4 * M_PI) *
                            std::tgamma(l - std::abs(m) + 1) /
                            std::tgamma(l + std::abs(m) + 1));
    // Associated Legendre (simple recursion, here use std::assoc_legendre if
    // C++17+)
    double Plm =
        SphericalHarmonics::associatedLegendre(l, std::abs(m), std::cos(theta));
    double value = norm * Plm;
    if (m > 0)
      value *= std::cos(m * phi) * std::sqrt(2.0);
    else if (m < 0)
      value *= std::sin(-m * phi) * std::sqrt(2.0);
    return value;
  };

  int state_index = 0;

  if (!useSpinOrbit) {
    // ---------------------------
    // Cartesian basis (old version)
    // ---------------------------
    for (int nx = 0; nx < n && state_index < nev; ++nx) {
      for (int ny = 0; ny < n && state_index < nev; ++ny) {
        for (int nz = 0; nz < n && state_index < nev; ++nz) {
          for (int s = 0; s < 2 && state_index < nev; ++s) {
            for (int i = 0; i < n; ++i) {
              for (int j = 0; j < n; ++j) {
                for (int k = 0; k < n; ++k) {
                  double x = grid.get_xs()[i];
                  double y = grid.get_ys()[j];
                  double z = grid.get_zs()[k];

                  double psi_x = ho_1d(nx, x);
                  double psi_y = ho_1d(ny, y);
                  double psi_z = ho_1d(nz, z);

                  guess(grid.idx(i, j, k, s), state_index) =
                      ComplexScalar(psi_x * psi_y * psi_z, 0.0);
                }
              }
            }
            guess.col(state_index).normalize();
            state_index++;
          }
        }
      }
    }
  } else {
    // ---------------------------
    // Spherical basis (intermediate level)
    // ---------------------------
    for (int n_rad = 0; n_rad < n && state_index < nev; ++n_rad) {
      for (int l = 0; l < n && state_index < nev; ++l) {
        for (int m = -l; m <= l && state_index < nev; ) {
          for (int s = 0; s < 2 && state_index < nev; ++s) {
            for (int i = 0; i < n; ++i) {
              for (int j = 0; j < n; ++j) {
                for (int k = 0; k < n; ++k) {
                  double x = grid.get_xs()[i];
                  double y = grid.get_ys()[j];
                  double z = grid.get_zs()[k];

                  double r = std::sqrt(x * x + y * y + z * z);
                  double theta = std::acos(z / (r + 1e-12));
                  double phi = std::atan2(y, x);

                  double Rnl = ho_radial(n_rad, l, r);
                  double Y = Ylm(l, m, theta, phi);

                  guess(grid.idx(i, j, k, s), state_index) =
                      ComplexScalar(Rnl * Y, 0.0);
                }
              }
            }
            guess.col(state_index).normalize();
            state_index++;
          }
                  if(m == 0)
                      break;
                  if(m > 0)
                      m = -(m - 1);
                  else
                      m = -m;
        }
      }
    }
  }

  std::cout << "Generated " << state_index << " states (requested " << nev
            << ")" << std::endl;
  return guess;
}
// ComplexDenseMatrix harmonic_oscillator_guess(const Grid &grid, int nev,
//                                              double a) {
//   int n = grid.get_n();
//   int total_points = grid.get_total_points();
//
//   ComplexDenseMatrix guess(total_points, nev);
//
//   // Hermite polynomials (physicist's definition) using recurrence
//   auto hermite = [](int n, double x) {
//     if (n == 0)
//       return 1.0;
//     if (n == 1)
//       return 2.0 * x;
//     double Hnm2 = 1.0;
//     double Hnm1 = 2.0 * x;
//     double Hn = 0.0;
//     for (int k = 2; k <= n; ++k) {
//       Hn = 2.0 * x * Hnm1 - 2.0 * (k - 1) * Hnm2;
//       Hnm2 = Hnm1;
//       Hnm1 = Hn;
//     }
//     return Hn;
//   };
//
//   // Normalized 1D HO wavefunction
//   auto ho_1d = [a, &hermite](int n, double x) {
//     double xi = x / a;
//     double norm =
//         1.0 / (std::sqrt((1 << n) * std::tgamma(n + 1) * std::sqrt(M_PI) *
//         a));
//     return norm * hermite(n, xi) * std::exp(-0.5 * xi * xi);
//   };
//
//   int state_index = 0;
//   for (int nx = 0; nx < n && state_index < nev; ++nx) {
//     for (int ny = 0; ny < n && state_index < nev; ++ny) {
//       for (int nz = 0; nz < n && state_index < nev; ++nz) {
//         for (int s = 0; s < 2 && state_index < nev; ++s) {
//           // costruiamo funzione d’onda tridimensionale
//           for (int i = 0; i < n; ++i) {
//             for (int j = 0; j < n; ++j) {
//               for (int k = 0; k < n; ++k) {
//                 double x = grid.get_xs()[i];
//                 double y = grid.get_ys()[j];
//                 double z = grid.get_zs()[k];
//
//                 double psi_x = ho_1d(nx, x);
//                 double psi_y = ho_1d(ny, y);
//                 double psi_z = ho_1d(nz, z);
//
//                 guess(grid.idx(i, j, k, s), state_index) =
//                     ComplexScalar(psi_x * psi_y * psi_z, 0.0);
//               }
//             }
//           }
//           guess.col(state_index).normalize();
//           state_index++;
//         }
//       }
//     }
//   }
//
//   std::cout << "Generated " << state_index << " states (requested " << nev
//             << ")" << std::endl;
//   return guess;
// }
//// ComplexDenseMatrix harmonic_oscillator_guess(const Grid &grid, int nev,
////                                              double omega) {
////
////   int n = grid.get_n();
////   int total_points = grid.get_total_points();
////
////   ComplexDenseMatrix guess(total_points, nev);
////   using namespace nuclearConstants;
////
////   // double a = sqrt(h_bar / omega / m);
////   double a = omega;
////
////   auto hermite = [](int n, double x) {
////     switch (n) {
////     case 0:
////       return 1.0;
////     case 1:
////       return 2.0 * x;
////     case 2:
////       return 4.0 * x * x - 2.0;
////     case 3:
////       return 8.0 * x * x * x - 12.0 * x;
////     case 4:
////       return 16.0 * x * x * x * x - 48.0 * x * x + 12.0;
////     case 5:
////       return 32.0 * x * x * x * x * x - 160.0 * x * x * x + 120.0 * x;
////     case 6:
////       return 64.0 * x * x * x * x * x * x - 480.0 * x * x * x * x +
////              720 * x * x - 120;
////     default:
////       return 0.0; // Implementazione più generale per n maggiore se
////       necessario
////     }
////   };
////
////   auto gaussian = [a](double x) {
////     return exp(-(x * x) / (2.0 * a * a)) / std::pow(M_PI * a * a, 0.25);
////   };
////
////   int state_index = 0;
////   for (int nx = 0; nx < n && state_index < nev; ++nx) {
////     for (int ny = 0; ny < n && state_index < nev; ++ny) {
////       for (int nz = 0; nz < n && state_index < nev; ++nz) {
////         for (int s = 0; s < 2; ++s) {
////           if (state_index < nev) {
////             for (int i = 0; i < n; ++i) {
////               for (int j = 0; j < n; ++j) {
////                 for (int k = 0; k < n; ++k) {
////                   double x = grid.get_xs()[i];
////                   double y = grid.get_ys()[j];
////                   double z = grid.get_zs()[k];
////
////                   double psi_x = hermite(nx, x / a) * gaussian(x);
////                   double psi_y = hermite(ny, y / a) * gaussian(y);
////                   double psi_z = hermite(nz, z / a) * gaussian(z);
////
////                   guess(grid.idx(i, j, k, s), state_index) =
////                       ComplexScalar(psi_x * psi_y * psi_z, 0.0);
////                 }
////               }
////             }
////             guess.col(state_index).normalize();
////             state_index++;
////           } else {
////             break;
////           }
////         }
////         if (state_index >= nev)
////           break;
////       }
////       if (state_index >= nev)
////         break;
////     }
////     if (state_index >= nev)
////       break;
////   }
////   std::cout << "state index: " << state_index << std::endl;
////
////   return guess;
//// }
// ComplexDenseMatrix anisotropic_gaussian_guess(const Grid &grid, int nev,
//                                               double a_x, double a_y,
//                                               double a_z) {
//   int n = grid.get_n();
//   ComplexDenseMatrix guess(grid.get_total_points(), nev);
//   for (int ev = 0; ev < nev; ++ev) {
//     for (int i = 0; i < n; ++i) {
//       for (int j = 0; j < n; ++j) {
//         for (int k = 0; k < n; ++k) {
//           // double r = grid.get_xs()[i]*grid.get_xs()[i] +
//           // grid.get_ys()[j]*grid.get_ys()[j] +
//           // grid.get_zs()[k]*grid.get_zs()[k]; r = (sqrt(r))/(5*a_x);
//           double r = grid.get_xs()[i] * grid.get_xs()[i] / (5 * a_x) +
//                      grid.get_ys()[j] * grid.get_ys()[j] / (5 * a_y) +
//                      grid.get_zs()[k] * grid.get_zs()[k] / (5 * a_z);
//           for (int s = 0; s < 2; ++s) {
//             guess(grid.idx(i, j, k, s), ev) =
//                 ComplexScalar(exp(-pow(r, ev)), 0);
//           }
//         }
//       }
//     }
//   }
//   // for (int ev = 0; ev < nev; ++ev) {
//   // guess.col(ev).normalize();
//   //}
//   return guess;
// }
