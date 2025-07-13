#include "guess.hpp"
#include "constants.hpp"
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
                                             double omega) {

  int n = grid.get_n();
  int total_points = grid.get_total_points();

  ComplexDenseMatrix guess(total_points, nev);
  using nuclearConstants::h_bar;

  // double a = sqrt(h_bar / omega);
  double a = omega;

  auto hermite = [](int n, double x) {
    switch (n) {
    case 0:
      return 1.0;
    case 1:
      return 2.0 * x;
    case 2:
      return 4.0 * x * x - 2.0;
    case 3:
      return 8.0 * x * x * x - 12.0 * x;
    case 4:
      return 16.0 * x * x * x * x - 48.0 * x * x + 12.0;
    case 5:
      return 32.0 * x * x * x * x * x - 160.0 * x * x * x + 120.0 * x;
    default:
      return 0.0; // Implementazione più generale per n maggiore se necessario
    }
  };

  auto gaussian = [a](double x) {
    return exp(-(x * x) / (2.0 * a * a)) / std::pow(M_PI * a * a, 0.25);
  };

  int state_index = 0;
  for (int nx = 0; nx < n && state_index < nev; ++nx) {
    for (int ny = 0; ny < n && state_index < nev; ++ny) {
      for (int nz = 0; nz < n && state_index < nev; ++nz) {
        for (int s = 0; s < 2; ++s) {
          if (state_index < nev) {
            for (int i = 0; i < n; ++i) {
              for (int j = 0; j < n; ++j) {
                for (int k = 0; k < n; ++k) {
                  double x = grid.get_xs()[i];
                  double y = grid.get_ys()[j];
                  double z = grid.get_zs()[k];

                  double psi_x = hermite(nx, x / a) * gaussian(x);
                  double psi_y = hermite(ny, y / a) * gaussian(y);
                  double psi_z = hermite(nz, z / a) * gaussian(z);

                  guess(grid.idx(i, j, k, s), state_index) =
                      ComplexScalar(psi_x * psi_y * psi_z, 0.0);
                }
              }
            }
            guess.col(state_index).normalize();
            state_index++;
          } else {
            break;
          }
        }
        if (state_index >= nev)
          break;
      }
      if (state_index >= nev)
        break;
    }
    if (state_index >= nev)
      break;
  }

  return guess;
}
ComplexDenseMatrix anisotropic_gaussian_guess(const Grid &grid, int nev,
                                              double a_x, double a_y,
                                              double a_z) {
  int n = grid.get_n();
  ComplexDenseMatrix guess(grid.get_total_points(), nev);
  for (int ev = 0; ev < nev; ++ev) {
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        for (int k = 0; k < n; ++k) {
          // double r = grid.get_xs()[i]*grid.get_xs()[i] +
          // grid.get_ys()[j]*grid.get_ys()[j] +
          // grid.get_zs()[k]*grid.get_zs()[k]; r = (sqrt(r))/(5*a_x);
          double r = grid.get_xs()[i] * grid.get_xs()[i] / (5 * a_x) +
                     grid.get_ys()[j] * grid.get_ys()[j] / (5 * a_y) +
                     grid.get_zs()[k] * grid.get_zs()[k] / (5 * a_z);
          for (int s = 0; s < 2; ++s) {
            guess(grid.idx(i, j, k, s), ev) =
                ComplexScalar(exp(-pow(r, ev)), 0);
          }
        }
      }
    }
  }
  // for (int ev = 0; ev < nev; ++ev) {
  // guess.col(ev).normalize();
  //}
  return guess;
}
