#include "operators/integral_operators.hpp"
#include <omp.h> // Include OpenMP header

std::complex<double> Operators::integral(const Eigen::VectorXcd &psi,
                                         const Grid &grid) {
  double h = grid.get_h();
  double hhh = h * h * h;

  std::complex<double> res(0.0, 0.0);
  for (int k = 0; k < grid.get_n(); ++k) {
    for (int j = 0; j < grid.get_n(); ++j) {
      for (int i = 0; i < grid.get_n(); ++i) {
        for (int s = 0; s < 2; ++s) {
          int idx = grid.idx(i, j, k, s);
          double w = 1.0;
          if (i == 0 || i == grid.get_n() - 1) {
            w *= 0.5;
          }
          if (j == 0 || j == grid.get_n() - 1) {
            w *= 0.5;
          }
          if (k == 0 || k == grid.get_n() - 1) {
            w *= 0.5;
          }
          res += psi(idx) * hhh * w;
        }
      }
    }
  }

  return res;
}

std::complex<double> Operators::integralNoSpin(const Eigen::VectorXcd &psi,
                                               const Grid &grid) {
  double h = grid.get_h();
  double hhh = h * h * h;
  std::complex<double> res = 0.0;
  for (int k = 0; k < grid.get_n(); ++k) {
    for (int j = 0; j < grid.get_n(); ++j) {
      for (int i = 0; i < grid.get_n(); ++i) {
        int idx = grid.idxNoSpin(i, j, k);
        double w = 1.0;

        if (i == 0 || i == grid.get_n() - 1) {
          w *= 0.5;
        }
        if (j == 0 || j == grid.get_n() - 1) {
          w *= 0.5;
        }
        if (k == 0 || k == grid.get_n() - 1) {
          w *= 0.5;
        }
        res += psi(idx) * hhh * w;
      }
    }
  }

  return res;
}

double Operators::integral(const Eigen::VectorXd &psi, const Grid &grid) {
  double h = grid.get_h();
  double hhh = h * h * h;
  double res = 0.0;
  for (int k = 0; k < grid.get_n(); ++k) {
    for (int j = 0; j < grid.get_n(); ++j) {
      for (int i = 0; i < grid.get_n(); ++i) {
        int idx = grid.idxNoSpin(i, j, k);
        double w = 1.0;

        if (i == 0 || i == grid.get_n() - 1) {
          w *= 0.5;
        }
        if (j == 0 || j == grid.get_n() - 1) {
          w *= 0.5;
        }
        if (k == 0 || k == grid.get_n() - 1) {
          w *= 0.5;
        }
        res += psi(idx) * hhh * w;
      }
    }
  }

  return res;
}
