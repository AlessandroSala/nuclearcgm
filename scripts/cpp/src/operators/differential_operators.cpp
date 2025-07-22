#include "operators/differential_operators.hpp"
#include "Eigen/src/Core/Matrix.h"
#include <iostream>
#include <omp.h> // Include OpenMP header

Eigen::VectorXd Operators::dvNoSpin(const Eigen::VectorXd &psi,
                                    const Grid &grid, char dir) {
  Eigen::VectorXd res(grid.get_total_spatial_points());
// Parallelize the outer loops. Each thread will handle a portion of the (i,j,k)
// iterations. The 'res(idx)' write is safe as 'idx' will be unique for each
// (i,j,k). collapse(3) treats the three nested loops as a single larger loop
// for parallelization, which can improve load balancing and reduce overhead if
// grid.get_n() is large enough.
#pragma omp parallel for collapse(3)
  for (int i = 0; i < grid.get_n(); ++i) {
    for (int j = 0; j < grid.get_n(); ++j) {
      for (int k = 0; k < grid.get_n(); ++k) {
        int idx = grid.idxNoSpin(i, j, k);
        res(idx) = Operators::derivativeNoSpin(psi, i, j, k, grid, dir);
      }
    }
  }

  for (int i = 0; i < res.rows(); i++) {
    if (std::isnan(res(i))) {
      res(i) = 0.0;
    }
  }

  return res;
}

Eigen::VectorXcd Operators::dvNoSpin(const Eigen::VectorXcd &psi,
                                     const Grid &grid, char dir) {
  Eigen::VectorXcd res(grid.get_total_spatial_points());

#pragma omp parallel for collapse(3)
  for (int i = 0; i < grid.get_n(); ++i) {
    for (int j = 0; j < grid.get_n(); ++j) {
      for (int k = 0; k < grid.get_n(); ++k) {
        int idx = grid.idxNoSpin(i, j, k);
        res(idx) = Operators::derivativeNoSpin(psi, i, j, k, grid, dir);
      }
    }
  }

  for (int i = 0; i < res.rows(); i++) {
    if (std::isnan(res(i).real()) || std::isnan(res(i).imag())) {
      res(i) = std::complex<double>(0.0, 0.0);
    }
  }

  return res;
}

Eigen::VectorXd Operators::dv2NoSpin(const Eigen::VectorXd &psi,
                                     const Grid &grid, char dir) {
  Eigen::VectorXd res(grid.get_total_points());
#pragma omp parallel for collapse(3)
  for (int i = 0; i < grid.get_n(); ++i) {
    for (int j = 0; j < grid.get_n(); ++j) {
      for (int k = 0; k < grid.get_n(); ++k) {
        int idx = grid.idxNoSpin(i, j, k);
        res(idx) = Operators::derivative2NoSpin(psi, i, j, k, grid, dir);
      }
    }
  }
  return res;
}

Eigen::VectorXcd Operators::dv(const Eigen::VectorXcd &psi, const Grid &grid,
                               char dir) {
  Eigen::VectorXcd res(grid.get_total_points());
  // Parallelize the (i,j,k) loops. The innermost loop over 's' (spin) is small
  // (size 2) and is kept sequential within each parallel iteration for
  // efficiency. The 'res(idx)' write is safe as 'idx' is unique for each
  // (i,j,k,s).
#pragma omp parallel for collapse(3)
  for (int i = 0; i < grid.get_n(); ++i) {
    for (int j = 0; j < grid.get_n(); ++j) {
      for (int k = 0; k < grid.get_n(); ++k) {
        for (int s = 0; s < 2; ++s) {
          int idx = grid.idx(i, j, k, s);
          res(idx) = Operators::derivative(psi, i, j, k, s, grid, dir);
        }
      }
    }
  }

  for (int i = 0; i < res.rows(); i++) {
    if (std::isnan(res(i).real()) || std::isnan(res(i).imag())) {
      res(i) = std::complex<double>(0.0, 0.0);
    }
  }
  return res;
}

Eigen::MatrixX3cd Operators::grad(const Eigen::VectorXcd &vec,
                                  const Grid &grid) {
  Eigen::MatrixX3cd res(vec.rows(), 3);
  res.setZero();
  auto dx = dv(vec, grid, 'x'); // This call is internally parallelized
  auto dy = dv(vec, grid, 'y'); // This call is internally parallelized
  auto dz = dv(vec, grid, 'z'); // This call is internally parallelized
  res.col(0) = dx;
  res.col(1) = dy;
  res.col(2) = dz;
  return res;
}
// gradNoSpin benefits from dvNoSpin being parallelized.
// No direct OpenMP pragmas are typically needed here if the called functions
// are parallel.
Eigen::Matrix<double, Eigen::Dynamic, 3>
Operators::gradNoSpin(const Eigen::VectorXd &vec, const Grid &grid) {
  Eigen::Matrix<double, -1, 3> res(vec.rows(), 3);
  res.setZero();
  auto dx = dvNoSpin(vec, grid, 'x'); // This call is internally parallelized
  auto dy = dvNoSpin(vec, grid, 'y'); // This call is internally parallelized
  auto dz = dvNoSpin(vec, grid, 'z'); // This call is internally parallelized
  res.col(0) = dx;
  res.col(1) = dy;
  res.col(2) = dz;
  return res;
}

Eigen::VectorXd Operators::lapNoSpin(const Eigen::VectorXd &vec,
                                     const Grid &grid) {
  Eigen::VectorXd res(vec.rows());
  res.setZero();
  res += dv2NoSpin(vec, grid, 'x');
  res += dv2NoSpin(vec, grid, 'y');
  res += dv2NoSpin(vec, grid, 'z');
  return res;
}

Eigen::VectorXcd Operators::divNoSpin(const Eigen::MatrixX3cd &vec,
                                      const Grid &grid) {

  Eigen::VectorXcd vx = vec.col(0);
  Eigen::VectorXcd vy = vec.col(1);
  Eigen::VectorXcd vz = vec.col(2);
  Eigen::VectorXcd dx = dvNoSpin(vx, grid, 'x');
  Eigen::VectorXcd dy = dvNoSpin(vy, grid, 'y');
  Eigen::VectorXcd dz = dvNoSpin(vz, grid, 'z');
  return dx + dy + dz;
}

Eigen::VectorXd Operators::divNoSpin(const Eigen::MatrixX3d &vec,
                                     const Grid &grid) {

  Eigen::VectorXd vx = vec.col(0);
  Eigen::VectorXd vy = vec.col(1);
  Eigen::VectorXd vz = vec.col(2);
  Eigen::VectorXd dx = dvNoSpin(vx, grid, 'x');
  Eigen::VectorXd dy = dvNoSpin(vy, grid, 'y');
  Eigen::VectorXd dz = dvNoSpin(vz, grid, 'z');
  return dx + dy + dz;
}
// The derivative functions (derivativeNoSpin, derivative, derivative2) are
// called by the parallelized loops. They don't contain loops that are
// themselves good candidates for internal parallelization in this context. They
// are thread-safe as they operate on local variables and read-only shared data
// (psi, grid) or data passed by value/const reference.

double Operators::derivativeNoSpin(const Eigen::VectorXd &psi, int i, int j,
                                   int k, const Grid &grid, char axis) {
  int n = grid.get_n();
  double h = grid.get_h();

  auto idx = [&](int ii, int jj, int kk) {
    return axis == 'x'   ? grid.idxNoSpin(ii, j, k)
           : axis == 'y' ? grid.idxNoSpin(i, jj, k)
                         : grid.idxNoSpin(i, j, kk);
  };

  int pos = axis == 'x' ? i : axis == 'y' ? j : k;

  if (pos == 0) {
    // Forward difference (2 punti)
    double f0 = psi(idx(pos, pos, pos));
    double f1 = psi(idx(pos + 1, pos + 1, pos + 1));
    return (f1 - f0) / h;
  } else if (pos == 1 || pos == n - 2) {
    // Derivata centrata a 3 punti
    double f_1 = psi(idx(pos - 1, pos - 1, pos - 1));
    double f1 = psi(idx(pos + 1, pos + 1, pos + 1));
    return (f1 - f_1) / (2.0 * h);
  } else if (pos == n - 1) {
    // Backward difference (2 punti)
    double f0 = psi(idx(pos, pos, pos));
    double f_1 = psi(idx(pos - 1, pos - 1, pos - 1));
    return (f0 - f_1) / h;
  } else {
    // Derivata centrata a 5 punti (ordine 4)
    double f_2 = psi(idx(pos - 2, pos - 2, pos - 2));
    double f_1 = psi(idx(pos - 1, pos - 1, pos - 1));
    double f1 = psi(idx(pos + 1, pos + 1, pos + 1));
    double f2 = psi(idx(pos + 2, pos + 2, pos + 2));
    return (-f2 + 8.0 * f1 - 8.0 * f_1 + f_2) / (12.0 * h);
  }
}

std::complex<double> Operators::derivativeNoSpin(const Eigen::VectorXcd &psi,
                                                 int i, int j, int k,
                                                 const Grid &grid, char axis) {
  int n = grid.get_n();
  double h = grid.get_h();

  auto idx = [&](int ii, int jj, int kk) {
    return axis == 'x'   ? grid.idxNoSpin(ii, j, k)
           : axis == 'y' ? grid.idxNoSpin(i, jj, k)
                         : grid.idxNoSpin(i, j, kk);
  };

  int pos = axis == 'x' ? i : axis == 'y' ? j : k;

  if (pos == 0) {
    // Forward difference (2 punti)
    std::complex<double> f0 = psi(idx(pos, pos, pos));
    std::complex<double> f1 = psi(idx(pos + 1, pos + 1, pos + 1));
    return (f1 - f0) / h;
  } else if (pos == 1 || pos == n - 2) {
    // Derivata centrata a 3 punti
    std::complex<double> f_1 = psi(idx(pos - 1, pos - 1, pos - 1));
    std::complex<double> f1 = psi(idx(pos + 1, pos + 1, pos + 1));
    return (f1 - f_1) / (2.0 * h);
  } else if (pos == n - 1) {
    // Backward difference (2 punti)
    std::complex<double> f0 = psi(idx(pos, pos, pos));
    std::complex<double> f_1 = psi(idx(pos - 1, pos - 1, pos - 1));
    return (f0 - f_1) / h;
  } else if (pos == 2 || pos == n - 3) {
    // Derivata centrata a 5 punti (ordine 4)
    std::complex<double> f_2 = psi(idx(pos - 2, pos - 2, pos - 2));
    std::complex<double> f_1 = psi(idx(pos - 1, pos - 1, pos - 1));
    std::complex<double> f1 = psi(idx(pos + 1, pos + 1, pos + 1));
    std::complex<double> f2 = psi(idx(pos + 2, pos + 2, pos + 2));
    return (-f2 + 8.0 * f1 - 8.0 * f_1 + f_2) / (12.0 * h);
  } else {
    std::complex<double> f_3 = psi(idx(pos - 3, pos - 3, pos - 3));
    std::complex<double> f_2 = psi(idx(pos - 2, pos - 2, pos - 2));
    std::complex<double> f_1 = psi(idx(pos - 1, pos - 1, pos - 1));
    std::complex<double> f0 = psi(idx(pos, pos, pos));
    std::complex<double> f1 = psi(idx(pos + 1, pos + 1, pos + 1));
    std::complex<double> f2 = psi(idx(pos + 2, pos + 2, pos + 2));
    std::complex<double> f3 = psi(idx(pos + 3, pos + 3, pos + 3));
    return (-f_3 + 9.0 * f_2 - 45.0 * f_1 + 45.0 * f1 - 9.0 * f2 + f3) /
           (60.0 * h);
  }
}

std::complex<double> Operators::derivative(const Eigen::VectorXcd &psi, int i,
                                           int j, int k, int s,
                                           const Grid &grid, char axis) {
  int n = grid.get_n();
  double h = grid.get_h();

  auto idx = [&](int ii, int jj, int kk) {
    return axis == 'x'   ? grid.idx(ii, j, k, s)
           : axis == 'y' ? grid.idx(i, jj, k, s)
                         : grid.idx(i, j, kk, s);
  };

  int pos = axis == 'x' ? i : axis == 'y' ? j : k;

  if (pos == 0) {
    // Forward difference (2 punti)
    std::complex<double> f0 = psi(idx(pos, pos, pos));
    std::complex<double> f1 = psi(idx(pos + 1, pos + 1, pos + 1));
    return (f1 - f0) / h;
  } else if (pos == 1 || pos == n - 2) {
    // Derivata centrata a 3 punti
    std::complex<double> f_1 = psi(idx(pos - 1, pos - 1, pos - 1));
    std::complex<double> f1 = psi(idx(pos + 1, pos + 1, pos + 1));
    return (f1 - f_1) / (2.0 * h);
  } else if (pos == n - 1) {
    // Backward difference (2 punti)
    std::complex<double> f0 = psi(idx(pos, pos, pos));
    std::complex<double> f_1 = psi(idx(pos - 1, pos - 1, pos - 1));
    return (f0 - f_1) / h;
  } else if (pos == 2 || pos == n - 3) {
    // Derivata centrata a 5 punti (ordine 4)
    std::complex<double> f_2 = psi(idx(pos - 2, pos - 2, pos - 2));
    std::complex<double> f_1 = psi(idx(pos - 1, pos - 1, pos - 1));
    std::complex<double> f1 = psi(idx(pos + 1, pos + 1, pos + 1));
    std::complex<double> f2 = psi(idx(pos + 2, pos + 2, pos + 2));
    return (-f2 + 8.0 * f1 - 8.0 * f_1 + f_2) / (12.0 * h);
  } else {
    std::complex<double> f_3 = psi(idx(pos - 3, pos - 3, pos - 3));
    std::complex<double> f_2 = psi(idx(pos - 2, pos - 2, pos - 2));
    std::complex<double> f_1 = psi(idx(pos - 1, pos - 1, pos - 1));
    std::complex<double> f0 = psi(idx(pos, pos, pos));
    std::complex<double> f1 = psi(idx(pos + 1, pos + 1, pos + 1));
    std::complex<double> f2 = psi(idx(pos + 2, pos + 2, pos + 2));
    std::complex<double> f3 = psi(idx(pos + 3, pos + 3, pos + 3));
    return (-f_3 + 9.0 * f_2 - 45.0 * f_1 + 45.0 * f1 - 9.0 * f2 + f3) /
           (60.0 * h);
  }
}

double Operators::derivative2NoSpin(const Eigen::VectorXd &psi, int i, int j,
                                    int k, const Grid &grid, char axis) {

  int n = grid.get_n();
  double h = grid.get_h();

  auto idx = [&](int ii, int jj, int kk) {
    return axis == 'x'   ? grid.idxNoSpin(ii, j, k)
           : axis == 'y' ? grid.idxNoSpin(i, jj, k)
                         : grid.idxNoSpin(i, j, kk);
  };

  int pos = axis == 'x' ? i : axis == 'y' ? j : k;

  // Estremi: schema a 2 punti (ordine 1)
  if (pos == 0) {
    double f0 = psi(idx(pos, pos, pos));
    double f1 = psi(idx(pos + 1, pos + 1, pos + 1));
    return (f1 - f0) / (h);
  } else if (pos == 1 || pos == n - 2) {
    // Centrata a 3 punti (ordine 2)
    double f_1 = psi(idx(pos - 1, pos - 1, pos - 1));
    double f0 = psi(idx(pos, pos, pos));
    double f1 = psi(idx(pos + 1, pos + 1, pos + 1));
    return (f_1 - 2.0 * f0 + f1) / (h * h);
  } else if (pos == n - 1) {
    // Note: Similar to pos == 0, this looks like a first derivative backward
    // stencil. Replicating original logic:
    double f0 = psi(idx(pos, pos, pos));
    double f_1 = psi(idx(pos - 1, pos - 1, pos - 1));
    return (f0 - f_1) / (h); // Stessa approssimazione rozza in coda
  } else if (pos == 2 || pos == n - 3) {
    // Centrata a 5 punti (ordine 4)
    double f_2 = psi(idx(pos - 2, pos - 2, pos - 2));
    double f_1 = psi(idx(pos - 1, pos - 1, pos - 1));
    double f0 = psi(idx(pos, pos, pos));
    double f1 = psi(idx(pos + 1, pos + 1, pos + 1));
    double f2 = psi(idx(pos + 2, pos + 2, pos + 2));
    return (-f2 + 16.0 * f1 - 30.0 * f0 + 16.0 * f_1 - f_2) / (12.0 * h * h);
  } else {
    double f_3 = psi(idx(pos - 3, pos - 3, pos - 3));
    double f_2 = psi(idx(pos - 2, pos - 2, pos - 2));
    double f_1 = psi(idx(pos - 1, pos - 1, pos - 1));
    double f0 = psi(idx(pos, pos, pos));
    double f1 = psi(idx(pos + 1, pos + 1, pos + 1));
    double f2 = psi(idx(pos + 2, pos + 2, pos + 2));
    double f3 = psi(idx(pos + 3, pos + 3, pos + 3));
    return (2.0 * f3 - 27.0 * f2 + 270.0 * f1 + -490.0 * f0 + 270.0 * f_1 -
            27.0 * f_2 + 2.0 * f_3) /
           (180.0 * h * h);
  }
}
