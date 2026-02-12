#include "operators/differential_operators.hpp"
#include <omp.h>
#include <unsupported/Eigen/CXX11/Tensor>

Eigen::VectorXd Operators::dvNoSpin(const Eigen::VectorXd &psi,
                                    const Grid &grid, char dir) {
  Eigen::VectorXd res(grid.get_total_spatial_points());
#pragma omp parallel for collapse(2)
  for (int k = 0; k < grid.get_n(); ++k) {
    for (int j = 0; j < grid.get_n(); ++j) {
      for (int i = 0; i < grid.get_n(); ++i) {
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

#pragma omp parallel for collapse(2)
  for (int k = 0; k < grid.get_n(); ++k) {
    for (int j = 0; j < grid.get_n(); ++j) {
      for (int i = 0; i < grid.get_n(); ++i) {
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

Eigen::VectorXcd Operators::dv2(const Eigen::VectorXcd &psi, const Grid &grid,
                                char dir) {
  Eigen::VectorXcd res(grid.get_total_points());
#pragma omp parallel for collapse(2)
  for (int k = 0; k < grid.get_n(); ++k) {
    for (int j = 0; j < grid.get_n(); ++j) {
      for (int i = 0; i < grid.get_n(); ++i) {
        for (int s = 0; s < 2; ++s) {
          int idx = grid.idx(i, j, k, s);
          res(idx) = Operators::derivative2(psi, i, j, k, s, grid, dir);
        }
      }
    }
  }
  return res;
}

Eigen::VectorXd Operators::dv2NoSpin(const Eigen::VectorXd &psi,
                                     const Grid &grid, char dir) {
  Eigen::VectorXd res(grid.get_total_points());
#pragma omp parallel for collapse(2)
  for (int k = 0; k < grid.get_n(); ++k) {
    for (int j = 0; j < grid.get_n(); ++j) {
      for (int i = 0; i < grid.get_n(); ++i) {
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
#pragma omp parallel for collapse(2)
  for (int k = 0; k < grid.get_n(); ++k) {
    for (int j = 0; j < grid.get_n(); ++j) {
      for (int i = 0; i < grid.get_n(); ++i) {
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

Eigen::MatrixXd lagrangeFirstDerivative(int N, double dx) {
  using Eigen::MatrixXd;

  MatrixXd D1 = MatrixXd::Zero(N, N);

  for (int s = 0; s < N; ++s) {
    // Loop over columns (t = basis function)
    for (int t = 0; t < N; ++t) {

      if (s == t) {
        D1(s, t) = 0.0;
      } else {
        double diff = static_cast<double>(t - s);
        double arg = M_PI * diff / N;
        double sin_arg = std::sin(arg);

        D1(s, t) = std::pow(-1.0, diff) * (M_PI / (N * dx)) * (1.0 / sin_arg);
      }
    }
  }
  return D1;
}

Eigen::MatrixXd lagrangeSecondDerivative(int N, double dx) {
  using Eigen::MatrixXd;

  MatrixXd D1 = MatrixXd::Zero(N, N);

  for (int s = 0; s < N; ++s) {
    // Loop over columns (t = basis function)
    for (int t = 0; t < N; ++t) {

      if (s == t) {
        D1(s, t) = std::pow(M_PI * M_PI / (3.0 * dx), 2) * (1.0 - 1.0 / N / N);
      } else {
        double diff = static_cast<double>(t - s);
        double arg = M_PI * diff / N;
        double sin_arg = std::sin(arg);
        double cos_arg = std::cos(arg);
        D1(s, t) = std::pow(-1.0, diff) * std::pow((2 * M_PI / (N * dx)), 2) *
                   (cos_arg / sin_arg / sin_arg);
      }
    }
  }
  return D1;
}

Eigen::MatrixX3d Operators::gradLagrangeNoSpin(const Eigen::VectorXd &vec) {
  Eigen::MatrixX3d res(vec.rows(), 3);
  res.setZero();
  Grid grid = *Grid::getInstance();
  Eigen::MatrixXd D1 = lagrangeFirstDerivative(grid.get_n(), grid.get_h());

  const int N = grid.get_n();

  using ColMajorTensor3 = Eigen::Tensor<double, 3, Eigen::ColMajor>;
  using ReadOnlyTensorMap = Eigen::TensorMap<const ColMajorTensor3>;
  using WriteableTensorMap = Eigen::TensorMap<ColMajorTensor3>;

  ReadOnlyTensorMap F_map(vec.data(), {N, N, N});

  WriteableTensorMap dFdx_map(res.col(0).data(), {N, N, N});
  WriteableTensorMap dFdy_map(res.col(1).data(), {N, N, N});
  WriteableTensorMap dFdz_map(res.col(2).data(), {N, N, N});

  Eigen::Tensor<double, 2> D_tensor(N, N);
  D_tensor = Eigen::TensorMap<const Eigen::Tensor<double, 2, Eigen::ColMajor>>(
      D1.data(), {N, N});

  Eigen::array<Eigen::IndexPair<int>, 1> contract_x = {
      Eigen::IndexPair<int>(0, 0)};
  Eigen::array<Eigen::IndexPair<int>, 1> contract_y = {
      Eigen::IndexPair<int>(0, 1)};
  Eigen::array<Eigen::IndexPair<int>, 1> contract_z = {
      Eigen::IndexPair<int>(0, 2)};

  dFdx_map = D_tensor.contract(F_map, contract_x);

  dFdy_map = D_tensor.contract(F_map, contract_y)
                 .shuffle(Eigen::array<int, 3>{1, 0, 2});

  dFdz_map = D_tensor.contract(F_map, contract_z)
                 .shuffle(Eigen::array<int, 3>{1, 2, 0});

  return res;
}

Eigen::VectorXd Operators::lapLagrangeNoSpin(const Eigen::VectorXd &vec) {
  Eigen::MatrixX3d res(vec.rows(), 3);
  res.setZero();
  Grid grid = *Grid::getInstance();
  Eigen::MatrixXd D1 = lagrangeFirstDerivative(grid.get_n(), grid.get_h()) *
                       lagrangeFirstDerivative(grid.get_n(), grid.get_h());

  const int N = grid.get_n();

  using ColMajorTensor3 = Eigen::Tensor<double, 3, Eigen::ColMajor>;
  using ReadOnlyTensorMap = Eigen::TensorMap<const ColMajorTensor3>;
  using WriteableTensorMap = Eigen::TensorMap<ColMajorTensor3>;

  ReadOnlyTensorMap F_map(vec.data(), {N, N, N});

  WriteableTensorMap dFdx_map(res.col(0).data(), {N, N, N});
  WriteableTensorMap dFdy_map(res.col(1).data(), {N, N, N});
  WriteableTensorMap dFdz_map(res.col(2).data(), {N, N, N});

  Eigen::Tensor<double, 2> D_tensor(N, N);
  D_tensor = Eigen::TensorMap<const Eigen::Tensor<double, 2, Eigen::ColMajor>>(
      D1.data(), {N, N});

  Eigen::array<Eigen::IndexPair<int>, 1> contract_x = {
      Eigen::IndexPair<int>(0, 0)};
  Eigen::array<Eigen::IndexPair<int>, 1> contract_y = {
      Eigen::IndexPair<int>(0, 1)};
  Eigen::array<Eigen::IndexPair<int>, 1> contract_z = {
      Eigen::IndexPair<int>(0, 2)};

  dFdx_map = D_tensor.contract(F_map, contract_x);

  dFdy_map = D_tensor.contract(F_map, contract_y)
                 .shuffle(Eigen::array<int, 3>{1, 0, 2});

  dFdz_map = D_tensor.contract(F_map, contract_z)
                 .shuffle(Eigen::array<int, 3>{1, 2, 0});

  return res.col(0) + res.col(1) + res.col(2);
}

Eigen::MatrixX3cd Operators::gradLagrange(const Eigen::VectorXcd &vec) {
  Eigen::MatrixX3cd res(vec.rows(), 3);
  res.setZero();
  Grid grid = *Grid::getInstance();
  auto D1 = lagrangeFirstDerivative(grid.get_n(), grid.get_h());

  const int N = grid.get_n();
  const int Nspin = 2;

  using dcomplex = std::complex<double>;
  using ColMajorTensor4 = Eigen::Tensor<dcomplex, 4, Eigen::ColMajor>;
  using ReadOnlyTensorMap = Eigen::TensorMap<const ColMajorTensor4>;
  using WriteableTensorMap = Eigen::TensorMap<ColMajorTensor4>;

  ReadOnlyTensorMap F_map(vec.data(), {Nspin, N, N, N});

  WriteableTensorMap dFdx_map(res.col(0).data(), {Nspin, N, N, N});
  WriteableTensorMap dFdy_map(res.col(1).data(), {Nspin, N, N, N});
  WriteableTensorMap dFdz_map(res.col(2).data(), {Nspin, N, N, N});

  Eigen::Tensor<double, 2> D_tensor(N, N);
  D_tensor = Eigen::TensorMap<const Eigen::Tensor<double, 2, Eigen::ColMajor>>(
      D1.data(), {N, N});

  Eigen::array<Eigen::IndexPair<int>, 1> contract_x = {
      Eigen::IndexPair<int>(1, 1)};
  Eigen::array<Eigen::IndexPair<int>, 1> contract_y = {
      Eigen::IndexPair<int>(1, 2)};
  Eigen::array<Eigen::IndexPair<int>, 1> contract_z = {
      Eigen::IndexPair<int>(1, 3)};

  dFdx_map = D_tensor.contract(F_map, contract_x)
                 .shuffle(Eigen::array<int, 4>{1, 0, 2, 3});

  dFdy_map = D_tensor.contract(F_map, contract_y)
                 .shuffle(Eigen::array<int, 4>{1, 2, 0, 3});

  dFdz_map = D_tensor.contract(F_map, contract_z)
                 .shuffle(Eigen::array<int, 4>{1, 2, 3, 0});

  return -res;
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

Eigen::VectorXcd Operators::lap(const Eigen::VectorXcd &vec, const Grid &grid) {
  Eigen::VectorXcd res(vec.rows());
  res.setZero();
  res += dv2(vec, grid, 'x');
  res += dv2(vec, grid, 'y');
  res += dv2(vec, grid, 'z');
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
  } else if (pos == 2 || pos == n - 3) {
    // Derivata centrata a 5 punti (ordine 4)
    double f_2 = psi(idx(pos - 2, pos - 2, pos - 2));
    double f_1 = psi(idx(pos - 1, pos - 1, pos - 1));
    double f1 = psi(idx(pos + 1, pos + 1, pos + 1));
    double f2 = psi(idx(pos + 2, pos + 2, pos + 2));
    return (-f2 + 8.0 * f1 - 8.0 * f_1 + f_2) / (12.0 * h);
  } else if (pos == 3 || pos == n - 4) {
    auto f_3 = psi(idx(pos - 3, pos - 3, pos - 3));
    auto f_2 = psi(idx(pos - 2, pos - 2, pos - 2));
    auto f_1 = psi(idx(pos - 1, pos - 1, pos - 1));
    auto f0 = psi(idx(pos, pos, pos));
    auto f1 = psi(idx(pos + 1, pos + 1, pos + 1));
    auto f2 = psi(idx(pos + 2, pos + 2, pos + 2));
    auto f3 = psi(idx(pos + 3, pos + 3, pos + 3));

    return (-f_3 + 9.0 * f_2 - 45.0 * f_1 + 45.0 * f1 - 9.0 * f2 + f3) /
           (60.0 * h);
  } else {
    auto f_4 = psi(idx(pos - 4, pos - 4, pos - 4));
    auto f_3 = psi(idx(pos - 3, pos - 3, pos - 3));
    auto f_2 = psi(idx(pos - 2, pos - 2, pos - 2));
    auto f_1 = psi(idx(pos - 1, pos - 1, pos - 1));
    auto f0 = psi(idx(pos, pos, pos));
    auto f1 = psi(idx(pos + 1, pos + 1, pos + 1));
    auto f2 = psi(idx(pos + 2, pos + 2, pos + 2));
    auto f3 = psi(idx(pos + 3, pos + 3, pos + 3));
    auto f4 = psi(idx(pos + 4, pos + 4, pos + 4));

    return (3.0 * f_4 - 32.0 * f_3 + 168.0 * f_2 - 672.0 * f_1 + 672.0 * f1 -
            168.0 * f2 + 32.0 * f3 - 3.0 * f4) /
           (840.0 * h);
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
  } else if (pos == 3 || pos == n - 4) {
    std::complex<double> f_3 = psi(idx(pos - 3, pos - 3, pos - 3));
    std::complex<double> f_2 = psi(idx(pos - 2, pos - 2, pos - 2));
    std::complex<double> f_1 = psi(idx(pos - 1, pos - 1, pos - 1));
    std::complex<double> f0 = psi(idx(pos, pos, pos));
    std::complex<double> f1 = psi(idx(pos + 1, pos + 1, pos + 1));
    std::complex<double> f2 = psi(idx(pos + 2, pos + 2, pos + 2));
    std::complex<double> f3 = psi(idx(pos + 3, pos + 3, pos + 3));
    return (-f_3 + 9.0 * f_2 - 45.0 * f_1 + 45.0 * f1 - 9.0 * f2 + f3) /
           (60.0 * h);
  } else {
    auto f_4 = psi(idx(pos - 4, pos - 4, pos - 4));
    auto f_3 = psi(idx(pos - 3, pos - 3, pos - 3));
    auto f_2 = psi(idx(pos - 2, pos - 2, pos - 2));
    auto f_1 = psi(idx(pos - 1, pos - 1, pos - 1));
    auto f0 = psi(idx(pos, pos, pos));
    auto f1 = psi(idx(pos + 1, pos + 1, pos + 1));
    auto f2 = psi(idx(pos + 2, pos + 2, pos + 2));
    auto f3 = psi(idx(pos + 3, pos + 3, pos + 3));
    auto f4 = psi(idx(pos + 4, pos + 4, pos + 4));

    return (3.0 * f_4 - 32.0 * f_3 + 168.0 * f_2 - 672.0 * f_1 + 672.0 * f1 -
            168.0 * f2 + 32.0 * f3 - 3.0 * f4) /
           (840.0 * h);
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
  } else if (pos == 3 || pos == n - 4) {
    std::complex<double> f_3 = psi(idx(pos - 3, pos - 3, pos - 3));
    std::complex<double> f_2 = psi(idx(pos - 2, pos - 2, pos - 2));
    std::complex<double> f_1 = psi(idx(pos - 1, pos - 1, pos - 1));
    std::complex<double> f0 = psi(idx(pos, pos, pos));
    std::complex<double> f1 = psi(idx(pos + 1, pos + 1, pos + 1));
    std::complex<double> f2 = psi(idx(pos + 2, pos + 2, pos + 2));
    std::complex<double> f3 = psi(idx(pos + 3, pos + 3, pos + 3));
    return (-f_3 + 9.0 * f_2 - 45.0 * f_1 + 45.0 * f1 - 9.0 * f2 + f3) /
           (60.0 * h);
  } else {
    auto f_4 = psi(idx(pos - 4, pos - 4, pos - 4));
    auto f_3 = psi(idx(pos - 3, pos - 3, pos - 3));
    auto f_2 = psi(idx(pos - 2, pos - 2, pos - 2));
    auto f_1 = psi(idx(pos - 1, pos - 1, pos - 1));
    auto f0 = psi(idx(pos, pos, pos));
    auto f1 = psi(idx(pos + 1, pos + 1, pos + 1));
    auto f2 = psi(idx(pos + 2, pos + 2, pos + 2));
    auto f3 = psi(idx(pos + 3, pos + 3, pos + 3));
    auto f4 = psi(idx(pos + 4, pos + 4, pos + 4));

    return (3.0 * f_4 - 32.0 * f_3 + 168.0 * f_2 - 672.0 * f_1 + 672.0 * f1 -
            168.0 * f2 + 32.0 * f3 - 3.0 * f4) /
           (840.0 * h);
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
  } else if (pos == 3 || pos == n - 4) {
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
  } else {
    auto f_4 = psi(idx(pos - 4, pos - 4, pos - 4));
    auto f_3 = psi(idx(pos - 3, pos - 3, pos - 3));
    auto f_2 = psi(idx(pos - 2, pos - 2, pos - 2));
    auto f_1 = psi(idx(pos - 1, pos - 1, pos - 1));
    auto f0 = psi(idx(pos, pos, pos));
    auto f1 = psi(idx(pos + 1, pos + 1, pos + 1));
    auto f2 = psi(idx(pos + 2, pos + 2, pos + 2));
    auto f3 = psi(idx(pos + 3, pos + 3, pos + 3));
    auto f4 = psi(idx(pos + 4, pos + 4, pos + 4));

    return (-9.0 * f_4 + 128.0 * f_3 - 1008.0 * f_2 + 8064.0 * f_1 -
            14350.0 * f0 + 8064.0 * f1 - 1008.0 * f2 + 128.0 * f3 - 9.0 * f4) /
           (5040.0 * h * h);
  }
}

std::complex<double> Operators::derivative2(const Eigen::VectorXcd &psi, int i,
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

  // Estremi: schema a 2 punti (ordine 1)
  if (pos == 0) {
    std::complex<double> f0 = psi(idx(pos, pos, pos));
    std::complex<double> f1 = psi(idx(pos + 1, pos + 1, pos + 1));
    return (f1 - f0) / (h);
  } else if (pos == 1 || pos == n - 2) {
    // Centrata a 3 punti (ordine 2)
    std::complex<double> f_1 = psi(idx(pos - 1, pos - 1, pos - 1));
    std::complex<double> f0 = psi(idx(pos, pos, pos));
    std::complex<double> f1 = psi(idx(pos + 1, pos + 1, pos + 1));
    return (f_1 - 2.0 * f0 + f1) / (h * h);
  } else if (pos == n - 1) {
    // Note: Similar to pos == 0, this looks like a first derivative backward
    // stencil. Replicating original logic:
    std::complex<double> f0 = psi(idx(pos, pos, pos));
    std::complex<double> f_1 = psi(idx(pos - 1, pos - 1, pos - 1));
    return (f0 - f_1) / (h); // Stessa approssimazione rozza in coda
  } else if (pos == 2 || pos == n - 3) {
    // Centrata a 5 punti (ordine 4)
    std::complex<double> f_2 = psi(idx(pos - 2, pos - 2, pos - 2));
    std::complex<double> f_1 = psi(idx(pos - 1, pos - 1, pos - 1));
    std::complex<double> f0 = psi(idx(pos, pos, pos));
    std::complex<double> f1 = psi(idx(pos + 1, pos + 1, pos + 1));
    std::complex<double> f2 = psi(idx(pos + 2, pos + 2, pos + 2));
    return (-f2 + 16.0 * f1 - 30.0 * f0 + 16.0 * f_1 - f_2) / (12.0 * h * h);
  } else if (pos == 3 || pos == n - 4) {
    std::complex<double> f_3 = psi(idx(pos - 3, pos - 3, pos - 3));
    std::complex<double> f_2 = psi(idx(pos - 2, pos - 2, pos - 2));
    std::complex<double> f_1 = psi(idx(pos - 1, pos - 1, pos - 1));
    std::complex<double> f0 = psi(idx(pos, pos, pos));
    std::complex<double> f1 = psi(idx(pos + 1, pos + 1, pos + 1));
    std::complex<double> f2 = psi(idx(pos + 2, pos + 2, pos + 2));
    std::complex<double> f3 = psi(idx(pos + 3, pos + 3, pos + 3));
    return (2.0 * f3 - 27.0 * f2 + 270.0 * f1 + -490.0 * f0 + 270.0 * f_1 -
            27.0 * f_2 + 2.0 * f_3) /
           (180.0 * h * h);
  } else {
    auto f_4 = psi(idx(pos - 4, pos - 4, pos - 4));
    auto f_3 = psi(idx(pos - 3, pos - 3, pos - 3));
    auto f_2 = psi(idx(pos - 2, pos - 2, pos - 2));
    auto f_1 = psi(idx(pos - 1, pos - 1, pos - 1));
    auto f0 = psi(idx(pos, pos, pos));
    auto f1 = psi(idx(pos + 1, pos + 1, pos + 1));
    auto f2 = psi(idx(pos + 2, pos + 2, pos + 2));
    auto f3 = psi(idx(pos + 3, pos + 3, pos + 3));
    auto f4 = psi(idx(pos + 4, pos + 4, pos + 4));

    return (-9.0 * f_4 + 128.0 * f_3 - 1008.0 * f_2 + 8064.0 * f_1 -
            14350.0 * f0 + 8064.0 * f1 - 1008.0 * f2 + 128.0 * f3 - 9.0 * f4) /
           (5040.0 * h * h);
  }
}
