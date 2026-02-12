#include "operators/common_operators.hpp"
#include "Eigen/src/Core/Matrix.h"
#include <iostream>
Eigen::VectorXcd Operators::P(const Eigen::VectorXcd &psi, const Grid &grid) {
  Eigen::VectorXcd res(grid.get_total_points());
  int n = grid.get_n();
  for (int k = 0; k < n; ++k) {
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < n; ++i) {
        for (int s = 0; s < 2; ++s) {
          int idx = grid.idx(i, j, k, s);

          int i1 = n - i - 1;
          int j1 = n - j - 1;
          int k1 = n - k - 1;
          res(idx) = psi(grid.idx(i1, j1, k1, s));
        }
      }
    }
  }
  return res;
}

Eigen::MatrixX3d Operators::leviCivita(Real2Tensor J) {

  Eigen::MatrixX3d res(J.rows(), 3);

  // J_x = J_yz - J_zy
  res.col(0) = J.col(3 * 1 + 2) - J.col(2 * 3 + 1);

  // J_y = J_zx - J_xz
  res.col(1) = J.col(3 * 2 + 0) - J.col(3 * 0 + 2);

  // J_z = J_xy - J_yx
  res.col(2) = J.col(3 * 0 + 1) - J.col(3 * 1 + 0);

  return res;
}

Eigen::VectorXd Operators::dot(Eigen::MatrixX3d x, Eigen::MatrixX3d y) {
  if (x.array().isNaN().any() || y.array().isNaN().any()) {
    std::cerr << "Attenzione: dot(x, y) ha prodotto un NaN!" << std::endl;
  }
  Eigen::MatrixX3d x1 = 1e10 * x;
  Eigen::MatrixX3d y1 = 1e10 * y;

  return (x1 * y1.adjoint()).diagonal() / 1e20;
  return (x.col(0).array() * y.col(0).array() +
          x.col(1).array() * y.col(1).array() +
          x.col(2).array() * y.col(2).array())
      .matrix();
}

Eigen::VectorXd Operators::mod2(Eigen::MatrixX3d x) {
  return Operators::dot(x, x);
}
