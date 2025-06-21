#include "operators/common_operators.hpp"

Eigen::VectorXcd Operators::P(const Eigen::VectorXcd &psi, const Grid &grid) {
  Eigen::VectorXcd res(grid.get_total_points());
  int n = grid.get_n();
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      for (int k = 0; k < n; ++k) {
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
