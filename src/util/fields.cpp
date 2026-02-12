#include "util/fields.hpp"

Eigen::VectorXd Fields::position() {
  auto grid = Grid::getInstance();
  Eigen::VectorXd res(grid->get_total_spatial_points());
  int n = grid->get_n();
  auto xs = grid->get_xs();
  auto ys = grid->get_ys();
  auto zs = grid->get_zs();
#pragma omp parallel for collapse(2)
  for (int k = 0; k < n; ++k) {
    for (int j = 0; j < n; ++j) {
      for (int i = 0; i < n; ++i) {
        int idx = grid->idxNoSpin(i, j, k);
        double x = xs[i];
        double y = ys[j];
        double z = zs[k];
        res(idx) = std::sqrt(x * x + y * y + z * z);
      }
    }
  }
  return res;
}
