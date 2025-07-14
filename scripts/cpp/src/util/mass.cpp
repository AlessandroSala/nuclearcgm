#include "util/mass.hpp"
#include "constants.hpp"
#include "operators/differential_operators.hpp"
#include <cmath>
#include <iostream>

Mass::Mass(std::shared_ptr<Grid> grid, std::shared_ptr<IterationData> data_,
           SkyrmeParameters p, NucleonType n_)
    : grid_ptr_(grid), params(p), n(n_), data(data_) {}
// TODO: ambiguità!!
double Mass::getMass(size_t i, size_t j, size_t k) const noexcept {
  using namespace nuclearConstants;
  double res = 0.0;
  double t1 = params.t1, t2 = params.t2;
  double x1 = params.x1, x2 = params.x2;
  int idx = grid_ptr_->idxNoSpin(i, j, k);

  // TODO: fix
  double rho_q = n == NucleonType::N ? (*data->rhoN)(idx) : (*data->rhoP)(idx);
  double rho = (*(data->rhoN))(idx) + (*(data->rhoP))(idx);

  res += h_bar * h_bar / (2 * m);

  // Stev. I same  as Chab
  res += 0.25 * (t1 + t2 + 0.5 * (t1 * x1 + t2 * x2)) * rho;
  // Original
  // res += (1.0 / 8.0) * (t2 * (1 + 2 * x2) - t1 * (1 + 2 * x1)) * rho_q;
  // Modified, should be correct
  res += (1.0 / 8.0) * (t2 * (1 + 2 * x2) - t1 * (1 + 2 * x1)) * rho_q;

  // Chabanat
  // res += (1.0 / 8.0) * (t1 * (2 + x1) + t2 * (2 + x2)) * rho;
  // res -= (1.0 / 8.0) * (t1 * (1 + 2 * x1) + t2 * (1 + 2 * x2)) * rho_q;
  //

  return res;
}

Eigen::Vector3d Mass::getGradient(size_t i, size_t j, size_t k) const noexcept {
  Eigen::Vector3d grad(3);
  grad.setZero();

  double t1 = params.t1, t2 = params.t2;
  double x1 = params.x1, x2 = params.x2;
  int idx = grid_ptr_->idxNoSpin(i, j, k);

  Eigen::Vector3d nablaRho_q = n == NucleonType::N ? data->nablaRhoN->row(idx)
                                                   : data->nablaRhoP->row(idx);
  Eigen::Vector3d nablaRho =
      data->nablaRhoN->row(idx) + data->nablaRhoP->row(idx);

  // stev
  grad += 0.25 * (t1 + t2 + 0.5 * (t1 * x1 + t2 * x2)) * nablaRho;
  // Original
  // grad += (1.0 / 8.0) * (t2 * (1 + 2 * x2) - t1 * (1 + 2 * x1)) * nablaRho_q;
  // Modified
  grad += (1.0 / 8.0) * (t2 * (1 + 2 * x2) - t1 * (1 + 2 * x1)) * nablaRho_q;

  // chab
  // grad += (1.0 / 8.0) * (t1 * (2 + x1) + t2 * (2 + x2)) * nablaRho;
  // grad -= (1.0 / 8.0) * (t1 * (1 + 2 * x1) + t2 * (1 + 2 * x2)) * nablaRho_q;
  return grad;
}
Eigen::VectorXd Mass::getMassVector() const noexcept {
  using namespace nuclearConstants;
  Eigen::VectorXd res(grid_ptr_->get_total_spatial_points());

#pragma omp parallel for collapse(3)
  for (int i = 0; i < grid_ptr_->get_n(); ++i) {
    for (int j = 0; j < grid_ptr_->get_n(); ++j) {
      for (int k = 0; k < grid_ptr_->get_n(); ++k) {
        int idx = grid_ptr_->idxNoSpin(i, j, k);
        res(idx) = getMass(i, j, k);
      }
    }
  }

  return res;
}
