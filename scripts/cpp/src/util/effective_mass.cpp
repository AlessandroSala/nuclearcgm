#include "util/effective_mass.hpp"
#include "constants.hpp"
#include "operators/differential_operators.hpp"
#include "util/iteration_data.hpp"

EffectiveMass::EffectiveMass(const Grid &grid_ptr, Eigen::VectorXd &rho,
                             Eigen::VectorXd &rho_q, Eigen::MatrixX3d &nablaRho,
                             Eigen::MatrixX3d &nabla_rho_q, double massCorr,
                             SkyrmeParameters p) {
  double t1 = p.t1, t2 = p.t2;
  double x1 = p.x1, x2 = p.x2;

  using namespace nuclearConstants;
  using Eigen::VectorXd;
  vector = Eigen::VectorXd(grid_ptr.get_total_spatial_points());
  gradient = Eigen::MatrixX3d(grid_ptr.get_total_spatial_points(), 3);
  vector.setZero();
  gradient.setZero();
  VectorXd ones = VectorXd::Ones(grid_ptr.get_total_spatial_points());

  vector += ones * h_bar * h_bar / (2 * massCorr);

  vector += 0.25 * (t1 + t2 + 0.5 * (t1 * x1 + t2 * x2)) * rho;
  vector += (1.0 / 8.0) * (t2 * (1 + 2 * x2) - t1 * (1 + 2 * x1)) * rho_q;

  gradient += 0.25 * (t1 + t2 + 0.5 * (t1 * x1 + t2 * x2)) * nablaRho;
  gradient +=
      (1.0 / 8.0) * (t2 * (1 + 2 * x2) - t1 * (1 + 2 * x1)) * nabla_rho_q;
}
