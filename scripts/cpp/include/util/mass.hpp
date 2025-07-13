#pragma once
#include "grid.hpp"
#include "input_parser.hpp"
#include "skyrme/skyrme_u.hpp"
#include "util/iteration_data.hpp"
#include <Eigen/Dense>
#include <memory>

/**
 * @brief Represents the mass term in the Skyrme interaction
 */
class Mass {
public:
  /**
   * @brief Constructs a mass term, function of the nucleon density
   *
   * @param rho nucleon density
   * @param grid_ptr pointer to the grid configuration
   * @param t1 first term of the Skyrme interaction
   * @param t2 second term of the Skyrme interaction
   */

  // TODO: Fix this, we should make the mass hierarchically dependent on the
  // iteration data
  Mass(std::shared_ptr<Grid> grid, std::shared_ptr<IterationData> data,
       SkyrmeParameters p, NucleonType n_);

  double getMass(size_t i, size_t j, size_t k) const noexcept;
  Eigen::VectorXd getMassVector() const noexcept;
  Eigen::Vector3d getGradient(size_t, size_t, size_t) const noexcept;

public:
  std::shared_ptr<Grid> grid_ptr_;
  NucleonType n;
  std::shared_ptr<IterationData> data;
  SkyrmeParameters params;
};
