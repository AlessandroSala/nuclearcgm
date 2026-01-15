
#pragma once
#include "grid.hpp"
#include "input_parser.hpp"
#include <Eigen/Dense>

class EDF;

/**
 * @brief Represents the mass term in the Skyrme interaction
 */
class EffectiveMass {
public:
  /**
   * @brief Constructs a mass term, function of the nucleon density
   *
   * @param rho nucleon density
   * @param grid_ptr pointer to the grid configuration
   * @param t1 first term of the Skyrme interaction
   * @param t2 second term of the Skyrme interaction
   */

  EffectiveMass(const Grid &grid, Eigen::VectorXd &rho, Eigen::VectorXd &rho_q,
                Eigen::MatrixX3d &nablaRho, Eigen::MatrixX3d &nabla_rho_q,
                double m, std::shared_ptr<EDF> p);

  Eigen::VectorXd vector;
  Eigen::MatrixX3d gradient;
};
