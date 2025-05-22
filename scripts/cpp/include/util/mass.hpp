#pragma once
#include <Eigen/Dense>
#include <memory>
#include "grid.hpp"
/**
 * @brief Represents the mass term in the Skyrme interaction
 */
class Mass {
public:
    /**
     * @brief Constructs a nuclear radius
     *
     * @param A Mass number
     */
    Mass(Eigen::VectorXd rho, std::shared_ptr<Grid> grid_ptr, double t1, double t3);

    // --- accessors ---

    double getMass(size_t i, size_t j, size_t k) const noexcept;
    Eigen::Vector3d getGradient(size_t, size_t, size_t) const noexcept;

public:
    Eigen::VectorXd mVec;
    std::shared_ptr<Grid> grid_ptr_; // Pointer to the grid configuration

};

