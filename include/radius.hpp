#pragma once
#include <Eigen/Dense>
/**
 * @brief Represents the nuclear radius
 */
class Radius {
public:
    /**
     * @brief Constructs a nuclear radius
     *
     * @param Beta Nuclear deformation
     * @param r_0 Radius coefficient
     * @param A Mass number
     */
    Radius(double Beta, double r_0, int A);

    // --- accessors ---

    double getRadius(double x, double y, double z) const noexcept;
    Eigen::VectorXd getGradient(double x, double y, double z) const noexcept;
    double r_0;

private:
    int A; 
    double Beta;
};

