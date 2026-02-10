#include "radius.hpp"
#include "spherical_harmonics.hpp"
#include <cmath>

Radius::Radius(double Beta_, double r_0_, int A_)
    : Beta(Beta_), r_0(r_0_), A(A_) {}

double Radius::getRadius(double x, double y, double z) const noexcept {
    double R = r_0 * pow(A, 1.0/3.0);
    double theta = acos(z/std::sqrt(x*x + y*y + z*z));
    return R*(1+Beta*SphericalHarmonics::Y20(theta));
}
Eigen::VectorXd Radius::getGradient(double x, double y, double z) const noexcept {
    double R = r_0 * pow(A, 1.0/3.0);

    return R*Beta*SphericalHarmonics::Y20Grad(x, y, z);
}
