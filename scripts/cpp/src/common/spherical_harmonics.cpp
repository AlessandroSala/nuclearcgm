#include "spherical_harmonics.hpp"
#include <cmath>

double SphericalHarmonics::Y20(double theta) {
  double pi = M_PI;
  return std::sqrt(5.0 / (16.0 * pi)) *
         (3.0 * std::cos(theta) * std::cos(theta) - 1.0);
}
double SphericalHarmonics::Y20(double x, double y, double z) {
  return Y20(std::acos(z / std::sqrt(x * x + y * y + z * z)));
}

double SphericalHarmonics::Y22(double theta, double phi) {
  double pi = M_PI;
  using std::cos;
  using std::sin;

  return std::sqrt(15.0 / (32.0 * pi)) * cos(2 * phi) * sin(theta) * sin(theta);
}
double SphericalHarmonics::Y22(double x, double y, double z) {
  return Y22(std::acos(z / std::sqrt(x * x + y * y + z * z)), std::atan2(y, x));
}
Eigen::VectorXd SphericalHarmonics::Y20Grad(double x, double y, double z) {
  Eigen::VectorXd grad(3);
  double r2 = x * x + y * y + z * z;
  double r = std::sqrt(r2);
  double theta = acos(z / r);
  grad.setIdentity();
  grad = grad * 1.0 / (r2 * std::sin(theta));
  grad = grad * std::sqrt(5.0 / (16.0 * M_PI)) *
         (-6.0 * std::sin(theta) * std::cos(theta));
  // grad(0) = grad(0) * (-4.0) * x * pow(r2, -3);
  // grad(1) = grad(1) * (-4.0) * y * pow(r2, -3);
  // grad(2) = grad(2) * (r + 4*z*z*pow(r, -3.0/2.0)) * z * pow(r, -2);
  grad(0) = -grad(0) * x * z;
  grad(1) = -grad(1) * y * z;
  grad(2) = grad(2) * (+x * x + y * y);

  return grad;
}
