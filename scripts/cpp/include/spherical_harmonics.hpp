#include "Eigen/Dense"
namespace SphericalHarmonics {
double Y20(double theta);
double Y20(double x, double y, double z);
double Y22(double theta, double phi);
double Y22(double x, double y, double z);
Eigen::VectorXd Y20Grad(double x, double y, double z);
} // namespace SphericalHarmonics
