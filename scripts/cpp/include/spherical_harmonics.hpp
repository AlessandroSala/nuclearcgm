#include "Eigen/Dense"
namespace SphericalHarmonics {
    double Y20(double theta);
    Eigen::VectorXd Y20Grad(double x, double y, double z);
}
