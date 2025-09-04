#include "Eigen/Dense"
#include "Eigen/src/Core/Matrix.h"
#include <complex>
namespace SphericalHarmonics {
double Y20(double theta);
double Y20(double x, double y, double z);
double Y22(double theta, double phi);
double Y22(double x, double y, double z);
double factorial(int n);
double associatedLegendre(int l, int m, double x);
Eigen::VectorXd Y20Grad(double x, double y, double z);
typedef struct angles {
  double theta, phi;
} angles;
std::complex<double> Y(int l, int m, double theta, double phi);
angles cart2spher(double x, double y, double z);
std::complex<double> Y(int l, int m, double x, double y, double z);
Eigen::VectorXcd Y(int l, int m);
Eigen::VectorXd X(int l, int m);
std::complex<double> Q(int l, int m, Eigen::VectorXd rho);
double massMult(int l, int m, Eigen::VectorXd rho);
} // namespace SphericalHarmonics
