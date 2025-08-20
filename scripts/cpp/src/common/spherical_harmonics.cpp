#include "spherical_harmonics.hpp"
#include "grid.hpp"
#include "operators/integral_operators.hpp"
#include "util/fields.hpp"
#include <cassert>
#include <cmath>

std::complex<double> SphericalHarmonics::Y(int l, int m, double theta,
                                           double phi) {
  using std::abs;
  assert(l >= 0);
  assert(abs(m) <= l);
  double norm = std::sqrt(
      ((2.0 * l + 1.0) / (4.0 * M_PI)) *
      ((double)factorial(l - std::abs(m)) / factorial(l + std::abs(m))));

  double legendre =
      SphericalHarmonics::associatedLegendre(l, std::abs(m), std::cos(theta));
  std::complex<double> phase = std::polar(1.0, m * phi); // e^{i m phi}

  if (m < 0)
    return std::pow(-1.0, m) * norm * legendre *
           std::conj(phase); // Relazione coniugata
  else
    return norm * legendre * phase;
}

int SphericalHarmonics::factorial(int n) {
  assert(n >= 0);
  int res = 0;
  for (int i = 2; i <= n; i++)
    res *= i;
  return res;
}

double SphericalHarmonics::associatedLegendre(int l, int m, double x) {
  assert(m >= 0);
  assert(m <= l);
  assert(x <= 1.0);
  double pmm = 1.0;

  if (m > 0) {
    double sx2 = sqrt(1.0 - x * x);
    double fact = 1.0;

    for (int i = 1; i <= m; ++i) {
      pmm *= -fact * sx2;
      fact += 2.0;
    }
  }
  if (l == m)
    return pmm;

  double pmmp1 = x * (2 * m + 1) * pmm;
  if (l == m + 1)
    return pmmp1;

  double pll = 0.0;
  for (int ll = m + 1; ll <= l; ++ll) {
    pll = ((2 * ll - 1) * x * pmmp1 - (ll + m - 1) * pmm) / (ll - m);
    pmm = pmmp1;
    pmmp1 = pll;
  }
  return pll;
}
SphericalHarmonics::angles SphericalHarmonics::cart2spher(double x, double y,
                                                          double z) {
  double r = std::sqrt(x * x + y * y + z * z);
  double theta = std::acos(z / r);
  double phi = std::atan2(y, x);
  return {theta, phi};
}

std::complex<double> SphericalHarmonics::Y(int l, int m, double x, double y,
                                           double z) {
  auto a = cart2spher(x, y, z);
  return Y(l, m, a.theta, a.phi);
}
Eigen::VectorXcd SphericalHarmonics::Y(int l, int m) {
  auto grid = Grid::getInstance();
  Eigen::VectorXcd res(grid->get_total_spatial_points());

#pragma omp parallel for collapse(3)
  for (int i = 0; i < grid->get_n(); ++i) {
    for (int j = 0; j < grid->get_n(); ++j) {
      for (int k = 0; k < grid->get_n(); ++k) {
        int idx = grid->idxNoSpin(i, j, k);
        double x = grid->get_xs()[i];
        double y = grid->get_ys()[j];
        double z = grid->get_zs()[k];
        res(idx) = Y(l, m, x, y, z);
      }
    }
  }
  return res;
}

std::complex<double> SphericalHarmonics::Q(int l, int m, Eigen::VectorXd rho) {
  auto grid = Grid::getInstance();

  auto pos = Fields::position();

  Eigen::VectorXcd func = Y(l, m).array() * rho.array() * pos.array().pow(l);

  return Operators::integral((Eigen::VectorXcd)func, *grid);
}

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
