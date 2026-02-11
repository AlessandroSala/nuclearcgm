#include "spherical_harmonics.hpp"
#include "grid.hpp"
#include "math.h"
#include "operators/integral_operators.hpp"
#include "util/fields.hpp"
#include <cassert>
#include <cmath>

// Factorial function (double to avoid overflow for large l)
double SphericalHarmonics::factorial(int n) {
  assert(n >= 0);
  double res = 1.0;
  for (int i = 2; i <= n; i++)
    res *= i;
  return res;
}

// Associated Legendre polynomial using Condon–Shortley
double SphericalHarmonics::associatedLegendre(int l, int m, double x) {
  assert(m >= 0);
  assert(m <= l);
  assert(std::fabs(x) <= 1.0);

  double pmm = 1.0;
  if (m > 0) {
    double sign = 1.0;
    double fact = 1.0;
    for (int i = 1; i <= 2 * m - 1; i += 2)
      fact *= i;
    pmm = sign * fact * std::pow(1.0 - x * x, m / 2.0);
  }

  if (l == m)
    return pmm;

  double pmmp1 = x * (2 * m + 1) * pmm;
  if (l == m + 1)
    return pmmp1;

  double pll = 0.0;
  double p1 = pmmp1;
  double p2 = pmm;
  for (int ll = m + 2; ll <= l; ++ll) {
    pll = ((2 * ll - 1) * x * p1 - (ll + m - 1) * p2) / (ll - m);
    p2 = p1;
    p1 = pll;
  }
  return pll;
}

Eigen::VectorXd SphericalHarmonics::X(int l, int m) {
  using SphericalHarmonics::Y;
  std::complex<double> img(0.0, 1.0);
  assert(l >= 0);
  assert(std::abs(m) <= l);
  if (m > 0)
    return (Y(l, -m) + Y(l, -m).conjugate()).real() / std::sqrt(2.0);
  if (m < 0)
    return (-img * (Y(l, m) - Y(l, m).conjugate())).real() / std::sqrt(2.0);
  return Y(l, 0).real();
}

std::complex<double> SphericalHarmonics::Y(int l, int m, double theta,
                                           double phi) {
  assert(l >= 0);
  assert(std::abs(m) <= l);

  int mm = std::abs(m);
  double norm = std::sqrt(((2.0 * l + 1.0) / (4.0 * M_PI)) *
                          (factorial(l - mm) / factorial(l + mm)));

  double legendre = associatedLegendre(l, mm, std::cos(theta));
  std::complex<double> phase = std::polar(1.0, mm * phi);

  std::complex<double> res = std::pow(-1.0, mm) * norm * legendre * phase;

  if (m < 0) {
    return std::pow(-1.0, mm) * std::conj(res);
  } else {
    return res;
  }
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

  Eigen::VectorXd pos = Fields::position();
  pos = pos.array().abs();

  using Eigen::VectorXcd;

  VectorXcd YLM = Y(l, m).conjugate();

  assert(pos.rows() == rho.rows() == YLM.rows());

  using comp = std::complex<double>;

  VectorXcd tmp = YLM.array() * rho.cast<comp>().array();
  VectorXcd func = tmp.array() * pos.cast<comp>().array().pow(l);

  return Operators::integralNoSpin(func, *grid);
}

double SphericalHarmonics::massMult(int l, int m, Eigen::VectorXd rho) {
  auto grid = Grid::getInstance();

  Eigen::VectorXd pos = Fields::position().array().abs();

  using Eigen::VectorXd;

  VectorXd XLM = SphericalHarmonics::X(l, m);

  assert(pos.rows() == rho.rows() == XLM.rows());

  using comp = std::complex<double>;

  VectorXd tmp = XLM.array() * rho.array();
  VectorXd func = tmp.array() * pos.array().pow(l);

  return Operators::integral(func, *grid);
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
  grad(0) = -grad(0) * x * z;
  grad(1) = -grad(1) * y * z;
  grad(2) = grad(2) * (+x * x + y * y);

  return grad;
}
