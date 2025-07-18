#include "skyrme/skyrme_u.hpp"
#include "constants.hpp"
#include "types.hpp"
#include <cmath>

SkyrmeU::SkyrmeU(SkyrmeParameters p, NucleonType n_,
                 std::shared_ptr<IterationData> data_)
    : params(p), data(data_), n(n_) {}

double SkyrmeU::getValue(double x, double y, double z) const { return 0.0; }

std::complex<double> SkyrmeU::getElement(int i, int j, int k, int s, int i1,
                                         int j1, int k1, int s1,
                                         const Grid &grid) const {
  return 0.0;
}
std::complex<double> SkyrmeU::getElement5p(int i, int j, int k, int s, int i1,
                                           int j1, int k1, int s1,
                                           const Grid &grid) const {
  if (i != i1 || j != j1 || k != k1 || s != s1) {
    return std::complex<double>(0.0, 0.0);
  }
  int idx = grid.idxNoSpin(i, j, k);
  double field = n == NucleonType::N ? (*data->UN)(idx) : (*data->UP)(idx);
  return std::complex<double>(field, 0.0);
  // TODO: pulire sotto

  // Parametri Skyrme
  double t0 = params.t0, t1 = params.t1, t2 = params.t2;
  double t3 = params.t3, x0 = params.x0, x1 = params.x1;
  double x2 = params.x2, x3 = params.x3, sigma = params.sigma;
  double W0 = params.W0;
  std::complex<double> spinPart = std::complex<double>(0.0, 0.0);
  Eigen::Vector3cd spin =
      n == NucleonType::N ? data->spinN->row(idx) : data->spinP->row(idx);
  auto pauli = nuclearConstants::getPauli();

  for (int i = 0; i < 3; ++i) {
    SpinMatrix res = (-0.5 * t0 * pauli[i] * spin(i));
    spinPart += res(s, s1);
  }
  // spinPart = std::complex<double>(0.0, 0.0);
  if (i != i1 || j != j1 || k != k1 || s != s1) {
    return spinPart;
  }

  // Densità e derivate
  double rho_p = (*data->rhoP)(idx);
  double rho_n = (*data->rhoN)(idx);
  double rho = rho_p + rho_n;
  double nabla2rho = (*data->nabla2RhoN)(idx) + (*data->nabla2RhoP)(idx);
  auto nabla2rho_q =
      n == NucleonType::N ? (*data->nabla2RhoN)(idx) : (*data->nabla2RhoP)(idx);
  auto rho_q = n == NucleonType::N ? rho_n : rho_p;

  double tau_p = (*data->tauP)(idx);
  double tau_n = (*data->tauN)(idx);
  double tau = tau_p + tau_n;
  auto tau_q = n == NucleonType::N ? tau_n : tau_p;
  std::complex<double> curr =
      n == NucleonType::N ? (*data->divJN)(idx) : (*data->divJP)(idx);

  double gRhoP2 = data->nablaRhoP->row(idx).norm(); // |∇ρ_p|²
  double gRhoN2 = data->nablaRhoN->row(idx).norm(); // |∇ρ_n|²
  double gRho2 =
      (data->nablaRhoP->row(idx) + data->nablaRhoN->row(idx)).norm(); // |∇ρ|²

  double res = 0.0;
  // Chabanat
  // res += 0.5 * t0 * ((2 + x0) * rho - (1 + 2 * x0) * rho_q);
  // res += (1.0 / 24.0) * t3 * ((2 + x3) * (2.0 + sigma) * pow(rho, sigma
  // + 1.0) - (2 * x3 + 1) * (2 * pow(rho, sigma) * rho_q + sigma * pow(rho,
  // sigma - 1.0) * (rho_n * rho_n + rho_p * rho_p))); res += (1.0 / 8.0) * (t1
  // * (2 + x1) + t2 * (2 + x2)) * tau; res += (1.0 / 8.0) * (t2 * (2 * x2 + 1)
  // - t1 * (2 * x1 + 1)) * tau_q; res += (1.0 / 16.0) * (t2 * (2 + x2) - 3 * t1
  // * (2 + x1)) * ((*data->nabla2RhoP)(idx) + (*data->nabla2RhoN)(idx)); res +=
  // (1.0 / 16.0) * (3 * t1 * (2 * x1 + 1) + t2 * (2 * x2 + 1)) * nabla2rho_q;

  // Engel
  res += t0 * rho - t0 * 0.5 * rho_q;
  // res += (1.0 / 8.0) * (t2 - 3 * t1) * nabla2rho;
  // res += (1.0 / 16.0) * (3 * t1 + t2) * nabla2rho_q;
  // res += (1.0 / 4.0) * (t1 + t2) * tau;
  // res += (1.0 / 8.0) * (t2 - t1) * tau_q;
  res += (t3 / 12.0) * pow(rho, sigma - 1) *
         ((sigma + 2) * pow(rho, sigma + 1) -
          0.5 * (sigma * pow(rho, sigma - 1) * rho_q * rho_q +
                 2 * pow(rho, sigma) * rho_q));

  // auto complexPart = (1.0/8.0)*(t1-t2)*curr -
  // (1.0/8.0)*(t1*x1+t2*x2)*(data->JN->row(idx).norm() +
  // data->JP->row(idx).norm()); res += (1.0/8.0)*(t1-t2)*

  // res += -0.75 * W0 * ((*data->rhoP)(idx) - (*data->rhoN)(idx));
  return std::complex<double>(res, 0.0) + spinPart;
}
