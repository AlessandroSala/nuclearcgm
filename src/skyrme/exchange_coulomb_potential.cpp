#include "skyrme/exchange_coulomb_potential.hpp"
#include "constants.hpp"
#include "omp.h"

ExchangeCoulombPotential::ExchangeCoulombPotential(
    std::shared_ptr<Eigen::VectorXd> rho_)
    : rho(rho_) {}
double ExchangeCoulombPotential::getValue(double x, double y, double z) const {
  return 0.0;
}
std::complex<double>
ExchangeCoulombPotential::getElement5p(int i, int j, int k, int s, int i1,
                                       int j1, int k1, int s1,
                                       const Grid &grid) const {
  std::complex<double> res = std::complex<double>(0.0, 0.0);
  if (i != i1 || j != j1 || k != k1 || s != s1) {
    return res;
  }
  double h = grid.get_h();
  using namespace nuclearConstants;

  res -= e2 * pow(3.0 / M_PI, 1.0 / 3.0) *
         pow((*rho)(grid.idxNoSpin(i, j, k)), 1.0 / 3.0);
  return res;
}
