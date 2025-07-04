#include "util/shell.hpp"
#include "constants.hpp"
#include "operators/angular_momentum.hpp"
#include "operators/common_operators.hpp"
#include "operators/integral_operators.hpp"
#include <complex>
double roundHI(double x) {
  int c = ceil(x * 2);
  int f = floor(x * 2);

  if (c % 2 == 0) {
    return f / 2.0;
  }
  return c / 2.0;
}
double momFromSq(double x) { return ((0.5 * (-1 + sqrt(1 + 4 * x)))); }

Shell::Shell(std::shared_ptr<Grid> grid_ptr,
             std::shared_ptr<Eigen::VectorXcd> psi_, double energy_)
    : grid(grid_ptr), psi(psi_), energy(energy_) {}

double Shell::l() {
  using namespace nuclearConstants;
  Eigen::VectorXcd psicL2psi =
      (*psi * (Operators::L2(*psi, *grid).adjoint())).diagonal();
  std::complex<double> L2 = Operators::integral(psicL2psi, *grid);
  return round(momFromSq(L2.real() / (h_bar * h_bar)));
}
double Shell::j() {
  using namespace nuclearConstants;
  Eigen::VectorXcd psicJ2psi =
      (*psi * (Operators::J2(*psi, *grid).adjoint())).diagonal();
  std::complex<double> J2 = Operators::integral(psicJ2psi, *grid);
  return roundHI(momFromSq(J2.real() / (h_bar * h_bar)));
}
double Shell::mj() {
  using namespace nuclearConstants;
  Eigen::VectorXcd psicJzpsi =
      (*psi * (Operators::Jz(Operators::Jz(*psi, *grid), *grid)).adjoint())
          .diagonal();
  auto mj =
      roundHI(sqrt(Operators::integral(psicJzpsi, *grid).real()) / (h_bar));
  return mj;
}
double Shell::P() {
  using namespace nuclearConstants;
  Eigen::VectorXcd par = Operators::P(*psi, *grid);
  auto res = par.dot(*psi);
  return res.real();
  return round((res / std::norm(res)).real());
}
