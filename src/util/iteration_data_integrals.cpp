#include "EDF.hpp"
#include "constants.hpp"
#include "operators/common_operators.hpp"
#include "operators/integral_operators.hpp"
#include "spherical_harmonics.hpp"
#include "util/iteration_data.hpp"
#include <cmath>

double IterationData::EpairP() {
  if (!input.pairing) {
    return 0.0;
  }
  if (!input.pairingP.active)
    return 0.0;
  if (input.pairingType == PairingType::hfb) {
    return HFBResultP.energy;
  }
  if (input.pairingType == PairingType::bcs) {
    return bcsP.Epair;
  }
  return 0.0;
}
double IterationData::EpairN() {
  if (!input.pairing) {
    return 0.0;
  }
  if (!input.pairingN.active)
    return 0.0;
  if (input.pairingType == PairingType::hfb) {
    return HFBResultN.energy;
  }
  if (input.pairingType == PairingType::bcs) {
    return bcsN.Epair;
  }
  return 0.0;
}

double IterationData::betaRealRadius() {

  Eigen::VectorXd rho = *rhoP + *rhoN;
  int A = input.getA();
  double a20 = SphericalHarmonics::Q(2, 0, rho).real();
  double beta = 4 * M_PI * a20 / (5 * A * radius());

  if (axis2Exp('x') > axis2Exp('z')) {
    beta = -beta;
  }

  return beta;
}
QuadrupoleDeformation IterationData::quadrupoleDeformation() {

  Eigen::VectorXd rho = *rhoP + *rhoN;
  int A = input.getA();
  int Z = input.getZ();
  int N = A - Z;
  double a20 = SphericalHarmonics::Q(2, 0, rho).real();
  double a22 = SphericalHarmonics::Q(2, 2, rho).real();
  double R = 1.2 * pow(A, 1.0 / 3.0);

  double gamma = atan2(std::sqrt(2) * a22, a20);

  double beta = 4 * M_PI * a20 / (3 * A * R * R);
  double betaSim = 4 * M_PI * a20 / (5 * A * radius());

  return {beta, gamma};
}

double IterationData::axis2Exp(char dir) {

  Eigen::VectorXd rho = *rhoP + *rhoN;
  auto grid = *Grid::getInstance();
  int n = grid.get_n();
  double res = 0.0;
  for (int k = 0; k < grid.get_n(); ++k) {
    for (int j = 0; j < grid.get_n(); ++j) {
      for (int i = 0; i < grid.get_n(); ++i) {
        int idx = grid.idxNoSpin(i, j, k);
        double ii = grid.get_xs()[i];
        double jj = grid.get_ys()[j];
        double kk = grid.get_zs()[k];
        if (dir == 'x') {
          res += ii * ii * rho(idx);
        } else if (dir == 'y') {
          res += jj * jj * rho(idx);
        } else if (dir == 'z') {
          res += kk * kk * rho(idx);
        }
      }
    }
  }
  return res / rho.sum();
}

double IterationData::C0RhoEnergy() {
  auto grid = *Grid::getInstance();

  Eigen::VectorXd ones = Eigen::VectorXd::Ones(grid.get_total_spatial_points());

  Eigen::VectorXd rho0 = *rhoN + *rhoP;
  Eigen::VectorXd energy0 =
      ((interaction->params.C0rr * ones.array() +
        interaction->params.C0Drr *
            rho0.array().pow(interaction->params.sigma))) *
      rho0.array().square();

  // Eigen::VectorXd energy0 =
  //     (((3.0 / 8.0) * t0 * ones +
  //       (3.0 / 48.0) * t3 * rho0.array().pow(sigma).matrix()) *
  //      rho0.array().pow(2).matrix().transpose())
  //         .diagonal();

  using Operators::integral;
  return integral(Eigen::VectorXd(energy0), grid);
}

double IterationData::C1RhoEnergy() {
  auto grid = *Grid::getInstance();

  Eigen::VectorXd rho0 = *rhoN + *rhoP;
  Eigen::VectorXd rho1 = *rhoN - *rhoP;
  using Operators::integral;
  Eigen::VectorXd ones = Eigen::VectorXd::Ones(grid.get_total_spatial_points());

  Eigen::VectorXd energy1 =
      ((interaction->params.C1rr * ones.array() +
        interaction->params.C1Drr *
            rho0.array().pow(interaction->params.sigma))) *
      rho1.array().square();

  return integral(energy1, grid);
}

double IterationData::C0TauEnergy() {
  Eigen::VectorXd tau0 = *tauN + *tauP;
  Eigen::VectorXd rho0 = *rhoN + *rhoP;

  Eigen::VectorXd prod = (tau0 * rho0.transpose()).diagonal();

  Eigen::VectorXd energy0c = interaction->params.C0rt * prod;

  using Operators::integral;
  auto grid = *Grid::getInstance();
  return integral(energy0c, grid);
}

double IterationData::C1TauEnergy() {

  Eigen::VectorXd tau1 = *tauN - *tauP;
  Eigen::VectorXd rho1 = *rhoN - *rhoP;

  Eigen::VectorXd prod = (tau1 * rho1.transpose()).diagonal();

  Eigen::VectorXd energy1c = interaction->params.C1rt * prod;

  using Operators::integral;
  auto grid = *Grid::getInstance();
  return integral(energy1c, grid);
}

// TODO: SUS !
double IterationData::C0nabla2RhoEnergy() {

  Eigen::VectorXd nabla2Rho0 = *nabla2RhoN + *nabla2RhoP;
  Eigen::VectorXd rho0 = *rhoN + *rhoP;

  Eigen::VectorXd prod = (rho0 * nabla2Rho0.transpose()).diagonal();

  Eigen::VectorXd energy0c = interaction->params.C0rdr * prod;

  using Operators::integral;
  auto grid = *Grid::getInstance();
  return integral(energy0c, grid);
}

double IterationData::C1nabla2RhoEnergy() {
  Eigen::VectorXd nabla2Rho1 = *nabla2RhoN - *nabla2RhoP;
  Eigen::VectorXd rho1 = *rhoN - *rhoP;
  Eigen::VectorXd prod = (nabla2Rho1 * rho1.transpose()).diagonal();

  Eigen::VectorXd energy1c = interaction->params.C1rdr * prod;

  using Operators::integral;
  auto grid = *Grid::getInstance();
  return integral(energy1c, grid);
}

double IterationData::C0J2Energy() {
  auto grid = *Grid::getInstance();
  Eigen::MatrixX3d J0 = (*JvecN + *JvecP);

  using Operators::mod2, Operators::integral;
  Eigen::VectorXd func = interaction->params.C0J2 * mod2(J0);

  return integral(func, grid);
}

double IterationData::C1J2Energy() {
  auto grid = *Grid::getInstance();
  Eigen::MatrixX3d J1 = (*JvecN - *JvecP);

  using Operators::mod2, Operators::integral;
  Eigen::VectorXd func = interaction->params.C1J2 * mod2(J1);

  return integral(func, grid);
}

double IterationData::C0rhoDivJEnergy() {
  Eigen::VectorXd rho = *rhoN + *rhoP;
  Eigen::VectorXd nablaJ = (*divJvecN + *divJvecP);
  Eigen::VectorXd prod = (nablaJ.array() * rho.array()).matrix();
  Eigen::VectorXd func = interaction->params.C0nJ * prod;

  using Operators::integral;

  auto grid = *Grid::getInstance();
  return integral(func, grid);
}

double IterationData::C1rhoDivJEnergy() {
  Eigen::VectorXd rho1 = *rhoN - *rhoP;
  Eigen::VectorXd nablaJ = (*divJvecN - *divJvecP);
  Eigen::VectorXd prod = (nablaJ.array() * rho1.array()).matrix();
  Eigen::VectorXd func = interaction->params.C1nJ * prod;

  using Operators::integral;

  auto grid = *Grid::getInstance();
  return integral(func, grid);
}

double IterationData::CoulombDirectEnergy(const Grid &grid) {

  double res = 0.0;
  double h = grid.get_h();
  int n = grid.get_n();
  using namespace nuclearConstants;

  if (input.useCoulomb == false)
    return 0.0;

  using Eigen::VectorXd;

  return 0.5 *
         Operators::integral((VectorXd)(rhoP->array() * UCDir.array()), grid);
}

double IterationData::SlaterCoulombEnergy(const Grid &grid) {
  using Eigen::VectorXd;
  using nuclearConstants::e2;
  using Operators::integral;

  if (input.useCoulomb == false)
    return 0.0;

  VectorXd func = rhoP->array().pow(4.0 / 3.0).matrix();

  func *= -3.0 * e2 / 4.0;
  func *= pow(3.0 / M_PI, 1.0 / 3.0);

  return integral(func, grid);
}

double IterationData::Hso() {
  using Operators::dot;

  Eigen::MatrixX3d nablaRho = (*nablaRhoN + *nablaRhoP);
  Eigen::MatrixX3d Jvec = (*JvecN + *JvecP);

  // TODO: rivedere questo calcolo
  //  Eigen::VectorXd func =
  //      0.5 * W0 *
  //      (dot(Jvec, nablaRho) + dot(*JvecP, *nablaRhoP) + dot(*JvecN,
  //      *nablaRhoN));

  // Eigen::VectorXd func = in
  // return Operators::integral(func, grid);
  return 0.0;
}

double IterationData::Hsg() {
  // TODO: rivedere questo calcolo
  if (!input.useJ)
    return 0.0;

  Eigen::MatrixX3d Jvec = (*JvecN + *JvecP);

  using Operators::mod2;

  // Eigen::VectorXd func = -(1.0 / 16.0) * (t1 * x1 + t2 * x2) * mod2(Jvec);
  // func += (1.0 / 16.0) * (t1 - t2) * (mod2(*JvecP) + mod2(*JvecN));

  // return Operators::integral(func, grid);
  return 0.0;
}
