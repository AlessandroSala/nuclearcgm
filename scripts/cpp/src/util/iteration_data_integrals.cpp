#include "constants.hpp"
#include <iostream>
#include "operators/common_operators.hpp"
#include "operators/integral_operators.hpp"
#include "util/iteration_data.hpp"
#include <cmath>
#include "spherical_harmonics.hpp"

double IterationData::betaRealRadius()
{

  Eigen::VectorXd rho = *rhoP + *rhoN;
  int A = input.getA();
  double a20 = SphericalHarmonics::Q(2, 0, rho).real();
  double beta = 4 * M_PI * a20 / (5 * A * radius());

  if (axis2Exp('x') > axis2Exp('z'))
  {
    beta = -beta;
  }

  return beta;
}
QuadrupoleDeformation IterationData::quadrupoleDeformation()
{

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
  std::cout << "beta: " << beta << ", betaSim: " << betaSim << std::endl;

  if (axis2Exp('x') > axis2Exp('z'))
  {
    beta = -beta;
  }

  return {beta, gamma};
}

double IterationData::axis2Exp(char dir)
{

  Eigen::VectorXd rho = *rhoP + *rhoN;
  auto grid = *Grid::getInstance();
  int n = grid.get_n();
  double res = 0.0;
  for (int i = 0; i < grid.get_n(); ++i)
  {
    for (int j = 0; j < grid.get_n(); ++j)
    {
      for (int k = 0; k < grid.get_n(); ++k)
      {
        int idx = grid.idxNoSpin(i, j, k);
        double ii = grid.get_xs()[i];
        double jj = grid.get_ys()[j];
        double kk = grid.get_zs()[k];
        if (dir == 'x')
        {
          res += ii * ii * rho(idx);
        }
        else if (dir == 'y')
        {
          res += jj * jj * rho(idx);
        }
        else if (dir == 'z')
        {
          res += kk * kk * rho(idx);
        }
      }
    }
  }
  return res / rho.sum();
}

double IterationData::C0RhoEnergy(SkyrmeParameters params, const Grid &grid)
{

  double t0 = params.t0;
  double t3 = params.t3;
  double x0 = params.x0;
  double x3 = params.x3;
  double sigma = params.sigma;

  Eigen::VectorXd ones = Eigen::VectorXd::Ones(grid.get_total_spatial_points());

  Eigen::VectorXd rho0 = *rhoN + *rhoP;

  Eigen::VectorXd energy0 =
      (((3.0 / 8.0) * t0 * ones +
        (3.0 / 48.0) * t3 * rho0.array().pow(sigma).matrix()) *
       rho0.array().pow(2).matrix().transpose())
          .diagonal();

  using Operators::integral;
  return integral(Eigen::VectorXd(energy0), grid);
}

double IterationData::C1RhoEnergy(SkyrmeParameters params, const Grid &grid)
{
  Eigen::VectorXd energy1 =
      Eigen::VectorXd::Zero(grid.get_total_spatial_points());
  double t0 = params.t0;
  double t3 = params.t3;
  double x0 = params.x0;
  double x3 = params.x3;
  double sigma = params.sigma;
  Eigen::VectorXd rho0 = *rhoN + *rhoP;
  Eigen::VectorXd rho1 = *rhoN - *rhoP;
  using Operators::integral;
  Eigen::VectorXd ones = Eigen::VectorXd::Ones(grid.get_total_spatial_points());
  energy1 +=
      ((-(1.0 / 4.0) * t0 * (0.5 + x0) * ones -
        (1.0 / 24.0) * t3 * (0.5 + x3) * rho0.array().pow(sigma).matrix()) *
       rho1.array().pow(2).matrix().transpose())
          .diagonal();
  return integral(energy1, grid);
}

double IterationData::C0TauEnergy(SkyrmeParameters params, const Grid &grid)
{

  double t1 = params.t1;
  double t2 = params.t2;
  double x2 = params.x2;

  Eigen::VectorXd tau0 = *tauN + *tauP;
  Eigen::VectorXd rho0 = *rhoN + *rhoP;

  Eigen::VectorXd prod = (tau0 * rho0.transpose()).diagonal();

  Eigen::VectorXd energy0c =
      +(1.0 / 16.0) * (3.0 * t1 + t2 * (5.0 + 4 * x2)) * prod;

  using Operators::integral;
  return integral(energy0c, grid);
}

double IterationData::C1TauEnergy(SkyrmeParameters params, const Grid &grid)
{

  double t1 = params.t1;
  double t2 = params.t2;
  double x1 = params.x1;
  double x2 = params.x2;

  Eigen::VectorXd tau1 = *tauN - *tauP;
  Eigen::VectorXd rho1 = *rhoN - *rhoP;

  Eigen::VectorXd prod = (tau1 * rho1.transpose()).diagonal();

  Eigen::VectorXd energy1c =
      (-0.125 * t1 * (0.5 + x1) + 0.125 * t2 * (0.5 + x2)) * prod;

  using Operators::integral;
  return integral(energy1c, grid);
}

// SUS !
double IterationData::C0nabla2RhoEnergy(SkyrmeParameters params,
                                        const Grid &grid)
{
  double t1 = params.t1;
  double t2 = params.t2;
  double x1 = params.x1;
  double x2 = params.x2;

  Eigen::VectorXd nabla2Rho0 = *nabla2RhoN + *nabla2RhoP;
  Eigen::VectorXd rho0 = *rhoN + *rhoP;

  Eigen::VectorXd prod = (rho0 * nabla2Rho0.transpose()).diagonal();
  Eigen::MatrixX3d nablaRho = *nablaRhoN + *nablaRhoP;
  Eigen::VectorXd nablaRhoSq = (nablaRho * nablaRho.adjoint()).diagonal();
  Eigen::VectorXd nablaRhoNSq =
      (*nablaRhoN * nablaRhoN->transpose()).diagonal();
  Eigen::VectorXd nablaRhoPSq =
      (*nablaRhoP * nablaRhoP->transpose()).diagonal();

  Eigen::VectorXd energy0c =
      ((-(9.0 / 64.0) * t1 + (1.0 / 16.0) * t2 * (1.25 + x2))) * prod;
  // Eigen::VectorXd energy0c =
  // -((1.0 / 64.0) * (9 * t1 - 5 * t2 - 4 * t2 * x2)) * prod;
  // Eigen::VectorXd energy0c =
  //     (1.0 / 32.0) * (3 * t1 * (2 + x1) - t2 * (2 + x2)) * nablaRhoSq;
  // energy0c += -(1.0 / 32.0) * (3 * t1 * (2 * x1 + 1) + t2 * (2 * x2 + 1)) *
  //             (nablaRhoNSq + nablaRhoPSq);

  using Operators::integral;
  return integral(energy0c, grid);
}

double IterationData::C1nabla2RhoEnergy(SkyrmeParameters params,
                                        const Grid &grid)
{
  double t1 = params.t1;
  double t2 = params.t2;
  double x1 = params.x1;
  double x2 = params.x2;

  Eigen::VectorXd nabla2Rho1 = *nabla2RhoN - *nabla2RhoP;
  Eigen::VectorXd rho1 = *rhoN - *rhoP;
  Eigen::VectorXd prod = (nabla2Rho1 * rho1.transpose()).diagonal();

  Eigen::VectorXd energy1c =
      ((3.0 / 32.0) * t1 * (0.5 + x1) + (1.0 / 32.0) * t2 * (0.5 + x2)) * prod;

  using Operators::integral;
  return integral(energy1c, grid);
}

double IterationData::CoulombDirectEnergy(const Grid &grid)
{

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

double IterationData::SlaterCoulombEnergy(const Grid &grid)
{
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

double IterationData::Hso(SkyrmeParameters params, const Grid &grid)
{
  double W0 = params.W0;
  using Operators::dot;

  Eigen::MatrixX3d nablaRho = (*nablaRhoN + *nablaRhoP);
  Eigen::MatrixX3d Jvec = (*JvecN + *JvecP);

  Eigen::VectorXd func =
      0.5 * W0 *
      (dot(Jvec, nablaRho) + dot(*JvecP, *nablaRhoP) + dot(*JvecN, *nablaRhoN));

  return Operators::integral(func, grid);
}

double IterationData::Hsg(SkyrmeParameters params, const Grid &grid)
{
  if (!input.useJ)
    return 0.0;
  double t0 = params.t0, t1 = params.t1, t2 = params.t2;
  double x1 = params.x1, x2 = params.x2;

  Eigen::MatrixX3d Jvec = (*JvecN + *JvecP);

  using Operators::mod2;

  Eigen::VectorXd func = -(1.0 / 16.0) * (t1 * x1 + t2 * x2) * mod2(Jvec);
  func += (1.0 / 16.0) * (t1 - t2) * (mod2(*JvecP) + mod2(*JvecN));

  return Operators::integral(func, grid);
}
