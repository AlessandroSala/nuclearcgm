#include "util/iteration_data.hpp"
#include "Eigen/src/Core/Matrix.h"
#include "constants.hpp"
#include "operators/differential_operators.hpp"
#include "operators/integral_operators.hpp"
#include "util/wavefunction.hpp"
#include <chrono>
#include <iostream>
#include <memory>
IterationData::IterationData(SkyrmeParameters params) : params(params) {}

double IterationData::totalEnergyIntegral(SkyrmeParameters params,
                                          const Grid &grid) {
  return C0RhoEnergy(params, grid) + C1RhoEnergy(params, grid) +
         C0nabla2RhoEnergy(params, grid) + C1nabla2RhoEnergy(params, grid) +
         C0TauEnergy(params, grid) + C1TauEnergy(params, grid);
}

double IterationData::densityUVPIntegral(const Grid &grid) {
  Eigen::VectorXd vecN = rhoN->array() * UN->array();
  Eigen::VectorXd vecP = rhoP->array() * UP->array();
  Eigen::VectorXd vec = vecN + vecP;
  return Operators::integral(vec, grid);
}

double IterationData::kineticEnergy(SkyrmeParameters params, const Grid &grid) {
  double t0 = params.t0;
  double t3 = params.t3;
  double sigma = params.sigma;

  using Eigen::VectorXd;
  using nuclearConstants::h_bar;
  using nuclearConstants::m;
  using Operators::integral;
  // VectorXd tN = massN->vector;
  // VectorXd tP = massP->vector;
  VectorXd tN = 0.5 * h_bar * h_bar * (*tauN) / m;
  VectorXd tP = 0.5 * h_bar * h_bar * (*tauP) / m;
  double kineticEnergy = integral((VectorXd)(tN + tP), grid);

  return kineticEnergy;
}

void IterationData::updateQuantities(const Eigen::MatrixXcd &neutronsShells,
                                     const Eigen::MatrixXcd &protonsShells,
                                     int A, int Z, const Grid &grid) {
  int N = A - Z;
  auto neutrons = neutronsShells(Eigen::all, Eigen::seq(0, N - 1));
  auto protons = protonsShells(Eigen::all, Eigen::seq(0, Z - 1));
  std::cout << neutrons.cols() << " " << protons.cols() << std::endl;

  rhoN =
      std::make_shared<Eigen::VectorXd>(Wavefunction::density(neutrons, grid));
  rhoP =
      std::make_shared<Eigen::VectorXd>(Wavefunction::density(protons, grid));
  Eigen::VectorXd rho = (*rhoN) + (*rhoP);

  nablaRhoN =
      std::make_shared<Eigen::MatrixX3d>(Operators::gradNoSpin(*rhoN, grid));
  nablaRhoP =
      std::make_shared<Eigen::MatrixX3d>(Operators::gradNoSpin(*rhoP, grid));

  Eigen::MatrixX3d nablaRho = (*nablaRhoN) + (*nablaRhoP);

  massN = std::make_shared<EffectiveMass>(
      EffectiveMass(grid, rho, *rhoN, nablaRho, *nablaRhoN, params));

  massP = std::make_shared<EffectiveMass>(
      EffectiveMass(grid, rho, *rhoP, nablaRho, *nablaRhoP, params));

  auto start = std::chrono::steady_clock::now();
  tauN = std::make_shared<Eigen::VectorXd>(
      Wavefunction::kineticDensity(neutrons, grid));
  tauP = std::make_shared<Eigen::VectorXd>(
      Wavefunction::kineticDensity(protons, grid));

  auto end = std::chrono::steady_clock::now();
  std::cout << "Time elapsed tau "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                     start)
                   .count()
            << " ms" << std::endl;

  std::cout << "Time elapsed grad "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                     start)
                   .count()
            << " ms" << std::endl;
  start = std::chrono::steady_clock::now();

  // nabla2RhoN =
  //     std::make_shared<Eigen::VectorXd>(Operators::divNoSpin(*nablaRhoN,
  //     grid));
  // nabla2RhoP =
  //     std::make_shared<Eigen::VectorXd>(Operators::divNoSpin(*nablaRhoP,
  //     grid));
  nabla2RhoN =
      std::make_shared<Eigen::VectorXd>(Operators::lapNoSpin(*rhoN, grid));
  nabla2RhoP =
      std::make_shared<Eigen::VectorXd>(Operators::lapNoSpin(*rhoP, grid));

  end = std::chrono::steady_clock::now();
  std::cout << "Time elapsed nabla 2 rho "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                     start)
                   .count()
            << " ms" << std::endl;

  // start = std::chrono::steady_clock::now();
  // JN = std::make_shared<RealDoubleTensor>(
  //     Wavefunction::soDensity(neutrons, grid));
  // JP = std::make_shared<RealDoubleTensor>(
  //     Wavefunction::soDensity(protons, grid));
  // end = std::chrono::steady_clock::now();
  // std::cout << "Time elapsed J "
  //           << std::chrono::duration_cast<std::chrono::milliseconds>(end
  //           -
  //                                                                    start)
  //                  .count()
  //           << " ms" << std::endl;

  // divJN = std::make_shared<Eigen::VectorXcd>(Operators::divNoSpin(*JN,
  // grid)); divJP =
  // std::make_shared<Eigen::VectorXcd>(Operators::divNoSpin(*JP, grid));

  double mu = 0.2;
  Eigen::VectorXd tau = *tauN + *tauP;
  Eigen::VectorXd nabla2rho = *nabla2RhoN + *nabla2RhoP;
  Eigen::VectorXd divJJQ =
      Eigen::VectorXd::Zero(grid.get_total_spatial_points());

  // TODO: fix Q & PN
  Eigen::VectorXd newFieldN = Wavefunction::field(
      rho, *rhoN, tau, *tauN, nabla2rho, *nabla2RhoN, divJJQ, grid, params);
  Eigen::VectorXd newFieldP = Wavefunction::field(
      rho, *rhoP, tau, *tauP, nabla2rho, *nabla2RhoP, divJJQ, grid, params);

  if (UP == nullptr) {
    UP = std::make_shared<Eigen::VectorXd>(newFieldP);
    UN = std::make_shared<Eigen::VectorXd>(newFieldN);
  } else {
    *UN = (*UN) * (1 - mu) + newFieldN * mu;
    *UP = (*UP) * (1 - mu) + newFieldP * mu;
  }

  if (BP == nullptr) {
    BP = std::make_shared<Eigen::MatrixX3d>(
        params.W0 * 0.5 * (*nablaRhoN + *nablaRhoP + *nablaRhoP));
    BN = std::make_shared<Eigen::MatrixX3d>(
        params.W0 * 0.5 * (*nablaRhoN + *nablaRhoP + *nablaRhoN));
  } else {
    auto newBN = params.W0 * 0.5 * (*nablaRhoN + *nablaRhoP + *nablaRhoP);
    auto newBP = params.W0 * 0.5 * (*nablaRhoN + *nablaRhoP + *nablaRhoN);
    BP = std::make_shared<Eigen::MatrixX3d>((*BP) * (1 - mu) + newBP * mu);
    BN = std::make_shared<Eigen::MatrixX3d>((*BN) * (1 - mu) + newBN * mu);
  }
  // start = std::chrono::steady_clock::now();
}
