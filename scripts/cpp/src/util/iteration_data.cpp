#include "util/iteration_data.hpp"
#include "Eigen/src/Core/Matrix.h"
#include "constants.hpp"
#include "operators/common_operators.hpp"
#include "operators/differential_operators.hpp"
#include "operators/integral_operators.hpp"
#include "util/effective_mass.hpp"
#include "util/wavefunction.hpp"
#include <chrono>
#include <iostream>
#include <memory>
IterationData::IterationData(InputParser input) : input(input) {
  params = input.skyrme;
  int A = input.getA();
  using nuclearConstants::m;

  massCorr = m * ((double)A) / ((double)(A - 1));
}

double IterationData::totalEnergyIntegral(SkyrmeParameters params,
                                          const Grid &grid) {
  return C0RhoEnergy(params, grid) + C1RhoEnergy(params, grid) +
         C0nabla2RhoEnergy(params, grid) + C1nabla2RhoEnergy(params, grid) +
         C0TauEnergy(params, grid) + C1TauEnergy(params, grid) +
         SlaterCoulombEnergy(grid) + CoulombDirectEnergy(grid);
}

double IterationData::HFEnergy(double SPE, const Grid &grid) {

  return totalEnergyIntegral(input.skyrme, grid) -
         0.5 * densityUVPIntegral(grid) +
         -0.5 * kineticEnergyEff(input.skyrme, grid) +
         +kineticEnergy(input.skyrme, grid) + 0.5 * SPE;
}

double IterationData::densityUVPIntegral(const Grid &grid) {
  Eigen::VectorXd vecN = rhoN->array() * UN->array();
  Eigen::VectorXd vecP = rhoP->array() * UP->array();
  Eigen::VectorXd coul = rhoP->array() * UCoul->array();
  Eigen::VectorXd vec = vecN + vecP + coul;

  return Operators::integral(vec, grid);
}

double IterationData::kineticEnergyEff(SkyrmeParameters params,
                                       const Grid &grid) {
  double t0 = params.t0;
  double t3 = params.t3;
  double sigma = params.sigma;

  using Eigen::VectorXd;
  using nuclearConstants::h_bar;
  using nuclearConstants::m;
  using Operators::integral;

  // VectorXd tN = 0.5 * h_bar * h_bar * (*tauN) / (massCorr);
  // VectorXd tP = 0.5 * h_bar * h_bar * (*tauP) / (massCorr);
  VectorXd tN = massN->vector.array() * tauN->array();
  VectorXd tP = massP->vector.array() * tauP->array();
  double kineticEnergy = integral((VectorXd)(tN.matrix() + tP.matrix()), grid);

  return kineticEnergy;
}

double IterationData::kineticEnergy(SkyrmeParameters params, const Grid &grid) {
  double t0 = params.t0;
  double t3 = params.t3;
  double sigma = params.sigma;

  using Eigen::VectorXd;
  using nuclearConstants::h_bar;
  using nuclearConstants::m;
  using Operators::integral;

  VectorXd tN = 0.5 * h_bar * h_bar * (*tauN) / (massCorr);
  VectorXd tP = 0.5 * h_bar * h_bar * (*tauP) / (massCorr);
  double kineticEnergy = integral((VectorXd)(tN.matrix() + tP.matrix()), grid);

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

  tauN = std::make_shared<Eigen::VectorXd>(
      Wavefunction::kineticDensity(neutrons, grid));
  tauP = std::make_shared<Eigen::VectorXd>(
      Wavefunction::kineticDensity(protons, grid));

  nabla2RhoN =
      std::make_shared<Eigen::VectorXd>(Operators::lapNoSpin(*rhoN, grid));
  nabla2RhoP =
      std::make_shared<Eigen::VectorXd>(Operators::lapNoSpin(*rhoP, grid));

  Eigen::VectorXd divJvecN =
      Eigen::VectorXd::Zero(grid.get_total_spatial_points());
  Eigen::VectorXd divJvecP =
      Eigen::VectorXd::Zero(grid.get_total_spatial_points());

  if (input.skyrme.W0 != 0.0) {
    JN = std::make_shared<Real2Tensor>(Wavefunction::soDensity(*rhoN, grid));
    JP = std::make_shared<Real2Tensor>(Wavefunction::soDensity(*rhoP, grid));

    JvecN = std::make_shared<Eigen::MatrixX3d>(Operators::leviCivita(*JN));
    JvecP = std::make_shared<Eigen::MatrixX3d>(Operators::leviCivita(*JP));

    divJvecN = Operators::divNoSpin(*JvecN, grid);
    divJvecP = Operators::divNoSpin(*JvecP, grid);
  }

  Eigen::VectorXd divJJQN = divJvecN + divJvecN + divJvecP;
  Eigen::VectorXd divJJQP = divJvecP + divJvecN + divJvecP;

  double mu = 0.2;
  Eigen::VectorXd tau = *tauN + *tauP;
  Eigen::VectorXd nabla2rho = *nabla2RhoN + *nabla2RhoP;

  EffectiveMass newMassN(grid, rho, *rhoN, nablaRho, *nablaRhoN, massCorr,
                         params);
  EffectiveMass newMassP(grid, rho, *rhoP, nablaRho, *nablaRhoP, massCorr,
                         params);

  Eigen::VectorXd newFieldN = Wavefunction::field(
      rho, *rhoN, tau, *tauN, nabla2rho, *nabla2RhoN, divJJQN, grid, params);
  Eigen::VectorXd newFieldP = Wavefunction::field(
      rho, *rhoP, tau, *tauP, nabla2rho, *nabla2RhoP, divJJQP, grid, params);

  Eigen::VectorXd newFieldCoul =
      input.useCoulomb ? Wavefunction::coulombField(rho, grid)
                       : Eigen::VectorXd::Zero(grid.get_total_spatial_points());

  if (massN == nullptr) {
    massN = std::make_shared<EffectiveMass>(newMassN);
    massP = std::make_shared<EffectiveMass>(newMassP);
  } else {
    std::cout << "Mass mixing" << std::endl;
    massN->vector = (massN->vector) * (1 - mu) + newMassN.vector * mu;
    massP->vector = (massP->vector) * (1 - mu) + newMassP.vector * mu;
    massN->gradient = (massN->gradient) * (1 - mu) + newMassN.gradient * mu;
    massP->gradient = (massP->gradient) * (1 - mu) + newMassP.gradient * mu;
  }
  if (UP == nullptr) {
    UP = std::make_shared<Eigen::VectorXd>(newFieldP);
    UN = std::make_shared<Eigen::VectorXd>(newFieldN);
  } else {
    *UN = (*UN) * (1 - mu) + newFieldN * mu;
    *UP = (*UP) * (1 - mu) + newFieldP * mu;
  }

  if (UCoul == nullptr) {
    UCoul = std::make_shared<Eigen::VectorXd>(newFieldCoul);
  } else {
    *UCoul = (*UCoul) * (1 - mu) + newFieldCoul * mu;
  }

  double t1 = params.t1, t2 = params.t2;
  double x1 = params.x1, x2 = params.x2;
  if (BP == nullptr) {
    BP = std::make_shared<Eigen::MatrixX3d>(
        params.W0 * 0.5 * (*nablaRhoP + *nablaRhoN + *nablaRhoP) +
        0.125 * ((t1 - t2) * (*JvecP) -
                 (t1 * x1 + t2 * x2) * ((*JvecN) + (*JvecP))));
    BN = std::make_shared<Eigen::MatrixX3d>(
        params.W0 * 0.5 * (*nablaRhoN + *nablaRhoN + *nablaRhoP) +
        0.125 * ((t1 - t2) * (*JvecN) -
                 (t1 * x1 + t2 * x2) * ((*JvecN) + (*JvecP))));
  } else {
    auto newBN = params.W0 * 0.5 * (*nablaRhoN + *nablaRhoP + *nablaRhoP);
    auto newBP = params.W0 * 0.5 * (*nablaRhoN + *nablaRhoP + *nablaRhoN);
    *BP = (*BP) * (1 - mu) + newBP * mu;
    *BN = (*BN) * (1 - mu) + newBN * mu;
  }
  std::cout << "Quantities updated" << std::endl;
  // start = std::chrono::steady_clock::now();
}
