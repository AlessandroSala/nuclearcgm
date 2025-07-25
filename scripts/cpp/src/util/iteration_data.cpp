#include "util/iteration_data.hpp"
#include "Eigen/src/Core/Matrix.h"
#include "constants.hpp"
#include "operators/common_operators.hpp"
#include "operators/differential_operators.hpp"
#include "operators/integral_operators.hpp"
#include "util/effective_mass.hpp"
#include "util/wavefunction.hpp"
#include <chrono>
#include <cmath>
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
  double energy_C0Rho = C0RhoEnergy(params, grid);
  if (std::isnan(energy_C0Rho)) {
    std::cerr << "Attenzione: C0RhoEnergy ha prodotto un NaN!" << std::endl;
  }

  double energy_C1Rho = C1RhoEnergy(params, grid);
  if (std::isnan(energy_C1Rho)) {
    std::cerr << "Attenzione: C1RhoEnergy ha prodotto un NaN!" << std::endl;
  }

  double energy_C0nabla2Rho = C0nabla2RhoEnergy(params, grid);
  if (std::isnan(energy_C0nabla2Rho)) {
    std::cerr << "Attenzione: C0nabla2RhoEnergy ha prodotto un NaN!"
              << std::endl;
  }

  double energy_C1nabla2Rho = C1nabla2RhoEnergy(params, grid);
  if (std::isnan(energy_C1nabla2Rho)) {
    std::cerr << "Attenzione: C1nabla2RhoEnergy ha prodotto un NaN!"
              << std::endl;
  }

  double energy_C0Tau = C0TauEnergy(params, grid);
  if (std::isnan(energy_C0Tau)) {
    std::cerr << "Attenzione: C0TauEnergy ha prodotto un NaN!" << std::endl;
  }

  double energy_C1Tau = C1TauEnergy(params, grid);
  if (std::isnan(energy_C1Tau)) {
    std::cerr << "Attenzione: C1TauEnergy ha prodotto un NaN!" << std::endl;
  }

  double energy_CoulombDirect = CoulombDirectEnergy(grid);
  if (std::isnan(energy_CoulombDirect)) {
    std::cerr << "Attenzione: CoulombDirectEnergy ha prodotto un NaN!"
              << std::endl;
  }

  double energy_Hso = input.spinOrbit ? Hso(params, grid) : 0.0;
  if (std::isnan(energy_Hso)) {
    std::cerr << "Attenzione: Hso ha prodotto un NaN!" << std::endl;
  }

  double energy_Hsg = Hsg(params, grid);
  if (std::isnan(energy_Hsg)) {
    std::cerr << "Attenzione: Hsg ha prodotto un NaN!" << std::endl;
  }

  return energy_C0Rho + energy_C1Rho + energy_C0nabla2Rho + energy_C1nabla2Rho +
         energy_C0Tau + energy_C1Tau + energy_CoulombDirect +
         0.5 * SlaterCoulombEnergy(grid) + energy_Hso + energy_Hsg;
}

//
// return C0RhoEnergy(params, grid) + C1RhoEnergy(params, grid) +
//        C0nabla2RhoEnergy(params, grid) + C1nabla2RhoEnergy(params, grid) +
//        C0TauEnergy(params, grid) + C1TauEnergy(params, grid) +
//        SlaterCoulombEnergy(grid) + CoulombDirectEnergy(grid) +
//        Hso(params, grid) + Hsg(params, grid);

double IterationData::Erear(const Grid &grid) {

  double t0 = params.t0;
  double t3 = params.t3;
  double x0 = params.x0;
  double x3 = params.x3;
  double sigma = params.sigma;

  Eigen::VectorXd ones = Eigen::VectorXd::Ones(grid.get_total_spatial_points());

  Eigen::VectorXd rho0 = *rhoN + *rhoP;
  Eigen::VectorXd rho1 = *rhoN - *rhoP;

  Eigen::VectorXd energy1 =
      sigma * rho1.array().pow(sigma) *
      ((1.0 / 12.0) * t3 * (1 + x3 / 2.0) * rho0.array().pow(2) -
       (1.0 / 12.0) * t3 * (0.5 + x3) *
           (rhoN->array().pow(2) + rhoP->array().pow(2)));
  if (energy1.array().isNaN().any()) {
    energy1 = Eigen::VectorXd::Zero(grid.get_total_spatial_points());
  }

  Eigen::VectorXd energy0 =
      sigma * rho0.array().pow(sigma) *
      ((1.0 / 12.0) * t3 * (1 + x3 / 2.0) * rho0.array().pow(2) -
       (1.0 / 12.0) * t3 * (0.5 + x3) *
           (rhoN->array().pow(2) + rhoP->array().pow(2)));

  using Operators::integral;
  return integral(Eigen::VectorXd(energy0 + energy1), grid);
}
double IterationData::HFEnergy(double SPE, const Grid &grid) {

  return 0.5 * (SPE + kineticEnergy(params, grid)) - 0.5 * Erear(grid) +
         0.5 * SlaterCoulombEnergy(grid);

  return 0.5 * Erear(grid) - 0.5 * densityUVPIntegral(grid) +
         +0.5 * (SPE + kineticEnergy(params, grid));

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
                                     Eigen::VectorXd &vksN,
                                     Eigen::VectorXd &vksP, const Grid &grid) {
  int A = input.getA();
  int Z = input.getZ();
  int N = A - Z;

  Eigen::MatrixXcd neutrons, protons;
  if (input.pairing) {
    std::cout << vksN << std::endl;
    neutrons = Eigen::MatrixXcd(neutronsShells.rows(), neutronsShells.cols());
    protons = Eigen::MatrixXcd(protonsShells.rows(), protonsShells.cols());

    for (int i = 0; i < neutronsShells.cols(); ++i) {
      neutrons.col(i) = neutronsShells.col(i) * vksN(i);
      protons.col(i) = protonsShells.col(i) * vksP(i);
    }

  } else {

    neutrons = neutronsShells(Eigen::all, Eigen::seq(0, N - 1));
    protons = protonsShells(Eigen::all, Eigen::seq(0, Z - 1));
  }
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

  using std::isnan;

  Eigen::VectorXd divJvecN =
      Eigen::VectorXd::Zero(grid.get_total_spatial_points());
  Eigen::VectorXd divJvecP =
      Eigen::VectorXd::Zero(grid.get_total_spatial_points());
  JvecN = std::make_shared<Eigen::MatrixX3d>(
      Eigen::MatrixX3d::Zero(grid.get_total_spatial_points(), 3));
  JvecP = std::make_shared<Eigen::MatrixX3d>(
      Eigen::MatrixX3d::Zero(grid.get_total_spatial_points(), 3));

  // if (input.useJ) {
  JN = std::make_shared<Real2Tensor>(Wavefunction::soDensity(neutrons, grid));
  JP = std::make_shared<Real2Tensor>(Wavefunction::soDensity(protons, grid));

  for (int j = 0; j < JN->cols(); j++) {
    for (int i = 0; i < JN->rows(); i++) {
      if (std::isnan((*JN)(i, j))) {
        (*JN)(i, j) = 0.0;
      }
      if (std::isnan((*JP)(i, j))) {
        (*JP)(i, j) = 0.0;
      }
    }
  }
  JvecN = std::make_shared<Eigen::MatrixX3d>(Operators::leviCivita(*JN));
  JvecP = std::make_shared<Eigen::MatrixX3d>(Operators::leviCivita(*JP));

  for (int j = 0; j < JvecN->cols(); j++) {
    for (int i = 0; i < JvecN->rows(); i++) {
      if (std::isnan((*JvecN)(i, j))) {
        (*JvecN)(i, j) = 0.0;
      }
      if (std::isnan((*JvecP)(i, j))) {
        (*JvecP)(i, j) = 0.0;
      }
    }
  }

  // divJvecN = Operators::divNoSpin(*JvecN, grid);
  // divJvecP = Operators::divNoSpin(*JvecP, grid);
  divJvecN = Wavefunction::divJ(neutrons, grid);
  divJvecP = Wavefunction::divJ(protons, grid);
  for (int j = 0; j < divJvecN.size(); j++) {
    if (std::isnan((divJvecN)(j))) {
      (divJvecN)(j) = 0.0;
    }
    if (std::isnan((divJvecP)(j))) {
      (divJvecP)(j) = 0.0;
    }
  }
  //  } else {
  //    JvecN = std::make_shared<Eigen::MatrixX3d>(
  //        Eigen::MatrixX3d::Zero(grid.get_total_spatial_points(), 3));
  //    JvecP = std::make_shared<Eigen::MatrixX3d>(
  //        Eigen::MatrixX3d::Zero(grid.get_total_spatial_points(), 3));
  //  }

  Eigen::VectorXd divJJQN;
  Eigen::VectorXd divJJQP;
  if (input.spinOrbit) {
    divJJQN = divJvecN + divJvecN + divJvecP;
    divJJQP = divJvecP + divJvecN + divJvecP;
  } else {
    divJJQN = Eigen::VectorXd::Zero(grid.get_total_spatial_points());
    divJJQP = Eigen::VectorXd::Zero(grid.get_total_spatial_points());
  }

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

  Eigen::VectorXd newFieldCoul;
  if (input.useCoulomb) {
    newFieldCoul = Wavefunction::coulombField(*rhoP, grid);
    newFieldCoul += Wavefunction::exchangeCoulombField(*rhoP, grid);
  } else {
    newFieldCoul = Eigen::VectorXd::Zero(grid.get_total_spatial_points());
  }

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

  Eigen::MatrixX3d newBN =
      Eigen::MatrixX3d::Zero(grid.get_total_spatial_points(), 3);
  Eigen::MatrixX3d newBP =
      Eigen::MatrixX3d::Zero(grid.get_total_spatial_points(), 3);

  if (input.spinOrbit) {
    newBN += params.W0 * 0.5 * (*nablaRhoN + *nablaRhoN + *nablaRhoP);
    newBP += params.W0 * 0.5 * (*nablaRhoN + *nablaRhoP + *nablaRhoP);
  }
  if (input.useJ) {
    newBN += 0.125 * ((t1 - t2) * (*JvecN) -
                      (t1 * x1 + t2 * x2) * ((*JvecN) + (*JvecP)));
    newBP += 0.125 * ((t1 - t2) * (*JvecP) -
                      (t1 * x1 + t2 * x2) * ((*JvecN) + (*JvecP)));
  }
  if (BN == nullptr) {
    BN = std::make_shared<Eigen::MatrixX3d>(newBN);
    BP = std::make_shared<Eigen::MatrixX3d>(newBP);
  } else {
    *BN = (*BN) * (1 - mu) + newBN * mu;
    *BP = (*BP) * (1 - mu) + newBP * mu;
  }

  //  if (BP == nullptr) {
  //
  //    if (!input.spinOrbit) {
  //      BP = std::make_shared<Eigen::MatrixX3d>(
  //          Eigen::MatrixX3d::Zero(grid.get_total_spatial_points(), 3));
  //      BN = std::make_shared<Eigen::MatrixX3d>(
  //          Eigen::MatrixX3d::Zero(grid.get_total_spatial_points(), 3));
  //    } else {
  //      BP = std::make_shared<Eigen::MatrixX3d>(
  //          params.W0 * 0.5 * (*nablaRhoP + *nablaRhoN + *nablaRhoP));
  //      if (input.useJ)
  //        *BP += 0.125 * ((t1 - t2) * (*JvecP) -
  //                        (t1 * x1 + t2 * x2) * ((*JvecN) + (*JvecP)));
  //
  //      BN = std::make_shared<Eigen::MatrixX3d>(
  //          params.W0 * 0.5 * (*nablaRhoN + *nablaRhoN + *nablaRhoP));
  //      if (input.useJ)
  //        *BN += 0.125 * ((t1 - t2) * (*JvecN) -
  //                        (t1 * x1 + t2 * x2) * ((*JvecN) + (*JvecP)));
  //    }
  //  } else {
  //    if (input.spinOrbit) {
  //      // TODO: fixare con J2!!
  //      auto newBN = params.W0 * 0.5 * (*nablaRhoN + *nablaRhoP + *nablaRhoP);
  //      auto newBP = params.W0 * 0.5 * (*nablaRhoN + *nablaRhoP + *nablaRhoN);
  //      *BP = (*BP) * (1 - mu) + newBP * mu;
  //      *BN = (*BN) * (1 - mu) + newBN * mu;
  //    }
  //  }
  std::cout << "Quantities updated" << std::endl << std::endl;
}
