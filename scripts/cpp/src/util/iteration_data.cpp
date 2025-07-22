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

  double energy_SlaterCoulomb = SlaterCoulombEnergy(grid);
  if (std::isnan(energy_SlaterCoulomb)) {
    std::cerr << "Attenzione: SlaterCoulombEnergy ha prodotto un NaN!"
              << std::endl;
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
         energy_C0Tau + energy_C1Tau + energy_SlaterCoulomb +
         energy_CoulombDirect + energy_Hso + energy_Hsg;
}
//
// return C0RhoEnergy(params, grid) + C1RhoEnergy(params, grid) +
//        C0nabla2RhoEnergy(params, grid) + C1nabla2RhoEnergy(params, grid) +
//        C0TauEnergy(params, grid) + C1TauEnergy(params, grid) +
//        SlaterCoulombEnergy(grid) + CoulombDirectEnergy(grid) +
//        Hso(params, grid) + Hsg(params, grid);

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
  // for (int i = 0; i < rhoN->size(); i++) {
  //   if ((*rhoN)(i) < 1e-15 || std::isnan((*rhoN)(i))) {
  //     (*rhoN)(i) = 0.0;
  //   }
  //   if ((*rhoP)(i) < 1e-15 || std::isnan((*rhoP)(i))) {
  //     (*rhoP)(i) = 0.0;
  //   }
  //   if ((*tauN)(i) < 1e-15 || std::isnan((*tauN)(i))) {
  //     (*tauN)(i) = 0.0;
  //   }
  //   if ((*tauP)(i) < 1e-15 || std::isnan((*tauP)(i))) {
  //     (*tauP)(i) = 0.0;
  //   }
  //   if ((*nabla2RhoN)(i) < 1e-15 || std::isnan((*nabla2RhoN)(i))) {
  //     (*nabla2RhoN)(i) = 0.0;
  //   }
  //   if ((*nabla2RhoP)(i) < 1e-15 || std::isnan((*nabla2RhoP)(i))) {
  //     (*nabla2RhoP)(i) = 0.0;
  //   }
  // }

  // for (int j = 0; j < nablaRhoN->cols(); j++) {

  //  for (int i = 0; i < nablaRhoN->rows(); i++) {
  //    if ((*nablaRhoN)(i, j) < 1e-15 || std::isnan((*nablaRhoN)(i, j))) {
  //      (*nablaRhoN)(i, j) = 0.0;
  //    }
  //    if ((*nablaRhoP)(i, j) < 1e-15 || std::isnan((*nablaRhoP)(i, j))) {
  //      (*nablaRhoP)(i, j) = 0.0;
  //    }
  //  }
  //}

  Eigen::VectorXd divJvecN =
      Eigen::VectorXd::Zero(grid.get_total_spatial_points());
  Eigen::VectorXd divJvecP =
      Eigen::VectorXd::Zero(grid.get_total_spatial_points());

  if (input.useJ) {
    JN = std::make_shared<Real2Tensor>(Wavefunction::soDensity(*rhoN, grid));
    JP = std::make_shared<Real2Tensor>(Wavefunction::soDensity(*rhoP, grid));

    for (int j = 0; j < JN->cols(); j++) {
      for (int i = 0; i < JN->rows(); i++) {
        if ((*JN)(i, j) < 1e-25 || std::isnan((*JN)(i, j))) {
          (*JN)(i, j) = 0.0;
        }
        if ((*JP)(i, j) < 1e-25 || std::isnan((*JP)(i, j))) {
          (*JP)(i, j) = 0.0;
        }
      }
    }
    JvecN = std::make_shared<Eigen::MatrixX3d>(Operators::leviCivita(*JN));
    JvecP = std::make_shared<Eigen::MatrixX3d>(Operators::leviCivita(*JP));

    for (int j = 0; j < JvecN->cols(); j++) {
      for (int i = 0; i < JvecN->rows(); i++) {
        if ((*JvecN)(i, j) < 1e-25 || std::isnan((*JvecN)(i, j))) {
          (*JvecN)(i, j) = 0.0;
        }
        if ((*JvecP)(i, j) < 1e-25 || std::isnan((*JvecP)(i, j))) {
          (*JvecP)(i, j) = 0.0;
        }
      }
    }

    divJvecN = Operators::divNoSpin(*JvecN, grid);
    divJvecP = Operators::divNoSpin(*JvecP, grid);
    for (int j = 0; j < divJvecN.size(); j++) {
      if ((divJvecN)(j) < 1e-25 || std::isnan((divJvecN)(j))) {
        (divJvecN)(j) = 0.0;
      }
      if ((divJvecP)(j) < 1e-25 || std::isnan((divJvecP)(j))) {
        (divJvecP)(j) = 0.0;
      }
    }
  } else {
    JvecN = std::make_shared<Eigen::MatrixX3d>(
        Eigen::MatrixX3d::Zero(grid.get_total_spatial_points(), 3));
    JvecP = std::make_shared<Eigen::MatrixX3d>(
        Eigen::MatrixX3d::Zero(grid.get_total_spatial_points(), 3));
  }

  Eigen::VectorXd divJJQN = divJvecN + divJvecN + divJvecP;
  Eigen::VectorXd divJJQP = divJvecP + divJvecN + divJvecP;

  double mu = 0.1;
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
      input.useCoulomb ? Wavefunction::coulombField(*rhoP, grid)
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

    if (input.skyrme.W0 == 0.0) {
      BP = std::make_shared<Eigen::MatrixX3d>(
          Eigen::MatrixX3d::Zero(grid.get_total_spatial_points(), 3));
      BN = std::make_shared<Eigen::MatrixX3d>(
          Eigen::MatrixX3d::Zero(grid.get_total_spatial_points(), 3));
    } else {
      BP = std::make_shared<Eigen::MatrixX3d>(
          params.W0 * 0.5 * (*nablaRhoP + *nablaRhoN + *nablaRhoP) +
          0.125 * ((t1 - t2) * (*JvecP) -
                   (t1 * x1 + t2 * x2) * ((*JvecN) + (*JvecP))));
      BN = std::make_shared<Eigen::MatrixX3d>(
          params.W0 * 0.5 * (*nablaRhoN + *nablaRhoN + *nablaRhoP) +
          0.125 * ((t1 - t2) * (*JvecN) -
                   (t1 * x1 + t2 * x2) * ((*JvecN) + (*JvecP))));
    }
  } else {
    if (input.spinOrbit) {
      auto newBN = params.W0 * 0.5 * (*nablaRhoN + *nablaRhoP + *nablaRhoP);
      auto newBP = params.W0 * 0.5 * (*nablaRhoN + *nablaRhoP + *nablaRhoN);
      *BP = (*BP) * (1 - mu) + newBP * mu;
      *BN = (*BN) * (1 - mu) + newBN * mu;
    }
  }
  std::cout << "Quantities updated" << std::endl << std::endl;
}
