#include "util/iteration_data.hpp"
#include "Eigen/src/Core/Matrix.h"
#include "constants.hpp"
#include "operators/differential_operators.hpp"
#include "operators/integral_operators.hpp"
#include "util/wavefunction.hpp"
#include <chrono>
#include <iostream>
IterationData::IterationData(SkyrmeParameters params) : params(params) {}

double IterationData::totalEnergy(SkyrmeParameters params, const Grid &grid) {
  Eigen::VectorXd energy0 =
      Eigen::VectorXd::Zero(grid.get_total_spatial_points());
  Eigen::VectorXd energy1 =
      Eigen::VectorXd::Zero(grid.get_total_spatial_points());
  double t0 = params.t0;
  double t3 = params.t3;
  double sigma = params.sigma;
  Eigen::VectorXd rho0 = *rhoN + *rhoP;
  Eigen::VectorXd rho1 = *rhoN - *rhoP;
  using Operators::integral;
  Eigen::VectorXd ones = Eigen::VectorXd::Ones(grid.get_total_spatial_points());
  energy0 += (((3.0 / 8.0) * t0 * ones +
               (3.0 / 48.0) * t3 * rho0.array().pow(sigma).matrix()) *
              rho0.array().pow(2).matrix().transpose())
                 .diagonal();
  energy1 += (-(1.0 / 4.0) * t0 * (0.5) * ones -
              (1.0 / 24.0) * t3 * (0.5) * rho0.array().pow(sigma).matrix()) *
             rho1.array().pow(2).matrix().transpose().diagonal();

  using nuclearConstants::h_bar;
  using nuclearConstants::m;
  Eigen::VectorXd tau = (*tauN + *tauP) * (pow(h_bar, 2) / (m * 2));
  double kineticEnergy = integral(tau, grid);

  std::cout << "Kinetic energy: " << kineticEnergy << std::endl;

  Eigen::VectorXd energyDensity = energy0 + energy1;

  return integral(energyDensity, grid);
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

  using nuclearConstants::h_bar;
  using nuclearConstants::m;
  using Operators::integral;
  Eigen::VectorXd tau = (*tauN + *tauP) * (pow(h_bar, 2) / (m * 2));
  double kineticEnergy = integral(tau, grid);

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

  spinN = std::make_shared<Eigen::MatrixX3d>(
      Wavefunction::spinDensity(neutrons, grid));
  spinP = std::make_shared<Eigen::MatrixX3d>(
      Wavefunction::spinDensity(protons, grid));

  double mu = 0.3;
  Eigen::VectorXd rho = *rhoN + *rhoP;
  Eigen::VectorXd newFieldN = Wavefunction::field(rho, *rhoN, grid, params);
  Eigen::VectorXd newFieldP = Wavefunction::field(rho, *rhoP, grid, params);
  if (UP == nullptr) {
    UP = std::make_shared<Eigen::VectorXd>(newFieldP);
    UN = std::make_shared<Eigen::VectorXd>(newFieldN);
  } else {
    UN = std::make_shared<Eigen::VectorXd>((*UN) * (1 - mu) + newFieldN * mu);
    UP = std::make_shared<Eigen::VectorXd>((*UP) * (1 - mu) + newFieldP * mu);
  }
  // start = std::chrono::steady_clock::now();

  // nablaRhoN =
  //     std::make_shared<Eigen::MatrixX3d>(Operators::gradNoSpin(*rhoN, grid));
  // nablaRhoP =
  //     std::make_shared<Eigen::MatrixX3d>(Operators::gradNoSpin(*rhoP, grid));
  // end = std::chrono::steady_clock::now();
  // std::cout << "Time elapsed grad "
  //           << std::chrono::duration_cast<std::chrono::milliseconds>(end -
  //                                                                    start)
  //                  .count()
  //           << " ms" << std::endl;
  // start = std::chrono::steady_clock::now();
  // nabla2RhoN = std::make_shared<Eigen::VectorXd>(
  //     Operators::divNoSpin(*nablaRhoN, grid).real());
  // nabla2RhoP = std::make_shared<Eigen::VectorXd>(
  //     Operators::divNoSpin(*nablaRhoP, grid).real());
  // end = std::chrono::steady_clock::now();
  // std::cout << "Time elapsed nabla 2 rho "
  //           << std::chrono::duration_cast<std::chrono::milliseconds>(end -
  //                                                                    start)
  //                  .count()
  //           << " ms" << std::endl;
  // start = std::chrono::steady_clock::now();
  // JN = std::make_shared<RealDoubleTensor>(
  //     Wavefunction::soDensity(neutrons, grid));
  // JP = std::make_shared<RealDoubleTensor>(
  //     Wavefunction::soDensity(protons, grid));
  // end = std::chrono::steady_clock::now();
  // std::cout << "Time elapsed J "
  //           << std::chrono::duration_cast<std::chrono::milliseconds>(end -
  //                                                                    start)
  //                  .count()
  //           << " ms" << std::endl;

  // divJN = std::make_shared<Eigen::VectorXcd>(Operators::divNoSpin(*JN,
  // grid)); divJP =
  // std::make_shared<Eigen::VectorXcd>(Operators::divNoSpin(*JP, grid));
}
