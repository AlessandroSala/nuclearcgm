#include "util/iteration_data.hpp"
#include "operators/differential_operators.hpp"
#include "util/wavefunction.hpp"
#include <chrono>
#include <iostream>
IterationData::IterationData() {}

void IterationData::updateQuantities(const Eigen::MatrixXcd &neutronsShells,
                                     const Eigen::MatrixXcd &protonsShells,
                                     int A, int Z, const Grid &grid) {
  int N = A - Z;
  auto neutrons = neutronsShells(Eigen::all, Eigen::seq(0, N - 1));
  auto protons = protonsShells(Eigen::all, Eigen::seq(0, Z - 1));

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
  start = std::chrono::steady_clock::now();
  nablaRhoN =
      std::make_shared<Eigen::MatrixX3d>(Operators::gradNoSpin(*rhoN, grid));
  nablaRhoP =
      std::make_shared<Eigen::MatrixX3d>(Operators::gradNoSpin(*rhoP, grid));
  end = std::chrono::steady_clock::now();
  std::cout << "Time elapsed grad "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                     start)
                   .count()
            << " ms" << std::endl;
  start = std::chrono::steady_clock::now();
  nabla2RhoN = std::make_shared<Eigen::VectorXd>(
      Operators::divNoSpin(*nablaRhoN, grid).real());
  nabla2RhoP = std::make_shared<Eigen::VectorXd>(
      Operators::divNoSpin(*nablaRhoP, grid).real());
  end = std::chrono::steady_clock::now();
  std::cout << "Time elapsed nabla 2 rho "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                     start)
                   .count()
            << " ms" << std::endl;
  start = std::chrono::steady_clock::now();
  JN = std::make_shared<RealDoubleTensor>(
      Wavefunction::soDensity(neutrons, grid));
  JP = std::make_shared<RealDoubleTensor>(
      Wavefunction::soDensity(protons, grid));
  end = std::chrono::steady_clock::now();
  std::cout << "Time elapsed J "
            << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                     start)
                   .count()
            << " ms" << std::endl;
  // divJN = std::make_shared<Eigen::VectorXcd>(Operators::divNoSpin(*JN,
  // grid)); divJP =
  // std::make_shared<Eigen::VectorXcd>(Operators::divNoSpin(*JP, grid));
}
