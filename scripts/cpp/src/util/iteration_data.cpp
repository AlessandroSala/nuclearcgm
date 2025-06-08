#include "util/iteration_data.hpp"
#include "util/wavefunction.hpp"
#include "operators/differential_operators.hpp"

IterationData::IterationData()
{

}

void IterationData::updateQuantities(const Eigen::MatrixXcd& neutronsShells, const Eigen::MatrixXcd& protonsShells, int A, int Z, const Grid& grid) {
    int N = A - Z;
    auto neutrons = neutronsShells(Eigen::all, Eigen::seq(0, N - 1));
    auto protons = protonsShells(Eigen::all, Eigen::seq(0, Z - 1));
    rhoN = std::make_shared<Eigen::VectorXd>(Wavefunction::density(neutrons, grid));
    rhoP = std::make_shared<Eigen::VectorXd>(Wavefunction::density(protons, grid));
    tauN = std::make_shared<Eigen::VectorXd>(Wavefunction::kineticDensity(neutrons, grid));
    tauP = std::make_shared<Eigen::VectorXd>(Wavefunction::kineticDensity(protons, grid));
    nablaRhoN = std::make_shared<Eigen::MatrixX3d>(Operators::gradNoSpin(*rhoN, grid));
    nablaRhoP = std::make_shared<Eigen::MatrixX3d>(Operators::gradNoSpin(*rhoP, grid));
    nabla2RhoN = std::make_shared<Eigen::VectorXd>(Operators::divNoSpin(*nablaRhoN, grid).real());
    nabla2RhoP = std::make_shared<Eigen::VectorXd>(Operators::divNoSpin(*nablaRhoP, grid).real());
    JN = std::make_shared<Eigen::MatrixX3cd>(Wavefunction::soDensity(neutrons, grid));
    JP = std::make_shared<Eigen::MatrixX3cd>(Wavefunction::soDensity(protons, grid));
    divJN = std::make_shared<Eigen::VectorXcd>(Operators::divNoSpin(*JN, grid));
    divJP = std::make_shared<Eigen::VectorXcd>(Operators::divNoSpin(*JP, grid));
}