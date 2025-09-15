#include "constants.hpp"
#include "grid.hpp"
#include "guess.hpp"
#include "hamiltonian.hpp"
#include "input_parser.hpp"
#include "kinetic/iso_kinetic_potential.hpp"
#include "kinetic/local_kinetic_potential.hpp"
#include "kinetic/non_local_kinetic_potential.hpp"
#include "radius.hpp"
#include "skyrme/axial_symmetry_constraint.hpp"
#include "skyrme/local_coulomb_potential.hpp"
#include "skyrme/octupole_constraint.hpp"
#include "skyrme/quadrupole_constraint.hpp"
#include "skyrme/x2my2_constraint.hpp"
#include "skyrme/xcm_constraint.hpp"
#include "skyrme/xy2_constraint.hpp"
#include "skyrme/ycm_constraint.hpp"
#include "skyrme/zcm_constraint.hpp"
#include "solver.hpp"
#include "spherical_harmonics.hpp"
#include "spin_orbit/deformed_spin_orbit.hpp"
#include "types.hpp"
#include "util/iteration_data.hpp"
#include "util/output.hpp"
#include "util/utilities.hpp"
#include "util/wavefunction.hpp"
#include "woods_saxon/deformed_woods_saxon.hpp"
#include <chrono>
#include <cmath>
#include <experimental/filesystem>
#include <filesystem>
#include <iostream>
#include <memory>
#include <skyrme/skyrme_so.hpp>
#include <skyrme/skyrme_u.hpp>
#include <vector>

int main(int argc, char **argv) {
  using namespace std;
  using namespace Eigen;
  using namespace nuclearConstants;
  using namespace Utilities;

  Eigen::initParallel();

  using recursive_directory_iterator =
      std::filesystem::recursive_directory_iterator;
  for (const auto &dirEntry : recursive_directory_iterator("input/exec")) {
    cout << "Reading input from " << dirEntry.path() << endl;
    InputParser input(dirEntry.path().string());
    Output out("output/" + input.outputDirectory);

    auto computationBegin = std::chrono::steady_clock::now();

    Grid grid = input.get_grid();
    Calculation calc = input.getCalculation();

    std::vector<double> betas;
    int A = input.getA();
    int Z = input.getZ();
    int N = A - Z;

    auto WS = input.getWS();
    auto WSSO = input.getWSSO();

    std::pair<ComplexDenseMatrix, DenseVector> eigenpair;
    vector<shared_ptr<Potential>> pots;

    Radius radius(0.00, WS.r0, A);

    pots.push_back(make_shared<DeformedWoodsSaxonPotential>(
        WS.V0, radius, WS.diffusivity, A, input.getZ(), WS.kappa));
    if (input.spinOrbit) {
      pots.push_back(make_shared<DeformedSpinOrbitPotential>(WSSO.V0, radius,
                                                             WS.diffusivity));
    }

    Hamiltonian initialWS(make_shared<Grid>(grid), pots);

    double nucRadius = pow(A, 0.3333333) * 1.27;
    auto guess =
        harmonic_oscillator_guess(grid, N + input.additional, nucRadius, true);
    pair<MatrixXcd, VectorXd> neutronsEigenpair =
        solve(initialWS.build_matrix5p(), calc.initialGCG, guess);
    std::cout << neutronsEigenpair.second << std::endl;

    pair<MatrixXcd, VectorXd> protonsEigenpair;
    if (N > Z) {
      protonsEigenpair.second = neutronsEigenpair.second;
      protonsEigenpair.first = neutronsEigenpair.first(all, seq(0, Z - 1));
    } else if (N < Z) {
      throw std::runtime_error("Protons cannot be smaller than neutrons");
    } else {
      protonsEigenpair = neutronsEigenpair;
    }

    Wavefunction::normalize(neutronsEigenpair.first, grid);
    Wavefunction::normalize(protonsEigenpair.first, grid);

    IterationData data(input);

    std::vector<double> integralEnergies;

    vector<double> mu20s;
    for (double mu = 33.0; mu < 33.5; mu += 1.0) {
      mu20s.push_back(mu);
    }
    vector<unique_ptr<Constraint>> constraints;
    Wavefunction::printShells(neutronsEigenpair, grid);

    for (int i = -1; i < (int)mu20s.size(); ++i) {
      int hfIter = 0;

      double integralEnergy = 0.0;
      double HFEnergy = 0.0;
      constraints.clear();
        auto mu = mu20s[i];
      if (i >= 0) {
        constraints.push_back(make_unique<XCMConstraint>(0.0));
        constraints.push_back(make_unique<YCMConstraint>(0.0));
        constraints.push_back(make_unique<ZCMConstraint>(0.0));
        constraints.push_back(make_unique<OctupoleConstraint>(0.0));
        constraints.push_back(make_unique<X2MY2Constraint>(0.0));
        constraints.push_back(make_unique<XY2Constraint>(0.0));
        constraints.push_back(make_unique<QuadrupoleConstraint>(mu));
      }
      for (hfIter = 0; hfIter < calc.hf.cycles; ++hfIter) {
        data.updateQuantities(neutronsEigenpair.first, protonsEigenpair.first,
                              hfIter, constraints);
        int maxIterGCGHF = calc.hf.gcg.maxIter;

        cout << "HF iteration: " << hfIter << endl;
        cout << "Neutrons " << endl;

        vector<shared_ptr<Potential>> pots;
        skyrmeHamiltonian(pots, input, NucleonType::N, data);

        Hamiltonian skyrmeHam(make_shared<Grid>(grid), pots);

        auto newNeutronsEigenpair = solve(skyrmeHam.buildMatrix(), calc.hf.gcg,
                                          neutronsEigenpair.first);

        pair<MatrixXcd, VectorXd> newProtonsEigenpair;
        if (input.useCoulomb || N != Z) {
          pots.clear();
          skyrmeHamiltonian(pots, input, NucleonType::P, data);

          std::cout << "Protons " << std::endl;
          pots.push_back(make_shared<LocalCoulombPotential>(data.UCoul));
          Hamiltonian skyrmeHam(make_shared<Grid>(grid), pots);

          newProtonsEigenpair = solve(skyrmeHam.buildMatrix(), calc.hf.gcg,
                                      protonsEigenpair.first);

        } else {
          newProtonsEigenpair.first =
              newNeutronsEigenpair.first(Eigen::all, Eigen::seq(0, Z - 1));
          newProtonsEigenpair.second =
              newNeutronsEigenpair.second(Eigen::seq(0, Z - 1));
        }

        neutronsEigenpair = newNeutronsEigenpair;
        protonsEigenpair = newProtonsEigenpair;

        Wavefunction::normalize(neutronsEigenpair.first, grid);
        Wavefunction::normalize(protonsEigenpair.first, grid);

        double newIntegralEnergy =
            data.totalEnergyIntegral(input.skyrme, grid) +
            data.kineticEnergy(input.skyrme, grid);

        double SPE = 0.5 * (neutronsEigenpair.second.sum() +
                            protonsEigenpair.second.sum());

        integralEnergies.push_back(newIntegralEnergy);
        double newHFEnergy = data.HFEnergy(2.0 * SPE, constraints);
        if (abs(newIntegralEnergy - integralEnergy) <
                input.getCalculation().hf.energyTol &&
            abs(newHFEnergy - HFEnergy) < input.getCalculation().hf.energyTol &&
            data.constraintEnergy(constraints) <
                input.getCalculation().hf.energyTol/100.0) {
          break;
        }
        data.energyDiff = std::abs(newHFEnergy - HFEnergy);
        integralEnergy = newIntegralEnergy;
        HFEnergy = newHFEnergy;
        data.logData(neutronsEigenpair, protonsEigenpair, constraints);
      }
      cout << "Neutrons " << endl;
      Wavefunction::printShells(neutronsEigenpair, grid);
      cout << "Protons " << endl;
      Wavefunction::printShells(protonsEigenpair, grid);

      std::chrono::steady_clock::time_point computationEnd =
          std::chrono::steady_clock::now();
      double cpuTime = chrono::duration_cast<chrono::seconds>(computationEnd -
                                                              computationBegin)
                           .count();
      data.lastConvergedIter = 0;
      out.shellsToFile("calc_output.csv", neutronsEigenpair, protonsEigenpair,
                       &data, input, hfIter, integralEnergies, cpuTime,
                       i == -1 ? 'w' : 'a', constraints);
    }
  }
  return 0;
}
