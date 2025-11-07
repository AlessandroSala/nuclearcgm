#include "constants.hpp"
#include "grid.hpp"
#include "guess.hpp"
#include "hamiltonian.hpp"
#include "input_parser.hpp"
#include "kinetic/iso_kinetic_potential.hpp"
#include "kinetic/local_kinetic_potential.hpp"
#include "kinetic/non_local_kinetic_potential.hpp"
#include "radius.hpp"
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

int main(int argc, char **argv)
{
  using namespace std;
  using namespace Eigen;
  using namespace nuclearConstants;
  using namespace Utilities;

  Eigen::initParallel();

  using recursive_directory_iterator =
      std::filesystem::recursive_directory_iterator;
  for (const auto &dirEntry : recursive_directory_iterator("input/exec"))
  {
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

    int orbitalsN = N + input.pairingParameters.additionalStates;
    int orbitalsP = Z + input.pairingParameters.additionalStates;

    auto WS = input.getWS();
    auto WSSO = input.getWSSO();

    std::pair<ComplexDenseMatrix, DenseVector> eigenpair;
    vector<shared_ptr<Potential>> pots;

    std::cout << "Initial beta: " << input.initialBeta << std::endl;
    Radius radius(input.initialBeta, WS.r0, A);

    pots.push_back(make_shared<DeformedWoodsSaxonPotential>(
        WS.V0, radius, WS.diffusivity, A, input.getZ(), WS.kappa, input.beta3));
    if (input.spinOrbit)
    {
      pots.push_back(make_shared<DeformedSpinOrbitPotential>(WSSO.V0, radius,
                                                             WS.diffusivity));
    }

    Hamiltonian initialWS(make_shared<Grid>(grid), pots);

    double nucRadius = pow(A, 0.3333333) * 1.20;
    int max = N > Z ? N + input.pairingParameters.additionalStates : Z + input.pairingParameters.additionalStates;
    auto guess =
        harmonic_oscillator_guess(grid, max, nucRadius, true);

    pair<MatrixXcd, VectorXd> firstEigenpair =
        solve(initialWS.build_matrix5p(), calc.initialGCG, guess);
    std::cout << firstEigenpair.second << std::endl;

    pair<MatrixXcd, VectorXd> protonsEigenpair;
    pair<MatrixXcd, VectorXd> neutronsEigenpair;
    if (N >= Z)
    {
      neutronsEigenpair = firstEigenpair;
      protonsEigenpair.second = neutronsEigenpair.second.head(orbitalsP);
      protonsEigenpair.first = neutronsEigenpair.first.leftCols(orbitalsP);
    }
    else if (N < Z)
    {
      protonsEigenpair = firstEigenpair;
      neutronsEigenpair.second = protonsEigenpair.second.head(orbitalsN);
      neutronsEigenpair.first = protonsEigenpair.first.leftCols(orbitalsN);
    }

    Wavefunction::normalize(neutronsEigenpair.first, grid);
    Wavefunction::normalize(protonsEigenpair.first, grid);

    IterationData data(input);

    std::vector<double> integralEnergies;
    std::vector<double> HFEnergies;

    vector<double> mu20s;
    mu20s.clear();
    std::cout << input.calculationType << std::endl;

    if (input.calculationType == CalculationType::deformation_curve)
    {
      std::cout << "=== Deformation curve, beta: [" << input.deformationCurve.start << ", " << input.deformationCurve.end << "], step: " << input.deformationCurve.step << " ===" << std::endl;
      double R0 = 1.2 * pow(A, 1.0 / 3.0);
      double startMu = Utilities::mu20FromBeta(input.deformationCurve.start, R0, A);
      double endMu = Utilities::mu20FromBeta(input.deformationCurve.end, R0, A);
      double stepMu = Utilities::mu20FromBeta(input.deformationCurve.step, R0, A);

      if(startMu < endMu) {
      for (double mu = startMu; mu <= endMu; mu += stepMu)
      {
        mu20s.push_back(mu);
      }
      }
      else {
      for (double mu = startMu; mu >= endMu; mu -= stepMu)
      {
        mu20s.push_back(mu);
      }
      }
      std::cout << "=== Deformation curve, mu: [" << startMu << ", " << endMu << "], step: " << stepMu << " ===" << std::endl;
      std::cout << mu20s.size() << " calculations to be done" << std::endl;
    }
    else
    {
      std::cout << "=== Ground state ===" << std::endl;
    }
    vector<unique_ptr<Constraint>> constraints;
    constraints.clear();
    // Wavefunction::printShells(neutronsEigenpair, grid);
    std::cout << "Start HF" << std::endl;

    // i=-1 to guarantee a first gs iteration if calculation is not deformation curve
    for (int i = -1; i < (int)mu20s.size(); ++i)
    {
      if (i == -1)
        i = 0;
      int hfIter = 0;

      double integralEnergy = 0.0;
      double HFEnergy = 0.0;
      //constraints.clear();
      if((int)mu20s.size() != 0){
        if ( i == 0)
        {
          auto mu = mu20s[i];
          constraints.push_back(make_unique<XCMConstraint>(0.0));
          constraints.push_back(make_unique<YCMConstraint>(0.0));
          constraints.push_back(make_unique<ZCMConstraint>(0.0));
          constraints.push_back(make_unique<X2MY2Constraint>(0.0));
          constraints.push_back(make_unique<XY2Constraint>(0.0));
          constraints.push_back(make_unique<OctupoleConstraint>(0.0));
          constraints.push_back(make_unique<QuadrupoleConstraint>(mu));
        } else {
          std::cout << "=== now using quadrupole constraint " << mu20s[i] << " ===" << std::endl;
          //TODO: fix this, always keep quadrupole constraint last!
          constraints.back()->target = mu20s[i];
        }
      }
      for (hfIter = 0; hfIter < calc.hf.cycles; ++hfIter)
      {
        data.updateQuantities(neutronsEigenpair, protonsEigenpair,
                              hfIter, constraints);
        int maxIterGCGHF = calc.hf.gcg.maxIter;

        cout << "HF iteration: " << hfIter << endl;
        cout << "Neutrons " << endl;

        vector<shared_ptr<Potential>> pots;
        auto dataPtr = make_shared<IterationData>(data);
        skyrmeHamiltonian(pots, input, NucleonType::N, dataPtr);

        Hamiltonian skyrmeHam(make_shared<Grid>(grid), pots);

        auto newNeutronsEigenpair = solve(skyrmeHam.buildMatrix(), calc.hf.gcg,
                                          neutronsEigenpair.first, orbitalsN);

        pair<MatrixXcd, VectorXd> newProtonsEigenpair;
        if (input.useCoulomb || N != Z)
        {
          pots.clear();
          skyrmeHamiltonian(pots, input, NucleonType::P, dataPtr);

          std::cout << "Protons " << std::endl;
          Hamiltonian skyrmeHam(make_shared<Grid>(grid), pots);

          newProtonsEigenpair = solve(skyrmeHam.buildMatrix(), calc.hf.gcg,
                                      protonsEigenpair.first, orbitalsP);
        }
        else
        {
          newProtonsEigenpair.first =
              newNeutronsEigenpair.first.leftCols(orbitalsP);
          newProtonsEigenpair.second =
              newNeutronsEigenpair.second.head(orbitalsP);
        }

        neutronsEigenpair = newNeutronsEigenpair;
        protonsEigenpair = newProtonsEigenpair;

        Wavefunction::normalize(neutronsEigenpair.first, grid);
        Wavefunction::normalize(protonsEigenpair.first, grid);

        double newIntegralEnergy =
            data.totalEnergyIntegral(input.skyrme, grid) +
            data.kineticEnergy(input.skyrme, grid);

        double SPE = (neutronsEigenpair.second.sum() +
                      protonsEigenpair.second.sum());

        integralEnergies.push_back(newIntegralEnergy);
        double newHFEnergy = 0.0;
        HFEnergies.push_back(newHFEnergy);
        if (abs(newIntegralEnergy - integralEnergy) < input.getCalculation().hf.energyTol &&
            abs(newHFEnergy - HFEnergy) < input.getCalculation().hf.energyTol &&
            abs(data.constraintEnergy(constraints)) <
                input.getCalculation().hf.energyTol)
        {
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
                       &data, input, hfIter, integralEnergies, HFEnergies, cpuTime,
                       'a', constraints);
    }
  }
  return 0;
}
