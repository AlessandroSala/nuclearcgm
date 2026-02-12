#include "constants.hpp"
#include "grid.hpp"
#include "guess.hpp"
#include "hamiltonian.hpp"
#include "input_parser.hpp"
#include "kinetic/local_kinetic_potential.hpp"
#include "kinetic/non_local_kinetic_potential.hpp"
#include "operators/integral_operators.hpp"
#include "radius.hpp"
#include "skyrme/local_coulomb_potential.hpp"
#include "skyrme/multipole_constraint.hpp"
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
    if (!input.check()) {
      continue;
    }

    Output out("output/" + input.outputDirectory);

    auto computationBegin = std::chrono::steady_clock::now();

    Grid grid = input.get_grid();
    Calculation calc = input.getCalculation();

    std::vector<double> betas;
    int A = input.getA();
    int Z = input.getZ();
    int N = A - Z;

    int orbitalsN = N + input.pairingN.additionalStates;
    int orbitalsP = Z + input.pairingP.additionalStates;

    auto WS = input.getWS();
    auto WSSO = input.getWSSO();

    std::pair<ComplexDenseMatrix, DenseVector> eigenpair;
    vector<shared_ptr<Potential>> pots;

    std::cout << std::endl;
    Radius radius(input.initialBeta, WS.r0, A);

    pots.push_back(make_shared<DeformedWoodsSaxonPotential>(
        WS.V0, radius, WS.diffusivity, A, input.getZ(), WS.kappa, input.beta3));
    if (input.spinOrbit) {
      pots.push_back(make_shared<DeformedSpinOrbitPotential>(WSSO.V0, radius,
                                                             WS.diffusivity));
    }

    Hamiltonian initialWS(make_shared<Grid>(grid), pots);

    double nucRadius = pow(A, 0.3333333) * 1.20;
    int max = orbitalsN > orbitalsP ? orbitalsN : orbitalsP;
    auto guess = harmonic_oscillator_guess(grid, max, nucRadius,
                                           input.initialBeta, true);

    std::cout << "=== Woods-Saxon guess ===" << std::endl;
    pair<MatrixXcd, VectorXd> firstEigenpair =
        solve(initialWS.build_matrix5p(), Eigen::MatrixXcd(0, 0),
              calc.initialGCG, guess);

    pair<MatrixXcd, VectorXd> protonsEigenpair;
    pair<MatrixXcd, VectorXd> neutronsEigenpair;
    if (N >= Z) {
      neutronsEigenpair = firstEigenpair;
      protonsEigenpair.second = neutronsEigenpair.second.head(orbitalsP);
      protonsEigenpair.first = neutronsEigenpair.first.leftCols(orbitalsP);
    } else if (N < Z) {
      protonsEigenpair = firstEigenpair;
      neutronsEigenpair.second = protonsEigenpair.second.head(orbitalsN);
      neutronsEigenpair.first = protonsEigenpair.first.leftCols(orbitalsN);
    }

    Wavefunction::normalize(neutronsEigenpair.first, grid);
    Wavefunction::normalize(protonsEigenpair.first, grid);

    IterationData data(input);

    std::vector<double> integralEnergies;
    std::vector<double> HFEnergies;
    std::vector<double> maxDispersions;

    vector<double> mu20s;
    mu20s.clear();

    if (input.calculationType == CalculationType::deformation_curve) {
      std::cout << "=== Deformation curve, beta: ["
                << input.deformationCurve.start << ", "
                << input.deformationCurve.end
                << "], step: " << input.deformationCurve.step
                << " ===" << std::endl;
      double R0 = 1.2 * pow(A, 1.0 / 3.0);
      double startMu =
          Utilities::mu20FromBeta(input.deformationCurve.start, R0, A);
      double endMu = Utilities::mu20FromBeta(input.deformationCurve.end, R0, A);
      double stepMu =
          Utilities::mu20FromBeta(input.deformationCurve.step, R0, A);

      if (startMu < endMu) {
        for (double mu = startMu; mu <= endMu; mu += stepMu) {
          mu20s.push_back(mu);
        }
      } else {
        for (double mu = startMu; mu >= endMu; mu -= stepMu) {
          mu20s.push_back(mu);
        }
      }
      std::cout << "=== Deformation curve, mu: [" << startMu << ", " << endMu
                << "], step: " << stepMu << " ===" << std::endl;
      std::cout << mu20s.size() << " calculations to be done" << std::endl;
    }
    vector<unique_ptr<Constraint>> constraints;
    constraints.clear();
    // Wavefunction::printShells(neutronsEigenpair, grid);

    // i=-1 to guarantee a first gs iteration if calculation is not deformation
    // curve
    Eigen::MatrixXcd ConjDirN(0, 0), ConjDirP(0, 0);
    std::vector<double> enErrors;
    for (int i = -1; i < (int)mu20s.size(); ++i) {
      if (i == -1)
        i = 0;
      int hfIter = 0;

      double integralEnergy = 0.0;
      double HFEnergy = 0.0;
      // constraints.push_back(make_unique<OctupoleConstraint>(20.0));
      // constraints.push_back(make_unique<XCMConstraint>(0.0));
      // constraints.push_back(make_unique<YCMConstraint>(0.0));
      // constraints.push_back(make_unique<ZCMConstraint>(0.0));
      //  constraints.clear();

      if (input.constrainCOM) {
        std::cout << "Constraining COM to the origin" << std::endl;
        constraints.push_back(make_unique<XCMConstraint>(0.0));
        constraints.push_back(make_unique<YCMConstraint>(0.0));
        constraints.push_back(make_unique<ZCMConstraint>(0.0));
      }
      if ((int)mu20s.size() > 0) {
        if (i == 0) {
          auto mu = mu20s[i];
          constraints.push_back(make_unique<XCMConstraint>(0.0));
          constraints.push_back(make_unique<YCMConstraint>(0.0));
          constraints.push_back(make_unique<ZCMConstraint>(0.0));
          constraints.push_back(make_unique<X2MY2Constraint>(0.0));
          constraints.push_back(make_unique<XY2Constraint>(0.0));
          constraints.push_back(make_unique<OctupoleConstraint>(0.0));
          constraints.push_back(make_unique<QuadrupoleConstraint>(mu));
        } else {
          std::cout << "=== now using quadrupole constraint " << mu20s[i]
                    << " ===" << std::endl;
          // TODO: fix this, always keep quadrupole constraint last!
          constraints.back()->target = mu20s[i];
        }
      }
      ConjDirN = Eigen::MatrixXcd(0, 0);
      ConjDirP = Eigen::MatrixXcd(0, 0);
      Eigen::VectorXd neutronSPEDiff(orbitalsN);
      Eigen::VectorXd protonSPEDiff(orbitalsN);
      neutronSPEDiff.setOnes();
      protonSPEDiff.setOnes();

      for (hfIter = 0; hfIter < calc.hf.cycles; ++hfIter) {
        if (hfIter == 1) {

          if (input.multipoleConstraints.size() > 0) {
            for (auto &c : input.multipoleConstraints) {
              constraints.push_back(
                  make_unique<MultipoleConstraint>(c.target, c.l, c.m, &data));
            }
          }
        }
        data.updateQuantities(neutronsEigenpair, protonsEigenpair, hfIter,
                              constraints);
        int maxIterGCGHF = calc.hf.gcg.maxIter;

        cout << endl;
        cout << "======= Iteration: " << hfIter << " =======" << endl;

        vector<shared_ptr<Potential>> pots;
        auto dataPtr = make_shared<IterationData>(data);
        skyrmeHamiltonian(pots, input, NucleonType::N, dataPtr);

        Hamiltonian skyrmeHam(make_shared<Grid>(grid), pots);

        auto hN = skyrmeHam.buildMatrix();

        auto newNeutronsEigenpair = solve(hN, ConjDirN, calc.hf.gcg,
                                          neutronsEigenpair.first, orbitalsN);

        pair<MatrixXcd, VectorXd> newProtonsEigenpair;
        auto hP = hN;
        if (input.useCoulomb || N != Z) {
          pots.clear();
          skyrmeHamiltonian(pots, input, NucleonType::P, dataPtr);

          Hamiltonian skyrmeHam(make_shared<Grid>(grid), pots);

          hP = skyrmeHam.buildMatrix();

          newProtonsEigenpair = solve(hP, ConjDirP, calc.hf.gcg,
                                      protonsEigenpair.first, orbitalsP);
        } else {
          newProtonsEigenpair.first =
              newNeutronsEigenpair.first.leftCols(orbitalsP);
          newProtonsEigenpair.second =
              newNeutronsEigenpair.second.head(orbitalsP);
        }

        neutronSPEDiff =
            (newNeutronsEigenpair.second - neutronsEigenpair.second)
                .array()
                .abs();
        protonSPEDiff = (newProtonsEigenpair.second - protonsEigenpair.second)
                            .array()
                            .abs();

        double maxSPEDiff =
            std::max(protonSPEDiff.maxCoeff(), neutronSPEDiff.maxCoeff());

        Wavefunction::normalize(newNeutronsEigenpair.first, grid);
        Wavefunction::normalize(newProtonsEigenpair.first, grid);

        if (input.pairingType != PairingType::none) {
          newNeutronsEigenpair.first =
              Wavefunction::TROrder(newNeutronsEigenpair.first);
          newProtonsEigenpair.first =
              Wavefunction::TROrder(newProtonsEigenpair.first);
        }

        if (hfIter > 0) {
          MatrixXcd tmp;
          tmp.noalias() =
              neutronsEigenpair.first.adjoint() * newNeutronsEigenpair.first;

          ConjDirN.noalias() =
              newNeutronsEigenpair.first - neutronsEigenpair.first * tmp;

          tmp.noalias() =
              protonsEigenpair.first.adjoint() * newProtonsEigenpair.first;

          ConjDirP.noalias() =
              newProtonsEigenpair.first - protonsEigenpair.first * tmp;
          ConjDirN.colwise().normalize();
          ConjDirP.colwise().normalize();
        }

        if (hfIter > 0 && false) {

          auto dispersionN =
              Utilities::computeDispersion(hN, newNeutronsEigenpair.first);
          auto dispersionP =
              Utilities::computeDispersion(hP, newProtonsEigenpair.first);
          auto dispersion = (dispersionN * N + dispersionP * Z) / (N + Z);
          cout << "Neutron dispersion: " << dispersionN << endl;
          cout << "Proton dispersion: " << dispersionP << endl;
          cout << "Total dispersion: "
               << (dispersionN * N + dispersionP * Z) / (N + Z) << endl;
        }

        neutronsEigenpair = newNeutronsEigenpair;
        protonsEigenpair = newProtonsEigenpair;

        double newIntegralEnergy =
            data.totalEnergyIntegral() + data.kineticEnergy();

        double SPE =
            (neutronsEigenpair.second.sum() + protonsEigenpair.second.sum());

        integralEnergies.push_back(newIntegralEnergy);
        double newHFEnergy = data.HFEnergy(SPE, constraints);
        bool constraintsConv = true;
        if (constraints.size() > 0) {
          std::cout << "Constraints errors: ";
          for (auto &&constraint : constraints) {
            if (constraint->error() != 0.0) {
              std::cout << constraint->error() << ", ";
              constraintsConv = constraintsConv &&
                                (constraint->error() < input.constraintsTol);
            }
          }
          // std::cout << "Constraints energies: ";
          // for (auto &&constraint : constraints) {
          //   std::cout << constraint->evaluate(&data) << ", ";
          // }
        }
        std::cout << std::endl;
        HFEnergies.push_back(newHFEnergy);

        double e_int_diff = std::abs(newIntegralEnergy - integralEnergy);
        double e_int_diff_rel = e_int_diff / newIntegralEnergy;
        std::cout << "ED diff (rel): " << e_int_diff_rel
                  << " (abs): " << e_int_diff << std::endl;
        enErrors.push_back((newIntegralEnergy - integralEnergy));
        double maxDiff = 0.0;

        double tol = input.getCalculation().hf.energyTol;
        if (abs(newIntegralEnergy - integralEnergy) < tol &&
            abs(newHFEnergy - HFEnergy) < tol && constraintsConv) {
          break;
        }
        data.energyDiff = std::abs(newHFEnergy - HFEnergy);
        integralEnergy = newIntegralEnergy;
        HFEnergy = newHFEnergy;
        data.logData(neutronsEigenpair, protonsEigenpair, constraints);
      }
      if (hfIter < input.getCalculation().hf.cycles) {
        std::cout << "Calculation converged in " << hfIter << " iterations."
                  << std::endl;
      } else {
        std::cout << "Calculation did not converge in " << hfIter
                  << " iterations." << std::endl;
      }

      // cout << "Neutrons " << endl;
      // Wavefunction::printShells(neutronsEigenpair, grid);
      // cout << "Protons " << endl;
      // Wavefunction::printShells(protonsEigenpair, grid);

      std::chrono::steady_clock::time_point computationEnd =
          std::chrono::steady_clock::now();

      double cpuTime = chrono::duration_cast<chrono::seconds>(computationEnd -
                                                              computationBegin)
                           .count();
      std::cout << "CPU time: " << cpuTime << std::endl;
      out.shellsToFile(neutronsEigenpair, protonsEigenpair, &data, input,
                       hfIter, integralEnergies, HFEnergies, cpuTime, 'a',
                       constraints);
    }
  }
  return 0;
}
