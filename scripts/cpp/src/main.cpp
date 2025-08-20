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
#include "solver.hpp"
#include "spin_orbit/deformed_spin_orbit.hpp"
#include "types.hpp"
#include "util/iteration_data.hpp"
#include "util/output.hpp"
#include "util/wavefunction.hpp"
#include "woods_saxon/deformed_woods_saxon.hpp"
#include <chrono>
#include <cmath>
#include <experimental/filesystem>
#include <iostream>
#include <memory>
#include <skyrme/skyrme_so.hpp>
#include <skyrme/skyrme_u.hpp>
#include <vector>

int main(int argc, char **argv) {
  using namespace std;
  using namespace Eigen;
  using namespace nuclearConstants;
  cout << "Using mass " << m << endl;
  Eigen::initParallel();
  InputParser input("input/input.json");
  ComplexDenseMatrix guess;
  Output out;

  auto computationBegin = std::chrono::steady_clock::now();

  Grid grid = input.get_grid();
  Calculation calc = input.getCalculation();

  std::vector<double> betas;
  int A = input.getA();
  int Z = input.getZ();
  int N = A - Z;

  auto WS = input.getWS();
  auto WSSO = input.getWSSO();

  std::cout << input.skyrme.t0 << " " << input.skyrme.t1 << " "
            << input.skyrme.t2 << " " << input.skyrme.t3 << " "
            << input.skyrme.W0 << std::endl;
  cout << "Initial CG tol: " << calc.initialGCG.cgTol << endl;
  cout << "Initial CG steps: " << calc.initialGCG.steps << endl;

  std::pair<ComplexDenseMatrix, DenseVector> eigenpair;
  vector<shared_ptr<Potential>> pots;

  Radius radius(0.0, WS.r0, A);

  pots.push_back(make_shared<DeformedWoodsSaxonPotential>(
      WS.V0, radius, WS.diffusivity, A, input.getZ(), WS.kappa));
  pots.push_back(make_shared<IsoKineticPotential>());
  if (input.spinOrbit) {
    pots.push_back(make_shared<DeformedSpinOrbitPotential>(WSSO.V0, radius,
                                                           WS.diffusivity));
  }

  Hamiltonian initialWS(make_shared<Grid>(grid), pots);
  // cout << ham.buildMatrix() << endl;
  //

  // double h_omega = 41.0 / (pow(A, 0.33333));
  double nucRadius = pow(A, 0.3333333) * 1.27;
  pair<MatrixXcd, VectorXd> neutronsEigenpair = gcgm_complex_no_B(
      initialWS.buildMatrix(),
      harmonic_oscillator_guess(grid, N + input.additional, nucRadius,
                                input.spinOrbit),
      N + input.additional, 60, calc.initialGCG.maxIter, 0.0001,
      calc.initialGCG.steps, calc.initialGCG.cgTol, false, 2);

  std::cout << neutronsEigenpair.second << std::endl;
  pair<MatrixXcd, VectorXd> protonsEigenpair;
  if (Z != N) {
    if (N > Z) {
      protonsEigenpair.second = neutronsEigenpair.second;
      protonsEigenpair.first = neutronsEigenpair.first(all, seq(0, Z - 1));
    } else {
      pots.clear();

      pots.push_back(make_shared<DeformedWoodsSaxonPotential>(
          WS.V0, radius, WS.diffusivity, A, input.getZ(), -WS.kappa));
      if (input.spinOrbit) {
        pots.push_back(make_shared<DeformedSpinOrbitPotential>(WSSO.V0, radius,
                                                               WS.diffusivity));
      }

      int pEigval = Z + input.additional;

      MatrixXcd pGuess =
          N > Z ? neutronsEigenpair.first(all, seq(0, pEigval - 1))
                : harmonic_oscillator_guess(grid, pEigval, grid.get_a());

      initialWS = Hamiltonian(make_shared<Grid>(grid), pots);
      protonsEigenpair = gcgm_complex_no_B(
          initialWS.build_matrix5p(),
          harmonic_oscillator_guess(grid, N + input.additional, grid.get_a()),
          pEigval, 35 + 0.01, calc.initialGCG.maxIter, 0.0001,
          grid.get_n() * 1.2, 1.0e-7 / grid.get_n(), false, 1);
    }
  } else {
    protonsEigenpair = neutronsEigenpair;
  }

  Wavefunction::normalize(neutronsEigenpair.first, grid);
  Wavefunction::normalize(protonsEigenpair.first, grid);

  IterationData data(input);

  std::vector<double> integralEnergies;

  double integralEnergy = 0.0;
  double HFEnergy = 0.0;

  data.updateQuantities(neutronsEigenpair.first, protonsEigenpair.first, grid);

  int hfIter = 0;
  for (hfIter = 0; hfIter < calc.hf.cycles; ++hfIter) {
    int maxIterGCGHF = calc.hf.gcg.maxIter;
    maxIterGCGHF = 14 - (hfIter % 5);
    //    if (hfIter == 0) {
    //    maxIterGCGHF = ;
    //  }

    cout << "HF iteration: " << hfIter << endl;

    std::cout << "Neutrons " << std::endl;
    vector<shared_ptr<Potential>> pots;
    pots.push_back(make_shared<SkyrmeU>(input.skyrme, NucleonType::N,
                                        make_shared<IterationData>(data)));
    pots.push_back(make_shared<NonLocalKineticPotential>(
        make_shared<IterationData>(data), NucleonType::N));
    pots.push_back(make_shared<LocalKineticPotential>(
        make_shared<IterationData>(data), NucleonType::N));
    if (input.spinOrbit)
      pots.push_back(make_shared<SkyrmeSO>(make_shared<IterationData>(data),
                                           NucleonType::N));

    Hamiltonian skyrmeHam(make_shared<Grid>(grid), pots);

    auto newNeutronsEigenpair = gcgm_complex_no_B(
        skyrmeHam.buildMatrix(), neutronsEigenpair.first, N + input.additional,
        35 + 0.01, maxIterGCGHF, calc.hf.gcg.tol, 40,
        1.0e-4 / (calc.initialGCG.nev), false, 1);

    pair<MatrixXcd, VectorXd> newProtonsEigenpair;
    if (input.useCoulomb) {
      pots.clear();
      pots.push_back(make_shared<LocalKineticPotential>(
          make_shared<IterationData>(data), NucleonType::P));
      pots.push_back(make_shared<NonLocalKineticPotential>(
          make_shared<IterationData>(data), NucleonType::P));
      pots.push_back(make_shared<SkyrmeU>(input.skyrme, NucleonType::P,
                                          make_shared<IterationData>(data)));
      if (input.spinOrbit)
        pots.push_back(make_shared<SkyrmeSO>(make_shared<IterationData>(data),
                                             NucleonType::P));
      std::cout << "Protons " << std::endl;
      pots.push_back(make_shared<LocalCoulombPotential>(data.UCoul));
      Hamiltonian skyrmeHam(make_shared<Grid>(grid), pots);
      newProtonsEigenpair =
          gcgm_complex_no_B(skyrmeHam.buildMatrix(), protonsEigenpair.first, Z,
                            35 + 0.01, maxIterGCGHF, calc.hf.gcg.tol, 40,
                            1.0e-4 / (calc.initialGCG.nev), false, 1);
      skyrmeHam = Hamiltonian(make_shared<Grid>(grid), pots);

    } else {
      newProtonsEigenpair = newNeutronsEigenpair;
    }

    neutronsEigenpair = newNeutronsEigenpair;
    protonsEigenpair = newProtonsEigenpair;

    Wavefunction::normalize(neutronsEigenpair.first, grid);
    Wavefunction::normalize(protonsEigenpair.first, grid);

    data.updateQuantities(neutronsEigenpair.first, protonsEigenpair.first,
                          grid);

    double newIntegralEnergy = data.totalEnergyIntegral(input.skyrme, grid) +
                               data.kineticEnergy(input.skyrme, grid);

    double SPE =
        0.5 * (neutronsEigenpair.second.sum() + protonsEigenpair.second.sum());

    integralEnergies.push_back(newIntegralEnergy);
    double newHFEnergy = data.HFEnergy(2.0 * SPE, grid);
    cout << "Total energy as integral: " << newIntegralEnergy << endl;
    cout << "Total energy as HF energy: " << newHFEnergy << endl;

    cout << "Direct coulomb energy: " << data.CoulombDirectEnergy(grid) << endl;
    cout << "Slater exchange energy: " << data.SlaterCoulombEnergy(grid)
         << endl;

    if (abs(newIntegralEnergy - integralEnergy) <
            input.getCalculation().hf.energyTol &&
        abs(newHFEnergy - HFEnergy) < input.getCalculation().hf.energyTol) {
      break;
    }
    integralEnergy = newIntegralEnergy;
    HFEnergy = newHFEnergy;
  }
  cout << "Neutrons " << endl;
  Wavefunction::printShells(neutronsEigenpair, grid);
  cout << "Protons " << endl;
  Wavefunction::printShells(protonsEigenpair, grid);

  std::chrono::steady_clock::time_point computationEnd =
      std::chrono::steady_clock::now();
  double cpuTime =
      chrono::duration_cast<chrono::seconds>(computationEnd - computationBegin)
          .count();
  out.shellsToFile("calc_output.csv", neutronsEigenpair, protonsEigenpair,
                   make_shared<IterationData>(data), input, hfIter,
                   integralEnergies, cpuTime, grid);
  return 0;
  // pots.push_back(
  //     make_shared<WoodsSaxonPotential>(V0, r_0 * pow(A, 1.0 / 3.0), 0.67));
  // pots.push_back(make_shared<DeformedWoodsSaxonPotential>(
  //    V0, Radius(0.000, r_0, A), 0.67, A, input.getZ(), input.getKappa()));
  // pots.push_back(
  //    make_shared<DeformedSpinOrbitPotential>(V0, Radius(0.000, r_0, A),
  //    0.67));

  // Hamiltonian hamDef = Hamiltonian(make_shared<Grid>(grid), pots);
  //// cout << ham.buildMatrix() << endl;
  // pair<MatrixXcd, VectorXd> defNeutronsEigenpair = gcgm_complex_no_B(
  //     hamDef.build_matrix7p(),
  //     harmonic_oscillator_guess(grid, calc.nev, grid.get_a()), calc.nev,
  //     35 + 0.01, calc.cycles, calc.tol * N, 40, 5.0e-8, false, 1);

  // std::chrono::steady_clock::time_point end =
  // std::chrono::steady_clock::now();

  // pots.clear();

  // Wavefunction::normalize(defNeutronsEigenpair.first, grid);
  // int n_betas = 11;
  // DenseMatrix energies(calc.nev, n_betas);
  // DenseMatrix ms(calc.nev, n_betas);
  // DenseMatrix J2s(calc.nev, n_betas);
  // DenseMatrix L2s(calc.nev, n_betas);
  // DenseMatrix Ps(calc.nev, n_betas);

  // for (int j = 0; j < calc.nev; ++j) {
  //   Shell shell(
  //       make_shared<Grid>(grid),
  //       make_shared<Eigen::VectorXcd>(defNeutronsEigenpair.first.col(j)),
  //       defNeutronsEigenpair.second(j));
  //   ms(j, (n_betas - 1) / 2) = shell.mj();
  //   L2s(j, (n_betas - 1) / 2) = shell.l();
  //   J2s(j, (n_betas - 1) / 2) = shell.j();
  //   Ps(j, (n_betas - 1) / 2) = shell.P();
  //   energies.col((n_betas - 1) / 2) = defNeutronsEigenpair.second;
  // }
  // cout << L2s << endl;

  // for (int i = 0; i < n_betas; i++) {
  //   double Beta = -0.1 * (n_betas - 1) / 2 + i * 0.1;
  //   if (i == (n_betas - 1) / 2)
  //     continue;
  //   cout << "Beta: " << Beta << endl;
  //   betas.push_back(Beta);
  //   vector<shared_ptr<Potential>> pots;
  //   Radius radius(Beta, r_0, A);
  //   pots.push_back(make_shared<DeformedSpinOrbitPotential>(
  //       DeformedSpinOrbitPotential(V0, Radius(Beta, r_0_so, A), 0.67)));
  //   pots.push_back(make_shared<DeformedWoodsSaxonPotential>(
  //       DeformedWoodsSaxonPotential(V0, Radius(Beta, r_0, A), 0.67, A,
  //                                   input.getZ(), input.getKappa())));

  //  Hamiltonian ham(make_shared<Grid>(grid), pots);
  //  string path = "output/def/";

  //  ComplexSparseMatrix ham_mat_5p = ham.build_matrix5p();
  //  eigenpair = gcgm_complex_no_B(ham_mat_5p, defNeutronsEigenpair.first,
  //                                calc.nev, 35 + 0.01, 20, calc.tol, 40,
  //                                1.0e-5 / (calc.nev), false, 1);
  //  energies.col(i) = eigenpair.second;
  //  Wavefunction::normalize(eigenpair.first, grid);
  //  for (int j = 0; j < calc.nev; ++j) {
  //    Shell shell(make_shared<Grid>(grid),
  //                make_shared<Eigen::VectorXcd>(eigenpair.first.col(j)),
  //                eigenpair.second(j));
  //    ms(j, i) = shell.mj();
  //    L2s(j, i) = shell.l();
  //    J2s(j, i) = shell.j();
  //    Ps(j, i) = shell.P();
  //  }
  //}

  // cout << "Shells computed, outputting" << endl;
  // cout << energies << endl;
  // cout << ms << endl;
  // std::string path = "output/";
  // ofstream file(path + "def_energies.csv");
  // file << energies << endl;
  //// return 0;
  // ofstream fileM(path + "m.csv");
  // fileM << ms << endl;

  // ofstream fileL2(path + "L2.csv");
  // fileL2 << L2s << endl;

  // ofstream fileJ2(path + "J2.csv");
  // fileJ2 << J2s << endl;

  // ofstream fileP(path + "P.csv");
  // fileP << Ps << endl;

  // return 0;
}
