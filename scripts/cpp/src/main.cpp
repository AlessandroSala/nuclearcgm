#include "constants.hpp"
#include "grid.hpp"
#include "guess.hpp"
#include "hamiltonian.hpp"
#include "harmonic_oscillator.hpp"
#include "input_parser.hpp"
#include "kinetic/local_kinetic_potential.hpp"
#include "kinetic/non_local_kinetic_potential.hpp"
#include "operators/differential_operators.hpp"
#include "operators/integral_operators.hpp"
#include "radius.hpp"
#include "skyrme/exchange_coulomb_potential.hpp"
#include "skyrme/local_coulomb_potential.hpp"
#include "solver.hpp"
#include "spherical_coulomb.hpp"
#include "spin_orbit.hpp"
#include "spin_orbit/deformed_spin_orbit.hpp"
#include "types.hpp"
#include "util/iteration_data.hpp"
#include "util/mass.hpp"
#include "util/output.hpp"
#include "util/shell.hpp"
#include "util/wavefunction.hpp"
#include "woods_saxon.hpp"
#include "woods_saxon/deformed_woods_saxon.hpp"
#include "json/json.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <memory>
#include <skyrme/skyrme_so.hpp>
#include <skyrme/skyrme_u.hpp>
#include <vector>

const double TOL = 1e-6;
const int MAX_ITER = 1000;

// Funzione per risolvere lambda imponendo N = 2 * sum v_i^2
double find_lambda(const Eigen::VectorXd &epsilon, const double Delta,
                   const int N_particles) {
  double lambda_low = epsilon(0) - 5;
  double lambda_high = epsilon(epsilon.size() - 1) + 5;
  double lambda = 0.0;

  for (int iter = 0; iter < 100; ++iter) {
    lambda = 0.5 * (lambda_low + lambda_high);
    double N_calc = 0.0;

    for (double e : epsilon) {
      double E = sqrt((e - lambda) * (e - lambda) + Delta * Delta);
      double v2 = 0.5 * (1.0 - (e - lambda) / E);
      N_calc += 2.0 * v2;
    }

    if (abs(N_calc - N_particles) < TOL)
      break;

    if (N_calc > N_particles)
      lambda_high = lambda;
    else
      lambda_low = lambda;
  }

  return lambda;
}
int main(int argc, char **argv) {
  using namespace std;
  using namespace Eigen;
  using namespace nuclearConstants;
  cout << "Using mass " << m << endl;
  cout << "Using C " << C << endl;
  Eigen::initParallel();
  InputParser input("input/input.json");
  ComplexDenseMatrix guess;
  Output out;

  const double G = 0.4;

  std::chrono::steady_clock::time_point computationBegin =
      std::chrono::steady_clock::now();

  Grid grid = input.get_grid();
  Calculation calc = input.getCalculation();
  calc.tol = grid.get_total_spatial_points() * 5e-9 * calc.tol;
  cout << "GCG tolerance: " << calc.tol << endl;
  std::vector<double> betas;
  int A = input.getA();
  int Z = input.getZ();
  int N = A - Z;
  double V0 = (input.get_json())["potentials"][1]["V0"];
  double r_0 = (input.get_json())["potentials"][1]["r0"];
  double r_0_so = (input.get_json())["potentials"][0]["r0"];
  std::cout << input.skyrme.t0 << " " << input.skyrme.t1 << " "
            << input.skyrme.t2 << " " << input.skyrme.t3 << " "
            << input.skyrme.W0 << std::endl;

  std::pair<ComplexDenseMatrix, DenseVector> eigenpair;
  vector<shared_ptr<Potential>> pots;

  pots.push_back(make_shared<DeformedWoodsSaxonPotential>(
      V0, Radius(0.000, r_0, A), 0.67, A, input.getZ(), input.getKappa()));
  if (input.spinOrbit) {
    pots.push_back(make_shared<DeformedSpinOrbitPotential>(
        V0, Radius(0.000, r_0, A), 0.55));
  }

  std::chrono::steady_clock::time_point begin =
      std::chrono::steady_clock::now();

  Hamiltonian hamNoKin(make_shared<Grid>(grid), pots);
  // cout << ham.buildMatrix() << endl;
  //

  double h_omega = 41.0 / (pow(A, 0.33333));
  pair<MatrixXcd, VectorXd> neutronsEigenpair = gcgm_complex_no_B(
      hamNoKin.build_matrix5p(),
      harmonic_oscillator_guess(grid, N + input.additional, grid.get_a()),
      N + input.additional, 35 + 0.01, calc.cycles, 0.0001, grid.get_n() * 1.2,
      1.0e-7 / grid.get_n(), false, 1);
  // calc.tol * N

  pair<MatrixXcd, VectorXd> protonsEigenpair = neutronsEigenpair;

  Wavefunction::normalize(neutronsEigenpair.first, grid);
  Wavefunction::normalize(protonsEigenpair.first, grid);

  IterationData data(input);

  std::vector<double> hfEnergies;

  Eigen::VectorXd vksN = Eigen::VectorXd::Zero(neutronsEigenpair.first.cols());
  Eigen::VectorXd vksP = Eigen::VectorXd::Zero(protonsEigenpair.first.cols());
  for (int i = 0; i < N; ++i) {
    vksN(i) = 1.0;
  }
  for (int i = 0; i < Z; ++i) {
    vksP(i) = 1.0;
  }

  double totalEnergy = 0.0;
  data.updateQuantities(neutronsEigenpair.first, protonsEigenpair.first, vksN,
                        vksP, grid);

  int hfIter = 0;
  for (hfIter = 0; hfIter < calc.hf.cycles; ++hfIter) {
    vector<shared_ptr<Potential>> pots;
    int maxIterGCGHF = calc.hf.gcg.maxIter;
    if (hfIter == 0) {
      maxIterGCGHF = 30;
    } else {
      maxIterGCGHF = 14 - (calc.hf.gcg.maxIter % 5);
    }

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

    auto newNeutronsEigenpair =
        gcgm_complex_no_B(skyrmeHam.buildMatrix(), neutronsEigenpair.first,
                          N + input.additional, 35 + 0.01, maxIterGCGHF,
                          calc.hf.gcg.tol, 40, 1.0e-4 / (calc.nev), false, 1);

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

    pair<MatrixXcd, VectorXd> newProtonsEigenpair;
    if (input.useCoulomb) {
      std::cout << "Protons " << std::endl;
      pots.push_back(make_shared<LocalCoulombPotential>(data.UCoul));
      // pots.push_back(make_shared<ExchangeCoulombPotential>(data.rhoP));
      Hamiltonian skyrmeHam(make_shared<Grid>(grid), pots);
      newProtonsEigenpair = gcgm_complex_no_B(
          skyrmeHam.buildMatrix(), protonsEigenpair.first, Z, 35 + 0.01,
          maxIterGCGHF, calc.hf.gcg.tol, 40, 1.0e-4 / (calc.nev), false, 1);
      skyrmeHam = Hamiltonian(make_shared<Grid>(grid), pots);

    } else {
      newProtonsEigenpair = newNeutronsEigenpair;
    }

    neutronsEigenpair = newNeutronsEigenpair;
    protonsEigenpair = newProtonsEigenpair;

    Wavefunction::normalize(neutronsEigenpair.first, grid);
    Wavefunction::normalize(protonsEigenpair.first, grid);

    data.updateQuantities(neutronsEigenpair.first, protonsEigenpair.first, vksN,
                          vksP, grid);
    double newEnergy = data.totalEnergyIntegral(input.skyrme, grid) +
                       data.kineticEnergy(input.skyrme, grid);

    double kinEn = data.kineticEnergy(input.skyrme, grid);
    double Erea = data.totalEnergyIntegral(input.skyrme, grid) -
                  0.5 * data.densityUVPIntegral(grid);
    double Ekin = kinEn * 0.5;
    double SPE =
        0.5 * ((neutronsEigenpair.second.array() * vksN.array()).sum() +
               (protonsEigenpair.second.array() * vksP.array()).sum());
    cout << "SPE calc: " << SPE << endl;
    cout << "Esg: " << data.Hsg(input.skyrme, grid) << endl;
    cout << "Eso: " << data.Hso(input.skyrme, grid) << endl;

    cout << "E (REA): " << Erea << endl;
    cout << "E (SPE): " << SPE << endl;
    cout << "E kin: " << Ekin << endl;

    hfEnergies.push_back(newEnergy);
    cout << "Total energy as integral: " << newEnergy << endl;
    cout << "Total energy as HF energy: " << data.HFEnergy(SPE * 2.0, grid)
         << endl;
    if (std::abs(newEnergy - totalEnergy) <
        input.getCalculation().hf.energyTol) {
      break;
    }
    totalEnergy = newEnergy;
  }
  cout << "Neutrons " << endl;
  Wavefunction::printShells(neutronsEigenpair, grid);
  cout << "Protons " << endl;
  Wavefunction::printShells(protonsEigenpair, grid);

  std::chrono::steady_clock::time_point computationEnd =
      std::chrono::steady_clock::now();
  double cpuTime = std::chrono::duration_cast<std::chrono::seconds>(
                       computationEnd - computationBegin)
                       .count();
  out.shellsToFile("calc_output.csv", neutronsEigenpair, protonsEigenpair,
                   make_shared<IterationData>(data), input, hfIter, hfEnergies,
                   cpuTime, grid);
  return 0;
  // pots.push_back(
  //     make_shared<WoodsSaxonPotential>(V0, r_0 * pow(A, 1.0 / 3.0), 0.67));
  pots.push_back(make_shared<DeformedWoodsSaxonPotential>(
      V0, Radius(0.000, r_0, A), 0.67, A, input.getZ(), input.getKappa()));
  pots.push_back(
      make_shared<DeformedSpinOrbitPotential>(V0, Radius(0.000, r_0, A), 0.67));

  Hamiltonian hamDef = Hamiltonian(make_shared<Grid>(grid), pots);
  // cout << ham.buildMatrix() << endl;
  pair<MatrixXcd, VectorXd> defNeutronsEigenpair = gcgm_complex_no_B(
      hamDef.build_matrix7p(),
      harmonic_oscillator_guess(grid, calc.nev, grid.get_a()), calc.nev,
      35 + 0.01, calc.cycles, calc.tol * N, 40, 5.0e-8, false, 1);

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

  pots.clear();

  Wavefunction::normalize(defNeutronsEigenpair.first, grid);
  int n_betas = 11;
  DenseMatrix energies(calc.nev, n_betas);
  DenseMatrix ms(calc.nev, n_betas);
  DenseMatrix J2s(calc.nev, n_betas);
  DenseMatrix L2s(calc.nev, n_betas);
  DenseMatrix Ps(calc.nev, n_betas);

  for (int j = 0; j < calc.nev; ++j) {
    Shell shell(
        make_shared<Grid>(grid),
        make_shared<Eigen::VectorXcd>(defNeutronsEigenpair.first.col(j)),
        defNeutronsEigenpair.second(j));
    ms(j, (n_betas - 1) / 2) = shell.mj();
    L2s(j, (n_betas - 1) / 2) = shell.l();
    J2s(j, (n_betas - 1) / 2) = shell.j();
    Ps(j, (n_betas - 1) / 2) = shell.P();
    energies.col((n_betas - 1) / 2) = defNeutronsEigenpair.second;
  }
  cout << L2s << endl;

  for (int i = 0; i < n_betas; i++) {
    double Beta = -0.1 * (n_betas - 1) / 2 + i * 0.1;
    if (i == (n_betas - 1) / 2)
      continue;
    cout << "Beta: " << Beta << endl;
    betas.push_back(Beta);
    vector<shared_ptr<Potential>> pots;
    Radius radius(Beta, r_0, A);
    pots.push_back(make_shared<DeformedSpinOrbitPotential>(
        DeformedSpinOrbitPotential(V0, Radius(Beta, r_0_so, A), 0.67)));
    pots.push_back(make_shared<DeformedWoodsSaxonPotential>(
        DeformedWoodsSaxonPotential(V0, Radius(Beta, r_0, A), 0.67, A,
                                    input.getZ(), input.getKappa())));

    Hamiltonian ham(make_shared<Grid>(grid), pots);
    string path = "output/def/";

    ComplexSparseMatrix ham_mat_5p = ham.build_matrix5p();
    eigenpair = gcgm_complex_no_B(ham_mat_5p, defNeutronsEigenpair.first,
                                  calc.nev, 35 + 0.01, 20, calc.tol, 40,
                                  1.0e-5 / (calc.nev), false, 1);
    energies.col(i) = eigenpair.second;
    Wavefunction::normalize(eigenpair.first, grid);
    for (int j = 0; j < calc.nev; ++j) {
      Shell shell(make_shared<Grid>(grid),
                  make_shared<Eigen::VectorXcd>(eigenpair.first.col(j)),
                  eigenpair.second(j));
      ms(j, i) = shell.mj();
      L2s(j, i) = shell.l();
      J2s(j, i) = shell.j();
      Ps(j, i) = shell.P();
    }
  }

  cout << "Shells computed, outputting" << endl;
  cout << energies << endl;
  cout << ms << endl;
  std::string path = "output/";
  ofstream file(path + "def_energies.csv");
  file << energies << endl;
  // return 0;
  ofstream fileM(path + "m.csv");
  fileM << ms << endl;

  ofstream fileL2(path + "L2.csv");
  fileL2 << L2s << endl;

  ofstream fileJ2(path + "J2.csv");
  fileJ2 << J2s << endl;

  ofstream fileP(path + "P.csv");
  fileP << Ps << endl;

  return 0;
}
