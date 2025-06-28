#include "constants.hpp"
#include "grid.hpp"
#include "guess.hpp"
#include "hamiltonian.hpp"
#include "harmonic_oscillator.hpp"
#include "input_parser.hpp"
#include "kinetic/local_kinetic_potential.hpp"
#include "kinetic/non_local_kinetic_potential.hpp"
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
#include <chrono>
#include <cmath>
#include <fstream>
#include <skyrme/skyrme_so.hpp>
#include <skyrme/skyrme_u.hpp>
#include <vector>

int main(int argc, char **argv) {
  using namespace std;
  using namespace Eigen;
  using namespace nuclearConstants;
  cout << "Using mass " << m << endl;
  cout << "Using C " << C << endl;
  // Eigen::initParallel();
  InputParser input("input/input.json");
  ComplexDenseMatrix guess;
  Output out;

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

  // pots.push_back(
  //     make_shared<WoodsSaxonPotential>(V0, r_0 * pow(A, 1.0 / 3.0), 0.67));
  pots.push_back(make_shared<DeformedWoodsSaxonPotential>(
      V0, Radius(0.000, r_0, A), 0.67, A, input.getZ(), input.getKappa()));
  pots.push_back(make_shared<DeformedSpinOrbitPotential>(
      V0, Radius(0.000, r_0_so, A), 0.67));

  Hamiltonian hamDef = Hamiltonian(make_shared<Grid>(grid), pots);
  // cout << ham.buildMatrix() << endl;
  pair<MatrixXcd, VectorXd> defNeutronsEigenpair = gcgm_complex_no_B(
      hamDef.build_matrix5p(),
      harmonic_oscillator_guess(grid, calc.nev, grid.get_a()), calc.nev,
      35 + 0.01, calc.cycles, calc.tol * N, 40, 5.0e-8, false, 1);

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

  pots.clear();

  int n_betas = 3;
  DenseMatrix energies(calc.nev, n_betas);
  DenseMatrix ms(calc.nev, n_betas);
  DenseMatrix J2s(calc.nev, n_betas);
  DenseMatrix L2s(calc.nev, n_betas);
  DenseMatrix Ps(calc.nev, n_betas);

  for (int j = 0; j < calc.nev; ++j) {
    Shell shell(make_shared<Grid>(grid),
                make_shared<Eigen::VectorXcd>(eigenpair.first.col(j)),
                eigenpair.second(j));
    ms(j, (n_betas - 1) / 2) = shell.mj();
    L2s(j, (n_betas - 1) / 2) = shell.l();
    J2s(j, (n_betas - 1) / 2) = shell.j();
    Ps(j, (n_betas - 1) / 2) = shell.P();
  }
  int i = 0;
  for (double Beta = -0.1 * (n_betas - 1) / 2;
       Beta < (0.1 * (n_betas - 1) / 2 + 0.01); Beta += 0.1) {
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
                                  calc.nev, 35 + 0.01, 10, calc.tol, 40,
                                  1.0e-4 / (calc.nev), false, 1);
    energies.col(i) = eigenpair.second;
    for (int j = 0; j < calc.nev; ++j) {
      Shell shell(make_shared<Grid>(grid),
                  make_shared<Eigen::VectorXcd>(eigenpair.first.col(j)),
                  eigenpair.second(j));
      ms(j, i) = shell.mj();
      L2s(j, i) = shell.l();
      J2s(j, i) = shell.j();
      Ps(j, i) = shell.P();
    }
    ++i;
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
  pots.push_back(make_shared<DeformedWoodsSaxonPotential>(
      V0, Radius(0.000, r_0, A), 0.55, A, input.getZ(), input.getKappa()));

  std::chrono::steady_clock::time_point begin =
      std::chrono::steady_clock::now();

  Hamiltonian hamNoKin(make_shared<Grid>(grid), pots);
  // cout << ham.buildMatrix() << endl;
  pair<MatrixXcd, VectorXd> neutronsEigenpair = gcgm_complex_no_B(
      hamNoKin.build_matrix5p(),
      harmonic_oscillator_guess(grid, N, grid.get_a()), N, 35 + 0.01,
      calc.cycles, calc.tol * N, 40, 5.0e-8, false, 1);

  pair<MatrixXcd, VectorXd> protonsEigenpair = neutronsEigenpair;

  Wavefunction::normalize(neutronsEigenpair.first, grid);
  Wavefunction::normalize(protonsEigenpair.first, grid);
  constexpr double alpha = 1.0;
  IterationData data(input.skyrme);

  std::vector<double> hfEnergies;

  double totalEnergy = 0.0;
  data.updateQuantities(neutronsEigenpair.first, protonsEigenpair.first, A, Z,
                        grid);
  int hfIter = 0;
  for (hfIter = 0; hfIter < calc.hf.cycles; ++hfIter) {
    vector<shared_ptr<Potential>> pots;

    Mass hfMassN(make_shared<Grid>(grid), make_shared<IterationData>(data),
                 input.skyrme, NucleonType::N);

    // pots.push_back(
    //     make_shared<LocalKineticPotential>(make_shared<Mass>(hfMassN)));
    // pots.push_back(
    //     make_shared<NonLocalKineticPotential>(make_shared<Mass>(hfMassN)));
    pots.push_back(make_shared<SkyrmeU>(input.skyrme, NucleonType::N,
                                        make_shared<IterationData>(data)));
    //  pots.push_back(make_shared<SkyrmeSO>(make_shared<IterationData>(data),
    //                                      NucleonType::N));

    Hamiltonian skyrmeHam(make_shared<Grid>(grid), pots);
    auto newNeutronsEigenpair =
        gcgm_complex_no_B(skyrmeHam.build_matrix5p(), neutronsEigenpair.first,
                          N, 35 + 0.01, calc.hf.gcg.maxIter, calc.hf.gcg.tol,
                          40, 1.0e-4 / (calc.nev), false, 1);

    Mass hfMassP(make_shared<Grid>(grid), make_shared<IterationData>(data),
                 input.skyrme, NucleonType::P);
    pots.clear();
    // pots.push_back(
    //     make_shared<LocalKineticPotential>(make_shared<Mass>(hfMassP)));
    // pots.push_back(
    //     make_shared<NonLocalKineticPotential>(make_shared<Mass>(hfMassP)));
    pots.push_back(make_shared<SkyrmeU>(input.skyrme, NucleonType::P,
                                        make_shared<IterationData>(data)));
    // pots.push_back(make_shared<SkyrmeSO>(make_shared<IterationData>(data),
    //                                      NucleonType::P));

    // pots.push_back(make_shared<LocalCoulombPotential>(data.rhoP));
    // pots.push_back(make_shared<ExchangeCoulombPotential>(data.rhoP));
    skyrmeHam = Hamiltonian(make_shared<Grid>(grid), pots);
    auto newProtonsEigenpair =
        gcgm_complex_no_B(skyrmeHam.build_matrix5p(), protonsEigenpair.first, Z,
                          35 + 0.01, calc.hf.gcg.maxIter, calc.hf.gcg.tol, 40,
                          1.0e-4 / (calc.nev), false, 1);

    // cout << "Residual: " << (newNeutronsEigenpair.first -
    // neutronsEigenpair.first).norm() << endl; cout << "Residual: " <<
    // (newProtonsEigenpair.first - protonsEigenpair.first).norm() << endl;
    neutronsEigenpair = newNeutronsEigenpair;
    protonsEigenpair = newProtonsEigenpair;

    Wavefunction::normalize(neutronsEigenpair.first, grid);
    Wavefunction::normalize(protonsEigenpair.first, grid);

    data.updateQuantities(neutronsEigenpair.first, protonsEigenpair.first, A, Z,
                          grid);
    double newEnergy = data.totalEnergy(input.skyrme, grid) +
                       data.kineticEnergy(input.skyrme, grid);
    hfEnergies.push_back(newEnergy);
    cout << "Total energy: " << newEnergy << endl;
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

  out.shellsToFile("calc_output.csv", neutronsEigenpair, protonsEigenpair,
                   make_shared<IterationData>(data), input, hfIter, hfEnergies,
                   grid);
  return 0;
  Wavefunction::printShells(eigenpair, grid);
  auto ham_time_no_B_5p =
      std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
          .count();

  cout << "Time elapsed 5 points: " << ham_time_no_B_5p << "[ms]" << endl;

  return 0;
}
