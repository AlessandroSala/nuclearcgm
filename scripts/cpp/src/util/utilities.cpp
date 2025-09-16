#include "util/utilities.hpp"
#include "hamiltonian.hpp"
#include "input_parser.hpp"
#include "kinetic/local_kinetic_potential.hpp"
#include "kinetic/non_local_kinetic_potential.hpp"
#include "skyrme/local_coulomb_potential.hpp"
#include "skyrme/skyrme_so.hpp"
#include "skyrme/skyrme_u.hpp"
#include "solver.hpp"

std::pair<Eigen::MatrixXcd, Eigen::VectorXd>
Utilities::solve(const ComplexSparseMatrix &matrix, GCGParameters &calc,
                 const Eigen::MatrixXcd &guess) {
  return gcgm_complex_no_B_lock(matrix, guess, guess.cols(), 0.0, calc.maxIter,
                                calc.tol, calc.steps, (calc.cgTol), false);
}

void Utilities::skyrmeHamiltonian(std::vector<std::shared_ptr<Potential>> &pots,
                                  InputParser input, NucleonType t,
                                  std::shared_ptr<IterationData> data) {
  using std::make_shared;
  auto grid = Grid::getInstance();
  pots.push_back(make_shared<SkyrmeU>(input.skyrme, t, data));
  pots.push_back(make_shared<NonLocalKineticPotential>(data, t));
  pots.push_back(make_shared<LocalKineticPotential>(data, t));
  if (input.spinOrbit)
    pots.push_back(make_shared<SkyrmeSO>(data, t));

  if (t == NucleonType::P && input.useCoulomb) {
    pots.push_back(make_shared<LocalCoulombPotential>(data->UCoul));
  }
}

// pots.push_back(
//     make_shared<WoodsSaxonPotential>(V0, r_0 * pow(A, 1.0 / 3.0),
//     0.67));
// pots.push_back(make_shared<DeformedWoodsSaxonPotential>(
//    V0, Radius(0.000, r_0, A), 0.67, A, input.getZ(),
//    input.getKappa()));
// pots.push_back(
//    make_shared<DeformedSpinOrbitPotential>(V0, Radius(0.000, r_0,
//    A), 0.67));

// Hamiltonian hamDef = Hamiltonian(make_shared<Grid>(grid), pots);
//// cout << ham.buildMatrix() << endl;
// pair<MatrixXcd, VectorXd> defNeutronsEigenpair = gcgm_complex_no_B(
//     hamDef.build_matrix7p(),
//     harmonic_oscillator_guess(grid, calc.nev, grid.get_a()),
//     calc.nev, 35 + 0.01, calc.cycles, calc.tol * N, 40, 5.0e-8,
//     false, 1);

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
//       DeformedSpinOrbitPotential(V0, Radius(Beta, r_0_so, A),
//       0.67)));
//   pots.push_back(make_shared<DeformedWoodsSaxonPotential>(
//       DeformedWoodsSaxonPotential(V0, Radius(Beta, r_0, A), 0.67,
//       A,
//                                   input.getZ(),
//                                   input.getKappa())));

//  Hamiltonian ham(make_shared<Grid>(grid), pots);
//  string path = "output/def/";

//  ComplexSparseMatrix ham_mat_5p = ham.build_matrix5p();
//  eigenpair = gcgm_complex_no_B(ham_mat_5p,
//  defNeutronsEigenpair.first,
//                                calc.nev, 35 + 0.01, 20, calc.tol,
//                                40, 1.0e-5 / (calc.nev), false, 1);
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
