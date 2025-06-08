#include "util/integral.hpp"
#include "constants.hpp"
#include "kinetic/non_local_kinetic_potential.hpp"
#include "operators/differential_operators.hpp"
#include "grid.hpp"
#include "guess.hpp"
#include "hamiltonian.hpp"
#include "harmonic_oscillator.hpp"
#include "input_parser.hpp"
#include "kinetic/local_kinetic_potential.hpp"
#include "operators/angular_momentum.hpp"
#include "operators/common_operators.hpp"
#include "radius.hpp"
#include "solver.hpp"
#include "spherical_coulomb.hpp"
#include "spin_orbit.hpp"
#include "spin_orbit/deformed_spin_orbit.hpp"
#include "types.hpp"
#include "util/mass.hpp"
#include "util/wavefunction.hpp"
#include "woods_saxon.hpp"
#include "woods_saxon/deformed_woods_saxon.hpp"
#include "json/json.hpp"
#include <chrono>
#include <cmath>
#include <fstream>
#include <vector>
#include <skyrme/skyrme_u.hpp>
#include <skyrme/skyrme_so.hpp>
#include "util/output.hpp"
#include "skyrme/local_coulomb_potential.hpp"
#include "skyrme/exchange_coulomb_potential.hpp"
#include "util/iteration_data.hpp"

int main(int argc, char **argv)
{
    using namespace std;
    using namespace Eigen;
    using namespace nuclearConstants;
    // Eigen::initParallel();
    InputParser input("input/input.json");
    ComplexDenseMatrix guess;
    Output out;



    Grid grid = input.get_grid();
    Calculation calc = input.getCalculation();


    std::vector<double> betas;
    int A = input.getA();
    int Z = input.getZ();
    int N = A - Z;
    double V0 = (input.get_json())["potentials"][1]["V0"];
    double r_0 = (input.get_json())["potentials"][1]["r0"];
    double r_0_so = (input.get_json())["potentials"][0]["r0"];
    std::cout << input.skyrme.t0 << " " << input.skyrme.t1 << " " << input.skyrme.t2 << " " << input.skyrme.t3 << " " << input.skyrme.W0 << std::endl;

    std::pair<ComplexDenseMatrix, DenseVector> eigenpair;
    vector<shared_ptr<Potential>> pots;
    //pots.push_back(make_shared<DeformedSpinOrbitPotential>(
        //DeformedSpinOrbitPotential(V0, Radius(0, (input.get_json())["potentials"][0]["r0"], A), 0.7)));
    pots.push_back(make_shared<DeformedWoodsSaxonPotential>(
        DeformedWoodsSaxonPotential(V0, Radius(0, r_0, A), 0.7, A, input.getZ(), input.getKappa())));
    std::chrono::steady_clock::time_point begin =
        std::chrono::steady_clock::now();

    Hamiltonian hamNoKin(make_shared<Grid>(grid), pots);
    // cout << ham.buildMatrix() << endl;
    pair<MatrixXcd, VectorXd> neutronsEigenpair = gcgm_complex_no_B(
        hamNoKin.build_matrix5p(),
        harmonic_oscillator_guess(grid, N, grid.get_a()), calc.nev,
        35 + 0.01, calc.cycles, 1.0e-3, 20, 1.0e-4 / (calc.nev), false, 1);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    pots.push_back(make_shared<SphericalCoulombPotential>(SphericalCoulombPotential(Z, r_0 * pow(A, 1.0 / 3.0))));
    hamNoKin = Hamiltonian(make_shared<Grid>(grid), pots);

    pair<MatrixXcd, VectorXd> protonsEigenpair = gcgm_complex_no_B(
        hamNoKin.build_matrix5p(),
        harmonic_oscillator_guess(grid, Z, grid.get_a()), calc.nev,
        35 + 0.01, calc.cycles, 1.0e-3, 20, 1.0e-4 / (calc.nev), false, 1);
    // std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();


    for (int i = 0; i < calc.hfCycles; ++i)
    {
        vector<shared_ptr<Potential>> pots;
        IterationData data;
        data.updateQuantities(neutronsEigenpair.first, protonsEigenpair.first, A, Z, grid);
        cout << "Density integrated: " << Integral::wholeSpace(data.rhoN->array(), grid) << endl;
        cout << "Density integrated protons: " << Integral::wholeSpace(data.rhoP->array(), grid) << endl;
        cout << "Kinetic density integrated: " << Integral::wholeSpace(data.tauN->array(), grid) << endl;


        Mass hfMassN(make_shared<Grid>(grid), make_shared<IterationData>(data), input.skyrme, NucleonType::N);

        pots.push_back(make_shared<LocalKineticPotential>(make_shared<Mass>(hfMassN)));
        pots.push_back(make_shared<NonLocalKineticPotential>(make_shared<Mass>(hfMassN)));
        pots.push_back(make_shared<SkyrmeU>(input.skyrme, NucleonType::N, make_shared<IterationData>(data)));
        //pots.push_back(make_shared<SkyrmeSO>(make_shared<IterationData>(data), NucleonType::N));

        Hamiltonian skyrmeHam(make_shared<Grid>(grid), pots);
        auto newNeutronsEigenpair = gcgm_complex_no_B(
            skyrmeHam.buildMatrix(),
            neutronsEigenpair.first, N, 35 + 0.01, calc.cycles*2, 1.0e-3, 20, 1.0e-4 / (calc.nev), false, 1);

        pots.clear();
        pots.push_back(make_shared<LocalKineticPotential>(make_shared<Mass>(hfMassN)));
        pots.push_back(make_shared<NonLocalKineticPotential>(make_shared<Mass>(hfMassN)));
        pots.push_back(make_shared<SkyrmeU>(input.skyrme, NucleonType::P, make_shared<IterationData>(data)));
        //pots.push_back(make_shared<SkyrmeSO>(make_shared<IterationData>(data), NucleonType::P));
        
        pots.push_back(make_shared<LocalCoulombPotential>(data.rhoP));
        pots.push_back(make_shared<ExchangeCoulombPotential>(data.rhoP));
        skyrmeHam = Hamiltonian(make_shared<Grid>(grid), pots);
        auto newProtonsEigenpair = gcgm_complex_no_B(
            skyrmeHam.buildMatrix(),
            protonsEigenpair.first, Z, 35 + 0.01, calc.cycles*2, 1.0e-3, 20, 1.0e-4 / (calc.nev), false, 1);

        cout << "Residual: " << (newNeutronsEigenpair.first - neutronsEigenpair.first).norm() << endl;
        cout << "Residual: " << (newProtonsEigenpair.first - protonsEigenpair.first).norm() << endl;

         Wavefunction::printShells(neutronsEigenpair.first, grid);
        // return 0;
        neutronsEigenpair = newNeutronsEigenpair;
        protonsEigenpair = newProtonsEigenpair;
    }
    out.matrixToFile("density.csv", Wavefunction::density(neutronsEigenpair.first, grid) + Wavefunction::density(protonsEigenpair.first, grid));
    return 0;
    Wavefunction::printShells(eigenpair.first, grid);
    auto ham_time_no_B_5p =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - begin)
            .count();

    cout << "Time elapsed 5 points: " << ham_time_no_B_5p << "[ms]" << endl;

    return 0;
    pots.clear();
    guess = eigenpair.first(all, seq(0, calc.nev - 1));

    // for(int i = 0; i < guess.cols(); ++i) {
    // cout << "m: " << Operators::Jz(guess(all, i), grid).norm()/h_bar<< endl;
    ////cout << "m: " << Operators::JzExp(guess(all, i), grid)/h_bar<< endl;

    //}
    // return 0;

    int n_betas = 11;
    DenseMatrix energies(calc.nev, n_betas);
    DenseMatrix ms(calc.nev, n_betas);
    DenseMatrix J2s(calc.nev, n_betas);
    DenseMatrix L2s(calc.nev, n_betas);
    DenseMatrix Ps(calc.nev, n_betas);
    int i = 0;
    for (double Beta = -0.5; Beta < 0.55; Beta += 0.1)
    {
        cout << "Beta: " << Beta << endl;
        betas.push_back(Beta);
        vector<shared_ptr<Potential>> pots;
        // pots.push_back(make_shared<SpinOrbitPotential>(SpinOrbitPotential(V0,
        // r_0, R)));
        Radius radius(Beta, r_0, A);
        pots.push_back(make_shared<DeformedSpinOrbitPotential>(
            DeformedSpinOrbitPotential(V0, Radius(Beta, r_0_so, A), 0.7)));
        pots.push_back(make_shared<DeformedWoodsSaxonPotential>(
            DeformedWoodsSaxonPotential(V0, Radius(Beta, r_0, A), 0.7, A, input.getZ(), input.getKappa())));
        // pots.push_back(make_shared<WoodsSaxonPotential>(WoodsSaxonPotential(V0,
        // R, 0.67))); double omega = 41*pow(A_val, -1.0/3.0)/h_bar; omega = omega;
        // double omega_y = omega * 4;
        // double omega_z = omega * 7.8;
        // cout << "r HO: " << pow((h_bar)/(m*omega), 0.5) << endl;
        // double epsilon = 0.1;
        // pots.push_back(make_shared<HarmonicOscillatorPotential>(HarmonicOscillatorPotential(omega,
        // omega_y, omega_z)));

        Hamiltonian ham(make_shared<Grid>(grid), pots);
        string path = "output/wd_3d/";

        guess = eigenpair.first(all, seq(0, calc.nev - 1));

        // guess = anisotropic_gaussian_guess(grid, k, a/4, a/2, a);
        // guess = random_orthonormal_matrix(grid.get_total_points(), k);

        ComplexSparseMatrix ham_mat_5p = ham.build_matrix5p();
        cout << "5 point matrix generated" << endl;
        // cout << ham_mat << endl;
        // SelfAdjointEigenSolver<ComplexSparseMatrix> eigensolver(ham_mat);
        // cout << "Exact eigenvalues: " << eigensolver.eigenvalues().transpose() <<
        // endl;

        std::chrono::steady_clock::time_point begin =
            std::chrono::steady_clock::now();
        std::chrono::steady_clock::time_point end =
            std::chrono::steady_clock::now();

        begin = std::chrono::steady_clock::now();

        eigenpair = gcgm_complex_no_B(ham_mat_5p, guess, calc.nev, 35 + 0.01, 10,
                                      1.0e-3, 40, 1.0e-4 / (calc.nev), false, 1);
        end = std::chrono::steady_clock::now();
        for (int j = 0; j < calc.nev; ++j)
        {
            double m = Operators::Jz(eigenpair.first(all, j), grid).norm() / h_bar;
            auto L2 = Operators::L2(eigenpair.first.col(j), grid)
                          .dot(eigenpair.first.col(j)) /
                      (h_bar * h_bar);
            auto J2 = eigenpair.first.col(j).dot(
                          Operators::J2(eigenpair.first.col(j), grid)) /
                      (h_bar * h_bar);
            auto P = eigenpair.first.col(j).dot(
                Operators::P(eigenpair.first.col(j), grid));

            ms(j, i) = m;
            L2s(j, i) = L2.real();
            J2s(j, i) = J2.real();
            Ps(j, i) = (P / norm(P)).real();

            cout << "m: " << m << " | ";
            cout << "L2: " << L2 << " | ";
            cout << "J2: " << J2 << " | ";
            cout << "P: " << P.real() << " | ";

            // cout << "m: " << Operators::JzExp(guess(all, i), grid)/h_bar<< endl;
        }
        // if(i==2) return 0;

        // double gs = h_bar * 0.5 * (omega + omega_y + omega_z);
        // cout << h_bar*omega << " " << h_bar*omega_y << " " << h_bar*omega_z <<
        // endl; cout << gs << endl; cout << gs + h_bar*omega << endl; cout << gs +
        // h_bar*omega*2 << endl; cout << gs + h_bar*omega_y << endl; cout << gs +
        // h_bar*omega_z << endl; cout << gs + h_bar*omega + h_bar*omega_y << endl;

        energies.col(i) = eigenpair.second;
        ++i;
    }

    cout << energies << endl;
    for (auto &b : betas)
    {
        cout << b << " ";
    }
    cout << endl;

    // return 0;

    std::string path = "output/";
    ofstream file(path + "def.csv");
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

    // file.close();
    //}

    // ofstream file2(path + "x.txt");
    // for(const auto &e : xs) {
    // file2 << e << endl;
    //}
    // file2.close();
    // cout << MatrixXd(mat) << endl;

    return 0;
}
