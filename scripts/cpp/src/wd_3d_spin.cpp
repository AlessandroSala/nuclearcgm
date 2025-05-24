#include "woods_saxon.hpp"
#include "woods_saxon/deformed_woods_saxon.hpp"
#include "spin_orbit/deformed_spin_orbit.hpp"
#include "radius.hpp"
#include "guess.hpp"
#include "grid.hpp"
#include <vector>
#include <cmath>
#include <fstream>
#include "solver.hpp"
#include "types.hpp"
#include "hamiltonian.hpp"
#include "spin_orbit.hpp"
#include "constants.hpp"
#include <chrono>
#include "spherical_coulomb.hpp"
#include "harmonic_oscillator.hpp"
#include "operators/angular_momentum.hpp"
#include "operators/common_operators.hpp"
#include "json/json.hpp"
#include "input_parser.hpp"
#include "util/mass.hpp"
#include "util/wavefunction.hpp"
#include "kinetic/local_kinetic_potential.hpp"

using namespace std;
using namespace Eigen;
using namespace nuclearConstants;

const double V0 = 49.6;
const double A_val = 44;
const double r_0 = 1.347;
const double r_0_so = 1.310;
const double R = r_0 * pow(A_val, 1.0 / 3.0);

int main(int argc, char **argv)
{
    Eigen::initParallel();
    InputParser input("input/input.json");
    ComplexDenseMatrix guess;

    Grid grid = input.get_grid();
    Calculation calc = input.getCalculation();
    cout << "Computing " << calc.nev << " eigenvalues" << endl;
    std::vector<double> betas;

    std::pair<ComplexDenseMatrix, DenseVector> eigenpair;
    vector<shared_ptr<Potential>> pots;
    //pots.push_back(make_shared<SpinOrbitPotential>(SpinOrbitPotential(V0, r_0, R)));
    //pots.push_back(make_shared<WoodsSaxonPotential>(WoodsSaxonPotential(V0, R, 0.7)));
    pots.push_back(make_shared<DeformedSpinOrbitPotential>(DeformedSpinOrbitPotential(V0, Radius(0, r_0_so, A_val), 0.7)));
    pots.push_back(make_shared<DeformedWoodsSaxonPotential>(DeformedWoodsSaxonPotential(V0, Radius(0, r_0, A_val), 0.7)));
    Hamiltonian ham(make_shared<Grid>(grid), pots);
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    //eigenpair = gcgm_complex_no_B(ham.build_matrix5p(), harmonic_oscillator_guess(grid, calc.nev, grid.get_a()), calc.nev, 35 + 0.01, calc.cycles, 1.0e-3, 20, 1.0e-4 / (calc.nev), false, 1);
    Eigen::VectorXd idVec(grid.get_total_points());
    idVec.setIdentity();

    Mass m(Wavefunction::density(idVec, grid), make_shared<Grid>(grid), 0.0, 0.0);
    
    pots.push_back(make_shared<LocalKineticPotential>(make_shared<Mass>(m)));
    Hamiltonian hamNoKin(make_shared<Grid>(grid), pots);
    //cout << ham.buildMatrix() << endl;
    eigenpair = gcgm_complex_no_B(hamNoKin.buildMatrix(), harmonic_oscillator_guess(grid, calc.nev, grid.get_a()), calc.nev, 35 + 0.01, calc.cycles, 1.0e-3, 20, 1.0e-4 / (calc.nev), false, 1);
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    return 0;
    for(int i = 0; i < eigenpair.first.cols(); ++i) {
        auto L2 =  Operators::L2(eigenpair.first.col(i), grid).dot(eigenpair.first.col(i))/(h_bar*h_bar);
        cout << "L2: " <<  L2.real()<< " | ";
        //cout << ( Operators::LS(eigenpair.first.col(i), grid).norm())/(h_bar*h_bar)<< " | ";
        //cout << "J2: " <<  eigenpair.first.col(i).adjoint() * Operators::J2(eigenpair.first.col(i), grid)/(h_bar*h_bar)<< endl;
        cout << "J2: " <<  (eigenpair.first.col(i).adjoint() * Operators::J2(eigenpair.first.col(i), grid)).norm()/(h_bar*h_bar)<< " | ";
        auto par =  (eigenpair.first.col(i).adjoint() * Operators::P(eigenpair.first.col(i), grid));
        cout << "P: " <<( par/par.norm()).real() << " | ";
        cout << "mz: " << Operators::Jz(eigenpair.first.col(i), grid).norm()/(h_bar) <<endl;

    }
    auto ham_time_no_B_5p = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

    cout << "Time elapsed 5 points: " << ham_time_no_B_5p << "[ms]" << endl;

    return 0;
    pots.clear();
    guess = eigenpair.first(all, seq(0, calc.nev - 1));

    //for(int i = 0; i < guess.cols(); ++i) {
        //cout << "m: " << Operators::Jz(guess(all, i), grid).norm()/h_bar<< endl;
        ////cout << "m: " << Operators::JzExp(guess(all, i), grid)/h_bar<< endl;

    //}
    //return 0;

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
        // pots.push_back(make_shared<SpinOrbitPotential>(SpinOrbitPotential(V0, r_0, R)));
        Radius radius(Beta, r_0, A_val);
        pots.push_back(make_shared<DeformedSpinOrbitPotential>(DeformedSpinOrbitPotential(V0, Radius(Beta, r_0_so, A_val), 0.7)));
        pots.push_back(make_shared<DeformedWoodsSaxonPotential>(DeformedWoodsSaxonPotential(V0, Radius(Beta, r_0, A_val), 0.7)));
        // pots.push_back(make_shared<WoodsSaxonPotential>(WoodsSaxonPotential(V0, R, 0.67)));
        // double omega = 41*pow(A_val, -1.0/3.0)/h_bar;
        // omega = omega;
        // double omega_y = omega * 4;
        // double omega_z = omega * 7.8;
        // cout << "r HO: " << pow((h_bar)/(m*omega), 0.5) << endl;
        // double epsilon = 0.1;
        // pots.push_back(make_shared<HarmonicOscillatorPotential>(HarmonicOscillatorPotential(omega, omega_y, omega_z)));


        Hamiltonian ham(make_shared<Grid>(grid), pots);
        string path = "output/wd_3d/";

        guess = eigenpair.first(all, seq(0, calc.nev - 1));

        // guess = anisotropic_gaussian_guess(grid, k, a/4, a/2, a);
        // guess = random_orthonormal_matrix(grid.get_total_points(), k);

        ComplexSparseMatrix ham_mat_5p = ham.build_matrix5p();
        cout << "5 point matrix generated" << endl;
        // cout << ham_mat << endl;
        // SelfAdjointEigenSolver<ComplexSparseMatrix> eigensolver(ham_mat);
        // cout << "Exact eigenvalues: " << eigensolver.eigenvalues().transpose() << endl;

        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();

        begin = std::chrono::steady_clock::now();

        eigenpair = gcgm_complex_no_B(ham_mat_5p, guess, calc.nev, 35 + 0.01, 10, 1.0e-3, 40, 1.0e-4 / (calc.nev), false, 1);
        end = std::chrono::steady_clock::now();
        for(int j = 0; j < calc.nev; ++j) {
            double m = Operators::Jz(eigenpair.first(all, j), grid).norm()/h_bar;
            auto L2 =  Operators::L2(eigenpair.first.col(j), grid).dot(eigenpair.first.col(j))/(h_bar*h_bar);
            auto J2 = eigenpair.first.col(j).dot(Operators::J2(eigenpair.first.col(j), grid))/(h_bar*h_bar);
            auto P = eigenpair.first.col(j).dot(Operators::P(eigenpair.first.col(j), grid));

            ms(j, i) = m;
            L2s(j, i) = L2.real();
            J2s(j, i) = J2.real();
            Ps(j, i) = (P/norm(P)).real(); 

            cout << "m: " << m << " | ";
            cout << "L2: " << L2 << " | ";
            cout << "J2: " << J2 << " | ";
            cout << "P: " << P.real() << " | ";

            //cout << "m: " << Operators::JzExp(guess(all, i), grid)/h_bar<< endl;

        }
        //if(i==2) return 0;

        // double gs = h_bar * 0.5 * (omega + omega_y + omega_z);
        // cout << h_bar*omega << " " << h_bar*omega_y << " " << h_bar*omega_z << endl;
        // cout << gs << endl;
        // cout << gs + h_bar*omega << endl;
        // cout << gs + h_bar*omega*2 << endl;
        // cout << gs + h_bar*omega_y << endl;
        // cout << gs + h_bar*omega_z << endl;
        // cout << gs + h_bar*omega + h_bar*omega_y << endl;

        energies.col(i) = eigenpair.second;
        ++i;
    }

    cout << energies << endl;
    for (auto &b : betas)
    {
        cout << b << " ";
    }
    cout << endl;
    
    //return 0;

    std::string path = "output/";
    ofstream file(path + "def.csv");
    file << energies << endl;
    //return 0;
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
