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

using namespace std;
using namespace Eigen;
using namespace nuclearConstants;

int n = 10;
int k = 7;
double a = 10.0;
// double h = 2.0 * a / (n-1);
int acgm_cycles = 50;
const double V0 = 49.6;
const double A_val = 24;
const double r_0 = 1.347;
const double r_0_so = 1.310;
const double R = r_0 * pow(A_val, 1.0 / 3.0);

int main(int argc, char **argv)
{
    Eigen::initParallel();

    if (argc >= 2)
    {
        a = atof(argv[1]);
    }
    if (argc >= 3)
    {
        k = atoi(argv[2]);
    }
    if (argc >= 4)
    {
        acgm_cycles = atoi(argv[3]);
    }
    if (argc >= 5)
    {
        n = atoi(argv[4]);
        // h = 2.0 * a / (n-1);
    }
    ComplexDenseMatrix guess;

    Grid grid(n, a);
    cout << "Grid domain: [-" << grid.get_a() << ", " << -grid.get_a() + grid.get_h() * (grid.get_n() - 1) << "]" << endl;
    std::vector<double> betas;

    std::pair<ComplexDenseMatrix, DenseVector> eigenpair;
    vector<shared_ptr<Potential>> pots;
    //pots.push_back(make_shared<SpinOrbitPotential>(SpinOrbitPotential(V0, r_0, R)));
    //pots.push_back(make_shared<WoodsSaxonPotential>(WoodsSaxonPotential(V0, R, 0.67)));
    pots.push_back(make_shared<DeformedSpinOrbitPotential>(DeformedSpinOrbitPotential(V0, Radius(0, r_0_so, A_val), 0.7)));
    pots.push_back(make_shared<DeformedWoodsSaxonPotential>(DeformedWoodsSaxonPotential(V0, Radius(0, r_0, A_val), 0.7)));

    Hamiltonian ham(make_shared<Grid>(grid), pots);
    eigenpair = gcgm_complex_no_B(ham.build_matrix5p(), gaussian_guess(grid, k, a), k, 35 + 0.01, acgm_cycles, 1.0e-3, 40, 1.0e-4 / (k), false, 1);
    //return 0;
    pots.clear();
    guess = eigenpair.first(all, seq(0, k - 1));

    //for(int i = 0; i < guess.cols(); ++i) {
        //cout << "m: " << Operators::Jz(guess(all, i), grid).norm()/h_bar<< endl;
        ////cout << "m: " << Operators::JzExp(guess(all, i), grid)/h_bar<< endl;

    //}
    //return 0;

    int n_betas = 11;
    DenseMatrix energies(k, n_betas);
    DenseMatrix ms(k, n_betas);
    int i = 0;
    for (double Beta = -0.5; Beta < 0.55; Beta += 0.1)
    {
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

        if (argc >= 6)
        {
            cout << "Electric charge " << (A_val / 2) << endl;
            pots.push_back(make_shared<SphericalCoulombPotential>(SphericalCoulombPotential(A_val / 2, R)));
            cout << "Adding coulomb potential" << endl;
        }

        Hamiltonian ham(make_shared<Grid>(grid), pots);
        string path = "output/wd_3d/";

        guess = eigenpair.first(all, seq(0, k - 1));

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

        eigenpair = gcgm_complex_no_B(ham_mat_5p, guess, k, 35 + 0.01, 10, 1.0e-3, 40, 1.0e-4 / (k), false, 1);
        end = std::chrono::steady_clock::now();
        for(int j = 0; j < k; ++j) {
            double m = Operators::Jz(eigenpair.first(all, j), grid).norm()/h_bar;
            ms(j, i) = m;
            cout << "m: " << m << endl;
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

        auto ham_time_no_B_5p = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();

        cout << "Time elapsed 5 points: " << ham_time_no_B_5p << "[ms]" << endl;

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
