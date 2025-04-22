#include "woods_saxon.hpp"
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

using namespace std;
using namespace Eigen;
using namespace nuclearConstants;

int n = 10;
int k = 7;
double a = 10.0;
//double h = 2.0 * a / (n-1);
int acgm_cycles = 50;
const double V0 = 51.0;
const double A_val = 16;
const double r_0 = 1.27;
const double R = r_0 * pow(A_val, 1.0/3.0);


int main(int argc, char** argv) {
    Eigen::initParallel();

    if(argc >= 2) {
        a = atof(argv[1]);
    } 
    if(argc >= 3) {
        k = atoi(argv[2]);
    } 
    if(argc >= 4) {
        acgm_cycles = atoi(argv[3]);
    }
    if(argc >= 5) {
        n = atoi(argv[4]);
        //h = 2.0 * a / (n-1);
    }
    ComplexDenseMatrix guess;

    Grid grid(n, a);
    vector<shared_ptr<Potential>> pots;
    pots.push_back(make_shared<SpinOrbitPotential>(SpinOrbitPotential(V0, r_0, R)));
    pots.push_back(make_shared<WoodsSaxonPotential>(WoodsSaxonPotential(V0, R, diff)));

    if(argc >= 6) {
        pots.push_back(make_shared<SphericalCoulombPotential>(SphericalCoulombPotential(A_val/2, R)));
    }

    Hamiltonian ham(make_shared<Grid>(grid), pots);
    string path = "output/wd_3d/";

    guess = gaussian_guess(grid, k, a);
    //guess = random_orthonormal_matrix(grid.get_total_points(), k);
    
    SparseMatrix<double> B(grid.get_total_points(), grid.get_total_points());
    ComplexSparseMatrix ham_mat = ham.build_matrix();
    B.setIdentity();

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    pair<MatrixXcd, VectorXd> eigenpairs_ham_no_B = gcgm_complex_no_B(ham_mat, guess, k, 35 + 0.01, acgm_cycles, 1.0e-3, 50, 1.0e-5/k, true, 1); 
    
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    auto ham_time_no_B = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
    
    
    cout << "Time elapsed no B: " << ham_time_no_B << "[ms]" << endl;

    //std::cout << "GCGM eigenvalues: " << eigenpairs.second << std::endl;


    double e_real = -38.842;
    cout << "Real GS energy: " << e_real << " MeV" << endl;
    cout << "Error: " << ((eigenpairs_ham_no_B.second(0))/e_real - 1)*100 << "%" << endl;


    //for(int ev = 0; ev < eigenpairs.second.size(); ++ev) {

        //ofstream file(path + "eigenvectors_" + std::to_string(ev) + ".txt");
        //for (int i = 0; i < eigenpairs.first.rows(); ++i) {
            //file << eigenpairs.first(i, ev) << endl;
        //}
        
        //file.close();
    //}

    //ofstream file2(path + "x.txt");
    //for(const auto &e : xs) {
        //file2 << e << endl;
    //}
    //file2.close();
    //cout << MatrixXd(mat) << endl;
    
    

    
    return 0;
}
