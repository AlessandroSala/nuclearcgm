#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <omp.h>
#include <fstream>
#include "constants.hpp"
#include "solver/solver.cpp"

using namespace std;
using namespace Eigen;
using namespace nuclearConstants;

int n = 10;
int k = 7;
int acgm_cycles = 50;
const double a = 10.0;
const double h = 2 * a / (n-1);
const double V0 = 51.0;
const double A_val = 16;
const double R = 1.27 * pow(A_val, 1.0/3.0);
const double diff = 0.67;
const int max_iter = 5000;
const double tol = 1e-29;
const double omega = 41*pow(A_val, -1.0/3.0)/h_bar;


inline int idx(int i, int j, int k) {
    return i + n * (j + n * k);
}

double pot(double x, double y, double z) {
    return -V0 / (1 + exp((sqrt(x*x + y*y + z*z) - R) / diff));
}
double A(int i, int j, int k, int i1, int j1, int k1, const vector<double>& xs, const vector<double>& ys, const vector<double>& zs) {
    if (i1 == i && j1 == j && k1 == k)
        return (pot(xs[i], ys[j], zs[k])*C*h*h - 6) / (C * h * h);
    else if ((i == i1 && k == k1 && (j == j1 + 1 || j == j1 - 1)) ||
             (j == j1 && k == k1 && (i == i1 + 1 || i == i1 - 1)) ||
             (i == i1 && j == j1 && (k == k1 + 1 || k == k1 - 1)))
        return 1 / (C * h * h);
    
    return 0;
}
SparseMatrix<double> matsetup_oscillator(int n, const vector<double>& xs, const vector<double>& ys, const vector<double>& zs) {
    SparseMatrix<double> mat(n * n * n, n * n * n);
    vector<Triplet<double>> tripletList;

    #pragma omp parallel for collapse(3)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                for (int i1 = max(0, i-1); i1 <= min(n-1, i+1); ++i1) {
                    for (int j1 = max(0, j-1); j1 <= min(n-1, j+1); ++j1) {
                        for (int k1 = max(0, k-1); k1 <= min(n-1, k+1); ++k1) {
                            int n0 = idx(i, j, k);
                            int n1 = idx(i1, j1, k1);
                            double val = A(i, j, k, i1, j1, k1, xs, ys, zs);
                            if (val != 0) {
                                #pragma omp critical
                                tripletList.emplace_back(n0, n1, val);
                            }
                        }
                    }
                }
            }
        }
    }
    mat.setFromTriplets(tripletList.begin(), tripletList.end());
    return mat;
}

int main(int argc, char** argv) {

    if(argc >= 1) {
        n = atoi(argv[1]);

    } 
    if(argc >= 2) {
        k = atoi(argv[2]);
    } 
    if(argc >= 3) {
        acgm_cycles = atoi(argv[3]);
    }

    vector<double> xs(n), ys(n), zs(n);
    string path = "output/wd_3d/";

    for (int i = 0; i < n; ++i) {
        xs[i] = ys[i] = zs[i] = -a + i * h;
    }
    
    SparseMatrix<double> mat = matsetup_oscillator(n, xs, ys, zs);
    cout << "Matrix generated" << endl;

    int N = mat.rows();
    
    pair<double, VectorXd> eigenpair = find_eigenpair_constrained(mat, VectorXd::Random(N).normalized(), 200, 1e-10);

    cout << "Smallest eigenvalue (RCGM): " << eigenpair.first << endl;

    SparseMatrix<double> B(N, N);
    B.setIdentity();
    pair<MatrixXd, VectorXd> eigenpairs = gcgm(mat, B, random_orthonormal_matrix(N, k), k, 35 + 0.01, acgm_cycles, 1.0e-9, 50 ); 

    std::cout << "GCGM eigenvalues: " << eigenpairs.second << std::endl;


    double e_real = -31.091;
    cout << "Real energy: " << e_real << " MeV" << endl;
    cout << "Error: " << ((eigenpair.first)/e_real - 1)*100 << "%" << endl;


    ofstream file(path + "eigenvectors.txt");
    for (int i = 0; i < eigenpair.second.size(); ++i) {
        file << eigenpair.second(i) << endl;
    }
    file.close();

    ofstream file2(path + "x.txt");
    for(const auto &e : xs) {
        file2 << e << endl;
    }
    file2.close();
    //cout << MatrixXd(mat) << endl;
    


    
    return 0;
}
