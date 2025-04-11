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

int n = 100;
const double a = 10.0;
const double h = 2 * a / (n-1);
const double V0 = 57.0;
const double A_val = 16;
const double R = 1.27 * pow(A_val, 1.0/3.0);
const double diff = 0.67;
const int max_iter = 5000;
const double tol = 1e-28;
const double omega = 41*pow(A_val, -1.0/3.0)/h_bar;

inline int idx(int i, int j, int k) {
    return i + n * (j + n * k);
}

double pot_osc(double x) {
    return 0.5*(x*x)*m*omega*omega;
}

SparseMatrix<double> matsetup_oscillator(int n, const vector<double>& xs, const vector<double>& ys, const vector<double>& zs) {
    SparseMatrix<double> mat(n, n);
    vector<Triplet<double>> tripletList;

    #pragma omp parallel for collapse(1)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double val = 0;
            if(i==j)
                val = (pot_osc(xs[i])*C*h*h - 2) / (C * h * h);
            if(i==j-1 || i==j+1)
                val = 1/(C*h*h);

            if (val != 0) {
                #pragma omp critical
                tripletList.emplace_back(i, j, val);
            }
        }
    }
    mat.setFromTriplets(tripletList.begin(), tripletList.end());
    return mat;
}

int main(int argc, char** argv) {
    int k = 3;
    int acgm_cycles = 5;
    int n = 100;
    if(argc >= 1){
        k = atoi(argv[1]);
        cout << "Calculating " << k << " eigenvalues" << endl;
    }
    if(argc >= 2) {
        acgm_cycles = atoi(argv[2]);
        cout << "ACGM cycles: " << acgm_cycles << endl;
    }
    if(argc >= 3) {
        n = atoi(argv[3]);
        cout << "n: " << n << endl;
    }
    vector<double> xs(n), ys(n), zs(n);
    string path = "output/ho_cpp/";
    for (int i = 0; i < n; ++i) {
        xs[i] = ys[i] = zs[i] = -a + i * h;
    }
    
    SparseMatrix<double> mat = matsetup_oscillator(n, xs, ys, zs);
    cout << "Matrix generated" << endl;

    
    pair<MatrixXd, VectorXd> gcgm_pair;
    //accelerated_eigenpair = accelerated_cgm(mat, random_orthonormal_matrix(n, k), 200, 1e-15, acgm_cycles);
    //accelerated_eigenpair = lobpcg(mat, random_orthonormal_matrix(n, k), 200, 1e-15, acgm_cycles);
    //std::cout << "LOBPCG eigenvalues: " << accelerated_eigenpair.first << std::endl;
    gcgm_pair = gcgm(mat, MatrixXd::Identity(n, n), random_orthonormal_matrix(n, k), k);
    //cout << "GCGM eigenvalues: " << gcgm_pair.second << endl;

    //cout << "Accelerated eigenvalues: " << accelerated_eigenpair.first << endl;
    double coeff = h_bar * omega;
    double en;
    for(int i=0; i<k; i++) {
        en = coeff * (i+0.5);
        cout << "Energy N=" << i << ": " << en << " MeV" << endl;
        cout << "Error: N=" << i << ": " << (gcgm_pair.second(i)/en - 1)*100 << "%" << endl;
    }

    ofstream file(path + "eigenvectors.txt");
    //for (int i = 0; i < eigenpair.second.size(); ++i) {
        //file << eigenpair.second(i) << endl;
    //}
    file.close();

    ofstream file2(path + "x.txt");
    for(const auto &e : xs) {
        file2 << e << endl;
    }
    file2.close();
    //cout << MatrixXd(mat) << endl;
    


    
    return 0;
}
