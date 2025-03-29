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

const int n = 1000;
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

    #pragma omp parallel for collapse(3)
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

int main() {
    vector<double> xs(n), ys(n), zs(n);
    string path = "output/ho_cpp/";
    for (int i = 0; i < n; ++i) {
        xs[i] = ys[i] = zs[i] = -a + i * h;
    }
    
    SparseMatrix<double> mat = matsetup_oscillator(n, xs, ys, zs);
    cout << "Matrix generated" << endl;

    
    pair<double, VectorXd> eigenpair = find_eigenpair(mat);
    cout << "Smallest eigenvalue (iterative): " << eigenpair.first << endl;
    double e_real = h_bar * omega * 0.5;
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
