
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

typedef Matrix<complex<double>, 2, 2> SpinMatrix;

int n = 10;
int k = 7;
const double a = 15.0;
double h = 2.0 * a / (n-1);
int acgm_cycles = 50;
const double V0 = 51.0;
const double A_val = 16;
const double r_0 = 1.27;
const double R = r_0 * pow(A_val, 1.0/3.0);
const double diff = 0.67;


inline int idx(int i, int j, int k, int s) {
    return s + 2*(i + n * (j + n * k));
}

double pot(double x, double y, double z) {
    return -V0 / (1 + exp((sqrt(x*x + y*y + z*z) - R) / diff));
}

double spinOrbitCoeff(double x, double y, double z) {
    double fac = 1;
    double r = sqrt(x*x + y*y + z*z);
    if(r > 1e-12) {
        double t = (r-R)/diff;
        fac = -pow(diff*r, -1)*exp(t)/pow(1+exp(t), 2);
    }
    return 0.44*V0*pow((r_0/h_bar), 2)*fac;
}
std::complex<double> A(int i, int j, int k, int s, int i1, int j1, int k1, int s1, const vector<double>& xs, const vector<double>& ys, const vector<double>& zs) {
    Matrix<complex<double>, 2, 2> spin(2, 2);
    spin.setZero();

    double v = 0;
    double x = xs[i], y = ys[j], z = zs[k];
    double ls = spinOrbitCoeff(x, y, z);
    if (i1 == i && j1 == j && k1 == k && s1 == s)
        v = (pot(xs[i], ys[j], zs[k])*C*h*h - 6) / (C * h * h);
    else if ((i == i1 && k == k1 && s==s1 && (j == j1 + 1 || j == j1 - 1)) ||
             (j == j1 && k == k1 && s==s1 && (i == i1 + 1 || i == i1 - 1)) ||
             (i == i1 && j == j1 && s==s1 && (k == k1 + 1 || k == k1 - 1)))
        v = 1 / (C * h * h);
    if((i + 1 == i1 || i - 1 == i1)&& j == j1 && k == k1)
        spin += (i1-i)*(pauli[1]*z - pauli[2]*y);

    else if(i == i1 && (j + 1 == j1 || j - 1 == j1) && k == k1)
        spin += (j1-j)*(-pauli[0]*z + pauli[2]*x);

    else if(i == i1 && j == j1 && (k + 1 == k1 || k - 1 == k1))
        spin += (k1-k)*(pauli[0]*y - pauli[1]*x);
    //ls = 0;
    spin = -pow(2*h, -1)*std::complex<double>(0, 1.0)*0.5*h_bar*h_bar*spin * ls;

    return spin(s, s1) + v;    
}
SparseMatrix<std::complex<double>> matsetup_oscillator(int n, const vector<double>& xs, const vector<double>& ys, const vector<double>& zs) {
    SparseMatrix<std::complex<double>> mat(n * n * n * 2, n * n * n * 2);
    vector<Triplet<std::complex<double>>> tripletList;

    #pragma omp parallel for collapse(4)
    for(int s = 0; s < 2; ++s) {
        for(int s1 = 0; s1 < 2; ++s1) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                for (int i1 = max(0, i-1); i1 <= min(n-1, i+1); ++i1) {
                    for (int j1 = max(0, j-1); j1 <= min(n-1, j+1); ++j1) {
                        for (int k1 = max(0, k-1); k1 <= min(n-1, k+1); ++k1) {
                            int n0 = idx(i, j, k, s);
                            int n1 = idx(i1, j1, k1, s1);
                            std::complex<double> val = A(i, j, k, s, i1, j1, k1, s1, xs, ys, zs);
                            if (val != std::complex<double>(0, 0)) {
                                #pragma omp critical
                                tripletList.emplace_back(n0, n1, val);
                            }
                        }
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

    std::cout << "Pauli matrices: " << std::endl;
    for(int i = 0; i < 3; ++i) {
        std::cout << pauli[i] << std::endl;
    }

    if(argc >= 2) {
        n = atoi(argv[1]);
        h = 2.0 * a / (n-1);
    } 
    if(argc >= 3) {
        k = atoi(argv[2]);
    } 
    if(argc >= 4) {
        acgm_cycles = atoi(argv[3]);
    }

    vector<double> xs(n), ys(n), zs(n);
    string path = "output/wd_3d/";

    for (int i = 0; i < n; ++i) {
        xs[i] = ys[i] = zs[i] = -a + i * h;
    }
    
    SparseMatrix<std::complex<double>> mat = matsetup_oscillator(n, xs, ys, zs);
    cout << "Matrix generated" << endl;

    cout << "Hermit: " << endl;
    //cout << mat.toDense() << endl;

    int N = mat.rows();

    
    SparseMatrix<double> B(N, N);
    B.setIdentity();
    pair<MatrixXcd, VectorXd> eigenpairs = gcgm_complex(mat, B, MatrixXcd(random_orthonormal_matrix(N, k)), k, 35 + 0.01, acgm_cycles, 1.0e-9, 100 ); 

    std::cout << "GCGM eigenvalues: " << eigenpairs.second << std::endl;


    double e_real = -31.091;
    cout << "Real GS energy: " << e_real << " MeV" << endl;
    cout << "Error: " << ((eigenpairs.second(0))/e_real - 1)*100 << "%" << endl;


    for(int ev = 0; ev < eigenpairs.second.size(); ++ev) {

        ofstream file(path + "eigenvectors_" + std::to_string(ev) + ".txt");
        for (int i = 0; i < eigenpairs.first.rows(); ++i) {
            file << eigenpairs.first(i, ev) << endl;
        }
        
        file.close();
    }

    ofstream file2(path + "x.txt");
    for(const auto &e : xs) {
        file2 << e << endl;
    }
    file2.close();
    //cout << MatrixXd(mat) << endl;
    
    

    
    return 0;
}
