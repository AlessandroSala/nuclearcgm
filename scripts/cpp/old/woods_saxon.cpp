#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Sparse>
#include <Eigen/Eigenvalues>
#include <omp.h>
#include <fstream>

using namespace std;
using namespace Eigen;

const int n = 30;
const double a = 12.0;
const double h = 2 * a / n;
const double V0 = 57.0;
const double A_val = 16;
const double R = 1.27 * pow(A_val, 1.0/3.0);
const double diff = 0.67;
const double h_bar = 6.62607015e-34;
const double m = 1.67262192e-27;
const double C = (-2*m/(h_bar*h_bar));
const int max_iter = 1000;
const double tol = 1e-28;
const double omega = 41*pow(A_val, -1.0/3.0)/h_bar;

inline int idx(int i, int j, int k) {
    return i + n * (j + n * k);
}

double pot(double x, double y, double z) {
    return -V0 / (1 + exp((sqrt(x*x + y*y + z*z) - R) / diff));
}
double pot_osc(double x, double y, double z) {
    return 0.5*(x*x + y*y + z*z)*m*omega*omega;
}
double A(int i, int j, int k, int i1, int j1, int k1, const vector<double>& xs, const vector<double>& ys, const vector<double>& zs) {
    if (i1 == i && j1 == j && k1 == k)
        return (pot_osc(xs[i], ys[j], zs[k])*C*h*h - 6) / (C * h * h);
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

double find_positive_root(const VectorXd& x, const SparseMatrix<double>& A, const VectorXd& p, const VectorXd& Ax, double xx) {
    VectorXd Ap = A * p;
    double xp = x.dot(p);
    double pp = p.dot(p);
    double pAp = p.dot(Ap);
    double xAp = x.dot(Ap);
    double xAx = x.dot(Ax);

    double a = (pAp) * (xp) - (xAp) * (pp);
    double b = (pAp) * (xx) - (xAx) * (pp);
    double c = (xAp) * (xx) - (xAx) * (xp);

    double delta = b * b - 4 * a * c;
    if (delta < 0) {
        return 0;
    }
    delta = sqrt(delta);
    double lambda1 = (-b + delta) / (2 * a);
    double lambda2 = (-b - delta) / (2 * a);
    if (lambda1 < 0 || lambda2 < 0) {
        return 0;
    }
    return lambda1 > lambda2 ? lambda1 : lambda2;

}

double compute_beta_FR(const VectorXd& r_new, const VectorXd& r_old) {
    return r_new.dot(r_new) / r_old.dot(r_old);
}

double f(const VectorXd& x, const VectorXd& Ax, double xx) {
    return x.dot(Ax) / xx;
}
VectorXd g(const VectorXd& x, const VectorXd& Ax, double xx) {
    return (Ax - f(x, Ax, xx) * x)/ xx;
}

pair<double, VectorXd> find_eigenpair(const SparseMatrix<double>& A) {
    int N = A.rows();
    VectorXd x = VectorXd::Random(N).normalized();
    VectorXd Ax(N);
    VectorXd grad(N);
    VectorXd d(N);
    VectorXd r_new(N);
    VectorXd r_old(N);
    double xx = x.dot(x);
    Ax = A * x;
    grad = g(x, Ax, xx);
    double g0 = grad.dot(grad);
    d = -grad;
    r_new = d;
    
    for (int iter = 0; iter < max_iter; ++iter) {
        r_old = r_new;
        x = x + d*find_positive_root(x, A, d, Ax, xx);
        Ax = A * x;
        xx = x.dot(x);
        grad = g(x, Ax, xx);
        
        if (grad.dot(grad) < tol*g0) break;

        r_new = -grad;
        d = compute_beta_FR(r_new, r_old) * d + r_new;
        cout << "Iteration " << iter << endl;
    }
    return {f(x, Ax, xx), x};
}


int main() {
    vector<double> xs(n), ys(n), zs(n);
    string path = "output/woods_saxon/";
    for (int i = 0; i < n; ++i) {
        xs[i] = ys[i] = zs[i] = -a + i * h;
    }
    
    SparseMatrix<double> mat = matsetup_oscillator(n, xs, ys, zs);
    cout << "Matrix generated" << endl;
    
    pair<double, VectorXd> eigenpair = find_eigenpair(mat);
    cout << "Smallest eigenvalue (iterative): " << eigenpair.first << endl;

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


    
    return 0;
}
